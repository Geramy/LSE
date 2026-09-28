#include <array>
#include <cmath>
#include <limits>

#include "harness.hpp"
#include "lse/backends/hrx/arch_database.hpp"
#include "lse/backends/hrx/hipc/hip_emitter.hpp"
#include "lse/backends/hrx/loomc/loom_emitter.hpp"
#include "lse/backends/hrx/loomc/loom_types.hpp"
#include "lse/graph/kernel_primitive.hpp"
#include "lse/graph/ops.hpp"
#include "lse/dispatch/cache.hpp"
using namespace lse;
using namespace lse::graph;
namespace {
Array leaf(Shape s, DType d = DType::kF32) {
  auto n = std::make_shared<Node>();
  n->shape = s;
  n->dtype = d;
  n->materialized = true;
  return Array(n);
}
FusionGroup solo(const Array& a) {
  FusionGroup g;
  g.nodes = {a.node()};
  g.outputs = g.nodes;
  g.inputs = a.node()->inputs;
  g.anchor = a.node()->kind;
  g.anchor_class = a.node()->fclass;
  return g;
}
struct Fixture {
  backend::DeviceInfo d;
  backend::AmdDeviceInfo amd;
  backend::HipEmitter hip;
  backend::LoomEmitter loom;
  Fixture() {
    d.arch = "gfx1201";
    d.compute_units = 64;
    d.wavefront_size = 32;
    d.max_threads_per_workgroup = 1024;
    d.lds_bytes_per_workgroup = 65536;
    backend::apply_arch_defaults(d, amd);
    d.extension_id = backend::AmdDeviceInfo::kExtensionId;
    d.extension = &amd;
  }
  void check(const FusionGroup& g, bool wave) {
    auto h = hip.emit(g, d), l = loom.emit(g, d);
    LSE_EXPECT(h.ok());
    LSE_EXPECT(l.ok());
    if (!h.ok()) std::fprintf(stderr, "HIP: %s\n", h.status().to_string().c_str());
    if (!l.ok()) std::fprintf(stderr, "Loom: %s\n", l.status().to_string().c_str());
    if (!h.ok() || !l.ok()) return;
    LSE_EXPECT((h->source.find("__shfl_xor") != std::string::npos) == wave);
    LSE_EXPECT((l->source.find("kernel.subgroup.shuffle") != std::string::npos) == wave);
    if (g.outputs.size() > 1) {
      LSE_EXPECT(h->source.find("__device__ float lse_l2_normalize_") != std::string::npos);
      for (const auto& output : g.outputs) {
        bool stored = false;
        for (std::size_t i = 0; i < h->binding_order.size(); ++i)
          if (h->binding_order[i] == output)
            stored = h->source.find("b" + std::to_string(i) + "[i] = ") != std::string::npos;
        LSE_EXPECT(stored);
      }
    }
    if (wave) {
      for (const auto* e : {&*h, &*l}) {
        LSE_EXPECT_EQ(e->dims.workgroup_size[0], 128u);
        LSE_EXPECT_EQ(e->dims.workgroup_count[0], 4u);
        LSE_EXPECT_EQ(e->dims.subgroup_size, 32u);
        LSE_EXPECT_EQ(e->lds_bytes, 0u);
      }
    }

  }
};
}  // namespace
LSE_TEST(wave32_l2_exact_selection_and_fallback_contract) {
  Fixture f;
  auto g = solo(l2_normalize(leaf({1, 1, 16, 128}), 1e-6f));
  f.check(g, true);
  auto generic = l2_normalize(leaf({1, 1, 16, 128}));
  LSE_EXPECT(generic.node()->attrs[0] == 1e-12f);
  f.check(solo(generic), true);
  for (auto sh : {Shape{1, 1, 16, 127}, Shape{1, 1, 16, 129}, Shape{1, 1, 15, 128},
                  Shape{1, 2, 16, 128}, Shape{16, 128}})
    f.check(solo(l2_normalize(leaf(sh))), false);
  for (auto dt : {DType::kBF16, DType::kF16}) {
    auto n = l2_normalize(leaf({1, 1, 16, 128}, dt));
    auto* p = dynamic_cast<const KernelPrimitiveBase*>(n.node()->prim);
    std::vector<Shape> in{n.node()->inputs[0]->shape};
    std::vector<DType> d{dt};
    KernelShapes s;
    s.inputs = in;
    s.input_dtypes = d;
    s.output = n.shape();
    s.output_dtype = dt;
    s.device = &f.d;
    s.types = backend::loom_types();
    auto intr = backend::loom_sources();
    s.intrinsics = &intr;
    s.attrs = n.node()->attrs;
    LSE_EXPECT(p->specialize(s) == p);
  }
  for (float eps : {0.0f, -1.0f, std::numeric_limits<float>::quiet_NaN(),
                    std::numeric_limits<float>::infinity()})
    f.check(solo(l2_normalize(leaf({1, 1, 16, 128}), eps)), false);
  auto empty = solo(l2_normalize(leaf({1, 1, 0, 128})));
  auto zero = f.loom.emit(empty, f.d);
  LSE_EXPECT(!zero.ok());
  if (!zero.ok()) LSE_EXPECT(zero.status().code() == StatusCode::kUnimplemented);
  f.d.arch = "gfx1151";
  f.check(g, false);
  f.d.arch = "gfx1201";
  f.d.wavefront_size = 64;
  f.check(g, false);
  f.d.wavefront_size = 32;
  f.d.max_threads_per_workgroup = 64;
  f.check(g, false);
}
LSE_TEST(wave32_l2_query_scale_epilogue_and_live_outputs) {
  Fixture f;
  auto n = l2_normalize(leaf({1, 1, 16, 128}), 1e-6f);
  auto q = n * Array::full({1}, DType::kF32, 1 / std::sqrt(128.0f));
  const NodePtr roots[]{q.node()};
  auto gs = Partitioner::partition(roots);
  bool found = false;
  for (auto& g : gs)
    if (g.anchor == OpKind::kL2Norm) {
      found = true;
      f.check(g, true);
      LSE_EXPECT(g.outputs.size() == 1 && g.outputs[0] == q.node());
      g.outputs = {n.node(), q.node()};
      f.check(g, false);
    }
  LSE_EXPECT(found);
}
LSE_TEST(wave32_l2_phase_keeps_virtual_scalar_indexing) {
  Fixture f;
  auto n = l2_normalize(leaf({1, 1, 16, 128}), 1e-6f);
  auto q = n * Array::full({1}, DType::kF32, 1 / std::sqrt(128.0f));
  auto r = repeat(q, 3, 2);
  const NodePtr roots[]{r.node()};
  bool found = false;
  for (const auto& phase : Partitioner::phases(roots)) {
    auto g = Partitioner::phase_group(phase, roots);
    for (auto& node : g.nodes)
      if (node->kind == OpKind::kL2Norm) {
        found = true;
        LSE_EXPECT(g.is_phase);
        auto h = f.hip.emit(g, f.d);
        LSE_EXPECT(h.ok());
        if (h.ok()) LSE_EXPECT(h->source.find("__shfl_xor") == std::string::npos);
        auto l = f.loom.emit(g, f.d);
        LSE_EXPECT(!l.ok());
        if (!l.ok()) LSE_EXPECT(l.status().code() == StatusCode::kUnimplemented);
        break;
      }
  }
  LSE_EXPECT(found);
}
LSE_TEST(wave32_l2_fp32_association_epsilon_and_index_coverage) {
  double worst_abs = 0, worst_rel = 0;
  unsigned rows = 0;
  for (unsigned pattern = 0; pattern < 8; ++pattern)
    for (unsigned r = 0; r < 128; ++r)
      for (float eps : {1e-12f, 1e-6f, 2.0f}) {
        std::array<float, 128> x{};
        std::array<float, 32> sum{};
        std::array<unsigned, 128> writes{};
        double ref = 0;
        float serial = 0;
        for (unsigned i = 0; i < 128; ++i) {
          float v = std::sin(float(i * 17 + r * 3 + 1)) * 3.0f;
          if (pattern == 1) v = 0;
          if (pattern == 2) v = (i == r ? 1000.0f : v * 0.001f);
          if (pattern == 3) v = std::ldexp(v, -30);
          if (pattern == 4) v = std::ldexp(v, 30);
          if (pattern == 5) v = (i % 2 ? 1 : -1) * std::ldexp(float(1 + i % 7), int(i % 30) - 15);
          if (pattern == 6) v = std::ldexp(v, -45);
          if (pattern == 7) v = std::ldexp(v, 45);
          x[i] = v;
          ref += double(v) * v;
          serial = std::fma(v, v, serial);
        }
        for (unsigned lane = 0; lane < 32; ++lane)
          for (unsigned j = 0; j < 4; ++j) {
            float v = x[lane + j * 32];
            sum[lane] = std::fma(v, v, sum[lane]);
          }
        for (unsigned mask = 1; mask < 32; mask *= 2) {
          auto before = sum;
          for (unsigned lane = 0; lane < 32; ++lane) sum[lane] = before[lane] + before[lane ^ mask];
        }
        float inv = 1.0f / std::max(std::sqrt(sum[0]), eps),
              si = 1.0f / std::max(std::sqrt(serial), eps);
        double ri = 1.0 / std::max(std::sqrt(ref), double(eps));
        for (unsigned lane = 0; lane < 32; ++lane)
          for (unsigned j = 0; j < 4; ++j) {
            unsigned i = lane + j * 32;
            float out = x[i] * inv;
            double want = double(x[i]) * ri;
            double delta = std::abs(double(out) - want);
            worst_abs = std::max(worst_abs, delta);
            if (want != 0) worst_rel = std::max(worst_rel, delta / std::abs(want));
            LSE_EXPECT(std::isfinite(out));
            LSE_EXPECT(delta <= 1e-6 + 1e-5 * std::abs(want));
            LSE_EXPECT(std::abs(out - x[i] * si) <= 1e-6f + 1e-5f * std::abs(out));
            ++writes[i];
          }
        for (auto count : writes) LSE_EXPECT_EQ(count, 1u);
        ++rows;
      }
  std::printf("       rows=%u worst_abs=%.9g worst_rel=%.9g\n", rows, worst_abs, worst_rel);
}
int main() { return lse::test::run_all(); }
