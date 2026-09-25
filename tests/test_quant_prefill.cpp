// Host-only contracts for the shared HIP/Loom Q6 prefill schedule.
#include "harness.hpp"
#include "lse/backends/hrx/arch_database.hpp"
#include "lse/backends/hrx/hipc/hip_emitter.hpp"
#include "lse/backends/hrx/hipc/hip_sources.hpp"
#include "lse/backends/hrx/hipc/hip_types.hpp"
#include "lse/backends/hrx/loomc/loom_emitter.hpp"
#include "lse/graph/graph.hpp"
#include "lse/graph/kernel_primitive.hpp"
#include "lse/graph/ops.hpp"
#include "lse/kernels/quant_panel.hpp"

#include <algorithm>
#include <array>
#include <bit>

namespace {
using namespace lse;
using namespace lse::graph;

struct Fixture {
  backend::DeviceInfo device;
  backend::AmdDeviceInfo amd;
  std::array<Shape, 4> inputs;
  std::array<DType, 4> dtypes{DType::kF32, DType::kU32,
                            DType::kBF16, DType::kBF16};
  DialectSourceTable intrinsics = backend::hip_sources();
  KernelShapes shapes;
  Fixture(int m, int k) {
    device.arch = "gfx1201";
    device.wavefront_size = 32;
    device.max_threads_per_workgroup = 1024;
    device.lds_bytes_per_workgroup = 65536;
    backend::apply_arch_defaults(device, amd);
    device.extension_id = backend::AmdDeviceInfo::kExtensionId;
    device.extension = &amd;
    inputs = {Shape{m, k}, Shape{17, k * 6 / 32},
              Shape{17, k / 64}, Shape{17, k / 64}};
    shapes.inputs = inputs;
    shapes.input_dtypes = dtypes;
    shapes.output = Shape{m, 17};
    shapes.iattrs = {6, 64, 0, 0};
    shapes.device = &device;
    shapes.types = backend::hip_types();
    shapes.intrinsics = &intrinsics;
  }
};
const KernelPrimitiveBase* primitive() {
  return dynamic_cast<const KernelPrimitiveBase*>(find_primitive("quant_linear"));
}
Array input(const Shape& shape, DType dtype) {
  auto node = std::make_shared<Node>();
  node->shape = shape;
  node->dtype = dtype;
  node->materialized = true;
  return Array(node);
}
}  // namespace

LSE_TEST(q6_prefill_plan_declares_multirow_resources_and_no_single_row_panel) {
  const auto* p = primitive();
  LSE_EXPECT(p != nullptr);
  if (!p) return;
  for (int m : {1, 2, 3, 4, 5, 17, 31, 32, 33, 63, 128, 256}) {
    for (int k : {64, 576, 1024, 1088, 4096, 5120, 17408}) {
      Fixture f(m, k);
      const unsigned rows = m >= 32 ? 8u : (m >= 4 ? 4u : (m >= 2 ? 2u : 1u));
      const auto plan = p->plan(f.shapes);
      LSE_EXPECT_EQ(plan.workgroup_count[0], 3u);
      LSE_EXPECT_EQ(plan.workgroup_count[1], (unsigned(m) + rows - 1) / rows);
      LSE_EXPECT_EQ(plan.lds_bytes, m > 1 ? rows * unsigned(std::min(k, rows >= 8 ? 512 : 1024)) * 4u
                                       : (k <= 16384 ? unsigned(k) * 4u : 0u));
      if (m > 1) LSE_EXPECT_EQ(p->staged_row(f.shapes).count, 0u);
    }
  }
}

LSE_TEST(q6_prefill_fallback_preserves_architecture_and_staging_contracts) {
  const auto* p = primitive();
  LSE_EXPECT(p != nullptr);
  if (!p) return;
  for (int reason = 0; reason < 8; ++reason) {
    Fixture f(5, 4096);
    if (reason == 0) f.device.arch = "gfx1200";
    if (reason == 1) f.device.wavefront_size = 64;
    if (reason == 2) f.device.lds_bytes_per_workgroup = 8192;
    if (reason == 3) f.dtypes[2] = f.dtypes[3] = DType::kF32;
    if (reason == 4) f.shapes.staged = {"caller_panel", 4096};
    if (reason == 5) f.shapes.staged_quant.codes = "caller_codes";
    if (reason == 6) {
      f.shapes.iattrs[1] = 128;
      f.inputs[2] = f.inputs[3] = Shape{17, 32};
    }
    if (reason == 7) f.device.lds_bytes_per_workgroup = 0;
    LSE_EXPECT_EQ(p->plan(f.shapes).workgroup_count[1], 5u);
  }
  // Indexed rows can select different matrices, so they cannot share weights.
  const auto* indexed = dynamic_cast<const KernelPrimitiveBase*>(
      find_primitive("quant_linear_indexed"));
  LSE_EXPECT(indexed != nullptr);
  if (!indexed) return;
  Fixture f(5, 4096);
  const Shape shapes[] = {Shape{5, 4096}, Shape{2, 17, 768},
                         Shape{2, 17, 64}, Shape{2, 17, 64}, Shape{5, 1}};
  const DType dtypes[] = {DType::kF32, DType::kU32, DType::kBF16,
                         DType::kBF16, DType::kF32};
  f.shapes.inputs = shapes;
  f.shapes.input_dtypes = dtypes;
  f.shapes.iattrs = {0, 6, 64, 0};
  LSE_EXPECT_EQ(indexed->plan(f.shapes).workgroup_count[1], 5u);
}

LSE_TEST(q6_prefill_hip_and_loom_emit_matching_launch_and_lds_requirements) {
  for (int m : {3, 5, 17, 31, 32, 33, 128}) {
    Fixture f(m, 1088);
    auto y = quant_linear(input(f.inputs[0], f.dtypes[0]),
                          input(f.inputs[1], f.dtypes[1]),
                          input(f.inputs[2], f.dtypes[2]),
                          input(f.inputs[3], f.dtypes[3]), 6, 64);
    const NodePtr roots[] = {y.node()};
    const auto groups = Partitioner::partition(roots);
    LSE_EXPECT_EQ(groups.size(), 1u);
    if (groups.size() != 1) continue;
    auto hip = backend::HipEmitter{}.emit(groups.front(), f.device);
    auto loom = backend::LoomEmitter{}.emit(groups.front(), f.device);
    LSE_EXPECT(hip.ok() && loom.ok());
    if (!hip.ok() || !loom.ok()) continue;
    const unsigned rows = m >= 32 ? 8u : (m >= 4 ? 4u : 2u);
    LSE_EXPECT_EQ(hip->dims.workgroup_count[1], (unsigned(m) + rows - 1) / rows);
    LSE_EXPECT_EQ(hip->lds_bytes, rows * (rows >= 8 ? 2048u : 4096u));
    LSE_EXPECT_EQ(loom->lds_bytes, hip->lds_bytes);
    LSE_EXPECT_EQ(loom->dims.workgroup_count[1], hip->dims.workgroup_count[1]);
    auto actual_shared = backend::HipEmitter::shared_bytes(hip->source);
    LSE_EXPECT(actual_shared.ok());
    if (actual_shared.ok()) LSE_EXPECT_EQ(*actual_shared, hip->lds_bytes);
  }
}

LSE_TEST(q6_panel_layout_is_bijective_and_spreads_strided_lane_reads) {
  for (unsigned count : {64u, 576u, 1024u, 4096u, 16384u}) {
    std::vector<unsigned> panel(count, count);
    for (unsigned i = 0; i < count; ++i) {
      const auto at = kernels::q6_panel_index(i);
      LSE_EXPECT(at < count);
      if (at >= count) continue;
      LSE_EXPECT_EQ(panel[at], count);
      panel[at] = i;
    }
    for (unsigned i = 0; i < count; ++i)
      LSE_EXPECT_EQ(panel[kernels::q6_panel_index(i)], i);
  }
  // This verifies address distribution, not a claim about measured stall count.
  for (unsigned banks : {32u, 64u}) {
    for (unsigned c = 0; c < 16; ++c) {
      std::array<unsigned, 64> hits{};
      for (unsigned lane = 0; lane < 32; ++lane)
        ++hits[kernels::q6_panel_index(lane * 16 + c) % banks];
      LSE_EXPECT_EQ(*std::max_element(hits.begin(), hits.end()), 1u);
    }
  }
}

namespace {
static float activation(unsigned row, unsigned k) {
  return std::sin(float(row * 17 + k * 3) * 0.03f) * 0.7f;
}
static float weight(unsigned col, unsigned k) {
  // BF16-representable scale/bias and all Q6 code values, including straddles.
  const float scale = float((col + k / 64) % 7 + 1) / 64.0f;
  const float bias = float(int((col * 3 + k / 64) % 9) - 4) / 8.0f;
  const auto code = (col * 17 + k * 13 + 7) % 64;
  return std::fma(float(code), scale, bias);
}
static float reduce(std::array<float, 32> values) {
  for (unsigned bit = 1; bit < 32; bit <<= 1) {
    const auto prior = values;
    for (unsigned lane = 0; lane < 32; ++lane)
      values[lane] = prior[lane] + prior[lane ^ bit];
  }
  return values[0];
}
int check_schedule() {
  for (unsigned tile : {64u, 128u, 576u, 1024u}) {
    std::vector<bool> seen(tile);
    for (unsigned at = 0; at < tile; ++at) {
      const auto mapped = lse::kernels::q6_panel_index(at);
      if (mapped >= tile || seen[mapped]) return 2;
      seen[mapped] = true;
    }
  }
  for (unsigned banks : {32u, 64u}) {
    for (unsigned c = 0; c < 16; ++c) {
      std::array<unsigned, 64> original{}, rotated{};
      for (unsigned lane = 0; lane < 32; ++lane) {
        ++original[(lane * 16 + c) % banks];
        ++rotated[lse::kernels::q6_panel_index(lane * 16 + c) % banks];
      }
      unsigned before = 0, after = 0;
      for (unsigned b = 0; b < banks; ++b) {
        before = std::max(before, original[b]);
        after = std::max(after, rotated[b]);
      }
      if (after != 1 || after >= before) return 3;
    }
  }
  std::puts("PASS physical panel bounds/permutation and per-instruction bank distribution (not measured stalls)");
  unsigned checked = 0;
  for (unsigned m : {2u, 3u, 4u, 5u, 17u, 31u, 32u, 33u, 63u, 128u}) {
    for (unsigned k : {64u, 576u, 1088u, 4096u, 17408u}) {
      constexpr unsigned n = 17;
      const unsigned rows = m >= 32 ? 8 : (m >= 4 ? 4 : 2);
      const unsigned tile = std::min(k, rows >= 8 ? 512u : 1024u);
      std::vector<float> expected(m * n), actual(m * n, NAN);
      std::uint64_t baseline_weights = 0, reused_weights = 0;
      for (unsigned row = 0; row < m; ++row) for (unsigned col = 0; col < n; ++col) {
        std::array<float, 32> sums{};
        for (unsigned lane = 0; lane < 32; ++lane)
          for (unsigned chunk = lane; chunk < k / 16; chunk += 32)
            for (unsigned c = 0; c < 16; ++c) {
              const auto at = chunk * 16 + c;
              sums[lane] = std::fma(activation(row, at), weight(col, at), sums[lane]);
              ++baseline_weights;
            }
        expected[row * n + col] = reduce(sums);
      }
      for (unsigned row = 0; row < m; row += rows) for (unsigned col = 0; col < n; ++col) {
        std::array<std::array<float, 32>, 8> sums{};
        for (unsigned base = 0; base < k; base += tile) {
          std::vector<float> panel(rows * tile);
          for (unsigned r = 0; r < rows; ++r)
            for (unsigned at = 0; at < tile; ++at)
              panel[r * tile + lse::kernels::q6_panel_index(at)] = row + r < m && base + at < k
                  ? activation(row + r, base + at) : 0.0f;
          for (unsigned lane = 0; lane < 32; ++lane)
            for (unsigned chunk = lane; chunk < tile / 16; chunk += 32) {
              if (base + chunk * 16 >= k) continue;
              for (unsigned c = 0; c < 16; ++c) {
                const auto at = base + chunk * 16 + c;
                const float w = weight(col, at);
                ++reused_weights;
                for (unsigned r = 0; r < rows; ++r) {
                  const float x = panel[r * tile + lse::kernels::q6_panel_index(chunk * 16 + c)];
                  sums[r][lane] = std::fma(x, w, sums[r][lane]);
                }
              }
            }
        }
        for (unsigned r = 0; r < rows && row + r < m; ++r)
          actual[(row + r) * n + col] = reduce(sums[r]);
      }
      for (unsigned i = 0; i < actual.size(); ++i) {
        if (std::bit_cast<std::uint32_t>(actual[i]) != std::bit_cast<std::uint32_t>(expected[i])) {
          std::fprintf(stderr, "FAIL M%u K%u output%u expected=%g actual=%g\n", m, k, i, expected[i], actual[i]);
          return 1;
        }
      }
      const auto expected_reads = std::uint64_t((m + rows - 1) / rows) * n * k;
      if (reused_weights != expected_reads || reused_weights >= baseline_weights) return 1;
      checked += actual.size();
      std::printf("PASS M%u K%u N17: %zu bit-exact FP32 results, weight-decodes %llu -> %llu\n",m,k,actual.size(),(unsigned long long)baseline_weights,(unsigned long long)reused_weights);
    }
  }
  std::printf("PASS %u FP32 schedule outputs; no GPU execution\n", checked);
  return 0;
}

}

LSE_TEST(q6_prefill_preserves_fp32_accumulation_order_with_ragged_tiles) {
  LSE_EXPECT_EQ(check_schedule(), 0);
}

LSE_TEST(q6_decode_rotation_preserves_resources_and_caller_panel_layout) {
  const auto* p = primitive();
  LSE_EXPECT(p != nullptr);
  if (!p) return;
  for (int k : {64, 576, 1024, 1088, 4096, 5120, 16384, 17408}) {
    Fixture f(1, k);
    const auto expected_lds = k <= 16384 ? unsigned(k) * 4u : 0u;
    LSE_EXPECT_EQ(p->plan(f.shapes).lds_bytes, expected_lds);
    LSE_EXPECT_EQ(p->plan(f.shapes).workgroup_count[1], 1u);
    LSE_EXPECT_EQ(p->staged_row(f.shapes).count, unsigned(k));
    // A prefilled caller panel must not be rotated or redeclared privately.
    f.shapes.staged = {"caller_panel", unsigned(k)};
    LSE_EXPECT_EQ(p->plan(f.shapes).lds_bytes, 0u);
    f.shapes.store = [](std::string_view i, std::string_view v) {
      return "out[" + std::string(i) + "] = " + std::string(v) + ";";
    };
    const auto body = p->emit_kernel(f.shapes);
    LSE_EXPECT(!body.empty());
    LSE_EXPECT(body.find("caller_panel") != std::string::npos);
    auto declared = backend::HipEmitter::shared_bytes(body);
    LSE_EXPECT(declared.ok());
    if (declared.ok()) LSE_EXPECT_EQ(*declared, 0u);
  }
}

LSE_TEST(q6_decode_quad_plan_matches_both_emitters_and_ragged_columns) {
  const auto* p = primitive();
  LSE_EXPECT(p != nullptr);
  if (!p) return;
  for (int n : {4, 32, 36, 64}) {
    for (int k : {64, 1088, 5120, 16384, 16448, 17408}) {
      Fixture f(1, k);
      f.inputs[1] = Shape{n, k * 6 / 32};
      f.inputs[2] = f.inputs[3] = Shape{n, k / 64};
      f.shapes.output = Shape{1, n};
      const auto plan = p->plan(f.shapes);
      LSE_EXPECT_EQ(plan.workgroup_count[0], unsigned((n + 31) / 32));
      LSE_EXPECT_EQ(plan.workgroup_count[1], 1u);
      LSE_EXPECT_EQ(plan.lds_bytes, k <= 16384 ? unsigned(k * 4) : 0u);
      LSE_EXPECT_EQ(p->staged_row(f.shapes).count, 0u);
      LSE_EXPECT_EQ(p->traffic(f.shapes).workgroups, plan.workgroup_count[0]);
      auto out = quant_linear(input(f.inputs[0], f.dtypes[0]),
                              input(f.inputs[1], f.dtypes[1]),
                              input(f.inputs[2], f.dtypes[2]),
                              input(f.inputs[3], f.dtypes[3]), 6, 64);
      const NodePtr roots[] = {out.node()};
      const auto groups = Partitioner::partition(roots);
      LSE_EXPECT_EQ(groups.size(), 1u);
      if (groups.size() != 1) continue;
      auto hip = backend::HipEmitter{}.emit(groups.front(), f.device);
      auto loom = backend::LoomEmitter{}.emit(groups.front(), f.device);
      LSE_EXPECT(hip.ok() && loom.ok());
      if (!hip.ok() || !loom.ok()) continue;
      LSE_EXPECT_EQ(hip->dims.workgroup_count[0], plan.workgroup_count[0]);
      LSE_EXPECT_EQ(loom->dims.workgroup_count[0], plan.workgroup_count[0]);
      LSE_EXPECT_EQ(hip->lds_bytes, plan.lds_bytes);
      LSE_EXPECT_EQ(loom->lds_bytes, plan.lds_bytes);
      LSE_EXPECT(hip->binding_order == loom->binding_order);
    }
  }
}

LSE_TEST(q6_decode_quad_excludes_odd_columns_and_external_panels) {
  const auto* p = primitive();
  LSE_EXPECT(p != nullptr);
  if (!p) return;
  for (int reason = 0; reason < 5; ++reason) {
    Fixture f(1, 5120);
    // All other refusal cases must satisfy the divisible-by-four condition,
    // otherwise that first gate hides regressions in the gate under test.
    const int n = reason == 0 ? 19 : 20;
    f.inputs[1] = Shape{n, 960};
    f.inputs[2] = f.inputs[3] = Shape{n, 80};
    f.shapes.output = Shape{1, n};
    if (reason == 1) f.device.arch = "gfx1200";
    if (reason == 2) f.shapes.staged = {"caller_panel", 5120};
    if (reason == 3) f.shapes.staged_quant.codes = "caller_codes";
    if (reason == 4) f.dtypes[2] = f.dtypes[3] = DType::kF32;
    LSE_EXPECT_EQ(p->plan(f.shapes).workgroup_count[0], 3u);
  }
}

LSE_TEST_MAIN()
