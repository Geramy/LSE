#include "kernels/gdn.hpp"

#include "lse/graph/kernel_primitive.hpp"
#include "lse/math.hpp"
#include "lse/backends/hrx/device_info.hpp"

#include <string>

namespace lse::backend::hrx_kernels {

using namespace lse::graph;
namespace math = lse::math;

namespace {

constexpr std::uint32_t kBlock = 256;

std::uint32_t wave_of(const DeviceInfo* device) {
  if (device == nullptr) return 32;
  const AmdDeviceInfo* amd = device_extension<AmdDeviceInfo>(*device);
  if (amd != nullptr && (amd->wavefront_size == 32 || amd->wavefront_size == 64)) {
    return amd->wavefront_size;
  }
  return 32;
}

enum class GdnWrite : std::uint8_t { kOut, kState, kBoth };

// One wave owns one row of S. Each lane holds D/wave (or 1) scalars — never
// a D-wide ext_vector, which is what spilled 64 floats to scratch.
// kBoth scans time once: write o[t] every step and S at the end.
std::string emit_gdn(const KernelShapes& s, GdnWrite mode) {
  const Shape& q = s.inputs[0];
  const auto batch = static_cast<std::uint32_t>(q.dim(0));
  const auto seq = static_cast<std::uint32_t>(q.dim(1));
  const auto heads = static_cast<std::uint32_t>(q.dim(2));
  const auto D = static_cast<std::uint32_t>(q.dim(3));
  const bool write_state = mode != GdnWrite::kOut;
  const bool write_out = mode != GdnWrite::kState;
  if (seq == 0 || heads == 0 || D == 0) return {};
  if (write_out && !s.store) return {};
  if (mode == GdnWrite::kBoth && s.inputs.size() < 7) return {};

  const std::uint32_t wave = wave_of(s.device);
  const std::uint32_t tile = (D + wave - 1) / wave;
  const std::uint32_t rows =
      write_state ? batch * heads * D : batch * seq * heads * D;

  kir::KernelBody k(s.types, *s.intrinsics);
  k.set_store(s.store);
  const auto qin = k.input<kir::f32>(0);
  const auto kin = k.input<kir::f32>(1);
  const auto vin = k.input<kir::f32>(2);
  const auto alpha = k.input<kir::f32>(3);
  const auto beta = k.input<kir::f32>(4);
  const auto sin = k.input<kir::f32>(5);
  const auto i = k.thread_id();
  const auto lane = k.let<kir::u32>("lane", i % wave);
  const auto wid = k.let<kir::u32>("wid", i / wave);
  k.ret_if(wid >= rows);

  const auto row = k.let<kir::u32>("row", wid % D);
  const auto h = k.let<kir::u32>("h", (wid / D) % heads);
  kir::Val<kir::u32> b, t_out;
  if (write_state) {
    b = k.let<kir::u32>("b", wid / (D * heads));
    t_out = k.let<kir::u32>("to", k.constant<kir::u32>(seq - 1));
  } else {
    t_out = k.let<kir::u32>("to", (wid / (D * heads)) % seq);
    b = k.let<kir::u32>("b", wid / (D * heads * seq));
  }
  kir::Buffer<kir::f32> sout;
  if (mode == GdnWrite::kBoth) sout = k.input<kir::f32>(6);

  k.statement("float s[" + std::to_string(tile) + "];");
  kir::Tile<kir::f32> srow(&k, &k.types(), "s", tile);

  k.unroll("e0", tile, [&](kir::Val<kir::u32> e) {
    const auto j = k.let<kir::u32>("j0", lane + e * wave);
    const auto idx =
        ((b * heads + h) * D + row) * D + j;
    srow[e] = kir::select(j < D, sin[idx].read(), k.lit(0.0f));
  });

  auto reduce = [&](std::string_view name, kir::Val<kir::f32> v) {
    auto acc = k.var<kir::f32>(name, v);
    for (std::uint32_t m = 1; m < wave; m <<= 1) {
      acc = acc.read() + math::shfl_xor(acc.read(), k.constant<kir::u32>(m));
    }
    return acc.read();
  };

  auto result = k.var<kir::f32>("oacc", k.lit(0.0f));
  k.loop("t", k.constant<kir::u32>(0), k.constant<kir::u32>(seq), 1,
         [&](kir::Val<kir::u32> t) {
           const auto sc = (b * seq + t) * heads + h;
           const auto vec = sc * D;
           const auto a = k.let<kir::f32>("al", alpha[sc].read());
           auto skp = k.var<kir::f32>("skp", k.lit(0.0f));
           k.unroll("e1", tile, [&](kir::Val<kir::u32> e) {
             const auto j = lane + e * wave;
             srow[e] = srow[e].read() * a;
             k.when(j < D, [&] {
               skp = math::fma(srow[e].read(), kin[vec + j].read(), skp.read());
             });
           });
           const auto sk = reduce("skr", skp.read());
           const auto bt = k.let<kir::f32>("bt", beta[sc].read());
           const auto delta = k.let<kir::f32>(
               "del", (vin[vec + row].read() - sk) * bt);
           auto accp = k.var<kir::f32>("acp", k.lit(0.0f));
           k.unroll("e2", tile, [&](kir::Val<kir::u32> e) {
             const auto j = lane + e * wave;
             k.when(j < D, [&] {
               srow[e] = math::fma(delta, kin[vec + j].read(), srow[e].read());
               accp = math::fma(srow[e].read(), qin[vec + j].read(),
                                accp.read());
             });
           });
           const auto acc = reduce("acr", accp.read());
           if (mode == GdnWrite::kBoth) {
             k.when(lane == 0, [&] {
               k.store(((b * seq + t) * heads + h) * D + row, acc);
             });
           } else {
             k.when(t == t_out, [&] { result = acc; });
           }
         });

  if (write_state) {
    k.unroll("ew", tile, [&](kir::Val<kir::u32> e) {
      const auto j = k.let<kir::u32>("jw", lane + e * wave);
      k.when(j < D, [&] {
        const auto idx = ((b * heads + h) * D + row) * D + j;
        if (mode == GdnWrite::kBoth) sout[idx] = srow[e].read();
        else k.store(idx, srow[e].read());
      });
    });
  } else {
    k.when(lane == 0, [&] {
      k.store(((b * seq + t_out) * heads + h) * D + row, result.read());
    });
  }
  return k.str();
}

ThreadPlan gdn_plan(const KernelShapes& s, bool write_state) {
  ThreadPlan tp;
  tp.workgroup_size[0] = kBlock;
  if (s.inputs.empty() || s.inputs[0].rank() != 4) {
    tp.workgroup_count[0] = 1;
    return tp;
  }
  const Shape& q = s.inputs[0];
  const auto batch = static_cast<std::uint32_t>(q.dim(0));
  const auto seq = static_cast<std::uint32_t>(q.dim(1));
  const auto heads = static_cast<std::uint32_t>(q.dim(2));
  const auto D = static_cast<std::uint32_t>(q.dim(3));
  const std::uint32_t wave = wave_of(s.device);
  const std::uint32_t rows =
      write_state ? batch * heads * D : batch * seq * heads * D;
  const std::uint32_t threads = rows * wave;
  tp.workgroup_count[0] = threads == 0 ? 1u : (threads + kBlock - 1) / kBlock;
  return tp;
}

}  // namespace

struct GdnKernel final : KernelPrimitive<GdnKernel> {
  static constexpr std::string_view kName = "gdn_chunk_scan";
  static constexpr std::string_view kEntry = "lse_gdn";
  static constexpr std::string_view kSource = {};

  std::size_t arity() const noexcept override { return 6; }
  bool owns_indexing() const noexcept override { return true; }

  std::string emit_kernel(const KernelShapes& s) const override {
    if (s.inputs.size() < 6 || s.types.scalar == nullptr ||
        s.intrinsics == nullptr || s.inputs[0].rank() != 4) {
      return {};
    }
    const auto dim = s.inputs[0].dim(3);
    if (dim != 16 && dim != 32 && dim != 64 && dim != 128) return {};
    return emit_gdn(s, s.iattrs[0] != 0 ? GdnWrite::kState : GdnWrite::kOut);
  }

  Result<Shape> infer_shape(std::span<const Shape> in) const override {
    if (in.size() < 6) {
      return LSE_ERROR(kInvalidArgument, "gdn_chunk_scan needs 6 inputs");
    }
    return in[0];
  }
  DType infer_dtype(std::span<const DType> in) const override {
    return in.empty() ? DType::kF32 : in[0];
  }
  static ThreadPlan plan_impl(const KernelShapes& s) {
    return gdn_plan(s, s.iattrs[0] != 0);
  }
};
LSE_REGISTER_PRIMITIVE(GdnKernel);

struct GdnPairKernel final : KernelPrimitive<GdnPairKernel> {
  static constexpr std::string_view kName = "gdn_chunk_scan.pair";
  static constexpr std::string_view kEntry = "lse_gdn_pair";
  static constexpr std::string_view kSource = {};

  std::size_t arity() const noexcept override { return 7; }
  bool owns_indexing() const noexcept override { return true; }

  std::string emit_kernel(const KernelShapes& s) const override {
    if (s.inputs.size() < 7 || s.types.scalar == nullptr ||
        s.intrinsics == nullptr || s.inputs[0].rank() != 4) {
      return {};
    }
    const auto dim = s.inputs[0].dim(3);
    if (dim != 16 && dim != 32 && dim != 64 && dim != 128) return {};
    return emit_gdn(s, GdnWrite::kBoth);
  }

  Result<Shape> infer_shape(std::span<const Shape> in) const override {
    if (in.empty()) return LSE_ERROR(kInvalidArgument, "gdn pair needs q");
    return in[0];
  }
  DType infer_dtype(std::span<const DType> in) const override {
    return in.empty() ? DType::kF32 : in[0];
  }
  static ThreadPlan plan_impl(const KernelShapes& s) {
    return gdn_plan(s, true);
  }
};

bool is_gdn_node(const Node* n) {
  return n != nullptr && n->kind == OpKind::kGDNChunkScan;
}

bool same_gdn_inputs(const Node& a, const Node& b) {
  if (a.inputs.size() != 6 || b.inputs.size() != 6) return false;
  for (std::size_t i = 0; i < 6; ++i) {
    if (a.inputs[i].get() != b.inputs[i].get()) return false;
  }
  return true;
}

const graph::KernelPrimitiveBase* gdn_pair_kernel() {
  static const GdnPairKernel k;
  return &k;
}

LinkedBinding gdn_pair_bindings(const FusionGroup& group) {
  LinkedBinding b;
  const Node* o = nullptr;
  const Node* st = nullptr;
  for (const NodePtr& n : group.nodes) {
    if (is_gdn_node(n.get())) {
      if (n->iattrs[0] == 0) o = n.get();
      else st = n.get();
      continue;
    }
    if (dynamic_cast<const KernelPrimitiveBase*>(n->prim) != nullptr) {
      return b;
    }
  }
  if (o == nullptr || st == nullptr || !same_gdn_inputs(*o, *st)) return b;
  b.inputs = {o->inputs[0].get(), o->inputs[1].get(), o->inputs[2].get(),
              o->inputs[3].get(), o->inputs[4].get(), o->inputs[5].get(), st};
  b.sink = o;
  b.ok = true;
  return b;
}

}  // namespace lse::backend::hrx_kernels
