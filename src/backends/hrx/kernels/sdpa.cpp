#include "lse/graph/kernel_primitive.hpp"
#include "lse/math.hpp"

#include <string>

namespace lse::backend::hrx_kernels {

using namespace lse::graph;
namespace math = lse::math;

// One thread per output [B, Hq, Tq, Dv]. Recomputes the key dots for softmax;
// S is small on the generate path (prefill tokens or cached length).
struct SdpaKernel final : KernelPrimitive<SdpaKernel> {
  static constexpr std::string_view kName = "attention";
  static constexpr std::string_view kEntry = "lse_sdpa";
  static constexpr std::string_view kSource = {};

  std::size_t arity() const noexcept override { return 3; }

  std::string emit_kernel(const KernelShapes& s) const override {
    if ((s.inputs.size() != 3 && s.inputs.size() != 4) ||
        s.types.scalar == nullptr || s.intrinsics == nullptr ||
        s.inputs[0].rank() != 4 || s.inputs[1].rank() != 4 ||
        s.inputs[2].rank() != 4) {
      return {};
    }
    const Shape& q = s.inputs[0];
    const Shape& ksh = s.inputs[1];
    const Shape& vsh = s.inputs[2];
    const auto bsz = static_cast<std::uint32_t>(q.dim(0));
    const auto qh = static_cast<std::uint32_t>(q.dim(1));
    const auto tq = static_cast<std::uint32_t>(q.dim(2));
    const auto dh = static_cast<std::uint32_t>(q.dim(3));
    const auto kvh = static_cast<std::uint32_t>(ksh.dim(1));
    const auto ts = static_cast<std::uint32_t>(ksh.dim(2));
    const auto dv = static_cast<std::uint32_t>(vsh.dim(3));
    if (qh == 0 || kvh == 0 || qh % kvh != 0 || dh == 0 || ts == 0 || dv == 0) {
      return {};
    }
    const auto group = qh / kvh;
    const float scale = s.attrs[0];
    const int mask = s.iattrs[0];
    const auto window = static_cast<std::uint32_t>(s.iattrs[1]);
    const auto baked_off = static_cast<std::uint32_t>(s.iattrs[2]);
    const bool live_off = s.inputs.size() >= 4;
    (void)bsz;

    kir::KernelBody k(s.types, *s.intrinsics);
    const auto qin = k.input<kir::f32>(0);
    const auto kin = k.input<kir::f32>(1);
    const auto vin = k.input<kir::f32>(2);
    const auto i = k.thread_id();
    const auto d = k.let<kir::u32>("d", i % dv);
    const auto qi = k.let<kir::u32>("qi", (i / dv) % tq);
    const auto h = k.let<kir::u32>("h", (i / (dv * tq)) % qh);
    const auto b = k.let<kir::u32>("b", i / (dv * tq * qh));
    const auto kh = k.let<kir::u32>("kh", h / group);
    kir::Val<kir::u32> offset = k.constant<kir::u32>(baked_off);
    if (live_off) {
      const auto offb = k.input<kir::f32>(3);
      offset = k.let<kir::u32>(
          "off", kir::cast<kir::u32>(offb[k.constant<kir::u32>(0)].read()));
    }
    const auto abs_i = k.let<kir::u32>("ai", offset + qi);
    const auto used = k.let<kir::u32>("used", offset + tq);
    const auto hi = k.let<kir::u32>(
        "hi", select(used < k.constant<kir::u32>(ts), used,
                     k.constant<kir::u32>(ts)));

    const auto m = k.var<kir::f32>("m", math::neg_inf());
    k.loop("j0", k.constant<kir::u32>(0), live_off ? hi : k.constant<kir::u32>(ts), 1,
           [&](kir::Val<kir::u32> j) {
             auto score = k.var<kir::f32>("sc0", k.lit(0.0f));
             k.loop("dd0", k.constant<kir::u32>(0), k.constant<kir::u32>(dh), 1,
                    [&](kir::Val<kir::u32> dd) {
                      const auto qb = ((b * qh + h) * tq + qi) * dh + dd;
                      const auto kb = ((b * kvh + kh) * ts + j) * dh + dd;
                      score = math::fma(qin[qb].read(), kin[kb].read(),
                                        score.read());
                    });
             score = score.read() * scale;
             auto take = [&] {
               m = math::max(m.read(), score.read());
             };
             if (mask == 0) {
               take();
             } else if (mask == 1) {
               k.when(j <= abs_i, take);
             } else {
               k.when(j <= abs_i && (abs_i - j) < window, take);
             }
           });

    auto denom = k.var<kir::f32>("den", k.lit(0.0f));
    auto acc = k.var<kir::f32>("acc", k.lit(0.0f));
    k.loop("j1", k.constant<kir::u32>(0), live_off ? hi : k.constant<kir::u32>(ts), 1,
           [&](kir::Val<kir::u32> j) {
             auto score = k.var<kir::f32>("sc1", k.lit(0.0f));
             k.loop("dd1", k.constant<kir::u32>(0), k.constant<kir::u32>(dh), 1,
                    [&](kir::Val<kir::u32> dd) {
                      const auto qb = ((b * qh + h) * tq + qi) * dh + dd;
                      const auto kb = ((b * kvh + kh) * ts + j) * dh + dd;
                      score = math::fma(qin[qb].read(), kin[kb].read(),
                                        score.read());
                    });
             score = score.read() * scale;
             auto w = k.var<kir::f32>("w", k.lit(0.0f));
             auto apply = [&] {
               w = math::exp(score.read() - m.read());
             };
             if (mask == 0) {
               apply();
             } else if (mask == 1) {
               k.when(j <= abs_i, apply);
             } else {
               k.when(j <= abs_i && (abs_i - j) < window, apply);
             }
             denom = denom.read() + w.read();
             const auto vb = ((b * kvh + kh) * ts + j) * dv + d;
             acc = math::fma(w.read(), vin[vb].read(), acc.read());
           });
    k.ret(acc.read() / select(denom.read() == 0.0f, k.lit(1.0f), denom.read()));
    return k.str();
  }

  Result<Shape> infer_shape(std::span<const Shape> in) const override {
    if (in.size() != 3 && in.size() != 4) {
      return LSE_ERROR(kInvalidArgument, "sdpa takes 3 or 4 inputs");
    }
    return Shape{in[0].dim(0), in[0].dim(1), in[0].dim(2), in[2].dim(3)};
  }
  DType infer_dtype(std::span<const DType> in) const override {
    return in.empty() ? DType::kF32 : in[0];
  }
  static ThreadPlan plan_impl(const KernelShapes& s) {
    ThreadPlan tp;
    const std::uint32_t threads =
        s.device && s.device->max_threads_per_workgroup >= 256 ? 256u : 64u;
    const auto elems = static_cast<std::uint32_t>(s.output.elem_count());
    tp.workgroup_size[0] = threads;
    tp.workgroup_count[0] = elems == 0 ? 1u : (elems + threads - 1) / threads;
    return tp;
  }
};
LSE_REGISTER_PRIMITIVE(SdpaKernel);

}  // namespace lse::backend::hrx_kernels
