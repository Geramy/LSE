#include "lse/graph/kernel_args.hpp"
#include "lse/graph/kernel_env.hpp"
#include "lse/graph/kernel_primitive.hpp"
#include "lse/kv/block.hpp"

#include <string>

namespace lse::backend::hrx_kernels {

using namespace lse::graph;

template <class E>
struct RopeArgs {
  env::In<kir::f32, E> x;
  env::In<kir::f32, E> cos;
  env::In<kir::f32, E> sin;
  env::In<kir::f32, E> off;  // optional: present only on 4-input nodes
  env::Out<kir::f32, E> out;
};

// Interleaved pairs: (x0,x1) -> (x0 c - x1 s, x1 c + x0 s). cos/sin are
// [max_T, D]; position is offset + (row % seq).
struct RopeKernel final : KernelPrimitive<RopeKernel> {
  static constexpr std::string_view kName = "rope";
  static constexpr std::string_view kEntry = "lse_rope";
  static constexpr std::string_view kSource = {};

  std::size_t arity() const noexcept override { return 3; }

  std::string emit_kernel(const KernelShapes& s) const override {
    if ((s.inputs.size() != 3 && s.inputs.size() != 4) ||
        s.types.scalar == nullptr || s.intrinsics == nullptr ||
        s.inputs[0].rank() < 2) {
      return {};
    }
    const Shape& x = s.inputs[0];
    const auto dim = static_cast<std::uint32_t>(x.dim(x.rank() - 1));
    const auto seq = static_cast<std::uint32_t>(x.dim(x.rank() - 2));
    const auto baked_off = static_cast<std::uint32_t>(s.iattrs[0]);
    const bool live_off = s.inputs.size() >= 4;
    if (dim < 2 || seq == 0) return {};
    // Rows per batch entry: [B, H, T, D] flattens to B*H*T rows of D, so a
    // thread's batch entry is its row over H*T.
    const auto batch = static_cast<std::uint32_t>(x.dim(0));
    const auto rows_total = static_cast<std::uint32_t>(x.elem_count()) / dim;
    if (batch == 0 || rows_total % batch != 0) return {};
    const auto rows_per_batch = rows_total / batch;
    // A batched pass hands RoPE the step descriptor, not a lone offset: rows
    // sit at different absolute positions, and one shared angle would rotate
    // every row but one to somebody else's position. A 1-element input is the
    // single-sequence form and stays exactly that.
    const bool ragged =
        live_off && s.inputs[3].elem_count() >=
                        static_cast<std::size_t>(kv::step_meta_elems(
                            static_cast<std::int32_t>(batch)));

    kir::KernelBody k(s.types, *s.intrinsics);
    RopeArgs<env::Emit> a;
    if (!env::bind(k, a, s)) return {};
    env::Emit e{&k};
    const auto i = e.thread_id();
    const auto d = e.let(i % dim);
    const auto r = e.let(i / dim);
    kir::Val<kir::u32> offset = e.u32(baked_off);
    if (ragged) {
      const auto b = e.let(r / rows_per_batch);
      offset = e.let(kir::cast<kir::u32>(
          a.off[e.u32(static_cast<std::uint32_t>(kv::kStepMetaHeader)) +
                b * e.u32(static_cast<std::uint32_t>(kv::kStepMetaPerRow))]));
    } else if (live_off) {
      offset = e.let(kir::cast<kir::u32>(a.off[0u]));
    }
    const auto t = e.let(offset + (r % seq));
    const auto pair = e.let((d / 2u) * 2u);
    const auto base = e.let(r * dim + pair);
    const auto ang = e.let(t * dim + pair);
    const auto x0 = e.let(a.x[base]);
    const auto x1 = e.let(a.x[base + 1u]);
    const auto c = e.let(a.cos[ang]);
    const auto s0 = e.let(a.sin[ang]);
    e.ret(select((d % 2u) == 0u, x0 * c - x1 * s0, x1 * c + x0 * s0));
    return k.str();
  }

  Result<Shape> infer_shape(std::span<const Shape> in) const override {
    if (in.size() != 3 && in.size() != 4) {
      return LSE_ERROR(kInvalidArgument, "rope takes 3 or 4 inputs");
    }
    return in[0];
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
LSE_REGISTER_PRIMITIVE(RopeKernel);

}  // namespace lse::backend::hrx_kernels
