#include "lse/graph/kernel_primitive.hpp"

#include <string>

#include "lse/graph/kernel_args.hpp"
#include "lse/graph/kernel_env.hpp"

namespace lse::kernels {

using namespace lse::graph;

// Concat takes 2-4 inputs; every possible slot is declared and the body only
// touches the first `parts` of them.
template <class E>
struct ConcatArgs {
  env::In<kir::f32, E> in0;
  env::In<kir::f32, E> in1;
  env::In<kir::f32, E> in2;
  env::In<kir::f32, E> in3;
  // Ret-style kernel: the element value is returned, not stored; the slot
  // exists to satisfy the binding contract.
  env::Out<kir::f32, E> out;
};

template <class E>
auto concat_element(E& e, ConcatArgs<E>& a, std::size_t parts,
                    const std::uint32_t (&lens)[4], std::uint32_t out_axis,
                    std::uint32_t inner) {
  const env::In<kir::f32, E>* src[4] = {&a.in0, &a.in1, &a.in2, &a.in3};
  auto i = e.thread_id();
  auto span = e.u32(out_axis * inner);
  auto o = e.let(i / span);
  auto rem = e.let(i % span);
  auto ax = e.let(rem / inner);
  auto ii = e.let(rem % inner);
  auto v = e.var(0.0f);
  std::uint32_t off = 0;
  for (std::size_t p = 0; p < parts; ++p) {
    auto idx = (o * lens[p] + (ax - off)) * inner + ii;
    if (auto owns = e.when(ax >= off && ax < off + lens[p])) {
      v = (*src[p])[idx];
    }
    off += lens[p];
  }
  return v;
}

// Pack along one axis. Split points are literals; the thread picks which
// input owns its output coordinate and copies that element.
struct ConcatKernel final : KernelPrimitive<ConcatKernel> {
  static constexpr std::string_view kName = "concat";
  static constexpr std::string_view kEntry = "lse_concat";
  static constexpr std::string_view kSource = {};

  std::size_t arity() const noexcept override { return 2; }

  std::string emit_kernel(const KernelShapes& s) const override {
    if (s.inputs.size() < 2 || s.inputs.size() > 4 ||
        s.types.scalar == nullptr || s.intrinsics == nullptr) {
      return {};
    }
    const auto axis = static_cast<std::size_t>(s.iattrs[0]);
    if (axis >= s.output.rank()) return {};

    std::uint32_t inner = 1;
    std::uint32_t out_axis = 1;
    for (std::size_t i = 0; i < s.output.rank(); ++i) {
      const auto d = static_cast<std::uint32_t>(s.output.dim(i));
      if (i > axis) inner *= d;
      else if (i == axis) out_axis = d;
    }
    if (inner == 0 || out_axis == 0) return {};

    std::uint32_t lens[4] = {};
    for (std::size_t p = 0; p < s.inputs.size(); ++p) {
      if (axis >= s.inputs[p].rank()) return {};
      lens[p] = static_cast<std::uint32_t>(s.inputs[p].dim(axis));
    }

    kir::KernelBody k(s.types, *s.intrinsics);
    ConcatArgs<env::Emit> a;
    if (!env::bind(k, a, s)) return {};
    env::Emit e{&k};
    const kir::Val<kir::f32> v =
        concat_element(e, a, s.inputs.size(), lens, out_axis, inner);
    e.ret(v);
    return k.str();
  }

  Result<Shape> infer_shape(std::span<const Shape> in) const override {
    if (in.size() < 2) return LSE_ERROR(kInvalidArgument, "concat needs 2+ inputs");
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

LSE_REGISTER_PRIMITIVE(ConcatKernel);

}  // namespace lse::kernels
