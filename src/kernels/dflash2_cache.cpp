#include "lse/graph/kernel_args.hpp"
#include "lse/graph/kernel_env.hpp"
#include "lse/graph/kernel_primitive.hpp"

#include <algorithm>
#include <cstring>

namespace lse::kernels {
using namespace lse::graph;
namespace {
// Inputs: the ring cache, a [1, heads, rows, dim] update, the ring position
// of its first row, and how many of its rows are real. A prompt pass is
// padded to its pass width; the padding rows past `valid` are never written,
// so one kernel serves every prompt length that pads to the same width.
Result<Shape> cache_shape(std::span<const Shape> in) {
  if (in.size() != 4 || in[0].rank() != 4 || in[1].rank() != 4 ||
      in[2].elem_count() != 1 || in[3].elem_count() != 1 ||
      in[0].dim(0) != 1 || in[1].dim(0) != 1 ||
      in[0].dim(1) != in[1].dim(1) || in[0].dim(3) != in[1].dim(3) ||
      in[1].dim(2) < 1 || in[1].dim(2) > in[0].dim(2) ||
      in[0].dim(1) < 1 || in[0].dim(3) < 1 ||
      in[0].elem_count() > UINT32_MAX || in[1].elem_count() > UINT32_MAX)
    return LSE_ERROR(kInvalidArgument, "invalid DFlash2 ring write geometry");
  return in[0];
}
template <class E>
struct CacheArgs {
  env::In<kir::f32,E> cache;
  env::In<kir::f32,E> update;
  env::In<kir::f32,E> first;
  env::In<kir::f32,E> valid;
  env::Out<kir::f32,E> out;
};
}
struct DFlash2CacheWriteKernel final : KernelPrimitive<DFlash2CacheWriteKernel> {
  static constexpr std::string_view kName = "dflash2.cache_write";
  static constexpr std::string_view kEntry = "lse_dflash2_cache_write";
  static constexpr std::string_view kSource = {};
  std::size_t arity() const noexcept override { return 4; }
  int inplace_input() const noexcept override { return 0; }
  bool owns_indexing() const noexcept override { return true; }
  bool supports_epilogue() const noexcept override { return false; }
  bool has_host_impl() const noexcept override { return true; }
  std::string emit_kernel(const KernelShapes& s) const override {
    auto shape = cache_shape(s.inputs);
    if (!shape.ok() || s.output != *shape || s.input_dtypes.size() != 4 ||
        s.output_dtype != DType::kF32 || !s.types.scalar || !s.intrinsics || !s.store) return {};
    for (auto type : s.input_dtypes) if (type != DType::kF32) return {};
    const auto heads = static_cast<std::uint32_t>(s.inputs[0].dim(1));
    const auto capacity = static_cast<std::uint32_t>(s.inputs[0].dim(2));
    const auto dim = static_cast<std::uint32_t>(s.inputs[0].dim(3));
    const auto rows = static_cast<std::uint32_t>(s.inputs[1].dim(2));
    if (s.attrs != std::array<float,4>{static_cast<float>(heads),static_cast<float>(capacity),
        static_cast<float>(dim),static_cast<float>(rows)}) return {};
    kir::KernelBody body(s.types,*s.intrinsics);
    body.set_store(s.store);
    CacheArgs<env::Emit> a;
    if (!env::bind(body,a,s)) return {};
    env::Emit e{&body};
    const auto at = e.let(e.thread_id());
    const auto first = e.let(kir::cast<kir::u32>(a.first[0u]) % capacity);
    const auto live = e.let(kir::cast<kir::u32>(a.valid[0u]));
    if (auto inside = e.when(at < static_cast<std::uint32_t>(s.inputs[1].elem_count()))) {
      const auto d = e.let(at % dim);
      const auto row = e.let((at / dim) % rows);
      const auto head = e.let(at / (dim * rows));
      if (auto real = e.when(row < live)) {
        const auto destination = e.let(((head * capacity + (first + row) % capacity) * dim) + d);
        e.store(destination,a.update[at]);
      }
    }
    return body.str();
  }
  void eval_cpu(std::span<const float* const> in,float* out,std::size_t count,
                const std::array<float,4>& attrs) const override {
    const auto heads = static_cast<std::size_t>(attrs[0]);
    const auto capacity = static_cast<std::size_t>(attrs[1]);
    const auto dim = static_cast<std::size_t>(attrs[2]);
    const auto rows = static_cast<std::size_t>(attrs[3]);
    const auto first = static_cast<std::size_t>(in[2][0]) % capacity;
    const auto live = std::min(rows, static_cast<std::size_t>(in[3][0]));
    std::memcpy(out,in[0],count*sizeof(float));
    for (std::size_t h=0;h<heads;++h)
      for (std::size_t r=0;r<live;++r)
        std::memcpy(out+(h*capacity+(first+r)%capacity)*dim,
                    in[1]+(h*rows+r)*dim,dim*sizeof(float));
  }
  Result<Shape> infer_shape(std::span<const Shape> in) const override { return cache_shape(in); }
  DType infer_dtype(std::span<const DType>) const override { return DType::kF32; }
  static ThreadPlan plan_impl(const KernelShapes& s) {
    ThreadPlan plan;
    const std::uint32_t threads = s.device && s.device->max_threads_per_workgroup >= 256 ? 256u : 1u;
    plan.workgroup_size[0] = threads;
    plan.workgroup_count[0] = s.inputs.size() > 1 ? static_cast<std::uint32_t>((s.inputs[1].elem_count()+threads-1)/threads) : 1u;
    return plan;
  }
};
LSE_REGISTER_PRIMITIVE(DFlash2CacheWriteKernel);
}  // namespace lse::kernels
