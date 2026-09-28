#include "lse/graph/kernel_args.hpp"
#include "lse/graph/kernel_env.hpp"
#include "lse/graph/kernel_primitive.hpp"
#include "lse/math.hpp"
namespace lse::kernels::splitk4_merge_probe {
using namespace graph;
namespace {
struct Args {
  env::In<kir::f32, env::Emit> partial;
  env::Out<kir::f32, env::Emit> out;
};
struct Merge final : KernelPrimitive<Merge> {
  static constexpr std::string_view kName = "quant_linear.splitk4.merge.v1";
  static constexpr std::string_view kEntry = "lse_quant_linear_splitk4_merge";
  static constexpr std::string_view kSource = {};
  std::size_t arity() const noexcept override { return 1; }
  bool owns_indexing() const noexcept override { return true; }
  DType infer_dtype(std::span<const DType>) const override { return DType::kF32; }
  Result<Shape> infer_shape(std::span<const Shape> in) const override {
    if (in.size()!=1 || in[0].rank()!=3 || in[0].dim(0)!=4)
      return LSE_ERROR(kInvalidArgument,"split-K merge requires [4,M,N]");
    return Shape{in[0].dim(1),in[0].dim(2)};
  }
  static bool valid(const KernelShapes& s) {
    return s.inputs.size()==1 && s.inputs[0].rank()==3 && s.inputs[0].dim(0)==4 &&
      s.input_dtypes.size()==1 && s.input_dtypes[0]==DType::kF32 &&
      s.output_dtype==DType::kF32 && s.output.elem_count()*4==s.inputs[0].elem_count();
  }
  std::string emit_kernel(const KernelShapes& s) const override {
    if(!valid(s)||!s.types.scalar||!s.intrinsics||!s.store)return {};
    const auto count=static_cast<std::uint32_t>(s.output.elem_count());
    kir::KernelBody kb(s.types,*s.intrinsics,0);kb.set_store(s.store);
    Args a;if(!env::bind(kb,a,s))return {};
    env::Emit e{&kb};const auto i=e.let(math::workgroup_id_x()*256u+math::local_id());
    if(auto live=e.when(i<count)) {
      auto sum=e.let(a.partial[i]);
      for(std::uint32_t split=1;split<4;++split)
        sum=e.let(sum+a.partial[e.let(split*count+i)]);
      e.store(i,sum);
    }
    return kb.str();
  }
  static ThreadPlan plan_impl(const KernelShapes& s) {
    ThreadPlan p;if(!valid(s))return p;
    p.workgroup_count[0]=static_cast<std::uint32_t>((s.output.elem_count()+255)/256);
    return p;
  }
};
}
const KernelPrimitiveBase* primitive(){static const Merge merge;return &merge;}
}
