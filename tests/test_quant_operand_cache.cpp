#include "lse/backends/hrx/arch_database.hpp"
#include "lse/backends/hrx/hipc/hip_emitter.hpp"
#include "lse/backends/hrx/loomc/loom_emitter.hpp"
#include "lse/graph/ops.hpp"
#include "lse/graph/kernel_primitive.hpp"
#include <cstdio>
#include <stdexcept>
using namespace lse;
using namespace lse::graph;
namespace {
void require(bool x,const char* message){if(!x)throw std::runtime_error(message);}
struct NamedKernel final:KernelPrimitiveBase {
 const KernelPrimitiveBase* base;std::string_view label;
 NamedKernel(const KernelPrimitiveBase* p,std::string_view n):base(p),label(n){}
 std::string_view name()const noexcept override{return label;}
 std::string_view entry_name()const noexcept override{return label;}
 size_t arity()const noexcept override{return base->arity();}
 bool supports(Dialect d)const noexcept override{return base->supports(d);}
 void eval_cpu(std::span<const float*const> in,float*out,size_t count,const std::array<float,4>&attrs)const override{base->eval_cpu(in,out,count,attrs);}
 Result<Shape> infer_shape(std::span<const Shape> a)const override{return base->infer_shape(a);}
 DType infer_dtype(std::span<const DType> a)const override{return base->infer_dtype(a);}
 std::string emit_kernel(const KernelShapes&s)const override{return base->specialize(s)->emit_kernel(s);}
 ThreadPlan plan(const KernelShapes&s)const override{auto p=base->specialize(s)->plan(s);p.workgroup_size[0]=label=="mock_quant_a"?128:256;return p;}
 bool owns_indexing()const noexcept override{return true;}
};
struct SelectingKernel final:KernelPrimitiveBase {
 const KernelPrimitiveBase*base;NamedKernel a,b;bool second=false;
 explicit SelectingKernel(const KernelPrimitiveBase*p):base(p),a(p,"mock_quant_a"),b(p,"mock_quant_b"){}
 std::string_view name()const noexcept override{return "unchanged_abstract_quant";}
 std::string_view entry_name()const noexcept override{return "abstract_quant";}
 size_t arity()const noexcept override{return base->arity();}
 bool supports(Dialect d)const noexcept override{return base->supports(d);}
 void eval_cpu(std::span<const float*const> in,float*out,size_t count,const std::array<float,4>&attrs)const override{base->eval_cpu(in,out,count,attrs);}
 Result<Shape> infer_shape(std::span<const Shape>a)const override{return base->infer_shape(a);}
 DType infer_dtype(std::span<const DType>a)const override{return base->infer_dtype(a);}
 std::string emit_kernel(const KernelShapes&s)const override{return specialize(s)->emit_kernel(s);}
 ThreadPlan plan(const KernelShapes&s)const override{return specialize(s)->plan(s);}
 const KernelPrimitiveBase*specialize(const KernelShapes&)const override{return second?&b:&a;}
};
Array leaf(Shape shape,DType type){auto n=std::make_shared<Node>();n->shape=shape;n->dtype=type;return Array(n);}
}
int main(){try{
 auto out=quant_linear(leaf({32,64},DType::kF32),leaf({17,12},DType::kU32),
                       leaf({17,1},DType::kBF16),leaf({17,1},DType::kBF16),6,64);
 auto* original=dynamic_cast<const KernelPrimitiveBase*>(out.node()->prim);require(original,"primitive absent");
 SelectingKernel selector(original);out.node()->prim=&selector;
 const NodePtr roots[]{out.node()};FusionGroup group;
 for(auto& g:Partitioner::partition(roots))if(g.anchor==OpKind::kQuantMatMul)group=g;
 require(!group.nodes.empty(),"quant group absent");
 backend::DeviceInfo device;backend::AmdDeviceInfo amd;
 device.arch="gfx1201";device.wavefront_size=32;device.compute_units=64;
 device.lds_bytes_per_workgroup=65536;device.max_threads_per_workgroup=1024;
 backend::apply_arch_defaults(device,amd);device.extension_id=backend::AmdDeviceInfo::kExtensionId;device.extension=&amd;
 backend::HipEmitter hip;backend::LoomEmitter loom;
 for(IKernelEmitter* e:{static_cast<IKernelEmitter*>(&hip),static_cast<IKernelEmitter*>(&loom)}){
  selector.second=false;auto k1=e->cache_key(group,device);auto a=e->emit(group,device);require(a.ok(),"first emit failed");
  // Same graph/profile, changed winner. These wrappers use the real Q6 body;
  // this is a cache test, not FP8 execution or a performance measurement.
  selector.second=true;auto k2=e->cache_key(group,device);auto b=e->emit(group,device);require(b.ok(),"second emit failed");
  require(k1!=k2,"persistent keys alias different selected implementation");
  require(a->dims.workgroup_size[0]!=b->dims.workgroup_size[0],"emission cache returned prior implementation plan");
  auto again=e->emit(group,device);require(again.ok()&&again->source==b->source,"same winner unstable");
  require(e->cache_key(group,device)==k2,"same winner persistent key unstable");
  selector.second=false;auto back=e->emit(group,device);require(back.ok()&&back->source==a->source,"return to first winner missed identity");
 }
 std::puts("PASS HIP/Loom actual selected-implementation emission and persistent cache identities; no GPU");
}catch(const std::exception&e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}}
