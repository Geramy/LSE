#include "harness.hpp"
#include "lse/backends/cpu/cpu_backend.hpp"
#include "lse/graph/graph.hpp"
#include "lse/graph/ops.hpp"
#include "lse/graph/kernel_primitive.hpp"
#include <algorithm>
#include <stdexcept>
using namespace lse;using namespace lse::graph;
struct OwnerTestKernel final:KernelPrimitive<OwnerTestKernel>{
 static constexpr std::string_view kName="owner_test",kEntry="owner_test",kSource="";int input=-1;
 explicit OwnerTestKernel(int a=-1):input(a){}int inplace_input()const noexcept override{return input;}
 std::size_t arity()const noexcept override{return 1;}bool owns_indexing()const noexcept override{return true;}
 Result<Shape>infer_shape(std::span<const Shape> s)const override{return s[0];}DType infer_dtype(std::span<const DType>)const override{return DType::kF32;}
 static ThreadPlan plan_impl(const KernelShapes&){return {};}
};
NodePtr node(const Primitive*prim,Shape shape,std::vector<NodePtr> inputs){auto n=std::make_shared<Node>();n->kind=OpKind::kCustom;n->prim=prim;n->shape=shape;n->dtype=DType::kF32;n->fclass=FusionClass::kBarrier;n->inputs=std::move(inputs);for(auto&i:n->inputs)++i->consumer_count;return n;}
void scenario(bool nested,bool shorter,bool escape){
 backend::BackendAdapter<backend::CpuBackend>b;LSE_EXPECT(b.init(0).ok());OwnerTestKernel ordinary,alias(0);
 auto input=Array::full(Shape{1024},DType::kF32,0).node();auto stage=node(&ordinary,{1024},{input});std::vector<NodePtr>order{stage};NodePtr source=stage;
 if(nested){source=reshape(Array(source),{32,32}).node();order.push_back(source);}
 auto repair=node(&alias,shorter?Shape{512}:Shape{1024},{source});order.push_back(repair);NodePtr view=repair;
 if(nested){view=reshape(Array(view),shorter?Shape{16,32}:Shape{32,32}).node();order.push_back(view);view=node(&alias,view->shape,{view});order.push_back(view);}
 auto temporary=node(&ordinary,{1024},{input});auto after=node(&ordinary,{1024},{temporary});auto late=node(&ordinary,{1024},{view,after});order.insert(order.end(),{temporary,after,late});
 Workgroup wg;for(auto&n:order)LSE_EXPECT(wg.try_add(n));std::vector<FusionGroup>launches;for(auto&n:order){FusionGroup g;g.nodes={n};g.inputs=n->inputs;g.outputs={n};launches.push_back(g);}
 std::vector<NodePtr>roots{late};if(escape)roots.push_back(view);wg.plan_slots(roots,launches);LSE_EXPECT(wg.bind_slots(b).ok());
 // Mirror actual scheduler's topological inplace binding after views exist.
 for(auto&n:order)if(n->kind==OpKind::kReshape || (n->prim&&n->prim->inplace_input()>=0)){n->buffer=n->inputs[0]->buffer;if(n->kind==OpKind::kReshape)n->buffer.size_bytes=n->element_count()*4;}
 LSE_EXPECT(stage->buffer.valid());LSE_EXPECT(view->buffer.valid());
 LSE_EXPECT(stage->buffer.ptr==view->buffer.ptr);LSE_EXPECT(view->buffer.size_bytes>=view->element_count()*4);
 LSE_EXPECT(stage->buffer.ptr!=temporary->buffer.ptr);LSE_EXPECT(stage->buffer.ptr!=after->buffer.ptr);LSE_EXPECT(stage->buffer.ptr!=late->buffer.ptr);
 auto* data=static_cast<float*>(stage->buffer.ptr);std::fill(data,data+1024,3.25f);auto* scratch=static_cast<float*>(temporary->buffer.ptr);std::fill(scratch,scratch+1024,-11.f);
 auto* actual=static_cast<float*>(view->buffer.ptr);for(size_t i=0;i<view->element_count();++i)LSE_EXPECT(actual[i]==3.25f);
}
LSE_TEST(inplace_alias_keeps_temporary_owner_live){for(bool nested:{false,true})for(bool shorter:{false,true})for(bool escape:{false,true})scenario(nested,shorter,escape);}
LSE_TEST_MAIN()
