#include "harness.hpp"
#include "lse/graph/gdn_pair.hpp"
#include "lse/graph/ops.hpp"
#include "lse/backends/hrx/loomc/loom_emitter.hpp"
#include "lse/backends/hrx/hipc/hip_emitter.hpp"
#include <array>
#include <unordered_set>
using namespace lse;
using namespace lse::graph;
namespace {
Array leaf(Shape shape) {
  auto n=std::make_shared<Node>();n->shape=shape;n->dtype=DType::kF32;n->materialized=true;
  return Array(n);
}
struct Fixture {
  std::array<Array,6> inputs;
  Array output,state;
  Fixture(int dim=128,int seq=1) {
    const Shape q{1,seq,48,dim},scalar{1,seq,48},s{1,48,dim,dim};
    inputs={leaf(q),leaf(q),leaf(q),leaf(scalar),leaf(scalar),leaf(s)};
    output=gated_delta_step(inputs[0],inputs[1],inputs[2],inputs[3],inputs[4],inputs[5],&state);
  }
  std::vector<FusionGroup> groups(bool reverse=false) {
    const NodePtr roots[]{reverse?state.node():output.node(),reverse?output.node():state.node()};
    return Partitioner::partition(roots);
  }
};
backend::DeviceInfo device() {backend::DeviceInfo d;d.arch="gfx1201";d.wavefront_size=32;d.max_threads_per_workgroup=1024;return d;}
}
LSE_TEST(exact_pair_coalesces_both_root_orders_and_binds_separate_writes) {
  for (int dim:{16,32,64,128}) for(int seq:{1,6}) for(bool reverse:{false,true}) for(unsigned wave:{32u,64u}) {
    auto dev=device();dev.wavefront_size=wave;
    Fixture f(dim,seq);auto groups=f.groups(reverse);LSE_EXPECT_EQ(groups.size(),1u);
    if(groups.size()!=1)continue;
    LSE_EXPECT_EQ(groups[0].launches,1u);LSE_EXPECT_EQ(groups[0].outputs.size(),2u);
    auto emitted=backend::LoomEmitter{}.emit(groups[0],dev);
    if(!emitted.ok()){test::fail(__FILE__,__LINE__,emitted.status().to_string());continue;}
    LSE_EXPECT_EQ(emitted->binding_order.size(),8u);
    for(unsigned i=0;i<6;++i)LSE_EXPECT(emitted->binding_order[i]==f.inputs[i].node());
    LSE_EXPECT(emitted->binding_order[6]==f.state.node());
    LSE_EXPECT(emitted->binding_order[7]==f.output.node());
    LSE_EXPECT(emitted->source.find("%in6_view")!=std::string::npos);
    LSE_EXPECT(emitted->source.find("%out_view")!=std::string::npos);
    LSE_EXPECT(emitted->source.find("buffer.assume.noalias")==std::string::npos);
    LSE_EXPECT(emitted->source.find("buffer.alloca")==std::string::npos);
    auto hip=backend::HipEmitter{}.emit(groups[0],dev);LSE_EXPECT(hip.ok());
    if(hip.ok())LSE_EXPECT_EQ(hip->dims.workgroup_count[0],emitted->dims.workgroup_count[0]);
  }
}
LSE_TEST(single_roots_and_materialized_siblings_remain_standalone) {
  Fixture f;
  for(const NodePtr& root:{f.output.node(),f.state.node()}) {
    const NodePtr roots[]{root};auto groups=Partitioner::partition(roots);LSE_EXPECT_EQ(groups.size(),1u);
    auto e=backend::LoomEmitter{}.emit(groups[0],device());LSE_EXPECT(e.ok());
    if(e.ok())LSE_EXPECT_EQ(e->binding_order.size(),7u);
  }
  f.state.node()->materialized=true;auto groups=f.groups();LSE_EXPECT_EQ(groups.size(),1u);
  LSE_EXPECT_EQ(groups[0].nodes.size(),1u);
}
LSE_TEST(only_identical_inputs_shapes_types_modes_and_placement_match) {
  for(int reason=0;reason<9;++reason) {
    Fixture f;
    if(reason==0)f.state.node()->inputs[0]=leaf(f.inputs[0].shape()).node();
    if(reason==1)f.state.node()->member=3;
    if(reason==2)f.state.node()->iattrs[0]=0;
    if(reason==3)f.state.node()->shape=Shape{1,48,128,127};
    if(reason==4)f.inputs[2].node()->shape=Shape{1,1,48,127};
    if(reason==5)f.inputs[3].node()->dtype=DType::kBF16;
    if(reason==6)f.output.node()->dtype=DType::kBF16;
    if(reason==7)f.state.node()->attrs[0]=1.0f;
    if(reason==8)f.state.node()->iattrs[1]=1;
    LSE_EXPECT(!exact_gdn_pair(f.output.node(),f.state.node()));
  }
}
LSE_TEST(other_consumers_keep_original_nodes_and_topological_order) {
  Fixture f;
  auto out_consumer=f.output+Array::full(Shape{1},DType::kF32,0.5f);
  auto state_consumer=f.state*Array::full(Shape{1},DType::kF32,2.0f);
  const NodePtr roots[]{out_consumer.node(),state_consumer.node()};
  auto groups=Partitioner::partition(roots);std::unordered_set<const Node*> done;
  bool sawOut=false,sawState=false;
  for(const auto& group:groups) {
    for(const auto& in:group.inputs) if(!in->materialized && in->fclass!=FusionClass::kLeaf)LSE_EXPECT(done.contains(in.get()));
    for(const auto& n:group.nodes){done.insert(n.get());sawOut|=n==f.output.node();sawState|=n==f.state.node();}
    auto emitted=backend::LoomEmitter{}.emit(group,device());LSE_EXPECT(emitted.ok());
  }
  LSE_EXPECT(sawOut&&sawState);LSE_EXPECT(done.contains(out_consumer.node().get())&&done.contains(state_consumer.node().get()));
}
LSE_TEST(cached_pair_rebinds_new_graph_and_preserves_alias_arguments) {
  backend::LoomEmitter emitter;Fixture first;auto a=first.groups();auto x=emitter.emit(a[0],device());LSE_EXPECT(x.ok());
  Fixture next;auto b=next.groups(true);auto y=emitter.emit(b[0],device());LSE_EXPECT(y.ok());
  if(y.ok()) {LSE_EXPECT(y->binding_order[6]==next.state.node());LSE_EXPECT(y->binding_order[7]==next.output.node());}
  // A repeated q/k operand consumes one physical binding but still occupies
  // two logical arguments in the typed recorder.
  Fixture alias;alias.output.node()->inputs[1]=alias.inputs[0].node();alias.state.node()->inputs[1]=alias.inputs[0].node();
  auto groups=alias.groups();auto emitted=emitter.emit(groups[0],device());LSE_EXPECT(emitted.ok());
  if(emitted.ok())LSE_EXPECT_EQ(emitted->binding_order.size(),7u);
}
LSE_TEST_MAIN()
