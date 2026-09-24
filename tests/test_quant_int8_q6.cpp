// Host contracts for explicit INT8 Q6 selection and packed operand formation.
#include "harness.hpp"
#include "lse/backends/hrx/arch_database.hpp"
#include "lse/backends/hrx/hipc/hip_emitter.hpp"
#include "lse/backends/hrx/hipc/hip_sources.hpp"
#include "lse/backends/hrx/hipc/hip_types.hpp"
#include "lse/backends/hrx/loomc/loom_emitter.hpp"
#include "lse/graph/graph.hpp"
#include "lse/graph/kernel_primitive.hpp"
#include "lse/graph/ops.hpp"
#include "lse/kernels/int8_policy.hpp"
#include "lse/quant/group_affine_codec.hpp"
#include <array>
namespace {
using namespace lse; using namespace lse::graph;
struct Arithmetic {
  unsigned u32(unsigned x) const { return x; }
  unsigned let(unsigned x) const { return x; }
};
struct Fixture {
  backend::DeviceInfo device;
  backend::AmdDeviceInfo amd;
  std::array<Shape,4> inputs;
  std::array<DType,4> dtypes{DType::kF32,DType::kU32,DType::kBF16,DType::kBF16};
  DialectSourceTable intrinsics=backend::hip_sources();
  KernelShapes s;
  Fixture(int m,int k,int n=36) {
    device.arch="gfx1201";device.wavefront_size=32;
    device.max_threads_per_workgroup=1024;device.lds_bytes_per_workgroup=65536;
    backend::apply_arch_defaults(device,amd);
    device.extension_id=backend::AmdDeviceInfo::kExtensionId;device.extension=&amd;
    inputs={Shape{m,k},Shape{n,k*6/32},Shape{n,k/64},Shape{n,k/64}};
    s.inputs=inputs;s.input_dtypes=dtypes;s.output=Shape{m,n};s.iattrs={6,64,0,0};
    s.device=&device;s.types=backend::hip_types();s.intrinsics=&intrinsics;
  }
};
Array leaf(Shape s,DType t) {
  auto n=std::make_shared<Node>();n->shape=s;n->dtype=t;n->materialized=true;return Array(n);
}
unsigned scratch(int k) {
  const auto chunks=unsigned(k/8),groups=unsigned(k/64);
  auto swz=[](unsigned x){return x+x/32+1;};
  return 2*(kir::Lds::align(swz(2*chunks)*4)+kir::Lds::align(swz(chunks)*4))+kir::Lds::align(groups*4)+kir::Lds::align(8*4);
}
}
LSE_TEST(q6_integer_operand_packing_exact_all_values_all_positions) {
  Arithmetic e;
  for(unsigned at=0;at<16;++at) for(unsigned value=0;value<64;++value) {
    std::array<unsigned,3> words{};
    std::array<unsigned,16> reference{};
    for(unsigned i=0;i<16;++i) {
      reference[i]=i==at?value:(i*19+value*7)%64;
      for(unsigned bit=0;bit<6;++bit)
        words[(i*6+bit)/32]|=((reference[i]>>bit)&1u)<<((i*6+bit)%32);
    }
    const auto planes=quant::dot4_q6_code_planes(e,words);
    for(unsigned h=0;h<2;++h) for(unsigned p=0;p<2;++p) for(unsigned b=0;b<4;++b)
      LSE_EXPECT_EQ((planes[h*2+p]>>(b*8))&255u,reference[h*8+b*2+p]);
  }
}
LSE_TEST(q6_integer_selection_uses_policy_capability_and_exact_resources) {
  const auto* p=dynamic_cast<const KernelPrimitiveBase*>(find_primitive("quant_linear"));
  LSE_EXPECT(p!=nullptr); if(!p)return;
  const bool enabled=kernels::activation_int8_enabled();
  for(int k:{64,576,5120,6144,17408,18432}) {
    Fixture f(1,k);const auto plan=p->plan(f.s);
    LSE_EXPECT_EQ(plan.workgroup_size[0],256u);
    LSE_EXPECT_EQ(plan.workgroup_count[0],2u);
    LSE_EXPECT_EQ(plan.workgroup_count[1],1u);
    LSE_EXPECT_EQ(plan.lds_bytes,enabled?scratch(k):(k<=16384?unsigned(k)*4:0u));
    LSE_EXPECT_EQ(p->staged_row(f.s).count,0u);
    auto y=quant_linear(leaf(f.inputs[0],f.dtypes[0]),leaf(f.inputs[1],f.dtypes[1]),
                       leaf(f.inputs[2],f.dtypes[2]),leaf(f.inputs[3],f.dtypes[3]),6,64);
    const NodePtr roots[]={y.node()};const auto groups=Partitioner::partition(roots);
    LSE_EXPECT_EQ(groups.size(),1u); if(groups.size()!=1)continue;
    auto hip=backend::HipEmitter{}.emit(groups[0],f.device);
    auto loom=backend::LoomEmitter{}.emit(groups[0],f.device);
    LSE_EXPECT(hip.ok());LSE_EXPECT(loom.ok());if(!hip.ok()||!loom.ok())continue;
    LSE_EXPECT_EQ(hip->lds_bytes,plan.lds_bytes);LSE_EXPECT_EQ(loom->lds_bytes,plan.lds_bytes);
    LSE_EXPECT_EQ(loom->source.find("vector.dot")!=std::string::npos,enabled);
    LSE_EXPECT_EQ(hip->source.find("__builtin_amdgcn_sudot4")!=std::string::npos,enabled);
    f.amd.has_dot4_iu8=false;
    LSE_EXPECT_EQ(p->plan(f.s).lds_bytes,k<=16384?unsigned(k)*4:0u);
  }
}
LSE_TEST(q6_integer_quad_refusals_preserve_canonical_fp32_geometry) {
  const auto* p=dynamic_cast<const KernelPrimitiveBase*>(find_primitive("quant_linear"));
  LSE_EXPECT(p!=nullptr);if(!p)return;
  for(int reason=0;reason<10;++reason) {
    Fixture f(1,5120);
    unsigned grid=5,lds=20480;
    if(reason==0) { f.amd.has_dot4_iu8=false;grid=2; }
    if(reason==1) f.device.arch="gfx1200";
    if(reason==2) { f.device.wavefront_size=64;grid=9; }
    if(reason==3) {
      f.s.iattrs[1]=128;
      f.inputs[2]=f.inputs[3]=Shape{36,40};
    }
    if(reason==4) f.dtypes[2]=f.dtypes[3]=DType::kF32;
    if(reason==5) {
      f.inputs[1]=Shape{35,960};f.inputs[2]=f.inputs[3]=Shape{35,80};
      f.s.output=Shape{1,35};
    }
    if(reason==6) { f.s.staged={"caller_panel",5120};lds=0; }
    if(reason==7) f.s.staged_quant.codes="caller_codes";
    if(reason==8) { f.device.lds_bytes_per_workgroup=4096;grid=2;lds=0; }
    if(reason==9) { f.s.intrinsics=nullptr;grid=2; }
    const auto plan=p->plan(f.s);
    LSE_EXPECT_EQ(plan.workgroup_count[0],grid);
    LSE_EXPECT_EQ(plan.lds_bytes,lds);
    if(reason!=6 && reason!=7 && reason!=9) {
      auto y=quant_linear(leaf(f.inputs[0],f.dtypes[0]),leaf(f.inputs[1],f.dtypes[1]),
                         leaf(f.inputs[2],f.dtypes[2]),leaf(f.inputs[3],f.dtypes[3]),
                         f.s.iattrs[0],f.s.iattrs[1]);
      const NodePtr roots[]={y.node()};const auto groups=Partitioner::partition(roots);
      LSE_EXPECT_EQ(groups.size(),1u);if(groups.size()!=1)continue;
      auto emitted=backend::LoomEmitter{}.emit(groups[0],f.device);
      LSE_EXPECT(emitted.ok());if(!emitted.ok())continue;
      LSE_EXPECT(emitted->source.find("vector.dot")==std::string::npos);
      LSE_EXPECT_EQ(emitted->dims.workgroup_count[0],grid);
      LSE_EXPECT_EQ(emitted->lds_bytes,lds);
    }
  }
  Fixture prefill(3,5120);
  LSE_EXPECT_EQ(p->plan(prefill.s).workgroup_count[0],5u);
  LSE_EXPECT_EQ(p->plan(prefill.s).workgroup_count[1],2u);
}
LSE_TEST(q6_integer_quad_rejects_indexed_matrices) {
  const auto* p=dynamic_cast<const KernelPrimitiveBase*>(find_primitive("quant_linear_indexed"));
  LSE_EXPECT(p!=nullptr);if(!p)return;
  Fixture f(1,5120);
  const Shape shapes[]={Shape{1,5120},Shape{2,36,960},Shape{2,36,80},Shape{2,36,80},Shape{1,1}};
  const DType types[]={DType::kF32,DType::kU32,DType::kBF16,DType::kBF16,DType::kF32};
  f.s.inputs=shapes;f.s.input_dtypes=types;f.s.iattrs={0,6,64,0};
  LSE_EXPECT_EQ(p->plan(f.s).workgroup_count[0],5u);
  LSE_EXPECT_EQ(p->plan(f.s).lds_bytes,20480u);
}
LSE_TEST_MAIN()
