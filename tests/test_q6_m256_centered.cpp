// Host-only selection contracts for the isolated M256 centered-affine selection.
#include "harness.hpp"
#include "lse/backends/hrx/arch_database.hpp"
#include "lse/backends/hrx/hipc/hip_sources.hpp"
#include "lse/backends/hrx/loomc/loom_sources.hpp"
#include "lse/graph/kernel_primitive.hpp"
#include "lse/kernels/wmma.hpp"
#include <array>
#include <cstdlib>
using namespace lse;
using namespace lse::graph;
struct Fixture {
  backend::DeviceInfo device;
  backend::AmdDeviceInfo amd;
  std::array<Shape,4> inputs;
  std::array<DType,4> dtypes{DType::kF32,DType::kU32,DType::kBF16,DType::kBF16};
  DialectSourceTable intrinsics;
  KernelShapes shapes;
  Fixture(int m,int n,int k,bool loom):intrinsics(loom?backend::loom_sources():backend::hip_sources()) {
    device.arch="gfx1201";device.compute_units=64;device.wavefront_size=32;
    device.max_threads_per_workgroup=1024;device.lds_bytes_per_workgroup=65536;
    backend::apply_arch_defaults(device,amd);device.extension_id=backend::AmdDeviceInfo::kExtensionId;device.extension=&amd;
    inputs={Shape{m,k},Shape{n,k*6/32},Shape{n,k/64},Shape{n,k/64}};
    shapes.inputs=inputs;shapes.input_dtypes=dtypes;shapes.output=Shape{m,n};shapes.iattrs={6,64,0,0};shapes.device=&device;shapes.intrinsics=&intrinsics;
  }
};
LSE_TEST(m256_profile_preserves_other_shape_selection) {
  unsetenv("LSE_WMMA");
  for(bool loom:{false,true})for(int m:{1,17,64,128,256,512})for(auto [n,k]:{std::pair{17408,5120},std::pair{5120,17408}}){
    Fixture f(m,n,k,loom);auto p=kernels::wmma_q6_linear_for(f.shapes);
    LSE_EXPECT_EQ(p!=nullptr,m==64||m==256||m==512);if(!p)continue;
    LSE_EXPECT_EQ(p->name().find("centered_affine")!=std::string_view::npos,m==256);
    LSE_EXPECT_EQ(p->name().find("centered_affine_v1")!=std::string_view::npos,m==256);
    const auto plan=p->plan(f.shapes);LSE_EXPECT_EQ(plan.lds_bytes,m==256?17968u:16384u);
  }
}
LSE_TEST(m256_profile_rejects_unmeasured_and_ineligible_requests) {
  for(bool loom:{false,true})for(int reason=0;reason<12;++reason){
    Fixture f(256,reason==0?19:17408,reason==11?1088:5120,loom);
    if(reason==1)f.device.arch="gfx1200";
    if(reason==2)f.device.wavefront_size=64;
    if(reason==3)f.device.compute_units=32;
    if(reason==4)f.device.max_threads_per_workgroup=64;
    if(reason==5)f.device.lds_bytes_per_workgroup=16384;
    if(reason==6)f.shapes.staged={"caller",5120};
    if(reason==7)f.shapes.staged_quant.codes="caller";
    if(reason==8)f.dtypes[0]=DType::kBF16;
    if(reason==9)f.dtypes[2]=DType::kF32;
    if(reason==10)f.intrinsics={};
    LSE_EXPECT(kernels::wmma_q6_linear_for(f.shapes)==nullptr);
  }
  setenv("LSE_WMMA","0",1);Fixture f(256,17408,5120,true);LSE_EXPECT(kernels::wmma_q6_linear_for(f.shapes)==nullptr);unsetenv("LSE_WMMA");
}
LSE_TEST_MAIN()
