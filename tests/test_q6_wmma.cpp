#include "harness.hpp"
#include "lse/backends/hrx/arch_database.hpp"
#include "lse/backends/hrx/hipc/hip_emitter.hpp"
#include "lse/backends/hrx/loomc/loom_emitter.hpp"
#include "lse/backends/hrx/loomc/loomc_compiler.hpp"
#include "lse/graph/ops.hpp"
#include "lse/kernels/quant_operand_policy.hpp"
#include <cstdlib>
using namespace lse;
namespace {
graph::FusionGroup make_group(int m,int n,int k) {
 using namespace graph;
 auto leaf=[](Shape shape,DType dt){auto p=std::make_shared<Node>();p->shape=shape;p->dtype=dt;return Array(p);};
 auto out=quant_linear(leaf({m,k},DType::kF32),leaf({n,k*6/32},DType::kU32),
   leaf({n,k/64},DType::kBF16),leaf({n,k/64},DType::kBF16),6,64);
 const NodePtr roots[]={out.node()};
 for(auto g:Partitioner::partition(roots))if(g.anchor==OpKind::kQuantMatMul)return g;
 std::abort();
}
struct Fixture {
 backend::DeviceInfo device;backend::AmdDeviceInfo amd;
 backend::HipEmitter hip;backend::LoomEmitter loom;backend::LoomcCompiler compiler;
 Fixture(){device.arch="gfx1201";device.wavefront_size=32;device.compute_units=64;
  device.max_threads_per_workgroup=1024;device.lds_bytes_per_workgroup=65536;
  backend::apply_arch_defaults(device,amd);device.extension_id=backend::AmdDeviceInfo::kExtensionId;device.extension=&amd;}
 void check(int m,int n,int k,bool matrix,bool native=true) {
  const auto group=make_group(m,n,k);
  unsetenv("LSE_Q6_WMMA");
  const auto hk=hip.cache_key(group,device),lk=loom.cache_key(group,device);
  auto h=hip.emit(group,device),l=loom.emit(group,device);
  LSE_EXPECT(h.ok());LSE_EXPECT(l.ok());if(!h.ok()||!l.ok())return;
  LSE_EXPECT((h->source.find("__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12")!=std::string::npos)==matrix);
  LSE_EXPECT((l->source.find("vector.mma")!=std::string::npos)==matrix);
  LSE_EXPECT(l->source.find("element_format=fp8")==std::string::npos);
  LSE_EXPECT(l->source.find("element_format=bf8")==std::string::npos);
  if(matrix){LSE_EXPECT(l->lds_bytes==16384);LSE_EXPECT(l->dims.workgroup_size[0]==128);LSE_EXPECT(l->source.find("element_format=bf16")!=std::string::npos);}
  // The removed switch cannot change either selection or persisted identity.
  for(const char* old:{"f16","bf16","f16-lds","bf16-lds","invalid"}) {
   setenv("LSE_Q6_WMMA",old,1);
   LSE_EXPECT_EQ(hip.cache_key(group,device),hk);LSE_EXPECT_EQ(loom.cache_key(group,device),lk);
   auto hh=hip.emit(group,device),ll=loom.emit(group,device);
   LSE_EXPECT(hh.ok());LSE_EXPECT(ll.ok());
   if(hh.ok())LSE_EXPECT(hh->source==h->source);
   if(ll.ok())LSE_EXPECT(ll->source==l->source);
  }
  unsetenv("LSE_Q6_WMMA");
  if(native){LSE_EXPECT(compiler.available());if(!compiler.available())return;
   auto code=compiler.compile(l->source,"gfx1201");LSE_EXPECT(code.ok());
   if(!code.ok())std::fprintf(stderr,"M%d N%d K%d: %s\n",m,n,k,code.status().to_string().c_str());
   else LSE_EXPECT(!code->code.empty());}
 }
};
}
LSE_TEST(q6_ranked_selection_uses_measured_bf16_winners_in_both_dialects) {
 unsetenv("LSE_WMMA");LSE_EXPECT(!kernels::kQuantOperandProfile.run_qualification_candidate);
 Fixture f;
 for(int m:{64,512}){f.check(m,17408,5120,true);f.check(m,5120,17408,true);}
}
LSE_TEST(q6_unknown_decode_and_unsupported_device_keep_scalar_without_old_override) {
 unsetenv("LSE_WMMA");Fixture f;
 f.check(17,19,1088,false);f.check(1,17408,5120,false);
 f.device.compute_units=32;f.check(64,17408,5120,false,false);
 f.device.compute_units=64;f.device.lds_bytes_per_workgroup=8192;f.check(64,17408,5120,false,false);
 f.device.lds_bytes_per_workgroup=65536;setenv("LSE_WMMA","0",1);f.check(64,17408,5120,false);
 unsetenv("LSE_WMMA");
}
LSE_TEST_MAIN()
