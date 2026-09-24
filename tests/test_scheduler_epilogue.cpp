#include <algorithm>

#include "harness.hpp"
// Host-only scheduler audit. Launches are recorded, never executed.
#include "lse/backends/cpu/cpu_backend.hpp"
#include "lse/backends/hrx/loomc/loom_emitter.hpp"
#include "lse/graph/codegen.hpp"
#include "lse/graph/graph.hpp"
#include "lse/graph/ops.hpp"
#include "lse/graph/program.hpp"
#include <cstdio>
#include "lse/backends/hrx/device_info.hpp"
#include "lse/backends/hrx/arch_database.hpp"
#include <map>
#include <unordered_set>
#include <cstdlib>
#include "lse/kv/block.hpp"
#include <stdexcept>
using namespace lse;
using namespace lse::graph;
struct CaptureEmitter final : IKernelEmitter {
  mutable backend::LoomEmitter real;
  mutable unsigned attempts=0, failures=0, pairs=0;
  Result<EmittedKernel> emit(const FusionGroup& g,const backend::DeviceInfo& d) const override {
    ++attempts;
    auto result=real.emit(g,d);
    std::printf("EMIT nodes=%zu outputs=%zu phase=%d ok=%d |",g.nodes.size(),g.outputs.size(),g.is_phase,result.ok());
    for(auto&n:g.nodes)std::printf(" %s",n->prim?std::string(n->prim->name()).c_str():"view");
    if(!result.ok()){++failures;std::printf(" => %s",result.status().to_string().c_str());}
    if(result.ok()&&g.nodes.size()==2&&g.nodes[0]->kind==OpKind::kGDNChunkScan)++pairs;
    std::puts("");return result;
  }
  Dialect dialect()const noexcept override{return Dialect::kLoom;}
  std::string_view prelude()const noexcept override{return {};}
  DialectSourceTable sources()const noexcept override{return real.sources();}
};
struct CaptureCompiler final : IKernelCompiler {
  Result<CompiledKernel> compile(std::string_view,std::string_view)const override{CompiledKernel k;k.code.push_back(std::byte{1});return k;}
  bool available()const override{return true;}
  std::string identity()const override{return "scheduler-capture-no-execution-v1";}
};
struct CaptureDevice:backend::CpuBackend {
  unsigned launches=0;
  mutable CaptureEmitter emitter;
  mutable CaptureCompiler compiler;
  mutable KernelToolchain chain{Dialect::kLoom,&emitter,&compiler};
  backend::DeviceInfo info;
  backend::AmdDeviceInfo amd;
  CaptureDevice(){info.arch="gfx1201";info.compute_units=64;info.wavefront_size=32;info.max_threads_per_workgroup=1024;info.lds_bytes_per_workgroup=65536;backend::apply_arch_defaults(info,amd);info.extension_id=backend::AmdDeviceInfo::kExtensionId;info.extension=&amd;}
  const backend::DeviceInfo& device_info()const noexcept{return info;}
  std::span<const KernelToolchain>toolchains()const noexcept{return {&chain,1};}
  Result<backend::KernelHandle>load_executable(std::string_view name,std::span<const std::byte>){return backend::KernelHandle{1,0,std::string(name)};}
  Status launch(const backend::KernelHandle&,const backend::LaunchDims&,const backend::DispatchArgs&,const backend::DispatchTarget&){++launches;return OkStatus();}
};
void check(Status s){if(!s.ok())throw std::runtime_error(s.to_string());}

LSE_TEST(actual_scheduler_epilogues_preserve_roots_fanout_and_replay) {
 for(int kind:{0,1,2}) for(int rows:{1,17,64}) for(int escapes:{0,1,2}) for(bool reverse:{false,true}) {
  backend::BackendAdapter<CaptureDevice> backend;check(backend.init(0));
  Scheduler scheduler(backend);scheduler.set_dialect(Dialect::kLoom);
  auto leaf=[&](Shape sh,DType dt=DType::kF32){auto b=backend.allocate(sh.elem_count()*4,backend::MemoryClass::kDevice,backend::kDefaultStream);if(!b.ok())throw std::runtime_error(b.status().to_string());return Array::from_buffer(std::move(*b),sh,dt);};
  Array producer;
  if(kind==0)producer=slice(leaf({rows,256}),-1,3,131);
  if(kind==1)producer=l2_normalize(leaf({rows,128}),1e-6f);
  if(kind==2)producer=quant_linear(leaf({rows,64}),leaf({17,12},DType::kU32),leaf({17,1},DType::kBF16),leaf({17,1},DType::kBF16),6,64);
  auto out=silu(producer);
  std::vector<NodePtr> roots{out.node()};Array other;
  if(escapes==1)roots.push_back(producer.node());
  if(escapes==2){other=neg(producer);roots.push_back(other.node());}
  if(reverse)std::reverse(roots.begin(),roots.end());
  Program program;check(scheduler.eval(roots,false,&program));
  const unsigned expected=escapes==0?1u:escapes==1?2u:3u;
  LSE_EXPECT_EQ(scheduler.last_trace().kernels_launched,expected);
  LSE_EXPECT_EQ(scheduler.last_trace().host_groups,0u);
  LSE_EXPECT_EQ(backend.impl().emitter.failures,0u);
  if(escapes)LSE_EXPECT(producer.node()->buffer.valid());
  program.reset_compute();check(scheduler.eval(roots,false,&program));
  LSE_EXPECT(scheduler.last_trace().replayed);
  LSE_EXPECT_EQ(scheduler.last_trace().kernels_launched,expected);
  LSE_EXPECT_EQ(scheduler.last_trace().host_groups,0u);
 }
}
LSE_TEST_MAIN()
