#include "harness.hpp"
// Host-only scheduler audit. Launches are recorded, never executed.
#include "lse/backends/cpu/cpu_backend.hpp"
#include "lse/backends/hrx/loomc/loom_emitter.hpp"
#include "lse/graph/codegen.hpp"
#include "lse/graph/graph.hpp"
#include "lse/graph/ops.hpp"
#include "lse/graph/program.hpp"
#include <cstdio>
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
  CaptureDevice(){info.arch="gfx1201";info.compute_units=64;info.wavefront_size=32;info.max_threads_per_workgroup=1024;info.lds_bytes_per_workgroup=65536;}
  const backend::DeviceInfo& device_info()const noexcept{return info;}
  std::span<const KernelToolchain>toolchains()const noexcept{return {&chain,1};}
  Result<backend::KernelHandle>load_executable(std::string_view name,std::span<const std::byte>){return backend::KernelHandle{1,0,std::string(name)};}
  Status launch(const backend::KernelHandle&,const backend::LaunchDims&,const backend::DispatchArgs&,const backend::DispatchTarget&){++launches;return OkStatus();}
};
void check(Status s){if(!s.ok())throw std::runtime_error(s.to_string());}

LSE_TEST(scheduler_preserves_both_pair_outputs_and_replay) {
  for (bool separated : {false,true}) for (bool reverse : {false,true}) {
    backend::BackendAdapter<CaptureDevice> backend;check(backend.init(0));
    Scheduler scheduler(backend);scheduler.set_dialect(Dialect::kLoom);
    auto leaf=[&](Shape sh){auto b=backend.allocate(sh.elem_count()*4,backend::MemoryClass::kDevice,backend::kDefaultStream);if(!b.ok())throw std::runtime_error(b.status().to_string());return Array::from_buffer(std::move(*b),sh,DType::kF32);};
    auto q=leaf({1,1,48,128}),k=leaf({1,1,48,128}),v=leaf({1,1,48,128});
    auto a=leaf({1,1,48}),b=leaf({1,1,48}),s=leaf({1,48,128,128});Array next;
    auto out=gated_delta_step(q,k,v,a,b,s,&next);
    auto visible=separated?silu(out):out;
    const NodePtr roots[]{reverse?next.node():visible.node(),reverse?visible.node():next.node()};
    Program program;check(scheduler.eval(roots,false,&program));
    LSE_EXPECT_EQ(scheduler.last_trace().kernels_launched,separated?2u:1u);
    LSE_EXPECT_EQ(scheduler.last_trace().host_groups,0u);
    LSE_EXPECT_EQ(backend.impl().emitter.failures,0u);
    LSE_EXPECT_EQ(backend.impl().emitter.pairs,1u);
    LSE_EXPECT(out.node()->buffer.valid());LSE_EXPECT(next.node()->buffer.valid());
    LSE_EXPECT(out.node()->buffer.ptr!=next.node()->buffer.ptr);
    program.reset_compute();check(scheduler.eval(roots,false,&program));
    LSE_EXPECT(scheduler.last_trace().replayed);
    LSE_EXPECT_EQ(scheduler.last_trace().kernels_launched,separated?2u:1u);
    LSE_EXPECT_EQ(scheduler.last_trace().host_groups,0u);
  }
}
LSE_TEST_MAIN()
