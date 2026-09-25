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
  mutable std::vector<bool> writes;
  Result<EmittedKernel> emit(const FusionGroup& g,const backend::DeviceInfo& d) const override {
    ++attempts;
    auto result=real.emit(g,d);
    writes.clear();
    if (result.ok()) for (const auto& n : result->binding_order)
      writes.push_back(std::find(g.outputs.begin(), g.outputs.end(), n) != g.outputs.end());
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
  bool audit_ranges=false, overlapped=false;
  mutable CaptureEmitter emitter;
  mutable CaptureCompiler compiler;
  mutable KernelToolchain chain{Dialect::kLoom,&emitter,&compiler};
  backend::DeviceInfo info;
  backend::AmdDeviceInfo amd;
  CaptureDevice(){info.arch="gfx1201";info.compute_units=64;info.wavefront_size=32;info.max_threads_per_workgroup=1024;info.lds_bytes_per_workgroup=65536;backend::apply_arch_defaults(info,amd);info.extension_id=backend::AmdDeviceInfo::kExtensionId;info.extension=&amd;}
  const backend::DeviceInfo& device_info()const noexcept{return info;}
  std::span<const KernelToolchain>toolchains()const noexcept{return {&chain,1};}
  Result<backend::KernelHandle>load_executable(std::string_view name,std::span<const std::byte>){return backend::KernelHandle{1,0,std::string(name)};}
  Status launch(const backend::KernelHandle&, const backend::LaunchDims&,
                const backend::DispatchArgs& args, const backend::DispatchTarget&) {
    ++launches;
    if (audit_ranges) {
      if (emitter.writes.size() != args.bindings.size())
        return LSE_ERROR(kInternal, "capture binding metadata mismatch");
      for (std::size_t i = 0; i < args.bindings.size(); ++i) {
        const auto& a = args.bindings[i];
        const auto first = reinterpret_cast<std::uintptr_t>(a.buffer->ptr) +
                           a.buffer->offset + a.offset;
        for (std::size_t j = i + 1; j < args.bindings.size(); ++j) {
          if (!emitter.writes[i] && !emitter.writes[j]) continue;
          const auto& b = args.bindings[j];
          const auto second = reinterpret_cast<std::uintptr_t>(b.buffer->ptr) +
                              b.buffer->offset + b.offset;
          if (first < second + b.length && second < first + a.length)
            overlapped = true;
        }
      }
    }
    return OkStatus();
  }
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

LSE_TEST(final_launch_slots_keep_reduction_input_live_through_fused_epilogue) {
  for (int rows : {1, 17, 256}) for (int heads : {1, 16})
      for (bool reduction : {false, true}) {
    backend::BackendAdapter<CaptureDevice> backend;
    check(backend.init(0));
    backend.impl().audit_ranges = true;
    Scheduler scheduler(backend);
    scheduler.set_dialect(Dialect::kLoom);
    const Shape parent_shape{1, rows, 5 * heads * 128};
    auto buffer = backend.allocate(parent_shape.elem_count() * sizeof(float),
                                    backend::MemoryClass::kDevice,
                                    backend::kDefaultStream);
    LSE_EXPECT(buffer.ok());
    if (!buffer.ok()) continue;
    auto parent = Array::from_buffer(buffer.release(), parent_shape, DType::kF32);
    // Unlike a preallocated leaf, this activation is produced inside the
    // phase and therefore eligible for slot recycling. Nested reshapes must
    // extend its underlying allocation's lifetime, not merely a view node's.
    auto activation = silu(slice(parent, -1, 0, heads * 128));
    auto flat = reshape(activation, Shape{rows, heads * 128});
    auto view = reshape(flat, Shape{1, rows, heads, 128});
    auto norm = reduction ? l2_normalize(view, 1e-6f) : sigmoid(view);
    auto scaled = norm * Array::full(Shape{1}, DType::kF32, 0.0883883476f);
    auto out = repeat(scaled, 3, 2);
    const NodePtr roots[] = {out.node()};
    Program program;
    check(scheduler.eval(roots, false, &program));
    LSE_EXPECT_EQ(scheduler.last_trace().kernels_launched, 3u);
    LSE_EXPECT_EQ(scheduler.last_trace().host_groups, 0u);
    LSE_EXPECT(!backend.impl().overlapped);
    LSE_EXPECT(activation.node()->buffer.ptr != scaled.node()->buffer.ptr);
    LSE_EXPECT(scheduler.last_trace().slots_reused > 0);
    program.reset_compute();
    check(scheduler.eval(roots, false, &program));
    LSE_EXPECT(scheduler.last_trace().replayed);
    LSE_EXPECT_EQ(scheduler.last_trace().kernels_launched, 3u);
    LSE_EXPECT(!backend.impl().overlapped);
  }
}

LSE_TEST(device_retained_overwrite_replays_without_repartitioning) {
  backend::BackendAdapter<CaptureDevice> backend;
  check(backend.init(0));
  Scheduler scheduler(backend);
  scheduler.set_dialect(Dialect::kLoom);
  auto leaf = [&](Shape shape) {
    auto buffer = backend.allocate(shape.elem_count() * sizeof(float),
        backend::MemoryClass::kDevice, backend::kDefaultStream);
    if (!buffer.ok()) throw std::runtime_error(buffer.status().to_string());
    return Array::from_buffer(buffer.release(), shape, DType::kF32);
  };
  auto dst = leaf({1, 1, 4, 2});
  auto src = leaf({1, 1, 1, 2});
  auto pos = leaf({1});
  auto out = overwrite_slice(dst, src, 2, pos);
  const NodePtr roots[] = {out.node()};
  Program program;
  check(scheduler.eval(roots, false, &program));
  LSE_EXPECT_EQ(scheduler.last_trace().kernels_launched, 1u);
  LSE_EXPECT_EQ(scheduler.last_trace().host_groups, 0u);
  LSE_EXPECT(program.holds(roots));
  const auto attempts = backend.impl().emitter.attempts;
  program.reset_compute();
  check(scheduler.eval(roots, false, &program));
  LSE_EXPECT(scheduler.last_trace().replayed);
  LSE_EXPECT_EQ(scheduler.last_trace().partition_passes, 0u);
  LSE_EXPECT_EQ(scheduler.last_trace().kernels_launched, 1u);
  LSE_EXPECT_EQ(scheduler.last_trace().host_groups, 0u);
  LSE_EXPECT_EQ(backend.impl().launches, 2u);
  LSE_EXPECT_EQ(backend.impl().emitter.failures, 0u);
  // The real emitter may rebind the held group; no repartition is permitted.
  LSE_EXPECT(backend.impl().emitter.attempts >= attempts);
}

LSE_TEST(device_view_validation_precedes_partition_and_launch) {
  backend::BackendAdapter<CaptureDevice> backend;
  check(backend.init(0));
  Scheduler scheduler(backend);
  scheduler.set_dialect(Dialect::kLoom);
  auto allocation = backend.allocate(8 * sizeof(float),
      backend::MemoryClass::kDevice, backend::kDefaultStream);
  LSE_EXPECT(allocation.ok());
  if (!allocation.ok()) return;
  auto window = allocation.release();
  window.offset = 2 * sizeof(float);
  window.size_bytes = 4 * sizeof(float);
  auto base = Array::from_buffer(std::move(window), Shape{4}, DType::kF32);
  auto evaluate = [&](Array& view) {
    const NodePtr roots[] = {view.node()};
    return scheduler.eval(roots, false);
  };
  auto valid = reshape(base, Shape{2, 2});
  check(evaluate(valid));
  LSE_EXPECT_EQ(valid.node()->buffer.handle, base.node()->buffer.handle);
  LSE_EXPECT_EQ(valid.node()->buffer.offset, 2 * sizeof(float));
  LSE_EXPECT(valid.node()->buffer.storage == base.node()->buffer.storage);
  auto wrong_count = reshape(base, Shape{5});
  LSE_EXPECT(evaluate(wrong_count).code() == StatusCode::kInvalidArgument);
  LSE_EXPECT(!wrong_count.node()->materialized);
  auto wrong_dtype = reshape(base, Shape{2, 2});
  wrong_dtype.node()->dtype = DType::kF16;
  LSE_EXPECT(evaluate(wrong_dtype).code() == StatusCode::kInvalidArgument);
  auto missing_input = reshape(base, Shape{2, 2});
  missing_input.node()->inputs.clear();
  LSE_EXPECT(evaluate(missing_input).code() == StatusCode::kInvalidArgument);
  auto null_input = reshape(base, Shape{2, 2});
  null_input.node()->inputs[0].reset();
  LSE_EXPECT(evaluate(null_input).code() == StatusCode::kInvalidArgument);
  auto extra_input = reshape(base, Shape{2, 2});
  extra_input.node()->inputs.push_back(base.node());
  LSE_EXPECT(evaluate(extra_input).code() == StatusCode::kInvalidArgument);
  auto nested = reshape(wrong_count, Shape{1, 5});
  LSE_EXPECT(evaluate(nested).code() == StatusCode::kInvalidArgument);
  LSE_EXPECT(!nested.node()->materialized);

  auto retained_view = reshape(base, Shape{2, 2});
  Program retained;
  const NodePtr roots[] = {retained_view.node()};
  check(scheduler.eval(roots, false, &retained));
  LSE_EXPECT(retained.holds(roots));
  retained.reset_compute();
  LSE_EXPECT(!retained_view.node()->materialized);
  base.node()->buffer.size_bytes = 3 * sizeof(float);
  LSE_EXPECT(scheduler.eval(roots, false, &retained).code() ==
             StatusCode::kOutOfRange);
  LSE_EXPECT(!retained_view.node()->materialized);

  auto short_view = reshape(base, Shape{2, 2});
  base.node()->buffer.size_bytes = 3 * sizeof(float);
  LSE_EXPECT(evaluate(short_view).code() == StatusCode::kOutOfRange);
  LSE_EXPECT(!short_view.node()->materialized);
  LSE_EXPECT_EQ(backend.impl().launches, 0u);
}

LSE_TEST_MAIN()
