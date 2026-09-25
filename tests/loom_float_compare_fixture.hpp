#pragma once
#include "lse/backends/hrx/loomc/loom_print.hpp"
#include "lse/backends/hrx/loomc/loom_types.hpp"
#include "lse/graph/kernel_env.hpp"
namespace float_compare_fixture {
inline lse::Result<lse::backend::LoomBody> body(bool not_equal) {
  namespace ir=lse::ir;
  auto types=lse::backend::loom_types();
  const ir::DialectSourceTable intrinsics{std::span<const ir::PrimitiveSource>{}};
  ir::KernelBody recorded(types,intrinsics);ir::env::Emit e{&recorded};
  ir::Buffer<ir::f32> a(&recorded,&types,"a"),b(&recorded,&types,"b");
  auto i=e.thread_id();auto x=e.let(a[i].read()),y=e.let(b[i].read());
  auto comparison=not_equal ? x!=y : x==y;
  e.ret(ir::select(comparison,e.f32(1.0f),e.f32(0.0f)));
  lse::backend::LoomPrintOptions options;
  for(const char* name:{"a","b"})options.buffers.emplace(name,lse::backend::LoomBufferView{ir::Scalar::kF32,32,std::string("%")+name+"_view"});
  return lse::backend::loom_print(recorded.ir(),options);
}
inline std::string kernel(const lse::backend::LoomBody& b) {
  return R"(kernel.def export("float_compare") @float_compare() {
 %one = index.constant 1 : index
 %lanes = index.constant 32 : index
 kernel.launch.config workgroups(%one, %one, %one) workgroup_size(%lanes, %one, %one) : index
} launch(%a: buffer, %b: buffer, %out: buffer) {
 %base = index.constant 0 : offset
 %a_view = buffer.view %a[%base] : buffer -> view<32xf32, #dense>
 %b_view = buffer.view %b[%base] : buffer -> view<32xf32, #dense>
 %out_view = buffer.view %out[%base] : buffer -> view<32xf32, #dense>
 %lane = kernel.workitem.id<x> : index
 %i = index.assume %lane [range(%lane, 0, 31)] : index
)"+b.text+" view.store "+b.result+", %out_view[%i] : f32, view<32xf32, #dense>\n kernel.return\n}\n";
}
} // namespace float_compare_fixture
