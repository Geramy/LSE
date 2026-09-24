#pragma once
#include "lse/backends/hrx/loomc/loom_print.hpp"
#include "lse/backends/hrx/loomc/loom_types.hpp"
#include "lse/backends/hrx/loomc/loom_sources.hpp"
#include "lse/backends/hrx/loomc/loom_emitter.hpp"
#include "lse/backends/hrx/arch_database.hpp"
#include "lse/graph/ops.hpp"
#include "lse/ir/env.hpp"
#include "lse/math.hpp"
namespace dot_fixture {
using namespace lse;
inline Result<backend::LoomBody> body(unsigned kind) {
  const auto types=backend::loom_types();
  const auto intrinsics=backend::loom_sources();
  ir::KernelBody recorded(types,intrinsics);
  ir::env::Emit e{&recorded};
  ir::Buffer<ir::i32> x(&recorded,&types,"x"), codes(&recorded,&types,"codes"), acc(&recorded,&types,"acc");
  ir::Buffer<ir::f32> values(&recorded,&types,"values");
  if(kind==0) {
    const auto at=e.thread_id();
    const auto signed_x=e.let(x[at].read());
    const auto bits=e.let(ir::cast<ir::u32>(signed_x));
    const auto signed_again=e.let(ir::cast<ir::i32>(bits));
    const auto dot=e.let(math::dot4_iu8(signed_again,codes[at].read(),acc[at].read()));
    e.ret(ir::cast<ir::f32>(dot));
  } else if(kind==1) {
    e.ret(math::rint(values[e.thread_id()].read()));
  } else {
    const auto index=e.let(math::workgroup_id_x());
    const auto lo=e.let(math::min(index,e.u32(2)));
    const auto hi=e.let(math::max(lo,e.u32(1)));
    e.ret(ir::cast<ir::f32>(hi));
  }
  backend::LoomPrintOptions opts;
  for(const auto* name:{"x","codes","acc"})opts.buffers.emplace(name,backend::LoomBufferView{ir::Scalar::kI32,128,std::string("%")+name+"_view"});
  opts.buffers.emplace("values",backend::LoomBufferView{ir::Scalar::kF32,128,"%values_view"});
  return backend::loom_print(recorded.ir(),opts);
}
inline std::string kernel(const backend::LoomBody& b) {
 return R"(kernel.def export("dot_fixture") @dot_fixture() {
 %one = index.constant 1 : index
 %groups = index.constant 4 : index
 %lanes = index.constant 32 : index
 kernel.launch.config workgroups(%groups, %one, %one) workgroup_size(%lanes, %one, %one) : index
} launch(%x: buffer, %codes: buffer, %acc: buffer, %values: buffer, %out: buffer) {
 %base = index.constant 0 : offset
 %x_view = buffer.view %x[%base] : buffer -> view<128xi32, #dense>
 %codes_view = buffer.view %codes[%base] : buffer -> view<128xi32, #dense>
 %acc_view = buffer.view %acc[%base] : buffer -> view<128xi32, #dense>
 %values_view = buffer.view %values[%base] : buffer -> view<128xf32, #dense>
 %out_view = buffer.view %out[%base] : buffer -> view<128xf32, #dense>
 %group = kernel.workgroup.id<x> : index
 %lane = kernel.workitem.id<x> : index
 %lanes = index.constant 32 : index
 %flat = index.madd %group, %lanes, %lane : index
 %i = index.assume %flat [range(%flat, 0, 127)] : index
)"+b.text+" view.store "+b.result+", %out_view[%i] : f32, view<128xf32, #dense>\n kernel.return\n}\n";
}
inline Result<graph::EmittedKernel> projection(const char* arch,int bits,bool disable_mixed=false) {
 using namespace graph;
 auto leaf=[](Shape shape,DType dtype){auto n=std::make_shared<Node>();n->shape=shape;n->dtype=dtype;return Array(n);};
 auto result=quant_linear(leaf({3,64},DType::kF32),leaf({17,64*bits/32},DType::kU32),leaf({17,1},DType::kBF16),leaf({17,1},DType::kBF16),bits,64);
 backend::DeviceInfo device;device.arch=arch;device.compute_units=64;device.max_threads_per_workgroup=1024;device.wavefront_size=32;device.lds_bytes_per_workgroup=65536;
 backend::AmdDeviceInfo amd;backend::apply_arch_defaults(device,amd);if(disable_mixed)amd.has_dot4_iu8=false;
 device.extension_id=backend::AmdDeviceInfo::kExtensionId;device.extension=&amd;
 backend::LoomEmitter emitter;const NodePtr roots[]={result.node()};
 for(const auto& group:Partitioner::partition(roots))if(group.anchor==OpKind::kQuantMatMul)return emitter.emit(group,device);
 return LSE_ERROR(kInternal,"projection group missing");
}
}
