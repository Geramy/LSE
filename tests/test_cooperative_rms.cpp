#include "harness.hpp"
#include "lse/backends/hrx/arch_database.hpp"
#include "lse/backends/hrx/hipc/hip_emitter.hpp"
#include "lse/backends/hrx/loomc/loom_emitter.hpp"
#include "lse/backends/hrx/loomc/loom_types.hpp"
#include "lse/graph/kernel_primitive.hpp"
#include "lse/graph/ops.hpp"
#include "lse/kernels/quant_operand_cache.hpp"
#include <array>
#include <bit>
#include <cmath>
#include <limits>
using namespace lse;
using namespace lse::graph;
namespace {
Array leaf(Shape shape,DType dtype=DType::kF32) {
 auto n=std::make_shared<Node>();n->shape=shape;n->dtype=dtype;n->materialized=true;return Array(n);
}
FusionGroup solo(const Array& a) {
 FusionGroup g;g.nodes={a.node()};g.outputs=g.nodes;g.inputs=a.node()->inputs;
 g.anchor=a.node()->kind;g.anchor_class=a.node()->fclass;return g;
}
struct Fixture {
 backend::DeviceInfo d;backend::AmdDeviceInfo amd;backend::HipEmitter hip;backend::LoomEmitter loom;
 Fixture(){d.arch="gfx1201";d.compute_units=64;d.wavefront_size=32;d.max_threads_per_workgroup=1024;d.lds_bytes_per_workgroup=65536;backend::apply_arch_defaults(d,amd);d.extension_id=backend::AmdDeviceInfo::kExtensionId;d.extension=&amd;}
 void check(const FusionGroup& g,bool cooperative) {
  auto h=hip.emit(g,d),l=loom.emit(g,d);LSE_EXPECT(h.ok());LSE_EXPECT(l.ok());
  if(!h.ok())std::fprintf(stderr,"HIP: %s\n",h.status().to_string().c_str());
  if(!l.ok())std::fprintf(stderr,"Loom: %s\n",l.status().to_string().c_str());
  if(!h.ok()||!l.ok())return;
  LSE_EXPECT((l->source.find("kernel.barrier<workgroup>")!=std::string::npos)==cooperative);
  LSE_EXPECT((h->source.find("__syncthreads")!=std::string::npos)==cooperative);
  if(cooperative){
   const auto& shape=g.nodes[0]->shape;auto rows=shape.elem_count()/static_cast<size_t>(shape.dim(shape.rank()-1));
   LSE_EXPECT_EQ(l->dims.workgroup_size[0],256u);LSE_EXPECT_EQ(l->dims.workgroup_count[0],rows);
   LSE_EXPECT_EQ(h->dims.workgroup_size[0],256u);LSE_EXPECT_EQ(h->dims.workgroup_count[0],rows);
   LSE_EXPECT_EQ(l->lds_bytes,1024u);
  }
 }
};
}
LSE_TEST(cooperative_rms_shared_selection_and_resource_limits) {
 Fixture f;
 for(auto dt:{DType::kF32,DType::kBF16,DType::kF16})for(int width:{32,127,128,129,256,513,1024,5120}) {
  f.check(solo(rms_norm(leaf({1,3,width}),leaf({width},dt),1e-6f,true)),true);
 }
 auto g=solo(rms_norm(leaf({1,1,5120}),leaf({5120},DType::kBF16),1e-6f,false));
 f.d.max_threads_per_workgroup=128;f.check(g,false);
 f.d.max_threads_per_workgroup=1024;f.d.lds_bytes_per_workgroup=512;f.check(g,false);
 f.d.lds_bytes_per_workgroup=65536;
 f.check(solo(rms_norm(leaf({3,17}),leaf({17}),1e-6f)),false);
}
LSE_TEST(cooperative_rms_epilogue_alias_and_multiple_output_fallback) {
 Fixture f;auto x=leaf({1,2,3,129});auto n=rms_norm(x,leaf({129},DType::kBF16),1e-6f,true);
 auto y=n*x;auto z=y+n;
 auto g=solo(n);g.nodes={n.node(),y.node(),z.node()};g.outputs={z.node()};f.check(g,true);
 g.outputs={n.node(),z.node()};f.check(g,false);
 // Repeated logical input aliases must retain the shared binding and gain index.
 auto shared=leaf({128});f.check(solo(rms_norm(shared,shared,1e-6f)),true);
 // A materialized transpose/view is consumed in its logical shape, with the
 // same seam as the scalar kernel. Scheduler tests cover materialization.
 auto t=transpose(leaf({1,3,2,129}),{0,2,1,3});t.node()->materialized=true;
 f.check(solo(rms_norm(t,leaf({129}),1e-6f)),true);
}
LSE_TEST(cooperative_rms_reduction_keeps_its_producer_dependency_cut) {
 Fixture f;auto x=leaf({1,2,3,129});auto pre=silu(x)+x;
 auto norm=rms_norm(pre,leaf({129},DType::kBF16),1e-6f,true);
 auto out=norm*leaf({1,2,3,129});const NodePtr roots[]={out.node()};
 bool found=false;
 for(const auto& group:Partitioner::partition(roots)) {
  if(group.anchor!=OpKind::kRMS)continue;
  found=true;
  for(const auto& n:group.nodes)LSE_EXPECT(n!=pre.node());
  bool bound=false;for(const auto& n:group.inputs)bound=bound||n==pre.node();
  LSE_EXPECT(bound);f.check(group,true);
 }
 LSE_EXPECT(found);
}
LSE_TEST(cooperative_rms_phase_uses_virtual_rows_instead_of_physical_blocks) {
 Fixture f;
 // Eight narrow rows previously reached the generic self-indexed phase path.
 // That path advances virtual i while the standalone RMS uses blockIdx.x,
 // so one phase workgroup repeatedly normalized the first physical row.
 auto x=leaf({1,1,8,64});auto gain=leaf({64},DType::kBF16);
 auto norm=rms_norm(x,gain,1e-6f,true);
 auto out=reshape(norm,{1,1,512})*leaf({1,1,512});
 const NodePtr roots[]={out.node()};
 const auto phases=Partitioner::phases(roots);
 LSE_EXPECT(!phases.empty());
 bool found=false;
 for(const auto& phase:phases) {
  auto g=Partitioner::phase_group(phase,roots);
  bool has_rms=false;for(const auto& n:g.nodes)has_rms=has_rms||n->kind==OpKind::kRMS;
  if(!has_rms)continue;
  found=true;LSE_EXPECT(g.is_phase);
  for(auto cus:{1u,64u}) {
   f.d.compute_units=static_cast<std::uint16_t>(cus);
   auto emitted=backend::HipEmitter::emit_phase(g,f.d);
   LSE_EXPECT(emitted.ok());
   if(emitted.ok()) {
    // The per-element RMS function consumes each virtual i supplied by the
    // phase. A standalone self-indexed body cannot provide this contract.
    LSE_EXPECT(emitted->source.find("__device__ float lse_rms_norm_")!=std::string::npos);
   }
   const auto key=f.hip.cache_key(g,f.d);
   f.d.lds_bytes_per_workgroup=512;
   LSE_EXPECT_EQ(f.hip.cache_key(g,f.d),key);
   f.d.lds_bytes_per_workgroup=65536;
  }
 }
 LSE_EXPECT(found);
}
LSE_TEST(cooperative_rms_implementation_identity_changes_cache_keys) {
 Fixture f;auto g=solo(rms_norm(leaf({1,5120}),leaf({5120},DType::kBF16),1e-6f));
 auto types=backend::loom_types();auto intr=backend::loom_sources();
 const auto cooperative=kernels::quant_operand_specialization_key(0,g,f.d,types,intr);
 f.d.lds_bytes_per_workgroup=512;
 const auto scalar=kernels::quant_operand_specialization_key(0,g,f.d,types,intr);
 LSE_EXPECT(cooperative!=scalar);
 f.d.lds_bytes_per_workgroup=65536;g.outputs.push_back(g.outputs[0]);
 LSE_EXPECT_EQ(kernels::quant_operand_specialization_key(0,g,f.d,types,intr),scalar);
}
LSE_TEST(cooperative_rms_parallel_association_and_index_coverage) {
 for(size_t d:{32u,127u,128u,129u,256u,513u,1024u,5120u})for(int pattern=0;pattern<5;++pattern){
  std::vector<float> x(d),gain(d);std::vector<unsigned> writes(d);
  std::array<float,256> parts{};double reference=0;
  for(size_t i=0;i<d;++i){
   float v=std::sin(float(i*17+3))*3.0f;
   if(pattern==1)v=0;
   if(pattern==2)v=(i==d/2?1000.0f:v*0.001f);
   if(pattern==3)v=std::ldexp(v,-30);
   if(pattern==4)v=std::ldexp(v,30);
   x[i]=v;reference+=double(v)*v;
   const float g=std::cos(float(i*3))*2.0f;
   auto bits=std::bit_cast<uint32_t>(g);bits+=0x7fffu+((bits>>16)&1u);
   gain[i]=std::bit_cast<float>(bits&0xffff0000u);
  }
  for(size_t lane=0;lane<256;++lane)for(size_t col=lane;col<d;col+=256)parts[lane]=std::fma(x[col],x[col],parts[lane]);
  for(size_t step=128;step;step/=2)for(size_t lane=0;lane<step;++lane)parts[lane]+=parts[lane+step];
  for(bool centered:{false,true}){
   const float scale=1/std::sqrt(parts[0]/float(d)+1e-6f);
   const double wantscale=1/std::sqrt(reference/double(d)+1e-6);
   for(size_t lane=0;lane<256;++lane)for(size_t col=lane;col<d;col+=256){
    const float w=gain[col]+(centered?1.0f:0.0f);
    const float actual=(x[col]*scale)*w;
    const double wanted=double(x[col])*wantscale*w;
    LSE_EXPECT(std::isfinite(actual));LSE_EXPECT(std::abs(double(actual)-wanted)<=1e-5+1e-5*std::abs(wanted));
    ++writes[col];
   }
  }
  for(auto count:writes)LSE_EXPECT_EQ(count,2u);
 }
}
LSE_TEST_MAIN()
