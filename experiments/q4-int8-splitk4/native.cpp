#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <vector>
#include "lse/backend/backend.hpp"
#include "lse/backends/hrx/arch_database.hpp"
#include "lse/backends/hrx/loomc/loom_emitter.hpp"
#include "lse/backends/hrx/loomc/loomc_compiler.hpp"
#include "lse/backends/hrx/loomc/loom_types.hpp"
#include "lse/graph/ops.hpp"
namespace lse::kernels::panelvector_probe { const graph::KernelPrimitiveBase* consumer(); }
namespace lse::kernels::panel_probe { const graph::KernelPrimitiveBase* producer(); }
namespace lse::kernels::i8_splitk4_probe { const graph::KernelPrimitiveBase* primitive(const graph::KernelShapes&); }
namespace lse::kernels::splitk4_merge_probe { const graph::KernelPrimitiveBase* primitive(); }
using namespace lse;
using namespace lse::graph;
namespace {
constexpr std::size_t kGuard=64;
constexpr std::uint8_t kSentinel=0xa5;
struct Case {int bits,m,n,k;const char*label;bool realistic=false;};
struct Allocation {NodePtr node;backend::DeviceBuffer buffer;std::vector<std::uint8_t> bytes;};
struct Arm {std::uint64_t cache_key=0;EmittedKernel e;backend::KernelHandle handle;NodePtr output;std::vector<backend::BufferRef> refs;std::array<std::byte,4> constants{};};
Array leaf(Shape s,DType t){auto n=std::make_shared<Node>();n->shape=s;n->dtype=t;n->materialized=true;return Array(n);}
bool check(Status s,const char*label){if(s.ok())return true;std::fprintf(stderr,"%s: %s\n",label,s.message().c_str());return false;}
int activation(int row,int group,int t) {
  if(group%3==1)return 0;
  if(t%8==7)return (t/8)%2==0 ? 127:-127;
  return ((row%16)*11+(group%8)*7+t*13)%31-15;
}
int weight(int col,int group,int t,int bits){return ((col%16)*43+(group%8)*67+t*17)%(bits==4?16:256);}
int bias_units(int col){return col%2==0 ? -4:4;}
float input_value(const Case& c, int row, int at) {
  if (!c.realistic) return static_cast<float>(activation(row,at/64,at%64))/128.0f;
  if ((at/8)%13 == 2) return 0;
  const float a = std::sin(static_cast<float>((row+1)*19+at*7)*0.071f);
  const float b = std::cos(static_cast<float>((row+3)*11+at*3)*0.037f);
  return (a*0.6137f + b*0.1923f) * (0.25f + static_cast<float>(at%7)*0.1131f);
}
std::vector<std::uint32_t> panel_oracle(const Case& c) {
  const int chunks=c.k/8,groups=c.k/64,stride=c.k*25/64;
  std::vector<std::uint32_t> out(static_cast<std::size_t>(c.m*stride));
  for(int r=0;r<c.m;++r) {
    std::vector<float> sums(static_cast<std::size_t>(chunks));
    for(int chunk=0;chunk<chunks;++chunk) {
      float max=0,sum=0;std::array<float,8> v;
      for(int j=0;j<8;++j) {
        v[static_cast<std::size_t>(j)]=input_value(c,r,chunk*8+j);
        max=std::max(max,std::fabs(v[static_cast<std::size_t>(j)]));
        sum=j==0?v[0]:sum+v[static_cast<std::size_t>(j)];
      }
      sums[static_cast<std::size_t>(chunk)]=sum;
      const float step=max*(1.0f/127.0f),inv=127.0f/std::max(max,1e-30f);
      std::array<std::uint32_t,2> words{};
      for(int j=0;j<8;++j) {
        const auto code=static_cast<std::int32_t>(std::rint(v[static_cast<std::size_t>(j)]*inv));
        words[static_cast<std::size_t>(j%2)] |= (static_cast<std::uint32_t>(code)&255u) << (8*(j/2));
      }
      const auto base=static_cast<std::size_t>(r*stride);
      out[base+static_cast<std::size_t>(chunk*2)]=words[0];
      out[base+static_cast<std::size_t>(chunk*2+1)]=words[1];
      out[base+static_cast<std::size_t>(c.k/4+chunk)]=std::bit_cast<std::uint32_t>(step);
    }
    for(int g=0;g<groups;++g) {
      std::array<float,8> tree;
      for(int u=0;u<8;++u) tree[static_cast<std::size_t>(u)]=sums[static_cast<std::size_t>(g*8+u)];
      for(int bit=1;bit<8;bit*=2) {
        const auto prior=tree;
        for(int u=0;u<8;++u) tree[static_cast<std::size_t>(u)]=prior[static_cast<std::size_t>(u)]+prior[static_cast<std::size_t>(u^bit)];
      }
      out[static_cast<std::size_t>(r*stride+c.k/4+chunks+g)]=std::bit_cast<std::uint32_t>(tree[0]);
    }
  }
  return out;
}
std::vector<float> oracle(const Case&c) {
  const auto panel=panel_oracle(c);
  const int stride=c.k*25/64;
  std::vector<float> out(static_cast<std::size_t>(c.m*16));
  for(int r=0;r<c.m;++r)for(int col=0;col<16;++col) {
    double want=0;
    for(int g=0;g<c.k/64;++g) {
      double dot=0;
      for(int chunk=0;chunk<8;++chunk) {
        const auto base=static_cast<std::size_t>(r*stride);
        const auto step=std::bit_cast<float>(panel[base+static_cast<std::size_t>(c.k/4+g*8+chunk)]);
        for(int j=0;j<8;++j) {
          const auto packed=panel[base+static_cast<std::size_t>((g*8+chunk)*2+j%2)];
          const auto byte=static_cast<std::uint8_t>(packed>>(8*(j/2)));
          const int a=static_cast<std::int8_t>(byte);
          dot+=static_cast<double>(a)*weight(col,g,chunk*8+j,c.bits)*static_cast<double>(step);
        }
      }
      const float sum=std::bit_cast<float>(panel[static_cast<std::size_t>(r*stride+3*c.k/8+g)]);
      want+=dot/1024.0+static_cast<double>(sum)*bias_units(col)/1024.0;
    }
    out[static_cast<std::size_t>(r*16+col)]=static_cast<float>(want);
  }
  return out;
}
std::vector<float> group64_oracle(const Case& c, int partition=-1) {
  std::vector<float> out(static_cast<std::size_t>(c.m*16));
  for(int r=0;r<c.m;++r) {
    std::vector<std::array<int,64>> codes(static_cast<std::size_t>(c.k/64));
    std::vector<float> steps(static_cast<std::size_t>(c.k/64)),sums(steps.size());
    for(int g=0;g<c.k/64;++g) {
      std::array<float,4> partials{},maxima{};
      std::array<float,64> raw;
      for(int slice=0;slice<4;++slice) {
        for(int j=0;j<16;++j) {
          const float v=input_value(c,r,g*64+slice*16+j);
          raw[static_cast<std::size_t>(slice*16+j)]=v;
          maxima[static_cast<std::size_t>(slice)]=std::max(maxima[static_cast<std::size_t>(slice)],std::fabs(v));
          partials[static_cast<std::size_t>(slice)]=j==0?v:partials[static_cast<std::size_t>(slice)]+v;
        }
      }
      const float max=*std::max_element(maxima.begin(),maxima.end());
      float sum=partials[0];for(int slice=1;slice<4;++slice)sum+=partials[static_cast<std::size_t>(slice)];
      const float inv=127.0f/std::max(max,1e-30f);
      steps[static_cast<std::size_t>(g)]=max*(1.0f/127.0f);sums[static_cast<std::size_t>(g)]=sum;
      for(int j=0;j<64;++j)codes[static_cast<std::size_t>(g)][static_cast<std::size_t>(j)]=static_cast<int>(std::rint(raw[static_cast<std::size_t>(j)]*inv));
    }
    for(int col=0;col<16;++col) {
      double sum=0;
      const int span=(c.k/64+3)/4;
      const int first=partition<0?0:partition*span;
      const int last=partition<0?c.k/64:std::min(first+span,c.k/64);
      for(int g=first;g<last;++g) {
        std::int32_t exact=0;
        for(int j=0;j<64;++j)exact+=codes[static_cast<std::size_t>(g)][static_cast<std::size_t>(j)]*weight(col,g,j,c.bits);
        sum+=static_cast<double>(steps[static_cast<std::size_t>(g)])*exact/1024.0+
             static_cast<double>(sums[static_cast<std::size_t>(g)])*bias_units(col)/1024.0;
      }
      out[static_cast<std::size_t>(r*16+col)]=static_cast<float>(sum);
    }
  }
  return out;
}
int run_case(const Case&c,backend::IBackend*be,const backend::DeviceInfo&device,const std::string&prefix,int count) {
  const auto expected=oracle(c),matrix_expected=group64_oracle(c);if(expected.size()!=static_cast<std::size_t>(c.m*16))return 3;
  const auto x=leaf({c.m,c.k},DType::kF32),w=leaf({c.n,c.k*c.bits/32},DType::kU32);
  const auto sc=leaf({c.n,c.k/64},DType::kBF16),bi=leaf(sc.shape(),DType::kBF16);
  const int padded_n=(c.n+15)/16*16;
  const auto wp=leaf({padded_n,c.k/8},DType::kU32),sp=leaf({padded_n,c.k/64},DType::kBF16),bp=leaf(sp.shape(),DType::kBF16);
  const auto panel=leaf({c.m,c.k*25/64},DType::kU32);
  const auto partial=leaf({4,c.m,c.n},DType::kF32);
  std::vector<float> partial_expected;
  for(int split=0;split<4;++split) {
    const auto values=group64_oracle(c,split);
    partial_expected.insert(partial_expected.end(),values.begin(),values.end());
  }
  std::vector<Allocation> allocations;allocations.reserve(16);
  double packing_ms=0;
  std::size_t packing_bytes=0;
  const auto find=[&](const NodePtr&n)->Allocation*{for(auto&a:allocations)if(a.node==n)return &a;return nullptr;};
  auto ensure=[&](const NodePtr&n)->bool {
    if(find(n))return true;
    Allocation a;a.node=n;
    if(be) {
      const auto bytes=n->element_count()*dtype_storage_bytes(n->dtype,1);
      a.bytes.resize(bytes+2*kGuard,kSentinel);
      auto got=be->allocate(a.bytes.size(),backend::MemoryClass::kDevice);
      if(!got.ok())return false;a.buffer=got.release();
      auto*content=a.bytes.data()+kGuard;
      if(n==x.node()) {
        auto*p=reinterpret_cast<float*>(content);
        for(int r=0;r<c.m;++r)for(int k=0;k<c.k;++k)p[static_cast<std::size_t>(r*c.k+k)]=input_value(c,r,k);
      } else if(n==w.node()) {
        auto*p=reinterpret_cast<std::uint32_t*>(content);const int codes=32/c.bits;
        for(int col=0;col<c.n;++col)for(int k=0;k<c.k;k+=codes) {
          std::uint32_t packed=0;
          for(int u=0;u<codes;++u)packed |= static_cast<std::uint32_t>(weight(col,(k+u)/64,(k+u)%64,c.bits))<<(c.bits*u);
          p[static_cast<std::size_t>(col*(c.k/codes)+k/codes)]=packed;
        }
      } else if(n==sc.node()||n==bi.node()) {
        auto*p=reinterpret_cast<std::uint16_t*>(content);
        for(int col=0;col<c.n;++col)for(int g=0;g<c.k/64;++g) {
          const float v=n==sc.node()?1.0f/1024.0f:static_cast<float>(bias_units(col))/1024.0f;
          p[static_cast<std::size_t>(col*(c.k/64)+g)]=static_cast<std::uint16_t>(std::bit_cast<std::uint32_t>(v)>>16);
        }
      } else if(n==wp.node()||n==sp.node()||n==bp.node()) {
        const auto started=std::chrono::steady_clock::now();
        const auto source=n==wp.node()?w.node():n==sp.node()?sc.node():bi.node();
        const auto*original=find(source);
        if(!original)return false;
        const int width=n==wp.node()?c.k/8:c.k/64;
        for(int col=0;col<padded_n;++col)for(int at=0;at<width;++at) {
          const auto dst=static_cast<std::size_t>((col/16)*width*16+at*16+col%16);
          const auto src=static_cast<std::size_t>((col<c.n?col:0)*width+at);
          if(n==wp.node()) {
            const auto*raw=reinterpret_cast<const std::uint32_t*>(original->bytes.data()+kGuard);
            reinterpret_cast<std::uint32_t*>(content)[dst]=raw[src];
          } else {
            const auto*raw=reinterpret_cast<const std::uint16_t*>(original->bytes.data()+kGuard);
            reinterpret_cast<std::uint16_t*>(content)[dst]=raw[src];
          }
        }
        packing_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
        packing_bytes+=n->element_count()*dtype_storage_bytes(n->dtype,1);
      } else {
        auto*p=reinterpret_cast<float*>(content);
        std::fill_n(p,n->element_count(),std::numeric_limits<float>::quiet_NaN());
      }
      if(!check(be->copy_h2d(a.bytes.data(),a.buffer,a.bytes.size(),0),"upload"))return false;
    }
    allocations.push_back(std::move(a));return true;
  };
  std::array<Arm,4> arms;
  backend::LoomEmitter emitter;backend::LoomcCompiler compiler;
  for(unsigned which=0;which<4;++which) {
    auto&arm=arms[which];
    if(which==0) {
      auto y=quant_linear(x,w,sc,bi,c.bits,64);arm.output=y.node();
      arm.output->prim=kernels::panelvector_probe::consumer();
      arm.output->inputs.push_back(panel.node());
      panel.node()->materialized=true;
    } else if(which==1) {
      arm.output=partial.node();arm.output->set_kind(OpKind::kQuantMatMul);
      arm.output->inputs={x.node(),wp.node(),sp.node(),bp.node()};
      arm.output->iattrs={c.bits,64,0,0};
      const std::array shapes{x.shape(),wp.shape(),sp.shape(),bp.shape()};
      const std::array dtypes{DType::kF32,DType::kU32,DType::kBF16,DType::kBF16};
      auto intrinsics=backend::loom_sources();KernelShapes shapes_info;
      shapes_info.inputs=shapes;shapes_info.input_dtypes=dtypes;
      shapes_info.output=partial.shape();shapes_info.iattrs={c.bits,64,0,0};
      shapes_info.device=&device;shapes_info.types=backend::loom_types();shapes_info.intrinsics=&intrinsics;
      arm.output->prim=kernels::i8_splitk4_probe::primitive(shapes_info);
      if(!arm.output->prim)return 4;
      if(be && (!ensure(w.node())||!ensure(sc.node())||!ensure(bi.node())))return 5;
    } else if(which==2) {
      arm.output=leaf({1,c.m,c.n},DType::kF32).node();
      arm.output->set_kind(OpKind::kCustom);arm.output->prim=kernels::splitk4_merge_probe::primitive();
      arm.output->inputs={partial.node()};partial.node()->materialized=true;
    } else {
      arm.output=panel.node();arm.output->set_kind(OpKind::kCustom);
      arm.output->prim=kernels::panel_probe::producer();arm.output->inputs={x.node()};
    }
    arm.output->materialized=false;
    const NodePtr roots[]{arm.output};auto groups=Partitioner::partition(roots);if(groups.size()!=1)return 4;
    arm.cache_key=emitter.cache_key(groups.front(),device);
    if(which && (arm.cache_key==arms[0].cache_key||(which>1&&arm.cache_key==arms[1].cache_key))){std::fprintf(stderr,"cache identity collision\n");return 4;}
    auto emitted=emitter.emit(groups.front(),device);if(!emitted.ok()){std::fprintf(stderr,"emit: %s\n",emitted.status().message().c_str());return 4;}arm.e=emitted.release();
    const auto stem=prefix+"-"+c.label+"-"+(which==3?"prep":which==2?"merge":which==1?"partial":"panel");
    std::ofstream(stem+".loom")<<arm.e.source;
    auto code=compiler.compile(arm.e.source,std::string(device.arch));
    if(!code.ok()){std::fprintf(stderr,"compile: %s\n",code.status().message().c_str());return 5;}
    std::ofstream co(stem+".co",std::ios::binary);co.write(reinterpret_cast<const char*>(code->code.data()),static_cast<std::streamsize>(code->code.size()));
    std::printf("prepared case=%s bits=%d M=%d N=%d K=%d arm=%s grid=%uxY%u WG=%u LDS=%u\n",c.label,c.bits,c.m,c.n,c.k,which==3?"prep":which==2?"merge":which==1?"partial":"panel",arm.e.dims.workgroup_count[0],arm.e.dims.workgroup_count[1],arm.e.dims.workgroup_size[0],arm.e.lds_bytes);std::fflush(stdout);
    if(be) {
      auto h=be->load_executable(arm.e.entry_name,code->code);if(!h.ok()){std::fprintf(stderr,"load: %s\n",h.status().message().c_str());return 5;}arm.handle=h.release();
      for(const auto&n:arm.e.binding_order){if(!ensure(n))return 5;auto*a=find(n);arm.refs.push_back({&a->buffer,kGuard,n->element_count()*dtype_storage_bytes(n->dtype,1)});}
    }
    const auto output_count=static_cast<std::uint32_t>(arm.output->element_count());std::memcpy(arm.constants.data(),&output_count,4);
  }
  if(!be){std::printf("PASS offline native/math case=%s no owner initialized\n",c.label);return 0;}
  std::uint64_t launched=0;
  const auto launch_one=[&](unsigned arm){auto&a=arms[arm];if(!check(be->launch(a.handle,a.e.dims,{a.refs,a.constants}),"native launch"))return false;++launched;return true;};
  const auto launch=[&](unsigned arm){return arm==0?(launch_one(3)&&launch_one(0)):(launch_one(1)&&launch_one(2));};
  const auto verify=[&](unsigned which)->bool {
    auto*a=find(arms[which==0?0:2].output);std::vector<std::uint8_t> data(a->bytes.size());
    if(!check(be->copy_d2h(a->buffer,data.data(),data.size(),0),"output download"))return false;
    const auto*p=reinterpret_cast<const float*>(data.data()+kGuard);
    for(int r=0;r<c.m;++r)for(int col=0;col<c.n;++col) {
      const auto at=static_cast<std::size_t>(r*c.n+col),ei=static_cast<std::size_t>(r*16+col%16);
      const auto want=which?matrix_expected[ei]:expected[ei];
      const auto tolerance=c.realistic?(2e-5f+std::fabs(want)*2e-5f):0.0f;
      if(!std::isfinite(p[at])||std::fabs(p[at]-want)>tolerance){std::fprintf(stderr,"oracle mismatch %s arm%u row%d col%d got%.9g expected%.9g\n",c.label,which,r,col,static_cast<double>(p[at]),static_cast<double>(want));return false;}
    }
    if(which==0) {
      auto*a_panel=find(panel.node());std::vector<std::uint8_t> bytes(a_panel->bytes.size());
      if(!check(be->copy_d2h(a_panel->buffer,bytes.data(),bytes.size(),0),"panel download"))return false;
      const auto wanted=panel_oracle(c);
      if(std::memcmp(bytes.data()+kGuard,wanted.data(),wanted.size()*4)!=0)return false;
    } else {
      auto*a_partial=find(partial.node());std::vector<std::uint8_t> bytes(a_partial->bytes.size());
      if(!check(be->copy_d2h(a_partial->buffer,bytes.data(),bytes.size(),0),"partial download"))return false;
      const auto*parts=reinterpret_cast<const float*>(bytes.data()+kGuard);
      for(int split=0;split<4;++split)for(int r=0;r<c.m;++r)for(int col=0;col<c.n;++col) {
        const auto at=static_cast<std::size_t>((split*c.m+r)*c.n+col);
        const auto ei=static_cast<std::size_t>((split*c.m+r)*16+col%16);
        const auto want=partial_expected[ei];
        const auto tolerance=c.realistic?(2e-5f+std::fabs(want)*2e-5f):0.0f;
        if(!std::isfinite(parts[at])||std::fabs(parts[at]-want)>tolerance){std::fprintf(stderr,"partial oracle mismatch %s split%d row%d col%d got%.9g expected%.9g\n",c.label,split,r,col,static_cast<double>(parts[at]),static_cast<double>(want));return false;}
      }
    }
    return true;
  };
  const unsigned warm_count=c.n>17?32u:3u;
  for(unsigned which=0;which<2;++which) {
    for(unsigned warm=0;warm<warm_count;++warm)if(!launch(which))return 6;
    if(!check(be->synchronize(),"warm sync")||!verify(which))return 6;
  }
  unsigned sample=0;
  for(unsigned which:{0u,1u,1u,0u}) {
    if(!check(be->synchronize(),"timing pre sync"))return 6;
    const auto started=std::chrono::steady_clock::now();
    for(int j=0;j<count;++j)if(!launch(which))return 6;
    if(!check(be->synchronize(),"timing sync"))return 6;
    const auto wall=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
    std::printf("timing case=%s arm=%s sample=%u warmed_calls=%d wall_ms=%.6f per_call_ms=%.6f measured_compiles=0\n",c.label,which==0?"panel":"splitk4",sample++,count,wall,wall/count);std::fflush(stdout);
    if(!verify(which))return 7;
  }
  for(auto&a:allocations) {
    std::vector<std::uint8_t> data(a.bytes.size());
    if(!check(be->copy_d2h(a.buffer,data.data(),data.size(),0),"guard/input download"))return 8;
    for(std::size_t j=0;j<kGuard;++j)if(data[j]!=kSentinel||data[data.size()-kGuard+j]!=kSentinel)return 8;
    if(a.node!=arms[0].output&&a.node!=arms[1].output&&a.node!=arms[2].output&&a.node!=arms[3].output&&data!=a.bytes){std::fprintf(stderr,"input modified\n");return 8;}
    be->deallocate(a.buffer);
  }
  std::printf("packing case=%s output_bytes=%zu host_reorder_ms=%.6f warm_dispatches_per_arm=%u one_time_only=true source_weights_retained=true\n",c.label,packing_bytes,packing_ms,warm_count);
  std::printf("PASS native case=%s complete_outputs=%zu oracle=independent_DOT4chunk8_and_matrixgroup64_partials repeats=finite_complete guards=preserved device_dispatches=%llu host_dispatches=0 fallback_dispatches=0 timing=steady_wall_only\n",c.label,arms[0].output->element_count(),static_cast<unsigned long long>(launched));std::fflush(stdout);
  return 0;
}
}  // namespace
int main(int argc,char**argv) {
  if(argc<3||argc>5)return 2;
  const std::string mode=argv[1],prefix=argv[2];
  const bool offline=argc>=4&&std::string(argv[3])=="--offline";
  const int count=argc==5?std::atoi(argv[4]):20;if(count<=0||count>20)return 2;
  std::vector<Case> cases;
  if(mode=="perf"||mode=="perf-m4")cases={{4,4,17408,5120,"q4_up_m4",true},
                                                       {4,4,5120,17408,"q4_down_m4",true}};
  else if(mode=="edge")cases={{4,4,17,64,"q4_empty_split",true},
                              {4,4,17,128,"q4_two_empty_splits",true},
                              {4,4,17,320,"q4_uneven_split",true},
                              {4,4,17,5120,"q4_realistic_m4",true}};
  else return 2;
  std::unique_ptr<backend::IBackend>be;backend::DeviceInfo synthetic;backend::AmdDeviceInfo amd;
  synthetic.arch="gfx1201";backend::apply_arch_defaults(synthetic,amd);
  synthetic.extension_id=backend::AmdDeviceInfo::kExtensionId;synthetic.extension=&amd;
  if(!offline){auto made=backend::create_backend("hrx");if(!made.ok())return 2;be=made.release();if(!check(be->init(0),"init"))return 2;}
  const auto&device=offline?synthetic:be->device_info();
  if(device.arch!="gfx1201")return 2;
  for(const auto&c:cases){const auto result=run_case(c,be.get(),device,prefix,count);if(result)return result;}
  std::printf("PASS cases=%zu mode=%s offline=%d\n",cases.size(),mode.c_str(),offline);return 0;
}
