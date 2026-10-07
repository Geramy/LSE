#include "harness.hpp"
#include "lse/dispatch/arch/tuning.hpp"
#include "lse/backends/hrx/arch_database.hpp"
#include "lse/backends/hrx/loomc/loom_emitter.hpp"
#include "lse/graph/ops.hpp"
#include "lse/kv/block.hpp"
#include <algorithm>
#include <bit>
#include <cstring>
#include <limits>
#include <vector>
using namespace lse;
using namespace lse::graph;
namespace {
backend::DeviceInfo gpu() {
  backend::DeviceInfo d;
  d.arch="gfx1201"; d.wavefront_size=32; d.max_threads_per_workgroup=1024;
  d.lds_bytes_per_workgroup=65536;
  static backend::AmdDeviceInfo amd;
  backend::apply_arch_defaults(d,amd);
  d.extension_id=backend::AmdDeviceInfo::kExtensionId; d.extension=&amd;
  return d;
}
Array leaf(Shape shape) {
  auto n=std::make_shared<Node>(); n->shape=shape; n->dtype=DType::kF32; n->materialized=true;
  return Array(n);
}
float narrow_bf16(float value) {
  auto bits=std::bit_cast<std::uint32_t>(value);
  bits += 0x7fffu + ((bits >> 16) & 1u);
  return std::bit_cast<float>(bits & 0xffff0000u);
}
}
LSE_TEST(sparse_attention_routing_is_opt_in_and_phase_shape_specific) {
  const auto d=gpu();
  backend::LoomEmitter emitter;
  for (int queries : {1,2,17}) {
    auto q=leaf({1,24,queries,256}), k=leaf({48,4,16,256}), v=leaf(k.shape());
    auto meta=leaf({5}), table=leaf({1,48});
    auto dense=sdpa_paged(q,k,v,.0625f,MaskKind::kCausal,0,meta,table,16,&d);
    auto sparse=sdpa_paged(q,k,v,.0625f,MaskKind::kCausal,0,meta,table,16,&d,
                           kv::CacheDType::kF32,{true,1.0f});
    LSE_EXPECT_EQ(dense.node()->attrs[3],0.0f);
    const auto sparse_body=queries==1?sparse.node()->inputs[0]:sparse.node();
    LSE_EXPECT_EQ(sparse_body->attrs[3],1.0f);
    LSE_EXPECT(sparse.node()->prim->name()==(queries==1 ? "attention.split_merge128.wg128c2.v1" : "attention.flash.wmma16.v3"));
    const NodePtr roots[]{sparse.node()};
    auto groups=Partitioner::partition(roots,&d);
    LSE_EXPECT_EQ(groups.size(),queries==1?2u:1u);
    if(groups.empty()) continue;
    auto result=emitter.emit(groups[0],d);
    LSE_EXPECT(result.ok());
    if(!result.ok()) { std::fprintf(stderr,"%s\n",result.status().to_string().c_str()); continue; }
    LSE_EXPECT(result->source.find("kernel.barrier")!=std::string::npos);
    LSE_EXPECT(result->source.find("scalar.expf")!=std::string::npos);
    auto zero=sdpa_paged(q,k,v,.0625f,MaskKind::kCausal,0,meta,table,16,&d,
                         kv::CacheDType::kF32,{true,0.0f});
    const NodePtr zr[]{zero.node()}; auto zg=Partitioner::partition(zr,&d);
    LSE_EXPECT(emitter.cache_key(groups[0],d)!=emitter.cache_key(zg[0],d));
  }
}

LSE_TEST(prompt_and_speculative_attention_use_explicit_phase) {
  using Phase = ops::AttentionExecutionPhase;
  ops::SparseAttentionOptions options;
  options.prefill = {false, .1f, true};
  options.decode = {true, 1.0f, false};
  const auto d = gpu();
  for (int queries : {1, 2, 4, 8, 17}) {
    const auto prompt = ops::attention_for_phase(options, Phase::kPrefill, queries);
    const auto verify = ops::attention_for_phase(options, Phase::kSpeculative, queries);
    LSE_EXPECT_EQ(prompt.flashprefill, queries > 1);
    LSE_EXPECT(!verify.enabled());
    LSE_EXPECT(ops::attention_for_phase(options, Phase::kDecode, queries).blasst);
    auto q = leaf({1, 4, queries, 256}), k = leaf({48, 2, 16, 256}), v = leaf(k.shape());
    auto meta = leaf({5}), table = leaf({1, 48});
    auto p = sdpa_paged(q, k, v, .0625f, MaskKind::kCausal, 0, meta, table, 16,
                        &d, kv::CacheDType::kF32, prompt);
    auto verification = sdpa_paged(q, k, v, .0625f, MaskKind::kCausal, 0, meta, table,
                                   16, &d, kv::CacheDType::kF32, verify);
    LSE_EXPECT_EQ(p.node()->prim->name() == "attention.flashprefill.wmma.v1", queries > 1);
    LSE_EXPECT(verification.node()->prim->name() != "attention.flashprefill.wmma.v1");
    LSE_EXPECT_EQ(verification.node()->attrs[3], 0.0f);
  }
}

LSE_TEST(flashprefill_native_graph_has_pool_selector_and_correction) {
  const auto d=gpu(); backend::LoomEmitter emitter;
  auto q=leaf({2,4,2,256}),k=leaf({80,2,16,256}),v=leaf(k.shape());
  auto meta=leaf({7}),table=leaf({2,80});
  auto sparse=sdpa_paged(q,k,v,.0625f,MaskKind::kCausal,0,meta,table,16,&d,kv::CacheDType::kF32,{false,.8f,true});
  const NodePtr roots[]{sparse.node()}; auto groups=Partitioner::partition(roots,&d);
  LSE_EXPECT_EQ(groups.size(),3u);
  for(auto& group:groups) {
    auto result=emitter.emit(group,d); LSE_EXPECT(result.ok());
    if(!result.ok()) continue;
    LSE_EXPECT(result->source.find("kernel.def")!=std::string::npos);
  }
}

int gpu_check() {
  auto* scheduler=default_scheduler();
  if(!scheduler) return 1;
  scheduler->set_mode(Scheduler::Mode::kDeviceFirst);
  scheduler->set_dialect(Dialect::kLoom);
  const auto& d=scheduler->backend().device_info();
  if(!dispatch::arch::tuning(d.arch).flash_prefill) { std::fprintf(stderr,"requires a part with FlashPrefill qualified\n"); return 1; }
  auto upload=[&](Shape shape,const std::vector<float>& data) {
    auto allocation=scheduler->backend().allocate(data.size()*4,backend::MemoryClass::kDevice);
    if(!allocation.ok()) return Array{};
    auto buf=allocation.release();
    LSE_EXPECT_OK(scheduler->backend().copy(buf,data.data(),data.size()*4));
    return Array::from_buffer(std::move(buf),shape,DType::kF32);
  };
  unsigned cases=0;
  // Nonmonotonic physical pages, partial blocks, nonzero causal offsets,
  // masked history, an empty sequence, and an inactive padded batch row.
  for (int queries : {1,17}) for (auto format : {kv::CacheDType::kF32,kv::CacheDType::kBF16})
    for (int scenario : {0,1,2,3}) for (float coefficient : {0.0f,1.0f}) {
      constexpr int capacity=768, heads=2, width=256;
      const int length=scenario==2 ? 0 : 529;
      const int offset=scenario==3 ? 0 : 512;
      const auto mask=scenario==1 ? MaskKind::kSlidingWindow : MaskKind::kCausal;
      constexpr int window=129;
      std::vector<float> q(2*heads*queries*width),k(capacity*width),v(k.size());
      std::vector<float> table(2*48);
      for(int page=0;page<48;++page) { table[page]=float(47-page); table[48+page]=0; }
      for(int h=0;h<heads;++h) for(int r=0;r<queries;++r) q[(h*queries+r)*width]=16;
      for(int j=0;j<capacity;++j) {
        const int physical=(47-j/16)*16+j%16;
        k[physical*width]=j<256?8.0f:j<512?-8.0f:7.0f;
        for(int c=0;c<width;++c) v[physical*width+c]=float((j+c)%17-8)*.125f;
      }
      // No query, table or metadata from padded row 1 is semantically live.
      auto aq=upload({2,heads,queries,width},q);
      auto ak=cast(upload({48,1,16,width},k),kv::storage_dtype(format));
      auto av=cast(upload({48,1,16,width},v),kv::storage_dtype(format));
      auto am=upload({7},{float(offset),float(length),1,float(offset),float(length),0,0});
      auto at=upload({2,48},table);
      auto output=sdpa_paged(aq,ak,av,.0625f,mask,window,am,at,16,&d,format,{true,coefficient});
      const NodePtr roots[]{output.node()};
      scheduler->reset_accumulated_trace();
      auto status=scheduler->eval(roots,false);
      if(!status.ok()) { std::fprintf(stderr,"GPU eval: %s\n",status.to_string().c_str()); return 1; }
      LSE_EXPECT_OK(scheduler->drain());
      const auto trace=scheduler->last_trace();
      LSE_EXPECT_EQ(trace.host_groups,0u); LSE_EXPECT_EQ(trace.host_fallbacks,0u);
      std::vector<float> actual(q.size());
      LSE_EXPECT_OK(output.to_host(actual.data(),actual.size()*4));
      const int tile=queries==1?128:256;
      std::vector<double> maximum(queries,-INFINITY),denom(queries),expected(queries*width);
      for(int begin=0;begin<length;begin+=tile) {
        std::vector<double> block_max(queries,-INFINITY);
        auto allowed=[&](int row,int j) { return j<length && j<=offset+row &&
            (mask!=MaskKind::kSlidingWindow || offset+row-j<window); };
        for(int row=0;row<queries;++row) for(int j=begin;j<std::min(begin+tile,length);++j)
          if(allowed(row,j)) block_max[row]=std::max(block_max[row],double(j<256?8:j<512?-8:7));
        // Each prefill workgroup votes independently across up to 16 rows.
        for(int q0=0;q0<queries;q0+=16) {
          bool retain=false;
          for(int row=q0;row<std::min(q0+16,queries);++row)
            retain |= std::isfinite(block_max[row]) &&
                std::exp(block_max[row]-std::max(maximum[row],block_max[row]))>=std::min(double(coefficient)/std::max(length,1),1.0);
          if(!retain) continue;
          for(int row=q0;row<std::min(q0+16,queries);++row) {
            const double updated=std::max(maximum[row],block_max[row]);
            if(!std::isfinite(updated)) continue;
            const double alpha=std::exp(maximum[row]-updated); maximum[row]=updated;
            denom[row]*=alpha;
            for(int c=0;c<width;++c) expected[row*width+c]*=alpha;
            for(int j=begin;j<std::min(begin+tile,length);++j) if(allowed(row,j)) {
              const double probability=std::exp(double(j<256?8:j<512?-8:7)-updated);
              denom[row]+=probability;
              const double operand=queries==1?probability:narrow_bf16(float(probability));
              for(int c=0;c<width;++c) expected[row*width+c]+=operand*float((j+c)%17-8)*.125f;
            }
          }
        }
      }
      double max_error=0;
      for(int h=0;h<heads;++h) for(int row=0;row<queries;++row) for(int c=0;c<width;++c) {
        const double want=expected[row*width+c]/(denom[row]>0?denom[row]:1);
        const float got=actual[(h*queries+row)*width+c];
        LSE_EXPECT(std::isfinite(got)); max_error=std::max(max_error,std::abs(double(got)-want));
      }
      LSE_EXPECT(max_error<2e-5);
      for(size_t i=actual.size()/2;i<actual.size();++i) LSE_EXPECT_EQ(actual[i],0.0f);
      std::printf("BLASST q=%d storage=%d case=%d a=%g max_error=%g\n",queries,int(format),scenario,double(coefficient),max_error);
      ++cases;
    }
  for (int queries : {1,17}) for (auto format : {kv::CacheDType::kF32,kv::CacheDType::kBF16}) {
    constexpr int length=257, capacity=272, width=256, heads=4, kvheads=2, offset=240;
    std::uint32_t seed=42;
    auto random=[&](float unit) { seed=seed*1664525u+1013904223u; return float(int((seed>>16)%17)-8)*unit; };
    std::vector<float> q(heads*queries*width),k(capacity*kvheads*width),v(k.size()),table(17);
    for(auto& x:q) x=random(1.0f/32);
    for(auto& x:k) x=random(1.0f/32);
    for(auto& x:v) x=random(1.0f/8);
    for(int i=0;i<17;++i) table[i]=float(16-i);
    auto aq=upload({1,heads,queries,width},q);
    auto ak=cast(upload({17,kvheads,16,width},k),kv::storage_dtype(format));
    auto av=cast(upload({17,kvheads,16,width},v),kv::storage_dtype(format));
    auto am=upload({5},{offset,length,1,offset,length}),at=upload({1,17},table);
    auto output=sdpa_paged(aq,ak,av,.0625f,MaskKind::kCausal,0,am,at,16,&d,format,{true,0});
    const NodePtr roots[]{output.node()}; scheduler->reset_accumulated_trace();
    auto status=scheduler->eval(roots,false);
    if(!status.ok()) { std::fprintf(stderr,"random GPU eval: %s\n",status.to_string().c_str()); return 1; }
    LSE_EXPECT_OK(scheduler->drain());
    LSE_EXPECT_EQ(scheduler->last_trace().host_fallbacks,0u);
    LSE_EXPECT_EQ(scheduler->last_trace().host_groups,0u);
    std::vector<float> actual(q.size());
    LSE_EXPECT_OK(output.to_host(actual.data(),actual.size()*4));
    double max_error=0;
    for(int h=0;h<heads;++h) for(int row=0;row<queries;++row) {
      std::vector<double> scores(length),acc(width);
      auto index=[&](int j,int c) { return (((16-j/16)*kvheads+h/2)*16+j%16)*width+c; };
      for(int j=0;j<length;++j) {
        scores[j]=-INFINITY;
        if(j<=offset+row) {
          double value=0;
          for(int c=0;c<width;++c) value+=double(q[(h*queries+row)*width+c])*k[index(j,c)];
          scores[j]=value*.0625;
        }
      }
      double maximum=-INFINITY,denom=0;
      const int tile=queries==1?128:256;
      for(int begin=0;begin<length;begin+=tile) {
        double block_max=-INFINITY;
        for(int j=begin;j<std::min(begin+tile,length);++j) block_max=std::max(block_max,scores[j]);
        const double updated=std::max(maximum,block_max);
        if(!std::isfinite(updated)) continue;
        const double alpha=std::exp(maximum-updated); maximum=updated;
        denom*=alpha; for(auto& value:acc) value*=alpha;
        for(int j=begin;j<std::min(begin+tile,length);++j) {
          const double probability=std::exp(scores[j]-updated);
          denom+=probability;
          const double operand=queries==1?probability:narrow_bf16(float(probability));
          for(int c=0;c<width;++c) acc[c]+=operand*v[index(j,c)];
        }
      }
      for(int c=0;c<width;++c) {
        const float value=actual[(h*queries+row)*width+c]; LSE_EXPECT(std::isfinite(value));
        max_error=std::max(max_error,std::abs(double(value)-acc[c]/denom));
      }
    }
    LSE_EXPECT(max_error<2e-5);
    std::printf("BLASST random dense-equivalence q=%d storage=%d max_error=%g\n",queries,int(format),max_error);
    ++cases;
  }
  for(auto format:{kv::CacheDType::kF32,kv::CacheDType::kBF16}) for(float coefficient:{0.0f,64.0f}) {
    constexpr int capacity=2304,length=2065,heads=4,kvheads=2,width=256,parts=3;
    std::vector<float> q(heads*width),k(capacity*kvheads*width),v(k.size()),table(144);
    for(int page=0;page<144;++page) table[page]=float(143-page);
    for(int h=0;h<heads;++h) q[h*width]=16;
    auto score=[](int j,int kh) { return float((j%1024<128?8:j%1024<256?-8:7)-2*(j/1024))+float(kh)*.5f; };
    auto value=[](int j,int c,int kh) { return float((j+c+kh)%17-8)*.125f; };
    for(int j=0;j<capacity;++j) for(int kh=0;kh<kvheads;++kh) {
      const int base=(((143-j/16)*kvheads+kh)*16+j%16)*width;
      k[base]=score(j,kh);
      for(int c=0;c<width;++c) v[base+c]=value(j,c,kh);
    }
    auto aq=upload({1,heads,1,width},q);
    auto ak=cast(upload({144,kvheads,16,width},k),kv::storage_dtype(format));
    auto av=cast(upload({144,kvheads,16,width},v),kv::storage_dtype(format));
    auto am=upload({5},{length-1,length,1,length-1,length}),at=upload({1,144},table);
    auto output=sdpa_paged(aq,ak,av,.0625f,MaskKind::kCausal,0,am,at,16,&d,format,{true,coefficient});
    const NodePtr roots[]{output.node()};scheduler->reset_accumulated_trace();
    auto status=scheduler->eval(roots,false);
    if(!status.ok()) { std::fprintf(stderr,"partition BLASST: %s\n",status.to_string().c_str());return 1; }
    LSE_EXPECT_OK(scheduler->drain());LSE_EXPECT_EQ(scheduler->last_trace().host_groups,0u);
    std::vector<float> actual(q.size());LSE_EXPECT_OK(output.to_host(actual.data(),actual.size()*4));
    Array partial(output.node()->inputs[0]);std::vector<float> records(partial.shape().elem_count());
    LSE_EXPECT_OK(partial.to_host(records.data(),records.size()*4));
    double error=0;
    for(int h=0;h<heads;++h) {
      std::vector<double> maxima(parts,-INFINITY),denoms(parts),values(parts*width);
      for(int p=0;p<parts;++p) {
        for(int begin=p*1024;begin<std::min((p+1)*1024,length);begin+=128) {
          double maximum=-INFINITY;
          for(int j=begin;j<std::min(begin+128,length);++j) maximum=std::max(maximum,double(score(j,h/2)));
          const double updated=std::max(maxima[p],maximum);
          if(std::exp(maximum-updated)<double(coefficient)/length) continue;
          const double alpha=std::exp(maxima[p]-updated);maxima[p]=updated;denoms[p]*=alpha;
          for(int c=0;c<width;++c) values[p*width+c]*=alpha;
          for(int j=begin;j<std::min(begin+128,length);++j) {
            const double weight=std::exp(score(j,h/2)-updated);denoms[p]+=weight;
            for(int c=0;c<width;++c) values[p*width+c]+=weight*value(j,c,h/2);
          }
        }
        LSE_EXPECT(records[(h*parts+p)*258+1]>0);
        LSE_EXPECT(std::abs(records[(h*parts+p)*258]-maxima[p])<1e-6);
      }
      const double maximum=*std::max_element(maxima.begin(),maxima.end());
      double denominator=0;for(int p=0;p<parts;++p) denominator+=std::exp(maxima[p]-maximum)*denoms[p];
      for(int c=0;c<width;++c) {
        double numerator=0;for(int p=0;p<parts;++p) numerator+=std::exp(maxima[p]-maximum)*values[p*width+c];
        error=std::max(error,std::abs(actual[h*width+c]-numerator/denominator));
      }
    }
    LSE_EXPECT(error<2e-5);++cases;
    std::printf("BLASST partitioned storage=%d a=%g max_error=%g\n",int(format),double(coefficient),error);
  }
  {
    constexpr int heads=24,queries=2,width=256,kvheads=4,capacity=64;
    std::vector<float> q(heads*queries*width),k(capacity*kvheads*width),v(k.size());
    for(int j=0;j<capacity;++j) for(int kh=0;kh<kvheads;++kh) for(int c=0;c<width;++c)
      v[(((3-j/16)*kvheads+kh)*16+j%16)*width+c]=float(j);
    auto aq=upload({1,heads,queries,width},q),ak=upload({4,kvheads,16,width},k),av=upload({4,kvheads,16,width},v);
    auto am=upload({5},{0,2,1,0,2}),at=upload({1,4},{3,2,1,0});
    auto out=sdpa_paged(aq,ak,av,.0625f,MaskKind::kCausal,0,am,at,16,&d,kv::CacheDType::kF32,{false,.8f,true});
    const NodePtr roots[]{out.node()};scheduler->reset_accumulated_trace();
    const auto status=scheduler->eval(roots,false);
    if(!status.ok()) { std::fprintf(stderr,"FPV2 warmup: %s\n",status.to_string().c_str());return 1; }
    LSE_EXPECT_OK(scheduler->drain());LSE_EXPECT_EQ(scheduler->last_trace().host_groups,0u);
    std::vector<float> actual(q.size());LSE_EXPECT_OK(out.to_host(actual.data(),actual.size()*4));
    for(int h=0;h<heads;++h) for(int r=0;r<queries;++r) for(int c=0;c<width;++c)
      LSE_EXPECT(std::abs(actual[(h*queries+r)*width+c]-float(r)*.5f)<1e-6f);
    ++cases;
  }
  for(int queries:{2,17}) for(auto format:{kv::CacheDType::kF32,kv::CacheDType::kBF16})
    for(float threshold:{0.0f,.8f}) for(int offset:{1024,1020}) {
      constexpr int length=1041,capacity=1280,heads=4,kvheads=2,width=256;
      std::vector<float> q(2*heads*queries*width),k(capacity*kvheads*width),v(k.size()),table(160);
      for(int page=0;page<80;++page) table[page]=float(79-page);
      for(int h=0;h<heads;++h) for(int row=0;row<queries;++row) q[(h*queries+row)*width]=h%2?32.0f:16.0f;
      for(int j=0;j<capacity;++j) for(int kh=0;kh<kvheads;++kh) {
        const int base=(((79-j/16)*kvheads+kh)*16+j%16)*width;
        k[base]=(j<256?8.0f:j<512?-8.0f:7.0f)+float(kh)*.5f+float(j%5-2)*.125f;
        for(int c=0;c<width;++c) v[base+c]=float((j/256+c+kh)%7-3)*.25f+float(j%11-5)*.0625f;
      }
      auto aq=upload({2,heads,queries,width},q);
      auto ak=cast(upload({80,kvheads,16,width},k),kv::storage_dtype(format));
      auto av=cast(upload({80,kvheads,16,width},v),kv::storage_dtype(format));
      auto am=upload({7},{float(offset),length,1,float(offset),length,0,0}),at=upload({2,80},table);
      auto output=sdpa_paged(aq,ak,av,.0625f,MaskKind::kCausal,0,am,at,16,&d,format,{false,threshold,true});
      const NodePtr roots[]{output.node()}; scheduler->reset_accumulated_trace();
      auto status=scheduler->eval(roots,false);
      if(!status.ok()) { std::fprintf(stderr,"FPV2 GPU eval: %s\n",status.to_string().c_str()); return 1; }
      LSE_EXPECT_OK(scheduler->drain());
      LSE_EXPECT_EQ(scheduler->last_trace().host_groups,0u);
      LSE_EXPECT_EQ(scheduler->last_trace().host_fallbacks,0u);
      std::vector<float> actual(q.size()); LSE_EXPECT_OK(output.to_host(actual.data(),actual.size()*4));
      Array selection(output.node()->inputs[6]);
      std::vector<float> selected(selection.shape().elem_count());
      LSE_EXPECT_OK(selection.to_host(selected.data(),selected.size()*4));
      Array pool_array(output.node()->inputs[5]);
      std::vector<float> pooled(pool_array.shape().elem_count());
      LSE_EXPECT_OK(pool_array.to_host(pooled.data(),pooled.size()*4));
      for(int kh=0;kh<kvheads;++kh) for(int block=0;block<5;++block) {
        const int count=std::max(0,std::min(256,length-block*256));
        const int destination=(kh*5+block)*513;
        LSE_EXPECT_EQ(pooled[destination+512],float(count));
        for(int c=0;c<width;++c) {
          double mean_k=0,mean_v=0;
          for(int j=block*256;j<block*256+count;++j) {
            const int base=(((79-j/16)*kvheads+kh)*16+j%16)*width;
            mean_k+=k[base+c];mean_v+=v[base+c];
          }
          LSE_EXPECT(std::abs(pooled[destination+c]-mean_k/count)<1e-6);
          LSE_EXPECT(std::abs(pooled[destination+256+c]-mean_v/count)<1e-6);
        }
      }
      const int tiles=(queries+15)/16;
      for(int h=0;h<heads;++h) for(int qt=0;qt<tiles;++qt) for(int block=0;block<5;++block)
        LSE_EXPECT_EQ(selected[(h*tiles+qt)*5+block],threshold>0 && block>0 && (block+2)*256<=offset+qt*16?0.0f:1.0f);
      double max_error=0;
      for(int h=0;h<heads;++h) for(int row=0;row<queries;++row) {
        double maximum=-INFINITY,denom=0; std::vector<double> acc(width);
        for(int block=0;block<5;++block) {
          const bool exact=selected[(h*tiles+row/16)*5+block]!=0;
          const int count=std::max(0,std::min({256,length-block*256,offset+row-block*256+1}));
          if(!count) continue;
          auto key_score=[&](int j) {
            const int base=(((79-j/16)*kvheads+h/2)*16+j%16)*width;
            return double(k[base])*(h%2?2:1);
          };
          double block_score=-INFINITY;
          if(exact) for(int j=block*256;j<block*256+count;++j) block_score=std::max(block_score,key_score(j));
          else {
            block_score=0;
            for(int j=block*256;j<block*256+count;++j) block_score+=key_score(j);
            block_score=block_score/count+std::log(double(count));
          }
          const double updated=std::max(maximum,block_score);
          const double alpha=std::exp(maximum-updated); maximum=updated;
          denom*=alpha; for(auto& value:acc) value*=alpha;
          for(int j=block*256;j<block*256+count;++j) {
            const int base=(((79-j/16)*kvheads+h/2)*16+j%16)*width;
            const double weight=exact?std::exp(key_score(j)-updated):std::exp(block_score-updated)/count;
            denom+=weight;
            const double operand=exact?double(narrow_bf16(float(weight))):weight;
            for(int c=0;c<width;++c) acc[c]+=operand*v[base+c];
          }
        }
        for(int c=0;c<width;++c) {
          const auto got=actual[(h*queries+row)*width+c]; LSE_EXPECT(std::isfinite(got));
          max_error=std::max(max_error,std::abs(double(got)-acc[c]/denom));
        }
      }
      LSE_EXPECT(max_error<2e-5);
      for(size_t i=actual.size()/2;i<actual.size();++i) LSE_EXPECT_EQ(actual[i],0.0f);
      std::printf("FlashPrefill V2 q=%d storage=%d offset=%d alpha=%g max_error=%g\n",queries,int(format),offset,double(threshold),max_error);
      ++cases;
    }
  // Dense 256-component probes and a block count above one workgroup width
  // catch lane ownership, wave reductions, and strided selector tails.
  for(int blocks:{5,129}) {
    constexpr int heads=2,queries=17,width=256,tiles=2;
    const int length=blocks*256-7,offset=length-queries;
    std::vector<float> q(2*heads*queries*width),pool(2*blocks*513);
    for(int h=0;h<heads;++h) for(int r=0;r<queries;++r) for(int c=0;c<width;++c)
      q[(h*queries+r)*width+c]=float(c%11-5)*.125f+float((r+h)%3)*.0625f;
    for(int b=0;b<blocks;++b) {
      pool[b*513+512]=float(b==blocks-1?249:256);
      for(int c=0;c<width;++c)
        pool[b*513+c]=float(b%9-4)*.25f*float(c%11-5)+float((c+b)%7-3)*.03125f;
    }
    auto aq=upload({2,heads,queries,width},q),ap=upload({2,1,blocks,513},pool);
    auto am=upload({7},{float(offset),float(length),1,float(offset),float(length),0,0});
    auto node=std::make_shared<Node>();node->kind=OpKind::kCustom;
    node->shape={2,heads,tiles,blocks};node->dtype=DType::kF32;
    node->prim=find_primitive("attention.flashprefill.select.v2");
    node->fclass=node->prim->fusion_class();node->attrs={.0625f,0,.8f,2};
    node->inputs={aq.node(),ap.node(),am.node()};
    for(auto& input:node->inputs) ++input->consumer_count;
    Array selected_array(node);const NodePtr roots[]{node};scheduler->reset_accumulated_trace();
    const auto status=scheduler->eval(roots,false);
    if(!status.ok()) { std::fprintf(stderr,"selector reference: %s\n",status.to_string().c_str());return 1; }
    LSE_EXPECT_OK(scheduler->drain());LSE_EXPECT_EQ(scheduler->last_trace().host_groups,0u);
    std::vector<float> actual(node->shape.elem_count());LSE_EXPECT_OK(selected_array.to_host(actual.data(),actual.size()*4));
    unsigned omitted=0;
    for(int h=0;h<heads;++h) for(int t=0;t<tiles;++t) {
      std::vector<double> mass(blocks);double maximum=-INFINITY;
      std::vector<double> logits(blocks*16,-INFINITY);
      for(int b=0;b<blocks;++b) for(int r=t*16;r<std::min(t*16+16,queries);++r) {
        double score=0;for(int c=0;c<width;++c) score+=double(q[(h*queries+r)*width+c])*pool[b*513+c];
        logits[b*16+r-t*16]=score*.0625;maximum=std::max(maximum,score*.0625);
      }
      for(int b=0;b<blocks;++b) for(int r=0;r<16;++r) mass[b]+=std::exp(logits[b*16+r]-maximum);
      const double peak=*std::max_element(mass.begin(),mass.end());
      for(int b=0;b<blocks;++b) {
        const bool keep=b==0 || (b+2)*256>offset+t*16 || b==blocks-1 || mass[b]>=peak*.8;
        LSE_EXPECT_EQ(actual[(h*tiles+t)*blocks+b],keep?1.0f:0.0f);omitted+=!keep;
      }
    }
    for(size_t i=actual.size()/2;i<actual.size();++i) LSE_EXPECT_EQ(actual[i],0.0f);
    LSE_EXPECT(omitted>0);++cases;
    std::printf("FlashPrefill Wave32 selector blocks=%d omitted=%u\n",blocks,omitted);
  }
  std::printf("Sparse attention GPU reference cases: %u\n",cases);
  return test::Registry::get().failures?1:0;
}
int main(int argc,char** argv) {
  if(argc==2 && std::string_view(argv[1])=="--gpu") return gpu_check();
  return test::run_all();
}
