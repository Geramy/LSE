#include "harness.hpp"
#include "lse/backends/cpu/cpu_backend.hpp"
#include "lse/graph/interpreter.hpp"
#include "lse/graph/ops.hpp"
#include "lse/model/hybrid_lm.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

using namespace lse;
using namespace lse::graph;
using namespace lse::model;
namespace {
Array upload(backend::IBackend& backend, Shape shape, const std::vector<float>& values) {
  auto allocation = backend.allocate((values.size()+2)*sizeof(float), backend::MemoryClass::kDevice);
  LSE_EXPECT(allocation.ok()); if (!allocation.ok()) return {};
  auto buffer = allocation.release();
  buffer.offset=2*sizeof(float); buffer.size_bytes=values.size()*sizeof(float);
  LSE_EXPECT_OK(backend.copy(buffer, values.data(), values.size()*sizeof(float)));
  return Array::from_buffer(std::move(buffer), std::move(shape), DType::kF32);
}
void write(backend::IBackend& backend, const Array& array, const std::vector<float>& values) {
  LSE_EXPECT_OK(backend.copy(array.node()->buffer, values.data(), values.size()*sizeof(float)));
  array.node()->host_dirty = false;
  array.node()->device_dirty = array.node()->buffer.ptr == nullptr;
}
std::vector<float> read(backend::IBackend& backend, const Array& array) {
  LSE_EXPECT_OK(graph::interpreter::sync_to_device(*array.node(), backend));
  std::vector<float> values(array.shape().elem_count());
  LSE_EXPECT_OK(backend.copy(values.data(), array.node()->buffer, values.size()*sizeof(float)));
  return values;
}
std::vector<float> patterned(std::size_t count, float phase) {
  std::vector<float> values(count);
  for (std::size_t i=0; i<count; ++i) values[i] = std::sin((static_cast<float>(i)+phase)*.37f)*.23f;
  return values;
}
void near(const std::vector<float>& a, const std::vector<float>& b) {
  LSE_EXPECT_EQ(a.size(), b.size());
  for (std::size_t i=0; i<std::min(a.size(), b.size()); ++i) {
    LSE_EXPECT(std::isfinite(a[i])); LSE_EXPECT_NEAR(a[i], b[i], 2e-6);
  }
}
// Column dot products implement S <- alpha*S + beta*(v-alpha*S*k)*k^T.
std::vector<float> recurrent_prefix(const std::vector<float>& initial,
    const std::vector<float>& key, const std::vector<float>& value,
    const std::vector<float>& alpha, const std::vector<float>& beta,
    std::size_t rows, std::size_t heads, std::size_t dim) {
  std::vector<double> state(initial.begin(), initial.end());
  for (std::size_t t=0; t<rows; ++t) for (std::size_t h=0; h<heads; ++h) {
    double* s = state.data()+h*dim*dim;
    const auto at = (t*heads+h)*dim;
    const double decay = static_cast<double>(alpha[t*heads+h]), gain = static_cast<double>(beta[t*heads+h]);
    for (std::size_t channel=0; channel<dim; ++channel) {
      double predicted = 0;
      for (std::size_t k=0; k<dim; ++k) predicted += s[channel*dim+k]*static_cast<double>(key[at+k]);
      const double correction = gain*(static_cast<double>(value[at+channel])-decay*predicted);
      for (std::size_t k=0; k<dim; ++k)
        s[channel*dim+k] = decay*s[channel*dim+k]+correction*static_cast<double>(key[at+k]);
    }
  }
  return {state.begin(), state.end()};
}
std::vector<float> convolution_prefix(const std::vector<float>& previous,
    const std::vector<float>& input, std::size_t rows, std::size_t channels) {
  auto history = previous;
  history.insert(history.end(), input.begin(), input.begin()+static_cast<std::ptrdiff_t>(rows*channels));
  return {history.end()-static_cast<std::ptrdiff_t>(previous.size()), history.end()};
}
struct StateFixture {
  static constexpr std::size_t rows=4, heads=2, dim=4, channels=5;
  Array key, value, alpha, beta, initial, previous, raw;
  MixerState state;
  PrefixStateCommit prefix;
  Program target;
  std::vector<NodePtr> roots;
  std::vector<float> k, v, a, b, s, tail, x;
  void record(backend::IBackend& backend, bool reshape_value=false) {
    k=patterned(rows*heads*dim, 1); v=patterned(k.size(), 4);
    s=patterned(heads*dim*dim, 7); tail=patterned(3*channels, 12); x=patterned(rows*channels, 19);
    a.assign(rows*heads, .89f); b.assign(rows*heads, .47f);
    key=upload(backend,{1,rows,heads,dim},k);
    value=upload(backend,reshape_value ? Shape{rows*heads,dim} : key.shape(),v);
    if (reshape_value) value=reshape(reshape(value,{1,rows*heads,dim}),key.shape());
    alpha=upload(backend,{1,rows,heads},a); beta=upload(backend,alpha.shape(),b);
    initial=upload(backend,{1,heads,dim,dim},s); previous=upload(backend,{1,3,channels},tail);
    raw=upload(backend,{1,rows,channels},x);
    (void)gated_delta_step(key,key,value,alpha,beta,initial,&state.gdn_state);
    state.gdn_conv_qkv=conv_tail(previous,raw);
    roots={state.gdn_state.node(),state.gdn_conv_qkv.node()};
    const MixerState states[]{state};
    LSE_EXPECT_OK(prefix.retain(states, rows, roots));
    LSE_EXPECT_EQ(roots.size(),9u);
  }
  void evaluate(Scheduler& scheduler) {
    LSE_EXPECT_OK(scheduler.eval(roots,false,&target));
    LSE_EXPECT_OK(scheduler.drain());
  }
  void check(backend::IBackend& backend, std::size_t count) {
    near(read(backend,state.gdn_state),recurrent_prefix(s,k,v,a,b,count,heads,dim));
    near(read(backend,state.gdn_conv_qkv),convolution_prefix(tail,x,count,channels));
    near(read(backend,initial),s); near(read(backend,previous),tail);
  }
};
class OpaqueBackend final : public backend::Backend<OpaqueBackend> {
 public:
  static constexpr std::string_view kName="fixture.prefix_opaque";
  Status init_impl(int) { return OkStatus(); }
  void shutdown_impl() noexcept {}
  const backend::DeviceInfo& device_info_impl() const noexcept { return info_; }
  Result<backend::DeviceBuffer> allocate_impl(std::size_t bytes,backend::MemoryClass,backend::Stream) {
    auto storage=std::make_shared<std::vector<std::byte>>(bytes);
    backend::DeviceBuffer buffer; buffer.handle=reinterpret_cast<std::uint64_t>(storage->data());
    buffer.size_bytes=bytes; buffer.storage=std::move(storage); return buffer;
  }
  void deallocate_impl(backend::DeviceBuffer& buffer) noexcept { buffer={}; }
  Status copy_h2d_impl(const void* src,backend::DeviceBuffer& dst,std::size_t bytes,std::size_t offset) {
    std::memcpy(address(dst)+offset,src,bytes); return OkStatus();
  }
  Status copy_d2h_impl(const backend::DeviceBuffer& src,void* dst,std::size_t bytes,std::size_t offset) {
    std::memcpy(dst,address(src)+offset,bytes); return OkStatus();
  }
  Result<backend::KernelHandle> load_executable_impl(std::string_view,std::span<const std::byte>) {
    return LSE_ERROR(kUnimplemented,"fixture has no compiler");
  }
  Status launch_impl(const backend::KernelHandle&,const backend::LaunchDims&,const backend::DispatchArgs&) {
    return LSE_ERROR(kUnimplemented,"fixture has no GPU");
  }
  Status synchronize_impl() { return OkStatus(); }
  std::span<const KernelToolchain> toolchains_impl() const noexcept { return {}; }
 private:
  static std::byte* address(const backend::DeviceBuffer& b) { return reinterpret_cast<std::byte*>(b.handle)+b.offset; }
  backend::DeviceInfo info_;
};
class RecurrentMixer final : public IMixer {
 public:
  Status load(WeightBinder&,std::string_view,const LayerContext&) override { return OkStatus(); }
  Result<Array> forward(const Array& x,MixerState* state,const LayerContext&) override {
    if (!state) return LSE_ERROR(kInvalidArgument,"fixture needs state");
    const auto old=state->gdn_conv_qkv;
    state->gdn_conv_qkv=conv_tail(old,x);
    const auto convolved=causal_conv1d(x,Array::full({4,3},DType::kF32,.2f),Array::zeros({4},DType::kF32),old);
    const auto vector=reshape(convolved,{1,x.shape().dim(1),1,4});
    const auto scalar=Shape{1,x.shape().dim(1),1};
    auto output=gated_delta_step(vector,vector,vector,Array::full(scalar,DType::kF32,.87f),
                                Array::full(scalar,DType::kF32,.41f),state->gdn_state,&state->gdn_state);
    return reshape(output,x.shape());
  }
  std::string_view name() const noexcept override { return "fixture.recurrent"; }
};
class ZeroFfn final : public IFeedForward {
 public:
  Status load(WeightBinder&,std::string_view,const LayerContext&) override { return OkStatus(); }
  Result<Array> forward(const Array& x,Array*,const LayerContext&) override { return Array::zeros(x.shape(),DType::kF32); }
  std::string_view name() const noexcept override { return "fixture.zero_ffn"; }
};
std::filesystem::path weights_file() {
  const auto path=std::filesystem::temp_directory_path()/"lse-prefix-commit-fixture.safetensors";
  std::string header=R"({"embed.weight":{"dtype":"F32","shape":[16,4],"data_offsets":[0,256]},"final_norm.weight":{"dtype":"F32","shape":[4],"data_offsets":[256,272]},"blocks.0.norm1.weight":{"dtype":"F32","shape":[4],"data_offsets":[272,288]},"blocks.0.norm2.weight":{"dtype":"F32","shape":[4],"data_offsets":[288,304]}})";
  header.append((8-header.size()%8)%8,' ');
  auto data=patterned(64,3); data.insert(data.end(),12,1);
  const std::uint64_t bytes=header.size(); std::ofstream out(path,std::ios::binary);
  out.write(reinterpret_cast<const char*>(&bytes),sizeof(bytes)); out.write(header.data(),static_cast<std::streamsize>(header.size()));
  out.write(reinterpret_cast<const char*>(data.data()),static_cast<std::streamsize>(data.size()*sizeof(float)));
  return path;
}
}
LSE_TEST(prefix_state_matches_each_m4_prefix_without_replacing_carry_allocations) {
  backend::BackendAdapter<backend::CpuBackend> backend; LSE_EXPECT_OK(backend.init(0));
  Scheduler scheduler(backend); scheduler.set_mode(Scheduler::Mode::kHostOnly);
  StateFixture f; f.record(backend); f.evaluate(scheduler);
  const auto state_node=f.state.gdn_state.node(),tail_node=f.state.gdn_conv_qkv.node();
  const auto state_handle=state_node->buffer.handle,tail_handle=tail_node->buffer.handle;
  for (std::size_t count=1; count<4; ++count) {
    LSE_EXPECT_OK(f.prefix.commit(count,scheduler)); f.check(backend,count);
    LSE_EXPECT_EQ(scheduler.last_trace().nodes_evaluated,2u);
    for (const auto& description : scheduler.last_trace().group_descriptions)
      LSE_EXPECT(description.find("slice") == std::string::npos);
    LSE_EXPECT(f.state.gdn_state.node()==state_node && f.state.gdn_conv_qkv.node()==tail_node);
    LSE_EXPECT_EQ(state_node->buffer.handle,state_handle); LSE_EXPECT_EQ(tail_node->buffer.handle,tail_handle);
  }
}
LSE_TEST(prefix_replay_refreshes_inputs_and_rebound_verifier_output_buffers) {
  backend::BackendAdapter<backend::CpuBackend> backend; LSE_EXPECT_OK(backend.init(0));
  Scheduler scheduler(backend); scheduler.set_mode(Scheduler::Mode::kHostOnly);
  StateFixture f; f.record(backend); f.evaluate(scheduler);
  LSE_EXPECT_OK(f.prefix.commit(2,scheduler)); f.check(backend,2);
  f.k=patterned(f.k.size(),43); f.v=patterned(f.v.size(),52); f.s=patterned(f.s.size(),61); f.x=patterned(f.x.size(),75);
  write(backend,f.key,f.k); write(backend,f.value,f.v); write(backend,f.initial,f.s); write(backend,f.raw,f.x);
  f.target.reset_compute();
  auto new_state=backend.allocate(f.state.gdn_state.node()->buffer.size_bytes,backend::MemoryClass::kDevice,backend::kDefaultStream);
  auto new_tail=backend.allocate(f.state.gdn_conv_qkv.node()->buffer.size_bytes,backend::MemoryClass::kDevice,backend::kDefaultStream);
  LSE_EXPECT(new_state.ok() && new_tail.ok()); if (!new_state.ok() || !new_tail.ok()) return;
  f.state.gdn_state.node()->buffer=new_state.release(); f.state.gdn_conv_qkv.node()->buffer=new_tail.release();
  f.evaluate(scheduler); LSE_EXPECT_OK(f.prefix.commit(2,scheduler)); f.check(backend,2);
}
LSE_TEST(prefix_preflight_refuses_invalid_counts_short_buffers_and_aliases_without_writes) {
  backend::BackendAdapter<backend::CpuBackend> backend; LSE_EXPECT_OK(backend.init(0));
  Scheduler scheduler(backend); scheduler.set_mode(Scheduler::Mode::kHostOnly);
  StateFixture f; f.record(backend); f.evaluate(scheduler);
  const auto before=read(backend,f.state.gdn_state),tail=read(backend,f.state.gdn_conv_qkv);
  LSE_EXPECT(!f.prefix.commit(0,scheduler).ok()); LSE_EXPECT(!f.prefix.commit(4,scheduler).ok());
  const auto saved=f.key.node()->buffer;
  f.key.node()->buffer.size_bytes=sizeof(float); LSE_EXPECT(!f.prefix.commit(2,scheduler).ok());
  f.key.node()->buffer=f.state.gdn_state.node()->buffer; LSE_EXPECT(!f.prefix.commit(2,scheduler).ok());
  f.key.node()->buffer=saved;
  near(read(backend,f.state.gdn_state),before); near(read(backend,f.state.gdn_conv_qkv),tail);
}
LSE_TEST(prefix_reads_materialized_reshape_chains_and_refuses_missing_views) {
  backend::BackendAdapter<backend::CpuBackend> backend; LSE_EXPECT_OK(backend.init(0));
  Scheduler scheduler(backend); scheduler.set_mode(Scheduler::Mode::kHostOnly);
  StateFixture f; f.record(backend,true); f.evaluate(scheduler);
  auto& view=*f.value.node(); auto& middle=*view.inputs[0]; auto& owner=*middle.inputs[0];
  view.materialized=false; middle.materialized=false;
  const NodePtr views[]{f.value.node()};
  LSE_EXPECT_OK(scheduler.eval(views,false));
  LSE_EXPECT_OK(f.prefix.commit(2,scheduler)); f.check(backend,2);
  f.v=patterned(f.v.size(),101);
  auto fresh=upload(backend,owner.shape,f.v);
  owner.buffer=fresh.node()->buffer;
  owner.materialized=true; owner.host_dirty=false;
  view.materialized=false; middle.materialized=false;
  LSE_EXPECT_OK(scheduler.eval(views,false));
  LSE_EXPECT_OK(f.prefix.commit(1,scheduler)); f.check(backend,1);
  LSE_EXPECT_EQ(view.buffer.handle,owner.buffer.handle);
  LSE_EXPECT_EQ(view.buffer.offset,owner.buffer.offset);
  const auto before=read(backend,f.state.gdn_state);
  view.materialized=false; LSE_EXPECT(!f.prefix.commit(2,scheduler).ok());
  view.materialized=true;
  const auto bytes=view.buffer.size_bytes; view.buffer.size_bytes=sizeof(float);
  LSE_EXPECT(!f.prefix.commit(2,scheduler).ok()); view.buffer.size_bytes=bytes;
  near(read(backend,f.state.gdn_state),before);
}
LSE_TEST(prefix_opaque_host_fallback_publishes_state_to_the_original_device_buffers) {
  backend::BackendAdapter<OpaqueBackend> backend; LSE_EXPECT_OK(backend.init(0));
  Scheduler scheduler(backend); scheduler.set_mode(Scheduler::Mode::kHostOnly);
  StateFixture f; f.record(backend,true); f.evaluate(scheduler);
  f.v=patterned(f.v.size(),91);
  auto& view=*f.value.node(); auto& owner=*view.inputs[0]->inputs[0];
  for (std::size_t i=0; i<f.v.size(); ++i) graph::interpreter::store_element(owner,i,f.v[i]);
  view.materialized=false; view.inputs[0]->materialized=false;
  const NodePtr views[]{f.value.node()};
  LSE_EXPECT_OK(scheduler.eval(views,false));
  LSE_EXPECT_OK(f.prefix.commit(3,scheduler));
  // Read device storage directly, bypassing the old verifier node's host mirror.
  std::vector<float> actual(f.s.size());
  LSE_EXPECT_OK(backend.copy(actual.data(),f.state.gdn_state.node()->buffer,actual.size()*sizeof(float)));
  near(actual,recurrent_prefix(f.s,f.k,f.v,f.a,f.b,3,f.heads,f.dim));
  f.check(backend,3);
  LSE_EXPECT_OK(f.prefix.commit(1,scheduler)); f.check(backend,1);
}
LSE_TEST(hybrid_prefix_commit_advances_cursor_and_preserves_same_width_carry_replay) {
  auto* scheduler=default_scheduler(); LSE_EXPECT(scheduler!=nullptr); if (!scheduler) return;
  scheduler->set_mode(Scheduler::Mode::kHostOnly);
  const auto path=weights_file(); auto weights=SafeTensors::open(path.string());
  LSE_EXPECT(weights.ok()); if (!weights.ok()) return;
  Config c; c.hidden_size=4; c.vocab_size=16; c.num_layers=1; c.kv_length=64;
  c.gdn_qk_heads=1; c.gdn_v_heads=1; c.gdn_head_dim=4; c.gdn_conv_kernel=3;
  HybridLMSpec spec; spec.zero_centered_norm=false; spec.gdn_conv_width=4;
  auto factory=[](std::int32_t)->Result<std::unique_ptr<HybridBlock>> {
    return std::make_unique<HybridBlock>(std::make_unique<RecurrentMixer>(),std::make_unique<ZeroFfn>(),false);
  };
  HybridLM actual(c,spec,factory),reference(c,spec,factory);
  WeightBinder ba(*weights),br(*weights); LSE_EXPECT_OK(actual.load(ba)); LSE_EXPECT_OK(reference.load(br));
  std::vector<MixerState> sa(1),sr(1);
  auto run=[&](HybridLM& model,std::vector<MixerState>& states,std::vector<float> tokens,bool retained=false) {
    auto ids=upload(scheduler->backend(),{1,static_cast<std::int64_t>(tokens.size())},tokens);
    auto hidden=model.hidden(ids,&states,nullptr,nullptr,nullptr,false,nullptr,retained,
                             retained ? ops::AttentionExecutionPhase::kSpeculative
                                      : ops::AttentionExecutionPhase::kDecode);
    LSE_EXPECT(hidden.ok()); if (!hidden.ok()) return Array{};
    LSE_EXPECT_OK(hidden->eval()); return hidden.release();
  };
  (void)run(actual,sa,{1}); (void)run(reference,sr,{1});
  (void)run(actual,sa,{2,3,4,5},true); const auto carry=sa[0].gdn_state.node();
  const auto handle=carry->buffer.handle;
  const auto verifier_end=sa[0].position;
  actual.rewind(sa,verifier_end-1);
  LSE_EXPECT(!actual.commit_prefix(sa,2).ok());
  LSE_EXPECT_EQ(sa[0].position,verifier_end-1);
  actual.rewind(sa,verifier_end);
  LSE_EXPECT_OK(actual.commit_prefix(sa,2)); (void)run(reference,sr,{2,3});
  LSE_EXPECT_EQ(sa[0].position,3); LSE_EXPECT_EQ(sa[0].gdn_state.node()->buffer.handle,handle);
  LSE_EXPECT(!actual.commit_prefix(sa,1).ok());
  near(read(scheduler->backend(),sa[0].gdn_state),read(scheduler->backend(),sr[0].gdn_state));
  near(read(scheduler->backend(),sa[0].gdn_conv_qkv),read(scheduler->backend(),sr[0].gdn_conv_qkv));
  (void)run(actual,sa,{4,5,6,7},true); LSE_EXPECT(sa[0].gdn_state.node()==carry);
  LSE_EXPECT_OK(actual.commit_prefix(sa,3)); (void)run(reference,sr,{4,5,6});
  LSE_EXPECT_EQ(sa[0].position,6);
  auto a=run(actual,sa,{8}),b=run(reference,sr,{8});
  if (a.valid() && b.valid()) near(read(scheduler->backend(),a),read(scheduler->backend(),b));
  near(read(scheduler->backend(),sa[0].gdn_state),read(scheduler->backend(),sr[0].gdn_state));
  LSE_EXPECT_EQ(sa[0].position,7);
  auto compare_state=[&]() {
    LSE_EXPECT_EQ(sa[0].position,sr[0].position);
    near(read(scheduler->backend(),sa[0].gdn_state),read(scheduler->backend(),sr[0].gdn_state));
    near(read(scheduler->backend(),sa[0].gdn_conv_qkv),read(scheduler->backend(),sr[0].gdn_conv_qkv));
  };
  // Full acceptance uses the ordinary carry fold, without a prefix replay.
  (void)run(actual,sa,{9,10,11,12},true); (void)run(reference,sr,{9,10,11,12}); compare_state();
  // Reject, change verifier width, then return to the original width.
  (void)run(actual,sa,{13,14,15,0},true);
  const auto cursor=sa[0].position;
  LSE_EXPECT(!actual.commit_prefix(sa,0).ok()); LSE_EXPECT(!actual.commit_prefix(sa,4).ok());
  LSE_EXPECT_EQ(sa[0].position,cursor);
  LSE_EXPECT_OK(actual.commit_prefix(sa,1)); (void)run(reference,sr,{13}); compare_state();
  (void)run(actual,sa,{14,15},true); LSE_EXPECT_OK(actual.commit_prefix(sa,1));
  (void)run(reference,sr,{14}); compare_state();
  (void)run(actual,sa,{15,0,1,2},true); LSE_EXPECT_OK(actual.commit_prefix(sa,2));
  (void)run(reference,sr,{15,0}); compare_state();
  std::error_code error; std::filesystem::remove(path,error);
}
LSE_TEST_MAIN()
