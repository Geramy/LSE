#include "harness.hpp"
#include "lse/backends/cpu/cpu_backend.hpp"
#include "lse/backends/hrx/hipc/hip_emitter.hpp"
#include "lse/backends/hrx/loomc/loom_emitter.hpp"
#include "lse/graph/fallback.hpp"
#include "lse/graph/interpreter.hpp"
#include "lse/graph/kernel_primitive.hpp"
#include "lse/graph/kernel_env.hpp"
#include "lse/graph/ops.hpp"

#include <array>
#include <bit>
#include <cstring>
#include <memory>
#include <vector>

using namespace lse;
using namespace lse::graph;
namespace {
enum class Write { kOutput, kInput, kWrongType, kNone };
template<Write W = Write::kOutput, bool Epilogue = false>
class WordKernel final : public KernelPrimitive<WordKernel<W, Epilogue>> {
 public:
  static constexpr std::string_view kName = Epilogue ? "fixture.words.epilogue" :
      W == Write::kInput ? "fixture.words.input" :
      W == Write::kWrongType ? "fixture.words.float" :
      W == Write::kNone ? "fixture.words.none" : "fixture.words.terminal";
  static constexpr std::string_view kEntry = "fixture_words";
  static constexpr std::string_view kSource = {};
  std::size_t arity() const noexcept override { return 1; }
  bool owns_indexing() const noexcept override { return true; }
  bool supports_epilogue() const noexcept override { return Epilogue; }
  bool has_typed_host_impl() const noexcept override { return true; }
  DType infer_dtype(std::span<const DType>) const override { return DType::kU32; }
  Result<Shape> infer_shape(std::span<const Shape> inputs) const override {
    return inputs.size() == 1 ? Result<Shape>(inputs[0]) :
        Result<Shape>(LSE_ERROR(kInvalidArgument, "fixture needs one input"));
  }
  static ThreadPlan plan_impl(const KernelShapes&) {
    ThreadPlan plan; plan.workgroup_size[0] = 32; return plan;
  }
  std::string emit_kernel(const KernelShapes& s) const override {
    if (!s.intrinsics) return {};
    kir::KernelBody body(s.types, *s.intrinsics);
    kir::Buffer<kir::u32> input(&body, &s.types, "in0");
    const auto i = body.thread_id();
    env::Emit emit{&body};
    if (auto live = emit.when(i < 4u)) {
      if constexpr (W == Write::kWrongType) {
        kir::Buffer<kir::f32> output(&body, &s.types, "out");
        output[i] = body.lit(1.0f);
      } else {
        kir::Buffer<kir::u32> output(&body, &s.types, "out");
        if constexpr (W == Write::kOutput) output[i] = input[i].read();
        if constexpr (W == Write::kInput) input[i] = output[i].read();
      }
    }
    return body.str();
  }
  Status eval_cpu_typed(std::span<const HostTensorView> inputs, HostOutputView output,
                       const std::array<float,4>& attrs,
                       const std::array<std::int32_t,4>& iattrs) const override {
    if (inputs.size() != 1 || inputs[0].dtype != DType::kU32 ||
        output.dtype != DType::kU32 || inputs[0].shape != output.shape ||
        inputs[0].bytes.size() != output.bytes.size())
      return LSE_ERROR(kInvalidArgument, "fixture typed views disagree");
    for (std::size_t i=0; i<output.shape.elem_count(); ++i) {
      std::uint32_t word;
      std::memcpy(&word, inputs[0].bytes.data()+i*sizeof(word), sizeof(word));
      word ^= std::bit_cast<std::uint32_t>(attrs[0]) ^ static_cast<std::uint32_t>(iattrs[0]);
      std::memcpy(output.bytes.data()+i*sizeof(word), &word, sizeof(word));
    }
    return OkStatus();
  }
};
class OpaqueBackend final : public backend::Backend<OpaqueBackend> {
 public:
  static constexpr std::string_view kName="fixture.typed_opaque";
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
Array leaf() {
  auto n=std::make_shared<Node>(); n->set_kind(OpKind::kBuffer);
  n->shape={4}; n->dtype=DType::kU32; n->materialized=true; return Array(n);
}
Array result(const Array& input, const Primitive& primitive) {
  auto n=std::make_shared<Node>(); n->set_kind(OpKind::kCustom);
  n->shape={4}; n->dtype=DType::kU32; n->inputs={input.node()};
  n->prim=&primitive; n->fclass=FusionClass::kBarrier; return Array(n);
}
FusionGroup solo(const Array& output) {
  FusionGroup g; g.nodes={output.node()}; g.outputs=g.nodes;
  g.inputs=output.node()->inputs; g.anchor=output.node()->kind;
  g.anchor_class=output.node()->fclass; return g;
}
backend::DeviceInfo device() {
  backend::DeviceInfo info; info.arch="gfx1201"; info.wavefront_size=32;
  info.max_threads_per_workgroup=1024; info.lds_bytes_per_workgroup=65536;
  return info;
}
}
LSE_TEST(typed_terminal_stores_emit_without_float_conversion_in_both_dialects) {
  WordKernel<> primitive; auto x=leaf(); auto y=result(x,primitive);
  backend::LoomEmitter loom; backend::HipEmitter hip;
  auto l=loom.emit(solo(y),device()), h=hip.emit(solo(y),device());
  LSE_EXPECT_OK(l.status()); LSE_EXPECT_OK(h.status());
  if (l.ok()) {
    LSE_EXPECT(l->source.find("view.store") != std::string::npos);
    LSE_EXPECT(l->source.find("scalar.sitofp") == std::string::npos);
    LSE_EXPECT(l->source.find("scalar.fptosi") == std::string::npos);
  }
  if (h.ok()) {
    LSE_EXPECT(h->source.find("unsigned int* __restrict__ out") != std::string::npos);
    LSE_EXPECT(h->source.find("out[") != std::string::npos);
  }
  LSE_EXPECT(!Partitioner::can_fuse(*y.node(), *cast(y,DType::kF32).node()));
}
LSE_TEST(typed_terminal_validation_refuses_epilogues_other_bindings_and_wrong_types) {
  WordKernel<Write::kOutput,true> epilogue;
  WordKernel<Write::kInput> input;
  WordKernel<Write::kWrongType> wrong;
  WordKernel<Write::kNone> missing;
  auto x=leaf();
  const std::array<const Primitive*,4> invalid{&epilogue,&input,&wrong,&missing};
  for (const Primitive* primitive : invalid) {
    auto y=result(x,*primitive);
    LSE_EXPECT(!backend::LoomEmitter{}.emit(solo(y),device()).ok());
    LSE_EXPECT(!backend::HipEmitter{}.emit(solo(y),device()).ok());
  }
}
LSE_TEST(typed_host_fallback_preserves_raw_words_offsets_and_immediates) {
  backend::BackendAdapter<backend::CpuBackend> backend; LSE_EXPECT_OK(backend.init(0));
  constexpr std::array<std::uint32_t,6> storage{0x12345678u,0xffffffffu,
      0x80000001u,0x3eaaaaabu,0x01000001u,0xdeadbeefu};
  auto allocation=backend.allocate(sizeof(storage),backend::MemoryClass::kDevice,
                                   backend::kDefaultStream);
  LSE_EXPECT_OK(allocation.status()); if (!allocation.ok()) return;
  auto buffer=allocation.release();
  LSE_EXPECT_OK(backend.copy(buffer,storage.data(),sizeof(storage)));
  buffer.offset=sizeof(std::uint32_t); buffer.size_bytes=4*sizeof(std::uint32_t);
  auto x=Array::from_buffer(buffer,{4},DType::kU32);
  WordKernel<> primitive; auto y=result(x,primitive);
  y.node()->attrs[0]=.125f; y.node()->iattrs[0]=37;
  LSE_EXPECT(!primitive.has_host_impl());
  const auto* fallback=default_fallback_chain().resolve(*y.node(),backend);
  LSE_EXPECT(fallback!=nullptr); if (!fallback) return;
  LSE_EXPECT_OK(fallback->execute(*y.node(),backend));
  std::array<std::uint32_t,4> output{};
  LSE_EXPECT_OK(interpreter::read_raw(*y.node(),output.data(),sizeof(output)));
  for (std::size_t i=0; i<output.size(); ++i)
    LSE_EXPECT_EQ(output[i],storage[i+1]^std::bit_cast<std::uint32_t>(.125f)^37u);
  std::array<std::uint32_t,6> preserved{};
  auto whole=buffer; whole.offset=0; whole.size_bytes=sizeof(storage);
  LSE_EXPECT_OK(backend.copy(preserved.data(),whole,sizeof(preserved)));
  LSE_EXPECT(preserved==storage);
  LSE_EXPECT(y.node()->materialized && y.node()->host_dirty && !y.node()->device_dirty);
  auto invalid=result(x,primitive); x.node()->buffer.size_bytes=sizeof(std::uint32_t);
  LSE_EXPECT(!interpreter::evaluate(invalid.node(),backend).ok());
  LSE_EXPECT(!invalid.node()->materialized);
}
LSE_TEST(panel_consumers_keep_the_raw_group_affine_host_reference_on_fallback) {
  backend::BackendAdapter<backend::CpuBackend> backend; LSE_EXPECT_OK(backend.init(0));
  auto upload=[&](Shape shape,DType dtype,const void* data,std::size_t bytes) {
    auto allocation=backend.allocate(bytes,backend::MemoryClass::kDevice,
                                     backend::kDefaultStream);
    LSE_EXPECT_OK(allocation.status()); if (!allocation.ok()) return Array{};
    auto buffer=allocation.release(); LSE_EXPECT_OK(backend.copy(buffer,data,bytes));
    return Array::from_buffer(buffer,shape,dtype);
  };
  std::array<float,64> values; values.fill(1.0f);
  std::array<std::uint32_t,8> words; words.fill(0x11111111u);
  const bfloat16_t scale(.5f),bias(.25f);
  const std::array<std::uint32_t,25> panel{};
  auto x=upload({1,64},DType::kF32,values.data(),sizeof(values));
  auto w=upload({1,8},DType::kU32,words.data(),sizeof(words));
  auto scales=upload({1,1},DType::kBF16,&scale,sizeof(scale));
  auto biases=upload({1,1},DType::kBF16,&bias,sizeof(bias));
  auto packed=upload({1,25},DType::kU32,panel.data(),sizeof(panel));
  auto output=quant_linear(x,w,scales,biases,4,64);
  output.node()->inputs.push_back(packed.node());
  const auto* fallback=default_fallback_chain().resolve(*output.node(),backend);
  LSE_EXPECT(fallback!=nullptr); if (!fallback) return;
  LSE_EXPECT_OK(fallback->execute(*output.node(),backend));
  LSE_EXPECT_EQ(interpreter::load_element(*output.node(),0),48.0f);
}
LSE_TEST(typed_host_fallback_publishes_exact_words_to_opaque_device_storage) {
  backend::BackendAdapter<OpaqueBackend> backend; LSE_EXPECT_OK(backend.init(0));
  constexpr std::array<std::uint32_t,4> words{0xffffffffu,0x80000001u,0x3eaaaaabu,0x01000001u};
  auto allocation=backend.allocate(sizeof(words),backend::MemoryClass::kDevice,
                                   backend::kDefaultStream);
  LSE_EXPECT_OK(allocation.status()); if (!allocation.ok()) return;
  auto buffer=allocation.release(); LSE_EXPECT_OK(backend.copy(buffer,words.data(),sizeof(words)));
  auto x=Array::from_buffer(buffer,{4},DType::kU32);
  WordKernel<> primitive; auto y=result(x,primitive);
  const auto* fallback=default_fallback_chain().resolve(*y.node(),backend);
  LSE_EXPECT(fallback!=nullptr); if (!fallback) return;
  LSE_EXPECT_OK(fallback->execute(*y.node(),backend));
  LSE_EXPECT(y.node()->materialized && y.node()->host_dirty && !y.node()->device_dirty);
  LSE_EXPECT_OK(interpreter::sync_to_device(*y.node(),backend));
  std::array<std::uint32_t,4> actual{};
  LSE_EXPECT_OK(backend.copy(actual.data(),y.node()->buffer,sizeof(actual)));
  LSE_EXPECT(actual==words);
  LSE_EXPECT(!y.node()->host_dirty);
  auto missing=Array::from_buffer(buffer,{4},DType::kU32);
  missing.node()->device_dirty=false;
  auto invalid=result(missing,primitive);
  LSE_EXPECT(!interpreter::evaluate(invalid.node(),backend).ok());
  LSE_EXPECT(missing.node()->host_mirror.empty());
  LSE_EXPECT(!invalid.node()->materialized);
}
LSE_TEST_MAIN()
