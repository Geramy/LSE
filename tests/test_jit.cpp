// Emitter -> comgr -> code object. Runs the full JIT path when ROCm is
// present; the source-shape checks run everywhere.
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <unistd.h>
#include <vector>

#include <array>
#include <cstring>
#include <memory>
#include <string>

#include "harness.hpp"
#include "lse/backend/backend.hpp"
#include "lse/core/debug.hpp"
#include "lse/backends/hrx/arch_database.hpp"
#include "lse/backends/hrx/comgr_compiler.hpp"
#include "lse/backends/hrx/device_info.hpp"
#include "lse/backends/hrx/hip_emitter.hpp"
#include "lse/backends/hrx/hip_sources.hpp"
#include "lse/backends/hrx/hip_types.hpp"
#include "lse/backends/hrx/kernels/wmma.hpp"
#include "lse/graph/kernel_args.hpp"
#include "lse/graph/kernel_env.hpp"
#include "lse/graph/kernel_primitive.hpp"
#include "lse/graph/jit.hpp"
#include "lse/graph/ops.hpp"
#include "lse/graph/primitive.hpp"
#include "lse/graph/primitive_library.hpp"
#include "lse/math.hpp"

using namespace lse;
using namespace lse::graph;

namespace {

struct JitSwish final : ElementwisePrimitive<JitSwish, 1> {
  static constexpr std::string_view kName = "jit.swish";
  static constexpr std::array<DialectExpr, 1> kSources{{{Dialect::kHip, "$0 / (1.0f + __expf(-$0))"}}};
  static float apply(float x) { return x / (1.0f + std::exp(-x)); }
};

struct JitHostOnly final : ElementwisePrimitive<JitHostOnly, 1> {
  static constexpr std::string_view kName = "jit.hostonly";
  static constexpr std::array<DialectExpr, 1> kSources{{{Dialect::kHip, "$0"}}};
  static float apply(float x) { return x; }
  bool has_device_impl() const noexcept override { return false; }
};

const backend::AmdDeviceInfo& gfx1151_amd() {
  static const backend::AmdDeviceInfo amd = [] {
    backend::AmdDeviceInfo a;
    a.matrix_core = backend::MatrixCore::kWMMA;
    a.has_bf16_arith = true;
    a.max_load_bytes = 16;
    a.max_store_bytes = 16;
    return a;
  }();
  return amd;
}

backend::DeviceInfo gfx1151() {
  backend::DeviceInfo d;
  d.name = "test";
  d.arch = "gfx1151";
  d.compute_units = 40;
  d.max_threads_per_workgroup = 1024;
  d.wavefront_size = 32;
  d.max_waves_per_cu = 32;
  d.lds_bytes_per_workgroup = 65536;
  d.extension_id = backend::AmdDeviceInfo::kExtensionId;
  d.extension = &gfx1151_amd();
  return d;
}

const backend::HipEmitter kEmitter;
const backend::ComgrCompiler kCompiler;

Result<EmittedKernel> emit_for(Array& root) {
  const NodePtr roots[] = {root.node()};
  auto groups = Partitioner::partition(roots);
  if (groups.empty()) return LSE_ERROR(kInternal, "no groups");
  return kEmitter.emit(groups.front(), gfx1151());
}

}  // namespace

LSE_REGISTER_PRIMITIVE(JitSwish);
LSE_REGISTER_PRIMITIVE(JitHostOnly);

LSE_TEST(builtin_primitives_emit_from_the_backend_source_table) {
  // A built-in carries no device text: the emitting backend supplies it. With
  // no table there is nothing to emit, which is what makes the group fall back
  // rather than compile source meant for another target.
  const std::string args[] = {"a", "b"};
  const graph::DialectSourceTable table = backend::hip_sources();
  for (const char* name : {"add", "mul", "silu", "gelu", "clamp"}) {
    const Primitive* p = find_primitive(name);
    LSE_EXPECT(p != nullptr);
    if (!p) continue;

    EmitContext bare;
    bare.inputs = std::span<const std::string>(args, p->arity());
    bare.out = "dst";
    LSE_EXPECT(p->emit_device(bare).empty());

    EmitContext ctx = bare;
    ctx.sources = &table;
    ctx.attrs = {-1.0f, 1.0f, 0.0f, 0.0f};
    const std::string src = p->emit_device(ctx);
    LSE_EXPECT(src.rfind("dst =", 0) == 0);
    LSE_EXPECT(!src.empty() && src.back() == ';');
    // Attr placeholders are spliced as literals, not left in the source.
    LSE_EXPECT(src.find("$a") == std::string::npos);
  }
}

LSE_TEST(emitter_produces_a_complete_translation_unit) {
  Array x = Array::full(Shape{256}, DType::kF32, 1.0f);
  Array y = x * x + x;
  auto e = emit_for(y);
  LSE_EXPECT(e.ok());
  if (!e.ok()) {
    std::printf("       %s\n", e.status().to_string().c_str());
    return;
  }

  LSE_EXPECT(e->source.find("#include <hip/hip_runtime.h>") != std::string::npos);
  LSE_EXPECT(e->source.find("extern \"C\" __global__ void") != std::string::npos);
  LSE_EXPECT(e->source.find(e->entry_name) != std::string::npos);
  LSE_EXPECT(e->source.find("if (i >= k.count) return;") != std::string::npos);
  LSE_EXPECT(e->entry_name.rfind("lse_fused_", 0) == 0);
}

LSE_TEST(phase_emits_syncthreads_only_on_cross_lane_deps) {
  auto nsync = [](const std::string& src) {
    std::size_t n = 0, pos = 0;
    while ((pos = src.find("__syncthreads", pos)) != std::string::npos) {
      ++n;
      pos += 13;
    }
    return n;
  };

  // One CU cannot host a persist grid; this is the __syncthreads path.
  backend::DeviceInfo one = gfx1151();
  one.compute_units = 1;

  Array x = Array::full(Shape{32}, DType::kF32, 1.0f);
  Array y = silu(x * x + x);
  const NodePtr elem_roots[] = {y.node()};
  const auto elem_wgs = Partitioner::phases(elem_roots);
  LSE_EXPECT(!elem_wgs.empty());
  if (!elem_wgs.empty()) {
    const FusionGroup g = Partitioner::phase_group(elem_wgs[0], elem_roots);
    auto e = backend::HipEmitter::emit_phase(g, one);
    LSE_EXPECT(e.ok());
    if (e.ok()) {
      LSE_EXPECT(!e->persist_grid);
      // No barrier after the last store; at most one per RAW edge.
      LSE_EXPECT(nsync(e->source) < 3u);
    }
  }

  Array h = Array::full(Shape{1, 32}, DType::kF32, 1.0f);
  Array rw = Array::full(Shape{32}, DType::kF32, 1.0f);
  Array w1 = Array::full(Shape{64, 32}, DType::kF32, 0.1f);
  Array w2 = Array::full(Shape{32, 64}, DType::kF32, 0.2f);
  Array z = linear(silu(linear(rms_norm(h, rw, 1e-6f), w1)), w2);
  const NodePtr ffn_roots[] = {z.node()};
  const auto ffn_wgs = Partitioner::phases(ffn_roots);
  LSE_EXPECT(!ffn_wgs.empty());
  if (ffn_wgs.empty()) return;
  const FusionGroup g = Partitioner::phase_group(ffn_wgs[0], ffn_roots);
  auto e = backend::HipEmitter::emit_phase(g, one);
  LSE_EXPECT(e.ok());
  if (!e.ok()) return;
  LSE_EXPECT(!e->persist_grid);
  const auto n = nsync(e->source);
  LSE_EXPECT(n >= 1u);
  LSE_EXPECT(n < 6u);
}

LSE_TEST(phase_dependent_kernel_stays_one_workgroup) {
  Array h = Array::full(Shape{1, 32}, DType::kF32, 1.0f);
  Array rw = Array::full(Shape{32}, DType::kF32, 1.0f);
  Array w1 = Array::full(Shape{64, 32}, DType::kF32, 0.1f);
  Array w2 = Array::full(Shape{32, 64}, DType::kF32, 0.2f);
  Array z = linear(silu(linear(rms_norm(h, rw, 1e-6f), w1)), w2);
  const NodePtr ffn_roots[] = {z.node()};
  const auto ffn_wgs = Partitioner::phases(ffn_roots);
  LSE_EXPECT(!ffn_wgs.empty());
  if (ffn_wgs.empty()) return;
  const FusionGroup g = Partitioner::phase_group(ffn_wgs[0], ffn_roots);
  auto e = backend::HipEmitter::emit_phase(g, gfx1151());
  LSE_EXPECT(e.ok());
  if (!e.ok()) return;
  LSE_EXPECT(!e->persist_grid);
  LSE_EXPECT(e->source.find("lse_grid_sync") == std::string::npos);
  LSE_EXPECT(e->dims.workgroup_count[0] == 1u);

  Array x = Array::full(Shape{1, 256}, DType::kF32, 1.0f);
  Array w = Array::full(Shape{256, 256}, DType::kF32, 0.1f);
  Array y = linear(x, w);
  const NodePtr lin_roots[] = {y.node()};
  const auto lin_wgs = Partitioner::phases(lin_roots);
  LSE_EXPECT(!lin_wgs.empty());
  if (lin_wgs.empty()) return;
  const FusionGroup lg = Partitioner::phase_group(lin_wgs[0], lin_roots);
  auto le = backend::HipEmitter::emit_phase(lg, gfx1151());
  LSE_EXPECT(le.ok());
  if (!le.ok()) return;
  LSE_EXPECT(!le->persist_grid);
  LSE_EXPECT(le->dims.workgroup_count[0] > 1u);
}

LSE_TEST(rdna_is_wave32_cdna_is_wave64_rdna4_can_be_either) {
  LSE_EXPECT(backend::arch_family("gfx1100") == backend::ArchFamily::kRdna3);
  LSE_EXPECT(backend::arch_family("gfx1151") == backend::ArchFamily::kRdna35);
  LSE_EXPECT(backend::arch_family("gfx1201") == backend::ArchFamily::kRdna4);
  LSE_EXPECT(backend::arch_family("gfx90a") == backend::ArchFamily::kCdna2);
  LSE_EXPECT(backend::arch_family("gfx942") == backend::ArchFamily::kCdna3);
  LSE_EXPECT(backend::arch_family("gfx950") == backend::ArchFamily::kCdna4);
  LSE_EXPECT(backend::wavefront_legal("gfx1100", 32));
  LSE_EXPECT(!backend::wavefront_legal("gfx1100", 64));
  LSE_EXPECT(backend::wavefront_legal("gfx1151", 32));
  LSE_EXPECT(!backend::wavefront_legal("gfx1151", 64));
  LSE_EXPECT(backend::wavefront_legal("gfx1201", 32));
  LSE_EXPECT(backend::wavefront_legal("gfx1201", 64));
  LSE_EXPECT(!backend::wavefront_legal("gfx90a", 32));
  LSE_EXPECT(backend::wavefront_legal("gfx942", 64));
  LSE_EXPECT(backend::wavefront_legal("gfx950", 64));

  const char* prev = std::getenv("LSE_WAVEFRONT");
  const std::string saved = prev ? prev : "";
  ::unsetenv("LSE_WAVEFRONT");
  LSE_EXPECT_EQ(backend::select_wavefront("gfx1151", 32), 32);
  LSE_EXPECT_EQ(backend::select_wavefront("gfx942", 64), 64);
  LSE_EXPECT_EQ(backend::select_wavefront("gfx1201", 32), 32);
  ::setenv("LSE_WAVEFRONT", "64", 1);
  LSE_EXPECT_EQ(backend::select_wavefront("gfx1201", 32), 64);
  LSE_EXPECT_EQ(backend::select_wavefront("gfx1151", 32), 32);
  LSE_EXPECT_EQ(backend::select_wavefront("gfx942", 64), 64);
  ::setenv("LSE_WAVEFRONT", "32", 1);
  LSE_EXPECT_EQ(backend::select_wavefront("gfx942", 64), 64);
  LSE_EXPECT_EQ(backend::select_wavefront("gfx1201", 64), 32);
  if (!saved.empty()) ::setenv("LSE_WAVEFRONT", saved.c_str(), 1);
  else ::unsetenv("LSE_WAVEFRONT");
}

LSE_TEST(hrx_reported_cus_are_not_overwritten_by_the_table) {
  backend::DeviceInfo info;
  info.arch = "gfx1151";
  info.compute_units = 7;
  info.max_threads_per_workgroup = 256;
  info.lds_bytes_per_workgroup = 12345;
  info.wavefront_size = 32;
  backend::AmdDeviceInfo amd;
  backend::apply_arch_defaults(info, amd);
  LSE_EXPECT_EQ(info.compute_units, 7);
  LSE_EXPECT_EQ(info.max_threads_per_workgroup, 256);
  LSE_EXPECT_EQ(info.lds_bytes_per_workgroup, 12345u);
  LSE_EXPECT_EQ(info.wavefront_size, 32);
  LSE_EXPECT(amd.matrix_core == backend::MatrixCore::kWMMA);
}

LSE_TEST(unknown_family_still_gets_isa_from_the_arch_string) {
  backend::DeviceInfo info;
  info.arch = "gfx1102";
  backend::AmdDeviceInfo amd;
  backend::apply_arch_defaults(info, amd);
  LSE_EXPECT(backend::arch_family(info.arch) == backend::ArchFamily::kRdna3);
  LSE_EXPECT_EQ(info.wavefront_size, 32);
  LSE_EXPECT(amd.matrix_core == backend::MatrixCore::kWMMA);
}

LSE_TEST(rdna4_profile_honours_lse_wavefront) {
  const char* prev = std::getenv("LSE_WAVEFRONT");
  const std::string saved = prev ? prev : "";
  ::unsetenv("LSE_WAVEFRONT");
  backend::DeviceInfo info;
  info.arch = "gfx1201";
  backend::AmdDeviceInfo amd;
  backend::apply_arch_defaults(info, amd);
  LSE_EXPECT_EQ(info.wavefront_size, 32);
  ::setenv("LSE_WAVEFRONT", "64", 1);
  amd = {};
  info.wavefront_size = 0;
  backend::apply_arch_defaults(info, amd);
  LSE_EXPECT_EQ(info.wavefront_size, 64);
  if (!saved.empty()) ::setenv("LSE_WAVEFRONT", saved.c_str(), 1);
  else ::unsetenv("LSE_WAVEFRONT");
}

LSE_TEST(emitter_chooses_a_legal_workgroup_size) {
  Array x = Array::full(Shape{1024}, DType::kF32, 1.0f);
  Array y = x + x;
  auto e = emit_for(y);
  LSE_EXPECT(e.ok());
  if (!e.ok()) return;

  const backend::DeviceInfo d = gfx1151();
  const std::uint32_t threads = e->dims.workgroup_size[0];
  LSE_EXPECT(threads > 0);
  LSE_EXPECT(threads <= d.max_threads_per_workgroup);
  LSE_EXPECT(threads % d.wavefront_size == 0);
  LSE_EXPECT(backend::occupancy_per_cu(d, threads, 0) > 0);

  // Enough workgroups to cover every element.
  LSE_EXPECT(e->dims.workgroup_count[0] * threads >= 1024u);
}

LSE_TEST(a_custom_primitive_lands_in_the_generated_source) {
  Array x = Array::full(Shape{64}, DType::kF32, 1.0f);
  auto s = custom("jit.swish", {x * x});
  LSE_EXPECT(s.ok());
  if (!s.ok()) return;
  Array y = *s + x;

  auto e = emit_for(y);
  LSE_EXPECT(e.ok());
  if (!e.ok()) return;
  // The primitive's body, inlined rather than called out to.
  LSE_EXPECT(e->source.find("__expf") != std::string::npos);
}

LSE_TEST(shared_device_helpers_are_emitted_once) {
  auto lib = PrimitiveLibrary::load_file("tests/fixtures/example_primitives.hip");
  if (!lib.ok()) return;

  Array x = Array::full(Shape{64}, DType::kF32, 1.0f);
  auto a = custom("example.mish", {x});
  auto b = custom("example.swiglu", {x, x});
  if (!a.ok() || !b.ok()) return;
  Array y = *a + *b;

  auto e = emit_for(y);
  LSE_EXPECT(e.ok());
  if (!e.ok()) return;

  // Both primitives come from one file, so its helpers appear exactly once.
  std::size_t count = 0;
  for (std::size_t p = e->source.find("lse_softplus(float");
       p != std::string::npos; p = e->source.find("lse_softplus(float", p + 1)) {
    ++count;
  }
  LSE_EXPECT_EQ(count, 1u);
  LSE_EXPECT(e->source.find("lse_mish") != std::string::npos);
  LSE_EXPECT(e->source.find("lse_swiglu") != std::string::npos);
}

// Stands in for a wave-cooperative kernel: it writes its own output and picks
// its own thread map, which is the part WMMA needs and the per-element
// contract cannot express.
struct SelfIndexed final : KernelPrimitive<SelfIndexed> {
  static constexpr std::string_view kName = "test.self_indexed";
  static constexpr std::string_view kEntry = "test_self_indexed";
  static constexpr std::string_view kSource =
      {};

  std::size_t arity() const noexcept override { return 1; }
  bool owns_indexing() const noexcept override { return true; }

  // Stores through the hook rather than assigning `out`: that is the contract
  // for a self-indexing primitive, and it is what lets an epilogue fuse in.
  std::string emit_kernel(const KernelShapes& s) const override {
    if (!s.store) return {};
    return "  if (i < k.count) {\n    " + s.store("i", "in0[i] * 2.0f") + "\n  }";
  }

  Result<Shape> infer_shape(std::span<const Shape> in) const override {
    return in.empty() ? Shape{} : in[0];
  }
  DType infer_dtype(std::span<const DType> in) const override {
    return in.empty() ? DType::kF32 : in[0];
  }
  static ThreadPlan plan_impl(const KernelShapes& s) {
    ThreadPlan tp;
    tp.workgroup_size[0] = 64;
    tp.workgroup_count[0] =
        (static_cast<std::uint32_t>(s.output.elem_count()) + 63) / 64;
    return tp;
  }
};
LSE_REGISTER_PRIMITIVE(SelfIndexed);

LSE_TEST(a_self_indexing_kernel_owns_its_body_and_thread_map) {
  Array x = Array::full(Shape{256}, DType::kF32, 3.0f);
  auto y = custom("test.self_indexed", {x});
  LSE_EXPECT(y.ok());
  if (!y.ok()) return;

  const NodePtr roots[] = {y->node()};
  auto groups = Partitioner::partition(roots);
  LSE_EXPECT(!groups.empty());
  if (groups.empty()) return;

  auto e = kEmitter.emit(groups.back(), gfx1151());
  LSE_EXPECT(e.ok());
  if (!e.ok()) {
    std::printf("       %s\n", e.status().to_string().c_str());
    return;
  }
  // The body goes straight into the entry point: no per-element device
  // function, and the value reaches memory through the store hook, which with
  // nothing fused is just the store.
  LSE_EXPECT(e->source.find("in0[i] * 2.0f") != std::string::npos);
  LSE_EXPECT(e->source.find("out[i] = ") != std::string::npos);
  LSE_EXPECT(e->source.find("__device__ float test_self_indexed(") ==
             std::string::npos);
  // And the thread map is the primitive's, not the emitter's occupancy pick.
  LSE_EXPECT(e->dims.workgroup_size[0] == 64u);
  LSE_EXPECT(e->dims.workgroup_count[0] == 4u);
}

LSE_TEST(a_row_normalization_emits_a_kernel) {
  // RMSNorm reduces over the last axis but produces a tensor the same shape as
  // its input, so it fits the per-output-element contract. This is what keeps
  // the model's 41 norms off the host.
  Array a = Array::full(Shape{4, 16}, DType::kF32, 1.0f);
  Array w = Array::full(Shape{16}, DType::kF32, 1.0f);
  Array c = rms_norm(a, w, 1e-6f, false);
  const NodePtr roots[] = {c.node()};
  auto groups = Partitioner::partition(roots);

  bool emitted = false;
  for (const auto& g : groups) {
    auto e = kEmitter.emit(g, gfx1151());
    if (!e.ok()) continue;
    // The helper's name carries the shapes and attrs it was specialized for,
    // so match the primitive's stem rather than the whole identifier.
    if (e->source.find("__device__ float lse_rms_norm_") == std::string::npos) {
      continue;
    }
    emitted = true;
    // The reduction loop and the extent baked in as a literal. Loop vars are
    // auto-named by the env, so match the bound, not the name.
    LSE_EXPECT(e->source.find(" < 16u") != std::string::npos);
    LSE_EXPECT(e->source.find("rsqrtf") != std::string::npos);
  }
  LSE_EXPECT(emitted);
}

LSE_TEST(a_shape_reducing_reduction_is_still_refused) {
  // sum() shrinks the shape, so one thread per output element cannot express
  // it. The refusal has to stay explicit so the group falls back to the host.
  Array a = Array::full(Shape{4, 16}, DType::kF32, 1.0f);
  Array c = sum(a, -1);
  const NodePtr roots[] = {c.node()};
  auto groups = Partitioner::partition(roots);

  bool saw_refusal = false;
  for (const auto& g : groups) {
    auto e = kEmitter.emit(g, gfx1151());
    if (!e.ok() && e.status().code() == StatusCode::kUnimplemented) {
      saw_refusal = true;
    }
  }
  LSE_EXPECT(saw_refusal);
}

LSE_TEST(a_matmul_kernel_becomes_a_device_function_the_emitter_wraps) {
  Array a = Array::full(Shape{4, 8}, DType::kF32, 1.0f);
  Array b = Array::full(Shape{8, 16}, DType::kF32, 2.0f);
  Array c = matmul(a, b);
  const NodePtr roots[] = {c.node()};
  auto groups = Partitioner::partition(roots);
  LSE_EXPECT(!groups.empty());
  if (groups.empty()) return;

  auto e = kEmitter.emit(groups.back(), gfx1151());
  LSE_EXPECT(e.ok());
  if (!e.ok()) {
    std::printf("       %s\n", e.status().to_string().c_str());
    return;
  }
  // The primitive supplies a body; the emitter owns the entry point.
  LSE_EXPECT(e->source.find("__device__ float lse_matmul_") != std::string::npos);
  LSE_EXPECT(e->source.find("extern \"C\" __global__ void lse_fused_") !=
             std::string::npos);
  LSE_EXPECT(e->source.find("lse_matmul_") != std::string::npos);
  // Extents baked in, not read from a constant; the env names the loop var.
  LSE_EXPECT(e->source.find(" < 8u") != std::string::npos);
}

LSE_TEST(elementwise_work_after_a_matmul_fuses_into_its_kernel) {
  // The epilogue check: silu/mul/add after a contraction must ride along in the
  // matmul's launch rather than each becoming a kernel of its own.
  Array a = Array::full(Shape{4, 8}, DType::kF32, 1.0f);
  Array b = Array::full(Shape{8, 16}, DType::kF32, 2.0f);
  Array bias = Array::full(Shape{16}, DType::kF32, 0.5f);
  Array y = graph::silu(matmul(a, b) + bias) * bias;

  const NodePtr roots[] = {y.node()};
  auto groups = Partitioner::partition(roots);

  std::size_t with_matmul = 0;
  std::size_t matmul_group = groups.size();
  for (std::size_t gi = 0; gi < groups.size(); ++gi) {
    for (const NodePtr& n : groups[gi].nodes) {
      if (n->kind != OpKind::kMatMul) continue;
      with_matmul = groups[gi].nodes.size();
      matmul_group = gi;
    }
  }
  std::printf("       matmul group holds %zu nodes across %zu group(s)\n",
              with_matmul, groups.size());
  // matmul + add + silu + mul.
  LSE_EXPECT(with_matmul >= 4u);

  if (matmul_group == groups.size()) return;
  auto e = kEmitter.emit(groups[matmul_group], gfx1151());
  LSE_EXPECT(e.ok());
  if (!e.ok()) {
    std::printf("       %s\n", e.status().to_string().c_str());
    return;
  }
  if (std::getenv("LSE_DUMP_KERNEL") != nullptr) {
    std::printf("---\n%s---\n", e->source.c_str());
  }
  // One kernel: the contraction call and the activation in the same body.
  LSE_EXPECT(e->source.find("lse_matmul_") != std::string::npos);
  LSE_EXPECT(e->source.find("__expf") != std::string::npos);
}

// ---------------------------------------------------------------------------
// Independent siblings in one launch
// ---------------------------------------------------------------------------

namespace {

// What the scheduler's join_wide_linear hands the emitter: independent wide
// linears off one activation, every one of them a group output.
FusionGroup sibling_group(std::initializer_list<Array> outs) {
  FusionGroup g;
  for (const Array& a : outs) {
    g.nodes.push_back(a.node());
    g.outputs.push_back(a.node());
  }
  auto member = [&](const Node* p) {
    for (const NodePtr& n : g.nodes) {
      if (n.get() == p) return true;
    }
    return false;
  };
  for (const NodePtr& n : g.nodes) {
    for (const NodePtr& in : n->inputs) {
      if (member(in.get())) continue;
      bool seen = false;
      for (const NodePtr& e : g.inputs) seen |= e.get() == in.get();
      if (!seen) g.inputs.push_back(in);
    }
  }
  g.anchor = g.nodes.front()->kind;
  g.anchor_class = g.nodes.front()->fclass;
  return g;
}

std::size_t count_of(const std::string& hay, std::string_view needle) {
  std::size_t n = 0;
  for (std::size_t at = hay.find(needle); at != std::string::npos;
       at = hay.find(needle, at + needle.size())) {
    ++n;
  }
  return n;
}

}  // namespace

// The sibling path emits one body per stage and walks `stages`, not
// `group.nodes`. A member that did not specialize to a self-indexing form
// would get no body, no binding and no store — while the scheduler marks
// every group member materialized, so the next group would read a buffer
// nothing ever wrote. Refusing costs one launch; guessing costs an argmax.
LSE_TEST(sibling_fusion_is_refused_when_stages_do_not_cover_the_group) {
  ::unsetenv("LSE_WMMA");
  Array x = Array::full(Shape{1, 64}, DType::kF32, 1.0f);
  Array wa = Array::full(Shape{128, 64}, DType::kF32, 0.5f);
  Array wb = Array::full(Shape{64, 64}, DType::kF32, 0.25f);
  // N below the GEMV's minimum and below one matrix-core fragment: this one
  // stays the scalar per-element `linear`, so it can never be a stage.
  Array wc = Array::full(Shape{8, 64}, DType::kF32, 0.125f);
  Array a = linear(x, wa);
  Array b = linear(x, wb);
  Array c = linear(x, wc);

  // Two self-indexing siblings do take the fused path: no per-element
  // scaffold, so no `i >= k.count` early return.
  auto fused = kEmitter.emit(sibling_group({a, b}), gfx1151());
  LSE_EXPECT(fused.ok());
  if (!fused.ok()) {
    std::printf("       %s\n", fused.status().to_string().c_str());
    return;
  }
  LSE_EXPECT(fused->source.find("if (i >= k.count) return;") ==
             std::string::npos);
  LSE_EXPECT(fused->binding_order.size() == 5u);

  // Adding a member that is not one of them must turn the path off rather
  // than drop it.
  const FusionGroup uncovered = sibling_group({a, b, c});
  auto e = kEmitter.emit(uncovered, gfx1151());
  LSE_EXPECT(e.ok());
  if (!e.ok()) {
    std::printf("       %s\n", e.status().to_string().c_str());
    return;
  }
  LSE_EXPECT(e->source.find("if (i >= k.count) return;") != std::string::npos);

  // Every output of the declined group still gets a binding and a store.
  auto stored = [&](const Array& o) {
    for (std::size_t i = 0; i < e->binding_order.size(); ++i) {
      if (e->binding_order[i].get() != o.node().get()) continue;
      return e->source.find("b" + std::to_string(i) + "[i] = ") !=
             std::string::npos;
    }
    return false;
  };
  LSE_EXPECT(stored(a));
  LSE_EXPECT(stored(b));
  LSE_EXPECT(stored(c));
}

// The per-element scaffold's `__device__` helper bakes the extents, the
// operand dtypes and iattrs in as literals while `entry_name()` is a constant
// per primitive. Deduplicating on the name alone made a sibling with N=32 call
// the body specialized for N=128 — a read far past the end of its weight — and
// made the slot-1 expert run the slot-0 body.
LSE_TEST(device_helper_identity_carries_shapes_and_attrs) {
  ::setenv("LSE_WMMA", "0", 1);
  // M >= 16 keeps both off the decode GEMV, and LSE_WMMA=0 keeps them off the
  // matrix core, so both land in the scaffold as device functions.
  Array x = Array::full(Shape{16, 64}, DType::kF32, 1.0f);
  Array wa = Array::full(Shape{128, 64}, DType::kF32, 0.5f);
  Array wb = Array::full(Shape{32, 64}, DType::kF32, 0.25f);
  auto shapes = kEmitter.emit(sibling_group({linear(x, wa), linear(x, wb)}),
                              gfx1151());

  // Same shapes, same dtypes, different expert slot.
  Array w = Array::full(Shape{4, 32, 64}, DType::kF32, 0.5f);
  Array idx = Array::full(Shape{16, 2}, DType::kF32, 1.0f);
  auto attrs = kEmitter.emit(
      sibling_group({linear_indexed(x, w, idx, 0), linear_indexed(x, w, idx, 1)}),
      gfx1151());
  ::unsetenv("LSE_WMMA");

  LSE_EXPECT(shapes.ok());
  if (!shapes.ok()) {
    std::printf("       %s\n", shapes.status().to_string().c_str());
    return;
  }
  // Two definitions, not one shared by both: the names cannot have collided.
  LSE_EXPECT(count_of(shapes->source, "__device__ float lse_linear_") == 2u);
  // And the call is guarded by the *caller's* own element count. Guarding only
  // the store leaves the helper computing a row index off the end of x.
  LSE_EXPECT(shapes->source.find("i < 512u ? lse_linear_") != std::string::npos);

  LSE_EXPECT(attrs.ok());
  if (!attrs.ok()) {
    std::printf("       %s\n", attrs.status().to_string().c_str());
    return;
  }
  LSE_EXPECT(count_of(attrs->source, "__device__ float lse_linear_indexed_") ==
             2u);

  if (!kCompiler.available()) return;
  for (const std::string* src : {&shapes->source, &attrs->source}) {
    auto code = kCompiler.compile(*src, "gfx1151");
    if (!code.ok()) {
      std::printf("       compile failed:\n%s\n",
                  code.status().message().c_str());
    }
    LSE_EXPECT(code.ok());
  }
}

LSE_TEST(device_only_primitive_cannot_be_emitted_without_device_impl) {
  // The inverse of the CPU case: a host-only primitive must not reach codegen.
  Array x = Array::full(Shape{8}, DType::kF32, 1.0f);
  auto h = custom("jit.hostonly", {x});
  if (!h.ok()) return;
  auto e = emit_for(*h);
  LSE_EXPECT(!e.ok());
  LSE_EXPECT(e.status().code() == StatusCode::kUnimplemented);
}

LSE_TEST(compiler_is_available_when_rocm_is) {
  if (!kCompiler.available()) {
    std::printf("       (skipped: comgr not in this build)\n");
    return;
  }
  LSE_EXPECT(kCompiler.available());
}

LSE_TEST(generated_source_compiles_to_a_real_code_object) {
  if (!kCompiler.available()) return;

  Array x = Array::full(Shape{256}, DType::kF32, 1.0f);
  auto s = custom("jit.swish", {x * x});
  if (!s.ok()) return;
  Array y = *s + x;

  auto e = emit_for(y);
  LSE_EXPECT(e.ok());
  if (!e.ok()) {
    std::printf("       emit: %s\n", e.status().to_string().c_str());
    return;
  }

  auto code = kCompiler.compile(e->source, "gfx1151");
  if (!code.ok()) {
    std::printf("       compile failed:\n%s\n", code.status().message().c_str());
    std::printf("       ---- source ----\n%s\n", e->source.c_str());
  }
  LSE_EXPECT(code.ok());
  if (!code.ok()) return;

  // An ELF code object, not an empty buffer.
  LSE_EXPECT(code->size() > 512u);
  LSE_EXPECT(static_cast<unsigned char>((*code)[0]) == 0x7F);
  LSE_EXPECT(static_cast<char>((*code)[1]) == 'E');
  LSE_EXPECT(static_cast<char>((*code)[2]) == 'L');
  LSE_EXPECT(static_cast<char>((*code)[3]) == 'F');
  std::printf("       code object: %zu bytes for gfx1151\n", code->size());
}

LSE_TEST(compile_errors_carry_the_compiler_diagnostics) {
  if (!kCompiler.available()) return;
  auto bad = kCompiler.compile("this is not valid hip {{{", "gfx1151");
  LSE_EXPECT(!bad.ok());
  LSE_EXPECT(bad.status().code() == StatusCode::kCompileError);
  // The message must contain the compiler's own text, not just a status code.
  LSE_EXPECT(bad.status().message().size() > 40u);
}

LSE_TEST(compile_rejects_empty_input) {
  auto e = kCompiler.compile("", "gfx1151");
  LSE_EXPECT(!e.ok());
  auto a = kCompiler.compile("__global__ void k(){}", "");
  LSE_EXPECT(!a.ok());
}

LSE_TEST(lds_calculator_enforces_the_workgroup_budget) {
  kir::Lds pad(64);
  LSE_EXPECT(pad.fits(64));
  LSE_EXPECT(pad.reserve(32));
  LSE_EXPECT_EQ(pad.used(), 32u);
  LSE_EXPECT(pad.reserve(16));
  LSE_EXPECT_EQ(pad.used(), 48u);
  LSE_EXPECT(!pad.reserve(32));
  LSE_EXPECT(!pad.ok());
  LSE_EXPECT_EQ(pad.budget(), 64u);
  LSE_EXPECT_EQ(pad.remaining(), 16u);
}

LSE_TEST(lds_unknown_budget_is_not_enforced) {
  kir::Lds pad(0);
  LSE_EXPECT(pad.reserve(1u << 20));
  LSE_EXPECT(pad.ok());
}

LSE_TEST(lds_aligns_reservations_to_16_bytes) {
  kir::Lds pad(64);
  LSE_EXPECT(pad.reserve(1));
  LSE_EXPECT_EQ(pad.used(), 16u);
  LSE_EXPECT(pad.reserve(1));
  LSE_EXPECT_EQ(pad.used(), 32u);
}

LSE_TEST(gfx1151_lds_budget_is_64k) {
  const backend::DeviceInfo d = gfx1151();
  LSE_EXPECT_EQ(backend::workgroup_lds_bytes(&d), 65536u);
  LSE_EXPECT(backend::workgroup_lds_bytes(nullptr) == 0u);
}

namespace {

struct CacheStubBackend final : backend::IBackend {
  backend::DeviceInfo info;
  CacheStubBackend() {
    info.arch = "gfx1151";
    info.name = "stub";
  }
  Status init(int) override { return OkStatus(); }
  void shutdown() noexcept override {}
  const backend::DeviceInfo& device_info() const noexcept override {
    return info;
  }
  Result<backend::DeviceBuffer> allocate(std::size_t,
                                         backend::MemoryClass) override {
    return LSE_ERROR(kUnimplemented, "stub");
  }
  void deallocate(backend::DeviceBuffer&) noexcept override {}
  Status copy_h2d(const void*, backend::DeviceBuffer&, std::size_t,
                  std::size_t) override {
    return LSE_ERROR(kUnimplemented, "stub");
  }
  Status copy_d2h(const backend::DeviceBuffer&, void*, std::size_t,
                  std::size_t) override {
    return LSE_ERROR(kUnimplemented, "stub");
  }
  Result<backend::KernelHandle> load_executable(
      std::string_view name, std::span<const std::byte>) override {
    backend::KernelHandle h;
    h.executable = 1;
    h.name = std::string(name);
    return h;
  }
  Status launch(const backend::KernelHandle&, const backend::LaunchDims&,
                const backend::DispatchArgs&,
                const backend::DispatchTarget&) override {
    return LSE_ERROR(kUnimplemented, "stub");
  }
  Status synchronize() override { return OkStatus(); }
  std::string_view name() const noexcept override { return "stub"; }
  const IKernelEmitter* emitter() const noexcept override { return nullptr; }
  const IKernelCompiler* compiler() const noexcept override { return nullptr; }
};

struct CountingCompiler final : IKernelCompiler {
  mutable int n = 0;
  Result<std::vector<std::byte>> compile(std::string_view,
                                         std::string_view) const override {
    ++n;
    return std::vector<std::byte>(16, std::byte{1});
  }
  bool available() const override { return true; }
  std::string identity() const override { return "counting-compiler"; }
};

}  // namespace

LSE_TEST(jit_compiles_only_on_miss_source_change_or_device_change) {
  namespace fs = std::filesystem;
  const fs::path dir =
      fs::temp_directory_path() / ("lse-jit-" + std::to_string(::getpid()));
  fs::create_directories(dir);

  CacheStubBackend be;
  CountingCompiler cc;
  EmittedKernel ek;
  ek.source = "kernel v1 { }";
  ek.entry_name = "k";
  const std::uint64_t sig = 42;

  {
    JitCache cache(be, cc, dir.string());
    auto a = cache.get_or_compile(sig, ek);
    LSE_EXPECT(a.ok());
    LSE_EXPECT_EQ(cc.n, 1);
    auto b = cache.get_or_compile(sig, ek);
    LSE_EXPECT(b.ok());
    LSE_EXPECT_EQ(cc.n, 1);
    LSE_EXPECT(cache.stats().memory_hits >= 1u);
    LSE_EXPECT(cache.try_get(sig) != nullptr);
  }
  {
    JitCache cache(be, cc, dir.string());
    auto c = cache.get_or_compile(sig, ek);
    LSE_EXPECT(c.ok());
    LSE_EXPECT_EQ(cc.n, 1);
    LSE_EXPECT(cache.stats().disk_hits >= 1u);
  }
  {
    JitCache cache(be, cc, dir.string());
    ek.source = "kernel v2 { }";
    auto d = cache.get_or_compile(sig, ek);
    LSE_EXPECT(d.ok());
    LSE_EXPECT_EQ(cc.n, 2);
  }
  {
    ek.source = "kernel v2 { }";
    be.info.arch = "gfx1201";
    JitCache cache(be, cc, dir.string());
    auto e = cache.get_or_compile(sig, ek);
    LSE_EXPECT(e.ok());
    LSE_EXPECT_EQ(cc.n, 3);
  }

  std::error_code ec;
  fs::remove_all(dir, ec);
}

LSE_TEST(debug_writes_generated_hip_for_review) {
  namespace fs = std::filesystem;
  const fs::path dump =
      fs::temp_directory_path() / ("lse-hip-" + std::to_string(::getpid()));
  const fs::path cache =
      fs::temp_directory_path() / ("lse-jit-" + std::to_string(::getpid()));
  fs::create_directories(dump);
  fs::create_directories(cache);
  ::setenv("LSE_HIP_DUMP", dump.string().c_str(), 1);

  CacheStubBackend be;
  CountingCompiler cc;
  EmittedKernel ek;
  ek.source = "extern \"C\" __global__ void lemonseed() {}";
  ek.entry_name = "lemonseed";
  JitCache jit(be, cc, cache.string());
  LSE_EXPECT(jit.get_or_compile(1, ek).ok());

  const fs::path hip = dump / "lemonseed.hip";
  LSE_EXPECT(fs::exists(hip));
  const auto bytes = fs::file_size(hip);
  LSE_EXPECT_EQ(bytes, ek.source.size());

  ::unsetenv("LSE_HIP_DUMP");
  std::error_code ec;
  fs::remove_all(dump, ec);
  fs::remove_all(cache, ec);
}

LSE_TEST_MAIN()

// linear is a barrier, so its operands materialize as their own groups first;
// the kernel under test is the last one.
static Result<EmittedKernel> emit_last_for(Array& root) {
  const NodePtr roots[] = {root.node()};
  auto groups = Partitioner::partition(roots);
  if (groups.empty()) return LSE_ERROR(kInternal, "no groups");
  return kEmitter.emit(groups.back(), gfx1151());
}

// The matrix-core kernel is authored in C++ against kir, so this asserts on
// what the tables were asked for rather than on the kernel's own text: the
// intrinsic spelling and the vector typedef must both have come from the HIP
// tables, and the launch must be sized in waves-per-tile, not output elements.
LSE_TEST(live_arch_picks_the_widest_load) {
  // HRX gives us gfx1151 at init; the width is that ISA's dwordx4, not a
  // compile-time #ifdef and not an HRX property.
  const backend::DeviceInfo d1151 = gfx1151();
  LSE_EXPECT_EQ(backend::max_load_bytes(d1151), 16u);
  LSE_EXPECT_EQ(backend::max_store_bytes(d1151), 16u);
  LSE_EXPECT_EQ(backend::max_vec_elems(&d1151, 4), 4u);

  backend::DeviceInfo unknown;
  unknown.arch = "unknown";
  LSE_EXPECT_EQ(backend::max_load_bytes(unknown), 4u);
  backend::DeviceInfo any_gfx;
  any_gfx.arch = "gfx1100";
  LSE_EXPECT_EQ(backend::max_load_bytes(any_gfx), 16u);
}

// Decode-shaped (M=1): still WMMA. One wave owns a 16-wide N tile;
// padding the M side is cheaper than a scalar column walk.
LSE_TEST(decode_linear_uses_coalesced_gemv_not_wmma) {
  ::unsetenv("LSE_WMMA");
  Array x = Array::full(Shape{1, 32}, DType::kF32, 1.0f);
  Array w = Array::full(Shape{64, 32}, DType::kF32, 1.0f);
  Array y = linear(x, w);
  auto e = emit_last_for(y);
  LSE_EXPECT(e.ok());
  if (!e.ok()) return;
  // M below one tile takes the wave-per-column GEMV: the padded WMMA tile
  // masks 15/16 rows and its K-strided weight loads measured ~9 GB/s on the
  // M=1 vocab head. The GEMV reduces across the wave, so shfl is the marker.
  LSE_EXPECT(e->source.find("__builtin_amdgcn_wmma_f32_16x16x16_f16_w32") ==
             std::string::npos);
  LSE_EXPECT(e->source.find("__shfl_xor") != std::string::npos);
  LSE_EXPECT(e->dims.workgroup_size[0] == 256u);
  // 8 waves to a group, one wave per output column → 64/8 workgroups.
  LSE_EXPECT(e->dims.workgroup_count[0] == 8u);
  LSE_EXPECT(e->dims.workgroup_count[1] == 1u);
  if (!kCompiler.available()) return;
  auto code = kCompiler.compile(e->source, "gfx1151");
  if (!code.ok()) {
    std::printf("       compile failed:\n%s\n", code.status().message().c_str());
    std::printf("       ---- source ----\n%s\n", e->source.c_str());
  }
  LSE_EXPECT(code.ok());
}

LSE_TEST(linear_emits_the_devices_widest_f32_load) {
  ::unsetenv("LSE_WMMA");
  Array x = Array::full(Shape{4, 16}, DType::kF32, 1.0f);
  Array w = Array::full(Shape{8, 16}, DType::kF32, 1.0f);
  Array y = linear(x, w);
  auto e = emit_last_for(y);
  LSE_EXPECT(e.ok());
  if (!e.ok()) return;
  LSE_EXPECT(e->source.find("ext_vector_type(4)") != std::string::npos);
  LSE_EXPECT(e->source.find("*(const") != std::string::npos);
}

LSE_TEST(matrix_core_linear_is_authored_in_cpp_and_spelled_by_the_tables) {
  ::unsetenv("LSE_WMMA");
  Array x = Array::full(Shape{32, 64}, DType::kF32, 1.0f);
  Array w = Array::full(Shape{48, 64}, DType::kF32, 1.0f);
  Array y = linear(x, w);
  auto e = emit_last_for(y);

  LSE_EXPECT(e.ok());
  if (!e.ok()) {
    std::printf("       emit: %s\n", e.status().to_string().c_str());
    return;
  }
  if (std::getenv("LSE_TRACE_WMMA") != nullptr) {
    std::printf("---- wmma source ----\n%s\n", e->source.c_str());
  }

  // From hip_sources, not from the kernel.
  LSE_EXPECT(e->source.find("__builtin_amdgcn_wmma_f32_16x16x16_f16_w32") !=
             std::string::npos);
  // From hip_types, not from the kernel.
  LSE_EXPECT(e->source.find("ext_vector_type(8)") != std::string::npos);
  LSE_EXPECT(e->source.find("ext_vector_type(16)") != std::string::npos);
  LSE_EXPECT(e->source.find("_Float16") != std::string::npos);

  // 2 row-tiles x 3 col-tiles = 6 waves, 8 waves to a 256-thread group.
  LSE_EXPECT(e->dims.workgroup_size[0] == 256u);
  LSE_EXPECT(e->dims.workgroup_count[0] == 1u);

  if (!kCompiler.available()) return;
  auto code = kCompiler.compile(e->source, "gfx1151");
  if (!code.ok()) {
    std::printf("       compile failed:\n%s\n", code.status().message().c_str());
    std::printf("       ---- source ----\n%s\n", e->source.c_str());
  }
  LSE_EXPECT(code.ok());
}

LSE_TEST(matrix_core_linear_defaults_on_for_gfx11) {
  ::unsetenv("LSE_WMMA");
  Array x = Array::full(Shape{32, 64}, DType::kF32, 1.0f);
  Array w = Array::full(Shape{48, 64}, DType::kF32, 1.0f);
  Array y = linear(x, w);
  auto e = emit_last_for(y);
  LSE_EXPECT(e.ok());
  if (!e.ok()) return;
  LSE_EXPECT(e->source.find("__builtin_amdgcn_wmma_f32_16x16x16_f16_w32") !=
             std::string::npos);
}

// LSE_WMMA=0 is the f32 oracle. The lemonseed differential uses it.
LSE_TEST(matrix_core_linear_is_off_when_disabled) {
  ::setenv("LSE_WMMA", "0", 1);
  Array x = Array::full(Shape{32, 64}, DType::kF32, 1.0f);
  Array w = Array::full(Shape{48, 64}, DType::kF32, 1.0f);
  Array y = linear(x, w);
  auto e = emit_last_for(y);
  ::unsetenv("LSE_WMMA");
  LSE_EXPECT(e.ok());
  if (!e.ok()) return;
  LSE_EXPECT(e->source.find("wmma") == std::string::npos);
}

// The point of the matrix-core path: a linear and the elementwise work after it
// are one launch, with the epilogue running on the accumulator still in
// register. Without this the WMMA kernel only ever fires for a lone linear,
// which is not how the model is shaped.
LSE_TEST(matrix_core_linear_fuses_its_epilogue_into_one_kernel) {
  Array x = Array::full(Shape{32, 64}, DType::kF32, 1.0f);
  Array w = Array::full(Shape{48, 64}, DType::kF32, 1.0f);
  Array scale = Array::full(Shape{32, 48}, DType::kF32, 0.5f);
  Array lin = linear(x, w);
  auto act = custom("jit.swish", {lin});
  if (!act.ok()) return;
  Array y = *act * scale;

  ::unsetenv("LSE_WMMA");
  const NodePtr roots[] = {y.node()};
  auto groups = Partitioner::partition(roots);
  const FusionGroup* fused = nullptr;
  for (const FusionGroup& g : groups) {
    for (const NodePtr& n : g.nodes) {
      if (n.get() == lin.node().get()) fused = &g;
    }
  }
  LSE_EXPECT(fused != nullptr);
  if (fused == nullptr) return;
  // The linear and the two elementwise ops are in one group, not three.
  LSE_EXPECT(fused->nodes.size() == 3u);
  auto e = kEmitter.emit(*fused, gfx1151());
  ::unsetenv("LSE_WMMA");

  LSE_EXPECT(e.ok());
  if (!e.ok()) {
    std::printf("       emit: %s\n", e.status().to_string().c_str());
    return;
  }
  if (std::getenv("LSE_TRACE_WMMA") != nullptr) {
    std::printf("---- fused wmma source ----\n%s\n", e->source.c_str());
  }

  // One kernel doing both: the matrix instruction and the epilogue.
  LSE_EXPECT(e->source.find("__builtin_amdgcn_wmma_f32_16x16x16_f16_w32") !=
             std::string::npos);
  LSE_EXPECT(e->source.find("__expf") != std::string::npos);
  // The primitive's operands are bound first, so the epilogue's extra input
  // cannot shift what in0/in1 mean inside the kernel.
  LSE_EXPECT(e->binding_order.size() == 4u);
  if (e->binding_order.size() != 4u) return;
  LSE_EXPECT(e->binding_order[0].get() == x.node().get());
  LSE_EXPECT(e->binding_order[1].get() == w.node().get());
  LSE_EXPECT(e->binding_order[2].get() == scale.node().get());
  // Still a wave per tile, not a thread per element.
  LSE_EXPECT(e->dims.workgroup_size[0] == 256u);
  LSE_EXPECT(e->dims.workgroup_count[0] == 1u);

  if (!kCompiler.available()) return;
  auto code = kCompiler.compile(e->source, "gfx1151");
  if (!code.ok()) {
    std::printf("       compile failed:\n%s\n", code.status().message().c_str());
    std::printf("       ---- source ----\n%s\n", e->source.c_str());
  }
  LSE_EXPECT(code.ok());
}

// ---------------------------------------------------------------------------
// The matrix-core descriptor
// ---------------------------------------------------------------------------

namespace {

namespace lmath = lse::math;

// The same question the consteval lookup answers, asked at runtime. The
// negative case cannot be asserted with a static_assert: an unmapped
// combination is not a substitution failure, it is an uncaught exception in an
// immediate invocation, so `matrix_core_row(kRdna3, kF32, kFp8, 16, 16, 16)`
// fails to compile rather than yielding false.
bool table_has_row(lmath::MatrixTarget g, lmath::MatrixElem a,
                   lmath::MatrixElem t, int m, int n, int k) {
  for (const auto& r : lmath::matrix_core_table()) {
    if (r.target == g && r.acc == a && r.operand == t && r.m == m &&
        r.n == n && r.k_step == k) {
      return true;
    }
  }
  return false;
}

// Four int8 to an i32 lane, low byte first. What `pack` describes; the device
// test below is what establishes it.
std::int32_t pack_i8x4(int a, int b, int c, int d) {
  return static_cast<std::int32_t>(
      (static_cast<std::uint32_t>(a) & 0xFFu) |
      ((static_cast<std::uint32_t>(b) & 0xFFu) << 8) |
      ((static_cast<std::uint32_t>(c) & 0xFFu) << 16) |
      ((static_cast<std::uint32_t>(d) & 0xFFu) << 24));
}

// Deterministic integers in [-span/2, span/2]; no <random>, so the fixture is
// the same everywhere.
int small_int(std::uint32_t seed, std::uint32_t span) {
  const std::uint32_t h = seed * 2654435761u;
  return static_cast<int>((h >> 24) % span) - static_cast<int>(span / 2);
}

// A tile over the int4 row. The only thing that differs from the int8 one the
// linear kernel builds is the MatrixElem it names.
struct Int4Tile
    : backend::hrx_kernels::MatrixTile<Int4Tile, lmath::MatrixTarget::kRdna3,
                                       lmath::MatrixElem::kI32,
                                       lmath::MatrixElem::kI4, 16, 16, 16> {
  std::uint32_t cols = 0;
  void emit_element(env::Emit& e, const kir::Val<kir::u32>& row,
                    const kir::Val<kir::u32>& col,
                    const kir::Val<kir::f32>& v) const {
    e.store(row * cols + col, v);
  }
};

// Four int4 to a byte, eight to an i32 lane, low nibble first.
std::int32_t pack_i4x8(const int* v) {
  std::uint32_t w = 0;
  for (int j = 0; j < 8; ++j) {
    w |= (static_cast<std::uint32_t>(v[j]) & 0xFu) << (4 * j);
  }
  return static_cast<std::int32_t>(w);
}

// Compile a tile body into a standalone kernel over two packed i32 operand
// buffers and a float output, run it on the live device, and hand back what it
// wrote. The wrapper is the emitter's own shape: buffers in binding order,
// then the constants block, with `i` the flat thread id.
std::vector<float> run_packed_tile(const std::string& body,
                                   const std::string& entry,
                                   const backend::DeviceInfo& dev,
                                   backend::IBackend& be,
                                   const std::vector<std::int32_t>& ax,
                                   const std::vector<std::int32_t>& bx,
                                   std::size_t out_elems,
                                   const ThreadPlan& tp) {
  const std::string source =
      "#include <hip/hip_runtime.h>\n"
      "struct LseConstants { unsigned int count; };\n"
      "extern \"C\" __global__ void " + entry + "(\n"
      "    const int* __restrict__ in0,\n"
      "    const int* __restrict__ in1,\n"
      "    float* __restrict__ out,\n"
      "    LseConstants kc) {\n"
      "  (void)kc;\n"
      "  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;\n" +
      body + "}\n";
  auto code = kCompiler.compile(source, std::string(dev.arch));
  if (!code.ok()) {
    std::printf("       compile failed:\n%s\n", code.status().message().c_str());
    std::printf("       ---- source ----\n%s\n", source.c_str());
    return {};
  }
  auto ab = be.allocate(ax.size() * 4, backend::MemoryClass::kDevice);
  auto bb = be.allocate(bx.size() * 4, backend::MemoryClass::kDevice);
  auto ob = be.allocate(out_elems * 4, backend::MemoryClass::kDevice);
  if (!ab.ok() || !bb.ok() || !ob.ok()) return {};
  backend::DeviceBuffer abuf = ab.release();
  backend::DeviceBuffer bbuf = bb.release();
  backend::DeviceBuffer obuf = ob.release();
  std::vector<float> got(out_elems, -1.0f);

  std::vector<float> out;
  auto handle = be.load_executable(entry, *code);
  if (handle.ok() &&
      be.copy_h2d(ax.data(), abuf, ax.size() * 4, 0).ok() &&
      be.copy_h2d(bx.data(), bbuf, bx.size() * 4, 0).ok() &&
      be.copy_h2d(got.data(), obuf, out_elems * 4, 0).ok()) {
    backend::LaunchDims dims;
    dims.workgroup_size[0] = tp.workgroup_size[0];
    dims.workgroup_count[0] = tp.workgroup_count[0];
    const backend::BufferRef binds[] = {
        {&abuf, 0, abuf.size_bytes},
        {&bbuf, 0, bbuf.size_bytes},
        {&obuf, 0, obuf.size_bytes},
    };
    const auto count = static_cast<std::uint32_t>(out_elems);
    std::array<std::byte, sizeof(std::uint32_t)> constants{};
    std::memcpy(constants.data(), &count, sizeof(count));
    backend::DispatchArgs args;
    args.bindings = binds;
    args.constants = constants;
    if (be.launch(*handle, dims, args).ok() && be.synchronize().ok() &&
        be.copy_d2h(obuf, got.data(), out_elems * 4, 0).ok()) {
      out = got;
    }
  }
  be.deallocate(abuf);
  be.deallocate(bbuf);
  be.deallocate(obuf);
  return out;
}

// Exact for every integer a K-deep int8 dot can reach below 2^24, so the
// matrix core either agrees with the host exactly or the row is wrong.
double worst_abs(const std::vector<float>& got, const std::vector<float>& want) {
  double worst = 0.0;
  int shown = 0;
  for (std::size_t i = 0; i < want.size(); ++i) {
    const double d =
        std::fabs(static_cast<double>(got[i]) - static_cast<double>(want[i]));
    if (d != 0.0 && shown < 4) {
      std::printf("       [%zu] got %.1f want %.1f\n", i,
                  static_cast<double>(got[i]), static_cast<double>(want[i]));
      ++shown;
    }
    worst = std::max(worst, d);
  }
  return worst;
}

backend::IBackend* live_hrx() {
  static std::unique_ptr<backend::IBackend> be = [] {
    auto b = backend::create_backend("hrx");
    if (!b.ok()) return std::unique_ptr<backend::IBackend>{};
    auto owned = b.release();
    if (!owned->init(0).ok()) return std::unique_ptr<backend::IBackend>{};
    return owned;
  }();
  return be.get();
}

}  // namespace

// One table, one row per variant, and an unmapped combination is a compile
// error rather than a fallthrough to whichever variant was there first.
LSE_TEST(matrix_core_table_is_the_only_place_a_variant_is_written_down) {
  using lmath::MatrixElem;
  using lmath::MatrixTarget;

  constexpr auto bf16 = lmath::matrix_core_row(
      MatrixTarget::kRdna3, MatrixElem::kF32, MatrixElem::kBF16, 16, 16, 16);
  static_assert(bf16.key == "wmma.f32.16x16x16.bf16");
  static_assert(bf16.a_len == 16 && bf16.b_len == 16 && bf16.c_len == 8);
  static_assert(bf16.pack == 1 && bf16.k_step == 16 && bf16.chained == 1);
  static_assert(bf16.wave == 32 && bf16.emittable());

  // Same shape, same accumulator, a different operand — and every width is
  // different. A kernel that spelled 16 and 8 would serve exactly one of them.
  constexpr auto iu8 = lmath::matrix_core_row(
      MatrixTarget::kRdna3, MatrixElem::kI32, MatrixElem::kI8, 16, 16, 16);
  static_assert(iu8.key == "wmma.i32.16x16x16.iu8");
  static_assert(iu8.a_len == 4 && iu8.c_len == 8 && iu8.pack == 4);
  constexpr auto iu4 = lmath::matrix_core_row(
      MatrixTarget::kRdna3, MatrixElem::kI32, MatrixElem::kI4, 16, 16, 16);
  static_assert(iu4.a_len == 2 && iu4.c_len == 8 && iu4.pack == 8);

  // A row may be more than one instruction: the RDNA4 int8 form reads 64 bits
  // per lane, so the efficient variant chains two and steps K by 32.
  constexpr auto fused = lmath::matrix_core_row(
      MatrixTarget::kRdna4, MatrixElem::kI32, MatrixElem::kI8, 16, 16, 32);
  static_assert(fused.chained == 2 && fused.k_step == 32 && fused.k == 16);
  static_assert(fused.a_len == 4);

  // Throughput is in the row because that is what a selector ranks by.
  static_assert(fused.throughput > bf16.throughput);
  static_assert(iu8.throughput > bf16.throughput);

  // Mapped, and unmapped. Asking the lookup for either of the last two is a
  // compile error, which is the property under test; what a test can assert is
  // the same fact about the table it reads.
  LSE_EXPECT(table_has_row(MatrixTarget::kRdna3, MatrixElem::kF32,
                           MatrixElem::kF16, 16, 16, 16));
  LSE_EXPECT(table_has_row(MatrixTarget::kCdna3, MatrixElem::kF32,
                           MatrixElem::kF16, 32, 32, 8));
  LSE_EXPECT(!table_has_row(MatrixTarget::kRdna3, MatrixElem::kF32,
                            MatrixElem::kFp8, 16, 16, 16));
  LSE_EXPECT(!table_has_row(MatrixTarget::kRdna3, MatrixElem::kF32,
                            MatrixElem::kBF16, 32, 32, 8));
}

// A row exists for gfx12 and for MFMA, with the right names and widths — and
// the gate refuses to emit any of them, because no layout has been measured on
// that hardware. An unverified row that declines is correct; one that emits is
// a silent wrong answer.
LSE_TEST(unmeasured_rows_are_described_but_never_emitted) {
  using lmath::MatrixElem;
  using lmath::MatrixTarget;

  constexpr auto rdna4 = lmath::matrix_core_row(
      MatrixTarget::kRdna4, MatrixElem::kF32, MatrixElem::kBF16, 16, 16, 16);
  static_assert(rdna4.key == "wmma12.f32.16x16x16.bf16");
  // Half the RDNA3 operand width: K splits across the half-waves.
  static_assert(rdna4.a_len == 8 && rdna4.c_len == 8 && rdna4.wave == 32);
  static_assert(!rdna4.emittable());

  constexpr auto cdna3 = lmath::matrix_core_row(
      MatrixTarget::kCdna3, MatrixElem::kF32, MatrixElem::kF16, 16, 16, 16);
  static_assert(cdna3.key == "mfma.f32.16x16x16.f16");
  static_assert(cdna3.wave == 64 && cdna3.a_len == 4 && cdna3.c_len == 4);
  static_assert(!cdna3.emittable());

  // Every row's spelling is in the HIP table even when the row cannot run
  // here, because spelling and availability are separate concerns.
  const auto sources = backend::hip_sources();
  for (const auto& r : lmath::matrix_core_table()) {
    if (sources.find(r.key).empty()) {
      std::printf("       no HIP spelling for %.*s\n",
                  static_cast<int>(r.key.size()), r.key.data());
    }
    LSE_EXPECT(!sources.find(r.key).empty());
  }

  // gfx1201 and gfx942 name a target — the family is known — but no row of
  // theirs is emittable, so the linear declines and the group falls back.
  backend::AmdDeviceInfo amd12;
  amd12.matrix_core = backend::MatrixCore::kWMMA;
  amd12.matrix_core_bf16 = true;
  amd12.max_load_bytes = 16;
  amd12.max_store_bytes = 16;
  backend::DeviceInfo d12;
  d12.arch = "gfx1201";
  d12.max_threads_per_workgroup = 1024;
  d12.wavefront_size = 32;
  d12.extension_id = backend::AmdDeviceInfo::kExtensionId;
  d12.extension = &amd12;
  LSE_EXPECT(backend::hrx_kernels::matrix_target(d12).has_value());
  LSE_EXPECT(*backend::hrx_kernels::matrix_target(d12) ==
             MatrixTarget::kRdna4);

  backend::AmdDeviceInfo amd942;
  amd942.matrix_core = backend::MatrixCore::kMFMA;
  amd942.matrix_core_bf16 = true;
  amd942.max_load_bytes = 16;
  amd942.max_store_bytes = 16;
  backend::DeviceInfo d942;
  d942.arch = "gfx942";
  d942.max_threads_per_workgroup = 1024;
  d942.wavefront_size = 64;
  d942.extension_id = backend::AmdDeviceInfo::kExtensionId;
  d942.extension = &amd942;
  LSE_EXPECT(*backend::hrx_kernels::matrix_target(d942) ==
             MatrixTarget::kCdna3);

  const Shape shapes[] = {Shape{32, 64}, Shape{48, 64}};
  const DType dts[] = {DType::kBF16, DType::kBF16};
  const auto tbl = backend::hip_sources();
  for (const backend::DeviceInfo* d : {&d12, &d942}) {
    KernelShapes s;
    s.inputs = shapes;
    s.input_dtypes = dts;
    s.output = Shape{32, 48};
    s.output_dtype = DType::kF32;
    s.device = d;
    s.types = backend::hip_types();
    s.intrinsics = &tbl;
    s.store = [](std::string_view i, std::string_view v) {
      return "out[" + std::string(i) + "] = " + std::string(v) + ";";
    };
    LSE_EXPECT(backend::hrx_kernels::wmma_linear_for(s) == nullptr);
  }
}

// The acceptance test for the whole design: int8 is a row, not a kernel.
//
// Nothing in wmma_linear.cpp names 4, 8 or 16 — the fragment widths, the K
// step and the accumulator mapping all come from the row — so this exercises
// the same body the bf16 prefill uses, against a host integer reference.
LSE_TEST(matrix_core_int8_linear_matches_a_host_integer_reference) {
  ::unsetenv("LSE_WMMA");
  constexpr int kM = 32;
  constexpr int kN = 48;
  constexpr int kLanes = 16;              // i32 lanes per row
  constexpr int kK = kLanes * 4;          // 64 int8 per row

  backend::IBackend* be = live_hrx();
  if (be == nullptr || !kCompiler.available()) return;
  const backend::DeviceInfo& dev = be->device_info();
  if (backend::arch_family(dev.arch) != backend::ArchFamily::kRdna3 &&
      backend::arch_family(dev.arch) != backend::ArchFamily::kRdna35) {
    return;  // no measured int8 layout for this device; the gate declines
  }

  const Shape shapes[] = {Shape{kM, kLanes}, Shape{kN, kLanes}};
  const DType in_dt[] = {DType::kI32, DType::kI32};
  const auto tbl = backend::hip_sources();
  KernelShapes s;
  s.inputs = shapes;
  s.input_dtypes = in_dt;
  s.output = Shape{kM, kN};
  s.output_dtype = DType::kF32;
  s.device = &dev;
  s.types = backend::hip_types();
  s.intrinsics = &tbl;
  s.store = [](std::string_view i, std::string_view v) {
    return "out[" + std::string(i) + "] = (float)(" + std::string(v) + ");";
  };

  const KernelPrimitiveBase* prim = backend::hrx_kernels::wmma_linear_for(s);
  LSE_EXPECT(prim != nullptr);
  if (prim == nullptr) return;
  const std::string body = prim->emit_kernel(s);
  LSE_EXPECT(!body.empty());
  if (body.empty()) return;

  // The row's widths, in the generated text: v4i32 operands, v8i32
  // accumulator, and the integer builtin from the HIP table.
  LSE_EXPECT(body.find("__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32") !=
             std::string::npos);
  LSE_EXPECT(body.find("typedef int lse_v4_i32") != std::string::npos);
  LSE_EXPECT(body.find("typedef int lse_v8_i32") != std::string::npos);

  constexpr std::size_t kMs = kM, kNs = kN, kKs = kK, kLs = kLanes;
  std::vector<int> a(kMs * kKs), b(kNs * kKs);
  for (std::size_t r = 0; r < kMs; ++r) {
    for (std::size_t t = 0; t < kKs; ++t) {
      a[r * kKs + t] =
          small_int(static_cast<std::uint32_t>(r * 977u + t * 31u), 101);
    }
  }
  for (std::size_t r = 0; r < kNs; ++r) {
    for (std::size_t t = 0; t < kKs; ++t) {
      b[r * kKs + t] =
          small_int(static_cast<std::uint32_t>(r * 613u + t * 71u + 7u), 101);
    }
  }
  std::vector<std::int32_t> ax(kMs * kLs), bx(kNs * kLs);
  for (std::size_t r = 0; r < kMs; ++r) {
    for (std::size_t l = 0; l < kLs; ++l) {
      const int* p = &a[r * kKs + l * 4];
      ax[r * kLs + l] = pack_i8x4(p[0], p[1], p[2], p[3]);
    }
  }
  for (std::size_t r = 0; r < kNs; ++r) {
    for (std::size_t l = 0; l < kLs; ++l) {
      const int* p = &b[r * kKs + l * 4];
      bx[r * kLs + l] = pack_i8x4(p[0], p[1], p[2], p[3]);
    }
  }
  std::vector<float> want(kMs * kNs, 0.0f);
  for (std::size_t r = 0; r < kMs; ++r) {
    for (std::size_t c = 0; c < kNs; ++c) {
      std::int64_t acc = 0;
      for (std::size_t t = 0; t < kKs; ++t) {
        acc += std::int64_t{a[r * kKs + t]} * b[c * kKs + t];
      }
      want[r * kNs + c] = static_cast<float>(acc);
    }
  }

  const std::vector<float> got = run_packed_tile(
      body, "lse_iu8_test", dev, *be, ax, bx, want.size(), prim->plan(s));
  LSE_EXPECT(got.size() == want.size());
  if (got.size() != want.size()) return;
  std::printf("       iu8 %dx%dx%d on %s\n", kM, kN, kK,
              std::string(dev.arch).c_str());
  LSE_EXPECT_EQ(worst_abs(got, want), 0.0);
}

// The same tile body, at half the fragment width again: int4 packs eight
// operands to a lane and the fragment is two registers instead of four. The
// engine has no int4 storage dtype for the selector to route, so the tile is
// instantiated directly — which is exactly the point, since the only thing
// that changes between this and the int8 case above is which row it reads.
LSE_TEST(matrix_core_int4_reuses_the_body_at_another_fragment_width) {
  constexpr int kM = 16;
  constexpr int kN = 16;
  constexpr int kLanes = 8;            // i32 lanes per row
  constexpr int kK = kLanes * 8;       // 64 int4 per row

  backend::IBackend* be = live_hrx();
  if (be == nullptr || !kCompiler.available()) return;
  const backend::DeviceInfo& dev = be->device_info();
  if (backend::arch_family(dev.arch) != backend::ArchFamily::kRdna3 &&
      backend::arch_family(dev.arch) != backend::ArchFamily::kRdna35) {
    return;
  }

  const auto sources = backend::hip_sources();
  const kir::TypeTable types = backend::hip_types();
  std::string body;
  {
    kir::KernelBody k(types, sources);
    k.set_store([](std::string_view i, std::string_view v) {
      return "out[" + std::string(i) + "] = (float)(" + std::string(v) + ");";
    });
    struct Args {
      env::In<std::int32_t, env::Emit> x;
      env::In<std::int32_t, env::Emit> w;
      env::Out<kir::f32, env::Emit> out;
    } a;
    const DType in_dt[] = {DType::kI32, DType::kI32};
    LSE_EXPECT(env::bind(k, a, in_dt, DType::kF32));
    env::Emit e{&k};
    Int4Tile tile;
    tile.cols = kN;
    tile.run(e, a.x, a.w, kM, kN, kLanes, backend::max_load_bytes(dev));
    body = k.str();
  }
  LSE_EXPECT(body.find("__builtin_amdgcn_wmma_i32_16x16x16_iu4_w32") !=
             std::string::npos);
  LSE_EXPECT(body.find("typedef int lse_v2_i32") != std::string::npos);

  constexpr std::size_t kMs = kM, kNs = kN, kKs = kK, kLs = kLanes;
  std::vector<int> a(kMs * kKs), b(kNs * kKs);
  for (std::size_t r = 0; r < kMs; ++r) {
    for (std::size_t t = 0; t < kKs; ++t) {
      a[r * kKs + t] =
          small_int(static_cast<std::uint32_t>(r * 397u + t * 13u), 15);
    }
  }
  for (std::size_t r = 0; r < kNs; ++r) {
    for (std::size_t t = 0; t < kKs; ++t) {
      b[r * kKs + t] =
          small_int(static_cast<std::uint32_t>(r * 211u + t * 53u + 3u), 15);
    }
  }
  std::vector<std::int32_t> ax(kMs * kLs), bx(kNs * kLs);
  for (std::size_t r = 0; r < kMs; ++r) {
    for (std::size_t l = 0; l < kLs; ++l) ax[r * kLs + l] = pack_i4x8(&a[r * kKs + l * 8]);
  }
  for (std::size_t r = 0; r < kNs; ++r) {
    for (std::size_t l = 0; l < kLs; ++l) bx[r * kLs + l] = pack_i4x8(&b[r * kKs + l * 8]);
  }
  std::vector<float> want(kMs * kNs, 0.0f);
  for (std::size_t r = 0; r < kMs; ++r) {
    for (std::size_t c = 0; c < kNs; ++c) {
      std::int64_t acc = 0;
      for (std::size_t t = 0; t < kKs; ++t) {
        acc += std::int64_t{a[r * kKs + t]} * b[c * kKs + t];
      }
      want[r * kNs + c] = static_cast<float>(acc);
    }
  }

  ThreadPlan tp;
  tp.workgroup_size[0] = 256;
  tp.workgroup_count[0] = 1;
  const std::vector<float> got =
      run_packed_tile(body, "lse_iu4_test", dev, *be, ax, bx, want.size(), tp);
  LSE_EXPECT(got.size() == want.size());
  if (got.size() != want.size()) return;
  LSE_EXPECT_EQ(worst_abs(got, want), 0.0);
}
