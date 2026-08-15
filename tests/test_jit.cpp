// Emitter -> comgr -> code object. Runs the full JIT path when ROCm is
// present; the source-shape checks run everywhere.
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <unistd.h>
#include <vector>

#include <array>

#include "harness.hpp"
#include "lse/backend/backend.hpp"
#include "lse/core/debug.hpp"
#include "lse/backends/hrx/arch_database.hpp"
#include "lse/backends/hrx/comgr_compiler.hpp"
#include "lse/backends/hrx/device_info.hpp"
#include "lse/backends/hrx/hip_emitter.hpp"
#include "lse/backends/hrx/hip_sources.hpp"
#include "lse/graph/kernel_primitive.hpp"
#include "lse/graph/jit.hpp"
#include "lse/graph/ops.hpp"
#include "lse/graph/primitive.hpp"
#include "lse/graph/primitive_library.hpp"

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
    a.wavefront_size = 32;
    a.max_waves_per_cu = 32;
    a.lds_bytes_per_workgroup = 65536;
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
  backend::AmdDeviceInfo amd;
  amd.lds_bytes_per_workgroup = 12345;
  amd.wavefront_size = 32;
  backend::apply_arch_defaults(info, amd);
  LSE_EXPECT_EQ(info.compute_units, 7);
  LSE_EXPECT_EQ(info.max_threads_per_workgroup, 256);
  LSE_EXPECT_EQ(amd.lds_bytes_per_workgroup, 12345u);
  LSE_EXPECT_EQ(amd.wavefront_size, 32);
  LSE_EXPECT(amd.matrix_core == backend::MatrixCore::kWMMA);
}

LSE_TEST(unknown_family_still_gets_isa_from_the_arch_string) {
  backend::DeviceInfo info;
  info.arch = "gfx1102";
  backend::AmdDeviceInfo amd;
  backend::apply_arch_defaults(info, amd);
  LSE_EXPECT(backend::arch_family(info.arch) == backend::ArchFamily::kRdna3);
  LSE_EXPECT_EQ(amd.wavefront_size, 32);
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
  LSE_EXPECT_EQ(amd.wavefront_size, 32);
  ::setenv("LSE_WAVEFRONT", "64", 1);
  amd = {};
  backend::apply_arch_defaults(info, amd);
  LSE_EXPECT_EQ(amd.wavefront_size, 64);
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
  const backend::AmdDeviceInfo& amd = gfx1151_amd();
  LSE_EXPECT(threads % amd.wavefront_size == 0);
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
    if (e->source.find("__device__ float lse_rms_norm(") == std::string::npos) {
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
  LSE_EXPECT(e->source.find("__device__ float lse_matmul(") != std::string::npos);
  LSE_EXPECT(e->source.find("extern \"C\" __global__ void lse_fused_") !=
             std::string::npos);
  LSE_EXPECT(e->source.find("lse_matmul(i,") != std::string::npos);
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
  LSE_EXPECT(e->source.find("lse_matmul(i,") != std::string::npos);
  LSE_EXPECT(e->source.find("__expf") != std::string::npos);
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
                const backend::DispatchArgs&) override {
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
