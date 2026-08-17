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
  // Fills what HRX has no query for — the CUs behind one LDS pool — from the
  // ISA row, so the model under test is the one a live device would get. Every
  // field above is already non-zero and is left alone.
  backend::AmdDeviceInfo amd = gfx1151_amd();
  backend::apply_arch_defaults(d, amd);
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

// The group that anchors on `kind`, not the first one: a root whose operands
// are themselves computed contributes a group per operand, and those come
// first in topological order.
Result<EmittedKernel> emit_anchor(Array& root, OpKind kind) {
  const NodePtr roots[] = {root.node()};
  auto groups = Partitioner::partition(roots);
  for (const FusionGroup& g : groups) {
    if (g.anchor == kind) return kEmitter.emit(g, gfx1151());
  }
  return LSE_ERROR(kNotFound, "no group anchored on ", to_string(kind));
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

// The row stride has to reach the activation load on both sides of the LDS
// budget. quant_linear stages the activation row in scratch when it fits and
// reads it from global when it does not; the global form had no `row * K`
// term, so a multi-row launch computed every row from row 0. This asks the
// emitter directly, so it holds on a machine with no device: gfx1151 reports
// 64 KiB, K * 4 B decides, and the shape below sits one group past the line.
// N, the lane count and the group count are all distinct from K, so the K
// stride can only have come from indexing the activation by its row.
LSE_TEST(quant_linear_indexes_the_activation_by_its_row) {
  constexpr int kBits = 8;
  constexpr std::int64_t kGroup = 64;
  constexpr std::int64_t kN = 16;
  const std::int64_t lds_floats = gfx1151().lds_bytes_per_workgroup / 4;

  for (const std::int64_t k :
       {std::int64_t{640}, (lds_floats / kGroup + 1) * kGroup}) {
    const std::int64_t lanes = k * kBits / 32;
    const std::int64_t groups = k / kGroup;
    Array x = Array::zeros(Shape{5, k}, DType::kF32);
    Array packed = Array::zeros(Shape{kN, lanes}, DType::kU32);
    Array scales = Array::zeros(Shape{kN, groups}, DType::kBF16);
    Array biases = Array::zeros(Shape{kN, groups}, DType::kBF16);

    Array y = quant_linear(x, packed, scales, biases, kBits, kGroup);
    auto e = emit_anchor(y, OpKind::kQuantMatMul);
    LSE_EXPECT(e.ok());
    if (!e.ok()) {
      std::printf("       K=%lld: %s\n", static_cast<long long>(k),
                  e.status().to_string().c_str());
      continue;
    }
    const std::string stride = "* " + std::to_string(k) + "u";
    LSE_EXPECT(e->source.find(stride) != std::string::npos);
  }
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

// The SwiGLU pair. Two fat stages over one element count where thread i writes
// element i and thread i reads it back: nothing in the chain crosses a thread,
// so a workgroup barrier orders it at any grid width and the pair belongs in
// one launch. Splitting it was 120 launches per token on lemonseed, 32% of
// them. A gather anywhere in the chain still has to split — __syncthreads is
// not a grid-wide barrier, and the launch boundary is the only one this path
// has.
LSE_TEST(a_fat_lane_chain_is_one_grid_launch) {
  auto nsync = [](const std::string& src) {
    std::size_t n = 0, pos = 0;
    while ((pos = src.find("__syncthreads", pos)) != std::string::npos) {
      ++n;
      pos += 13;
    }
    return n;
  };
  const backend::DeviceInfo dev = gfx1151();
  const backend::HipEmitter emitter;

  Array gate = Array::full(Shape{1, 1, 2176}, DType::kF32, 0.5f);
  Array up = Array::full(Shape{1, 1, 2176}, DType::kF32, 0.25f);
  Array act = silu(gate);
  Array hid = act * up;
  const NodePtr lane_roots[] = {hid.node()};
  const auto lane_wgs = Partitioner::phases(lane_roots);
  LSE_EXPECT(!lane_wgs.empty());
  if (lane_wgs.empty()) return;
  const FusionGroup lane = Partitioner::phase_group(lane_wgs[0], lane_roots);
  auto e = backend::HipEmitter::emit_phase(lane, dev);
  LSE_EXPECT(e.ok());
  if (!e.ok()) return;
  // 2176 elements over 256 threads is nine workgroups, not one.
  LSE_EXPECT_EQ(e->dims.workgroup_count[0], 9u);
  LSE_EXPECT(!e->persist_grid);
  // Exactly one, and it must stay: bindings are __restrict__, so two nodes on
  // one slot look like distinct pointers and dropping the barrier lets the
  // compiler reorder the two stages' accesses.
  LSE_EXPECT_EQ(nsync(e->source), 1u);

  // The scheduler splits the chain on the same predicate the emitter uses to
  // pick the geometry. If the two ever disagree the pair still merges and then
  // lands on ONE workgroup, which measured slower than the two launches it
  // replaced — so the agreement is the thing under test.
  const graph::IPhaseStaging* staging = emitter.staging();
  LSE_EXPECT(staging != nullptr);
  if (staging == nullptr) return;
  LSE_EXPECT(staging->lane_stage(*act.node()));
  LSE_EXPECT(staging->lane_stage(*hid.node()));
  LSE_EXPECT(staging->lane_aligned(*act.node(), *hid.node()));

  // A reduction reading the same chain: thread i now needs an element some
  // other thread wrote, so the split has to survive.
  Array w = Array::full(Shape{2176}, DType::kF32, 1.0f);
  Array pre = silu(gate);
  Array normed = rms_norm(pre, w, 1e-6f);
  const NodePtr gather_roots[] = {normed.node()};
  const auto gather_wgs = Partitioner::phases(gather_roots);
  LSE_EXPECT(!gather_wgs.empty());
  if (gather_wgs.empty()) return;
  const FusionGroup gather =
      Partitioner::phase_group(gather_wgs[0], gather_roots);
  auto ge = backend::HipEmitter::emit_phase(gather, dev);
  LSE_EXPECT(ge.ok());
  if (!ge.ok()) return;
  LSE_EXPECT_EQ(ge->dims.workgroup_count[0], 1u);
  LSE_EXPECT(nsync(ge->source) >= 1u);
  LSE_EXPECT(!staging->lane_aligned(*pre.node(), *normed.node()));

  // Equal element count is part of the question, not a detail: a broadcast
  // consumer reads an index its own thread never wrote.
  Array scalar = Array::full(Shape{1, 1, 1}, DType::kF32, 2.0f);
  Array scaled = silu(gate) * scalar;
  const NodePtr bcast_roots[] = {scaled.node()};
  const auto bcast_wgs = Partitioner::phases(bcast_roots);
  LSE_EXPECT(!bcast_wgs.empty());
  if (bcast_wgs.empty()) return;
  LSE_EXPECT(!staging->lane_aligned(*scalar.node(), *scaled.node()));
}

LSE_TEST(a_stage_that_reads_a_broadcast_operand_is_not_a_lane_stage) {
  // The edge test above only catches a broadcast whose SOURCE is in the chunk.
  // When the broadcast operand comes from outside — a per-head scale, a slice
  // already materialized — no edge is recorded and only lane_stage() stands
  // between the chunk and a fat grid. `stage_use` answers kLane for every
  // elementwise node without looking at its operands, so this is the case where
  // "thread i touches element i and nothing else" has to be checked rather than
  // assumed: slot recycling puts unrelated nodes on one allocation, and a
  // cross-workgroup write-after-read there is not something __syncthreads can
  // order. Shapes mirror what lemonseed actually emits: 4096 elements reading a
  // 4-element operand at (i / 1024) % 4.
  auto nsync = [](const std::string& src) {
    std::size_t n = 0, pos = 0;
    while ((pos = src.find("__syncthreads", pos)) != std::string::npos) {
      ++n;
      pos += 13;
    }
    return n;
  };
  const backend::DeviceInfo dev = gfx1151();
  const backend::HipEmitter emitter;
  const graph::IPhaseStaging* staging = emitter.staging();
  LSE_EXPECT(staging != nullptr);
  if (staging == nullptr) return;

  Array x = Array::full(Shape{1, 4, 1024}, DType::kF32, 0.5f);
  Array scale = Array::full(Shape{1, 4, 1}, DType::kF32, 2.0f);
  Array res = Array::full(Shape{1, 4, 1024}, DType::kF32, 0.25f);
  Array gated = x * scale;
  Array out = res + gated;

  LSE_EXPECT(!staging->lane_stage(*gated.node()));
  LSE_EXPECT(staging->lane_stage(*out.node()));
  // The consumer reads the producer at its own index, so the EDGE looks fine;
  // the producer's own operand is what disqualifies the chunk.
  LSE_EXPECT(!staging->lane_aligned(*gated.node(), *out.node()));

  const NodePtr roots[] = {out.node()};
  const auto wgs = Partitioner::phases(roots);
  LSE_EXPECT(!wgs.empty());
  if (wgs.empty()) return;
  const FusionGroup g = Partitioner::phase_group(wgs[0], roots);
  auto e = backend::HipEmitter::emit_phase(g, dev);
  LSE_EXPECT(e.ok());
  if (!e.ok()) return;
  // One workgroup, where __syncthreads is the whole grid. Merged onto 16 the
  // barrier orders one sixteenth of the threads that need ordering.
  LSE_EXPECT_EQ(e->dims.workgroup_count[0], 1u);
  LSE_EXPECT(nsync(e->source) >= 1u);
  // And the read really is at a computed index, not at `i` — the reason the
  // geometry above is the right answer rather than a conservative one.
  LSE_EXPECT(e->source.find("/ 1024") != std::string::npos ||
             e->source.find("/1024") != std::string::npos);
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
  LSE_EXPECT(backend::occupancy_per_lds_pool(d, threads, 0) > 0);

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

// One member of a device set: its own geometry, its own executables. The
// executable id is per instance so a test can tell which device a handle was
// loaded on, which is the thing a shared cache must never get wrong.
struct SetStubBackend final : backend::IBackend {
  backend::DeviceInfo info;
  const IKernelCompiler* cc = nullptr;
  std::uint64_t exec_id = 1;

  SetStubBackend(std::uint64_t id, const IKernelCompiler* compiler)
      : cc(compiler), exec_id(id) {
    info.arch = "gfx1151";
    info.name = "stub";
    info.compute_units = 40;
    info.lds_bytes_per_workgroup = 65536;
    info.max_threads_per_workgroup = 1024;
    info.wavefront_size = 32;
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
    h.executable = exec_id;
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
  const IKernelCompiler* compiler() const noexcept override { return cc; }
};

struct StubSet final : backend::IDeviceSet {
  std::vector<backend::IBackend*> members;
  std::size_t size() const noexcept override { return members.size(); }
  backend::IBackend& device(std::size_t i) const override {
    return *members[i];
  }
  std::size_t primary() const noexcept override { return 0; }
  std::size_t member_of(backend::DeviceIndex) const noexcept override {
    return members.size();
  }
  Status may_read(backend::DeviceIndex, std::size_t) const override {
    return OkStatus();
  }
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
    auto a = cache.get_or_compile(0, sig, ek);
    LSE_EXPECT(a.ok());
    LSE_EXPECT_EQ(cc.n, 1);
    auto b = cache.get_or_compile(0, sig, ek);
    LSE_EXPECT(b.ok());
    LSE_EXPECT_EQ(cc.n, 1);
    LSE_EXPECT(cache.stats().memory_hits >= 1u);
    LSE_EXPECT(cache.try_get(0, sig) != nullptr);
  }
  {
    JitCache cache(be, cc, dir.string());
    auto c = cache.get_or_compile(0, sig, ek);
    LSE_EXPECT(c.ok());
    LSE_EXPECT_EQ(cc.n, 1);
    LSE_EXPECT(cache.stats().disk_hits >= 1u);
  }
  {
    JitCache cache(be, cc, dir.string());
    ek.source = "kernel v2 { }";
    auto d = cache.get_or_compile(0, sig, ek);
    LSE_EXPECT(d.ok());
    LSE_EXPECT_EQ(cc.n, 2);
  }
  {
    ek.source = "kernel v2 { }";
    be.info.arch = "gfx1201";
    JitCache cache(be, cc, dir.string());
    auto e = cache.get_or_compile(0, sig, ek);
    LSE_EXPECT(e.ok());
    LSE_EXPECT_EQ(cc.n, 3);
  }

  std::error_code ec;
  fs::remove_all(dir, ec);
}

LSE_TEST(one_cache_serves_two_devices_without_mixing_their_executables) {
  namespace fs = std::filesystem;
  const fs::path dir = fs::temp_directory_path() /
                       ("lse-jit-set-" + std::to_string(::getpid()));
  std::error_code pre;
  fs::remove_all(dir, pre);
  fs::create_directories(dir);

  CountingCompiler cc;
  SetStubBackend a(0xAA, &cc);
  SetStubBackend b(0xBB, &cc);
  StubSet set;
  set.members = {&a, &b};

  EmittedKernel ek;
  ek.source = "kernel same { }";
  ek.entry_name = "k";
  const std::uint64_t sig = 7;

  JitCache cache(set, dir.string());

  auto first = cache.get_or_compile(0, sig, ek);
  LSE_EXPECT_OK(first.status());
  LSE_EXPECT_EQ(cc.n, 1);
  LSE_EXPECT_EQ(first->executable, std::uint64_t{0xAA});

  // Same arch, same geometry: the source is byte-identical, so the OBJECT is
  // shared and compiling it twice would be ~350 ms spent on bytes already on
  // disk. What is not shared is the loaded executable.
  auto second = cache.get_or_compile(1, sig, ek);
  LSE_EXPECT_OK(second.status());
  LSE_EXPECT_EQ(cc.n, 1);
  LSE_EXPECT_EQ(second->executable, std::uint64_t{0xBB});
  LSE_EXPECT(cache.stats().disk_hits >= 1u);

  // A live handle belongs to the device it was loaded on. Serving member 1 the
  // handle member 0 loaded is a wrong-device dispatch no runtime reports.
  const backend::KernelHandle* on_a = cache.try_get(0, sig);
  const backend::KernelHandle* on_b = cache.try_get(1, sig);
  LSE_EXPECT(on_a != nullptr && on_b != nullptr);
  if (on_a != nullptr && on_b != nullptr) {
    LSE_EXPECT_EQ(on_a->executable, std::uint64_t{0xAA});
    LSE_EXPECT_EQ(on_b->executable, std::uint64_t{0xBB});
  }
  LSE_EXPECT(cache.try_get(2, sig) == nullptr);
  LSE_EXPECT(!cache.get_or_compile(2, sig, ek).ok());

  std::error_code ec;
  fs::remove_all(dir, ec);
}

LSE_TEST(two_same_arch_devices_with_different_geometry_do_not_collide) {
  // Arch is not enough between two parts of one ISA: the emitter picks
  // workgroup dimensions and an LDS budget from the CU count and the LDS pool,
  // so the same signature is handed different source on each. Sharing a key
  // made the source-hash guard force a recompile per alternation AND each
  // device delete the other's object as dead.
  namespace fs = std::filesystem;
  const fs::path dir = fs::temp_directory_path() /
                       ("lse-jit-geom-" + std::to_string(::getpid()));
  std::error_code pre;
  fs::remove_all(dir, pre);
  fs::create_directories(dir);

  CountingCompiler cc;
  SetStubBackend wide(0xAA, &cc);
  SetStubBackend narrow(0xBB, &cc);
  narrow.info.compute_units = 20;   // same gfx1151 string, half the CUs
  StubSet set;
  set.members = {&wide, &narrow};

  EmittedKernel wide_src;
  wide_src.source = "kernel tuned_for_40_cu { }";
  wide_src.entry_name = "k";
  EmittedKernel narrow_src;
  narrow_src.source = "kernel tuned_for_20_cu { }";
  narrow_src.entry_name = "k";
  const std::uint64_t sig = 11;

  JitCache cache(set, dir.string());
  LSE_EXPECT_OK(cache.get_or_compile(0, sig, wide_src).status());
  LSE_EXPECT_EQ(cc.n, 1);
  LSE_EXPECT_OK(cache.get_or_compile(1, sig, narrow_src).status());
  LSE_EXPECT_EQ(cc.n, 2);

  // And back again: two live entries, not one that keeps being rebuilt.
  LSE_EXPECT_OK(cache.get_or_compile(0, sig, wide_src).status());
  LSE_EXPECT_EQ(cc.n, 2);
  LSE_EXPECT_OK(cache.get_or_compile(1, sig, narrow_src).status());
  LSE_EXPECT_EQ(cc.n, 2);

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
  LSE_EXPECT(jit.get_or_compile(0, 1, ek).ok());

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

// Names of the workgroup arrays a kernel declares, in source order.
std::vector<std::string> shared_arrays(const std::string& src) {
  std::vector<std::string> out;
  const std::string decl = "__shared__ float ";
  for (std::size_t at = src.find(decl); at != std::string::npos;
       at = src.find(decl, at)) {
    const std::size_t from = at + decl.size();
    const std::size_t open = src.find('[', from);
    if (open == std::string::npos) break;
    out.push_back(src.substr(from, open - from));
    at = open;
  }
  return out;
}

// Assignments into `name[...]`, which for a staged activation row is its fill —
// as opposed to the reads, which are the same subscript on the other side.
std::size_t fills_of(const std::string& src, const std::string& name) {
  std::size_t n = 0;
  const std::string open = name + "[";
  for (std::size_t at = src.find(open); at != std::string::npos;
       at = src.find(open, at + open.size())) {
    const std::size_t close = src.find(']', at);
    if (close != std::string::npos && src.compare(close, 4, "] = ") == 0) ++n;
  }
  return n;
}

// Compile an emitted kernel, give every binding a device buffer, run it, and
// hand back what each binding holds afterwards, in binding order. `host`
// supplies the bytes for the input bindings by node; anything without an entry
// starts zeroed, which is what an output wants.
std::vector<std::vector<float>> run_emitted(
    const EmittedKernel& e, const backend::DeviceInfo& dev,
    backend::IBackend& be,
    const std::vector<std::pair<const Node*, std::vector<float>>>& host) {
  auto code = kCompiler.compile(e.source, std::string(dev.arch));
  if (!code.ok()) {
    std::printf("       compile failed:\n%s\n", code.status().message().c_str());
    return {};
  }
  auto handle = be.load_executable(e.entry_name, *code);
  if (!handle.ok()) return {};

  const std::size_t n = e.binding_order.size();
  std::vector<backend::DeviceBuffer> bufs(n);
  std::vector<std::vector<float>> data(n);
  std::vector<backend::BufferRef> binds(n);
  bool ok = true;
  std::size_t made = 0;
  for (; made < n && ok; ++made) {
    const Node* b = e.binding_order[made].get();
    data[made].assign(b->element_count(), 0.0f);
    for (const auto& [node, v] : host) {
      if (node == b && v.size() == data[made].size()) data[made] = v;
    }
    auto a = be.allocate(data[made].size() * 4, backend::MemoryClass::kDevice);
    ok = a.ok();
    if (!ok) break;
    bufs[made] = a.release();
    binds[made] = {&bufs[made], 0, bufs[made].size_bytes};
    ok = be.copy_h2d(data[made].data(), bufs[made], data[made].size() * 4, 0)
             .ok();
  }

  if (ok) {
    std::uint32_t count = 0;
    for (const std::vector<float>& v : data) {
      count = std::max(count, static_cast<std::uint32_t>(v.size()));
    }
    std::array<std::byte, sizeof(std::uint32_t)> constants{};
    std::memcpy(constants.data(), &count, sizeof(count));
    backend::DispatchArgs args;
    args.bindings = binds;
    args.constants = constants;
    ok = be.launch(*handle, e.dims, args).ok() && be.synchronize().ok();
    for (std::size_t i = 0; i < n && ok; ++i) {
      ok = be.copy_d2h(bufs[i], data[i].data(), data[i].size() * 4, 0).ok();
    }
  }
  for (std::size_t i = 0; i < made; ++i) be.deallocate(bufs[i]);
  return ok ? data : std::vector<std::vector<float>>{};
}

}  // namespace

// A fused run stages the shared activation row ONCE, by construction: the
// emitter declares one array, fills it under a workgroup-uniform row guard and
// barriers behind it, and every sibling reads that array. Before this the stages
// each staged their own copy and `lds_fold` merged them afterwards, which left
// the surviving barriers behind and depended on the widest stage happening to
// come first — a narrow stage first and the fills could not be folded at all.
LSE_TEST(one_staging_serves_every_sibling_that_shares_the_row) {
  ::unsetenv("LSE_WMMA");
  Array x = Array::full(Shape{1, 256}, DType::kF32, 1.0f);
  Array wa = Array::full(Shape{128, 256}, DType::kF32, 0.5f);
  // Narrower N first: this one covers 2 tiles where the others cover 16, so a
  // fill under its own tile guard would leave 14 tiles reading a row nothing
  // wrote. The staging is guarded on the row alone for exactly that reason.
  Array wb = Array::full(Shape{16, 256}, DType::kF32, 0.25f);
  Array wc = Array::full(Shape{128, 256}, DType::kF32, 0.125f);
  auto e = kEmitter.emit(
      sibling_group({linear(x, wb), linear(x, wa), linear(x, wc)}), gfx1151());
  LSE_EXPECT(e.ok());
  if (!e.ok()) {
    std::printf("       %s\n", e.status().to_string().c_str());
    return;
  }

  const std::vector<std::string> arrays = shared_arrays(e->source);
  LSE_EXPECT_EQ(arrays.size(), 1u);
  if (arrays.empty()) {
    std::printf("---- source ----\n%s\n", e->source.c_str());
    return;
  }
  LSE_EXPECT_EQ(fills_of(e->source, arrays[0]), 1u);
  // One barrier publishes the row; the other two are the convoy between the
  // three stages, which is a memory-locality decision and not a dependence.
  // Three stages that each published their own row would need three.
  LSE_EXPECT_EQ(count_of(e->source, "__syncthreads()"), 3u);
  // Three stages, three reads of the one array, and every one of them reaches
  // it without a row term: the row offset is in the panel.
  LSE_EXPECT(count_of(e->source, arrays[0] + "[") >= 5u);
  LSE_EXPECT(e->source.find(arrays[0] + "[((") == std::string::npos);
  // The group's scratch is one row of 64 floats, 16-byte aligned — not the four
  // copies a max over the stage plans used to report as one.
  LSE_EXPECT_EQ(e->lds_bytes, 1024u);

  // And the fill does carry the row stride, which is the other half of the
  // contract: the panel holds row blockIdx.y, not row 0.
  LSE_EXPECT(e->source.find("* 256u)") != std::string::npos);

  if (!kCompiler.available()) return;
  auto code = kCompiler.compile(e->source, "gfx1151");
  if (!code.ok()) {
    std::printf("       %s\n", code.status().message().c_str());
  }
  LSE_EXPECT(code.ok());
}

// Siblings that do not share the row must not share the staging. The predicate
// is the activation node, its length and its row count — not a shape that
// merely agrees — so two runs over two different buffers of the same K get one
// panel each and the run is priced for both.
LSE_TEST(siblings_over_different_rows_get_their_own_staging) {
  ::unsetenv("LSE_WMMA");
  Array x0 = Array::full(Shape{1, 256}, DType::kF32, 1.0f);
  Array x1 = Array::full(Shape{1, 256}, DType::kF32, 2.0f);
  Array wa = Array::full(Shape{128, 256}, DType::kF32, 0.5f);
  Array wb = Array::full(Shape{128, 256}, DType::kF32, 0.25f);
  Array wc = Array::full(Shape{128, 256}, DType::kF32, 0.125f);
  Array wd = Array::full(Shape{128, 256}, DType::kF32, 0.0625f);
  auto e = kEmitter.emit(sibling_group({linear(x0, wa), linear(x1, wb),
                                        linear(x0, wc), linear(x1, wd)}),
                         gfx1151());
  LSE_EXPECT(e.ok());
  if (!e.ok()) {
    std::printf("       %s\n", e.status().to_string().c_str());
    return;
  }

  const std::vector<std::string> arrays = shared_arrays(e->source);
  LSE_EXPECT_EQ(arrays.size(), 2u);
  if (arrays.size() != 2) {
    std::printf("---- source ----\n%s\n", e->source.c_str());
    return;
  }
  LSE_EXPECT(arrays[0] != arrays[1]);
  for (const std::string& a : arrays) LSE_EXPECT_EQ(fills_of(e->source, a), 1u);
  // Two publications, one per row, plus three convoy barriers for four stages.
  LSE_EXPECT_EQ(count_of(e->source, "__syncthreads()"), 5u);
  // Two rows, both charged to the run.
  LSE_EXPECT_EQ(e->lds_bytes, 2048u);
  // Each panel is filled from its own buffer, and the two buffers are the
  // first two bindings.
  LSE_EXPECT(e->source.find(arrays[0] + "[p0_") != std::string::npos);
  LSE_EXPECT(e->source.find(arrays[1] + "[p1_") != std::string::npos);

  if (!kCompiler.available()) return;
  auto code = kCompiler.compile(e->source, "gfx1151");
  if (!code.ok()) {
    std::printf("       %s\n", code.status().message().c_str());
  }
  LSE_EXPECT(code.ok());
}

// The occupancy model must be the hardware's, in the hardware's unit. On RDNA a
// WGP pairs two CUs behind ONE 64 KiB pool, which is what HIP calls a
// multiprocessor, so `budget / request` counts workgroups per pool and describing
// it as per-CU overstates residency by 2x. Every row below is
// hipOccupancyMaxActiveBlocksPerMultiprocessor on gfx1151, measured.
LSE_TEST(the_occupancy_model_reproduces_the_measured_hardware_table) {
  const backend::DeviceInfo d = gfx1151();
  LSE_EXPECT_EQ(backend::cus_per_lds_pool(d), 2u);

  struct Row {
    std::uint32_t threads;
    std::uint32_t lds;
    std::uint32_t blocks;
  };
  // 256 threads: the 2048-thread cap allows 8 per pool, and LDS takes over from
  // 12288 up. 512 threads: the thread cap is 4 and LDS binds from 32768.
  constexpr Row kMeasured[] = {
      {256, 4096, 8},  {256, 8192, 8},   {256, 12288, 5}, {256, 16384, 4},
      {256, 20480, 3}, {256, 26112, 2},  {256, 32768, 2}, {256, 65536, 1},
      {512, 4096, 4},  {512, 16384, 4},  {512, 32768, 2},
  };
  for (const Row& r : kMeasured) {
    const std::uint32_t got =
        backend::occupancy_per_lds_pool(d, r.threads, r.lds);
    LSE_EXPECT_EQ(got, r.blocks);
    if (got != r.blocks) {
      std::printf("       threads=%u lds=%u got=%u want=%u\n", r.threads, r.lds,
                  got, r.blocks);
    }
  }
  // Past the device's own limit there is no legal launch at all.
  LSE_EXPECT_EQ(backend::occupancy_per_lds_pool(d, 256, 65552), 0u);
  // Half the budget is where residency reaches exactly one workgroup per CU:
  // 2 per pool, and a pool is 2 CUs. That is the number the fusion gate is held
  // to, stated here in the unit it is derived from.
  LSE_EXPECT_EQ(backend::occupancy_per_lds_pool(d, 256, 32768),
                backend::cus_per_lds_pool(d));
  LSE_EXPECT(backend::occupancy_per_lds_pool(d, 256, 40960) <
             backend::cus_per_lds_pool(d));
}

// A run is admitted on a number, and that number has to be what its text asks
// for. Two panels at exactly half the device budget each fill it and must still
// be emitted; one line of f32 more and the run cannot be launched at all, so it
// is refused rather than emitted over budget and left for the compiler to reject
// — a JIT failure is not fatal here, it drops the group onto the host
// interpreter for the rest of the run.
LSE_TEST(a_sibling_run_over_the_workgroup_budget_is_refused_not_emitted) {
  ::unsetenv("LSE_WMMA");
  const backend::DeviceInfo dev = gfx1151();
  const std::uint32_t budget = dev.lds_bytes_per_workgroup;
  LSE_EXPECT_EQ(budget, 65536u);

  // Two distinct activations, each staging half the budget: 8192 f32 = 32768 B.
  {
    Array x0 = Array::full(Shape{1, 8192}, DType::kF32, 1.0f);
    Array x1 = Array::full(Shape{1, 8192}, DType::kF32, 2.0f);
    Array wa = Array::full(Shape{16, 8192}, DType::kF32, 0.5f);
    Array wb = Array::full(Shape{16, 8192}, DType::kF32, 0.25f);
    auto e = kEmitter.emit(sibling_group({linear(x0, wa), linear(x1, wb)}), dev);
    LSE_EXPECT(e.ok());
    if (!e.ok()) {
      std::printf("       %s\n", e.status().to_string().c_str());
      return;
    }
    LSE_EXPECT_EQ(shared_arrays(e->source).size(), 2u);
    // Summed to exactly the budget. A max over the stages would say 32768 and
    // would keep saying it while a third panel walked off the end.
    LSE_EXPECT_EQ(e->lds_bytes, budget);
    // And the number is the text's, not a prediction sitting beside it.
    auto counted = backend::HipEmitter::shared_bytes(e->source);
    LSE_EXPECT(counted.ok());
    if (counted.ok()) LSE_EXPECT_EQ(*counted, e->lds_bytes);
    if (kCompiler.available()) {
      auto code = kCompiler.compile(e->source, "gfx1151");
      if (!code.ok()) {
        std::printf("       %s\n", code.status().message().c_str());
      }
      LSE_EXPECT(code.ok());
    }
  }

  // One f32 line over: 32768 + 32784 = 65552. The run is not emitted as a fused
  // body, and whatever is emitted instead stays inside the budget.
  {
    Array x0 = Array::full(Shape{1, 8192}, DType::kF32, 1.0f);
    Array x1 = Array::full(Shape{1, 8196}, DType::kF32, 2.0f);
    Array wa = Array::full(Shape{16, 8192}, DType::kF32, 0.5f);
    Array wb = Array::full(Shape{16, 8196}, DType::kF32, 0.25f);
    auto e = kEmitter.emit(sibling_group({linear(x0, wa), linear(x1, wb)}), dev);
    LSE_EXPECT(e.ok());
    if (!e.ok()) return;
    LSE_EXPECT(e->lds_bytes <= budget);
    LSE_EXPECT(shared_arrays(e->source).size() < 2u);
    auto counted = backend::HipEmitter::shared_bytes(e->source);
    LSE_EXPECT(counted.ok());
    if (counted.ok()) LSE_EXPECT_EQ(*counted, e->lds_bytes);
  }
}

// A phase is several stage bodies concatenated, each built against the whole
// device budget and blind to the others, so its workgroup scratch is the SUM of
// what they declare. It used to report a constant 1024 for every geometry, which
// is 17x under for a two-GEMV phase — and an over-budget phase does not fail to
// launch, it fails to compile, and the scheduler then runs the group on the host
// interpreter for the rest of the run.
LSE_TEST(a_phase_is_priced_for_every_row_its_stages_stage) {
  auto phase_of = [](Array& root) {
    const NodePtr roots[] = {root.node()};
    const auto wgs = Partitioner::phases(roots);
    return wgs.empty() ? FusionGroup{}
                       : Partitioner::phase_group(wgs[0], roots);
  };

  // One staged GEMV. A phase only puts a GEMV on a grid at N >= 256, and only a
  // grid launch has a workgroup-constant row to stage: K=1024 f32 = 4096 B.
  Array x = Array::full(Shape{1, 1024}, DType::kF32, 1.0f);
  Array w2 = Array::full(Shape{256, 1024}, DType::kF32, 0.2f);
  Array solo = linear(x, w2);
  const FusionGroup g1 = phase_of(solo);
  auto e1 = backend::HipEmitter::emit_phase(g1, gfx1151());
  LSE_EXPECT(e1.ok());
  if (!e1.ok()) {
    std::printf("       %s\n", e1.status().to_string().c_str());
    return;
  }
  LSE_EXPECT_EQ(shared_arrays(e1->source).size(), 1u);
  LSE_EXPECT_EQ(e1->lds_bytes, 4096u);

  // Two staged GEMVs in one phase, over two different activations, so neither
  // can reuse the other's row. The declarations sit in disjoint braces and still
  // sum: the hardware gives each its own LDS offset.
  Array w1 = Array::full(Shape{1024, 1024}, DType::kF32, 0.1f);
  Array chain = linear(linear(x, w1), w2);
  const FusionGroup g2 = phase_of(chain);
  auto e2 = backend::HipEmitter::emit_phase(g2, gfx1151());
  LSE_EXPECT(e2.ok());
  if (!e2.ok()) {
    std::printf("       %s\n", e2.status().to_string().c_str());
    return;
  }
  const std::size_t rows_two = shared_arrays(e2->source).size();
  LSE_EXPECT_EQ(rows_two, 2u);
  if (rows_two != 2) return;
  LSE_EXPECT_EQ(e2->lds_bytes, 8192u);
  // Read off the text, which is where a concatenated phase's total lives.
  auto counted = backend::HipEmitter::shared_bytes(e2->source);
  LSE_EXPECT(counted.ok());
  if (counted.ok()) LSE_EXPECT_EQ(*counted, e2->lds_bytes);
  // Not the max, which is what a per-geometry constant amounts to.
  LSE_EXPECT(e2->lds_bytes > e1->lds_bytes);
}

// Width invariance: a token must not depend on how many rows are in the pass.
// The staged and unstaged activation reads index differently — the panel holds
// the row, so it is read from 0, while global is read from `row * K` — and a
// staging change that gets that wrong makes every row read row 0. That is the
// defect this whole path can reproduce, so it is checked by running the fused
// kernel at four widths against a host reference rather than by reading source.
LSE_TEST(a_shared_staged_row_holds_its_own_row_at_every_pass_width) {
  ::unsetenv("LSE_WMMA");
  backend::IBackend* be = live_hrx();
  if (be == nullptr || !kCompiler.available()) return;
  const backend::DeviceInfo& dev = be->device_info();
  if (backend::device_extension<backend::AmdDeviceInfo>(dev) == nullptr) return;

  constexpr std::int64_t kK = 256;
  constexpr std::int64_t kN = 32;
  // Rows must differ, or reading row 0 for everything would pass.
  auto xrow = [](std::size_t r, std::size_t t) {
    return static_cast<float>(
        small_int(static_cast<std::uint32_t>(r * 1013u + t * 37u + 5u), 9));
  };
  std::vector<float> wa(kN * kK), wb(kN * kK);
  for (std::size_t c = 0; c < static_cast<std::size_t>(kN); ++c) {
    for (std::size_t t = 0; t < static_cast<std::size_t>(kK); ++t) {
      wa[c * kK + t] = static_cast<float>(
          small_int(static_cast<std::uint32_t>(c * 409u + t * 17u), 7));
      wb[c * kK + t] = static_cast<float>(
          small_int(static_cast<std::uint32_t>(c * 811u + t * 53u + 3u), 7));
    }
  }

  for (const std::int64_t m : {std::int64_t{1}, std::int64_t{2},
                               std::int64_t{3}, std::int64_t{5}}) {
    Array x = Array::zeros(Shape{m, kK}, DType::kF32);
    Array w0 = Array::zeros(Shape{kN, kK}, DType::kF32);
    Array w1 = Array::zeros(Shape{kN, kK}, DType::kF32);
    Array ya = linear(x, w0);
    Array yb = linear(x, w1);
    auto e = kEmitter.emit(sibling_group({ya, yb}), dev);
    LSE_EXPECT(e.ok());
    if (!e.ok()) {
      std::printf("       m=%lld: %s\n", static_cast<long long>(m),
                  e.status().to_string().c_str());
      continue;
    }
    // One row of the pass in scratch whatever the width is.
    LSE_EXPECT_EQ(shared_arrays(e->source).size(), 1u);

    std::vector<float> xs(static_cast<std::size_t>(m * kK));
    for (std::size_t r = 0; r < static_cast<std::size_t>(m); ++r) {
      for (std::size_t t = 0; t < static_cast<std::size_t>(kK); ++t) {
        xs[r * kK + t] = xrow(r, t);
      }
    }
    const std::vector<std::vector<float>> got = run_emitted(
        *e, dev, *be,
        {{x.node().get(), xs},
         {w0.node().get(), wa},
         {w1.node().get(), wb}});
    LSE_EXPECT(got.size() == e->binding_order.size());
    if (got.size() != e->binding_order.size()) continue;

    // Every row against a host dot product over that row's own activation.
    // Small integers, so f32 is exact and the comparison is too.
    for (int which = 0; which < 2; ++which) {
      const Node* out = (which == 0 ? ya : yb).node().get();
      const std::vector<float>& w = which == 0 ? wa : wb;
      std::size_t at = e->binding_order.size();
      for (std::size_t i = 0; i < e->binding_order.size(); ++i) {
        if (e->binding_order[i].get() == out) at = i;
      }
      LSE_EXPECT(at < e->binding_order.size());
      if (at >= e->binding_order.size()) continue;
      std::vector<float> want(static_cast<std::size_t>(m * kN), 0.0f);
      for (std::size_t r = 0; r < static_cast<std::size_t>(m); ++r) {
        for (std::size_t c = 0; c < static_cast<std::size_t>(kN); ++c) {
          float acc = 0.0f;
          for (std::size_t t = 0; t < static_cast<std::size_t>(kK); ++t) {
            acc += xs[r * kK + t] * w[c * kK + t];
          }
          want[r * kN + c] = acc;
        }
      }
      const double worst = worst_abs(got[at], want);
      if (worst != 0.0) {
        std::printf("       m=%lld output %d worst %.3f\n",
                    static_cast<long long>(m), which, worst);
      }
      LSE_EXPECT_EQ(worst, 0.0);
    }
  }
}

// The shape that actually runs: lemonseed's routed experts are `linear_indexed`
// siblings off one normed activation, and at a prefill width they are the only
// sibling run in the program that carries a shared panel over more than one row.
//
// Two row offsets have to be right at once here and they are read from different
// buffers — the panel holds row `blockIdx.y` of the activation and is therefore
// subscripted from 0, while the expert index is still read out of global at
// `row * keep + slot`. Dropping either one leaves every row on row 0's data,
// which is fluent and wrong, so both are checked by running the fused kernel
// against a host reference at four widths with a different expert per row.
LSE_TEST(a_shared_row_and_a_per_row_expert_agree_at_every_pass_width) {
  ::unsetenv("LSE_WMMA");
  backend::IBackend* be = live_hrx();
  if (be == nullptr || !kCompiler.available()) return;
  const backend::DeviceInfo& dev = be->device_info();
  if (backend::device_extension<backend::AmdDeviceInfo>(dev) == nullptr) return;

  constexpr std::int64_t kK = 256;
  constexpr std::int64_t kN = 32;
  constexpr std::int64_t kE = 4;
  constexpr std::int64_t kKeep = 2;
  // Every expert plane differs, or routing to the wrong one would still agree.
  std::vector<float> ws(static_cast<std::size_t>(kE * kN * kK));
  for (std::size_t e = 0; e < static_cast<std::size_t>(kE); ++e) {
    for (std::size_t c = 0; c < static_cast<std::size_t>(kN); ++c) {
      for (std::size_t t = 0; t < static_cast<std::size_t>(kK); ++t) {
        ws[(e * kN + c) * kK + t] = static_cast<float>(small_int(
            static_cast<std::uint32_t>(e * 2003u + c * 409u + t * 17u), 7));
      }
    }
  }
  auto xrow = [](std::size_t r, std::size_t t) {
    return static_cast<float>(
        small_int(static_cast<std::uint32_t>(r * 1013u + t * 37u + 5u), 9));
  };
  // Distinct per row AND per slot, so neither a lost row offset nor a lost slot
  // can land on the right expert.
  auto expert = [](std::size_t r, std::size_t slot) {
    return (r * 3u + slot * 2u + 1u) % static_cast<std::size_t>(kE);
  };

  for (const std::int64_t m : {std::int64_t{1}, std::int64_t{2},
                               std::int64_t{3}, std::int64_t{5}}) {
    Array x = Array::zeros(Shape{m, kK}, DType::kF32);
    Array w = Array::zeros(Shape{kE, kN, kK}, DType::kF32);
    Array idx = Array::zeros(Shape{m, kKeep}, DType::kF32);
    Array y0 = linear_indexed(x, w, idx, 0);
    Array y1 = linear_indexed(x, w, idx, 1);
    auto e = kEmitter.emit(sibling_group({y0, y1}), dev);
    LSE_EXPECT(e.ok());
    if (!e.ok()) {
      std::printf("       m=%lld: %s\n", static_cast<long long>(m),
                  e.status().to_string().c_str());
      continue;
    }
    LSE_EXPECT_EQ(shared_arrays(e->source).size(), 1u);

    std::vector<float> xs(static_cast<std::size_t>(m * kK));
    for (std::size_t r = 0; r < static_cast<std::size_t>(m); ++r) {
      for (std::size_t t = 0; t < static_cast<std::size_t>(kK); ++t) {
        xs[r * kK + t] = xrow(r, t);
      }
    }
    std::vector<float> ids(static_cast<std::size_t>(m * kKeep));
    for (std::size_t r = 0; r < static_cast<std::size_t>(m); ++r) {
      for (std::size_t s = 0; s < static_cast<std::size_t>(kKeep); ++s) {
        ids[r * kKeep + s] = static_cast<float>(expert(r, s));
      }
    }

    const std::vector<std::vector<float>> got =
        run_emitted(*e, dev, *be,
                    {{x.node().get(), xs},
                     {w.node().get(), ws},
                     {idx.node().get(), ids}});
    LSE_EXPECT(got.size() == e->binding_order.size());
    if (got.size() != e->binding_order.size()) continue;

    for (std::size_t slot = 0; slot < static_cast<std::size_t>(kKeep); ++slot) {
      const Node* out = (slot == 0 ? y0 : y1).node().get();
      std::size_t at = e->binding_order.size();
      for (std::size_t i = 0; i < e->binding_order.size(); ++i) {
        if (e->binding_order[i].get() == out) at = i;
      }
      LSE_EXPECT(at < e->binding_order.size());
      if (at >= e->binding_order.size()) continue;
      std::vector<float> want(static_cast<std::size_t>(m * kN), 0.0f);
      for (std::size_t r = 0; r < static_cast<std::size_t>(m); ++r) {
        const std::size_t ex = expert(r, slot);
        for (std::size_t c = 0; c < static_cast<std::size_t>(kN); ++c) {
          float acc = 0.0f;
          for (std::size_t t = 0; t < static_cast<std::size_t>(kK); ++t) {
            acc += xs[r * kK + t] * ws[(ex * kN + c) * kK + t];
          }
          want[r * kN + c] = acc;
        }
      }
      const double worst = worst_abs(got[at], want);
      if (worst != 0.0) {
        std::printf("       m=%lld slot %zu worst %.3f\n",
                    static_cast<long long>(m), slot, worst);
      }
      LSE_EXPECT_EQ(worst, 0.0);
    }
  }
}

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

// Width invariance for the merged SwiGLU pair, on the device. Both stages bake
// the element count and the grid width into the body, so an M-row pass has to
// compute every row from its own data — the recurring defect in this tree is a
// kernel that reads row 0 for all of them and still produces fluent text. Row r
// is compared bit-for-bit across pass widths, which needs no reference model and
// no tolerance; a host reference then catches every row being wrong the same
// way. The grid must also grow with M: a body that kept one row's count would
// launch the same nine workgroups at every width and leave rows 1.. at zero.
LSE_TEST(the_fused_swiglu_pair_holds_its_own_row_at_every_pass_width) {
  backend::IBackend* be = live_hrx();
  if (be == nullptr || !kCompiler.available()) return;
  const backend::DeviceInfo& dev = be->device_info();
  if (backend::device_extension<backend::AmdDeviceInfo>(dev) == nullptr) return;

  constexpr std::int64_t kInner = 2176;
  // Rows must differ, or reading row 0 for everything would pass.
  auto gval = [](std::size_t r, std::size_t t) {
    return static_cast<float>(
               small_int(static_cast<std::uint32_t>(r * 1013u + t * 37u), 9)) *
           0.25f;
  };
  auto uval = [](std::size_t r, std::size_t t) {
    return static_cast<float>(
               small_int(static_cast<std::uint32_t>(r * 617u + t * 53u + 7u),
                         9)) *
           0.5f;
  };

  auto run = [&](std::int64_t m) -> std::vector<float> {
    Array g = Array::zeros(Shape{m, kInner}, DType::kF32);
    Array u = Array::zeros(Shape{m, kInner}, DType::kF32);
    Array hid = silu(g) * u;
    const NodePtr roots[] = {hid.node()};
    const auto wgs = Partitioner::phases(roots);
    LSE_EXPECT(!wgs.empty());
    if (wgs.empty()) return {};
    const FusionGroup grp = Partitioner::phase_group(wgs[0], roots);
    auto e = backend::HipEmitter::emit_phase(grp, dev);
    LSE_EXPECT(e.ok());
    if (!e.ok()) {
      std::printf("       m=%lld: %s\n", static_cast<long long>(m),
                  e.status().to_string().c_str());
      return {};
    }
    // One launch for the pair, and a grid that covers the whole pass.
    const auto want_wgs =
        static_cast<std::uint32_t>((m * kInner + 255) / 256);
    LSE_EXPECT_EQ(e->dims.workgroup_count[0], want_wgs);

    const auto n = static_cast<std::size_t>(m * kInner);
    std::vector<float> gs(n), us(n);
    for (std::size_t r = 0; r < static_cast<std::size_t>(m); ++r) {
      for (std::size_t t = 0; t < static_cast<std::size_t>(kInner); ++t) {
        gs[r * kInner + t] = gval(r, t);
        us[r * kInner + t] = uval(r, t);
      }
    }
    const std::vector<std::vector<float>> got = run_emitted(
        *e, dev, *be, {{g.node().get(), gs}, {u.node().get(), us}});
    if (got.size() != e->binding_order.size()) {
      LSE_EXPECT(false);
      return {};
    }
    for (std::size_t i = 0; i < e->binding_order.size(); ++i) {
      if (e->binding_order[i].get() == hid.node().get()) return got[i];
    }
    LSE_EXPECT(false);
    return {};
  };

  const std::vector<float> wide = run(5);
  LSE_EXPECT(wide.size() == static_cast<std::size_t>(5 * kInner));
  if (wide.size() != static_cast<std::size_t>(5 * kInner)) return;

  for (const std::int64_t m : {std::int64_t{1}, std::int64_t{2},
                               std::int64_t{3}, std::int64_t{17}}) {
    const std::vector<float> got = run(m);
    LSE_EXPECT(got.size() == static_cast<std::size_t>(m * kInner));
    if (got.size() != static_cast<std::size_t>(m * kInner)) continue;
    const auto span = static_cast<std::ptrdiff_t>((m < 5 ? m : 5) * kInner);
    std::vector<float> a(got.begin(), got.begin() + span);
    std::vector<float> b(wide.begin(), wide.begin() + span);
    const double worst = worst_abs(a, b);
    if (worst != 0.0) {
      std::printf("       m=%lld differs from the 5-row pass by %g\n",
                  static_cast<long long>(m), worst);
    }
    LSE_EXPECT_EQ(worst, 0.0);
  }

  // Every row wrong the same way would survive the comparison above.
  std::vector<float> want(static_cast<std::size_t>(5 * kInner), 0.0f);
  for (std::size_t r = 0; r < 5; ++r) {
    for (std::size_t t = 0; t < static_cast<std::size_t>(kInner); ++t) {
      const float x = gval(r, t);
      want[r * kInner + t] = x / (1.0f + std::exp(-x)) * uval(r, t);
    }
  }
  double worst = 0.0;
  for (std::size_t i = 0; i < want.size(); ++i) {
    worst = std::max(worst, std::fabs(static_cast<double>(wide[i]) -
                                      static_cast<double>(want[i])));
  }
  // __expf is the device's fast exponential, so this is an agreement bound and
  // not an equality: measured worst here is ~1e-7.
  if (worst > 1e-5) std::printf("       host disagreement %g\n", worst);
  LSE_EXPECT(worst <= 1e-5);
}
