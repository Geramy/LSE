// Emitter -> comgr -> code object. Runs the full JIT path when ROCm is
// present; the source-shape checks run everywhere.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#include <array>
#include <cstring>
#include <memory>
#include <optional>
#include <string>

#include "harness.hpp"
#include "lse/backend/backend.hpp"
#include "lse/core/debug.hpp"
#include "lse/backends/cpu/cpu_backend.hpp"
#include "lse/backend/resources.hpp"
#include "lse/opt/arrangement.hpp"
#include "lse/opt/occupancy.hpp"
#include "lse/opt/fusion.hpp"
#include "lse/opt/measurements.hpp"
#include "lse/opt/traffic.hpp"
#include "lse/backends/hrx/arch_database.hpp"
#include "lse/backends/hrx/code_object.hpp"
#include "lse/backends/hrx/hrx_backend.hpp"
#include "lse/backends/hrx/hipc/comgr_compiler.hpp"
#include "lse/backends/hrx/device_info.hpp"
#include "lse/backends/hrx/hipc/hip_emitter.hpp"
#include "lse/backends/hrx/hipc/hip_sources.hpp"
#include "lse/backends/hrx/hipc/hip_types.hpp"
#include "lse/backends/hrx/loomc/loom_emitter.hpp"
#include "lse/backends/hrx/loomc/loomc_compiler.hpp"
#include "lse/kernels/wmma.hpp"
#include "lse/graph/kernel_args.hpp"
#include "lse/graph/kernel_env.hpp"
#include "lse/graph/kernel_primitive.hpp"
#include "lse/graph/graph.hpp"
#include "lse/graph/interpreter.hpp"
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
    // What the kRdna35 ISA row says, which is what apply_arch_defaults hands
    // a live gfx1151. Stated here because this object is the extension the
    // fixture publishes and nothing fills it in on the way.
    a.has_dot4_i8 = true;
    a.has_dot4_iu8 = true;
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
  // The same source a live device's facts come from, so the occupancy model
  // under test is the one that will run: the compiler's ISA table where it
  // answers, the family row where it does not.
  d.arch_facts = backend::arch_facts_for(d);
  return d;
}

const backend::HipEmitter kEmitter;
const backend::ComgrCompiler kCompiler;
const backend::LoomcCompiler kLoom;

// A complete Loom kernel for one concrete matmul shape: one thread per output
// element, a sequential fmaf walk over K. Bit-exact against the HIP kernel for
// the same shape, so it is the fixture the loom half of the toolchain is
// checked against.
std::string loom_matmul_source(int m, int k, int n, unsigned threads) {
  const auto elems = static_cast<unsigned>(m * n);
  const std::string md = std::to_string(m);
  const std::string kd = std::to_string(k);
  const std::string nd = std::to_string(n);
  const std::string xt = "view<" + md + "x" + kd + "xf32, #dense>";
  const std::string yt = "view<" + kd + "x" + nd + "xf32, #dense>";
  const std::string ot = "view<" + md + "x" + nd + "xf32, #dense>";
  std::string s;
  s += "kernel.def export(\"lse_matmul_loom\") @lse_matmul_loom() {\n";
  s += "  %unit = index.constant 1 : index\n";
  s += "  %wg = index.constant " + std::to_string(threads) + " : index\n";
  s += "  %groups = index.constant " +
       std::to_string((elems + threads - 1) / threads) + " : index\n";
  s += "  kernel.launch.config workgroups(%groups, %unit, %unit) "
       "workgroup_size(%wg, %unit, %unit) : index\n";
  s += "} launch(%x: buffer, %y: buffer, %out: buffer, %count: i32) {\n";
  s += "  %base = index.constant 0 : offset\n";
  s += "  %zero = index.constant 0 : index\n";
  s += "  %unit = index.constant 1 : index\n";
  s += "  %wg = index.constant " + std::to_string(threads) + " : index\n";
  s += "  %kdim = index.constant " + kd + " : index\n";
  s += "  %cols = index.constant " + nd + " : index\n";
  s += "  %zero_f32 = scalar.constant 0.0 : f32\n";
  s += "  %group = kernel.workgroup.id<x> : index\n";
  s += "  %lane = kernel.workitem.id<x> : index\n";
  s += "  %i = index.madd %group, %wg, %lane : index\n";
  s += "  %elems = index.constant " + std::to_string(elems) + " : index\n";
  s += "  %limit0 = index.cast %count : i32 to index\n";
  s += "  %limit = index.assume %limit0 [range(%limit0, 0, " +
       std::to_string(elems) + "), le(%limit0, %elems)] : index\n";
  s += "  %live = index.cmp ult, %i, %limit : index\n";
  s += "  %xn, %yn, %on = buffer.assume.noalias %x, %y, %out : buffer, buffer, "
       "buffer\n";
  s += "  %xv = buffer.view %xn[%base] : buffer -> " + xt + "\n";
  s += "  %yv = buffer.view %yn[%base] : buffer -> " + yt + "\n";
  s += "  %ov = buffer.view %on[%base] : buffer -> " + ot + "\n";
  s += "  scf.if %live {\n";
  s += "    %row = index.div %i, %cols : index\n";
  s += "    %col = index.rem %i, %cols : index\n";
  s += "    %acc = scf.for %t = [%zero to %kdim step %unit](%a = %zero_f32 : "
       "f32) -> (f32) {\n";
  s += "      %xval = view.load %xv[%row, %t] : " + xt + " -> f32\n";
  s += "      %yval = view.load %yv[%t, %col] : " + yt + " -> f32\n";
  s += "      %next = scalar.fmaf %xval, %yval, %a : f32\n";
  s += "      scf.yield %next : f32\n";
  s += "    }\n";
  s += "    view.store %acc, %ov[%row, %col] : f32, " + ot + "\n";
  s += "  }\n";
  s += "  kernel.return\n";
  s += "}\n";
  return s;
}

// Value of `key=` in an identity string, up to the next space.
std::string identity_field(const std::string& id, std::string_view key) {
  const std::size_t at = id.find(key);
  if (at == std::string::npos) return {};
  const std::size_t begin = at + key.size();
  const std::size_t end = id.find(' ', begin);
  return id.substr(begin, end == std::string::npos ? std::string::npos
                                                   : end - begin);
}

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
  // The launch geometry is stated, not left at the target's "unknown" default
  // of 1024 — see kernel_signature in hip_emitter.cpp — and the number is the
  // flat size the dims say it will be dispatched at.
  const std::uint32_t flat = e->dims.workgroup_size[0] *
                             e->dims.workgroup_size[1] *
                             e->dims.workgroup_size[2];
  LSE_EXPECT(e->source.find("extern \"C\" __global__ __launch_bounds__(" +
                            std::to_string(flat) + ") void") !=
             std::string::npos);
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
  constexpr std::int64_t kGroup = 64;
  constexpr std::int64_t kN = 16;
  const std::int64_t lds_floats = gfx1151().lds_bytes_per_workgroup / 4;

  // Both widths: 4 bits reads the row to quantize it and 8 bits reads it to
  // multiply it, and the row term has to survive either way.
  for (const int bits : {4, 8}) {
    for (const std::int64_t k :
         {std::int64_t{640}, (lds_floats / kGroup + 1) * kGroup}) {
      const std::int64_t lanes = k * bits / 32;
      const std::int64_t groups = k / kGroup;
      Array x = Array::zeros(Shape{5, k}, DType::kF32);
      Array packed = Array::zeros(Shape{kN, lanes}, DType::kU32);
      Array scales = Array::zeros(Shape{kN, groups}, DType::kBF16);
      Array biases = Array::zeros(Shape{kN, groups}, DType::kBF16);

      Array y = quant_linear(x, packed, scales, biases, bits, kGroup);
      auto e = emit_anchor(y, OpKind::kQuantMatMul);
      LSE_EXPECT(e.ok());
      if (!e.ok()) {
        std::printf("       bits=%d K=%lld: %s\n", bits,
                    static_cast<long long>(k), e.status().to_string().c_str());
        continue;
      }
      const std::string stride = "* " + std::to_string(k) + "u";
      LSE_EXPECT(e->source.find(stride) != std::string::npos);
    }
  }
}

// A 4-bit contraction on a device whose arch database reports the mixed-
// signedness dot product must emit it. This is the path's only static proof
// that it is reachable: the numerics tests pass whichever inner loop ran, and
// a capability check that quietly never fires would leave the integer path as
// code that cannot execute.
LSE_TEST(quant_linear_4bit_contracts_with_the_dot_product) {
  constexpr std::int64_t kGroup = 64;
  constexpr std::int64_t kN = 16;
  constexpr std::int64_t kK = 1024;
  constexpr std::string_view kDot = "__builtin_amdgcn_sudot4";

  const auto emit_at = [&](int bits) -> std::string {
    const std::int64_t lanes = kK * bits / 32;
    const std::int64_t groups = kK / kGroup;
    Array x = Array::zeros(Shape{2, kK}, DType::kF32);
    Array packed = Array::zeros(Shape{kN, lanes}, DType::kU32);
    Array scales = Array::zeros(Shape{kN, groups}, DType::kBF16);
    Array biases = Array::zeros(Shape{kN, groups}, DType::kBF16);
    Array y = quant_linear(x, packed, scales, biases, bits, kGroup);
    auto e = emit_anchor(y, OpKind::kQuantMatMul);
    return e.ok() ? e->source : std::string{};
  };

  const auto count = [](const std::string& src, std::string_view what) {
    std::size_t n = 0, at = 0;
    while ((at = src.find(what, at)) != std::string::npos) {
      ++n;
      at += what.size();
    }
    return n;
  };

  const std::string four = emit_at(4);
  LSE_EXPECT(!four.empty());
  LSE_EXPECT(four.find(kDot) != std::string::npos);

  // 8 bits stays on the float codec — its packed word is already four unsigned
  // bytes, so the win there is a different change and is not made here.
  const std::string eight = emit_at(8);
  LSE_EXPECT(!eight.empty());
  LSE_EXPECT(eight.find(kDot) == std::string::npos);

  // The structural claim, not just the presence of a builtin: the float codec
  // spends two fmaf per weight — one to place the code, one to accumulate it —
  // and the integer path spends none. A lane decodes 16 codes per step at 8
  // bits, so that side is 32 of them; the integer side keeps only the group
  // scale, the group bias and the per-chunk activation step, and it keeps one
  // of each for every row the workgroup covers. This x is two rows, so the
  // bound is per row — what the claim rests on is that neither side of it
  // grows with K.
  constexpr std::size_t kRows = 2;
  LSE_EXPECT(count(eight, "fmaf(") >= 32);
  LSE_EXPECT(count(four, "fmaf(") < 8 * kRows);

  // And a device without the capability takes the float codec at 4 bits too.
  // A fresh emitter, because the emit cache keys on the group and the wave
  // width and would otherwise hand back the source emitted just above.
  const backend::HipEmitter fresh;
  backend::AmdDeviceInfo no_dot = gfx1151_amd();
  no_dot.has_dot4_iu8 = false;
  backend::DeviceInfo plain = gfx1151();
  plain.extension = &no_dot;
  const std::int64_t lanes = kK * 4 / 32;
  const std::int64_t groups = kK / kGroup;
  Array x = Array::zeros(Shape{2, kK}, DType::kF32);
  Array packed = Array::zeros(Shape{kN, lanes}, DType::kU32);
  Array scales = Array::zeros(Shape{kN, groups}, DType::kBF16);
  Array biases = Array::zeros(Shape{kN, groups}, DType::kBF16);
  Array y = quant_linear(x, packed, scales, biases, 4, kGroup);
  const NodePtr roots[] = {y.node()};
  for (const FusionGroup& g : Partitioner::partition(roots)) {
    if (g.anchor != OpKind::kQuantMatMul) continue;
    auto e = fresh.emit(g, plain);
    LSE_EXPECT(e.ok());
    if (e.ok()) {
      LSE_EXPECT(e->source.find(kDot) == std::string::npos);
      LSE_EXPECT(count(e->source, "fmaf(") > 32);
    }
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
  opt::KernelDemand demand;
  demand.threads = threads;
  LSE_EXPECT(opt::occupancy(opt::DeviceCapacity::of(d), demand).seated());

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
  LSE_EXPECT(e->source.find("__global__ __launch_bounds__(") !=
             std::string::npos);
  LSE_EXPECT(e->source.find(") void lse_fused_") != std::string::npos);
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
  const std::vector<std::byte>& obj = code->code;
  LSE_EXPECT(obj.size() > 512u);
  LSE_EXPECT(static_cast<unsigned char>(obj[0]) == 0x7F);
  LSE_EXPECT(static_cast<char>(obj[1]) == 'E');
  LSE_EXPECT(static_cast<char>(obj[2]) == 'L');
  LSE_EXPECT(static_cast<char>(obj[3]) == 'F');
  std::printf("       code object: %zu bytes for gfx1151\n", obj.size());
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

// --- facts and measurement -------------------------------------------------

LSE_TEST(an_unanswered_fact_is_not_a_zero) {
  // The whole contract of this file in one check: a fact nobody answered holds
  // a zero so a careless reader gets something spottable, and known() is false
  // so a careful one never spends it.
  backend::DeviceFact<std::uint32_t> nothing;
  LSE_EXPECT(!nothing.known());
  LSE_EXPECT_EQ(nothing.value, 0u);
  LSE_EXPECT_EQ(nothing.value_or(77u), 77u);
  LSE_EXPECT_EQ(backend::DeviceFact<std::uint32_t>::queried(0u).value_or(77u), 0u);
  LSE_EXPECT(backend::DeviceFact<std::uint32_t>::queried(0u).known());

  // And the case it exists for: a reported zero and an unreported count are
  // different answers about spilling.
  backend::KernelResources silent;
  LSE_EXPECT(silent.spilled() == backend::SpillState::kUnknown);
  backend::KernelResources clean;
  clean.vector_spills = backend::DeviceFact<std::uint32_t>::queried(0);
  clean.scalar_spills = backend::DeviceFact<std::uint32_t>::queried(0);
  LSE_EXPECT(clean.spilled() == backend::SpillState::kNone);
  backend::KernelResources spilling = clean;
  spilling.vector_spills = backend::DeviceFact<std::uint32_t>::queried(4);
  LSE_EXPECT(spilling.spilled() == backend::SpillState::kSpilled);
  // Scratch is not spill evidence: a kernel in this tree's own cache carries
  // 192 B of private segment with both counts reported and zero.
  backend::KernelResources scratchy = clean;
  scratchy.private_segment_bytes =
      backend::DeviceFact<std::uint32_t>::queried(192);
  LSE_EXPECT(scratchy.spilled() == backend::SpillState::kNone);
}

LSE_TEST(isa_facts_are_queried_from_the_compilers_own_target_table) {
  if (!kCompiler.available()) {
    std::printf("       (skipped: comgr not in this build)\n");
    return;
  }
  const backend::ArchFacts f = backend::query_isa_facts("gfx1151");
  LSE_EXPECT(f.any());
  LSE_EXPECT(f.vector_registers_per_simd.known());
  LSE_EXPECT(f.vector_register_alloc_granule.known());
  LSE_EXPECT(f.vector_registers_addressable_per_wave.known());
  LSE_EXPECT(f.lds_bytes_addressable_per_workgroup.known());
  LSE_EXPECT(f.max_flat_workgroup_size.known());
  LSE_EXPECT(f.vector_registers_per_simd.source == backend::FactSource::kQueried);
  // Sanity, not a hardcoded expectation: a register file that does not divide
  // into whole allocation granules would mean one of the two was misread.
  LSE_EXPECT(f.vector_register_alloc_granule.value > 0u);
  LSE_EXPECT(f.vector_registers_per_simd.value >=
             f.vector_registers_addressable_per_wave.value);
  std::printf("       gfx1151 %u vgprs/SIMD, granule %u, %u addressable, "
              "%u B LDS/wg, %u banks\n",
              f.vector_registers_per_simd.value,
              f.vector_register_alloc_granule.value,
              f.vector_registers_addressable_per_wave.value,
              f.lds_bytes_addressable_per_workgroup.value,
              f.lds_banks.value);

  // A target nothing knows answers nothing, rather than answering zero.
  const backend::ArchFacts none = backend::query_isa_facts("gfx0000");
  LSE_EXPECT(!none.any());
  LSE_EXPECT(!none.vector_registers_per_simd.known());
}

LSE_TEST(isa_facts_are_real_per_target_data_not_one_constant) {
  if (!kCompiler.available()) return;
  // If these were compiler-model constants rather than per-target facts they
  // would be identical across targets, and querying them would buy nothing
  // over a hardcoded row. Two RDNA generations and one CDNA part disagree, so
  // the query is load-bearing. (This is also why MaxWavesPerCU and EUsPerCU
  // are deliberately not among the facts: those two ARE identical everywhere.)
  const backend::ArchFacts a = backend::query_isa_facts("gfx1151");
  const backend::ArchFacts b = backend::query_isa_facts("gfx1030");
  const backend::ArchFacts c = backend::query_isa_facts("gfx942");
  LSE_EXPECT(a.vector_registers_per_simd.known());
  LSE_EXPECT(b.vector_registers_per_simd.known());
  LSE_EXPECT(c.vector_registers_per_simd.known());
  LSE_EXPECT(a.vector_registers_per_simd.value !=
             b.vector_registers_per_simd.value);
  LSE_EXPECT(a.vector_registers_addressable_per_wave.value !=
             c.vector_registers_addressable_per_wave.value);
  std::printf("       vgprs/SIMD: gfx1151 %u, gfx1030 %u, gfx942 %u\n",
              a.vector_registers_per_simd.value,
              b.vector_registers_per_simd.value,
              c.vector_registers_per_simd.value);
}

LSE_TEST(the_arch_table_and_the_compiler_agree_where_both_answer) {
  if (!kCompiler.available()) return;
  // Two sources for one number is two places for it to be wrong. Where the
  // compiler answers, the family row must match it; when this fails after a
  // ROCm bump, the compiler is right and the row is stale.
  int checked = 0;
  for (const char* arch : {"gfx1151", "gfx1100", "gfx1201", "gfx1030",
                           "gfx90a", "gfx942"}) {
    const backend::ArchFacts f = backend::query_isa_facts(arch);
    const backend::FamilyIsa* row =
        backend::family_isa(backend::arch_family(arch));
    if (!f.lds_bytes_addressable_per_workgroup.known() || row == nullptr) {
      continue;
    }
    LSE_EXPECT_EQ(f.lds_bytes_addressable_per_workgroup.value,
                  row->lds_bytes_per_workgroup);
    LSE_EXPECT_EQ(f.max_flat_workgroup_size.value,
                  static_cast<std::uint32_t>(row->max_threads_per_workgroup));
    ++checked;
  }
  LSE_EXPECT(checked >= 4);
}

LSE_TEST(a_compiled_hip_kernel_carries_what_the_compiler_measured) {
  if (!kCompiler.available()) return;

  const std::string entry = "lse_measure_probe";
  const std::string source =
      "#include <hip/hip_runtime.h>\n"
      "extern \"C\" __global__ void " + entry +
      "(float* __restrict__ out, const float* __restrict__ in, unsigned n) {\n"
      "  __shared__ float tile[512];\n"
      "  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;\n"
      "  tile[threadIdx.x & 511u] = i < n ? in[i] : 0.0f;\n"
      "  __syncthreads();\n"
      "  if (i < n) out[i] = tile[threadIdx.x & 511u] * 2.0f;\n"
      "}\n";
  auto built = kCompiler.compile(source, "gfx1151");
  LSE_EXPECT_OK(built.status());
  if (!built.ok()) return;

  const backend::KernelResources* r = built->resources_for(entry);
  LSE_EXPECT(r != nullptr);
  if (r == nullptr) return;
  LSE_EXPECT(r->entry == entry);
  LSE_EXPECT(r->vector_registers.known());
  LSE_EXPECT(r->vector_registers.value > 0u);
  LSE_EXPECT(r->scalar_registers.known());
  LSE_EXPECT(r->workgroup_segment_bytes.known());
  // The kernel asks for exactly 512 floats of workgroup scratch, so this is
  // the compiler's own answer to a question we know the answer to — which is
  // what makes it a measurement of the reader and not just of the kernel.
  LSE_EXPECT_EQ(r->workgroup_segment_bytes.value, 2048u);
  LSE_EXPECT(r->private_segment_bytes.known());
  LSE_EXPECT(r->wavefront_size.known());
  // The HIP path's producer emits spill counts even when they are zero, so
  // this dialect can always answer the question.
  LSE_EXPECT(r->spilled() != backend::SpillState::kUnknown);
  std::printf("       %s", r->describe().c_str());
}

// The census reader, on a kernel whose traffic is known by construction: one
// 16-byte load and one 16-byte store per thread, staged through workgroup
// scratch. Nothing loops and nothing is conditional, so the static count is the
// whole body and every number below is arithmetic anyone can redo by hand.
LSE_TEST(an_object_census_counts_what_the_kernel_was_told_to_move) {
  if (!kCompiler.available()) return;

  const std::string entry = "lse_census_probe";
  const std::string source =
      "#include <hip/hip_runtime.h>\n"
      "extern \"C\" __global__ void " + entry +
      "(float4* __restrict__ out, const float4* __restrict__ in) {\n"
      "  __shared__ float4 tile[256];\n"
      "  const unsigned i = threadIdx.x;\n"
      "  tile[i] = in[i];\n"
      "  __syncthreads();\n"
      "  out[i] = tile[255u - i];\n"
      "}\n";
  auto built = kCompiler.compile(source, "gfx1151");
  LSE_EXPECT_OK(built.status());
  if (!built.ok()) return;

  const backend::KernelCensus* c = built->census_for(entry);
  LSE_EXPECT(c != nullptr);
  if (c == nullptr) return;
  LSE_EXPECT(c->any());
  LSE_EXPECT(c->instructions.value > 0u);
  // Every instruction in this body is one the classifier knows. A census that
  // cannot name what it read is not evidence about what it did not find.
  LSE_EXPECT_EQ(c->unclassified.value, 0u);
  LSE_EXPECT(c->straight_line());

  LSE_EXPECT_EQ(c->global_loads.bytes(), 16u);
  LSE_EXPECT_EQ(c->global_stores.bytes(), 16u);
  LSE_EXPECT_EQ(c->shared_loads.bytes(), 16u);
  LSE_EXPECT_EQ(c->shared_stores.bytes(), 16u);
  // The load is 16 bytes wide because the source asked for a float4; a reader
  // that split it into four dwords would report the same bytes and four times
  // the instructions.
  LSE_EXPECT_EQ(c->global_loads.count(), 1u);
  // One load, and one wait that retires it: with a single load in flight the
  // batch depth is 1 by construction, which is what pins the wait accounting.
  LSE_EXPECT_EQ(c->memory_waits.value, 1u);
  LSE_EXPECT_EQ(c->deepest_load_batch.value, 1u);
  LSE_EXPECT_EQ(c->serializing_waits.value, 1u);
  // The kernarg pointers arrive through the scalar path, once per wave.
  LSE_EXPECT(c->scalar_loads.count() > 0u);

  // And the engine reads it as bytes for a whole workgroup without knowing a
  // single instruction name.
  const opt::EmittedTraffic t = opt::emitted_traffic(*c, 256);
  LSE_EXPECT(t.known);
  LSE_EXPECT(t.exact);
  LSE_EXPECT_EQ(t.global_read, 16u * 256u);
  LSE_EXPECT_EQ(t.global_written, 16u * 256u);
  std::printf("       %s", c->describe().c_str());
}

// A loop is the difference between a count and a measurement, so the census has
// to say which one it produced. The body below reads one dword per iteration of
// a loop whose length the kernel is told at runtime: the static count is one
// load and the launch's traffic is not.
LSE_TEST(a_census_says_which_of_its_counts_can_repeat) {
  if (!kCompiler.available()) return;

  const std::string entry = "lse_census_loop_probe";
  const std::string source =
      "#include <hip/hip_runtime.h>\n"
      "extern \"C\" __global__ void " + entry +
      "(float* __restrict__ out, const float* __restrict__ in, unsigned n) {\n"
      "  float acc = 0.0f;\n"
      "  for (unsigned j = threadIdx.x; j < n; j += 256u) acc += in[j];\n"
      "  out[threadIdx.x] = acc;\n"
      "}\n";
  auto built = kCompiler.compile(source, "gfx1151");
  LSE_EXPECT_OK(built.status());
  if (!built.ok()) return;
  const backend::KernelCensus* c = built->census_for(entry);
  LSE_EXPECT(c != nullptr);
  if (c == nullptr) return;

  LSE_EXPECT(!c->straight_line());
  LSE_EXPECT(c->backward_branches.value > 0u);
  LSE_EXPECT(c->global_loads.loops());
  // The store is outside the loop, so it is the whole of the floor.
  LSE_EXPECT_EQ(c->global_loads.straight_line_bytes(), 0u);
  LSE_EXPECT_EQ(c->global_stores.straight_line_bytes(), 4u);
  const opt::EmittedTraffic t = opt::emitted_traffic(*c, 256);
  LSE_EXPECT(!t.exact);
  LSE_EXPECT_EQ(t.global_floor, 4u * 256u);
  std::printf("       %s", c->describe().c_str());
}

// The engine's own arithmetic, at the two shapes that decide the 27B: the hidden
// contraction at K=5120 and the projection at K=17408. The point of the split is
// that a ROW tile divides only the activation term and a COLUMN tile only the
// weight term, so the balance says which axis is worth moving — and the balance
// is shape dependent, which is why it is computed and not tabulated.
LSE_TEST(the_traffic_split_prices_both_terms_of_a_contraction) {
  struct Site {
    const char* what;
    std::uint64_t k;
    std::uint32_t rows;
  };
  const Site sites[] = {{"hidden K=5120", 5120, 8}, {"proj K=17408", 17408, 8}};
  for (const Site& site : sites) {
    const std::uint64_t groups = site.k / 64;
    const opt::TrafficModel m = opt::contraction_traffic(
        site.rows, 8, site.k, 4, 4, 8 * groups * 2 * 2,
        static_cast<std::uint64_t>(site.rows) * 8 * 4);
    LSE_EXPECT(m.stated);
    const std::uint64_t w = m.bytes_read(opt::OperandClass::kWeight);
    const std::uint64_t a = m.bytes_read(opt::OperandClass::kActivation);
    // Both derivable by hand: 8 columns of K half-bytes against R rows of K
    // floats. The weight term is the smaller one at every reachable tile, which
    // is the whole finding.
    LSE_EXPECT_EQ(w, 8u * site.k / 2u);
    LSE_EXPECT_EQ(a, site.rows * site.k * 4u);
    LSE_EXPECT(a > w);
    std::printf("       %s R=%u: weight %llu B, activation %llu B, "
                "activation %.0f%% of reads, %.3f B/mac\n",
                site.what, site.rows, static_cast<unsigned long long>(w),
                static_cast<unsigned long long>(a),
                m.read_share(opt::OperandClass::kActivation) * 100.0,
                m.bytes_per_work());
  }

  // The row tile only ever divides the smaller term, so its whole reach is the
  // ratio below — the same ratio at both widths, which is what makes it a rule
  // about the schedule rather than a fact about one matrix.
  for (const std::uint64_t k : {std::uint64_t{5120}, std::uint64_t{17408}}) {
    const opt::TrafficModel r1 = opt::contraction_traffic(1, 8, k, 4, 4, 0, 0);
    const opt::TrafficModel r8 = opt::contraction_traffic(8, 8, k, 4, 4, 0, 0);
    LSE_EXPECT(r8.bytes_per_work() < r1.bytes_per_work());
    // And the column axis, which nothing has ever moved: it divides the term
    // that dominates, so it reaches further than the row axis ever could.
    const opt::TrafficModel c16 = opt::contraction_traffic(8, 16, k, 4, 4, 0, 0);
    LSE_EXPECT(c16.bytes_per_work() < r8.bytes_per_work());
    std::printf("       K=%llu B/mac: R=1 %.3f, R=8 %.3f, R=8 c=16 %.3f\n",
                static_cast<unsigned long long>(k), r1.bytes_per_work(),
                r8.bytes_per_work(), c16.bytes_per_work());
  }
}

// THE DISAGREEMENT CHECK, on kernels this engine really emits.
//
// The failure it exists to catch is on record: a traffic model that priced the
// weight and forgot the activation, which made a row tile look like a 26.6% win
// that was never available. Here the same model is built both ways against the
// same compiled object — correctly, and weight-only — and the object has to
// contradict the second one.
//
// HOW BADLY IT CONTRADICTS IS THE ROW TILE'S OWN FACTOR, so which K clears the
// instrument's measured 3x spread moves when the tile does. Weight-only
// understates by the ratio of the whole workgroup's reads to its weight reads,
// which is set by the rows the tile covers; at K=17408 the tile is two rows and
// the object reads 4.1x the model, decidably more, while at K=5120 it is four
// and the 2.5x sits inside the spread. So the assertion is in two parts: the
// weight-only model must read high at EVERY shape — that part is a property of
// the model and holds whatever the tile is — and it must be called a
// disagreement wherever the gap clears what the instrument can resolve.
LSE_TEST(a_traffic_model_that_forgets_a_term_disagrees_with_the_object) {
  if (!kCompiler.available()) return;
  constexpr std::int64_t kGroup = 64;
  constexpr std::int64_t kN = 16;
  constexpr std::int64_t kRows = 8;
  int caught = 0;

  for (const std::int64_t k : {std::int64_t{5120}, std::int64_t{17408}}) {
    const std::int64_t lanes = k * 4 / 32;
    const std::int64_t groups = k / kGroup;
    Array x = Array::zeros(Shape{kRows, k}, DType::kF32);
    Array packed = Array::zeros(Shape{kN, lanes}, DType::kU32);
    Array scales = Array::zeros(Shape{kN, groups}, DType::kBF16);
    Array biases = Array::zeros(Shape{kN, groups}, DType::kBF16);
    Array y = quant_linear(x, packed, scales, biases, 4, kGroup);
    auto e = emit_anchor(y, OpKind::kQuantMatMul);
    LSE_EXPECT(e.ok());
    if (!e.ok()) continue;

    // The emitter states what the run means to move, split by class.
    const opt::TrafficModel& intended = e->traffic;
    LSE_EXPECT(intended.stated);
    if (!intended.stated) continue;
    LSE_EXPECT(intended.workgroup_threads > 0u);
    const std::uint64_t weight =
        intended.bytes_read(opt::OperandClass::kWeight);
    const std::uint64_t activation =
        intended.bytes_read(opt::OperandClass::kActivation);
    LSE_EXPECT(weight > 0u);
    LSE_EXPECT(activation > weight);

    auto built = kCompiler.compile(e->source, "gfx1151");
    LSE_EXPECT_OK(built.status());
    if (!built.ok()) continue;
    const backend::KernelCensus* c = built->census_for(e->entry_name);
    LSE_EXPECT(c != nullptr);
    if (c == nullptr) continue;
    LSE_EXPECT_EQ(c->unclassified.value, 0u);
    // The integer inner loop is the reason this shape is interesting at all.
    LSE_EXPECT(c->dot_products.value > 0u);

    const opt::TrafficCheck honest =
        opt::check_traffic(intended, *c, intended.workgroup_threads);
    LSE_EXPECT(!honest.disagrees());

    // Now the model as it was when the row tile was mispriced: weights only.
    opt::TrafficModel weights_only;
    weights_only.add_read(opt::OperandClass::kWeight, weight);
    weights_only.work = intended.work;
    const opt::TrafficCheck wrong =
        opt::check_traffic(weights_only, *c, intended.workgroup_threads);

    std::printf("       K=%lld weight %llu B activation %llu B "
                "(activation %.0f%% of reads)\n",
                static_cast<long long>(k),
                static_cast<unsigned long long>(weight),
                static_cast<unsigned long long>(activation),
                intended.read_share(opt::OperandClass::kActivation) * 100.0);
    std::printf("       honest %s", honest.describe().c_str());
    std::printf("       weight-only %s", wrong.describe().c_str());
    // Reads high at every shape, and is never mistaken for agreement.
    LSE_EXPECT(wrong.intensity_ratio() > honest.intensity_ratio());
    LSE_EXPECT(wrong.verdict != opt::TrafficVerdict::kAgrees);
    // And is caught outright wherever the gap clears the instrument's spread.
    if (wrong.intensity_ratio() > opt::kTrafficDisagreementFactor) {
      LSE_EXPECT(wrong.verdict == opt::TrafficVerdict::kEmitsMore);
      ++caught;
    }
  }
  // A check that could never fire would pass on a tree with no instrument at
  // all, so at least one of the shapes has to have been decided.
  LSE_EXPECT(caught > 0);
}

LSE_TEST(a_compiled_loom_kernel_reports_registers_and_declines_spills) {
  if (!kLoom.available() || !kCompiler.available()) return;

  auto built = kLoom.compile(loom_matmul_source(32, 16, 32, 64), "gfx1151");
  LSE_EXPECT_OK(built.status());
  if (!built.ok()) return;
  LSE_EXPECT(!built->resources.empty());
  if (built->resources.empty()) return;

  const backend::KernelResources& r = built->resources.front();
  // loomc's HSACO carries the same amdhsa.kernels note, so one reader serves
  // both dialects: this is the evidence, not an assumption.
  LSE_EXPECT(!r.entry.empty());
  LSE_EXPECT(r.vector_registers.known());
  LSE_EXPECT(r.scalar_registers.known());
  LSE_EXPECT(r.workgroup_segment_bytes.known());
  LSE_EXPECT(r.wavefront_size.known());
  // And what it does NOT carry. If loomc ever starts emitting spill counts
  // this flips, which is worth being told about rather than silently gaining
  // a fact nothing was checking for.
  LSE_EXPECT(!r.vector_spills.known());
  LSE_EXPECT(!r.scalar_spills.known());
  LSE_EXPECT(r.spilled() == backend::SpillState::kUnknown);
  // The compensating asymmetry: loomc states the kernel's own launch geometry
  // where the HIP path does not.
  LSE_EXPECT(r.required_workgroup_size.known());
  std::printf("       %s", r.describe().c_str());
}

LSE_TEST(the_loom_compiler_reports_available_when_loomc_is_linked) {
  if (!kLoom.available()) {
    std::printf("       (skipped: loomc not in this build)\n");
    LSE_EXPECT(kLoom.identity() == "no-compiler");
    return;
  }
  LSE_EXPECT(kLoom.available());
  std::printf("       %s\n", kLoom.identity().c_str());
}

LSE_TEST(a_loom_kernel_compiles_to_an_amdgpu_code_object) {
  if (!kLoom.available()) return;

  const std::string src = loom_matmul_source(32, 16, 32, 64);
  auto code = kLoom.compile(src, "gfx1151");
  if (!code.ok()) {
    std::printf("       compile failed:\n%s\n", code.status().message().c_str());
    std::printf("       ---- source ----\n%s\n", src.c_str());
  }
  LSE_EXPECT(code.ok());
  if (!code.ok()) return;

  LSE_EXPECT(code->code.size() > 512u);
  const auto* b = reinterpret_cast<const unsigned char*>(code->code.data());
  LSE_EXPECT(b[0] == 0x7F && b[1] == 'E' && b[2] == 'L' && b[3] == 'F');
  // ELFCLASS64, and EM_AMDGPU rather than whatever host object a misrouted
  // emitter would hand back.
  LSE_EXPECT_EQ(static_cast<int>(b[4]), 2);
  const int machine = b[18] | (b[19] << 8);
  LSE_EXPECT_EQ(machine, 224);
  std::printf("       code object: %zu bytes, e_machine=%d for gfx1151\n",
              code->code.size(), machine);
}

LSE_TEST(target_id_features_in_the_arch_string_reach_the_code_object) {
  if (!kLoom.available()) return;

  // `gfx942:sramecc+:xnack-` is a legal thing for HRX to report and the
  // suffixes select a different object, so they cannot be dropped on the way
  // into the target profile.
  const std::string src =
      "kernel.def export(\"lse_loom_store\") @lse_loom_store() {\n"
      "  %unit = index.constant 1 : index\n"
      "  %wg = index.constant 64 : index\n"
      "  kernel.launch.config workgroups(%unit, %unit, %unit) "
      "workgroup_size(%wg, %unit, %unit) : index\n"
      "} launch(%out: buffer) {\n"
      "  %base = index.constant 0 : offset\n"
      "  %lane = kernel.workitem.id<x> : index\n"
      "  %v = scalar.constant 1.0 : f32\n"
      "  %ov = buffer.view %out[%base] : buffer -> view<64xf32, #dense>\n"
      "  view.store %v, %ov[%lane] : f32, view<64xf32, #dense>\n"
      "  kernel.return\n"
      "}\n";

  // e_flags of an ELF64 header, where AMDGPU records the target id.
  const auto flags = [](const std::vector<std::byte>& code) {
    unsigned v = 0;
    for (int i = 3; i >= 0; --i) {
      v = (v << 8) | static_cast<unsigned char>(code[48u + static_cast<unsigned>(i)]);
    }
    return v;
  };

  auto plain = kLoom.compile(src, "gfx942");
  auto featured = kLoom.compile(src, "gfx942:sramecc+:xnack-");
  LSE_EXPECT(plain.ok());
  LSE_EXPECT(featured.ok());
  if (!plain.ok() || !featured.ok()) {
    if (!featured.ok()) {
      std::printf("       %s\n", featured.status().message().c_str());
    }
    return;
  }
  LSE_EXPECT(flags(plain->code) != flags(featured->code));
  std::printf("       gfx942 e_flags 0x%x vs sramecc+/xnack- 0x%x\n",
              flags(plain->code), flags(featured->code));
}

LSE_TEST(loom_identity_carries_the_install_and_every_option) {
  if (!kLoom.available()) return;
  const std::string id = kLoom.identity();

  // Stable across calls: the JIT cache reads it once per device and would
  // otherwise invalidate itself.
  LSE_EXPECT(id == kLoom.identity());
  // Arch is mixed in by the cache, not here, so it must not appear.
  LSE_EXPECT(id.find("gfx") == std::string::npos);

  // Every option the invocation runs with is named, so editing one moves the
  // key. The struct they come from is the same one compile() reads.
  LSE_EXPECT(id.find("loomc.") == 0u);
  LSE_EXPECT(id.find("pipeline=prepared_low") != std::string::npos);
  LSE_EXPECT(id.find("control_flow=cfg") != std::string::npos);
  LSE_EXPECT(id.find("max_errors=") != std::string::npos);
  LSE_EXPECT(id.find("compile_artifacts=") != std::string::npos);
  LSE_EXPECT(id.find("runtime_globals=") != std::string::npos);
  LSE_EXPECT(id.find("format=amdgpu-hsaco") != std::string::npos);
  LSE_EXPECT(id.find("manifest=none") != std::string::npos);

  // The install stamp is real: loomc publishes no version query and its .so
  // carries no build-id, so identity() stats the library it resolved. A rebuild
  // at the same package version has to move this or the cache serves objects a
  // different compiler produced.
  const std::string lib = identity_field(id, "lib=");
  LSE_EXPECT(!lib.empty());
  struct ::stat st{};
  LSE_EXPECT_EQ(::stat(lib.c_str(), &st), 0);
  LSE_EXPECT(identity_field(id, "size=") ==
             std::to_string(static_cast<long long>(st.st_size)));
  LSE_EXPECT(identity_field(id, "mtime=") ==
             std::to_string(static_cast<long long>(st.st_mtime)));

  // Two dialects on one device must not share a compiler identity.
  LSE_EXPECT(kLoom.identity() != kCompiler.identity());
}

LSE_TEST(loom_compile_errors_name_the_problem_instead_of_crashing) {
  if (!kLoom.available()) return;

  auto syntax = kLoom.compile("kernel.def @broken( {\n", "gfx1151");
  LSE_EXPECT(!syntax.ok());
  LSE_EXPECT(syntax.status().code() == StatusCode::kCompileError);
  // The diagnostic code and message, not a bare status name.
  LSE_EXPECT(syntax.status().message().find("PARSE/") != std::string::npos);

  // A well-formed module with no kernel.def has nothing to specialize, and that
  // is a different failure from a parse error.
  auto no_kernel = kLoom.compile(
      "func.def inline @helper(%a: f32) -> (f32) {\n"
      "  func.return %a : f32\n"
      "}\n",
      "gfx1151");
  LSE_EXPECT(!no_kernel.ok());
  LSE_EXPECT(no_kernel.status().message().find("kernel.def") !=
             std::string::npos);

  // A type error survives the parse and fails in lowering, so the stage the
  // message names has to change with it.
  const std::string bad_type =
      "kernel.def @bad_type() {\n"
      "  %unit = index.constant 1 : index\n"
      "  kernel.launch.config workgroups(%unit, %unit, %unit) "
      "workgroup_size(%unit, %unit, %unit) : index\n"
      "} launch(%out: buffer) {\n"
      "  %base = index.constant 0 : offset\n"
      "  %zero = index.constant 0 : index\n"
      "  %v = scalar.constant 1.0 : f32\n"
      "  %ov = buffer.view %out[%base] : buffer -> view<4xi32, #dense>\n"
      "  view.store %v, %ov[%zero] : f32, view<4xi32, #dense>\n"
      "  kernel.return\n"
      "}\n";
  auto typed = kLoom.compile(bad_type, "gfx1151");
  LSE_EXPECT(!typed.ok());
  LSE_EXPECT(typed.status().message().find("element type") != std::string::npos);
  std::printf("       %s\n", typed.status().message().c_str());

  auto no_arch = kLoom.compile(loom_matmul_source(4, 4, 4, 16), "");
  LSE_EXPECT(!no_arch.ok());
  LSE_EXPECT(no_arch.status().code() == StatusCode::kInvalidArgument);
  auto empty = kLoom.compile("", "gfx1151");
  LSE_EXPECT(!empty.ok());
  LSE_EXPECT(empty.status().code() == StatusCode::kInvalidArgument);

  // An architecture no loom target table knows is a real failure, and it must
  // arrive as a Status naming the arch.
  auto bad_arch = kLoom.compile(loom_matmul_source(4, 4, 4, 16), "gfx0000");
  LSE_EXPECT(!bad_arch.ok());
  LSE_EXPECT(bad_arch.status().message().find("gfx0000") != std::string::npos);

  // The toolchain is still usable after every one of those.
  auto good = kLoom.compile(loom_matmul_source(8, 8, 8, 32), "gfx1151");
  LSE_EXPECT(good.ok());
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
  Result<backend::DeviceBuffer> allocate(std::size_t, backend::MemoryClass,
                                         backend::Stream) override {
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
  std::span<const KernelToolchain> toolchains() const noexcept override {
    return {};
  }
};

// One member of a device set: its own geometry, its own executables. The
// executable id is per instance so a test can tell which device a handle was
// loaded on, which is the thing a shared cache must never get wrong.
struct SetStubBackend final : backend::IBackend {
  backend::DeviceInfo info;
  const IKernelCompiler* cc = nullptr;
  std::uint64_t exec_id = 1;
  std::array<KernelToolchain, 1> tcs{};

  SetStubBackend(std::uint64_t id, const IKernelCompiler* compiler)
      : cc(compiler), exec_id(id) {
    tcs[0] = KernelToolchain{Dialect::kHip, nullptr, cc};
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
  Result<backend::DeviceBuffer> allocate(std::size_t, backend::MemoryClass,
                                         backend::Stream) override {
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
  std::span<const KernelToolchain> toolchains() const noexcept override {
    return tcs;
  }
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
  Result<CompiledKernel> compile(std::string_view,
                                 std::string_view) const override {
    ++n;
    return CompiledKernel{std::vector<std::byte>(16, std::byte{1}), {}, {}};
  }
  bool available() const override { return true; }
  std::string identity() const override { return "counting-compiler"; }
};

struct NamedCompiler final : IKernelCompiler {
  std::string id;
  explicit NamedCompiler(std::string s) : id(std::move(s)) {}
  Result<CompiledKernel> compile(std::string_view,
                                 std::string_view) const override {
    return CompiledKernel{std::vector<std::byte>(16, std::byte{2}), {}, {}};
  }
  bool available() const override { return true; }
  std::string identity() const override { return id; }
};

struct NamedEmitter final : IKernelEmitter {
  Dialect d;
  explicit NamedEmitter(Dialect dialect) : d(dialect) {}
  Result<EmittedKernel> emit(const FusionGroup&,
                            const backend::DeviceInfo&) const override {
    EmittedKernel k;
    k.dialect = d;
    k.entry_name = "stub";
    return k;
  }
  Dialect dialect() const noexcept override { return d; }
  std::string_view prelude() const noexcept override { return {}; }
  DialectSourceTable sources() const noexcept override { return {}; }
};

// A device that declares two dialects. Nothing in the engine selects one yet;
// what this proves is that the seam can carry the second at all, and that the
// dialect-blind accessors keep answering with the first.
struct TwoDialectBackend final : backend::IBackend {
  backend::DeviceInfo info;
  NamedEmitter hip_emitter{Dialect::kHip};
  NamedEmitter loom_emitter{Dialect::kLoom};
  NamedCompiler hip_compiler{"hip-compiler"};
  NamedCompiler loom_compiler{"loom-compiler"};
  std::array<KernelToolchain, 2> tcs{};

  TwoDialectBackend() {
    info.arch = "gfx1151";
    info.name = "stub";
    tcs[0] = KernelToolchain{Dialect::kHip, &hip_emitter, &hip_compiler};
    tcs[1] = KernelToolchain{Dialect::kLoom, &loom_emitter, &loom_compiler};
  }
  Status init(int) override { return OkStatus(); }
  void shutdown() noexcept override {}
  const backend::DeviceInfo& device_info() const noexcept override {
    return info;
  }
  Result<backend::DeviceBuffer> allocate(std::size_t, backend::MemoryClass,
                                         backend::Stream) override {
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
      std::string_view, std::span<const std::byte>) override {
    return LSE_ERROR(kUnimplemented, "stub");
  }
  Status launch(const backend::KernelHandle&, const backend::LaunchDims&,
                const backend::DispatchArgs&,
                const backend::DispatchTarget&) override {
    return LSE_ERROR(kUnimplemented, "stub");
  }
  Status synchronize() override { return OkStatus(); }
  std::string_view name() const noexcept override { return "stub"; }
  std::span<const KernelToolchain> toolchains() const noexcept override {
    return tcs;
  }
};

// A compiler whose object is a function of its source, and a device that loads
// the first byte of an object as the executable id. Together they let a test
// say WHICH source's object a handle came from, which is the only way to ask
// whether the cache served the right dialect's bytes.
struct EchoCompiler final : IKernelCompiler {
  std::string id;
  explicit EchoCompiler(std::string s) : id(std::move(s)) {}
  Result<CompiledKernel> compile(std::string_view src,
                                 std::string_view) const override {
    return CompiledKernel{
        std::vector<std::byte>{
            static_cast<std::byte>(src.empty() ? 0 : src.front())},
        {},
        {}};
  }
  bool available() const override { return true; }
  std::string identity() const override { return id; }
};

// A compiler that reports a resource set with one fact deliberately missing,
// so a round trip through the cache can be asked whether the hole survived.
struct MeasuringCompiler final : IKernelCompiler {
  Result<CompiledKernel> compile(std::string_view src,
                                 std::string_view) const override {
    CompiledKernel out;
    out.code.assign(1, static_cast<std::byte>(src.empty() ? 0 : src.front()));
    backend::KernelResources r;
    r.entry = "lse_measured";
    r.vector_registers = backend::DeviceFact<std::uint32_t>::queried(102);
    r.scalar_registers = backend::DeviceFact<std::uint32_t>::queried(28);
    r.workgroup_segment_bytes =
        backend::DeviceFact<std::uint32_t>::queried(25600);
    r.private_segment_bytes = backend::DeviceFact<std::uint32_t>::queried(0);
    r.scalar_spills = backend::DeviceFact<std::uint32_t>::queried(0);
    // vector_spills deliberately left unanswered.
    r.required_workgroup_size =
        backend::DeviceFact<std::array<std::uint32_t, 3>>::queried({256, 1, 1});
    out.resources.push_back(std::move(r));
    out.census = census(out.code);
    return out;
  }
  std::vector<backend::KernelCensus> census(
      std::span<const std::byte>) const override {
    backend::KernelCensus c;
    c.entry = "lse_measured";
    c.instructions = backend::DeviceFact<std::uint32_t>::queried(1453);
    c.vector_alu = backend::DeviceFact<std::uint32_t>::queried(1031);
    c.dot_products = backend::DeviceFact<std::uint32_t>::queried(160);
    c.multiply_accumulates =
        backend::DeviceFact<std::uint32_t>::queried(640);
    c.backward_branches = backend::DeviceFact<std::uint32_t>::queried(10);
    c.memory_waits = backend::DeviceFact<std::uint32_t>::queried(26);
    c.deepest_load_batch = backend::DeviceFact<std::uint32_t>::queried(3);
    // scalar_alu and the rest deliberately left unanswered, the same hole the
    // resource half carries.
    c.global_loads.add(16, false);
    c.global_loads.add(16, true);
    c.global_loads.add(2, false);
    c.shared_stores.add(4, true);
    return {c};
  }
  bool available() const override { return true; }
  std::string identity() const override { return "measuring-compiler"; }
};

struct EchoBackend final : backend::IBackend {
  backend::DeviceInfo info;
  std::array<KernelToolchain, 2> tcs{};

  EchoBackend(const IKernelCompiler& hip, const IKernelCompiler& loom) {
    info.arch = "gfx1151";
    info.name = "echo";
    tcs[0] = KernelToolchain{Dialect::kHip, nullptr, &hip};
    tcs[1] = KernelToolchain{Dialect::kLoom, nullptr, &loom};
  }
  Status init(int) override { return OkStatus(); }
  void shutdown() noexcept override {}
  const backend::DeviceInfo& device_info() const noexcept override {
    return info;
  }
  Result<backend::DeviceBuffer> allocate(std::size_t, backend::MemoryClass,
                                         backend::Stream) override {
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
      std::string_view name, std::span<const std::byte> code) override {
    backend::KernelHandle h;
    h.executable = code.empty()
                       ? 0
                       : static_cast<std::uint64_t>(std::to_integer<unsigned char>(code.front()));
    h.name = std::string(name);
    return h;
  }
  Status launch(const backend::KernelHandle&, const backend::LaunchDims&,
                const backend::DispatchArgs&,
                const backend::DispatchTarget&) override {
    return LSE_ERROR(kUnimplemented, "stub");
  }
  Status synchronize() override { return OkStatus(); }
  std::string_view name() const noexcept override { return "echo"; }
  std::span<const KernelToolchain> toolchains() const noexcept override {
    return tcs;
  }
};

// The same signature, emitted twice in two languages. What a cache must never
// do is answer one of these with the other's object.
struct DialectPair {
  EmittedKernel hip;
  EmittedKernel loom;
  std::uint64_t signature = 0x5eed;

  DialectPair() {
    hip.dialect = Dialect::kHip;
    hip.source = "Hhip source";
    hip.entry_name = "k";
    loom.dialect = Dialect::kLoom;
    loom.source = "Lloom source";
    loom.entry_name = "k";
  }
};

std::size_t count_suffix(const std::filesystem::path& dir,
                         std::string_view suffix) {
  std::size_t n = 0;
  std::error_code ec;
  for (std::filesystem::directory_iterator it(dir, ec), end; !ec && it != end;
       ++it) {
    const std::string name = it->path().filename().string();
    if (name.size() >= suffix.size() &&
        name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0) {
      ++n;
    }
  }
  return n;
}

}  // namespace

LSE_TEST(a_device_declares_its_dialects_and_the_default_is_the_first) {
  TwoDialectBackend be;
  LSE_EXPECT_EQ(be.toolchains().size(), 2u);
  // The dialect-blind accessors every existing caller uses answer with the
  // first declared entry, unchanged by the second being there.
  LSE_EXPECT(be.emitter() == &be.hip_emitter);
  LSE_EXPECT(be.compiler() == &be.hip_compiler);
  LSE_EXPECT(be.emitter()->dialect() == Dialect::kHip);

  const KernelToolchain* loom = be.toolchain_for(Dialect::kLoom);
  LSE_EXPECT(loom != nullptr);
  if (loom == nullptr) return;
  LSE_EXPECT(loom->emitter == &be.loom_emitter);
  LSE_EXPECT(loom->compiler == &be.loom_compiler);
  // Two dialects are two JIT identities, which is what keeps their objects out
  // of each other's cache slots.
  LSE_EXPECT(loom->compiler->identity() != be.compiler()->identity());
}

LSE_TEST(an_undeclared_dialect_is_absent_not_substituted) {
  TwoDialectBackend be;
  LSE_EXPECT(be.toolchain_for(Dialect::kSpirv) == nullptr);
  LSE_EXPECT(be.toolchain_for(Dialect::kCuda) == nullptr);
}

LSE_TEST(a_device_with_no_codegen_declares_no_dialect) {
  backend::BackendAdapter<backend::CpuBackend> cpu;
  LSE_EXPECT(cpu.toolchains().empty());
  LSE_EXPECT(cpu.emitter() == nullptr);
  LSE_EXPECT(cpu.compiler() == nullptr);
  LSE_EXPECT(cpu.toolchain_for(Dialect::kHip) == nullptr);
}

LSE_TEST(emitted_source_carries_the_dialect_of_its_emitter) {
  Array x = Array::full(Shape{64}, DType::kF32, 1.0f);
  Array y = x * x + x;
  auto e = emit_for(y);
  LSE_EXPECT(e.ok());
  if (!e.ok()) return;
  LSE_EXPECT(e->dialect == kEmitter.dialect());
  LSE_EXPECT(e->dialect == Dialect::kHip);
}

LSE_TEST(the_hrx_device_declares_two_dialects_with_hip_in_front) {
  backend::BackendAdapter<backend::HrxBackend> hrx;
  LSE_EXPECT_EQ(hrx.toolchains().size(), 2u);
  // kHip stays the front entry, which is the whole of what keeps every caller
  // that asks the device for "its" emitter unchanged.
  LSE_EXPECT(hrx.toolchains().front().dialect == Dialect::kHip);
  LSE_EXPECT(hrx.emitter() != nullptr);
  LSE_EXPECT(hrx.compiler() != nullptr);
  LSE_EXPECT(hrx.emitter()->dialect() == Dialect::kHip);
  LSE_EXPECT(hrx.toolchain_for(Dialect::kHip) == &hrx.toolchains().front());

  const KernelToolchain* loom = hrx.toolchain_for(Dialect::kLoom);
  LSE_EXPECT(loom != nullptr);
  if (loom == nullptr) return;
  LSE_EXPECT(loom->emitter != nullptr && loom->compiler != nullptr);
  LSE_EXPECT(loom->emitter->dialect() == Dialect::kLoom);
  // Both halves of one dialect, and neither is the other dialect's.
  LSE_EXPECT(loom->emitter != hrx.emitter());
  LSE_EXPECT(loom->compiler != hrx.compiler());
  LSE_EXPECT(hrx.toolchain_for(Dialect::kCuda) == nullptr);
}

// The whole of the negotiation, exercised: no opinion takes the front entry, a
// named dialect that is declared is handed over, and one that is not declared
// degrades to the front entry rather than failing the run.
LSE_TEST(a_dialect_preference_resolves_or_degrades_to_the_front_entry) {
  TwoDialectBackend be;
  LSE_EXPECT(be.toolchain(std::nullopt) == &be.toolchains().front());
  LSE_EXPECT(be.toolchain(Dialect::kHip) == &be.toolchains().front());

  const KernelToolchain* loom = be.toolchain(Dialect::kLoom);
  LSE_EXPECT(loom != nullptr && loom->dialect == Dialect::kLoom);
  LSE_EXPECT(loom == &be.toolchains()[1]);

  // Undeclared. A preference names a resource; a member that lacks it still
  // has to run, so this is the front entry and not an error.
  LSE_EXPECT(be.toolchain(Dialect::kSpirv) == &be.toolchains().front());
  LSE_EXPECT(be.toolchain(Dialect::kMetal) == &be.toolchains().front());

  // A device with no codegen has no front entry to degrade to either.
  backend::BackendAdapter<backend::CpuBackend> cpu;
  LSE_EXPECT(cpu.toolchain(std::nullopt) == nullptr);
  LSE_EXPECT(cpu.toolchain(Dialect::kHip) == nullptr);
}

// A run names a dialect on the scheduler, and what the scheduler hands the
// member is that dialect's BOTH halves.
LSE_TEST(a_run_that_names_a_dialect_is_given_that_dialects_emitter) {
  backend::BackendAdapter<backend::HrxBackend> hrx;
  Scheduler sched(hrx);
  LSE_EXPECT(!sched.dialect().has_value());
  const KernelToolchain* front = sched.toolchain(0);
  LSE_EXPECT(front != nullptr);
  if (front == nullptr) return;
  LSE_EXPECT(front->dialect == Dialect::kHip);

  sched.set_dialect(Dialect::kLoom);
  const KernelToolchain* chosen = sched.toolchain(0);
  LSE_EXPECT(chosen != nullptr);
  if (chosen == nullptr) return;
  LSE_EXPECT(chosen->dialect == Dialect::kLoom);
  LSE_EXPECT(chosen->emitter != nullptr &&
             chosen->emitter->dialect() == Dialect::kLoom);
  LSE_EXPECT(chosen->compiler != nullptr);
  LSE_EXPECT(chosen->emitter != hrx.emitter());
  LSE_EXPECT(chosen->compiler != hrx.compiler());

  sched.set_dialect(Dialect::kSpirv);
  const KernelToolchain* degraded = sched.toolchain(0);
  LSE_EXPECT(degraded != nullptr && degraded->dialect == Dialect::kHip);
  sched.clear_dialect();
  const KernelToolchain* none = sched.toolchain(0);
  LSE_EXPECT(none != nullptr && none->dialect == Dialect::kHip);
}

// Two dialects, one member, one signature. The cache must build both and hand
// each caller back the object built from ITS OWN text.
LSE_TEST(the_cache_never_serves_one_dialects_object_for_the_other) {
  namespace fs = std::filesystem;
  const fs::path dir = fs::temp_directory_path() /
                       ("lse-jit-dialect-" + std::to_string(::getpid()));
  fs::remove_all(dir);
  fs::create_directories(dir);

  EchoCompiler hip_cc("comgr-like");
  EchoCompiler loom_cc("loomc-like");
  EchoBackend be(hip_cc, loom_cc);
  StubSet set;
  set.members.push_back(&be);
  DialectPair k;

  JitCache cache(set, dir.string());
  auto a = cache.get_or_compile(0, k.signature, k.hip);
  auto b = cache.get_or_compile(0, k.signature, k.loom);
  LSE_EXPECT_OK(a.status());
  LSE_EXPECT_OK(b.status());
  // Two compiles, not one and a hit: the second ask was for a different
  // language and there was nothing to serve it.
  LSE_EXPECT_EQ(cache.stats().compiles, 2u);
  LSE_EXPECT_EQ(cache.stats().memory_hits, 0u);

  const backend::KernelHandle* hip = cache.try_get(0, k.signature, Dialect::kHip);
  const backend::KernelHandle* loom =
      cache.try_get(0, k.signature, Dialect::kLoom);
  LSE_EXPECT(hip != nullptr && loom != nullptr);
  if (hip == nullptr || loom == nullptr) return;
  // The echo compiler puts the source's first byte in the object and the echo
  // device loads that byte as the executable id, so this says which text each
  // handle was actually built from.
  LSE_EXPECT_EQ(hip->executable, static_cast<std::uint64_t>('H'));
  LSE_EXPECT_EQ(loom->executable, static_cast<std::uint64_t>('L'));

  // Two toolchain identities are two disk slots, so neither overwrites the
  // other's metadata and a later process finds both.
  LSE_EXPECT_EQ(count_suffix(dir, ".meta"), 2u);
  LSE_EXPECT_EQ(count_suffix(dir, ".co"), 2u);

  // A second process finds both on disk, still not mixed up.
  {
    JitCache warm(set, dir.string());
    auto wa = warm.get_or_compile(0, k.signature, k.hip);
    auto wb = warm.get_or_compile(0, k.signature, k.loom);
    LSE_EXPECT_OK(wa.status());
    LSE_EXPECT_OK(wb.status());
    LSE_EXPECT_EQ(warm.stats().compiles, 0u);
    LSE_EXPECT_EQ(warm.stats().disk_hits, 2u);
    LSE_EXPECT_EQ(wa->executable, static_cast<std::uint64_t>('H'));
    LSE_EXPECT_EQ(wb->executable, static_cast<std::uint64_t>('L'));
  }
  fs::remove_all(dir);
}

LSE_TEST(measured_resources_survive_a_warm_start_holes_included) {
  namespace fs = std::filesystem;
  const fs::path dir = fs::temp_directory_path() /
                       ("lse-jit-measured-" + std::to_string(::getpid()));
  fs::remove_all(dir);
  fs::create_directories(dir);

  MeasuringCompiler cc;
  EchoBackend be(cc, cc);
  StubSet set;
  set.members.push_back(&be);
  DialectPair k;

  const auto check = [&](const JitCache& cache, const char* when) {
    const backend::KernelResources* r =
        cache.resources(0, k.signature, Dialect::kHip);
    LSE_EXPECT(r != nullptr);
    if (r == nullptr) {
      std::printf("       no resources %s\n", when);
      return;
    }
    LSE_EXPECT(r->entry == "lse_measured");
    LSE_EXPECT_EQ(r->vector_registers.value, 102u);
    LSE_EXPECT_EQ(r->scalar_registers.value, 28u);
    LSE_EXPECT_EQ(r->workgroup_segment_bytes.value, 25600u);
    LSE_EXPECT(r->private_segment_bytes.known());
    LSE_EXPECT_EQ(r->private_segment_bytes.value, 0u);
    // The hole is the point: a compiler that never answered must not come back
    // from disk having answered zero.
    LSE_EXPECT(!r->vector_spills.known());
    LSE_EXPECT(r->spilled() == backend::SpillState::kNone);
    LSE_EXPECT(r->required_workgroup_size.known());
    LSE_EXPECT_EQ(r->required_workgroup_size.value[0], 256u);
    // Facts nobody ever reported stay unreported through the round trip.
    LSE_EXPECT(!r->accum_registers.known());
    LSE_EXPECT(!r->wavefront_size.known());
  };

  {
    JitCache cache(set, dir.string());
    LSE_EXPECT_OK(cache.get_or_compile(0, k.signature, k.hip).status());
    LSE_EXPECT_EQ(cache.stats().compiles, 1u);
    check(cache, "after compiling");
  }
  {
    // A second process: nothing compiles, and the numbers still arrive.
    JitCache warm(set, dir.string());
    LSE_EXPECT_OK(warm.get_or_compile(0, k.signature, k.hip).status());
    LSE_EXPECT_EQ(warm.stats().compiles, 0u);
    LSE_EXPECT_EQ(warm.stats().disk_hits, 1u);
    check(warm, "after a warm start");
  }
  fs::remove_all(dir);
}

// The census travels with the object the same way the register counts do, and
// an object cached before anything counted instructions gets counted from its
// own bytes rather than staying undescribed forever.
LSE_TEST(an_emitted_census_survives_a_warm_start_and_an_older_note) {
  namespace fs = std::filesystem;
  const fs::path dir = fs::temp_directory_path() /
                       ("lse-jit-census-" + std::to_string(::getpid()));
  fs::remove_all(dir);
  fs::create_directories(dir);

  MeasuringCompiler cc;
  EchoBackend be(cc, cc);
  StubSet set;
  set.members.push_back(&be);
  DialectPair k;

  const auto check = [&](const JitCache& cache, const char* when) {
    const backend::KernelCensus* c =
        cache.census(0, k.signature, Dialect::kHip);
    LSE_EXPECT(c != nullptr);
    if (c == nullptr) {
      std::printf("       no census %s\n", when);
      return;
    }
    LSE_EXPECT(c->entry == "lse_measured");
    LSE_EXPECT_EQ(c->instructions.value, 1453u);
    LSE_EXPECT_EQ(c->dot_products.value, 160u);
    LSE_EXPECT_EQ(c->multiply_accumulates.value, 640u);
    // Widths and their loop split are the part a scalar line cannot carry.
    LSE_EXPECT_EQ(c->global_loads.count(), 3u);
    LSE_EXPECT_EQ(c->global_loads.bytes(), 34u);
    LSE_EXPECT_EQ(c->global_loads.straight_line_bytes(), 18u);
    LSE_EXPECT_EQ(c->shared_stores.bytes(), 4u);
    LSE_EXPECT(!c->straight_line());
    // A hole stays a hole: nothing counted the scalar arithmetic, and a round
    // trip must not answer zero for it.
    LSE_EXPECT(!c->scalar_alu.known());
  };

  {
    JitCache cache(set, dir.string());
    LSE_EXPECT_OK(cache.get_or_compile(0, k.signature, k.hip).status());
    LSE_EXPECT_EQ(cache.stats().compiles, 1u);
    check(cache, "after compiling");
  }
  {
    JitCache warm(set, dir.string());
    LSE_EXPECT_OK(warm.get_or_compile(0, k.signature, k.hip).status());
    LSE_EXPECT_EQ(warm.stats().compiles, 0u);
    LSE_EXPECT_EQ(warm.stats().disk_hits, 1u);
    check(warm, "after a warm start");
  }

  // Now the note as an older build wrote it: resources, no counts. The bytes
  // are still there, so the counts have to come back without a compile.
  for (const fs::directory_entry& e : fs::directory_iterator(dir)) {
    if (e.path().extension() != ".meta") continue;
    std::ifstream in(e.path());
    std::string kept;
    for (std::string line; std::getline(in, line);) {
      if (line.rfind("cen ", 0) == 0 || line.rfind("acc ", 0) == 0) continue;
      kept += line;
      kept += '\n';
    }
    in.close();
    std::ofstream out(e.path(), std::ios::trunc);
    out << kept;
  }
  {
    JitCache older(set, dir.string());
    LSE_EXPECT_OK(older.get_or_compile(0, k.signature, k.hip).status());
    LSE_EXPECT_EQ(older.stats().compiles, 0u);
    check(older, "from an older note");
  }

  // And the counts reach the optimizer's own store under the entry name, which
  // is the identity a decision site can spell before the next kernel exists.
  const backend::KernelCensus stored =
      opt::KernelMeasurements::instance().census("lse_measured");
  LSE_EXPECT(stored.any());
  LSE_EXPECT_EQ(stored.multiply_accumulates.value, 640u);
  fs::remove_all(dir);
}

// The store is what carries a fact from one compile to the next decision, so
// both halves of the comparison have to survive it: the counts the object
// yielded, and the traffic the emitter meant. Nothing here touches a device —
// this is the seam, not the measurement.
LSE_TEST(the_measurement_store_carries_both_halves_of_the_comparison) {
  opt::KernelMeasurements& store = opt::KernelMeasurements::instance();
  const std::string entry = "lse_traffic_store_probe";
  LSE_EXPECT(!store.census_known(entry));
  LSE_EXPECT(!store.traffic(entry).stated);

  backend::KernelCensus c;
  c.entry = entry;
  c.instructions = backend::DeviceFact<std::uint32_t>::queried(64);
  c.multiply_accumulates = backend::DeviceFact<std::uint32_t>::queried(256);
  c.backward_branches = backend::DeviceFact<std::uint32_t>::queried(0);
  c.global_loads.add(16, false);
  store.record(entry, c);

  opt::TrafficModel m = opt::contraction_traffic(1, 8, 64, 4, 4, 0, 0);
  m.workgroup_threads = 64;
  store.record(entry, m);

  LSE_EXPECT(store.census_known(entry));
  const opt::TrafficModel back = store.traffic(entry);
  LSE_EXPECT(back.stated);
  LSE_EXPECT_EQ(back.bytes_read(opt::OperandClass::kWeight), 8u * 64u / 2u);
  LSE_EXPECT_EQ(back.bytes_read(opt::OperandClass::kActivation), 64u * 4u);

  // The whole point of keeping them together: the comparison composes out of
  // the store, with nothing held over from the compile that produced them.
  const opt::TrafficCheck check = opt::check_traffic(
      back, store.census(entry), back.workgroup_threads);
  LSE_EXPECT(check.verdict != opt::TrafficVerdict::kUnknown);
  LSE_EXPECT(check.emitted_bytes == 16u * 64u);
  LSE_EXPECT(check.emitted_exact);
}

// The case the compiler identity cannot separate: one compiler declared for
// both dialects, so the two share a cache key. They must still not share an
// entry — the memory table is one map per dialect, which is what stops try_get
// from answering a Loom ask with the HIP object it happens to have.
LSE_TEST(two_dialects_sharing_a_compiler_still_do_not_share_a_slot) {
  namespace fs = std::filesystem;
  const fs::path dir = fs::temp_directory_path() /
                       ("lse-jit-shared-" + std::to_string(::getpid()));
  fs::remove_all(dir);
  fs::create_directories(dir);

  EchoCompiler shared("one-compiler-for-both");
  EchoBackend be(shared, shared);
  StubSet set;
  set.members.push_back(&be);
  DialectPair k;

  JitCache cache(set, dir.string());
  LSE_EXPECT_OK(cache.get_or_compile(0, k.signature, k.hip).status());
  LSE_EXPECT_OK(cache.get_or_compile(0, k.signature, k.loom).status());
  LSE_EXPECT_EQ(cache.stats().compiles, 2u);

  const backend::KernelHandle* hip = cache.try_get(0, k.signature, Dialect::kHip);
  const backend::KernelHandle* loom =
      cache.try_get(0, k.signature, Dialect::kLoom);
  LSE_EXPECT(hip != nullptr && loom != nullptr);
  if (hip == nullptr || loom == nullptr) return;
  LSE_EXPECT_EQ(hip->executable, static_cast<std::uint64_t>('H'));
  LSE_EXPECT_EQ(loom->executable, static_cast<std::uint64_t>('L'));
  fs::remove_all(dir);
}

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
    LSE_EXPECT(cache.try_get(0, sig, Dialect::kHip) != nullptr);
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
  const backend::KernelHandle* on_a = cache.try_get(0, sig, Dialect::kHip);
  const backend::KernelHandle* on_b = cache.try_get(1, sig, Dialect::kHip);
  LSE_EXPECT(on_a != nullptr && on_b != nullptr);
  if (on_a != nullptr && on_b != nullptr) {
    LSE_EXPECT_EQ(on_a->executable, std::uint64_t{0xAA});
    LSE_EXPECT_EQ(on_b->executable, std::uint64_t{0xBB});
  }
  LSE_EXPECT(cache.try_get(2, sig, Dialect::kHip) == nullptr);
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

// ---------------------------------------------------------------------------
// HIP against Loom, on the same graph and the same bytes.
//
// The dialect pair is the only variable: identical nodes, identical inputs,
// identical device. A group Loom declines falls to the host, so a case reports
// how much of it reached the device in each dialect; a case that reached the
// device in both and disagrees is a Loom codegen bug, and there is no other
// explanation available to it.
namespace dialect_diff {

// The shape decides how much is written: a case that hands a longer pool of
// noise than its shape holds must fill the array, not walk off it.
Array host_array(const std::vector<float>& values, Shape shape) {
  Array a = Array::zeros(shape, DType::kF32);
  (void)a.eval();
  const std::size_t n = std::min(values.size(), a.shape().elem_count());
  for (std::size_t i = 0; i < n; ++i) {
    interpreter::store_element(*a.node(), i, values[i]);
  }
  return a;
}

std::vector<float> drain(Array& a) {
  (void)a.eval();
  std::vector<float> out(a.shape().elem_count());
  for (std::size_t i = 0; i < out.size(); ++i) {
    out[i] = interpreter::load_element(*a.node(), i);
  }
  return out;
}

// A deterministic spread in [-1, 1); the same generator test_graph uses, so a
// failure here and a failure there are on the same numbers.
float spread(std::uint64_t i) {
  std::uint64_t x = i * 0x9E3779B97F4A7C15ull + 0xBF58476D1CE4E5B9ull;
  x ^= x >> 30;
  x *= 0xBF58476D1CE4E5B9ull;
  x ^= x >> 27;
  return static_cast<float>(static_cast<double>(x >> 40) / 8388608.0) - 1.0f;
}

std::vector<float> noise(std::size_t n, std::uint64_t seed) {
  std::vector<float> v(n);
  for (std::size_t i = 0; i < n; ++i) v[i] = spread(i + seed);
  return v;
}

struct Side {
  std::vector<float> out;
  std::uint32_t device_groups = 0;
  std::uint32_t host_groups = 0;
};

// One evaluation under one dialect, with the trace of what it did.
template <class Build>
Side run_under(Scheduler* sc, std::optional<Dialect> d, Build&& build) {
  const auto prev_mode = sc->mode();
  const DialectPreference prev = sc->dialect();
  sc->set_mode(Scheduler::Mode::kDeviceFirst);
  if (d.has_value()) {
    sc->set_dialect(*d);
  } else {
    sc->clear_dialect();
  }
  Array y = build();
  Side s;
  s.out = drain(y);
  s.device_groups = sc->last_trace().device_groups;
  s.host_groups = sc->last_trace().host_groups;
  sc->set_mode(prev_mode);
  if (prev.has_value()) {
    sc->set_dialect(*prev);
  } else {
    sc->clear_dialect();
  }
  return s;
}

// The worst disagreement, relative to the magnitude of the HIP answer. An
// absolute difference on a K=1024 dot product says more about f32 than about
// the kernel.
double worst_rel(const std::vector<float>& a, const std::vector<float>& b) {
  double worst = 0.0;
  for (std::size_t i = 0; i < a.size() && i < b.size(); ++i) {
    const double scale = std::max(1e-3, std::fabs(static_cast<double>(a[i])));
    worst = std::max(worst, std::fabs(static_cast<double>(a[i] - b[i])) / scale);
  }
  return worst;
}

// Every case is (name, builder). The builder is called once per dialect, and
// the report says how much of it each one put on the device: a case Loom
// declined runs its group on the host interpreter, so it is still compared —
// against a looser bound, because the host arm is a different summation order
// and not the same instruction stream.
struct Tally {
  int emitted = 0;
  int declined = 0;
};
Tally& tally() {
  static Tally t;
  return t;
}

template <class Build>
void compare(const char* name, Build&& build, double tol = 1e-5) {
  Scheduler* sc = default_scheduler();
  if (sc == nullptr || sc->backend().emitter() == nullptr) return;
  const Side hip = run_under(sc, Dialect::kHip, build);
  const Side loom = run_under(sc, Dialect::kLoom, build);
  LSE_EXPECT_EQ(hip.out.size(), loom.out.size());
  if (hip.out.size() != loom.out.size()) return;

  const bool declined = loom.host_groups != 0;
  if (declined) {
    ++tally().declined;
    tol = std::max(tol, 1e-4);
  } else {
    ++tally().emitted;
  }
  const double rel = worst_rel(hip.out, loom.out);
  std::printf("       %-22s hip dev=%u | loom dev=%u host=%u %-8s worst rel "
              "%.3e\n",
              name, hip.device_groups, loom.device_groups, loom.host_groups,
              declined ? "DECLINE" : "emit", rel);
  LSE_EXPECT(rel <= tol);
  if (rel > tol) {
    std::size_t at = 0;
    double worst = 0.0;
    for (std::size_t i = 0; i < hip.out.size(); ++i) {
      const double d = std::fabs(static_cast<double>(hip.out[i] - loom.out[i]));
      if (d > worst) { worst = d; at = i; }
    }
    std::printf("         !! %s disagrees worst at [%zu]: hip=%.9g loom=%.9g\n",
                name, at, static_cast<double>(hip.out[at]),
                static_cast<double>(loom.out[at]));
    for (std::size_t i = 0; i < std::min<std::size_t>(8, hip.out.size()); ++i) {
      std::printf("            [%zu] hip=%.9g loom=%.9g\n", i,
                  static_cast<double>(hip.out[i]),
                  static_cast<double>(loom.out[i]));
    }
  }
}

}  // namespace dialect_diff

LSE_TEST(loom_matches_hip_on_the_elementwise_vocabulary) {
  using namespace dialect_diff;
  const std::vector<float> a = noise(256, 11);
  const std::vector<float> b = noise(256, 977);
  compare("mul+add+silu", [&] {
    Array x = host_array(a, Shape{256});
    Array y = host_array(b, Shape{256});
    return silu(x * y + x);
  });
  compare("sigmoid+exp+neg", [&] {
    Array x = host_array(a, Shape{256});
    return sigmoid(exp(neg(x)));
  });
  compare("softplus+clamp", [&] {
    Array x = host_array(a, Shape{256});
    return clamp(softplus(x), -2.0f, 2.0f);
  });
  compare("softmax", [&] {
    Array x = host_array(a, Shape{8, 32});
    return softmax(x, -1);
  });
  compare("l2_normalize", [&] {
    Array x = host_array(a, Shape{8, 32});
    return l2_normalize(x);
  });
}

LSE_TEST(loom_matches_hip_on_the_shape_kernels) {
  using namespace dialect_diff;
  const std::vector<float> a = noise(512, 4241);
  compare("transpose", [&] {
    Array x = host_array(a, Shape{16, 32});
    return transpose(x, {1, 0});
  });
  compare("slice", [&] {
    Array x = host_array(a, Shape{16, 32});
    return slice(x, 1, 4, 20);
  });
  compare("embedding", [&] {
    Array t = host_array(a, Shape{16, 32});
    Array ids = host_array({3, 0, 9, 15}, Shape{4});
    return embedding(t, ids);
  });
  // The router's own shape: 8 experts, 2 active, which is what lemonseed and
  // the Qwen MoE checkpoints both route with.
  compare("topk 8->2", [&] {
    Array x = host_array(a, Shape{4, 8});
    Array idx;
    return topk(x, 2, -1, &idx);
  });
}

LSE_TEST(loom_matches_hip_on_the_linear_kernels) {
  using namespace dialect_diff;
  // The three shapes that pick the three gemv bodies a decode step uses: a
  // narrow K that stays in one wave, the LDS-panel tile, and a K wide enough
  // to run the packed loop.
  for (const auto& [n, k] : std::initializer_list<std::pair<int, int>>{
           {8, 64}, {64, 512}, {32, 1024}}) {
    const std::vector<float> xs = noise(static_cast<std::size_t>(k), 7);
    const std::vector<float> ws =
        noise(static_cast<std::size_t>(n) * static_cast<std::size_t>(k), 33);
    char label[64];
    std::snprintf(label, sizeof(label), "linear %dx%d", n, k);
    compare(label, [&] {
      Array x = host_array(xs, Shape{1, k});
      Array w = host_array(ws, Shape{n, k});
      return linear(x, w);
    }, 2e-4);
  }
  // The checkpoint's own storage. lemonseed's weights are bf16 and the gemv
  // reads them in that format, so an f32-only differential would never touch
  // the widening path the real model runs.
  for (const DType dt : {DType::kBF16, DType::kF16}) {
    const std::vector<float> xw = noise(512, 7);
    const std::vector<float> ww = noise(64 * 512, 33);
    char label[64];
    std::snprintf(label, sizeof(label), "linear 64x512 %s",
                  dt == DType::kBF16 ? "bf16" : "f16");
    compare(label, [&] {
      Array x = host_array(xw, Shape{1, 512});
      Array w = cast(host_array(ww, Shape{64, 512}), dt);
      return linear(x, w);
    }, 2e-2);
  }
  const std::vector<float> xs = noise(512, 91);
  const std::vector<float> ws = noise(4 * 32 * 512, 137);
  compare("linear_indexed", [&] {
    Array x = host_array(xs, Shape{1, 512});
    Array w = host_array(ws, Shape{4, 32, 512});
    Array idx = host_array({2, 0}, Shape{1, 2});
    return linear_indexed(x, w, idx, 0);
  }, 2e-4);
}

LSE_TEST(loom_matches_hip_on_a_gemv_with_an_epilogue) {
  using namespace dialect_diff;
  const std::vector<float> xs = noise(512, 7);
  const std::vector<float> ws = noise(64 * 512, 33);
  const std::vector<float> gs = noise(64, 51);
  compare("linear*vec", [&] {
    Array x = host_array(xs, Shape{1, 512});
    Array w = host_array(ws, Shape{64, 512});
    Array g = host_array(gs, Shape{64});
    return linear(x, w) * g;
  }, 2e-4);
  // The shape the gated paths actually multiply by: a trailing axis of 1
  // stretched over the row, which is a strided reindex of the operand rather
  // than a whole-array or a same-shape read.
  compare("mul by [1,4,1]", [&] {
    Array x = host_array(noise(4096, 13), Shape{1, 4, 1024});
    Array g = host_array(noise(4, 991), Shape{1, 4, 1});
    return x * g;
  });
  compare("mul by [1,1,1]", [&] {
    Array x = host_array(noise(1024, 13), Shape{1, 1, 1024});
    Array g = host_array(noise(1, 991), Shape{1, 1, 1});
    return x * g;
  });
  compare("linear*scalar", [&] {
    Array x = host_array(xs, Shape{1, 512});
    Array w = host_array(ws, Shape{64, 512});
    Array g = host_array({0.37f}, Shape{1});
    return linear(x, w) * g;
  }, 2e-4);
  // The six fused shapes lemonseed's decode step actually builds, weights in
  // the checkpoint's own bf16. These are the groups the model runs; the
  // isolated ones above are not the same code.
  {
    const std::vector<float> hid = noise(1024, 401);
    const std::vector<float> big = noise(2176 * 1024, 403);
    const std::vector<float> gate = noise(2176, 407);
    compare("gemv2176*vec bf16", [&] {
      Array x = host_array(hid, Shape{1, 1, 1024});
      Array w = cast(host_array(big, Shape{2176, 1024}), DType::kBF16);
      Array g = host_array(gate, Shape{1, 1, 2176});
      return linear(x, w) * g;
    }, 2e-2);
    compare("sigmoid*x", [&] {
      Array x = host_array(hid, Shape{1, 1, 1024});
      Array y = host_array(noise(1024, 409), Shape{1, 1, 1024});
      return sigmoid(x) * y;
    });
    compare("mul[1,1,1]+add", [&] {
      Array x = host_array(hid, Shape{1, 1, 1024});
      Array g = host_array(noise(1, 411), Shape{1, 1, 1});
      Array y = host_array(noise(1024, 413), Shape{1, 1, 1024});
      return x * g + y;
    });
    // Every operand is materialized first, so the group the scheduler sees is
    // the one the model builds: a kernel primitive, a reshape and an
    // elementwise consumer over FOUR bindings. With the operands unevaluated
    // the same expression partitions into three groups and never exercises it.
    compare("rms+reshape*x", [&] {
      Array x = host_array(noise(512, 415), Shape{1, 1, 8, 64});
      Array nw = cast(host_array(noise(64, 417), Shape{64}), DType::kBF16);
      (void)nw.eval();
      Array y = host_array(noise(512, 419), Shape{1, 1, 512});
      return reshape(rms_norm(x, nw, 1e-6f, true), Shape{1, 1, 512}) * y;
    }, 2e-2);
    compare("router chain", [&] {
      Array x = host_array(hid, Shape{1, 1, 1024});
      Array w = cast(host_array(noise(8 * 1024, 421), Shape{8, 1024}),
                     DType::kBF16);
      Array b = cast(host_array(noise(8, 423), Shape{8}), DType::kBF16);
      Array s = host_array(noise(8, 425), Shape{8});
      return exp(neg(softplus(linear(x, w) + b) * s));
    }, 2e-2);
  }
  // A one-column gemv: the wave reduction still runs, but every lane after the
  // first has no column of its own. lemonseed's per-layer gate is this shape.
  compare("gemv N=1 bf16", [&] {
    Array x = host_array(noise(1024, 601), Shape{1, 1, 1024});
    Array w = cast(host_array(noise(1024, 603), Shape{1, 1024}), DType::kBF16);
    Array b = cast(host_array(noise(1, 605), Shape{1}), DType::kBF16);
    return sigmoid(linear(x, w) + b);
  }, 2e-2);
  // Four gemvs over one activation row: the shared staged panel, which is the
  // only path where one kernel's LDS is filled for siblings it does not own.
  compare("four siblings 512", [&] {
    Array x = host_array(noise(1024, 607), Shape{1, 1, 1024});
    Array a1 = cast(host_array(noise(512 * 1024, 609), Shape{512, 1024}),
                    DType::kBF16);
    Array a2 = cast(host_array(noise(512 * 1024, 611), Shape{512, 1024}),
                    DType::kBF16);
    return linear(x, a1) + linear(x, a2);
  }, 2e-2);
  compare("swiglu+linear", [&] {
    Array x = host_array(xs, Shape{1, 512});
    Array w1 = host_array(ws, Shape{64, 512});
    Array w2 = host_array(noise(64 * 512, 909), Shape{64, 512});
    return silu(linear(x, w1)) * linear(x, w2);
  }, 2e-4);
  compare("rms_norm+linear", [&] {
    Array x = host_array(xs, Shape{1, 512});
    Array nw = host_array(noise(512, 71), Shape{512});
    Array w = host_array(ws, Shape{64, 512});
    return linear(rms_norm(x, nw, 1e-6f, true), w);
  }, 2e-4);
}

LSE_TEST(loom_matches_hip_on_the_attention_and_recurrent_kernels) {
  using namespace dialect_diff;
  compare("rms_norm", [&] {
    Array x = host_array(noise(4 * 128, 5), Shape{4, 128});
    Array w = host_array(noise(128, 61), Shape{128});
    return rms_norm(x, w, 1e-6f, true);
  });
  // [B, H, T, Dh] against a [T, Dh] angle table. The primitive takes T from
  // x.dim(rank-2), so a table with fewer rows than that is read past its end —
  // in C that is a silent out-of-bounds load, and in Loom the printer's
  // `index.assume` on the same subscript becomes a false promise. The shapes
  // here are the ones the kernel's own contract admits.
  compare("rope", [&] {
    Array x = host_array(noise(1 * 4 * 2 * 64, 17), Shape{1, 4, 2, 64});
    Array cs = host_array(noise(2 * 64, 23), Shape{2, 64});
    Array sn = host_array(noise(2 * 64, 29), Shape{2, 64});
    return rope(x, cs, sn, 0);
  });
  compare("causal_conv1d", [&] {
    Array x = host_array(noise(2 * 8 * 16, 3), Shape{2, 8, 16});
    Array w = host_array(noise(16 * 4, 71), Shape{16, 4});
    Array b = host_array(noise(16, 83), Shape{16});
    return causal_conv1d(x, w, b);
  });
  // The three kernels lemonseed runs that Loom declines. They are in the
  // differential so the decline is measured rather than assumed, and so a
  // later change that makes one of them emit is compared the same way.
  compare("conv_tail", [&] {
    Array tail = host_array(noise(2 * 3 * 16, 201), Shape{2, 3, 16});
    Array x = host_array(noise(2 * 8 * 16, 203), Shape{2, 8, 16});
    return conv_tail(tail, x);
  });
  compare("gated_delta_step", [&] {
    Array q = host_array(noise(1 * 1 * 2 * 16, 211), Shape{1, 1, 2, 16});
    Array k = host_array(noise(1 * 1 * 2 * 16, 213), Shape{1, 1, 2, 16});
    Array v = host_array(noise(1 * 1 * 2 * 16, 215), Shape{1, 1, 2, 16});
    Array al = host_array(noise(1 * 1 * 2, 217), Shape{1, 1, 2});
    Array be = host_array(noise(1 * 1 * 2, 219), Shape{1, 1, 2});
    Array st = host_array(noise(1 * 2 * 16 * 16, 221), Shape{1, 2, 16, 16});
    return gated_delta_step(q, k, v, al, be, st, nullptr);
  }, 1e-3);
  compare("kv_page_write", [&] {
    Array dst = host_array(noise(4 * 2 * 8 * 16, 231), Shape{4, 2, 8, 16});
    Array src = host_array(noise(1 * 2 * 1 * 16, 233), Shape{1, 2, 1, 16});
    Array meta = host_array({0, 1, 1}, Shape{3});
    Array table = host_array({0, 1, 2, 3}, Shape{1, 4});
    return kv_page_write(dst, src, meta, table, 8);
  });
  compare("sdpa", [&] {
    Array q = host_array(noise(1 * 4 * 1 * 32, 101), Shape{1, 4, 1, 32});
    Array k = host_array(noise(1 * 4 * 8 * 32, 103), Shape{1, 4, 8, 32});
    Array v = host_array(noise(1 * 4 * 8 * 32, 107), Shape{1, 4, 8, 32});
    return sdpa(q, k, v, 0.176776695f, MaskKind::kCausal);
  }, 1e-4);
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
LSE_TEST(loom_reports_what_fraction_of_the_cases_it_emitted) {
  using namespace dialect_diff;
  const Tally& t = tally();
  if (t.emitted + t.declined == 0) return;
  std::printf("       %d of %d differential cases emit through Loom; %d "
              "decline and fall to the host\n",
              t.emitted, t.emitted + t.declined, t.declined);
}

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
    : kernels::MatrixTile<Int4Tile, lmath::MatrixTarget::kRdna3,
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
  auto handle = be.load_executable(entry, code->code);
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
std::vector<std::vector<float>> run_emitted_with(
    const graph::IKernelCompiler& cc, const EmittedKernel& e,
    const backend::DeviceInfo& dev, backend::IBackend& be,
    const std::vector<std::pair<const Node*, std::vector<float>>>& host) {
  auto code = cc.compile(e.source, std::string(dev.arch));
  if (!code.ok()) {
    std::printf("       compile failed:\n%s\n", code.status().message().c_str());
    return {};
  }
  auto handle = be.load_executable(e.entry_name, code->code);
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

std::vector<std::vector<float>> run_emitted(
    const EmittedKernel& e, const backend::DeviceInfo& dev,
    backend::IBackend& be,
    const std::vector<std::pair<const Node*, std::vector<float>>>& host) {
  return run_emitted_with(kCompiler, e, dev, be, host);
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

// The occupancy model must be the hardware's, in the hardware's unit, and it
// must be a MINIMUM over every limit that applies.
//
// The table this replaced was hipOccupancyMaxActiveBlocksPerMultiprocessor's,
// which returns floor(65536 / lds) and is 2x low on this part: it says 1
// workgroup at 40000 B where a live co-residency probe counts 3, and 1 at
// 64000 B where the probe counts 2. Every row below is that probe — blocks
// observed concurrently resident, divided by the 20 workgroup processors the
// device reports — and it closes only on a 131072 B pool over 4 SIMDs with a
// 1024 B allocation granule.
LSE_TEST(the_occupancy_model_reproduces_the_measured_hardware_table) {
  const backend::DeviceInfo d = gfx1151();
  LSE_EXPECT_EQ(backend::cus_per_lds_pool(d), 2u);
  const opt::DeviceCapacity cap = opt::DeviceCapacity::of(d);
  LSE_EXPECT(cap.usable());
  LSE_EXPECT_EQ(cap.lds_bytes_per_pool.value_or(0u), 131072u);
  LSE_EXPECT_EQ(cap.simds_per_lds_pool.value_or(0u), 4u);
  LSE_EXPECT_EQ(cap.wave_slots_per_simd.value_or(0u), 16u);
  LSE_EXPECT_EQ(cap.lds_alloc_granule_bytes.value_or(0u), 1024u);

  struct Row {
    std::uint32_t threads;
    std::uint32_t lds;
    std::uint32_t wgs_per_pool;
  };
  // 256-thread rows: 8 waves each, so 64 wave slots per pool cap residency at
  // 8 and scratch takes over above 16384 B. 32-thread rows: one wave, so the
  // slot cap is 64 and every row is the granule's answer — which is what makes
  // 5200 B (21, not 23) pin the granule at 1024 rather than 512.
  constexpr Row kMeasured[] = {
      {256, 0, 8},     {256, 16384, 8}, {256, 32768, 4}, {256, 36000, 3},
      {256, 40000, 3}, {256, 48000, 2}, {256, 64000, 2}, {256, 65536, 2},
      {32, 2600, 42},  {32, 4900, 25},  {32, 5000, 25},  {32, 5200, 21},
      {32, 8192, 16},  {32, 9000, 14},  {32, 12000, 10}, {32, 16000, 8},
  };
  for (const Row& r : kMeasured) {
    const opt::Occupancy occ =
        opt::occupancy(cap, opt::KernelDemand::counted(r.threads, r.lds));
    LSE_EXPECT_EQ(occ.workgroups_per_pool, r.wgs_per_pool);
    if (occ.workgroups_per_pool != r.wgs_per_pool) {
      std::printf("       threads=%u lds=%u got=%u want=%u\n", r.threads,
                  r.lds, occ.workgroups_per_pool, r.wgs_per_pool);
    }
  }
  // Past the device's own per-workgroup cap there is no legal launch at all,
  // and that is a different answer from "one workgroup fits".
  LSE_EXPECT(!opt::occupancy(cap, opt::KernelDemand::counted(256, 65552))
                  .seated());
}

// THE RULE: occupancy is the minimum over every limit that applies, and the
// answer must name which one bound. Nothing here is a threshold — each row is
// a resource made scarce on its own while the others are left alone.
LSE_TEST(occupancy_is_the_minimum_over_the_limits_that_apply) {
  const opt::DeviceCapacity cap = opt::DeviceCapacity::of(gfx1151());

  // Nothing scarce: the wave slots are the ceiling, as they always are.
  opt::KernelDemand plain = opt::KernelDemand::counted(256, 0);
  const opt::Occupancy slots = opt::occupancy(cap, plain);
  LSE_EXPECT(slots.binding == opt::OccupancyLimit::kWaveSlots);
  LSE_EXPECT_EQ(slots.waves_per_simd, 16u);

  // Scratch scarce, registers not.
  opt::KernelDemand lds = plain;
  lds.lds_bytes = backend::DeviceFact<std::uint32_t>::queried(48000);
  const opt::Occupancy by_lds = opt::occupancy(cap, lds);
  LSE_EXPECT(by_lds.binding == opt::OccupancyLimit::kWorkgroupScratch);
  LSE_EXPECT_EQ(by_lds.workgroups_per_pool, 2u);

  if (!cap.vector_registers_per_simd.known()) {
    std::printf("       (register arm skipped: no ISA register facts)\n");
    return;
  }

  // Registers scarce, scratch not. 102 vgprs round to 120 on a 24-register
  // granule, and 1536/120 is 12 — which is the number the compiler itself
  // printed for the kernel that motivated this model, while the engine was
  // busy cutting its workgroup scratch for nothing.
  opt::KernelDemand regs = plain;
  regs.vector_registers = backend::DeviceFact<std::uint32_t>::queried(102);
  const opt::Occupancy by_regs = opt::occupancy(cap, regs);
  LSE_EXPECT(by_regs.binding == opt::OccupancyLimit::kVectorRegisters);
  LSE_EXPECT_EQ(by_regs.waves_per_simd, 12u);
  LSE_EXPECT(by_regs.registers_counted);
  LSE_EXPECT(by_regs.exact);
  // And it really is a minimum: cutting scratch cannot buy back a wave the
  // register file already refused.
  opt::KernelDemand regs_no_lds = regs;
  regs_no_lds.lds_bytes = backend::DeviceFact<std::uint32_t>::queried(0);
  LSE_EXPECT_EQ(opt::occupancy(cap, regs_no_lds).waves_per_simd, 12u);

  // Both scarce: the tighter one wins, and the loser is still reported.
  // 200 vgprs allocate 216 and give 7 waves/SIMD, i.e. 3 workgroups of 8 waves
  // per pool; 32000 B of scratch rounds to 32768 and gives 4. Registers bind.
  opt::KernelDemand both = plain;
  both.lds_bytes = backend::DeviceFact<std::uint32_t>::queried(32000);
  both.vector_registers = backend::DeviceFact<std::uint32_t>::queried(200);
  const opt::Occupancy min = opt::occupancy(cap, both);
  LSE_EXPECT(min.binding == opt::OccupancyLimit::kVectorRegisters);
  LSE_EXPECT_EQ(min.workgroups_per_pool, 3u);
  LSE_EXPECT_EQ(min.waves_per_simd, 6u);
  LSE_EXPECT_EQ(min.waves_by_scratch, 8u);
  LSE_EXPECT(min.waves_per_simd < min.waves_by_scratch);
}

// An unanswered fact must make the model DEGRADE, not guess. A limit it cannot
// count drops out of the min and the answer says it is no longer exact, which
// is what stops a decision spending a number nobody established.
LSE_TEST(an_unknown_capacity_degrades_rather_than_guessing) {
  const opt::DeviceCapacity full = opt::DeviceCapacity::of(gfx1151());

  // No register file reported: the register arm does not participate at all,
  // rather than being read as "zero registers" or "unlimited but counted".
  opt::DeviceCapacity no_regs = full;
  no_regs.vector_registers_per_simd = {};
  opt::KernelDemand d;
  d.threads = 256;
  d.vector_registers = backend::DeviceFact<std::uint32_t>::queried(200);
  const opt::Occupancy blind = opt::occupancy(no_regs, d);
  LSE_EXPECT(!blind.registers_counted);
  LSE_EXPECT_EQ(blind.waves_by_registers, opt::Occupancy::kNoLimit);
  LSE_EXPECT_EQ(blind.waves_per_simd, 16u);

  // No allocation granule: the request is charged unrounded, which UNDERstates
  // it, so the answer is marked inexact and a tie it cannot vouch for is
  // refused instead of guessed.
  opt::DeviceCapacity no_granule = full;
  no_granule.lds_alloc_granule_bytes = {};
  const auto ask = [](std::uint32_t bytes) {
    return opt::KernelDemand::counted(256, bytes);
  };
  LSE_EXPECT(opt::occupancy(full, ask(40000)).exact);
  LSE_EXPECT(!opt::occupancy(no_granule, ask(40000)).exact);

  // 40000 and 40960 bytes seat the same 3 workgroups per pool either way. The
  // measured granule KNOWS they round to the same allocation, so the larger
  // request is admitted on the tie; without it the model declines to say, and
  // refuses. Same two numbers, and the difference is only what was measured.
  LSE_EXPECT(opt::prefer(opt::occupancy(full, ask(40960)),
                         opt::occupancy(full, ask(40000))));
  LSE_EXPECT(!opt::prefer(opt::occupancy(no_granule, ask(40960)),
                          opt::occupancy(no_granule, ask(40000))));
  // A request that did not grow is still admitted: no rounding step, known or
  // unknown, separates two identical asks.
  LSE_EXPECT(opt::prefer(opt::occupancy(no_granule, ask(40000)),
                         opt::occupancy(no_granule, ask(40000))));

  // Nothing known at all: no answer, rather than a plausible one.
  const opt::DeviceCapacity nothing;
  LSE_EXPECT(!nothing.usable());
  LSE_EXPECT(!opt::occupancy(nothing, ask(40000)).seated());
}

// ---------------------------------------------------------------------------
// Pricing an arrangement: the traffic it moves and the residency it achieves,
// in one figure.
//
// Every case below is a RULE, not a number. Each one makes exactly one of the
// two terms decisive and checks that the other cannot overturn it, so a change
// to the measured curve moves the figures without moving any of these answers.
// ---------------------------------------------------------------------------

namespace {

// One row tile of a 4-bit contraction: `rows` activation rows against `cols`
// weight columns over `depth`, and the grid such a tile dispatches.
opt::Arrangement tile(std::uint32_t rows, std::uint64_t depth,
                      std::uint64_t n, std::uint64_t m,
                      std::uint32_t lds_bytes) {
  constexpr std::uint32_t kCols = 8;   // one column per wave of a 256-thread group
  constexpr std::uint32_t kThreads = 256;
  opt::Arrangement a;
  a.traffic = opt::contraction_traffic(
      rows, kCols, depth, /*weight_bits=*/4, /*activation_bytes=*/4,
      /*scale_bytes=*/kCols * (depth / 64) * 2 * 2,
      /*output_bytes=*/rows * kCols * 4);
  const auto blocks = static_cast<std::uint32_t>((m + rows - 1) / rows);
  a.traffic.workgroups =
      static_cast<std::uint32_t>((n + kCols - 1) / kCols) * blocks;
  a.traffic.workgroup_threads = kThreads;
  a.demand = opt::KernelDemand::counted(kThreads, lds_bytes);
  return a;
}

}  // namespace

// THE RULE, both halves at once: an arrangement that seats one workgroup on a
// pool loses to one that seats more when the traffic is comparable, and a
// two-row tile survives at two rows because the traffic it saves outweighs the
// residency it costs.
//
// The second half is the one that matters most in this tree. Multi-token
// prediction verifies two tokens per step, and its measured 1.96x rests on
// those two tokens costing ONE token of weight traffic — which is a two-row
// tile at m=2. A rule that scored residency alone would answer one row at every
// shape and take that speedup away; this test is what stops it.
LSE_TEST(an_arrangement_is_priced_on_both_traffic_and_residency) {
  const opt::DeviceCapacity cap = opt::DeviceCapacity::of(gfx1151());
  LSE_EXPECT(cap.residency_bandwidth.known());

  // Same traffic, different residency: more workgroups on the pool wins, and
  // it wins because the bytes are charged at a lower share, not by fiat.
  {
    opt::Arrangement roomy = tile(2, 5120, 17408, 128, 16000);
    opt::Arrangement tight = roomy;
    tight.demand = opt::KernelDemand::counted(256, 64000);
    const opt::ArrangementCost a = opt::arrangement_cost(cap, roomy);
    const opt::ArrangementCost b = opt::arrangement_cost(cap, tight);
    LSE_EXPECT_EQ(a.launch_bytes, b.launch_bytes);
    LSE_EXPECT(a.residency.workgroups_per_pool >
               b.residency.workgroups_per_pool);
    LSE_EXPECT(a.charged_bytes < b.charged_bytes);
    LSE_EXPECT(opt::prefer(a, b));
    LSE_EXPECT(!opt::prefer(b, a));
  }

  // Two rows at m=2 over the widest projection in the 27B: the tile halves the
  // weight traffic and costs a halving of residency, and the traffic wins.
  {
    const opt::Arrangement one = tile(1, 17408, 5120, 2, 27200);
    const opt::Arrangement two = tile(2, 17408, 5120, 2, 54400);
    const opt::ArrangementCost a = opt::arrangement_cost(cap, one);
    const opt::ArrangementCost b = opt::arrangement_cost(cap, two);
    // The residency really does fall — this is not a case where the tile was
    // free — and the arrangement is taken anyway.
    LSE_EXPECT(b.residency.workgroups_per_pool <
               a.residency.workgroups_per_pool);
    LSE_EXPECT(b.launch_bytes < a.launch_bytes);
    LSE_EXPECT(b.charged_bytes < a.charged_bytes);
    const opt::Arrangement ladder[] = {one, two};
    LSE_EXPECT_EQ(opt::best_arrangement(cap, ladder), std::size_t{1});
  }

  // And the converse, at the same shape and a row count further up the ladder:
  // a tile whose traffic saving no longer covers what it gives up in residency
  // is refused. Nothing here is a cap — eight rows FIT the part's addressable
  // scratch, and the model declines them anyway.
  {
    const opt::Arrangement ladder[] = {
        tile(1, 5120, 17408, 128, 8000), tile(2, 5120, 17408, 128, 16000),
        tile(4, 5120, 17408, 128, 32000), tile(8, 5120, 17408, 128, 64000)};
    const std::size_t best = opt::best_arrangement(cap, ladder);
    LSE_EXPECT(best < 4);
    // The largest rung moves the fewest bytes and is still not the answer.
    const opt::ArrangementCost widest = opt::arrangement_cost(cap, ladder[3]);
    const opt::ArrangementCost chosen = opt::arrangement_cost(cap, ladder[best]);
    LSE_EXPECT(widest.launch_bytes < chosen.launch_bytes);
    LSE_EXPECT(chosen.charged_bytes < widest.charged_bytes);
    LSE_EXPECT(chosen.residency.workgroups_per_pool >
               widest.residency.workgroups_per_pool);
  }
}

// An unmeasured fact degrades rather than guessing, here as everywhere else.
//
// With no residency curve the arm drops out, the comparison runs on traffic
// alone, and the answer is the largest tile — which is what a rule that only
// asked whether the tile FIT already gave. A part nobody measured keeps the
// behaviour it had instead of inheriting a curve measured on another part.
LSE_TEST(an_unmeasured_residency_curve_degrades_to_traffic_alone) {
  opt::DeviceCapacity blind = opt::DeviceCapacity::of(gfx1151());
  blind.residency_bandwidth = {};
  LSE_EXPECT(!blind.residency_bandwidth.known());

  const opt::Arrangement ladder[] = {
      tile(1, 5120, 17408, 128, 8000), tile(2, 5120, 17408, 128, 16000),
      tile(4, 5120, 17408, 128, 32000), tile(8, 5120, 17408, 128, 64000)};
  const opt::ArrangementCost c = opt::arrangement_cost(blind, ladder[3]);
  LSE_EXPECT(!c.charged);
  LSE_EXPECT_EQ(static_cast<std::uint64_t>(c.charged_bytes), c.launch_bytes);
  LSE_EXPECT_EQ(opt::best_arrangement(blind, ladder), std::size_t{3});

  // The same ladder on the same part WITH the curve answers differently, so the
  // difference is the measurement and nothing else.
  const opt::DeviceCapacity seeing = opt::DeviceCapacity::of(gfx1151());
  LSE_EXPECT(opt::best_arrangement(seeing, ladder) != std::size_t{3});

  // An arrangement that never said what it moves is not a free one: it is not
  // priced and never wins, however little scratch it asks for.
  opt::Arrangement silent;
  silent.demand = opt::KernelDemand::counted(256, 0);
  const opt::ArrangementCost none = opt::arrangement_cost(seeing, silent);
  LSE_EXPECT(!none.stated);
  LSE_EXPECT(!opt::prefer(none, opt::arrangement_cost(seeing, ladder[0])));
}

// A real tie is settled, and settled in an order that says why.
//
// Two arrangements moving the same bytes at the same share are separated by
// residency; two that also seat the same are separated by what they ask the
// pool for, because the room one leaves is what the next fusion decision has to
// spend. And a spilling arrangement loses whatever it moves — that is a
// different regime, not a smaller number.
LSE_TEST(a_tie_between_arrangements_is_settled_by_residency_then_by_the_ask) {
  const opt::DeviceCapacity cap = opt::DeviceCapacity::of(gfx1151());

  opt::Arrangement lean = tile(2, 5120, 17408, 128, 12000);
  opt::Arrangement fat = lean;
  fat.demand = opt::KernelDemand::counted(256, 16000);
  const opt::ArrangementCost a = opt::arrangement_cost(cap, lean);
  const opt::ArrangementCost b = opt::arrangement_cost(cap, fat);
  // Both are held by the wave slots, so residency cannot separate them.
  LSE_EXPECT_EQ(a.residency.workgroups_per_pool,
                b.residency.workgroups_per_pool);
  LSE_EXPECT_EQ(a.charged_bytes, b.charged_bytes);
  LSE_EXPECT(opt::prefer(a, b));
  LSE_EXPECT(!opt::prefer(b, a));

  // Spilling decides first and outright, in both directions.
  opt::ArrangementCost spilling = a;
  spilling.residency.spill = backend::SpillState::kSpilled;
  opt::ArrangementCost clean = b;
  clean.residency.spill = backend::SpillState::kNone;
  LSE_EXPECT(!opt::prefer(spilling, clean));
  LSE_EXPECT(opt::prefer(clean, spilling));
}

// The mirror of the test above, and the one that was missing: an unanswered
// DEMAND, not an unanswered capacity.
//
// A toolchain that reports no workgroup segment has said nothing about what
// the kernel asks for. Read as zero, that kernel gets the best occupancy the
// part can give — asserted as exact — and beats a measured arrangement that
// honestly declared 32000 bytes. The unknown must lose the comparison, not win
// it.
LSE_TEST(an_unknown_demand_degrades_rather_than_scoring_best) {
  const opt::DeviceCapacity cap = opt::DeviceCapacity::of(gfx1151());

  // A code object with registers and spill counts but no workgroup segment:
  // exactly what a toolchain that does not emit the field produces.
  backend::KernelResources silent;
  silent.vector_registers = backend::DeviceFact<std::uint32_t>::queried(64);
  silent.vector_spills = backend::DeviceFact<std::uint32_t>::queried(0);
  silent.scalar_spills = backend::DeviceFact<std::uint32_t>::queried(0);
  LSE_EXPECT(!silent.workgroup_segment_bytes.known());

  const opt::Occupancy unknown =
      opt::occupancy(cap, opt::KernelDemand::measured(256, silent));
  // The scratch arm did not run, and the answer says so rather than claiming
  // a count it does not have.
  LSE_EXPECT(!unknown.exact);
  LSE_EXPECT_EQ(unknown.waves_by_scratch, opt::Occupancy::kNoLimit);
  LSE_EXPECT(!unknown.scratch_request.known());

  // Zero bytes is a DIFFERENT demand: it is an answer, the arm runs, and the
  // figure is exact. Same occupancy, different confidence — which is the whole
  // distinction.
  backend::KernelResources none = silent;
  none.workgroup_segment_bytes = backend::DeviceFact<std::uint32_t>::queried(0);
  const opt::Occupancy zero =
      opt::occupancy(cap, opt::KernelDemand::measured(256, none));
  LSE_EXPECT(zero.exact);
  LSE_EXPECT_EQ(zero.workgroups_per_pool, unknown.workgroups_per_pool);

  // And the comparison: the unmeasured kernel must not displace a counted one
  // on a tie it was never scored for.
  backend::KernelResources heavy = none;
  heavy.workgroup_segment_bytes =
      backend::DeviceFact<std::uint32_t>::queried(32000);
  const opt::Occupancy counted =
      opt::occupancy(cap, opt::KernelDemand::measured(256, heavy));
  LSE_EXPECT_EQ(counted.workgroups_per_pool, 4u);
  LSE_EXPECT(!opt::prefer(unknown, zero));
  // Nor may it win by looking roomier than an arrangement that declared more.
  LSE_EXPECT(opt::prefer(unknown, counted) ==
             (unknown.workgroups_per_pool > counted.workgroups_per_pool));
  LSE_EXPECT(!opt::prefer(counted, unknown));
}

// A pool's SIZE and the CU count it was stated for are one fact. The SIMD
// count follows the live part, so a part that pairs its cores differently from
// the family row must not inherit a block sized for the row's pairing — that
// is a silent factor of two in every residency answer it gives.
LSE_TEST(a_pool_size_is_not_read_at_a_cu_count_nobody_stated_it_for) {
  backend::DeviceInfo paired = gfx1151();
  LSE_EXPECT(paired.arch_facts.lds_bytes_per_pool.known());
  LSE_EXPECT_EQ(paired.arch_facts.lds_bytes_per_pool.value, 131072u);
  LSE_EXPECT_EQ(paired.arch_facts.simds_per_lds_pool.value, 4u);

  // The same part reporting one CU behind a pool: two SIMDs draw on the block,
  // and nothing states how big that block is. The fact stays unknown and the
  // scratch limit drops out of the min rather than charging the request
  // against twice the block that exists.
  backend::DeviceInfo solo_cu = paired;
  solo_cu.cus_per_lds_pool = 1;
  solo_cu.arch_facts = backend::arch_facts_for(solo_cu);
  LSE_EXPECT_EQ(solo_cu.arch_facts.simds_per_lds_pool.value, 2u);
  LSE_EXPECT(!solo_cu.arch_facts.lds_bytes_per_pool.known());

  const opt::DeviceCapacity cap = opt::DeviceCapacity::of(solo_cu);
  const opt::Occupancy occ =
      opt::occupancy(cap, opt::KernelDemand::counted(256, 40000));
  LSE_EXPECT(!occ.exact);
  LSE_EXPECT_EQ(occ.waves_by_scratch, opt::Occupancy::kNoLimit);
}

// A kernel that SPILLS is a different regime, not a lower number. It must not
// be reachable by making some other arrangement's occupancy worse.
LSE_TEST(a_spilling_kernel_is_reported_as_spilling_not_as_less_occupancy) {
  const opt::DeviceCapacity cap = opt::DeviceCapacity::of(gfx1151());

  backend::KernelResources clean;
  clean.vector_registers = backend::DeviceFact<std::uint32_t>::queried(64);
  clean.workgroup_segment_bytes =
      backend::DeviceFact<std::uint32_t>::queried(48000);
  clean.vector_spills = backend::DeviceFact<std::uint32_t>::queried(0);
  clean.scalar_spills = backend::DeviceFact<std::uint32_t>::queried(0);

  backend::KernelResources spilling = clean;
  spilling.workgroup_segment_bytes =
      backend::DeviceFact<std::uint32_t>::queried(0);
  spilling.vector_spills = backend::DeviceFact<std::uint32_t>::queried(12);

  const opt::Occupancy a =
      opt::occupancy(cap, opt::KernelDemand::measured(256, clean));
  const opt::Occupancy b =
      opt::occupancy(cap, opt::KernelDemand::measured(256, spilling));
  // The spilling one has the BETTER occupancy — no scratch at all — so a model
  // that folded spilling into the number would prefer it.
  LSE_EXPECT(b.workgroups_per_pool > a.workgroups_per_pool);
  LSE_EXPECT(b.spill == backend::SpillState::kSpilled);
  LSE_EXPECT(a.spill == backend::SpillState::kNone);
  LSE_EXPECT(!opt::prefer(b, a));
  LSE_EXPECT(opt::prefer(a, b));

  // A toolchain that reports no spill counts has said nothing, and nothing
  // must not read back as "did not spill": the state stays unknown, and only a
  // REPORTED spill decides a comparison. (Reading silence as spilling would
  // disable every fusion on a toolchain that never emits the counts, which is
  // the whole Loom path.)
  backend::KernelResources silent = clean;
  silent.vector_spills = {};
  silent.scalar_spills = {};
  const opt::Occupancy c =
      opt::occupancy(cap, opt::KernelDemand::measured(256, silent));
  LSE_EXPECT(c.spill == backend::SpillState::kUnknown);
  LSE_EXPECT(c.spill != backend::SpillState::kNone);
  LSE_EXPECT(opt::prefer(c, b));
}

// THE RULE the 13x prefill regression violated: a fusion that would seat fewer
// workgroups per scratch pool than the arrangement it replaces is refused. The
// number is not written here — the two sides are computed and compared.
LSE_TEST(a_fusion_that_costs_residency_is_refused_by_the_model) {
  const opt::DeviceCapacity cap = opt::DeviceCapacity::of(gfx1151());
  const auto at = [&](std::uint32_t bytes) {
    return opt::occupancy(cap, opt::KernelDemand::counted(256, bytes));
  };

  // The measured failure: two ~32000-byte panels merged into one 64000-byte
  // run. It FITS the 65536-byte per-workgroup cap, which is why a fitting test
  // admitted it, and it seats half as many workgroups.
  const opt::Occupancy unfused = at(32000);
  const opt::Occupancy fused = at(64000);
  LSE_EXPECT(fused.workgroups_per_pool < unfused.workgroups_per_pool);
  LSE_EXPECT_EQ(fused.workgroups_per_pool, 2u);
  LSE_EXPECT_EQ(unfused.workgroups_per_pool, 4u);
  LSE_EXPECT(!opt::prefer(fused, unfused));

  // The nearby case that must NOT be got wrong: siblings sharing one staged
  // row ask for exactly what one of them asks for alone, so the merge is
  // residency-neutral and is admitted on equality. The 27B's hidden is 5120,
  // whose f32 row is 20480 bytes.
  LSE_EXPECT(opt::prefer(at(20480), at(20480)));
  // And a merge that seats MORE is never refused.
  LSE_EXPECT(opt::prefer(at(16384), at(32000)));
}

// A verdict must not move under a running model. The engine re-emits every
// group every step, so an answer that changed when a measurement landed would
// change the kernel mid-run — and the first answer for an arrangement is
// therefore the one that stands for the process.
LSE_TEST(a_fusion_verdict_does_not_move_once_it_is_taken) {
  const opt::DeviceCapacity cap = opt::DeviceCapacity::of(gfx1151());

  opt::FusionCandidate c;
  c.threads = 256;
  c.fused_scratch_bytes = 8192;
  c.worst_solo_scratch_bytes = 8192;
  c.fused_entry = "lse_fused_verdict_probe";
  LSE_EXPECT(opt::admit_fusion(cap, c).admit);

  // Now make the same arrangement look terrible, in both the ways a later
  // answer could differ: a much larger request, and a measurement saying the
  // kernel spills. Neither may move the answer for this process.
  opt::KernelMeasurements& measured = opt::KernelMeasurements::instance();
  backend::KernelResources spilling;
  spilling.vector_registers = backend::DeviceFact<std::uint32_t>::queried(180);
  spilling.vector_spills = backend::DeviceFact<std::uint32_t>::queried(64);
  spilling.scalar_spills = backend::DeviceFact<std::uint32_t>::queried(0);
  measured.record(c.fused_entry, spilling);

  opt::FusionCandidate worse = c;
  worse.fused_scratch_bytes = 64000;
  LSE_EXPECT(opt::admit_fusion(cap, worse).admit);

  // An arrangement nobody has asked about yet is scored freshly, spill and
  // all — so the freeze is per arrangement, not a global off switch.
  opt::FusionCandidate other = worse;
  other.fused_entry = "lse_fused_verdict_probe_2";
  const opt::FusionVerdict fresh = opt::admit_fusion(cap, other);
  LSE_EXPECT(!fresh.admit);
  LSE_EXPECT(fresh.fused.workgroups_per_pool < fresh.unfused.workgroups_per_pool);

  // And a spilling measurement on a residency-neutral arrangement refuses it,
  // which is the one thing a measurement of ONE side is allowed to decide:
  // spilling disqualifies an arrangement outright rather than being compared
  // against anything. The verdict is not `measured` — nothing was measured for
  // the launches this merge replaces, so the residency figures on both sides
  // are still the emitter's own counts.
  measured.record("lse_fused_verdict_probe_3", spilling);
  opt::FusionCandidate spills = c;
  spills.fused_entry = "lse_fused_verdict_probe_3";
  const opt::FusionVerdict v = opt::admit_fusion(cap, spills);
  LSE_EXPECT(!v.measured);
  LSE_EXPECT(v.fused.spill == backend::SpillState::kSpilled);
  LSE_EXPECT(!v.admit);
}

// A MEASUREMENT ON ONE SIDE OF A COMPARISON IS NOT A MEASUREMENT. The fused
// kernel's workgroup segment is the whole body's, every stage's own staging
// included; the unfused side, before anyone has built it, can only be counted.
// Substituting one against the other compares two different quantities and
// turns the verdict on that difference, so the substitution is all of it or
// none of it.
LSE_TEST(a_measured_verdict_needs_both_sides_measured) {
  const opt::DeviceCapacity cap = opt::DeviceCapacity::of(gfx1151());
  opt::KernelMeasurements& measured = opt::KernelMeasurements::instance();

  const auto segment = [](std::uint32_t bytes) {
    backend::KernelResources r;
    r.workgroup_segment_bytes =
        backend::DeviceFact<std::uint32_t>::queried(bytes);
    r.vector_spills = backend::DeviceFact<std::uint32_t>::queried(0);
    r.scalar_spills = backend::DeviceFact<std::uint32_t>::queried(0);
    return r;
  };

  // The fused object is on record and the solo ones are not. The counts, not
  // the measurement, decide — and they say the merge is neutral.
  measured.record("lse_fused_both_sides", segment(64000));
  const std::vector<std::string> solo = {"lse_solo_a", "lse_solo_b"};
  opt::FusionCandidate half;
  half.threads = 256;
  half.fused_scratch_bytes = 8192;
  half.worst_solo_scratch_bytes = 8192;
  half.fused_entry = "lse_fused_both_sides";
  half.solo_entries = solo;
  const opt::FusionVerdict counted = opt::admit_fusion(cap, half);
  LSE_EXPECT(!counted.measured);
  LSE_EXPECT(counted.admit);

  // Both sides on record: 64000 bytes fused against 16000 solo is 2
  // workgroups per pool against 8, and the merge is refused on the measured
  // figures rather than on the counted ones, which claimed a tie.
  measured.record("lse_solo_a", segment(16000));
  measured.record("lse_solo_b", segment(8000));
  opt::FusionCandidate full = half;
  full.fused_entry = "lse_fused_both_sides_2";
  measured.record(full.fused_entry, segment(64000));
  const opt::FusionVerdict both = opt::admit_fusion(cap, full);
  LSE_EXPECT(both.measured);
  LSE_EXPECT_EQ(both.fused.workgroups_per_pool, 2u);
  // The TIGHTEST of the launches it replaces, not the loosest: 16000 rounds to
  // 16384 and seats 8, which the wave slots cap at 8 anyway.
  LSE_EXPECT_EQ(both.unfused.workgroups_per_pool, 8u);
  LSE_EXPECT(!both.admit);
}

// BOTH SIDES OF THE COMPARISON MUST BE THE SAME QUANTITY, and the quantity is
// what the emitted body declares.
//
// This is the arrangement that is live on Qwen3.8-27B-4bit today: two wide
// 4-bit linears off one 5120-long f32 activation, decode shape. The old
// pricing handed the model align(5120*4) = 20480 for BOTH arrangements — the
// fused body's hoisted panel, quoted back as the price of a solo launch that
// never allocates one — so `prefer` saw a tie on every run in every model and
// the gate could not fire at all.
//
// The true figures, and they are not close. The merged body declares the f32
// panel plus the int8 form it hoists beside it: 20480 + 8000 = 28480. The same
// stage alone declares no panel at all — the integer path quantizes straight
// out of global — so it declares only its own staging, 8000. 28480 rounds to
// 28672 and seats 4 workgroups on a 131072-byte pool; 8000 rounds to 8192 and
// seats 16, which the wave slots cap at 8. Four times the residency, and the
// model was being told it was a tie.
LSE_TEST(a_merged_run_is_priced_by_what_it_declares_not_by_its_panel) {
  ::unsetenv("LSE_WMMA");
  const backend::DeviceInfo dev = gfx1151();
  constexpr std::int64_t kK = 5120;
  constexpr std::int64_t kN = 640;
  constexpr std::int64_t kGroup = 64;
  constexpr int kBits = 4;
  constexpr std::int64_t kLanes = kK * kBits / 32;
  constexpr std::int64_t kGroups = kK / kGroup;

  Array x = Array::zeros(Shape{1, kK}, DType::kF32);
  const auto wide = [&] {
    Array packed = Array::zeros(Shape{kN, kLanes}, DType::kU32);
    Array scales = Array::zeros(Shape{kN, kGroups}, DType::kBF16);
    Array biases = Array::zeros(Shape{kN, kGroups}, DType::kBF16);
    return quant_linear(x, packed, scales, biases, kBits, kGroup);
  };
  Array q = wide();
  Array k = wide();
  const std::vector<graph::NodePtr> run = {q.node(), k.node()};

  const graph::IKernelEmitter::RunScratch cost = kEmitter.run_scratch(run, dev);
  LSE_EXPECT_EQ(cost.threads, 256u);
  // The f32 panel, the int8 panel hoisted beside it, and nothing per stage:
  // with both hoisted, neither body declares scratch of its own.
  LSE_EXPECT_EQ(cost.fused, 28480u);
  // What ONE of them declares launching alone. Not the panel: the solo body
  // has no panel.
  LSE_EXPECT_EQ(cost.worst_solo, 8000u);

  const opt::DeviceCapacity cap = opt::DeviceCapacity::of(dev);
  const opt::Occupancy fused =
      opt::occupancy(cap, opt::KernelDemand::counted(cost.threads, cost.fused));
  const opt::Occupancy solo = opt::occupancy(
      cap, opt::KernelDemand::counted(cost.threads, cost.worst_solo));
  LSE_EXPECT_EQ(fused.workgroups_per_pool, 4u);
  LSE_EXPECT_EQ(solo.workgroups_per_pool, 8u);
  LSE_EXPECT(fused.binding == opt::OccupancyLimit::kWorkgroupScratch);

  opt::FusionCandidate candidate;
  candidate.threads = cost.threads;
  candidate.fused_scratch_bytes = cost.fused;
  candidate.worst_solo_scratch_bytes = cost.worst_solo;
  LSE_EXPECT(!opt::admit_fusion(cap, candidate).admit);

  // And the old pricing, restated, so the regression is a number and not a
  // memory: quoting the panel on both sides ties, and a tie is admitted.
  opt::FusionCandidate as_it_was = candidate;
  as_it_was.fused_scratch_bytes = 20480;
  as_it_was.worst_solo_scratch_bytes = 20480;
  LSE_EXPECT(opt::admit_fusion(cap, as_it_was).admit);
}

// The prediction the gate is fed and the allocation the body makes are the
// same number, not two numbers that agree today. A drift between them is a run
// admitted against a launch that never happens, which is the whole failure
// this area exists for, so it is pinned rather than trusted.
LSE_TEST(the_run_price_is_the_bytes_the_merged_body_declares) {
  ::unsetenv("LSE_WMMA");
  const backend::DeviceInfo dev = gfx1151();

  // Dense f32 siblings over one 512-wide activation: small enough that the
  // wave slots bind and the run is admitted, so there is an emitted body to
  // compare against.
  Array x = Array::full(Shape{1, 512}, DType::kF32, 1.0f);
  Array wa = Array::full(Shape{16, 512}, DType::kF32, 0.5f);
  Array wb = Array::full(Shape{16, 512}, DType::kF32, 0.25f);
  Array a = linear(x, wa);
  Array b = linear(x, wb);
  const std::vector<graph::NodePtr> run = {a.node(), b.node()};

  const graph::IKernelEmitter::RunScratch cost = kEmitter.run_scratch(run, dev);
  auto e = kEmitter.emit(sibling_group({a, b}), dev);
  LSE_EXPECT(e.ok());
  if (!e.ok()) {
    std::printf("       %s\n", e.status().to_string().c_str());
    return;
  }
  LSE_EXPECT(cost.fused != 0u);
  LSE_EXPECT_EQ(cost.fused, e->lds_bytes);
  // And the arrangement is named the same way at both sites. The decision is
  // taken before the group exists, so the price is looked up under a name
  // reconstructed from the run; a name that did not match would silently never
  // find the measurement it is there to find.
  LSE_EXPECT(cost.fused_entry == e->entry_name);
  LSE_EXPECT_EQ(cost.solo_entries.size(), 2u);
  // e->lds_bytes is itself read off the body, and the text agrees with it.
  auto counted = backend::HipEmitter::shared_bytes(e->source);
  LSE_EXPECT(counted.ok());
  if (counted.ok()) LSE_EXPECT_EQ(*counted, cost.fused);
  // One shared row for two stages, and the solo price is that same row, since
  // a dense f32 GEMV alone stages exactly what the run hoists for it. Equal,
  // and equality is admitted — this is the case a stricter rule would break.
  LSE_EXPECT_EQ(cost.worst_solo, 2048u);
  LSE_EXPECT_EQ(cost.fused, 2048u);
}

// A run is admitted on RESIDENCY, and refused on it. Three cases, one rule:
// the merged body must seat at least as many workgroups on one scratch pool as
// the tightest launch it replaces.
//
// The middle case is the measured failure. Two panels at exactly half the
// per-workgroup cap sum to the whole cap, so a fitting test admits them — and
// they seat 2 workgroups per pool where either alone seats 4. That is the same
// shape as the 64000-byte family still sitting in this machine's kernel cache,
// and the 141-token prefill that went 9.4 s to 126.8 s.
LSE_TEST(a_sibling_run_is_admitted_and_refused_on_residency_not_on_fitting) {
  ::unsetenv("LSE_WMMA");
  const backend::DeviceInfo dev = gfx1151();
  const std::uint32_t budget = dev.lds_bytes_per_workgroup;
  LSE_EXPECT_EQ(budget, 65536u);

  // Two distinct activations, 512 f32 = 2048 B each. Scratch is not the
  // binding resource at that size — 8 workgroups of 256 threads fill the wave
  // slots first, whether the run asks for 2048 bytes or 4096 — so the merge
  // costs nothing and is emitted as one body with both panels. A rule that
  // simply forbade a second distinct panel would refuse this.
  {
    Array x0 = Array::full(Shape{1, 512}, DType::kF32, 1.0f);
    Array x1 = Array::full(Shape{1, 512}, DType::kF32, 2.0f);
    Array wa = Array::full(Shape{16, 512}, DType::kF32, 0.5f);
    Array wb = Array::full(Shape{16, 512}, DType::kF32, 0.25f);
    auto e = kEmitter.emit(sibling_group({linear(x0, wa), linear(x1, wb)}), dev);
    LSE_EXPECT(e.ok());
    if (!e.ok()) {
      std::printf("       %s\n", e.status().to_string().c_str());
      return;
    }
    LSE_EXPECT_EQ(shared_arrays(e->source).size(), 2u);
    // Summed, not maximized: one launch, one allocation.
    LSE_EXPECT_EQ(e->lds_bytes, 4096u);
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

  // Two distinct activations, 8192 f32 = 32768 B each. The sum is exactly the
  // per-workgroup cap, so it FITS — and it halves residency, so it is refused.
  {
    Array x0 = Array::full(Shape{1, 8192}, DType::kF32, 1.0f);
    Array x1 = Array::full(Shape{1, 8192}, DType::kF32, 2.0f);
    Array wa = Array::full(Shape{16, 8192}, DType::kF32, 0.5f);
    Array wb = Array::full(Shape{16, 8192}, DType::kF32, 0.25f);
    auto e = kEmitter.emit(sibling_group({linear(x0, wa), linear(x1, wb)}), dev);
    LSE_EXPECT(e.ok());
    if (!e.ok()) return;
    // The refusal is the model's, and it is not a constant: state both sides.
    const opt::DeviceCapacity cap = opt::DeviceCapacity::of(dev);
    const opt::KernelDemand merged = opt::KernelDemand::counted(256, 65536);
    const opt::KernelDemand alone = opt::KernelDemand::counted(256, 32768);
    LSE_EXPECT_EQ(opt::occupancy(cap, merged).workgroups_per_pool, 2u);
    LSE_EXPECT_EQ(opt::occupancy(cap, alone).workgroups_per_pool, 4u);
    LSE_EXPECT(65536u <= budget);  // it fits, and fitting is not the question
    LSE_EXPECT(shared_arrays(e->source).size() < 2u);
    LSE_EXPECT(e->lds_bytes < 65536u);
    auto counted = backend::HipEmitter::shared_bytes(e->source);
    LSE_EXPECT(counted.ok());
    if (counted.ok()) LSE_EXPECT_EQ(*counted, e->lds_bytes);
  }

  // One f32 line over the cap: 32768 + 32784 = 65552. Not a worse arrangement
  // but an illegal one — no workgroup can be seated at all — and it is refused
  // rather than emitted over budget for the compiler to reject.
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
  LSE_EXPECT(kernels::matrix_target(d12).has_value());
  LSE_EXPECT(*kernels::matrix_target(d12) ==
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
  LSE_EXPECT(*kernels::matrix_target(d942) ==
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
    LSE_EXPECT(kernels::wmma_linear_for(s) == nullptr);
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

  const KernelPrimitiveBase* prim = kernels::wmma_linear_for(s);
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

// ---------------------------------------------------------------------------
// The loom generator: the same fusion group, through the other dialect.
//
// Every check here is DIFFERENTIAL. A Loom kernel is only interesting if it is
// the same kernel: same bindings, same launch, same bytes out. Where the two
// dialects cannot be bit-identical the difference is measured and printed
// rather than asserted away.
// ---------------------------------------------------------------------------
namespace {

const backend::LoomEmitter kLoomEmitter;

FusionGroup group_anchored(Array& root, OpKind kind) {
  const NodePtr roots[] = {root.node()};
  for (const FusionGroup& g : Partitioner::partition(roots)) {
    if (g.anchor == kind) return g;
  }
  return FusionGroup{};
}

struct Diff {
  bool ran = false;
  std::size_t compared = 0;
  std::size_t nonzero = 0;
  std::size_t mismatched = 0;
  double worst_abs = 0.0;
  double worst_rel = 0.0;
};

// A real device buffer behind a graph input. Array::zeros/full record a
// kConstant, which the emitter folds into the source as a literal — so a
// differential over those would be comparing two kernels with no inputs.
Array device_input(backend::IBackend& be, Shape shape,
                   const std::vector<float>& host) {
  auto buf = be.allocate(host.size() * 4, backend::MemoryClass::kDevice);
  if (!buf.ok()) return Array::zeros(shape, DType::kF32);
  backend::DeviceBuffer owned = buf.release();
  (void)be.copy_h2d(host.data(), owned, host.size() * 4, 0);
  return Array::from_buffer(std::move(owned), shape, DType::kF32);
}

// Emit `group` both ways, compile each with its own dialect's compiler, run
// both against identical inputs and compare what the OUTPUT bindings hold.
Diff differential(const FusionGroup& group,
                  const std::vector<std::pair<const Node*, std::vector<float>>>& host,
                  const char* label) {
  Diff d;
  backend::IBackend* be = live_hrx();
  if (be == nullptr || !kLoom.available() || !kCompiler.available()) return d;
  const backend::DeviceInfo& dev = be->device_info();

  auto hip = kEmitter.emit(group, dev);
  auto loom = kLoomEmitter.emit(group, dev);
  if (!hip.ok()) {
    std::printf("       %s: hip declined: %s\n", label,
                hip.status().message().c_str());
    return d;
  }
  if (!loom.ok()) {
    std::printf("       %s: loom declined: %s\n", label,
                loom.status().message().c_str());
    return d;
  }
  LSE_EXPECT(loom->dialect == Dialect::kLoom);
  LSE_EXPECT_EQ(loom->binding_order.size(), hip->binding_order.size());
  for (std::size_t i = 0; i < loom->binding_order.size() &&
                          i < hip->binding_order.size(); ++i) {
    LSE_EXPECT(loom->binding_order[i] == hip->binding_order[i]);
  }
  LSE_EXPECT_EQ(loom->constants.total_bytes, hip->constants.total_bytes);
  for (int k = 0; k < 3; ++k) {
    LSE_EXPECT_EQ(loom->dims.workgroup_size[k], hip->dims.workgroup_size[k]);
    LSE_EXPECT_EQ(loom->dims.workgroup_count[k], hip->dims.workgroup_count[k]);
  }

  const auto a = run_emitted_with(kCompiler, *hip, dev, *be, host);
  const auto b = run_emitted_with(kLoom, *loom, dev, *be, host);
  if (a.empty() || b.empty() || a.size() != b.size()) {
    if (b.empty()) {
      const std::string path = std::string("/tmp/lse-loom-") + label + ".loom";
      if (FILE* f = std::fopen(path.c_str(), "w")) {
        std::fwrite(loom->source.data(), 1, loom->source.size(), f);
        std::fclose(f);
        std::printf("       %s: loom source written to %s\n", label,
                    path.c_str());
      }
    }
    return d;
  }
  std::unordered_set<const Node*> outs;
  for (const NodePtr& o : group.outputs) outs.insert(o.get());
  d.ran = true;
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (outs.count(hip->binding_order[i].get()) == 0) continue;
    for (std::size_t j = 0; j < a[i].size() && j < b[i].size(); ++j) {
      ++d.compared;
      const float x = a[i][j];
      const float y = b[i][j];
      if (x != 0.0f) ++d.nonzero;
      if (std::memcmp(&x, &y, sizeof(float)) == 0) continue;
      ++d.mismatched;
      const double abs_err =
          std::fabs(static_cast<double>(x) - static_cast<double>(y));
      d.worst_abs = std::max(d.worst_abs, abs_err);
      const double mag = std::fabs(static_cast<double>(x));
      if (mag > 0.0) d.worst_rel = std::max(d.worst_rel, abs_err / mag);
    }
  }
  std::printf("       %s: %zu output values (%zu nonzero), %zu differ, "
              "worst |abs| %.3g, worst rel %.3g\n",
              label, d.compared, d.nonzero, d.mismatched, d.worst_abs,
              d.worst_rel);
  return d;
}

}  // namespace

// The simplest real kernel: self-indexing, no workgroup scratch, no loop, one
// store through the hook — and an in-place input, so it is also the case where
// `buffer.assume.noalias` would be a miscompile.
LSE_TEST(loom_overwrite_slice_is_bit_exact_against_hip) {
  Array dst = Array::zeros(Shape{1, 64, 32}, DType::kF32);
  Array src = Array::full(Shape{1, 1, 32}, DType::kF32, 2.5f);
  Array begin = Array::full(Shape{1}, DType::kF32, 7.0f);
  Array y = overwrite_slice(dst, src, 1, begin);
  const FusionGroup g = group_anchored(y, OpKind::kOverwriteSlice);
  LSE_EXPECT(!g.nodes.empty());
  if (g.nodes.empty()) return;

  std::vector<float> dst_host(64 * 32);
  for (std::size_t i = 0; i < dst_host.size(); ++i) {
    dst_host[i] = static_cast<float>(i) * 0.5f;
  }
  const std::vector<float> src_host(32, 2.5f);
  const std::vector<float> begin_host(1, 7.0f);

  auto* be = live_hrx();
  if (be == nullptr) {
    std::printf("       (skipped: no hrx device)\n");
    return;
  }
  auto loom = kLoomEmitter.emit(g, be->device_info());
  LSE_EXPECT(loom.ok());
  if (!loom.ok()) {
    std::printf("       %s\n", loom.status().message().c_str());
    return;
  }
  // The source states the whole ABI: buffers in binding order, then the
  // dispatch constant, and the guard the primitive wrote as an early return.
  LSE_EXPECT(loom->source.find("kernel.def export(\"") == 0u);
  LSE_EXPECT(loom->source.find("launch(%in0: buffer, %in1: buffer, "
                               "%in2: buffer, %out: buffer, %count: i32)") !=
             std::string::npos);
  // in0 IS out. Asserting they do not alias would be a miscompile.
  LSE_EXPECT(loom->source.find("buffer.assume.noalias") == std::string::npos);
  LSE_EXPECT(loom->source.find("scalar.xori") != std::string::npos);

  const Diff d = differential(g,
                              {{dst.node().get(), dst_host},
                               {src.node().get(), src_host},
                               {begin.node().get(), begin_host}},
                              "overwrite_slice");
  LSE_EXPECT(d.ran);
  LSE_EXPECT_EQ(d.mismatched, 0u);
}

// The other shape a group takes: no primitive owns the indexing, so the
// scaffold loads every input at the broadcast index, runs the primitive rows
// and stores. Arithmetic only — the two dialects spell the same operations.
LSE_TEST(loom_elementwise_fusion_is_bit_exact_against_hip) {
  auto* be = live_hrx();
  if (be == nullptr) {
    std::printf("       (skipped: no hrx device)\n");
    return;
  }
  std::vector<float> xs(256);
  std::vector<float> ys(256);
  for (std::size_t i = 0; i < xs.size(); ++i) {
    xs[i] = static_cast<float>(i) * 0.125f - 8.0f;
    ys[i] = static_cast<float>(i) * -0.0625f + 3.0f;
  }
  Array x = device_input(*be, Shape{256}, xs);
  Array y = device_input(*be, Shape{256}, ys);
  Array z = x * y + x;
  const FusionGroup g = group_anchored(z, OpKind::kMul);
  LSE_EXPECT(!g.nodes.empty());
  if (g.nodes.empty()) return;

  const Diff d = differential(
      g, {{x.node().get(), xs}, {y.node().get(), ys}}, "mul+add");
  LSE_EXPECT(d.ran);
  LSE_EXPECT(d.nonzero > 0u);
  LSE_EXPECT_EQ(d.mismatched, 0u);
}

// A broadcast operand, so the scaffold's index arithmetic is exercised rather
// than the identity map.
LSE_TEST(loom_broadcast_operand_is_bit_exact_against_hip) {
  auto* be = live_hrx();
  if (be == nullptr) {
    std::printf("       (skipped: no hrx device)\n");
    return;
  }
  std::vector<float> xs(256);
  std::vector<float> bs(32);
  for (std::size_t i = 0; i < xs.size(); ++i) xs[i] = static_cast<float>(i);
  for (std::size_t i = 0; i < bs.size(); ++i) {
    bs[i] = static_cast<float>(i) * 0.25f;
  }
  Array x = device_input(*be, Shape{8, 32}, xs);
  Array bias = device_input(*be, Shape{1, 32}, bs);
  Array z = x + bias;
  const FusionGroup g = group_anchored(z, OpKind::kAdd);
  LSE_EXPECT(!g.nodes.empty());
  if (g.nodes.empty()) return;

  const Diff d = differential(
      g, {{x.node().get(), xs}, {bias.node().get(), bs}}, "broadcast add");
  LSE_EXPECT(d.ran);
  LSE_EXPECT(d.nonzero > 0u);
  LSE_EXPECT_EQ(d.mismatched, 0u);
}

// The transcendental rows. HIP's `__expf` and Loom's `scalar.expf<afn>` are
// both the hardware approximation, so the point of the test is to MEASURE the
// gap rather than to assume there is none.
LSE_TEST(loom_silu_matches_hip_across_the_range) {
  auto* be = live_hrx();
  if (be == nullptr) {
    std::printf("       (skipped: no hrx device)\n");
    return;
  }
  std::vector<float> xs(1024);
  for (std::size_t i = 0; i < xs.size(); ++i) {
    xs[i] = static_cast<float>(i) * 0.02f - 10.0f;
  }
  Array x = device_input(*be, Shape{1024}, xs);
  Array z = silu(x);
  const FusionGroup g = group_anchored(z, OpKind::kSiLU);
  LSE_EXPECT(!g.nodes.empty());
  if (g.nodes.empty()) return;

  const Diff d = differential(g, {{x.node().get(), xs}}, "silu");
  LSE_EXPECT(d.ran);
  LSE_EXPECT(d.nonzero > 0u);
  if (!d.ran) return;
  // A relative error past 1e-5 would mean the two rows are not the same
  // function, rather than that one rounded a hardware approximation
  // differently.
  LSE_EXPECT(d.worst_rel < 1e-5);
}

// A self-indexing primitive with a fused elementwise epilogue: the store hook
// runs in Loom, at the stored index, on the value still in register.
LSE_TEST(loom_matmul_with_an_epilogue_matches_hip) {
  auto* be = live_hrx();
  if (be == nullptr) {
    std::printf("       (skipped: no hrx device)\n");
    return;
  }
  ::unsetenv("LSE_WMMA");
  std::vector<float> xs(4 * 64);
  std::vector<float> ws(64 * 32);
  for (std::size_t i = 0; i < xs.size(); ++i) {
    xs[i] = static_cast<float>((i % 13)) * 0.125f - 0.5f;
  }
  for (std::size_t i = 0; i < ws.size(); ++i) {
    ws[i] = static_cast<float>((i % 7)) * 0.25f - 0.75f;
  }
  Array x = device_input(*be, Shape{4, 64}, xs);
  Array w = device_input(*be, Shape{64, 32}, ws);
  Array z = relu(matmul(x, w));
  const FusionGroup g = group_anchored(z, OpKind::kMatMul);
  LSE_EXPECT(!g.nodes.empty());
  if (g.nodes.empty()) return;

  const Diff d = differential(
      g, {{x.node().get(), xs}, {w.node().get(), ws}}, "matmul+relu");
  LSE_EXPECT(d.ran);
  LSE_EXPECT(d.nonzero > 0u);
  LSE_EXPECT_EQ(d.mismatched, 0u);
}

// Both toolchains must choose the same launch for the same group, or a
// differential is comparing two different kernels. choose_dims is duplicated
// between the two emitters until their shared half is extracted; this is what
// keeps the copies honest.
LSE_TEST(both_emitters_choose_the_same_launch) {
  for (std::int64_t n : {std::int64_t{1}, std::int64_t{63}, std::int64_t{64},
                         std::int64_t{4096}, std::int64_t{1 << 20}}) {
    Array a = Array::zeros(Shape{n}, DType::kF32);
    Array b = Array::zeros(Shape{n}, DType::kF32);
    Array z = a * b;
    const FusionGroup g = group_anchored(z, OpKind::kMul);
    if (g.nodes.empty()) continue;
    auto hip = kEmitter.emit(g, gfx1151());
    auto loom = kLoomEmitter.emit(g, gfx1151());
    LSE_EXPECT(hip.ok() && loom.ok());
    if (!hip.ok() || !loom.ok()) continue;
    for (int k = 0; k < 3; ++k) {
      LSE_EXPECT_EQ(loom->dims.workgroup_size[k], hip->dims.workgroup_size[k]);
      LSE_EXPECT_EQ(loom->dims.workgroup_count[k],
                    hip->dims.workgroup_count[k]);
    }
    LSE_EXPECT_EQ(loom->dims.subgroup_size, hip->dims.subgroup_size);
  }
}

// Every kernel primitive in the tree, at a representative shape, emitted both
// ways and RUN both ways. Coverage is a measurement: what declines prints the
// reason, and what emits has to produce the same bytes as the HIP path or the
// test fails. Nothing here is asserted from the source text.
LSE_TEST(loom_differential_over_every_kernel_primitive) {
  auto* be = live_hrx();
  if (be == nullptr || !kLoom.available()) {
    std::printf("       (skipped: no hrx device or no loomc)\n");
    return;
  }

  auto ramp = [](std::size_t n, float scale, float bias) {
    std::vector<float> v(n);
    for (std::size_t i = 0; i < n; ++i) {
      v[i] = static_cast<float>(i % 17) * scale + bias;
    }
    return v;
  };

  // The group that produced `root`, whichever op the partitioner anchored it
  // on — the anchor is the partitioner's business, not this test's.
  auto group_for = [](Array& root) {
    const NodePtr roots[] = {root.node()};
    FusionGroup found;
    for (const FusionGroup& g : Partitioner::partition(roots)) {
      for (const NodePtr& o : g.outputs) {
        if (o.get() == root.node().get()) found = g;
      }
    }
    return found;
  };

  struct Case {
    const char* name;
    FusionGroup group;
    std::vector<std::pair<const Node*, std::vector<float>>> host;
  };
  std::vector<Case> cases;

  auto one_in = [&](const char* name, Shape shape, auto&& build) {
    const std::vector<float> h = ramp(shape.elem_count(), 0.125f, -1.0f);
    Array in = device_input(*be, shape, h);
    Array root = build(in);
    cases.push_back(Case{name, group_for(root), {{in.node().get(), h}}});
  };
  auto two_in = [&](const char* name, Shape a, Shape b, auto&& build) {
    const std::vector<float> ha = ramp(a.elem_count(), 0.125f, -1.0f);
    const std::vector<float> hb = ramp(b.elem_count(), 0.25f, -0.5f);
    Array x = device_input(*be, a, ha);
    Array y = device_input(*be, b, hb);
    Array root = build(x, y);
    cases.push_back(Case{name,
                         group_for(root),
                         {{x.node().get(), ha}, {y.node().get(), hb}}});
  };

  ::unsetenv("LSE_WMMA");
  two_in("elementwise", Shape{256}, Shape{256},
         [](Array& a, Array& b) { return a * b + a; });
  two_in("matmul", Shape{4, 64}, Shape{64, 32},
         [](Array& a, Array& b) { return matmul(a, b); });
  two_in("linear", Shape{4, 64}, Shape{32, 64},
         [](Array& a, Array& b) { return linear(a, b); });
  one_in("transpose", Shape{4, 64},
         [](Array& a) { return transpose(a, {1, 0}); });
  one_in("slice", Shape{4, 64}, [](Array& a) { return slice(a, 1, 8, 40); });
  two_in("concat", Shape{4, 64}, Shape{2, 64},
         [](Array& a, Array& b) { return concat({a, b}, 0); });
  one_in("sum", Shape{4, 64}, [](Array& a) { return sum(a, 1); });
  one_in("max", Shape{4, 64}, [](Array& a) { return max(a, 1); });
  one_in("softmax", Shape{4, 64}, [](Array& a) { return softmax(a, 1); });
  one_in("silu", Shape{256}, [](Array& a) { return silu(a); });
  one_in("gelu", Shape{256}, [](Array& a) { return gelu(a); });
  one_in("argmax", Shape{4, 64}, [](Array& a) { return argmax(a); });
  two_in("rms_norm", Shape{4, 64}, Shape{64},
         [](Array& a, Array& b) { return rms_norm(a, b, 1e-5f); });

  // Cases whose operand list is not one or two plain f32 tensors.
  {
    const std::vector<float> hw = ramp(32 * 64, 0.0625f, -0.25f);
    const std::vector<float> hr = {3.0f, 0.0f, 31.0f, 12.0f};
    Array w = device_input(*be, Shape{32, 64}, hw);
    Array rows = device_input(*be, Shape{4}, hr);
    Array root = gather_rows(w, rows);
    cases.push_back(Case{"gather_rows",
                         group_for(root),
                         {{w.node().get(), hw}, {rows.node().get(), hr}}});
  }
  {
    const std::vector<float> hx = ramp(4 * 64, 0.125f, -1.0f);
    const std::vector<float> ht = ramp(4 * 64, 0.03125f, 0.5f);
    Array x = device_input(*be, Shape{4, 64}, hx);
    Array cs = device_input(*be, Shape{4, 64}, ht);
    Array sn = device_input(*be, Shape{4, 64}, ht);
    Array root = rope(x, cs, sn, 0);
    cases.push_back(Case{"rope",
                         group_for(root),
                         {{x.node().get(), hx},
                          {cs.node().get(), ht},
                          {sn.node().get(), ht}}});
  }
  {
    const std::vector<float> hd = ramp(64 * 32, 0.5f, 0.0f);
    const std::vector<float> hs = ramp(32, 2.0f, 1.0f);
    const std::vector<float> hb = {7.0f};
    Array dst = device_input(*be, Shape{1, 64, 32}, hd);
    Array src = device_input(*be, Shape{1, 1, 32}, hs);
    Array begin = device_input(*be, Shape{1}, hb);
    Array root = overwrite_slice(dst, src, 1, begin);
    cases.push_back(Case{"overwrite_slice",
                         group_for(root),
                         {{dst.node().get(), hd},
                          {src.node().get(), hs},
                          {begin.node().get(), hb}}});
  }

  std::size_t emitted = 0;
  std::size_t exact = 0;
  std::size_t declined = 0;
  for (const Case& c : cases) {
    if (c.group.nodes.empty()) {
      std::printf("       %-16s no group\n", c.name);
      continue;
    }
    auto e = kLoomEmitter.emit(c.group, be->device_info());
    if (!e.ok()) {
      ++declined;
      std::printf("       %-16s DECLINES: %s\n", c.name,
                  e.status().message().c_str());
      continue;
    }
    ++emitted;
    const Diff d = differential(c.group, c.host, c.name);
    LSE_EXPECT(d.ran);
    if (d.ran && d.mismatched == 0) ++exact;
    // Bit-exact everywhere the two dialects spell the same instructions. The
    // exception is the transcendental rows: HIP's `tanhf` is libm's and Loom's
    // is `scalar.tanhf<afn>`, the hardware approximation, because the exact
    // f32 form does not lower on this target at all. One ULP, measured, not
    // assumed — anything larger means the rows are not the same function.
    LSE_EXPECT(d.worst_rel < 2e-7);
  }
  std::printf("       -- %zu emit (%zu bit-exact against hip), %zu decline, "
              "%zu of %zu cases\n",
              emitted, exact, declined, emitted + declined, cases.size());
  LSE_EXPECT(emitted > 0);
}

// Compile time, the one number this dialect exists for. Same process, same
// group, the two toolchains interleaved so a thermal or scheduling drift lands
// on both, medians of 24. This is a COMPILER measurement: it says nothing
// about how fast the resulting kernel runs.
LSE_TEST(loom_compiles_the_same_group_far_faster_than_comgr) {
  auto* be = live_hrx();
  if (be == nullptr || !kLoom.available() || !kCompiler.available()) {
    std::printf("       (skipped: no hrx device or no loomc)\n");
    return;
  }
  const backend::DeviceInfo& dev = be->device_info();
  const std::string arch(dev.arch);

  ::unsetenv("LSE_WMMA");
  std::vector<float> xs(4 * 64, 1.0f);
  std::vector<float> ws(64 * 32, 0.5f);
  Array x = device_input(*be, Shape{4, 64}, xs);
  Array w = device_input(*be, Shape{64, 32}, ws);
  Array z = relu(matmul(x, w));
  const NodePtr roots[] = {z.node()};
  FusionGroup g;
  for (const FusionGroup& c : Partitioner::partition(roots)) {
    for (const NodePtr& o : c.outputs) {
      if (o.get() == z.node().get()) g = c;
    }
  }
  if (g.nodes.empty()) return;

  auto hip = kEmitter.emit(g, dev);
  auto loom = kLoomEmitter.emit(g, dev);
  LSE_EXPECT(hip.ok() && loom.ok());
  if (!hip.ok() || !loom.ok()) return;

  auto median = [](std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.empty() ? 0.0 : v[v.size() / 2];
  };
  auto time_one = [&](const graph::IKernelCompiler& cc, const std::string& src) {
    const auto t0 = std::chrono::steady_clock::now();
    auto r = cc.compile(src, arch);
    const auto t1 = std::chrono::steady_clock::now();
    LSE_EXPECT(r.ok());
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
  };

  const double cold_loom = time_one(kLoom, loom->source);
  const double cold_hip = time_one(kCompiler, hip->source);

  std::vector<double> loom_ms;
  std::vector<double> hip_ms;
  for (int i = 0; i < 24; ++i) {
    loom_ms.push_back(time_one(kLoom, loom->source));
    hip_ms.push_back(time_one(kCompiler, hip->source));
  }
  const double lm = median(loom_ms);
  const double hm = median(hip_ms);
  std::printf("       matmul+relu on %s, medians of 24, interleaved:\n"
              "         loomc  cold %.2f ms  steady %.3f ms\n"
              "         comgr  cold %.2f ms  steady %.3f ms\n"
              "         ratio  %.1fx\n",
              arch.c_str(), cold_loom, lm, cold_hip, hm,
              lm > 0.0 ? hm / lm : 0.0);
  LSE_EXPECT(lm > 0.0 && hm > 0.0);
}

LSE_TEST(the_device_publishes_its_capacity_facts) {
  backend::IBackend* be = live_hrx();
  if (be == nullptr) {
    std::printf("       (skipped: no HRX device)\n");
    return;
  }
  const backend::DeviceInfo& info = be->device_info();
  const backend::ArchFacts& f = info.arch_facts;
  // A device the engine can see must publish an addressable LDS budget one way
  // or another; the register facts need a compiler and may be absent.
  LSE_EXPECT(f.lds_bytes_addressable_per_workgroup.known());
  LSE_EXPECT(f.max_flat_workgroup_size.known());
  if (kCompiler.available()) {
    LSE_EXPECT(f.vector_registers_per_simd.known());
    LSE_EXPECT(f.vector_register_alloc_granule.known());
    LSE_EXPECT(f.vector_registers_per_simd.source ==
               backend::FactSource::kQueried);
  }
  // The runtime's own LDS answer and the compiler's must not disagree.
  if (info.lds_bytes_per_workgroup != 0 &&
      f.lds_bytes_addressable_per_workgroup.known()) {
    LSE_EXPECT_EQ(f.lds_bytes_addressable_per_workgroup.value,
                  info.lds_bytes_per_workgroup);
  }
  std::printf("       %s\n%s", info.arch.c_str(), f.describe().c_str());
}
