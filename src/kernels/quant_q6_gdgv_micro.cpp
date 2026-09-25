// TEST-ONLY decode GEMV geometry variants for the PP512 decode-TPS microbench
// (cand-decode-gemv-micro). No production selector references any symbol in
// this translation unit: the five primitives are reached exclusively through
// the micro fixture's explicit `prim` assignment on its synthetic
// quant_linear-shaped nodes. `git status` of the candidate tree shows this as
// the only candidate addition.
//
// Variant table (production baseline = `emit_q6_decode_quad`,
// quant_linear.cpp:887-949 at the pin, selected by `q6_decode_columns`
// :335-343, planned by `gemv_plan` :179-197):
//
//   q6_gdgv_baseline   verbatim copy of emit_q6_decode_quad. 4 cols/wave,
//                      256-thread WGs, LDS x-panel + barrier when it fits,
//                      3 scalar u32 code loads per chunk, shfl_xor x5,
//                      lane 0 stores 4 outputs. Timed as the in-bench
//                      baseline.
//   q6_gdgv_8col       8 cols/wave instead of 4: half the workgroups (grid
//                      N/64), each wave owns 8 output columns and doubles
//                      the per-chunk scale/bias fetches and code loads.
//                      Hypothesis: fewer, fatter workgroups keep more
//                      weight-stream loads in flight; the cost question is
//                      24 accumulators + 8x(3 u32 + 2 f32) live registers.
//   q6_gdgv_ksplit     each wave accumulates two HALVES of K in separate
//                      accumulators (K=5120: two 256-chunk halves) and adds
//                      them before the 5-step shfl_xor wave reduce. Same
//                      grid as baseline. Doubles the live accumulators and
//                      the scale/bias traffic; the per-column fma order
//                      within each half is unchanged, so this re-associates
//                      the K sum (arithmetic-different, see build report).
//   q6_gdgv_nopanel    baseline schedule, but the x-panel is ALWAYS read
//                      from global memory: no LDS, no barrier, no stage
//                      loop. The in-kernel form of the LSE_LDS_BUDGET=0
//                      probe: at 544 WGs the 20 KB per-WG scratch plus its
//                      serial stage+barrier is suspected to cap residency
//                      (64 KB per-CU budget -> 3 WGs/CU, below the 8.5 the
//                      grid wants) and to serialize the front of every WG.
//
// Grids (M=1, all 256-thread WGs, workgroup_count[1] = 1):
//   baseline  / ksplit / nopanel : ceil(N/32)
//   8col                          : ceil(N/64)
// N=17408 -> 544 / 272; N=5120 -> 160 / 80; N=10240 -> 320 / 160;
// N=248320 -> 7760 / 3880.

#include <array>
#include <cstdint>
#include <string>
#include <string_view>

#include "lse/graph/kernel_args.hpp"
#include "lse/graph/kernel_primitive.hpp"
#include "lse/kernels/quant_panel.hpp"
#include "lse/math.hpp"
#include "lse/quant/group_affine.hpp"

namespace lse::kernels {
namespace {

namespace env = graph::env;
namespace kir = graph::kir;
using namespace graph;
using quant::GroupAffine;
using lse::math::fma;
using lse::math::local_id;
using lse::math::shfl_xor;
using lse::math::widen;
using lse::math::workgroup_id_x;

constexpr std::uint32_t kWave = 32;
constexpr std::uint32_t kBlock = 256;
constexpr std::uint32_t kVals = 16;    // 6-bit values_per_chunk
constexpr std::uint32_t kWords = 3;    // 6-bit words_per_chunk
constexpr std::uint32_t kGroup = 64;   // the only group size this bench tests
constexpr std::uint32_t kDeviceLds = 65536;  // gfx1201 workgroup LDS

// The 4-operand quant_linear contract: x [1, K] f32, packed [N, K*6/32] u32,
// scales and biases [N, K/64] bf16, iattrs [6, 64]. Mirrors the production
// `dims_of` in quant_linear.cpp for the M=1 shape only, which is the only
// shape the bench exists to measure.
struct GdgvDims {
  std::uint32_t n = 0, k = 0, lanes = 0, groups = 0;
  bool valid = false;
};

GdgvDims gdgv_dims_of(const KernelShapes& s) {
  GdgvDims d;
  if (s.inputs.size() != 4 || s.input_dtypes.size() < 4) return d;
  if (s.input_dtypes[0] != DType::kF32 || s.input_dtypes[1] != DType::kU32 ||
      s.input_dtypes[2] != DType::kBF16 || s.input_dtypes[3] != DType::kBF16)
    return d;
  if (s.inputs[1].rank() != 2 || s.inputs[2].rank() != 2 ||
      s.inputs[3].rank() != 2 || s.inputs[0].rank() < 1)
    return d;
  if (s.iattrs.size() < 2 || s.iattrs[0] != 6 || s.iattrs[1] != kGroup)
    return d;
  const std::int64_t n = s.inputs[1].dim(0);
  const std::int64_t lanes = s.inputs[1].dim(1);
  const std::int64_t k = s.inputs[0].dim(s.inputs[0].rank() - 1);
  const std::int64_t groups = s.inputs[2].dim(1);
  if (n <= 0 || k <= 0 || lanes <= 0 || groups <= 0) return d;
  if (lanes * 32 != k * 6) return d;
  if (groups * kGroup != k) return d;
  if (s.inputs[2].dim(0) != n || s.inputs[3].dim(0) != n) return d;
  if (static_cast<std::int64_t>(s.output.elem_count()) != n) return d;  // M must be 1
  d.n = static_cast<std::uint32_t>(n);
  d.k = static_cast<std::uint32_t>(k);
  d.lanes = static_cast<std::uint32_t>(lanes);
  d.groups = static_cast<std::uint32_t>(groups);
  d.valid = true;
  return d;
}

struct GdgvArgs {
  env::In<kir::f32, env::Emit> x;
  env::In<std::uint32_t, env::Emit> packed;
  env::In<lse::bf16, env::Emit> scales;
  env::In<lse::bf16, env::Emit> biases;
  env::Out<kir::f32, env::Emit> out;
};

// The 6-bit dequant of one code of a chunk, straight from the production
// body (quant_linear.cpp:931-938). `packed` holds the chunk's three u32
// words already loaded.
auto decode_code(const std::array<kir::Val<kir::u32>, kWords>& packed,
                 const GroupAffine& spec, int code, env::Emit& e) {
  const auto word = static_cast<std::size_t>(spec.chunk_word(code));
  const int offset = spec.chunk_bit(code);
  const int carry = spec.chunk_carry(code);
  auto value = packed[word] / (1u << offset);
  if (carry > 0)
    value = value +
            (packed[word + 1] % (1u << carry)) * (1u << (32 - offset));
  else
    value = value % 64u;
  return e.let(value);
}

// ---- variant 1: baseline, verbatim copy of emit_q6_decode_quad -----------
std::string emit_gdgv_baseline(const KernelShapes& s, const GdgvDims& d) {
  const auto n = d.n;
  const auto k = d.k;
  const auto lanes = d.lanes;
  const auto groups = d.groups;
  const GroupAffine spec = GroupAffine::make(6, kGroup).value_or(GroupAffine{});
  constexpr std::uint32_t columns = 4;
  const auto chunks = k / kVals;
  const auto aligned = chunks / kWave * kWave;
  kir::KernelBody kb(s.types, *s.intrinsics, kDeviceLds);
  kb.set_store(s.store);
  GdgvArgs a;
  if (!env::bind(kb, a, s)) return {};
  env::Emit e{&kb};
  const auto lid = e.let(local_id());
  const auto lane = e.let(lid % kWave);
  const auto col = e.let((workgroup_id_x() * (kBlock / kWave) +
                          lid / kWave) *
                         columns);
  kir::Tile<kir::f32> xs;
  if (e.lds_fits<kir::f32>(k)) {
    xs = e.lds<kir::f32>(k);
    for (auto at : e.range(lid, e.u32(k), kBlock))
      xs[e.let(q6_panel_index(at))] = a.x[at];
    e.barrier();
  }
  std::vector<kir::LValue<kir::f32>> acc{e.var(0.0f), e.var(0.0f),
                                          e.var(0.0f), e.var(0.0f)};
  if (auto live = e.when(col < n)) {
    auto accumulate = [&](const kir::Val<kir::u32>& chunk) {
      std::array<kir::Val<kir::f32>, columns> scales, biases;
      std::array<std::array<kir::Val<kir::u32>, kWords>, columns> packed;
      for (std::uint32_t output = 0; output < columns; ++output) {
        const auto current = e.let(col + output);
        const auto group = e.let(current * groups + chunk / 4u);
        scales[output] = e.let(widen(a.scales[group]));
        biases[output] = e.let(widen(a.biases[group]));
        const auto first = e.let(current * lanes + chunk * kWords);
        for (std::uint32_t word = 0; word < kWords; ++word)
          packed[output][word] = e.let(a.packed[first + word]);
      }
      const auto first_x = e.let(chunk * kVals);
      for (int code = 0; code < int(kVals); ++code) {
        const auto index = e.let(first_x + static_cast<std::uint32_t>(code));
        const auto x = e.let(xs ? xs[e.let(q6_panel_index(index))].read()
                                : a.x[index]);
        for (std::uint32_t output = 0; output < columns; ++output) {
          const auto decoded = decode_code(packed[output], spec, code, e);
          const auto weight = fma(cast<kir::f32>(decoded),
                                        scales[output], biases[output]);
          acc[output] = fma(x, weight, acc[output].read());
        }
      }
    };
    for (auto c0 : e.range(0u, aligned, kWave)) accumulate(e.let(c0 + lane));
    if (aligned < chunks)
      for (auto chunk :
           e.range(e.u32(aligned) + lane, e.u32(chunks), kWave))
        accumulate(chunk);
  }
  for (std::uint32_t output = 0; output < columns; ++output) {
    for (std::uint32_t bit = 1; bit < kWave; bit <<= 1)
      acc[output] = acc[output].read() +
                    shfl_xor(acc[output].read(), e.u32(bit));
    if (auto writer = e.when(lane == 0u && col < n))
      e.store(col + output, acc[output].read());
  }
  return kb.lds().ok() ? kb.str() : std::string{};
}

// ---- variant 2: 8 columns per wave ---------------------------------------
std::string emit_gdgv_8col(const KernelShapes& s, const GdgvDims& d) {
  const auto n = d.n;
  const auto k = d.k;
  const auto lanes = d.lanes;
  const auto groups = d.groups;
  const GroupAffine spec = GroupAffine::make(6, kGroup).value_or(GroupAffine{});
  constexpr std::uint32_t columns = 8;
  const auto chunks = k / kVals;
  const auto aligned = chunks / kWave * kWave;
  kir::KernelBody kb(s.types, *s.intrinsics, kDeviceLds);
  kb.set_store(s.store);
  GdgvArgs a;
  if (!env::bind(kb, a, s)) return {};
  env::Emit e{&kb};
  const auto lid = e.let(local_id());
  const auto lane = e.let(lid % kWave);
  // 8 columns per wave: each WG owns 8*8 = 64 output columns. The
  // wave base (gid*8 + lid/32) is multiplied by the full column count
  // so that WG g covers columns [g*64, g*64+64) and wave w within the
  // WG covers the sub-range [g*64 + w*8, g*64 + w*8 + 8). The 8col
  // variant's prebuild had col = gid*8 + lid/32 (missing the *8),
  // which collapsed the wave sub-ranges into the same 8-column window
  // and produced out-of-bounds reads (NaN at output[2183] on the first
  // run). The 4-col baseline's col = (gid*8 + lid/32) * 4 is the
  // correct pattern generalised to 8 cols: *8 instead of *4.
  const auto col = e.let((workgroup_id_x() * (kBlock / kWave) +
                          lid / kWave) *
                         columns);
  kir::Tile<kir::f32> xs;
  if (e.lds_fits<kir::f32>(k)) {
    xs = e.lds<kir::f32>(k);
    for (auto at : e.range(lid, e.u32(k), kBlock))
      xs[e.let(q6_panel_index(at))] = a.x[at];
    e.barrier();
  }
  std::vector<kir::LValue<kir::f32>> acc;
  acc.reserve(columns);
  for (std::uint32_t i = 0; i < columns; ++i) acc.push_back(e.var(0.0f));
  if (auto live = e.when(col < n)) {
    auto accumulate = [&](const kir::Val<kir::u32>& chunk) {
      std::array<kir::Val<kir::f32>, columns> scales, biases;
      std::array<std::array<kir::Val<kir::u32>, kWords>, columns> packed;
      for (std::uint32_t output = 0; output < columns; ++output) {
        const auto current = e.let(col + output);
        const auto group = e.let(current * groups + chunk / 4u);
        scales[output] = e.let(widen(a.scales[group]));
        biases[output] = e.let(widen(a.biases[group]));
        const auto first = e.let(current * lanes + chunk * kWords);
        for (std::uint32_t word = 0; word < kWords; ++word)
          packed[output][word] = e.let(a.packed[first + word]);
      }
      const auto first_x = e.let(chunk * kVals);
      for (int code = 0; code < int(kVals); ++code) {
        const auto index = e.let(first_x + static_cast<std::uint32_t>(code));
        const auto x = e.let(xs ? xs[e.let(q6_panel_index(index))].read()
                                : a.x[index]);
        for (std::uint32_t output = 0; output < columns; ++output) {
          const auto decoded = decode_code(packed[output], spec, code, e);
          const auto weight = fma(cast<kir::f32>(decoded),
                                        scales[output], biases[output]);
          acc[output] = fma(x, weight, acc[output].read());
        }
      }
    };
    for (auto c0 : e.range(0u, aligned, kWave)) accumulate(e.let(c0 + lane));
    if (aligned < chunks)
      for (auto chunk :
           e.range(e.u32(aligned) + lane, e.u32(chunks), kWave))
        accumulate(chunk);
  }
  for (std::uint32_t output = 0; output < columns; ++output) {
    for (std::uint32_t bit = 1; bit < kWave; bit <<= 1)
      acc[output] = acc[output].read() +
                    shfl_xor(acc[output].read(), e.u32(bit));
    if (auto writer = e.when(lane == 0u && col < n))
      e.store(col + output, acc[output].read());
  }
  return kb.lds().ok() ? kb.str() : std::string{};
}

// ---- variant 3: K split across the wave, 2 halves per wave ---------------
// Same grid as baseline. Each wave owns the full K but walks it in two
// halves of K/2 values (256 chunks each for K=5120); the half sums live in
// separate per-column accumulators and are added before the shfl_xor wave
// reduce. Within each half the per-column fma order is unchanged, so the
// variant re-associates the K sum (see the build report's arithmetic flag).
std::string emit_gdgv_ksplit(const KernelShapes& s, const GdgvDims& d) {
  const auto n = d.n;
  const auto k = d.k;
  const auto lanes = d.lanes;
  const auto groups = d.groups;
  const GroupAffine spec = GroupAffine::make(6, kGroup).value_or(GroupAffine{});
  constexpr std::uint32_t columns = 4;
  constexpr std::uint32_t halves = 2;
  const auto chunks = k / kVals;
  const auto half_chunks = k / 2 / kVals;  // 256 for K=5120
  kir::KernelBody kb(s.types, *s.intrinsics, kDeviceLds);
  kb.set_store(s.store);
  GdgvArgs a;
  if (!env::bind(kb, a, s)) return {};
  env::Emit e{&kb};
  const auto lid = e.let(local_id());
  const auto lane = e.let(lid % kWave);
  const auto col = e.let((workgroup_id_x() * (kBlock / kWave) +
                          lid / kWave) *
                         columns);
  kir::Tile<kir::f32> xs;
  if (e.lds_fits<kir::f32>(k)) {
    xs = e.lds<kir::f32>(k);
    for (auto at : e.range(lid, e.u32(k), kBlock))
      xs[e.let(q6_panel_index(at))] = a.x[at];
    e.barrier();
  }
  std::vector<kir::LValue<kir::f32>> acc[halves];
  for (std::uint32_t h = 0; h < halves; ++h) {
    acc[h].reserve(columns);
    for (std::uint32_t c = 0; c < columns; ++c) acc[h].push_back(e.var(0.0f));
  }
  if (auto live = e.when(col < n)) {
    auto accumulate = [&](std::uint32_t h, const kir::Val<kir::u32>& chunk) {
      const auto absolute = e.let(h * half_chunks + chunk);
      std::array<kir::Val<kir::f32>, columns> scales, biases;
      std::array<std::array<kir::Val<kir::u32>, kWords>, columns> packed;
      for (std::uint32_t output = 0; output < columns; ++output) {
        const auto current = e.let(col + output);
        const auto group = e.let(current * groups + absolute / 4u);
        scales[output] = e.let(widen(a.scales[group]));
        biases[output] = e.let(widen(a.biases[group]));
        const auto first = e.let(current * lanes + absolute * kWords);
        for (std::uint32_t word = 0; word < kWords; ++word)
          packed[output][word] = e.let(a.packed[first + word]);
      }
      const auto first_x = e.let(absolute * kVals);
      for (int code = 0; code < int(kVals); ++code) {
        const auto index = e.let(first_x + static_cast<std::uint32_t>(code));
        const auto x = e.let(xs ? xs[e.let(q6_panel_index(index))].read()
                                : a.x[index]);
        for (std::uint32_t output = 0; output < columns; ++output) {
          const auto decoded = decode_code(packed[output], spec, code, e);
          const auto weight = fma(cast<kir::f32>(decoded),
                                        scales[output], biases[output]);
          acc[h][output] = fma(x, weight, acc[h][output].read());
        }
      }
    };
    for (std::uint32_t h = 0; h < halves; ++h) {
      // One chunk per lane in stride-32 order: the aligned run of 256
      // chunks per half is exactly half_chunks / kWave iterations of 32
      // lanes (256 % 32 == 0), so the loop is straight-line with no tail.
      for (auto c : e.range(e.u32(h * half_chunks),
                            e.u32((h + 1) * half_chunks), kWave))
        accumulate(h, e.let(c - h * half_chunks + lane));
    }
  }
  (void)chunks;
  for (std::uint32_t output = 0; output < columns; ++output) {
    auto sum = acc[0][output].read();
    for (std::uint32_t h = 1; h < halves; ++h)
      sum = sum + acc[h][output].read();
    for (std::uint32_t bit = 1; bit < kWave; bit <<= 1)
      sum = sum + shfl_xor(sum, e.u32(bit));
    if (auto writer = e.when(lane == 0u && col < n))
      e.store(col + output, sum);
  }
  return kb.lds().ok() ? kb.str() : std::string{};
}

// ---- variant 4: baseline schedule, no LDS x-panel -------------------------
std::string emit_gdgv_nopanel(const KernelShapes& s, const GdgvDims& d) {
  const auto n = d.n;
  const auto k = d.k;
  const auto lanes = d.lanes;
  const auto groups = d.groups;
  const GroupAffine spec = GroupAffine::make(6, kGroup).value_or(GroupAffine{});
  constexpr std::uint32_t columns = 4;
  const auto chunks = k / kVals;
  const auto aligned = chunks / kWave * kWave;
  kir::KernelBody kb(s.types, *s.intrinsics, kDeviceLds);
  kb.set_store(s.store);
  GdgvArgs a;
  if (!env::bind(kb, a, s)) return {};
  env::Emit e{&kb};
  const auto lid = e.let(local_id());
  const auto lane = e.let(lid % kWave);
  const auto col = e.let((workgroup_id_x() * (kBlock / kWave) +
                          lid / kWave) *
                         columns);
  std::vector<kir::LValue<kir::f32>> acc{e.var(0.0f), e.var(0.0f),
                                          e.var(0.0f), e.var(0.0f)};
  if (auto live = e.when(col < n)) {
    auto accumulate = [&](const kir::Val<kir::u32>& chunk) {
      std::array<kir::Val<kir::f32>, columns> scales, biases;
      std::array<std::array<kir::Val<kir::u32>, kWords>, columns> packed;
      for (std::uint32_t output = 0; output < columns; ++output) {
        const auto current = e.let(col + output);
        const auto group = e.let(current * groups + chunk / 4u);
        scales[output] = e.let(widen(a.scales[group]));
        biases[output] = e.let(widen(a.biases[group]));
        const auto first = e.let(current * lanes + chunk * kWords);
        for (std::uint32_t word = 0; word < kWords; ++word)
          packed[output][word] = e.let(a.packed[first + word]);
      }
      const auto first_x = e.let(chunk * kVals);
      for (int code = 0; code < int(kVals); ++code) {
        const auto index = e.let(first_x + static_cast<std::uint32_t>(code));
        const auto x = e.let(a.x[index]);
        for (std::uint32_t output = 0; output < columns; ++output) {
          const auto decoded = decode_code(packed[output], spec, code, e);
          const auto weight = fma(cast<kir::f32>(decoded),
                                        scales[output], biases[output]);
          acc[output] = fma(x, weight, acc[output].read());
        }
      }
    };
    for (auto c0 : e.range(0u, aligned, kWave)) accumulate(e.let(c0 + lane));
    if (aligned < chunks)
      for (auto chunk :
           e.range(e.u32(aligned) + lane, e.u32(chunks), kWave))
        accumulate(chunk);
  }
  for (std::uint32_t output = 0; output < columns; ++output) {
    for (std::uint32_t bit = 1; bit < kWave; bit <<= 1)
      acc[output] = acc[output].read() +
                    shfl_xor(acc[output].read(), e.u32(bit));
    if (auto writer = e.when(lane == 0u && col < n))
      e.store(col + output, acc[output].read());
  }
  return kb.lds().ok() ? kb.str() : std::string{};
}

// Dispatch on entry name so every variant class shares the emitters and the
// plan helper below (the plan and the body must agree exactly, as in the
// production `gemv_plan` / `body_lds_bytes_at` pair).
std::string gdgv_emit(const char* entry, const KernelShapes& s,
                      const GdgvDims& d) {
  if (!d.valid || s.types.scalar == nullptr || !s.store ||
      s.intrinsics == nullptr)
    return {};
  if (std::string_view(entry) == "lse_q6_gdgv_baseline")
    return emit_gdgv_baseline(s, d);
  if (std::string_view(entry) == "lse_q6_gdgv_8col") return emit_gdgv_8col(s, d);
  if (std::string_view(entry) == "lse_q6_gdgv_ksplit")
    return emit_gdgv_ksplit(s, d);
  if (std::string_view(entry) == "lse_q6_gdgv_nopanel")
    return emit_gdgv_nopanel(s, d);
  return {};
}

ThreadPlan gdgv_plan(const char* entry, const GdgvDims& d) {
  ThreadPlan tp;
  tp.workgroup_size[0] = kBlock;
  tp.workgroup_size[1] = 1;
  tp.workgroup_size[2] = 1;
  tp.workgroup_count[0] = 1;
  tp.workgroup_count[1] = 1;
  tp.workgroup_count[2] = 1;
  tp.lds_bytes = 0;
  if (!d.valid) return tp;
  // The production `gemv_plan` (quant_linear.cpp:179-197) computes
  //   workgroup_count[0] = (n + waves - 1) / waves
  // with waves = kBlock/wave * cols = 8 * cols. 4 cols -> 32 cols/WG ->
  // ceil(N/32); 8 cols -> 64 cols/WG -> ceil(N/64). The unparenthesized
  // `(n + 31) / 32 * m`-style rewrites are the documented precedence trap,
  // so the single ceil factor stays explicit here.
  if (std::string_view(entry) == "lse_q6_gdgv_8col")
    tp.workgroup_count[0] = static_cast<std::uint32_t>((d.n + 64 - 1) / 64);
  else
    tp.workgroup_count[0] = static_cast<std::uint32_t>((d.n + 32 - 1) / 32);
  // LDS exactly as the body declares it: the panel exists in baseline/8col/
  // ksplit only when K*4 fits the per-CU budget (K=17408 -> 69,632 B does
  // NOT fit, so those variants run panel-less there, exactly like the
  // production `lds_fits` arm). The no-panel variant declares none.
  const bool has_panel = std::string_view(entry) == "lse_q6_gdgv_baseline" ||
                         std::string_view(entry) == "lse_q6_gdgv_8col" ||
                         std::string_view(entry) == "lse_q6_gdgv_ksplit";
  if (has_panel && d.k * 4u <= kDeviceLds) tp.lds_bytes = d.k * 4u;
  return tp;
}

}  // namespace

// The four test-only primitives. They are reachable only through the micro
// fixture's explicit `prim` assignment: no production selector (the
// `quant_linear` KernelPrimitive's `specialize`) names any of these.
#define LSE_GDGV_PRIMITIVE(ClassName, VariantName, EntryName)     \
  struct ClassName final : KernelPrimitive<ClassName> {           \
    static constexpr std::string_view kName = VariantName;        \
    static constexpr std::string_view kEntry = EntryName;         \
    static constexpr std::string_view kSource = {};               \
    std::size_t arity() const noexcept override { return 4; }     \
    bool owns_indexing() const noexcept override { return true; } \
    std::string emit_kernel(const KernelShapes& s) const override \
        {                                                         \
      return gdgv_emit(kEntry.data(), s,        \
                                       gdgv_dims_of(s)); \
    }                                                             \
    Result<Shape> infer_shape(std::span<const Shape> in) const    \
        override {                                                \
      if (in.size() != 4 || in[1].rank() != 2)                    \
        return LSE_ERROR(kInvalidArgument, "gdgv variant takes " \
                                           "x, packed, scales, " \
                                           "biases");             \
      Shape out;                                                  \
      for (std::size_t i = 0; i + 1 < in[0].rank(); ++i)          \
        out.push_back(in[0].dim(i));                              \
      out.push_back(in[1].dim(0));                                \
      return out;                                                 \
    }                                                             \
    DType infer_dtype(std::span<const DType>) const override {    \
      return DType::kF32;                                         \
    }                                                             \
    static ThreadPlan plan_impl(const KernelShapes& s) {          \
      return gdgv_plan(kEntry.data(),            \
                                       gdgv_dims_of(s)); \
    }                                                             \
  };                                                              \
  LSE_REGISTER_PRIMITIVE(ClassName)

LSE_GDGV_PRIMITIVE(GdgvBaseline, "quant_linear.q6_gdgv_baseline",
                   "lse_q6_gdgv_baseline");
LSE_GDGV_PRIMITIVE(GdgvEightCol, "quant_linear.q6_gdgv_8col",
                   "lse_q6_gdgv_8col");
LSE_GDGV_PRIMITIVE(GdgvKSplit, "quant_linear.q6_gdgv_ksplit",
                   "lse_q6_gdgv_ksplit");
LSE_GDGV_PRIMITIVE(GdgvNoPanel, "quant_linear.q6_gdgv_nopanel",
                   "lse_q6_gdgv_nopanel");

}  // namespace lse::kernels
