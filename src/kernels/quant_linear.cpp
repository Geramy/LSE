#include <algorithm>
#include <array>
#include <cstdlib>
#include <span>
#include <string>
#include <type_traits>

#include "lse/backends/hrx/device_info.hpp"
#include "lse/kernels/lds_linear.hpp"
#include "lse/dispatch/quant.hpp"
#include "lse/dispatch/q8_matrix.hpp"
#include "lse/kernels/wmma_q8_linear.hpp"
#include "lse/kernels/quant_panel.hpp"
#include "lse/kernels/vec_mem.hpp"
#include "lse/kernels/wmma.hpp"
#include "lse/graph/kernel_args.hpp"
#include "lse/graph/kernel_env.hpp"
#include "lse/graph/kernel_primitive.hpp"
#include "lse/math.hpp"
#include "lse/opt/arrangement.hpp"
#include "lse/quant/group_affine_codec.hpp"

namespace lse::kernels {

// Forward declaration: the Q6/WGMMA selector lives in wmma_q6_linear.cpp
// (same target, lse_kernels), which defines it after the anonymous namespace.
const graph::KernelPrimitiveBase *wmma_q6_linear_for(const graph::KernelShapes &);

// These name device facts, which the backend supplies.
using backend::AmdDeviceInfo;
using backend::DeviceInfo;
using backend::device_extension;
using backend::workgroup_lds_bytes;

using namespace lse::graph;
namespace math = lse::math;

namespace {

constexpr std::uint32_t kBlock = 256;
// Codes decoded per lane iteration. Past this the straight-line block stops
// paying for itself in registers.
constexpr std::uint32_t kMaxUnrolledCodes = 32;

std::uint32_t wave_of(const DeviceInfo* device) {
  if (device == nullptr) return 32;
  const std::uint32_t wave = device->wavefront_size;
  return (wave == 32 || wave == 64) ? wave : 32u;
}

struct QuantDims {
  quant::GroupAffine spec{};
  std::int64_t m = 0;
  std::int64_t n = 0;
  std::int64_t k = 0;
  std::int64_t lanes = 0;   // packed u32 per weight row
  std::int64_t groups = 0;  // scale/bias entries per weight row
  std::int64_t experts = 0;  // 0 when the weight is a single matrix
  std::uint32_t keep = 0;    // width of the index row
  std::uint32_t slot = 0;
  bool valid = false;
};
QuantDims dims_of(const KernelShapes& s, bool indexed) {
  QuantDims d;
  const std::size_t want = indexed ? 5u : 4u;
  if (s.inputs.size() != want || s.input_dtypes.size() < want) return d;
  if (s.input_dtypes[1] != DType::kU32) return d;
  if (s.input_dtypes[2] != s.input_dtypes[3]) return d;
  if (indexed && s.input_dtypes[4] != DType::kF32) return d;
  auto spec = indexed ? quant::GroupAffine::make(s.iattrs[1], s.iattrs[2])
                      : quant::GroupAffine::make(s.iattrs[0], s.iattrs[1]);
  if (!spec.ok()) return d;
  d.spec = *spec;

  const std::size_t rank = indexed ? 3u : 2u;
  const Shape& x = s.inputs[0];
  const Shape& w = s.inputs[1];
  const Shape& sc = s.inputs[2];
  const Shape& bi = s.inputs[3];
  if (x.rank() == 0 || w.rank() != rank || sc.rank() != rank ||
      bi.rank() != rank) {
    return d;
  }
  d.k = x.dim(x.rank() - 1);
  if (indexed) {
    d.experts = w.dim(0);
    d.n = w.dim(1);
    d.lanes = w.dim(2);
    d.groups = sc.dim(2);
    if (d.experts <= 0) return d;
    if (sc.dim(0) != d.experts || bi.dim(0) != d.experts) return d;
    if (sc.dim(1) != d.n || bi.dim(1) != d.n || bi.dim(2) != d.groups) return d;
  } else {
    d.n = w.dim(0);
    d.lanes = w.dim(1);
    d.groups = sc.dim(1);
    if (sc.dim(0) != d.n || bi.dim(0) != d.n || bi.dim(1) != d.groups) return d;
  }
  if (d.k <= 0 || d.n <= 0) return d;
  // The plane must describe exactly this K at exactly this bit width; a shape
  // that solves to another width is a different tensor, not a slower path.
  if (d.lanes * 32 != d.k * d.spec.bits) return d;
  if (d.groups * d.spec.group_size != d.k) return d;

  const auto elems = static_cast<std::int64_t>(s.output.elem_count());
  if (elems <= 0 || elems % d.n != 0) return d;
  d.m = elems / d.n;
  if (d.m <= 0) return d;

  if (indexed) {
    const Shape& ix = s.inputs[4];
    if (ix.rank() == 0) return d;
    d.keep = static_cast<std::uint32_t>(ix.dim(ix.rank() - 1));
    d.slot = static_cast<std::uint32_t>(s.iattrs[0]);
    if (d.keep == 0 || d.slot >= d.keep) return d;
    // One index row per output row, or `row * keep` walks off the end of a
    // buffer whose extent nothing else here constrains.
    if (static_cast<std::int64_t>(ix.elem_count()) != d.m * d.keep) return d;
    const std::int64_t plane_elems = d.experts * d.n * d.lanes;
    if (plane_elems > static_cast<std::int64_t>(0xffffffffll)) return d;
  }
  d.valid = true;
  return d;
}

bool device_fits(const KernelShapes& s) {
  if (s.device == nullptr || s.intrinsics == nullptr) return false;
  if (s.intrinsics->find("wave.shfl_xor").empty()) return false;
  for (std::string_view sym : quant::kGroupAffineSymbols) {
    if (s.intrinsics->find(sym).empty()) return false;
  }
  return device_extension<AmdDeviceInfo>(*s.device) != nullptr;
}
bool shape_ok(const QuantDims& d) { return d.valid; }
std::uint32_t chunks_per_step(const QuantDims& d, std::uint32_t max_bytes) {
  const auto words = static_cast<std::uint32_t>(d.spec.words_per_chunk());
  const auto vals = static_cast<std::uint32_t>(d.spec.values_per_chunk());
  const std::uint32_t cap =
      row_pack(static_cast<std::uint32_t>(d.lanes), max_bytes, 4);
  std::uint32_t cpl = 1;
  if ((words & (words - 1)) == 0 && words <= cap) cpl = cap / words;
  while (cpl > 1 && cpl * vals > kMaxUnrolledCodes) cpl >>= 1;
  return cpl;
}

ThreadPlan gemv_plan(const QuantDims& d, std::uint32_t wave,
                     std::uint32_t lds_bytes, std::uint32_t rows_per_wg,
                     std::uint32_t cols_per_wave = 1) {
  if (wave != 32 && wave != 64) wave = 32;
  const std::uint32_t waves = kBlock / wave * cols_per_wave;
  ThreadPlan tp;
  tp.workgroup_size[0] = kBlock;
  tp.workgroup_count[0] =
      static_cast<std::uint32_t>((d.n + waves - 1) / waves);
  const std::uint32_t rows = rows_per_wg != 0 ? rows_per_wg : 1;
  const auto m = static_cast<std::uint32_t>(d.m > 0 ? d.m : 1);
  tp.workgroup_count[1] = (m + rows - 1) / rows;
  tp.workgroup_count[2] = 1;
  tp.lds_bytes = lds_bytes;
  return tp;
}

template <class S>
struct QuantLinearArgs {
  env::In<kir::f32, env::Emit> x;
  env::In<std::uint32_t, env::Emit> packed;
  env::In<S, env::Emit> scales;
  env::In<S, env::Emit> biases;
  env::Out<kir::f32, env::Emit> out;
};

template <class S>
struct QuantLinearIndexedArgs {
  env::In<kir::f32, env::Emit> x;
  env::In<std::uint32_t, env::Emit> packed;
  env::In<S, env::Emit> scales;
  env::In<S, env::Emit> biases;
  env::In<kir::f32, env::Emit> idx;
  env::Out<kir::f32, env::Emit> out;
};

template <class A>
concept Indexed = requires(A& a) { a.idx; };
struct DotActs {
  kir::Tile<kir::u32> codes;
  kir::Tile<kir::f32> scale;
  kir::Tile<kir::f32> sum;
};
inline std::uint32_t swz_words(std::uint32_t n) { return n + n / 32u + 1u; }
inline kir::Val<kir::u32> swz(env::Emit& e, const kir::Val<kir::u32>& i) {
  return e.let(i + i / 32u);
}

std::uint32_t dot_lds_bytes_span(std::uint32_t nchunks, std::uint32_t groups) {
  constexpr std::uint32_t w = kir::pack_elem_bytes<kir::u32>();
  return kir::Lds::align(swz_words(2 * nchunks) * w) +
         kir::Lds::align(swz_words(nchunks) * w) + kir::Lds::align(groups * w);
}

std::uint32_t dot_lds_bytes(const QuantDims& d) {
  return dot_lds_bytes_span(
      static_cast<std::uint32_t>(d.k / d.spec.values_per_chunk()),
      static_cast<std::uint32_t>(d.groups));
}
std::uint32_t dot_ksplits(const QuantDims& d, std::uint32_t rows,
                          std::uint32_t budget) {
  const auto nchunks =
      static_cast<std::uint32_t>(d.k / d.spec.values_per_chunk());
  const auto groups = static_cast<std::uint32_t>(d.groups);
  const auto cpg =
      static_cast<std::uint32_t>(d.spec.group_size / d.spec.values_per_chunk());
  for (std::uint32_t t = 1; t <= 4; t <<= 1) {
    if (nchunks % t != 0 || groups % t != 0) continue;
    if ((nchunks / t) % cpg != 0) continue;
    if (budget == 0 ||
        rows * dot_lds_bytes_span(nchunks / t, groups / t) <= budget) {
      return t;
    }
  }
  return 0;
}
bool dot_ok(const KernelShapes& s, const QuantDims& d, std::uint32_t wave,
            std::uint32_t cpl) {
  if (!dispatch::quant_plan(s).int8_activations) return false;
  if (!d.valid || d.spec.bits != quant::kDot4Bits) return false;
  const auto cpg = static_cast<std::uint32_t>(d.spec.group_size) /
                   static_cast<std::uint32_t>(d.spec.values_per_chunk());
  return cpl <= cpg && cpg % cpl == 0 && cpg <= wave && kBlock % cpg == 0;
}
bool body_dot(const KernelShapes& s, const QuantDims& d) {
  if (!d.valid) return false;
  const std::uint32_t cpl = chunks_per_step(d, device_load_bytes(s.device));
  if (!dot_ok(s, d, wave_of(s.device), cpl)) return false;
  const std::uint32_t budget = workgroup_lds_bytes(s.device);
  const bool quant_hoisted = s.staged_quant.matches(
      static_cast<std::uint32_t>(d.k), d.spec.group_size, d.spec.bits);
  return quant_hoisted || budget == 0 || dot_lds_bytes(d) <= budget;
}

std::uint32_t q6_decode_columns(const KernelShapes& s, const QuantDims&,
                                bool indexed) {
  return dispatch::quant_plan(s, indexed).decode_columns;
}

std::uint32_t q6_prefill_tile(std::uint32_t k, std::uint32_t rows) {
  // Eight rows retain the validated four-row panel's 16 KiB maximum LDS.
  // 512 is exactly one 32-lane wave of 16-code chunks, preserving K order.
  return std::min(rows >= 8 ? 512u : 1024u, k);
}

std::uint32_t q6_prefill_rows(const KernelShapes& s, const QuantDims&,
                              bool indexed) {
  return dispatch::quant_plan(s, indexed).prefill_rows;
}
std::uint32_t body_lds_bytes_at(const KernelShapes& s, const QuantDims& d,
                                std::uint32_t rows) {
  if (!d.valid || rows == 0) return 0;
  if (rows > 1 && d.spec.bits == 6 && !body_dot(s, d)) {
    return rows * q6_prefill_tile(static_cast<std::uint32_t>(d.k), rows) * 4u;
  }
  const auto k = static_cast<std::uint32_t>(d.k);
  const std::uint32_t budget = workgroup_lds_bytes(s.device);
  const bool quant_hoisted =
      s.staged_quant.matches(k, d.spec.group_size, d.spec.bits);
  if (body_dot(s, d)) {
    // A hoisted int8 panel holds ONE row; a workgroup covering several reads
    // its siblings' activations from global and stages them itself.
    if (quant_hoisted && rows == 1) return 0;
    const std::uint32_t splits = dot_ksplits(d, rows, budget);
    if (splits == 0) return 0;
    const auto nchunks =
        static_cast<std::uint32_t>(d.k / d.spec.values_per_chunk());
    return rows * dot_lds_bytes_span(nchunks / splits,
                                     static_cast<std::uint32_t>(d.groups) /
                                         splits);
  }
  if (rows == 1 && s.staged.count == k && !s.staged.name.empty()) return 0;
  const std::uint32_t need =
      kir::Lds::align(k * kir::pack_elem_bytes<kir::f32>());
  return (budget == 0 || need <= budget) ? need : 0;
}
opt::TrafficModel traffic_at(const KernelShapes& s, const QuantDims& d,
                             bool indexed, std::uint32_t rows) {
  if (!shape_ok(d) || !device_fits(s) || rows == 0) return {};
  const std::uint32_t wave = wave_of(s.device);
  const std::uint32_t columns = q6_decode_columns(s, d, indexed);
  const std::uint32_t cols = kBlock / wave * columns;
  const auto scale_elem =
      static_cast<std::uint32_t>(dtype_storage_bytes(s.input_dtypes[2], 1));
  // One scale and one bias per group, for every column the workgroup owns.
  const auto scale_bytes = static_cast<std::uint64_t>(cols) *
                           static_cast<std::uint64_t>(d.groups) * 2 *
                           scale_elem;
  opt::TrafficModel m = opt::contraction_traffic(
      rows, cols, static_cast<std::uint64_t>(d.k),
      static_cast<std::uint32_t>(d.spec.bits), sizeof(float), scale_bytes,
      static_cast<std::uint64_t>(rows) * cols * sizeof(float));
  if (indexed) {
    m.add_read(opt::OperandClass::kIndex,
               static_cast<std::uint64_t>(rows) * d.keep * sizeof(float));
  }
  const ThreadPlan plan =
      gemv_plan(d, wave, body_lds_bytes_at(s, d, rows), rows, columns);
  m.workgroups = plan.workgroup_count[0] * plan.workgroup_count[1];
  m.workgroup_threads = plan.workgroup_size[0];
  return m;
}
constexpr std::uint32_t kMaxRowsPerGroup = 8;
// Most weight runs a decode lane issues before it accumulates the first.
constexpr std::uint32_t kPrefetchRuns = 8;
std::uint32_t rows_per_group(const KernelShapes& s, const QuantDims& d,
                             bool indexed) {
  const auto exact_rows = q6_prefill_rows(s, d, indexed);
  if (exact_rows > 1) return exact_rows;
  if (!d.valid || indexed || d.m <= 1 || !body_dot(s, d)) return 1;
  if (s.device == nullptr) return 1;
  // A part's own shared-panel row that states its rows: the contraction
  // reads the panel, not workgroup scratch, so the scratch the pricing below
  // charges per row is not what it costs, and the part measured the rows.
  if (const auto* panel = dispatch::q4_shared_panel_rule(s);
      panel != nullptr && !panel->arch.empty() && panel->rows > 1 &&
      dispatch::quant_plan(s, indexed).shared_activation_panel)
    return panel->rows;
  const auto m = static_cast<std::uint32_t>(d.m);
  const auto row_limit = std::max(m, dispatch::quant_plan(s, indexed).row_ladder_ceiling);
  const std::uint32_t budget = workgroup_lds_bytes(s.device);

  std::array<opt::Arrangement, 4> priced{};
  std::array<std::uint32_t, 4> rung{};
  std::size_t n = 0;
  for (std::uint32_t r = 1; r <= kMaxRowsPerGroup && r <= row_limit; r <<= 1) {
    const std::uint32_t splits = dot_ksplits(d, r, budget);
    if (splits == 0) break;
    priced[n].traffic = traffic_at(s, d, indexed, r);
    // COUNTED, not measured, and counted as the plan will report it: a panel a
    // fused run hoists belongs to the run and is priced by the fusion gate.
    priced[n].demand =
        opt::KernelDemand::counted(kBlock, body_lds_bytes_at(s, d, r));
    rung[n] = r;
    ++n;
  }
  if (n == 0) return 1;
  const opt::DeviceCapacity cap = opt::DeviceCapacity::of(*s.device);
  const std::size_t best = opt::best_arrangement(
      cap, std::span<const opt::Arrangement>(priced.data(), n));
  return best < n ? rung[best] : 1;
}

// R for this shape, or 1 when the integer path does not apply. The plan and
// the body must agree on it exactly: it sets the grid and the scratch.
std::uint32_t dot_rows(const KernelShapes& s, const QuantDims& d,
                       bool indexed) {
  if (!d.valid) return 1;
  return rows_per_group(s, d, indexed);
}
std::uint32_t body_lds_bytes(const KernelShapes& s, const QuantDims& d,
                             bool indexed) {
  if (!d.valid) return 0;
  return body_lds_bytes_at(s, d, dot_rows(s, d, indexed));
}

opt::TrafficModel quant_traffic(const KernelShapes& s, bool indexed) {
  const QuantDims d = dims_of(s, indexed);
  if (!shape_ok(d) || !device_fits(s)) return {};
  return traffic_at(s, d, indexed, dot_rows(s, d, indexed));
}
template <class Fetch>
void stage_dot_acts_from(env::Emit& e, const Fetch& fetch, const DotActs& q,
                         const kir::Val<kir::u32>& lid, std::uint32_t nchunks,
                         std::uint32_t cpg) {
  constexpr std::uint32_t kCodes = quant::kDot4ChunkCodes;
  // An all-zero group would divide by zero; the floor is below any activation
  // that carries information and every code in such a group is 0 regardless.
  constexpr float kAmaxFloor = 1e-30f;
  for (auto c : e.range(lid, e.u32(nchunks), kBlock)) {
    std::array<kir::Val<kir::f32>, kCodes> v;
    fetch(c, v);
    auto amax = e.let(math::abs(v[0]));
    auto sum = v[0];
    for (std::uint32_t j = 1; j < kCodes; ++j) {
      amax = e.let(math::max(amax, math::abs(v[j])));
      sum = e.let(sum + v[j]);
    }
    for (std::uint32_t bit = 1; bit < cpg; bit <<= 1) {
      sum = e.let(sum + math::shfl_xor(sum, e.u32(bit)));
    }
    // 127, not 128, so the codes stay symmetric: -x and x quantize to the
    // same magnitude, which is what keeps the bias term unbiased.
    const auto step = e.let(amax * (1.0f / 127.0f));
    const auto inv = e.let(127.0f / math::max(amax, e.f32(kAmaxFloor)));
    // |x| <= amax by construction, so the rounded code is inside
    // [-127, 127] and no clamp is needed.
    const auto byte_of = [&](int j) {
      const auto code = e.let(math::rint(v[static_cast<std::size_t>(j)] * inv));
      return e.let(kir::cast<kir::u32>(kir::cast<kir::i32>(code)) % 256u);
    };
    const auto slot = e.let(c * 2u);
    q.codes[swz(e, slot)] = quant::dot4_activation_word(e, byte_of, 0);
    q.codes[swz(e, e.let(slot + 1u))] =
        quant::dot4_activation_word(e, byte_of, 1);
    q.scale[swz(e, c)] = step;
    if (auto lead = e.when(c % cpg == 0u)) {
      q.sum[e.let(c / cpg)] = sum;
    }
  }
  e.barrier();
}
template <class A>
void stage_dot_acts_all(env::Emit& e, const A& a, const kir::Tile<kir::f32>* xs,
                        const kir::Val<kir::u32>& row,
                        std::uint32_t rows, std::uint32_t m_rows,
                        std::span<const DotActs> q,
                        const kir::Val<kir::u32>& lid,
                        std::uint32_t chunk_begin, std::uint32_t chunk_end,
                        std::uint32_t cpg, std::uint32_t max_bytes,
                        std::uint32_t k, bool staged_row) {
  constexpr std::uint32_t kCodes = quant::kDot4ChunkCodes;
  constexpr float kAmaxFloor = 1e-30f;
  const std::uint32_t wide = row_pack(kCodes, max_bytes, 4);
  for (auto c : e.range(e.u32(chunk_begin) + lid, e.u32(chunk_end), kBlock)) {
    // The arrays hold ONE split's chunks; every staged index is relative to
    // the split so the same scratch serves each split in turn.
    const auto rel = e.let(c - chunk_begin);
    std::vector<std::array<kir::Val<kir::f32>, kCodes>> v(rows);
    for (std::uint32_t r = 0; r < rows; ++r) {
      // A row past the end fetches row 0\'s bytes -- in bounds, never stored.
      const auto x_base =
          e.let(staged_row ? e.u32(0)
                           : e.let(math::min(row + r, e.u32(m_rows - 1)) * k));
      const auto base = e.let(x_base + c * kCodes);
      if (xs != nullptr) {
        for (std::uint32_t j = 0; j < kCodes; ++j) {
          v[r][j] = e.let((*xs)[e.let(base + j)].read());
        }
      } else {
        for (std::uint32_t j = 0; j < kCodes; j += wide) {
          const auto pack = e.load(a.x, e.let(base + j), max_bytes);
          for (std::uint32_t t = 0; t < wide; ++t) {
            v[r][j + t] = e.let(pack[t]);
          }
        }
      }
    }
    for (std::uint32_t r = 0; r < rows; ++r) {
      if (auto live = e.when(row + r < m_rows)) {
        auto amax = e.let(math::abs(v[r][0]));
        auto sum = v[r][0];
        for (std::uint32_t j = 1; j < kCodes; ++j) {
          amax = e.let(math::max(amax, math::abs(v[r][j])));
          sum = e.let(sum + v[r][j]);
        }
        for (std::uint32_t bit = 1; bit < cpg; bit <<= 1) {
          sum = e.let(sum + math::shfl_xor(sum, e.u32(bit)));
        }
        const auto step = e.let(amax * (1.0f / 127.0f));
        const auto inv = e.let(127.0f / math::max(amax, e.f32(kAmaxFloor)));
        const auto byte_of = [&](int j) {
          const auto code =
              e.let(math::rint(v[r][static_cast<std::size_t>(j)] * inv));
          return e.let(kir::cast<kir::u32>(kir::cast<kir::i32>(code)) % 256u);
        };
        const auto slot = e.let(rel * 2u);
        q[r].codes[swz(e, slot)] = quant::dot4_activation_word(e, byte_of, 0);
        q[r].codes[swz(e, e.let(slot + 1u))] =
            quant::dot4_activation_word(e, byte_of, 1);
        q[r].scale[swz(e, rel)] = step;
        if (auto lead = e.when(c % cpg == 0u)) {
          q[r].sum[e.let(rel / cpg)] = sum;
        }
      }
    }
  }
  e.barrier();
}
// stage_dot_acts_all for one k split, in two phases: `fetch` issues every
// activation read a thread makes, `finish` quantizes and stores what they
// returned. The decode body issues its weight reads between the two, so the
// activation reads -- issued first -- are the ones the staging waits on.
// Same values, same order, same scratch layout as stage_dot_acts_all.
struct SplitStage {
  std::uint32_t steps = 0;
  std::vector<std::vector<std::array<kir::Val<kir::f32>, quant::kDot4ChunkCodes>>> v;
};
template <class A>
SplitStage stage_dot_acts_fetch(env::Emit& e, const A& a,
                                const kir::Tile<kir::f32>* xs,
                                const kir::Val<kir::u32>& row,
                                std::uint32_t rows, std::uint32_t m_rows,
                                const kir::Val<kir::u32>& lid,
                                std::uint32_t chunk_end, std::uint32_t max_bytes,
                                std::uint32_t k, bool staged_row) {
  constexpr std::uint32_t kCodes = quant::kDot4ChunkCodes;
  const std::uint32_t wide = row_pack(kCodes, max_bytes, 4);
  SplitStage out;
  out.steps = (chunk_end + kBlock - 1) / kBlock;
  for (std::uint32_t step = 0; step < out.steps; ++step) {
    // A thread past the last chunk reads the last chunk; finish never
    // stores it.
    const auto c = e.let(lid + step * kBlock);
    const auto safe_c = (step + 1u) * kBlock <= chunk_end
                            ? c
                            : e.let(select(c < chunk_end, c, e.u32(chunk_end - 1u)));
    std::vector<std::array<kir::Val<kir::f32>, kCodes>> v(rows);
    for (std::uint32_t r = 0; r < rows; ++r) {
      const auto x_base =
          e.let(staged_row ? e.u32(0)
                           : e.let(math::min(row + r, e.u32(m_rows - 1)) * k));
      const auto base = e.let(x_base + safe_c * kCodes);
      if (xs != nullptr) {
        for (std::uint32_t j = 0; j < kCodes; ++j)
          v[r][j] = e.let((*xs)[e.let(base + j)].read());
      } else {
        for (std::uint32_t j = 0; j < kCodes; j += wide) {
          const auto pack = e.load(a.x, e.let(base + j), max_bytes);
          for (std::uint32_t t = 0; t < wide; ++t) v[r][j + t] = e.let(pack[t]);
        }
      }
    }
    out.v.push_back(std::move(v));
  }
  return out;
}
inline void stage_dot_acts_finish(env::Emit& e, const SplitStage& fetched,
                                  const kir::Val<kir::u32>& row,
                                  std::uint32_t rows, std::uint32_t m_rows,
                                  std::span<const DotActs> q,
                                  const kir::Val<kir::u32>& lid,
                                  std::uint32_t chunk_end, std::uint32_t cpg) {
  constexpr std::uint32_t kCodes = quant::kDot4ChunkCodes;
  constexpr float kAmaxFloor = 1e-30f;
  for (std::uint32_t step = 0; step < fetched.steps; ++step) {
    const auto c = e.let(lid + step * kBlock);
    const auto body = [&] {
      for (std::uint32_t r = 0; r < rows; ++r) {
        const auto& v = fetched.v[step][r];
        if (auto live = e.when(row + r < m_rows)) {
          auto amax = e.let(math::abs(v[0]));
          auto sum = v[0];
          for (std::uint32_t j = 1; j < kCodes; ++j) {
            amax = e.let(math::max(amax, math::abs(v[j])));
            sum = e.let(sum + v[j]);
          }
          for (std::uint32_t bit = 1; bit < cpg; bit <<= 1) {
            sum = e.let(sum + math::shfl_xor(sum, e.u32(bit)));
          }
          const auto qstep = e.let(amax * (1.0f / 127.0f));
          const auto inv = e.let(127.0f / math::max(amax, e.f32(kAmaxFloor)));
          const auto byte_of = [&](int j) {
            const auto code =
                e.let(math::rint(v[static_cast<std::size_t>(j)] * inv));
            return e.let(kir::cast<kir::u32>(kir::cast<kir::i32>(code)) % 256u);
          };
          const auto slot = e.let(c * 2u);
          q[r].codes[swz(e, slot)] = quant::dot4_activation_word(e, byte_of, 0);
          q[r].codes[swz(e, e.let(slot + 1u))] =
              quant::dot4_activation_word(e, byte_of, 1);
          q[r].scale[swz(e, c)] = qstep;
          if (auto lead = e.when(c % cpg == 0u)) {
            q[r].sum[e.let(c / cpg)] = sum;
          }
        }
      }
    };
    if ((step + 1u) * kBlock <= chunk_end) {
      body();
    } else if (auto live = e.when(c < chunk_end)) {
      body();
    }
  }
  e.barrier();
}
template <class A>
void stage_dot_acts(env::Emit& e, const A& a, const kir::Tile<kir::f32>* xs,
                    const kir::Val<kir::u32>& x_base, const DotActs& q,
                    const kir::Val<kir::u32>& lid, std::uint32_t nchunks,
                    std::uint32_t cpg, std::uint32_t max_bytes) {
  constexpr std::uint32_t kCodes = quant::kDot4ChunkCodes;
  const std::uint32_t wide = row_pack(kCodes, max_bytes, 4);
  stage_dot_acts_from(
      e,
      [&](const kir::Val<kir::u32>& c,
          std::array<kir::Val<kir::f32>, kCodes>& v) {
        const auto base = e.let(x_base + c * kCodes);
        if (xs != nullptr) {
          for (std::uint32_t j = 0; j < kCodes; ++j) {
            v[j] = e.let((*xs)[e.let(base + j)].read());
          }
        } else {
          for (std::uint32_t j = 0; j < kCodes; j += wide) {
            const auto pack = e.load(a.x, e.let(base + j), max_bytes);
            for (std::uint32_t u = 0; u < wide; ++u) {
              v[j + u] = e.let(pack[static_cast<int>(u)]);
            }
          }
        }
      },
      q, lid, nchunks, cpg);
}
// One lane's run of `count` chunks from weight words and a group scale it
// already holds. emit_run_dot reads them first; the decode body issues them
// ahead so their latency overlaps the activation staging and earlier runs.
template <class A>
void emit_run_dot_loaded(env::Emit& e, std::span<const DotActs> q,
                         const kir::Pack<kir::u32>& words,
                         const kir::Val<kir::f32>& wscale,
                         const kir::Val<kir::u32>& chunk0, std::uint32_t count,
                         std::span<const kir::LValue<kir::f32>> acc,
                         std::uint32_t stage_begin) {
  std::vector<kir::LValue<kir::f32>> facc;
  facc.reserve(acc.size());
  for (std::size_t r = 0; r < acc.size(); ++r) facc.push_back(e.var(e.f32(0.0f)));
  for (auto uu : e.unroll(count)) {
    const auto chunk = e.let(chunk0 + uu);
    // Weights and the global scale index by the ABSOLUTE chunk; the staged
    // arrays hold one k-split and index relative to it.
    const auto srel = e.let(chunk - stage_begin);
    const auto word = e.let(words[uu]);
    const auto slot = e.let(srel * 2u);
    std::array<kir::Val<kir::u32>, 2> planes;
    for (std::size_t p = 0; p < 2; ++p) {
      planes[p] = quant::dot4_code_plane(e, word, static_cast<int>(p));
    }
    for (std::size_t r = 0; r < acc.size(); ++r) {
      auto iacc = e.var(kir::cast<kir::i32>(e.u32(0)));
      for (std::size_t p = 0; p < 2; ++p) {
        const auto x = e.let(
            q[r].codes[swz(e, e.let(slot + static_cast<std::uint32_t>(p)))]
                .read());
        iacc = math::dot4_iu8(kir::cast<kir::i32>(x),
                              kir::cast<kir::i32>(planes[p]), iacc.read());
      }
      facc[r] = math::fma(q[r].scale[swz(e, srel)].read(),
                          kir::cast<kir::f32>(iacc.read()), facc[r].read());
    }
  }
  for (std::size_t r = 0; r < acc.size(); ++r) {
    acc[r] = math::fma(wscale, facc[r].read(), acc[r].read());
  }
}
template <class A>
void emit_run_dot(env::Emit& e, const A& a, std::span<const DotActs> q,
                  const kir::Val<kir::u32>& row_base,
                  const kir::Val<kir::u32>& scale_base,
                  const kir::Val<kir::u32>& chunk0, std::uint32_t count,
                  std::span<const kir::LValue<kir::f32>> acc,
                  std::uint32_t cpg, std::uint32_t stage_begin) {
  const auto words = e.load(a.packed, row_base + chunk0, count * 4u);
  const auto group = e.let(chunk0 / cpg);
  const auto wscale = e.let(math::widen(a.scales[scale_base + group]));
  emit_run_dot_loaded<A>(e, q, words, wscale, chunk0, count, acc, stage_begin);
}
template <class A>
void emit_bias(env::Emit& e, const A& a, std::span<const DotActs> q,
               const kir::Val<kir::u32>& scale_base,
               const kir::Val<kir::u32>& lane, std::uint32_t g_begin,
               std::uint32_t g_end, std::uint32_t wave,
               std::span<const kir::LValue<kir::f32>> acc) {
  for (auto g : e.range(e.u32(g_begin) + lane, e.u32(g_end), wave)) {
    const auto b = e.let(math::widen(a.biases[scale_base + g]));
    const auto grel = e.let(g - g_begin);
    for (std::size_t r = 0; r < acc.size(); ++r) {
      acc[r] = math::fma(b, q[r].sum[grel].read(), acc[r].read());
    }
  }
}

// acc += sum over the codes of one chunk. `xs` is the activation row staged in
// LDS, or null when it did not fit and the row is read from global.
template <class A>
void emit_chunk(env::Emit& e, const A& a, const kir::Tile<kir::f32>* xs,
                const quant::GroupAffine& spec,
                const kir::Val<kir::u32>& row_base,
                const kir::Val<kir::u32>& scale_base,
                const kir::Val<kir::u32>& x_base,
                const kir::Val<kir::u32>& chunk,
                const kir::LValue<kir::f32>& acc,
                std::uint32_t chunks_per_group, bool rotate_panel) {
  const auto vals = static_cast<std::uint32_t>(spec.values_per_chunk());
  const auto words = static_cast<std::uint32_t>(spec.words_per_chunk());
  const auto group = e.let(chunk / chunks_per_group);
  const auto scale = e.let(math::widen(a.scales[scale_base + group]));
  const auto bias = e.let(math::widen(a.biases[scale_base + group]));
  const auto k_base = e.let(chunk * vals);
  quant::dequant_chunk(
      e, a.packed, spec, e.let(row_base + chunk * words), scale, bias,
      [&](int c, const kir::Val<kir::f32>& w) {
        const auto idx = e.let(k_base + static_cast<std::uint32_t>(c));
        const auto panel_idx = rotate_panel ? e.let(q6_panel_index(idx)) : idx;
        const auto xe = xs != nullptr ? (*xs)[panel_idx].read() : a.x[x_base + idx];
        acc = math::fma(xe, w, acc.read());
      });
}
template <class A>
std::string emit_q6_prefill(const KernelShapes& s, const QuantDims& d,
                            std::uint32_t rows) {
  const auto n = static_cast<std::uint32_t>(d.n);
  const auto m = static_cast<std::uint32_t>(d.m);
  const auto k = static_cast<std::uint32_t>(d.k);
  const auto lanes = static_cast<std::uint32_t>(d.lanes);
  const auto groups = static_cast<std::uint32_t>(d.groups);
  const auto tile_k = q6_prefill_tile(k, rows);
  constexpr std::uint32_t wave = 32;
  constexpr std::uint32_t values = 16;
  constexpr std::uint32_t words = 3;
  constexpr std::uint32_t chunks_per_group = 4;
  kir::KernelBody kb(s.types, *s.intrinsics, workgroup_lds_bytes(s.device));
  kb.set_store(s.store);
  A a;
  if (!env::bind(kb, a, s)) return {};
  env::Emit e{&kb};
  const auto lid = e.let(math::local_id());
  const auto lane = e.let(lid % wave);
  const auto col = e.let(math::workgroup_id_x() * (kBlock / wave) + lid / wave);
  const auto row = e.let(math::workgroup_id_y() * rows);
  const auto panel = e.lds<kir::f32>(rows * tile_k);
  std::vector<kir::LValue<kir::f32>> acc;
  for (std::uint32_t r = 0; r < rows; ++r) acc.push_back(e.var(0.0f));
  for (auto base : e.range(0u, k, tile_k)) {
    for (auto at : e.range(lid, e.u32(tile_k), kBlock)) {
      for (std::uint32_t r = 0; r < rows; ++r) {
        auto x = e.var(0.0f);
        if (auto live = e.when(row + r < m && base + at < k)) {
          x = a.x[(row + r) * k + base + at];
        }
        panel[r * tile_k + q6_panel_index(at)] = x.read();
      }
    }
    e.barrier();
    if (auto live_col = e.when(col < n)) {
      for (auto chunk : e.range(lane, e.u32(tile_k / values), wave)) {
        const auto logical_k = e.let(base + chunk * values);
        if (auto live_k = e.when(logical_k < k)) {
          const auto global_chunk = e.let(logical_k / values);
          const auto group = e.let(global_chunk / chunks_per_group);
          const auto scale = e.let(math::widen(a.scales[col * groups + group]));
          const auto bias = e.let(math::widen(a.biases[col * groups + group]));
          quant::dequant_chunk(e, a.packed, d.spec,
              e.let(col * lanes + global_chunk * words), scale, bias,
              [&](int c, const kir::Val<kir::f32>& weight) {
                for (std::uint32_t r = 0; r < rows; ++r) {
                  const auto at = e.let(r * tile_k + q6_panel_index(
                      chunk * values + static_cast<std::uint32_t>(c)));
                  acc[r] = math::fma(panel[at].read(), weight, acc[r].read());
                }
              });
        }
      }
    }
    // All readers retire before any wave refills the shared panel.
    e.barrier();
  }
  for (std::uint32_t r = 0; r < rows; ++r) {
    for (std::uint32_t bit = 1; bit < wave; bit <<= 1) {
      acc[r] = acc[r].read() + math::shfl_xor(acc[r].read(), e.u32(bit));
    }
    if (auto writer = e.when(lane == 0 && col < n && row + r < m)) {
      e.store((row + r) * n + col, acc[r].read());
    }
  }
  return kb.lds().ok() ? kb.str() : std::string{};
}

template <class A>
std::string emit_q6_decode_quad(const KernelShapes& s, const QuantDims& d) {
  const auto n = static_cast<std::uint32_t>(d.n);
  const auto k = static_cast<std::uint32_t>(d.k);
  const auto lanes = static_cast<std::uint32_t>(d.lanes);
  const auto groups = static_cast<std::uint32_t>(d.groups);
  constexpr std::uint32_t wave = 32, columns = 4;
  const std::uint32_t vals = static_cast<std::uint32_t>(d.spec.values_per_chunk());
  const std::uint32_t words = static_cast<std::uint32_t>(d.spec.words_per_chunk());
  const std::uint32_t chunks_per_group =
      static_cast<std::uint32_t>(d.spec.group_size) / vals;
  const std::uint32_t mask = 1u << d.spec.bits;
  const auto chunks = k / vals;
  const auto aligned = chunks / wave * wave;
  kir::KernelBody kb(s.types, *s.intrinsics, workgroup_lds_bytes(s.device));
  kb.set_store(s.store);
  A a;
  if (!env::bind(kb, a, s)) return {};
  env::Emit e{&kb};
  const auto lid = e.let(math::local_id());
  const auto lane = e.let(lid % wave);
  const auto col = e.let((math::workgroup_id_x() * (kBlock / wave) + lid / wave) * columns);
  kir::Tile<kir::f32> xs;
  if (e.lds_fits<kir::f32>(k)) {
    xs = e.lds<kir::f32>(k);
    for (auto at : e.range(lid, e.u32(k), kBlock)) xs[e.let(q6_panel_index(at))] = a.x[at];
    e.barrier();
  }
  std::array<kir::LValue<kir::f32>, columns> acc{e.var(0.0f), e.var(0.0f), e.var(0.0f), e.var(0.0f)};
  if (auto live = e.when(col < n)) {
    auto accumulate = [&](const kir::Val<kir::u32>& chunk) {
      std::array<kir::Val<kir::f32>, columns> scales, biases;
      std::array<std::array<kir::Val<kir::u32>, 3>, columns> packed;
      for (std::uint32_t output = 0; output < columns; ++output) {
        const auto current = e.let(col + output);
        const auto group = e.let(current * groups + chunk / chunks_per_group);
        scales[output] = e.let(math::widen(a.scales[group]));
        biases[output] = e.let(math::widen(a.biases[group]));
        const auto first = e.let(current * lanes + chunk * words);
        for (std::uint32_t word = 0; word < words; ++word)
          packed[output][word] = e.let(a.packed[first + word]);
      }
      const auto first_x = e.let(chunk * vals);
      for (int code = 0; code < int(vals); ++code) {
        const auto index = e.let(first_x + static_cast<std::uint32_t>(code));
        const auto x = e.let(xs ? xs[e.let(q6_panel_index(index))].read() : a.x[index]);
        const auto word = static_cast<std::size_t>(d.spec.chunk_word(code));
        const int offset = d.spec.chunk_bit(code), carry = d.spec.chunk_carry(code);
        for (std::uint32_t output = 0; output < columns; ++output) {
          auto value = packed[output][word] / (1u << offset);
          if (carry > 0)
            value = value + (packed[output][word + 1] % (1u << carry)) * (1u << (32 - offset));
          else value = value % mask;
          const auto weight = math::fma(math::cast<kir::f32>(e.let(value)), scales[output], biases[output]);
          acc[output] = math::fma(x, weight, acc[output].read());
        }
      }
    };
    for (auto c0 : e.range(0u, aligned, wave)) accumulate(e.let(c0 + lane));
    if (aligned < chunks)
      for (auto chunk : e.range(e.u32(aligned) + lane, e.u32(chunks), wave)) accumulate(chunk);
  }
  for (std::uint32_t output = 0; output < columns; ++output) {
    for (std::uint32_t bit = 1; bit < wave; bit <<= 1)
      acc[output] = acc[output].read() + math::shfl_xor(acc[output].read(), e.u32(bit));
    if (auto writer = e.when(lane == 0u && col < n)) e.store(col + output, acc[output].read());
  }
  return kb.lds().ok() ? kb.str() : std::string{};
}

// Whether the single-row decode dot product takes the prefetching body: one
// k split, whole runs, and the integer path.
bool decode_prefetch(const KernelShapes& s, const QuantDims& d, bool indexed) {
  if (indexed || !d.valid || d.m != 1 || !s.device) return false;
  if (q6_decode_columns(s, d, false) > 1 || q6_prefill_rows(s, d, false) > 1 ||
      !body_dot(s, d) || dot_rows(s, d, false) != 1)
    return false;
  if (dot_ksplits(d, 1, workgroup_lds_bytes(s.device)) != 1) return false;
  const auto vals = static_cast<std::uint32_t>(d.spec.values_per_chunk());
  const auto nchunks = static_cast<std::uint32_t>(d.k) / vals;
  const auto span = wave_of(s.device) * chunks_per_step(d, device_load_bytes(s.device));
  return nchunks >= span && nchunks % span == 0 && nchunks / span <= kPrefetchRuns;
}

// Single-row decode over 4-bit weights with int8 activations. emit_body's dot
// path waits on each run's weight words, then on its group scale, then on each
// bias in turn: a column costs a dozen memory round trips in series. This body
// issues the activation reads, then every weight word, group scale and bias
// the lane will use, and only then quantizes the activations, so a column
// costs about one round trip. The arithmetic and its order are emit_body's:
// the lane map, the per-run accumulation, the bias pass and the shuffle tree.
template <class A>
std::string emit_decode_dot(const KernelShapes& s, const QuantDims& d) {
  const auto n = static_cast<std::uint32_t>(d.n);
  const auto k = static_cast<std::uint32_t>(d.k);
  const auto lanes = static_cast<std::uint32_t>(d.lanes);
  const auto groups = static_cast<std::uint32_t>(d.groups);
  const auto vals = static_cast<std::uint32_t>(d.spec.values_per_chunk());
  const std::uint32_t nchunks = k / vals;
  const std::uint32_t cpg = static_cast<std::uint32_t>(d.spec.group_size) / vals;
  const std::uint32_t wave = wave_of(s.device);
  const std::uint32_t waves = kBlock / wave;
  const std::uint32_t cpl = chunks_per_step(d, device_load_bytes(s.device));
  const std::uint32_t span = wave * cpl;
  const std::uint32_t runs = nchunks / span;
  const std::uint32_t bias_steps = (groups + wave - 1) / wave;
  const std::uint32_t ntiles = (n + waves - 1) / waves;

  kir::KernelBody kb(s.types, *s.intrinsics, workgroup_lds_bytes(s.device));
  kb.set_store(s.store);
  A a;
  if (!env::bind(kb, a, s)) return {};
  env::Emit e{&kb};

  const auto lid = e.let(math::local_id());
  const auto wave_id = e.let(lid / wave);
  const auto lane = e.let(lid % wave);
  const auto tile = e.let(math::workgroup_id_x());
  const auto col = e.let(tile * waves + wave_id);
  const auto row = e.u32(0);
  const bool use_hoisted = s.staged_quant.matches(k, d.spec.group_size, d.spec.bits);
  kir::Tile<kir::f32> xs;
  if (s.staged.count == k && !s.staged.name.empty())
    xs = kir::Tile<kir::f32>(&kb, &kb.types(), std::string(s.staged.name), k);
  const bool stage = static_cast<bool>(xs);
  DotActs q;
  if (use_hoisted) {
    const auto& sq = s.staged_quant;
    q.codes = kir::Tile<kir::u32>(&kb, &kb.types(), std::string(sq.codes),
                                  swz_words(2 * nchunks));
    q.scale = kir::Tile<kir::f32>(&kb, &kb.types(), std::string(sq.scale),
                                  swz_words(nchunks));
    q.sum = kir::Tile<kir::f32>(&kb, &kb.types(), std::string(sq.sum), groups);
  } else {
    q.codes = e.lds<kir::u32>(swz_words(2 * nchunks));
    q.scale = e.lds<kir::f32>(swz_words(nchunks));
    q.sum = e.lds<kir::f32>(groups);
  }
  const std::span<const DotActs> qs{&q, 1};

  auto acc = e.var(0.0f);
  const std::span<const kir::LValue<kir::f32>> accs{&acc, 1};
  if (auto in_grid = e.when(tile < ntiles)) {
    // Activation reads first: the staging waits on them alone.
    SplitStage fetched;
    if (!use_hoisted)
      fetched = stage_dot_acts_fetch<A>(e, a, stage ? &xs : nullptr, row, 1, 1,
                                        lid, nchunks, device_load_bytes(s.device),
                                        k, stage);
    // A wave past the last column reads the last column; it never stores.
    const auto safe_col = e.let(select(col < n, col, e.u32(n - 1u)));
    const auto row_base = e.let(safe_col * lanes);
    const auto scale_base = e.let(safe_col * groups);
    std::vector<kir::Pack<kir::u32>> words;
    std::vector<kir::Val<kir::u32>> chunk;
    using ScaleT = std::remove_cvref_t<decltype(a.scales[e.u32(0)])>;
    std::vector<ScaleT> scale, bias;
    for (std::uint32_t run = 0; run < runs; ++run) {
      chunk.push_back(e.let(lane * cpl + run * span));
      words.push_back(e.load(a.packed, row_base + chunk.back(), cpl * 4u));
      scale.push_back(e.let(a.scales[scale_base + e.let(chunk.back() / cpg)]));
    }
    for (std::uint32_t t = 0; t < bias_steps; ++t) {
      const auto g = e.let(lane + t * wave);
      const auto safe_g = (t + 1u) * wave <= groups
                              ? g
                              : e.let(select(g < groups, g, e.u32(groups - 1u)));
      bias.push_back(e.let(a.biases[scale_base + safe_g]));
    }
    if (!use_hoisted)
      stage_dot_acts_finish(e, fetched, row, 1, 1, qs, lid, nchunks, cpg);
    if (auto in_cols = e.when(col < n)) {
      for (std::uint32_t run = 0; run < runs; ++run)
        emit_run_dot_loaded<A>(e, qs, words[run], e.let(math::widen(scale[run])),
                               chunk[run], cpl, accs, 0);
      for (std::uint32_t t = 0; t < bias_steps; ++t) {
        const auto g = e.let(lane + t * wave);
        const auto apply = [&] {
          acc = math::fma(e.let(math::widen(bias[t])), q.sum[g].read(), acc.read());
        };
        if ((t + 1u) * wave <= groups) {
          apply();
        } else if (auto live = e.when(g < groups)) {
          apply();
        }
      }
    }
  }
  // Outside every guard: shfl_xor is wave-cooperative.
  for (std::uint32_t bit = 1; bit < wave; bit <<= 1)
    acc = acc.read() + math::shfl_xor(acc.read(), e.u32(bit));
  if (auto lane0 = e.when(lane == 0 && col < n)) e.store(col, acc.read());
  if (!kb.lds().ok()) return {};
  return kb.str();
}

template <class A>
std::string emit_body(const KernelShapes& s, const QuantDims& d) {
  if constexpr (!Indexed<A>) {
    if (decode_prefetch(s, d, false)) return emit_decode_dot<A>(s, d);
  }
  if (q6_decode_columns(s, d, Indexed<A>) == 4) return emit_q6_decode_quad<A>(s, d);
  if (const auto rows = q6_prefill_rows(s, d, Indexed<A>); rows > 1) {
    return emit_q6_prefill<A>(s, d, rows);
  }
  const auto n = static_cast<std::uint32_t>(d.n);
  const auto k = static_cast<std::uint32_t>(d.k);
  const auto m = static_cast<std::uint32_t>(d.m);
  const auto lanes = static_cast<std::uint32_t>(d.lanes);
  const auto groups = static_cast<std::uint32_t>(d.groups);
  const auto vals = static_cast<std::uint32_t>(d.spec.values_per_chunk());
  const std::uint32_t nchunks = k / vals;
  const std::uint32_t chunks_per_group =
      static_cast<std::uint32_t>(d.spec.group_size) / vals;
  const std::uint32_t wave = wave_of(s.device);
  const std::uint32_t waves = kBlock / wave;
  const std::uint32_t cpl = chunks_per_step(d, device_load_bytes(s.device));
  const std::uint32_t span = wave * cpl;
  const std::uint32_t aligned = (nchunks / span) * span;
  const std::uint32_t ntiles = (n + waves - 1) / waves;

  kir::KernelBody kb(s.types, *s.intrinsics, workgroup_lds_bytes(s.device));
  kb.set_store(s.store);
  A a;
  if (!env::bind(kb, a, s)) return {};
  env::Emit e{&kb};

  const auto lid = e.let(math::local_id());
  const auto wave_id = e.let(lid / wave);
  const auto lane = e.let(lid % wave);
  const auto tile = e.let(math::workgroup_id_x());
  const auto col = e.let(tile * waves + wave_id);
  const bool quant_hoisted =
      s.staged_quant.matches(k, d.spec.group_size, d.spec.bits);
  const bool dot = body_dot(s, d);
  const std::uint32_t rows = dot_rows(s, d, Indexed<A>);
  const std::uint32_t ksplits =
      body_dot(s, d) ? dot_ksplits(d, rows, workgroup_lds_bytes(s.device))
                     : 1u;
  if (ksplits == 0) return {};
  const bool use_hoisted = quant_hoisted && rows == 1;
  const auto row = e.let(math::workgroup_id_y() * rows);
  kir::Tile<kir::f32> xs;
  bool fill = false;
  if (rows == 1 && s.staged.count == k && !s.staged.name.empty()) {
    xs = kir::Tile<kir::f32>(&kb, &kb.types(), std::string(s.staged.name), k);
  } else if (!dot && e.lds_fits<kir::f32>(k)) {
    xs = e.lds<kir::f32>(k);
    fill = true;
  }
  const bool stage = static_cast<bool>(xs);
  // A caller-owned panel retains its declared linear layout. Decode can rotate
  // only the panel this body fills, using the same map for stores and reads.
  const bool rotate_panel = fill && dispatch::quant_plan(s).rotate_decode_panel;
  std::vector<DotActs> q;
  if (dot) {
    const auto& sq = s.staged_quant;
    if (use_hoisted) {
      DotActs one;
      one.codes = kir::Tile<kir::u32>(&kb, &kb.types(), std::string(sq.codes),
                                      swz_words(2 * nchunks));
      one.scale = kir::Tile<kir::f32>(&kb, &kb.types(), std::string(sq.scale),
                                      swz_words(nchunks));
      one.sum =
          kir::Tile<kir::f32>(&kb, &kb.types(), std::string(sq.sum), groups);
      q.push_back(one);
    } else {
      const std::uint32_t nchunk_t = nchunks / ksplits;
      const std::uint32_t groups_t = groups / ksplits;
      const auto codes = e.lds<kir::u32>(swz_words(2 * nchunk_t) * rows);
      const auto scale = e.lds<kir::f32>(swz_words(nchunk_t) * rows);
      const auto sum = e.lds<kir::f32>(groups_t * rows);
      for (std::uint32_t r = 0; r < rows; ++r) {
        DotActs one;
        one.codes =
            codes.slice(swz_words(2 * nchunk_t) * r, swz_words(2 * nchunk_t));
        one.scale = scale.slice(swz_words(nchunk_t) * r, swz_words(nchunk_t));
        one.sum = sum.slice(groups_t * r, groups_t);
        q.push_back(one);
      }
    }
  }
  const std::span<const DotActs> qs{q};

  std::vector<kir::LValue<kir::f32>> acc;
  acc.reserve(rows);
  for (std::uint32_t r = 0; r < rows; ++r) acc.push_back(e.var(0.0f));
  const std::span<const kir::LValue<kir::f32>> accs{acc};
  if (auto in_grid = e.when(tile < ntiles && row < m)) {
    if (fill) {
      for (auto t : e.range(lid, e.u32(k), kBlock)) {
        const auto at = rotate_panel ? e.let(q6_panel_index(t)) : t;
        xs[at] = a.x[row * k + t];
      }
      e.barrier();
    }
    for (std::uint32_t ks = 0; ks < ksplits; ++ks) {
    const std::uint32_t cb = ks * (nchunks / ksplits);
    const std::uint32_t ce = cb + nchunks / ksplits;
    if (dot && !use_hoisted) {
      if (ks > 0) e.barrier();
      stage_dot_acts_all<A>(e, a, stage ? &xs : nullptr, row, rows,
                            static_cast<std::uint32_t>(m), qs, lid, cb, ce,
                            chunks_per_group, device_load_bytes(s.device),
                            static_cast<std::uint32_t>(d.k), stage);
    }
    if (auto in_cols = e.when(col < n)) {
      auto plane = e.u32(0);
      if constexpr (Indexed<A>) {
        plane = e.let(
            kir::cast<kir::u32>(a.idx[row * d.keep + e.u32(d.slot)]) * n);
      }
      const auto row_base = e.let((plane + col) * lanes);
      const auto scale_base = e.let((plane + col) * groups);
      const auto x_base = e.let(stage ? e.u32(0) : row * k);
      const std::uint32_t ntile = ce - cb;
      const std::uint32_t aligned_t = (ntile / span) * span;
      for (auto c0 : e.range(cb, cb + aligned_t, span)) {
        const auto chunk0 = e.let(c0 + lane * cpl);
        if (dot) {
          emit_run_dot<A>(e, a, qs, row_base, scale_base, chunk0, cpl, accs,
                          chunks_per_group, cb);
        } else {
          for (std::uint32_t u = 0; u < cpl; ++u) {
            emit_chunk<A>(e, a, stage ? &xs : nullptr, d.spec, row_base,
                          scale_base, x_base, e.let(chunk0 + u), acc[0],
                          chunks_per_group, rotate_panel);
          }
        }
      }
      if (aligned_t < ntile) {
        for (auto chunk :
             e.range(e.u32(cb + aligned_t) + lane, e.u32(ce), wave)) {
          if (dot) {
            emit_run_dot<A>(e, a, qs, row_base, scale_base, chunk, 1, accs,
                            chunks_per_group, cb);
          } else {
            emit_chunk<A>(e, a, stage ? &xs : nullptr, d.spec, row_base,
                          scale_base, x_base, chunk, acc[0], chunks_per_group,
                          rotate_panel);
          }
        }
      }
      // The bias half of the algebra, once per group rather than once per
      // weight. The scale half rode the integer accumulator above.
      if (dot) {
        emit_bias<A>(e, a, qs, scale_base, lane, cb / chunks_per_group,
                     ce / chunks_per_group, wave, accs);
      }
    }
    }  // ks: next k-split restages over the same scratch
  }

  // Outside every guard: shfl_xor is wave-cooperative, and the guards above are
  // workgroup-uniform, so a wave that owns no column still has to take part.
  for (std::uint32_t r = 0; r < rows; ++r) {
    for (std::uint32_t bit = 1; bit < wave; bit <<= 1) {
      acc[r] = acc[r].read() + math::shfl_xor(acc[r].read(), e.u32(bit));
    }
  }
  for (std::uint32_t r = 0; r < rows; ++r) {
    const auto rr = e.let(row + r);
    if (auto lane0 = e.when(lane == 0 && col < n && rr < m)) {
      e.store(rr * n + col, acc[r].read());
    }
  }
  if (!kb.lds().ok()) return {};
  return kb.str();
}

}  // namespace

Dot4Schedule dot4_schedule(const graph::KernelShapes& s) {
  const auto d = dims_of(s, false);
  if (!d.valid || !body_dot(s, d)) return {};
  const auto rows = dot_rows(s, d, false);
  return {rows, chunks_per_step(d, device_load_bytes(s.device)),
          dot_ksplits(d, rows, workgroup_lds_bytes(s.device)), wave_of(s.device)};
}

DotStagingPlan dot_staging_plan(const KernelShapes& s, bool indexed) {
  const QuantDims d = dims_of(s, indexed);
  if (!d.valid) return {};
  const std::uint32_t cpl = chunks_per_step(d, device_load_bytes(s.device));
  if (!dot_ok(s, d, wave_of(s.device), cpl)) return {};
  return DotStagingPlan{static_cast<std::uint32_t>(d.k), d.spec.group_size,
                        d.spec.bits};
}

std::uint32_t staged_dot_bytes(std::uint32_t count, std::int32_t group_size,
                               std::uint32_t block) {
  constexpr std::uint32_t kCodes = quant::kDot4ChunkCodes;
  if (count == 0 || block != kBlock || group_size <= 0) return 0;
  const auto gs = static_cast<std::uint32_t>(group_size);
  if (count % gs != 0 || gs % kCodes != 0) return 0;
  const std::uint32_t nchunks = count / kCodes;
  const std::uint32_t groups = count / gs;
  constexpr std::uint32_t w = kir::pack_elem_bytes<kir::u32>();
  return kir::Lds::align(swz_words(2 * nchunks) * w) +
         kir::Lds::align(swz_words(nchunks) * w) + kir::Lds::align(groups * w);
}

StagedQuantNames emit_staged_dot_acts(kir::KernelBody& k, std::string_view row,
                                      std::uint32_t count,
                                      std::int32_t group_size,
                                      std::uint32_t rows,
                                      std::uint32_t block) {
  StagedQuantNames out;
  // The fill below strides by kBlock, which is what the bodies that read it
  // also assume. A run of another width would leave chunks unwritten.
  if (count == 0 || rows == 0 || block != kBlock || group_size <= 0) return out;
  constexpr std::uint32_t kCodes = quant::kDot4ChunkCodes;
  const auto gs = static_cast<std::uint32_t>(group_size);
  if (count % gs != 0 || gs % kCodes != 0) return out;
  const std::uint32_t nchunks = count / kCodes;
  const std::uint32_t cpg = gs / kCodes;
  const std::uint32_t groups = count / gs;

  env::Emit e{&k};
  if (!e.lds_fits<kir::u32>(swz_words(2 * nchunks)) ||
      !e.lds_fits<kir::f32>(swz_words(nchunks)) ||
      !e.lds_fits<kir::f32>(groups)) {
    return out;
  }
  const auto xs = kir::Tile<kir::f32>(&k, &k.types(), std::string(row), count);
  if (!xs) return out;
  DotActs q;
  q.codes = e.lds<kir::u32>(swz_words(2 * nchunks));
  q.scale = e.lds<kir::f32>(swz_words(nchunks));
  q.sum = e.lds<kir::f32>(groups);
  if (!q.codes || !q.scale || !q.sum) return out;

  const auto lid = e.let(math::local_id());
  const auto wg_row = e.let(math::workgroup_id_y());
  if (auto in_rows = e.when(wg_row < rows)) {
    stage_dot_acts_from(
        e,
        [&](const kir::Val<kir::u32>& c,
            std::array<kir::Val<kir::f32>, kCodes>& v) {
          const auto base = e.let(c * kCodes);
          for (std::uint32_t j = 0; j < kCodes; ++j) {
            v[j] = e.let(xs[e.let(base + j)].read());
          }
        },
        q, lid, nchunks, cpg);
  }
  out.codes = q.codes.name();
  out.scale = q.scale.name();
  out.sum = q.sum.name();
  return out;
}

struct QuantLinearKernel final : KernelPrimitive<QuantLinearKernel> {
  static constexpr std::string_view kName = "quant_linear";
  static constexpr std::string_view kEntry = "lse_quant_linear";
  static constexpr std::string_view kSource = {};

  std::size_t arity() const noexcept override { return 4; }
  bool owns_indexing() const noexcept override { return true; }

  StagedRow staged_row(const KernelShapes& s) const override {
    const QuantDims d = dims_of(s, false);
    if (!shape_ok(d) || !device_fits(s)) return {};
    // This schedule owns several rows and stages K tiles itself; advertising a
    // single-row hoisted panel would let fusion change its launch coverage.
    if (q6_prefill_rows(s, d, false) > 1 || q6_decode_columns(s, d, false) > 1) return {};
    return {0, static_cast<std::uint32_t>(d.k),
            static_cast<std::uint32_t>(d.m)};
  }
  const KernelPrimitiveBase* specialize(const KernelShapes& s) const override {
    if (const auto rows = dispatch::q8_matrix_rows(s); rows != 0) {
      if (const KernelPrimitiveBase* w = wmma_q8_linear_for(s, rows)) return w;
    }
    if (const KernelPrimitiveBase* w = wmma_q6_linear_for(s)) return w;
    if (const KernelPrimitiveBase* w = wmma_quant_linear_for(s)) return w;
    return this;
  }

  std::string emit_kernel(const KernelShapes& s) const override {
    const QuantDims d = dims_of(s, false);
    if (!shape_ok(d) || !device_fits(s) || s.types.scalar == nullptr ||
        !s.store) {
      return {};
    }
    return with_elem(s.input_dtypes[2], [&]<class S>() -> std::string {
      return emit_body<QuantLinearArgs<S>>(s, d);
    });
  }

  Result<Shape> infer_shape(std::span<const Shape> in) const override {
    if (in.size() != 4 || in[1].rank() != 2) {
      return LSE_ERROR(kInvalidArgument,
                       "quant_linear takes x, packed[N, lanes], scales, biases");
    }
    Shape out;
    for (std::size_t i = 0; i + 1 < in[0].rank(); ++i) out.push_back(in[0].dim(i));
    out.push_back(in[1].dim(0));
    return out;
  }

  // The codes carry the weight and the scale only places it, so the product is
  // an activation whatever the planes are stored as.
  DType infer_dtype(std::span<const DType>) const override {
    return DType::kF32;
  }

  opt::TrafficModel traffic(const KernelShapes& s) const override {
    return quant_traffic(s, false);
  }

  static ThreadPlan plan_impl(const KernelShapes& s) {
    const QuantDims d = dims_of(s, false);
    return gemv_plan(d, wave_of(s.device), body_lds_bytes(s, d, false),
                     dot_rows(s, d, false), q6_decode_columns(s, d, false));
  }
};

LSE_REGISTER_PRIMITIVE(QuantLinearKernel);
struct QuantLinearIndexedKernel final
    : KernelPrimitive<QuantLinearIndexedKernel> {
  static constexpr std::string_view kName = "quant_linear_indexed";
  static constexpr std::string_view kEntry = "lse_quant_linear_indexed";
  static constexpr std::string_view kSource = {};

  std::size_t arity() const noexcept override { return 5; }
  bool owns_indexing() const noexcept override { return true; }

  StagedRow staged_row(const KernelShapes& s) const override {
    const QuantDims d = dims_of(s, true);
    if (!shape_ok(d) || !device_fits(s)) return {};
    return {0, static_cast<std::uint32_t>(d.k),
            static_cast<std::uint32_t>(d.m)};
  }

  std::string emit_kernel(const KernelShapes& s) const override {
    const QuantDims d = dims_of(s, true);
    if (!shape_ok(d) || !device_fits(s) || s.types.scalar == nullptr ||
        !s.store) {
      return {};
    }
    return with_elem(s.input_dtypes[2], [&]<class S>() -> std::string {
      return emit_body<QuantLinearIndexedArgs<S>>(s, d);
    });
  }

  Result<Shape> infer_shape(std::span<const Shape> in) const override {
    if (in.size() != 5 || in[1].rank() != 3) {
      return LSE_ERROR(kInvalidArgument,
                       "quant_linear_indexed takes x, packed[E, N, lanes], "
                       "scales, biases, idx");
    }
    Shape out;
    for (std::size_t i = 0; i + 1 < in[0].rank(); ++i) out.push_back(in[0].dim(i));
    out.push_back(in[1].dim(1));
    return out;
  }

  DType infer_dtype(std::span<const DType>) const override {
    return DType::kF32;
  }

  opt::TrafficModel traffic(const KernelShapes& s) const override {
    return quant_traffic(s, true);
  }

  static ThreadPlan plan_impl(const KernelShapes& s) {
    const QuantDims d = dims_of(s, true);
    return gemv_plan(d, wave_of(s.device), body_lds_bytes(s, d, true),
                     dot_rows(s, d, true));
  }
};

LSE_REGISTER_PRIMITIVE(QuantLinearIndexedKernel);

}  // namespace lse::kernels
