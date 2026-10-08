#include "lse/kernels/gdn.hpp"

#include "lse/graph/kernel_args.hpp"
#include "lse/graph/kernel_env.hpp"
#include "lse/graph/kernel_primitive.hpp"
#include "lse/math.hpp"
#include "lse/backends/hrx/device_info.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <array>
#include <string>
#include <vector>

namespace lse::kernels {

// These name device facts, which the backend supplies.
using backend::DeviceInfo;

using namespace lse::graph;
namespace math = lse::math;

namespace {

constexpr std::uint32_t kBlock = 256;
// Timesteps whose inputs one scan wave reads together before it updates.
constexpr std::uint32_t kStepBlock = 8;
// Scans this short (a speculative verify, a commit of its accepted prefix)
// whose head width leaves lanes partly idle still load every timestep's
// inputs before the recurrence starts.
constexpr std::uint32_t kGdnPreloadSteps = 8;

std::uint32_t wave_of(const DeviceInfo* device) {
  if (device == nullptr) return 32;
  const std::uint32_t wave = device->wavefront_size;
  return (wave == 32 || wave == 64) ? wave : 32u;
}

enum class GdnWrite : std::uint8_t { kOut, kState, kBoth };

template <class E>
struct GdnArgs {
  env::In<kir::f32, E> q;
  env::In<kir::f32, E> k;
  env::In<kir::f32, E> v;
  env::In<kir::f32, E> alpha;
  env::In<kir::f32, E> beta;
  env::In<kir::f32, E> s;
  // Input slot 6, written by the .pair variant to carry final state.
  env::InOut<kir::f32, E> sout;
  // Never indexed: this kernel owns indexing and stores through the emitter's
  // epilogue hook, but the binding contract wants exactly one Out.
  env::Out<kir::f32, E> out;
};

// One wave owns one row of S. Each lane holds D/wave (or 1) scalars — never
// a D-wide ext_vector, which is what spilled 64 floats to scratch.
// kBoth scans time once: write o[t] every step and S at the end.
std::string emit_gdn(const KernelShapes& s, GdnWrite mode) {
  const Shape& q = s.inputs[0];
  const auto batch = static_cast<std::uint32_t>(q.dim(0));
  const auto seq = static_cast<std::uint32_t>(q.dim(1));
  // Value heads index the state, v, alpha and beta; q and k carry
  // key_heads, each shared by heads / key_heads value heads.
  const auto heads = static_cast<std::uint32_t>(s.inputs[2].dim(2));
  const auto key_heads = static_cast<std::uint32_t>(q.dim(2));
  const auto D = static_cast<std::uint32_t>(q.dim(3));
  if (key_heads == 0 || heads % key_heads != 0 || s.inputs[1].dim(2) != q.dim(2)) return {};
  const auto share = heads / key_heads;
  const bool write_state = mode != GdnWrite::kOut;
  const bool write_out = mode != GdnWrite::kState;
  if (seq == 0 || heads == 0 || D == 0) return {};
  if (write_out && !s.store) return {};
  if (mode == GdnWrite::kBoth && s.inputs.size() < 7) return {};

  const std::uint32_t wave = wave_of(s.device);
  const std::uint32_t tile = (D + wave - 1) / wave;
  // One scan per (sequence, head, state row), never one per output element.
  // The recurrence has to be walked from the start to reach any timestep, so a
  // thread per timestep replays the whole thing and writes one value of it:
  // seq times the work, and seq here is the prompt.
  const std::uint32_t rows = batch * heads * D;

  kir::KernelBody k(s.types, *s.intrinsics);
  k.set_store(s.store);
  GdnArgs<env::Emit> a;
  if (!env::bind(k, a, s)) return {};
  env::Emit e{&k};

  const auto i = e.thread_id();
  const auto lane = e.let(i % wave);
  const auto wid = e.let(i / wave);
  (void)e.ret_if(wid >= rows);

  const auto row = e.let(wid % D);
  const auto h = e.let((wid / D) % heads);
  const auto b = e.let(wid / (D * heads));

  // Each lane owns at most four state scalars. Record these as mutable IR
  // locals so every dialect sees their definitions and loop-carried updates;
  // a raw C array declaration leaves only an unresolved storage symbol in IR.
  std::vector<kir::LValue<kir::f32>> srow;
  srow.reserve(tile);
  for (std::uint32_t ei = 0; ei < tile; ++ei) {
    srow.emplace_back(e.var(0.0f));
  }

  for (std::uint32_t ei = 0; ei < tile; ++ei) {
    const auto j = e.let(lane + ei * wave);
    const auto idx = ((b * heads + h) * D + row) * D + j;
    if (auto in = e.when(j < D)) {
      srow[ei] = a.s[idx];
    }
  }

  auto reduce = [&](kir::Val<kir::f32> acc) {
    for (std::uint32_t m = 1; m < wave; m <<= 1) {
      acc = e.let(acc + math::shfl_xor(acc, e.u32(m)));
    }
    return acc;
  };

  // One timestep of the recurrence, given that step's inputs already in
  // registers.
  struct StepIn {
    kir::Val<kir::f32> al, bt, v;
    std::vector<kir::Val<kir::f32>> k, q;
  };
  const auto load_step = [&](const kir::Val<kir::u32>& t) {
    StepIn in;
    const auto sc = e.let((b * seq + t) * heads + h);
    const auto vec = e.let(sc * D);
    const auto kvec = e.let(((b * seq + t) * key_heads + h / share) * D);
    in.al = e.let(a.alpha[sc]);
    in.bt = e.let(a.beta[sc]);
    in.v = e.let(a.v[vec + row]);
    for (std::uint32_t ei = 0; ei < tile; ++ei) {
      const auto j = e.let(lane + ei * wave);
      in.k.push_back(e.let(a.k[kvec + j]));
      in.q.push_back(e.let(a.q[kvec + j]));
    }
    return in;
  };
  const auto run_step = [&](const kir::Val<kir::u32>& t, const StepIn& in) {
    auto skp = e.var(0.0f);
    for (std::uint32_t ei = 0; ei < tile; ++ei) {
      srow[ei] = srow[ei].read() * in.al;
      skp = math::fma(srow[ei].read(), in.k[ei], skp);
    }
    const auto sk = reduce(skp);
    const auto delta = e.let((in.v - sk) * in.bt);
    auto accp = e.var(0.0f);
    for (std::uint32_t ei = 0; ei < tile; ++ei) {
      srow[ei] = math::fma(delta, in.k[ei], srow[ei].read());
      accp = math::fma(srow[ei].read(), in.q[ei], accp);
    }
    if (write_out) {
      // The scan passes every timestep on its way to the end, so it publishes
      // each one as it goes rather than being restarted to reach it.
      const auto acc = reduce(accp);
      if (auto in_lane = e.when(lane == 0)) {
        e.store(((b * seq + t) * heads + h) * D + row, acc);
      }
    }
  };

  const bool preload = seq <= kGdnPreloadSteps && tile <= 4;
  std::vector<kir::LValue<kir::f32>> pre_al, pre_bt, pre_v;
  std::vector<std::vector<kir::LValue<kir::f32>>> pre_k, pre_q;

  if (D % wave == 0) {
    // Every lane owns whole elements, so no step needs a guard and a block
    // of steps can read all its inputs before the first dependent update:
    // the loads do not depend on the state, and issuing them together pays
    // their latency once per block instead of once per step.
    const auto block = [&](const kir::Val<kir::u32>& t0, std::uint32_t steps) {
      std::vector<StepIn> ins;
      ins.reserve(steps);
      for (std::uint32_t u = 0; u < steps; ++u)
        ins.push_back(load_step(e.let(t0 + u)));
      for (std::uint32_t u = 0; u < steps; ++u)
        run_step(e.let(t0 + u), ins[u]);
    };
    // Whole blocks in a loop; the remainder, shorter than a block, in
    // straight-line code. A loop is emitted only when it repeats: a
    // one-trip loop and a short tail loop beside it are a control shape the
    // target's branch lowering refuses.
    const std::uint32_t blocks = seq / kStepBlock;
    if (blocks > 1) {
      for (auto tb : e.range(0u, blocks, 1u)) block(e.let(tb * kStepBlock), kStepBlock);
    } else if (blocks == 1) {
      block(e.u32(0), kStepBlock);
    }
    if (const std::uint32_t tail = seq - blocks * kStepBlock; tail != 0)
      block(e.u32(blocks * kStepBlock), tail);
  } else if (preload) {
    // A verify-width scan whose head width leaves some lanes without an
    // element: every timestep's inputs are loaded up front, behind the lane
    // guard, before the first state update (the recurrence, and its
    // arithmetic order, are the loop below).
    for (std::uint32_t t = 0; t < seq; ++t) {
      const auto sc = e.let((b * seq + t) * heads + h);
      const auto vec = e.let(sc * D);
      const auto kvec = e.let(((b * seq + t) * key_heads + h / share) * D);
      pre_al.push_back(e.var(a.alpha[sc]));
      pre_bt.push_back(e.var(a.beta[sc]));
      pre_v.push_back(e.var(a.v[vec + row]));
      pre_k.emplace_back(); pre_q.emplace_back();
      for (std::uint32_t ei = 0; ei < tile; ++ei) {
        const auto j = e.let(lane + ei * wave);
        pre_k.back().push_back(e.var(0.0f));
        pre_q.back().push_back(e.var(0.0f));
        if (auto in = e.when(j < D)) {
          pre_k.back()[ei] = a.k[kvec + j];
          pre_q.back()[ei] = a.q[kvec + j];
        }
      }
    }
    for (std::uint32_t t = 0; t < seq; ++t) {
      const auto al = e.let(pre_al[t].read());
      auto skp = e.var(0.0f);
      for (std::uint32_t ei = 0; ei < tile; ++ei) {
        const auto j = lane + ei * wave;
        srow[ei] = srow[ei].read() * al;
        if (auto in = e.when(j < D)) {
          skp = math::fma(srow[ei].read(), pre_k[t][ei].read(), skp);
        }
      }
      const auto sk = reduce(skp);
      const auto bt = e.let(pre_bt[t].read());
      const auto delta = e.let((pre_v[t].read() - sk) * bt);
      auto accp = e.var(0.0f);
      for (std::uint32_t ei = 0; ei < tile; ++ei) {
        const auto j = lane + ei * wave;
        if (auto in = e.when(j < D)) {
          srow[ei] = math::fma(delta, pre_k[t][ei].read(), srow[ei].read());
          accp = math::fma(srow[ei].read(), pre_q[t][ei].read(), accp);
        }
      }
      if (write_out) {
        const auto acc = reduce(accp);
        if (auto in = e.when(lane == 0)) {
          e.store(((b * seq + t) * heads + h) * D + row, acc);
        }
      }
    }
  } else {
  for (auto t : e.range(seq)) {
    const auto sc = (b * seq + t) * heads + h;
    const auto vec = sc * D;
    const auto kvec = e.let(((b * seq + t) * key_heads + h / share) * D);
    const auto al = e.let(a.alpha[sc]);
    auto skp = e.var(0.0f);
    for (std::uint32_t ei = 0; ei < tile; ++ei) {
      const auto j = lane + ei * wave;
      srow[ei] = srow[ei].read() * al;
      if (auto in = e.when(j < D)) {
        skp = math::fma(srow[ei].read(), a.k[kvec + j], skp);
      }
    }
    const auto sk = reduce(skp);
    const auto bt = e.let(a.beta[sc]);
    const auto delta = e.let((a.v[vec + row] - sk) * bt);
    auto accp = e.var(0.0f);
    for (std::uint32_t ei = 0; ei < tile; ++ei) {
      const auto j = lane + ei * wave;
      if (auto in = e.when(j < D)) {
        srow[ei] = math::fma(delta, a.k[kvec + j], srow[ei].read());
        accp = math::fma(srow[ei].read(), a.q[kvec + j], accp);
      }
    }
    if (write_out) {
      const auto acc = reduce(accp);
      if (auto in = e.when(lane == 0)) {
        e.store(((b * seq + t) * heads + h) * D + row, acc);
      }
    }
  }
  }

  if (write_state) {
    for (std::uint32_t ei = 0; ei < tile; ++ei) {
      const auto j = e.let(lane + ei * wave);
      if (auto in = e.when(j < D)) {
        const auto idx = ((b * heads + h) * D + row) * D + j;
        if (mode == GdnWrite::kBoth) a.sout[idx] = srow[ei].read();
        else e.store(idx, srow[ei].read());
      }
    }
  }
  return k.str();
}

// The prefill form: one thread per state row, the whole row in registers.
//
// Every row of S evolves independently -- decay, the row's own dot with k,
// its own delta, its own dot with q -- so a thread that owns a row needs no
// cross-lane reduction at all, where the wave-per-row form spends two
// five-step shuffle trees per timestep on each row. A workgroup owns one
// head's rows; the head's k and q for a block of timesteps are staged in
// workgroup scratch once and read there by every row as broadcasts.
// Selected for prompts of at least kRowScanMinSeq timesteps (decode keeps
// the wave form) and a row that fits the register budget.
constexpr std::uint32_t kRowScanMinSeq = 16;
constexpr std::uint32_t kRowScanMaxDim = 128;
constexpr std::uint32_t kRowScanBlock = 16;
// Lanes sharing one row: each holds D / kRowScanLanes of it, and the two
// dots per step finish with one shuffle per halving. Two lanes put eight
// waves on a 128-row head, which hides the per-step dependence chain that a
// single wave per SIMD leaves exposed.
constexpr std::uint32_t kRowScanLanes = 2;
// Rows per workgroup. Rows are independent, so a head's rows are spread over
// several workgroups, each staging the head's k and q for itself: more,
// smaller workgroups put more waves in flight than one per head does
// (measured on gfx1201 at 1024 steps and 48 heads: 64 rows 1.57 ms, 32 rows
// 1.68, a whole head of 128 rows 1.90, the wave-per-row scan 1.80).
constexpr std::uint32_t kRowScanRows = 64;
std::uint32_t row_scan_rows(std::uint32_t D) {
  return D % kRowScanRows == 0 && kRowScanRows * kRowScanLanes >= kRowScanBlock
             ? kRowScanRows : D;
}

bool row_scan_fits(const KernelShapes& s) {
  if (s.inputs.size() < 6 || s.inputs[0].rank() != 4 || !s.device) return false;
  const auto seq = s.inputs[0].dim(1);
  const auto D = s.inputs[0].dim(3);
  const std::uint32_t wave = wave_of(s.device);
  return seq >= kRowScanMinSeq && D > 0 && D <= kRowScanMaxDim &&
         D % wave == 0 && D % (4 * kRowScanLanes) == 0 &&
         static_cast<std::uint32_t>(D) * kRowScanLanes <= s.device->max_threads_per_workgroup &&
         !s.intrinsics->find("wave.shfl_xor").empty() &&
         backend::workgroup_lds_bytes(s.device) >=
             (2u * kRowScanBlock * static_cast<std::uint32_t>(D) + 2u * kRowScanBlock) * 4u &&
         s.intrinsics != nullptr && !s.intrinsics->find("barrier").empty();
}

std::string emit_gdn_rows(const KernelShapes& s, GdnWrite mode) {
  const Shape& q = s.inputs[0];
  const auto batch = static_cast<std::uint32_t>(q.dim(0));
  const auto seq = static_cast<std::uint32_t>(q.dim(1));
  // Value heads index the state, v, alpha and beta; q and k carry key_heads,
  // each shared by heads / key_heads value heads.
  const auto heads = static_cast<std::uint32_t>(s.inputs[2].dim(2));
  const auto key_heads = static_cast<std::uint32_t>(q.dim(2));
  const auto D = static_cast<std::uint32_t>(q.dim(3));
  if (key_heads == 0 || heads % key_heads != 0 || s.inputs[1].dim(2) != q.dim(2)) return {};
  const auto share = heads / key_heads;
  const bool write_state = mode != GdnWrite::kOut;
  const bool write_out = mode != GdnWrite::kState;
  if (write_out && !s.store) return {};
  if (mode == GdnWrite::kBoth && s.inputs.size() < 7) return {};
  (void)batch;

  kir::KernelBody k(s.types, *s.intrinsics, backend::workgroup_lds_bytes(s.device));
  k.set_store(s.store);
  GdnArgs<env::Emit> a;
  if (!env::bind(k, a, s)) return {};
  env::Emit e{&k};
  const auto ks = e.lds<kir::f32>(kRowScanBlock * D);
  const auto qs = e.lds<kir::f32>(kRowScanBlock * D);
  const auto als = e.lds<kir::f32>(kRowScanBlock);
  const auto bts = e.lds<kir::f32>(kRowScanBlock);
  if (!ks || !qs || !als || !bts) return {};

  constexpr std::uint32_t L = kRowScanLanes;
  const std::uint32_t E = D / L;  // elements of the row this lane holds
  const std::uint32_t R = row_scan_rows(D);
  const std::uint32_t groups_per_head = D / R;
  const std::uint32_t threads = R * L;
  const auto lid = e.let(math::local_id());
  const auto wg = e.let(math::workgroup_id_x());
  const auto row = e.let((wg % groups_per_head) * R + lid / L);
  const auto part_of_row = e.let(lid % L);
  const auto col0 = e.let(part_of_row * E);
  const auto h = e.let((wg / groups_per_head) % heads);
  const auto b = e.let(wg / (groups_per_head * heads));
  const auto state_row = e.let(((b * heads + h) * D + row) * D + col0);

  std::vector<kir::LValue<kir::f32>> srow;
  srow.reserve(E);
  for (std::uint32_t j = 0; j < E; j += 4) {
    const auto p = e.load(a.s, e.let(state_row + j), 16u);
    for (int u = 0; u < 4; ++u) srow.emplace_back(e.var(p[u]));
  }
  // The row's dot from its lanes' partial sums; every lane gets the total.
  const auto row_sum = [&](kir::Val<kir::f32> x) {
    for (std::uint32_t m = 1; m < L; m <<= 1) x = e.let(x + math::shfl_xor(x, e.u32(m)));
    return x;
  };

  // Stage kRowScanBlock timesteps from t0: thread `row` moves element `row`
  // of each step's k and q, and the first threads each step's decay and
  // beta. A step past the end reads a real step's vectors (its index wraps)
  // and gets decay 1 and beta 0, which leaves every row exactly as it was.
  const auto stage = [&](const kir::Val<kir::u32>& t0) {
    e.barrier();
    // The workgroup's threads move the block's k and q, element idx % D of
    // step idx / D, a workgroup's worth of elements per pass.
    for (std::uint32_t i0 = 0; i0 < kRowScanBlock * D; i0 += threads) {
      const auto idx = e.let(lid + i0);
      const auto u = e.let(idx / D);
      const auto j = e.let(idx % D);
      const auto tt = e.let((t0 + u) % seq);
      const auto kvec = e.let(((b * seq + tt) * key_heads + h / share) * D);
      ks[e.let(j + u * D)] = a.k[e.let(kvec + j)];
      qs[e.let(j + u * D)] = a.q[e.let(kvec + j)];
    }
    if (auto lead = e.when(lid < kRowScanBlock)) {
      const auto live = e.let(t0 + lid < seq);
      const auto sc = e.let((b * seq + (t0 + lid) % seq) * heads + h);
      als[lid] = select(live, a.alpha[sc], e.f32(1.0f));
      bts[lid] = select(live, a.beta[sc], e.f32(0.0f));
    }
    e.barrier();
  };
  const auto step = [&](const kir::Val<kir::u32>& t, const kir::Val<kir::u32>& u) {
    const auto sc = e.let((b * seq + t % seq) * heads + h);
    const auto al = e.let(als[u].read());
    const auto bt = e.let(bts[u].read());
    const auto vt = e.let(a.v[e.let(sc * D + row)]);
    const auto base = e.let(u * D + col0);
    // Four partial sums keep the dot's dependence chain a quarter as long.
    std::array<kir::LValue<kir::f32>, 4> part{e.var(0.0f), e.var(0.0f),
                                              e.var(0.0f), e.var(0.0f)};
    std::vector<kir::Val<kir::f32>> kj;
    kj.reserve(E);
    for (std::uint32_t j = 0; j < E; j += 4) {
      const auto kp = ks.load(e.let(base + j), 16u);
      for (int x = 0; x < 4; ++x) {
        kj.push_back(e.let(kp[x]));
        srow[j + x] = srow[j + x].read() * al;
        part[x] = math::fma(srow[j + x].read(), kj.back(), part[x].read());
      }
    }
    const auto sk = row_sum(e.let((part[0].read() + part[1].read()) +
                                  (part[2].read() + part[3].read())));
    const auto delta = e.let((vt - sk) * bt);
    for (int x = 0; x < 4; ++x) part[x] = 0.0f;
    for (std::uint32_t j = 0; j < E; j += 4) {
      const auto qp = qs.load(e.let(base + j), 16u);
      for (int x = 0; x < 4; ++x) {
        srow[j + x] = math::fma(delta, kj[j + x], srow[j + x].read());
        part[x] = math::fma(srow[j + x].read(), qp[x], part[x].read());
      }
    }
    if (write_out) {
      const auto o = row_sum(e.let((part[0].read() + part[1].read()) +
                                   (part[2].read() + part[3].read())));
      // One compare for both conditions: lane 0 of the row (part 0) at a
      // live step. part * seq + t < seq exactly when part is 0 and t < seq.
      if (auto in = e.when(part_of_row * seq + t < seq)) e.store(e.let(sc * D + row), o);
    }
  };

  // One loop over whole blocks; the last block's surplus steps are inert.
  // A second loop for a remainder doubles the live state the target has to
  // allocate across loops, and it spills.
  const std::uint32_t blocks = (seq + kRowScanBlock - 1u) / kRowScanBlock;
  for (auto tb : e.range(0u, blocks, 1u)) {
    const auto t0 = e.let(tb * kRowScanBlock);
    stage(t0);
    for (auto u : e.range(0u, kRowScanBlock, 1u)) step(e.let(t0 + u), u);
  }

  if (write_state) {
    for (std::uint32_t j = 0; j < E; ++j) {
      const auto idx = e.let(state_row + j);
      if (mode == GdnWrite::kBoth) a.sout[idx] = srow[j].read();
      else e.store(idx, srow[j].read());
    }
  }
  return k.str();
}

ThreadPlan gdn_plan(const KernelShapes& s, bool write_state) {
  ThreadPlan tp;
  tp.workgroup_size[0] = kBlock;
  if (s.inputs.empty() || s.inputs[0].rank() != 4) {
    tp.workgroup_count[0] = 1;
    return tp;
  }
  const Shape& q = s.inputs[0];
  const auto batch = static_cast<std::uint32_t>(q.dim(0));
  const auto heads = static_cast<std::uint32_t>(
      s.inputs.size() > 2 && s.inputs[2].rank() == 4 ? s.inputs[2].dim(2) : q.dim(2));
  const auto D = static_cast<std::uint32_t>(q.dim(3));
  const std::uint32_t wave = wave_of(s.device);
  (void)write_state;
  if (row_scan_fits(s)) {
    tp.workgroup_size[0] = row_scan_rows(D) * kRowScanLanes;
    tp.workgroup_count[0] = batch * heads * (D / row_scan_rows(D));
    tp.lds_bytes = (2u * kRowScanBlock * D + 2u * kRowScanBlock) * 4u;
    return tp;
  }
  const std::uint32_t rows = batch * heads * D;
  const std::uint32_t threads = rows * wave;
  tp.workgroup_count[0] = threads == 0 ? 1u : (threads + kBlock - 1) / kBlock;
  return tp;
}

}  // namespace

struct GdnKernel final : KernelPrimitive<GdnKernel> {
  static constexpr std::string_view kName = "gdn_chunk_scan";
  static constexpr std::string_view kEntry = "lse_gdn";
  static constexpr std::string_view kSource = {};

  std::size_t arity() const noexcept override { return 6; }
  bool owns_indexing() const noexcept override { return true; }

  std::string emit_kernel(const KernelShapes& s) const override {
    if (s.inputs.size() < 6 || s.types.scalar == nullptr ||
        s.intrinsics == nullptr || s.inputs[0].rank() != 4) {
      return {};
    }
    const auto dim = s.inputs[0].dim(3);
    if (dim != 16 && dim != 32 && dim != 64 && dim != 128) return {};
    const GdnWrite mode = s.iattrs[0] != 0 ? GdnWrite::kState : GdnWrite::kOut;
    return row_scan_fits(s) ? emit_gdn_rows(s, mode) : emit_gdn(s, mode);
  }

  Result<Shape> infer_shape(std::span<const Shape> in) const override {
    if (in.size() < 6) {
      return LSE_ERROR(kInvalidArgument, "gdn_chunk_scan needs 6 inputs");
    }
    return in[2];
  }
  DType infer_dtype(std::span<const DType> in) const override {
    return in.empty() ? DType::kF32 : in[0];
  }
  static ThreadPlan plan_impl(const KernelShapes& s) {
    return gdn_plan(s, s.iattrs[0] != 0);
  }
};
LSE_REGISTER_PRIMITIVE(GdnKernel);

struct GdnPairKernel final : KernelPrimitive<GdnPairKernel> {
  static constexpr std::string_view kName = "gdn_chunk_scan.pair";
  static constexpr std::string_view kEntry = "lse_gdn_pair";
  static constexpr std::string_view kSource = {};

  std::size_t arity() const noexcept override { return 7; }
  bool owns_indexing() const noexcept override { return true; }

  std::string emit_kernel(const KernelShapes& s) const override {
    if (s.inputs.size() < 7 || s.types.scalar == nullptr ||
        s.intrinsics == nullptr || s.inputs[0].rank() != 4) {
      return {};
    }
    const auto dim = s.inputs[0].dim(3);
    if (dim != 16 && dim != 32 && dim != 64 && dim != 128) return {};
    return row_scan_fits(s) ? emit_gdn_rows(s, GdnWrite::kBoth)
                            : emit_gdn(s, GdnWrite::kBoth);
  }

  Result<Shape> infer_shape(std::span<const Shape> in) const override {
    if (in.size() < 3) return LSE_ERROR(kInvalidArgument, "gdn pair needs q, k and v");
    return in[2];
  }
  DType infer_dtype(std::span<const DType> in) const override {
    return in.empty() ? DType::kF32 : in[0];
  }
  static ThreadPlan plan_impl(const KernelShapes& s) {
    return gdn_plan(s, true);
  }
};

bool is_gdn_node(const Node* n) {
  return n != nullptr && n->kind == OpKind::kGDNChunkScan;
}

bool same_gdn_inputs(const Node& a, const Node& b) {
  if (a.inputs.size() != 6 || b.inputs.size() != 6) return false;
  for (std::size_t i = 0; i < 6; ++i) {
    if (a.inputs[i].get() != b.inputs[i].get()) return false;
  }
  return true;
}

const graph::KernelPrimitiveBase* gdn_pair_kernel() {
  static const GdnPairKernel k;
  return &k;
}

LinkedBinding gdn_pair_bindings(const FusionGroup& group) {
  LinkedBinding b;
  const Node* o = nullptr;
  const Node* st = nullptr;
  for (const NodePtr& n : group.nodes) {
    if (is_gdn_node(n.get())) {
      if (n->iattrs[0] == 0) o = n.get();
      else st = n.get();
      continue;
    }
    if (dynamic_cast<const KernelPrimitiveBase*>(n->prim) != nullptr) {
      return b;
    }
  }
  if (o == nullptr || st == nullptr || !same_gdn_inputs(*o, *st)) return b;
  b.inputs = {o->inputs[0].get(), o->inputs[1].get(), o->inputs[2].get(),
              o->inputs[3].get(), o->inputs[4].get(), o->inputs[5].get(), st};
  b.sink = o;
  b.ok = true;
  return b;
}


// ---------------------------------------------------------------------------
// Draft trees (runtime::DraftTree).
//
// A tree verify pass carries its nodes as rows in depth-first preorder, so a
// row's parent is the most recent row one level up. The scan keeps one state
// per depth -- level 0 the state the pass starts from, level d + 1 the state
// after the latest row at depth d -- and each row continues from its parent's
// level and leaves its own result one level down. That is the state-resident
// serial form SpecLA (arXiv 2607.16673) uses for chains, applied to a tree, and
// the per-node recurrence TreeWY (arXiv 2608.20961) writes in closed form: no
// per-node state is stored. A row's arithmetic is the wave-per-row scan's
// (emit_gdn) step for step, so a node's output is bit-identical to the chain
// scan over its path.
//
// Path descriptors (gdn.path_state, conv_tail.rows, kv_page_write.rows):
// f32 [2 + rows]: {count, first position, row 0, row 1, ...}.

namespace {

// Depth levels the tree scan keeps: the starting state and one per depth,
// rows of depth 0..8 (model::TreeLayout::kMaxDepth).
constexpr std::uint32_t kTreeLevels = 10;

template <class E>
struct GdnTreeArgs {
  env::In<kir::f32, E> q;
  env::In<kir::f32, E> k;
  env::In<kir::f32, E> v;
  env::In<kir::f32, E> alpha;
  env::In<kir::f32, E> beta;
  env::In<kir::f32, E> s;
  env::In<kir::f32, E> tree;   // [T] depth of each row
  env::Out<kir::f32, E> out;
};

template <class E>
struct GdnPathArgs {
  env::In<kir::f32, E> k;
  env::In<kir::f32, E> v;
  env::In<kir::f32, E> alpha;
  env::In<kir::f32, E> beta;
  env::In<kir::f32, E> s;
  env::In<kir::f32, E> path;   // {count, first, rows...}
  env::Out<kir::f32, E> out;
};

bool gdn_tree_shapes(const KernelShapes& s, std::size_t q_index) {
  if (!s.device || !s.types.scalar || !s.intrinsics || !s.store ||
      s.inputs.size() != q_index + 6 || s.intrinsics->find("wave.shfl_xor").empty())
    return false;
  for (const DType dtype : s.input_dtypes)
    if (dtype != DType::kF32) return false;
  const Shape& k = s.inputs[q_index];
  const Shape& v = s.inputs[q_index + 1];
  const Shape& st = s.inputs[q_index + 4];
  if (k.rank() != 4 || v.rank() != 4 || st.rank() != 4 || k.dim(0) != 1 || v.dim(0) != 1 ||
      k.dim(1) != v.dim(1) || k.dim(3) != v.dim(3) || k.dim(2) == 0 || v.dim(2) % k.dim(2) != 0 ||
      st != Shape{1, v.dim(2), v.dim(3), v.dim(3)} ||
      s.inputs[q_index + 2] != Shape{1, v.dim(1), v.dim(2)} ||
      s.inputs[q_index + 3] != Shape{1, v.dim(1), v.dim(2)})
    return false;
  const auto D = static_cast<std::uint32_t>(v.dim(3));
  return D % wave_of(s.device) == 0 && D / wave_of(s.device) <= 4;
}

std::string emit_gdn_tree(const KernelShapes& s) {
  if (!gdn_tree_shapes(s, 1) || s.inputs[0] != s.inputs[1]) return {};
  const Shape& q = s.inputs[0];
  const auto seq = static_cast<std::uint32_t>(q.dim(1));
  const auto heads = static_cast<std::uint32_t>(s.inputs[2].dim(2));
  const auto key_heads = static_cast<std::uint32_t>(q.dim(2));
  const auto D = static_cast<std::uint32_t>(q.dim(3));
  const auto share = heads / key_heads;
  if (s.inputs[6].elem_count() < seq || s.output != s.inputs[2]) return {};
  const std::uint32_t wave = wave_of(s.device);
  const std::uint32_t tile = D / wave;

  kir::KernelBody k(s.types, *s.intrinsics);
  k.set_store(s.store);
  GdnTreeArgs<env::Emit> a;
  if (!env::bind(k, a, s)) return {};
  env::Emit e{&k};
  const auto i = e.thread_id();
  const auto lane = e.let(i % wave);
  const auto wid = e.let(i / wave);
  (void)e.ret_if(wid >= heads * D);
  const auto row = e.let(wid % D);
  const auto h = e.let(wid / D);

  std::vector<std::vector<kir::LValue<kir::f32>>> level(kTreeLevels);
  for (std::uint32_t l = 0; l < kTreeLevels; ++l)
    for (std::uint32_t ei = 0; ei < tile; ++ei) level[l].emplace_back(e.var(0.0f));
  for (std::uint32_t ei = 0; ei < tile; ++ei)
    level[0][ei] = a.s[e.let((h * D + row) * D + lane + ei * wave)];
  auto reduce = [&](kir::Val<kir::f32> acc) {
    for (std::uint32_t m = 1; m < wave; m <<= 1) acc = e.let(acc + math::shfl_xor(acc, e.u32(m)));
    return acc;
  };
  // A step's inputs do not depend on the state, so a block of steps loads
  // them all before the first update and pays their latency once (emit_gdn
  // does the same).
  struct StepIn {
    kir::Val<kir::u32> depth, sc;
    kir::Val<kir::f32> al, bt, vt;
    std::vector<kir::Val<kir::f32>> kj, qj;
  };
  const auto load_step = [&](const kir::Val<kir::u32>& t) {
    StepIn in{e.let(kir::cast<kir::u32>(a.tree[t])), e.let(t * heads + h), {}, {}, {}, {}, {}};
    const auto kvec = e.let((t * key_heads + h / share) * D);
    in.al = e.let(a.alpha[in.sc]);
    in.bt = e.let(a.beta[in.sc]);
    in.vt = e.let(a.v[in.sc * D + row]);
    for (std::uint32_t ei = 0; ei < tile; ++ei) {
      in.kj.push_back(e.let(a.k[kvec + lane + ei * wave]));
      in.qj.push_back(e.let(a.q[kvec + lane + ei * wave]));
    }
    return in;
  };
  const auto run_step = [&](const StepIn& in) {
    const auto& depth = in.depth;
    const auto& sc = in.sc;
    const auto& al = in.al;
    const auto& bt = in.bt;
    const auto& vt = in.vt;
    const auto& kj = in.kj;
    const auto& qj = in.qj;
    // The parent's state: the level of this row's depth.
    std::vector<kir::LValue<kir::f32>> srow;
    for (std::uint32_t ei = 0; ei < tile; ++ei) {
      srow.emplace_back(e.var(level[0][ei].read()));
      for (std::uint32_t l = 1; l < kTreeLevels; ++l)
        srow[ei] = select(depth == l, level[l][ei].read(), srow[ei].read());
    }
    auto skp = e.var(0.0f);
    for (std::uint32_t ei = 0; ei < tile; ++ei) {
      srow[ei] = srow[ei].read() * al;
      skp = math::fma(srow[ei].read(), kj[ei], skp);
    }
    const auto sk = reduce(skp);
    const auto delta = e.let((vt - sk) * bt);
    auto accp = e.var(0.0f);
    for (std::uint32_t ei = 0; ei < tile; ++ei) {
      srow[ei] = math::fma(delta, kj[ei], srow[ei].read());
      accp = math::fma(srow[ei].read(), qj[ei], accp);
    }
    const auto acc = reduce(accp);
    if (auto first = e.when(lane == 0u)) e.store(e.let(sc * D + row), acc);
    // This row's state is where its children continue from.
    for (std::uint32_t l = 1; l < kTreeLevels; ++l)
      for (std::uint32_t ei = 0; ei < tile; ++ei)
        level[l][ei] = select(depth + 1u == l, srow[ei].read(), level[l][ei].read());
  };
  const auto block = [&](const kir::Val<kir::u32>& t0, std::uint32_t steps) {
    std::vector<StepIn> ins;
    ins.reserve(steps);
    for (std::uint32_t u = 0; u < steps; ++u) ins.push_back(load_step(e.let(t0 + u)));
    for (std::uint32_t u = 0; u < steps; ++u) run_step(ins[u]);
  };
  const std::uint32_t blocks = seq / kStepBlock;
  if (blocks > 1) {
    for (auto tb : e.range(0u, blocks, 1u)) block(e.let(tb * kStepBlock), kStepBlock);
  } else if (blocks == 1) {
    block(e.u32(0), kStepBlock);
  }
  if (const std::uint32_t tail = seq - blocks * kStepBlock; tail != 0)
    block(e.u32(blocks * kStepBlock), tail);
  return k.str();
}

std::string emit_gdn_path(const KernelShapes& s) {
  if (!gdn_tree_shapes(s, 0)) return {};
  const Shape& kk = s.inputs[0];
  const auto seq = static_cast<std::uint32_t>(kk.dim(1));
  const auto heads = static_cast<std::uint32_t>(s.inputs[1].dim(2));
  const auto key_heads = static_cast<std::uint32_t>(kk.dim(2));
  const auto D = static_cast<std::uint32_t>(kk.dim(3));
  const auto share = heads / key_heads;
  const auto most = static_cast<std::uint32_t>(s.inputs[5].elem_count()) - 2u;
  if (s.inputs[5].elem_count() < 3 || most > seq || s.output != s.inputs[4]) return {};
  const std::uint32_t wave = wave_of(s.device);
  const std::uint32_t tile = D / wave;

  kir::KernelBody k(s.types, *s.intrinsics);
  k.set_store(s.store);
  GdnPathArgs<env::Emit> a;
  if (!env::bind(k, a, s)) return {};
  env::Emit e{&k};
  const auto i = e.thread_id();
  const auto lane = e.let(i % wave);
  const auto wid = e.let(i / wave);
  (void)e.ret_if(wid >= heads * D);
  const auto row = e.let(wid % D);
  const auto h = e.let(wid / D);
  const auto count = e.let(kir::cast<kir::u32>(a.path[0u]));
  std::vector<kir::LValue<kir::f32>> srow;
  for (std::uint32_t ei = 0; ei < tile; ++ei)
    srow.emplace_back(e.var(a.s[e.let((h * D + row) * D + lane + ei * wave)]));
  auto reduce = [&](kir::Val<kir::f32> acc) {
    for (std::uint32_t m = 1; m < wave; m <<= 1) acc = e.let(acc + math::shfl_xor(acc, e.u32(m)));
    return acc;
  };
  for (std::uint32_t step = 0; step < most; ++step) {
    if (auto live = e.when(count > step)) {
      const auto t = e.let(kir::cast<kir::u32>(a.path[2u + step]));
      const auto sc = e.let(t * heads + h);
      const auto kvec = e.let((t * key_heads + h / share) * D);
      const auto al = e.let(a.alpha[sc]);
      const auto bt = e.let(a.beta[sc]);
      const auto vt = e.let(a.v[sc * D + row]);
      std::vector<kir::Val<kir::f32>> kj;
      for (std::uint32_t ei = 0; ei < tile; ++ei) kj.push_back(e.let(a.k[kvec + lane + ei * wave]));
      auto skp = e.var(0.0f);
      for (std::uint32_t ei = 0; ei < tile; ++ei) {
        srow[ei] = srow[ei].read() * al;
        skp = math::fma(srow[ei].read(), kj[ei], skp);
      }
      const auto sk = reduce(skp);
      const auto delta = e.let((vt - sk) * bt);
      for (std::uint32_t ei = 0; ei < tile; ++ei)
        srow[ei] = math::fma(delta, kj[ei], srow[ei].read());
    }
  }
  for (std::uint32_t ei = 0; ei < tile; ++ei)
    e.store(e.let((h * D + row) * D + lane + ei * wave), srow[ei].read());
  return k.str();
}

ThreadPlan gdn_wave_rows_plan(const KernelShapes& s, std::size_t value_index) {
  ThreadPlan tp;
  tp.workgroup_size[0] = kBlock;
  const std::uint32_t wave = wave_of(s.device);
  const auto rows = s.inputs.size() > value_index && s.inputs[value_index].rank() == 4
      ? static_cast<std::uint32_t>(s.inputs[value_index].dim(2) * s.inputs[value_index].dim(3))
      : 1u;
  tp.workgroup_count[0] = (rows * wave + kBlock - 1) / kBlock;
  return tp;
}

// The host references, step for step: the device reduces each dot with a
// shuffle tree, so these agree with it to rounding, not bit for bit.
void host_gdn_step(float* srow, const float* k, const float* q, float v, float al,
                   float bt, std::size_t D, float* out) {
  float sk = 0;
  for (std::size_t j = 0; j < D; ++j) {
    srow[j] *= al;
    sk = std::fma(srow[j], k[j], sk);
  }
  const float delta = (v - sk) * bt;
  float acc = 0;
  for (std::size_t j = 0; j < D; ++j) {
    srow[j] = std::fma(delta, k[j], srow[j]);
    if (q) acc = std::fma(srow[j], q[j], acc);
  }
  if (out) *out = acc;
}

const float* host_f32(const HostTensorView& v) {
  return reinterpret_cast<const float*>(v.bytes.data());
}

}  // namespace

struct GdnTreeKernel final : KernelPrimitive<GdnTreeKernel> {
  static constexpr std::string_view kName = "gdn.tree_scan.v1";
  static constexpr std::string_view kEntry = "lse_gdn_tree_scan_v1";
  static constexpr std::string_view kSource = {};
  std::size_t arity() const noexcept override { return 7; }
  bool owns_indexing() const noexcept override { return true; }
  bool supports_epilogue() const noexcept override { return false; }
  std::string emit_kernel(const KernelShapes& s) const override { return emit_gdn_tree(s); }
  Result<Shape> infer_shape(std::span<const Shape> in) const override {
    if (in.size() != 7) return LSE_ERROR(kInvalidArgument, "gdn.tree_scan.v1 takes 7 inputs");
    return in[2];
  }
  DType infer_dtype(std::span<const DType>) const override { return DType::kF32; }
  static ThreadPlan plan_impl(const KernelShapes& s) { return gdn_wave_rows_plan(s, 2); }
  bool has_typed_host_impl() const noexcept override { return true; }
  Status eval_cpu_typed(std::span<const HostTensorView> in, HostOutputView out,
                        const std::array<float, 4>&,
                        const std::array<std::int32_t, 4>&) const override {
    if (in.size() != 7) return LSE_ERROR(kInvalidArgument, "gdn.tree_scan.v1 takes 7 inputs");
    const Shape& q = in[0].shape;
    const auto T = static_cast<std::size_t>(q.dim(1)), KH = static_cast<std::size_t>(q.dim(2)),
               D = static_cast<std::size_t>(q.dim(3)), H = static_cast<std::size_t>(in[2].shape.dim(2));
    const float *qq = host_f32(in[0]), *kk = host_f32(in[1]), *vv = host_f32(in[2]),
                *al = host_f32(in[3]), *bt = host_f32(in[4]), *s0 = host_f32(in[5]),
                *depth = host_f32(in[6]);
    auto* o = reinterpret_cast<float*>(out.bytes.data());
    std::vector<float> levels(kTreeLevels * D);
    for (std::size_t h = 0; h < H; ++h)
      for (std::size_t r = 0; r < D; ++r) {
        std::copy_n(s0 + (h * D + r) * D, D, levels.begin());
        for (std::size_t t = 0; t < T; ++t) {
          const auto d = static_cast<std::size_t>(depth[t]);
          if (d + 1 >= kTreeLevels) return LSE_ERROR(kInvalidArgument, "tree deeper than the scan's levels");
          std::vector<float> srow(levels.begin() + d * D, levels.begin() + (d + 1) * D);
          const std::size_t kv = (t * KH + h / (H / KH)) * D;
          host_gdn_step(srow.data(), kk + kv, qq + kv, vv[(t * H + h) * D + r], al[t * H + h],
                        bt[t * H + h], D, &o[(t * H + h) * D + r]);
          std::copy(srow.begin(), srow.end(), levels.begin() + (d + 1) * D);
        }
      }
    return OkStatus();
  }
};
LSE_REGISTER_PRIMITIVE(GdnTreeKernel);

struct GdnPathKernel final : KernelPrimitive<GdnPathKernel> {
  static constexpr std::string_view kName = "gdn.path_state.v1";
  static constexpr std::string_view kEntry = "lse_gdn_path_state_v1";
  static constexpr std::string_view kSource = {};
  std::size_t arity() const noexcept override { return 6; }
  bool owns_indexing() const noexcept override { return true; }
  bool supports_epilogue() const noexcept override { return false; }
  std::string emit_kernel(const KernelShapes& s) const override { return emit_gdn_path(s); }
  Result<Shape> infer_shape(std::span<const Shape> in) const override {
    if (in.size() != 6) return LSE_ERROR(kInvalidArgument, "gdn.path_state.v1 takes 6 inputs");
    return in[4];
  }
  DType infer_dtype(std::span<const DType>) const override { return DType::kF32; }
  static ThreadPlan plan_impl(const KernelShapes& s) { return gdn_wave_rows_plan(s, 1); }
  bool has_typed_host_impl() const noexcept override { return true; }
  Status eval_cpu_typed(std::span<const HostTensorView> in, HostOutputView out,
                        const std::array<float, 4>&,
                        const std::array<std::int32_t, 4>&) const override {
    if (in.size() != 6) return LSE_ERROR(kInvalidArgument, "gdn.path_state.v1 takes 6 inputs");
    const Shape& kshape = in[0].shape;
    const auto KH = static_cast<std::size_t>(kshape.dim(2)), D = static_cast<std::size_t>(kshape.dim(3)),
               H = static_cast<std::size_t>(in[1].shape.dim(2));
    const float *kk = host_f32(in[0]), *vv = host_f32(in[1]), *al = host_f32(in[2]),
                *bt = host_f32(in[3]), *s0 = host_f32(in[4]), *path = host_f32(in[5]);
    const auto count = static_cast<std::size_t>(path[0]);
    if (count + 2 > in[5].shape.elem_count())
      return LSE_ERROR(kInvalidArgument, "gdn.path_state.v1 path longer than its descriptor");
    auto* o = reinterpret_cast<float*>(out.bytes.data());
    for (std::size_t h = 0; h < H; ++h)
      for (std::size_t r = 0; r < D; ++r) {
        float* srow = o + (h * D + r) * D;
        std::copy_n(s0 + (h * D + r) * D, D, srow);
        for (std::size_t step = 0; step < count; ++step) {
          const auto t = static_cast<std::size_t>(path[2 + step]);
          const std::size_t kv = (t * KH + h / (H / KH)) * D;
          host_gdn_step(srow, kk + kv, nullptr, vv[(t * H + h) * D + r], al[t * H + h],
                        bt[t * H + h], D, nullptr);
        }
      }
    return OkStatus();
  }
};
LSE_REGISTER_PRIMITIVE(GdnPathKernel);

}  // namespace lse::kernels
