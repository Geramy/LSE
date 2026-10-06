#include "lse/kernels/gdn.hpp"

#include "lse/graph/kernel_args.hpp"
#include "lse/graph/kernel_env.hpp"
#include "lse/graph/kernel_primitive.hpp"
#include "lse/math.hpp"
#include "lse/backends/hrx/device_info.hpp"

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

}  // namespace lse::kernels
