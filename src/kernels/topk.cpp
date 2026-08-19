#include "lse/graph/kernel_primitive.hpp"

#include <string>
#include <vector>

#include "lse/graph/kernel_args.hpp"
#include "lse/graph/kernel_env.hpp"
#include "lse/math.hpp"

namespace lse::kernels {

using namespace lse::graph;
namespace math = lse::math;

// AITER AdaptiveTopK: bitonic in registers when n is small (BlockTopkSort);
// radix's LDS histogram is a fixed cost that only wins at large n and K.
// MoE here is n=8, k=2 — always the bitonic side. n>64 falls back to
// iterative max+mask (Loom topk_gate), which is still O(k n).
namespace {

constexpr std::uint32_t kBitonicLimit = 64;

std::uint32_t bitonic_n(std::uint32_t n) {
  std::uint32_t p = 1;
  while (p < n) p <<= 1;
  return p;
}

void emit_swap(env::Emit& e, kir::LValue<kir::f32>& va, kir::LValue<kir::u32>& ia,
               kir::LValue<kir::f32>& vb, kir::LValue<kir::u32>& ib) {
  const auto tv = e.let(va.read());
  const auto ti = e.let(ia.read());
  va = vb.read();
  ia = ib.read();
  vb = tv;
  ib = ti;
}

// Descending: higher value first, smaller index on a tie.
void emit_cmp_swap(env::Emit& e, std::vector<kir::LValue<kir::f32>>& pv,
                   std::vector<kir::LValue<kir::u32>>& pi, int a, int b,
                   bool want_a_greater) {
  const auto va = pv[static_cast<std::size_t>(a)].read();
  const auto vb = pv[static_cast<std::size_t>(b)].read();
  const auto ia = pi[static_cast<std::size_t>(a)].read();
  const auto ib = pi[static_cast<std::size_t>(b)].read();
  const auto a_better = (va > vb) || (va == vb && ia < ib);
  const auto b_better = (vb > va) || (vb == va && ib < ia);
  if (auto g = e.when(want_a_greater ? b_better : a_better)) {
    emit_swap(e, pv[static_cast<std::size_t>(a)],
              pi[static_cast<std::size_t>(a)], pv[static_cast<std::size_t>(b)],
              pi[static_cast<std::size_t>(b)]);
  }
}

void emit_bitonic(env::Emit& e, std::vector<kir::LValue<kir::f32>>& pv,
                  std::vector<kir::LValue<kir::u32>>& pi, std::uint32_t n) {
  for (std::uint32_t ksz = 2; ksz <= n; ksz <<= 1) {
    for (std::uint32_t j = ksz >> 1; j > 0; j >>= 1) {
      for (std::uint32_t i = 0; i < n; ++i) {
        const std::uint32_t l = i ^ j;
        if (l <= i) continue;
        const bool ascending = (i & ksz) == 0;
        emit_cmp_swap(e, pv, pi, static_cast<int>(i), static_cast<int>(l),
                      !ascending);
      }
    }
  }
}

void emit_band(env::Emit& e, std::vector<kir::LValue<kir::f32>>& pv,
               std::uint32_t k, float score_band) {
  if (score_band >= 1.0f || k == 0) return;
  const auto thresh = e.let(pv[0].read() * e.f32(1.0f - score_band));
  auto total = e.var(1e-9f);
  for (std::uint32_t s = 0; s < k; ++s) {
    if (auto g = e.when(pv[s].read() < thresh)) {
      pv[s] = 0.0f;
    }
    total = total.read() + pv[s].read();
  }
  for (std::uint32_t s = 0; s < k; ++s) {
    pv[s] = pv[s].read() / total.read();
  }
}

}  // namespace

// Not self-indexing: the element comes back through `ret` and the emitter
// stores it, so `out` is bound but never written here.
template <class E>
struct TopKArgs {
  env::In<kir::f32, E> x;
  env::Out<kir::f32, E> out;
};

struct TopKKernel final : KernelPrimitive<TopKKernel> {
  static constexpr std::string_view kName = "topk";
  static constexpr std::string_view kEntry = "lse_topk";
  static constexpr std::string_view kSource = {};

  std::size_t arity() const noexcept override { return 1; }

  std::string emit_kernel(const KernelShapes& s) const override {
    if (s.inputs.size() != 1 || s.types.scalar == nullptr ||
        s.intrinsics == nullptr || s.inputs[0].rank() == 0 ||
        s.output.rank() == 0) {
      return {};
    }
    if (static_cast<std::size_t>(s.iattrs[0]) + 1 != s.inputs[0].rank()) {
      return {};
    }
    const auto n = static_cast<std::uint32_t>(
        s.inputs[0].dim(s.inputs[0].rank() - 1));
    const auto k = static_cast<std::uint32_t>(s.iattrs[1]);
    if (k == 0 || k > n) return {};

    const bool write_idx = s.iattrs[2] != 0;
    const float score_band = s.attrs[0] == 0.0f ? 1.0f : s.attrs[0];

    kir::KernelBody body(s.types, *s.intrinsics);
    TopKArgs<env::Emit> a;
    if (!env::bind(body, a, s)) return {};
    env::Emit e{&body};
    const auto i = e.thread_id();
    const auto row = e.let((i / k) * n);
    const auto slot = e.let(i % k);
    auto outv = e.var(0.0f);

    if (n <= kBitonicLimit) {
      const auto N = bitonic_n(n);
      std::vector<kir::LValue<kir::f32>> pv;
      std::vector<kir::LValue<kir::u32>> pi;
      pv.reserve(N);
      pi.reserve(N);
      for (std::uint32_t el = 0; el < N; ++el) {
        auto v = el < n ? e.var(a.x[row + el]) : e.var(math::neg_inf());
        pv.push_back(v);
        pi.push_back(e.var(e.u32(el)));
      }
      emit_bitonic(e, pv, pi, N);
      // Network is ascending (smallest at 0). Top-k sits at the high end.
      std::vector<kir::LValue<kir::f32>> topv;
      std::vector<kir::LValue<kir::u32>> topi;
      topv.reserve(k);
      topi.reserve(k);
      for (std::uint32_t sl = 0; sl < k; ++sl) {
        topv.push_back(pv[N - 1 - sl]);
        topi.push_back(pi[N - 1 - sl]);
      }
      if (!write_idx) emit_band(e, topv, k, score_band);
      for (std::uint32_t sl = 0; sl < k; ++sl) {
        if (auto g = e.when(slot == sl)) {
          outv = write_idx ? cast<kir::f32>(topi[sl].read()) : topv[sl].read();
        }
      }
      e.ret(outv.read());
      return body.str();
    }

    std::vector<kir::LValue<kir::u32>> win_e;
    std::vector<kir::LValue<kir::f32>> win_v;
    win_e.reserve(k);
    win_v.reserve(k);
    for (std::uint32_t p = 0; p < k; ++p) {
      auto bv = e.var(math::neg_inf());
      auto be = e.var(e.u32(0));
      for (auto el : e.range(n)) {
        auto taken = e.var(e.u32(0));
        for (std::uint32_t q = 0; q < p; ++q) {
          if (auto g = e.when(el == win_e[static_cast<std::size_t>(q)].read())) {
            taken = e.u32(1);
          }
        }
        const auto v = e.let(a.x[row + el]);
        if (auto g = e.when(taken.read() == 0u &&
                            (v > bv.read() ||
                             (v == bv.read() && el < be.read())))) {
          bv = v;
          be = el;
        }
      }
      win_e.push_back(be);
      win_v.push_back(bv);
    }
    if (!write_idx) emit_band(e, win_v, k, score_band);
    for (std::uint32_t sl = 0; sl < k; ++sl) {
      if (auto g = e.when(slot == sl)) {
        outv = write_idx ? cast<kir::f32>(win_e[sl].read()) : win_v[sl].read();
      }
    }
    e.ret(outv.read());
    return body.str();
  }

  Result<Shape> infer_shape(std::span<const Shape> in) const override {
    if (in.size() != 1) {
      return LSE_ERROR(kInvalidArgument, "topk takes 1 input");
    }
    return in[0];
  }
  DType infer_dtype(std::span<const DType> in) const override {
    return in.empty() ? DType::kF32 : in[0];
  }

  static ThreadPlan plan_impl(const KernelShapes& s) {
    ThreadPlan tp;
    const std::uint32_t threads =
        s.device && s.device->max_threads_per_workgroup >= 256 ? 256u : 64u;
    const auto elems = static_cast<std::uint32_t>(s.output.elem_count());
    tp.workgroup_size[0] = threads;
    tp.workgroup_count[0] = elems == 0 ? 1u : (elems + threads - 1) / threads;
    return tp;
  }
};
LSE_REGISTER_PRIMITIVE(TopKKernel);

}  // namespace lse::kernels
