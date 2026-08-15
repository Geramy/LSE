#include "lse/graph/kernel_primitive.hpp"

#include <string>
#include <vector>

#include "lse/math.hpp"

namespace lse::backend::hrx_kernels {

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

void emit_swap(kir::KernelBody& body, kir::LValue<kir::f32>& va,
               kir::LValue<kir::u32>& ia, kir::LValue<kir::f32>& vb,
               kir::LValue<kir::u32>& ib, const std::string& tag) {
  auto tv = body.var<kir::f32>("tv" + tag, va.read());
  auto ti = body.var<kir::u32>("ti" + tag, ia.read());
  va = vb.read();
  ia = ib.read();
  vb = tv.read();
  ib = ti.read();
}

// Descending: higher value first, smaller index on a tie.
void emit_cmp_swap(kir::KernelBody& body, std::vector<kir::LValue<kir::f32>>& pv,
                   std::vector<kir::LValue<kir::u32>>& pi, int a, int b,
                   bool want_a_greater, const std::string& tag) {
  const auto va = pv[static_cast<std::size_t>(a)].read();
  const auto vb = pv[static_cast<std::size_t>(b)].read();
  const auto ia = pi[static_cast<std::size_t>(a)].read();
  const auto ib = pi[static_cast<std::size_t>(b)].read();
  const auto a_better = (va > vb) || (va == vb && ia < ib);
  const auto b_better = (vb > va) || (vb == va && ib < ia);
  body.when(want_a_greater ? b_better : a_better, [&] {
    emit_swap(body, pv[static_cast<std::size_t>(a)],
              pi[static_cast<std::size_t>(a)], pv[static_cast<std::size_t>(b)],
              pi[static_cast<std::size_t>(b)], tag);
  });
}

void emit_bitonic(kir::KernelBody& body, std::vector<kir::LValue<kir::f32>>& pv,
                  std::vector<kir::LValue<kir::u32>>& pi, std::uint32_t n) {
  int step = 0;
  for (std::uint32_t ksz = 2; ksz <= n; ksz <<= 1) {
    for (std::uint32_t j = ksz >> 1; j > 0; j >>= 1) {
      for (std::uint32_t i = 0; i < n; ++i) {
        const std::uint32_t l = i ^ j;
        if (l <= i) continue;
        const bool ascending = (i & ksz) == 0;
        emit_cmp_swap(body, pv, pi, static_cast<int>(i), static_cast<int>(l),
                      !ascending, std::to_string(step++));
      }
    }
  }
}

void emit_band(kir::KernelBody& body, std::vector<kir::LValue<kir::f32>>& pv,
               std::uint32_t k, float score_band) {
  if (score_band >= 1.0f || k == 0) return;
  const auto thresh =
      body.let<kir::f32>("thr", pv[0].read() * body.lit(1.0f - score_band));
  auto total = body.var<kir::f32>("tot", body.lit(1e-9f));
  for (std::uint32_t s = 0; s < k; ++s) {
    body.when(pv[s].read() < thresh, [&] { pv[s] = 0.0f; });
    total = total.read() + pv[s].read();
  }
  for (std::uint32_t s = 0; s < k; ++s) {
    pv[s] = pv[s].read() / total.read();
  }
}

}  // namespace

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
    const auto x = body.input<kir::f32>(0);
    const auto i = body.thread_id();
    const auto row = body.let<kir::u32>("row", (i / k) * n);
    const auto slot = body.let<kir::u32>("slot", i % k);
    auto outv = body.var<kir::f32>("outv", body.lit(0.0f));

    if (n <= kBitonicLimit) {
      const auto N = bitonic_n(n);
      std::vector<kir::LValue<kir::f32>> pv;
      std::vector<kir::LValue<kir::u32>> pi;
      pv.reserve(N);
      pi.reserve(N);
      for (std::uint32_t e = 0; e < N; ++e) {
        const std::string es = std::to_string(e);
        if (e < n) {
          pv.push_back(body.var<kir::f32>("pv" + es, x[row + e].read()));
          pi.push_back(body.var<kir::u32>("pi" + es, body.constant<kir::u32>(e)));
        } else {
          pv.push_back(body.var<kir::f32>("pv" + es, math::neg_inf()));
          pi.push_back(body.var<kir::u32>("pi" + es, body.constant<kir::u32>(e)));
        }
      }
      emit_bitonic(body, pv, pi, N);
      // Network is ascending (smallest at 0). Top-k sits at the high end.
      std::vector<kir::LValue<kir::f32>> topv;
      std::vector<kir::LValue<kir::u32>> topi;
      topv.reserve(k);
      topi.reserve(k);
      for (std::uint32_t sl = 0; sl < k; ++sl) {
        topv.push_back(pv[N - 1 - sl]);
        topi.push_back(pi[N - 1 - sl]);
      }
      if (!write_idx) emit_band(body, topv, k, score_band);
      for (std::uint32_t sl = 0; sl < k; ++sl) {
        body.when(slot == body.constant<kir::u32>(sl), [&] {
          outv = write_idx ? cast<kir::f32>(topi[sl].read()) : topv[sl].read();
        });
      }
      body.ret(outv.read());
      return body.str();
    }

    std::vector<kir::LValue<kir::u32>> win_e;
    std::vector<kir::LValue<kir::f32>> win_v;
    win_e.reserve(k);
    win_v.reserve(k);
    for (std::uint32_t p = 0; p < k; ++p) {
      const std::string ps = std::to_string(p);
      auto bv = body.var<kir::f32>("bv" + ps, math::neg_inf());
      auto be = body.var<kir::u32>("be" + ps, body.constant<kir::u32>(0));
      body.loop("e", body.constant<kir::u32>(0), body.constant<kir::u32>(n), 1,
                [&](kir::Val<kir::u32> e) {
                  auto taken =
                      body.var<kir::u32>("tk" + ps, body.constant<kir::u32>(0));
                  for (std::uint32_t q = 0; q < p; ++q) {
                    body.when(e == win_e[static_cast<std::size_t>(q)].read(),
                              [&] { taken = body.constant<kir::u32>(1); });
                  }
                  const auto v =
                      body.let<kir::f32>("v" + ps, x[row + e].read());
                  body.when(taken.read() == body.constant<kir::u32>(0) &&
                                (v > bv.read() ||
                                 (v == bv.read() && e < be.read())),
                            [&] {
                              bv = v;
                              be = e;
                            });
                });
      win_e.push_back(be);
      win_v.push_back(bv);
    }
    if (!write_idx) emit_band(body, win_v, k, score_band);
    for (std::uint32_t sl = 0; sl < k; ++sl) {
      body.when(slot == body.constant<kir::u32>(sl), [&] {
        outv = write_idx ? cast<kir::f32>(win_e[sl].read()) : win_v[sl].read();
      });
    }
    body.ret(outv.read());
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

}  // namespace lse::backend::hrx_kernels
