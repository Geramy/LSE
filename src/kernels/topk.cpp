#include "lse/graph/kernel_primitive.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "lse/backends/hrx/device_info.hpp"
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

namespace {
constexpr std::uint32_t kChunkThreads = 256;
constexpr std::uint32_t kChunkElements = 2 * kChunkThreads;
constexpr float kMissingIndex = 16777215.0f;

struct TopKPair { float value, index; };
// NaNs remain visible to callers; valid ties keep the original smaller index.
bool pair_better(const TopKPair& a, const TopKPair& b) {
  const bool an = std::isnan(a.value), bn = std::isnan(b.value);
  if (an != bn) return an;
  return a.value > b.value || ((a.value == b.value || an) && a.index < b.index);
}

template <class E>
auto pair_better(E& e, kir::Val<kir::f32> av, kir::Val<kir::f32> ai,
                 kir::Val<kir::f32> bv, kir::Val<kir::f32> bi) {
  const auto an = e.let(av != av), bn = e.let(bv != bv);
  return (an && (bv == bv)) || (an && bn && ai < bi) ||
      ((av == av) && (bv == bv) && ((av > bv) || (av == bv && ai < bi)));
}
}

struct TopKChunkKernel final : KernelPrimitive<TopKChunkKernel> {
  static constexpr std::string_view kName = "topk.chunk.v2";
  static constexpr std::string_view kEntry = "lse_topk_chunk_v2";
  static constexpr std::string_view kSource = {};
  std::size_t arity() const noexcept override { return 1; }
  bool owns_indexing() const noexcept override { return true; }
  bool supports_epilogue() const noexcept override { return false; }
  bool has_host_impl() const noexcept override { return true; }

  std::string emit_kernel(const KernelShapes& s) const override {
    if (s.inputs.size() != 1 || s.output.rank() < 3 || !s.device || !s.store ||
        !s.types.scalar || !s.intrinsics || s.input_dtypes[0] != DType::kF32 ||
        s.output_dtype != DType::kF32 ||
        s.device->max_threads_per_workgroup < kChunkThreads) return {};
    const auto top = static_cast<std::uint32_t>(s.attrs[0]);
    const auto count = static_cast<std::uint32_t>(s.attrs[1]);
    const bool pairs = s.attrs[2] != 0.0f;
    const auto chunks = static_cast<std::uint32_t>(s.output.dim(s.output.rank() - 3));
    if (!top || top > 16 || !count || chunks != (count + kChunkElements - 1) / kChunkElements ||
        s.output.dim(s.output.rank() - 2) != static_cast<std::int64_t>(top) || s.output.dim(s.output.rank() - 1) != 2 ||
        2ull * kChunkThreads * top * sizeof(float) > backend::workgroup_lds_bytes(s.device)) return {};
    kir::KernelBody body(s.types, *s.intrinsics, backend::workgroup_lds_bytes(s.device));
    body.set_store(s.store);
    TopKArgs<env::Emit> a;
    if (!env::bind(body, a, s)) return {};
    env::Emit e{&body};
    const auto lid = e.let(math::local_id());
    const auto chunk = e.let(math::workgroup_id_x());
    const auto row = e.let(math::workgroup_id_y());
    auto v0 = e.var(math::neg_inf()), v1 = e.var(math::neg_inf());
    auto i0 = e.var(kMissingIndex), i1 = e.var(kMissingIndex);
    for (unsigned part = 0; part < 2; ++part) {
      const auto at = e.let(chunk * kChunkElements + lid + part * kChunkThreads);
      if (auto valid = e.when(at < count)) {
        const auto offset = e.let((row * count + at) * (pairs ? 2u : 1u));
        auto& value = part == 0 ? v0 : v1;
        auto& index = part == 0 ? i0 : i1;
        value = a.x[offset];
        index = pairs ? a.x[offset + 1u] : kir::cast<kir::f32>(at);
      }
    }
    if (auto swap = e.when(pair_better(e, v1.read(), i1.read(), v0.read(), i0.read()))) {
      const auto saved_value = e.let(v0.read()), saved_index = e.let(i0.read());
      v0 = v1.read(); i0 = i1.read(); v1 = saved_value; i1 = saved_index;
    }
    const auto values = e.lds<kir::f32>(kChunkThreads * top);
    const auto indices = e.lds<kir::f32>(kChunkThreads * top);
    const auto base = e.let(lid);
    for (std::uint32_t p = 0; p < top; ++p) {
      values[base + p * kChunkThreads] = p == 0 ? v0.read() : p == 1 ? v1.read() : math::neg_inf();
      indices[base + p * kChunkThreads] = p == 0 ? i0.read() : p == 1 ? i1.read() : e.f32(kMissingIndex);
    }
    e.barrier();
    std::vector<kir::LValue<kir::f32>> merged_values, merged_indices;
    for (std::uint32_t p = 0; p < top; ++p) {
      merged_values.push_back(e.var(0.0f));
      merged_indices.push_back(e.var(0.0f));
    }
    for (std::uint32_t distance = kChunkThreads / 2; distance; distance >>= 1) {
      if (auto active = e.when(lid < distance)) {
        auto left = e.var(e.u32(0)), right = e.var(e.u32(0));
        const auto other = e.let(lid + distance);
        for (std::uint32_t p = 0; p < top; ++p) {
          auto lv = e.var(math::neg_inf()), rv = e.var(math::neg_inf());
          auto li = e.var(kMissingIndex), ri = e.var(kMissingIndex);
          if (auto valid = e.when(left.read() < top)) {
            lv = values[base + left.read() * kChunkThreads].read(); li = indices[base + left.read() * kChunkThreads].read();
          }
          if (auto valid = e.when(right.read() < top)) {
            rv = values[other + right.read() * kChunkThreads].read(); ri = indices[other + right.read() * kChunkThreads].read();
          }
          const auto take_right = e.let(pair_better(e, rv.read(), ri.read(), lv.read(), li.read()));
          merged_values[p] = select(take_right, rv.read(), lv.read());
          merged_indices[p] = select(take_right, ri.read(), li.read());
          right = right.read() + select(take_right, e.u32(1), e.u32(0));
          left = left.read() + select(take_right, e.u32(0), e.u32(1));
        }
        for (std::uint32_t p = 0; p < top; ++p) {
          values[base + p * kChunkThreads] = merged_values[p].read();
          indices[base + p * kChunkThreads] = merged_indices[p].read();
        }
      }
      e.barrier();
    }
    if (auto leader = e.when(lid == 0)) {
      const auto output = e.let((row * chunks + chunk) * top * 2u);
      for (std::uint32_t p = 0; p < top; ++p) {
        e.store(output + p * 2u, values[e.u32(p * kChunkThreads)].read());
        e.store(output + p * 2u + 1u, indices[e.u32(p * kChunkThreads)].read());
      }
    }
    return body.lds().ok() ? body.str() : std::string{};
  }
  void eval_cpu(std::span<const float* const> in, float* out, std::size_t elements,
                const std::array<float, 4>& attrs) const override {
    const auto top = static_cast<std::size_t>(attrs[0]);
    const auto count = static_cast<std::size_t>(attrs[1]);
    const bool pairs = attrs[2] != 0.0f;
    const auto chunks = (count + kChunkElements - 1) / kChunkElements;
    const auto rows = elements / (chunks * top * 2);
    std::vector<TopKPair> candidates;
    candidates.reserve(kChunkElements);
    for (std::size_t row = 0; row < rows; ++row) {
      for (std::size_t chunk = 0; chunk < chunks; ++chunk) {
        candidates.clear();
        const auto begin = chunk * kChunkElements;
        const auto end = std::min(count, begin + kChunkElements);
        for (auto at = begin; at < end; ++at) {
          const auto offset = (row * count + at) * (pairs ? 2u : 1u);
          candidates.push_back({in[0][offset], pairs ? in[0][offset + 1] : static_cast<float>(at)});
        }
        std::sort(candidates.begin(), candidates.end(),
                  [](const TopKPair& a, const TopKPair& b) { return pair_better(a, b); });
        const auto base = (row * chunks + chunk) * top * 2;
        for (std::size_t p = 0; p < top; ++p) {
          const auto candidate = p < candidates.size() ? candidates[p]
              : TopKPair{-std::numeric_limits<float>::infinity(), kMissingIndex};
          out[base + 2 * p] = candidate.value;
          out[base + 2 * p + 1] = candidate.index;
        }
      }
    }
  }
  Result<Shape> infer_shape(std::span<const Shape> in) const override { return in[0]; }
  DType infer_dtype(std::span<const DType>) const override { return DType::kF32; }
  static ThreadPlan plan_impl(const KernelShapes& s) {
    ThreadPlan plan;
    const auto chunks = static_cast<std::uint32_t>(s.output.dim(s.output.rank() - 3));
    const auto top = static_cast<std::uint32_t>(s.attrs[0]);
    plan.workgroup_size[0] = kChunkThreads;
    plan.workgroup_count[0] = chunks;
    plan.workgroup_count[1] = static_cast<std::uint32_t>(s.output.elem_count() / (chunks * top * 2));
    plan.lds_bytes = 2 * kChunkThreads * top * static_cast<std::uint32_t>(sizeof(float));
    return plan;
  }
};
LSE_REGISTER_PRIMITIVE(TopKChunkKernel);

struct TopKExtractKernel final : KernelPrimitive<TopKExtractKernel> {
  static constexpr std::string_view kName = "topk.extract";
  static constexpr std::string_view kEntry = "lse_topk_extract";
  static constexpr std::string_view kSource = {};
  std::size_t arity() const noexcept override { return 1; }
  bool owns_indexing() const noexcept override { return true; }
  bool supports_epilogue() const noexcept override { return false; }
  bool has_host_impl() const noexcept override { return true; }
  std::string emit_kernel(const KernelShapes& s) const override {
    if (s.inputs.size() != 1 || !s.store || !s.types.scalar || !s.intrinsics ||
        s.input_dtypes[0] != DType::kF32 || s.output_dtype != DType::kF32) return {};
    const auto top = static_cast<std::uint32_t>(s.attrs[0]);
    const bool index = s.attrs[1] != 0.0f;
    const float band = s.attrs[2] == 0.0f ? 1.0f : s.attrs[2];
    if (!top || top > 16) return {};
    kir::KernelBody body(s.types, *s.intrinsics);
    body.set_store(s.store);
    TopKArgs<env::Emit> a;
    if (!env::bind(body, a, s)) return {};
    env::Emit e{&body};
    const auto at = e.let(e.thread_id());
    if (auto valid = e.when(at < static_cast<std::uint32_t>(s.output.elem_count()))) {
      const auto row = e.let(at / top);
      const auto slot = e.let(at % top);
      const auto base = e.let(row * top * 2u);
      auto result = e.var(a.x[base + slot * 2u + (index ? 1u : 0u)]);
      if (!index && band < 1.0f) {
        const auto threshold = e.let(a.x[base] * (1.0f - band));
        auto total = e.var(0.0f);
        for (std::uint32_t p = 0; p < top; ++p) {
          const auto value = e.let(a.x[base + p * 2u]);
          total = total.read() + select(value < threshold, e.f32(0.0f), value);
        }
        result = select(result.read() < threshold, e.f32(0.0f), result.read()) / (total.read() + 1e-9f);
      }
      e.store(at, result.read());
    }
    return body.str();
  }
  void eval_cpu(std::span<const float* const> in, float* out, std::size_t count,
                const std::array<float, 4>& attrs) const override {
    const auto top = static_cast<std::size_t>(attrs[0]);
    const bool index = attrs[1] != 0.0f;
    const float band = attrs[2] == 0.0f ? 1.0f : attrs[2];
    for (std::size_t row = 0; row < count / top; ++row) {
      const auto base = row * top * 2;
      float total = 0.0f;
      const float threshold = in[0][base] * (1.0f - band);
      if (!index && band < 1.0f) {
        for (std::size_t p = 0; p < top; ++p) {
          const float value = in[0][base + p * 2];
          total += value < threshold ? 0.0f : value;
        }
        total += 1e-9f;
      }
      for (std::size_t p = 0; p < top; ++p) {
        const float value = in[0][base + p * 2 + (index ? 1u : 0u)];
        out[row * top + p] = !index && band < 1.0f ? (value < threshold ? 0.0f : value) / total : value;
      }
    }
  }
  Result<Shape> infer_shape(std::span<const Shape> in) const override { return in[0]; }
  DType infer_dtype(std::span<const DType>) const override { return DType::kF32; }
  static ThreadPlan plan_impl(const KernelShapes& s) {
    ThreadPlan plan;
    plan.workgroup_size[0] = kChunkThreads;
    plan.workgroup_count[0] = static_cast<std::uint32_t>((s.output.elem_count() + kChunkThreads - 1) / kChunkThreads);
    return plan;
  }
};
LSE_REGISTER_PRIMITIVE(TopKExtractKernel);

}  // namespace lse::kernels
