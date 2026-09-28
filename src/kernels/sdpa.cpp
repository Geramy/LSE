#include "lse/graph/kernel_args.hpp"
#include "lse/graph/kernel_env.hpp"
#include "lse/graph/kernel_primitive.hpp"
#include "lse/kv/block.hpp"
#include "lse/kernels/sdpa.hpp"
#include "lse/dispatch/attention.hpp"
#include "lse/dispatch/attention_shapes.hpp"
#include "lse/math.hpp"
#include "lse/backends/hrx/device_info.hpp"

#include <string>
#include <vector>

namespace lse::kernels {

using namespace lse::graph;
namespace math = lse::math;

namespace {

constexpr bool is_pow2(std::uint32_t v) noexcept {
  return v >= 2 && (v & (v - 1)) == 0;
}

}  // namespace

template <class E>
struct SdpaArgs {
  env::In<kir::f32, E> q;
  env::In<kir::f32, E> k;
  env::In<kir::f32, E> v;
  // Optional 4th input. Contiguous form: [1], the live cache offset. Paged
  // form: the step descriptor, kv::step_meta_elems(rows) floats — see
  // kv/block.hpp.
  env::In<kir::f32, E> meta;
  // Optional 5th input, paged form only: [rows, stride] block ids.
  env::In<kir::f32, E> table;
  env::Out<kir::f32, E> out;
};

// Shared scores preserve the sequential FP32 QK and value accumulation order.
template <bool SharedExp>
struct DecodeSdpaKernel final : KernelPrimitive<DecodeSdpaKernel<SharedExp>> {
  static constexpr std::string_view kName = SharedExp
      ? "attention.decode_shared_exp" : "attention.decode_shared";
  static constexpr std::string_view kEntry = SharedExp
      ? "lse_sdpa_decode_shared_exp" : "lse_sdpa_decode_shared";
  static constexpr std::string_view kSource = {};
  static constexpr std::uint32_t kThreads = 256;
  std::size_t arity() const noexcept override { return 5; }
  bool owns_indexing() const noexcept override { return true; }

  std::string emit_kernel(const KernelShapes &s) const override {
    if (!dispatch::shared_decode_supported(s) || !s.types.scalar || !s.intrinsics || !s.store)
      return {};
    const auto heads = static_cast<std::uint32_t>(s.inputs[0].dim(1));
    const auto kvheads = static_cast<std::uint32_t>(s.inputs[1].dim(1));
    const auto block = static_cast<std::uint32_t>(s.inputs[1].dim(2));
    const auto stride = static_cast<std::uint32_t>(s.inputs[4].dim(1));
    const auto capacity = stride * block;
    kir::KernelBody k(s.types, *s.intrinsics, workgroup_lds_bytes(s.device));
    k.set_store(s.store);
    SdpaArgs<env::Emit> a;
    if (!env::bind(k, a, s))
      return {};
    env::Emit e{&k};
    const auto scores = e.lds<kir::f32>(capacity);
    const auto lane = e.let(math::local_id());
    const auto wg = e.let(math::workgroup_id_x());
    const auto h = e.let(wg % heads), b = e.let(wg / heads);
    const auto kh = e.let(h / (heads / kvheads));
    const auto output = e.let((b * heads + h) * 256u + lane);
    const auto rows = e.runtime_extent("rows", kir::cast<kir::u32>(a.meta[2u]));
    if (auto pad = e.when(b >= rows))
      e.store(output, e.f32(0.0f));
    (void)e.ret_if(b >= rows); // Uniform across the whole workgroup.
    const auto mb =
        e.let(e.u32(kv::kStepMetaHeader) + b * e.u32(kv::kStepMetaPerRow));
    const auto offset = e.let(kir::cast<kir::u32>(a.meta[mb]));
    const auto loaded_len = e.let(kir::cast<kir::u32>(a.meta[mb + 1u]));
    const auto row_len =
        e.let(select(loaded_len < capacity, loaded_len, e.u32(capacity)));
    const auto loaded_max = e.let(kir::cast<kir::u32>(a.meta[1u]));
    const auto kv_len = e.runtime_extent(
        "kv_len", select(loaded_max < capacity, loaded_max, e.u32(capacity)));
    const auto tb = e.let(b * stride), qb = e.let((b * heads + h) * 256u);
    // All lanes reach the barrier, including those beyond the last live key.
    for (auto j : e.range(lane, kv_len, kThreads)) {
      scores[j] = math::neg_inf();
      if (auto live = e.when(j < row_len)) {
        const auto blk = e.let(kir::cast<kir::u32>(a.table[tb + j / block]));
        const auto kb =
            e.let(((blk * kvheads + kh) * block + j % block) * 256u);
        auto score = e.var(0.0f);
        for (auto dd : e.range(256u))
          score = math::fma(a.q[qb + dd], a.k[kb + dd], score.read());
        scores[j] = score.read() * s.attrs[0];
      }
    }
    e.barrier();
    const auto nblk = e.let((kv_len + e.u32(block - 1)) / e.u32(block));
    const auto row_blk = e.let((row_len + e.u32(block - 1)) / e.u32(block));
    const auto maximum = e.var(math::neg_inf());
    auto allowed = [&](auto j) {
      if (s.iattrs[0] == 0)
        return e.u32(1) == e.u32(1);
      if (s.iattrs[0] == 1)
        return j <= offset;
      // Dispatch bounds make key + window nonwrapping.
      return j <= offset &&
             j + static_cast<std::uint32_t>(s.iattrs[1]) > offset;
    };
    for (auto bi : e.range(nblk))
      if (auto mine = e.when(bi < row_blk)) {
        for (auto jj : e.range(block)) {
          const auto j = e.let(bi * block + jj);
          if (auto live = e.when(j < row_len && j < kv_len))
            if (auto mask = e.when(allowed(j)))
              maximum = math::max(maximum.read(), scores[j].read());
        }
      }
    if constexpr (SharedExp) {
      // The maximum is identical in every lane. Each key's exponential is
      // therefore identical too: compute it once, then share the result with
      // the output channels while retaining their serial FP32 sum order.
      // All lanes must finish reading the scores for max before any lane
      // overwrites them with probabilities.
      e.barrier();
      for (auto j : e.range(lane, kv_len, kThreads)) {
        auto weight = e.var(0.0f);
        if (auto live = e.when(j < row_len && allowed(j)))
          weight = math::exp(scores[j].read() - maximum.read());
        scores[j] = weight.read();
      }
      e.barrier();
    }
    auto denom = e.var(0.0f), acc = e.var(0.0f);
    for (auto bi : e.range(nblk))
      if (auto mine = e.when(bi < row_blk)) {
        const auto blk = e.let(kir::cast<kir::u32>(a.table[tb + bi]));
        const auto vb = e.let(((blk * kvheads + kh) * block) * 256u + lane);
        for (auto jj : e.range(block)) {
          const auto j = e.let(bi * block + jj);
          if (auto live = e.when(j < row_len && j < kv_len)) {
            auto weight = e.var(0.0f);
            if constexpr (SharedExp) {
              weight = scores[j].read();
            } else {
              if (auto mask = e.when(allowed(j)))
                weight = math::exp(scores[j].read() - maximum.read());
            }
            denom = denom.read() + weight.read();
            acc = math::fma(weight.read(), a.v[vb + jj * 256u], acc.read());
          }
        }
      }
    e.store(output, acc.read() / select(denom.read() == 0.0f, e.f32(1.0f),
                                        denom.read()));
    return k.lds().ok() ? k.str() : std::string{};
  }
  Result<Shape> infer_shape(std::span<const Shape> in) const override {
    if (in.size() != 5)
      return LSE_ERROR(kInvalidArgument, "decode attention takes five inputs");
    return Shape{in[0].dim(0), in[0].dim(1), 1, in[2].dim(3)};
  }
  DType infer_dtype(std::span<const DType> in) const override {
    return in.empty() ? DType::kF32 : in[0];
  }
  static ThreadPlan plan_impl(const KernelShapes &s) {
    ThreadPlan tp;
    tp.workgroup_size[0] = kThreads;
    tp.lds_bytes =
        static_cast<std::uint32_t>(s.inputs[4].dim(1) * s.inputs[1].dim(2)) *
        sizeof(float);
    tp.workgroup_count[0] =
        static_cast<std::uint32_t>(s.inputs[0].dim(0) * s.inputs[0].dim(1));
    return tp;
  }
};
using DecodeSdpaKernelBase = DecodeSdpaKernel<false>;
using DecodeSdpaKernelSharedExp = DecodeSdpaKernel<true>;
LSE_REGISTER_PRIMITIVE(DecodeSdpaKernelBase);
LSE_REGISTER_PRIMITIVE(DecodeSdpaKernelSharedExp);
const DecodeSdpaKernelSharedExp kDecodeSharedExp{};


// Partition records contain (maximum, denominator, FP32 value numerator[256]).
constexpr std::uint32_t kSplitKeys = 128;
constexpr std::uint32_t kSplitRecord = 258;

struct SplitDecodePartial final : KernelPrimitive<SplitDecodePartial> {
  static constexpr std::string_view kName = "attention.decode_partial128.v1";
  static constexpr std::string_view kEntry = "lse_sdpa_decode_partial128_v1";
  static constexpr std::string_view kSource = {};
  std::size_t arity() const noexcept override { return 5; }
  bool owns_indexing() const noexcept override { return true; }
  bool supports_epilogue() const noexcept override { return false; }
  const KernelPrimitiveBase* specialize(const KernelShapes& s) const override {
    return dispatch::split_decode_supported(s) ? this : nullptr;
  }

  std::string emit_kernel(const KernelShapes& s) const override {
    if (!dispatch::split_decode_supported(s) || !s.types.scalar || !s.intrinsics || !s.store)
      return {};
    const auto heads = static_cast<std::uint32_t>(s.inputs[0].dim(1));
    const auto kvheads = static_cast<std::uint32_t>(s.inputs[1].dim(1));
    const auto block = static_cast<std::uint32_t>(s.inputs[1].dim(2));
    const auto stride = static_cast<std::uint32_t>(s.inputs[4].dim(1));
    const auto capacity = stride * block;
    const auto parts = (capacity + kSplitKeys - 1u) / kSplitKeys;
    kir::KernelBody k(s.types, *s.intrinsics, workgroup_lds_bytes(s.device));
    k.set_store(s.store);
    SdpaArgs<env::Emit> a;
    if (!env::bind(k, a, s)) return {};
    env::Emit e{&k};
    const auto scores = e.lds<kir::f32>(kSplitKeys);
    const auto lane = e.let(math::local_id());
    const auto wg = e.let(math::workgroup_id_x());
    const auto part = e.let(wg % parts);
    const auto h = e.let((wg / parts) % heads), b = e.let(wg / (parts * heads));
    const auto kh = e.let(h / (heads / kvheads));
    const auto output = e.let(wg * kSplitRecord);
    const auto rows = e.runtime_extent("rows", kir::cast<kir::u32>(a.meta[2u]));
    const auto mb = e.let(e.u32(kv::kStepMetaHeader) + b * e.u32(kv::kStepMetaPerRow));
    const auto offset = e.let(kir::cast<kir::u32>(a.meta[mb]));
    const auto loaded_len = e.let(kir::cast<kir::u32>(a.meta[mb + 1u]));
    const auto row_len = e.let(select(loaded_len < capacity, loaded_len, e.u32(capacity)));
    const auto loaded_max = e.let(kir::cast<kir::u32>(a.meta[1u]));
    const auto kv_len = e.runtime_extent("kv_len", select(loaded_max < capacity, loaded_max, e.u32(capacity)));
    const auto begin = e.let(part * kSplitKeys);
    const auto tb = e.let(b * stride), qb = e.let((b * heads + h) * 256u);
    auto allowed = [&](auto j) {
      if (s.iattrs[0] == 0) return e.u32(1) == e.u32(1);
      if (s.iattrs[0] == 1) return j <= offset;
      return j <= offset && j + static_cast<std::uint32_t>(s.iattrs[1]) > offset;
    };
    if (auto owns = e.when(lane < kSplitKeys)) {
      const auto j = e.let(begin + lane);
      scores[lane] = math::neg_inf();
      if (auto live = e.when(b < rows && j < kv_len && j < row_len && allowed(j))) {
        const auto blk = e.let(kir::cast<kir::u32>(a.table[tb + j / block]));
        const auto kb = e.let(((blk * kvheads + kh) * block + j % block) * 256u);
        auto score = e.var(0.0f);
        for (auto dd : e.range(256u))
          score = math::fma(a.q[qb + dd], a.k[kb + dd], score.read());
        scores[lane] = score.read() * s.attrs[0];
      }
    }
    e.barrier();
    auto maximum = e.var(math::neg_inf());
    for (auto jj : e.range(kSplitKeys))
      maximum = math::max(maximum.read(), scores[jj].read());
    e.barrier();
    if (auto owns = e.when(lane < kSplitKeys)) {
      const auto j = e.let(begin + lane);
      auto weight = e.var(0.0f);
      if (auto live = e.when(b < rows && j < kv_len && j < row_len && allowed(j)))
        weight = math::exp(scores[lane].read() - maximum.read());
      scores[lane] = weight.read();
    }
    e.barrier();
    auto denom = e.var(0.0f), acc = e.var(0.0f);
    for (auto jj : e.range(kSplitKeys)) {
      const auto j = e.let(begin + jj);
      if (auto live = e.when(b < rows && j < kv_len && j < row_len && allowed(j))) {
        const auto weight = e.let(scores[jj].read());
        const auto blk = e.let(kir::cast<kir::u32>(a.table[tb + j / block]));
        const auto vb = e.let(((blk * kvheads + kh) * block + j % block) * 256u + lane);
        denom = denom.read() + weight;
        acc = math::fma(weight, a.v[vb], acc.read());
      }
    }
    e.store(output + 2u + lane, acc.read());
    if (auto first = e.when(lane == 0u)) {
      e.store(output, maximum.read());
      e.store(output + 1u, denom.read());
    }
    return k.lds().ok() ? k.str() : std::string{};
  }
  Result<Shape> infer_shape(std::span<const Shape> in) const override {
    if (in.size() != 5 || in[0].rank() != 4 || in[1].rank() != 4 || in[4].rank() != 2)
      return LSE_ERROR(kInvalidArgument, "split decode attention takes five paged inputs");
    const auto capacity = in[1].dim(2) * in[4].dim(1);
    return Shape{in[0].dim(0), in[0].dim(1), (capacity + kSplitKeys - 1) / kSplitKeys, kSplitRecord};
  }
  DType infer_dtype(std::span<const DType>) const override { return DType::kF32; }
  static ThreadPlan plan_impl(const KernelShapes& s) {
    ThreadPlan tp;
    tp.workgroup_size[0] = 256;
    tp.workgroup_count[0] = static_cast<std::uint32_t>(s.output.dim(0) * s.output.dim(1) * s.output.dim(2));
    tp.lds_bytes = kSplitKeys * sizeof(float);
    return tp;
  }
};

template<class E> struct SplitMergeArgs {
  env::In<kir::f32, E> partial;
  env::Out<kir::f32, E> out;
};
struct SplitDecodeMerge final : KernelPrimitive<SplitDecodeMerge> {
  static constexpr std::string_view kName = "attention.decode_merge128.v1";
  static constexpr std::string_view kEntry = "lse_sdpa_decode_merge128_v1";
  static constexpr std::string_view kSource = {};
  std::size_t arity() const noexcept override { return 1; }
  bool owns_indexing() const noexcept override { return true; }
  std::string emit_kernel(const KernelShapes& s) const override {
    if (!dispatch::split_decode_merge_supported(s) || !s.types.scalar || !s.intrinsics || !s.store)
      return {};
    const auto parts = static_cast<std::uint32_t>(s.inputs[0].dim(2));
    kir::KernelBody k(s.types, *s.intrinsics, workgroup_lds_bytes(s.device));
    k.set_store(s.store);
    SplitMergeArgs<env::Emit> a;
    if (!env::bind(k, a, s)) return {};
    env::Emit e{&k};
    const auto weights = e.lds<kir::f32>(parts);
    const auto lane = e.let(math::local_id()), wg = e.let(math::workgroup_id_x());
    const auto base = e.let(wg * parts * kSplitRecord);
    auto maximum = e.var(math::neg_inf());
    for (auto p : e.range(parts))
      if (auto valid = e.when(a.partial[base + p * kSplitRecord + 1u] > 0.0f))
        maximum = math::max(maximum.read(), a.partial[base + p * kSplitRecord]);
    if (auto owns = e.when(lane < parts)) {
      const auto pb = e.let(base + lane * kSplitRecord);
      auto weight = e.var(0.0f);
      // A zero denominator marks empty and padded partitions; skip their
      // sentinel maximum so an all-empty row never subtracts two infinities.
      if (auto valid = e.when(a.partial[pb + 1u] > 0.0f))
        weight = math::exp(a.partial[pb] - maximum.read());
      weights[lane] = weight.read();
    }
    e.barrier();
    auto denom = e.var(0.0f), acc = e.var(0.0f);
    for (auto p : e.range(parts)) {
      const auto pb = e.let(base + p * kSplitRecord);
      const auto weight = e.let(weights[p].read());
      denom = math::fma(weight, a.partial[pb + 1u], denom.read());
      acc = math::fma(weight, a.partial[pb + 2u + lane], acc.read());
    }
    e.store(wg * 256u + lane, acc.read() / select(denom.read() == 0.0f, e.f32(1.0f), denom.read()));
    return k.lds().ok() ? k.str() : std::string{};
  }
  Result<Shape> infer_shape(std::span<const Shape> in) const override {
    if (in.size() != 1 || in[0].rank() != 4 || in[0].dim(3) != kSplitRecord)
      return LSE_ERROR(kInvalidArgument, "split decode merge takes one partial record array");
    return Shape{in[0].dim(0), in[0].dim(1), 1, 256};
  }
  DType infer_dtype(std::span<const DType>) const override { return DType::kF32; }
  static ThreadPlan plan_impl(const KernelShapes& s) {
    ThreadPlan tp;
    tp.workgroup_size[0] = 256;
    tp.workgroup_count[0] = static_cast<std::uint32_t>(s.output.dim(0) * s.output.dim(1));
    tp.lds_bytes = static_cast<std::uint32_t>(s.inputs[0].dim(2)) * sizeof(float);
    return tp;
  }
};
LSE_REGISTER_PRIMITIVE(SplitDecodePartial);
LSE_REGISTER_PRIMITIVE(SplitDecodeMerge);

template <bool ShortQuery>
struct SplitPartialWg128C2 final : KernelPrimitive<SplitPartialWg128C2<ShortQuery>> {
  static constexpr std::string_view kName = ShortQuery
      ? "attention.short_partial128.wg128c2.v2" : "attention.decode_partial128.wg128c2.v2";
  static constexpr std::string_view kEntry = ShortQuery
      ? "lse_sdpa_short_partial128_wg128c2_v2" : "lse_sdpa_decode_partial128_wg128c2_v2";
  static constexpr std::string_view kSource = {};
  std::size_t arity() const noexcept override { return 5; }
  bool owns_indexing() const noexcept override { return true; }
  bool supports_epilogue() const noexcept override { return false; }
  const KernelPrimitiveBase* specialize(const KernelShapes& s) const override {
    return (ShortQuery ? dispatch::split_short_supported(s) : dispatch::split_decode_supported(s)) ? this : nullptr;
  }

  std::string emit_kernel(const KernelShapes& s) const override {
    if (!(ShortQuery ? dispatch::split_short_supported(s) : dispatch::split_decode_supported(s)) || !s.types.scalar || !s.intrinsics || !s.store)
      return {};
    const auto heads = static_cast<std::uint32_t>(s.inputs[0].dim(1));
    const auto queries = ShortQuery ? static_cast<std::uint32_t>(s.inputs[0].dim(2)) : 1u;
    const auto kvheads = static_cast<std::uint32_t>(s.inputs[1].dim(1));
    const auto block = static_cast<std::uint32_t>(s.inputs[1].dim(2));
    const auto stride = static_cast<std::uint32_t>(s.inputs[4].dim(1));
    const auto capacity = stride * block;
    const auto parts = (capacity + kSplitKeys - 1u) / kSplitKeys;
    kir::KernelBody k(s.types, *s.intrinsics, workgroup_lds_bytes(s.device));
    k.set_store(s.store);
    SdpaArgs<env::Emit> a;
    if (!env::bind(k, a, s)) return {};
    env::Emit e{&k};
    const auto query_tile = ShortQuery ? dispatch::attention_shapes::short_query_tile(queries, capacity) : 1u;
    if (query_tile > 1u) {
      const auto scores = e.lds<kir::f32>(query_tile * kSplitKeys);
      const auto lane = e.let(math::local_id());
      const auto wg = e.let(math::workgroup_id_x());
      const auto query_tiles = (queries + query_tile - 1u) / query_tile;
      const auto part = e.let(wg % parts);
      const auto q0 = e.let(((wg / parts) % query_tiles) * query_tile);
      const auto h = e.let((wg / (parts * query_tiles)) % heads);
      const auto b = e.let(wg / (parts * query_tiles * heads));
      const auto kh = e.let(h / (heads / kvheads));
      const auto rows = e.runtime_extent("rows", kir::cast<kir::u32>(a.meta[2u]));
      const auto mb = e.let(e.u32(kv::kStepMetaHeader) + b * e.u32(kv::kStepMetaPerRow));
      const auto offset = e.let(kir::cast<kir::u32>(a.meta[mb]));
      const auto loaded_len = e.let(kir::cast<kir::u32>(a.meta[mb + 1u]));
      const auto row_len = e.let(select(loaded_len < capacity, loaded_len, e.u32(capacity)));
      const auto loaded_max = e.let(kir::cast<kir::u32>(a.meta[1u]));
      const auto kv_len = e.runtime_extent("kv_len", select(loaded_max < capacity, loaded_max, e.u32(capacity)));
      const auto begin = e.let(part * kSplitKeys), tb = e.let(b * stride);
      if (dispatch::attention_shapes::short_skips_empty_partitions(queries, capacity)) {
        const auto no_keys = e.let(begin >= kv_len);
        if (auto empty = e.when(no_keys)) {
          for (std::uint32_t r = 0; r < query_tile; ++r) {
            if (auto row = e.when(q0 + r < queries)) {
              const auto output = e.let((((b * heads + h) * queries + q0 + r) * parts + part) * kSplitRecord);
              e.store(output + 2u + lane * 2u, e.f32(0.0f));
              e.store(output + 3u + lane * 2u, e.f32(0.0f));
              if (auto first = e.when(lane == 0u)) {
                e.store(output, math::neg_inf());
                e.store(output + 1u, e.f32(0.0f));
              }
            }
          }
        }
        // The condition is identical for every lane in this workgroup.
        (void)e.ret_if(no_keys);
      }
      auto allowed = [&](auto j, std::uint32_t r) {
        if (s.iattrs[0] == 0) return e.u32(1) == e.u32(1);
        const auto position = e.let(kir::cast<std::int64_t>(offset) +
                                     kir::cast<std::int64_t>(q0 + r));
        const auto key = e.let(kir::cast<std::int64_t>(j));
        if (s.iattrs[0] == 1) return key <= position;
        return key <= position && position - key < kir::cast<std::int64_t>(e.u32(static_cast<std::uint32_t>(s.iattrs[1])));
      };
      auto valid_rows = [&](auto j) {
        std::vector<kir::Val<kir::boolean>> valid;
        for (std::uint32_t r = 0; r < query_tile; ++r)
          valid.push_back(e.let(q0 + r < queries && b < rows && j < kv_len && j < row_len && allowed(j, r)));
        return valid;
      };
      auto any_valid = [&](const auto& valid) {
        auto any = valid[0];
        for (std::uint32_t r = 1; r < query_tile; ++r) any = any || valid[r];
        return e.let(any);
      };
      const auto wl = e.let(lane % 32u), wi = e.let(lane / 32u);
      std::vector<kir::LValue<kir::f32>> query_values;
      for (std::uint32_t r = 0; r < query_tile; ++r) {
        const auto qb = e.let(((b * heads + h) * queries + q0 + r) * 256u);
        for (std::uint32_t d = 0; d < 8u; ++d) {
          query_values.push_back(e.var(0.0f));
          if (auto live = e.when(q0 + r < queries && b < rows))
            query_values.back() = a.q[qb + wl + d * 32u];
        }
      }
      for (auto tile : e.range(kSplitKeys / 4u)) {
        const auto key_lane = e.let(tile * 4u + wi), j = e.let(begin + key_lane);
        const auto valid = valid_rows(j);
        std::vector<kir::LValue<kir::f32>> row_scores;
        for (std::uint32_t r = 0; r < query_tile; ++r) row_scores.push_back(e.var(0.0f));
        if (auto live = e.when(any_valid(valid))) {
          const auto blk = e.let(kir::cast<kir::u32>(a.table[tb + j / block]));
          const auto kb = e.let(((blk * kvheads + kh) * block + j % block) * 256u);
          for (std::uint32_t d = 0; d < 8u; ++d) {
            const auto key = e.let(a.k[kb + wl + d * 32u]);
            for (std::uint32_t r = 0; r < query_tile; ++r)
              row_scores[r] = math::fma(query_values[r * 8u + d].read(), key, row_scores[r].read());
          }
        }
        for (std::uint32_t shift = 16u; shift; shift >>= 1u)
          for (std::uint32_t r = 0; r < query_tile; ++r)
            row_scores[r] = row_scores[r].read() + math::shfl_xor(row_scores[r].read(), e.u32(shift));
        if (auto first = e.when(wl == 0u))
          for (std::uint32_t r = 0; r < query_tile; ++r)
            scores[r * kSplitKeys + key_lane] = select(valid[r], row_scores[r].read() * s.attrs[0], math::neg_inf());
      }
      e.barrier();
      std::vector<kir::LValue<kir::f32>> maximum;
      for (std::uint32_t r = 0; r < query_tile; ++r) maximum.push_back(e.var(math::neg_inf()));
      for (auto jj : e.range(kSplitKeys))
        for (std::uint32_t r = 0; r < query_tile; ++r)
          maximum[r] = math::max(maximum[r].read(), scores[r * kSplitKeys + jj].read());
      e.barrier();
      if (auto owns = e.when(lane < kSplitKeys)) {
        const auto j = e.let(begin + lane);
        const auto valid = valid_rows(j);
        for (std::uint32_t r = 0; r < query_tile; ++r) {
          auto weight = e.var(0.0f);
          if (auto live = e.when(valid[r]))
            weight = math::exp(scores[r * kSplitKeys + lane].read() - maximum[r].read());
          scores[r * kSplitKeys + lane] = weight.read();
        }
      }
      e.barrier();
      std::vector<kir::LValue<kir::f32>> denom, acc0, acc1;
      for (std::uint32_t r = 0; r < query_tile; ++r) {
        denom.push_back(e.var(0.0f));
        acc0.push_back(e.var(0.0f));
        acc1.push_back(e.var(0.0f));
      }
      for (auto jj : e.range(kSplitKeys)) {
        const auto j = e.let(begin + jj);
        const auto valid = valid_rows(j);
        if (auto live = e.when(any_valid(valid))) {
          const auto blk = e.let(kir::cast<kir::u32>(a.table[tb + j / block]));
          const auto vb = e.let(((blk * kvheads + kh) * block + j % block) * 256u + lane * 2u);
          const auto values = e.load(a.v, vb, 8u);
          for (std::uint32_t r = 0; r < query_tile; ++r) {
            if (auto row_live = e.when(valid[r])) {
              const auto weight = e.let(scores[r * kSplitKeys + jj].read());
              denom[r] = denom[r].read() + weight;
              acc0[r] = math::fma(weight, values[0], acc0[r].read());
              acc1[r] = math::fma(values[1], weight, acc1[r].read());
            }
          }
        }
      }
      for (std::uint32_t r = 0; r < query_tile; ++r) {
        if (auto row = e.when(q0 + r < queries)) {
          const auto output = e.let((((b * heads + h) * queries + q0 + r) * parts + part) * kSplitRecord);
          e.store(output + 2u + lane * 2u, acc0[r].read());
          e.store(output + 3u + lane * 2u, acc1[r].read());
          if (auto first = e.when(lane == 0u)) {
            e.store(output, maximum[r].read());
            e.store(output + 1u, denom[r].read());
          }
        }
      }
    } else {
      const auto scores = e.lds<kir::f32>(kSplitKeys);
      const auto lane = e.let(math::local_id());
      const auto wg = e.let(math::workgroup_id_x());
      const auto part = e.let(wg % parts);
      const auto query = [&] {
        if constexpr (ShortQuery) return e.let((wg / parts) % queries);
        else return wg;
      }();
      const auto h = e.let((wg / (parts * queries)) % heads);
      const auto b = e.let(wg / (parts * queries * heads));
      const auto kh = e.let(h / (heads / kvheads));
      const auto output = e.let(wg * kSplitRecord);
      const auto rows = e.runtime_extent("rows", kir::cast<kir::u32>(a.meta[2u]));
      const auto mb = e.let(e.u32(kv::kStepMetaHeader) + b * e.u32(kv::kStepMetaPerRow));
      const auto offset = e.let(kir::cast<kir::u32>(a.meta[mb]));
      const auto loaded_len = e.let(kir::cast<kir::u32>(a.meta[mb + 1u]));
      const auto row_len = e.let(select(loaded_len < capacity, loaded_len, e.u32(capacity)));
      const auto loaded_max = e.let(kir::cast<kir::u32>(a.meta[1u]));
      const auto kv_len = e.runtime_extent("kv_len", select(loaded_max < capacity, loaded_max, e.u32(capacity)));
      const auto begin = e.let(part * kSplitKeys);
      const auto tb = e.let(b * stride);
      const auto qb = [&] {
        if constexpr (ShortQuery) return e.let(((b * heads + h) * queries + query) * 256u);
        else return e.let((b * heads + h) * 256u);
      }();
      auto allowed = [&](auto j) {
        if (s.iattrs[0] == 0) return e.u32(1) == e.u32(1);
        if constexpr (ShortQuery) {
          const auto position = e.let(kir::cast<std::int64_t>(offset) +
                                       kir::cast<std::int64_t>(query));
          const auto key = e.let(kir::cast<std::int64_t>(j));
          if (s.iattrs[0] == 1) return key <= position;
          return key <= position && position - key < kir::cast<std::int64_t>(e.u32(static_cast<std::uint32_t>(s.iattrs[1])));
        } else {
          if (s.iattrs[0] == 1) return j <= offset;
          return j <= offset && j + static_cast<std::uint32_t>(s.iattrs[1]) > offset;
        }
      };
      if constexpr (ShortQuery) {
        const auto wl = e.let(lane % 32u);
        const auto wi = e.let(lane / 32u);
        std::vector<kir::Val<kir::f32>> query_values;
        for (std::uint32_t d = 0; d < 8u; ++d)
          query_values.push_back(e.let(a.q[qb + wl + d * 32u]));
        for (auto tile : e.range(kSplitKeys / 4u)) {
          const auto key_lane = e.let(tile * 4u + wi);
          const auto j = e.let(begin + key_lane);
          auto score = e.var(0.0f);
          const auto valid = e.let(b < rows && j < kv_len && j < row_len && allowed(j));
          if (auto live = e.when(valid)) {
            const auto blk = e.let(kir::cast<kir::u32>(a.table[tb + j / block]));
            const auto kb = e.let(((blk * kvheads + kh) * block + j % block) * 256u);
            for (std::uint32_t d = 0; d < 8u; ++d)
              score = math::fma(query_values[d], a.k[kb + wl + d * 32u], score.read());
          }
          for (std::uint32_t shift = 16u; shift; shift >>= 1u)
            score = score.read() + math::shfl_xor(score.read(), e.u32(shift));
          if (auto first = e.when(wl == 0u))
            scores[key_lane] = select(valid, score.read() * s.attrs[0], math::neg_inf());
        }
      } else {
        if (auto owns = e.when(lane < kSplitKeys)) {
          const auto j = e.let(begin + lane);
          scores[lane] = math::neg_inf();
          if (auto live = e.when(b < rows && j < kv_len && j < row_len && allowed(j))) {
            const auto blk = e.let(kir::cast<kir::u32>(a.table[tb + j / block]));
            const auto kb = e.let(((blk * kvheads + kh) * block + j % block) * 256u);
            auto score = e.var(0.0f);
            for (auto dd : e.range(256u))
              score = math::fma(a.q[qb + dd], a.k[kb + dd], score.read());
            scores[lane] = score.read() * s.attrs[0];
          }
        }
      }
      e.barrier();
      auto maximum = e.var(math::neg_inf());
      for (auto jj : e.range(kSplitKeys))
        maximum = math::max(maximum.read(), scores[jj].read());
      e.barrier();
      if (auto owns = e.when(lane < kSplitKeys)) {
        const auto j = e.let(begin + lane);
        auto weight = e.var(0.0f);
        if (auto live = e.when(b < rows && j < kv_len && j < row_len && allowed(j)))
          weight = math::exp(scores[lane].read() - maximum.read());
        scores[lane] = weight.read();
      }
      e.barrier();
      auto denom = e.var(0.0f), acc0 = e.var(0.0f), acc1 = e.var(0.0f);
      for (auto jj : e.range(kSplitKeys)) {
        const auto j = e.let(begin + jj);
        if (auto live = e.when(b < rows && j < kv_len && j < row_len && allowed(j))) {
          const auto weight = e.let(scores[jj].read());
          const auto blk = e.let(kir::cast<kir::u32>(a.table[tb + j / block]));
          const auto vb = e.let(((blk * kvheads + kh) * block + j % block) * 256u + lane * 2u);
          denom = denom.read() + weight;
          const auto values = e.load(a.v, vb, 8u);
          acc0 = math::fma(weight, values[0], acc0.read());
          acc1 = math::fma(values[1], weight, acc1.read());
        }
      }
      e.store(output + 2u + lane * 2u, acc0.read());
      e.store(output + 3u + lane * 2u, acc1.read());
      if (auto first = e.when(lane == 0u)) {
        e.store(output, maximum.read());
        e.store(output + 1u, denom.read());
      }
    }
    return k.lds().ok() ? k.str() : std::string{};
  }
  Result<Shape> infer_shape(std::span<const Shape> in) const override {
    if (in.size() != 5 || in[0].rank() != 4 || in[1].rank() != 4 || in[4].rank() != 2)
      return LSE_ERROR(kInvalidArgument, "split decode attention takes five paged inputs");
    const auto capacity = in[1].dim(2) * in[4].dim(1);
    if constexpr (ShortQuery)
      return Shape{in[0].dim(0), in[0].dim(1), in[0].dim(2),
                   (capacity + kSplitKeys - 1) / kSplitKeys, kSplitRecord};
    else
      return Shape{in[0].dim(0), in[0].dim(1),
                   (capacity + kSplitKeys - 1) / kSplitKeys, kSplitRecord};
  }
  DType infer_dtype(std::span<const DType>) const override { return DType::kF32; }
  static ThreadPlan plan_impl(const KernelShapes& s) {
    ThreadPlan tp;
    tp.workgroup_size[0] = 128;
    auto query_tile = 1u;
    if constexpr (ShortQuery) {
      query_tile = dispatch::attention_shapes::short_query_tile(
          static_cast<std::uint32_t>(s.inputs[0].dim(2)),
          static_cast<std::uint32_t>(s.inputs[4].dim(1) * s.inputs[1].dim(2)));
      tp.workgroup_count[0] = static_cast<std::uint32_t>(s.output.dim(0) * s.output.dim(1) *
          ((s.output.dim(2) + query_tile - 1) / query_tile) * s.output.dim(3));
    } else {
      tp.workgroup_count[0] = static_cast<std::uint32_t>(s.output.elem_count() / kSplitRecord);
    }
    tp.lds_bytes = query_tile * kSplitKeys * sizeof(float);
    return tp;
  }
};


template <bool ShortQuery>
struct SplitMergeWg128C2 final : KernelPrimitive<SplitMergeWg128C2<ShortQuery>> {
  static constexpr std::string_view kName = ShortQuery
      ? "attention.short_merge128.wg128c2.v1" : "attention.decode_merge128.wg128c2.v2";
  static constexpr std::string_view kEntry = ShortQuery
      ? "lse_sdpa_short_merge128_wg128c2_v1" : "lse_sdpa_decode_merge128_wg128c2_v2";
  static constexpr std::string_view kSource = {};
  std::size_t arity() const noexcept override { return 1; }
  bool owns_indexing() const noexcept override { return true; }
  std::string emit_kernel(const KernelShapes& s) const override {
    if (!(ShortQuery ? dispatch::split_short_merge_supported(s) : dispatch::split_decode_merge_supported(s)) || !s.types.scalar || !s.intrinsics || !s.store)
      return {};
    const auto parts = static_cast<std::uint32_t>(s.inputs[0].dim(ShortQuery ? 3 : 2));
    kir::KernelBody k(s.types, *s.intrinsics, workgroup_lds_bytes(s.device));
    k.set_store(s.store);
    SplitMergeArgs<env::Emit> a;
    if (!env::bind(k, a, s)) return {};
    env::Emit e{&k};
    const auto weights = e.lds<kir::f32>(parts);
    const auto lane = e.let(math::local_id()), wg = e.let(math::workgroup_id_x());
    const auto base = e.let(wg * parts * kSplitRecord);
    auto maximum = e.var(math::neg_inf());
    for (auto p : e.range(parts))
      if (auto valid = e.when(a.partial[base + p * kSplitRecord + 1u] > 0.0f))
        maximum = math::max(maximum.read(), a.partial[base + p * kSplitRecord]);
    if (auto owns = e.when(lane < parts)) {
      const auto pb = e.let(base + lane * kSplitRecord);
      auto weight = e.var(0.0f);
      // A zero denominator marks empty and padded partitions; skip their
      // sentinel maximum so an all-empty row never subtracts two infinities.
      if (auto valid = e.when(a.partial[pb + 1u] > 0.0f))
        weight = math::exp(a.partial[pb] - maximum.read());
      weights[lane] = weight.read();
    }
    e.barrier();
    auto denom = e.var(0.0f), acc0 = e.var(0.0f), acc1 = e.var(0.0f);
    for (auto p : e.range(parts)) {
      const auto pb = e.let(base + p * kSplitRecord);
      const auto weight = e.let(weights[p].read());
      denom = math::fma(weight, a.partial[pb + 1u], denom.read());
      const auto values = e.load(a.partial, e.let(pb + 2u + lane * 2u), 8u);
      acc0 = math::fma(weight, values[0], acc0.read());
      acc1 = math::fma(values[1], weight, acc1.read());
    }
    const auto divisor = e.let(select(denom.read() == 0.0f, e.f32(1.0f), denom.read()));
    e.store(wg * 256u + lane * 2u, acc0.read() / divisor);
    e.store(wg * 256u + lane * 2u + 1u, acc1.read() / divisor);
    return k.lds().ok() ? k.str() : std::string{};
  }
  Result<Shape> infer_shape(std::span<const Shape> in) const override {
    if (in.size() != 1 || in[0].rank() != (ShortQuery ? 5 : 4) ||
        in[0].dim(ShortQuery ? 4 : 3) != kSplitRecord)
      return LSE_ERROR(kInvalidArgument, "split decode merge takes one partial record array");
    return Shape{in[0].dim(0), in[0].dim(1), ShortQuery ? in[0].dim(2) : 1, 256};
  }
  DType infer_dtype(std::span<const DType>) const override { return DType::kF32; }
  static ThreadPlan plan_impl(const KernelShapes& s) {
    ThreadPlan tp;
    tp.workgroup_size[0] = 128;
    tp.workgroup_count[0] = static_cast<std::uint32_t>(s.output.elem_count() / 256u);
    tp.lds_bytes = static_cast<std::uint32_t>(s.inputs[0].dim(ShortQuery ? 3 : 2)) * sizeof(float);
    return tp;
  }
};

using SplitDecodePartialWg128C2 = SplitPartialWg128C2<false>;
using SplitShortPartialWg128C2 = SplitPartialWg128C2<true>;
using SplitDecodeMergeWg128C2 = SplitMergeWg128C2<false>;
using SplitShortMergeWg128C2 = SplitMergeWg128C2<true>;
LSE_REGISTER_PRIMITIVE(SplitDecodePartialWg128C2);
LSE_REGISTER_PRIMITIVE(SplitShortPartialWg128C2);
LSE_REGISTER_PRIMITIVE(SplitShortMergeWg128C2);
LSE_REGISTER_PRIMITIVE(SplitDecodeMergeWg128C2);

// Scalar attention supports contiguous (3/4 inputs) and paged (5 inputs) KV.
struct SdpaKernel final : KernelPrimitive<SdpaKernel> {
  static constexpr std::string_view kName = "attention";
  static constexpr std::string_view kEntry = "lse_sdpa";
  static constexpr std::string_view kSource = {};

  std::size_t arity() const noexcept override { return 3; }

  const KernelPrimitiveBase* specialize(const KernelShapes& s) const override {
    switch (dispatch::attention_plan(s)) {
      case dispatch::AttentionPlan::kSharedExp: return &kDecodeSharedExp;
      case dispatch::AttentionPlan::kFlash8:
      case dispatch::AttentionPlan::kFlash12: return flash_sdpa_for(s);
      case dispatch::AttentionPlan::kScalar: return this;
    }
    return this;
  }

  std::string emit_kernel(const KernelShapes& s) const override {
    if (s.inputs.size() < 3 || s.inputs.size() > 5 ||
        s.types.scalar == nullptr || s.intrinsics == nullptr ||
        s.inputs[0].rank() != 4 || s.inputs[1].rank() != 4 ||
        s.inputs[2].rank() != 4) {
      return {};
    }
    const Shape& q = s.inputs[0];
    const Shape& ksh = s.inputs[1];
    const Shape& vsh = s.inputs[2];
    const auto bsz = static_cast<std::uint32_t>(q.dim(0));
    const auto qh = static_cast<std::uint32_t>(q.dim(1));
    const auto tq = static_cast<std::uint32_t>(q.dim(2));
    const auto dh = static_cast<std::uint32_t>(q.dim(3));
    const auto kvh = static_cast<std::uint32_t>(ksh.dim(1));
    // Contiguous: the allocated sequence length. Paged: the block size.
    const auto ts = static_cast<std::uint32_t>(ksh.dim(2));
    const auto dv = static_cast<std::uint32_t>(vsh.dim(3));
    if (qh == 0 || kvh == 0 || qh % kvh != 0 || dh == 0 || ts == 0 || dv == 0) {
      return {};
    }
    const auto group = qh / kvh;
    const float scale = s.attrs[0];
    const int mask = s.iattrs[0];
    const auto window = static_cast<std::uint32_t>(s.iattrs[1]);
    const auto baked_off = static_cast<std::uint32_t>(s.iattrs[2]);
    const bool live_off = s.inputs.size() >= 4;
    const bool paged = s.inputs.size() == 5;

    std::uint32_t stride = 0;
    if (paged) {
      if (s.inputs[4].rank() < 2) return {};
      // The descriptor must contain each row's offset and length.
      if (s.inputs[3].elem_count() <
          static_cast<std::size_t>(
              kv::step_meta_elems(static_cast<std::int32_t>(bsz)))) {
        return {};
      }
      stride = static_cast<std::uint32_t>(s.inputs[4].dim(s.inputs[4].rank() - 1));
      const auto want = static_cast<std::uint32_t>(s.iattrs[3]);
      // The node block size must agree with its pool.
      if (stride == 0 || want != ts) return {};
      // Paged addressing requires a power-of-two block size.
      if (!is_pow2(ts)) return {};
    }

    kir::KernelBody k(s.types, *s.intrinsics);
    SdpaArgs<env::Emit> a;
    if (!env::bind(k, a, s)) return {};
    env::Emit e{&k};
    const auto i = e.thread_id();
    const auto d = e.let(i % dv);
    const auto qi = e.let((i / dv) % tq);
    const auto h = e.let((i / (dv * tq)) % qh);
    const auto b = e.let(i / (dv * tq * qh));
    const auto kh = e.let(h / group);
    const auto qb0 = e.let(((b * qh + h) * tq + qi) * dh);

    // Paged metadata is per row; contiguous metadata is one pass offset.
    kir::Val<kir::u32> offset = e.u32(baked_off);
    kir::Val<kir::u32> row_len = e.u32(0);
    if (paged) {
      const auto mb =
          e.let(e.u32(static_cast<std::uint32_t>(kv::kStepMetaHeader)) +
                b * e.u32(static_cast<std::uint32_t>(kv::kStepMetaPerRow)));
      offset = e.let(kir::cast<kir::u32>(a.meta[mb]));
      row_len = e.let(kir::cast<kir::u32>(a.meta[mb + 1u]));
    } else if (live_off) {
      offset = e.let(kir::cast<kir::u32>(a.meta[0u]));
    }
    const auto abs_i = e.let(offset + qi);

    if (!paged) {
      const auto used = e.let(offset + tq);
      const auto hi = e.let(select(used < e.u32(ts), used, e.u32(ts)));

      const auto m = e.var(math::neg_inf());
      for (auto j : e.range(live_off ? hi : e.u32(ts))) {
        auto score = e.var(0.0f);
        for (auto dd : e.range(dh)) {
          const auto kb = ((b * kvh + kh) * ts + j) * dh + dd;
          score = math::fma(a.q[qb0 + dd], a.k[kb], score.read());
        }
        score = score.read() * scale;
        auto take = [&] { m = math::max(m.read(), score.read()); };
        if (mask == 0) {
          take();
        } else if (mask == 1) {
          if (auto g = e.when(j <= abs_i)) take();
        } else {
          if (auto g = e.when(j <= abs_i && (abs_i - j) < window)) take();
        }
      }

      auto denom = e.var(0.0f);
      auto acc = e.var(0.0f);
      for (auto j : e.range(live_off ? hi : e.u32(ts))) {
        auto score = e.var(0.0f);
        for (auto dd : e.range(dh)) {
          const auto kb = ((b * kvh + kh) * ts + j) * dh + dd;
          score = math::fma(a.q[qb0 + dd], a.k[kb], score.read());
        }
        score = score.read() * scale;
        auto w = e.var(0.0f);
        auto apply = [&] { w = math::exp(score.read() - m.read()); };
        if (mask == 0) {
          apply();
        } else if (mask == 1) {
          if (auto g = e.when(j <= abs_i)) apply();
        } else {
          if (auto g = e.when(j <= abs_i && (abs_i - j) < window)) apply();
        }
        denom = denom.read() + w.read();
        const auto vb = ((b * kvh + kh) * ts + j) * dv + d;
        acc = math::fma(w.read(), a.v[vb], acc.read());
      }
      e.ret(acc.read() /
            select(denom.read() == 0.0f, e.f32(1.0f), denom.read()));
      return k.str();
    }

    // Padded batch rows. A pass is widened to a bucket so the bucket, not the
    // true row count, is what reaches the JIT key; the rows past `rows` read
    // real blocks through their own table row and answer zero. Returning here
    // rather than masking inside the address arithmetic is the width-invariance
    // rule: no real row's result may depend on how many rows shared the pass.
    const auto rows = e.runtime_extent("rows", kir::cast<kir::u32>(a.meta[2u]));
    auto result = e.var(0.0f);
    if (auto live_row = e.when(b < rows)) {
      // The longest live KV in the pass. One code object serves every sequence
      // length: this is the block loop's trip count and nothing else, and an
      // outermost trip count is exactly where ExtentBinding::kRuntime is legal.
      // Raggedness rides underneath it in `row_len`, which is a guard.
      const auto kv_len = e.runtime_extent("kv_len", kir::cast<kir::u32>(a.meta[1u]));
      const auto nblk = e.let((kv_len + e.u32(ts - 1)) / e.u32(ts));
      // Blocks this row actually holds. A row shorter than the longest one spends
      // the remaining iterations on one compare — it loads no table entry and
      // touches no key — so its accumulator sees the same terms in the same order
      // it would have seen decoding alone.
      const auto row_blk = e.let((row_len + e.u32(ts - 1)) / e.u32(ts));
      const auto tb = e.let(b * stride);

      const auto m = e.var(math::neg_inf());
      for (auto bi : e.range(nblk)) {
        if (auto mine = e.when(bi < row_blk)) {
          // One table read per block, not per key: the whole reason a block is 16
          // positions wide is that this load amortizes over them.
          const auto blk = e.let(kir::cast<kir::u32>(a.table[tb + bi]));
          const auto kb0 = e.let(((blk * kvh + kh) * ts) * dh);
          const auto j0 = e.let(bi * e.u32(ts));
          for (auto jj : e.range(ts)) {
            const auto j = e.let(j0 + jj);
            if (auto live = e.when(j < row_len)) {
              auto score = e.var(0.0f);
              for (auto dd : e.range(dh)) {
                score = math::fma(a.q[qb0 + dd], a.k[kb0 + jj * dh + dd],
                                  score.read());
              }
              score = score.read() * scale;
              auto take = [&] { m = math::max(m.read(), score.read()); };
              if (mask == 0) {
                take();
              } else if (mask == 1) {
                if (auto g = e.when(j <= abs_i)) take();
              } else {
                if (auto g = e.when(j <= abs_i && (abs_i - j) < window)) take();
              }
            }
          }
        }
      }

      auto denom = e.var(0.0f);
      auto acc = e.var(0.0f);
      for (auto bi : e.range(nblk)) {
        if (auto mine = e.when(bi < row_blk)) {
          const auto blk = e.let(kir::cast<kir::u32>(a.table[tb + bi]));
          const auto kb0 = e.let(((blk * kvh + kh) * ts) * dh);
          const auto vb0 = e.let(((blk * kvh + kh) * ts) * dv + d);
          const auto j0 = e.let(bi * e.u32(ts));
          for (auto jj : e.range(ts)) {
            const auto j = e.let(j0 + jj);
            if (auto live = e.when(j < row_len)) {
              auto score = e.var(0.0f);
              for (auto dd : e.range(dh)) {
                score = math::fma(a.q[qb0 + dd], a.k[kb0 + jj * dh + dd],
                                  score.read());
              }
              score = score.read() * scale;
              auto w = e.var(0.0f);
              auto apply = [&] { w = math::exp(score.read() - m.read()); };
              if (mask == 0) {
                apply();
              } else if (mask == 1) {
                if (auto g = e.when(j <= abs_i)) apply();
              } else {
                if (auto g = e.when(j <= abs_i && (abs_i - j) < window)) apply();
              }
              denom = denom.read() + w.read();
              acc = math::fma(w.read(), a.v[vb0 + jj * dv], acc.read());
            }
          }
        }
      }
      // A row holding no sequence has row_len 0, so it accumulated nothing and
      // this is the zero it must answer. No separate pad case, and therefore no
      // pad case that can drift out of step with the live one.
      result = acc.read() / select(denom.read() == 0.0f, e.f32(1.0f), denom.read());
    }
    e.ret(result.read());
    return k.str();
  }

  Result<Shape> infer_shape(std::span<const Shape> in) const override {
    if (in.size() < 3 || in.size() > 5) {
      return LSE_ERROR(kInvalidArgument, "sdpa takes 3, 4 or 5 inputs");
    }
    return Shape{in[0].dim(0), in[0].dim(1), in[0].dim(2), in[2].dim(3)};
  }
  DType infer_dtype(std::span<const DType> in) const override {
    return in.empty() ? DType::kF32 : in[0];
  }
  static ThreadPlan plan_impl(const KernelShapes& s) {
    ThreadPlan tp;
    const std::uint32_t threads =
        dispatch::scalar_threads(s);
    const auto elems = static_cast<std::uint32_t>(s.output.elem_count());
    tp.workgroup_size[0] = threads;
    tp.workgroup_count[0] = elems == 0 ? 1u : (elems + threads - 1) / threads;
    return tp;
  }
};
LSE_REGISTER_PRIMITIVE(SdpaKernel);

template <class E>
struct KvPageWriteArgs {
  env::In<kir::f32, E> dst;  // inplace pool; written through the store hook
  env::In<kir::f32, E> src;
  env::In<kir::f32, E> meta;
  env::In<kir::f32, E> table;
  env::Out<kir::f32, E> out;
};

// Write each row to its absolute paged KV position; leave other pool bytes intact.
struct KvPageWriteKernel final : KernelPrimitive<KvPageWriteKernel> {
  static constexpr std::string_view kName = "kv_page_write";
  static constexpr std::string_view kEntry = "lse_kv_page_write";
  static constexpr std::string_view kSource = {};

  std::size_t arity() const noexcept override { return 4; }
  bool owns_indexing() const noexcept override { return true; }
  bool supports_epilogue() const noexcept override { return false; }
  int inplace_input() const noexcept override { return 0; }

  std::string emit_kernel(const KernelShapes& s) const override {
    if (s.inputs.size() != 4 || s.types.scalar == nullptr ||
        s.intrinsics == nullptr || !s.store || s.inputs[0].rank() != 4 ||
        s.inputs[1].rank() != 4 || s.inputs[3].rank() < 2) {
      return {};
    }
    const Shape& dst = s.inputs[0];
    const Shape& src = s.inputs[1];
    const auto kvh = static_cast<std::uint32_t>(dst.dim(1));
    const auto bs = static_cast<std::uint32_t>(dst.dim(2));
    const auto width = static_cast<std::uint32_t>(dst.dim(3));
    const auto t = static_cast<std::uint32_t>(src.dim(2));
    const auto stride =
        static_cast<std::uint32_t>(s.inputs[3].dim(s.inputs[3].rank() - 1));
    if (kvh == 0 || bs == 0 || width == 0 || t == 0 || stride == 0) return {};
    if (static_cast<std::uint32_t>(src.dim(1)) != kvh ||
        static_cast<std::uint32_t>(src.dim(3)) != width) {
      return {};
    }
    if (static_cast<std::uint32_t>(s.iattrs[0]) != bs) return {};
    if (!is_pow2(bs)) return {};
    const auto src_n = static_cast<std::uint32_t>(src.elem_count());
    if (src_n == 0) return {};
    if (s.inputs[2].elem_count() <
        static_cast<std::size_t>(
            kv::step_meta_elems(static_cast<std::int32_t>(src.dim(0))))) {
      return {};
    }

    kir::KernelBody k(s.types, *s.intrinsics);
    k.set_store(s.store);
    KvPageWriteArgs<env::Emit> a;
    if (!env::bind(k, a, s)) return {};
    env::Emit e{&k};
    const auto i = e.thread_id();
    (void)e.ret_if(i >= src_n);

    const auto w = e.let(i % width);
    const auto tt = e.let((i / width) % t);
    const auto h = e.let((i / (width * t)) % kvh);
    const auto r = e.let(i / (width * t * kvh));
    // The same padded-bucket rule the attention kernel follows: pad rows do no
    // work rather than writing somewhere harmless, so a real row's blocks
    // cannot be reached by a row that is not in the batch.
    const auto rows = e.runtime_extent("rows", kir::cast<kir::u32>(a.meta[2u]));
    (void)e.ret_if(r >= rows);
    const auto mb =
        e.let(e.u32(static_cast<std::uint32_t>(kv::kStepMetaHeader)) +
              r * e.u32(static_cast<std::uint32_t>(kv::kStepMetaPerRow)));
    // A row holding no sequence has zero live length. It must not write: the
    // blocks its table row names belong to whoever held the slot last.
    (void)e.ret_if(kir::cast<kir::u32>(a.meta[mb + 1u]) == e.u32(0));

    // Per row, like the attention read: rows of one pass sit at different
    // absolute positions, and a shared one would have every row but one
    // overwrite somebody else's slot.
    const auto pos = e.let(kir::cast<kir::u32>(a.meta[mb]));
    const auto abs = e.let(pos + tt);
    const auto blk =
        e.let(kir::cast<kir::u32>(a.table[r * stride + abs / bs]));
    const auto slot = e.let(abs % bs);
    const auto dest = e.let(((blk * kvh + h) * bs + slot) * width + w);
    e.store(dest, a.src[i]);
    return k.str();
  }

  Result<Shape> infer_shape(std::span<const Shape> in) const override {
    if (in.size() != 4) {
      return LSE_ERROR(kInvalidArgument, "kv_page_write takes 4 inputs");
    }
    return in[0];
  }
  DType infer_dtype(std::span<const DType> in) const override {
    return in.empty() ? DType::kF32 : in[0];
  }

  static ThreadPlan plan_impl(const KernelShapes& s) {
    ThreadPlan tp;
    const std::uint32_t threads =
        dispatch::scalar_threads(s);
    const auto elems = s.inputs.size() > 1
                           ? static_cast<std::uint32_t>(s.inputs[1].elem_count())
                           : 1u;
    tp.workgroup_size[0] = threads;
    tp.workgroup_count[0] = elems == 0 ? 1u : (elems + threads - 1) / threads;
    return tp;
  }
};
LSE_REGISTER_PRIMITIVE(KvPageWriteKernel);

}  // namespace lse::kernels
