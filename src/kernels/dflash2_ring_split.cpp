#include <string>
#include <string_view>
#include <vector>

#include "lse/backends/hrx/device_info.hpp"
#include "lse/graph/kernel_args.hpp"
#include "lse/graph/kernel_env.hpp"
#include "lse/graph/kernel_primitive.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

#include "lse/math.hpp"

namespace lse::kernels {

using namespace lse::graph;
namespace math = lse::math;

namespace {

constexpr std::uint32_t kThreads = 256;

constexpr std::uint32_t kKWin = kThreads;
struct Dims {
  std::uint32_t bsz = 1, qh = 0, kvh = 0, tq = 0, dh = 0, dv = 0;
  std::uint32_t group = 0, capacity = 0, length = 0, window = 0;
  float scale = 0;
  bool valid = false;
};
Dims dimensions(const KernelShapes &s) {
  Dims d;
  if (s.inputs.size() != 6 || s.inputs[0].rank() != 4 ||
      s.inputs[1].rank() != 4 || s.inputs[2] != s.inputs[1] ||
      s.inputs[5].elem_count() != 3 || s.inputs[0].dim(0) != 1 ||
      s.inputs[1].dim(0) != 1 || s.inputs[0].dim(2) < 1 ||
      s.inputs[0].dim(2) > 8 || s.inputs[1].dim(1) < 1 ||
      s.inputs[0].dim(1) % s.inputs[1].dim(1) ||
      s.inputs[0].dim(3) != s.inputs[1].dim(3) ||
      s.inputs[1].dim(2) < s.inputs[0].dim(2) || s.attrs[3] < 2 ||
      s.inputs[3] != Shape{1, s.inputs[1].dim(1), s.inputs[0].dim(2),
                           s.inputs[0].dim(3)} ||
      s.inputs[4] != s.inputs[3])
    return d;
  for (const Shape &shape : s.inputs)
    for (std::size_t a = 0; a < shape.rank(); ++a)
      if (shape.dim(a) <= 0 || shape.dim(a) > INT32_MAX)
        return d;
  if (s.output_dtype != DType::kF32 || s.input_dtypes.size() != 6)
    return d;
  for (DType type : s.input_dtypes)
    if (type != DType::kF32)
      return d;
  d.qh = static_cast<std::uint32_t>(s.inputs[0].dim(1));
  d.kvh = static_cast<std::uint32_t>(s.inputs[1].dim(1));
  d.tq = static_cast<std::uint32_t>(s.inputs[0].dim(2));
  d.dh = d.dv = static_cast<std::uint32_t>(s.inputs[0].dim(3));
  d.capacity = static_cast<std::uint32_t>(s.inputs[1].dim(2));
  d.length = d.capacity + d.tq;
  d.group = d.qh / d.kvh;
  d.window = static_cast<std::uint32_t>(s.attrs[3]);
  d.scale = 1.0f / std::sqrt(static_cast<float>(d.dh));
  d.valid = true;
  return d;
}

template <class E> struct FlashArgs {
  env::In<kir::f32, E> q;
  env::In<kir::f32, E> k;
  env::In<kir::f32, E> v;
  env::In<kir::f32, E> proposal_k;
  env::In<kir::f32, E> proposal_v;
  env::In<kir::f32, E> meta;
  env::Out<kir::f32, E> out;
};

struct DFlash2RingPartial final : KernelPrimitive<DFlash2RingPartial> {
  static constexpr std::uint32_t QTile = 8;
  static constexpr std::string_view kName = "dflash2.ring_partial256.v1";
  static constexpr std::string_view kEntry = "lse_dflash2_ring_partial256_v1";
  static constexpr std::string_view kSource = {};

  std::size_t arity() const noexcept override { return 6; }
  bool owns_indexing() const noexcept override { return true; }
  bool supports_epilogue() const noexcept override { return false; }

  std::string emit_kernel(const KernelShapes &s) const override {
    const Dims d = dimensions(s);
    if (!d.valid || !s.device ||
        s.device->max_threads_per_workgroup < kThreads ||
        static_cast<std::uint64_t>(QTile) * (d.dh + 2ull * kKWin + 4) *
                sizeof(float) >
            workgroup_lds_bytes(s.device) ||
        s.types.scalar == nullptr || s.intrinsics == nullptr || !s.store) {
      return {};
    }
    const std::uint32_t parts = (d.length + kKWin - 1u) / kKWin;
    const std::uint32_t qchunks = (QTile * d.dh + kThreads - 1u) / kThreads;
    const std::uint32_t dpt = (d.dv + kThreads - 1u) / kThreads;

    kir::KernelBody k(s.types, *s.intrinsics, workgroup_lds_bytes(s.device));
    k.set_store(s.store);
    FlashArgs<env::Emit> a;
    if (!env::bind(k, a, s))
      return {};
    env::Emit e{&k};

    const auto qs = e.lds<kir::f32>(QTile * d.dh);
    const auto sc = e.lds<kir::f32>(QTile * kKWin);
    const auto red = e.lds<kir::f32>(QTile * kKWin);
    const auto mrow = e.lds<kir::f32>(QTile);
    const auto drow = e.lds<kir::f32>(QTile);
    const auto arow = e.lds<kir::f32>(QTile);
    const auto srow = e.lds<kir::f32>(QTile);

    const auto lid = e.let(math::local_id());
    const auto wg = e.let(math::workgroup_id_x());
    const auto qt = e.u32(0);
    const auto part = e.let(wg % parts);
    const auto h = e.let((wg / parts) % d.qh);
    const auto b = e.let(wg / (parts * d.qh));
    const auto kh = e.let(h / d.group);
    const auto q0 = e.let(qt * QTile);
    const auto obase = e.let((h * parts + part) * d.tq * (d.dv + 2u));

    const auto loaded_live = e.let(kir::cast<kir::u32>(a.meta[0u]));
    const auto context_live =
        e.let(select(loaded_live < d.capacity, loaded_live, e.u32(d.capacity)));
    const auto begin = e.let(kir::cast<std::int64_t>(e.u32(d.capacity)) -
                             kir::cast<std::int64_t>(context_live));
    const auto row_len = e.u32(d.length);
    const auto ring_end = e.let(kir::cast<kir::u32>(a.meta[2u]) % d.capacity);
    const auto load_key = [&](const kir::Val<kir::u32> &j,
                              const kir::Val<kir::u32> &dd) {
      auto value = e.var(0.0f);
      if (auto context = e.when(j < d.capacity))
        value =
            a.k[((kh * d.capacity + (ring_end + j) % d.capacity) * d.dh) + dd];
      if (auto proposal = e.when(j >= d.capacity))
        value = a.proposal_k[((kh * d.tq + j % d.capacity) * d.dh) + dd];
      return value.read();
    };
    const auto load_value = [&](const kir::Val<kir::u32> &j,
                                const kir::Val<kir::u32> &dd) {
      auto value = e.var(0.0f);
      if (auto context = e.when(j < d.capacity))
        value =
            a.v[((kh * d.capacity + (ring_end + j) % d.capacity) * d.dv) + dd];
      if (auto proposal = e.when(j >= d.capacity))
        value = a.proposal_v[((kh * d.tq + j % d.capacity) * d.dv) + dd];
      return value.read();
    };
    for (std::uint32_t c = 0; c < qchunks; ++c) {
      const auto idx = e.let(lid + c * kThreads);
      if (auto inb = e.when(idx < QTile * d.dh)) {
        const auto r = e.let(idx / d.dh);
        const auto dd = e.let(idx % d.dh);
        const auto qrow = e.let(q0 + r);
        qs[idx] = e.f32(0.0f);
        if (auto g = e.when(qrow < d.tq)) {
          qs[idx] = a.q[e.let(((b * d.qh + h) * d.tq + qrow) * d.dh + dd)];
        }
      }
    }
    if (auto g = e.when(lid < QTile)) {
      mrow[lid] = math::neg_inf();
      drow[lid] = e.f32(0.0f);
    }

    std::vector<kir::LValue<kir::f32>> o;
    o.reserve(QTile * dpt);
    for (std::uint32_t i = 0; i < QTile * dpt; ++i)
      o.push_back(e.var(0.0f));
    e.barrier();

    // Keep reductions and value accumulation inside the uniform window region.
    const auto nwin = e.u32((d.length + kKWin - 1u) / kKWin);
    for (auto w : e.range(nwin)) {
      if (auto selected = e.when(w == part)) {
        const auto wbase = e.let(w * kKWin);
        const auto j = e.let(wbase + lid);

        for (std::uint32_t r = 0; r < QTile; ++r) {
          sc[e.let(r * kKWin + lid)] = math::neg_inf();
        }
        if (auto live =
                e.when(j < row_len && kir::cast<std::int64_t>(j) >= begin)) {
          std::vector<kir::LValue<kir::f32>> scores;
          scores.reserve(QTile);
          for (std::uint32_t r = 0; r < QTile; ++r)
            scores.push_back(e.var(0.0f));
          for (auto dd : e.range(d.dh)) {
            const auto key = e.let(load_key(j, dd));
            for (std::uint32_t r = 0; r < QTile; ++r)
              scores[r] = math::fma(qs[e.let(r * d.dh + dd)].read(), key,
                                    scores[r].read());
          }
          for (std::uint32_t r = 0; r < QTile; ++r) {
            const auto abs_i = e.let(e.u32(d.capacity) + q0 + r);
            const auto sv = e.let(scores[r].read() * d.scale);
            const auto at = e.let(r * kKWin + lid);
            if (auto g = e.when(j >= d.capacity ||
                                (kir::cast<std::int64_t>(j) >= begin &&
                                 j + d.window > abs_i))) {
              sc[at] = sv;
            }
          }
        }
        e.barrier();

        for (std::uint32_t r = 0; r < QTile; ++r) {
          red[e.let(r * kKWin + lid)] = sc[e.let(r * kKWin + lid)].read();
        }
        e.barrier();
        for (std::uint32_t half = kKWin / 2u; half >= 1u; half /= 2u) {
          if (auto g = e.when(lid < half)) {
            for (std::uint32_t r = 0; r < QTile; ++r) {
              const auto at = e.let(r * kKWin + lid);
              red[at] = math::max(red[at].read(), red[e.let(at + half)].read());
            }
          }
          e.barrier();
        }

        if (auto g = e.when(lid < QTile)) {
          const auto wmax = e.let(red[e.let(lid * kKWin)].read());
          const auto mold = e.let(mrow[lid].read());
          const auto newm = e.let(math::max(mold, wmax));
          const auto empty = e.let(newm == math::neg_inf());
          const auto msafe = e.let(select(empty, e.f32(0.0f), newm));
          srow[lid] = msafe;
          arow[lid] = select(empty, e.f32(1.0f), math::exp(mold - msafe));
          mrow[lid] = newm;
        }
        e.barrier();

        for (std::uint32_t r = 0; r < QTile; ++r) {
          const auto at = e.let(r * kKWin + lid);
          const auto p =
              e.let(math::exp(sc[at].read() - srow[e.u32(r)].read()));
          sc[at] = p;
          red[at] = p;
        }
        e.barrier();
        for (std::uint32_t half = kKWin / 2u; half >= 1u; half /= 2u) {
          if (auto g = e.when(lid < half)) {
            for (std::uint32_t r = 0; r < QTile; ++r) {
              const auto at = e.let(r * kKWin + lid);
              red[at] = red[at].read() + red[e.let(at + half)].read();
            }
          }
          e.barrier();
        }
        if (auto g = e.when(lid < QTile)) {
          drow[lid] = math::fma(drow[lid].read(), arow[lid].read(),
                                red[e.let(lid * kKWin)].read());
        }

        for (std::uint32_t r = 0; r < QTile; ++r) {
          const auto al = e.let(arow[e.u32(r)].read());
          for (std::uint32_t p = 0; p < dpt; ++p) {
            o[r * dpt + p] = o[r * dpt + p].read() * al;
          }
        }

        for (auto slot : e.range(kKWin)) {
          const auto jv = e.let(wbase + slot);
          if (auto held = e.when(jv < row_len &&
                                 kir::cast<std::int64_t>(jv) >= begin)) {
            for (std::uint32_t p = 0; p < dpt; ++p) {
              const auto dd = e.let(lid + p * kThreads);
              if (auto g = e.when(dd < d.dv)) {
                const auto vv = e.let(load_value(jv, dd));
                for (std::uint32_t r = 0; r < QTile; ++r) {
                  o[r * dpt + p] = math::fma(sc[e.let(r * kKWin + slot)].read(),
                                             vv, o[r * dpt + p].read());
                }
              }
            }
          }
        }
        e.barrier();
      }
    }

    if (auto row = e.when(lid < d.tq)) {
      e.store(obase + lid * (d.dv + 2u), mrow[lid].read());
      e.store(obase + lid * (d.dv + 2u) + 1u, drow[lid].read());
    }
    for (std::uint32_t r = 0; r < QTile; ++r) {
      const auto qrow = e.let(q0 + r);
      for (std::uint32_t p = 0; p < dpt; ++p) {
        const auto dd = e.let(lid + p * kThreads);
        if (auto g = e.when(qrow < d.tq && dd < d.dv))
          e.store(obase + qrow * (d.dv + 2u) + 2u + dd, o[r * dpt + p].read());
      }
    }
    if (!k.lds().ok())
      return {};
    return k.str();
  }

  Result<Shape> infer_shape(std::span<const Shape> in) const override {
    if (in.size() != 6 || in[0].rank() != 4 || in[1].rank() != 4 ||
        in[2] != in[1] || in[0].dim(0) != 1 || in[1].dim(0) != 1 ||
        in[0].dim(1) < 1 || in[1].dim(1) < 1 || in[0].dim(1) % in[1].dim(1) ||
        in[0].dim(3) != in[1].dim(3) || in[0].dim(2) < 1 || in[0].dim(2) > 8 ||
        in[1].dim(2) < in[0].dim(2) || in[5].elem_count() != 3 ||
        in[3] != Shape{1, in[1].dim(1), in[0].dim(2), in[0].dim(3)} ||
        in[4] != in[3]) {
      return LSE_ERROR(kInvalidArgument, "invalid DFlash2 attention geometry");
    }
    return Shape{1, in[0].dim(1),
                 (in[1].dim(2) + in[0].dim(2) + kKWin - 1) / kKWin,
                 in[0].dim(2), in[0].dim(3) + 2};
  }
  DType infer_dtype(std::span<const DType> in) const override {
    return in.empty() ? DType::kF32 : in[0];
  }

  bool has_host_impl() const noexcept override { return true; }
  void eval_cpu(std::span<const float *const> in, float *out, std::size_t count,
                const std::array<float, 4> &attrs) const override {
    const auto heads = static_cast<std::size_t>(attrs[0]);
    const auto kv_heads = static_cast<std::size_t>(attrs[1]);
    const auto dim = static_cast<std::size_t>(attrs[2]);
    const auto window = static_cast<std::size_t>(attrs[3]);
    const auto capacity = static_cast<std::size_t>(in[5][1]);
    const auto live = std::min(capacity, static_cast<std::size_t>(in[5][0]));
    const auto ring_end = static_cast<std::size_t>(in[5][2]) % capacity;
    std::size_t queries = 0, parts = 0;
    for (std::size_t q = 1; q <= 8; ++q) {
      const auto n = (capacity + q + kKWin - 1) / kKWin;
      if (count == heads * n * q * (dim + 2)) {
        queries = q;
        parts = n;
        break;
      }
    }
    if (!queries) {
      std::fill_n(out, count, 0.0f);
      return;
    }
    const auto key = [&](std::size_t kh, std::size_t j, std::size_t d) {
      return j < capacity
                 ? in[1][(kh * capacity + (ring_end + j) % capacity) * dim + d]
                 : in[3][(kh * queries + j - capacity) * dim + d];
    };
    const auto value = [&](std::size_t kh, std::size_t j, std::size_t d) {
      return j < capacity
                 ? in[2][(kh * capacity + (ring_end + j) % capacity) * dim + d]
                 : in[4][(kh * queries + j - capacity) * dim + d];
    };
    const float scale = 1.0f / std::sqrt(static_cast<float>(dim));
    std::vector<float> scores(kKWin);
    for (std::size_t h = 0; h < heads; ++h)
      for (std::size_t p = 0; p < parts; ++p)
        for (std::size_t q = 0; q < queries; ++q) {
          const auto start = p * kKWin,
                     end = std::min(start + kKWin, capacity + queries);
          float maximum = -std::numeric_limits<float>::infinity();
          std::fill(scores.begin(), scores.end(), maximum);
          for (std::size_t j = start; j < end; ++j) {
            if (j < capacity &&
                (j < capacity - live || j + window <= capacity + q))
              continue;
            float score = 0;
            for (std::size_t d = 0; d < dim; ++d)
              score = std::fma(in[0][(h * queries + q) * dim + d],
                               key(h / (heads / kv_heads), j, d), score);
            scores[j - start] = score * scale;
            maximum = std::max(maximum, scores[j - start]);
          }
          const auto base = ((h * parts + p) * queries + q) * (dim + 2);
          out[base] = maximum;
          out[base + 1] = 0;
          std::fill_n(out + base + 2, dim, 0.0f);
          if (maximum == -std::numeric_limits<float>::infinity())
            continue;
          for (std::size_t j = start; j < end; ++j) {
            const float weight = std::exp(scores[j - start] - maximum);
            out[base + 1] += weight;
            if (weight != 0)
              for (std::size_t d = 0; d < dim; ++d)
                out[base + 2 + d] =
                    std::fma(weight, value(h / (heads / kv_heads), j, d),
                             out[base + 2 + d]);
          }
        }
  }
  static ThreadPlan plan_impl(const KernelShapes &s) {
    ThreadPlan tp;
    const Dims d = dimensions(s);
    tp.workgroup_size[0] = kThreads;
    tp.workgroup_count[0] =
        d.valid ? d.qh * ((d.length + kKWin - 1u) / kKWin) : 1u;
    tp.lds_bytes = d.valid ? QTile * (d.dh + 2 * kKWin + 4) * sizeof(float) : 0;
    return tp;
  }
};

template <class E> struct MergeArgs {
  env::In<kir::f32, E> partial;
  env::Out<kir::f32, E> output;
};
struct DFlash2RingMerge final : KernelPrimitive<DFlash2RingMerge> {
  static constexpr std::string_view kName = "dflash2.ring_merge256.v1";
  static constexpr std::string_view kEntry = "lse_dflash2_ring_merge256_v1";
  static constexpr std::string_view kSource = {};
  std::size_t arity() const noexcept override { return 1; }
  bool owns_indexing() const noexcept override { return true; }
  bool supports_epilogue() const noexcept override { return false; }
  Result<Shape> infer_shape(std::span<const Shape> in) const override {
    if (in.size() != 1 || in[0].rank() != 5 || in[0].dim(0) != 1 ||
        in[0].dim(1) < 1 || in[0].dim(2) < 1 || in[0].dim(3) < 1 ||
        in[0].dim(3) > 8 || in[0].dim(4) < 3)
      return LSE_ERROR(kInvalidArgument, "ring merge requires partial records");
    return Shape{1, in[0].dim(1), in[0].dim(3), in[0].dim(4) - 2};
  }
  DType infer_dtype(std::span<const DType>) const override {
    return DType::kF32;
  }
  std::string emit_kernel(const KernelShapes &s) const override {
    if (!infer_shape(s.inputs).ok() || s.input_dtypes.size() != 1 ||
        s.input_dtypes[0] != DType::kF32 || s.output_dtype != DType::kF32 ||
        !s.types.scalar || !s.intrinsics || !s.store || !s.device ||
        s.device->max_threads_per_workgroup < 128)
      return {};
    const auto parts = static_cast<std::uint32_t>(s.inputs[0].dim(2)),
               queries = static_cast<std::uint32_t>(s.inputs[0].dim(3)),
               record = static_cast<std::uint32_t>(s.inputs[0].dim(4));
    const auto dim = record - 2u;
    kir::KernelBody k(s.types, *s.intrinsics, workgroup_lds_bytes(s.device));
    k.set_store(s.store);
    MergeArgs<env::Emit> a;
    if (!env::bind(k, a, s))
      return {};
    env::Emit e{&k};
    const auto index = e.let(math::workgroup_id_x() * 128u + math::local_id());
    if (auto valid = e.when(index < s.output.elem_count())) {
      const auto row = e.let(index / dim), head = e.let(row / queries),
                 q = e.let(row % queries), d = e.let(index % dim);
      const auto base = e.let((head * parts * queries + q) * record);
      auto maximum = e.var(math::neg_inf());
      for (auto p : e.range(parts)) {
        const auto pb = e.let(base + p * queries * record);
        if (auto live = e.when(a.partial[pb + 1u] > 0.0f))
          maximum = math::max(maximum.read(), a.partial[pb]);
      }
      auto denominator = e.var(0.0f), numerator = e.var(0.0f);
      for (auto p : e.range(parts)) {
        const auto pb = e.let(base + p * queries * record);
        auto weight = e.var(0.0f);
        if (auto live = e.when(a.partial[pb + 1u] > 0.0f))
          weight = math::exp(a.partial[pb] - maximum.read());
        denominator =
            math::fma(weight.read(), a.partial[pb + 1u], denominator.read());
        numerator =
            math::fma(weight.read(), a.partial[pb + 2u + d], numerator.read());
      }
      const auto divisor = e.let(
          select(denominator.read() == 0.0f, e.f32(1.0f), denominator.read()));
      e.store(index, numerator.read() / divisor);
    }
    return k.str();
  }
  bool has_host_impl() const noexcept override { return true; }
  void eval_cpu(std::span<const float *const> in, float *out, std::size_t count,
                const std::array<float, 4> &attrs) const override {
    const auto queries = static_cast<std::size_t>(attrs[1]),
               dim = static_cast<std::size_t>(attrs[2]),
               parts = static_cast<std::size_t>(attrs[3]);
    for (std::size_t row = 0; row < count / dim; ++row) {
      const auto base =
          ((row / queries) * parts * queries + row % queries) * (dim + 2);
      float maximum = -std::numeric_limits<float>::infinity();
      for (std::size_t p = 0; p < parts; ++p)
        if (in[0][base + p * queries * (dim + 2) + 1] > 0)
          maximum = std::max(maximum, in[0][base + p * queries * (dim + 2)]);
      float denominator = 0;
      std::fill_n(out + row * dim, dim, 0.0f);
      for (std::size_t p = 0; p < parts; ++p) {
        const auto pb = base + p * queries * (dim + 2);
        const float weight =
            in[0][pb + 1] > 0 ? std::exp(in[0][pb] - maximum) : 0.0f;
        denominator = std::fma(weight, in[0][pb + 1], denominator);
        for (std::size_t d = 0; d < dim; ++d)
          out[row * dim + d] =
              std::fma(weight, in[0][pb + 2 + d], out[row * dim + d]);
      }
      for (std::size_t d = 0; d < dim; ++d)
        out[row * dim + d] /= denominator == 0 ? 1 : denominator;
    }
  }
  static ThreadPlan plan_impl(const KernelShapes &s) {
    ThreadPlan tp;
    tp.workgroup_size[0] = 128;
    tp.workgroup_count[0] =
        static_cast<std::uint32_t>((s.output.elem_count() + 127) / 128);
    return tp;
  }
};
LSE_REGISTER_PRIMITIVE(DFlash2RingMerge);
LSE_REGISTER_PRIMITIVE(DFlash2RingPartial);
} // namespace
} // namespace lse::kernels
