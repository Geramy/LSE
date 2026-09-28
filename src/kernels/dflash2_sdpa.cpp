#include <string_view>
#include <string>
#include <vector>

#include "lse/backends/hrx/device_info.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include "lse/graph/kernel_args.hpp"
#include "lse/graph/kernel_env.hpp"
#include "lse/graph/kernel_primitive.hpp"

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
Dims dimensions(const KernelShapes& s) {
  Dims d;
  if (s.inputs.size() != 4 || s.inputs[0].rank() != 4 ||
      s.inputs[1].rank() != 4 || s.inputs[2] != s.inputs[1] ||
      s.inputs[3].elem_count() != 2 || s.inputs[0].dim(0) != 1 ||
      s.inputs[1].dim(0) != 1 || s.inputs[0].dim(2) < 1 ||
      s.inputs[0].dim(2) > 8 || s.inputs[1].dim(1) < 1 ||
      s.inputs[0].dim(1) % s.inputs[1].dim(1) ||
      s.inputs[0].dim(3) != s.inputs[1].dim(3) ||
      s.inputs[1].dim(2) <= s.inputs[0].dim(2) || s.attrs[3] < 2) return d;
  for (const Shape& shape : s.inputs)
    for (std::size_t a = 0; a < shape.rank(); ++a)
      if (shape.dim(a) <= 0 || shape.dim(a) > INT32_MAX) return d;
  if (s.output_dtype != DType::kF32 || s.input_dtypes.size() != 4) return d;
  for (DType type : s.input_dtypes) if (type != DType::kF32) return d;
  d.qh = static_cast<std::uint32_t>(s.inputs[0].dim(1));
  d.kvh = static_cast<std::uint32_t>(s.inputs[1].dim(1));
  d.tq = static_cast<std::uint32_t>(s.inputs[0].dim(2));
  d.dh = d.dv = static_cast<std::uint32_t>(s.inputs[0].dim(3));
  d.length = static_cast<std::uint32_t>(s.inputs[1].dim(2));
  d.capacity = d.length - d.tq;
  d.group = d.qh / d.kvh;
  d.window = static_cast<std::uint32_t>(s.attrs[3]);
  d.scale = 1.0f / std::sqrt(static_cast<float>(d.dh));
  d.valid = true;
  return d;
}

template <class E>
struct FlashArgs {
  env::In<kir::f32, E> q;
  env::In<kir::f32, E> k;
  env::In<kir::f32, E> v;
  env::In<kir::f32, E> meta;
  env::Out<kir::f32, E> out;
};

struct DFlash2SdpaKernel final : KernelPrimitive<DFlash2SdpaKernel> {
  static constexpr std::uint32_t QTile = 8;
  static constexpr std::string_view kName = "dflash2.attention";
  static constexpr std::string_view kEntry = "lse_dflash2_attention";
  static constexpr std::string_view kSource = {};

  std::size_t arity() const noexcept override { return 4; }
  bool owns_indexing() const noexcept override { return true; }
  bool supports_epilogue() const noexcept override { return false; }

  std::string emit_kernel(const KernelShapes& s) const override {
    const Dims d = dimensions(s);
    if (!d.valid || !s.device || s.device->max_threads_per_workgroup < kThreads ||
        static_cast<std::uint64_t>(QTile) * (d.dh + 2ull * kKWin + 4) * sizeof(float) > workgroup_lds_bytes(s.device) || s.types.scalar == nullptr || s.intrinsics == nullptr ||
        !s.store) {
      return {};
    }
    const std::uint32_t ntiles = (d.tq + QTile - 1u) / QTile;
    const std::uint32_t qchunks = (QTile * d.dh + kThreads - 1u) / kThreads;
    const std::uint32_t dpt = (d.dv + kThreads - 1u) / kThreads;

    kir::KernelBody k(s.types, *s.intrinsics, workgroup_lds_bytes(s.device));
    k.set_store(s.store);
    FlashArgs<env::Emit> a;
    if (!env::bind(k, a, s)) return {};
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
    const auto qt = e.let(wg % ntiles);
    const auto h = e.let((wg / ntiles) % d.qh);
    const auto b = e.let(wg / (ntiles * d.qh));
    const auto kh = e.let(h / d.group);
    const auto q0 = e.let(qt * QTile);
    const auto obase = e.let(((b * d.qh + h) * d.tq) * d.dv);

    const auto loaded_live = e.let(kir::cast<kir::u32>(a.meta[0u]));
    const auto context_live = e.let(select(loaded_live < d.capacity, loaded_live, e.u32(d.capacity)));
    const auto begin = e.let(kir::cast<std::int64_t>(e.u32(d.capacity)) -
                             kir::cast<std::int64_t>(context_live));
    const auto row_len = e.u32(d.length);
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
    for (std::uint32_t i = 0; i < QTile * dpt; ++i) o.push_back(e.var(0.0f));
    e.barrier();

    const auto nwin = e.u32((d.length + kKWin - 1u) / kKWin);
    for (auto w : e.range(nwin)) {
      const auto wbase = e.let(w * kKWin);
      const auto j = e.let(wbase + lid);

      for (std::uint32_t r = 0; r < QTile; ++r) {
        sc[e.let(r * kKWin + lid)] = math::neg_inf();
      }
      if (auto live = e.when(j < row_len)) {
        const auto kb0 = e.let(((b * d.kvh + kh) * d.length + j) * d.dh);
        for (std::uint32_t r = 0; r < QTile; ++r) {
          const auto abs_i = e.let(e.u32(d.capacity) + q0 + r);
          auto score = e.var(0.0f);
          for (auto dd : e.range(d.dh)) {
            score = math::fma(qs[e.let(r * d.dh + dd)].read(),
                              a.k[e.let(kb0 + dd)], score.read());
          }
          const auto sv = e.let(score.read() * d.scale);
          const auto at = e.let(r * kKWin + lid);
          if (auto g = e.when(j >= d.capacity ||
                              (kir::cast<std::int64_t>(j) >= begin && j + d.window > abs_i))) {
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
            red[at] = math::max(red[at].read(),
                                red[e.let(at + half)].read());
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
        const auto p = e.let(math::exp(sc[at].read() - srow[e.u32(r)].read()));
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
        if (auto held = e.when(jv < row_len && kir::cast<std::int64_t>(jv) >= begin)) {
          const auto vb0 = e.let(((b * d.kvh + kh) * d.length + jv) * d.dv);
          for (std::uint32_t p = 0; p < dpt; ++p) {
            const auto dd = e.let(lid + p * kThreads);
            if (auto g = e.when(dd < d.dv)) {
              const auto vv = e.let(a.v[e.let(vb0 + dd)]);
              for (std::uint32_t r = 0; r < QTile; ++r) {
                o[r * dpt + p] = math::fma(
                    sc[e.let(r * kKWin + slot)].read(), vv, o[r * dpt + p].read());
              }
            }
          }
        }
      }
      e.barrier();
    }

    for (std::uint32_t r = 0; r < QTile; ++r) {
      const auto qrow = e.let(q0 + r);
      const auto den = e.let(drow[e.u32(r)].read());
      const auto inv =
          e.let(select(den == 0.0f, e.f32(1.0f), den));
      for (std::uint32_t p = 0; p < dpt; ++p) {
        const auto dd = e.let(lid + p * kThreads);
        if (auto g = e.when(qrow < d.tq && dd < d.dv)) {
          e.store(obase + qrow * d.dv + dd, o[r * dpt + p].read() / inv);
        }
      }
    }
    if (!k.lds().ok()) return {};
    return k.str();
  }

  Result<Shape> infer_shape(std::span<const Shape> in) const override {
    if (in.size() != 4 || in[0].rank() != 4 || in[1].rank() != 4 ||
        in[2] != in[1] || in[0].dim(0) != 1 || in[1].dim(0) != 1 ||
        in[0].dim(1) < 1 || in[1].dim(1) < 1 ||
        in[0].dim(1) % in[1].dim(1) || in[0].dim(3) != in[1].dim(3) ||
        in[0].dim(2) < 1 || in[0].dim(2) > 8 ||
        in[1].dim(2) <= in[0].dim(2) || in[3].elem_count() != 2) {
      return LSE_ERROR(kInvalidArgument, "invalid DFlash2 attention geometry");
    }
    return in[0];
  }
  DType infer_dtype(std::span<const DType> in) const override {
    return in.empty() ? DType::kF32 : in[0];
  }

  bool has_host_impl() const noexcept override { return true; }
  void eval_cpu(std::span<const float* const> in, float* out,
                std::size_t count, const std::array<float, 4>& attrs) const override {
    const auto heads = static_cast<std::size_t>(attrs[0]);
    const auto kv_heads = static_cast<std::size_t>(attrs[1]);
    const auto dim = static_cast<std::size_t>(attrs[2]);
    const auto window = static_cast<std::size_t>(attrs[3]);
    const auto queries = count / (heads * dim);
    const auto capacity = static_cast<std::size_t>(in[3][1]);
    const auto live = std::min(capacity, static_cast<std::size_t>(in[3][0]));
    const auto length = capacity + queries;
    const float scale = 1.0f / std::sqrt(static_cast<float>(dim));
    std::vector<float> scores(length);
    for (std::size_t h = 0; h < heads; ++h) {
      const auto kh = h / (heads / kv_heads);
      for (std::size_t q = 0; q < queries; ++q) {
        float maximum = -std::numeric_limits<float>::infinity();
        for (std::size_t j = 0; j < length; ++j) {
          scores[j] = -std::numeric_limits<float>::infinity();
          if (j < capacity && (j < capacity - live || j + window <= capacity + q)) continue;
          float score = 0;
          for (std::size_t d = 0; d < dim; ++d)
            score = std::fma(in[0][(h * queries + q) * dim + d],
                             in[1][(kh * length + j) * dim + d], score);
          scores[j] = score * scale;
          maximum = std::max(maximum, scores[j]);
        }
        float denominator = 0;
        for (float& score : scores) { score = std::exp(score - maximum); denominator += score; }
        for (std::size_t d = 0; d < dim; ++d) {
          float value = 0;
          for (std::size_t j = 0; j < length; ++j)
            if (scores[j] != 0.0f) value = std::fma(scores[j], in[2][(kh * length + j) * dim + d], value);
          out[(h * queries + q) * dim + d] = value / denominator;
        }
      }
    }
  }
  static ThreadPlan plan_impl(const KernelShapes& s) {
    ThreadPlan tp;
    const Dims d = dimensions(s);
    tp.workgroup_size[0] = kThreads;
    tp.workgroup_count[0] = d.valid ? d.qh : 1u;
    tp.lds_bytes = d.valid ? QTile * (d.dh + 2 * kKWin + 4) * sizeof(float) : 0;
    return tp;
  }
};
LSE_REGISTER_PRIMITIVE(DFlash2SdpaKernel);
}  // namespace
}  // namespace lse::kernels
