// Attention with the score computed once.
//
// The thread-per-output kernel beside this one gives every one of the Dv
// threads of a row its own copy of the whole score vector, and a two-pass
// softmax makes it compute that copy twice: the same Q.K dot is evaluated
// 2*Dv times. At Dv 256 that is a five-hundred-fold arithmetic overhead, which
// does not show on the generate path -- one query, a handful of keys -- and is
// the entire prefill on a long prompt.
//
// Here a workgroup owns one (sequence, head, query tile). A score is computed
// by one thread, published in LDS, and read by every thread that needs it, so
// the arithmetic is the 2*S*Dh the algebra actually calls for. The softmax is
// the running form: one pass over the keys carrying a max and a denominator,
// rescaling the accumulator when a window raises the max.
#include <string>
#include <vector>

#include "lse/backends/hrx/device_info.hpp"
#include "lse/graph/kernel_args.hpp"
#include "lse/graph/kernel_env.hpp"
#include "lse/graph/kernel_primitive.hpp"
#include "lse/kv/block.hpp"
#include "lse/math.hpp"

namespace lse::kernels {

using namespace lse::graph;
namespace math = lse::math;

namespace {

constexpr std::uint32_t kThreads = 256;
// Query rows a workgroup carries. Every one of them reuses the key the window
// already loaded, so this is what amortizes the KV read.
constexpr std::uint32_t kQTile = 8;
// Keys a window covers, one per thread: a thread owns a key for the score and
// a channel for the accumulation, and neither mapping needs a cross-lane
// primitive this IR does not have.
constexpr std::uint32_t kKWin = kThreads;

std::string dispatch_u32(const kir::KernelBody& k, const kir::TypeTable& types,
                         std::size_t input, unsigned element) {
  return "(" + std::string(types.scalar(kir::Scalar::kU32)) + ")(" +
         k.input_name(input) + "[" + std::to_string(element) + "u])";
}

constexpr bool is_pow2(std::uint32_t v) noexcept {
  return v >= 2 && (v & (v - 1)) == 0;
}

struct Dims {
  std::uint32_t bsz = 0, qh = 0, tq = 0, dh = 0, kvh = 0, ts = 0, dv = 0;
  std::uint32_t group = 0, stride = 0, window = 0;
  float scale = 0.0f;
  int mask = 0;
  bool valid = false;
};

Dims dims_of(const KernelShapes& s) {
  Dims d;
  // The paged form only. The contiguous form is the single-sequence path and
  // carries no block table to walk.
  if (s.inputs.size() != 5) return d;
  if (s.inputs[0].rank() != 4 || s.inputs[1].rank() != 4 ||
      s.inputs[2].rank() != 4 || s.inputs[4].rank() < 2) {
    return d;
  }
  const Shape& q = s.inputs[0];
  const Shape& ksh = s.inputs[1];
  const Shape& vsh = s.inputs[2];
  d.bsz = static_cast<std::uint32_t>(q.dim(0));
  d.qh = static_cast<std::uint32_t>(q.dim(1));
  d.tq = static_cast<std::uint32_t>(q.dim(2));
  d.dh = static_cast<std::uint32_t>(q.dim(3));
  d.kvh = static_cast<std::uint32_t>(ksh.dim(1));
  d.ts = static_cast<std::uint32_t>(ksh.dim(2));
  d.dv = static_cast<std::uint32_t>(vsh.dim(3));
  if (d.qh == 0 || d.kvh == 0 || d.qh % d.kvh != 0) return d;
  if (d.dh == 0 || d.ts == 0 || d.dv == 0 || d.tq == 0) return d;
  if (!is_pow2(d.ts) || kKWin % d.ts != 0) return d;
  d.group = d.qh / d.kvh;
  d.stride = static_cast<std::uint32_t>(s.inputs[4].dim(s.inputs[4].rank() - 1));
  if (d.stride == 0) return d;
  if (static_cast<std::uint32_t>(s.iattrs[3]) != d.ts) return d;
  if (s.inputs[3].elem_count() <
      static_cast<std::size_t>(
          kv::step_meta_elems(static_cast<std::int32_t>(d.bsz)))) {
    return d;
  }
  d.scale = s.attrs[0];
  d.mask = s.iattrs[0];
  d.window = static_cast<std::uint32_t>(s.iattrs[1]);
  d.valid = true;
  return d;
}

std::uint32_t lds_floats(const Dims& d) {
  return kQTile * d.dh + 2u * kQTile * kKWin + 4u * kQTile;
}

template <class E>
struct FlashArgs {
  env::In<kir::f32, E> q;
  env::In<kir::f32, E> k;
  env::In<kir::f32, E> v;
  env::In<kir::f32, E> meta;
  env::In<kir::f32, E> table;
  env::Out<kir::f32, E> out;
};

struct FlashSdpaKernel final : KernelPrimitive<FlashSdpaKernel> {
  static constexpr std::string_view kName = "attention.flash";
  static constexpr std::string_view kEntry = "lse_sdpa_flash";
  static constexpr std::string_view kSource = {};

  std::size_t arity() const noexcept override { return 3; }
  bool owns_indexing() const noexcept override { return true; }
  bool supports_epilogue() const noexcept override { return false; }

  std::string emit_kernel(const KernelShapes& s) const override {
    const Dims d = dims_of(s);
    if (!d.valid || s.types.scalar == nullptr || s.intrinsics == nullptr ||
        !s.store) {
      return {};
    }
    const std::uint32_t ntiles = (d.tq + kQTile - 1u) / kQTile;
    const std::uint32_t qchunks = (kQTile * d.dh + kThreads - 1u) / kThreads;
    const std::uint32_t dpt = (d.dv + kThreads - 1u) / kThreads;
    const std::uint32_t blocks_per_win = kKWin / d.ts;

    kir::KernelBody k(s.types, *s.intrinsics, workgroup_lds_bytes(s.device));
    k.set_store(s.store);
    FlashArgs<env::Emit> a;
    if (!env::bind(k, a, s)) return {};
    env::Emit e{&k};

    const auto qs = e.lds<kir::f32>(kQTile * d.dh);
    // The window's scores, then the window's probabilities in place.
    const auto sc = e.lds<kir::f32>(kQTile * kKWin);
    // The tree reduction needs the scores intact while it consumes a copy.
    const auto red = e.lds<kir::f32>(kQTile * kKWin);
    const auto mrow = e.lds<kir::f32>(kQTile);
    const auto drow = e.lds<kir::f32>(kQTile);
    const auto arow = e.lds<kir::f32>(kQTile);
    const auto srow = e.lds<kir::f32>(kQTile);

    const auto lid = e.let(math::local_id());
    const auto wg = e.let(math::workgroup_id_x());
    const auto qt = e.let(wg % ntiles);
    const auto h = e.let((wg / ntiles) % d.qh);
    const auto b = e.let(wg / (ntiles * d.qh));
    const auto kh = e.let(h / d.group);
    const auto q0 = e.let(qt * kQTile);
    const auto obase = e.let(((b * d.qh + h) * d.tq) * d.dv);

    const auto rows = e.runtime_extent("rows", dispatch_u32(k, s.types, 3, 2));
    // A pass widened to a bucket carries rows that are not in the batch. They
    // answer zero, the same as the kernel this replaces, rather than reading a
    // block table row that belongs to nobody.
    if (auto pad = e.when(b >= rows)) {
      for (std::uint32_t r = 0; r < kQTile; ++r) {
        for (std::uint32_t p = 0; p < dpt; ++p) {
          const auto qrow = e.let(q0 + r);
          const auto dd = e.let(lid + p * kThreads);
          if (auto g = e.when(qrow < d.tq && dd < d.dv)) {
            e.store(obase + qrow * d.dv + dd, e.f32(0.0f));
          }
        }
      }
    }
    (void)e.ret_if(b >= rows);

    const auto mb =
        e.let(e.u32(static_cast<std::uint32_t>(kv::kStepMetaHeader)) +
              b * e.u32(static_cast<std::uint32_t>(kv::kStepMetaPerRow)));
    const auto offset = e.let(kir::cast<kir::u32>(a.meta[mb]));
    const auto row_len = e.let(kir::cast<kir::u32>(a.meta[mb + 1u]));
    const auto tb = e.let(b * d.stride);

    // The query tile, read once into LDS and then read by every key.
    for (std::uint32_t c = 0; c < qchunks; ++c) {
      const auto idx = e.let(lid + c * kThreads);
      // The tile is kQTile * dh floats, which a narrow head leaves smaller
      // than the workgroup. Without this the surplus threads write past `qs`
      // and into the arrays behind it.
      if (auto inb = e.when(idx < kQTile * d.dh)) {
        const auto r = e.let(idx / d.dh);
        const auto dd = e.let(idx % d.dh);
        const auto qrow = e.let(q0 + r);
        if (auto g = e.when(qrow < d.tq)) {
          qs[idx] = a.q[e.let(((b * d.qh + h) * d.tq + qrow) * d.dh + dd)];
        } else {
          qs[idx] = e.f32(0.0f);
        }
      }
    }
    if (auto g = e.when(lid < kQTile)) {
      mrow[lid] = math::neg_inf();
      drow[lid] = e.f32(0.0f);
    }

    std::vector<kir::LValue<kir::f32>> o;
    o.reserve(kQTile * dpt);
    for (std::uint32_t i = 0; i < kQTile * dpt; ++i) o.push_back(e.var(0.0f));
    e.barrier();

    // The longest live KV in the pass sets the trip count -- an outermost one,
    // which is the only place a runtime extent is legal. A shorter row spends
    // the remaining windows on compares.
    const auto kv_len = e.runtime_extent("kv_len", dispatch_u32(k, s.types, 3, 1));
    const auto nwin = e.let((kv_len + e.u32(kKWin - 1u)) / e.u32(kKWin));
    for (auto w : e.range(nwin)) {
      const auto wbase = e.let(w * kKWin);
      const auto j = e.let(wbase + lid);

      // One thread, one key, one score per query row of the tile.
      for (std::uint32_t r = 0; r < kQTile; ++r) {
        sc[e.let(r * kKWin + lid)] = math::neg_inf();
      }
      if (auto live = e.when(j < row_len)) {
        const auto blk =
            e.let(kir::cast<kir::u32>(a.table[e.let(tb + j / d.ts)]));
        const auto kb0 =
            e.let(((blk * d.kvh + kh) * d.ts + j % d.ts) * d.dh);
        for (std::uint32_t r = 0; r < kQTile; ++r) {
          const auto abs_i = e.let(offset + (q0 + r));
          auto score = e.var(0.0f);
          for (auto dd : e.range(d.dh)) {
            score = math::fma(qs[e.let(r * d.dh + dd)].read(),
                              a.k[e.let(kb0 + dd)], score.read());
          }
          const auto sv = e.let(score.read() * d.scale);
          const auto at = e.let(r * kKWin + lid);
          if (d.mask == 0) {
            sc[at] = sv;
          } else if (d.mask == 1) {
            if (auto g = e.when(j <= abs_i)) sc[at] = sv;
          } else {
            if (auto g = e.when(j <= abs_i && (abs_i - j) < d.window)) {
              sc[at] = sv;
            }
          }
        }
      }
      e.barrier();

      // Window max. Every row reduces on the same step, so the tree costs the
      // barriers of one reduction rather than of kQTile of them.
      for (std::uint32_t r = 0; r < kQTile; ++r) {
        red[e.let(r * kKWin + lid)] = sc[e.let(r * kKWin + lid)].read();
      }
      e.barrier();
      for (std::uint32_t half = kKWin / 2u; half >= 1u; half /= 2u) {
        if (auto g = e.when(lid < half)) {
          for (std::uint32_t r = 0; r < kQTile; ++r) {
            const auto at = e.let(r * kKWin + lid);
            red[at] = math::max(red[at].read(),
                                red[e.let(at + half)].read());
          }
        }
        e.barrier();
      }

      // A row whose keys are all masked has no max to subtract. Subtracting
      // zero instead leaves exp(-inf) = 0, where subtracting -inf is a NaN
      // that would poison the accumulator for every later window.
      if (auto g = e.when(lid < kQTile)) {
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

      for (std::uint32_t r = 0; r < kQTile; ++r) {
        const auto at = e.let(r * kKWin + lid);
        const auto p = e.let(math::exp(sc[at].read() - srow[e.u32(r)].read()));
        sc[at] = p;
        red[at] = p;
      }
      e.barrier();
      for (std::uint32_t half = kKWin / 2u; half >= 1u; half /= 2u) {
        if (auto g = e.when(lid < half)) {
          for (std::uint32_t r = 0; r < kQTile; ++r) {
            const auto at = e.let(r * kKWin + lid);
            red[at] = red[at].read() + red[e.let(at + half)].read();
          }
        }
        e.barrier();
      }
      if (auto g = e.when(lid < kQTile)) {
        drow[lid] = math::fma(drow[lid].read(), arow[lid].read(),
                              red[e.let(lid * kKWin)].read());
      }

      // The accumulator carries the old max; rescale it to the new one before
      // this window's terms go in.
      for (std::uint32_t r = 0; r < kQTile; ++r) {
        const auto al = e.let(arow[e.u32(r)].read());
        for (std::uint32_t p = 0; p < dpt; ++p) {
          o[r * dpt + p] = o[r * dpt + p].read() * al;
        }
      }

      // One table read per block, and the value it names is spent on every
      // query row of the tile before it is dropped.
      for (std::uint32_t bi = 0; bi < blocks_per_win; ++bi) {
        const auto j0 = e.let(wbase + bi * d.ts);
        if (auto held = e.when(j0 < row_len)) {
          const auto blk =
              e.let(kir::cast<kir::u32>(a.table[e.let(tb + j0 / d.ts)]));
          const auto vb0 = e.let(((blk * d.kvh + kh) * d.ts) * d.dv);
          for (std::uint32_t jj = 0; jj < d.ts; ++jj) {
            const std::uint32_t slot = bi * d.ts + jj;
            for (std::uint32_t p = 0; p < dpt; ++p) {
              const auto dd = e.let(lid + p * kThreads);
              if (auto g = e.when(dd < d.dv)) {
                const auto vv = e.let(a.v[e.let(vb0 + jj * d.dv + dd)]);
                for (std::uint32_t r = 0; r < kQTile; ++r) {
                  o[r * dpt + p] =
                      math::fma(sc[e.u32(r * kKWin + slot)].read(), vv,
                                o[r * dpt + p].read());
                }
              }
            }
          }
        }
      }
      e.barrier();
    }

    for (std::uint32_t r = 0; r < kQTile; ++r) {
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
    if (in.size() != 5) {
      return LSE_ERROR(kInvalidArgument, "flash sdpa takes 5 inputs");
    }
    return Shape{in[0].dim(0), in[0].dim(1), in[0].dim(2), in[2].dim(3)};
  }
  DType infer_dtype(std::span<const DType> in) const override {
    return in.empty() ? DType::kF32 : in[0];
  }

  static ThreadPlan plan_impl(const KernelShapes& s) {
    ThreadPlan tp;
    const Dims d = dims_of(s);
    const std::uint32_t ntiles = d.valid ? (d.tq + kQTile - 1u) / kQTile : 1u;
    tp.workgroup_size[0] = kThreads;
    tp.workgroup_count[0] = d.valid ? d.bsz * d.qh * ntiles : 1u;
    tp.workgroup_count[1] = 1;
    tp.workgroup_count[2] = 1;
    return tp;
  }
};
LSE_REGISTER_PRIMITIVE(FlashSdpaKernel);

const FlashSdpaKernel kFlash{};

}  // namespace

// Prefill only. One query row per workgroup would leave the key it loaded
// unshared, so the tile has to be worth filling; the generate path is one row
// against a long cache and stays on the kernel that streams it.
const KernelPrimitiveBase* flash_sdpa_for(const KernelShapes& s) {
  const Dims d = dims_of(s);
  if (!d.valid) return nullptr;
  if (d.tq < kQTile) return nullptr;
  if (s.device == nullptr) return nullptr;
  if (s.device->max_threads_per_workgroup < kThreads) return nullptr;
  if (lds_floats(d) * 4u > workgroup_lds_bytes(s.device)) return nullptr;
  return &kFlash;
}

}  // namespace lse::kernels
