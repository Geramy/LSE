#include "lse/kernels/kv_storage.hpp"
#include "lse/kernels/sdpa.hpp"
// Tiled paged attention shares FP32 scores and carries online softmax state.
#include <string_view>
#include <string>
#include <vector>

#include "lse/backends/hrx/device_info.hpp"
#include "lse/dispatch/attention.hpp"
#include "lse/dispatch/attention_shapes.hpp"
#include "lse/graph/kernel_args.hpp"
#include "lse/graph/kernel_env.hpp"
#include "lse/graph/kernel_primitive.hpp"
#include "lse/kv/block.hpp"
#include "lse/math.hpp"

namespace lse::kernels {

using namespace lse::graph;
namespace math = lse::math;

namespace {

constexpr std::uint32_t kThreads = dispatch::attention_shapes::kFlashThreads;
constexpr std::uint32_t kDefaultQTile = dispatch::attention_shapes::kFlashDefaultQueryTile;
constexpr std::uint32_t kPrefillQTile = dispatch::attention_shapes::kFlashPrefillQueryTile;
constexpr std::uint32_t kKWin = dispatch::attention_shapes::kFlashKeyWindow;
using Dims = dispatch::FlashDims;

template <class E, kv::CacheDType Storage = kv::CacheDType::kF32>
struct FlashArgs {
  env::In<kir::f32, E> q;
  env::In<KvElement<Storage>, E> k;
  env::In<KvElement<Storage>, E> v;
  env::In<kir::f32, E> meta;
  env::In<kir::f32, E> table;
  env::Out<kir::f32, E> out;
};

template <std::uint32_t QTile>
struct FlashSdpaKernel final : KernelPrimitive<FlashSdpaKernel<QTile>> {
  static constexpr std::string_view kName =
      QTile == 12 ? "attention.flash.qtile12.v2" : "attention.flash.v2";
  static constexpr std::string_view kEntry =
      QTile == 12 ? "lse_sdpa_flash_qtile12_v2" : "lse_sdpa_flash_v2";
  static constexpr std::string_view kSource = {};

  std::size_t arity() const noexcept override { return 3; }
  bool owns_indexing() const noexcept override { return true; }
  bool supports_epilogue() const noexcept override { return false; }

  std::string emit_kernel(const KernelShapes& s) const override {
    if (s.input_dtypes.size() < 3) return {};
    return with_kv_storage(s.input_dtypes[1], s.attrs[1], [&]<kv::CacheDType Storage>() {
      return emit_storage<Storage>(s);
    });
  }

  template <kv::CacheDType Storage>
  std::string emit_storage(const KernelShapes& s) const {
    const Dims d = dispatch::flash_dimensions(s);
    if (!dispatch::flash_supported(s, QTile) || s.types.scalar == nullptr || s.intrinsics == nullptr ||
        !s.store) {
      return {};
    }
    const std::uint32_t ntiles = (d.tq + QTile - 1u) / QTile;
    const std::uint32_t qchunks = (QTile * d.dh + kThreads - 1u) / kThreads;
    const std::uint32_t dpt = (d.dv + kThreads - 1u) / kThreads;
    const std::uint32_t blocks_per_win = kKWin / d.ts;

    kir::KernelBody k(s.types, *s.intrinsics, workgroup_lds_bytes(s.device));
    k.set_store(s.store);
    FlashArgs<env::Emit, Storage> a;
    if (!env::bind(k, a, s)) return {};
    env::Emit e{&k};

    const auto qs = e.lds<kir::f32>(QTile * d.dh);
    // The window's scores, then the window's probabilities in place.
    const auto sc = e.lds<kir::f32>(QTile * kKWin);
    // The tree reduction needs the scores intact while it consumes a copy.
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

    const auto rows = e.runtime_extent("rows", kir::cast<kir::u32>(a.meta[2u]));
    // Padded rows write zero without reading a block-table row.
    if (auto pad = e.when(b >= rows)) {
      for (std::uint32_t r = 0; r < QTile; ++r) {
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
    // Clamp live metadata to the block-table capacity.
    const auto capacity = e.u32(d.stride * d.ts);
    const auto loaded_len = e.let(kir::cast<kir::u32>(a.meta[mb + 1u]));
    const auto row_len = e.let(select(loaded_len < capacity, loaded_len, capacity));
    const auto tb = e.let(b * d.stride);

    // The query tile, read once into LDS and then read by every key.
    for (std::uint32_t c = 0; c < qchunks; ++c) {
      const auto idx = e.let(lid + c * kThreads);
      // The tile is QTile * dh floats, which a narrow head leaves smaller
      // than the workgroup. Without this the surplus threads write past `qs`
      // and into the arrays behind it.
      if (auto inb = e.when(idx < QTile * d.dh)) {
        const auto r = e.let(idx / d.dh);
        const auto dd = e.let(idx % d.dh);
        const auto qrow = e.let(q0 + r);
        // Recording guards have no C++ else arm: initialize the padding
        // before conditionally loading a live query row.
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

    // The longest live KV in the pass sets the trip count -- an outermost one,
    // which is the only place a runtime extent is legal. A shorter row spends
    // the remaining windows on compares.
    const auto loaded_max = e.let(kir::cast<kir::u32>(a.meta[1u]));
    const auto kv_len = e.runtime_extent(
        "kv_len", select(loaded_max < capacity, loaded_max, capacity));
    const auto nwin = e.let((kv_len + e.u32(kKWin - 1u)) / e.u32(kKWin));
    for (auto w : e.range(nwin)) {
      const auto wbase = e.let(w * kKWin);
      const auto j = e.let(wbase + lid);

      // One thread, one key, one score per query row of the tile.
      for (std::uint32_t r = 0; r < QTile; ++r) {
        sc[e.let(r * kKWin + lid)] = math::neg_inf();
      }
      if (auto live = e.when(j < row_len)) {
        const auto blk =
            e.let(kv_block_index<Storage>(a.table[e.let(tb + j / d.ts)]));
        const auto kb0 =
            e.let(((blk * d.kvh + kh) * d.ts + j % d.ts) * d.dh);
        if constexpr (dispatch::attention_shapes::flash_reuses_keys(QTile)) {
          // Reuse each key across rows; each row keeps ascending FP32 FMA order.
          std::vector<kir::LValue<kir::f32>> scores;
          scores.reserve(QTile);
          for (std::uint32_t r = 0; r < QTile; ++r) scores.push_back(e.var(0.0f));
          for (auto dd : e.range(d.dh)) {
            const auto kval = e.let(kv_load<Storage>(e, a.k, e.let(kb0 + dd), d.dh));
            for (std::uint32_t r = 0; r < QTile; ++r) {
              scores[r] = math::fma(qs[e.let(r * d.dh + dd)].read(),
                                    kval, scores[r].read());
            }
          }
          for (std::uint32_t r = 0; r < QTile; ++r) {
            const auto abs_i = e.let(offset + (q0 + r));
            const auto sv = e.let(scores[r].read() * d.scale);
            const auto at = e.let(r * kKWin + lid);
            if (d.mask == 0) {
              sc[at] = sv;
            } else if (d.mask == 1) {
              if (auto g = e.when(j <= abs_i)) sc[at] = sv;
            } else {
              const auto distance = e.let(kir::cast<std::int64_t>(abs_i) -
                                           kir::cast<std::int64_t>(j));
              if (auto g = e.when(j <= abs_i &&
                                  distance < kir::cast<std::int64_t>(e.u32(d.window)))) {
                sc[at] = sv;
              }
            }
          }
        } else {
          for (std::uint32_t r = 0; r < QTile; ++r) {
            const auto abs_i = e.let(offset + (q0 + r));
            auto score = e.var(0.0f);
            for (auto dd : e.range(d.dh)) {
              score = math::fma(qs[e.let(r * d.dh + dd)].read(),
                                kv_load<Storage>(e, a.k, e.let(kb0 + dd), d.dh), score.read());
            }
            const auto sv = e.let(score.read() * d.scale);
            const auto at = e.let(r * kKWin + lid);
            if (d.mask == 0) {
              sc[at] = sv;
            } else if (d.mask == 1) {
              if (auto g = e.when(j <= abs_i)) sc[at] = sv;
            } else {
              const auto distance = e.let(kir::cast<std::int64_t>(abs_i) -
                                           kir::cast<std::int64_t>(j));
              if (auto g = e.when(j <= abs_i &&
                                  distance < kir::cast<std::int64_t>(e.u32(d.window)))) {
                sc[at] = sv;
              }
            }
          }
        }
      }
      e.barrier();

      // Window max. Every row reduces on the same step, so the tree costs the
      // barriers of one reduction rather than of QTile of them.
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

      // A row whose keys are all masked has no max to subtract. Subtracting
      // zero instead leaves exp(-inf) = 0, where subtracting -inf is a NaN
      // that would poison the accumulator for every later window.
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

      // The accumulator carries the old max; rescale it to the new one before
      // this window's terms go in.
      for (std::uint32_t r = 0; r < QTile; ++r) {
        const auto al = e.let(arow[e.u32(r)].read());
        for (std::uint32_t p = 0; p < dpt; ++p) {
          o[r * dpt + p] = o[r * dpt + p].read() * al;
        }
      }

      // Keep page traversal in IR; the fixed key body preserves FMA order.
      for (auto bi : e.range(blocks_per_win)) {
        const auto j0 = e.let(wbase + bi * d.ts);
        if (auto held = e.when(j0 < row_len)) {
          const auto blk =
              e.let(kv_block_index<Storage>(a.table[e.let(tb + j0 / d.ts)]));
          const auto vb0 = e.let(((blk * d.kvh + kh) * d.ts) * d.dv);
          if constexpr (kv::packed_cache(Storage)) {
            // Keep packed decoding inside a loop to fit native branch offsets.
            for (auto jj : e.range(d.ts)) {
              const auto slot = e.let(bi * d.ts + jj);
              if (auto live_value = e.when(j0 + jj < row_len)) {
                for (std::uint32_t p = 0; p < dpt; ++p) {
                  const auto dd = e.let(lid + p * kThreads);
                  if (auto g = e.when(dd < d.dv)) {
                    const auto vv = e.let(kv_load<Storage>(
                        e, a.v, e.let(vb0 + jj * d.dv + dd), d.dv));
                    for (std::uint32_t r = 0; r < QTile; ++r) {
                      o[r * dpt + p] = math::fma(
                          sc[e.let(r * kKWin + slot)].read(), vv,
                          o[r * dpt + p].read());
                    }
                  }
                }
              }
            }
          } else {
            for (std::uint32_t jj = 0; jj < d.ts; ++jj) {
              const auto slot = e.let(bi * d.ts + jj);
              if (auto live_value = e.when(j0 + jj < row_len)) {
                for (std::uint32_t p = 0; p < dpt; ++p) {
                  const auto dd = e.let(lid + p * kThreads);
                  if (auto g = e.when(dd < d.dv)) {
                    const auto vv = e.let(kv_load<Storage>(e, a.v, e.let(vb0 + jj * d.dv + dd), d.dv));
                    for (std::uint32_t r = 0; r < QTile; ++r) {
                      o[r * dpt + p] =
                          math::fma(sc[e.let(r * kKWin + slot)].read(), vv,
                                    o[r * dpt + p].read());
                    }
                  }
                }
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
    const Dims d = dispatch::flash_dimensions(s);
    const std::uint32_t ntiles = d.valid ? (d.tq + QTile - 1u) / QTile : 1u;
    tp.workgroup_size[0] = kThreads;
    tp.workgroup_count[0] = d.valid ? d.bsz * d.qh * ntiles : 1u;
    tp.workgroup_count[1] = 1;
    tp.workgroup_count[2] = 1;
    return tp;
  }
};
using FlashSdpaKernel8 = FlashSdpaKernel<kDefaultQTile>;
using FlashSdpaKernel12 = FlashSdpaKernel<kPrefillQTile>;
LSE_REGISTER_PRIMITIVE(FlashSdpaKernel8);
LSE_REGISTER_PRIMITIVE(FlashSdpaKernel12);

const FlashSdpaKernel8 kFlash8{};
const FlashSdpaKernel12 kFlash12{};

}  // namespace

const KernelPrimitiveBase* flash_sdpa_for(const KernelShapes& s) {
  switch (dispatch::attention_plan(s)) {
    case dispatch::AttentionPlan::kFlashWmmaF16: return flash_wmma_sdpa_for(s);
    case dispatch::AttentionPlan::kFlash12: return &kFlash12;
    case dispatch::AttentionPlan::kFlash8: return &kFlash8;
    default: return nullptr;
  }
}

}  // namespace lse::kernels
