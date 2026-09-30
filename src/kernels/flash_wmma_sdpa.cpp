// Typed paged KV; FP32 matrix accumulators and online softmax state.
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
#include "lse/kernels/wmma.hpp"
#include "lse/kernels/kv_storage.hpp"
#include "lse/kernels/sdpa.hpp"

namespace lse::kernels {

using namespace lse::graph;
namespace math = lse::math;

namespace {

constexpr std::uint32_t kThreads = dispatch::attention_shapes::kFlashThreads;
constexpr std::uint32_t QTile = dispatch::attention_shapes::kFlashQueryTile;
constexpr std::uint32_t kKWin = dispatch::attention_shapes::kFlashKeyWindow;
using Dims = dispatch::FlashDims;

template <class E, kv::CacheDType Storage>
struct FlashArgs {
  env::In<kir::f32, E> q;
  env::In<KvElement<Storage>, E> k;
  env::In<KvElement<Storage>, E> v;
  env::In<kir::f32, E> meta;
  env::In<kir::f32, E> table;
  env::Out<kir::f32, E> out;
};

// Q/P and decoded KV use F16 operands for F16 storage, BF16 otherwise.
// Both matrix accumulators, softmax state and output remain FP32.
struct FlashWmma final : KernelPrimitive<FlashWmma> {
  static constexpr std::string_view kName = "attention.flash.wmma16.v2";
  static constexpr std::string_view kEntry = "lse_flash_wmma16_v2";
  static constexpr std::string_view kSource = {};

  std::size_t arity() const noexcept override { return 5; }
  bool owns_indexing() const noexcept override { return true; }
  bool supports_epilogue() const noexcept override { return false; }

  std::string emit_kernel(const KernelShapes& s) const override {
    if (s.input_dtypes.size() != 5 || !dispatch::flash_wmma_supported(s) || !s.store)
      return {};
    return with_kv_storage(s.input_dtypes[1], s.attrs[1],
                           [&]<kv::CacheDType Storage>() { return emit_storage<Storage>(s); });
  }

  template <kv::CacheDType Storage>
  std::string emit_storage(const KernelShapes& s) const {
    constexpr auto operand = Storage == kv::CacheDType::kF16
                                 ? math::MatrixElem::kF16 : math::MatrixElem::kBF16;
    using Mma = math::op::Mma<math::MatrixTarget::kRdna4, math::MatrixElem::kF32,
                              operand, 16, 16, 16>;
    using Narrow = math::matrix_scalar_t<Mma::kRow.a_elem>;
    const Dims d = dispatch::flash_dimensions(s);
    const auto mask = d.mask == 2 && d.window == 0u ? 1 : d.mask;
    const std::uint32_t ntiles = (d.tq + QTile - 1u) / QTile;
    const std::uint32_t padded_depth = (d.dh + 15u) / 16u * 16u;
    const std::uint32_t qchunks = (QTile * padded_depth + kThreads - 1u) / kThreads;
    const std::uint32_t dpt = (d.dv + kThreads - 1u) / kThreads;
    const std::uint32_t value_tiles = (d.dv + 127u) / 128u;

    kir::KernelBody k(s.types, *s.intrinsics, workgroup_lds_bytes(s.device));
    k.set_store(s.store);
    FlashArgs<env::Emit, Storage> a;
    if (!env::bind(k, a, s)) return {};
    env::Emit e{&k};

    const auto qs = e.lds<Narrow>(QTile * padded_depth);
    // The window's scores, then the window's probabilities in place.
    const auto sc = e.lds<kir::f32>(QTile * kKWin);
    const auto mrow = e.lds<kir::f32>(QTile);
    const auto drow = e.lds<kir::f32>(QTile);
    const auto arow = e.lds<kir::f32>(QTile);

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

    // Stage each query once.
    for (std::uint32_t c = 0; c < qchunks; ++c) {
      const auto idx = e.let(lid + c * kThreads);
      if (auto inb = e.when(idx < QTile * padded_depth)) {
        const auto r = e.let(idx / padded_depth);
        const auto dd = e.let(idx % padded_depth);
        const auto qrow = e.let(q0 + r);
        qs[idx] = math::narrow<Narrow>(e.f32(0.0f));
        if (auto g = e.when(qrow < d.tq && dd < d.dh)) {
          const auto qvalue = e.let(a.q[e.let(((b * d.qh + h) * d.tq + qrow) * d.dh + dd)]);
          qs[idx] = math::narrow<Narrow>(qvalue);
        }
      }
    }
    if (auto g = e.when(lid < QTile)) {
      mrow[lid] = math::neg_inf();
      drow[lid] = e.f32(0.0f);
    }

    const auto wave = e.let(lid / 32u);
    const auto lane = e.let(lid % 32u);
    const auto lane_lo = e.let(lane % 16u);
    const auto lane_hi = e.let(lane / 16u);
    std::vector<decltype(e.local<kir::f32, 8>())> o;
    o.reserve(value_tiles);
    for (std::uint32_t tile = 0; tile < value_tiles; ++tile)
      o.push_back(e.local<kir::f32, 8>());
    for (auto& fragment : o)
      for (auto f : e.unroll(8u)) fragment[f] = e.f32(0.0f);
    const auto last_query = e.let(select(q0 + (QTile - 1u) < d.tq,
                                        q0 + (QTile - 1u), e.u32(d.tq - 1u)));
    const auto first_position = e.let(kir::cast<std::int64_t>(offset) +
                                      kir::cast<std::int64_t>(q0));
    const auto last_position = e.let(kir::cast<std::int64_t>(offset) +
                                     kir::cast<std::int64_t>(last_query));
    e.barrier();

    // The window bound is shared by the workgroup.
    const auto loaded_max = e.let(kir::cast<kir::u32>(a.meta[1u]));
    const auto kv_len = e.runtime_extent(
        "kv_len", select(loaded_max < capacity, loaded_max, capacity));
    const auto nwin = e.let((kv_len + e.u32(kKWin - 1u)) / e.u32(kKWin));
    for (auto w : e.range(nwin)) {
      const auto wbase = e.let(w * kKWin);

      const auto signed_start = e.let(kir::cast<std::int64_t>(wbase));
      const auto useful = mask == 0 ? wbase < row_len
                                   : wbase < row_len && signed_start <= last_position;
      if (auto live_window = e.when(useful)) {
        auto write_score = [&](const kir::Val<kir::u32>& row,
                               const kir::Val<kir::u32>& key,
                               const kir::Val<kir::f32>& value) {
          const auto position = e.let(kir::cast<std::int64_t>(offset) +
                                      kir::cast<std::int64_t>(q0) + kir::cast<std::int64_t>(row));
          const auto signed_key = e.let(kir::cast<std::int64_t>(key));
          const auto slot = e.let(row * kKWin + key - wbase);
          sc[slot] = math::neg_inf();
          if (auto live_score = e.when(key < row_len && q0 + row < d.tq)) {
            if (mask == 0) sc[slot] = value;
            else if (mask == 1) {
              if (auto causal = e.when(signed_key <= position)) sc[slot] = value;
            } else {
              const auto distance = e.let(position - signed_key);
              if (auto sliding = e.when(signed_key <= position && distance < kir::cast<std::int64_t>(e.u32(d.window))))
                sc[slot] = value;
            }
          }
        };
        for (std::uint32_t column_tile = 0; column_tile < 2; ++column_tile) {
          const auto key = e.let(wbase + (wave + column_tile * 8u) * 16u + lane_lo);
          auto acc = e.local<kir::f32, 8>();
          for (auto f : e.unroll(8u)) acc[f] = e.f32(0.0f);
          for (auto depth : e.range(0u, padded_depth, 16u)) {
            auto af = e.local<Narrow, 8>();
            auto bf = e.local<Narrow, 8>();
            const auto kk = e.let(depth + lane_hi * 8u);
            for (auto f : e.unroll(8u)) {
              af[f] = qs[e.let(lane_lo * padded_depth + kk + f)].read();
              bf[f] = math::narrow<Narrow>(e.f32(0.0f));
            }
            if (auto key_live = e.when(key < row_len)) {
              const auto page = e.let(kir::cast<kir::u32>(a.table[e.let(tb + key / d.ts)]));
              const auto vector = e.let((page * d.kvh + kh) * d.ts + key % d.ts);
              const auto base = e.let(vector * d.dh + kk);
              if constexpr (Storage == kv::CacheDType::kF16 || Storage == kv::CacheDType::kBF16) {
                if (d.dh % 16u == 0) {
                  const auto packed = e.load(a.k, base, 16u);
                  for (auto f : e.unroll(8u)) bf[f] = packed[f];
                } else {
                  for (auto f : e.unroll(8u))
                    if (auto tail = e.when(kk + f < d.dh)) bf[f] = a.k[e.let(base + f)];
                }
              } else {
                for (auto f : e.unroll(8u)) {
                  auto load_key = [&] {
                    bf[f] = math::narrow<Narrow>(kv_load_vector<Storage>(e, a.k, vector, e.let(kk + f), d.dh));
                  };
                  if (d.dh % 16u == 0) load_key();
                  else if (auto tail = e.when(kk + f < d.dh)) load_key();
                }
              }
            }
            acc = math::mma<Mma>(af.value(), bf.value(), acc.value());
          }
          for (auto f : e.unroll(8u))
            write_score(e.let(f + lane_hi * 8u), key, e.let(acc[f].read() * d.scale));
        }
        e.barrier();

        // A wave owns two rows; each lane owns eight contiguous-bank keys.
        for (std::uint32_t wave_row = 0; wave_row < 2; ++wave_row) {
          const auto row = e.let(wave * 2u + wave_row);
          auto maximum = e.var(math::neg_inf());
          for (auto f : e.unroll(8u)) {
            const auto slot = e.let(lane + f * 32u);
            maximum = math::max(maximum.read(), sc[e.let(row * kKWin + slot)].read());
          }
          for (std::uint32_t shift = 16u; shift > 0; shift /= 2u)
            maximum = math::max(maximum.read(), math::shfl_xor(maximum.read(), e.u32(shift)));
          const auto old_max = e.let(mrow[row].read());
          const auto new_max = e.let(math::max(old_max, maximum.read()));
          const auto empty = e.let(new_max == math::neg_inf());
          const auto safe_max = e.let(select(empty, e.f32(0.0f), new_max));
          const auto alpha = e.let(select(empty, e.f32(1.0f), math::exp(old_max - safe_max)));
          auto total = e.var(0.0f);
          for (auto f : e.unroll(8u)) {
            const auto slot = e.let(lane + f * 32u);
            const auto index = e.let(row * kKWin + slot);
            const auto probability = e.let(math::exp(sc[index].read() - safe_max));
            sc[index] = probability;
            total = total.read() + probability;
          }
          for (std::uint32_t shift = 16u; shift > 0; shift /= 2u)
            total = total.read() + math::shfl_xor(total.read(), e.u32(shift));
          if (auto leader = e.when(lane == 0u)) {
            mrow[row] = new_max;
            arow[row] = alpha;
            drow[row] = math::fma(drow[row].read(), alpha, total.read());
          }
        }
        e.barrier();

        // A wave owns one D16 tile per 128-channel slice; C remains FP32.
        for (std::uint32_t column_tile = 0; column_tile < value_tiles; ++column_tile) {
          const auto dimension = e.let((wave + column_tile * 8u) * 16u + lane_lo);
          for (auto f : e.unroll(8u)) {
            const auto row = e.let(f + lane_hi * 8u);
            o[column_tile][f] = o[column_tile][f].read() * arow[row].read();
          }
          for (auto tile : e.range(0u, kKWin, 16u)) {
            auto af = e.local<Narrow, 8>();
            auto bf = e.local<Narrow, 8>();
            const auto key_half = e.let(lane_hi * 8u);
            for (auto f : e.unroll(8u)) {
              const auto key_slot = e.let(tile + key_half + f);
              const auto key = e.let(wbase + key_slot);
              af[f] = math::narrow<Narrow>(sc[e.let(lane_lo * kKWin + key_slot)].read());
              bf[f] = math::narrow<Narrow>(e.f32(0.0f));
              auto load_value = [&] {
                const auto page = e.let(kir::cast<kir::u32>(a.table[e.let(tb + key / d.ts)]));
                const auto vector = e.let((page * d.kvh + kh) * d.ts + key % d.ts);
                if constexpr (Storage == kv::CacheDType::kF16 || Storage == kv::CacheDType::kBF16)
                  bf[f] = a.v[e.let(vector * d.dv + dimension)];
                else
                  bf[f] = math::narrow<Narrow>(kv_load_vector<Storage>(e, a.v, vector, dimension, d.dv));
              };
              if (auto live_key = e.when(key < row_len && dimension < d.dv)) {
                if (mask == 0) load_value();
                else {
                  const auto signed_key = e.let(kir::cast<std::int64_t>(key));
                  if (mask == 1) {
                    if (auto used = e.when(signed_key <= last_position)) load_value();
                  } else {
                    if (auto used = e.when(signed_key <= last_position &&
                          first_position - signed_key < kir::cast<std::int64_t>(e.u32(d.window))))
                      load_value();
                  }
                }
              }
            }
            o[column_tile] = math::mma<Mma>(af.value(), bf.value(), o[column_tile].value());
          }
        }
        e.barrier();
      }
    }

    for (std::uint32_t column_tile = 0; column_tile < value_tiles; ++column_tile) {
      const auto dimension = e.let((wave + column_tile * 8u) * 16u + lane_lo);
      for (auto f : e.unroll(8u)) {
        const auto row = e.let(f + lane_hi * 8u);
        const auto qrow = e.let(q0 + row);
        const auto den = e.let(drow[row].read());
        const auto inv = e.let(select(den == 0.0f, e.f32(1.0f), den));
        if (auto live_query = e.when(qrow < d.tq && dimension < d.dv))
          e.store(obase + qrow * d.dv + dimension, o[column_tile][f].read() / inv);
      }
    }
    if (!k.lds().ok()) return {};
    return k.str();
  }

  Result<Shape> infer_shape(std::span<const Shape>) const override {
    // The attention builder knows the packed storage tag and logical V width.
    return LSE_ERROR(kInvalidArgument, "WMMA attention is selected through sdpa_paged");
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
LSE_REGISTER_PRIMITIVE(FlashWmma);
const FlashWmma kFlashWmma{};
}  // namespace

const KernelPrimitiveBase* flash_wmma_sdpa() { return &kFlashWmma; }
}  // namespace lse::kernels
