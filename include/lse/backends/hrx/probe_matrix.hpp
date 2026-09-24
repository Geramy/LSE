#pragma once
#include "lse/kernels/wmma.hpp"
#include "lse/graph/kernel_env.hpp"
#include "lse/graph/kernel_args.hpp"

namespace lse::backend::hrx_kernels::probe_matrix {
using namespace lse::graph;
using namespace lse::kernels;
// The shape the matrix rows are rated at.
//
// Deliberately small enough that both operands stay resident in L2: the tile
// re-reads its A row and B column for every output tile, so a shape that spills
// to DRAM would measure the same roofline the streaming probe already reports
// and tell the cost model nothing new about the matrix core. K is long so the
// instruction loop, not the tile setup, dominates.
inline constexpr std::uint32_t kRateM = 256;
inline constexpr std::uint32_t kRateN = 256;
inline constexpr std::uint32_t kRateK = 1024;
inline constexpr int kRateReps = 8;

inline constexpr int kTileM = 16;
inline constexpr int kTileN = 16;
inline constexpr int kTileK = 16;

template <class E, class X = kir::f32, class W = kir::f32>
struct RateArgs {
  env::In<X, E> x;
  env::In<W, E> w;
  env::Out<kir::f32, E> out;
};

// The shared tile writes its final accumulator directly in the standalone
// HIP wrapper, or through the existing output hook in the Loom graph emitter.
// The arithmetic and logical output index are identical in both cases.
template <math::MatrixTarget G, math::MatrixElem A, math::MatrixElem T>
struct RateTile : MatrixTile<RateTile<G, A, T>, G, A, T, kTileM, kTileN, kTileK> {
  const env::Out<kir::f32, env::Emit>* out = nullptr;
  std::uint32_t cols = 0;
  bool store_through_hook = false;

  void emit_element(env::Emit& e, const kir::Val<kir::u32>& row,
                    const kir::Val<kir::u32>& col,
                    const kir::Val<kir::f32>& v) const {
    if (store_through_hook) e.store(row * cols + col, v);
    else (*out)[row * cols + col] = v;
  }
};

// The generations whose lane layouts have been measured here. Same pack the
// linear kernel carries, for the same reason: an unmeasured layout is a silent
// wrong answer, and a probe that emitted one would be timing nonsense.
template <math::MatrixTarget... Live, class F>
[[nodiscard]] std::string for_live_target(math::MatrixTarget t, F&& fn) {
  std::string out;
  const auto one = [&]<math::MatrixTarget L>() {
    if (t == L) out = fn.template operator()<L>();
  };
  (one.template operator()<Live>(), ...);
  return out;
}

inline std::string rate_body(const DeviceInfo& info,
    const math::MatrixCoreRow& row, DType storage,
    const kir::TypeTable& types, const DialectSourceTable& table,
    const KernelShapes* shapes = nullptr) {
  const std::uint32_t kb = static_cast<std::uint32_t>(
      kRateK / static_cast<std::uint32_t>(row.pack));

  return for_live_target<math::MatrixTarget::kRdna3,
                                   math::MatrixTarget::kRdna4>(
      row.target, [&]<math::MatrixTarget G>() -> std::string {
        return with_matrix_operand<std::string>(
            storage,
            [&]<class X, class W, math::MatrixElem A, math::MatrixElem T>()
                -> std::string {
              if (A != row.acc || T != row.operand) return {};
              kir::KernelBody k(types, table);
              RateArgs<env::Emit, X, W> a;
              const DType ins[] = {env::elem_dtype<X>::value,
                                   env::elem_dtype<W>::value};
              if (shapes != nullptr) {
                k.set_store(shapes->store);
                if (!env::bind(k, a, *shapes)) return {};
              } else if (!env::bind(k, a, ins, DType::kF32)) return {};
              env::Emit e{&k};
              RateTile<G, A, T> tile;
              tile.out = &a.out;
              tile.cols = kRateN;
              tile.store_through_hook = shapes != nullptr;
              tile.run(e, a.x, a.w, kRateM, kRateN, kb,
                       device_load_bytes(&info));
              return k.str();
            });
      });

}
inline LaunchDims rate_dims(const DeviceInfo& info, const math::MatrixCoreRow& row) {
  const auto lanes = static_cast<std::uint32_t>(row.wave);
  const std::uint32_t tiles_n = (kRateN + kTileN - 1) / kTileN;
  const std::uint32_t tiles = ((kRateM + kTileM - 1) / kTileM) * tiles_n;
  const std::uint32_t cap = info.max_threads_per_workgroup >= 256 ? 256u : 64u;
  const std::uint32_t waves = cap / lanes;
  LaunchDims dims;
  dims.workgroup_size[0] = waves * lanes;
  dims.workgroup_count[0] = (tiles + waves - 1) / waves;
  dims.subgroup_size = lanes;
  return dims;
}
} // namespace lse::backend::hrx_kernels::probe_matrix
