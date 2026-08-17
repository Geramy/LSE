// `linear` / `linear_indexed` for the decode-shaped case: few rows, wide N.
//
// W is [N, K] (or [E, N, K]). One thread per column walks a whole row, so
// neighbouring threads are K floats apart. One wave per column, lanes on
// consecutive K, is a coalesced row read; the wave xor-reduces in registers.
#include "lse/backends/hrx/kernels/lds_linear.hpp"

#include <string>

#include "lse/backends/hrx/kernels/vec_mem.hpp"
#include "lse/backends/hrx/device_info.hpp"
#include "lse/graph/kernel_args.hpp"
#include "lse/graph/kernel_env.hpp"
#include "lse/math.hpp"

namespace lse::backend::hrx_kernels {

using namespace lse::graph;
namespace math = lse::math;

namespace {

// One wave owns one output column. A 256-thread group is 8 waves of 32
// or 4 waves of 64.
constexpr std::uint32_t kBlock = 256;
constexpr std::int64_t kMaxRows = 16;

std::uint32_t wave_of(const DeviceInfo* device) {
  if (device == nullptr) return 32;
  const std::uint32_t wave = device->wavefront_size;
  return (wave == 32 || wave == 64) ? wave : 32u;
}

struct LinearDims {
  std::int64_t m = 0;
  std::int64_t n = 0;
  std::int64_t k = 0;
  bool valid = false;
};

LinearDims dims_of_linear(const KernelShapes& s) {
  LinearDims d;
  if (s.inputs.size() != 2) return d;
  const Shape& x = s.inputs[0];
  const Shape& w = s.inputs[1];
  if (x.rank() == 0 || w.rank() != 2) return d;
  d.k = x.dim(x.rank() - 1);
  d.n = w.dim(0);
  if (d.k <= 0 || d.n <= 0 || w.dim(1) != d.k) return d;
  const std::int64_t elems = static_cast<std::int64_t>(s.output.elem_count());
  if (elems <= 0 || elems % d.n != 0) return d;
  d.m = elems / d.n;
  d.valid = d.m > 0;
  return d;
}

LinearDims dims_of_indexed(const KernelShapes& s) {
  LinearDims d;
  if (s.inputs.size() != 3) return d;
  const Shape& x = s.inputs[0];
  const Shape& w = s.inputs[1];
  if (x.rank() == 0 || w.rank() != 3) return d;
  d.k = x.dim(x.rank() - 1);
  d.n = w.dim(1);
  if (d.k <= 0 || d.n <= 0 || w.dim(2) != d.k) return d;
  const std::int64_t elems = static_cast<std::int64_t>(s.output.elem_count());
  if (elems <= 0 || elems % d.n != 0) return d;
  d.m = elems / d.n;
  d.valid = d.m > 0;
  return d;
}

bool device_fits(const KernelShapes& s) {
  if (s.device == nullptr || s.intrinsics == nullptr) return false;
  if (s.intrinsics->find("wave.shfl_xor").empty()) return false;
  return device_extension<AmdDeviceInfo>(*s.device) != nullptr;
}

bool lds_shape_ok(const LinearDims& d) {
  return d.valid && d.m > 0 && d.m < kMaxRows && d.n >= 16 && d.k >= 16;
}

ThreadPlan gemv_plan(const LinearDims& d, std::uint32_t wave) {
  if (wave != 32 && wave != 64) wave = 32;
  const std::uint32_t waves = kBlock / wave;
  ThreadPlan tp;
  tp.workgroup_size[0] = kBlock;
  tp.workgroup_size[1] = 1;
  tp.workgroup_size[2] = 1;
  tp.workgroup_count[0] =
      static_cast<std::uint32_t>((d.n + waves - 1) / waves);
  tp.workgroup_count[1] = static_cast<std::uint32_t>(d.m);
  tp.workgroup_count[2] = 1;
  return tp;
}

using F32In = env::In<kir::f32, env::Emit>;
template <class W>
using WIn = env::In<W, env::Emit>;

// `xs` is the activation row already staged in workgroup scratch; null means
// read it from global, which is what a non-grid launch still does.
template <class W>
void emit_tile(env::Emit& e, const F32In& x, const kir::Tile<kir::f32>* xs,
               const WIn<W>& w,
               const kir::Val<kir::u32>& w_base, const kir::Val<kir::u32>& tile,
               const kir::Val<kir::u32>& row, const kir::Val<kir::u32>& wave_id,
               const kir::Val<kir::u32>& lane, std::uint32_t N, std::uint32_t K,
               std::uint32_t wave, std::uint32_t load_bytes) {
  const auto waves = kBlock / wave;
  const auto col = e.let(tile * waves + wave_id);
  constexpr std::uint32_t we = kir::pack_elem_bytes<W>();
  const auto step = row_pack(K, load_bytes, we);
  const auto span = wave * step;
  const auto aligned = (K / span) * span;

  auto acc = e.var(0.0f);
  if (auto in_cols = e.when(col < N)) {
    for (auto k0 : e.range(0u, aligned, span)) {
      const auto kk = e.let(k0 + lane * step);
      // The weight load must stay the only global load in this loop: adding a
      // second one splits the load clause and costs a full DRAM round trip
      // (measured 41 -> 68 us). The activation comes from LDS for that reason.
      const auto wv = e.load(w, (w_base + col) * K + kk, step * we);
      for (auto elem : e.unroll(step)) {
        const auto xe = xs != nullptr ? (*xs)[kk + elem].read()
                                      : x[row * K + kk + elem];
        acc = math::fma(xe, math::widen(wv[elem]), acc.read());
      }
    }
    if (aligned < K) {
      for (auto kt : e.range(e.u32(aligned) + lane, e.u32(K), wave)) {
        const auto xe =
            xs != nullptr ? (*xs)[kt].read() : x[row * K + kt];
        acc = math::fma(xe, math::widen(w[(w_base + col) * K + kt]),
                        acc.read());
      }
    }
  }

  // Wave xor-reduce: columns stay independent, no LDS, no WG barrier.
  for (std::uint32_t m = 1; m < wave; m <<= 1) {
    acc = acc.read() + math::shfl_xor(acc.read(), e.u32(m));
  }
  if (auto lane0 = e.when(lane == 0 && col < N)) {
    e.store(row * N + col, acc.read());
  }
}

kir::Val<kir::u32> gemv_w_base(env::Emit& e, const F32In* idx,
                               std::uint32_t keep, std::uint32_t slot,
                               const kir::Val<kir::u32>& row, std::uint32_t N) {
  if (idx == nullptr) return e.u32(0);
  const auto expert =
      e.let(kir::cast<kir::u32>((*idx)[row * keep + e.u32(slot)]));
  return e.let(expert * N);
}

}  // namespace

template <class W>
void emit_gemv(env::Emit& e, const F32In& x, const WIn<W>& w, const F32In* idx,
               std::uint32_t keep, std::uint32_t slot, std::uint32_t N,
               std::uint32_t K, std::uint32_t M, std::uint32_t load_bytes,
               bool grid, std::uint32_t wave, bool persist,
               std::uint32_t persist_wgs) {
  if (wave != 32 && wave != 64) wave = 32;
  if (persist_wgs == 0) persist_wgs = 1;
  const auto lid = e.let(math::local_id());
  const auto wave_id = e.let(lid / wave);
  const auto lane = e.let(lid % wave);
  const auto ntiles = (N + (kBlock / wave) - 1) / (kBlock / wave);

  auto run = [&](const kir::Val<kir::u32>& tile, const kir::Val<kir::u32>& row) {
    const auto w_base = gemv_w_base(e, idx, keep, slot, row, N);
    emit_tile(e, x, nullptr, w, w_base, tile, row, wave_id, lane, N, K, wave,
              load_bytes);
  };

  if (persist) {
    for (auto row : e.range(M)) {
      for (auto tile :
           e.range(math::workgroup_id_x(), e.u32(ntiles), persist_wgs)) {
        run(tile, row);
      }
    }
    return;
  }
  if (grid) {
    // One workgroup owns one row of x and every column in its tile. Left in
    // global, that row is re-read by every wave inside the K loop, and those
    // loads hold the same outstanding-request capacity the weight stream needs
    // — measured 66.2 -> 41.3 us (134 -> 216 GB/s) at N=2176 K=1024. Only a
    // grid launch has a workgroup-constant row, so only it stages.
    kir::Tile<kir::f32> xs;
    const bool stage = e.lds_fits<kir::f32>(K);
    if (stage) xs = e.lds<kir::f32>(K);
    const auto tile = e.let(math::workgroup_id_x());
    const auto row = e.let(math::workgroup_id_y());
    if (auto in_grid = e.when(tile < ntiles && row < M)) {
      if (stage) {
        for (auto t : e.range(lid, e.u32(K), kBlock)) {
          xs[t] = x[row * K + t];
        }
        e.barrier();
      }
      const auto w_base = gemv_w_base(e, idx, keep, slot, row, N);
      emit_tile(e, x, stage ? &xs : nullptr, w, w_base, tile, row, wave_id,
                lane, N, K, wave, load_bytes);
    }
    return;
  }
  for (auto row : e.range(M)) {
    for (auto tile : e.range(ntiles)) {
      run(tile, row);
    }
  }
}

template <class W>
void emit_gemv(kir::KernelBody& k, const kir::Buffer<kir::f32>& x,
               const kir::Buffer<W>& w, const kir::Buffer<kir::f32>* idx,
               std::uint32_t keep, std::uint32_t slot, std::uint32_t N,
               std::uint32_t K, std::uint32_t M, std::uint32_t load_bytes,
               bool grid, std::uint32_t wave, bool persist,
               std::uint32_t persist_wgs) {
  env::Emit e{&k};
  const F32In xi{x, &k.types()};
  const WIn<W> wi{w, &k.types()};
  F32In ii;
  const F32In* idxp = nullptr;
  if (idx != nullptr) {
    ii = F32In{*idx, &k.types()};
    idxp = &ii;
  }
  emit_gemv<W>(e, xi, wi, idxp, keep, slot, N, K, M, load_bytes, grid, wave,
               persist, persist_wgs);
}

#define LSE_GEMV_INSTANTIATE(W_)                                               \
  template void emit_gemv<W_>(                                                 \
      env::Emit&, const F32In&, const WIn<W_>&, const F32In*, std::uint32_t,   \
      std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t,              \
      std::uint32_t, bool, std::uint32_t, bool, std::uint32_t);                \
  template void emit_gemv<W_>(                                                 \
      kir::KernelBody&, const kir::Buffer<kir::f32>&, const kir::Buffer<W_>&,  \
      const kir::Buffer<kir::f32>*, std::uint32_t, std::uint32_t,              \
      std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t, bool,        \
      std::uint32_t, bool, std::uint32_t);
LSE_GEMV_INSTANTIATE(kir::f32)
LSE_GEMV_INSTANTIATE(lse::bf16)
LSE_GEMV_INSTANTIATE(lse::f16)
#undef LSE_GEMV_INSTANTIATE

namespace {

// The output leaves through the emitter's store hook; the Out member only
// names the slot the binding contract requires. `W` is the weight's storage
// element, so one body serves every checkpoint format.
template <class E, class W = kir::f32>
struct LinearLdsArgs {
  env::In<kir::f32, E> x;
  env::In<W, E> w;
  env::Out<kir::f32, E> out;
};

struct LinearLdsKernel final : KernelPrimitive<LinearLdsKernel> {
  static constexpr std::string_view kName = "linear.lds";
  static constexpr std::string_view kEntry = "lse_linear_lds";
  static constexpr std::string_view kSource = {};

  std::size_t arity() const noexcept override { return 2; }
  bool owns_indexing() const noexcept override { return true; }

  std::string emit_kernel(const KernelShapes& s) const override {
    const LinearDims d = dims_of_linear(s);
    if (!lds_shape_ok(d) || !device_fits(s) || s.types.scalar == nullptr ||
        !s.store || s.input_dtypes.size() < 2) {
      return {};
    }
    return with_elem(s.input_dtypes[1], [&]<class W>() -> std::string {
      kir::KernelBody k(s.types, *s.intrinsics, workgroup_lds_bytes(s.device));
      k.set_store(s.store);
      LinearLdsArgs<env::Emit, W> a;
      if (!env::bind(k, a, s)) return {};
      env::Emit e{&k};
      emit_gemv<W>(e, a.x, a.w, nullptr, 0, 0, static_cast<std::uint32_t>(d.n),
                   static_cast<std::uint32_t>(d.k),
                   static_cast<std::uint32_t>(d.m), device_load_bytes(s.device),
                   true, wave_of(s.device));
      if (!k.lds().ok()) return {};
      return k.str();
    });
  }

  Result<Shape> infer_shape(std::span<const Shape> in) const override {
    if (in.size() != 2) return LSE_ERROR(kInvalidArgument, "linear takes 2 inputs");
    Shape out;
    for (std::size_t i = 0; i + 1 < in[0].rank(); ++i) out.push_back(in[0].dim(i));
    out.push_back(in[1].dim(0));
    return out;
  }
  DType infer_dtype(std::span<const DType> in) const override {
    return in.empty() ? DType::kF32 : in[0];
  }
  static ThreadPlan plan_impl(const KernelShapes& s) {
    return gemv_plan(dims_of_linear(s), wave_of(s.device));
  }
};

template <class E, class W = kir::f32>
struct LinearIndexedLdsArgs {
  env::In<kir::f32, E> x;
  env::In<W, E> w;
  env::In<kir::f32, E> idx;
  env::Out<kir::f32, E> out;
};

struct LinearIndexedLdsKernel final : KernelPrimitive<LinearIndexedLdsKernel> {
  static constexpr std::string_view kName = "linear_indexed.lds";
  static constexpr std::string_view kEntry = "lse_linear_indexed_lds";
  static constexpr std::string_view kSource = {};

  std::size_t arity() const noexcept override { return 3; }
  bool owns_indexing() const noexcept override { return true; }

  std::string emit_kernel(const KernelShapes& s) const override {
    const LinearDims d = dims_of_indexed(s);
    if (!lds_shape_ok(d) || !device_fits(s) || s.types.scalar == nullptr ||
        !s.store || s.inputs.size() != 3 || s.inputs[2].rank() == 0 ||
        s.input_dtypes.size() < 3) {
      return {};
    }
    const auto keep = static_cast<std::uint32_t>(
        s.inputs[2].dim(s.inputs[2].rank() - 1));
    const auto slot = static_cast<std::uint32_t>(s.iattrs[0]);
    if (keep == 0 || slot >= keep) return {};

    return with_elem(s.input_dtypes[1], [&]<class W>() -> std::string {
      kir::KernelBody k(s.types, *s.intrinsics, workgroup_lds_bytes(s.device));
      k.set_store(s.store);
      LinearIndexedLdsArgs<env::Emit, W> a;
      if (!env::bind(k, a, s)) return {};
      env::Emit e{&k};
      emit_gemv<W>(e, a.x, a.w, &a.idx, keep, slot,
                   static_cast<std::uint32_t>(d.n),
                   static_cast<std::uint32_t>(d.k),
                   static_cast<std::uint32_t>(d.m), device_load_bytes(s.device),
                   true, wave_of(s.device));
      if (!k.lds().ok()) return {};
      return k.str();
    });
  }

  Result<Shape> infer_shape(std::span<const Shape> in) const override {
    if (in.size() != 3 || in[1].rank() != 3) {
      return LSE_ERROR(kInvalidArgument, "linear_indexed takes x, W[E,N,K], idx");
    }
    Shape out;
    for (std::size_t i = 0; i + 1 < in[0].rank(); ++i) out.push_back(in[0].dim(i));
    out.push_back(in[1].dim(1));
    return out;
  }
  DType infer_dtype(std::span<const DType> in) const override {
    return in.empty() ? DType::kF32 : in[0];
  }
  static ThreadPlan plan_impl(const KernelShapes& s) {
    return gemv_plan(dims_of_indexed(s), wave_of(s.device));
  }
};

}  // namespace

const KernelPrimitiveBase* lds_linear_for(const KernelShapes& s) {
  static const LinearLdsKernel kKernel;
  if (!device_fits(s) || !lds_shape_ok(dims_of_linear(s))) return nullptr;
  return &kKernel;
}

const KernelPrimitiveBase* lds_linear_indexed_for(const KernelShapes& s) {
  static const LinearIndexedLdsKernel kKernel;
  if (!device_fits(s) || !lds_shape_ok(dims_of_indexed(s))) return nullptr;
  return &kKernel;
}

}  // namespace lse::backend::hrx_kernels
