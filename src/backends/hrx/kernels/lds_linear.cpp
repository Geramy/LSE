// `linear` / `linear_indexed` for the decode-shaped case: few rows, wide N.
//
// W is [N, K] (or [E, N, K]). One thread per column walks a whole row, so
// neighbouring threads are K floats apart. One wave per column, lanes on
// consecutive K, is a coalesced row read; the wave xor-reduces in registers.
#include "lse/backends/hrx/kernels/lds_linear.hpp"

#include <string>

#include "lse/backends/hrx/kernels/vec_mem.hpp"
#include "lse/backends/hrx/device_info.hpp"
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
  const AmdDeviceInfo* amd = device_extension<AmdDeviceInfo>(*device);
  if (amd != nullptr && (amd->wavefront_size == 32 || amd->wavefront_size == 64)) {
    return amd->wavefront_size;
  }
  return 32;
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

void emit_tile(kir::KernelBody& k, const kir::Buffer<kir::f32>& x,
               const kir::Buffer<kir::f32>& w, const kir::Val<kir::u32>& w_base,
               const kir::Val<kir::u32>& tile, const kir::Val<kir::u32>& row,
               const kir::Val<kir::u32>& wave_id, const kir::Val<kir::u32>& lane,
               std::uint32_t N, std::uint32_t K, std::uint32_t wave,
               std::uint32_t load_bytes) {
  const auto waves = kBlock / wave;
  const auto col = k.let<kir::u32>("col", tile * waves + wave_id);
  const auto step = kir::pack_n(load_bytes, kir::pack_elem_bytes<kir::f32>());
  const auto span = wave * step;
  const auto aligned = (K / span) * span;

  auto acc = k.var<kir::f32>("acc", k.lit(0.0f));
  k.when(col < N, [&] {
    k.loop("k0", k.constant<kir::u32>(0), k.constant<kir::u32>(aligned), span,
           [&](kir::Val<kir::u32> k0) {
             const auto kk = k.let<kir::u32>("kk", k0 + lane * step);
             const auto xv = math::load(x, row * K + kk, load_bytes);
             const auto wv = math::load(w, (w_base + col) * K + kk, load_bytes);
             k.unroll("e", step, [&](kir::Val<kir::u32> e) {
               acc = math::fma(xv[e], wv[e], acc.read());
             });
           });
    if (aligned < K) {
      k.loop("kt", k.constant<kir::u32>(aligned) + lane,
             k.constant<kir::u32>(K), wave, [&](kir::Val<kir::u32> kt) {
               acc = math::fma(x[row * K + kt].read(),
                               w[(w_base + col) * K + kt].read(), acc.read());
             });
    }
  });

  // Wave xor-reduce: columns stay independent, no LDS, no WG barrier.
  for (std::uint32_t m = 1; m < wave; m <<= 1) {
    acc = acc.read() + math::shfl_xor(acc.read(), k.constant<kir::u32>(m));
  }
  k.when(lane == 0 && col < N, [&] { k.store(row * N + col, acc.read()); });
}

kir::Val<kir::u32> gemv_w_base(kir::KernelBody& k,
                               const kir::Buffer<kir::f32>* idx,
                               std::uint32_t keep, std::uint32_t slot,
                               const kir::Val<kir::u32>& row, std::uint32_t N) {
  if (idx == nullptr) return k.constant<kir::u32>(0);
  const auto e = k.let<kir::u32>(
      "e", kir::cast<kir::u32>(
               (*idx)[row * keep + k.constant<kir::u32>(slot)].read()));
  return k.let<kir::u32>("wb", e * N);
}

}  // namespace

void emit_gemv(kir::KernelBody& k, const kir::Buffer<kir::f32>& x,
               const kir::Buffer<kir::f32>& w, const kir::Buffer<kir::f32>* idx,
               std::uint32_t keep, std::uint32_t slot, std::uint32_t N,
               std::uint32_t K, std::uint32_t M, std::uint32_t load_bytes,
               bool grid, std::uint32_t wave, bool persist,
               std::uint32_t persist_wgs) {
  if (wave != 32 && wave != 64) wave = 32;
  if (persist_wgs == 0) persist_wgs = 1;
  const auto lid = k.let<kir::u32>("lid", math::local_id());
  const auto wave_id = k.let<kir::u32>("wave", lid / wave);
  const auto lane = k.let<kir::u32>("lane", lid % wave);
  const auto ntiles = (N + (kBlock / wave) - 1) / (kBlock / wave);

  auto run = [&](const kir::Val<kir::u32>& tile, const kir::Val<kir::u32>& row) {
    const auto w_base = gemv_w_base(k, idx, keep, slot, row, N);
    emit_tile(k, x, w, w_base, tile, row, wave_id, lane, N, K, wave,
              load_bytes);
  };

  if (persist) {
    k.loop("row", k.constant<kir::u32>(0), k.constant<kir::u32>(M), 1,
           [&](kir::Val<kir::u32> row) {
             k.loop("tile", math::workgroup_id_x(),
                    k.constant<kir::u32>(ntiles),
                    k.constant<kir::u32>(persist_wgs),
                    [&](kir::Val<kir::u32> tile) { run(tile, row); });
           });
    return;
  }
  if (grid) {
    const auto tile = k.let<kir::u32>("tile", math::workgroup_id_x());
    const auto row = k.let<kir::u32>("row", math::workgroup_id_y());
    k.when(tile < ntiles && row < M, [&] { run(tile, row); });
    return;
  }
  k.loop("row", k.constant<kir::u32>(0), k.constant<kir::u32>(M), 1,
         [&](kir::Val<kir::u32> row) {
           k.loop("tile", k.constant<kir::u32>(0), k.constant<kir::u32>(ntiles),
                  1, [&](kir::Val<kir::u32> tile) { run(tile, row); });
         });
}

namespace {

struct LinearLdsKernel final : KernelPrimitive<LinearLdsKernel> {
  static constexpr std::string_view kName = "linear.lds";
  static constexpr std::string_view kEntry = "lse_linear_lds";
  static constexpr std::string_view kSource = {};

  std::size_t arity() const noexcept override { return 2; }
  bool owns_indexing() const noexcept override { return true; }

  std::string emit_kernel(const KernelShapes& s) const override {
    const LinearDims d = dims_of_linear(s);
    if (!lds_shape_ok(d) || !device_fits(s) || s.types.scalar == nullptr ||
        !s.store) {
      return {};
    }
    kir::KernelBody k(s.types, *s.intrinsics, workgroup_lds_bytes(s.device));
    k.set_store(s.store);
    const auto x = k.input<kir::f32>(0);
    const auto w = k.input<kir::f32>(1);
    emit_gemv(k, x, w, nullptr, 0, 0, static_cast<std::uint32_t>(d.n),
              static_cast<std::uint32_t>(d.k),
              static_cast<std::uint32_t>(d.m), device_load_bytes(s.device),
              true, wave_of(s.device));
    if (!k.lds().ok()) return {};
    return k.str();
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

struct LinearIndexedLdsKernel final : KernelPrimitive<LinearIndexedLdsKernel> {
  static constexpr std::string_view kName = "linear_indexed.lds";
  static constexpr std::string_view kEntry = "lse_linear_indexed_lds";
  static constexpr std::string_view kSource = {};

  std::size_t arity() const noexcept override { return 3; }
  bool owns_indexing() const noexcept override { return true; }

  std::string emit_kernel(const KernelShapes& s) const override {
    const LinearDims d = dims_of_indexed(s);
    if (!lds_shape_ok(d) || !device_fits(s) || s.types.scalar == nullptr ||
        !s.store || s.inputs.size() != 3 || s.inputs[2].rank() == 0) {
      return {};
    }
    const auto keep = static_cast<std::uint32_t>(
        s.inputs[2].dim(s.inputs[2].rank() - 1));
    const auto slot = static_cast<std::uint32_t>(s.iattrs[0]);
    if (keep == 0 || slot >= keep) return {};

    kir::KernelBody k(s.types, *s.intrinsics, workgroup_lds_bytes(s.device));
    k.set_store(s.store);
    const auto x = k.input<kir::f32>(0);
    const auto w = k.input<kir::f32>(1);
    const auto idx = k.input<kir::f32>(2);
    emit_gemv(k, x, w, &idx, keep, slot, static_cast<std::uint32_t>(d.n),
              static_cast<std::uint32_t>(d.k),
              static_cast<std::uint32_t>(d.m), device_load_bytes(s.device),
              true, wave_of(s.device));
    if (!k.lds().ok()) return {};
    return k.str();
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
