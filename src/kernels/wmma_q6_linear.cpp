#include "lse/graph/graph.hpp"
// Measured floating-operand matrix contraction over unchanged MLX affine Q6.
// Weight storage remains packed; staged operands feed the measured matrix core.
#include "lse/graph/kernel_args.hpp"
#include "lse/kernels/quant_operand_policy.hpp"
#include "lse/kernels/wmma.hpp"
#include "lse/math/fp8.hpp"
#include "lse/quant/group_affine_codec.hpp"
#include "lse/quant/centered_affine.hpp"
#include <cstdlib>
#include <cstring>
#include <limits>
#include <limits>
namespace lse::kernels {
namespace {
namespace env = graph::env;
namespace kir = graph::kir;
using namespace graph;
namespace math = lse::math;
struct Dims {
  std::uint32_t m = 0, n = 0, k = 0, words = 0, groups = 0, group = 0;
  bool valid = false;
};
Dims dims_of(const KernelShapes &s) {
  Dims d;
  if (s.inputs.size() != 4 || s.input_dtypes.size() != 4 ||
      s.iattrs.size() < 2 || s.iattrs[0] != 6 || s.iattrs[1] <= 0 ||
      s.iattrs[1] % 32 || s.inputs[1].rank() != 2 || s.inputs[0].rank() == 0)
    return d;
  const auto n = s.inputs[1].dim(0), words = s.inputs[1].dim(1);
  const std::int64_t group = s.iattrs[1];
  if (n <= 0 || words <= 0 || words % 3 || n > UINT32_MAX ||
      words > UINT32_MAX / 32)
    return d;
  const auto k = words * 32 / 6;
  if (k % group || s.inputs[0].dim(s.inputs[0].rank() - 1) != k ||
      s.output.elem_count() % n)
    return d;
  const auto m = s.output.elem_count() / n;
  if (m < 16 || m > UINT32_MAX || m * k > UINT32_MAX ||
      n * words > UINT32_MAX || m * n > UINT32_MAX)
    return d;
  if (s.inputs[2].elem_count() != n * (k / group) ||
      s.inputs[3].elem_count() != n * (k / group) ||
      s.input_dtypes[0] != DType::kF32 || s.input_dtypes[1] != DType::kU32 ||
      s.input_dtypes[2] != DType::kBF16 || s.input_dtypes[3] != DType::kBF16)
    return d;
  d = {static_cast<std::uint32_t>(m),
       static_cast<std::uint32_t>(n),
       static_cast<std::uint32_t>(k),
       static_cast<std::uint32_t>(words),
       static_cast<std::uint32_t>(k / group),
       static_cast<std::uint32_t>(group),
       true};
  return d;
}
template <math::MatrixElem T>
using Base = MatrixTile<struct UnusedQ6Tile, math::MatrixTarget::kRdna4,
                        math::MatrixElem::kF32, T, 16, 16, 16>;
struct Args {
  env::In<kir::f32, env::Emit> x;
  env::In<std::uint32_t, env::Emit> packed;
  env::In<lse::bf16, env::Emit> scales, biases;
  env::Out<kir::f32, env::Emit> out;
};
// A workgroup covers64x64 output values. Four waves each retain a32x32
// quadrant, reusing two A and two B fragments across four accumulators.
// Packed model storage is unchanged; only the staged operands are narrowed.
std::string emit_staged_bf16(const KernelShapes &s, const Dims &d) {
  using Tile = Base<math::MatrixElem::kBF16>;
  using F = typename Tile::AFrag;
  using Op = typename Tile::Op;
  constexpr auto geo = geometry_of(Tile::kRow);
  static_assert(geo.wave == 32 && geo.split_k && geo.lane_k == 8);
  kir::KernelBody body(s.types, *s.intrinsics, workgroup_lds_bytes(s.device));
  body.set_store(s.store);
  Args a;
  if (!env::bind(body, a, s))
    return {};
  env::Emit e{&body};
  const auto xs = e.lds<F>(64u * 64u), ws = e.lds<F>(64u * 64u);
  const auto lid = e.let(math::local_id()), lane = e.let(lid % 32u);
  const auto wave = e.let(lid / 32u), lo = e.let(lane % 16u),
             hi = e.let(lane / 16u);
  const auto wg = e.let(math::workgroup_id_x());
  const auto nblocks = (d.n + 63) / 64;
  const auto mbase = e.let((wg / nblocks) * 64u),
             nbase = e.let((wg % nblocks) * 64u);
  const auto am = e.let((wave % 2u) * 32u), bn = e.let((wave / 2u) * 32u);
  // Rotate eight-half spans by row. Every span remains contiguous, while
  // neighboring lanes read different LDS bank groups rather than stride64.
  const auto address = [&](const kir::Val<kir::u32> &r,
                           const kir::Val<kir::u32> &k) {
    return r * 64u + ((k / 8u + r % 8u) % 8u) * 8u + k % 8u;
  };
  std::vector<kir::Local<kir::f32, 8>> acc;
  for (unsigned i = 0; i < 4; ++i) {
    acc.push_back(e.local<kir::f32, 8>());
    for (auto j : e.unroll(8u))
      acc.back()[j] = e.f32(0);
  }
  for (auto kb : e.range(0u, d.k, 64u)) {
    // Each lane stages two16-value row fragments. All128 lanes execute
    // both barriers, including lanes covering padded M/N output edges.
    for (auto block : e.unroll(2u)) {
      const auto r = e.let(lid / 4u + block * 32u),
                 kc = e.let((lid % 4u) * 16u);
      const auto ar = e.let(mbase + r), bc = e.let(nbase + r);
      const auto av = e.local<F, 16>(), bv = e.local<F, 16>();
      for (auto j : e.unroll(16u)) {
        av[j] = math::narrow<F>(e.f32(0));
        bv[j] = math::narrow<F>(e.f32(0));
      }
      if (auto active = e.when(ar < d.m)) {
        for (unsigned v = 0; v < 4; ++v) {
          const auto loaded = e.load(a.x, ar * d.k + kb + kc + v * 4u, 16u);
          for (unsigned j = 0; j < 4; ++j)
            av[v * 4 + j] = math::narrow<F>(loaded[j]);
        }
      }
      if (auto active = e.when(bc < d.n)) {
        const auto gi = e.let(bc * d.groups + (kb + kc) / d.group);
        const auto scale = e.let(math::widen(a.scales[gi])),
                   bias = e.let(math::widen(a.biases[gi]));
        const auto wb = e.let(bc * d.words + ((kb + kc) / 16u) * 3u);
        const std::array<kir::Val<kir::u32>, 3> packed{
            e.let(a.packed[wb]), e.let(a.packed[wb + 1u]),
            e.let(a.packed[wb + 2u])};
        for (unsigned j = 0; j < 16; ++j) {
          const unsigned off = (j * 6) % 32, wi = j * 6 / 32;
          auto code = packed[wi] / (1u << off);
          if (off > 26)
            code = code +
                   (packed[wi + 1] % (1u << (off - 26))) * (1u << (32 - off));
          else
            code = code % 64u;
          bv[j] = math::narrow<F>(
              math::fma(math::cast<kir::f32>(e.let(code)), scale, bias));
        }
      }
      for (auto j : e.unroll(16u)) {
        const auto index = e.let(address(r, kc + j));
        xs[index] = av[j].read();
        ws[index] = bv[j].read();
      }
    }
    e.barrier();
    for (auto slice : e.unroll(4u)) {
      std::vector<kir::Local<F, 8>> af, bf;
      for (unsigned i = 0; i < 2; ++i) {
        af.push_back(e.local<F, 8>());
        bf.push_back(e.local<F, 8>());
        const auto ar = e.let(am + i * 16u + lo), bc = e.let(bn + i * 16u + lo);
        for (auto j : e.unroll(8u)) {
          const auto k = e.let(slice * 16u + hi * 8u + j);
          af.back()[j] = xs[e.let(address(ar, k))];
          bf.back()[j] = ws[e.let(address(bc, k))];
        }
      }
      for (unsigned m = 0; m < 2; ++m)
        for (unsigned n = 0; n < 2; ++n)
          acc[m * 2 + n] = math::mma<Op>(af[m].value(), bf[n].value(),
                                         acc[m * 2 + n].value());
    }
    e.barrier();
  }
  for (unsigned m = 0; m < 2; ++m)
    for (unsigned n = 0; n < 2; ++n) {
      const auto col = e.let(nbase + bn + n * 16u + lo);
      for (auto j : e.unroll(8u)) {
        const auto rr = e.let(mbase + am + m * 16u + j * geo.slot_step +
                              hi * geo.half_rows);
        if (auto active = e.when(rr < d.m && col < d.n))
          e.store(rr * d.n + col, acc[m * 2 + n][j].read());
      }
    }
  if (!body.lds().ok())
    return {};
  return body.str();
}
std::string emit_staged_bf16_centered(const KernelShapes &s, const Dims &d,
                                      bool deferred_repair = false) {
  using Tile = Base<math::MatrixElem::kBF16>;
  using F = typename Tile::AFrag;
  using Op = typename Tile::Op;
  constexpr auto geo = geometry_of(Tile::kRow);
  static_assert(geo.wave == 32 && geo.split_k && geo.lane_k == 8);
  kir::KernelBody body(s.types, *s.intrinsics, workgroup_lds_bytes(s.device));
  body.set_store(s.store);
  Args a;
  if (!env::bind(body, a, s))
    return {};
  env::Emit e{&body};
  const auto xs = e.lds<F>(64u * 64u), ws = e.lds<F>(64u * 64u);
  const auto xl = e.lds<F>(64u * 64u);
  const auto row_partials = e.lds<kir::f32>(64u * 4u);
  const auto group_scales = e.lds<kir::f32>(64u);
  const auto group_biases = e.lds<kir::f32>(64u);
  const auto flags = e.lds<kir::f32>(12u);
  constexpr float product_headroom = quant::kCenteredProductHeadroom;
  constexpr float min_normal = std::numeric_limits<float>::min();
  constexpr float max_bf16 = 3.3895313892515355e38f;
  const auto lid = e.let(math::local_id()), lane = e.let(lid % 32u);
  const auto wave = e.let(lid / 32u), lo = e.let(lane % 16u),
             hi = e.let(lane / 16u);
  const auto wg = e.let(math::workgroup_id_x());
  const auto nblocks = (d.n + 63) / 64;
  const auto mbase = e.let((wg / nblocks) * 64u),
             nbase = e.let((wg % nblocks) * 64u);
  const auto am = e.let((wave % 2u) * 32u), bn = e.let((wave / 2u) * 32u);
  // Rotate eight-half spans by row. Every span remains contiguous, while
  // neighboring lanes read different LDS bank groups rather than stride64.
  const auto address = [&](const kir::Val<kir::u32> &r,
                           const kir::Val<kir::u32> &k) {
    return r * 64u + ((k / 8u + r % 8u) % 8u) * 8u + k % 8u;
  };
  // Compact original-FP32 fallback: consume the same little-endian 96 bits
  // in increasing weight order without emitting sixteen copies of the body.
  // The three DWORD loads are exact; no row-end overread or extra alignment.
  const auto scalar_chunk = [&](const auto& word_base, const auto& scale,
                                const auto& bias, auto&& sink) {
    auto low=e.var(a.packed[word_base]);
    auto middle=e.var(a.packed[word_base+1u]);
    auto high=e.var(a.packed[word_base+2u]);
    for(auto q:e.range(0u,16u,1u)) {
      const auto code=e.let(low.read()%64u);
      sink(q,math::fma(math::cast<kir::f32>(code),scale,bias));
      low=low.read()/64u+(middle.read()%64u)*67108864u;
      middle=middle.read()/64u+(high.read()%64u)*67108864u;
      high=high.read()/64u;
    }
  };
  auto largest_group = e.var(0.0f);
  std::vector<kir::Local<kir::f32, 8>> acc;
  for (unsigned i = 0; i < 4; ++i) {
    acc.push_back(e.local<kir::f32, 8>());
    for (auto j : e.unroll(8u))
      acc.back()[j] = e.f32(0);
  }
  for (auto kb : e.range(0u, d.k, 64u)) {
    auto bad = e.var(0.0f);
    for(unsigned i=0;i<4;++i)for(auto j:e.unroll(8u))
      if(auto large=e.when(math::abs(acc[i][j].read()) > quant::kCenteredAccumulatorHeadroom))
        bad=e.f32(1.0f);
    auto x_max = e.var(0.0f), coefficient_max = e.var(0.0f);
    const auto mark_unsafe = [&](const kir::Val<kir::f32>& value) {
      auto invalid=e.var(1.0f);
      if(auto safe=e.when(math::abs(value)<=max_bf16 &&
          (value==0.0f || math::abs(value)>=min_normal)))invalid=e.f32(0);
      bad=math::max(bad.read(),invalid.read());
    };
    // Each lane stages two16-value row fragments. All128 lanes execute
    // both barriers, including lanes covering padded M/N output edges.
    for (auto block : e.unroll(2u)) {
      const auto r = e.let(lid / 4u + block * 32u),
                 kc = e.let((lid % 4u) * 16u);
      const auto ar = e.let(mbase + r), bc = e.let(nbase + r);
      const auto av = e.local<F, 16>(), al = e.local<F, 16>(), bv = e.local<F, 16>();
      auto row_sum = e.var(0.0f);
      for (auto j : e.unroll(16u)) {
        av[j] = math::narrow<F>(e.f32(0));
        al[j] = math::narrow<F>(e.f32(0));
        bv[j] = math::narrow<F>(e.f32(0));
      }
      if (auto active = e.when(ar < d.m)) {
        for (unsigned v = 0; v < 4; ++v) {
          const auto loaded = e.load(a.x, ar * d.k + kb + kc + v * 4u, 16u);
          for (unsigned j = 0; j < 4; ++j) {
            const auto original=e.let(loaded[j]);
            mark_unsafe(original);
            x_max = math::max(x_max.read(), math::abs(original));
            av[v * 4 + j] = math::narrow<F>(original);
            const auto residual=e.let(original-math::widen(av[v * 4 + j].read()));
            mark_unsafe(residual);
            al[v * 4 + j] = math::narrow<F>(residual);
            // Retain the original FP32 activation in the inexpensive bias sum.
            // The matrix dot receives its rounded high and residual separately.
            row_sum = row_sum.read() + original;
          }
        }
      }
      if (auto owner = e.when(lid % 4u == 0u)) {
        group_scales[r] = e.f32(0.0f);
        group_biases[r] = e.f32(0.0f);
      }
      if (auto active = e.when(bc < d.n)) {
        const auto gi = e.let(bc * d.groups + (kb + kc) / d.group);
        const auto scale = e.let(math::widen(a.scales[gi])),
                   bias = e.let(math::widen(a.biases[gi]));
        mark_unsafe(scale); mark_unsafe(bias);
        const auto coefficients = quant::centered_affine(e, scale, bias);
        const auto center = coefficients.center;
        const auto residual_bias = coefficients.residual_bias;
        mark_unsafe(residual_bias);
        // This bounds cancellation in the separated affine terms by three.
        // Zero scale is supported here only when every weight is zero.
        if (auto ill_conditioned = e.when(
                math::abs(residual_bias) > math::abs(scale) * 0.5f))
          bad = e.f32(1.0f);
        coefficient_max = math::max(coefficient_max.read(),
            math::max(math::abs(scale), math::abs(residual_bias)));
        if (auto owner = e.when(lid % 4u == 0u)) {
          group_scales[r] = scale;
          group_biases[r] = residual_bias;
        }
        const auto wb = e.let(bc * d.words + ((kb + kc) / 16u) * 3u);
        const std::array<kir::Val<kir::u32>, 3> packed{
            e.let(a.packed[wb]), e.let(a.packed[wb + 1u]),
            e.let(a.packed[wb + 2u])};
        for (unsigned j = 0; j < 16; ++j) {
          const unsigned off = (j * 6) % 32, wi = j * 6 / 32;
          auto code = packed[wi] / (1u << off);
          if (off > 26)
            code = code +
                   (packed[wi + 1] % (1u << (off - 26))) * (1u << (32 - off));
          else
            code = code % 64u;
          const auto original=e.let(math::fma(math::cast<kir::f32>(e.let(code)), scale, bias));
          mark_unsafe(original);
          // Integer centered codes are exactly representable in BF16.
          bv[j] = math::narrow<F>(math::cast<kir::f32>(e.let(code)) - center);
        }
      }
      for (auto j : e.unroll(16u)) {
        const auto index = e.let(address(r, kc + j));
        xs[index] = av[j].read();
        xl[index] = al[j].read();
        ws[index] = bv[j].read();
      }
      row_partials[r * 4u + lid % 4u] = row_sum.read();
    }
    for(unsigned bit:{1u,2u,4u,8u,16u}) {
      bad=math::max(bad.read(),math::shfl_xor(bad.read(),e.u32(bit)));
      x_max=math::max(x_max.read(),math::shfl_xor(x_max.read(),e.u32(bit)));
      coefficient_max=math::max(coefficient_max.read(),
          math::shfl_xor(coefficient_max.read(),e.u32(bit)));
    }
    if(auto leader=e.when(lane==0u)) {
      flags[wave]=bad.read();
      flags[4u+wave]=x_max.read();
      flags[8u+wave]=coefficient_max.read();
    }
    e.barrier();
    auto block_bad=e.var(flags[0u].read()+flags[1u].read()+flags[2u].read()+flags[3u].read());
    const auto block_x = e.let(math::max(math::max(flags[4u].read(), flags[5u].read()),
                                       math::max(flags[6u].read(), flags[7u].read())));
    const auto block_coefficient = e.let(math::max(math::max(flags[8u].read(), flags[9u].read()),
                                                 math::max(flags[10u].read(), flags[11u].read())));
    // Guard the unscaled group dot and its separate affine products, even
    // when cancellation or a tiny scale makes the reference result finite.
    if (auto overflow = e.when(block_x > product_headroom)) block_bad=e.f32(1.0f);
    if (auto nonzero = e.when(block_coefficient > 0.0f))
      if (auto overflow = e.when(block_x > e.f32(product_headroom) / block_coefficient))
        block_bad=e.f32(1.0f);
    if(auto regular=e.when(block_bad.read()==0.0f)) {
    // A small group total can hide cancellation inside the matrix dot itself.
    // Keep a scale for that case even when every group contribution is zero.
    // The existing overflow guard makes this product finite. This conservative
    // trigger requests original ordered FP32; it is not an error certificate.
    largest_group=math::max(largest_group.read(),
        (block_x*block_coefficient)*64.0f);
    std::vector<kir::Local<kir::f32, 8>> group_acc;
    for (unsigned i=0;i<4;++i) {
      group_acc.push_back(e.local<kir::f32,8>());
      for(auto j:e.unroll(8u)) group_acc.back()[j]=e.f32(0.0f);
    }
    for (auto slice : e.unroll(4u)) {
      std::vector<kir::Local<F, 8>> af, alf, bf;
      for (unsigned i = 0; i < 2; ++i) {
        af.push_back(e.local<F, 8>());
        alf.push_back(e.local<F, 8>());
        bf.push_back(e.local<F, 8>());
        const auto ar = e.let(am + i * 16u + lo), bc = e.let(bn + i * 16u + lo);
        // K-major fragments occupy eight contiguous BF16 elements. The
        // bank rotation preserves the eight-element alignment at each start.
        const auto k = e.let(slice * 16u + hi * 8u);
        const auto ap = xs.load(e.let(address(ar, k)), 16u);
        const auto alp = xl.load(e.let(address(ar, k)), 16u);
        const auto bp = ws.load(e.let(address(bc, k)), 16u);
        for (auto j : e.unroll(8u)) {
          af.back()[j] = ap[j];
          alf.back()[j] = alp[j];
          bf.back()[j] = bp[j];
        }
      }
      for (unsigned m = 0; m < 2; ++m)
        for (unsigned n = 0; n < 2; ++n) {
          group_acc[m * 2 + n] = math::mma<Op>(af[m].value(), bf[n].value(),
                                               group_acc[m * 2 + n].value());
          group_acc[m * 2 + n] = math::mma<Op>(alf[m].value(), bf[n].value(),
                                               group_acc[m * 2 + n].value());
        }
    }
    for(unsigned m=0;m<2;++m)for(unsigned n=0;n<2;++n) {
      const auto col=e.let(bn+n*16u+lo);
      const auto scale=e.let(group_scales[col].read());
      const auto bias=e.let(group_biases[col].read());
      for(auto j:e.unroll(8u)) {
        const auto row=e.let(am+m*16u+j*geo.slot_step+hi*geo.half_rows);
        const auto base=e.let(row*4u);
        const auto sum=e.let(((row_partials[base].read()+row_partials[base+1u].read())+
                              row_partials[base+2u].read())+row_partials[base+3u].read());
        const auto contribution=e.let(math::fma(scale,group_acc[m*2+n][j].read(),bias*sum));
        largest_group=math::max(largest_group.read(),math::abs(contribution));
        acc[m*2+n][j]=acc[m*2+n][j].read()+contribution;
      }
    }
    }
    if(auto exceptional=e.when(block_bad.read()!=0.0f)) {
      for(unsigned m=0;m<2;++m)for(unsigned n=0;n<2;++n){
        const auto col=e.let(nbase+bn+n*16u+lo);
        for(auto j:e.unroll(8u)){
          const auto rr=e.let(mbase+am+m*16u+j*geo.slot_step+hi*geo.half_rows);
          if(auto active=e.when(rr<d.m && col<d.n)){
            for(auto chunk:e.range(kb/16u,kb/16u+4u,1u)){
              const auto gi=e.let(col*d.groups+(chunk*16u)/d.group);
              const auto scale=e.let(math::widen(a.scales[gi])),bias=e.let(math::widen(a.biases[gi]));
              scalar_chunk(e.let(col*d.words+chunk*3u),scale,bias,
                [&](const auto& q,const kir::Val<kir::f32>& weight){
                  const auto x=a.x[rr*d.k+chunk*16u+q];
                  acc[m*2+n][j]=math::fma(x,weight,acc[m*2+n][j].read());
                });
            }
          }
        }
      }
    }
    e.barrier();
  }
  for (unsigned m = 0; m < 2; ++m)
    for (unsigned n = 0; n < 2; ++n) {
      const auto col = e.let(nbase + bn + n * 16u + lo);
      for (auto j : e.unroll(8u)) {
        const auto rr = e.let(mbase + am + m * 16u + j * geo.slot_step +
                              hi * geo.half_rows);
        if (auto active = e.when(rr < d.m && col < d.n)) {
          auto value=e.var(acc[m * 2 + n][j].read());
          // Near-cancelling group totals can hide the original ordered-FMA
          // residual. Recompute only these outputs, after all LDS barriers.
          const auto needs_repair=e.let(largest_group.read()>0.0f &&
              math::abs(value.read()) <= quant::kCenteredCancellationRatio*largest_group.read());
          if (deferred_repair) {
            e.store(d.m*d.n+rr*d.n+col,
                    math::select(needs_repair,e.f32(1.0f),e.f32(0.0f)));
          } else if (auto cancellation=e.when(needs_repair)) {
            value=e.f32(0.0f);
            for(auto chunk:e.range(0u,d.k/16u,1u)) {
              const auto gi=e.let(col*d.groups+(chunk*16u)/d.group);
              const auto scale=e.let(math::widen(a.scales[gi]));
              const auto bias=e.let(math::widen(a.biases[gi]));
              scalar_chunk(e.let(col*d.words+chunk*3u),scale,bias,
                [&](const auto& q,const kir::Val<kir::f32>& weight) {
                  value=math::fma(a.x[rr*d.k+chunk*16u+q],
                                  weight,value.read());
                });
            }
          }
          e.store(rr*d.n+col,value.read());
        }
      }
    }
  if (!body.lds().ok())
    return {};
  return body.str();
}
#include "q6_fp8_residual.inc"
struct StagedBF16Kernel final : KernelPrimitive<StagedBF16Kernel> {
  static constexpr std::string_view kName = "quant_linear.q6_wmma_bf16_reuse";
  static constexpr std::string_view kEntry = "lse_q6_wmma_bf16_reuse";
  static constexpr std::string_view kSource = {};
  std::size_t arity() const noexcept override { return 4; }
  bool owns_indexing() const noexcept override { return true; }
  std::string emit_kernel(const KernelShapes &s) const override {
    const auto d = dims_of(s);
    if (!d.valid || !s.store || !s.types.scalar || !s.intrinsics)
      return {};
    return emit_staged_bf16(s, d);
  }
  Result<Shape> infer_shape(std::span<const Shape> in) const override {
    if (in.size() != 4 || in[1].rank() != 2)
      return LSE_ERROR(kInvalidArgument, "Q6 matrix requires four operands");
    Shape o;
    for (std::size_t i = 0; i + 1 < in[0].rank(); ++i)
      o.push_back(in[0].dim(i));
    o.push_back(in[1].dim(0));
    return o;
  }
  DType infer_dtype(std::span<const DType>) const override {
    return DType::kF32;
  }
  static ThreadPlan plan_impl(const KernelShapes &s) {
    ThreadPlan p;
    auto d = dims_of(s);
    if (!d.valid)
      return p;
    p.workgroup_size[0] = 128;
    p.workgroup_count[0] = ((d.m + 63) / 64) * ((d.n + 63) / 64);
    p.lds_bytes = 16384;
    p.workgroup_count[1] = p.workgroup_count[2] = 1;
    return p;
  }
};
const KernelPrimitiveBase *select_staged_bf16(const KernelShapes &s) {
  constexpr auto r = Base<math::MatrixElem::kBF16>::kRow;
  if (!r.emittable() || !math::has_cap(device_matrix_caps(*s.device), r.cap) ||
      s.intrinsics->find(r.key).empty())
    return nullptr;
  static const StagedBF16Kernel kernel;
  return &kernel;
}

struct CenteredAffineKernel final : KernelPrimitive<CenteredAffineKernel> {
  static constexpr std::string_view kName = "quant_linear.q6_wmma_bf16_centered_activation_residual2_v3";
  static constexpr std::string_view kEntry = "lse_q6_wmma_bf16_centered_activation_residual2_v3";
  static constexpr std::string_view kSource = {};
  std::size_t arity() const noexcept override { return 4; }
  bool owns_indexing() const noexcept override { return true; }
  std::string emit_kernel(const KernelShapes &s) const override {
    const auto d = dims_of(s);
    if (!d.valid || !s.store || !s.types.scalar || !s.intrinsics)
      return {};
    return emit_staged_bf16_centered(s, d);
  }
  Result<Shape> infer_shape(std::span<const Shape> in) const override {
    if (in.size() != 4 || in[1].rank() != 2)
      return LSE_ERROR(kInvalidArgument, "Q6 matrix requires four operands");
    Shape o;
    for (std::size_t i = 0; i + 1 < in[0].rank(); ++i)
      o.push_back(in[0].dim(i));
    o.push_back(in[1].dim(0));
    return o;
  }
  DType infer_dtype(std::span<const DType>) const override {
    return DType::kF32;
  }
  static ThreadPlan plan_impl(const KernelShapes &s) {
    ThreadPlan p;
    auto d = dims_of(s);
    if (!d.valid)
      return p;
    p.workgroup_size[0] = 128;
    p.workgroup_count[0] = ((d.m + 63) / 64) * ((d.n + 63) / 64);
    p.lds_bytes = 26160;
    p.workgroup_count[1] = p.workgroup_count[2] = 1;
    return p;
  }
};
const KernelPrimitiveBase *select_centered_affine(const KernelShapes &s) {
  if (s.intrinsics->find("rint").empty() || s.intrinsics->find("min").empty()) return nullptr;
  constexpr auto r = Base<math::MatrixElem::kBF16>::kRow;
  if (!r.emittable() || !math::has_cap(device_matrix_caps(*s.device), r.cap) ||
      s.intrinsics->find(r.key).empty())
    return nullptr;
  static const CenteredAffineKernel kernel;
  return &kernel;
}

QuantOperandRequest residual_request(const KernelShapes &s, const Dims &d) {
  return {s.device->arch,
          s.device->wavefront_size,
          6,
          d.group,
          d.m,
          d.n,
          d.k,
          workgroup_lds_bytes(s.device),
          true,
          true,
          false,
          !s.staged.name.empty() || !s.staged_quant.codes.empty()};
}
template <math::MatrixElem T>
struct ResidualKernel final : KernelPrimitive<ResidualKernel<T>> {
  static constexpr std::string_view kName =
      T == math::MatrixElem::kFp8 ? "quant_linear.q6_fp8_residual3_v2"
                                  : "quant_linear.q6_bf8_residual3_v2";
  static constexpr std::string_view kEntry = T == math::MatrixElem::kFp8
                                                 ? "lse_q6_fp8_residual3_v2"
                                                 : "lse_q6_bf8_residual3_v2";
  static constexpr std::string_view kSource = {};
  std::size_t arity() const noexcept override { return 4; }
  bool owns_indexing() const noexcept override { return true; }
  std::string emit_kernel(const KernelShapes &s) const override {
    const auto d = dims_of(s);
    if (!d.valid || !s.store || !s.types.scalar || !s.intrinsics)
      return {};
    const auto request = residual_request(s, d);
    QuantOperandDecision decision;
    decision.operand =
        T == math::MatrixElem::kFp8 ? QuantOperand::kE4M3 : QuantOperand::kE5M2;
    decision.strategy = QuantOperandStrategy::kResidualThreeProduct;
    decision.reason = OperandReason::kSelected;
    decision.qualification = kQuantOperandProfile.qualification;
    decision.relative_l2_limit_ppm = kQuantOperandProfile.relative_l2_limit_ppm;
    return "// " + quant_operand_diagnostic(request, decision) + "\n" +
           emit_residual<T>(s, d);
  }
  Result<Shape> infer_shape(std::span<const Shape> in) const override {
    if (in.size() != 4 || in[1].rank() != 2 || !in[0].rank())
      return LSE_ERROR(kInvalidArgument,
                       "Q6 residual matrix requires four operands");
    Shape out;
    for (std::size_t i = 0; i + 1 < in[0].rank(); ++i)
      out.push_back(in[0].dim(i));
    out.push_back(in[1].dim(0));
    return out;
  }
  DType infer_dtype(std::span<const DType>) const override {
    return DType::kF32;
  }
  static ThreadPlan plan_impl(const KernelShapes &s) {
    ThreadPlan p;
    const auto d = dims_of(s);
    if (!d.valid)
      return p;
    p.workgroup_size[0] = 128;
    p.workgroup_count[0] = ((d.m + 63) / 64) * ((d.n + 63) / 64);
    p.workgroup_count[1] = p.workgroup_count[2] = 1;
    p.lds_bytes = kResidualLdsBytes;
    return p;
  }
};
template <math::MatrixElem T>
QuantOperandKernel residual_descriptor(const KernelShapes &s) {
  constexpr auto row = Base<T>::kRow;
  const auto operand =
      T == math::MatrixElem::kFp8 ? QuantOperand::kE4M3 : QuantOperand::kE5M2;
  const bool matrix = row.emittable() &&
                      math::has_cap(device_matrix_caps(*s.device), row.cap) &&
                      !s.intrinsics->find(row.key).empty();
  // The backend supplies the same conversion semantics to both dialects.
  bool conversion = !s.intrinsics->find(math::Fp8Format<T>::pack_key).empty();
  for (const auto key : math::Fp8Format<T>::value_keys)
    conversion = conversion && !s.intrinsics->find(key).empty();
  QuantOperandKernel kernel;
  kernel.operand = operand;
  kernel.strategy = QuantOperandStrategy::kResidualThreeProduct;
  kernel.implementation_id =
      quant_operand_implementation_id(ResidualKernel<T>::kName);
  kernel.implementation_revision = 2;
  kernel.implemented = true;
  kernel.matrix_intrinsic = matrix;
  kernel.conversion_intrinsic = conversion;
  kernel.block_absmax_scaling = true;
  kernel.fp32_exceptional_block_fallback = true;
  kernel.lds_bytes = kResidualLdsBytes;
  return kernel;
}
#include "q6_centered_repair.inc"

// Private M256 qualification route. Synthetic timing is not an accepted
// model-quality record. Both dialects select the same two projection shapes.
const KernelPrimitiveBase* select_centered_affine_candidate(
    const KernelShapes& s, const Dims& dims) {
  if (dims.m != 256 ||
      !((dims.n == 17408 && dims.k == 5120) ||
        (dims.n == 5120 && dims.k == 17408))) return nullptr;
  QuantOperandKernel kernel;
  kernel.operand = QuantOperand::kBF16;
  kernel.strategy = QuantOperandStrategy::kNative;
  kernel.implementation_id =
      quant_operand_implementation_id(CenteredAffineKernel::kName);
  kernel.implemented = true;
  kernel.matrix_intrinsic = select_centered_affine(s) != nullptr;
  kernel.conversion_intrinsic = true;
  kernel.fp32_exceptional_block_fallback = true;
  kernel.lds_bytes = 26160;
  QuantOperandProfile profile;
  profile.preferred = QuantOperand::kBF16;
  profile.strategy = QuantOperandStrategy::kNative;
  profile.min_m = profile.max_m = dims.m;
  profile.run_qualification_candidate = true;
  const auto decision = select_quant_operand(residual_request(s, dims), kernel,
                                            profile);
  return decision.reason == OperandReason::kSelected
      ? select_centered_affine(s) : nullptr;
}
} // namespace
bool expand_q6_centered_repair(graph::Node& node, const graph::KernelShapes& shapes) {
  const auto* selected = wmma_q6_linear_for(shapes);
  if (!selected || selected->name() != CenteredAffineKernel::kName ||
      node.materialized || node.inputs.size() != 4) return false;
  static const CenteredStage stage_primitive;
  static const CenteredRepair repair_primitive;
  const auto dims = dims_of(shapes);
  auto stage = std::make_shared<graph::Node>();
  stage->kind = graph::OpKind::kQuantMatMul;
  stage->prim = &stage_primitive;
  stage->fclass = graph::FusionClass::kBarrier;
  stage->dtype = DType::kF32;
  stage->shape = Shape{2ll * dims.m, dims.n};
  stage->inputs = node.inputs;
  stage->iattrs = node.iattrs;
  for (const auto& input : stage->inputs) ++input->consumer_count;
  stage->consumer_count = 1;
  node.inputs.insert(node.inputs.begin(), std::move(stage));
  node.kind = graph::OpKind::kCustom;
  node.prim = &repair_primitive;
  node.fclass = graph::FusionClass::kBarrier;
  return true;
}

const graph::KernelPrimitiveBase *
wmma_q6_linear_for(const graph::KernelShapes &s) {
  const auto dims = dims_of(s);
  const auto *global = std::getenv("LSE_WMMA");
  if (!s.device || !s.intrinsics || !dims.valid ||
      s.device->max_threads_per_workgroup < 128 ||
      s.device->compute_units != 64 ||
      (global && std::strcmp(global, "0") == 0))
    return nullptr;
  if (dims.m == 256) return select_centered_affine_candidate(s, dims);
  const auto request = residual_request(s, dims);
  // Matched driver195 projection experiments: eight post-warm host
  // eval+retire intervals per implementation. These are not device timestamps.
  struct Measured {
    uint32_t m, n, k;
    uint64_t bf16, e4m3, e5m2;
  };
  static constexpr Measured records[] = {
      {64, 17408, 5120, 1877865, 4170963, 1987625},
      {512, 17408, 5120, 6711219, 18451005, 9670537},
      {64, 5120, 17408, 3005271, 4375042, 3662411},
      {512, 5120, 17408, 8312625, 13622156, 11045698}};
  const Measured *record = nullptr;
  for (const auto &r : records)
    if (r.m == dims.m && r.n == dims.n && r.k == dims.k)
      record = &r;
  if (!record)
    return nullptr; // Unknown shapes and M1 retain the scalar path.
  constexpr uint64_t cohort = 0x202609230001ull;
  std::array<QuantOperandOption, 3> options{};
  auto &bf16 = options[0];
  bf16.kernel.operand = QuantOperand::kBF16;
  bf16.kernel.strategy = QuantOperandStrategy::kNative;
  bf16.kernel.implementation_id =
      quant_operand_implementation_id(StagedBF16Kernel::kName);
  bf16.kernel.implemented = true;
  bf16.kernel.matrix_intrinsic = select_staged_bf16(s) != nullptr;
  bf16.kernel.conversion_intrinsic = true; // Shared F32→BF16 cast lowering.
  bf16.kernel.lds_bytes = 16384;
  bf16.profile.preferred = QuantOperand::kBF16;
  bf16.profile.strategy = QuantOperandStrategy::kNative;
  bf16.profile.revision = 2;
  bf16.profile.qualification = OperandQualification::kAccepted;
  // Fifteen guarded projection fixtures plus complete-model logits. The
  // worst original-FP32 projection error was0.41996%; full-model error
  // was0.30484%. Cancellation-safe absolute checks and all guards passed.
  bf16.profile.measured_cases = 16;
  bf16.profile.measured_relative_l2_ppm = 4200;
  bf16.profile.absolute_error_pass = true;
  bf16.profile.nonfinite_pass = true;
  bf16.profile.model_quality_pass = true;
  bf16.profile.performance_pass = true;
  bf16.profile.run_qualification_candidate = false;
  options[1].kernel = residual_descriptor<math::MatrixElem::kFp8>(s);
  options[2].kernel = residual_descriptor<math::MatrixElem::kBf8>(s);
  for (size_t i = 1; i < options.size(); ++i) {
    options[i].profile.preferred = options[i].kernel.operand;
    options[i].profile.strategy = QuantOperandStrategy::kResidualThreeProduct;
    options[i].profile.run_qualification_candidate = false;
    // Both implementations exist, but neither wins the measured throughput
    // comparison. BF8 additionally fails the outlier quality qualification.
    // Keep them ineligible until an accepted quality/performance record exists.
    options[i].profile.performance_pass = false;
  }
  const uint64_t costs[] = {record->bf16, record->e4m3, record->e5m2};
  for (size_t i = 0; i < options.size(); ++i) {
    auto &c = options[i].cost;
    c.arch = "gfx1201";
    c.wave = 32;
    c.bits = 6;
    c.group = 64;
    c.m = dims.m;
    c.n = dims.n;
    c.k = dims.k;
    c.cohort = cohort;
    c.samples = 8;
    c.cost_ns = costs[i];
    c.accepted_profile_revision = options[i].profile.revision;
    c.implementation_id = options[i].kernel.implementation_id;
    c.implementation_revision = options[i].kernel.implementation_revision;
  }
  const auto winner = rank_quant_operands(request, options, cohort,
                                          OperandCostClock::kHostEvalRetire);
  if (winner.option == 0)
    return select_staged_bf16(s);
  if (winner.option == 1) {
    static const ResidualKernel<math::MatrixElem::kFp8> kernel;
    return &kernel;
  }
  if (winner.option == 2) {
    static const ResidualKernel<math::MatrixElem::kBf8> kernel;
    return &kernel;
  }
  return nullptr;
}

} // namespace lse::kernels
