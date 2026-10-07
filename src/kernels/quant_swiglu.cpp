#include "lse/dispatch/quant.hpp"
#include "lse/graph/kernel_args.hpp"
#include "lse/graph/kernel_env.hpp"
#include "lse/graph/kernel_primitive.hpp"
#include "lse/kernels/lds_linear.hpp"
#include "lse/math.hpp"
#include "lse/quant/group_affine_codec.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <vector>
namespace lse::kernels {
using namespace graph;
namespace {
struct Silu {
  static constexpr std::string_view key = "silu";
  using result = lse::f32;
};
struct Args {
  env::In<kir::f32, env::Emit> x;
  env::In<kir::u32, env::Emit> gw;
  env::In<lse::bf16, env::Emit> gs, gb;
  env::In<kir::u32, env::Emit> uw;
  env::In<lse::bf16, env::Emit> us, ub;
  env::In<kir::u32, env::Emit> panel;
  env::Out<kir::f32, env::Emit> out;
};
void dot(env::Emit &e, const Args &a, const kir::Val<kir::u32> &wb,
         const kir::Val<kir::u32> &sb, const kir::Val<kir::u32> &chunk,
         std::uint32_t count, std::span<const kir::LValue<kir::f32>> ga,
         std::span<const kir::LValue<kir::f32>> ua) {
  std::vector<kir::LValue<kir::f32>> gf, uf;
  for (std::size_t r = 0; r < ga.size(); ++r) {
    gf.push_back(e.var(0.0f));
    uf.push_back(e.var(0.0f));
  }
  auto gwords = e.load(a.gw, wb + chunk, count * 4u);
  auto uwords = e.load(a.uw, wb + chunk, count * 4u);
  std::vector<std::array<kir::Val<kir::u32>, 2>> gp, up;
  for (std::uint32_t j = 0; j < count; ++j) {
    std::array<kir::Val<kir::u32>, 2> g, u;
    for (int p = 0; p < 2; ++p) {
      g[p] = quant::dot4_code_plane(e, e.let(gwords[static_cast<int>(j)]), p);
      u[p] = quant::dot4_code_plane(e, e.let(uwords[static_cast<int>(j)]), p);
    }
    gp.push_back(g);
    up.push_back(u);
  }
  for (std::size_t rp = 0; rp < ga.size(); rp += 2) {
    std::vector<kir::Pack<kir::u32>> first, second, steps;
    const auto pair_rows = std::min<std::size_t>(2, ga.size() - rp);
    for (std::size_t at = 0; at < pair_rows; ++at) {
      auto base = e.u32(static_cast<std::uint32_t>((rp + at) * 2000));
      first.push_back(
          e.load(a.panel, e.let(base + chunk * 2u), count == 4 ? 16u : 8u));
      if (count == 4)
        second.push_back(e.load(a.panel, e.let(base + chunk * 2u + 4u), 16u));
      steps.push_back(e.load(a.panel, e.let(base + 1280u + chunk), count * 4u));
    }
    for (std::uint32_t j = 0; j < count; ++j) {
      for (std::size_t at = 0; at < pair_rows; ++at) {
        auto gi = e.var(kir::cast<kir::i32>(e.u32(0)));
        auto ui = e.var(kir::cast<kir::i32>(e.u32(0)));
        for (int p = 0; p < 2; ++p) {
          auto x =
              e.let((j < 2 ? first[at]
                           : second[at])[static_cast<int>((j % 2) * 2u) + p]);
          gi = math::dot4_iu8(kir::cast<kir::i32>(x),
                              kir::cast<kir::i32>(gp[j][p]), gi.read());
          ui = math::dot4_iu8(kir::cast<kir::i32>(x),
                              kir::cast<kir::i32>(up[j][p]), ui.read());
        }
        auto step =
            e.let(math::from_bits<lse::f32>(steps[at][static_cast<int>(j)]));
        gf[rp + at] =
            math::fma(step, kir::cast<kir::f32>(gi.read()), gf[rp + at].read());
        uf[rp + at] =
            math::fma(step, kir::cast<kir::f32>(ui.read()), uf[rp + at].read());
      }
    }
  }
  auto group = e.let(chunk / 8u);
  auto gs = e.let(math::widen(a.gs[sb + group]));
  auto us = e.let(math::widen(a.us[sb + group]));
  for (std::size_t r = 0; r < ga.size(); ++r) {
    ga[r] = math::fma(gs, gf[r].read(), ga[r].read());
    ua[r] = math::fma(us, uf[r].read(), ua[r].read());
  }
}
KernelShapes original(const KernelShapes &s) {
  auto t = s;
  t.inputs = s.inputs.first(4);
  t.input_dtypes = s.input_dtypes.first(4);
  return t;
}
struct Pair final : KernelPrimitive<Pair> {
  static constexpr std::string_view kName = "quant_swiglu.q4_shared_panel.v2";
  static constexpr std::string_view kEntry =
      "lse_quant_swiglu_q4_shared_panel_v2";
  static constexpr std::string_view kSource = {};
  std::size_t arity() const noexcept override { return 8; }
  bool owns_indexing() const noexcept override { return true; }
  DType infer_dtype(std::span<const DType>) const override {
    return DType::kF32;
  }
  bool supports_epilogue() const noexcept override { return false; }
  bool has_typed_host_impl() const noexcept override { return true; }
  Result<Shape> infer_shape(std::span<const Shape> in) const override {
    if (in.size() != 8 || in[0].rank() != 3 || in[1].rank() != 2)
      return LSE_ERROR(kInvalidArgument, "invalid Q4 SwiGLU shape");
    static constexpr std::array types{DType::kF32,  DType::kU32, DType::kBF16,
                                      DType::kBF16, DType::kU32, DType::kBF16,
                                      DType::kBF16, DType::kU32};
    KernelShapes s;
    s.inputs = in;
    s.input_dtypes = types;
    s.iattrs = {4, 64, 0, 0};
    s.output = Shape{in[0].dim(0), in[0].dim(1), in[1].dim(0)};
    if (!dispatch::q4_swiglu_shape(s))
      return LSE_ERROR(kInvalidArgument, "unqualified Q4 SwiGLU shape");
    return s.output;
  }
  Status eval_cpu_typed(std::span<const HostTensorView>, HostOutputView,
                        const std::array<float, 4> &,
                        const std::array<std::int32_t, 4> &) const override;
  static bool valid(const KernelShapes &s) {
    const auto *rule = dispatch::q4_swiglu_rule(s);
    if (!rule)
      return false;
    const auto schedule = dot4_schedule(original(s));
    // The grid has no row axis: one workgroup row must cover every row.
    return schedule.valid() && schedule.rows >= rule->m &&
           schedule.wave == rule->wave &&
           schedule.chunks_per_lane == rule->chunks_per_lane &&
           schedule.k_splits == rule->k_splits;
  }
  std::string emit_kernel(const KernelShapes &s) const override {
    if (!valid(s) || !s.types.scalar || !s.store || !s.intrinsics)
      return {};
    auto sched = dot4_schedule(original(s));
    // A pass shorter than the rule's rows (dispatch::verify_rows) computes
    // its own rows alone, each exactly as the rule's pass computes it.
    const auto rows = std::min<std::uint32_t>(
        sched.rows, static_cast<std::uint32_t>(s.inputs[0].dim(1)));
    kir::KernelBody kb(s.types, *s.intrinsics, 0);
    kb.set_store(s.store);
    Args a;
    if (!env::bind(kb, a, s))
      return {};
    env::Emit e{&kb};
    auto lid = e.let(math::local_id());
    auto lane = e.let(lid % 32u);
    auto col = e.let(math::workgroup_id_x() * 8u + lid / 32u);
    std::vector<kir::LValue<kir::f32>> ga, ua;
    for (std::uint32_t r = 0; r < rows; ++r) {
      ga.push_back(e.var(0.0f));
      ua.push_back(e.var(0.0f));
    }
    if (auto active = e.when(col < 17408u)) {
      auto wb = e.let(col * 640u);
      auto sb = e.let(col * 80u);
      for (std::uint32_t ks = 0; ks < sched.k_splits; ++ks) {
        auto cb = ks * (640u / sched.k_splits), ce = cb + 640u / sched.k_splits;
        auto aligned = ((ce - cb) / 128u) * 128u;
        for (auto c : e.range(cb, cb + aligned, 128u))
          dot(e, a, wb, sb, e.let(c + lane * 4u), 4, ga, ua);
        if (aligned < ce - cb)
          for (auto c : e.range(e.u32(cb + aligned) + lane, e.u32(ce), 32u))
            dot(e, a, wb, sb, c, 1, ga, ua);
        auto accumulate_bias = [&](auto g) {
          auto gb = e.let(math::widen(a.gb[sb + g]));
          auto ub = e.let(math::widen(a.ub[sb + g]));
          for (std::uint32_t r = 0; r < rows; ++r) {
            auto sum = e.let(math::from_bits<lse::f32>(
                a.panel[e.u32(static_cast<std::uint32_t>(r * 2000 + 1920)) +
                        g]));
            ga[r] = math::fma(gb, sum, ga[r].read());
            ua[r] = math::fma(ub, sum, ua[r].read());
          }
        };
        if (rows == 1 && sched.k_splits == 1) {
          accumulate_bias(lane);
          accumulate_bias(e.let(lane + 32u));
          if (auto tail = e.when(lane < 16u))
            accumulate_bias(e.let(lane + 64u));
        } else {
          for (auto g : e.range(e.u32(cb / 8u) + lane, e.u32(ce / 8u), 32u))
            accumulate_bias(g);
        }
      }
    }
    for (std::uint32_t r = 0; r < rows; ++r) {
      for (std::uint32_t bit = 1; bit < 32; bit <<= 1)
        ga[r] = ga[r].read() + math::shfl_xor(ga[r].read(), e.u32(bit));
      for (std::uint32_t bit = 1; bit < 32; bit <<= 1)
        ua[r] = ua[r].read() + math::shfl_xor(ua[r].read(), e.u32(bit));
    }
    if (auto lane0 = e.when(lane == 0u && col < 17408u))
      for (std::uint32_t r = 0; r < rows; ++r) {
        auto g = ga[r].read();
        auto silu = e.let(math::emit<Silu>(g));
        e.store(e.u32(static_cast<std::uint32_t>(r * 17408)) + col,
                silu * ua[r].read());
      }
    return kb.str();
  }
  static ThreadPlan plan_impl(const KernelShapes &s) {
    ThreadPlan p;
    p.workgroup_count[0] = 0;
    if (valid(s)) {
      p.workgroup_size[0] = 256;
      p.workgroup_count[0] = 2176;
    }
    return p;
  }
};
Status Pair::eval_cpu_typed(std::span<const HostTensorView> in,
                            HostOutputView out, const std::array<float, 4> &,
                            const std::array<std::int32_t, 4> &iattrs) const {
  std::vector<Shape> shapes;
  std::vector<DType> types;
  for (const auto &v : in) {
    shapes.push_back(v.shape);
    types.push_back(v.dtype);
  }
  KernelShapes s;
  s.inputs = shapes;
  s.input_dtypes = types;
  s.output = out.shape;
  s.output_dtype = out.dtype;
  s.iattrs = iattrs;
  const auto *rule = dispatch::q4_swiglu_shape(s);
  if (!rule || out.bytes.size() != out.shape.elem_count() * 4)
    return LSE_ERROR(kInvalidArgument, "invalid Q4 SwiGLU host output");
  for (const auto &v : in)
    if (v.bytes.size() != v.shape.elem_count() * dtype_info(v.dtype).size_bytes)
      return LSE_ERROR(kInvalidArgument, "invalid Q4 SwiGLU host input");
  const auto word = [&](std::size_t slot, std::size_t at) {
    std::uint32_t v;
    std::memcpy(&v, in[slot].bytes.data() + at * 4, 4);
    return v;
  };
  const auto affine = [&](std::size_t slot, std::size_t at) {
    std::uint16_t v;
    std::memcpy(&v, in[slot].bytes.data() + at * 2, 2);
    return std::bit_cast<float>(static_cast<std::uint32_t>(v) << 16);
  };
  const auto chunks = static_cast<std::size_t>(rule->k / 8);
  const auto groups = static_cast<std::size_t>(rule->k / rule->group);
  const auto stride = groups * 25;
  const auto pass_rows = static_cast<std::size_t>(in[0].shape.dim(1));
  for (std::size_t row = 0; row < pass_rows; ++row)
    for (std::size_t col = 0; col < static_cast<std::size_t>(rule->n); ++col) {
      std::array<std::array<float, 32>, 2> acc{};
      for (std::size_t split = 0; split < rule->k_splits; ++split) {
        const auto cb = split * (chunks / rule->k_splits),
                   ce = cb + chunks / rule->k_splits;
        const auto aligned = ((ce - cb) / 128) * 128;
        for (std::size_t lane = 0; lane < 32; ++lane) {
          const auto add = [&](std::size_t chunk, std::size_t count) {
            std::array<float, 2> partial{};
            for (std::size_t j = 0; j < count; ++j) {
              const auto c = chunk + j;
              std::array<std::int32_t, 2> dot{};
              const std::array weights{word(1, col * chunks + c),
                                       word(4, col * chunks + c)};
              for (std::size_t plane = 0; plane < 2; ++plane) {
                const auto codes = word(7, row * stride + c * 2 + plane);
                for (std::size_t byte = 0; byte < 4; ++byte) {
                  const auto x = std::bit_cast<std::int8_t>(
                      static_cast<std::uint8_t>(codes >> (byte * 8)));
                  for (std::size_t p = 0; p < 2; ++p)
                    dot[p] +=
                        x * static_cast<std::int32_t>(
                                (weights[p] >> ((byte * 2 + plane) * 4)) & 15);
                }
              }
              const auto step =
                  std::bit_cast<float>(word(7, row * stride + rule->k / 4 + c));
              for (std::size_t p = 0; p < 2; ++p)
                partial[p] =
                    std::fma(step, static_cast<float>(dot[p]), partial[p]);
            }
            for (std::size_t p = 0; p < 2; ++p)
              acc[p][lane] =
                  std::fma(affine(2 + p * 3, col * groups + chunk / 8),
                           partial[p], acc[p][lane]);
          };
          for (auto c = cb; c < cb + aligned; c += 128)
            add(c + lane * 4, 4);
          for (auto c = cb + aligned + lane; c < ce; c += 32)
            add(c, 1);
          for (auto g = cb / 8 + lane; g < ce / 8; g += 32) {
            const auto sum = std::bit_cast<float>(
                word(7, row * stride + 3 * rule->k / 8 + g));
            for (std::size_t p = 0; p < 2; ++p)
              acc[p][lane] = std::fma(affine(3 + p * 3, col * groups + g), sum,
                                      acc[p][lane]);
          }
        }
      }
      for (std::size_t p = 0; p < 2; ++p)
        for (std::size_t bit = 1; bit < 32; bit *= 2) {
          const auto before = acc[p];
          for (std::size_t lane = 0; lane < 32; ++lane)
            acc[p][lane] = before[lane] + before[lane ^ bit];
        }
      const float value =
          (acc[0][0] / (1.0f + std::exp(-acc[0][0]))) * acc[1][0];
      std::memcpy(out.bytes.data() + (row * rule->n + col) * 4, &value, 4);
    }
  return Status{};
}

} // namespace
LSE_REGISTER_PRIMITIVE(Pair);
} // namespace lse::kernels
