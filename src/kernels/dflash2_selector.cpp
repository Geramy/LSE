#include "lse/graph/kernel_args.hpp"
#include "lse/graph/kernel_env.hpp"
#include "lse/graph/kernel_primitive.hpp"
#include "lse/math.hpp"
#include "lse/dispatch/dflash2.hpp"

#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>

namespace lse::kernels {
using namespace lse::graph;
namespace math = lse::math;
namespace {
Result<Shape> selector_shape(std::span<const Shape> in) {
  if (in.size() != 4 || in[0].rank() != 4 || in[1].rank() != 3 ||
      in[2].rank() != 4 || in[3].rank() != 3)
    return LSE_ERROR(kInvalidArgument, "DFlash2 selector needs predecessor, gate, successor, and unary tensors");
  const auto batch = in[0].dim(0), positions = in[0].dim(1);
  const auto predecessors = in[0].dim(2), rank = in[0].dim(3);
  const auto successors = in[2].dim(2);
  if (batch <= 0 || positions <= 0 || predecessors <= 0 || successors <= 0 || rank <= 0 ||
      in[1] != Shape{batch, positions, rank} ||
      in[2] != Shape{batch, positions, successors, rank} ||
      in[3] != Shape{batch, positions, successors})
    return LSE_ERROR(kInvalidArgument, "DFlash2 selector tensor geometry does not match");
  return Shape{batch, positions, predecessors, successors};
}
template <class E>
struct SelectorArgs {
  env::In<kir::f32, E> predecessor;
  env::In<kir::f32, E> gate;
  env::In<kir::f32, E> successor;
  env::In<kir::f32, E> unary;
  env::Out<kir::f32, E> out;
};
}
struct DFlash2SelectorKernel final : KernelPrimitive<DFlash2SelectorKernel> {
  static constexpr std::string_view kName = "dflash2.selector";
  static constexpr std::string_view kEntry = "lse_dflash2_selector";
  static constexpr std::string_view kSource = {};
  std::size_t arity() const noexcept override { return 4; }
  bool owns_indexing() const noexcept override { return true; }
  bool supports_epilogue() const noexcept override { return false; }
  bool has_host_impl() const noexcept override { return true; }
  std::string emit_kernel(const KernelShapes& s) const override {
    auto shape = selector_shape(s.inputs);
    if (!shape.ok() || s.output != *shape || s.input_dtypes.size() != 4 ||
        s.output_dtype != DType::kF32 || !s.types.scalar || !s.intrinsics || !s.store) return {};
    for (auto type : s.input_dtypes) if (type != DType::kF32) return {};
    const auto positions = static_cast<std::uint32_t>(s.inputs[0].dim(1));
    const auto predecessors = static_cast<std::uint32_t>(s.inputs[0].dim(2));
    const auto successors = static_cast<std::uint32_t>(s.inputs[2].dim(2));
    const auto rank = static_cast<std::uint32_t>(s.inputs[0].dim(3));
    if (s.attrs != std::array<float, 4>{static_cast<float>(positions), static_cast<float>(predecessors),
        static_cast<float>(successors), static_cast<float>(rank)}) return {};
    kir::KernelBody body(s.types, *s.intrinsics);
    body.set_store(s.store);
    SelectorArgs<env::Emit> a;
    if (!env::bind(body, a, s)) return {};
    env::Emit e{&body};
    const auto at = e.let(e.thread_id());
    if (auto valid = e.when(at < static_cast<std::uint32_t>(s.output.elem_count()))) {
      const auto successor = e.let(at % successors);
      const auto predecessor = e.let((at / successors) % predecessors);
      const auto row = e.let(at / (successors * predecessors));
      const auto pbase = e.let((row * predecessors + predecessor) * rank);
      const auto sbase = e.let((row * successors + successor) * rank);
      const auto gbase = e.let(row * rank);
      auto score = e.var(0.0f);
      for (auto r : e.range(rank)) {
        const auto conditioned = e.let(a.predecessor[pbase + r] * a.gate[gbase + r]);
        score = math::fma(conditioned, a.successor[sbase + r], score.read());
      }
      e.store(at, score.read() + a.unary[row * successors + successor]);
    }
    return body.str();
  }
  void eval_cpu(std::span<const float* const> in, float* out, std::size_t count,
                const std::array<float, 4>& attrs) const override {
    const auto predecessors = static_cast<std::size_t>(attrs[1]);
    const auto successors = static_cast<std::size_t>(attrs[2]);
    const auto rank = static_cast<std::size_t>(attrs[3]);
    for (std::size_t at = 0; at < count; ++at) {
      const auto successor = at % successors, predecessor = (at / successors) % predecessors;
      const auto row = at / (successors * predecessors);
      const auto pbase = (row * predecessors + predecessor) * rank;
      const auto sbase = (row * successors + successor) * rank;
      const auto gbase = row * rank;
      float score = 0;
      for (std::size_t r = 0; r < rank; ++r)
        score = std::fma(in[0][pbase + r] * in[1][gbase + r], in[2][sbase + r], score);
      out[at] = score + in[3][row * successors + successor];
    }
  }
  Result<Shape> infer_shape(std::span<const Shape> in) const override { return selector_shape(in); }
  DType infer_dtype(std::span<const DType>) const override { return DType::kF32; }
  static ThreadPlan plan_impl(const KernelShapes& s) {
    ThreadPlan plan;
    const std::uint32_t threads = s.device && s.device->max_threads_per_workgroup >= 256 ? 256u : 64u;
    plan.workgroup_size[0] = threads;
    plan.workgroup_count[0] = static_cast<std::uint32_t>((s.output.elem_count() + threads - 1) / threads);
    return plan;
  }
};
LSE_REGISTER_PRIMITIVE(DFlash2SelectorKernel);

namespace {
Result<Shape> walk_shape(std::span<const Shape> in) {
  if (in.size() != 2 || in[0].rank() != 4 || in[1].rank() != 3)
    return LSE_ERROR(kInvalidArgument, "DFlash2 walk needs scores and candidates");
  const auto b = in[1].dim(0), p = in[1].dim(1), k = in[1].dim(2);
  if (b <= 0 || p <= 0 || p > 7 || k <= 0 || k > (1 << 24) ||
      static_cast<std::uint64_t>(b) * static_cast<std::uint64_t>(p + 1) > UINT32_MAX ||
      in[0] != Shape{b, p, k, k})
    return LSE_ERROR(kInvalidArgument, "invalid DFlash2 walk geometry");
  for (const auto& shape : in) {
    std::uint64_t count = 1;
    for (std::size_t axis = 0; axis < shape.rank(); ++axis) {
      const auto extent = static_cast<std::uint64_t>(shape.dim(axis));
      if (extent > UINT32_MAX / count)
        return LSE_ERROR(kInvalidArgument, "DFlash2 walk exceeds indexing bounds");
      count *= extent;
    }
  }
  return Shape{b, p + 1};
}

bool walk_contract(const KernelShapes& s) {
  const auto expected = walk_shape(s.inputs);
  return expected.ok() && s.output == *expected &&
         s.input_dtypes.size() == 2 &&
         s.input_dtypes[0] == DType::kF32 &&
         s.input_dtypes[1] == DType::kF32 && s.output_dtype == DType::kU32 &&
         s.attrs[0] == static_cast<float>(s.inputs[1].dim(1)) &&
         s.attrs[1] == static_cast<float>(s.inputs[1].dim(2)) &&
         std::isfinite(s.attrs[2]) && s.attrs[2] >= s.attrs[1] &&
         s.attrs[2] <= static_cast<float>(1 << 24) &&
         std::floor(s.attrs[2]) == s.attrs[2] && s.attrs[3] == 0.0f;
}

struct WalkArgs {
  env::In<kir::f32, env::Emit> scores;
  env::In<kir::f32, env::Emit> candidates;
  env::Out<kir::u32, env::Emit> out;
};
}  // namespace

struct DFlash2WalkKernel final : KernelPrimitive<DFlash2WalkKernel> {
  static constexpr std::string_view kName = "dflash2.selector_walk.v1";
  static constexpr std::string_view kEntry = "lse_dflash2_selector_walk_v1";
  static constexpr std::string_view kSource = {};
  std::size_t arity() const noexcept override { return 2; }
  bool owns_indexing() const noexcept override { return true; }
  bool supports_epilogue() const noexcept override { return false; }
  bool has_typed_host_impl() const noexcept override { return true; }
  const KernelPrimitiveBase* specialize(const KernelShapes& s) const override {
    return walk_contract(s) && dispatch::selector_walk_wave(s) ? this : nullptr;
  }
  DType infer_dtype(std::span<const DType>) const override { return DType::kU32; }
  Result<Shape> infer_shape(std::span<const Shape> in) const override {
    return walk_shape(in);
  }
  static ThreadPlan plan_impl(const KernelShapes& s) {
    ThreadPlan plan;
    const auto wave = dispatch::selector_walk_wave(s);
    plan.workgroup_size[0] = wave ? wave : 32;
    plan.workgroup_count[0] = s.inputs.size() == 2 && s.inputs[1].rank() == 3
                                ? static_cast<std::uint32_t>(s.inputs[1].dim(0))
                                : 1u;
    return plan;
  }
  std::string emit_kernel(const KernelShapes& s) const override {
    const auto wave = dispatch::selector_walk_wave(s);
    if (!wave || !walk_contract(s)) return {};
    const auto positions = static_cast<std::uint32_t>(s.inputs[1].dim(1));
    const auto top = static_cast<std::uint32_t>(s.inputs[1].dim(2));
    const auto batches = static_cast<std::uint32_t>(s.inputs[1].dim(0));
    kir::KernelBody body(s.types, *s.intrinsics);
    body.set_store(s.store);
    WalkArgs a;
    if (!env::bind(body, a, s)) return {};
    env::Emit e{&body};
    const auto lane = e.let(math::local_id());
    const auto batch = e.let(math::workgroup_id_x());
    if (auto live = e.when(batch < batches)) {
      auto previous = e.var(e.u32(0));
      auto invalid_id = e.var(0.0f), invalid_score = e.var(0.0f);
      for (auto p : e.range(positions)) {
        auto best = e.var(math::neg_inf());
        auto token = e.var(s.attrs[2]);
        auto index = e.var(static_cast<float>(top));
        const auto cb = e.let((batch * positions + p) * top);
        const auto sb = e.let((batch * positions + p) * top * top + previous.read() * top);
        for (auto chunk : e.range((top + wave - 1u) / wave)) {
          const auto k = e.let(chunk * wave + lane);
          if (auto valid = e.when(k < top)) {
            const auto id = e.let(a.candidates[cb + k]);
            auto safe_id = e.var(s.attrs[2]);
            if (auto range = e.when(id >= 0.0f && id < s.attrs[2])) {
              if (auto integer = e.when(math::rint(id) == id)) safe_id = id;
            }
            if (auto bad = e.when(safe_id.read() == s.attrs[2])) invalid_id = 1.0f;
            const auto score = e.let(a.scores[sb + k]);
            const auto finite = e.let(score >= -std::numeric_limits<float>::max() &&
                                      score <= std::numeric_limits<float>::max());
            invalid_score = math::max(invalid_score.read(),
                select(finite, e.f32(0.0f), e.f32(1.0f)));
            if (auto good = e.when(finite)) {
              const auto ki = e.let(kir::cast<kir::f32>(k));
              if (auto better = e.when(score > best.read() ||
                  (score == best.read() && (safe_id.read() < token.read() ||
                   (safe_id.read() == token.read() && ki < index.read()))))) {
                best = score; token = safe_id.read(); index = ki;
              }
            }
          }
        }
        for (std::uint32_t bit = 1; bit < wave; bit <<= 1) {
          const auto other_score = e.let(math::shfl_xor(best.read(), e.u32(bit)));
          const auto other_token = e.let(math::shfl_xor(token.read(), e.u32(bit)));
          const auto other_index = e.let(math::shfl_xor(index.read(), e.u32(bit)));
          if (auto better = e.when(other_score > best.read() ||
              (other_score == best.read() && (other_token < token.read() ||
               (other_token == token.read() && other_index < index.read()))))) {
            best = other_score; token = other_token; index = other_index;
          }
        }
        previous = select(index.read() < static_cast<float>(top),
                          kir::cast<kir::u32>(index.read()), e.u32(0));
        if (auto writer = e.when(lane == 0u)) {
          auto word = e.var(kir::cast<kir::i32>(e.f32(0.0f)));
          if (auto valid = e.when(token.read() < s.attrs[2]))
            word = kir::cast<kir::i32>(token.read());
          a.out[batch * (positions + 1u) + p] = kir::cast<kir::u32>(word.read());
        }
      }
      for (std::uint32_t bit = 1; bit < wave; bit <<= 1) {
        invalid_id = math::max(invalid_id.read(), math::shfl_xor(invalid_id.read(), e.u32(bit)));
        invalid_score = math::max(invalid_score.read(), math::shfl_xor(invalid_score.read(), e.u32(bit)));
      }
      if (auto writer = e.when(lane == 0u)) {
        a.out[batch * (positions + 1u) + positions] =
            kir::cast<kir::u32>(kir::cast<kir::i32>(invalid_id.read()) * 2 +
                                kir::cast<kir::i32>(invalid_score.read()));
      }
    }
    return body.str();
  }
  Status eval_cpu_typed(std::span<const HostTensorView> in, HostOutputView out,
                       const std::array<float, 4>& attrs,
                       const std::array<std::int32_t, 4>&) const override {
    if (in.size() != 2)
      return LSE_ERROR(kInvalidArgument, "invalid DFlash2 walk inputs");
    const std::array shapes{in[0].shape, in[1].shape};
    const std::array types{in[0].dtype, in[1].dtype};
    KernelShapes s; s.inputs = shapes; s.input_dtypes = types;
    s.output = out.shape; s.output_dtype = out.dtype; s.attrs = attrs;
    if (!walk_contract(s) || in[0].bytes.size() != shapes[0].elem_count() * 4 ||
        in[1].bytes.size() != shapes[1].elem_count() * 4 ||
        out.bytes.size() != out.shape.elem_count() * 4)
      return LSE_ERROR(kInvalidArgument, "invalid DFlash2 walk buffers");
    const auto load = [&](std::size_t input, std::size_t at) {
      float value; std::memcpy(&value, in[input].bytes.data() + at * 4, 4); return value;
    };
    const auto store = [&](std::size_t at, std::uint32_t value) {
      std::memcpy(out.bytes.data() + at * 4, &value, 4);
    };
    const auto batches = static_cast<std::size_t>(shapes[1].dim(0));
    const auto positions = static_cast<std::size_t>(shapes[1].dim(1));
    const auto top = static_cast<std::size_t>(shapes[1].dim(2));
    for (std::size_t batch = 0; batch < batches; ++batch) {
      std::size_t previous = 0;
      std::uint32_t status = 0;
      for (std::size_t p = 0; p < positions; ++p) {
        float best = -std::numeric_limits<float>::infinity(), token = attrs[2];
        std::size_t winner = top;
        for (std::size_t k = 0; k < top; ++k) {
          const auto id = load(1, (batch * positions + p) * top + k);
          const bool valid_id = std::isfinite(id) && id >= 0 && id < attrs[2] && std::floor(id) == id;
          if (!valid_id) status |= 2;
          const float safe_id = valid_id ? id : attrs[2];
          const auto score = load(0, ((batch * positions + p) * top + previous) * top + k);
          if (!std::isfinite(score)) { status |= 1; continue; }
          if (score > best || (score == best &&
              (safe_id < token || (safe_id == token && k < winner)))) {
            best = score; token = safe_id; winner = k;
          }
        }
        store(batch * (positions + 1) + p, token < attrs[2] ? static_cast<std::uint32_t>(token) : 0u);
        previous = winner < top ? winner : 0;
      }
      store(batch * (positions + 1) + positions, status);
    }
    return OkStatus();
  }
};
LSE_REGISTER_PRIMITIVE(DFlash2WalkKernel);
}  // namespace lse::kernels
