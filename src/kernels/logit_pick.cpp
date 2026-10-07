// Per-row log-softmax statistics for scoring a known sequence.
//
// One workgroup per logit row. The row is read twice: a max pass, then a pass
// summing exp(x - max), each tree-reduced through LDS. Each row has P picked
// token ids (the next token, and for a KL-divergence comparison the base run's
// top-k ids). The kernel writes P + 2 floats per row: the max M, the sum S and
// the logit of each picked id. The caller forms log p(id) = logit - M - log(S)
// on the host in double precision, so the device returns (P + 2) * 4 bytes per
// scored token instead of the whole vocabulary row.
//
// Picks ride as f32, the engine's convention for indices, and must be integers
// in [0, vocab).
#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "lse/backends/hrx/device_info.hpp"
#include "lse/graph/kernel_args.hpp"
#include "lse/graph/kernel_env.hpp"
#include "lse/graph/kernel_primitive.hpp"
#include "lse/math.hpp"

namespace lse::kernels {

using backend::workgroup_lds_bytes;
using namespace lse::graph;
namespace math = lse::math;

namespace {

constexpr std::uint32_t kBlock = 256;

// [.., V] logits and [.., P] picks with the same leading dims -> [.., P + 2].
Result<Shape> pick_shape(std::span<const Shape> in) {
  if (in.size() != 2 || in[0].rank() == 0 || in[1].rank() != in[0].rank())
    return LSE_ERROR(kInvalidArgument, "logits.lse_pick takes [.., V] logits and [.., P] picks");
  const auto rank = in[0].rank();
  for (std::size_t i = 0; i + 1 < rank; ++i)
    if (in[0].dim(i) != in[1].dim(i))
      return LSE_ERROR(kInvalidArgument, "logits.lse_pick needs one pick row per logit row");
  if (in[0].dim(rank - 1) <= 0 || in[1].dim(rank - 1) <= 0)
    return LSE_ERROR(kInvalidArgument, "logits.lse_pick over an empty vocabulary or pick list");
  Shape out;
  for (std::size_t i = 0; i + 1 < rank; ++i) out.push_back(in[0].dim(i));
  out.push_back(in[1].dim(rank - 1) + 2);
  return out;
}

template <class E>
struct LogitPickArgs {
  env::In<kir::f32, E> x;
  env::In<kir::f32, E> targets;
  env::Out<kir::f32, E> out;
};

}  // namespace

struct LogitPickKernel final : KernelPrimitive<LogitPickKernel> {
  static constexpr std::string_view kName = "logits.lse_pick.v1";
  static constexpr std::string_view kEntry = "lse_logits_lse_pick_v1";
  static constexpr std::string_view kSource = {};

  std::size_t arity() const noexcept override { return 2; }
  bool owns_indexing() const noexcept override { return true; }
  bool supports_epilogue() const noexcept override { return false; }
  bool has_typed_host_impl() const noexcept override { return true; }

  std::string emit_kernel(const KernelShapes& s) const override {
    if (s.inputs.size() != 2 || s.types.scalar == nullptr || s.intrinsics == nullptr ||
        !s.store || !pick_shape(s.inputs).ok())
      return {};
    const auto vocab = static_cast<std::uint32_t>(s.inputs[0].dim(s.inputs[0].rank() - 1));
    const auto picks = static_cast<std::uint32_t>(s.inputs[1].dim(s.inputs[1].rank() - 1));
    if (picks > kBlock) return {};  // one lane per pick

    kir::KernelBody k(s.types, *s.intrinsics, workgroup_lds_bytes(s.device));
    k.set_store(s.store);
    LogitPickArgs<env::Emit> a;
    if (!env::bind(k, a, s)) return {};
    env::Emit e{&k};

    const auto lid = e.let(math::local_id());
    const auto row = e.let(math::workgroup_id_x());
    const auto base = e.let(row * vocab);

    auto sm = e.lds<kir::f32>(kBlock);
    auto ss = e.lds<kir::f32>(kBlock);
    if (!sm || !ss) return {};

    auto m = e.var(0.0f);
    m = math::neg_inf();
    for (auto t : e.range(lid, e.u32(vocab), kBlock)) {
      m = math::max(m.read(), a.x[base + t]);
    }
    sm[lid] = m.read();
    e.barrier();
    for (std::uint32_t off = kBlock / 2; off > 0; off >>= 1) {
      if (auto low = e.when(lid < off)) {
        sm[lid] = math::max(sm[lid].read(), sm[lid + off].read());
      }
      e.barrier();
    }
    const auto top = e.let(sm[0].read());

    auto acc = e.var(0.0f);
    for (auto t : e.range(lid, e.u32(vocab), kBlock)) {
      acc = acc.read() + math::exp(a.x[base + t] - top);
    }
    ss[lid] = acc.read();
    e.barrier();
    for (std::uint32_t off = kBlock / 2; off > 0; off >>= 1) {
      if (auto low = e.when(lid < off)) {
        ss[lid] = ss[lid].read() + ss[lid + off].read();
      }
      e.barrier();
    }
    const auto sum = e.let(ss[0].read());
    const auto slot = e.let(row * e.u32(picks + 2));
    if (auto lead = e.when(lid == 0)) {
      e.store(slot, top);
      e.store(slot + e.u32(1), sum);
    }
    // One lane per pick: a guarded store, not a lane-divergent loop, which
    // Loom's branch lowering does not take.
    if (auto lane = e.when(lid < e.u32(picks))) {
      const auto id = e.let(kir::cast<kir::u32>(kir::cast<kir::i32>(a.targets[row * picks + lid])));
      e.store(slot + e.u32(2) + lid, a.x[base + id]);
    }
    if (!k.lds().ok()) return {};
    return k.str();
  }

  Result<Shape> infer_shape(std::span<const Shape> in) const override { return pick_shape(in); }
  DType infer_dtype(std::span<const DType>) const override { return DType::kF32; }

  static ThreadPlan plan_impl(const KernelShapes& s) {
    ThreadPlan tp;
    tp.workgroup_size[0] = kBlock;
    const auto width = s.output.rank() == 0 ? 1u
        : static_cast<std::uint32_t>(s.output.dim(s.output.rank() - 1));
    const auto rows = static_cast<std::uint32_t>(s.output.elem_count() / width);
    tp.workgroup_count[0] = rows == 0 ? 1u : rows;
    tp.lds_bytes = 2 * kBlock * static_cast<std::uint32_t>(sizeof(float));
    return tp;
  }

  Status eval_cpu_typed(std::span<const HostTensorView> in, HostOutputView out,
                        const std::array<float, 4>&,
                        const std::array<std::int32_t, 4>&) const override {
    const std::array shapes{in.size() > 0 ? in[0].shape : Shape{},
                            in.size() > 1 ? in[1].shape : Shape{}};
    LSE_ASSIGN_OR(Shape expect, pick_shape(shapes));
    if (in[0].dtype != DType::kF32 || in[1].dtype != DType::kF32 || out.shape != expect ||
        in[0].bytes.size() != shapes[0].elem_count() * 4 ||
        in[1].bytes.size() != shapes[1].elem_count() * 4 ||
        out.bytes.size() != expect.elem_count() * 4)
      return LSE_ERROR(kInvalidArgument, "logits.lse_pick needs f32 buffers of the declared shapes");
    const auto vocab = static_cast<std::size_t>(shapes[0].dim(shapes[0].rank() - 1));
    const auto picks = static_cast<std::size_t>(shapes[1].dim(shapes[1].rank() - 1));
    const std::size_t rows = shapes[0].elem_count() / vocab;
    const auto load = [&](std::size_t input, std::size_t at) {
      float v;
      std::memcpy(&v, in[input].bytes.data() + at * 4, 4);
      return v;
    };
    std::vector<float> values(picks + 2);
    for (std::size_t r = 0; r < rows; ++r) {
      float top = -std::numeric_limits<float>::infinity();
      for (std::size_t v = 0; v < vocab; ++v) top = std::max(top, load(0, r * vocab + v));
      double sum = 0.0;
      for (std::size_t v = 0; v < vocab; ++v)
        sum += std::exp(static_cast<double>(load(0, r * vocab + v)) - static_cast<double>(top));
      values[0] = top;
      values[1] = static_cast<float>(sum);
      for (std::size_t j = 0; j < picks; ++j) {
        const float id = load(1, r * picks + j);
        if (!(id >= 0.0f) || id >= static_cast<float>(vocab) || std::floor(id) != id)
          return LSE_ERROR(kInvalidArgument, "logits.lse_pick id ", std::to_string(id),
                           " is not a token id below ", std::to_string(vocab));
        values[2 + j] = load(0, r * vocab + static_cast<std::size_t>(id));
      }
      std::memcpy(out.bytes.data() + r * (picks + 2) * 4, values.data(), values.size() * 4);
    }
    return OkStatus();
  }
};
LSE_REGISTER_PRIMITIVE(LogitPickKernel);

}  // namespace lse::kernels
