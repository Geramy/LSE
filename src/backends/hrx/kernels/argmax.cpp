// Row argmax in two launches, so greedy decode reads back 4 bytes instead of
// the whole logit row.
//
// The partial stage gives one workgroup per chunk of the row: each thread
// scans a strided slice, the workgroup tree-reduces through LDS, and lane 0
// writes (max value, index) for its chunk. The final stage is one thread per
// row folding the partials. Indices ride as f32, the engine's convention
// (see topk), and ties take the smaller index everywhere — the comparator
// must match runtime::argmax and the host interpreter exactly.
#include <string>

#include "lse/backends/hrx/device_info.hpp"
#include "lse/graph/kernel_args.hpp"
#include "lse/graph/kernel_env.hpp"
#include "lse/graph/kernel_primitive.hpp"
#include "lse/math.hpp"

namespace lse::backend::hrx_kernels {

using namespace lse::graph;
namespace math = lse::math;

namespace {

constexpr std::uint32_t kBlock = 256;

}  // namespace

// The output leaves through the emitter's store hook; the Out member only
// names the slot the binding contract requires.
template <class E>
struct ArgMaxPartialArgs {
  env::In<kir::f32, E> x;
  env::Out<kir::f32, E> out;
};

struct ArgMaxPartialKernel final : KernelPrimitive<ArgMaxPartialKernel> {
  static constexpr std::string_view kName = "argmax.partial";
  static constexpr std::string_view kEntry = "lse_argmax_partial";
  static constexpr std::string_view kSource = {};

  std::size_t arity() const noexcept override { return 1; }
  bool owns_indexing() const noexcept override { return true; }

  std::string emit_kernel(const KernelShapes& s) const override {
    if (s.inputs.size() != 1 || s.types.scalar == nullptr ||
        s.intrinsics == nullptr || !s.store || s.inputs[0].rank() == 0 ||
        s.output.rank() < 2) {
      return {};
    }
    const auto n = static_cast<std::uint32_t>(
        s.inputs[0].dim(s.inputs[0].rank() - 1));
    const auto chunk = static_cast<std::uint32_t>(s.iattrs[1]);
    if (n == 0 || chunk == 0) return {};
    const auto nchunks =
        static_cast<std::uint32_t>(s.output.dim(s.output.rank() - 2));
    if (nchunks != (n + chunk - 1) / chunk) return {};

    kir::KernelBody k(s.types, *s.intrinsics, workgroup_lds_bytes(s.device));
    k.set_store(s.store);
    ArgMaxPartialArgs<env::Emit> a;
    env::bind(k, a);
    env::Emit e{&k};

    const auto lid = e.let(math::local_id());
    const auto c = e.let(math::workgroup_id_x());
    const auto row = e.let(math::workgroup_id_y());
    const auto start = e.let(c * chunk);
    auto stop = e.var(start + e.u32(chunk));
    if (auto clip = e.when(stop.read() > e.u32(n))) {
      stop = e.u32(n);
    }

    // Sentinel index n never survives: every chunk holds at least one element
    // and -inf == -inf ties resolve toward the real (smaller) index.
    auto bv = e.var(0.0f);
    bv = math::neg_inf();
    auto bi = e.var(static_cast<float>(n));
    for (auto t : e.range(start + lid, stop.read(), kBlock)) {
      const auto v = e.let(a.x[row * n + t]);
      const auto idx = e.let(cast<kir::f32>(t));
      if (auto take = e.when(v > bv.read() ||
                             (v == bv.read() && idx < bi.read()))) {
        bv = v;
        bi = idx;
      }
    }

    auto sv = e.lds<kir::f32>(kBlock);
    auto si = e.lds<kir::f32>(kBlock);
    if (!sv || !si) return {};
    sv[lid] = bv.read();
    si[lid] = bi.read();
    e.barrier();
    for (std::uint32_t off = kBlock / 2; off > 0; off >>= 1) {
      if (auto low = e.when(lid < off)) {
        const auto ov = e.let(sv[lid + off].read());
        const auto oi = e.let(si[lid + off].read());
        const auto cv = e.let(sv[lid].read());
        const auto ci = e.let(si[lid].read());
        if (auto take = e.when(ov > cv || (ov == cv && oi < ci))) {
          sv[lid] = ov;
          si[lid] = oi;
        }
      }
      e.barrier();
    }
    if (auto lead = e.when(lid == 0)) {
      const auto slot = e.let((row * nchunks + c) * e.u32(2));
      e.store(slot, sv[0].read());
      e.store(slot + e.u32(1), si[0].read());
    }
    if (!k.lds().ok()) return {};
    return k.str();
  }

  Result<Shape> infer_shape(std::span<const Shape> in) const override {
    if (in.size() != 1) {
      return LSE_ERROR(kInvalidArgument, "argmax.partial takes 1 input");
    }
    return in[0];
  }
  DType infer_dtype(std::span<const DType>) const override {
    return DType::kF32;
  }

  static ThreadPlan plan_impl(const KernelShapes& s) {
    ThreadPlan tp;
    tp.workgroup_size[0] = kBlock;
    const auto rank = s.output.rank();
    const auto nchunks =
        rank >= 2 ? static_cast<std::uint32_t>(s.output.dim(rank - 2)) : 1u;
    const auto elems = static_cast<std::uint32_t>(s.output.elem_count());
    tp.workgroup_count[0] = nchunks == 0 ? 1u : nchunks;
    tp.workgroup_count[1] =
        nchunks == 0 ? 1u : (elems + 2 * nchunks - 1) / (2 * nchunks);
    tp.lds_bytes = 2 * kBlock * static_cast<std::uint32_t>(sizeof(float));
    return tp;
  }
};
LSE_REGISTER_PRIMITIVE(ArgMaxPartialKernel);

// Not self-indexing: one thread folds its row's partials and hands the index
// back through `ret`, so `out` is bound but never written here.
template <class E>
struct ArgMaxFinalArgs {
  env::In<kir::f32, E> x;
  env::Out<kir::f32, E> out;
};

struct ArgMaxFinalKernel final : KernelPrimitive<ArgMaxFinalKernel> {
  static constexpr std::string_view kName = "argmax.final";
  static constexpr std::string_view kEntry = "lse_argmax_final";
  static constexpr std::string_view kSource = {};

  std::size_t arity() const noexcept override { return 1; }

  std::string emit_kernel(const KernelShapes& s) const override {
    if (s.inputs.size() != 1 || s.types.scalar == nullptr ||
        s.intrinsics == nullptr || s.inputs[0].rank() < 2) {
      return {};
    }
    const auto nchunks = static_cast<std::uint32_t>(
        s.inputs[0].dim(s.inputs[0].rank() - 2));
    if (nchunks == 0 || s.inputs[0].dim(s.inputs[0].rank() - 1) != 2) {
      return {};
    }

    kir::KernelBody k(s.types, *s.intrinsics);
    ArgMaxFinalArgs<env::Emit> a;
    env::bind(k, a);
    env::Emit e{&k};
    const auto base = e.let(e.thread_id() * (2 * nchunks));
    auto bv = e.var(0.0f);
    bv = math::neg_inf();
    auto bi = e.var(3.4e38f);
    for (auto c : e.range(nchunks)) {
      const auto v = e.let(a.x[base + c * 2u]);
      const auto idx = e.let(a.x[base + c * 2u + 1u]);
      if (auto take = e.when(v > bv.read() ||
                             (v == bv.read() && idx < bi.read()))) {
        bv = v;
        bi = idx;
      }
    }
    e.ret(bi.read());
    return k.str();
  }

  Result<Shape> infer_shape(std::span<const Shape> in) const override {
    if (in.size() != 1 || in[0].rank() < 2) {
      return LSE_ERROR(kInvalidArgument, "argmax.final takes [.., nchunks, 2]");
    }
    Shape out;
    for (std::size_t i = 0; i + 2 < in[0].rank(); ++i) out.push_back(in[0].dim(i));
    if (out.rank() == 0) out.push_back(1);
    return out;
  }
  DType infer_dtype(std::span<const DType>) const override {
    return DType::kF32;
  }

  static ThreadPlan plan_impl(const KernelShapes& s) {
    ThreadPlan tp;
    const std::uint32_t threads =
        s.device && s.device->max_threads_per_workgroup >= 256 ? 256u : 64u;
    const auto elems = static_cast<std::uint32_t>(s.output.elem_count());
    tp.workgroup_size[0] = threads;
    tp.workgroup_count[0] = elems == 0 ? 1u : (elems + threads - 1) / threads;
    return tp;
  }
};
LSE_REGISTER_PRIMITIVE(ArgMaxFinalKernel);

}  // namespace lse::backend::hrx_kernels
