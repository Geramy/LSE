// `linear` on the matrix core.
//
// One wave owns one output tile and walks K in the step its table row names,
// accumulating in the accumulator fragment that row describes. Nothing below
// spells a fragment width, a K step, a lane index or an instruction: those all
// come from `MatrixCoreRow`, so a new operand format — f16, bf16, int8, int4,
// a future fp8 or a different device generation — is a row in `lse::math`'s
// table and nothing here changes.
//
// The measured layouts the gfx1151 rows carry were established on the device
// rather than taken from documentation, because they are not guessable and the
// generations disagree. A row whose layout has not been measured is still a
// real row — name, widths, capability, shape — but it is not emittable, so the
// gate below declines it and the group falls back visibly. That is what keeps
// an unverified gfx12 or CDNA layout from becoming a silent wrong answer.
//
// The operand element follows the weight's storage: a bf16 checkpoint feeds the
// bf16 form of the instruction directly, which is both the arithmetic it was
// trained in and the one that keeps f32's exponent range. f32 weights have no
// matching operand form and keep the f16 one. Default is on when the live
// device has the row's capability; LSE_WMMA=0 keeps the f32 scalar oracle,
// which is what the lemonseed differential in test_reference compares against.
#include "lse/kernels/wmma.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "lse/backends/hrx/device_info.hpp"
#include "lse/kernels/vec_mem.hpp"
#include "lse/graph/kernel_args.hpp"
#include "lse/graph/kernel_env.hpp"
#include "lse/math.hpp"

namespace lse::kernels {

// These name device facts, which the backend supplies.

using namespace lse::graph;
namespace math = lse::math;

namespace {

// The output tile one wave owns. Blocking is the author's choice, so it is
// named here — once — and everything else about the instruction comes from the
// row this shape selects.
constexpr int kTileM = 16;
constexpr int kTileN = 16;
constexpr int kTileK = 16;

// The generations whose layouts have been measured here, and therefore the
// ones a body may be emitted for. Bringing gfx1201 or gfx942 up is measuring
// its rows and adding its target to this pack — the body itself already reads
// everything it needs from the row.
template <math::MatrixTarget... Live, class F>
[[nodiscard]] std::string for_live_target(math::MatrixTarget t, F&& fn) {
  std::string out;
  const auto one = [&]<math::MatrixTarget L>() {
    if (t == L) out = fn.template operator()<L>();
  };
  (one.template operator()<Live>(), ...);
  return out;
}

// LSE_WMMA=0 forces the scalar loop. Anything else, including unset, follows
// the device.
bool wmma_forced_off() {
  const char* v = std::getenv("LSE_WMMA");
  return v != nullptr && std::strcmp(v, "0") == 0;
}

struct LinearDims {
  std::int64_t m = 0;
  std::int64_t n = 0;
  // Row length in *buffer* elements. Equal to K for an unpacked operand; K/4
  // for int8 riding four to an i32.
  std::int64_t kb = 0;
  bool valid = false;
};

LinearDims dims_of(const KernelShapes& s) {
  LinearDims d;
  if (s.inputs.size() != 2) return d;
  const Shape& x = s.inputs[0];
  const Shape& w = s.inputs[1];
  if (x.rank() == 0 || w.rank() != 2) return d;
  d.kb = x.dim(x.rank() - 1);
  d.n = w.dim(0);
  if (d.kb <= 0 || d.n <= 0 || w.dim(1) != d.kb) return d;
  const std::int64_t elems = static_cast<std::int64_t>(s.output.elem_count());
  if (elems <= 0 || elems % d.n != 0) return d;
  d.m = elems / d.n;
  d.valid = d.m > 0;
  return d;
}

// The row this invocation would use, or null. One place the table is consulted
// at runtime — the gate, the thread plan and the body all come through here,
// keyed identically, so they cannot pick different rows.
//
// Four separate reasons to decline, all of them loud: the device speaks no
// matrix family, the storage format has no operand form, the row exists but
// its layout has never been measured on hardware, or the row is measured but
// this device lacks the capability. None of them converts an operand to fit an
// instruction the device does have.
const math::MatrixCoreRow* row_for(const KernelShapes& s) {
  if (s.device == nullptr || s.intrinsics == nullptr) return nullptr;
  if (s.input_dtypes.size() < 2) return nullptr;
  const std::optional<math::MatrixTarget> target = matrix_target(*s.device);
  if (!target.has_value()) return nullptr;

  struct Key {
    bool named = false;
    math::MatrixElem acc{};
    math::MatrixElem operand{};
  };
  const Key key = with_matrix_operand<Key>(
      s.input_dtypes[1],
      []<class, class, math::MatrixElem A, math::MatrixElem T>() {
        return Key{true, A, T};
      });
  if (!key.named) return nullptr;

  const std::uint32_t caps = device_matrix_caps(*s.device);
  for (const math::MatrixCoreRow& r : math::matrix_core_table()) {
    if (r.target != *target || r.acc != key.acc || r.operand != key.operand) {
      continue;
    }
    if (r.m != kTileM || r.n != kTileN || r.k_step != kTileK) continue;
    if (!r.emittable()) continue;
    if (!math::has_cap(caps, r.cap)) continue;
    if (s.intrinsics->find(r.key).empty()) continue;
    return &r;
  }
  return nullptr;
}

template <class E, class X = kir::f32, class W = kir::f32>
struct MatrixLinearArgs {
  env::In<X, E> x;
  env::In<W, E> w;
  // Unused directly: owns_indexing stores go through the emitter hook, but the
  // binding contract still names the output slot.
  env::Out<kir::f32, E> out;
};

// The one thing this kernel adds to the shared tile: where a finished
// accumulator slot goes. Through the emitter's hook, not into `out`, because
// that is where a fused silu/mul/add runs — on the value still in register.
template <math::MatrixTarget G, math::MatrixElem A, math::MatrixElem T>
struct LinearTile
    : MatrixTile<LinearTile<G, A, T>, G, A, T, kTileM, kTileN, kTileK> {
  std::uint32_t cols = 0;

  void emit_element(env::Emit& e, const kir::Val<kir::u32>& row,
                    const kir::Val<kir::u32>& col,
                    const kir::Val<kir::f32>& v) const {
    e.store(row * cols + col, v);
  }
};

struct MatrixLinearKernel final : KernelPrimitive<MatrixLinearKernel> {
  static constexpr std::string_view kName = "linear.wmma";
  static constexpr std::string_view kEntry = "lse_linear_wmma";
  static constexpr std::string_view kSource = {};

  std::size_t arity() const noexcept override { return 2; }
  bool owns_indexing() const noexcept override { return true; }

  std::string emit_kernel(const KernelShapes& s) const override {
    const LinearDims d = dims_of(s);
    const math::MatrixCoreRow* row = row_for(s);
    if (!d.valid || row == nullptr || s.types.scalar == nullptr ||
        s.intrinsics == nullptr || !s.store) {
      if (std::getenv("LSE_DEBUG_WMMA") != nullptr) {
        std::fprintf(stderr, "[wmma] decline: d=%d row=%d scalar=%d intr=%d store=%d\n",
                     (int)d.valid, (int)(row != nullptr),
                     (int)(s.types.scalar != nullptr),
                     (int)(s.intrinsics != nullptr), (int)(bool)s.store);
      }
      return {};
    }
    return for_live_target<math::MatrixTarget::kRdna3,
                                   math::MatrixTarget::kRdna4>(
        row->target, [&]<math::MatrixTarget G>() -> std::string {
          return with_matrix_operand<std::string>(
              s.input_dtypes[1],
              [&]<class X, class W, math::MatrixElem A, math::MatrixElem T>()
                  -> std::string { return emit_body<G, X, W, A, T>(s, d); });
        });
  }

  // The kernel, in C++. Nothing here names a HIP type, a builtin or a width:
  // types come from s.types, the instruction from s.intrinsics, and every
  // shape from the row.
  template <math::MatrixTarget G, class X, class W, math::MatrixElem A,
            math::MatrixElem T>
  std::string emit_body(const KernelShapes& s, const LinearDims& d) const {
    const auto M = static_cast<std::uint32_t>(d.m);
    const auto N = static_cast<std::uint32_t>(d.n);
    const auto KB = static_cast<std::uint32_t>(d.kb);

    kir::KernelBody k(s.types, *s.intrinsics);
    k.set_store(s.store);
    MatrixLinearArgs<env::Emit, X, W> a;
    if (!env::bind(k, a, s)) {
      if (std::getenv("LSE_DEBUG_WMMA") != nullptr) {
        std::fprintf(stderr, "[wmma] decline: bind failed\n");
      }
      return {};
    }
    env::Emit e{&k};

    LinearTile<G, A, T> tile;
    tile.cols = N;
    tile.run(e, a.x, a.w, M, N, KB, device_load_bytes(s.device));
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

  // A tile is a wave, not an output element, so the launch is sized in waves —
  // in the row's own tile and the row's own wave, which is how a wave64 MFMA
  // row would launch correctly without a second plan.
  static ThreadPlan plan_impl(const KernelShapes& s) {
    ThreadPlan tp;
    const math::MatrixCoreRow* row = row_for(s);
    if (row == nullptr) return tp;  // never selected; nothing to size
    const LinearDims d = dims_of(s);
    const auto lanes = static_cast<std::uint32_t>(row->wave);
    const std::int64_t tiles_n = (d.n + row->n - 1) / row->n;
    const std::int64_t tiles = ((d.m + row->m - 1) / row->m) * tiles_n;

    const std::uint32_t cap =
        s.device != nullptr && s.device->max_threads_per_workgroup >= 256
            ? 256u : 64u;
    const std::uint32_t waves = cap / lanes;
    tp.workgroup_size[0] = waves * lanes;
    tp.workgroup_count[0] = static_cast<std::uint32_t>(
        (tiles + waves - 1) / waves);
    return tp;
  }
};

}  // namespace

const KernelPrimitiveBase* wmma_linear_for(const KernelShapes& s) {
  static const MatrixLinearKernel kKernel;
  if (wmma_forced_off()) return nullptr;
  const math::MatrixCoreRow* row = row_for(s);
  if (row == nullptr) return nullptr;
  const LinearDims d = dims_of(s);
  if (!d.valid) return nullptr;
  // One tile is the smallest unit the instruction can do. A K that is not a
  // whole number of steps is fine — the fill guards zero the tail — but a K
  // below one step would be all tail.
  if (d.kb < row->k_step / row->pack || d.n < row->n) return nullptr;
  return &kKernel;
}

}  // namespace lse::kernels
