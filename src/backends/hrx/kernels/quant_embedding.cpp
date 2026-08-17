// `quant_embedding`: rows of a group-affine table, dequantized in register.
//
// A tied LM head reads the same table twice — as a matrix in quant_linear and
// as rows here — so widening it for the gather would widen it for both, and
// for this checkpoint that is 143 MB of packed planes becoming 1 GB of f32.
// One thread owns one chunk, the smallest run of lanes holding a whole number
// of codes, so every shift inside it is a compile-time constant.
#include <string>

#include "lse/backends/hrx/device_info.hpp"
#include "lse/backends/hrx/kernels/vec_mem.hpp"
#include "lse/graph/kernel_args.hpp"
#include "lse/graph/kernel_env.hpp"
#include "lse/graph/kernel_primitive.hpp"
#include "lse/math.hpp"
#include "lse/quant/group_affine_codec.hpp"

namespace lse::backend::hrx_kernels {

using namespace lse::graph;
namespace math = lse::math;

namespace {

struct EmbedDims {
  quant::GroupAffine spec{};
  std::int64_t dim = 0;     // weights per table row
  std::int64_t lanes = 0;   // packed u32 per table row
  std::int64_t groups = 0;  // scale/bias entries per table row
  std::int64_t slots = 0;   // token ids to gather
  bool valid = false;
};

EmbedDims dims_of(const KernelShapes& s) {
  EmbedDims d;
  if (s.inputs.size() != 4 || s.input_dtypes.size() < 4) return d;
  if (s.input_dtypes[0] != DType::kU32) return d;
  if (s.input_dtypes[1] != s.input_dtypes[2]) return d;
  auto spec = quant::GroupAffine::make(s.iattrs[0], s.iattrs[1]);
  if (!spec.ok()) return d;
  d.spec = *spec;

  const Shape& packed = s.inputs[0];
  const Shape& scales = s.inputs[1];
  const Shape& biases = s.inputs[2];
  if (packed.rank() != 2 || scales.rank() != 2 || biases.rank() != 2) return d;
  if (s.output.rank() == 0) return d;
  d.lanes = packed.dim(1);
  d.groups = scales.dim(1);
  d.dim = s.output.dim(s.output.rank() - 1);
  if (scales.dim(0) != packed.dim(0) || biases.dim(0) != packed.dim(0)) return d;
  if (biases.dim(1) != d.groups) return d;
  if (d.lanes * 32 != d.dim * d.spec.bits) return d;
  if (d.groups * d.spec.group_size != d.dim) return d;

  const auto elems = static_cast<std::int64_t>(s.output.elem_count());
  if (elems <= 0 || d.dim <= 0 || elems % d.dim != 0) return d;
  d.slots = elems / d.dim;
  d.valid = d.slots > 0;
  return d;
}

bool device_fits(const KernelShapes& s) {
  if (s.device == nullptr || s.intrinsics == nullptr) return false;
  for (std::string_view sym : quant::kGroupAffineSymbols) {
    if (s.intrinsics->find(sym).empty()) return false;
  }
  return true;
}

template <class S>
struct QuantEmbedArgs {
  env::In<std::uint32_t, env::Emit> packed;
  env::In<S, env::Emit> scales;
  env::In<S, env::Emit> biases;
  env::In<kir::f32, env::Emit> ids;
  env::Out<kir::f32, env::Emit> out;
};

template <class S>
std::string emit_body(const KernelShapes& s, const EmbedDims& d) {
  const auto lanes = static_cast<std::uint32_t>(d.lanes);
  const auto groups = static_cast<std::uint32_t>(d.groups);
  const auto dim = static_cast<std::uint32_t>(d.dim);
  const auto vals = static_cast<std::uint32_t>(d.spec.values_per_chunk());
  const auto words = static_cast<std::uint32_t>(d.spec.words_per_chunk());
  const std::uint32_t nchunks = dim / vals;
  const std::uint32_t chunks_per_group =
      static_cast<std::uint32_t>(d.spec.group_size) / vals;
  const auto total = static_cast<std::uint32_t>(d.slots) * nchunks;

  kir::KernelBody kb(s.types, *s.intrinsics, workgroup_lds_bytes(s.device));
  kb.set_store(s.store);
  QuantEmbedArgs<S> a;
  if (!env::bind(kb, a, s)) return {};
  env::Emit e{&kb};

  const auto i = e.let(e.thread_id());
  if (auto in_range = e.when(i < total)) {
    const auto slot = e.let(i / nchunks);
    const auto chunk = e.let(i % nchunks);
    const auto id = e.let(math::cast<kir::u32>(a.ids[slot]));
    const auto group = e.let(id * groups + chunk / chunks_per_group);
    const auto scale = e.let(math::widen(a.scales[group]));
    const auto bias = e.let(math::widen(a.biases[group]));
    const auto base = e.let(slot * dim + chunk * vals);
    quant::dequant_chunk(e, a.packed, d.spec, e.let(id * lanes + chunk * words),
                         scale, bias,
                         [&](int c, const kir::Val<kir::f32>& w) {
                           e.store(base + static_cast<std::uint32_t>(c), w);
                         });
  }
  if (!kb.lds().ok()) return {};
  return kb.str();
}

}  // namespace

struct QuantEmbeddingKernel final : KernelPrimitive<QuantEmbeddingKernel> {
  static constexpr std::string_view kName = "quant_embedding";
  static constexpr std::string_view kEntry = "lse_quant_embedding";
  static constexpr std::string_view kSource = {};

  std::size_t arity() const noexcept override { return 4; }
  bool owns_indexing() const noexcept override { return true; }

  std::string emit_kernel(const KernelShapes& s) const override {
    const EmbedDims d = dims_of(s);
    if (!d.valid || !device_fits(s) || s.types.scalar == nullptr || !s.store) {
      return {};
    }
    return with_elem(s.input_dtypes[1], [&]<class S>() -> std::string {
      return emit_body<S>(s, d);
    });
  }

  // The row width is lanes * 32 / bits and bits is a node attribute, not a
  // shape, so the shapes alone do not determine the output. graph::embedding
  // builds the node with the width it read from the weight's own geometry;
  // this path exists for the generic custom-op builder, which has no bits.
  Result<Shape> infer_shape(std::span<const Shape>) const override {
    return LSE_ERROR(kInvalidArgument,
                     "quant_embedding's row width depends on its bit width, "
                     "which is a node attribute rather than an input shape");
  }

  DType infer_dtype(std::span<const DType>) const override {
    return DType::kF32;
  }

  static ThreadPlan plan_impl(const KernelShapes& s) {
    const EmbedDims d = dims_of(s);
    const std::uint32_t threads =
        s.device && s.device->max_threads_per_workgroup >= 256 ? 256u : 64u;
    const auto vals = d.valid
                          ? static_cast<std::uint32_t>(d.spec.values_per_chunk())
                          : 1u;
    const auto chunks =
        static_cast<std::uint32_t>(s.output.elem_count()) / (vals != 0 ? vals : 1u);
    ThreadPlan tp;
    tp.workgroup_size[0] = threads;
    tp.workgroup_count[0] =
        chunks == 0 ? 1u : (chunks + threads - 1) / threads;
    return tp;
  }
};

LSE_REGISTER_PRIMITIVE(QuantEmbeddingKernel);

}  // namespace lse::backend::hrx_kernels
