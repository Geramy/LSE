// Offline compilation of the shared matrix tile; never opens a device.
//
// compile_loom_matrix <dir> [arch]. The arch (gfx1201 when omitted) picks the
// matrix generation from the device tables -- gfx12 parts take the RDNA4 rows,
// gfx11 parts the RDNA3/3.5 ones -- and every emittable single-instruction
// wave32 row of that generation is emitted through the Loom path and compiled
// for that arch. The loomc build must include the arch's target.
#include "lse/backends/hrx/arch_database.hpp"
#include "lse/backends/hrx/loomc/loom_emitter.hpp"
#include "lse/backends/hrx/loomc/loomc_compiler.hpp"
#include "lse/graph/graph.hpp"
#include "lse/graph/kernel_args.hpp"
#include "lse/dispatch/quant.hpp"
#include "lse/graph/kernel_primitive.hpp"
#include "lse/kernels/wmma.hpp"
#include <cstdio>
#include <fstream>
using namespace lse;
namespace kir = graph::kir;
namespace env = graph::env;
template <class X, class W> struct Args {
  env::In<X, env::Emit> x;
  env::In<W, env::Emit> w;
};
template <math::MatrixTarget G, math::MatrixElem T, math::MatrixElem C>
struct Tile final : kernels::MatrixTile<Tile<G, T, C>, G, C, T, 16, 16, 16> {
  void emit_element(env::Emit &e, const kir::Val<kir::u32> &row,
                    const kir::Val<kir::u32> &col,
                    const kir::Val<kir::f32> &v) const {
    e.store(row * 16u + col, v);
  }
};
template <math::MatrixTarget G, math::MatrixElem T, math::MatrixElem C>
struct Primitive final : graph::KernelPrimitive<Primitive<G, T, C>> {
  static constexpr std::string_view kName = "matrix_parity",
                                    kEntry = "matrix_parity", kSource = "";
  using Op = math::op::Mma<G, C, T, 16, 16, 16>;
  static constexpr auto row = Op::kRow;
  using X = math::matrix_scalar_t<row.a_elem>;
  using W = math::matrix_scalar_t<row.b_elem>;
  size_t arity() const noexcept override { return 2; }
  bool owns_indexing() const noexcept override { return true; }
  Result<Shape> infer_shape(std::span<const Shape>) const override {
    return Shape{16, 16};
  }
  DType infer_dtype(std::span<const DType>) const override {
    return DType::kF32;
  }
  std::string emit_kernel(const graph::KernelShapes &s) const override {
    kir::KernelBody k(s.types, *s.intrinsics);
    k.set_store(s.store);
    Args<X, W> a;
    if (!env::bind(k, a, s))
      return {};
    env::Emit e{&k};
    Tile<G, T, C> tile;
    tile.run(e, a.x, a.w, 16, 16, 32 / row.pack, 16);
    return k.str();
  }
  static graph::ThreadPlan plan_impl(const graph::KernelShapes &) {
    graph::ThreadPlan p;
    p.workgroup_size[0] = 32;
    p.workgroup_count[0] = 1;
    return p;
  }
};
template <math::MatrixTarget G, math::MatrixElem T, math::MatrixElem C>
bool compile(backend::DeviceInfo &info, backend::LoomcCompiler &compiler,
             const std::string &path, const char *label) {
  static Primitive<G, T, C> primitive;
  constexpr auto row = Primitive<G, T, C>::row;
  constexpr auto geometry = kernels::geometry_of(row);
  std::ofstream layout(path + "/" + label + ".layout");
  layout << row.wave << " " << row.pack << " " << row.a_len << " " << row.b_len
         << " " << row.c_len << "\n";
  for (int lane = 0; lane < row.wave; ++lane) {
    for (int element = 0; element < row.a_len * row.pack; ++element)
      layout << "lhs " << lane << " " << element << " " << lane % row.n << " "
             << (lane / row.n) * geometry.lane_k + element << "\n";
    for (int element = 0; element < row.b_len * row.pack; ++element)
      layout << "rhs " << lane << " " << element << " " << lane % row.n << " "
             << (lane / row.n) * geometry.lane_k + element << "\n";
    for (int element = 0; element < row.c_len; ++element)
      layout << "result " << lane << " " << element << " "
             << element * geometry.slot_step +
                    (lane / row.n) * geometry.half_rows
             << " " << lane % row.n << "\n";
  }
  auto leaf = [](Shape shape, DType type) {
    auto n = std::make_shared<graph::Node>();
    n->set_kind(graph::OpKind::kBuffer);
    n->shape = shape;
    n->dtype = type;
    return n;
  };
  auto x = leaf(Shape{16, 32 / row.pack},
                env::elem_dtype<typename Primitive<G, T, C>::X>::value);
  auto w = leaf(Shape{16, 32 / row.pack},
                env::elem_dtype<typename Primitive<G, T, C>::W>::value);
  auto out = leaf(Shape{16, 16}, DType::kF32);
  out->set_kind(graph::OpKind::kCustom);
  out->prim = &primitive;
  out->inputs = {x, w};
  graph::FusionGroup group;
  group.nodes = {out};
  group.inputs = {x, w};
  group.outputs = {out};
  group.anchor = graph::OpKind::kCustom;
  group.anchor_class = graph::FusionClass::kBarrier;
  backend::LoomEmitter emitter;
  auto emitted = emitter.emit(group, info);
  if (!emitted.ok()) {
    std::fprintf(stderr, "%s emit: %s\n", label,
                 emitted.status().to_string().c_str());
    return false;
  }
  std::ofstream(path + "/" + label + ".loom") << emitted->source;
  auto obj = compiler.compile(emitted->source, info.arch);
  if (!obj.ok()) {
    std::fprintf(stderr, "%s compile: %s\n", label,
                 obj.status().to_string().c_str());
    return false;
  }
  std::ofstream file(path + "/" + label + ".hsaco", std::ios::binary);
  file.write(reinterpret_cast<const char *>(obj->code.data()),
             obj->code.size());
  std::printf("PASS shared MatrixTile %s %s %.*s bytes=%zu\n",
              info.arch.c_str(), label, static_cast<int>(row.key.size()),
              row.key.data(), obj->code.size());
  return true;
}
// The tiled 4-bit prefill GEMM at one prompt length, the way the graph
// builds it for this device: the K slices it would cut, the tile the shape
// selects, and the matrix row the device's generation spells.
bool compile_q4_gemm(backend::DeviceInfo &info, backend::LoomcCompiler &compiler,
                     const std::string &path, std::int64_t m, std::int64_t n,
                     std::int64_t k) {
  auto leaf = [](Shape shape, DType type) {
    auto node = std::make_shared<graph::Node>();
    node->set_kind(graph::OpKind::kBuffer);
    node->shape = shape;
    node->dtype = type;
    return node;
  };
  constexpr std::int64_t kGroup = 64;
  auto x = leaf(Shape{m, k}, DType::kF32);
  auto packed = leaf(Shape{n, k / 8}, DType::kU32);
  auto scales = leaf(Shape{n, k / kGroup}, DType::kBF16);
  auto biases = leaf(Shape{n, k / kGroup}, DType::kBF16);
  auto panel = leaf(Shape{m, k}, DType::kF16);
  const std::uint32_t slices = dispatch::q4_gemm_slices(
      static_cast<std::uint64_t>(m), static_cast<std::uint64_t>(n),
      static_cast<std::uint64_t>(k), info.compute_units,
      dispatch::q4_gemm_fragment_registers(dispatch::q4_gemm_row(info)));
  auto gemm = std::make_shared<graph::Node>();
  gemm->set_kind(graph::OpKind::kCustom);
  gemm->dtype = DType::kF32;
  gemm->iattrs = {4, static_cast<std::int32_t>(kGroup),
                  static_cast<std::int32_t>(slices), 0};
  gemm->shape = slices > 1 ? Shape{static_cast<std::int64_t>(slices), m, n}
                           : Shape{m, n};
  gemm->prim = graph::find_primitive(slices > 1
                                         ? "quant_linear.q4_gemm_f16.slices.v1"
                                         : "quant_linear.q4_gemm_f16.v1");
  gemm->inputs = {x, packed, scales, biases, panel};
  if (gemm->prim == nullptr) return false;
  graph::FusionGroup group;
  group.nodes = {gemm};
  group.inputs = {x, packed, scales, biases, panel};
  group.outputs = {gemm};
  group.anchor = graph::OpKind::kCustom;
  group.anchor_class = graph::FusionClass::kBarrier;
  const std::string label = "q4_gemm_m" + std::to_string(m) + "_n" +
                            std::to_string(n) + "_k" + std::to_string(k);
  backend::LoomEmitter emitter;
  auto emitted = emitter.emit(group, info);
  if (!emitted.ok()) {
    std::fprintf(stderr, "%s %s emit: %s\n", info.arch.c_str(), label.c_str(),
                 emitted.status().to_string().c_str());
    return false;
  }
  std::ofstream(path + "/" + label + ".loom") << emitted->source;
  auto obj = compiler.compile(emitted->source, info.arch);
  if (!obj.ok()) {
    std::fprintf(stderr, "%s %s compile: %s\n", info.arch.c_str(),
                 label.c_str(), obj.status().to_string().c_str());
    return false;
  }
  std::ofstream file(path + "/" + label + ".hsaco", std::ios::binary);
  file.write(reinterpret_cast<const char *>(obj->code.data()),
             obj->code.size());
  std::printf("PASS q4 prefill GEMM %s %s slices=%u bytes=%zu\n",
              info.arch.c_str(), label.c_str(), slices, obj->code.size());
  return true;
}

// Every emittable single-instruction row of one generation, by the operand
// format it takes. A row the table does not hold is skipped at compile time;
// a row it holds but has not measured is skipped too, since it never emits.
template <math::MatrixTarget G>
bool compile_target(backend::DeviceInfo &info, backend::LoomcCompiler &compiler,
                    const std::string &path) {
  bool ok = true;
  std::size_t compiled = 0;
  const auto one = [&]<math::MatrixElem T, math::MatrixElem C>(
                       const char *label) {
    if constexpr (math::has_matrix_core_row(G, C, T, 16, 16, 16)) {
      constexpr auto row = math::matrix_core_row(G, C, T, 16, 16, 16);
      if constexpr (row.emittable() && row.chained == 1 && row.wave == 32) {
        ok = compile<G, T, C>(info, compiler, path, label) && ok;
        ++compiled;
      }
    }
  };
  using E = math::MatrixElem;
  one.template operator()<E::kI8, E::kI32>("i8");
  one.template operator()<E::kSU8, E::kI32>("su8");
  one.template operator()<E::kI4, E::kI32>("i4");
  one.template operator()<E::kFp8, E::kF32>("fp8");
  one.template operator()<E::kBf8, E::kF32>("bf8");
  one.template operator()<E::kF16, E::kF32>("f16");
  one.template operator()<E::kBF16, E::kF32>("bf16");
  return ok && compiled != 0;
}

int main(int argc, char **argv) {
  if (argc != 2 && argc != 3)
    return 2;
  backend::DeviceInfo info;
  info.arch = argc == 3 ? argv[2] : "gfx1201";
  info.wavefront_size = 32;
  info.max_threads_per_workgroup = 1024;
  info.lds_bytes_per_workgroup = 65536;
  backend::AmdDeviceInfo amd;
  backend::apply_arch_defaults(info, amd);
  info.extension_id = backend::AmdDeviceInfo::kExtensionId;
  info.extension = &amd;
  const auto target = kernels::matrix_target(info);
  if (!target) {
    std::fprintf(stderr, "%s has no wave32 matrix generation\n",
                 info.arch.c_str());
    return 2;
  }
  backend::LoomcCompiler compiler;
  bool ok = kernels::with_matrix_target<bool>(
      *target, [&]<math::MatrixTarget G>() {
        return compile_target<G>(info, compiler, argv[1]);
      });
  // Each tile family the prefill GEMM selects, on both projection shapes.
  if (dispatch::q4_gemm_device(info))
    for (const std::int64_t m : {16, 137, 512, 1024})
      for (const auto [n, k] : {std::pair<std::int64_t, std::int64_t>{17408, 5120},
                                {5120, 17408}})
        ok = compile_q4_gemm(info, compiler, argv[1], m, n, k) && ok;
  return ok ? 0 : 1;
}
