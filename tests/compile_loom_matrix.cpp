// Offline compilation of the shared matrix tile; never opens a device.
#include "lse/backends/hrx/arch_database.hpp"
#include "lse/backends/hrx/loomc/loom_emitter.hpp"
#include "lse/backends/hrx/loomc/loomc_compiler.hpp"
#include "lse/graph/graph.hpp"
#include "lse/graph/kernel_args.hpp"
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
template <math::MatrixElem T, math::MatrixElem C>
struct Tile final : kernels::MatrixTile<Tile<T, C>, math::MatrixTarget::kRdna4,
                                        C, T, 16, 16, 16> {
  void emit_element(env::Emit &e, const kir::Val<kir::u32> &row,
                    const kir::Val<kir::u32> &col,
                    const kir::Val<kir::f32> &v) const {
    e.store(row * 16u + col, v);
  }
};
template <math::MatrixElem T, math::MatrixElem C>
struct Primitive final : graph::KernelPrimitive<Primitive<T, C>> {
  static constexpr std::string_view kName = "matrix_parity",
                                    kEntry = "matrix_parity", kSource = "";
  using Op = math::op::Mma<math::MatrixTarget::kRdna4, C, T, 16, 16, 16>;
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
    Tile<T, C> tile;
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
template <math::MatrixElem T, math::MatrixElem C>
bool compile(backend::DeviceInfo &info, backend::LoomcCompiler &compiler,
             const std::string &path, const char *label) {
  static Primitive<T, C> primitive;
  constexpr auto row = Primitive<T, C>::row;
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
                env::elem_dtype<typename Primitive<T, C>::X>::value);
  auto w = leaf(Shape{16, 32 / row.pack},
                env::elem_dtype<typename Primitive<T, C>::W>::value);
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
  auto obj = compiler.compile(emitted->source, "gfx1201");
  if (!obj.ok()) {
    std::fprintf(stderr, "%s compile: %s\n", label,
                 obj.status().to_string().c_str());
    return false;
  }
  std::ofstream file(path + "/" + label + ".hsaco", std::ios::binary);
  file.write(reinterpret_cast<const char *>(obj->code.data()),
             obj->code.size());
  std::printf("PASS shared MatrixTile %s bytes=%zu\n", label, obj->code.size());
  return true;
}
int main(int argc, char **argv) {
  if (argc != 2)
    return 2;
  backend::DeviceInfo info;
  info.arch = "gfx1201";
  info.compute_units = 64;
  info.wavefront_size = 32;
  info.max_threads_per_workgroup = 1024;
  info.lds_bytes_per_workgroup = 65536;
  backend::AmdDeviceInfo amd;
  backend::apply_arch_defaults(info, amd);
  info.extension_id = backend::AmdDeviceInfo::kExtensionId;
  info.extension = &amd;
  backend::LoomcCompiler compiler;
  bool ok = true;
  ok = compile<math::MatrixElem::kI8, math::MatrixElem::kI32>(info, compiler,
                                                              argv[1], "i8") &&
       ok;
  ok = compile<math::MatrixElem::kSU8, math::MatrixElem::kI32>(
           info, compiler, argv[1], "su8") &&
       ok;
  ok = compile<math::MatrixElem::kFp8, math::MatrixElem::kF32>(
           info, compiler, argv[1], "fp8") &&
       ok;
  ok = compile<math::MatrixElem::kBf8, math::MatrixElem::kF32>(
           info, compiler, argv[1], "bf8") &&
       ok;
  ok = compile<math::MatrixElem::kF16, math::MatrixElem::kF32>(
           info, compiler, argv[1], "f16") &&
       ok;
  ok = compile<math::MatrixElem::kBF16, math::MatrixElem::kF32>(
           info, compiler, argv[1], "bf16") &&
       ok;
  return ok ? 0 : 1;
}
