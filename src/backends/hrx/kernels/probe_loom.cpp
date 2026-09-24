#include "lse/backends/hrx/probe_emit.hpp"
#include "lse/backends/hrx/probe_matrix.hpp"

#include "lse/backends/hrx/loomc/loom_emitter.hpp"
#include "lse/graph/graph.hpp"
#include "lse/graph/kernel_primitive.hpp"
#include "lse/probe/probe_kernels.hpp"

namespace lse::backend::hrx_kernels {
namespace {
using namespace graph;

struct StreamProbe final : KernelPrimitive<StreamProbe> {
  static constexpr std::string_view kName = "probe.stream";
  static constexpr std::string_view kEntry = "lse_probe_stream";
  static constexpr std::string_view kSource = {};
  std::size_t arity() const noexcept override { return 1; }
  bool owns_indexing() const noexcept override { return true; }
  Result<Shape> infer_shape(std::span<const Shape> shapes) const override {
    return shapes.empty() ? Shape{} : shapes[0];
  }
  DType infer_dtype(std::span<const DType>) const override { return DType::kF32; }
  std::string emit_kernel(const KernelShapes& s) const override {
    if (s.inputs.size() != 1 || s.iattrs.size() < 2 || !s.intrinsics) return {};
    kir::KernelBody k(s.types, *s.intrinsics);
    k.set_store(s.store);
    probe::StreamArgs<env::Emit> args;
    if (!env::bind(k, args, s)) return {};
    env::Emit e{&k};
    probe::stream_read<env::Emit, true>(e, args,
        static_cast<std::uint32_t>(s.inputs[0].elem_count()),
        static_cast<std::uint32_t>(s.output.elem_count()),
        static_cast<std::uint32_t>(s.iattrs[1]));
    return k.str();
  }
  static ThreadPlan plan_impl(const KernelShapes& s) {
    ThreadPlan p;
    p.workgroup_size[0] = static_cast<std::uint32_t>(s.iattrs[0]);
    p.workgroup_count[0] = static_cast<std::uint32_t>(s.output.elem_count()) /
                            p.workgroup_size[0];
    return p;
  }
};

struct TouchProbe final : KernelPrimitive<TouchProbe> {
  static constexpr std::string_view kName = "probe.touch";
  static constexpr std::string_view kEntry = "lse_probe_touch";
  static constexpr std::string_view kSource = {};
  std::size_t arity() const noexcept override { return 0; }
  bool owns_indexing() const noexcept override { return true; }
  Result<Shape> infer_shape(std::span<const Shape>) const override { return Shape{1}; }
  DType infer_dtype(std::span<const DType>) const override { return DType::kF32; }
  std::string emit_kernel(const KernelShapes& s) const override {
    if (!s.intrinsics) return {};
    kir::KernelBody k(s.types, *s.intrinsics);
    k.set_store(s.store);
    probe::TouchArgs<env::Emit> args;
    if (!env::bind(k, args, s)) return {};
    env::Emit e{&k};
    probe::touch_one<env::Emit, true>(e, args);
    return k.str();
  }
  static ThreadPlan plan_impl(const KernelShapes& s) {
    ThreadPlan p;
    p.workgroup_size[0] = s.device->wavefront_size;
    p.workgroup_count[0] = 1;
    return p;
  }
};

struct MatrixRateProbe final : KernelPrimitive<MatrixRateProbe> {
  static constexpr std::string_view kName = "probe.matrix_rate";
  static constexpr std::string_view kEntry = "lse_probe_matrix_rate";
  static constexpr std::string_view kSource = {};
  std::size_t arity() const noexcept override { return 2; }
  bool owns_indexing() const noexcept override { return true; }
  Result<Shape> infer_shape(std::span<const Shape>) const override {
    return Shape{probe_matrix::kRateM, probe_matrix::kRateN};
  }
  DType infer_dtype(std::span<const DType>) const override { return DType::kF32; }
  std::string emit_kernel(const KernelShapes& s) const override {
    if (!s.intrinsics || s.iattrs.size() < 2 || !s.device) return {};
    const auto rows = math::matrix_core_table();
    const auto index = static_cast<std::size_t>(s.iattrs[0]);
    if (index >= rows.size()) return {};
    return probe_matrix::rate_body(*s.device, rows[index],
        static_cast<DType>(s.iattrs[1]), s.types, *s.intrinsics, &s);
  }
  static ThreadPlan plan_impl(const KernelShapes& s) {
    ThreadPlan p;
    const auto dims = probe_matrix::rate_dims(*s.device,
        math::matrix_core_table()[static_cast<std::size_t>(s.iattrs[0])]);
    p.workgroup_size[0] = dims.workgroup_size[0];
    p.workgroup_count[0] = dims.workgroup_count[0];
    return p;
  }
};

NodePtr node(Shape shape, const Primitive* primitive = nullptr) {
  auto n = std::make_shared<Node>();
  n->set_kind(primitive ? OpKind::kCustom : OpKind::kBuffer);
  n->shape = std::move(shape);
  n->dtype = DType::kF32;
  n->prim = primitive;
  return n;
}
Result<EmittedKernel> emit(const NodePtr& output, const DeviceInfo& device) {
  FusionGroup group;
  group.nodes = {output};
  group.inputs = output->inputs;
  group.outputs = {output};
  group.anchor = OpKind::kCustom;
  group.anchor_class = FusionClass::kBarrier;
  const LoomEmitter emitter;
  return emitter.emit(group, device);
}
}  // namespace

Result<graph::EmittedKernel> emit_loom_stream_probe(
    const DeviceInfo& device, std::uint32_t elements, std::uint32_t threads,
    std::uint32_t load_bytes) {
  if (device.max_threads_per_workgroup < 256 || device.wavefront_size == 0 ||
      elements == 0 || threads == 0 || threads % 256 != 0 ||
      (load_bytes != 4 && load_bytes != 8 && load_bytes != 16) ||
      static_cast<std::uint64_t>(threads) * (load_bytes / 4) > elements ||
      elements % (static_cast<std::uint64_t>(threads) * (load_bytes / 4)) != 0) {
    return LSE_ERROR(kInvalidArgument, "unsupported streaming-probe geometry");
  }
  static const StreamProbe primitive;
  auto out = node(Shape{threads}, &primitive);
  out->inputs = {node(Shape{elements})};
  out->iattrs = {256, static_cast<std::int32_t>(load_bytes)};
  return emit(out, device);
}

Result<graph::EmittedKernel> emit_loom_touch_probe(const DeviceInfo& device) {
  if (device.wavefront_size == 0 ||
      device.wavefront_size > device.max_threads_per_workgroup) {
    return LSE_ERROR(kInvalidArgument, "unknown touch-probe wavefront geometry");
  }
  static const TouchProbe primitive;
  return emit(node(Shape{1}, &primitive), device);
}
Result<graph::EmittedKernel> emit_loom_matrix_rate_probe(
    const DeviceInfo& device, const math::MatrixCoreRow& requested, DType storage) {
  using namespace probe_matrix;
  const auto target = kernels::matrix_target(device);
  const auto table = math::matrix_core_table();
  std::size_t index = table.size();
  for (std::size_t i = 0; i < table.size(); ++i) {
    if (table[i].key == requested.key && table[i].acc == requested.acc &&
        table[i].operand == requested.operand) { index = i; break; }
  }
  if (index == table.size() || !target || table[index].target != *target ||
      !table[index].emittable() || table[index].m != kTileM ||
      table[index].n != kTileN || table[index].k_step != kTileK ||
      table[index].wave != device.wavefront_size ||
      device.max_threads_per_workgroup < 64 ||
      !math::has_cap(kernels::device_matrix_caps(device), table[index].cap) ||
      loom_sources().find(table[index].key).empty()) {
    return LSE_ERROR(kUnimplemented, "no supported Loom matrix probe row");
  }
  const auto& row = table[index];
  const auto elements = static_cast<std::int64_t>(kRateK / row.pack);
  static const MatrixRateProbe primitive;
  auto out = node(Shape{kRateM, kRateN}, &primitive);
  bool bound = kernels::with_matrix_operand<bool>(storage,
      [&]<class X, class W, math::MatrixElem A, math::MatrixElem T>() {
        if (A != row.acc || T != row.operand) return false;
        auto x = node(Shape{kRateM, elements}); x->dtype = env::elem_dtype<X>::value;
        auto w = node(Shape{kRateN, elements}); w->dtype = env::elem_dtype<W>::value;
        out->inputs = {x,w}; return true;
      });
  if (!bound) return LSE_ERROR(kUnimplemented, "matrix probe has no matching storage arm");
  out->iattrs = {static_cast<std::int32_t>(index), static_cast<std::int32_t>(storage)};
  return emit(out, device);
}
}  // namespace lse::backend::hrx_kernels
