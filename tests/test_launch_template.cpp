// Launch templates (graph/launch_template.hpp), host only: launches are
// described by the real Loom emitter and recorded, never executed.
#include <cstdlib>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "harness.hpp"
#include "lse/backends/cpu/cpu_backend.hpp"
#include "lse/backends/hrx/arch_database.hpp"
#include "lse/backends/hrx/device_info.hpp"
#include "lse/backends/hrx/loomc/loom_emitter.hpp"
#include "lse/graph/codegen.hpp"
#include "lse/graph/graph.hpp"
#include "lse/graph/launch_template.hpp"
#include "lse/graph/ops.hpp"
#include "lse/graph/primitive.hpp"
#include "lse/graph/program.hpp"

using namespace lse;
using namespace lse::graph;

namespace {

struct RecordingEmitter final : IKernelEmitter {
  mutable backend::LoomEmitter real;
  Result<EmittedKernel> emit(const FusionGroup& g, const backend::DeviceInfo& d) const override {
    return real.emit(g, d);
  }
  Result<EmittedKernel> emit_launch(const FusionGroup& g,
                                    const backend::DeviceInfo& d) const override {
    return real.emit_launch(g, d);
  }
  bool stores_epilogue_output(const FusionGroup& g, const Node& o,
                              const backend::DeviceInfo& d) const override {
    return real.stores_epilogue_output(g, o, d);
  }
  bool joins_reduction(const FusionGroup& g, const NodePtr& next,
                       const backend::DeviceInfo& d) const override {
    return real.joins_reduction(g, next, d);
  }
  Dialect dialect() const noexcept override { return Dialect::kLoom; }
  std::string_view prelude() const noexcept override { return {}; }
  DialectSourceTable sources() const noexcept override { return real.sources(); }
};
struct RecordingCompiler final : IKernelCompiler {
  Result<CompiledKernel> compile(std::string_view, std::string_view) const override {
    CompiledKernel k;
    k.code.push_back(std::byte{1});
    return k;
  }
  bool available() const override { return true; }
  std::string identity() const override { return "launch-template-test-v1"; }
};
// Every launch's kernel name and grid, in order.
struct RecordingDevice : backend::CpuBackend {
  std::vector<std::string> launched;
  mutable RecordingEmitter emitter;
  mutable RecordingCompiler compiler;
  mutable KernelToolchain chain{Dialect::kLoom, &emitter, &compiler};
  backend::DeviceInfo info;
  backend::AmdDeviceInfo amd;
  RecordingDevice() {
    info.arch = "gfx1201";
    info.compute_units = 64;
    info.wavefront_size = 32;
    info.max_threads_per_workgroup = 1024;
    info.lds_bytes_per_workgroup = 65536;
    backend::apply_arch_defaults(info, amd);
    info.extension_id = backend::AmdDeviceInfo::kExtensionId;
    info.extension = &amd;
  }
  const backend::DeviceInfo& device_info() const noexcept { return info; }
  std::span<const KernelToolchain> toolchains() const noexcept { return {&chain, 1}; }
  Result<backend::KernelHandle> load_executable(std::string_view name,
                                                std::span<const std::byte>) {
    return backend::KernelHandle{1, 0, std::string(name)};
  }
  Result<std::vector<backend::KernelHandle>> load_executables(
      std::span<const std::string> names, std::span<const std::byte>) {
    std::vector<backend::KernelHandle> handles;
    for (const std::string& name : names) handles.push_back(backend::KernelHandle{1, 0, name});
    return handles;
  }
  Status launch(const backend::KernelHandle& k, const backend::LaunchDims& dims,
                const backend::DispatchArgs&, const backend::DispatchTarget&) {
    launched.push_back(k.name + " " + std::to_string(dims.workgroup_count[0]) + "x" +
                       std::to_string(dims.workgroup_size[0]));
    return OkStatus();
  }
};

void check(Status s) {
  if (!s.ok()) throw std::runtime_error(s.to_string());
}

// A layer as a pass builds it: a norm, two projections and their gated
// product, the down projection and the residual, then a view and an
// elementwise op on it.
struct Layer {
  std::vector<Array> leaves;
  Array out;
};
Layer layer(backend::BackendAdapter<RecordingDevice>& be, std::int64_t rows,
            float eps = 1e-6f) {
  Layer l;
  const auto leaf = [&](Shape sh, DType dt = DType::kF32) {
    auto b = be.allocate(sh.elem_count() * 4, backend::MemoryClass::kDevice,
                         backend::kDefaultStream);
    if (!b.ok()) throw std::runtime_error(b.status().to_string());
    l.leaves.push_back(Array::from_buffer(std::move(*b), sh, dt));
    return l.leaves.back();
  };
  const std::int64_t d = 256, f = 512;
  auto x = leaf({1, rows, d});
  auto norm = rms_norm(x, leaf({d}), eps);
  auto gate = quant_linear(norm, leaf({f, d / 8}, DType::kU32), leaf({f, d / 64}, DType::kBF16),
                           leaf({f, d / 64}, DType::kBF16), 4, 64);
  auto up = quant_linear(x, leaf({f, d / 8}, DType::kU32), leaf({f, d / 64}, DType::kBF16),
                         leaf({f, d / 64}, DType::kBF16), 4, 64);
  auto act = silu(gate) * up;
  auto down = quant_linear(act, leaf({d, f / 8}, DType::kU32), leaf({d, f / 64}, DType::kBF16),
                           leaf({d, f / 64}, DType::kBF16), 4, 64);
  auto head = slice(down + x, -1, 0, 128);
  l.out = sigmoid(head);
  return l;
}

struct AutotuneOff {
  AutotuneOff() { setenv("LSE_AUTOTUNE", "off", 1); }
} autotune_off;

}  // namespace

// A request rebuilds its pass graph over new state; the second graph takes
// the first one's template, and the check mode splits, plans and describes it
// afresh beside the template and fails on any difference. The launches equal
// those of a scheduler with no templates at all.
LSE_TEST(a_template_on_a_fresh_graph_equals_a_fresh_split_and_emit) {
  for (const std::int64_t rows : {17, 64, 304}) {
    backend::BackendAdapter<RecordingDevice> be;
    check(be.init(0));
    Scheduler scheduler(be);
    scheduler.set_dialect(Dialect::kLoom);
    scheduler.set_template_check(true);
    std::vector<std::vector<std::string>> runs;
    for (int request = 0; request < 3; ++request) {
      Layer l = layer(be, rows);
      Program program;
      be.impl().launched.clear();
      const NodePtr roots[] = {l.out.node()};
      check(scheduler.eval(roots, false, &program));
      LSE_EXPECT_EQ(scheduler.last_trace().host_groups, 0u);
      runs.push_back(be.impl().launched);
    }
    const auto stats = scheduler.template_stats();
    LSE_EXPECT_EQ(stats.recorded, 1u);
    LSE_EXPECT_EQ(stats.applied, 2u);
    LSE_EXPECT(!runs[0].empty());
    LSE_EXPECT(runs[1] == runs[0]);
    LSE_EXPECT(runs[2] == runs[0]);

    backend::BackendAdapter<RecordingDevice> plain_be;
    check(plain_be.init(0));
    Scheduler plain(plain_be);
    plain.set_dialect(Dialect::kLoom);
    plain.set_launch_templates(false);
    Layer l = layer(plain_be, rows);
    Program program;
    const NodePtr roots[] = {l.out.node()};
    check(plain.eval(roots, false, &program));
    LSE_EXPECT(plain_be.impl().launched == runs[0]);
    LSE_EXPECT_EQ(plain.template_stats().applied, 0u);
  }
}

// A server prepares every pass shape at load; the first eval of each takes
// the template its preparation recorded.
LSE_TEST(a_prepared_shape_takes_its_template_at_the_first_eval) {
  backend::BackendAdapter<RecordingDevice> be;
  check(be.init(0));
  Scheduler scheduler(be);
  scheduler.set_dialect(Dialect::kLoom);
  scheduler.set_template_check(true);
  {
    Layer l = layer(be, 304);
    const NodePtr roots[] = {l.out.node()};
    check(scheduler.prepare(roots));
  }
  LSE_EXPECT_EQ(scheduler.template_stats().recorded, 1u);
  Layer l = layer(be, 304);
  Program program;
  const NodePtr roots[] = {l.out.node()};
  check(scheduler.eval(roots, false, &program));
  LSE_EXPECT_EQ(scheduler.template_stats().applied, 1u);
  LSE_EXPECT_EQ(scheduler.last_trace().host_groups, 0u);

  backend::BackendAdapter<RecordingDevice> plain_be;
  check(plain_be.init(0));
  Scheduler plain(plain_be);
  plain.set_dialect(Dialect::kLoom);
  plain.set_launch_templates(false);
  Layer p = layer(plain_be, 304);
  Program plain_program;
  const NodePtr plain_roots[] = {p.out.node()};
  check(plain.eval(plain_roots, false, &plain_program));
  LSE_EXPECT(plain_be.impl().launched == be.impl().launched);
}

// A norm feeding a prefill GEMM stores the GEMM's f16 activation panel from
// its own epilogue (graph::join_epilogue_output): the norm and the panel are
// one launch, the GEMM the other. Without a norm in front, the panel is a
// launch of its own. The panel and the GEMM are built as graph::quant_linear
// builds them for a device with the tiled GEMM.
LSE_TEST(a_norm_stores_the_panel_its_gemm_reads) {
  for (const bool norm_first : {true, false}) {
    backend::BackendAdapter<RecordingDevice> be;
    check(be.init(0));
    Scheduler scheduler(be);
    scheduler.set_dialect(Dialect::kLoom);
    const auto leaf = [&](Shape sh, DType dt = DType::kF32) {
      auto b = be.allocate(sh.elem_count() * 4, backend::MemoryClass::kDevice,
                           backend::kDefaultStream);
      if (!b.ok()) throw std::runtime_error(b.status().to_string());
      return Array::from_buffer(std::move(*b), sh, dt);
    };
    const std::int64_t m = 64, k = 256, n = 512;
    auto x = leaf({1, m, k});
    auto activation = norm_first ? rms_norm(x, leaf({k}), 1e-6f) : neg(x);
    auto panel = std::make_shared<Node>();
    panel->set_kind(OpKind::kCustom);
    panel->prim = find_primitive("quant_activation.f16_panel.v1");
    LSE_EXPECT(panel->prim != nullptr);
    if (panel->prim == nullptr) return;
    panel->fclass = panel->prim->fusion_class();
    panel->shape = Shape{m, k};
    panel->dtype = DType::kF16;
    panel->inputs = {activation.node()};
    ++activation.node()->consumer_count;
    auto gemm = std::make_shared<Node>();
    gemm->set_kind(OpKind::kCustom);
    gemm->prim = find_primitive("quant_linear.q4_gemm_f16.v1");
    gemm->fclass = gemm->prim->fusion_class();
    gemm->shape = Shape{1, m, n};
    gemm->dtype = DType::kF32;
    gemm->iattrs[0] = 4;
    gemm->iattrs[1] = 64;
    gemm->inputs = {activation.node(), leaf({n, k / 8}, DType::kU32).node(),
                    leaf({n, k / 64}, DType::kBF16).node(),
                    leaf({n, k / 64}, DType::kBF16).node(), panel};
    ++activation.node()->consumer_count;
    ++panel->consumer_count;
    Program program;
    const NodePtr roots[] = {gemm};
    check(scheduler.eval(roots, false, &program));
    LSE_EXPECT_EQ(scheduler.last_trace().host_groups, 0u);
    LSE_EXPECT_EQ(scheduler.last_trace().kernels_launched, norm_first ? 2u : 3u);
    bool fused = false;
    for (const FusionGroup& g : program.groups())
      fused = fused || (g.outputs.size() == 2 && g.outputs[0] == activation.node() &&
                        g.outputs[1] == panel);
    LSE_EXPECT_EQ(fused, norm_first);
  }
}

// A layer's residual sum over a K-sliced projection, the norm of it and that
// norm's panel are one launch (graph::exact_residual_norm), the next GEMM the
// other; x stays an output for the next residual.
LSE_TEST(a_residual_sum_its_norm_and_panel_are_one_launch) {
  backend::BackendAdapter<RecordingDevice> be;
  check(be.init(0));
  Scheduler scheduler(be);
  scheduler.set_dialect(Dialect::kLoom);
  const auto leaf = [&](Shape sh, DType dt = DType::kF32) {
    auto b = be.allocate(sh.elem_count() * 4, backend::MemoryClass::kDevice,
                         backend::kDefaultStream);
    if (!b.ok()) throw std::runtime_error(b.status().to_string());
    return Array::from_buffer(std::move(*b), sh, dt);
  };
  const std::int64_t m = 64, k = 512, n = 256;
  auto partial = leaf({2, 1, m, k});
  auto sum = std::make_shared<Node>();
  sum->set_kind(OpKind::kCustom);
  sum->prim = find_primitive("quant_linear.q4_gemm_f16.slice_sum.v1");
  LSE_EXPECT(sum->prim != nullptr);
  if (sum->prim == nullptr) return;
  sum->fclass = sum->prim->fusion_class();
  sum->shape = Shape{1, m, k};
  sum->dtype = DType::kF32;
  sum->inputs = {partial.node()};
  auto residual = leaf({1, m, k});
  auto x = residual + Array(sum);
  auto norm = rms_norm(x, leaf({k}), 1e-6f);
  auto panel = std::make_shared<Node>();
  panel->set_kind(OpKind::kCustom);
  panel->prim = find_primitive("quant_activation.f16_panel.v1");
  panel->fclass = panel->prim->fusion_class();
  panel->shape = Shape{m, k};
  panel->dtype = DType::kF16;
  panel->inputs = {norm.node()};
  ++norm.node()->consumer_count;
  auto gemm = std::make_shared<Node>();
  gemm->set_kind(OpKind::kCustom);
  gemm->prim = find_primitive("quant_linear.q4_gemm_f16.v1");
  gemm->fclass = gemm->prim->fusion_class();
  gemm->shape = Shape{1, m, n};
  gemm->dtype = DType::kF32;
  gemm->iattrs[0] = 4;
  gemm->iattrs[1] = 64;
  gemm->inputs = {norm.node(), leaf({n, k / 8}, DType::kU32).node(),
                  leaf({n, k / 64}, DType::kBF16).node(), leaf({n, k / 64}, DType::kBF16).node(),
                  panel};
  ++norm.node()->consumer_count;
  ++panel->consumer_count;
  // x is read again later, by the next residual add.
  auto next = x + leaf({1, m, k});
  Program program;
  const NodePtr roots[] = {gemm, next.node()};
  check(scheduler.eval(roots, false, &program));
  LSE_EXPECT_EQ(scheduler.last_trace().host_groups, 0u);
  bool fused = false;
  for (const FusionGroup& g : program.groups())
    fused = fused || (g.nodes.size() == 4 && g.outputs.size() == 3 &&
                      g.outputs[0] == x.node() && g.outputs[1] == norm.node() &&
                      g.outputs[2] == panel);
  LSE_EXPECT(fused);
  // The fused kernel, the GEMM and the later add.
  LSE_EXPECT_EQ(scheduler.last_trace().kernels_launched, 3u);
}

// Another width is another key: no template crosses shapes.
LSE_TEST(a_template_is_taken_only_by_its_own_shape) {
  backend::BackendAdapter<RecordingDevice> be;
  check(be.init(0));
  Scheduler scheduler(be);
  scheduler.set_dialect(Dialect::kLoom);
  for (const std::int64_t rows : {288, 304, 288}) {
    Layer l = layer(be, rows);
    Program program;
    const NodePtr roots[] = {l.out.node()};
    check(scheduler.eval(roots, false, &program));
  }
  LSE_EXPECT_EQ(scheduler.template_stats().recorded, 2u);
  LSE_EXPECT_EQ(scheduler.template_stats().applied, 1u);
}

// Two graphs of one structure over different buffers share a key; changing
// any one thing a split or a description reads gives another key.
LSE_TEST(different_structures_never_share_a_key) {
  backend::BackendAdapter<RecordingDevice> be;
  check(be.init(0));
  const auto key_of = [&](const Array& out) {
    const NodePtr roots[] = {out.node()};
    const auto nodes = template_nodes(roots);
    return launch_template_key(nodes, roots, "gfx1201|test");
  };
  const auto leaf = [&](Shape sh, DType dt = DType::kF32) {
    auto b = be.allocate(sh.elem_count() * 4, backend::MemoryClass::kDevice,
                         backend::kDefaultStream);
    if (!b.ok()) throw std::runtime_error(b.status().to_string());
    return Array::from_buffer(std::move(*b), sh, dt);
  };
  const std::string base = key_of(layer(be, 17).out);
  LSE_EXPECT(key_of(layer(be, 17).out) == base);
  LSE_EXPECT(key_of(layer(be, 18).out) != base);           // a shape
  LSE_EXPECT(key_of(layer(be, 17, 1e-5f).out) != base);    // a float attribute
  LSE_EXPECT(launch_template_key(template_nodes(std::vector<NodePtr>{layer(be, 17).out.node()}),
                                 std::vector<NodePtr>{}, "gfx1201|test") != base);  // the roots
  {
    const NodePtr roots[] = {layer(be, 17).out.node()};
    LSE_EXPECT(launch_template_key(template_nodes(roots), roots, "gfx1151|test") != base);
  }

  // One op against each change of it.
  auto a = leaf({4, 64});
  auto b = leaf({4, 64});
  const std::string sum = key_of(a + b);
  LSE_EXPECT(key_of(leaf({4, 64}) + leaf({4, 64})) == sum);
  LSE_EXPECT(key_of(a * b) != sum);                              // the op
  LSE_EXPECT(key_of(a + a) != sum);                              // the wiring
  LSE_EXPECT(key_of(leaf({4, 64}, DType::kF16) + b) != sum);     // a dtype
  LSE_EXPECT(key_of(leaf({4, 32}) + leaf({4, 32})) != sum);      // a leaf's shape
  LSE_EXPECT(key_of(slice(a, -1, 0, 32) + slice(b, -1, 0, 32)) !=
             key_of(slice(a, -1, 32, 64) + slice(b, -1, 0, 32)));  // an integer attribute
  {
    // Two leaves that share one allocation against two that do not.
    auto whole = leaf({8, 64});
    auto top = Array::from_buffer(whole.node()->buffer, Shape{4, 64}, DType::kF32);
    auto shared = Array::from_buffer(whole.node()->buffer, Shape{4, 64}, DType::kF32);
    LSE_EXPECT(key_of(top + shared) != sum);
  }
  {
    // The same leaf shape, one materialized and one still to be computed.
    auto computed = neg(leaf({4, 64}));
    LSE_EXPECT(key_of(computed + b) != key_of(leaf({4, 64}) + b));
  }
  {
    // A node placed on another member.
    auto x = leaf({4, 64});
    auto y = leaf({4, 64});
    auto placed = x + y;
    placed.node()->member = 1;
    LSE_EXPECT(key_of(placed) != sum);
  }
}

LSE_TEST_MAIN()
