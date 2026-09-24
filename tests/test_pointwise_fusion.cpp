#include "harness.hpp"
#include "lse/backends/hrx/arch_database.hpp"
#include "lse/backends/hrx/loomc/loom_emitter.hpp"
#include "lse/graph/ops.hpp"
#include "lse/graph/pointwise_fusion.hpp"
using namespace lse;
using namespace lse::graph;
namespace {
Array leaf(Shape s = {1, 128}, DType dtype = DType::kF32) {
  auto n = std::make_shared<Node>();
  n->shape = s;
  n->dtype = dtype;
  n->materialized = true;
  return Array(n);
}
FusionGroup solo(const Array &a) {
  FusionGroup g;
  g.nodes = {a.node()};
  g.outputs = g.nodes;
  g.inputs = a.node()->inputs;
  g.anchor = a.node()->kind;
  g.anchor_class = a.node()->fclass;
  return g;
}
bool join(FusionGroup &g, const Array &next,
          std::span<const NodePtr> roots = {}) {
  return join_pointwise_chain(g, next.node(), roots, backend::loom_sources());
}
} // namespace
LSE_TEST(gates_fuse_and_keep_external_bindings) {
  auto x = leaf(), v = leaf();
  auto sig = sigmoid(x), out = sig * v;
  auto g = solo(sig);
  const NodePtr roots[] = {out.node()};
  LSE_EXPECT(Partitioner::can_fuse(*sig.node(), *out.node()));
  LSE_EXPECT(join(g, out, roots));
  LSE_EXPECT(g.nodes.size() == 2);
  LSE_EXPECT(g.outputs.size() == 1 && g.outputs[0] == out.node());
  LSE_EXPECT(g.inputs.size() == 2 && g.inputs[0] == x.node() &&
             g.inputs[1] == v.node());
  backend::DeviceInfo device;
  device.arch = "gfx1201";
  device.compute_units = 64;
  device.max_threads_per_workgroup = 1024;
  device.wavefront_size = 32;
  device.lds_bytes_per_workgroup = 65536;
  backend::AmdDeviceInfo amd;
  backend::apply_arch_defaults(device, amd);
  device.extension_id = backend::AmdDeviceInfo::kExtensionId;
  device.extension = &amd;
  backend::LoomEmitter emitter;
  auto emitted = emitter.emit(g, device);
  LSE_EXPECT(emitted.ok());
  if (!emitted.ok())
    return;
  LSE_EXPECT(emitted->binding_order.size() == 3);
  LSE_EXPECT(emitted->source.find("scalar.expf") != std::string::npos);
  LSE_EXPECT(emitted->source.find("scalar.mulf") != std::string::npos);
  LSE_EXPECT(emitted->source.find("kernel.barrier") == std::string::npos);
}
LSE_TEST(three_node_chain_and_duplicate_edges) {
  auto x = leaf();
  auto a = softplus(x), b = neg(a), c = exp(b);
  auto g = solo(a);
  LSE_EXPECT(join(g, b));
  LSE_EXPECT(join(g, c));
  LSE_EXPECT(g.nodes.size() == 3 && g.inputs.size() == 1 &&
             g.outputs[0] == c.node());
  auto d = sigmoid(x), e = d * d;
  auto repeated = solo(d);
  LSE_EXPECT(d.node()->consumer_count == 1);
  LSE_EXPECT(join(repeated, e));
  LSE_EXPECT(repeated.inputs.size() == 1);
}
LSE_TEST(escaping_roots_and_fanout_stay_materialized) {
  auto x = leaf();
  auto a = sigmoid(x), b = a * x;
  auto g = solo(a);
  const NodePtr roots[] = {a.node(), b.node()};
  LSE_EXPECT(!join(g, b, roots));
  LSE_EXPECT(g.nodes.size() == 1);
  auto c = neg(a);
  (void)c;
  LSE_EXPECT(a.node()->consumer_count == 2);
  LSE_EXPECT(!join(g, b));
  LSE_EXPECT(g.outputs[0] == a.node());
}
LSE_TEST(reductions_and_indexed_kernels_cannot_join) {
  auto x = leaf();
  auto a = sigmoid(x);
  auto r = sum(a, -1, true);
  auto g = solo(a);
  LSE_EXPECT(!join(g, r));
  auto n = rms_norm(x, leaf(Shape{128}), 1e-6f, false);
  auto out = n * x;
  auto norm = solo(n);
  LSE_EXPECT(!join(norm, out));
}
LSE_TEST(views_and_shape_changes_cannot_join) {
  auto x = leaf();
  auto a = sigmoid(x), v = reshape(a, Shape{128});
  auto g = solo(a);
  LSE_EXPECT(!join(g, v));
  auto small = sigmoid(leaf(Shape{1}));
  auto broad = small * x;
  auto sg = solo(small);
  LSE_EXPECT(!join(sg, broad));
}
LSE_TEST(placement_and_narrowing_boundaries_remain) {
  auto x = leaf();
  auto a = sigmoid(x), b = a * x;
  auto g = solo(a);
  a.node()->member = 0;
  b.node()->member = 1;
  LSE_EXPECT(!join(g, b));
  b.node()->member = Node::kAnyMember;
  LSE_EXPECT(!join(g, b));
  auto narrow = leaf(Shape{1, 128}, DType::kBF16);
  auto c = sigmoid(narrow), d = c * narrow;
  auto ng = solo(c);
  LSE_EXPECT(!join(ng, d));
}
LSE_TEST(no_cross_phase_or_unrelated_merge) {
  auto x = leaf();
  auto a = sigmoid(x), b = a * x;
  auto g = solo(a);
  g.is_phase = true;
  LSE_EXPECT(!join(g, b));
  g.is_phase = false;
  auto unrelated = neg(x);
  LSE_EXPECT(!join(g, unrelated));
  a.node()->materialized = true;
  LSE_EXPECT(!join(g, b));
}
LSE_TEST(missing_dialect_spelling_refuses_before_dispatch) {
  auto x = leaf();
  auto a = sigmoid(x), b = a * x;
  auto g = solo(a);
  DialectSourceTable empty(std::span<const PrimitiveSource>{});
  LSE_EXPECT(!join_pointwise_chain(g, b.node(), {}, empty));
}
int main() { return lse::test::run_all(); }
