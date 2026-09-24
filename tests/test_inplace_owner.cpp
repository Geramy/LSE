#include "harness.hpp"
#include "lse/backends/cpu/cpu_backend.hpp"
#include "lse/graph/graph.hpp"
#include "lse/graph/kernel_primitive.hpp"
#include "lse/graph/ops.hpp"
#include <algorithm>
using namespace lse;
using namespace lse::graph;
namespace {
struct AllocationTestKernel final : KernelPrimitive<AllocationTestKernel> {
  static constexpr std::string_view kName = "allocation_test";
  static constexpr std::string_view kEntry = "allocation_test";
  static constexpr std::string_view kSource = "";
  explicit AllocationTestKernel(int input = -1) : alias_input(input) {}
  int inplace_input() const noexcept override { return alias_input; }
  std::size_t arity() const noexcept override { return 1; }
  bool owns_indexing() const noexcept override { return true; }
  Result<Shape> infer_shape(std::span<const Shape> inputs) const override {
    return inputs[0];
  }
  DType infer_dtype(std::span<const DType>) const override {
    return DType::kF32;
  }
  static ThreadPlan plan_impl(const KernelShapes &) { return {}; }
  int alias_input;
};
NodePtr make_test_node(const Primitive *primitive, Shape shape,
                       std::vector<NodePtr> inputs) {
  auto node = std::make_shared<Node>();
  node->kind = OpKind::kCustom;
  node->prim = primitive;
  node->shape = shape;
  node->dtype = DType::kF32;
  node->fclass = FusionClass::kBarrier;
  node->inputs = std::move(inputs);
  for (auto &input : node->inputs)
    ++input->consumer_count;
  return node;
}
std::vector<FusionGroup> separate_launches(const std::vector<NodePtr> &nodes) {
  std::vector<FusionGroup> launches;
  for (const auto &node : nodes) {
    FusionGroup group;
    group.nodes = {node};
    group.inputs = node->inputs;
    group.outputs = {node};
    launches.push_back(std::move(group));
  }
  return launches;
}
void check_alias_lifetime(bool nested, bool shorter, bool escapes, int index) {
  backend::BackendAdapter<backend::CpuBackend> backend;
  LSE_EXPECT(backend.init(0).ok());
  AllocationTestKernel ordinary, alias(index);
  auto input = Array::full(Shape{1024}, DType::kF32, 0).node();
  auto stage = make_test_node(&ordinary, {1024}, {input});
  std::vector<NodePtr> order{stage};
  NodePtr source = stage;
  if (nested) {
    source = reshape(Array(source), {32, 32}).node();
    order.push_back(source);
  }
  auto alias_inputs = [&](NodePtr owner) {
    return index == 0 ? std::vector<NodePtr>{owner}
                      : std::vector<NodePtr>{input, owner};
  };
  auto repair = make_test_node(&alias, shorter ? Shape{512} : Shape{1024},
                               alias_inputs(source));
  order.push_back(repair);
  NodePtr view = repair;
  if (nested) {
    view = reshape(Array(view), shorter ? Shape{16, 32} : Shape{32, 32}).node();
    order.push_back(view);
    view = make_test_node(&alias, view->shape, alias_inputs(view));
    order.push_back(view);
  }
  auto temporary = make_test_node(&ordinary, {1024}, {input});
  auto after = make_test_node(&ordinary, {1024}, {temporary});
  auto late = make_test_node(&ordinary, {1024}, {view, after});
  order.insert(order.end(), {temporary, after, late});
  Workgroup workgroup;
  for (auto &node : order)
    LSE_EXPECT(workgroup.try_add(node));
  std::vector<NodePtr> roots{late};
  if (escapes)
    roots.push_back(view);
  workgroup.plan_slots(roots, separate_launches(order));
  LSE_EXPECT(workgroup.bind_slots(backend).ok());
  // Mirror the scheduler's topological binding of logical aliases. A shorter
  // inplace result retains the physical input allocation; reshape narrows its
  // logical buffer window. The planner must protect the full owning slot.
  for (auto &node : order) {
    if (node->kind == OpKind::kReshape) {
      node->buffer = node->inputs[0]->buffer;
      node->buffer.size_bytes = node->element_count() * sizeof(float);
    } else if (node->prim && node->prim->inplace_input() >= 0) {
      node->buffer = node->inputs[node->prim->inplace_input()]->buffer;
    }
  }
  LSE_EXPECT(stage->buffer.valid());
  LSE_EXPECT(view->buffer.valid());
  LSE_EXPECT(stage->buffer.ptr == view->buffer.ptr);
  LSE_EXPECT(view->buffer.size_bytes >= view->element_count() * sizeof(float));
  LSE_EXPECT(stage->buffer.ptr != temporary->buffer.ptr);
  LSE_EXPECT(stage->buffer.ptr != after->buffer.ptr);
  LSE_EXPECT(stage->buffer.ptr != late->buffer.ptr);
  auto *data = static_cast<float *>(stage->buffer.ptr);
  std::fill(data, data + 1024, 3.25f);
  auto *scratch = static_cast<float *>(temporary->buffer.ptr);
  std::fill(scratch, scratch + 1024, -11.f);
  auto *actual = static_cast<float *>(view->buffer.ptr);
  for (size_t i = 0; i < view->element_count(); ++i)
    LSE_EXPECT(actual[i] == 3.25f);
}
} // namespace
LSE_TEST(inplace_alias_keeps_temporary_owner_live) {
  for (bool nested : {false, true})
    for (bool shorter : {false, true})
      for (bool escapes : {false, true})
        for (int index : {0, 1})
          check_alias_lifetime(nested, shorter, escapes, index);
}
LSE_TEST(malformed_inplace_cycle_disables_slot_recycling) {
  AllocationTestKernel ordinary, alias(1);
  auto input = Array::full(Shape{1024}, DType::kF32, 0).node();
  auto stage = make_test_node(&ordinary, {1024}, {input});
  auto cycle = make_test_node(&alias, {1024}, {stage, input});
  cycle->inputs[1] = cycle;
  auto temporary = make_test_node(&ordinary, {1024}, {input});
  auto late = make_test_node(&ordinary, {1024}, {cycle, temporary});
  std::vector<NodePtr> nodes{stage, cycle, temporary, late};
  Workgroup workgroup;
  for (auto &node : nodes)
    LSE_EXPECT(workgroup.try_add(node));
  const NodePtr roots[]{late};
  workgroup.plan_slots(roots, separate_launches(nodes));
  LSE_EXPECT_EQ(workgroup.reused_slots(), 0u);
  cycle->inputs[1] = stage; // Release the intentionally malformed shared cycle.
}
LSE_TEST_MAIN()
