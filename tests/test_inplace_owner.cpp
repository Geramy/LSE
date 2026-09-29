#include "harness.hpp"
#include "lse/backends/cpu/cpu_backend.hpp"
#include "lse/graph/graph.hpp"
#include "lse/graph/kernel_primitive.hpp"
#include "lse/graph/ops.hpp"
#include "lse/graph/interpreter.hpp"
#include "lse/graph/program.hpp"
#include "lse/graph/stream_plan.hpp"
#include "lse/graph/view.hpp"
#include <algorithm>
#include <cstring>
#include <limits>
#include <numeric>
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

namespace {
class ViewOpaqueBackend final : public backend::Backend<ViewOpaqueBackend> {
 public:
  static constexpr std::string_view kName = "opaque-view-test";
  Status init_impl(int ordinal) {
    if (ordinal != 0) return LSE_ERROR(kInvalidArgument, "invalid test ordinal");
    info_.arch = "host";
    info_.compute_units = 1;
    info_.max_threads_per_workgroup = 1;
    return OkStatus();
  }
  void shutdown_impl() noexcept {}
  const backend::DeviceInfo& device_info_impl() const noexcept { return info_; }
  Result<backend::DeviceBuffer> allocate_impl(std::size_t bytes,
      backend::MemoryClass, backend::Stream) {
    if (!bytes) return LSE_ERROR(kInvalidArgument, "empty test buffer");
    backend::DeviceBuffer buffer;
    buffer.storage = std::shared_ptr<void>(new std::byte[bytes],
        [](void* p) { delete[] static_cast<std::byte*>(p); });
    buffer.handle = reinterpret_cast<std::uint64_t>(buffer.storage.get());
    buffer.size_bytes = bytes;
    return buffer;
  }
  void deallocate_impl(backend::DeviceBuffer& b) noexcept { b = {}; }
  Status copy_h2d_impl(const void* source, backend::DeviceBuffer& target,
                      std::size_t bytes, std::size_t offset) {
    if (offset > target.size_bytes || bytes > target.size_bytes - offset)
      return LSE_ERROR(kOutOfRange, "test upload exceeds buffer");
    std::memcpy(static_cast<std::byte*>(target.storage.get()) + target.offset + offset,
                source, bytes);
    return OkStatus();
  }
  Status copy_d2h_impl(const backend::DeviceBuffer& source, void* target,
                      std::size_t bytes, std::size_t offset) {
    if (offset > source.size_bytes || bytes > source.size_bytes - offset)
      return LSE_ERROR(kOutOfRange, "test download exceeds buffer");
    std::memcpy(target, static_cast<std::byte*>(source.storage.get()) + source.offset + offset,
                bytes);
    return OkStatus();
  }
  Status copy_peer_impl(const backend::DeviceBuffer& source,
      backend::DeviceBuffer& target, std::size_t bytes,
      std::size_t source_offset, std::size_t target_offset) {
    if (source_offset > source.size_bytes || bytes > source.size_bytes - source_offset ||
        target_offset > target.size_bytes || bytes > target.size_bytes - target_offset)
      return LSE_ERROR(kOutOfRange, "test device copy exceeds buffer");
    std::memmove(static_cast<std::byte*>(target.storage.get()) + target.offset + target_offset,
                 static_cast<std::byte*>(source.storage.get()) + source.offset + source_offset,
                 bytes);
    return OkStatus();
  }
  Result<backend::KernelHandle> load_executable_impl(std::string_view,
      std::span<const std::byte>) { return LSE_ERROR(kUnimplemented, "host test only"); }
  Status launch_impl(const backend::KernelHandle&, const backend::LaunchDims&,
      const backend::DispatchArgs&) { return LSE_ERROR(kUnimplemented, "host test only"); }
  Status synchronize_impl() { return OkStatus(); }
  std::span<const graph::KernelToolchain> toolchains_impl() const noexcept { return {}; }
 private:
  backend::DeviceInfo info_;
};

Array view_input(backend::IBackend& backend, Shape shape, std::size_t offset = 12) {
  const auto bytes = shape.elem_count() * sizeof(float);
  auto allocation = backend.allocate(bytes + offset, backend::MemoryClass::kDevice);
  LSE_EXPECT(allocation.ok());
  if (!allocation.ok()) return {};
  auto buffer = allocation.release();
  buffer.offset = offset;
  buffer.size_bytes = bytes;
  std::vector<float> values(shape.elem_count());
  std::iota(values.begin(), values.end(), 1.0f);
  LSE_EXPECT_OK(backend.copy(buffer, values.data(), bytes));
  return Array::from_buffer(std::move(buffer), shape, DType::kF32);
}
std::vector<float> view_read(const Array& array) {
  std::vector<float> values(array.shape().elem_count());
  LSE_EXPECT_OK(interpreter::read_raw(*array.node(), values.data(), values.size() * sizeof(float)));
  return values;
}
}

LSE_TEST(contiguous_slices_are_offset_views_and_refresh_after_rebinding) {
  backend::BackendAdapter<backend::CpuBackend> backend;
  LSE_EXPECT_OK(backend.init(0));
  Scheduler scheduler(backend);
  auto input = view_input(backend, {1, 4, 8});
  auto first = slice(input, 1, 1, 4);
  auto flat = reshape(first, {3, 8});
  auto second = slice(flat, 0, 1, 2);
  const NodePtr roots[]{second.node()};
  Program program;
  LSE_EXPECT_OK(scheduler.eval(roots, true, &program));
  LSE_EXPECT_EQ(scheduler.last_trace().device_groups, 0u);
  LSE_EXPECT_EQ(scheduler.last_trace().host_groups, 0u);
  LSE_EXPECT(second.node()->buffer.storage == input.node()->buffer.storage);
  LSE_EXPECT_EQ(second.node()->buffer.offset, 12u + 16u * sizeof(float));
  const auto actual = view_read(second);
  for (std::size_t i = 0; i < actual.size(); ++i) LSE_EXPECT_EQ(actual[i], 17.0f + i);

  auto replacement = view_input(backend, input.shape(), 20);
  std::vector<float> changed(input.shape().elem_count(), -7.25f);
  LSE_EXPECT_OK(backend.copy(replacement.node()->buffer, changed.data(), changed.size() * sizeof(float)));
  input.node()->buffer = replacement.node()->buffer;
  program.reset_compute();
  LSE_EXPECT_OK(scheduler.eval(roots, true, &program));
  LSE_EXPECT(second.node()->buffer.storage == replacement.node()->buffer.storage);
  LSE_EXPECT_EQ(second.node()->buffer.offset, 20u + 16u * sizeof(float));
  for (float value : view_read(second)) LSE_EXPECT_EQ(value, -7.25f);
}

LSE_TEST(noncontiguous_slice_keeps_its_indexed_copy_and_pointwise_fusion) {
  backend::BackendAdapter<backend::CpuBackend> backend;
  LSE_EXPECT_OK(backend.init(0));
  Scheduler scheduler(backend);
  auto input = view_input(backend, {2, 4});
  auto window = slice(input, 1, 1, 3);
  auto activation = silu(window);
  LSE_EXPECT(!is_buffer_view(*window.node()));
  LSE_EXPECT(Partitioner::can_fuse(*window.node(), *activation.node()));
  const NodePtr roots[]{window.node()};
  LSE_EXPECT_OK(scheduler.eval(roots, true));
  LSE_EXPECT(window.node()->buffer.storage != input.node()->buffer.storage);
  const auto actual = view_read(window);
  const float expected[]{2, 3, 6, 7};
  for (std::size_t i = 0; i < actual.size(); ++i) LSE_EXPECT_EQ(actual[i], expected[i]);
}

LSE_TEST(escaped_contiguous_slice_pins_the_upstream_slot_owner) {
  backend::BackendAdapter<backend::CpuBackend> backend;
  LSE_EXPECT_OK(backend.init(0));
  AllocationTestKernel ordinary;
  auto input = Array::full({1024}, DType::kF32, 0).node();
  auto stage = make_test_node(&ordinary, {1024}, {input});
  auto first = slice(Array(stage), 0, 256, 768).node();
  auto flat = reshape(Array(first), {2, 256}).node();
  auto escaped = slice(Array(flat), 0, 1, 2).node();
  auto temporary = make_test_node(&ordinary, {1024}, {input});
  auto later = make_test_node(&ordinary, {1024}, {temporary});
  std::vector<NodePtr> order{stage, first, flat, escaped, temporary, later};
  Workgroup workgroup;
  for (const auto& node : order) LSE_EXPECT(workgroup.try_add(node));
  const NodePtr roots[]{escaped, later};
  workgroup.plan_slots(roots, separate_launches(order));
  LSE_EXPECT_OK(workgroup.bind_slots(backend));
  LSE_EXPECT(stage->buffer.storage != temporary->buffer.storage);
  LSE_EXPECT(stage->buffer.storage != later->buffer.storage);
  LSE_EXPECT(escaped->buffer.storage == stage->buffer.storage);
  LSE_EXPECT_EQ(escaped->buffer.offset, 512u * sizeof(float));
  LSE_EXPECT_EQ(escaped->buffer.size_bytes, 256u * sizeof(float));
}

LSE_TEST(owned_materialization_keeps_a_snapshot_across_producer_replay) {
  backend::BackendAdapter<backend::CpuBackend> backend;
  LSE_EXPECT_OK(backend.init(0));
  Scheduler scheduler(backend);
  auto input = view_input(backend, {1, 4, 8});
  auto tail = reshape(slice(input, 1, 3, 4), {1, 8});
  LSE_EXPECT_OK(interpreter::ensure_owned_output_buffer(*tail.node(), backend));
  const NodePtr roots[]{tail.node()};
  LSE_EXPECT_OK(scheduler.eval(roots, true));
  LSE_EXPECT(tail.node()->buffer.storage != input.node()->buffer.storage);
  LSE_EXPECT(tail.node()->inputs[0]->requires_owned_storage);
  auto detached = Array::from_buffer(tail.node()->buffer, tail.shape(), tail.dtype());
  std::vector<float> overwrite(input.shape().elem_count(), -19.0f);
  LSE_EXPECT_OK(backend.copy(input.node()->buffer, overwrite.data(), overwrite.size() * sizeof(float)));
  for (float value : view_read(detached)) LSE_EXPECT(value >= 25.0f && value <= 32.0f);
}

LSE_TEST(contiguous_view_preserves_opaque_dirty_state_and_raw_integer_payloads) {
  auto source = std::make_shared<Node>();
  source->shape = {1, 4, 2};
  source->dtype = DType::kU32;
  source->materialized = true;
  source->buffer.handle = 123;
  source->buffer.offset = 16;
  source->buffer.size_bytes = 8 * sizeof(std::uint32_t);
  source->host_dirty = true;
  const std::uint32_t words[]{0xff800001u, 0x12345678u, 0xffffffffu, 0x7fc01234u,
                             0x80000001u, 0x00ffffffu, 0xaabbccddu, 0x98765432u};
  source->host_mirror.resize(sizeof(words));
  std::memcpy(source->host_mirror.data(), words, sizeof(words));
  auto window = slice(Array(source), 1, 1, 3);
  LSE_EXPECT_OK(bind_buffer_view(*window.node()));
  LSE_EXPECT_EQ(window.node()->buffer.offset, 24u);
  LSE_EXPECT(window.node()->host_dirty);
  LSE_EXPECT(!window.node()->device_dirty);
  LSE_EXPECT(std::memcmp(window.node()->host_mirror.data(), words + 2, 4 * sizeof(std::uint32_t)) == 0);
  source->host_dirty = false;
  source->device_dirty = false;
  window.node()->materialized = false;
  LSE_EXPECT_OK(bind_buffer_view(*window.node()));
  LSE_EXPECT(!window.node()->device_dirty);
  LSE_EXPECT(std::memcmp(window.node()->host_mirror.data(), words + 2,
                         4 * sizeof(std::uint32_t)) == 0);
  source->host_mirror.clear();
  LSE_EXPECT_OK(bind_buffer_view(*window.node()));
  LSE_EXPECT(window.node()->device_dirty);
  source->device_dirty = true;
  window.node()->materialized = false;
  LSE_EXPECT_OK(bind_buffer_view(*window.node()));
  LSE_EXPECT(window.node()->host_mirror.empty());
  LSE_EXPECT(window.node()->device_dirty);
}

LSE_TEST(contiguous_view_validates_bounds_shapes_dtype_and_byte_overflow) {
  auto source = Array::full({1, 4, 8}, DType::kF16, 0);
  auto window = slice(source, 1, 1, 3);
  auto layout = buffer_view_window(*window.node());
  LSE_EXPECT(layout.ok() && layout->has_value());
  if (layout.ok() && layout->has_value()) {
    LSE_EXPECT_EQ((*layout)->offset, 16u);
    LSE_EXPECT_EQ((*layout)->bytes, 32u);
  }
  window.node()->iattrs[1] = -1;
  LSE_EXPECT(!buffer_view_window(*window.node()).ok());
  window.node()->iattrs[1] = 1;
  window.node()->dtype = DType::kF32;
  LSE_EXPECT(!buffer_view_window(*window.node()).ok());
  window.node()->dtype = source.dtype();
  window.node()->shape = {1, 3, 8};
  LSE_EXPECT(!buffer_view_window(*window.node()).ok());
  auto huge = Array::full({std::numeric_limits<std::int64_t>::max(), 2}, DType::kF32, 0);
  auto shaped = reshape(huge, huge.shape());
  LSE_EXPECT(!buffer_view_window(*shaped.node()).ok());
  source.node()->buffer.handle = 1;
  source.node()->buffer.size_bytes = 4;
  source.node()->materialized = true;
  auto short_storage = slice(source, 1, 1, 3);
  LSE_EXPECT(!bind_buffer_view(*short_storage.node()).ok());
  auto same = slice(source, 1, 0, 4);
  auto alias = slice(source, 1, 1, 3);
  const NodePtr bindings[]{same.node(), alias.node()};
  LSE_EXPECT(bindings_may_alias(bindings));
}


LSE_TEST(opaque_owned_slice_preserves_raw_bytes_and_upload_authority) {
  backend::BackendAdapter<ViewOpaqueBackend> backend;
  LSE_EXPECT_OK(backend.init(0));
  auto allocation = backend.allocate(8 * sizeof(std::uint32_t), backend::MemoryClass::kDevice, backend::kDefaultStream);
  LSE_EXPECT(allocation.ok());
  if (!allocation.ok()) return;
  auto source = Array::from_buffer(allocation.release(), {2, 4}, DType::kU32);
  const std::uint32_t words[]{0x12345678u, 0xffffffffu, 0x7fc01234u, 0x80000001u,
                             0xaabbccddu, 0xfedcba98u, 0x87654321u, 0x00ffffffu};
  LSE_EXPECT_OK(backend.copy(source.node()->buffer, words, sizeof(words)));
  auto window = slice(source, 1, 1, 3);
  LSE_EXPECT(!is_buffer_view(*window.node()));
  LSE_EXPECT_OK(interpreter::evaluate(window.node(), backend));
  LSE_EXPECT(window.node()->host_dirty && !window.node()->device_dirty);
  LSE_EXPECT_OK(interpreter::sync_to_device(*window.node(), backend));
  std::uint32_t result[4]{};
  LSE_EXPECT_OK(backend.copy_d2h(window.node()->buffer, result, sizeof(result), 0));
  const std::uint32_t expected[]{words[1], words[2], words[5], words[6]};
  LSE_EXPECT(std::memcmp(result, expected, sizeof(result)) == 0);

  auto clean = slice(source, 0, 0, 1);
  LSE_EXPECT(!source.node()->device_dirty && !source.node()->host_dirty);
  LSE_EXPECT_OK(bind_buffer_view(*clean.node()));
  LSE_EXPECT_OK(interpreter::read_raw(*clean.node(), result, sizeof(result)));
  LSE_EXPECT(std::memcmp(result, words, sizeof(result)) == 0);
  const std::uint32_t changed = 0xdeadbeefu;
  std::memcpy(clean.node()->host_mirror.data(), &changed, sizeof(changed));
  clean.node()->host_dirty = true;
  LSE_EXPECT_OK(interpreter::ensure_owned_output_buffer(*clean.node(), backend));
  LSE_EXPECT(clean.node()->buffer.storage != source.node()->buffer.storage);
  LSE_EXPECT_OK(backend.copy_d2h(clean.node()->buffer, result, sizeof(result), 0));
  LSE_EXPECT_EQ(result[0], changed);
  LSE_EXPECT_EQ(result[1], words[1]);
}

LSE_TEST(public_owned_materialization_detaches_a_pure_reshape_snapshot) {
  LSE_EXPECT(!Array{}.materialize_owned().ok());
  auto* scheduler = default_scheduler();
  LSE_EXPECT(scheduler != nullptr);
  if (!scheduler) return;
  auto& backend = scheduler->backend();
  auto source = view_input(backend, {2, 8});
  auto view = reshape(source, {1, 16});
  LSE_EXPECT_OK(view.materialize_owned());
  LSE_EXPECT(view.node()->kind == OpKind::kBuffer);
  LSE_EXPECT(view.node()->buffer.storage != source.node()->buffer.storage);
  std::vector<float> changed(16, -19.0f);
  LSE_EXPECT_OK(backend.copy(source.node()->buffer, changed.data(), changed.size() * sizeof(float)));
  LSE_EXPECT_OK(interpreter::sync_from_device(*view.node(), backend));
  const auto actual = view_read(view);
  for (std::size_t i = 0; i < actual.size(); ++i) LSE_EXPECT_EQ(actual[i], 1.0f + i);
}

LSE_TEST(carried_reshape_swaps_its_owning_slice_allocation) {
  backend::BackendAdapter<backend::CpuBackend> backend;
  LSE_EXPECT_OK(backend.init(0));
  Scheduler scheduler(backend);
  auto source = view_input(backend, {1, 4, 8});
  auto owned = slice(source, 1, 3, 4);
  LSE_EXPECT_OK(interpreter::ensure_owned_output_buffer(*owned.node(), backend));
  auto output = reshape(owned, {1, 8});
  const NodePtr roots[]{output.node()};
  LSE_EXPECT_OK(scheduler.eval(roots, true));
  auto input = view_input(backend, {1, 8});
  const auto prior_input = input.node()->buffer.storage;
  const auto produced = owned.node()->buffer.storage;
  const NodePtr order[]{owned.node(), output.node()};
  Program program;
  program.retain(roots, {}, {}, order);
  program.set_carries({{input.node(), output.node()}});
  program.fold_carries();
  LSE_EXPECT(input.node()->buffer.storage == produced);
  LSE_EXPECT(owned.node()->buffer.storage == prior_input);
  LSE_EXPECT(output.node()->buffer.storage == prior_input);
  for (float value : view_read(input)) LSE_EXPECT(value >= 25.0f && value <= 32.0f);
  LSE_EXPECT_OK(refresh_buffer_aliases(order));
  LSE_EXPECT(output.node()->buffer.storage == prior_input);
  program.hold_carries();
  LSE_EXPECT(input.node()->materialized);
  LSE_EXPECT(!output.node()->materialized);
  LSE_EXPECT(program.holds(roots));
  owned.node()->requires_owned_storage = false;
  LSE_EXPECT(!program.holds(roots));
}

LSE_TEST(view_rebinding_refreshes_cross_stream_dependencies_before_placement) {
  AllocationTestKernel primitive;
  auto source = make_test_node(&primitive, {1, 4, 8}, {});
  source->buffer.handle = 1;
  source->buffer.size_bytes = 32 * sizeof(float);
  auto view = slice(Array(source), 1, 1, 3).node();
  const NodePtr order[]{source, view};
  LSE_EXPECT_OK(refresh_buffer_aliases(order));
  LSE_EXPECT(!view->materialized);
  source->buffer.handle = 2;
  source->buffer.offset = 64;
  LSE_EXPECT_OK(refresh_buffer_aliases(order));
  LSE_EXPECT_EQ(view->buffer.handle, 2u);
  LSE_EXPECT_EQ(view->buffer.offset, 64u + 8u * sizeof(float));
  auto result = make_test_node(&primitive, view->shape, {view});
  result->buffer.handle = 3;
  result->buffer.size_bytes = view->buffer.size_bytes;
  const auto groups = separate_launches({source, view, result});
  backend::StreamCapabilities capabilities;
  capabilities.stream_count = capabilities.concurrent_streams = 2;
  const std::uint32_t streams[]{0, 0, 1};
  const auto plan = plan_streams(groups, capabilities, backend::DeviceInfo{}, {}, streams);
  LSE_EXPECT(plan.waits[2] == std::vector<std::uint32_t>{0});
  LSE_EXPECT_EQ(plan.record_after[0], 1u);
}

LSE_TEST(view_replay_preserves_previously_materialized_producer_boundaries) {
  backend::BackendAdapter<backend::CpuBackend> backend;
  LSE_EXPECT_OK(backend.init(0));
  Scheduler scheduler(backend);
  auto source = view_input(backend, {4, 8});
  auto owner = cast(cast(source, DType::kBF16), DType::kF32);
  const NodePtr owner_roots[]{owner.node()};
  LSE_EXPECT_OK(scheduler.eval(owner_roots, true));
  LSE_EXPECT(owner.node()->kind == OpKind::kCast && owner.node()->materialized);
  auto view = slice(owner, 0, 1, 3);
  const NodePtr roots[]{view.node()};
  Program program;
  LSE_EXPECT_OK(scheduler.eval(roots, true, &program));
  LSE_EXPECT_EQ(scheduler.last_trace().host_groups, 0u);
  LSE_EXPECT_EQ(program.compute_count(), 1u);
  LSE_EXPECT(program.node_count() > program.compute_count());
  program.reset_compute();
  LSE_EXPECT(owner.node()->materialized);
  LSE_EXPECT(!view.node()->materialized);
  LSE_EXPECT_OK(scheduler.eval(roots, true, &program));
  LSE_EXPECT_EQ(scheduler.last_trace().host_groups, 0u);
  const auto first = view_read(view);
  for (std::size_t i = 0; i < first.size(); ++i) LSE_EXPECT_EQ(first[i], 9.0f + i);
  auto replacement = view_input(backend, owner.shape(), 20);
  std::vector<float> values(owner.shape().elem_count(), -7.25f);
  LSE_EXPECT_OK(backend.copy(replacement.node()->buffer, values.data(), values.size() * sizeof(float)));
  owner.node()->buffer = replacement.node()->buffer;
  program.reset_compute();
  LSE_EXPECT(owner.node()->materialized);
  LSE_EXPECT_OK(scheduler.eval(roots, true, &program));
  LSE_EXPECT_EQ(scheduler.last_trace().host_groups, 0u);
  LSE_EXPECT(view.node()->buffer.storage == replacement.node()->buffer.storage);
  for (float value : view_read(view)) LSE_EXPECT_EQ(value, -7.25f);
}

LSE_TEST_MAIN()
