#include "harness.hpp"
#include "lse/backends/cpu/cpu_backend.hpp"
#include "lse/graph/codegen.hpp"
#include "lse/graph/graph.hpp"
#include "lse/graph/ops.hpp"
#include "lse/graph/program.hpp"
#include "lse/runtime/decode_sample.hpp"
#include <cstring>

namespace {
using namespace lse;
using namespace lse::graph;
class AddEmitter final : public IKernelEmitter {
 public:
  Result<EmittedKernel> emit(const FusionGroup& group, const backend::DeviceInfo&) const override {
    if (group.inputs.size() != 2 || group.outputs.size() != 1 || group.nodes.size() != 1)
      return LSE_ERROR(kInvalidArgument, "host replay fixture expects one binary add");
    EmittedKernel out;
    out.dialect = Dialect::kLoom;
    out.source = "host-only-replay-add";
    out.entry_name = "host_add";
    out.binding_order = group.inputs;
    out.binding_order.push_back(group.outputs[0]);
    out.constants.add("count", 4);
    return out;
  }
  Dialect dialect() const noexcept override { return Dialect::kLoom; }
  std::string_view prelude() const noexcept override { return {}; }
  DialectSourceTable sources() const noexcept override { return {}; }
};
class HostCompiler final : public IKernelCompiler {
 public:
  Result<CompiledKernel> compile(std::string_view, std::string_view) const override {
    CompiledKernel out;
    out.code.push_back(std::byte{1});
    return out;
  }
  bool available() const override { return true; }
  std::string identity() const override { return "host-replay-test-v1"; }
};
struct HostDevice : backend::CpuBackend {
  unsigned launches = 0;
  std::span<const KernelToolchain> toolchains() const noexcept {
    static const AddEmitter emitter;
    static const HostCompiler compiler;
    static const KernelToolchain chain{Dialect::kLoom, &emitter, &compiler};
    return std::span(&chain, 1);
  }
  Result<backend::KernelHandle> load_executable(std::string_view name, std::span<const std::byte>) {
    return backend::KernelHandle{1, 0, std::string(name)};
  }
  Status launch(const backend::KernelHandle&, const backend::LaunchDims&,
                const backend::DispatchArgs& args, const backend::DispatchTarget&) {
    if (args.bindings.size() != 3 || args.constants.size() != 4)
      return LSE_ERROR(kInvalidArgument, "unexpected host add ABI");
    std::uint32_t count;
    std::memcpy(&count, args.constants.data(), sizeof(count));
    auto data = [&](unsigned i) {
      const auto& ref = args.bindings[i];
      return reinterpret_cast<float*>(static_cast<std::byte*>(ref.buffer->ptr) +
                                      ref.buffer->offset + ref.offset);
    };
    for (std::uint32_t i = 0; i < count; ++i) data(2)[i] = data(0)[i] + data(1)[i];
    ++launches;
    return OkStatus();
  }
};
}

LSE_TEST(decode_sample_rejects_actual_cold_or_host_work) {
  using lse::runtime::DecodeSampleCounters;
  using lse::runtime::warm_decode_sample;
  DecodeSampleCounters before{3, 4, 5, 6};
  LSE_EXPECT(warm_decode_sample(before, before));
  for (unsigned field = 0; field < 4; ++field) {
    auto after = before;
    if (field == 0) ++after.compiles;
    if (field == 1) ++after.disk_loads;
    if (field == 2) ++after.partition_passes;
    if (field == 3) ++after.host_groups;
    LSE_EXPECT(!warm_decode_sample(before, after));
  }
}

LSE_TEST(decode_sample_retained_replay_does_not_count_a_partition) {
  backend::BackendAdapter<HostDevice> backend;
  LSE_EXPECT_OK(backend.init(0));
  Scheduler scheduler(backend);
  scheduler.set_dialect(Dialect::kLoom);
  auto make_leaf = [&](float value) {
    auto buffer = backend.allocate(4 * sizeof(float), backend::MemoryClass::kDevice, backend::kDefaultStream);
    LSE_EXPECT(buffer.ok());
    float values[4]{value, value + 1, value + 2, value + 3};
    LSE_EXPECT_OK(backend.copy_h2d(values, *buffer, sizeof(values), 0));
    return Array::from_buffer(std::move(*buffer), Shape{4}, DType::kF32);
  };
  auto a = make_leaf(1), b = make_leaf(10);
  auto output = a + b;
  const NodePtr roots[]{output.node()};
  Program program;
  LSE_EXPECT_OK(scheduler.eval(roots, true, &program));
  LSE_EXPECT(scheduler.last_trace().partition_passes > 0);
  LSE_EXPECT_EQ(scheduler.last_trace().host_groups, 0u);
  const auto partitions = scheduler.accumulated_trace().partition_passes;
  LSE_EXPECT(program.holds(roots));
  for (unsigned pass = 0; pass < 4; ++pass) {
    float input[]{float(20 + pass), 2, 3, 4};
    LSE_EXPECT_OK(backend.copy_h2d(input, a.node()->buffer, sizeof(input), 0));
    program.reset_compute();
    LSE_EXPECT_OK(scheduler.eval(roots, true, &program));
    LSE_EXPECT(scheduler.last_trace().replayed);
    LSE_EXPECT_EQ(scheduler.last_trace().partition_passes, 0u);
    LSE_EXPECT_EQ(scheduler.accumulated_trace().partition_passes, partitions);
    LSE_EXPECT_EQ(scheduler.last_trace().host_groups, 0u);
    float result[4]{};
    LSE_EXPECT_OK(backend.copy_d2h(output.node()->buffer, result, sizeof(result), 0));
    LSE_EXPECT_EQ(result[0], float(30 + pass));
    LSE_EXPECT_EQ(result[3], 17.0f);
  }
  LSE_EXPECT_EQ(backend.impl().launches, 5u);
}
LSE_TEST_MAIN()
