#include "harness.hpp"
#include "lse/backends/cpu/cpu_backend.hpp"
#include "lse/backends/hrx/hipc/hip_emitter.hpp"
#include "lse/backends/hrx/loomc/loom_emitter.hpp"
#include "lse/graph/interpreter.hpp"
#include "lse/graph/kernel_primitive.hpp"
#include "lse/graph/ops.hpp"
#include "lse/model/dflash2.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

using namespace lse;
using namespace lse::graph;
namespace {
constexpr std::string_view kWalk = "dflash2.selector_walk.v1";
struct Fixture {
  std::size_t batches = 2, positions = 7, top = 16;
  float vocab = 248320.0f;
  std::vector<float> scores, ids;
  explicit Fixture(std::size_t k = 16)
      : top(k), scores(batches * positions * top * top),
        ids(batches * positions * top) {
    for (std::size_t b = 0; b < batches; ++b)
      for (std::size_t p = 0; p < positions; ++p) {
        for (std::size_t k = 0; k < top; ++k)
          ids[(b * positions + p) * top + k] =
              static_cast<float>(200000 + b * 1000 + p * 100 + top - k);
        for (std::size_t predecessor = 0; predecessor < top; ++predecessor)
          for (std::size_t k = 0; k < top; ++k)
            scores[((b * positions + p) * top + predecessor) * top + k] =
                static_cast<float>((k + predecessor * 3 + p * 5) % top);
      }
  }
  Shape score_shape() const {
    return {static_cast<std::int64_t>(batches), static_cast<std::int64_t>(positions),
            static_cast<std::int64_t>(top), static_cast<std::int64_t>(top)};
  }
  Shape id_shape() const {
    return {static_cast<std::int64_t>(batches), static_cast<std::int64_t>(positions),
            static_cast<std::int64_t>(top)};
  }
  std::array<float, 4> attrs() const {
    return {static_cast<float>(positions), static_cast<float>(top), vocab, 0.0f};
  }
};
const KernelPrimitiveBase* primitive() {
  return dynamic_cast<const KernelPrimitiveBase*>(find_primitive(kWalk));
}
std::vector<std::uint32_t> typed_reference(const Fixture& fx) {
  const auto* kernel = primitive();
  LSE_EXPECT(kernel != nullptr);
  if (!kernel) return {};
  std::vector<std::uint32_t> output(fx.batches * (fx.positions + 1), 0xdeadbeefu);
  const std::array inputs{
      HostTensorView{std::as_bytes(std::span(fx.scores)), fx.score_shape(), DType::kF32},
      HostTensorView{std::as_bytes(std::span(fx.ids)), fx.id_shape(), DType::kF32}};
  HostOutputView out{std::as_writable_bytes(std::span(output)),
                    {static_cast<std::int64_t>(fx.batches),
                     static_cast<std::int64_t>(fx.positions + 1)}, DType::kU32};
  LSE_EXPECT_OK(kernel->eval_cpu_typed(inputs, out, fx.attrs(), {}));
  return output;
}
void compare_oracle(const Fixture& fx, const std::vector<std::uint32_t>& output) {
  LSE_EXPECT_EQ(output.size(), fx.batches * (fx.positions + 1));
  if (output.size() != fx.batches * (fx.positions + 1)) return;
  for (std::size_t b = 0; b < fx.batches; ++b) {
    std::vector<std::uint32_t> ids(fx.positions * fx.top);
    for (std::size_t i = 0; i < ids.size(); ++i)
      ids[i] = static_cast<std::uint32_t>(fx.ids[b * ids.size() + i]);
    const auto scores = std::span(fx.scores).subspan(
        b * fx.positions * fx.top * fx.top, fx.positions * fx.top * fx.top);
    const auto expected = model::dflash2_select_path(
        scores, ids, static_cast<std::uint32_t>(fx.positions),
        static_cast<std::uint32_t>(fx.top));
    LSE_EXPECT_OK(expected.status());
    if (!expected.ok()) return;
    for (std::size_t p = 0; p < fx.positions; ++p)
      LSE_EXPECT_EQ(output[b * (fx.positions + 1) + p], (*expected)[p]);
    LSE_EXPECT_EQ(output[b * (fx.positions + 1) + fx.positions], 0u);
  }
}
Array leaf(Shape shape) {
  auto n = std::make_shared<Node>(); n->set_kind(OpKind::kBuffer);
  n->shape = std::move(shape); n->dtype = DType::kF32; n->materialized = true;
  return Array(n);
}
FusionGroup solo(const Array& output) {
  FusionGroup g; g.nodes = {output.node()}; g.outputs = g.nodes;
  g.inputs = output.node()->inputs; g.anchor = output.node()->kind;
  g.anchor_class = output.node()->fclass; return g;
}
Array upload(backend::IBackend& backend, Shape shape, std::span<const float> values) {
  auto allocation = backend.allocate(values.size_bytes(), backend::MemoryClass::kDevice,
                                     backend::kDefaultStream);
  LSE_EXPECT_OK(allocation.status()); if (!allocation.ok()) return {};
  auto buffer = allocation.release();
  LSE_EXPECT_OK(backend.copy(buffer, values.data(), values.size_bytes()));
  return Array::from_buffer(std::move(buffer), std::move(shape), DType::kF32);
}
}  // namespace

LSE_TEST(walk_matches_current_oracle_across_predecessors_and_partial_waves) {
  for (std::size_t top : {1u, 16u, 33u, 65u}) {
    Fixture fx(top);
    compare_oracle(fx, typed_reference(fx));
  }
}
LSE_TEST(walk_ties_choose_lowest_token_then_first_duplicate_index) {
  Fixture fx;
  std::fill(fx.scores.begin(), fx.scores.end(), -3.0f);
  for (std::size_t b = 0; b < fx.batches; ++b) {
    const auto base = b * fx.positions * fx.top;
    fx.ids[base + 4] = 1.0f; fx.ids[base + 11] = 1.0f;
    // Duplicate ids must retain predecessor index4, rather than index11.
    fx.scores[((b * fx.positions + 1) * fx.top + 4) * fx.top + 6] = 99.0f;
    fx.scores[((b * fx.positions + 1) * fx.top + 11) * fx.top + 8] = 100.0f;
  }
  compare_oracle(fx, typed_reference(fx));
}
LSE_TEST(walk_rejects_nonfinite_selected_scores_and_all_malformed_ids) {
  for (float invalid : {std::numeric_limits<float>::quiet_NaN(),
                        std::numeric_limits<float>::infinity(),
                        -std::numeric_limits<float>::infinity()}) {
    Fixture fx;
    fx.scores[2] = invalid;
    const auto output = typed_reference(fx);
    LSE_EXPECT_EQ(output[fx.positions], 1u);
    LSE_EXPECT_EQ(output[2 * fx.positions + 1], 0u);
    fx = Fixture{};
    fx.scores[fx.top + 2] = invalid; // Unvisited predecessor rows remain irrelevant.
    compare_oracle(fx, typed_reference(fx));
  }
  for (float invalid : {-1.0f, 0.25f, 248320.0f,
                        std::numeric_limits<float>::quiet_NaN(),
                        std::numeric_limits<float>::infinity()}) {
    Fixture fx;
    fx.ids[(fx.positions - 1) * fx.top + 3] = invalid;
    const auto output = typed_reference(fx);
    LSE_EXPECT_EQ(output[fx.positions] & 2u, 2u);
    LSE_EXPECT_EQ(output[2 * fx.positions + 1], 0u);
  }
}
LSE_TEST(walk_typed_cpu_graph_reads_device_only_inputs_and_preserves_u32_output) {
  backend::BackendAdapter<backend::CpuBackend> backend;
  LSE_EXPECT_OK(backend.init(0));
  Fixture fx;
  auto scores = upload(backend, fx.score_shape(), fx.scores);
  auto ids = upload(backend, fx.id_shape(), fx.ids);
  auto output = custom(kWalk, {scores, ids}, fx.attrs());
  LSE_EXPECT_OK(output.status()); if (!output.ok()) return;
  LSE_EXPECT(output->dtype() == DType::kU32);
  LSE_EXPECT_OK(interpreter::evaluate(output->node(), backend));
  std::vector<std::uint32_t> words(fx.batches * (fx.positions + 1));
  LSE_EXPECT_OK(interpreter::read_raw(*output->node(), words.data(), words.size() * 4));
  compare_oracle(fx, words);
  auto invalid = output->node(); invalid->attrs[2] = 0.5f;
  invalid->materialized = false;
  LSE_EXPECT(!interpreter::evaluate(invalid, backend).ok());
}
LSE_TEST(walk_emits_typed_terminal_wave_plan_in_both_backends_and_declines_bad_geometry) {
  Fixture fx;
  auto output = custom(kWalk, {leaf(fx.score_shape()), leaf(fx.id_shape())}, fx.attrs());
  LSE_EXPECT_OK(output.status()); if (!output.ok()) return;
  backend::DeviceInfo gpu; gpu.arch = "gfx1201"; gpu.wavefront_size = 32;
  gpu.max_threads_per_workgroup = 1024; gpu.lds_bytes_per_workgroup = 65536;
  backend::LoomEmitter loom; backend::HipEmitter hip;
  for (auto* emitter : {static_cast<IKernelEmitter*>(&loom), static_cast<IKernelEmitter*>(&hip)}) {
    const auto group = solo(*output);
    const auto key = emitter->cache_key(group, gpu);
    const auto emitted = emitter->emit(group, gpu);
    LSE_EXPECT_OK(emitted.status()); if (!emitted.ok()) continue;
    LSE_EXPECT_EQ(emitted->dims.workgroup_size[0], 32u);
    LSE_EXPECT_EQ(emitted->dims.workgroup_count[0], 2u);
    LSE_EXPECT_EQ(emitted->lds_bytes, 0u);
    LSE_EXPECT_EQ(emitted->binding_order.size(), 3u);
    LSE_EXPECT(emitted->source.find(emitter->dialect() == Dialect::kLoom
        ? "view.store" : "unsigned int* __restrict__ out") != std::string::npos);
    gpu.max_threads_per_workgroup = 31;
    LSE_EXPECT(key != emitter->cache_key(group, gpu));
    LSE_EXPECT(!emitter->emit(group, gpu).ok());
    gpu.max_threads_per_workgroup = 1024;
    LSE_EXPECT_EQ(key, emitter->cache_key(group, gpu));
    LSE_EXPECT_OK(emitter->emit(group, gpu).status());
  }
  const auto* kernel = primitive();
  LSE_EXPECT(kernel != nullptr); if (!kernel) return;
  const std::array invalid{Shape{2, 7, 15, 16}, fx.id_shape()};
  LSE_EXPECT(!kernel->infer_shape(invalid).ok());
}

namespace {
int gpu_walk() {
  auto* scheduler = default_scheduler();
  if (!scheduler) return 1;
  scheduler->set_mode(Scheduler::Mode::kDeviceFirst);
  scheduler->set_dialect(Dialect::kLoom);
  scheduler->reset_accumulated_trace();
  for (int variant = 0; variant < 5; ++variant) {
    Fixture fx;
    if (variant == 1) std::fill(fx.scores.begin(), fx.scores.end(), -0.0f);
    if (variant == 2) fx.scores[1] = std::numeric_limits<float>::quiet_NaN();
    if (variant == 3) fx.ids.back() = 0.25f;
    if (variant == 4) fx.scores[fx.top + 1] = std::numeric_limits<float>::infinity();
    const auto expected = typed_reference(fx);
    auto scores = upload(scheduler->backend(), fx.score_shape(), fx.scores);
    auto ids = upload(scheduler->backend(), fx.id_shape(), fx.ids);
    auto output = custom(kWalk, {scores, ids}, fx.attrs());
    LSE_EXPECT_OK(output.status()); if (!output.ok()) return 1;
    std::vector<std::uint32_t> actual(expected.size(), 0xdeadbeefu);
    LSE_EXPECT_OK(output->to_host(actual.data(), actual.size() * 4));
    LSE_EXPECT(actual == expected);
    std::vector<float> preserved(fx.ids.size());
    LSE_EXPECT_OK(ids.to_host(preserved.data(), preserved.size() * 4));
    LSE_EXPECT(std::memcmp(preserved.data(), fx.ids.data(), preserved.size() * 4) == 0);
  }
  const auto trace = scheduler->accumulated_trace();
  LSE_EXPECT(trace.device_groups >= 5u);
  LSE_EXPECT_EQ(trace.host_groups, 0u); LSE_EXPECT_EQ(trace.host_fallbacks, 0u);
  std::printf("DFlash2 greedy typed walk device_groups=%u host_groups=%u host_fallbacks=%u\n",
              trace.device_groups, trace.host_groups, trace.host_fallbacks);
  return lse::test::Registry::get().failures ? 1 : 0;
}
}  // namespace
int main(int argc, char** argv) {
  if (argc == 2 && std::string(argv[1]) == "--gpu-walk") return gpu_walk();
  return lse::test::run_all();
}
