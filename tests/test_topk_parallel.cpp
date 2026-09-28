#include "harness.hpp"
#include "lse/backends/hrx/loomc/loom_emitter.hpp"
#include "lse/graph/kernel_primitive.hpp"
#include "lse/graph/ops.hpp"
#include "lse/graph/program.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

using namespace lse;
using namespace lse::graph;
namespace {
Array filled(Shape shape, const std::vector<float>& values) {
  auto* scheduler = default_scheduler();
  if (!scheduler) return {};
  auto storage = scheduler->backend().allocate(values.size() * sizeof(float), backend::MemoryClass::kDevice);
  if (!storage.ok()) return {};
  auto buffer = storage.release();
  if (!scheduler->backend().copy(buffer, values.data(), values.size() * sizeof(float)).ok()) return {};
  return Array::from_buffer(std::move(buffer), std::move(shape), DType::kF32);
}
std::vector<float> read(Array array) {
  std::vector<float> values(array.shape().elem_count());
  LSE_EXPECT_OK(array.to_host(values.data(), values.size() * sizeof(float)));
  return values;
}
struct Expected { std::vector<float> values, indices; };
Expected reference(const std::vector<float>& input, std::size_t width, std::size_t k, float band = 1.0f) {
  Expected result;
  std::vector<std::size_t> ids(width);
  for (std::size_t row = 0; row < input.size() / width; ++row) {
    for (std::size_t i = 0; i < width; ++i) ids[i] = i;
    std::sort(ids.begin(), ids.end(), [&](std::size_t a, std::size_t b) {
      const float x = input[row * width + a], y = input[row * width + b];
      const bool xn = std::isnan(x), yn = std::isnan(y);
      if (xn != yn) return xn;
      if (x != y && !xn) return x > y;
      return a < b;
    });
    std::vector<float> selected;
    for (std::size_t p = 0; p < k; ++p) {
      selected.push_back(input[row * width + ids[p]]);
      result.indices.push_back(static_cast<float>(ids[p]));
    }
    if (band > 0 && band < 1) {
      const float threshold = selected[0] * (1 - band);
      float sum = 0;
      for (auto& value : selected) { if (value < threshold) value = 0; sum += value; }
      sum += 1e-9f;
      for (auto& value : selected) value /= sum;
    }
    result.values.insert(result.values.end(), selected.begin(), selected.end());
  }
  return result;
}
void verify(const std::vector<float>& values, const std::vector<float>& indices, const Expected& expected) {
  LSE_EXPECT_EQ(values.size(), expected.values.size());
  LSE_EXPECT_EQ(indices.size(), expected.indices.size());
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (std::isnan(expected.values[i])) LSE_EXPECT(std::isnan(values[i]));
    else LSE_EXPECT_EQ(values[i], expected.values[i]);
    LSE_EXPECT_EQ(indices[i], expected.indices[i]);
  }
}
Array legacy_topk(Array input, int k, bool indices) {
  auto node = std::make_shared<Node>();
  node->set_kind(OpKind::kTopK);
  for (std::size_t i = 0; i < input.shape().rank(); ++i)
    node->shape.push_back(i + 1 == input.shape().rank() ? k : input.shape().dim(i));
  node->dtype = DType::kF32; node->inputs = {input.node()};
  ++input.node()->consumer_count;
  node->iattrs[0] = static_cast<std::int32_t>(input.shape().rank() - 1);
  node->iattrs[1] = k; node->iattrs[2] = indices ? 1 : 0;
  node->attrs[0] = 1; node->prim = find_primitive("topk");
  return Array(node);
}
}

LSE_TEST(parallel_topk_matches_finite_ties_and_infinities) {
  constexpr std::size_t width = 4131, rows = 3;
  std::vector<float> input(rows * width);
  for (std::size_t i = 0; i < input.size(); ++i) input[i] = static_cast<float>((i * 37) % 997) * .03125f - 17;
  for (std::size_t row = 0; row < rows; ++row)
    for (std::size_t i : {1u, 511u, 512u, 1025u, 4129u}) input[row * width + i] = 100;
  input[width + 73] = std::numeric_limits<float>::infinity();
  input[width + 1024] = std::numeric_limits<float>::infinity();
  std::fill(input.begin() + 2 * width, input.end(), -std::numeric_limits<float>::infinity());
  for (int k : {1, 3, 16}) {
    Array ids;
    auto values = topk(filled({rows, width}, input), k, -1, &ids);
    LSE_EXPECT(values.node()->prim->name() == "topk.extract");
    verify(read(values), read(ids), reference(input, width, static_cast<std::size_t>(k)));
  }
}
LSE_TEST(parallel_topk_propagates_nan_with_original_index_order) {
  std::vector<float> input(4096, 0);
  input[17] = std::numeric_limits<float>::quiet_NaN();
  input[1027] = std::numeric_limits<float>::quiet_NaN();
  input[4095] = std::numeric_limits<float>::infinity();
  Array ids;
  auto values = topk(filled({4096}, input), 16, -1, &ids);
  verify(read(values), read(ids), reference(input, 4096, 16));
}
LSE_TEST(parallel_topk_score_band_and_small_row_policy) {
  std::vector<float> input(4096, .01f);
  input[12] = .80f; input[3077] = .15f;
  Array ids;
  auto values = topk(filled({4096}, input), 3, -1, &ids, .15f);
  verify(read(values), read(ids), reference(input, 4096, 3, .15f));
  auto small = topk(filled({4}, {.1f, .8f, .2f, .3f}), 2);
  LSE_EXPECT(small.node()->prim->name() == "topk");
}
LSE_TEST(parallel_topk_loom_emits_ragged_three_stage_vocabulary) {
  auto leaf = std::make_shared<Node>(); leaf->shape = {1, 7, 248320};
  leaf->dtype = DType::kF32; leaf->materialized = true;
  Array ids;
  auto values = topk(Array(leaf), 16, -1, &ids);
  const NodePtr roots[] = {values.node(), ids.node()};
  const auto groups = Partitioner::partition(roots);
  backend::DeviceInfo device;
  device.arch = "gfx1201"; device.wavefront_size = 32;
  device.max_threads_per_workgroup = 1024; device.lds_bytes_per_workgroup = 65536;
  backend::LoomEmitter emitter;
  unsigned chunks = 0, extracts = 0;
  for (const auto& group : groups) {
    auto emitted = emitter.emit(group, device);
    LSE_EXPECT(emitted.ok());
    if (!emitted.ok()) { std::fprintf(stderr, "%s\n", emitted.status().to_string().c_str()); continue; }
    const auto name = group.outputs[0]->prim->name();
    if (name == "topk.chunk.v2") {
      const unsigned expected_chunks[] = {485, 16, 1};
      LSE_EXPECT(chunks < 3);
      if (chunks < 3) LSE_EXPECT_EQ(emitted->dims.workgroup_count[0], expected_chunks[chunks]);
      LSE_EXPECT_EQ(emitted->dims.workgroup_count[1], 7u);
      LSE_EXPECT_EQ(emitted->lds_bytes, 32768u);
      LSE_EXPECT(emitted->source.find("kernel.barrier") != std::string::npos);
      ++chunks;
    } else if (name == "topk.extract") ++extracts;
  }
  LSE_EXPECT_EQ(chunks, 3u); LSE_EXPECT_EQ(extracts, 2u);
}

int gpu_topk() {
  auto* scheduler = default_scheduler();
  if (!scheduler) return 1;
  scheduler->set_mode(Scheduler::Mode::kDeviceFirst);
  scheduler->set_dialect(Dialect::kLoom);
  constexpr std::size_t rows = 7, width = 248320, k = 16;
  std::vector<float> input(rows * width);
  for (std::size_t i = 0; i < input.size(); ++i)
    input[i] = std::sin(static_cast<float>(i) * .00317f) * 11;
  for (std::size_t row = 0; row < rows; ++row)
    for (std::size_t i : {1u, 511u, 512u, 1025u, 248319u}) input[row * width + i] = 20;
  const auto expected = reference(input, width, k);
  auto x = filled({1, rows, width}, input);
  for (bool parallel : {false, true}) {
    Array ids;
    auto values = parallel ? topk(x, static_cast<int>(k), -1, &ids) : legacy_topk(x, static_cast<int>(k), false);
    if (!parallel) ids = legacy_topk(x, static_cast<int>(k), true);
    const std::vector<NodePtr> roots{values.node(), ids.node()};
    Program program;
    LSE_EXPECT_OK(scheduler->eval(roots, false, &program));
    LSE_EXPECT_OK(scheduler->drain());
    verify(read(values), read(ids), expected);
    std::vector<double> elapsed;
    scheduler->reset_accumulated_trace();
    for (int iteration = 0; iteration < 5; ++iteration) {
      program.reset_compute();
      const auto begin = std::chrono::steady_clock::now();
      LSE_EXPECT_OK(scheduler->eval(roots, false, &program));
      LSE_EXPECT_OK(scheduler->drain());
      elapsed.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count());
    }
    const auto trace = scheduler->accumulated_trace();
    LSE_EXPECT(trace.device_groups > 0); LSE_EXPECT(trace.kernels_launched > 0);
    LSE_EXPECT_EQ(trace.host_groups, 0u); LSE_EXPECT_EQ(trace.host_fallbacks, 0u);
    verify(read(values), read(ids), expected);
    std::sort(elapsed.begin(), elapsed.end());
    std::printf("topk %s shape=7x248320 k=16 warm_median_ms=%.6f device_groups=%u host_groups=%u host_fallbacks=%u\n",
        parallel ? "parallel" : "legacy", elapsed[elapsed.size() / 2], trace.device_groups, trace.host_groups, trace.host_fallbacks);
  }
  return test::Registry::get().failures ? 1 : 0;
}
int gpu_topk_edges() {
  auto* scheduler = default_scheduler();
  if (!scheduler) return 1;
  scheduler->set_mode(Scheduler::Mode::kDeviceFirst);
  scheduler->set_dialect(Dialect::kLoom);
  constexpr std::size_t rows = 4, width = 4097, k = 16;
  const float infinity = std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  std::vector<float> input(rows * width);
  for (std::size_t i = 0; i < input.size(); ++i)
    input[i] = static_cast<float>((i * 37) % 997) * .03125f - 17;
  input[73] = infinity; input[513] = infinity;
  input[17] = -infinity; input[514] = -infinity;
  for (std::size_t i : {511u, 512u, 1025u, 4096u}) input[i] = 100;
  for (std::size_t i : {511u, 512u, 4096u}) input[width + i] = nan;
  input[width + 1023] = infinity; input[width + 1024] = -infinity;
  std::fill(input.begin() + 2 * width, input.begin() + 3 * width, -infinity);
  std::fill(input.begin() + 3 * width, input.end(), -9);
  for (std::size_t i : {0u, 511u, 512u, 1025u, 2048u, 4096u}) input[3 * width + i] = 7;
  const auto expected = reference(input, width, k);
  Array ids;
  auto values = topk(filled({rows, width}, input), static_cast<int>(k), -1, &ids);
  LSE_EXPECT(values.shape() == Shape{rows,k});
  LSE_EXPECT(ids.shape() == Shape{rows,k});
  const std::vector<NodePtr> roots{values.node(),ids.node()};
  scheduler->reset_accumulated_trace();
  LSE_EXPECT_OK(scheduler->eval(roots,false));
  LSE_EXPECT_OK(scheduler->drain());
  const auto trace = scheduler->accumulated_trace();
  LSE_EXPECT(trace.device_groups > 0); LSE_EXPECT(trace.kernels_launched > 0);
  LSE_EXPECT_EQ(trace.host_groups,0u); LSE_EXPECT_EQ(trace.host_fallbacks,0u);
  verify(read(values),read(ids),expected);
  std::printf("topk edges shape=4x4097 k=16 device_groups=%u host_groups=%u host_fallbacks=%u\n",
      trace.device_groups,trace.host_groups,trace.host_fallbacks);
  return test::Registry::get().failures ? 1 : 0;
}
int main(int argc, char** argv) {
  if (argc == 2 && std::string(argv[1]) == "--gpu-topk") return gpu_topk();
  if (argc == 2 && std::string(argv[1]) == "--gpu-topk-edges") return gpu_topk_edges();
  if (auto* scheduler = default_scheduler()) scheduler->set_mode(Scheduler::Mode::kHostOnly);
  return test::run_all();
}
