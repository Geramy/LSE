// Sampler, session cache, and the decode loop.
//
// The load-bearing test here is cached_decode_matches_a_full_forward_pass: a
// KV cache is only correct if stepping one token at a time gives the same
// logits as running the whole sequence at once. Everything else in M7 rests on
// that.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "harness.hpp"
#include "lse/backend/backend.hpp"
#include "lse/backends/hrx/loomc/loom_types.hpp"
#include "lse/dispatch/arch/tuning.hpp"
#include "lse/dispatch/attention_tuneconfig.h"
#include "lse/graph/interpreter.hpp"
#include "lse/graph/ops.hpp"
#include "lse/kv/allocator.hpp"
#include "lse/kv/block.hpp"
#include "lse/kv/memory.hpp"
#include "lse/kv/policy.hpp"
#include "lse/model/config.hpp"
#include "lse/ops/attention.hpp"
#include "lse/ops/rope.hpp"
#include "lse/model/hybrid_lm.hpp"
#include "lse/model/lemonseed.hpp"
#include "lse/model/mtp.hpp"
#include "lse/model/registry.hpp"
#include "lse/model/weights.hpp"
#include "lse/runtime/batch.hpp"
#include "lse/runtime/generator.hpp"
#include "lse/runtime/sampler.hpp"
#include "lse/runtime/session.hpp"

using namespace lse;
using namespace lse::runtime;

namespace {

std::string model_dir() {
  if (const char* p = std::getenv("LSE_TEST_MODEL")) return p;
  const char* home = std::getenv("HOME");
  return home ? std::string(home) + "/Documents/Dev/LDE/model" : "";
}

bool have_model() {
  std::error_code ec;
  return !model_dir().empty() && std::filesystem::is_directory(model_dir(), ec);
}

}  // namespace

LSE_TEST(fragmented_f16_scalar_attention_matches_contiguous_and_reference) {
  auto* scheduler = graph::default_scheduler();
  LSE_EXPECT(scheduler != nullptr);
  if (!scheduler) return;
  auto& backend = scheduler->backend();
  const auto* toolchain = backend.toolchain(graph::Dialect::kLoom);
  if (!backend.emitter() || !toolchain)
    LSE_SKIP("requires native Loom attention dispatch");
  struct RestoreScheduler {
    graph::Scheduler& scheduler;
    graph::Scheduler::Mode mode;
    graph::DialectPreference dialect;
    ~RestoreScheduler() {
      scheduler.set_mode(mode);
      if (dialect) scheduler.set_dialect(*dialect);
      else scheduler.clear_dialect();
    }
  } restore{*scheduler, scheduler->mode(), scheduler->dialect()};
  scheduler->set_mode(graph::Scheduler::Mode::kDeviceFirst);
  scheduler->set_dialect(graph::Dialect::kLoom);
  LSE_EXPECT(scheduler->toolchain(0)->dialect == graph::Dialect::kLoom);
  const auto run = [&]() -> Status {
    std::fprintf(stderr, "       F16 fragmented check: begin arch=%s\n",
        std::string(backend.device_info().arch).c_str());
    constexpr int rows = 2, heads = 4, kvheads = 2, dim = 32, block = 16;
    constexpr int blocks = 256, stride = 256;
    const Shape pool_shape{blocks, kvheads, block, dim};
    const auto bytes = pool_shape.elem_count() * sizeof(std::uint16_t);
    std::vector<std::uint16_t> keys(pool_shape.elem_count()), values(keys.size());
    for (std::size_t i = 0; i < keys.size(); ++i) {
      keys[i] = float16_t::from_float(static_cast<float>(static_cast<int>(i % 97) - 48) / 64.f);
      values[i] = float16_t::from_float(static_cast<float>(static_cast<int>(i % 83) - 41) / 64.f);
    }
    auto upload = [&](const Shape& shape, DType dtype, const void* data) -> Result<graph::Array> {
      const auto size = dtype_storage_bytes(dtype, shape.elem_count());
      LSE_ASSIGN_OR(auto buffer, backend.allocate(size + 32, backend::MemoryClass::kDevice));
      LSE_ASSIGN_OR(auto base_address, backend.device_pointer(buffer));
      buffer.offset = 32;
      buffer.size_bytes = size;
      LSE_ASSIGN_OR(auto view_address, backend.device_pointer(buffer));
      LSE_EXPECT_EQ(reinterpret_cast<std::uintptr_t>(view_address),
                    reinterpret_cast<std::uintptr_t>(base_address) + 32);
      std::fprintf(stderr, "       upload bytes=%zu address=%p offset=%zu\n",
          size, view_address, buffer.offset);
      LSE_RETURN_IF_ERROR(backend.copy_h2d(data, buffer, size, 0));
      return graph::Array::from_buffer(std::move(buffer), shape, dtype);
    };
    LSE_ASSIGN_OR(auto contiguous_k, upload(pool_shape, DType::kF16, keys.data()));
    LSE_ASSIGN_OR(auto contiguous_v, upload(pool_shape, DType::kF16, values.data()));
    auto manager = kv::MemoryManager::create();
    auto key_storage = std::make_shared<kv::FragmentStorage>(manager, backend, backend::kDefaultStream);
    auto value_storage = std::make_shared<kv::FragmentStorage>(manager, backend, backend::kDefaultStream);
    // Each pool spans two fragments; interleave them to prohibit assuming
    // that consecutive logical fragments occupy consecutive GPU addresses.
    std::fprintf(stderr, "       F16 fragmented check: grow pools\n");
    LSE_RETURN_IF_ERROR(key_storage->grow(kv::kFragmentBytes));
    LSE_RETURN_IF_ERROR(value_storage->grow(kv::kFragmentBytes));
    LSE_RETURN_IF_ERROR(key_storage->grow(bytes));
    LSE_RETURN_IF_ERROR(value_storage->grow(bytes));
    std::fprintf(stderr, "       F16 fragmented check: initialize pools\n");
    LSE_RETURN_IF_ERROR(key_storage->write(keys.data(), bytes));
    LSE_RETURN_IF_ERROR(value_storage->write(values.data(), bytes));
    LSE_ASSIGN_OR(auto key_binding, key_storage->binding());
    LSE_ASSIGN_OR(auto value_binding, value_storage->binding());
    auto fragmented_k = graph::Array::from_buffer(std::move(key_binding), pool_shape, DType::kF16);
    auto fragmented_v = graph::Array::from_buffer(std::move(value_binding), pool_shape, DType::kF16);
    fragmented_k.node()->kv_fragments = key_storage;
    fragmented_v.node()->kv_fragments = value_storage;
    std::vector<std::uint64_t> key_addresses(key_storage->fragment_count());
    LSE_RETURN_IF_ERROR(backend.copy_d2h(fragmented_k.node()->buffer,
        key_addresses.data(), key_addresses.size() * sizeof(std::uint64_t), 0));
    for (std::size_t i = 0; i < key_addresses.size(); ++i)
      std::fprintf(stderr, "       K fragment[%zu]=0x%llx\n", i,
          static_cast<unsigned long long>(key_addresses[i]));
    std::vector<float> table_values(rows * stride);
    for (int row = 0; row < rows; ++row)
      for (int i = 0; i < stride; ++i)
        table_values[row * stride + i] = static_cast<float>((i * 13 + 7 + row * 126) % blocks);
    LSE_ASSIGN_OR(auto table, upload({rows, stride}, DType::kF32, table_values.data()));
    const float write_values[]{255, 258, rows, 255, 258, 127, 130};
    LSE_ASSIGN_OR(auto write_meta, upload({7}, DType::kF32, write_values));
    std::vector<float> additions(rows * kvheads * 3 * dim);
    for (std::size_t i = 0; i < additions.size(); ++i)
      additions[i] = static_cast<float>(static_cast<int>(i % 19) - 9) / 32.f;
    LSE_ASSIGN_OR(auto source, upload({rows, kvheads, 3, dim}, DType::kF32, additions.data()));
    for (int row = 0; row < rows; ++row)
      for (int h = 0; h < kvheads; ++h)
        for (int t = 0; t < 3; ++t)
          for (int d = 0; d < dim; ++d) {
            const int position = (row == 0 ? 255 : 127) + t;
            const auto physical = static_cast<int>(table_values[row * stride + position / block]);
            const auto at = static_cast<std::size_t>(((physical * kvheads + h) * block + position % block) * dim + d);
            keys[at] = float16_t::from_float(additions[((row * kvheads + h) * 3 + t) * dim + d]);
          }
    std::vector<float> query(rows * heads * dim);
    for (std::size_t i = 0; i < query.size(); ++i)
      query[i] = static_cast<float>(static_cast<int>(i % 71) - 35) / 64.f;
    LSE_ASSIGN_OR(auto q, upload({rows, heads, 1, dim}, DType::kF32, query.data()));
    const float metadata[]{272, 273, rows, 272, 273, 129, 130};
    LSE_ASSIGN_OR(auto meta, upload({7}, DType::kF32, metadata));
    std::vector<float> contiguous;
    for (bool fragmented : {false, true}) {
      // Q=1 and D=32 deliberately select scalar attention on both gfx11 and
      // gfx12, independently of short-query matrix/split specialization.
      auto written = graph::kv_page_write(fragmented ? fragmented_k : contiguous_k,
          source, write_meta, table, block, kv::CacheDType::kF16);
      auto output = graph::sdpa_paged(q, written,
          fragmented ? fragmented_v : contiguous_v, .125f,
          graph::MaskKind::kCausal, 0, meta, table, block,
          &backend.device_info(), kv::CacheDType::kF16);
      std::fprintf(stderr, "       F16 fragmented check: eval fragmented=%d\n", fragmented);
      LSE_RETURN_IF_ERROR(output.eval());
      const auto trace = scheduler->last_trace();
      LSE_EXPECT(trace.device_groups >= 2);
      LSE_EXPECT_EQ(trace.host_groups, 0u);
      LSE_EXPECT_EQ(trace.host_fallbacks, 0u);
      std::vector<float> got(query.size());
      LSE_RETURN_IF_ERROR(output.to_host(got.data(), got.size() * sizeof(float)));
      std::fprintf(stderr, "       F16 fragmented check: read output fragmented=%d\n", fragmented);
      std::vector<std::uint16_t> actual(keys.size());
      if (fragmented) LSE_RETURN_IF_ERROR(key_storage->read(actual.data(), bytes));
      else LSE_RETURN_IF_ERROR(backend.copy_d2h(contiguous_k.node()->buffer, actual.data(), bytes, 0));
      if (actual != keys) {
        const auto mismatch = std::mismatch(actual.begin(), actual.end(), keys.begin());
        std::fprintf(stderr, "       F16 page write fragmented=%d index=%zu actual_bits=%u expected_bits=%u\n",
            fragmented, static_cast<std::size_t>(mismatch.first - actual.begin()),
            static_cast<unsigned>(*mismatch.first), static_cast<unsigned>(*mismatch.second));
        std::size_t unchanged = 0, changed = 0, bad = 0;
        for (std::size_t i = 0; i < actual.size(); ++i) {
          const auto original = float16_t::from_float(
              static_cast<float>(static_cast<int>(i % 97) - 48) / 64.f);
          if (actual[i] == original) ++unchanged;
          else if (changed++ < 8)
            std::fprintf(stderr, "       changed K[%zu]=%u expected=%u original=%u\n",
                i, static_cast<unsigned>(actual[i]), static_cast<unsigned>(keys[i]),
                static_cast<unsigned>(original));
          bad += actual[i] != keys[i];
        }
        std::fprintf(stderr, "       write changed=%zu unchanged=%zu incorrect=%zu\n",
            changed, unchanged, bad);
        LSE_EXPECT(actual == keys);
      }
      LSE_EXPECT_EQ(got.size(), query.size());
      if (got.size() != query.size()) return LSE_ERROR(kInternal, "incomplete attention output");
      double worst = 0;
      std::size_t worst_index = 0;
      double worst_expected = 0;
      auto decode = [](std::uint16_t bits) { float16_t value; value.bits = bits; return value.to_float(); };
      for (int row = 0; row < rows; ++row)
        for (int h = 0; h < heads; ++h) {
          const int live = row == 0 ? 273 : 130;
          std::vector<double> scores(static_cast<std::size_t>(live));
          double maximum = -std::numeric_limits<double>::infinity();
          auto at = [&](int key, int d) {
            const int physical = static_cast<int>(table_values[row * stride + key / block]);
            return static_cast<std::size_t>(((physical * kvheads + h / (heads / kvheads)) * block + key % block) * dim + d);
          };
          for (int key = 0; key < live; ++key) {
            double score = 0;
            for (int d = 0; d < dim; ++d)
              score += static_cast<double>(query[(row * heads + h) * dim + d]) * decode(keys[at(key, d)]);
            maximum = std::max(maximum, scores[key] = score * .125);
          }
          double denominator = 0;
          for (auto& score : scores) { score = std::exp(score - maximum); denominator += score; }
          for (int d = 0; d < dim; ++d) {
            double numerator = 0;
            for (int key = 0; key < live; ++key) numerator += scores[key] * decode(values[at(key, d)]);
            const double expected = numerator / denominator;
            const auto index = static_cast<std::size_t>((row * heads + h) * dim + d);
            LSE_EXPECT(std::isfinite(got[index]));
            const double error = std::abs(static_cast<double>(got[index]) - expected);
            if (error > worst) { worst = error; worst_index = index; worst_expected = expected; }
          }
        }
      std::fprintf(stderr, "       F16 scalar arch=%s fragmented=%d max_abs=%.9g index=%zu actual=%.9g expected=%.9g\n",
          std::string(backend.device_info().arch).c_str(), fragmented, worst, worst_index,
          static_cast<double>(got[worst_index]), worst_expected);
      LSE_EXPECT(worst < 2e-6);
      if (!fragmented) contiguous = got;
      else LSE_EXPECT(std::memcmp(contiguous.data(), got.data(), got.size() * sizeof(float)) == 0);
    }
    return OkStatus();
  };
  LSE_EXPECT_OK(run());
}

LSE_TEST(native_partial_rope_to_paged_keys_matches_reference) {
  auto* scheduler = graph::default_scheduler();
  if (!scheduler || !scheduler->backend().emitter())
    LSE_SKIP("requires native kernel dispatch");
  auto& backend = scheduler->backend();
  struct Restore {
    graph::Scheduler& scheduler;
    graph::Scheduler::Mode mode;
    graph::DialectPreference dialect;
    ~Restore() {
      scheduler.set_mode(mode);
      if (dialect) scheduler.set_dialect(*dialect);
      else scheduler.clear_dialect();
    }
  } restore{*scheduler, scheduler->mode(), scheduler->dialect()};
  scheduler->set_mode(graph::Scheduler::Mode::kDeviceFirst);
  for (const auto dialect : {graph::Dialect::kHip, graph::Dialect::kLoom}) {
    const auto* toolchain = backend.toolchain(dialect);
    if (!toolchain || toolchain->dialect != dialect || !toolchain->compiler ||
        !toolchain->compiler->available()) continue;
    scheduler->set_dialect(dialect);
    const auto run = [&]() -> Status {
      auto upload = [&](Shape shape, DType dtype, const void* data) -> Result<graph::Array> {
        const auto bytes = dtype_storage_bytes(dtype, shape.elem_count());
        LSE_ASSIGN_OR(auto buffer, backend.allocate(bytes, backend::MemoryClass::kDevice));
        LSE_RETURN_IF_ERROR(backend.copy_h2d(data, buffer, bytes, 0));
        return graph::Array::from_buffer(std::move(buffer), shape, dtype);
      };
      constexpr int dim = 16, rotary = 4, block = 16;
      LSE_ASSIGN_OR(auto angles, ops::build_rope(rotary, block, 10000000.f));
      for (const int heads : {1, 2, 4}) for (const int queries : {1, 4}) {
        constexpr int first = 4;
        const int live = first + queries;
        const float positions[]{first, static_cast<float>(live), 1, first, static_cast<float>(live)};
        const float block_id[]{0};
        LSE_ASSIGN_OR(auto meta, upload({5}, DType::kF32, positions));
        LSE_ASSIGN_OR(auto table, upload({1, 1}, DType::kF32, block_id));
        const Shape shape{1, heads, queries, dim};
        std::vector<float> input(shape.elem_count()), gain(dim, 1.f);
        for (std::size_t i = 0; i < input.size(); ++i)
          input[i] = static_cast<float>(static_cast<int>(i % 29) - 14) / 32.f;
        LSE_ASSIGN_OR(auto x, upload(shape, DType::kF32, input.data()));
        LSE_ASSIGN_OR(auto weight, upload({dim}, DType::kF32, gain.data()));
        auto norm = graph::rms_norm(x, weight, 1e-6f, false);
        LSE_ASSIGN_OR(auto rotated, ops::apply_rope(norm, angles, meta));
        std::vector<std::uint16_t> expected(heads * block * dim);
        LSE_ASSIGN_OR(auto pool, upload({1, heads, block, dim}, DType::kF16, expected.data()));
        auto written = graph::kv_page_write(pool, rotated, meta, table, block, kv::CacheDType::kF16);
        LSE_RETURN_IF_ERROR(written.eval());
        const auto trace = scheduler->last_trace();
        LSE_EXPECT(trace.device_groups > 0);
        LSE_EXPECT_EQ(trace.host_groups, 0u);
        LSE_EXPECT_EQ(trace.host_fallbacks, 0u);
        for (int h = 0; h < heads; ++h) for (int t = 0; t < queries; ++t) {
          const auto base = (h * queries + t) * dim;
          double squares = 0;
          for (int d = 0; d < dim; ++d) squares += static_cast<double>(input[base + d]) * input[base + d];
          const float scale = static_cast<float>(1.0 / std::sqrt(squares / dim + 1e-6));
          for (int d = 0; d < dim; ++d) {
            float value = input[base + d] * scale;
            if (d < rotary) {
              const int pair = d / 2 * 2;
              const double angle = (first + t) / std::pow(10000000.0, static_cast<double>(pair / 2) / (rotary / 2));
              const float c = static_cast<float>(std::cos(angle)), s = static_cast<float>(std::sin(angle));
              const float a = input[base + pair] * scale, b = input[base + pair + 1] * scale;
              value = d % 2 ? b * c + a * s : a * c - b * s;
            }
            expected[(h * block + first + t) * dim + d] = float16_t::from_float(value);
          }
        }
        std::vector<std::uint16_t> got(expected.size());
        LSE_RETURN_IF_ERROR(backend.copy_d2h(written.node()->buffer, got.data(), got.size() * sizeof(got[0]), 0));
        auto decode = [](std::uint16_t bits) { float16_t v; v.bits = bits; return v.to_float(); };
        double worst = 0; std::size_t at = 0;
        for (std::size_t i = 0; i < got.size(); ++i) {
          LSE_EXPECT(std::isfinite(decode(got[i])));
          const double error = std::abs(static_cast<double>(decode(got[i])) - decode(expected[i]));
          if (error > worst) { worst = error; at = i; }
        }
        std::fprintf(stderr, "       %s partial RoPE to K heads=%d queries=%d max_abs=%.9g index=%zu actual=%.9g expected=%.9g device=%u host=%u\n",
            dialect == graph::Dialect::kHip ? "hip" : "loom", heads, queries, worst, at,
            decode(got[at]), decode(expected[at]), trace.device_groups, trace.host_groups);
        LSE_EXPECT(worst < .002);
      }
      return OkStatus();
    };
    LSE_EXPECT_OK(run());
  }
}

LSE_TEST(native_f16_paged_prefill_and_decode_match_reference) {
  auto* scheduler = graph::default_scheduler();
  if (!scheduler || !scheduler->backend().emitter())
    LSE_SKIP("requires native kernel dispatch");
  auto& backend = scheduler->backend();
  struct Restore {
    graph::Scheduler& scheduler;
    graph::Scheduler::Mode mode;
    graph::DialectPreference dialect;
    ~Restore() {
      scheduler.set_mode(mode);
      if (dialect) scheduler.set_dialect(*dialect);
      else scheduler.clear_dialect();
    }
  } restore{*scheduler, scheduler->mode(), scheduler->dialect()};
  scheduler->set_mode(graph::Scheduler::Mode::kDeviceFirst);
  for (const auto dialect : {graph::Dialect::kHip, graph::Dialect::kLoom}) {
    const auto* toolchain = backend.toolchain(dialect);
    if (!toolchain || toolchain->dialect != dialect || !toolchain->compiler ||
        !toolchain->compiler->available()) continue;
    scheduler->set_dialect(dialect);
    const auto run = [&]() -> Status {
      // Exceed the matrix attention width limit so both compilers execute the
      // same FP32 scalar reduction even for the four-query prefill.
      constexpr int heads = 4, kvheads = 2, dim = 528, block = 16, blocks = 2;
      const Shape pool_shape{blocks, kvheads, block, dim};
      std::vector<std::uint16_t> keys(pool_shape.elem_count()), values(keys.size());
      auto upload = [&](Shape shape, DType dtype, const void* data) -> Result<graph::Array> {
        const auto bytes = dtype_storage_bytes(dtype, shape.elem_count());
        LSE_ASSIGN_OR(auto buffer, backend.allocate(bytes, backend::MemoryClass::kDevice));
        LSE_RETURN_IF_ERROR(backend.copy_h2d(data, buffer, bytes, 0));
        return graph::Array::from_buffer(std::move(buffer), shape, dtype);
      };
      LSE_ASSIGN_OR(auto key_pool, upload(pool_shape, DType::kF16, keys.data()));
      LSE_ASSIGN_OR(auto value_pool, upload(pool_shape, DType::kF16, values.data()));
      const float block_ids[]{1, 0};
      LSE_ASSIGN_OR(auto table, upload({1, blocks}, DType::kF32, block_ids));
      for (const int first : {0, 4}) {
        const int queries = first == 0 ? 4 : 1;
        const int live = first + queries;
        const float positions[]{static_cast<float>(first), static_cast<float>(live), 1,
                                static_cast<float>(first), static_cast<float>(live)};
        LSE_ASSIGN_OR(auto meta, upload({5}, DType::kF32, positions));
        std::vector<float> q(heads * queries * dim), k(kvheads * queries * dim), v(k.size());
        for (std::size_t i = 0; i < q.size(); ++i)
          q[i] = static_cast<float>(static_cast<int>((i + first) % 31) - 15) / 32.f;
        for (std::size_t i = 0; i < k.size(); ++i) {
          k[i] = static_cast<float>(static_cast<int>((i + first) % 29) - 14) / 32.f;
          v[i] = static_cast<float>(static_cast<int>((i + first) % 23) - 11) / 32.f;
        }
        LSE_ASSIGN_OR(auto query, upload({1, heads, queries, dim}, DType::kF32, q.data()));
        LSE_ASSIGN_OR(auto key_source, upload({1, kvheads, queries, dim}, DType::kF32, k.data()));
        LSE_ASSIGN_OR(auto value_source, upload({1, kvheads, queries, dim}, DType::kF32, v.data()));
        // Keep the producers lazy: model K/V writes consume generated values,
        // and HIP may stage these producers with the in-place stores.
        key_source = key_source + graph::Array::full({1}, DType::kF32, .03125f);
        value_source = value_source - graph::Array::full({1}, DType::kF32, .03125f);
        auto written_k = graph::kv_page_write(key_pool, key_source, meta, table, block, kv::CacheDType::kF16);
        auto written_v = graph::kv_page_write(value_pool, value_source, meta, table, block, kv::CacheDType::kF16);
        auto output = graph::sdpa_paged(query, written_k, written_v, .25f,
            graph::MaskKind::kCausal, 0, meta, table, block, &backend.device_info(), kv::CacheDType::kF16);
        LSE_RETURN_IF_ERROR(output.eval());
        const auto trace = scheduler->last_trace();
        LSE_EXPECT(trace.device_groups > 0);
        LSE_EXPECT_EQ(trace.host_groups, 0u);
        LSE_EXPECT_EQ(trace.host_fallbacks, 0u);
        const auto index = [&](int h, int t, int d) {
          return static_cast<std::size_t>(((static_cast<int>(block_ids[t / block]) * kvheads + h) * block + t % block) * dim + d);
        };
        for (int h = 0; h < kvheads; ++h)
          for (int t = 0; t < queries; ++t)
            for (int d = 0; d < dim; ++d) {
              const auto src = static_cast<std::size_t>((h * queries + t) * dim + d);
              keys[index(h, first + t, d)] = float16_t::from_float(k[src] + .03125f);
              values[index(h, first + t, d)] = float16_t::from_float(v[src] - .03125f);
            }
        std::vector<std::uint16_t> actual(keys.size());
        const auto check_pool = [&](const graph::Array& pool, const auto& expected, const char* label) -> Status {
          LSE_RETURN_IF_ERROR(backend.copy_d2h(pool.node()->buffer, actual.data(), actual.size() * 2, 0));
          if (actual != expected) {
            const auto at = std::mismatch(actual.begin(), actual.end(), expected.begin()).first - actual.begin();
            std::fprintf(stderr, "       %s %s write first=%d index=%zu actual=%u expected=%u\n",
                std::string(to_string(dialect)).c_str(), label, first, static_cast<std::size_t>(at),
                unsigned(actual[at]), unsigned(expected[at]));
          }
          LSE_EXPECT(actual == expected);
          return OkStatus();
        };
        LSE_RETURN_IF_ERROR(check_pool(key_pool, keys, "K"));
        LSE_RETURN_IF_ERROR(check_pool(value_pool, values, "V"));
        std::vector<float> got(q.size());
        LSE_RETURN_IF_ERROR(output.to_host(got.data(), got.size() * sizeof(float)));
        auto decode = [](std::uint16_t bits) { float16_t x; x.bits = bits; return x.to_float(); };
        double worst = 0;
        for (int h = 0; h < heads; ++h)
          for (int t = 0; t < queries; ++t) {
            std::vector<double> scores(static_cast<std::size_t>(first + t + 1));
            double maximum = -std::numeric_limits<double>::infinity();
            for (std::size_t j = 0; j < scores.size(); ++j) {
              double score = 0;
              for (int d = 0; d < dim; ++d)
                score += q[(h * queries + t) * dim + d] * decode(keys[index(h / 2, static_cast<int>(j), d)]);
              maximum = std::max(maximum, scores[j] = score * .25);
            }
            double denominator = 0;
            for (auto& score : scores) { score = std::exp(score - maximum); denominator += score; }
            for (int d = 0; d < dim; ++d) {
              double numerator = 0;
              for (std::size_t j = 0; j < scores.size(); ++j)
                numerator += scores[j] * decode(values[index(h / 2, static_cast<int>(j), d)]);
              const auto at = static_cast<std::size_t>((h * queries + t) * dim + d);
              LSE_EXPECT(std::isfinite(got[at]));
              worst = std::max(worst, std::abs(got[at] - numerator / denominator));
            }
          }
        std::fprintf(stderr, "       %s F16 paged first=%d queries=%d max_abs=%.9g device=%u host=%u\n",
            std::string(to_string(dialect)).c_str(), first, queries, worst,
            unsigned(trace.device_groups), unsigned(trace.host_groups));
        LSE_EXPECT(worst < 2e-6);
      }
      return OkStatus();
    };
    LSE_EXPECT_OK(run());
  }
}

LSE_TEST(greedy_sampling_ignores_every_other_knob) {
  SamplingParams p;
  p.temperature = 0.0f;
  p.top_k = 1;
  p.top_p = 0.1f;
  Sampler s(p);

  std::vector<float> logits{0.1f, 9.0f, 0.2f, 3.0f};
  LSE_EXPECT_EQ(s.sample(logits, {}), 1u);
  LSE_EXPECT_EQ(argmax(logits), 1u);
}

namespace {
// A policy that has measured every verify width: `base` ns for one row and
// `per_row` more for each further row, drafts of `draft` ns, and a long-run
// rate of `rate_tps` tokens per second.
DraftWidthPolicy measured_policy(double base, double per_row, double draft, double rate_tps) {
  DraftWidthPolicy policy;
  for (std::uint64_t i = 0; i < DraftWidthPolicy::kWarmupSteps; ++i)
    policy.observe_verify(DraftWidthPolicy::kMaxRows, 1);  // warm-up: priced nothing
  for (std::uint32_t pass = 0; pass < DraftWidthPolicy::kExploreSamples; ++pass)
    for (std::uint32_t rows = 1; rows <= DraftWidthPolicy::kMaxRows; ++rows)
      policy.observe_verify(rows, static_cast<std::uint64_t>(base + per_row * (rows - 1)));
  if (draft > 0) policy.observe_draft(static_cast<std::uint64_t>(draft));
  policy.observe_step(1, static_cast<std::uint64_t>(1e9 / rate_tps));
  return policy;
}
}  // namespace

LSE_TEST(draft_width_policy_measures_every_width_first) {
  DraftWidthPolicy policy;
  // While the clocks ramp up: the full width, no measuring.
  const std::array<double, 7> unsure{0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
  LSE_EXPECT_EQ(policy.exploring(), 0u);
  LSE_EXPECT_EQ(policy.proposals(unsure), 7u);
  LSE_EXPECT_EQ(policy.depth(7, 3), 3u);
  for (std::uint64_t i = 0; i < DraftWidthPolicy::kWarmupSteps; ++i)
    policy.observe_verify(DraftWidthPolicy::kMaxRows, 99'000'000);  // priced nothing
  // Width 1 is a plain step: no draft at all.
  LSE_EXPECT_EQ(policy.exploring(), 1u);
  LSE_EXPECT(!policy.draft_next());
  const std::array<double, 7> sure{1, 1, 1, 1, 1, 1, 1};
  for (std::uint32_t pass = 0; pass < DraftWidthPolicy::kExploreSamples; ++pass) {
    for (std::uint32_t rows = 1; rows <= DraftWidthPolicy::kMaxRows; ++rows) {
      LSE_EXPECT_EQ(policy.exploring(), rows);
      if (rows > 1) {
        LSE_EXPECT(policy.draft_next());
        LSE_EXPECT_EQ(policy.proposals(sure), rows - 1);
      }
      policy.observe_verify(rows, 30'000'000 + 1'000'000 * rows);
    }
  }
  LSE_EXPECT_EQ(policy.exploring(), 0u);
  LSE_EXPECT_NEAR(policy.verify_ns(8), 38e6, 1.0);
  LSE_EXPECT_NEAR(policy.verify_ns(1), 31e6, 1.0);
}

LSE_TEST(draft_width_policy_verifies_what_pays_for_its_rows) {
  // 32 ms for one row, 1 ms per further row, an 8 ms draft, 50 tok/s.
  DraftWidthPolicy policy = measured_policy(32e6, 1e6, 8e6, 50.0);
  const std::array<double, 7> sure{0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99};
  LSE_EXPECT_EQ(policy.proposals(sure), 7u);
  // Confident first proposals, then a draft that has lost the thread, after
  // the target has shown that such proposals are rarely accepted: a proposal
  // reached with a few percent chance does not buy a millisecond at 50
  // tok/s (0.05 tokens per millisecond).
  for (int i = 0; i < 1000; ++i) {
    policy.observe_acceptance(0.02, 2, i % 50 == 0);
    policy.observe_acceptance(0.99, 0, i % 50 != 0);
  }
  const std::array<double, 7> fading{0.99, 0.99, 0.02, 0.02, 0.02, 0.02, 0.02};
  const std::uint32_t k = policy.proposals(fading);
  LSE_EXPECT(k >= 2 && k <= 3);
  // Free rows: everything with any chance is verified.
  DraftWidthPolicy flat = measured_policy(32e6, 0.0, 8e6, 50.0);
  LSE_EXPECT_EQ(flat.proposals(fading), 7u);
}

LSE_TEST(draft_width_policy_is_a_stopping_rule) {
  // Whether proposal j is verified may depend on the draft's distributions at
  // positions up to j only: confidence after the stopping point never changes
  // the choice. That is what keeps rejection sampling exact.
  DraftWidthPolicy policy = measured_policy(32e6, 2e6, 8e6, 40.0);
  std::mt19937_64 rng(7);
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  for (int trial = 0; trial < 2000; ++trial) {
    std::array<double, 7> c{};
    for (double& v : c) v = unit(rng);
    const std::uint32_t k = policy.proposals(c);
    for (int variant = 0; variant < 4; ++variant) {
      std::array<double, 7> other = c;
      for (std::uint32_t j = k + 1; j < other.size(); ++j) other[j] = unit(rng);
      LSE_EXPECT_EQ(policy.proposals(other), k);
    }
  }
}

LSE_TEST(draft_width_policy_skips_a_draft_that_does_not_pay) {
  DraftWidthPolicy policy = measured_policy(32e6, 1e6, 8e6, 30.0);
  // Drafts whose proposals are almost never right: drafting costs more than
  // it returns, so plain steps follow, with a probing draft every
  // kProbeAfter of them.
  const std::array<double, 1> none{0.0};
  for (int i = 0; i < 32; ++i) policy.observe_draft_value(none, 1);
  std::uint32_t drafted = 0;
  const std::uint32_t steps = 4 * (DraftWidthPolicy::kProbeAfter + 1);
  for (std::uint32_t i = 0; i < steps; ++i) drafted += policy.draft_next() ? 1 : 0;
  LSE_EXPECT_EQ(drafted, 4u);
  // Proposals that are usually right make drafting worth it again.
  const std::array<double, 7> good{0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9};
  for (int i = 0; i < 64; ++i) policy.observe_draft_value(good, 7);
  LSE_EXPECT(policy.draft_next());
  LSE_EXPECT(policy.draft_next());
}

LSE_TEST(draft_width_policy_chains_as_deep_as_pays) {
  // An MTP-like chain: each proposal costs a 2 ms pass, verify rows 1 ms.
  DraftWidthPolicy policy = measured_policy(32e6, 1e6, 0.0, 50.0);
  for (std::uint32_t d = 0; d <= DraftWidthPolicy::kMaxProposals; ++d)
    policy.observe_draft(static_cast<std::uint64_t>(2e6 + 2e6 * d), d);
  // Positions that are almost always accepted: chain all the way.
  for (int i = 0; i < 4000; ++i)
    policy.observe_acceptance(0.999, static_cast<std::uint32_t>(i % 7), i % 50 != 0);
  LSE_EXPECT_EQ(policy.depth(7, 3), 7u);
  LSE_EXPECT_EQ(policy.depth(3, 3), 3u);
  // Positions that are rarely accepted, with 5 ms passes: a 6 ms proposal
  // returning about a tenth of a token loses at 50 tok/s (0.3 tokens).
  DraftWidthPolicy weak = measured_policy(32e6, 1e6, 0.0, 50.0);
  for (std::uint32_t d = 0; d <= DraftWidthPolicy::kMaxProposals; ++d)
    weak.observe_draft(static_cast<std::uint64_t>(2e6 + 5e6 * d), d);
  for (int i = 0; i < 4000; ++i)
    weak.observe_acceptance(0.5, static_cast<std::uint32_t>(i % 7), i % 40 == 0);
  LSE_EXPECT_EQ(weak.depth(7, 3), 0u);
}

LSE_TEST(draft_width_policy_follows_a_change_of_text_within_steps) {
  // Prose: proposals of confidence 0.8 accepted half the time. Then code:
  // the same confidence accepted 95% of the time. Within a few dozen checks
  // the estimate has moved most of the way, before the bin itself has.
  DraftWidthPolicy policy;
  for (int i = 0; i < 3000; ++i) policy.observe_acceptance(0.8, 0, i % 2 == 0);
  const double prose = policy.acceptance(0.8);
  LSE_EXPECT_NEAR(prose, 0.5, 0.05);
  for (int i = 0; i < 60; ++i) policy.observe_acceptance(0.8, 0, i % 20 != 0);
  LSE_EXPECT(policy.acceptance(0.8) > 0.8);
}

LSE_TEST(draft_width_policy_measures_stale_widths_only_when_cheap) {
  // Every width measured, then only the full block for a long stretch.
  const auto run_full_block = [](double accepted) {
    DraftWidthPolicy policy = measured_policy(32e6, 1e6, 8e6, 50.0);
    for (int i = 0; i < 2000; ++i)
      policy.observe_acceptance(0.9, static_cast<std::uint32_t>(i % 7), (i % 100) < accepted * 100);
    for (std::uint64_t i = 0; i <= DraftWidthPolicy::kRefreshSteps; ++i)
      policy.observe_verify(DraftWidthPolicy::kMaxRows, 39'000'000);
    return policy;
  };
  // Code: every narrower width would throw away most of a step.
  LSE_EXPECT_EQ(run_full_block(0.98).refresh(DraftWidthPolicy::kMaxProposals, false), 0u);
  // Prose: some narrower width costs next to nothing, and gets measured.
  const std::uint32_t rows = run_full_block(0.3).refresh(DraftWidthPolicy::kMaxProposals, false);
  LSE_EXPECT(rows >= 1 && rows < DraftWidthPolicy::kMaxRows);
}

LSE_TEST(draft_width_policy_calibrates_confidence_to_acceptance) {
  DraftWidthPolicy policy;
  // Unobserved, a bin believes its own confidence (its middle: 0.925 falls in
  // [0.9, 0.944), a quarter decade of 1 - c).
  LSE_EXPECT_NEAR(policy.acceptance(0.925), 1.0 - std::pow(10.0, -1.125), 1e-12);
  LSE_EXPECT_EQ(DraftWidthPolicy::bin_of(0.0), 0u);
  LSE_EXPECT_EQ(DraftWidthPolicy::bin_of(0.995), 9u);
  LSE_EXPECT_EQ(DraftWidthPolicy::bin_of(1.0), DraftWidthPolicy::kBins - 1);
  // A draft that claims 0.9 but is right half the time is believed at 0.5.
  for (int i = 0; i < 2000; ++i) policy.observe_acceptance(0.91, 0, i % 2 == 0);
  LSE_EXPECT_NEAR(policy.acceptance(0.91), 0.5, 0.02);
  // Other confidences keep their own estimate.
  LSE_EXPECT_NEAR(policy.acceptance(0.9999), 1.0 - std::pow(10.0, -4.125), 1e-3);
}

// A tree policy past warm-up that has measured every tree rung: `base` ns
// at four rows and `per_row` more for each row past four.
DraftWidthPolicy measured_tree_policy(double base, double per_row, double draft) {
  DraftWidthPolicy policy;
  for (std::uint64_t i = 0; i < DraftWidthPolicy::kWarmupSteps; ++i)
    policy.observe_tree_step(DraftWidthPolicy::kTreeRows[0], 1);
  for (std::uint32_t pass = 0; pass < DraftWidthPolicy::kExploreSamples; ++pass) {
    policy.observe_verify(1, static_cast<std::uint64_t>(base));
    for (const std::uint32_t rows : DraftWidthPolicy::kTreeRows)
      policy.observe_tree_step(rows, static_cast<std::uint64_t>(draft + base + per_row * (rows - 4)));
  }
  policy.observe_draft(static_cast<std::uint64_t>(draft));
  policy.observe_step(3, static_cast<std::uint64_t>(60e6));
  return policy;
}

LSE_TEST(draft_width_policy_sizes_trees_by_measured_cost) {
  // Expected accepted nodes of the best b-node tree: strongly diminishing.
  std::vector<double> prefix{0.0};
  for (std::uint32_t b = 1; b <= DraftWidthPolicy::kMaxTreeNodes; ++b)
    prefix.push_back(prefix.back() + 0.8 * std::pow(0.85, b - 1));
  // While warming up, the chain's rung; then every rung is measured first.
  DraftWidthPolicy fresh;
  LSE_EXPECT_EQ(fresh.tree_nodes(prefix), 7u);
  // Rows that cost nothing: the widest tree.
  DraftWidthPolicy flat = measured_tree_policy(40e6, 0.0, 8e6);
  LSE_EXPECT_EQ(flat.tree_nodes(prefix), 31u);
  // Rows that cost a lot: the narrowest.
  DraftWidthPolicy steep = measured_tree_policy(40e6, 8e6, 8e6);
  LSE_EXPECT_EQ(steep.tree_nodes(prefix), 3u);
  // In between, the rung with the most tokens per second.
  DraftWidthPolicy mid = measured_tree_policy(40e6, 0.5e6, 8e6);
  std::uint32_t best = 0;
  double best_rate = 0;
  for (const std::uint32_t rows : DraftWidthPolicy::kTreeRows) {
    const double rate = (1.0 + prefix[rows - 1]) / (8e6 + 40e6 + 0.5e6 * (rows - 4));
    if (rate > best_rate) best_rate = rate, best = rows - 1;
  }
  LSE_EXPECT_EQ(mid.tree_nodes(prefix), best);
  // A short expansion caps the tree.
  LSE_EXPECT_EQ(flat.tree_nodes(std::span(prefix).first(6)), 5u);
}

LSE_TEST(draft_width_policy_prefers_a_chain_that_decodes_faster) {
  std::vector<double> prefix{0.0};
  for (std::uint32_t b = 1; b <= DraftWidthPolicy::kMaxTreeNodes; ++b)
    prefix.push_back(prefix.back() + 0.8 * std::pow(0.85, b - 1));
  DraftWidthPolicy policy = measured_tree_policy(40e6, 1e6, 8e6);
  // The chain's width is measured before it is compared.
  LSE_EXPECT_EQ(policy.tree_nodes(prefix, 6.5, 8), 0u);
  for (std::uint32_t i = 0; i < DraftWidthPolicy::kExploreSamples; ++i)
    policy.observe_verify(8, static_cast<std::uint64_t>(42e6));
  // A top path accepted almost whole (code): the chain, two rows cheaper per
  // pass than the eight-row tree, wins.
  LSE_EXPECT_EQ(policy.tree_nodes(prefix, 6.8, 8), 0u);
  // A doubtful top path (prose): a tree.
  LSE_EXPECT(policy.tree_nodes(prefix, 1.9, 8) > 0u);
  // Without a chain to compare, always a tree.
  LSE_EXPECT(policy.tree_nodes(prefix) > 0u);
}

LSE_TEST(draft_width_policy_calibrates_tree_candidates) {
  DraftWidthPolicy policy;
  // Unobserved, a candidate is believed at its own probability.
  LSE_EXPECT_NEAR(policy.candidate(0.37), 0.37, 1e-12);
  // A draft whose 0.5 candidates are answered one time in five.
  for (int i = 0; i < 4000; ++i) policy.observe_candidate(0.52, i % 5 == 0);
  LSE_EXPECT_NEAR(policy.candidate(0.52), 0.2, 0.02);
  LSE_EXPECT_NEAR(policy.candidate(0.07), 0.07, 1e-12);
}

LSE_TEST(a_draft_side_stopping_rule_keeps_the_target_distribution) {
  // Two positions over a three-token vocabulary. The draft's second
  // conditional depends on its first token; the verifier checks proposal 2
  // only when the draft's second distribution is confident, which depends on
  // the first drafted token but not on the second. Rejection sampling must
  // still emit (t1, t2) with the target's joint distribution.
  const auto dist = [](std::vector<double> p) {
    DiscreteDistribution d;
    d.ids = {0, 1, 2};
    d.probabilities = std::move(p);
    return d;
  };
  const DiscreteDistribution p1 = dist({0.5, 0.3, 0.2});
  const std::array<DiscreteDistribution, 3> p2{dist({0.6, 0.3, 0.1}), dist({0.1, 0.2, 0.7}),
                                               dist({0.3, 0.4, 0.3})};
  const DiscreteDistribution q1 = dist({0.2, 0.5, 0.3});
  const std::array<DiscreteDistribution, 3> q2{dist({0.9, 0.05, 0.05}), dist({0.3, 0.4, 0.3}),
                                               dist({0.05, 0.05, 0.9})};
  std::mt19937_64 rng(2026);
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  const auto draw = [&](const DiscreteDistribution& d) {
    auto t = sample_distribution(d, unit(rng));
    return t.ok() ? *t : 0u;
  };
  std::array<std::array<double, 3>, 3> seen{};
  const int trials = 400000;
  for (int trial = 0; trial < trials; ++trial) {
    const std::uint32_t x1 = draw(q1), x2 = draw(q2[x1]);
    const double top2 = *std::max_element(q2[x1].probabilities.begin(), q2[x1].probabilities.end());
    const bool check_second = top2 > 0.5;
    auto first = verify_proposal(p1, q1, x1, unit(rng), unit(rng));
    if (!first.ok()) { LSE_EXPECT(false); return; }
    std::uint32_t t2 = 0;
    if (first->accepted && check_second) {
      auto second = verify_proposal(p2[first->token], q2[x1], x2, unit(rng), unit(rng));
      if (!second.ok()) { LSE_EXPECT(false); return; }
      t2 = second->token;
    } else {
      t2 = draw(p2[first->token]);
    }
    seen[first->token][t2] += 1.0;
  }
  for (std::uint32_t a = 0; a < 3; ++a)
    for (std::uint32_t b = 0; b < 3; ++b)
      LSE_EXPECT_NEAR(seen[a][b] / trials, p1.probabilities[a] * p2[a].probabilities[b], 0.004);
}

LSE_TEST(top_k_of_one_is_deterministic_whatever_the_seed) {
  SamplingParams p;
  p.temperature = 1.0f;
  p.top_k = 1;
  for (std::uint64_t seed = 0; seed < 8; ++seed) {
    p.seed = seed;
    Sampler s(p);
    std::vector<float> logits{1.0f, 5.0f, 2.0f, 0.0f};
    LSE_EXPECT_EQ(s.sample(logits, {}), 1u);
  }
}

LSE_TEST(the_same_seed_gives_the_same_stream) {
  SamplingParams p;
  p.temperature = 1.0f;
  p.seed = 12345;

  const auto run = [&p] {
    Sampler s(p);
    std::vector<std::uint32_t> out;
    for (int i = 0; i < 32; ++i) {
      std::vector<float> logits{1.0f, 1.2f, 0.9f, 1.1f};
      out.push_back(s.sample(logits, {}));
    }
    return out;
  };
  LSE_EXPECT(run() == run());

  // A different seed must not give the same 32 draws.
  const auto a = run();
  p.seed = 999;
  const auto b = run();
  LSE_EXPECT(a != b);
}

LSE_TEST(top_p_keeps_the_smallest_prefix_reaching_the_threshold) {
  // One token holds ~95% of the mass, so any p <= 0.95 must select only it.
  SamplingParams p;
  p.temperature = 1.0f;
  p.top_p = 0.5f;
  p.seed = 7;
  Sampler s(p);
  for (int i = 0; i < 64; ++i) {
    std::vector<float> logits{10.0f, 0.0f, 0.0f, 0.0f};
    LSE_EXPECT_EQ(s.sample(logits, {}), 0u);
  }
}

LSE_TEST(repetition_penalty_pushes_a_seen_token_down) {
  SamplingParams p;
  p.temperature = 0.0f;  // greedy, so the effect is visible in the choice
  p.repetition_penalty = 2.0f;
  Sampler s(p);

  const std::vector<std::uint32_t> history{0};
  std::vector<float> logits{2.0f, 1.5f};
  // Without the penalty token 0 wins; halved to 1.0 it loses to 1.5.
  LSE_EXPECT_EQ(s.sample(logits, history), 1u);
}

LSE_TEST(repetition_penalty_does_not_promote_negative_logits) {
  // Dividing a negative logit raises it. A sign-blind implementation would
  // *encourage* the repeat it is meant to suppress.
  SamplingParams p;
  p.temperature = 0.0f;
  p.repetition_penalty = 2.0f;
  Sampler s(p);

  const std::vector<std::uint32_t> history{0};
  std::vector<float> logits{-1.0f, -1.2f};
  LSE_EXPECT_EQ(s.sample(logits, history), 1u);
}

LSE_TEST(unfiltered_sampling_excludes_zero_weight_at_rng_endpoints) {
  SamplingParams p;
  p.temperature = 1.0f;
  for (const std::uint64_t seed : {14258097010372255221ull,
                                  2295574122455614247ull}) {
    // The first SplitMix64 uniforms are exactly 0 and 1 - 2^-24.
    p.seed = seed;
    Sampler sampler(p);
    std::vector<float> logits{-1000.0f, 0.0f, -1000.0f};
    LSE_EXPECT_EQ(sampler.sample(logits, {}), 1u);
  }
}

LSE_TEST(filtered_sampling_preserves_the_seeded_stream) {
  SamplingParams p;
  p.temperature = 0.85f;
  p.top_k = 20;
  p.top_p = 0.95f;
  p.seed = 12345;
  Sampler sampler(p);
  std::vector<float> logits(32);
  for (std::size_t i = 0; i < logits.size(); ++i) {
    logits[i] = 0.1f * static_cast<float>((i * 13) % 32) - 2.0f;
  }
  const std::uint32_t expected[] = {
      27, 7, 22, 19, 27, 17, 9, 14, 16, 29, 7, 26, 11, 9, 27, 24,
      14, 26, 27, 4, 27, 29, 4, 14, 17, 12, 27, 19, 22, 21, 17, 29};
  for (const auto id : expected) {
    LSE_EXPECT_EQ(sampler.sample(logits, {}), id);
  }
}

LSE_TEST(unfiltered_sampling_reseed_repeats_the_stream) {
  SamplingParams p;
  p.temperature = 1.0f;
  for (const std::uint64_t seed : {0ull, 12345ull}) {
    p.seed = seed;
    Sampler sampler(p);
    std::vector<float> logits{-2.0f, 1.0f, 0.0f, -0.5f, 0.5f};
    std::vector<std::uint32_t> first;
    for (int i = 0; i < 32; ++i) first.push_back(sampler.sample(logits, {}));
    sampler.reseed(seed);
    std::vector<float> empty;
    LSE_EXPECT_EQ(sampler.sample(empty, {}), 0u);
    for (const auto id : first) {
      LSE_EXPECT_EQ(sampler.sample(logits, {}), id);
    }
  }
}

LSE_TEST(unfiltered_sampling_matches_the_softmax_distribution) {
  SamplingParams p;
  p.temperature = 1.0f;
  p.seed = 12345;
  Sampler sampler(p);
  std::vector<float> logits{-2.0f, 1.0f, 0.0f, -0.5f, 0.5f};
  std::array<std::uint32_t, 5> counts{};
  std::array<double, 5> probabilities{};
  double total = 0.0;
  for (std::size_t i = 0; i < logits.size(); ++i) {
    probabilities[i] = std::exp(static_cast<double>(logits[i]) - 1.0);
    total += probabilities[i];
  }
  constexpr std::uint32_t draws = 20000;
  for (std::uint32_t i = 0; i < draws; ++i) {
    const auto id = sampler.sample(logits, {});
    LSE_EXPECT(id < counts.size());
    if (id < counts.size()) ++counts[id];
  }
  for (std::size_t i = 0; i < counts.size(); ++i) {
    const double expected = probabilities[i] / total;
    const double observed = static_cast<double>(counts[i]) / draws;
    const double bound =
        6.0 * std::sqrt(expected * (1.0 - expected) / draws) + 2e-5;
    LSE_EXPECT(std::abs(observed - expected) <= bound);
  }
}

LSE_TEST(a_session_reports_and_releases_its_cache) {
  SessionStore store(4, /*budget_bytes=*/0);
  Session& a = store.get_or_create("a");
  LSE_EXPECT(a.cache_bytes() == 0u);
  LSE_EXPECT(a.position() == 0);

  a.advance(8);
  LSE_EXPECT(a.position() == 8);
  a.clear();
  LSE_EXPECT(a.position() == 0);
  LSE_EXPECT(a.history().empty());

  LSE_EXPECT(store.find("a") != nullptr);
  LSE_EXPECT(store.find("missing") == nullptr);
  store.erase("a");
  LSE_EXPECT(store.find("a") == nullptr);
}

LSE_TEST(the_store_keeps_sessions_apart) {
  SessionStore store(2, 0);
  store.get_or_create("alice").history().push_back(1);
  store.get_or_create("bob").history().push_back(2);

  LSE_EXPECT_EQ(store.size(), 2u);
  LSE_EXPECT_EQ(store.get_or_create("alice").history().size(), 1u);
  LSE_EXPECT_EQ(store.get_or_create("alice").history()[0], 1u);
  LSE_EXPECT_EQ(store.get_or_create("bob").history()[0], 2u);
}

LSE_TEST(cached_decode_matches_a_full_forward_pass) {
  // Step token by token through a session, then run the same tokens in one
  // pass with no cache. The final logits must agree: if they do not, the cache
  // is feeding attention the wrong keys and every generation is wrong.
  if (!have_model()) return;

  auto paths = model::resolve_model(model_dir());
  if (!paths.ok()) return;
  auto ckpt = model::SafeTensors::open(paths->weights);
  auto cfg = model::Config::from_json_file(paths->config);
  if (!ckpt.ok() || !cfg.ok()) return;

  auto lm = model::make_lemonseed(*cfg);
  model::WeightBinder binder(*ckpt);
  if (!lm->load(binder).ok()) return;

  const std::vector<std::uint32_t> tokens{1, 42, 1337, 7};

  const auto logits_of = [&](const std::vector<std::uint32_t>& ids,
                             std::vector<model::MixerState>* states) {
    graph::Array a = graph::Array::zeros(
        Shape{1, static_cast<std::int64_t>(ids.size())}, DType::kF32);
    std::vector<float> out;
    if (!a.eval().ok()) return out;
    for (std::size_t i = 0; i < ids.size(); ++i) {
      graph::interpreter::store_element(*a.node(), i, static_cast<float>(ids[i]));
    }
    auto h = lm->hidden(a, states, nullptr);
    if (!h.ok()) return out;
    auto last = Generator::last_hidden(h.release());
    if (!last.ok()) return out;
    auto lg = lm->lm_head(*last);
    if (!lg.ok()) return out;
    graph::Array logits = lg.release();
    out.resize(logits.shape().elem_count());
    if (!logits.to_host(out.data(), out.size() * sizeof(float)).ok()) {
      out.clear();
      return out;
    }
    return out;
  };

  // One shot, no cache.
  const std::vector<float> whole = logits_of(tokens, nullptr);
  LSE_EXPECT(!whole.empty());
  if (whole.empty()) return;

  // Same tokens, one at a time, through a session's cache.
  Session session("diff", lm->num_layers());
  std::vector<float> stepped;
  const auto cap = cfg->kv_capacity();
  for (std::uint32_t id : tokens) {
    stepped = logits_of({id}, &session.states());
    LSE_EXPECT(!stepped.empty());
    if (stepped.empty()) return;
    for (const auto& st : session.states()) {
      if (!st.key_cache.valid()) continue;
      // Paged: the pool is [blocks, kv_heads, block_size, head_dim], so the
      // position axis is one block wide and the pool is sized by a rung, never
      // by the engine capacity.
      LSE_EXPECT_EQ(st.key_cache.shape().rank(), 4u);
      LSE_EXPECT_EQ(st.key_cache.shape().dim(2), kv::kBlockSize);
      LSE_EXPECT_EQ(st.value_cache.shape().dim(2), kv::kBlockSize);
      LSE_EXPECT_EQ(st.key_cache.shape().dim(0), kv::kMinPoolBlocks);
      LSE_EXPECT(st.key_cache.shape().dim(0) * kv::kBlockSize < cap);
      LSE_EXPECT_EQ(st.paged.stride(),
                    std::min(kv::blocks_for(cap, kv::kBlockSize),
                             static_cast<std::int32_t>(st.key_cache.shape().dim(0))));
      break;
    }
  }

  double max_abs = 0.0;
  double ref_absmax = 0.0;
  for (std::size_t i = 0; i < whole.size(); ++i) {
    max_abs = std::max(max_abs, std::abs(static_cast<double>(stepped[i] - whole[i])));
    ref_absmax = std::max(ref_absmax, std::abs(static_cast<double>(whole[i])));
  }
  const double max_rel = ref_absmax > 0.0 ? max_abs / ref_absmax : max_abs;
  std::printf("       cached vs whole: max_abs=%.3e max_rel=%.3e over %zu\n",
              max_abs, max_rel, whole.size());

  // Also the decision, not just the numbers. WMMA decode is f16×f16
  // accumulated in f32; a full-seq tile and a padded M=1 tile disagree
  // around 1e-3, which is still the same argmax.
  LSE_EXPECT_EQ(argmax(stepped), argmax(whole));
  LSE_EXPECT(max_rel < 2e-3);
}

LSE_TEST(a_second_decode_step_replays_the_held_program) {
  if (!have_model()) return;

  auto paths = model::resolve_model(model_dir());
  if (!paths.ok()) return;
  auto ckpt = model::SafeTensors::open(paths->weights);
  auto cfg = model::Config::from_json_file(paths->config);
  if (!ckpt.ok() || !cfg.ok()) return;

  auto lm = model::make_lemonseed(*cfg);
  model::WeightBinder binder(*ckpt);
  if (!lm->load(binder).ok()) return;

  graph::Scheduler* sched = graph::default_scheduler();
  LSE_EXPECT(sched != nullptr);
  if (sched == nullptr) return;

  Session session("replay", lm->num_layers());
  const auto run = [&](std::uint32_t id) {
    graph::Array a =
        graph::Array::zeros(Shape{1, 1}, DType::kF32);
    if (!a.eval().ok()) return false;
    graph::interpreter::store_element(*a.node(), 0, static_cast<float>(id));
    auto h = lm->hidden(a, &session.states(), nullptr);
    return h.ok();
  };

  LSE_EXPECT(run(1));
  for (const auto& st : session.states()) {
    if (!st.key_cache.valid()) continue;
    LSE_EXPECT_EQ(st.key_cache.shape().dim(2), kv::kBlockSize);
    break;
  }

  sched->reset_accumulated_trace();
  LSE_EXPECT(run(42));
  LSE_EXPECT(sched->last_trace().replayed);
  LSE_EXPECT(sched->last_trace().partition_ns < 100000ull);

  session.clear();
  sched->reset_accumulated_trace();
  LSE_EXPECT(run(7));
  LSE_EXPECT(!sched->last_trace().replayed);
}

LSE_TEST(graph_argmax_matches_the_host_sampler_argmax) {
  // 5000 elements crosses the 4096-per-chunk partial stage, so both the
  // in-chunk reduce and the cross-chunk combine are exercised. The duplicated
  // maximum checks the tie rule: smallest index, exactly like runtime::argmax.
  const std::int64_t n = 5000;
  std::vector<float> row(static_cast<std::size_t>(n));
  for (std::size_t i = 0; i < row.size(); ++i) {
    row[i] = 3.0f * std::sin(static_cast<float>(i) * 0.7f);
  }
  row[1234] = 9.5f;
  row[4321] = 9.5f;

  const auto pick_of = [](const std::vector<float>& values) -> std::uint32_t {
    graph::Array a = graph::Array::zeros(
        Shape{1, static_cast<std::int64_t>(values.size())}, DType::kF32);
    if (!a.eval().ok()) return 0xffffffffu;
    for (std::size_t i = 0; i < values.size(); ++i) {
      graph::interpreter::store_element(*a.node(), i, values[i]);
    }
    graph::Array pick = graph::argmax(a);
    if (!pick.valid()) return 0xffffffffu;
    auto v = pick.item();
    if (!v.ok()) return 0xffffffffu;
    return static_cast<std::uint32_t>(*v);
  };

  LSE_EXPECT_EQ(pick_of(row), argmax(row));
  LSE_EXPECT_EQ(pick_of(row), 1234u);

  // Maximum inside the trailing, partial chunk.
  row[4321] = 11.0f;
  LSE_EXPECT_EQ(pick_of(row), argmax(row));
  LSE_EXPECT_EQ(pick_of(row), 4321u);

  // All-negative row: nothing beats index ordering on the way down.
  for (float& v : row) v = -std::abs(v) - 1.0f;
  LSE_EXPECT_EQ(pick_of(row), argmax(row));
}

LSE_TEST(softmax_top_matches_the_largest_softmax_probability) {
  // Two rows over three chunks (the last one partial): the in-chunk reduce,
  // the cross-chunk fold, and a chunk of -inf that must add nothing.
  const std::int64_t n = 9000;
  std::vector<float> rows(static_cast<std::size_t>(2 * n));
  for (std::size_t i = 0; i < rows.size(); ++i)
    rows[i] = 2.0f * std::sin(static_cast<float>(i) * 0.37f);
  rows[17] = 12.0f;                       // row 0: one dominant logit
  for (std::int64_t i = 0; i < 4096; ++i)  // row 1: first chunk all -inf
    rows[static_cast<std::size_t>(n + i)] = -std::numeric_limits<float>::infinity();
  rows[static_cast<std::size_t>(n + 8999)] = 4.0f;
  const auto reference = [&](std::int64_t r) {
    const float* x = rows.data() + r * n;
    const double top = *std::max_element(x, x + n);
    double sum = 0.0;
    for (std::int64_t i = 0; i < n; ++i) sum += std::exp(static_cast<double>(x[i]) - top);
    return 1.0 / sum;
  };
  graph::Array a = graph::Array::zeros(Shape{2, n}, DType::kF32);
  LSE_EXPECT_OK(a.eval());
  for (std::size_t i = 0; i < rows.size(); ++i)
    graph::interpreter::store_element(*a.node(), i, rows[i]);
  auto partial = graph::custom("softmax_top.partial", {a},
                               {static_cast<float>(n), 0.0f, 0.0f, 0.0f});
  LSE_EXPECT_OK(partial.status());
  if (!partial.ok()) return;
  LSE_EXPECT(partial->shape() == (Shape{2, 3, 2}));
  auto top = graph::custom("softmax_top.final", {*partial}, {3.0f, 0.0f, 0.0f, 0.0f});
  LSE_EXPECT_OK(top.status());
  if (!top.ok()) return;
  std::vector<float> out(2);
  LSE_EXPECT_OK(top->to_host(out.data(), out.size() * sizeof(float)));
  for (std::int64_t r = 0; r < 2; ++r)
    LSE_EXPECT_NEAR(out[static_cast<std::size_t>(r)], reference(r), 2e-6 * reference(r) + 1e-7);
}

LSE_TEST(greedy_device_decode_matches_the_host_argmax_path) {
  // The device path reads back one f32 index; it must pick exactly the token
  // the host sampler picks from the full logit row, step for step.
  if (!have_model()) return;

  auto paths = model::resolve_model(model_dir());
  if (!paths.ok()) return;
  auto ckpt = model::SafeTensors::open(paths->weights);
  auto cfg = model::Config::from_json_file(paths->config);
  if (!ckpt.ok() || !cfg.ok()) return;

  auto lm = model::make_lemonseed(*cfg);
  model::WeightBinder binder(*ckpt);
  if (!lm->load(binder).ok()) return;

  const std::vector<std::uint32_t> prompt{1, 42, 1337};
  constexpr std::int32_t kSteps = 4;

  const auto logits_of = [&](const std::vector<std::uint32_t>& ids,
                             std::vector<model::MixerState>* states) {
    graph::Array a = graph::Array::zeros(
        Shape{1, static_cast<std::int64_t>(ids.size())}, DType::kF32);
    std::vector<float> out;
    if (!a.eval().ok()) return out;
    for (std::size_t i = 0; i < ids.size(); ++i) {
      graph::interpreter::store_element(*a.node(), i, static_cast<float>(ids[i]));
    }
    auto h = lm->hidden(a, states, nullptr);
    if (!h.ok()) return out;
    auto last = Generator::last_hidden(h.release());
    if (!last.ok()) return out;
    auto lg = lm->lm_head(*last);
    if (!lg.ok()) return out;
    graph::Array logits = lg.release();
    out.resize(logits.shape().elem_count());
    if (!logits.to_host(out.data(), out.size() * sizeof(float)).ok()) out.clear();
    return out;
  };

  Session ref("greedy-ref", lm->num_layers());
  std::vector<std::uint32_t> want;
  std::vector<float> logits = logits_of(prompt, &ref.states());
  LSE_EXPECT(!logits.empty());
  if (logits.empty()) return;
  std::uint32_t next = argmax(logits);
  want.push_back(next);
  for (std::int32_t i = 1; i < kSteps; ++i) {
    logits = logits_of({next}, &ref.states());
    LSE_EXPECT(!logits.empty());
    if (logits.empty()) return;
    next = argmax(logits);
    want.push_back(next);
  }

  SamplingParams sp;
  sp.temperature = 0.0f;  // repetition_penalty stays at its 1.0 no-op
  Generator gen(*lm, sp);
  Session dev("greedy-dev", lm->num_layers());
  GenerationLimits limits;
  limits.max_tokens = kSteps;
  auto got = gen.generate(dev, prompt, limits);
  LSE_EXPECT(got.ok());
  if (!got.ok()) return;
  LSE_EXPECT(*got == want);
}

LSE_TEST(a_continued_session_only_scores_the_new_tokens) {
  if (!have_model()) return;

  auto paths = model::resolve_model(model_dir());
  if (!paths.ok()) return;
  auto ckpt = model::SafeTensors::open(paths->weights);
  auto cfg = model::Config::from_json_file(paths->config);
  if (!ckpt.ok() || !cfg.ok()) return;

  auto lm = model::make_lemonseed(*cfg);
  model::WeightBinder binder(*ckpt);
  if (!lm->load(binder).ok()) return;

  SamplingParams sp;
  sp.temperature = 0.0f;
  Generator gen(*lm, sp);

  Session session("chat", lm->num_layers());
  GenerationLimits limits;
  limits.max_tokens = 2;

  auto first = gen.generate(session, {1, 42, 1337}, limits);
  LSE_EXPECT(first.ok());
  if (!first.ok()) return;
  LSE_EXPECT_EQ(gen.stats().prompt_tokens, 3);

  // The next turn extends what the cache already holds, so only the tail is
  // scored — this is the whole reason the session owns the cache. The tail is
  // everything past position(), which trails history() by one: the last token
  // sampled was never fed back through the model.
  std::vector<std::uint32_t> next = session.history();
  next.push_back(99);
  const auto expected =
      static_cast<std::int32_t>(next.size()) - session.position();
  auto second = gen.generate(session, next, limits);
  LSE_EXPECT(second.ok());
  if (!second.ok()) return;
  std::printf("       continued turn scored %d of %zu prompt token(s)\n",
              gen.stats().prompt_tokens, next.size());
  LSE_EXPECT(gen.stats().prompt_tokens == expected);
  LSE_EXPECT(gen.stats().prompt_tokens < static_cast<std::int32_t>(next.size()));
}

// --- device clock, against whatever device this box actually has ------------

LSE_TEST(the_device_clock_is_named_and_rated_or_cleanly_refused) {
  // The process-wide backend the rest of this suite already runs on, rather
  // than a fresh one: bringing the GPU up is a process-wide act, and a second
  // create+init returns "GPU accelerator already initialized" -- a failure with
  // nothing to do with clocks that would silently skip this whole case.
  //
  // The contract is the same whichever backend answers, and neither half of it
  // is optional: a clock that is claimed is fully specified, a timestamp that
  // is handed over carries a usable clock and moves forward, and anything
  // absent is kUnimplemented rather than a device error or a half-filled
  // struct. It passes today because hrx declines the timestamp, and it keeps
  // passing unchanged on the day hrx can answer -- which is the point of
  // writing it against the contract rather than against today's gap.
  graph::Scheduler* sched = graph::default_scheduler();
  LSE_EXPECT(sched != nullptr);
  if (sched == nullptr) return;
  backend::IBackend& be = sched->backend();
  const std::string name(be.name());

  auto clock = be.device_clock();
  if (clock.ok()) {
    LSE_EXPECT(clock->known());
    LSE_EXPECT(clock->domain != backend::ClockDomain::kUnknown);
    LSE_EXPECT(clock->ticks_per_second > 0);
    LSE_EXPECT(clock->valid_bits > 0);
  } else {
    LSE_EXPECT(clock.status().code() == StatusCode::kUnimplemented);
  }

  // hrx cannot read a tick today, but it can say which counter a tick would be
  // on and how fast that counter runs, and that answer is a live device query
  // rather than a constant. This guards the query: a wrong attribute ordinal
  // returns some neighbouring field's value, a plausible-looking number that
  // nothing else in the system would catch.
  if (name == "hrx") {
    LSE_EXPECT(clock.ok());
    if (clock.ok()) {
      LSE_EXPECT(clock->domain == backend::ClockDomain::kDeviceAgent);
      // A tick rate, which is neither the engine clock in MHz nor the host's
      // in GHz -- both values this could be confused with, and both outside
      // this decade.
      LSE_EXPECT(clock->ticks_per_second >= 10'000'000);
      LSE_EXPECT(clock->ticks_per_second <= 500'000'000);
      LSE_EXPECT_EQ(clock->valid_bits, 64u);
    }
  }
  if (clock.ok()) {
    std::printf("       %s device clock: %s, %llu Hz, %u bits\n", name.c_str(),
                std::string(backend::clock_domain_name(clock->domain)).c_str(),
                static_cast<unsigned long long>(clock->ticks_per_second),
                clock->valid_bits);
  } else {
    std::printf("       %s device clock: declined (%s)\n", name.c_str(),
                clock.status().to_string().c_str());
  }

  auto first = be.sample_device_time();
  if (!first.ok()) {
    LSE_EXPECT(first.status().code() == StatusCode::kUnimplemented);
    std::printf("       %s device timestamp: declined (%s)\n", name.c_str(),
                first.status().to_string().c_str());
    return;
  }

  // A timestamp exists, so it must behave like one: usable clock, moving
  // forward, and a known sleep coming back as that interval in the clock's own
  // units. The band is wide against scheduler slop and still an order of
  // magnitude tighter than the ~10x error a mistaken tick rate produces, which
  // is the failure it is here to catch.
  LSE_EXPECT(first->valid());
  constexpr double kSleepNs = 20'000'000.0;
  std::this_thread::sleep_for(
      std::chrono::nanoseconds(static_cast<std::int64_t>(kSleepNs)));
  auto second = be.sample_device_time();
  LSE_EXPECT(second.ok());
  if (!second.ok()) return;

  auto elapsed = backend::nanoseconds_between(*first, *second);
  LSE_EXPECT(elapsed.ok());
  if (elapsed.ok()) {
    LSE_EXPECT(*elapsed > 0.0);
    LSE_EXPECT_NEAR(*elapsed, kSleepNs, kSleepNs * 0.25);
  }
}


// --- paged KV -----------------------------------------------------------------

LSE_TEST(the_block_pool_refcounts_and_reports_exhaustion) {
  kv::BlockAllocator pool(3);
  LSE_EXPECT_EQ(pool.total(), 3);
  LSE_EXPECT_EQ(pool.free_count(), 3);
  LSE_EXPECT_EQ(pool.used(), 0);

  auto a = pool.acquire();
  LSE_EXPECT_OK(a.status());
  auto b = pool.acquire();
  LSE_EXPECT_OK(b.status());
  LSE_EXPECT_EQ(pool.used(), 2);
  LSE_EXPECT_EQ(*pool.refcount(*a), 1);

  // A second holder of the same block: this is what makes prefix sharing a
  // refcount bump rather than a copy.
  LSE_EXPECT_OK(pool.retain(*a));
  LSE_EXPECT_EQ(*pool.refcount(*a), 2);
  auto first = pool.release(*a);
  LSE_EXPECT_OK(first.status());
  LSE_EXPECT(!*first);  // still held by the other reference
  LSE_EXPECT_EQ(pool.free_count(), 1);
  auto last = pool.release(*a);
  LSE_EXPECT_OK(last.status());
  LSE_EXPECT(*last);
  LSE_EXPECT_EQ(pool.free_count(), 2);

  // Exhaustion names the pool size rather than handing out a stale block.
  LSE_EXPECT_OK(pool.acquire().status());
  LSE_EXPECT_OK(pool.acquire().status());
  auto dry = pool.acquire();
  LSE_EXPECT(!dry.ok());
  LSE_EXPECT(dry.status().code() == StatusCode::kOutOfMemory);
  LSE_EXPECT(dry.status().message().find("all 3 blocks") != std::string::npos);

  // A free block cannot be retained and a held block cannot be double-released.
  kv::BlockAllocator fresh(2);
  LSE_EXPECT(!fresh.retain(0).ok());
  auto one = fresh.acquire();
  LSE_EXPECT_OK(one.status());
  LSE_EXPECT_OK(fresh.release(*one).status());
  LSE_EXPECT(!fresh.release(*one).ok());
  LSE_EXPECT(!fresh.release(7).ok());
}

LSE_TEST(a_block_table_covers_positions_and_locates_them) {
  kv::BlockAllocator pool(8);
  kv::BlockTable table(kv::kBlockSize);
  LSE_EXPECT_OK(pool.cover(table, 1));
  LSE_EXPECT_EQ(table.size(), 1);
  LSE_EXPECT_OK(pool.cover(table, kv::kBlockSize));
  LSE_EXPECT_EQ(table.size(), 1);  // still one block: 16 tokens fit
  LSE_EXPECT_OK(pool.cover(table, kv::kBlockSize + 1));
  LSE_EXPECT_EQ(table.size(), 2);

  auto slot = table.locate(kv::kBlockSize);
  LSE_EXPECT_OK(slot.status());
  LSE_EXPECT_EQ(slot->block, table.blocks()[1]);
  LSE_EXPECT_EQ(slot->offset, 0);
  LSE_EXPECT(!table.locate(2 * kv::kBlockSize).ok());

  // Covering more than the pool holds leaves the table and the pool untouched.
  const std::int32_t held = table.size();
  const std::int32_t free_before = pool.free_count();
  LSE_EXPECT(!pool.cover(table, 4096).ok());
  LSE_EXPECT_EQ(table.size(), held);
  LSE_EXPECT_EQ(pool.free_count(), free_before);

  LSE_EXPECT_OK(pool.release_all(table));
  LSE_EXPECT_EQ(pool.used(), 0);
  LSE_EXPECT(table.empty());

  // The device image: padded rows and unreached slots take the pad block, so
  // every row runs the identical address arithmetic.
  kv::BlockTable one(kv::kBlockSize);
  LSE_EXPECT_OK(pool.cover(one, 2 * kv::kBlockSize));
  std::vector<kv::BlockTable> rows{one};
  std::vector<float> image(2 * 4, -1.0f);
  LSE_EXPECT_OK(kv::write_table_rows(rows, 4, /*pad=*/0, image));
  LSE_EXPECT_EQ(image[0], static_cast<float>(one.blocks()[0]));
  LSE_EXPECT_EQ(image[1], static_cast<float>(one.blocks()[1]));
  LSE_EXPECT_EQ(image[2], 0.0f);
  LSE_EXPECT_EQ(image[4], 0.0f);  // the padded row
  LSE_EXPECT(!kv::write_table_rows(rows, 4, kv::kNoBlock, image).ok());
}

LSE_TEST(the_pool_ladder_bounds_the_shapes_a_context_can_compile) {
  // Rungs, not exact fits: the pool block count is a tensor dimension the JIT
  // keys on.
  LSE_EXPECT_EQ(kv::pool_rung(1, 256), 8);
  LSE_EXPECT_EQ(kv::pool_rung(8, 256), 8);
  LSE_EXPECT_EQ(kv::pool_rung(9, 256), 16);
  LSE_EXPECT_EQ(kv::pool_rung(200, 256), 256);
  LSE_EXPECT_EQ(kv::pool_rung(999, 256), 256);
  LSE_EXPECT_EQ(kv::pool_rung(2048, 16384), 2048);
  LSE_EXPECT_EQ(kv::pool_rung(2049, 16384), 2304);
  LSE_EXPECT_EQ(kv::pool_rung(2305, 16384), 2560);
  LSE_EXPECT_EQ(kv::pool_rung(2305, 2400), 2400);
  // 4096 tokens at 16 per block is 256 blocks, so the whole ladder a session can
  // ever walk is six rungs.
  int rungs = 0;
  for (std::int32_t b = 1; b <= 256; ++b) {
    if (b == 1 || kv::pool_rung(b, 256) != kv::pool_rung(b - 1, 256)) ++rungs;
  }
  LSE_EXPECT_EQ(rungs, 6);
}

LSE_TEST(admission_preempts_the_oldest_and_swaps_while_host_room_lasts) {
  kv::BlockPolicy policy(kv::kBlockSize, /*reserve=*/1, /*host_blocks=*/2);
  kv::BlockAllocator pool(8);

  kv::SequenceDemand want{"new", 0, 3 * kv::kBlockSize, 10};
  LSE_EXPECT_EQ(policy.shortfall(want), 3);
  {
    const kv::Admission a = policy.admit(want, pool, {});
    LSE_EXPECT(a.verdict == kv::Verdict::kAdmit);
    LSE_EXPECT_EQ(a.blocks_needed, 3);
  }

  // Fill the pool, then ask for three more with two older sequences resident.
  for (int i = 0; i < 8; ++i) LSE_EXPECT_OK(pool.acquire().status());
  const std::vector<kv::SequenceDemand> resident{
      {"old", 4, 4 * kv::kBlockSize, 1},
      {"newer", 4, 4 * kv::kBlockSize, 5},
  };
  const kv::Admission a = policy.admit(want, pool, resident);
  LSE_EXPECT(a.verdict == kv::Verdict::kPreempt);
  LSE_EXPECT_EQ(a.preempt.size(), 1u);
  LSE_EXPECT(a.preempt[0].id == "old");  // oldest first
  // 4 blocks needed to land, only 2 of host room: it drops rather than pretending
  // to swap.
  LSE_EXPECT(a.preempt[0].how == kv::Eviction::kDrop);

  kv::BlockPolicy roomy(kv::kBlockSize, 1, /*host_blocks=*/16);
  const kv::Admission b = roomy.admit(want, pool, resident);
  LSE_EXPECT(b.preempt[0].how == kv::Eviction::kSwapOut);

  // Nothing resident and nothing free: not transient, so it refuses.
  const kv::Admission c = policy.admit(want, pool, {});
  LSE_EXPECT(c.verdict == kv::Verdict::kRefuse);
  LSE_EXPECT(c.reason.find("needs 3") != std::string::npos);
}

LSE_TEST(the_batch_ladder_selects_a_bucket_and_refuses_what_fits_none) {
  LSE_EXPECT_EQ(*model::batch_bucket(1), 1);
  LSE_EXPECT_EQ(*model::batch_bucket(2), 2);
  LSE_EXPECT_EQ(*model::batch_bucket(3), 4);
  LSE_EXPECT_EQ(*model::batch_bucket(9), 16);
  LSE_EXPECT_EQ(*model::batch_bucket(16), 16);
  LSE_EXPECT_EQ(*model::batch_bucket(17), 32);
  LSE_EXPECT_EQ(*model::batch_bucket(32), 32);

  auto none = model::batch_bucket(33);
  LSE_EXPECT(!none.ok());
  LSE_EXPECT(none.status().code() == StatusCode::kOutOfRange);
  // The error names the size that did not fit and the top of the ladder.
  LSE_EXPECT(none.status().message().find("batch of 33") != std::string::npos);
  LSE_EXPECT(none.status().message().find("32") != std::string::npos);
  LSE_EXPECT(!model::batch_bucket(0).ok());
}

namespace {

// A device Array holding exactly `v`.
graph::Array filled(Shape shape, const std::vector<float>& v) {
  graph::Array a = graph::Array::zeros(shape, DType::kF32);
  graph::Scheduler* sched = graph::default_scheduler();
  if (sched == nullptr) return {};
  graph::Node& n = *a.node();
  if (!graph::interpreter::ensure_output_buffer(n, sched->backend()).ok()) {
    return {};
  }
  for (std::size_t i = 0; i < v.size(); ++i) {
    graph::interpreter::store_element(n, i, v[i]);
  }
  n.materialized = true;
  if (!graph::interpreter::sync_to_device(n, sched->backend()).ok()) return {};
  return a;
}

std::vector<float> read_all(graph::Array& a) {
  std::vector<float> out(static_cast<std::size_t>(a.shape().elem_count()));
  if (!a.to_host(out.data(), out.size() * sizeof(float)).ok()) out.clear();
  return out;
}

// The step descriptor kv/block.hpp specifies, from one {first query position,
// live KV length} pair per row. A length of 0 is a row holding no sequence.
graph::Array step_meta(
    const std::vector<std::pair<std::int32_t, std::int32_t>>& rows) {
  const auto n = static_cast<std::int32_t>(rows.size());
  std::vector<float> m(
      static_cast<std::size_t>(kv::step_meta_elems(n)), 0.0f);
  std::int32_t pos = 0;
  std::int32_t len = 0;
  std::int32_t live = 0;
  for (std::size_t r = 0; r < rows.size(); ++r) {
    if (rows[r].second == 0) continue;
    const auto at = static_cast<std::size_t>(kv::kStepMetaHeader) +
                    r * static_cast<std::size_t>(kv::kStepMetaPerRow);
    m[at] = static_cast<float>(rows[r].first);
    m[at + 1] = static_cast<float>(rows[r].second);
    pos = std::max(pos, rows[r].first);
    len = std::max(len, rows[r].second);
    live = static_cast<std::int32_t>(r) + 1;
  }
  m[0] = static_cast<float>(pos);
  m[1] = static_cast<float>(len);
  m[2] = static_cast<float>(live);
  return filled(Shape{static_cast<std::int64_t>(m.size())}, m);
}

// Deterministic and not symmetric, so an index mistake shows up as a value
// mistake rather than cancelling out.
float noise(std::size_t i) {
  return static_cast<float>(static_cast<double>((i * 2654435761u) % 2003) /
                                1000.0 -
                            1.0);
}

}  // namespace

LSE_TEST(a_paged_read_matches_a_contiguous_read_exactly) {
  graph::Scheduler* sched = graph::default_scheduler();
  LSE_EXPECT(sched != nullptr);
  if (sched == nullptr) return;

  constexpr std::int64_t kQh = 4;
  constexpr std::int64_t kKvh = 2;
  constexpr std::int64_t kHd = 8;
  constexpr std::int64_t kTq = 1;
  const std::int32_t bs = kv::kBlockSize;

  // Two lengths: a whole number of blocks, and one that leaves the last block
  // part-filled so the guard on the live length is the thing under test.
  for (std::int32_t live : {3 * kv::kBlockSize, 3 * kv::kBlockSize - 3}) {
    const std::int32_t nblk = kv::blocks_for(live, bs);
    const std::int32_t pool_blocks = kv::kMinPoolBlocks;
    const std::int32_t offset = live - static_cast<std::int32_t>(kTq);

    std::vector<float> q(static_cast<std::size_t>(kQh * kTq * kHd));
    for (std::size_t i = 0; i < q.size(); ++i) q[i] = noise(i + 7);

    // Contiguous [1, kvh, live, hd] and a pool [pool_blocks, kvh, bs, hd] with
    // the same logical content, plus junk in the slots past `live` and in the
    // blocks the table does not name.
    std::vector<float> flat(static_cast<std::size_t>(kKvh * live * kHd));
    std::vector<float> flatv(flat.size());
    for (std::size_t i = 0; i < flat.size(); ++i) {
      flat[i] = noise(i + 101);
      flatv[i] = noise(i + 9001);
    }
    std::vector<float> pool(
        static_cast<std::size_t>(pool_blocks * kKvh * bs * kHd), 1e30f);
    std::vector<float> poolv(pool.size(), -1e30f);
    // Logical block b lives in physical block `phys(b)`, which is deliberately
    // not b: an identity table is satisfied by a kernel that ignores the table
    // and walks the pool linearly, which is the whole thing under test. The
    // three values below are distinct for pool_blocks == kMinPoolBlocks.
    const auto phys = [pool_blocks](std::int32_t b) {
      return (b * 5 + 3) % pool_blocks;
    };
    for (std::int32_t blk = 0; blk < nblk; ++blk) {
      for (std::int64_t h = 0; h < kKvh; ++h) {
        for (std::int32_t sl = 0; sl < bs; ++sl) {
          const std::int32_t j = blk * bs + sl;
          if (j >= live) continue;
          for (std::int64_t d = 0; d < kHd; ++d) {
            const auto dst = static_cast<std::size_t>(
                ((phys(blk) * kKvh + h) * bs + sl) * kHd + d);
            const auto src = static_cast<std::size_t>((h * live + j) * kHd + d);
            pool[dst] = flat[src];
            poolv[dst] = flatv[src];
          }
        }
      }
    }

    graph::Array qa = filled(Shape{1, kQh, kTq, kHd}, q);
    graph::Array ka = filled(Shape{1, kKvh, live, kHd}, flat);
    graph::Array va = filled(Shape{1, kKvh, live, kHd}, flatv);
    graph::Array kp = filled(Shape{pool_blocks, kKvh, bs, kHd}, pool);
    graph::Array vp = filled(Shape{pool_blocks, kKvh, bs, kHd}, poolv);

    const std::int32_t stride = 16;
    std::vector<float> table(static_cast<std::size_t>(stride), 0.0f);
    for (std::int32_t i = 0; i < nblk; ++i) table[static_cast<std::size_t>(i)] =
        static_cast<float>(phys(i));
    graph::Array ta = filled(Shape{1, stride}, table);
    graph::Array meta = step_meta({{offset, live}});

    const float scale = 0.125f;
    graph::Array contig = graph::sdpa(qa, ka, va, scale,
                                      graph::MaskKind::kCausal, 0, offset);
    graph::Array pagedo =
        graph::sdpa_paged(qa, kp, vp, scale, graph::MaskKind::kCausal, 0, meta,
                          ta, bs);
    LSE_EXPECT_OK(contig.eval());
    LSE_EXPECT_OK(pagedo.eval());
    const std::vector<float> a = read_all(contig);
    const std::vector<float> b = read_all(pagedo);
    LSE_EXPECT(!a.empty() && a.size() == b.size());
    if (a.size() != b.size()) return;
    std::size_t differ = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
      if (std::memcmp(&a[i], &b[i], sizeof(float)) != 0) ++differ;
    }
    std::printf("       paged vs contiguous at live=%d: %zu of %zu differ\n",
                live, differ, a.size());
    LSE_EXPECT_EQ(differ, 0u);
  }
}

LSE_TEST(a_padded_batch_row_reads_its_own_blocks_and_pad_rows_answer_zero) {
  // The width-invariance case for the batch axis: two rows whose block tables
  // point at different blocks must give different answers, and each must equal
  // what it gives alone. Rows past the live count must answer zero without
  // touching another row's blocks.
  graph::Scheduler* sched = graph::default_scheduler();
  LSE_EXPECT(sched != nullptr);
  if (sched == nullptr) return;

  constexpr std::int64_t kQh = 2;
  constexpr std::int64_t kKvh = 2;
  constexpr std::int64_t kHd = 8;
  const std::int32_t bs = kv::kBlockSize;
  const std::int32_t live = bs;  // one block per row
  const std::int32_t pool_blocks = kv::kMinPoolBlocks;
  const std::int32_t stride = 8;
  const std::int32_t bucket = 4;

  std::vector<float> pool(
      static_cast<std::size_t>(pool_blocks * kKvh * bs * kHd));
  std::vector<float> poolv(pool.size());
  for (std::size_t i = 0; i < pool.size(); ++i) {
    pool[i] = noise(i + 31);
    poolv[i] = noise(i + 555);
  }
  graph::Array kp = filled(Shape{pool_blocks, kKvh, bs, kHd}, pool);
  graph::Array vp = filled(Shape{pool_blocks, kKvh, bs, kHd}, poolv);

  // Row 0 -> block 2, row 1 -> block 5. Rows 2 and 3 are padding and point at
  // block 0, which is a real block neither live row uses.
  std::vector<float> table(static_cast<std::size_t>(bucket * stride), 0.0f);
  table[0] = 2.0f;
  table[static_cast<std::size_t>(stride)] = 5.0f;
  graph::Array ta = filled(Shape{bucket, stride}, table);
  graph::Array meta =
      step_meta({{0, live}, {0, live}, {0, 0}, {0, 0}});

  std::vector<float> q(static_cast<std::size_t>(bucket * kQh * 1 * kHd));
  for (std::size_t i = 0; i < q.size(); ++i) q[i] = noise(i + 77);
  graph::Array qa = filled(Shape{bucket, kQh, 1, kHd}, q);

  graph::Array out = graph::sdpa_paged(qa, kp, vp, 0.25f,
                                      graph::MaskKind::kCausal, 0, meta, ta, bs);
  LSE_EXPECT_OK(out.eval());
  const std::vector<float> got = read_all(out);
  const auto per_row = static_cast<std::size_t>(kQh * kHd);
  LSE_EXPECT_EQ(got.size(), per_row * static_cast<std::size_t>(bucket));
  if (got.size() != per_row * static_cast<std::size_t>(bucket)) return;

  // Row 1 is not row 0. This is the assertion the quant_linear row-offset defect
  // would have failed while still producing fluent text.
  std::size_t same = 0;
  for (std::size_t i = 0; i < per_row; ++i) {
    if (got[i] == got[per_row + i]) ++same;
  }
  std::printf("       row0 vs row1: %zu of %zu elements identical\n", same,
              per_row);
  LSE_EXPECT(same < per_row);

  // Padded rows answered zero and did not read a live row's block.
  for (std::size_t r = 2; r < static_cast<std::size_t>(bucket); ++r) {
    for (std::size_t i = 0; i < per_row; ++i) {
      LSE_EXPECT_EQ(got[r * per_row + i], 0.0f);
    }
  }

  // Each live row alone gives what it gave in the batch: the same token cannot
  // depend on how many rows shared the pass.
  for (std::int32_t r = 0; r < 2; ++r) {
    std::vector<float> one_q(per_row);
    for (std::size_t i = 0; i < per_row; ++i) {
      one_q[i] = q[static_cast<std::size_t>(r) * per_row + i];
    }
    std::vector<float> one_table(static_cast<std::size_t>(stride), 0.0f);
    one_table[0] = table[static_cast<std::size_t>(r) * stride];
    graph::Array q1 = filled(Shape{1, kQh, 1, kHd}, one_q);
    graph::Array t1 = filled(Shape{1, stride}, one_table);
    graph::Array m1 = step_meta({{0, live}});
    graph::Array o1 = graph::sdpa_paged(q1, kp, vp, 0.25f,
                                        graph::MaskKind::kCausal, 0, m1, t1, bs);
    LSE_EXPECT_OK(o1.eval());
    const std::vector<float> alone = read_all(o1);
    LSE_EXPECT_EQ(alone.size(), per_row);
    if (alone.size() != per_row) return;
    std::size_t differ = 0;
    for (std::size_t i = 0; i < per_row; ++i) {
      if (std::memcmp(&alone[i], &got[static_cast<std::size_t>(r) * per_row + i],
                      sizeof(float)) != 0) {
        ++differ;
      }
    }
    std::printf("       row %d batched vs alone: %zu of %zu differ\n", r, differ,
                per_row);
    LSE_EXPECT_EQ(differ, 0u);
  }
}

namespace {

// One ragged attention pass: `rows` gives each row's {first query position,
// live KV length} and `blocks` its block list. Returns the whole [bucket, Hq,
// 1, Hd] output.
struct RaggedRow {
  std::int32_t first = 0;
  std::int32_t len = 0;
  std::vector<std::int32_t> blocks;
};

constexpr std::int64_t kRagQh = 2;
constexpr std::int64_t kRagKvh = 2;
constexpr std::int64_t kRagHd = 8;
constexpr std::int32_t kRagStride = 8;
constexpr std::int32_t kRagPool = 16;

std::vector<float> ragged_pass(const std::vector<RaggedRow>& rows,
                               const std::vector<float>& q,
                               const graph::Array& kp, const graph::Array& vp) {
  const auto bucket = static_cast<std::int64_t>(rows.size());
  std::vector<float> table(
      static_cast<std::size_t>(bucket * kRagStride), 0.0f);
  std::vector<std::pair<std::int32_t, std::int32_t>> meta;
  for (std::size_t r = 0; r < rows.size(); ++r) {
    for (std::size_t i = 0; i < rows[r].blocks.size(); ++i) {
      table[r * static_cast<std::size_t>(kRagStride) + i] =
          static_cast<float>(rows[r].blocks[i]);
    }
    meta.push_back({rows[r].first, rows[r].len});
  }
  graph::Array ta = filled(Shape{bucket, kRagStride}, table);
  graph::Array ma = step_meta(meta);
  graph::Array qa = filled(Shape{bucket, kRagQh, 1, kRagHd}, q);
  graph::Array out = graph::sdpa_paged(qa, kp, vp, 0.25f,
                                       graph::MaskKind::kCausal, 0, ma, ta,
                                       kv::kBlockSize);
  if (!out.eval().ok()) return {};
  return read_all(out);
}

}  // namespace

LSE_TEST(a_row_gets_the_same_bits_whoever_shares_its_step) {
  // The acceptance gate for continuous batching, and the shape of the defect
  // that has bitten this codebase three times: a token's answer must not depend
  // on how many rows shared its pass, on where those rows sit, or on how long
  // they are. Here every row is at a different absolute position with a
  // different live length and its own blocks — the case a shared meta[0]/meta[1]
  // gets wrong three separate ways (causal mask, softmax bound, RoPE origin)
  // while still producing fluent text.
  graph::Scheduler* sched = graph::default_scheduler();
  LSE_EXPECT(sched != nullptr);
  if (sched == nullptr) return;

  std::vector<float> pool(static_cast<std::size_t>(
      kRagPool * kRagKvh * kv::kBlockSize * kRagHd));
  std::vector<float> poolv(pool.size());
  for (std::size_t i = 0; i < pool.size(); ++i) {
    pool[i] = noise(i + 17);
    poolv[i] = noise(i + 4211);
  }
  graph::Array kp =
      filled(Shape{kRagPool, kRagKvh, kv::kBlockSize, kRagHd}, pool);
  graph::Array vp =
      filled(Shape{kRagPool, kRagKvh, kv::kBlockSize, kRagHd}, poolv);

  // Deliberately ragged: one row inside its first block, one three blocks deep,
  // one exactly on a block boundary, one holding no sequence at all.
  std::vector<RaggedRow> rows{
      {4, 5, {2}},
      {39, 40, {5, 1, 7}},
      {15, 16, {11}},
      {0, 0, {}},
  };
  const auto per_row = static_cast<std::size_t>(kRagQh * kRagHd);
  std::vector<float> q(per_row * rows.size());
  for (std::size_t i = 0; i < q.size(); ++i) q[i] = noise(i + 88);

  const std::vector<float> batched = ragged_pass(rows, q, kp, vp);
  LSE_EXPECT_EQ(batched.size(), per_row * rows.size());
  if (batched.size() != per_row * rows.size()) return;

  // 1. Each live row alone, at its own position and length, gives the same bits.
  for (std::size_t r = 0; r + 1 < rows.size(); ++r) {
    const std::vector<RaggedRow> solo{rows[r]};
    const std::vector<float> one_q(q.begin() + static_cast<std::ptrdiff_t>(r * per_row),
                                   q.begin() + static_cast<std::ptrdiff_t>((r + 1) * per_row));
    const std::vector<float> alone = ragged_pass(solo, one_q, kp, vp);
    LSE_EXPECT_EQ(alone.size(), per_row);
    if (alone.size() != per_row) return;
    std::size_t differ = 0;
    for (std::size_t i = 0; i < per_row; ++i) {
      if (std::memcmp(&alone[i], &batched[r * per_row + i], sizeof(float)) != 0) {
        ++differ;
      }
    }
    std::printf("       row %zu (pos %d, len %d) batched vs alone: %zu of %zu differ\n",
                r, rows[r].first, rows[r].len, differ, per_row);
    LSE_EXPECT_EQ(differ, 0u);
  }

  // 2. A row holding no sequence answers zero and reads nobody's blocks.
  for (std::size_t i = 0; i < per_row; ++i) {
    LSE_EXPECT_EQ(batched[3 * per_row + i], 0.0f);
  }

  // 3. Row 0 does not move when the rest of the batch is rewritten under it.
  // This is the assertion an index that lost its row term fails while every
  // single-row test still passes.
  std::vector<RaggedRow> other{
      rows[0],
      {7, 8, {3}},
      {0, 0, {}},
      {60, 61, {9, 12, 6, 14}},
  };
  std::vector<float> q2 = q;
  for (std::size_t i = per_row; i < q2.size(); ++i) q2[i] = noise(i + 9999);
  const std::vector<float> shuffled = ragged_pass(other, q2, kp, vp);
  LSE_EXPECT_EQ(shuffled.size(), batched.size());
  if (shuffled.size() != batched.size()) return;
  std::size_t moved = 0;
  for (std::size_t i = 0; i < per_row; ++i) {
    if (std::memcmp(&shuffled[i], &batched[i], sizeof(float)) != 0) ++moved;
  }
  std::printf("       row 0 under a different batch: %zu of %zu moved\n", moved,
              per_row);
  LSE_EXPECT_EQ(moved, 0u);

  // 4. The rows are not each other: a kernel that read one row's descriptor for
  // every row would pass 1-3 above if the rows happened to agree, so say it.
  std::size_t same = 0;
  for (std::size_t i = 0; i < per_row; ++i) {
    if (batched[i] == batched[per_row + i]) ++same;
  }
  LSE_EXPECT(same < per_row);
}

LSE_TEST(a_ragged_write_puts_each_row_at_its_own_position) {
  // The write side of the same rule. Two rows at different absolute positions
  // must land in their own blocks at their own slots; a shared position would
  // have one of them overwrite the other's KV, which reads as a model that
  // slowly forgets under load.
  graph::Scheduler* sched = graph::default_scheduler();
  LSE_EXPECT(sched != nullptr);
  if (sched == nullptr) return;

  const std::int32_t bs = kv::kBlockSize;
  constexpr std::int64_t kKvh = 2;
  constexpr std::int64_t kW = 4;
  const std::int32_t pool_blocks = 6;
  const std::int32_t stride = 4;

  std::vector<float> zero(
      static_cast<std::size_t>(pool_blocks * kKvh * bs * kW), 0.0f);
  graph::Array pool = filled(Shape{pool_blocks, kKvh, bs, kW}, zero);

  std::vector<float> src(static_cast<std::size_t>(3 * kKvh * 1 * kW));
  for (std::size_t i = 0; i < src.size(); ++i) src[i] = noise(i + 61) + 3.0f;
  graph::Array sa = filled(Shape{3, kKvh, 1, kW}, src);

  // Row 0 writes position 2 of block 1; row 1 writes position 17, which is slot
  // 1 of its *second* block, block 4; row 2 holds no sequence.
  std::vector<float> table(static_cast<std::size_t>(3 * stride), 0.0f);
  table[0] = 1.0f;
  table[static_cast<std::size_t>(stride)] = 5.0f;
  table[static_cast<std::size_t>(stride) + 1] = 4.0f;
  table[static_cast<std::size_t>(2 * stride)] = 3.0f;
  graph::Array ta = filled(Shape{3, stride}, table);
  graph::Array meta = step_meta({{2, 3}, {17, 18}, {0, 0}});

  graph::Array written = graph::kv_page_write(pool, sa, meta, ta, bs);
  LSE_EXPECT_OK(written.eval());
  const std::vector<float> got = read_all(written);
  LSE_EXPECT_EQ(got.size(), zero.size());
  if (got.size() != zero.size()) return;

  const auto at = [&](std::int32_t blk, std::int64_t h, std::int32_t slot,
                      std::int64_t w) {
    return static_cast<std::size_t>(((blk * kKvh + h) * bs + slot) * kW + w);
  };
  for (std::int64_t h = 0; h < kKvh; ++h) {
    for (std::int64_t w = 0; w < kW; ++w) {
      LSE_EXPECT_EQ(got[at(1, h, 2, w)],
                    src[static_cast<std::size_t>((0 * kKvh + h) * kW + w)]);
      LSE_EXPECT_EQ(got[at(4, h, 1, w)],
                    src[static_cast<std::size_t>((1 * kKvh + h) * kW + w)]);
    }
  }
  // Block 3 belongs to the row holding no sequence, and blocks 0/2/5 were never
  // a write target: all of them stay zero.
  for (std::int32_t blk : {0, 2, 3, 5}) {
    for (std::size_t i = 0; i < static_cast<std::size_t>(kKvh * bs * kW); ++i) {
      LSE_EXPECT_EQ(got[static_cast<std::size_t>(blk * kKvh * bs * kW) + i],
                    0.0f);
    }
  }
}

LSE_TEST(the_paged_write_lands_where_the_block_table_says) {
  graph::Scheduler* sched = graph::default_scheduler();
  LSE_EXPECT(sched != nullptr);
  if (sched == nullptr) return;

  const std::int32_t bs = kv::kBlockSize;
  constexpr std::int64_t kKvh = 2;
  constexpr std::int64_t kW = 4;
  const std::int32_t pool_blocks = 4;
  const std::int32_t stride = 4;
  const std::int64_t t = 3;
  const std::int32_t pos = bs - 1;  // straddles the block boundary

  std::vector<float> zero(
      static_cast<std::size_t>(pool_blocks * kKvh * bs * kW), 0.0f);
  graph::Array pool = filled(Shape{pool_blocks, kKvh, bs, kW}, zero);

  std::vector<float> src(static_cast<std::size_t>(2 * kKvh * t * kW));
  for (std::size_t i = 0; i < src.size(); ++i) src[i] = noise(i + 13) + 2.0f;
  graph::Array sa = filled(Shape{2, kKvh, t, kW}, src);

  // Row 0 -> blocks 1 then 3; row 1 is padding and must write nothing.
  std::vector<float> table(static_cast<std::size_t>(2 * stride), 0.0f);
  table[0] = 1.0f;
  table[1] = 3.0f;
  table[static_cast<std::size_t>(stride)] = 2.0f;
  graph::Array ta = filled(Shape{2, stride}, table);
  graph::Array meta = step_meta(
      {{pos, pos + static_cast<std::int32_t>(t)}, {0, 0}});

  graph::Array written = graph::kv_page_write(pool, sa, meta, ta, bs);
  LSE_EXPECT_OK(written.eval());
  const std::vector<float> got = read_all(written);
  LSE_EXPECT_EQ(got.size(), zero.size());
  if (got.size() != zero.size()) return;

  for (std::int64_t h = 0; h < kKvh; ++h) {
    for (std::int64_t j = 0; j < t; ++j) {
      const std::int32_t abs = pos + static_cast<std::int32_t>(j);
      const std::int32_t blk = abs < bs ? 1 : 3;
      const std::int32_t slot = abs % bs;
      for (std::int64_t w = 0; w < kW; ++w) {
        const auto dst = static_cast<std::size_t>(
            ((blk * kKvh + h) * bs + slot) * kW + w);
        const auto s_i = static_cast<std::size_t>((h * t + j) * kW + w);
        LSE_EXPECT_EQ(got[dst], src[s_i]);
      }
    }
  }
  // Block 2 is the padded row's target and block 0 was never named: both stay
  // zero, so a pad row cannot corrupt the pool.
  for (std::int32_t blk : {0, 2}) {
    for (std::size_t i = 0; i < static_cast<std::size_t>(kKvh * bs * kW); ++i) {
      LSE_EXPECT_EQ(got[static_cast<std::size_t>(blk * kKvh * bs * kW) + i],
                    0.0f);
    }
  }
}

LSE_TEST(the_same_token_gets_the_same_logits_at_every_pass_width) {
  // Width invariance across the row axis: prefill the same prompt through
  // different pass plans and the final logits must agree. Extents are baked, so
  // each plan is a different set of kernels reading the same paged KV.
  if (!have_model()) return;

  auto paths = model::resolve_model(model_dir());
  if (!paths.ok()) return;
  auto ckpt = model::SafeTensors::open(paths->weights);
  auto cfg = model::Config::from_json_file(paths->config);
  if (!ckpt.ok() || !cfg.ok()) return;

  auto lm = model::make_lemonseed(*cfg);
  model::WeightBinder binder(*ckpt);
  if (!lm->load(binder).ok()) return;

  const std::vector<std::uint32_t> prompt{2, 3, 5, 7, 11, 13, 17, 19};

  const auto run = [&](const std::vector<std::size_t>& plan) {
    Session session("width", lm->num_layers());
    std::vector<float> out;
    std::size_t at = 0;
    for (std::size_t take : plan) {
      std::vector<float> ids(take);
      for (std::size_t i = 0; i < take; ++i) {
        ids[i] = static_cast<float>(prompt[at + i]);
      }
      graph::Array tokens =
          filled(Shape{1, static_cast<std::int64_t>(take)}, ids);
      auto h = lm->hidden(tokens, &session.states(), nullptr);
      if (!h.ok()) return out;
      at += take;
      if (at != prompt.size()) continue;
      const Shape& hs = h->shape();
      graph::Array row = graph::slice(*h, static_cast<int>(hs.rank()) - 2,
                                     hs.dim(hs.rank() - 2) - 1,
                                     hs.dim(hs.rank() - 2));
      auto lg = lm->lm_head(row);
      if (!lg.ok()) return out;
      out = read_all(*lg);
    }
    return out;
  };

  const std::vector<float> ones = run({1, 1, 1, 1, 1, 1, 1, 1});
  LSE_EXPECT(!ones.empty());
  if (ones.empty()) return;
  for (const std::vector<std::size_t>& plan :
       std::vector<std::vector<std::size_t>>{{8}, {4, 4}, {1, 2, 4, 1}, {2, 2, 4}}) {
    const std::vector<float> got = run(plan);
    LSE_EXPECT_EQ(got.size(), ones.size());
    if (got.size() != ones.size()) continue;
    double worst = 0.0;
    double ref = 0.0;
    for (std::size_t i = 0; i < got.size(); ++i) {
      worst = std::max(worst, std::abs(static_cast<double>(got[i] - ones[i])));
      ref = std::max(ref, std::abs(static_cast<double>(ones[i])));
    }
    const double rel = ref > 0.0 ? worst / ref : worst;
    std::printf("       plan of %zu pass(es): max_rel=%.3e\n", plan.size(), rel);
    LSE_EXPECT_EQ(argmax(got), argmax(ones));
    LSE_EXPECT(rel < 2e-3);
  }
}

LSE_TEST(a_paged_session_holds_only_the_blocks_it_reached) {
  if (!have_model()) return;

  auto paths = model::resolve_model(model_dir());
  if (!paths.ok()) return;
  auto ckpt = model::SafeTensors::open(paths->weights);
  auto cfg = model::Config::from_json_file(paths->config);
  if (!ckpt.ok() || !cfg.ok()) return;

  auto lm = model::make_lemonseed(*cfg);
  model::WeightBinder binder(*ckpt);
  if (!lm->load(binder).ok()) return;

  Session session("budget", lm->num_layers());
  std::vector<float> ids{2.0f, 3.0f, 5.0f, 7.0f, 11.0f};
  graph::Array tokens = filled(Shape{1, 5}, ids);
  auto h = lm->hidden(tokens, &session.states(), nullptr);
  LSE_EXPECT_OK(h.status());
  if (!h.ok()) return;

  // 5 tokens is one block per attention layer, in a pool at the smallest rung.
  std::size_t attn_layers = 0;
  for (const auto& st : session.states()) {
    if (!st.paged.valid()) continue;
    ++attn_layers;
    LSE_EXPECT_EQ(st.paged.tables.size(), 1u);
    LSE_EXPECT_EQ(st.paged.tables[0].size(), 1);
    LSE_EXPECT_EQ(st.key_cache.shape().dim(0), kv::kMinPoolBlocks);
  }
  LSE_EXPECT(attn_layers > 0u);
  LSE_EXPECT_EQ(session.kv_blocks(), attn_layers);

  // What a contiguous cache at the engine length would have cost, against what
  // the pools actually hold.
  const auto cap = static_cast<std::size_t>(cfg->kv_capacity());
  std::size_t paged_bytes = 0;
  std::size_t contiguous_bytes = 0;
  for (const auto& st : session.states()) {
    if (!st.paged.valid()) continue;
    paged_bytes += st.paged.pool_bytes();
    const Shape& p = st.key_cache.shape();
    const auto per_token = static_cast<std::size_t>(p.dim(1) * p.dim(3)) *
                           sizeof(float) * 2;
    contiguous_bytes += per_token * cap;
  }
  std::printf("       KV pools: %.3f MiB paged vs %.3f MiB contiguous (%.1fx)\n",
              static_cast<double>(paged_bytes) / 1048576.0,
              static_cast<double>(contiguous_bytes) / 1048576.0,
              static_cast<double>(contiguous_bytes) /
                  static_cast<double>(paged_bytes));
  LSE_EXPECT(paged_bytes * 8 <= contiguous_bytes);
}

namespace {

// The lemonseed fixture: the model plus the memory-mapped checkpoint its
// weights are views into, which therefore has to outlive it. Empty when the
// checkpoint is not on this box.
struct Lemonseed {
  std::unique_ptr<model::SafeTensors> weights;
  std::unique_ptr<model::HybridLM> lm;
  explicit operator bool() const noexcept { return lm != nullptr; }
};

Lemonseed load_lemonseed() {
  Lemonseed out;
  if (!have_model()) return out;
  auto paths = model::resolve_model(model_dir());
  if (!paths.ok()) return out;
  auto ckpt = model::SafeTensors::open(paths->weights);
  auto cfg = model::Config::from_json_file(paths->config);
  if (!ckpt.ok() || !cfg.ok()) return out;
  out.weights = std::make_unique<model::SafeTensors>(ckpt.release());
  auto lm = model::make_lemonseed(*cfg);
  model::WeightBinder binder(*out.weights);
  if (!lm->load(binder).ok()) return out;
  out.lm = std::move(lm);
  return out;
}

SamplingParams greedy_params() {
  SamplingParams p;
  p.temperature = 0.0f;
  p.repetition_penalty = 1.0f;
  return p;
}

std::string ids_to_string(const std::vector<std::uint32_t>& v) {
  std::string s;
  for (std::uint32_t id : v) {
    if (!s.empty()) s += ' ';
    s += std::to_string(id);
  }
  return s;
}

}  // namespace

LSE_TEST(a_batch_of_one_says_exactly_what_the_single_session_path_says) {
  // The batch driver is not a second engine. One sequence through it must give
  // the same tokens as Generator gives the same sequence, because that path is
  // where every baseline in WORK.md was measured: a batch of one that differed
  // would mean the baselines no longer describe the engine.
  Lemonseed fx = load_lemonseed();
  if (!fx) return;
  model::HybridLM& lm = *fx.lm;

  const std::vector<std::uint32_t> prompt{11u, 907u, 40u, 5u, 82u};
  constexpr std::int32_t kWant = 24;

  Generator gen(lm, greedy_params());
  Session session("solo", lm.num_layers());
  GenerationLimits glimits;
  glimits.max_tokens = kWant;
  auto want = gen.generate(session, prompt, glimits);
  LSE_EXPECT_OK(want.status());
  if (!want.ok()) return;

  BatchLimits limits;
  limits.max_batch = 1;
  limits.max_tokens = kWant;
  BatchScheduler one(lm, greedy_params(), limits);
  LSE_EXPECT_OK(one.submit({"solo", prompt, kWant}));
  auto got = one.run();
  LSE_EXPECT_OK(got.status());
  if (!got.ok() || got->empty()) return;

  LSE_EXPECT_EQ(one.bucket(), 1);
  std::printf("       generator [%s]\n       batch-of-1 [%s] | %.1f tok/s "
              "aggregate, %.1f tok/s per session\n",
              ids_to_string(*want).c_str(),
              ids_to_string((*got)[0].generated).c_str(),
              one.stats().aggregate_tokens_per_second(),
              (*got)[0].tokens_per_second());
  LSE_EXPECT((*got)[0].generated == *want);
}

LSE_TEST(a_sequence_decodes_the_same_whoever_shares_its_batch) {
  // The acceptance gate for the driver, through the whole model rather than one
  // kernel: run three sequences of different prompt lengths together, then run
  // each of them alone in an engine of the same width, and the token streams
  // must be identical.
  //
  // Three sequences into two rows on purpose. Rows admitted in the same step
  // advance in lockstep, so they share an absolute position however different
  // their prompts are, and a shared-scalar kernel is accidentally right about
  // them. The third sequence joins when a row frees, at a position the other
  // live row is nowhere near — which is the case a single meta[0] gets wrong
  // while still producing fluent text.
  Lemonseed fx = load_lemonseed();
  if (!fx) return;
  model::HybridLM& lm = *fx.lm;

  const std::vector<std::vector<std::uint32_t>> prompts{
      {11u, 907u, 40u, 5u, 82u},
      {3u, 19u},
      {77u, 4u, 913u},
  };
  constexpr std::int32_t kWant = 6;

  BatchLimits limits;
  limits.max_batch = 2;
  limits.max_tokens = kWant;

  BatchScheduler together(lm, greedy_params(), limits);
  for (std::size_t i = 0; i < prompts.size(); ++i) {
    LSE_EXPECT_OK(together.submit({"s" + std::to_string(i), prompts[i], kWant}));
  }
  auto batched = together.run();
  LSE_EXPECT_OK(batched.status());
  if (!batched.ok()) return;
  LSE_EXPECT_EQ(batched->size(), prompts.size());
  if (batched->size() != prompts.size()) return;

  std::printf("       batch of %zu: %.2f tok/s aggregate over %d step(s), "
              "occupancy %.2f\n",
              prompts.size(), together.stats().aggregate_tokens_per_second(),
              together.stats().steps, together.stats().occupancy());

  for (const SequenceResult& r : *batched) {
    const std::size_t i = static_cast<std::size_t>(r.id[1] - '0');
    BatchScheduler alone(lm, greedy_params(), limits);
    LSE_EXPECT_OK(alone.submit({r.id, prompts[i], kWant}));
    auto solo = alone.run();
    LSE_EXPECT_OK(solo.status());
    if (!solo.ok() || solo->empty()) return;
    std::printf("       %s batched [%s] vs alone [%s] | %.1f tok/s per session, "
                "ttft %.1f ms\n",
                r.id.c_str(), ids_to_string(r.generated).c_str(),
                ids_to_string((*solo)[0].generated).c_str(),
                r.tokens_per_second(),
                static_cast<double>(r.ttft_ns) / 1e6);
    LSE_EXPECT_EQ(r.generated.size(), static_cast<std::size_t>(kWant));
    LSE_EXPECT(r.generated == (*solo)[0].generated);
  }
}

LSE_TEST(a_sequence_that_joins_a_running_batch_gets_its_own_answer) {
  // Sessions join and leave between steps. Six sequences through four rows, with
  // different token budgets so they retire at different steps and the ones still
  // waiting are admitted into the rows that free up. A slot that keeps the
  // previous occupant's recurrent state would answer this wrong, fluently.
  Lemonseed fx = load_lemonseed();
  if (!fx) return;
  model::HybridLM& lm = *fx.lm;

  const std::vector<std::vector<std::uint32_t>> prompts{
      {11u, 907u}, {3u},        {77u, 4u, 913u},
      {5u, 6u},    {820u, 12u}, {41u},
  };
  const std::vector<std::int32_t> budget{2, 5, 3, 6, 4, 5};

  BatchLimits limits;
  limits.max_batch = 4;
  limits.max_tokens = 8;

  BatchScheduler mixed(lm, greedy_params(), limits);
  for (std::size_t i = 0; i < prompts.size(); ++i) {
    LSE_EXPECT_OK(mixed.submit(
        {"j" + std::to_string(i), prompts[i], budget[i]}));
  }
  auto got = mixed.run();
  LSE_EXPECT_OK(got.status());
  if (!got.ok()) return;
  LSE_EXPECT_EQ(got->size(), prompts.size());
  if (got->size() != prompts.size()) return;
  // More sequences than rows, so at least one was admitted into a row somebody
  // else had already used.
  LSE_EXPECT(mixed.stats().admissions >
             static_cast<std::int32_t>(mixed.bucket()));

  for (const SequenceResult& r : *got) {
    const std::size_t i = static_cast<std::size_t>(r.id[1] - '0');
    BatchScheduler alone(lm, greedy_params(), limits);
    LSE_EXPECT_OK(alone.submit({r.id, prompts[i], budget[i]}));
    auto solo = alone.run();
    LSE_EXPECT_OK(solo.status());
    if (!solo.ok() || solo->empty()) return;
    std::printf("       %s joined mid-flight [%s] vs alone [%s]\n",
                r.id.c_str(), ids_to_string(r.generated).c_str(),
                ids_to_string((*solo)[0].generated).c_str());
    LSE_EXPECT_EQ(r.generated.size(), static_cast<std::size_t>(budget[i]));
    LSE_EXPECT(r.generated == (*solo)[0].generated);
  }
}

LSE_TEST(churning_sessions_hand_every_block_back) {
  // ops::ensure_paged used to reassign a layer's block tables without releasing
  // the old ones, so every row that went away leaked its blocks. At a batch of
  // one that path never ran; with sessions retiring and being replaced it runs
  // constantly, and a leak shows up only much later as a pool that will not
  // admit anything. So watch the free list directly: many sequences through few
  // rows, twice, and the pool must come back to full both times without having
  // had to grow the second time.
  Lemonseed fx = load_lemonseed();
  if (!fx) return;
  model::HybridLM& lm = *fx.lm;

  std::vector<std::vector<std::uint32_t>> prompts;
  std::vector<std::int32_t> budget;
  for (std::uint32_t i = 0; i < 10u; ++i) {
    prompts.push_back({11u + i, 40u + i * 7u, 900u - i * 13u});
    budget.push_back(2 + static_cast<std::int32_t>(i % 5u));
  }

  BatchLimits limits;
  limits.max_batch = 4;
  limits.max_tokens = 8;
  limits.kv_blocks = 8;
  limits.kv_reserve = 0;

  BatchScheduler churn(lm, greedy_params(), limits);
  std::int32_t low_water = std::numeric_limits<std::int32_t>::max();
  auto wave = [&](const char* tag) {
    for (std::size_t i = 0; i < prompts.size(); ++i) {
      LSE_EXPECT_OK(churn.submit(
          {std::string(tag) + std::to_string(i), prompts[i], budget[i]}));
    }
    auto got = churn.run([&](const std::string&, std::uint32_t) {
      low_water = std::min(low_water, churn.kv_blocks_free());
      return true;
    });
    LSE_EXPECT_OK(got.status());
    if (got.ok()) LSE_EXPECT_EQ(got->size(), prompts.size());
    std::printf("       %s: %zu sequence(s), %d admission(s), %d preemption(s),"
                " pool %d block(s), free %d, low water %d\n",
                tag, prompts.size(), churn.stats().admissions,
                churn.stats().preemptions, churn.kv_blocks_total(),
                churn.kv_blocks_free(), low_water);
    // Every sequence has left, so every block it held is back.
    LSE_EXPECT_EQ(churn.kv_blocks_free(), churn.kv_blocks_total());
  };

  wave("w0");
  const std::int32_t pool_after_first = churn.kv_blocks_total();
  const std::int32_t admissions_first = churn.stats().admissions;
  // Blocks were genuinely held while the batch ran, so a full free list at the
  // end is a return rather than a pool that was never used.
  LSE_EXPECT(low_water < pool_after_first);

  wave("w1");
  // The second wave found the pool exactly as the first left it: a leak would
  // have forced it to a bigger rung or to more preemptions to fit.
  LSE_EXPECT_EQ(churn.kv_blocks_total(), pool_after_first);
  LSE_EXPECT(churn.stats().admissions >= 2 * admissions_first);
}

LSE_TEST(an_exhausted_block_pool_preempts_instead_of_failing) {
  // Exhaustion is a decision. With a budget too small for every sequence at
  // once, kv::BlockPolicy names victims oldest-first, they hand their blocks
  // back, and they are re-admitted later with their history as the prompt — so
  // every sequence still finishes, and finishes with the answer it would have
  // given alone.
  Lemonseed fx = load_lemonseed();
  if (!fx) return;
  model::HybridLM& lm = *fx.lm;

  const std::vector<std::vector<std::uint32_t>> prompts{
      {11u, 907u, 40u}, {3u, 19u}, {77u, 4u}, {820u, 12u, 6u},
  };
  constexpr std::int32_t kWant = 30;

  BatchLimits limits;
  limits.max_batch = 4;
  limits.max_tokens = kWant;
  // A block covers 16 tokens, so each of these reaches three of them. Four rows
  // want twelve and the pool holds six: they fit while they are short and stop
  // fitting as they grow, which is when a running sequence — not a waiting one —
  // is the thing that cannot get a block.
  limits.kv_blocks = 6;
  limits.kv_reserve = 0;

  BatchScheduler tight(lm, greedy_params(), limits);
  for (std::size_t i = 0; i < prompts.size(); ++i) {
    LSE_EXPECT_OK(tight.submit({"p" + std::to_string(i), prompts[i], kWant}));
  }
  auto got = tight.run();
  LSE_EXPECT_OK(got.status());
  if (!got.ok()) return;
  std::printf("       %d block(s) for %zu sequence(s): %d preemption(s), "
              "%d admission(s)\n",
              tight.block_ceiling(), prompts.size(),
              tight.stats().preemptions, tight.stats().admissions);
  LSE_EXPECT_EQ(got->size(), prompts.size());
  if (got->size() != prompts.size()) return;
  LSE_EXPECT(tight.stats().preemptions > 0);

  BatchLimits roomy = limits;
  roomy.kv_blocks = 0;
  for (const SequenceResult& r : *got) {
    const std::size_t i = static_cast<std::size_t>(r.id[1] - '0');
    LSE_EXPECT_EQ(r.generated.size(), static_cast<std::size_t>(kWant));
    BatchScheduler alone(lm, greedy_params(), roomy);
    LSE_EXPECT_OK(alone.submit({r.id, prompts[i], kWant}));
    auto solo = alone.run();
    LSE_EXPECT_OK(solo.status());
    if (!solo.ok() || solo->empty()) return;
    std::printf("       %s preempted %d time(s) [%s] vs untouched [%s]\n",
                r.id.c_str(), r.preemptions,
                ids_to_string(r.generated).c_str(),
                ids_to_string((*solo)[0].generated).c_str());
    LSE_EXPECT(r.generated == (*solo)[0].generated);
  }
}

LSE_TEST(a_pool_that_cannot_hold_one_sequence_refuses_it_by_name) {
  // The other half of the same decision: a request that does not fit even with
  // the pool empty is not a transient condition, so it is refused with the size
  // rather than queued forever or allowed to fail mid-step.
  Lemonseed fx = load_lemonseed();
  if (!fx) return;
  model::HybridLM& lm = *fx.lm;

  BatchLimits limits;
  limits.max_batch = 2;
  limits.max_tokens = 2;
  limits.kv_blocks = 1;  // 16 tokens
  limits.kv_reserve = 0;

  BatchScheduler tiny(lm, greedy_params(), limits);
  std::vector<std::uint32_t> long_prompt(40u, 7u);
  LSE_EXPECT_OK(tiny.submit({"big", long_prompt, 2}));
  auto got = tiny.run();
  LSE_EXPECT(!got.ok());
  std::printf("       refusal: %s\n", got.status().message().c_str());
  LSE_EXPECT(got.status().message().find("big") != std::string::npos);
  LSE_EXPECT(got.status().message().find("block") != std::string::npos);
}

LSE_TEST(a_batch_that_fits_no_bucket_is_refused_by_size) {
  Lemonseed fx = load_lemonseed();
  if (!fx) return;
  model::HybridLM& lm = *fx.lm;
  BatchLimits limits;
  limits.max_batch = 64;
  BatchScheduler over(lm, greedy_params(), limits);
  const Status s = over.submit({"x", {1u}, 1});
  LSE_EXPECT(!s.ok());
  LSE_EXPECT(s.message().find("64") != std::string::npos);
  LSE_EXPECT(s.message().find("fits no bucket") != std::string::npos);
}

LSE_TEST(a_two_row_pass_gives_each_row_what_it_gets_alone) {
  // Width invariance for the batch axis through the whole model, not just the
  // attention kernel: MatmulKernel::specialize() picks GEMV vs WMMA by row count
  // and MoE picks per expert, so a row's answer must not depend on how many rows
  // shared the pass. Row 1 must also differ from row 0 — the quant_linear defect
  // that read row 0 for every token produced fluent text and passed everything
  // that did not check this.
  if (!have_model()) return;

  auto paths = model::resolve_model(model_dir());
  if (!paths.ok()) return;
  auto ckpt = model::SafeTensors::open(paths->weights);
  auto cfg = model::Config::from_json_file(paths->config);
  if (!ckpt.ok() || !cfg.ok()) return;

  auto lm = model::make_lemonseed(*cfg);
  model::WeightBinder binder(*ckpt);
  if (!lm->load(binder).ok()) return;

  const std::vector<float> ids{11.0f, 907.0f};

  Session pair("pair", lm->num_layers());
  graph::Array tokens = filled(Shape{2, 1}, ids);
  const model::StepRows plan{{0, 0}};
  auto both = lm->hidden(tokens, &pair.states(), nullptr, nullptr, &plan);
  LSE_EXPECT_OK(both.status());
  if (!both.ok()) return;
  const std::vector<float> got = read_all(*both);
  const std::size_t width = got.size() / 2;
  LSE_EXPECT(width > 0u);
  if (width == 0u) return;

  std::size_t same = 0;
  for (std::size_t i = 0; i < width; ++i) {
    if (got[i] == got[width + i]) ++same;
  }
  std::printf("       hidden row0 vs row1: %zu of %zu identical\n", same, width);
  LSE_EXPECT(same < width / 2);

  for (std::size_t r = 0; r < 2; ++r) {
    Session solo("solo", lm->num_layers());
    graph::Array one = filled(Shape{1, 1}, {ids[r]});
    auto alone = lm->hidden(one, &solo.states(), nullptr);
    LSE_EXPECT_OK(alone.status());
    if (!alone.ok()) return;
    const std::vector<float> ref = read_all(*alone);
    LSE_EXPECT_EQ(ref.size(), width);
    if (ref.size() != width) return;
    double worst = 0.0;
    double scale = 0.0;
    for (std::size_t i = 0; i < width; ++i) {
      worst = std::max(worst,
                       std::abs(static_cast<double>(got[r * width + i] - ref[i])));
      scale = std::max(scale, std::abs(static_cast<double>(ref[i])));
    }
    const double rel = scale > 0.0 ? worst / scale : worst;
    std::printf("       row %zu in a pair vs alone: max_rel=%.3e\n", r, rel);
    LSE_EXPECT(rel < 2e-3);
  }

  // An off-ladder batch is refused by name, not silently widened.
  Session odd("odd", lm->num_layers());
  std::vector<float> three(3, 5.0f);
  graph::Array wide = filled(Shape{3, 1}, three);
  auto bad = lm->hidden(wide, &odd.states(), nullptr);
  LSE_EXPECT(!bad.ok());
  LSE_EXPECT(bad.status().message().find("not a batch bucket") !=
             std::string::npos);
}

LSE_TEST(q4_mtp_rows_match_a_scalar_reference_on_device) {
  graph::Scheduler* sched = graph::default_scheduler();
  LSE_EXPECT(sched != nullptr);
  if (sched == nullptr) return;
  const auto& device = sched->backend().device_info();
  if (sched->backend().emitter() == nullptr || device.arch != "gfx1201" ||
      device.wavefront_size != 32)
    LSE_SKIP("requires gfx1201 wave32 Q4 dispatch");
  struct RestoreMode {
    graph::Scheduler& scheduler;
    graph::Scheduler::Mode saved;
    ~RestoreMode() { scheduler.set_mode(saved); }
  } restore{*sched, sched->mode()};
  sched->set_mode(graph::Scheduler::Mode::kDeviceFirst);
  constexpr std::int64_t kN = 17;
  const auto upload = [&](Shape shape, DType dtype, const auto& values) {
    auto a = graph::Array::zeros(shape, dtype);
    auto& node = *a.node();
    if (!graph::interpreter::ensure_output_buffer(node, sched->backend()).ok())
      return graph::Array{};
    std::memcpy(graph::interpreter::host_bytes(node), values.data(),
                values.size() * sizeof(values[0]));
    node.materialized = true;
    node.host_dirty = true;
    node.device_dirty = false;
    if (!graph::interpreter::sync_to_device(node, sched->backend()).ok())
      return graph::Array{};
    return a;
  };
  for (std::int64_t m : {2, 3, 7, 8}) {
    for (std::int64_t k : {64, 576}) {
      const auto chunks = static_cast<std::size_t>(k / 8);
      const auto groups = static_cast<std::size_t>(k / 64);
      std::vector<float> x(static_cast<std::size_t>(m * k));
      std::vector<std::uint32_t> packed(static_cast<std::size_t>(kN) * chunks);
      std::vector<std::uint16_t> scales(static_cast<std::size_t>(kN) * groups);
      std::vector<std::uint16_t> biases(scales.size());
      for (std::int64_t row = 0; row < m; ++row) {
        for (std::int64_t at = 0; at < k; ++at) {
          const int code = at % 8 == row % 8 ? 127
              : static_cast<int>((row * 37 + at * 19 + 11) % 255) - 127;
          x[static_cast<std::size_t>(row * k + at)] = (at / 8) % 5 == 4 ? 0.0f
              : std::ldexp(static_cast<float>(code),
                            -static_cast<int>(8 + row % 3));
        }
      }
      for (std::size_t col = 0; col < static_cast<std::size_t>(kN); ++col) {
        for (std::size_t chunk = 0; chunk < chunks; ++chunk) {
          std::uint32_t word = 0;
          for (unsigned i = 0; i < 8; ++i)
            word |= static_cast<std::uint32_t>((col * 5 + chunk * 7 + i * 3) % 16)
                    << (4 * i);
          packed[col * chunks + chunk] = word;
        }
        for (std::size_t group = 0; group < groups; ++group) {
          scales[col * groups + group] = bfloat16_t::from_float(
              static_cast<float>((col + group) % 7 + 1) / 1024.0f);
          biases[col * groups + group] = bfloat16_t::from_float(
              static_cast<float>(static_cast<int>((col * 3 + group) % 9) - 4) / 64.0f);
        }
      }
      auto xa = upload(Shape{m, k}, DType::kF32, x);
      auto pa = upload(Shape{kN, k / 8}, DType::kU32, packed);
      auto sa = upload(Shape{kN, k / 64}, DType::kBF16, scales);
      auto ba = upload(Shape{kN, k / 64}, DType::kBF16, biases);
      LSE_EXPECT(xa.valid() && pa.valid() && sa.valid() && ba.valid());
      if (!xa.valid() || !pa.valid() || !sa.valid() || !ba.valid()) return;
      auto out = graph::quant_linear(xa, pa, sa, ba, 4, 64);
      const auto status = out.eval();
      LSE_EXPECT_OK(status);
      if (!status.ok()) return;
      const auto trace = sched->last_trace();
      LSE_EXPECT_EQ(trace.host_groups, 0u);
      LSE_EXPECT_EQ(trace.host_fallbacks, 0u);
      LSE_EXPECT(trace.device_groups > 0u && trace.kernels_launched > 0u);
      const auto got = read_all(out);
      LSE_EXPECT_EQ(got.size(), static_cast<std::size_t>(m * kN));
      if (got.size() != static_cast<std::size_t>(m * kN)) return;
      double max_abs = 0.0;
      for (std::size_t row = 0; row < static_cast<std::size_t>(m); ++row) {
        for (std::size_t col = 0; col < static_cast<std::size_t>(kN); ++col) {
          double expected = 0.0;
          for (std::size_t at = 0; at < static_cast<std::size_t>(k); ++at) {
            const auto group = at / 64;
            const auto code = (packed[col * chunks + at / 8] >> (4 * (at % 8))) & 15u;
            bfloat16_t stored_scale, stored_bias;
            stored_scale.bits = scales[col * groups + group];
            stored_bias.bits = biases[col * groups + group];
            const auto scale = stored_scale.to_float();
            const auto bias = stored_bias.to_float();
            expected += static_cast<double>(x[row * static_cast<std::size_t>(k) + at]) *
                        std::fma(static_cast<float>(code), scale, bias);
          }
          const auto value = got[row * static_cast<std::size_t>(kN) + col];
          LSE_EXPECT(std::isfinite(value));
          max_abs = std::max(max_abs, std::abs(static_cast<double>(value) - expected));
        }
        if (row) LSE_EXPECT(got[row * static_cast<std::size_t>(kN)] !=
                             got[(row - 1) * static_cast<std::size_t>(kN)]);
      }
      std::printf("       Q4 M%lld K%lld N17 max_abs=%.3e device=%u host=%u fallback=%u\n",
                  static_cast<long long>(m), static_cast<long long>(k), max_abs,
                  trace.device_groups, trace.host_groups, trace.host_fallbacks);
      LSE_EXPECT(max_abs < 1e-5);
    }
  }
}

static void check_split_attention_masks_and_replay(int queries) {
  auto* scheduler = graph::default_scheduler();
  LSE_EXPECT(scheduler != nullptr);
  if (!scheduler) return;
  if (!scheduler->backend().emitter()) LSE_SKIP("requires native attention dispatch");
  struct RestoreMode {
    graph::Scheduler& scheduler;
    graph::Scheduler::Mode mode;
    ~RestoreMode() { scheduler.set_mode(mode); }
  } restore{*scheduler, scheduler->mode()};
  scheduler->set_mode(graph::Scheduler::Mode::kDeviceFirst);
  auto& backend = scheduler->backend();
  const auto& device = backend.device_info();
  const auto rules = dispatch::arch::tuning(device.arch).decode;
  if (!std::any_of(rules.begin(), rules.end(), [&](const auto& rule) {
        return device.arch == rule.arch && device.wavefront_size == rule.wave &&
               device.max_threads_per_workgroup >= rule.threads;
      }))
    LSE_SKIP("requires a device in the split decode attention shape table");
  constexpr int batch = 3, dim = 256, block = 16;
  const float poison = std::numeric_limits<float>::quiet_NaN();
  auto read = [&](graph::Array array) { return read_all(array); };
  auto rewrite = [&](const graph::Array& array, const std::vector<float>& values) {
    auto& node = *array.node();
    for (std::size_t i = 0; i < values.size(); ++i)
      graph::interpreter::store_element(node, i, values[i]);
    LSE_EXPECT_OK(graph::interpreter::sync_to_device(node, backend));
  };
  for (int heads : {2, 6})
  for (int capacity : {16, 512, 8192, 16384, 32768, 262144}) {
    if (queries > 1 && (heads != 6 || capacity != 8192)) continue;
    const int live = queries > 1 ? 43 : std::min(capacity, 513);
    const int lengths[]{live, std::max(1, live / 2), 0};
    const int blocks = (live + block - 1) / block;
    const int pool_blocks = 2 * blocks + 1, stride = capacity / block;
    std::vector<float> query(batch * heads * queries * dim), keys(pool_blocks * block * dim, poison);
    std::vector<float> values(keys.size(), poison), table(batch * stride, 999999.0f);
    for (std::size_t i = 0; i < query.size(); ++i) query[i] = noise(i + 137) * .25f;
    std::fill(query.begin() + 2 * heads * queries * dim, query.end(), poison);
    for (int row = 0; row < 2; ++row) {
      for (int bi = 0; bi < blocks; ++bi)
        table[row * stride + bi] = static_cast<float>(1 + row * blocks + blocks - bi - 1);
      for (int key = 0; key < lengths[row]; ++key) {
        const int physical = static_cast<int>(table[row * stride + key / block]);
        for (int d = 0; d < dim; ++d) {
          const auto address = static_cast<std::size_t>((physical * block + key % block) * dim + d);
          keys[address] = noise(address + 3101) * .25f;
          values[address] = noise(address + 7103) * .25f;
        }
      }
    }
    auto q = filled(Shape{batch, heads, queries, dim}, query);
    auto k = filled(Shape{pool_blocks, 1, block, dim}, keys);
    auto v = filled(k.shape(), values);
    auto t = filled(Shape{batch, stride}, table);
    std::vector<float> metadata{static_cast<float>(live - queries), static_cast<float>(capacity + 17), 2,
                               static_cast<float>(live - queries), static_cast<float>(lengths[0]),
                               static_cast<float>(lengths[1] - queries), static_cast<float>(lengths[1]), 0, 0};
    auto meta = filled(Shape{kv::step_meta_elems(batch)}, metadata);
    for (const auto [mask, window] : {
         std::pair{graph::MaskKind::kCausal, 0},
         std::pair{graph::MaskKind::kSlidingWindow, 7},
         std::pair{graph::MaskKind::kSlidingWindow, 0}}) {
      graph::Array out;
      if (queries == 1) {
        out = graph::sdpa_paged(q, k, v, .0625f, mask, window, meta, t, block, &device);
      } else {
        auto made_partial = graph::custom("attention.split_partial128.wg128c2.v1",
            {q, k, v, meta, t}, {.0625f, 0, 0, 0});
        LSE_EXPECT_OK(made_partial.status());
        if (!made_partial.ok()) return;
        made_partial->node()->iattrs = {static_cast<int>(mask), window, 0, block};
        auto made_out = graph::custom("attention.split_merge128.wg128c2.v1", {*made_partial});
        LSE_EXPECT_OK(made_out.status());
        if (!made_out.ok()) return;
        out = *made_out;
      }
      LSE_EXPECT(out.node()->prim->name() == "attention.split_merge128.wg128c2.v1");
      if (out.node()->prim->name() != "attention.split_merge128.wg128c2.v1") return;
      auto partial = graph::Array(out.node()->inputs[0]);
      graph::Program program;
      const graph::NodePtr roots[]{out.node()};
      const auto status = scheduler->eval(roots, false, &program);
      LSE_EXPECT_OK(status);
      if (!status.ok()) return;
      const auto trace = scheduler->last_trace();
      LSE_EXPECT_EQ(trace.device_groups, 2u);
      LSE_EXPECT_EQ(trace.host_groups, 0u);
      LSE_EXPECT_EQ(trace.host_fallbacks, 0u);
      LSE_EXPECT_EQ(program.groups().size(), 2u);
      const auto actual = read(out);
      LSE_EXPECT_EQ(actual.size(), query.size());
      if (actual.size() != query.size()) return;
      double max_abs = 0;
      for (int row = 0; row < batch; ++row) {
        for (int h = 0; h < heads; ++h) for (int qi = 0; qi < queries; ++qi) {
          const int offset = row < 2 ? lengths[row] - queries + qi : 0;
          std::vector<double> score(static_cast<std::size_t>(lengths[row]),
                                    -std::numeric_limits<double>::infinity());
          double maximum = -std::numeric_limits<double>::infinity();
          for (int key = 0; key < lengths[row]; ++key) {
            if (key > offset || (mask == graph::MaskKind::kSlidingWindow && offset - key >= window)) continue;
            double dot = 0;
            const int physical = static_cast<int>(table[row * stride + key / block]);
            const auto base = static_cast<std::size_t>((physical * block + key % block) * dim);
            for (int d = 0; d < dim; ++d)
              dot += static_cast<double>(query[((row * heads + h) * queries + qi) * dim + d]) * keys[base + d];
            score[key] = dot * .0625;
            maximum = std::max(maximum, score[key]);
          }
          std::vector<double> numerator(dim, 0);
          double denominator = 0;
          for (int key = 0; key < lengths[row]; ++key) {
            if (!std::isfinite(score[key])) continue;
            const auto weight = std::exp(score[key] - maximum);
            denominator += weight;
            const int physical = static_cast<int>(table[row * stride + key / block]);
            const auto base = static_cast<std::size_t>((physical * block + key % block) * dim);
            for (int d = 0; d < dim; ++d) numerator[d] += weight * values[base + d];
          }
          for (int d = 0; d < dim; ++d) {
            const auto expected = denominator > 0 ? numerator[d] / denominator : 0;
            const auto got = actual[((row * heads + h) * queries + qi) * dim + d];
            LSE_EXPECT(std::isfinite(got));
            max_abs = std::max(max_abs, std::abs(static_cast<double>(got) - expected));
          }
        }
      }
      LSE_EXPECT(max_abs < 2e-6);
      const auto q_after = read(q), k_after = read(k), v_after = read(v), t_after = read(t);
      LSE_EXPECT(q_after.size() == query.size() && std::memcmp(q_after.data(), query.data(), query.size() * sizeof(float)) == 0);
      LSE_EXPECT(k_after.size() == keys.size() && std::memcmp(k_after.data(), keys.data(), keys.size() * sizeof(float)) == 0);
      LSE_EXPECT(v_after.size() == values.size() && std::memcmp(v_after.data(), values.data(), values.size() * sizeof(float)) == 0);
      LSE_EXPECT(t_after.size() == table.size() && std::memcmp(t_after.data(), table.data(), table.size() * sizeof(float)) == 0);

      // Poison retained output slots, then replay with no live rows. Every
      // partition must overwrite its complete record before the merge reads it.
      rewrite(partial, std::vector<float>(partial.shape().elem_count(), poison));
      rewrite(out, std::vector<float>(out.shape().elem_count(), poison));
      rewrite(meta, std::vector<float>(metadata.size(), 0));
      program.reset_compute();
      const auto replay = scheduler->eval(roots, false, &program);
      LSE_EXPECT_OK(replay);
      if (!replay.ok()) return;
      const auto empty = read(out), records = read(partial);
      for (float value : empty) LSE_EXPECT_EQ(value, 0.0f);
      LSE_EXPECT_EQ(records.size(), partial.shape().elem_count());
      for (std::size_t at = 0; at < records.size(); at += 258) {
        LSE_EXPECT(std::isinf(records[at]) && records[at] < 0);
        for (std::size_t field = 1; field < 258; ++field) LSE_EXPECT_EQ(records[at + field], 0.0f);
      }
      LSE_EXPECT_EQ(scheduler->last_trace().host_groups, 0u);
      LSE_EXPECT_EQ(scheduler->last_trace().host_fallbacks, 0u);
      rewrite(meta, metadata);
      std::printf("       split T%d capacity=%d live=%d mask=%d window=%d max_abs=%.3e device=2 host=0 fallback=0 empty-replay=pass\n",
                  queries, capacity, live, static_cast<int>(mask), window, max_abs);
    }
  }
}

LSE_TEST(single_token_split_attention_covers_long_tables_and_empty_replay) {
  check_split_attention_masks_and_replay(1);
}

LSE_TEST(joint_head_query_attention_preserves_masks_and_empty_replay) {
  check_split_attention_masks_and_replay(7);
}

LSE_TEST(short_flash_queries_match_reference_with_ragged_and_padded_rows) {
  graph::Scheduler* sched = graph::default_scheduler();
  LSE_EXPECT(sched != nullptr);
  if (sched == nullptr) return;
  if (sched->backend().emitter() == nullptr)
    LSE_SKIP("requires native attention dispatch");
  struct RestoreMode {
    graph::Scheduler& scheduler;
    graph::Scheduler::Mode saved;
    ~RestoreMode() { scheduler.set_mode(saved); }
  } restore{*sched, sched->mode()};
  constexpr std::int64_t kRows = 3, kHeads = 2, kDim = 256, kPool = 4;
  constexpr std::int32_t kStride = 4;
  const float scale = 1.0f / 16.0f;
  const float poison = std::numeric_limits<float>::quiet_NaN();
  for (std::int32_t tq : {2, 3, 7}) {
    sched->set_mode(graph::Scheduler::Mode::kDeviceFirst);
    const std::int32_t lengths[]{32 + tq, 3 + tq};
    std::vector<float> q(static_cast<std::size_t>(kRows * kHeads * tq * kDim));
    std::vector<float> keys(static_cast<std::size_t>(kPool * kv::kBlockSize * kDim));
    std::vector<float> values(keys.size());
    for (std::size_t i = 0; i < q.size(); ++i) q[i] = noise(i + 37) * 0.25f;
    for (std::size_t i = 0; i < keys.size(); ++i) {
      keys[i] = noise(i + 301) * 0.25f;
      values[i] = noise(i + 7901) * 0.25f;
    }
    std::fill(q.begin() + static_cast<std::ptrdiff_t>(2 * kHeads * tq * kDim),
               q.end(), poison);
    for (int row = 0; row < 2; ++row) {
      const int last_block = row == 0 ? 3 : 1;
      for (int slot = lengths[row] % kv::kBlockSize; slot < kv::kBlockSize; ++slot)
        for (std::int64_t d = 0; d < kDim; ++d) {
          const auto i = static_cast<std::size_t>(
              (last_block * kv::kBlockSize + slot) * kDim + d);
          keys[i] = values[i] = poison;
        }
    }
    auto qa = filled(Shape{kRows, kHeads, tq, kDim}, q);
    auto ka = filled(Shape{kPool, 1, kv::kBlockSize, kDim}, keys);
    auto va = filled(Shape{kPool, 1, kv::kBlockSize, kDim}, values);
    // The padded table row is deliberately invalid and must never be read.
    const std::vector<float> pages{2, 0, 3, 0, 1, 0, 0, 0, 99999, 99999, 99999, 99999};
    auto table = filled(Shape{kRows, kStride}, pages);
    auto meta = step_meta({{32, lengths[0]}, {3, lengths[1]}, {0, 0}});
    for (auto mask : {graph::MaskKind::kCausal, graph::MaskKind::kSlidingWindow}) {
      sched->set_mode(graph::Scheduler::Mode::kDeviceFirst);
      auto native = graph::sdpa_paged(qa, ka, va, scale, mask, 5, meta, table,
                                      kv::kBlockSize);
      const auto status = native.eval();
      LSE_EXPECT_OK(status);
      if (!status.ok()) return;
      for (const auto& reason : sched->last_trace().host_group_reasons)
        std::fprintf(stderr, "short Flash Tq=%d mask=%d fallback: %s\n", tq,
                      static_cast<int>(mask), reason.c_str());
      LSE_EXPECT_EQ(sched->last_trace().host_groups, 0u);
      LSE_EXPECT(sched->last_trace().kernels_launched > 0u);
      const auto got = read_all(native);
      sched->set_mode(graph::Scheduler::Mode::kHostOnly);
      auto host = graph::sdpa_paged(qa, ka, va, scale, mask, 5, meta, table,
                                    kv::kBlockSize);
      auto expected = read_all(host);
      std::vector<Shape> shapes;
      std::vector<DType> dtypes;
      for (const auto& input : native.node()->inputs) {
        shapes.push_back(input->shape);
        dtypes.push_back(input->dtype);
      }
      graph::KernelShapes invocation;
      invocation.inputs = shapes;
      invocation.input_dtypes = dtypes;
      invocation.output = native.shape();
      invocation.output_dtype = native.dtype();
      invocation.attrs = native.node()->attrs;
      invocation.iattrs = native.node()->iattrs;
      invocation.device = &sched->backend().device_info();
      const auto sources = sched->backend().emitter()->sources();
      invocation.intrinsics = &sources;
      invocation.types = backend::loom_types();
      const bool matrix = dispatch::attention_plan(invocation) ==
                          dispatch::AttentionPlan::kFlashWmma;
      if (matrix) {
        // F32 storage uses BF16 Q/K/P/V operands; softmax and accumulation stay F32.
        const auto narrow = [](float value) { return bfloat16_t(value).to_float(); };
        std::fill(expected.begin(), expected.end(), 0.0f);
        const auto physical_base = [&](int row, int key) {
          const auto slot = static_cast<std::size_t>(row * kStride + key / kv::kBlockSize);
          const int page = static_cast<int>(pages[slot]);
          return static_cast<std::size_t>(page * kv::kBlockSize + key % kv::kBlockSize) * kDim;
        };
        for (int row = 0; row < 2; ++row) {
          for (int head = 0; head < kHeads; ++head) {
            for (int query = 0; query < tq; ++query) {
              const int position = (row == 0 ? 32 : 3) + query;
              std::vector<float> scores(static_cast<std::size_t>(lengths[row]),
                                        -std::numeric_limits<float>::infinity());
              const auto qbase = static_cast<std::size_t>((row * kHeads + head) * tq + query) * kDim;
              float maximum = -std::numeric_limits<float>::infinity();
              for (int key = 0; key < lengths[row]; ++key) {
                if (key > position || (mask == graph::MaskKind::kSlidingWindow &&
                                       position - key >= 5)) continue;
                const auto base = physical_base(row, key);
                float dot = 0.0f;
                for (std::size_t d = 0; d < static_cast<std::size_t>(kDim); ++d)
                  dot = std::fma(narrow(q[qbase + d]), narrow(keys[base + d]), dot);
                scores[static_cast<std::size_t>(key)] = dot * scale;
                maximum = std::max(maximum, scores[static_cast<std::size_t>(key)]);
              }
              float denominator = 0.0f;
              for (float& score : scores) {
                score = std::exp(score - maximum);
                denominator += score;
              }
              for (int key = 0; key < lengths[row]; ++key) {
                if (scores[static_cast<std::size_t>(key)] == 0.0f) continue;
                const auto base = physical_base(row, key);
                for (std::size_t d = 0; d < static_cast<std::size_t>(kDim); ++d)
                  expected[qbase + d] = std::fma(narrow(scores[static_cast<std::size_t>(key)]), narrow(values[base + d]),
                                                 expected[qbase + d]);
              }
              for (std::size_t d = 0; d < static_cast<std::size_t>(kDim); ++d) expected[qbase + d] /= denominator;
            }
          }
        }
      }
      LSE_EXPECT_EQ(got.size(), q.size());
      LSE_EXPECT_EQ(expected.size(), got.size());
      if (got.size() != q.size() || expected.size() != got.size()) return;
      for (std::size_t i = 0; i < got.size(); ++i) {
        LSE_EXPECT(std::isfinite(got[i]) && std::isfinite(expected[i]));
        LSE_EXPECT_NEAR(got[i], expected[i], matrix ? 2e-5 : 2e-4);
        if (i >= static_cast<std::size_t>(2 * kHeads * tq * kDim))
          LSE_EXPECT_EQ(got[i], 0.0f);
      }
    }
  }
}

namespace {
class OpaqueHostBackend final : public backend::Backend<OpaqueHostBackend> {
 public:
  static constexpr std::string_view kName = "opaque-host-test";
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
}

LSE_TEST(host_inplace_writes_preserve_opaque_pool_contents_and_owner_mirrors) {
  backend::BackendAdapter<OpaqueHostBackend> be;
  LSE_EXPECT_OK(be.init(0));
  graph::Scheduler scheduler(be);
  scheduler.set_mode(graph::Scheduler::Mode::kHostOnly);
  const auto upload = [&](Shape shape, const std::vector<float>& values) {
    auto allocated = be.allocate(values.size() * sizeof(float), backend::MemoryClass::kDevice,
                                 backend::kDefaultStream);
    if (!allocated.ok()) return graph::Array{};
    auto buffer = allocated.release();
    if (!be.copy(buffer, values.data(), values.size() * sizeof(float)).ok()) return graph::Array{};
    auto array = graph::Array::from_buffer(std::move(buffer), std::move(shape), DType::kF32);
    array.node()->device_dirty = true;
    return array;
  };
  const auto read = [&](graph::Array array) {
    const graph::NodePtr roots[] = {array.node()};
    LSE_EXPECT_OK(scheduler.eval(roots, true));
    std::vector<float> values(array.shape().elem_count());
    for (std::size_t i = 0; i < values.size(); ++i)
      values[i] = graph::interpreter::load_element(*array.node(), i);
    return values;
  };
  std::vector<float> expected(32);
  for (std::size_t i = 0; i < expected.size(); ++i) expected[i] = static_cast<float>(i + 1);
  auto pool = upload({1, 1, 16, 2}, expected);
  auto table = upload({1, 1}, {0});
  auto first = graph::kv_page_write(pool, upload({1, 1, 2, 2}, {101, 102, 103, 104}),
                                  upload({5}, {0, 2, 1, 0, 2}), table, 16);
  expected[0] = 101; expected[1] = 102; expected[2] = 103; expected[3] = 104;
  LSE_EXPECT(read(first) == expected);
  LSE_EXPECT(read(pool) == expected);
  auto next = graph::kv_page_write(pool, upload({1, 1, 1, 2}, {201, 202}),
                                  upload({5}, {3, 4, 1, 3, 4}), table, 16);
  expected[6] = 201; expected[7] = 202;
  LSE_EXPECT(read(next) == expected);
  LSE_EXPECT(read(pool) == expected);
  LSE_EXPECT_OK(graph::interpreter::sync_to_device(*pool.node(), be));
  std::vector<float> device(expected.size());
  LSE_EXPECT_OK(be.copy(device.data(), pool.node()->buffer, device.size() * sizeof(float)));
  LSE_EXPECT(device == expected);

  auto owner = upload({6}, {1, 2, 3, 4, 5, 6});
  auto overwrite = graph::overwrite_slice(owner, upload({2}, {10, 20}), 0,
                                         upload({1}, {2}));
  const std::vector<float> sliced{1, 2, 10, 20, 5, 6};
  LSE_EXPECT(read(overwrite) == sliced);
  LSE_EXPECT(read(owner) == sliced);
}

LSE_TEST(a_large_context_uses_small_paged_tables_and_grows_without_losing_keys) {
  if (std::getenv("LSE_KV_PREALLOC") != nullptr)
    LSE_SKIP("requires the default growing KV pool");
  graph::Scheduler* sched = graph::default_scheduler();
  LSE_EXPECT(sched != nullptr);
  if (sched == nullptr) return;
  struct HostMode {
    graph::Scheduler& scheduler;
    graph::Scheduler::Mode saved;
    ~HostMode() { scheduler.set_mode(saved); }
  } mode{*sched, sched->mode()};
  sched->set_mode(graph::Scheduler::Mode::kHostOnly);

  constexpr std::int32_t kCapacity = 262100;
  constexpr std::int32_t kTokens = 257;
  ops::GatedAttentionSpec spec;
  spec.q_heads = spec.kv_heads = 1;
  spec.head_dim = 2;
  spec.kv_length = kCapacity;
  ops::GatedAttentionWeights weights;
  weights.q_proj = weights.k_proj = weights.v_proj = weights.o_proj =
      filled(Shape{2, 2}, {1, 0, 0, 1});
  weights.g_proj = filled(Shape{2, 2}, {0, 0, 0, 0});
  weights.q_norm = weights.k_norm = filled(Shape{2}, {0, 0});
  auto rope = ops::build_rope(2, kTokens, 10000.0f);
  LSE_EXPECT_OK(rope.status());
  if (!rope.ok()) return;
  std::vector<float> inputs(static_cast<std::size_t>(2 * kTokens));
  for (std::size_t i = 0; i < inputs.size(); ++i) inputs[i] = noise(i + 71);
  auto whole = ops::gated_attention(
      filled(Shape{1, kTokens, 2}, inputs), weights, spec, *rope, 0);
  LSE_EXPECT_OK(whole.status());
  if (!whole.ok()) return;
  const auto reference = read_all(*whole);
  LSE_EXPECT_EQ(reference.size(), inputs.size());
  if (reference.size() != inputs.size()) return;
  for (float value : reference) LSE_EXPECT(std::isfinite(value));

  ops::PagedKvLayer layer;
  ops::AttentionCache cache;
  cache.paged = &layer;
  cache.capacity = kCapacity;
  std::int32_t at = 0;
  for (std::int32_t count : {127, 1, 1, 128}) {
    if (at == 128) {
      const auto table_before = read_all(layer.table);
      const auto used_before = layer.alloc.used();
      const auto blocks_before = std::vector<kv::BlockId>(
          layer.tables[0].blocks().begin(), layer.tables[0].blocks().end());
      auto needs_growth = ops::extend_paged(layer, at + count);
      LSE_EXPECT_OK(needs_growth.status());
      LSE_EXPECT(needs_growth.ok() && *needs_growth);
      LSE_EXPECT_EQ(layer.alloc.used(), used_before);
      LSE_EXPECT(std::equal(blocks_before.begin(), blocks_before.end(),
                            layer.tables[0].blocks().begin(),
                            layer.tables[0].blocks().end()));
      LSE_EXPECT(read_all(layer.table) == table_before);
    }
    const auto end = at + count;
    cache.used = at;
    cache.meta = step_meta({{at, end}});
    std::vector<float> chunk(inputs.begin() + 2 * at, inputs.begin() + 2 * end);
    auto pass = ops::gated_attention(filled(Shape{1, count, 2}, chunk),
                                      weights, spec, *rope, at, &cache);
    LSE_EXPECT_OK(pass.status());
    if (!pass.ok()) return;
    const auto got = read_all(*pass);
    LSE_EXPECT_EQ(got.size(), chunk.size());
    if (got.size() != chunk.size()) return;
    for (std::size_t i = 0; i < got.size(); ++i) {
      LSE_EXPECT(std::isfinite(got[i]));
      LSE_EXPECT_NEAR(got[i], reference[static_cast<std::size_t>(2 * at) + i],
                      2e-6);
    }
    const auto expected_pool = kv::pool_rung(
        kv::blocks_for(end, kv::kBlockSize),
        kv::blocks_for(kCapacity, kv::kBlockSize));
    LSE_EXPECT_EQ(layer.stride(), expected_pool);
    LSE_EXPECT_EQ(layer.keys.shape().dim(0), expected_pool);
    LSE_EXPECT_EQ(layer.tables[0].size(), kv::blocks_for(end, kv::kBlockSize));
    LSE_EXPECT_EQ(cache.capacity, kCapacity);
    LSE_EXPECT(layer.stride() < kv::blocks_for(kCapacity, kv::kBlockSize));
    at = end;
  }
  LSE_EXPECT_EQ(at, kTokens);
  const auto stride = layer.stride();
  const auto pool = layer.keys.node();
  auto full_context = ops::extend_paged(layer, kCapacity);
  LSE_EXPECT_OK(full_context.status());
  LSE_EXPECT(full_context.ok() && *full_context);
  LSE_EXPECT_EQ(layer.tables[0].size(), kv::blocks_for(kTokens, kv::kBlockSize));
  LSE_EXPECT_OK(ops::release_row(layer, 0));
  layer.row_tokens = {1};
  cache.used = 0;
  cache.meta = step_meta({{0, 1}});
  auto shorter = ops::gated_attention(filled(Shape{1, 1, 2}, {0.25f, -0.5f}),
                                     weights, spec, *rope, 0, &cache);
  LSE_EXPECT_OK(shorter.status());
  LSE_EXPECT_EQ(layer.stride(), stride);
  LSE_EXPECT(layer.keys.node() == pool);
  LSE_EXPECT_EQ(layer.tables[0].size(), 1);
  LSE_EXPECT_EQ(cache.capacity, kCapacity);
  auto fits = ops::extend_paged(layer, 1);
  LSE_EXPECT_OK(fits.status());
  LSE_EXPECT(fits.ok() && !*fits);

  // Ragged rows share the pool rung, while retaining independent block lists.
  ops::PagedKvLayer ragged;
  ops::AttentionCache batch;
  batch.paged = &ragged;
  batch.capacity = kCapacity;
  ragged.row_tokens = {17, 0};
  batch.meta = step_meta({{16, 17}, {0, 0}});
  const auto x = filled(Shape{2, 1, 2}, {1, 2, 3, 4});
  auto first = ops::gated_attention(x, weights, spec, *rope, 16, &batch);
  LSE_EXPECT_OK(first.status());
  if (!first.ok()) return;
  LSE_EXPECT_EQ(ragged.stride(), kv::kMinPoolBlocks);
  LSE_EXPECT_EQ(ragged.tables[0].size(), 2);
  LSE_EXPECT(ragged.tables[1].empty());
  const auto retained = std::vector<kv::BlockId>(ragged.tables[0].blocks().begin(),
                                               ragged.tables[0].blocks().end());
  const auto table_before = read_all(ragged.table);
  ragged.row_tokens = {17, 129};
  auto replay = ops::extend_paged(ragged, 129);
  LSE_EXPECT_OK(replay.status());
  LSE_EXPECT(replay.ok() && *replay);
  LSE_EXPECT(ragged.tables[1].empty());
  LSE_EXPECT_EQ(ragged.alloc.used(), 2);
  LSE_EXPECT(read_all(ragged.table) == table_before);
  batch.used = 128;
  batch.meta = step_meta({{16, 17}, {128, 129}});
  auto grown = ops::gated_attention(x, weights, spec, *rope, 128, &batch);
  LSE_EXPECT_OK(grown.status());
  LSE_EXPECT_EQ(ragged.stride(), 16);
  LSE_EXPECT_EQ(ragged.keys.shape().dim(0), 16);
  LSE_EXPECT_EQ(ragged.tables[1].size(), 9);
  LSE_EXPECT(std::equal(retained.begin(), retained.end(),
                        ragged.tables[0].blocks().begin(),
                        ragged.tables[0].blocks().end()));
}

LSE_TEST(a_context_that_outgrows_its_pool_keeps_the_keys_it_wrote) {
  // Crossing a pool rung reallocates the block pool and copies the used prefix.
  // Two pass plans over the same 132 tokens cross it at different points; if the
  // copy or the block table were wrong the two would disagree, and a prompt long
  // enough to grow would quietly read the wrong keys.
  //
  // 132 tokens is 9 blocks, one past the smallest rung of 8.
  if (!have_model()) return;

  auto paths = model::resolve_model(model_dir());
  if (!paths.ok()) return;
  auto ckpt = model::SafeTensors::open(paths->weights);
  auto cfg = model::Config::from_json_file(paths->config);
  if (!ckpt.ok() || !cfg.ok()) return;

  auto lm = model::make_lemonseed(*cfg);
  model::WeightBinder binder(*ckpt);
  if (!lm->load(binder).ok()) return;

  constexpr std::size_t kTokens = 132;
  std::vector<float> prompt(kTokens);
  for (std::size_t i = 0; i < kTokens; ++i) {
    prompt[i] = static_cast<float>(3 + (i * 37) % 900);
  }

  // `widths` cycles pass by pass, so a plan can mix widths and still keep every
  // pass on the ladder the engine already compiled.
  const auto run = [&](const std::vector<std::size_t>& widths,
                       std::int32_t* blocks_out) {
    Session session("grow", lm->num_layers());
    std::vector<float> out;
    std::size_t at = 0;
    for (std::size_t pass = 0; at < kTokens; ++pass) {
      const std::size_t n = widths[pass % widths.size()];
      if (at + n > kTokens) break;
      std::vector<float> ids(prompt.begin() + static_cast<std::ptrdiff_t>(at),
                             prompt.begin() + static_cast<std::ptrdiff_t>(at + n));
      graph::Array tokens =
          filled(Shape{1, static_cast<std::int64_t>(n)}, ids);
      auto h = lm->hidden(tokens, &session.states(), nullptr);
      if (!h.ok()) return out;
      at += n;
      if (at != kTokens) continue;
      const Shape& hs = h->shape();
      graph::Array row = graph::slice(*h, static_cast<int>(hs.rank()) - 2,
                                     hs.dim(hs.rank() - 2) - 1,
                                     hs.dim(hs.rank() - 2));
      auto lg = lm->lm_head(row);
      if (!lg.ok()) return out;
      out = read_all(*lg);
    }
    if (blocks_out != nullptr) {
      *blocks_out = 0;
      for (const auto& st : session.states()) {
        if (!st.paged.valid()) continue;
        *blocks_out = st.paged.tables[0].size();
        LSE_EXPECT(st.key_cache.shape().dim(0) > kv::kMinPoolBlocks);
        break;
      }
    }
    return out;
  };

  std::int32_t blocks = 0;
  const std::vector<float> by_four = run({4}, &blocks);
  LSE_EXPECT(!by_four.empty());
  if (by_four.empty()) return;
  std::printf("       132 tokens held in %d block(s) after growth\n", blocks);
  LSE_EXPECT_EQ(blocks, kv::blocks_for(132, kv::kBlockSize));

  for (const std::vector<std::size_t>& widths :
       std::vector<std::vector<std::size_t>>{{1}, {2}, {1, 2, 1}, {2, 1, 1}}) {
    const std::vector<float> got = run(widths, nullptr);
    LSE_EXPECT_EQ(got.size(), by_four.size());
    if (got.size() != by_four.size()) continue;
    double worst = 0.0;
    double ref = 0.0;
    for (std::size_t i = 0; i < got.size(); ++i) {
      worst = std::max(worst, std::abs(static_cast<double>(got[i] - by_four[i])));
      ref = std::max(ref, std::abs(static_cast<double>(by_four[i])));
    }
    const double rel = ref > 0.0 ? worst / ref : worst;
    std::printf("       across a pool rung, %zu width(s) cycling: max_rel=%.3e\n",
                widths.size(), rel);
    LSE_EXPECT_EQ(argmax(got), argmax(by_four));
    LSE_EXPECT(rel < 2e-3);
  }
}

LSE_TEST_MAIN()

// Staged lowering joins the sibling projections; non-staged lowering launches
// them separately. Materialized intermediates provide an FP32 order reference.
LSE_TEST(swiglu_chain_preserves_values_and_backend_launch_contract) {
  graph::Scheduler* sched = graph::default_scheduler();
  LSE_EXPECT(sched != nullptr);
  if (sched == nullptr) return;
  if (sched->backend().emitter() == nullptr) return;

  constexpr std::int64_t kHidden = 1024;
  constexpr std::int64_t kInter = 2176;
  std::vector<float> values(static_cast<std::size_t>(kHidden));
  std::vector<float> gate_weights(static_cast<std::size_t>(kInter * kHidden));
  std::vector<float> up_weights(gate_weights.size());
  std::vector<float> down_weights(gate_weights.size());
  for (std::size_t i = 0; i < values.size(); ++i)
    values[i] = 0.07f * static_cast<float>(static_cast<int>(i % 11) - 5);
  for (std::size_t c = 0; c < static_cast<std::size_t>(kInter); ++c) {
    const std::size_t row = c * static_cast<std::size_t>(kHidden);
    gate_weights[row + c % values.size()] = 0.125f;
    up_weights[row + (c + 3) % values.size()] = -0.25f;
  }
  for (std::size_t c = 0; c < values.size(); ++c)
    down_weights[c * static_cast<std::size_t>(kInter) + c] = 0.5f;
  graph::Array h = filled({1, 1, kHidden}, values);
  graph::Array wg = filled({kInter, kHidden}, gate_weights);
  graph::Array wu = filled({kInter, kHidden}, up_weights);
  graph::Array wd = filled({kHidden, kInter}, down_weights);
  auto reference_gate = graph::linear(h, wg);
  auto reference_up = graph::linear(h, wu);
  LSE_EXPECT_OK(reference_gate.eval());
  LSE_EXPECT_OK(reference_up.eval());
  auto reference_activation = graph::silu(reference_gate);
  LSE_EXPECT_OK(reference_activation.eval());
  auto reference_hidden = reference_activation * reference_up;
  LSE_EXPECT_OK(reference_hidden.eval());
  auto reference_out = graph::linear(reference_hidden, wd);
  const std::vector<float> reference = read_all(reference_out);
  LSE_EXPECT_EQ(reference.size(), static_cast<std::size_t>(kHidden));
  // Built in the model's order: both projections, then the pair over them.
  // The phase splitter walks the nodes as they were recorded, so building the
  // gate's silu before the up projection is a different graph to the one
  // lemonseed records and would not exercise this.
  graph::Array gate = graph::linear(h, wg);
  graph::Array up = graph::linear(h, wu);
  graph::Array hid = graph::silu(gate) * up;
  graph::Array out = graph::linear(hid, wd);

  sched->reset_accumulated_trace();
  const Status ev = out.eval();
  LSE_EXPECT(ev.ok());
  if (!ev.ok()) return;
  const auto& t = sched->last_trace();
  LSE_EXPECT_EQ(t.host_groups, 0u);
  const unsigned expected_launches =
      sched->backend().emitter()->staging() != nullptr ? 3u : 4u;
  LSE_EXPECT_EQ(t.kernels_launched, expected_launches);
  const std::vector<float> got = read_all(out);
  LSE_EXPECT_EQ(got.size(), reference.size());
  if (got.size() == reference.size() && !got.empty()) {
    LSE_EXPECT(std::memcmp(got.data(), reference.data(),
                           got.size() * sizeof(float)) == 0);
    for (std::size_t c = 0; c < got.size(); ++c) {
      const float g = 0.125f * values[c % values.size()];
      const float u = -0.25f * values[(c + 3) % values.size()];
      const float expected = 0.5f * (g / (1.0f + std::exp(-g))) * u;
      LSE_EXPECT(std::isfinite(got[c]));
      LSE_EXPECT_NEAR(got[c], expected, 2e-6f);
    }
  }

}

namespace {

// Adversarial ragged-attention checks. `ragged_pass` above is the batched
// entry; these need the pools rewritten between passes and the host reference
// run on the identical graph, so they build the pass themselves.
struct RagPass {
  std::vector<float> out;
  std::size_t per_row = 0;
};

RagPass rag_run(const std::vector<RaggedRow>& rows, const std::vector<float>& q,
                const std::vector<float>& kbuf, const std::vector<float>& vbuf,
                graph::MaskKind mask, int window, bool host_only) {
  RagPass r;
  r.per_row = static_cast<std::size_t>(kRagQh * kRagHd);
  const auto bucket = static_cast<std::int64_t>(rows.size());
  std::vector<float> table(static_cast<std::size_t>(bucket * kRagStride), 0.0f);
  std::vector<std::pair<std::int32_t, std::int32_t>> meta;
  for (std::size_t i = 0; i < rows.size(); ++i) {
    for (std::size_t b = 0; b < rows[i].blocks.size(); ++b) {
      table[i * static_cast<std::size_t>(kRagStride) + b] =
          static_cast<float>(rows[i].blocks[b]);
    }
    meta.push_back({rows[i].first, rows[i].len});
  }
  graph::Array kp =
      filled(Shape{kRagPool, kRagKvh, kv::kBlockSize, kRagHd}, kbuf);
  graph::Array vp =
      filled(Shape{kRagPool, kRagKvh, kv::kBlockSize, kRagHd}, vbuf);
  graph::Array ta = filled(Shape{bucket, kRagStride}, table);
  graph::Array ma = step_meta(meta);
  graph::Array qa = filled(Shape{bucket, kRagQh, 1, kRagHd}, q);
  graph::Array out =
      graph::sdpa_paged(qa, kp, vp, 0.25f, mask, window, ma, ta, kv::kBlockSize);
  graph::Scheduler* sched = graph::default_scheduler();
  const auto saved = sched->mode();
  if (host_only) sched->set_mode(graph::Scheduler::Mode::kHostOnly);
  const bool ok = out.eval().ok();
  sched->set_mode(saved);
  if (!ok) return r;
  r.out = read_all(out);
  return r;
}

std::size_t bitdiff(const std::vector<float>& a, std::size_t ao,
                    const std::vector<float>& b, std::size_t bo, std::size_t n) {
  std::size_t d = 0;
  for (std::size_t i = 0; i < n; ++i) {
    if (std::memcmp(&a[ao + i], &b[bo + i], sizeof(float)) != 0) ++d;
  }
  return d;
}

std::vector<float> rag_noise(std::size_t seed) {
  std::vector<float> v(static_cast<std::size_t>(kRagPool * kRagKvh *
                                                kv::kBlockSize * kRagHd));
  for (std::size_t i = 0; i < v.size(); ++i) v[i] = noise(i + seed);
  return v;
}

// Overwrite one block of a [blocks, Hkv, block_size, Hd] pool.
void poison_block(std::vector<float>& pool, std::int32_t blk, float base) {
  const auto per = static_cast<std::size_t>(kRagKvh * kv::kBlockSize * kRagHd);
  const std::size_t at = static_cast<std::size_t>(blk) * per;
  for (std::size_t i = 0; i < per; ++i) {
    pool[at + i] = base + static_cast<float>(i % 13) * 0.75f;
  }
}

}  // namespace

LSE_TEST(verify_a_pad_row_anywhere_in_the_bucket_changes_nothing) {
  // A pad row must be inert, and not only when it sits after every live row:
  // a session that retires mid-batch leaves a hole below the last live row, so
  // the case that matters is a pad *between* two live rows and a pad *before*
  // them. Both must leave every live row bit-identical to the tight batch.
  if (graph::default_scheduler() == nullptr) return;
  const std::vector<float> kb = rag_noise(17);
  const std::vector<float> vb = rag_noise(4211);
  const RaggedRow a{4, 5, {2}};
  const RaggedRow b{39, 40, {5, 1, 7}};
  const RaggedRow pad{0, 0, {}};

  const auto per = static_cast<std::size_t>(kRagQh * kRagHd);
  std::vector<float> q2(per * 2);
  for (std::size_t i = 0; i < q2.size(); ++i) q2[i] = noise(i + 88);
  const auto qrow = [&](std::size_t r) {
    return std::vector<float>(q2.begin() + static_cast<std::ptrdiff_t>(r * per),
                              q2.begin() +
                                  static_cast<std::ptrdiff_t>((r + 1) * per));
  };

  const RagPass tight = rag_run({a, b}, q2, kb, vb, graph::MaskKind::kCausal, 0,
                                false);
  LSE_EXPECT_EQ(tight.out.size(), per * 2);
  if (tight.out.size() != per * 2) return;

  struct Layout {
    const char* name;
    std::vector<RaggedRow> rows;
    std::vector<int> live;  // row index in this layout for a, then for b
  };
  std::vector<Layout> cases{
      {"trailing pads", {a, b, pad, pad}, {0, 1}},
      {"a hole between them", {a, pad, b}, {0, 2}},
      {"pads before and between", {pad, a, pad, b, pad}, {1, 3}},
  };
  for (const Layout& c : cases) {
    std::vector<float> q(per * c.rows.size(), 0.0f);
    for (std::size_t i = 0; i < c.rows.size(); ++i) {
      // Pad rows carry live-looking queries: inertness must come from the
      // descriptor, not from the row happening to be zero.
      const std::vector<float> src =
          static_cast<int>(i) == c.live[0]   ? qrow(0)
          : static_cast<int>(i) == c.live[1] ? qrow(1)
                                             : qrow(i % 2);
      std::copy(src.begin(), src.end(), q.begin() + static_cast<std::ptrdiff_t>(i * per));
    }
    const RagPass got =
        rag_run(c.rows, q, kb, vb, graph::MaskKind::kCausal, 0, false);
    LSE_EXPECT_EQ(got.out.size(), per * c.rows.size());
    if (got.out.size() != per * c.rows.size()) return;
    for (std::size_t k = 0; k < 2; ++k) {
      const std::size_t d =
          bitdiff(got.out, static_cast<std::size_t>(c.live[k]) * per, tight.out,
                  k * per, per);
      std::printf("       %-24s live row %zu: %zu of %zu differ\n", c.name, k, d,
                  per);
      LSE_EXPECT_EQ(d, 0u);
    }
    for (std::size_t i = 0; i < c.rows.size(); ++i) {
      if (static_cast<int>(i) == c.live[0] || static_cast<int>(i) == c.live[1]) {
        continue;
      }
      for (std::size_t e = 0; e < per; ++e) {
        LSE_EXPECT_EQ(got.out[i * per + e], 0.0f);
      }
    }
  }
}

LSE_TEST(verify_a_short_row_cannot_see_a_long_rows_keys) {
  // Leakage, stated as a dependency: rewrite every block the short row does not
  // own — the long row's three blocks and block 0, which is what the block
  // table's pad resolves to — and the short row's answer must not move a bit.
  // The long row must move, or the poison never reached the kernel.
  if (graph::default_scheduler() == nullptr) return;
  std::vector<float> kb = rag_noise(17);
  std::vector<float> vb = rag_noise(4211);
  const RaggedRow shortr{2, 3, {2}};
  const RaggedRow longr{39, 40, {5, 1, 7}};

  const auto per = static_cast<std::size_t>(kRagQh * kRagHd);
  std::vector<float> q(per * 2);
  for (std::size_t i = 0; i < q.size(); ++i) q[i] = noise(i + 505);

  const RagPass before =
      rag_run({shortr, longr}, q, kb, vb, graph::MaskKind::kCausal, 0, false);
  LSE_EXPECT_EQ(before.out.size(), per * 2);
  if (before.out.size() != per * 2) return;

  for (std::int32_t blk : {0, 1, 5, 7}) {
    poison_block(kb, blk, 40.0f + static_cast<float>(blk));
    poison_block(vb, blk, -70.0f - static_cast<float>(blk));
  }
  const RagPass after =
      rag_run({shortr, longr}, q, kb, vb, graph::MaskKind::kCausal, 0, false);
  LSE_EXPECT_EQ(after.out.size(), per * 2);
  if (after.out.size() != per * 2) return;

  const std::size_t moved_short = bitdiff(before.out, 0, after.out, 0, per);
  const std::size_t moved_long =
      bitdiff(before.out, per, after.out, per, per);
  std::printf("       poisoned blocks 0/1/5/7: short row moved %zu of %zu, "
              "long row moved %zu of %zu\n",
              moved_short, per, moved_long, per);
  LSE_EXPECT_EQ(moved_short, 0u);
  LSE_EXPECT(moved_long > 0);
}

LSE_TEST(verify_the_causal_mask_bounds_every_row_at_its_own_position) {
  // The mask is the whole correctness argument for raggedness: a row at
  // position p must see nothing past p whatever its live length says. Stated
  // exactly: giving a row 40 live keys when it sits at position 5 must be
  // bit-identical to giving it 6, because keys 6..39 are masked away and a
  // masked term adds 0.0f to the denominator and fma(0, v, acc) to the sum.
  //
  // Checked with the row under test in slot 1 and again in slot 0, at two
  // different positions, so a kernel right about row 0 alone fails it.
  if (graph::default_scheduler() == nullptr) return;
  const std::vector<float> kb = rag_noise(17);
  const std::vector<float> vb = rag_noise(4211);
  const auto per = static_cast<std::size_t>(kRagQh * kRagHd);
  std::vector<float> q(per * 2);
  for (std::size_t i = 0; i < q.size(); ++i) q[i] = noise(i + 313);

  struct Case {
    const char* name;
    std::int32_t under;  // which row is the one being bounded
    std::vector<RaggedRow> wide;
    std::vector<RaggedRow> tight;
    std::vector<RaggedRow> cut;  // one key short of the diagonal
  };
  const RaggedRow other{30, 31, {4, 6}};
  std::vector<Case> cases{
      {"row 1 at position 5",
       1,
       {other, {5, 40, {5, 1, 7}}},
       {other, {5, 6, {5, 1, 7}}},
       {other, {5, 5, {5, 1, 7}}}},
      {"row 0 at position 20",
       0,
       {{20, 40, {5, 1, 7}}, other},
       {{20, 21, {5, 1, 7}}, other},
       {{20, 20, {5, 1, 7}}, other}},
  };
  for (const Case& c : cases) {
    const RagPass wide =
        rag_run(c.wide, q, kb, vb, graph::MaskKind::kCausal, 0, false);
    const RagPass tight =
        rag_run(c.tight, q, kb, vb, graph::MaskKind::kCausal, 0, false);
    const RagPass cut =
        rag_run(c.cut, q, kb, vb, graph::MaskKind::kCausal, 0, false);
    if (wide.out.size() != per * 2 || tight.out.size() != per * 2 ||
        cut.out.size() != per * 2) {
      LSE_EXPECT(false);
      return;
    }
    const std::size_t at = static_cast<std::size_t>(c.under) * per;
    const std::size_t past = bitdiff(wide.out, at, tight.out, at, per);
    const std::size_t diag = bitdiff(wide.out, at, cut.out, at, per);
    std::printf("       %-20s keys past the diagonal: %zu of %zu differ; "
                "dropping the diagonal: %zu of %zu differ\n",
                c.name, past, per, diag, per);
    LSE_EXPECT_EQ(past, 0u);
    // The other row shares the pass and must not have moved either.
    const std::size_t sibling =
        static_cast<std::size_t>(c.under == 0 ? 1 : 0) * per;
    LSE_EXPECT_EQ(bitdiff(wide.out, sibling, tight.out, sibling, per), 0u);
    // Negative control: the identity above is not vacuous.
    LSE_EXPECT(diag > 0);
  }
}

LSE_TEST(verify_the_ragged_kernel_agrees_with_the_host_reference) {
  // The JIT kernel against interpreter.cpp, which walks the block table in
  // plain C++ and shares no code with the generator. Rows at four different
  // positions and lengths, causal and sliding-window.
  if (graph::default_scheduler() == nullptr) return;
  const std::vector<float> kb = rag_noise(17);
  const std::vector<float> vb = rag_noise(4211);
  const std::vector<RaggedRow> rows{
      {4, 5, {2}}, {39, 40, {5, 1, 7}}, {15, 16, {11}}, {0, 0, {}},
  };
  const auto per = static_cast<std::size_t>(kRagQh * kRagHd);
  std::vector<float> q(per * rows.size());
  for (std::size_t i = 0; i < q.size(); ++i) q[i] = noise(i + 88);

  struct Mode {
    const char* name;
    graph::MaskKind mask;
    int window;
  };
  for (const Mode& m : {Mode{"causal", graph::MaskKind::kCausal, 0},
                        Mode{"window 8", graph::MaskKind::kSlidingWindow, 8}}) {
    const RagPass dev = rag_run(rows, q, kb, vb, m.mask, m.window, false);
    const RagPass ref = rag_run(rows, q, kb, vb, m.mask, m.window, true);
    LSE_EXPECT_EQ(dev.out.size(), per * rows.size());
    LSE_EXPECT_EQ(ref.out.size(), dev.out.size());
    if (dev.out.size() != ref.out.size() || dev.out.empty()) return;
    for (std::size_t r = 0; r < rows.size(); ++r) {
      double max_abs = 0.0;
      double max_rel = 0.0;
      for (std::size_t i = 0; i < per; ++i) {
        const double a = dev.out[r * per + i];
        const double b = ref.out[r * per + i];
        const double e = std::fabs(a - b);
        max_abs = std::max(max_abs, e);
        const double mag = std::max(std::fabs(a), std::fabs(b));
        if (mag > 1e-6) max_rel = std::max(max_rel, e / mag);
      }
      std::printf("       %-9s row %zu (pos %d, len %2d): max_abs %.3e "
                  "max_rel %.3e\n",
                  m.name, r, rows[r].first, rows[r].len, max_abs, max_rel);
      LSE_EXPECT(max_abs < 1e-5);
      LSE_EXPECT(max_rel < 1e-5);
    }
  }
}

LSE_TEST(verify_a_one_token_prompt_decodes_the_same_beside_a_long_one) {
  // The widest length spread the fixture allows, end to end: a one-token prompt
  // and a forty-token one in the same batch, generating past a block boundary
  // so both rows cross into a second block at different steps. Each must give
  // the tokens it gives alone.
  Lemonseed fx = load_lemonseed();
  if (!fx) return;
  model::HybridLM& lm = *fx.lm;

  std::vector<std::uint32_t> longp;
  for (std::uint32_t i = 0; i < 40u; ++i) longp.push_back(100u + i * 7u);
  const std::vector<std::vector<std::uint32_t>> prompts{
      {41u}, longp, {7u, 8u}, {500u, 21u, 3u, 90u, 12u, 6u, 77u},
  };
  constexpr std::int32_t kWant = 24;  // > kv::kBlockSize, so blocks are added

  for (std::int32_t width : {2, 4}) {
    BatchLimits limits;
    limits.max_batch = width;
    limits.max_tokens = kWant;

    BatchScheduler together(lm, greedy_params(), limits);
    for (std::size_t i = 0; i < prompts.size(); ++i) {
      LSE_EXPECT_OK(
          together.submit({"v" + std::to_string(i), prompts[i], kWant}));
    }
    auto batched = together.run();
    LSE_EXPECT_OK(batched.status());
    if (!batched.ok()) return;
    LSE_EXPECT_EQ(batched->size(), prompts.size());
    if (batched->size() != prompts.size()) return;

    for (const SequenceResult& r : *batched) {
      const std::size_t i = static_cast<std::size_t>(r.id[1] - '0');
      BatchScheduler alone(lm, greedy_params(), limits);
      LSE_EXPECT_OK(alone.submit({r.id, prompts[i], kWant}));
      auto solo = alone.run();
      LSE_EXPECT_OK(solo.status());
      if (!solo.ok() || solo->empty()) return;
      const bool same = r.generated == (*solo)[0].generated;
      std::printf("       width %d  %s (prompt %zu): %s\n", width, r.id.c_str(),
                  prompts[i].size(), same ? "identical" : "DIFFERS");
      if (!same) {
        std::printf("         batched [%s]\n         alone   [%s]\n",
                    ids_to_string(r.generated).c_str(),
                    ids_to_string((*solo)[0].generated).c_str());
      }
      LSE_EXPECT_EQ(r.generated.size(), static_cast<std::size_t>(kWant));
      LSE_EXPECT(same);
    }
  }
}

LSE_TEST(verify_a_long_session_survives_short_ones_churning_beside_it) {
  // The point of the feature, as a diff. One session decoding 40 tokens holds a
  // row while eight two-token sessions take the other row in turn: each retires
  // mid-flight, hands its blocks back, and the next one is admitted into the
  // slot at a position the long row is nowhere near. The long session's tokens
  // must be exactly the ones it produces alone.
  //
  // Two rows and a pool small enough that the short sessions' blocks are
  // recycled, so the long row is decoding against a free list that is being
  // handed back and re-acquired under it every few steps.
  Lemonseed fx = load_lemonseed();
  if (!fx) return;
  model::HybridLM& lm = *fx.lm;

  const std::vector<std::uint32_t> longp{11u, 907u, 40u, 5u, 82u, 313u, 7u};
  constexpr std::int32_t kLong = 40;  // crosses three block boundaries

  BatchLimits limits;
  limits.max_batch = 2;
  limits.max_tokens = kLong;
  limits.kv_blocks = 8;
  limits.kv_reserve = 0;

  BatchScheduler mixed(lm, greedy_params(), limits);
  LSE_EXPECT_OK(mixed.submit({"long", longp, kLong}));
  for (std::uint32_t i = 0; i < 8u; ++i) {
    LSE_EXPECT_OK(mixed.submit(
        {"s" + std::to_string(i), {200u + i * 31u, 5u + i}, 2}));
  }
  auto got = mixed.run();
  LSE_EXPECT_OK(got.status());
  if (!got.ok()) return;
  LSE_EXPECT_EQ(got->size(), 9u);
  if (got->size() != 9u) return;

  BatchScheduler solo(lm, greedy_params(), limits);
  LSE_EXPECT_OK(solo.submit({"long", longp, kLong}));
  auto alone = solo.run();
  LSE_EXPECT_OK(alone.status());
  if (!alone.ok() || alone->empty()) return;

  const SequenceResult* batched = nullptr;
  for (const SequenceResult& r : *got) {
    if (r.id == "long") batched = &r;
  }
  LSE_EXPECT(batched != nullptr);
  if (batched == nullptr) return;

  // Both directions. The long row is always the furthest along, so a kernel
  // that shared one position across the batch would take *its* position and be
  // accidentally right about it; the rows that join behind it are the ones such
  // a kernel gets wrong. Checking only the long row is not a gate.
  for (const SequenceResult& r : *got) {
    if (r.id == "long") continue;
    const std::size_t i = static_cast<std::size_t>(r.id[1] - '0');
    BatchScheduler one(lm, greedy_params(), limits);
    LSE_EXPECT_OK(one.submit(
        {r.id, {200u + static_cast<std::uint32_t>(i) * 31u,
                5u + static_cast<std::uint32_t>(i)}, 2}));
    auto solo_short = one.run();
    LSE_EXPECT_OK(solo_short.status());
    if (!solo_short.ok() || solo_short->empty()) return;
    const bool same = r.generated == (*solo_short)[0].generated;
    if (!same) {
      std::printf("       short %s batched [%s] vs alone [%s]\n", r.id.c_str(),
                  ids_to_string(r.generated).c_str(),
                  ids_to_string((*solo_short)[0].generated).c_str());
    }
    LSE_EXPECT(same);
  }

  std::size_t first_diff = kLong;
  for (std::size_t i = 0; i < batched->generated.size() &&
                          i < (*alone)[0].generated.size();
       ++i) {
    if (batched->generated[i] != (*alone)[0].generated[i]) {
      first_diff = i;
      break;
    }
  }
  std::printf("       long row: %d admission(s), %d preemption(s), free %d of "
              "%d block(s) at the end; first differing token: %s\n",
              mixed.stats().admissions, mixed.stats().preemptions,
              mixed.kv_blocks_free(), mixed.kv_blocks_total(),
              first_diff == kLong ? "none" : std::to_string(first_diff).c_str());
  if (first_diff != static_cast<std::size_t>(kLong)) {
    std::printf("         batched [%s]\n         alone   [%s]\n",
                ids_to_string(batched->generated).c_str(),
                ids_to_string((*alone)[0].generated).c_str());
  }
  LSE_EXPECT_EQ(batched->generated.size(), static_cast<std::size_t>(kLong));
  LSE_EXPECT(batched->generated == (*alone)[0].generated);
  // The short rows genuinely came and went through the slot beside it.
  LSE_EXPECT(mixed.stats().admissions >= 9);
  LSE_EXPECT_EQ(mixed.kv_blocks_free(), mixed.kv_blocks_total());
}

LSE_TEST(verify_two_rows_prefill_together_from_different_positions) {
  // The multi-token ragged pass. Every other case here has T == 1, because a
  // row that is decoding has one pending token and the step width is the batch
  // minimum. Two rows both mid-prompt at different absolute positions is the
  // one arrangement that gives T > 1 *and* raggedness, and it is the only case
  // that exercises the per-row origin of RoPE and of the paged write across a
  // span of positions rather than a single one.
  //
  // Arranged, not hoped for: a one-token sequence takes a row for the first
  // step and leaves, so the sequence admitted into the freed row starts its
  // prompt while the other row is already two tokens deep.
  Lemonseed fx = load_lemonseed();
  if (!fx) return;
  model::HybridLM& lm = *fx.lm;

  std::vector<std::uint32_t> pa;
  for (std::uint32_t i = 0; i < 48u; ++i) pa.push_back(60u + i * 5u);
  std::vector<std::uint32_t> pb;
  for (std::uint32_t i = 0; i < 32u; ++i) pb.push_back(900u - i * 11u);
  // Sixteen tokens, so the row that joins after it starts a full block behind
  // the row already running. A shift of a token or two is invisible: RoPE and
  // the causal mask are both relative, so a batch whose rows are uniformly
  // displaced still answers correctly. It is the block the displaced row then
  // over-reads that does the damage, and that needs the gap to be a block.
  const std::vector<std::uint32_t> hog{4u,  9u,  21u, 33u, 44u, 51u, 62u, 70u,
                                       81u, 93u, 14u, 25u, 36u, 47u, 58u, 69u};
  constexpr std::int32_t kWant = 12;
  // Integration coverage, not the gate. Synthetic prompts drive this model into
  // a repeating attractor whose argmax survives quite large perturbations, so a
  // token diff here is evidence of a fault but agreement is not evidence of
  // correctness. The gate for T > 1 is
  // verify_a_multi_token_pass_is_ragged_too, which compares bits.

  BatchLimits limits;
  limits.max_batch = 2;
  limits.max_tokens = 64;

  BatchScheduler together(lm, greedy_params(), limits);
  LSE_EXPECT_OK(together.submit({"hog", hog, 1}));
  LSE_EXPECT_OK(together.submit({"a", pa, kWant}));
  LSE_EXPECT_OK(together.submit({"b", pb, kWant}));
  auto got = together.run();
  LSE_EXPECT_OK(got.status());
  if (!got.ok()) return;
  LSE_EXPECT_EQ(got->size(), 3u);
  if (got->size() != 3u) return;

  // Every token of both prompts, plus what they generated, in far fewer steps
  // than there are tokens: the prompts went in several at a time.
  const std::int32_t steps = together.stats().steps;
  std::printf("       %d step(s) for %zu prompt token(s) + %d generated\n",
              steps, pa.size() + pb.size() + hog.size(), 2 * kWant + 1);
  LSE_EXPECT(steps < static_cast<std::int32_t>(pa.size()));

  for (const SequenceResult& r : *got) {
    if (r.id == "hog") continue;
    const std::vector<std::uint32_t>& prompt = r.id == "a" ? pa : pb;
    BatchScheduler alone(lm, greedy_params(), limits);
    LSE_EXPECT_OK(alone.submit({r.id, prompt, kWant}));
    auto solo = alone.run();
    LSE_EXPECT_OK(solo.status());
    if (!solo.ok() || solo->empty()) return;
    const bool same = r.generated == (*solo)[0].generated;
    std::printf("       %s (prompt %zu): %s\n", r.id.c_str(), prompt.size(),
                same ? "identical" : "DIFFERS");
    if (!same) {
      std::printf("         batched [%s]\n         alone   [%s]\n",
                  ids_to_string(r.generated).c_str(),
                  ids_to_string((*solo)[0].generated).c_str());
    }
    LSE_EXPECT_EQ(r.generated.size(), static_cast<std::size_t>(kWant));
    LSE_EXPECT(same);
  }
}

LSE_TEST(verify_a_multi_token_pass_is_ragged_too) {
  // Every other ragged check here has T == 1, because a decoding row has one
  // pending token and the step width is the batch minimum. A row still feeding
  // its prompt beside a row that started earlier gives T > 1 at two different
  // origins, and that is the only arrangement in which the per-query term of
  // the causal mask, the per-row origin of RoPE and the span of positions the
  // paged write covers are all exercised at once.
  //
  // Bits, at the kernel, because the end-to-end form of this cannot be a gate:
  // a uniform displacement of one row is invisible to both RoPE and the causal
  // mask, which are relative, so the tokens can agree while the row is reading
  // the wrong slots.
  graph::Scheduler* sched = graph::default_scheduler();
  LSE_EXPECT(sched != nullptr);
  if (sched == nullptr) return;

  constexpr std::int64_t kT = 32;
  const std::vector<float> kb = rag_noise(17);
  const std::vector<float> vb = rag_noise(4211);
  graph::Array kp =
      filled(Shape{kRagPool, kRagKvh, kv::kBlockSize, kRagHd}, kb);
  graph::Array vp =
      filled(Shape{kRagPool, kRagKvh, kv::kBlockSize, kRagHd}, vb);

  // Row 0 is starting from nothing; row 1 is a full block further on. Row 2
  // holds no sequence.
  struct Row {
    std::int32_t first;
    std::int32_t len;
    std::vector<std::int32_t> blocks;
  };
  const std::vector<Row> rows{
      {0, 32, {3, 9}}, {16, 48, {5, 1, 7}}, {0, 0, {}},
  };
  const auto per_row = static_cast<std::size_t>(kRagQh * kT * kRagHd);
  std::vector<float> q(per_row * rows.size());
  for (std::size_t i = 0; i < q.size(); ++i) q[i] = noise(i + 1201);

  const auto run = [&](const std::vector<Row>& rs,
                       const std::vector<float>& qv) {
    const auto n = static_cast<std::int64_t>(rs.size());
    std::vector<float> table(static_cast<std::size_t>(n * kRagStride), 0.0f);
    std::vector<std::pair<std::int32_t, std::int32_t>> meta;
    for (std::size_t r = 0; r < rs.size(); ++r) {
      for (std::size_t i = 0; i < rs[r].blocks.size(); ++i) {
        table[r * static_cast<std::size_t>(kRagStride) + i] =
            static_cast<float>(rs[r].blocks[i]);
      }
      meta.push_back({rs[r].first, rs[r].len});
    }
    graph::Array ta = filled(Shape{n, kRagStride}, table);
    graph::Array ma = step_meta(meta);
    graph::Array qa = filled(Shape{n, kRagQh, kT, kRagHd}, qv);
    graph::Array o = graph::sdpa_paged(qa, kp, vp, 0.25f,
                                       graph::MaskKind::kCausal, 0, ma, ta,
                                       kv::kBlockSize);
    std::vector<float> out;
    if (o.eval().ok()) out = read_all(o);
    return out;
  };

  const std::vector<float> batched = run(rows, q);
  LSE_EXPECT_EQ(batched.size(), per_row * rows.size());
  if (batched.size() != per_row * rows.size()) return;

  for (std::size_t r = 0; r + 1 < rows.size(); ++r) {
    const std::vector<float> one(
        q.begin() + static_cast<std::ptrdiff_t>(r * per_row),
        q.begin() + static_cast<std::ptrdiff_t>((r + 1) * per_row));
    const std::vector<float> alone = run({rows[r]}, one);
    LSE_EXPECT_EQ(alone.size(), per_row);
    if (alone.size() != per_row) return;
    const std::size_t d = bitdiff(alone, 0, batched, r * per_row, per_row);
    std::printf("       T=%lld row %zu (pos %2d, len %2d): %zu of %zu differ\n",
                static_cast<long long>(kT), r, rows[r].first, rows[r].len, d,
                per_row);
    LSE_EXPECT_EQ(d, 0u);
  }
  for (std::size_t i = 0; i < per_row; ++i) {
    LSE_EXPECT_EQ(batched[2 * per_row + i], 0.0f);
  }
  // The two rows are not each other, so the comparison above had something to
  // catch.
  std::size_t same = 0;
  for (std::size_t i = 0; i < per_row; ++i) {
    if (batched[i] == batched[per_row + i]) ++same;
  }
  LSE_EXPECT(same < per_row);
}

LSE_TEST(verify_a_multi_token_write_covers_each_rows_own_span) {
  // The write side at T > 1: a row's span of positions crosses a block boundary
  // at its own offset, not at the batch's. Row 0 writes 14,15,16 — two blocks —
  // and row 1 writes 5,6,7 inside one.
  if (graph::default_scheduler() == nullptr) return;
  const std::int32_t bs = kv::kBlockSize;
  constexpr std::int64_t kKvh = 2;
  constexpr std::int64_t kW = 4;
  constexpr std::int64_t kT = 3;
  const std::int32_t pool_blocks = 6;
  const std::int32_t stride = 4;

  std::vector<float> zero(
      static_cast<std::size_t>(pool_blocks * kKvh * bs * kW), 0.0f);
  graph::Array pool = filled(Shape{pool_blocks, kKvh, bs, kW}, zero);
  std::vector<float> src(static_cast<std::size_t>(3 * kKvh * kT * kW));
  for (std::size_t i = 0; i < src.size(); ++i) src[i] = noise(i + 909) + 5.0f;
  graph::Array sa = filled(Shape{3, kKvh, kT, kW}, src);

  std::vector<float> table(static_cast<std::size_t>(3 * stride), 0.0f);
  table[0] = 1.0f;  // row 0: positions 14,15 -> block 1
  table[1] = 4.0f;  //         position  16   -> block 4
  table[static_cast<std::size_t>(stride)] = 2.0f;      // row 1: 5,6,7 -> block 2
  table[static_cast<std::size_t>(2 * stride)] = 3.0f;  // row 2 holds nothing
  graph::Array ta = filled(Shape{3, stride}, table);
  graph::Array meta = step_meta({{14, 17}, {5, 8}, {0, 0}});

  graph::Array written = graph::kv_page_write(pool, sa, meta, ta, bs);
  LSE_EXPECT_OK(written.eval());
  const std::vector<float> got = read_all(written);
  LSE_EXPECT_EQ(got.size(), zero.size());
  if (got.size() != zero.size()) return;

  const auto at = [&](std::int32_t blk, std::int64_t h, std::int32_t slot,
                      std::int64_t w) {
    return static_cast<std::size_t>(((blk * kKvh + h) * bs + slot) * kW + w);
  };
  const auto from = [&](std::int64_t r, std::int64_t h, std::int64_t t,
                        std::int64_t w) {
    return src[static_cast<std::size_t>(((r * kKvh + h) * kT + t) * kW + w)];
  };
  for (std::int64_t h = 0; h < kKvh; ++h) {
    for (std::int64_t w = 0; w < kW; ++w) {
      LSE_EXPECT_EQ(got[at(1, h, 14, w)], from(0, h, 0, w));
      LSE_EXPECT_EQ(got[at(1, h, 15, w)], from(0, h, 1, w));
      LSE_EXPECT_EQ(got[at(4, h, 0, w)], from(0, h, 2, w));
      for (std::int32_t t = 0; t < 3; ++t) {
        LSE_EXPECT_EQ(got[at(2, h, 5 + t, w)], from(1, h, t, w));
      }
    }
  }
  // Block 3 is the row that holds no sequence; block 0 is the table's pad and
  // the row 0 slot the span never reaches. Neither was written.
  for (std::int32_t blk : {0, 3, 5}) {
    for (std::size_t i = 0; i < static_cast<std::size_t>(kKvh * bs * kW); ++i) {
      LSE_EXPECT_EQ(got[static_cast<std::size_t>(blk * kKvh * bs * kW) + i],
                    0.0f);
    }
  }
}

LSE_TEST(verify_rope_rotates_each_row_to_its_own_position) {
  // RoPE reads the same descriptor. With T > 1 a row's angles run from its own
  // origin across its span; one shared origin rotates every row but the
  // furthest-along one to somebody else's position.
  if (graph::default_scheduler() == nullptr) return;
  constexpr std::int64_t kH = 2;
  constexpr std::int64_t kT = 4;
  constexpr std::int64_t kD = 8;
  constexpr std::int64_t kMaxT = 64;

  auto tables = ops::build_rope(static_cast<std::int32_t>(kD), kMaxT, 10000.0f);
  LSE_EXPECT_OK(tables.status());
  if (!tables.ok()) return;

  const auto per_row = static_cast<std::size_t>(kH * kT * kD);
  std::vector<float> x(per_row * 2);
  for (std::size_t i = 0; i < x.size(); ++i) x[i] = noise(i + 4004);

  graph::Array xa = filled(Shape{2, kH, kT, kD}, x);
  graph::Array off = step_meta({{0, 4}, {23, 27}});
  graph::Array both = graph::rope(xa, tables->cos, tables->sin, off);
  LSE_EXPECT_OK(both.eval());
  const std::vector<float> got = read_all(both);
  LSE_EXPECT_EQ(got.size(), per_row * 2);
  if (got.size() != per_row * 2) return;

  for (std::int32_t r = 0; r < 2; ++r) {
    const std::int32_t origin = r == 0 ? 0 : 23;
    std::vector<float> one(
        x.begin() + static_cast<std::ptrdiff_t>(static_cast<std::size_t>(r) * per_row),
        x.begin() + static_cast<std::ptrdiff_t>((static_cast<std::size_t>(r) + 1) * per_row));
    graph::Array x1 = filled(Shape{1, kH, kT, kD}, one);
    // The single-sequence form: a baked offset, which is what one sequence has.
    graph::Array o1 = graph::rope(x1, tables->cos, tables->sin, origin);
    LSE_EXPECT_OK(o1.eval());
    const std::vector<float> alone = read_all(o1);
    LSE_EXPECT_EQ(alone.size(), per_row);
    if (alone.size() != per_row) return;
    const std::size_t d =
        bitdiff(alone, 0, got, static_cast<std::size_t>(r) * per_row, per_row);
    std::printf("       rope row %d at position %d: %zu of %zu differ\n", r,
                origin, d, per_row);
    LSE_EXPECT_EQ(d, 0u);
  }
}


// --- multi-token prediction --------------------------------------------------
//
// The module ships beside a checkpoint and is not one: no embedding table, no
// head, one layer. These run against a synthetic Qwen3.5 pair, so they cover
// the wiring — load, rollback, accept, and token identity — on a bare machine.
// Acceptance rate is a property of trained weights and is not testable here.

namespace {

struct NamedTensor {
  std::string name;
  std::vector<std::int64_t> dims;
};

std::size_t tensor_elems(const NamedTensor& t) {
  std::size_t n = 1;
  for (std::int64_t d : t.dims) n *= static_cast<std::size_t>(d);
  return n;
}

std::string safetensors_header(const std::vector<NamedTensor>& tensors,
                               std::size_t* total) {
  std::string header = "{";
  std::size_t offset = 0;
  for (const NamedTensor& t : tensors) {
    std::string dims;
    for (std::int64_t d : t.dims) {
      if (!dims.empty()) dims += ",";
      dims += std::to_string(d);
    }
    const std::size_t bytes = tensor_elems(t) * 4;
    if (header.size() > 1) header += ",";
    header += "\"" + t.name + "\":{\"dtype\":\"F32\",\"shape\":[" + dims +
              "],\"data_offsets\":[" + std::to_string(offset) + "," +
              std::to_string(offset + bytes) + "]}";
    offset += bytes;
  }
  header += "}";
  while (header.size() % 8 != 0) header += " ";
  *total = offset;
  return header;
}

// `value(name, index)` fills each tensor, so a fixture can hand one tensor a
// structure and let the rest take filler.
template <typename Fill>
void write_shaped(const std::filesystem::path& path,
                  const std::vector<NamedTensor>& tensors, const Fill& value) {
  std::size_t total = 0;
  const std::string header = safetensors_header(tensors, &total);
  std::ofstream out(path, std::ios::binary);
  const std::uint64_t n = header.size();
  out.write(reinterpret_cast<const char*>(&n), sizeof(n));
  out.write(header.data(), static_cast<std::streamsize>(header.size()));
  std::vector<float> data;
  for (const NamedTensor& t : tensors) {
    data.resize(tensor_elems(t));
    for (std::size_t i = 0; i < data.size(); ++i) data[i] = value(t.name, i);
    out.write(reinterpret_cast<const char*>(data.data()),
              static_cast<std::streamsize>(data.size() * 4));
  }
}

// Deterministic and not symmetric: an all-zero module drafts token 0 every
// step, which would make a rejection test pass for the wrong reason.
float filler(std::size_t i) {
  return 0.03f * static_cast<float>(static_cast<int>((i * 7) % 23) - 11);
}

// `gdn_head_dim` is a knob because the device Gated DeltaNet kernel only emits
// at 16, 32, 64 and 128: any other width puts the scan on the host, which is
// the one shape of pass a speculative rollback cannot replace.
model::Config mtp_test_config(std::int32_t gdn_head_dim = 16) {
  model::Config c;
  c.vocab_size = 64;
  c.hidden_size = 32;
  c.num_layers = 4;
  c.full_attention_interval = 4;
  c.global_attention_layers.clear();
  c.sliding_window = 0;
  c.attn_q_heads = 2;
  c.attn_kv_heads = 1;
  c.attn_head_dim = 16;
  c.rope_dim = 4;
  c.rope_theta = 10000000.0f;
  c.gdn_qk_heads = 2;
  c.gdn_v_heads = 2;
  c.gdn_head_dim = gdn_head_dim;
  c.gdn_conv_kernel = 4;
  c.mlp_intermediate = 64;
  c.num_experts = 0;
  c.num_active_experts = 0;
  c.num_shared_experts = 0;
  c.expert_intermediate = 0;
  c.tie_word_embeddings = true;
  c.dtype = "float32";
  c.kv_length = 64;
  c.mtp_layers = 1;
  return c;
}

std::vector<NamedTensor> qwen_dense_tensors(const model::Config& c) {
  const std::int64_t h = c.hidden_size;
  const std::int64_t qh = c.attn_q_heads, kvh = c.attn_kv_heads;
  const std::int64_t ahd = c.attn_head_dim;
  const std::int64_t kh = c.gdn_qk_heads, vh = c.gdn_v_heads;
  const std::int64_t ghd = c.gdn_head_dim;
  const std::int64_t conv_dim = 2 * kh * ghd + vh * ghd;

  std::vector<NamedTensor> t;
  t.push_back({"language_model.model.embed_tokens.weight", {c.vocab_size, h}});
  t.push_back({"language_model.model.norm.weight", {h}});
  for (std::int32_t i = 0; i < c.num_layers; ++i) {
    const std::string p =
        "language_model.model.layers." + std::to_string(i) + ".";
    t.push_back({p + "input_layernorm.weight", {h}});
    t.push_back({p + "post_attention_layernorm.weight", {h}});
    if (c.is_attention_layer(i)) {
      const std::string a = p + "self_attn.";
      t.push_back({a + "q_proj.weight", {2 * qh * ahd, h}});
      t.push_back({a + "k_proj.weight", {kvh * ahd, h}});
      t.push_back({a + "v_proj.weight", {kvh * ahd, h}});
      t.push_back({a + "o_proj.weight", {h, qh * ahd}});
      t.push_back({a + "q_norm.weight", {ahd}});
      t.push_back({a + "k_norm.weight", {ahd}});
    } else {
      const std::string g = p + "linear_attn.";
      t.push_back({g + "in_proj_qkv.weight", {conv_dim, h}});
      t.push_back({g + "in_proj_z.weight", {vh * ghd, h}});
      t.push_back({g + "in_proj_a.weight", {vh, h}});
      t.push_back({g + "in_proj_b.weight", {vh, h}});
      t.push_back({g + "conv1d.weight", {conv_dim, c.gdn_conv_kernel, 1}});
      t.push_back({g + "A_log", {vh}});
      t.push_back({g + "dt_bias", {vh}});
      t.push_back({g + "norm.weight", {ghd}});
      t.push_back({g + "out_proj.weight", {h, vh * ghd}});
    }
    const std::string m = p + "mlp.";
    t.push_back({m + "gate_proj.weight", {c.mlp_intermediate, h}});
    t.push_back({m + "up_proj.weight", {c.mlp_intermediate, h}});
    t.push_back({m + "down_proj.weight", {h, c.mlp_intermediate}});
  }
  return t;
}

// The module's own file, named exactly as mlx-community/Qwen3.8-27B-MTP-4bit
// names it: no `language_model.model` prefix and no layer index above zero.
std::vector<NamedTensor> mtp_tensors(const model::Config& c) {
  const std::int64_t h = c.hidden_size;
  const std::int64_t qh = c.attn_q_heads, kvh = c.attn_kv_heads;
  const std::int64_t ahd = c.attn_head_dim;
  return {
      {"fc.weight", {h, 2 * h}},
      {"pre_fc_norm_hidden.weight", {h}},
      {"pre_fc_norm_embedding.weight", {h}},
      {"norm.weight", {h}},
      {"layers.0.input_layernorm.weight", {h}},
      {"layers.0.post_attention_layernorm.weight", {h}},
      {"layers.0.self_attn.q_proj.weight", {2 * qh * ahd, h}},
      {"layers.0.self_attn.k_proj.weight", {kvh * ahd, h}},
      {"layers.0.self_attn.v_proj.weight", {kvh * ahd, h}},
      {"layers.0.self_attn.o_proj.weight", {h, qh * ahd}},
      {"layers.0.self_attn.q_norm.weight", {ahd}},
      {"layers.0.self_attn.k_norm.weight", {ahd}},
      {"layers.0.mlp.gate_proj.weight", {c.mlp_intermediate, h}},
      {"layers.0.mlp.up_proj.weight", {c.mlp_intermediate, h}},
      {"layers.0.mlp.down_proj.weight", {h, c.mlp_intermediate}},
  };
}

std::string mtp_config_json(const model::Config& c) {
  return std::string("{\"tie_word_embeddings\": true, \"text_config\": {") +
         "\"vocab_size\": " + std::to_string(c.vocab_size) +
         ", \"hidden_size\": " + std::to_string(c.hidden_size) +
         ", \"num_hidden_layers\": " + std::to_string(c.num_layers) +
         ", \"rms_norm_eps\": 1e-06" +
         ", \"full_attention_interval\": " +
         std::to_string(c.full_attention_interval) +
         ", \"num_attention_heads\": " + std::to_string(c.attn_q_heads) +
         ", \"num_key_value_heads\": " + std::to_string(c.attn_kv_heads) +
         ", \"head_dim\": " + std::to_string(c.attn_head_dim) +
         ", \"linear_num_key_heads\": " + std::to_string(c.gdn_qk_heads) +
         ", \"linear_num_value_heads\": " + std::to_string(c.gdn_v_heads) +
         ", \"linear_conv_kernel_dim\": " +
         std::to_string(c.gdn_conv_kernel) +
         ", \"linear_key_head_dim\": " + std::to_string(c.gdn_head_dim) +
         ", \"linear_value_head_dim\": " + std::to_string(c.gdn_head_dim) +
         ", \"intermediate_size\": " + std::to_string(c.mlp_intermediate) +
         ", \"rope_parameters\": {\"rope_theta\": 10000000.0,"
         " \"partial_rotary_factor\": 0.25}" +
         ", \"max_position_embeddings\": 128, \"dtype\": \"float32\"" +
         ", \"mtp_num_hidden_layers\": 1" +
         ", \"mtp_use_dedicated_embeddings\": false}}";
}

struct MtpFixture {
  model::Config config;
  model::SafeTensors weights;
  std::unique_ptr<model::HybridLM> lm;
  std::unique_ptr<model::MtpModule> mtp;
  std::string module_dir;
  bool ok = false;
};

// `passthrough` writes a module that drafts the decoder's own next token
// rather than the one after it: fc keeps the hidden half and drops the
// embedding half, and the layer's projections are zero so the block is exactly
// its residual. On a sequence that has reached a fixed point those two are the
// same token, which is how the accept path gets exercised without trained
// weights.
MtpFixture build_mtp_fixture(bool passthrough = false,
                             std::int32_t gdn_head_dim = 16,
                             bool recurrent_only = false,
                             std::int32_t kv_length = 64,
                             bool attention_only = false,
                             bool transitions = false) {
  MtpFixture fx;
  fx.config = mtp_test_config(gdn_head_dim);
  fx.config.kv_length = kv_length;
  if (recurrent_only) fx.config.full_attention_interval = fx.config.num_layers + 1;
  if (attention_only) fx.config.full_attention_interval = 1;
  if (transitions) fx.config.tie_word_embeddings = false;
  const std::int64_t h = fx.config.hidden_size;
  std::error_code ec;
  const std::filesystem::path base =
      std::filesystem::temp_directory_path() /
      (passthrough ? "lse-mtp-fixture-pt"
                   : "lse-mtp-fixture-" + std::to_string(gdn_head_dim) +
                         (recurrent_only ? "-recurrent" : "") +
                         (attention_only ? "-attention" : "") +
                         (transitions ? "-transitions" : ""));
  std::filesystem::create_directories(base / "mtp", ec);
  fx.module_dir = (base / "mtp").string();

  // An untrained head puts the top two logits within a few ULP of each other
  // and the greedy choice then follows the last bit of an f32 sum, which two
  // differently shaped kernels are not required to agree on. Spacing the tied
  // embedding's rows makes most of the fixture's greedy steps a property of
  // the model rather than of the rounding.
  const auto parent = [&](const std::string& name, std::size_t i) {
    const std::size_t width = static_cast<std::size_t>(h);
    if (transitions) {
      const auto row = i / width, col = i % width;
      if (name == "language_model.model.embed_tokens.weight")
        return row < width && col == row ? 1.0f : 0.0f;
      if (name == "language_model.lm_head.weight")
        return row < width && col == (row + width - 1) % width ? 1.0f : 0.0f;
      if (name.find("norm") != std::string::npos) return 1.0f;
      return 0.0f;
    }
    float v = filler(i);
    if (name == "language_model.model.embed_tokens.weight" &&
        i % width == (i / width) % width) {
      v += 2.0f;
    }
    return v;
  };
  const auto module = [&](const std::string& name, std::size_t i) {
    if (!passthrough && !transitions) return filler(i);
    if (name == "fc.weight") {
      const std::size_t width = 2 * static_cast<std::size_t>(h);
      const std::size_t row = i / width;
      const std::size_t col = i % width;
      const auto half = transitions && !passthrough ? 0 : static_cast<std::size_t>(h);
      return col == half + row ? 1.0f : 0.0f;
    }
    if (name.find(".weight") != std::string::npos &&
        name.find("norm") != std::string::npos) {
      return 1.0f;
    }
    return 0.0f;
  };

  auto target_tensors = qwen_dense_tensors(fx.config);
  if (transitions)
    target_tensors.push_back({"language_model.lm_head.weight", {fx.config.vocab_size, h}});
  write_shaped(base / "model.safetensors", target_tensors, parent);
  write_shaped(base / "mtp" / "model.safetensors", mtp_tensors(fx.config),
               module);
  {
    std::ofstream out(base / "mtp" / "config.json");
    out << mtp_config_json(fx.config);
  }

  auto st = model::SafeTensors::open((base / "model.safetensors").string());
  if (!st.ok()) return fx;
  fx.weights = st.release();
  // The attention-only fixture deliberately lacks the hybrid GDN marker.
  auto built = model::build_model(fx.config, fx.weights,
                                 attention_only ? "qwen3.5" : "");
  if (!built.ok()) return fx;
  fx.lm = built.release();
  model::WeightBinder binder(fx.weights);
  if (!fx.lm->load(binder).ok()) return fx;

  auto mod = model::MtpModule::open(fx.module_dir, fx.config, *fx.lm);
  if (!mod.ok()) {
    std::printf("       MTP open failed: %s\n", mod.status().to_string().c_str());
    return fx;
  }
  fx.mtp = mod.release();
  fx.ok = true;
  return fx;
}

std::vector<float> array_to_host(const graph::Array& a) {
  std::vector<float> v;
  if (!a.valid()) return v;
  graph::Scheduler* sched = graph::default_scheduler();
  if (sched == nullptr) return v;
  // The pass left it on the device and nothing asked for it on the host, so
  // the mirror is stale until this moves it.
  if (!graph::interpreter::sync_from_device(*a.node(), sched->backend()).ok()) {
    return v;
  }
  v.resize(a.shape().elem_count());
  if (!graph::interpreter::read_raw(*a.node(), v.data(),
                                    v.size() * sizeof(float))
           .ok()) {
    v.clear();
  }
  return v;
}

// Every buffer a pass can leave behind: each Gated DeltaNet layer's recurrent
// state and conv tail, and each attention layer's paged key/value pools.
std::vector<std::vector<float>> states_image(
    std::vector<model::MixerState>& states) {
  std::vector<std::vector<float>> out;
  for (model::MixerState& st : states) {
    out.push_back(array_to_host(st.gdn_state));
    out.push_back(array_to_host(st.gdn_conv_qkv));
    out.push_back(array_to_host(st.key_cache));
    out.push_back(array_to_host(st.value_cache));
  }
  return out;
}

std::size_t images_differ(const std::vector<std::vector<float>>& a,
                          const std::vector<std::vector<float>>& b) {
  if (a.size() != b.size()) return a.size() + b.size();
  std::size_t n = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (a[i].size() != b[i].size()) {
      ++n;
      continue;
    }
    for (std::size_t j = 0; j < a[i].size(); ++j) {
      if (a[i][j] != b[i][j]) {
        ++n;
        break;
      }
    }
  }
  return n;
}

graph::Array ids_array(const std::vector<std::uint32_t>& ids) {
  graph::Array a = graph::Array::zeros(
      Shape{1, static_cast<std::int64_t>(ids.size())}, DType::kF32);
  if (!a.eval().ok()) return {};
  for (std::size_t i = 0; i < ids.size(); ++i) {
    graph::interpreter::store_element(*a.node(), i, static_cast<float>(ids[i]));
  }
  return a;
}

}  // namespace

LSE_TEST(the_mtp_module_loads_beside_a_checkpoint_and_is_not_one_itself) {
  MtpFixture fx = build_mtp_fixture();
  LSE_EXPECT(fx.ok);
  if (!fx.ok) return;
  LSE_EXPECT_EQ(fx.mtp->position(), 0);
  LSE_EXPECT_EQ(fx.mtp->config().num_layers, 1);
  // ...and its one layer attends, which is what its self_attn tensors describe.
  // A layer index that answered otherwise would build a Gated DeltaNet against
  // them and fail on a tensor name rather than on the shape that is wrong.
  LSE_EXPECT(fx.mtp->config().is_attention_layer(0));

  // The registry is right to refuse the module standalone, and that refusal is
  // why it needs its own load path rather than an architecture entry.
  auto st = model::SafeTensors::open(fx.module_dir + "/model.safetensors");
  LSE_EXPECT(st.ok());
  if (!st.ok()) return;
  auto arch = model::detect_architecture(fx.config, *st);
  LSE_EXPECT(!arch.ok());
}

LSE_TEST(mtp_device_hidden_matches_host_input_across_depth3_and_width_reuse) {
  MtpFixture fx = build_mtp_fixture();
  LSE_EXPECT(fx.ok);
  if (!fx.ok) return;
  const auto run = [&](bool device_input) {
    fx.mtp->reset();
    std::vector<std::uint32_t> proposals;
    for (std::int64_t rows : {3, 1, 3, 1, 7, 1}) {
      std::vector<float> values(static_cast<std::size_t>(rows * fx.config.hidden_size));
      for (std::size_t i = 0; i < values.size(); ++i) {
        values[i] = filler(i + static_cast<std::size_t>(fx.mtp->position()) * 17);
      }
      std::vector<std::uint32_t> ids(static_cast<std::size_t>(rows));
      for (std::size_t i = 0; i < ids.size(); ++i) ids[i] = 2u + static_cast<std::uint32_t>(i);
      graph::Array input = graph::Array::zeros(
          Shape{1, rows, fx.config.hidden_size}, DType::kF32);
      LSE_EXPECT_OK(input.materialize());
      auto* sched = graph::default_scheduler();
      if (sched == nullptr) return std::vector<std::uint32_t>{};
      for (std::size_t i = 0; i < values.size(); ++i) {
        graph::interpreter::store_element(*input.node(), i, values[i]);
      }
      input.node()->host_dirty = true;
      LSE_EXPECT_OK(graph::interpreter::sync_to_device(*input.node(), sched->backend()));
      const auto first = fx.mtp->position();
      auto got = device_input
                     ? fx.mtp->draft_chain(input, ids, first, 3)
                     : fx.mtp->draft_chain(std::span<const float>(values), ids, first, 3);
      LSE_EXPECT(got.ok());
      if (!got.ok()) return std::vector<std::uint32_t>{};
      proposals.insert(proposals.end(), got->begin(), got->end());
      LSE_EXPECT_EQ(fx.mtp->position(), first + rows + 2);
      LSE_EXPECT(array_to_host(input) == values);
    }
    return proposals;
  };
  const auto host = run(false);
  const auto device = run(true);
  LSE_EXPECT_EQ(host.size(), 18u);
  LSE_EXPECT(device == host);
  LSE_EXPECT(run(true) == device);
}

LSE_TEST(mtp_device_hidden_rejects_invalid_inputs_before_changing_position) {
  MtpFixture fx = build_mtp_fixture();
  LSE_EXPECT(fx.ok);
  if (!fx.ok) return;
  const std::uint32_t token = 2;
  const std::span<const std::uint32_t> ids(&token, 1);
  for (const graph::Array& bad : {
           graph::Array{},
           graph::Array::zeros(Shape{1, 1, fx.config.hidden_size}, DType::kBF16),
           graph::Array::zeros(Shape{1, 2, fx.config.hidden_size}, DType::kF32)}) {
    auto got = fx.mtp->draft(bad, ids, 0);
    LSE_EXPECT(!got.ok());
    LSE_EXPECT(got.status().code() == StatusCode::kInvalidArgument);
    LSE_EXPECT_EQ(fx.mtp->position(), 0);
  }
  auto empty = fx.mtp->draft_chain(graph::Array{}, ids, 0, 0);
  LSE_EXPECT(empty.ok() && empty->empty());
  LSE_EXPECT_EQ(fx.mtp->position(), 0);
}

LSE_TEST(a_rejected_draft_leaves_the_caches_where_a_clean_pass_would) {
  // A replacement pass needs the device's held carry inputs. The CPU
  // interpreter mutates its carry in place and correctly refuses that pass.
  if (graph::default_scheduler()->backend().emitter() == nullptr) {
    LSE_SKIP("speculative rollback requires device replay");
  }
  // The paged KV rolls back by cursor — the redo overwrites the same slots —
  // but the Gated DeltaNet state does not: it is a value the pass replaces.
  // What makes that safe is that a `replaces_previous` pass starts from the
  // same carried input the discarded one did, so the two arms below must agree
  // bit for bit. Both end on the same two-row pass, so nothing here depends on
  // how a two-row kernel rounds against a one-row one.
  MtpFixture fx = build_mtp_fixture();
  LSE_EXPECT(fx.ok);
  if (!fx.ok) return;
  const std::vector<std::uint32_t> prompt{2, 11, 33};
  const auto at = static_cast<std::int32_t>(prompt.size());

  const auto run = [&](bool reject_first, std::vector<float>* hidden,
                       std::vector<std::vector<float>>* image) {
    std::vector<model::MixerState> st = fx.lm->make_states();
    auto pre = fx.lm->hidden(ids_array(prompt), &st, nullptr);
    if (!pre.ok()) return false;
    if (reject_first) {
      // The rejected pass: a second token the decoder did not choose.
      auto bad = fx.lm->hidden(ids_array({19, 7}), &st, nullptr);
      if (!bad.ok()) return false;
      fx.lm->rewind(st, at);
    }
    auto got = fx.lm->hidden(ids_array({19, 18}), &st, nullptr, nullptr, nullptr,
                             reject_first);
    if (!got.ok()) return false;
    *hidden = array_to_host(*got);
    *image = states_image(st);
    return true;
  };

  std::vector<float> clean_hidden, redo_hidden;
  std::vector<std::vector<float>> clean_image, redo_image;
  LSE_EXPECT(run(false, &clean_hidden, &clean_image));
  LSE_EXPECT(run(true, &redo_hidden, &redo_image));
  if (clean_hidden.empty() || redo_hidden.empty()) return;

  LSE_EXPECT_EQ(redo_hidden.size(), clean_hidden.size());
  std::size_t hidden_diff = 0;
  for (std::size_t i = 0; i < clean_hidden.size() && i < redo_hidden.size(); ++i) {
    if (clean_hidden[i] != redo_hidden[i]) ++hidden_diff;
  }
  LSE_EXPECT_EQ(hidden_diff, 0u);
  LSE_EXPECT_EQ(images_differ(clean_image, redo_image), 0u);
}

LSE_TEST(a_rebuild_cannot_pretend_to_replace_the_pass_before_it) {
  // The rollback is a replay of a held program. A rebuild would record the
  // graph from the state the discarded pass produced, which is the one thing
  // it must not start from, so it is refused rather than silently continued.
  MtpFixture fx = build_mtp_fixture();
  LSE_EXPECT(fx.ok);
  if (!fx.ok) return;
  std::vector<model::MixerState> st = fx.lm->make_states();
  auto pre = fx.lm->hidden(ids_array({2, 11, 33}), &st, nullptr);
  LSE_EXPECT(pre.ok());
  if (!pre.ok()) return;
  // A width the cache has never held, so no program can be replayed for it.
  auto refused = fx.lm->hidden(ids_array({19, 18}), &st, nullptr, nullptr,
                               nullptr, /*replaces_previous=*/true);
  LSE_EXPECT(!refused.ok());
}

LSE_TEST(a_pass_that_ran_on_the_host_cannot_be_replaced) {
  // The rollback rests on the decoder's carried state still being where the
  // discarded pass found it, and that is a property of the device path: a
  // group the host ran leaves it somewhere else. A head dim the Gated DeltaNet
  // kernel does not emit at is the cheapest way to produce one, and the point
  // is that the answer is a refusal rather than a continuation from the state
  // the pass was meant to discard.
  MtpFixture fx = build_mtp_fixture(/*passthrough=*/false, /*gdn_head_dim=*/8);
  LSE_EXPECT(fx.ok);
  if (!fx.ok) return;
  std::vector<model::MixerState> st = fx.lm->make_states();
  LSE_EXPECT(fx.lm->hidden(ids_array({2, 11, 33}), &st, nullptr).ok());
  LSE_EXPECT(fx.lm->hidden(ids_array({19, 18}), &st, nullptr).ok());
  fx.lm->rewind(st, 3);
  auto refused = fx.lm->hidden(ids_array({19, 18}), &st, nullptr, nullptr,
                               nullptr, /*replaces_previous=*/true);
  LSE_EXPECT(!refused.ok());
  if (refused.ok()) return;
  LSE_EXPECT(refused.status().message().find("host") != std::string::npos);
}

LSE_TEST(speculating_gives_the_tokens_a_plain_decode_gives) {
  if (graph::default_scheduler()->backend().emitter() == nullptr) {
    LSE_SKIP("speculative rollback requires device replay");
  }
  // A one-hot target has separated logits. The hidden-dependent module drafts
  // the previous prediction, so rejection is independent of reduction rounding.
  MtpFixture fx = build_mtp_fixture(true, 16, false, 64, false, true);
  LSE_EXPECT(fx.ok);
  if (!fx.ok) return;

  GenerationLimits limits;
  limits.mtp_depth = 1;
  // Odd, so the last speculative step lands on its second half and the session
  // ends holding exactly the text it emitted.
  limits.max_tokens = 7;
  GenerationLimits prefill_only;
  prefill_only.max_tokens = 0;

  const std::vector<std::vector<std::uint32_t>> prompts{
      {2, 11, 13},    {1, 5, 9, 17},  {7, 7, 7},        {20, 3},
      {30, 1, 2, 3, 4}, {12, 24, 5, 6}, {13, 2, 19}, {8, 8, 9, 10, 11}};

  std::size_t compared = 0;
  std::uint32_t rejections = 0;
  for (const std::vector<std::uint32_t>& prompt : prompts) {
    Generator plain(*fx.lm, greedy_params());
    Session ps("plain", fx.lm->num_layers());
    auto want = plain.generate(ps, prompt, limits);
    LSE_EXPECT(want.ok());
    if (!want.ok()) return;
    std::vector<std::uint32_t> expected;
    for (std::uint32_t token = (prompt.back() + 1) % 32;
         expected.size() < static_cast<std::size_t>(limits.max_tokens);
         token = (token + 1) % 32) {
      expected.push_back(token);
    }
    LSE_EXPECT(*want == expected);

    Generator split(*fx.lm, greedy_params());
    Session ss("split", fx.lm->num_layers());
    const std::vector<std::uint32_t> head(prompt.begin(), prompt.end() - 1);
    LSE_EXPECT(split.generate(ss, head, prefill_only).ok());
    auto again = split.generate(ss, prompt, limits);
    LSE_EXPECT(again.ok());
    if (!again.ok()) return;
    LSE_EXPECT(*again == *want);

    Generator spec(*fx.lm, greedy_params());
    spec.use_mtp(*fx.mtp);
    Session sp("spec", fx.lm->num_layers());
    auto got = spec.generate(sp, prompt, limits);
    LSE_EXPECT(got.ok());
    if (!got.ok()) return;
    LSE_EXPECT(spec.stats().spec_steps > 0);
    rejections += spec.stats().spec_tested - spec.stats().spec_accepted;
    ++compared;
    LSE_EXPECT(*got == *want);
    if (*got != *want) {
      std::printf("       [%s] plain %s  spec %s\n",
                  ids_to_string(prompt).c_str(), ids_to_string(*want).c_str(),
                  ids_to_string(*got).c_str());
    }
  }
  LSE_EXPECT_EQ(compared, prompts.size());
  // A run in which nothing was ever rejected would not have exercised the redo.
  LSE_EXPECT(rejections > 0);
}

LSE_TEST(an_accepted_draft_still_gives_the_decoders_own_tokens) {
  if (graph::default_scheduler()->backend().emitter() == nullptr) {
    LSE_SKIP("speculative rollback requires device replay");
  }
  // The other half of the loop: a proposal the decoder agrees with is taken
  // without a second pass, and the token that rides along with it is the one
  // the decoder's own second row produced.
  MtpFixture fx = build_mtp_fixture(/*passthrough=*/true);
  LSE_EXPECT(fx.ok);
  if (!fx.ok) return;

  GenerationLimits limits;
  limits.mtp_depth = 1;
  limits.max_tokens = 7;
  const std::vector<std::uint32_t> prompt{2, 11, 33};

  Generator plain(*fx.lm, greedy_params());
  Session ps("plain", fx.lm->num_layers());
  auto want = plain.generate(ps, prompt, limits);
  LSE_EXPECT(want.ok());
  if (!want.ok()) return;

  Generator spec(*fx.lm, greedy_params());
  spec.use_mtp(*fx.mtp);
  Session sp("spec", fx.lm->num_layers());
  auto got = spec.generate(sp, prompt, limits);
  LSE_EXPECT(got.ok());
  if (!got.ok()) return;

  LSE_EXPECT(spec.stats().spec_accepted > 0);
  LSE_EXPECT(*got == *want);
  if (*got != *want) {
    std::printf("       accepted %u/%u  plain %s  spec %s\n",
                spec.stats().spec_accepted, spec.stats().spec_steps,
                ids_to_string(*want).c_str(), ids_to_string(*got).c_str());
  }
}

namespace {
void expect_mtp_resident(const MtpFixture& fixture, Session& session, std::int32_t covered);
}  // namespace

LSE_TEST(adaptive_verify_widths_give_the_tokens_a_plain_decode_gives) {
  if (graph::default_scheduler()->backend().emitter() == nullptr) {
    LSE_SKIP("speculative rollback requires device replay");
  }
  // A sampled request whose sampler keeps only the top token decodes the
  // argmax through the adaptive path: each step chains as deep as the width
  // policy picks and verifies the prefix it keeps, so passes of every width
  // from a plain step to a full chain alternate, swapping carried buffers
  // between their programs. The tokens must be the plain decode's, and the
  // MTP and session state must end where the text does. The target's logits
  // are separated (a one-hot transition), so no width can flip a near-tie;
  // over the filler fixture even a fixed MTP depth differs from a plain
  // decode in the last bits. One module drafts the previous prediction
  // (rejected), the other passes the right token through (accepted).
  SamplingParams top_one;
  top_one.temperature = 1.0f;
  top_one.top_k = 1;
  const std::vector<std::vector<std::uint32_t>> prompts{
      {2, 11, 13}, {1, 5, 9, 17}, {7, 7, 7}, {20, 3}, {30, 1, 2, 3, 4}, {12, 24, 5, 6}};
  std::uint64_t steps = 0, proposed = 0, plain_steps = 0, tested = 0, accepted = 0;
  std::mt19937_64 rng(4171);
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  for (const bool passthrough : {true, false}) {
    MtpFixture fx = build_mtp_fixture(passthrough, 16, false, 64, false, true);
    LSE_EXPECT(fx.ok);
    if (!fx.ok) return;
    for (const std::vector<std::uint32_t>& prompt : prompts) {
      GenerationLimits limits;
      limits.max_tokens = 23;
      Generator plain(*fx.lm, greedy_params());
      Session ps("plain", fx.lm->num_layers());
      auto want = plain.generate(ps, prompt, limits);
      LSE_EXPECT(want.ok());
      if (!want.ok()) return;
      for (int variant = 0; variant < 4; ++variant) {
        using P = DraftWidthPolicy;
        DraftWidthPolicy policy;
        for (std::uint64_t i = 0; i < P::kWarmupSteps; ++i) policy.observe_verify(P::kMaxRows, 1);
        const double base = 1e5 + 1e6 * unit(rng), per_row = 2e5 * unit(rng);
        for (std::uint32_t pass = 0; pass < P::kExploreSamples; ++pass)
          for (std::uint32_t rows = 1; rows <= P::kMaxRows; ++rows)
            policy.observe_verify(rows, static_cast<std::uint64_t>(base + per_row * (rows - 1)));
        const double per_draft = 5e5 * unit(rng);
        for (std::uint32_t d = 0; d <= P::kMaxProposals; ++d)
          policy.observe_draft(static_cast<std::uint64_t>(1e5 + per_draft * d), d);
        policy.observe_step(1, static_cast<std::uint64_t>(1e9 / (100.0 + 3000.0 * unit(rng))));
        for (int i = 0; i < 200; ++i)
          policy.observe_acceptance(unit(rng), static_cast<std::uint32_t>(i % 7), unit(rng) < 0.7);
        Generator spec(*fx.lm, top_one);
        spec.use_mtp(*fx.mtp, &policy);
        Session sp("adaptive", fx.lm->state_slots());
        limits.mtp_depth = 3;
        auto got = spec.generate(sp, prompt, limits);
        LSE_EXPECT(got.ok());
        if (!got.ok()) { LSE_EXPECT_OK(got.status()); return; }
        LSE_EXPECT(spec.stats().spec_adaptive);
        LSE_EXPECT(*got == *want);
        if (*got != *want)
          std::printf("       [%s] plain %s  adaptive %s\n", ids_to_string(prompt).c_str(),
                      ids_to_string(*want).c_str(), ids_to_string(*got).c_str());
        expect_mtp_resident(fx, sp, static_cast<std::int32_t>(sp.history().size()) - 1);
        const auto& st = spec.stats();
        steps += st.spec_steps;
        proposed += st.spec_proposed;
        plain_steps += st.spec_plain_steps;
        tested += st.spec_tested;
        accepted += st.spec_accepted;
      }
    }
  }
  std::printf("       passes=%llu rows/pass=%.2f plain=%llu accepted=%llu/%llu\n",
              static_cast<unsigned long long>(steps),
              steps ? static_cast<double>(proposed + steps) / static_cast<double>(steps) : 0.0,
              static_cast<unsigned long long>(plain_steps),
              static_cast<unsigned long long>(accepted), static_cast<unsigned long long>(tested));
  // Narrow, wide and plain passes all happened.
  LSE_EXPECT(plain_steps > 0);
  LSE_EXPECT(accepted > 0 && accepted < tested);
  LSE_EXPECT(proposed > steps);
}

LSE_TEST(generation_timing_excludes_prefill_token_and_early_stop) {
  MtpFixture fx = build_mtp_fixture();
  LSE_EXPECT(fx.ok);
  if (!fx.ok) return;
  const std::vector<std::uint32_t> prompt{2, 11, 33};
  for (bool speculative : {false, true}) {
    for (int count : {0, 1, 3}) {
      // CPU interpreter fixtures cannot replay speculative verification. The
      // prefill-only/first-token/cancel paths still exercise speculative timing.
      if (speculative && count > 1) continue;
      Generator gen(*fx.lm, greedy_params());
      if (speculative) gen.use_mtp(*fx.mtp);
      GenerationLimits limits;
      limits.max_tokens = count;
      auto got = gen.generate(prompt, limits);
      LSE_EXPECT(got.ok());
      if (!got.ok()) return;
      LSE_EXPECT_EQ(got->size(), static_cast<std::size_t>(count));
      LSE_EXPECT_EQ(gen.stats().generated_tokens, count);
      LSE_EXPECT_EQ(gen.stats().decoded_tokens(), count > 0 ? count - 1 : 0);
      if (count <= 1) {
        LSE_EXPECT_EQ(gen.stats().decode_ns, 0u);
        LSE_EXPECT_EQ(gen.stats().decode_tokens_per_second(), 0.0);
      } else {
        LSE_EXPECT(gen.stats().decode_ns > 0);
        LSE_EXPECT(gen.stats().decode_tokens_per_second() > 0);
      }
    }
    Generator cancelled(*fx.lm, greedy_params());
    if (speculative) cancelled.use_mtp(*fx.mtp);
    GenerationLimits limits;
    limits.max_tokens = 8;
    auto got = cancelled.generate(prompt, limits, [](std::uint32_t) { return false; });
    LSE_EXPECT(got.ok());
    LSE_EXPECT_EQ(cancelled.stats().generated_tokens, 1);
    LSE_EXPECT_EQ(cancelled.stats().decode_ns, 0u);
    LSE_EXPECT_EQ(cancelled.stats().decode_tokens_per_second(), 0.0);
  }
}

namespace {
void expect_mtp_resident(const MtpFixture& fixture, Session& session,
                         std::int32_t covered) {
  LSE_EXPECT_EQ(session.position(), covered);
  LSE_EXPECT_EQ(fixture.mtp->position(), covered);
  LSE_EXPECT(session.mtp_context_matches(*fixture.mtp));
  LSE_EXPECT(session.mtp_tail().valid());
  LSE_EXPECT(session.history().size() == static_cast<std::size_t>(covered) ||
             session.history().size() == static_cast<std::size_t>(covered + 1));
  for (const auto& state : session.states())
    LSE_EXPECT_EQ(state.position, covered);
}

void mtp_terminal_next_request(bool rejected, std::int32_t max_tokens,
                              std::size_t cancel_after, std::uint32_t stop) {
  auto fixture = build_mtp_fixture(rejected, 16, false, 256, true, true);
  LSE_EXPECT(fixture.ok); if (!fixture.ok) return;
  Session session("mtp-resident", fixture.lm->state_slots());
  GenerationLimits limits;
  limits.max_tokens = max_tokens;
  if (stop != std::numeric_limits<std::uint32_t>::max()) limits.stop_tokens = {stop};
  std::vector<std::uint32_t> delivered;
  {
    Generator first(*fixture.lm, greedy_params());
    first.use_mtp(*fixture.mtp);
    auto result = first.generate(session, {2, 3}, limits, [&](std::uint32_t token) {
      delivered.push_back(token);
      return cancel_after == 0 || delivered.size() < cancel_after;
    });
    LSE_EXPECT(result.ok()); if (!result.ok()) { LSE_EXPECT_OK(result.status()); return; }
    LSE_EXPECT(*result == delivered);
    if (rejected && delivered.size() >= 3)
      LSE_EXPECT(first.stats().spec_tested > first.stats().spec_accepted);
    if (!rejected && delivered.size() >= 5) LSE_EXPECT(first.stats().spec_accepted > 0);
  }
  std::vector<std::uint32_t> expected;
  for (std::uint32_t token = 4; expected.size() < static_cast<std::size_t>(max_tokens); ++token) {
    if (token == stop) break;
    expected.push_back(token);
    if (cancel_after != 0 && expected.size() == cancel_after) break;
  }
  LSE_EXPECT(delivered == expected);
  const auto covered = 2 + static_cast<std::int32_t>(delivered.size()) -
      (stop == 4 + delivered.size() ? 0 : delivered.empty() ? 0 : 1);
  expect_mtp_resident(fixture, session, covered);

  auto prompt = session.history();
  prompt.insert(prompt.end(), {9, 10});
  GenerationLimits next_limits;
  next_limits.max_tokens = 4;
  std::vector<std::uint32_t> warm;
  {
    Generator next(*fixture.lm, greedy_params());
    next.use_mtp(*fixture.mtp);
    auto result = next.generate(session, prompt, next_limits);
    LSE_EXPECT(result.ok()); if (!result.ok()) { LSE_EXPECT_OK(result.status()); return; }
    warm = *result;
    LSE_EXPECT_EQ(next.stats().prompt_tokens, static_cast<std::int32_t>(prompt.size()) - covered);
  }
  expect_mtp_resident(fixture, session, static_cast<std::int32_t>(prompt.size()) + 3);
  auto cold = build_mtp_fixture(rejected, 16, false, 256, true, true);
  LSE_EXPECT(cold.ok); if (!cold.ok) return;
  Session fresh("mtp-cold", cold.lm->state_slots());
  Generator next(*cold.lm, greedy_params());
  next.use_mtp(*cold.mtp);
  auto result = next.generate(fresh, prompt, next_limits);
  LSE_EXPECT(result.ok()); if (!result.ok()) { LSE_EXPECT_OK(result.status()); return; }
  LSE_EXPECT(*result == warm);
  LSE_EXPECT(fresh.history() == session.history());
}
}

LSE_TEST(mtp_generation_without_a_cap_stops_cleanly_at_a_full_context) {
  // The transitions fixture counts upward forever and never emits a stop
  // token; with kv_length 32 and no max_tokens only the context ends it.
  for (const bool rejected : {false, true}) {
    auto fixture = build_mtp_fixture(rejected, 16, false, 32, true, true);
    LSE_EXPECT(fixture.ok); if (!fixture.ok) return;
    Session session("mtp-full", fixture.lm->state_slots());
    Generator gen(*fixture.lm, greedy_params());
    gen.use_mtp(*fixture.mtp);
    auto out = gen.generate(session, {2, 3}, GenerationLimits{});
    LSE_EXPECT(out.ok()); if (!out.ok()) { LSE_EXPECT_OK(out.status()); return; }
    LSE_EXPECT_EQ(out->size(), 30u);
    LSE_EXPECT(gen.stats().stop_reason == StopReason::kContextFull);
    LSE_EXPECT_EQ(gen.stats().context_tokens, 32);
    LSE_EXPECT_EQ(session.history().size(), 32u);
    auto plain_fixture = build_mtp_fixture(rejected, 16, false, 32, true, true);
    LSE_EXPECT(plain_fixture.ok); if (!plain_fixture.ok) return;
    Generator plain(*plain_fixture.lm, greedy_params());
    auto reference = plain.generate({2, 3}, GenerationLimits{});
    LSE_EXPECT(reference.ok() && *reference == *out);
    LSE_EXPECT(plain.stats().stop_reason == StopReason::kContextFull);
    // A stop token still ends it first, and an explicit limit is honoured.
    GenerationLimits stop;
    stop.stop_tokens = {10};
    Generator stopped(*plain_fixture.lm, greedy_params());
    auto until = stopped.generate({2, 3}, stop);
    LSE_EXPECT(until.ok() && until->size() == 6u);
    LSE_EXPECT(stopped.stats().stop_reason == StopReason::kStopToken);
    GenerationLimits five;
    five.max_tokens = 5;
    Generator capped(*plain_fixture.lm, greedy_params());
    auto limited = capped.generate({2, 3}, five);
    LSE_EXPECT(limited.ok() && limited->size() == 5u);
    LSE_EXPECT(capped.stats().stop_reason == StopReason::kMaxTokens);
  }
}

LSE_TEST(mtp_next_request_reuses_full_acceptance_prefix) {
  mtp_terminal_next_request(false, 7, 0, std::numeric_limits<std::uint32_t>::max());
}
LSE_TEST(mtp_next_request_reuses_rejected_prefix) {
  mtp_terminal_next_request(true, 7, 0, std::numeric_limits<std::uint32_t>::max());
}
LSE_TEST(mtp_next_request_reuses_one_terminal_row_after_cancel) {
  mtp_terminal_next_request(false, 7, 2, std::numeric_limits<std::uint32_t>::max());
}
LSE_TEST(mtp_next_request_reuses_two_terminal_rows_after_cancel) {
  mtp_terminal_next_request(false, 7, 3, std::numeric_limits<std::uint32_t>::max());
}
LSE_TEST(mtp_next_request_reuses_first_token_cancellation) {
  mtp_terminal_next_request(false, 7, 1, std::numeric_limits<std::uint32_t>::max());
}
LSE_TEST(mtp_next_request_reuses_prefill_stop) {
  mtp_terminal_next_request(false, 7, 0, 4);
}
LSE_TEST(mtp_next_request_reuses_stop_inside_verifier) {
  mtp_terminal_next_request(false, 7, 0, 6);
}
LSE_TEST(mtp_next_request_reuses_stop_after_rejection) {
  mtp_terminal_next_request(true, 7, 0, 6);
}
LSE_TEST(mtp_next_request_reuses_max_one_token) {
  mtp_terminal_next_request(false, 1, 0, std::numeric_limits<std::uint32_t>::max());
}

LSE_TEST(mtp_unrelated_prompt_coldstarts_both_caches) {
  auto fixture = build_mtp_fixture(false, 16, false, 256, true, true);
  LSE_EXPECT(fixture.ok); if (!fixture.ok) return;
  Session session("unrelated-mtp", fixture.lm->state_slots());
  GenerationLimits limits;
  limits.max_tokens = 1;
  {
    Generator first(*fixture.lm, greedy_params());
    first.use_mtp(*fixture.mtp);
    auto result = first.generate(session, {2, 3}, limits);
    LSE_EXPECT(result.ok()); if (!result.ok()) return;
  }
  Generator next(*fixture.lm, greedy_params());
  next.use_mtp(*fixture.mtp);
  auto result = next.generate(session, {7, 8}, limits);
  LSE_EXPECT(result.ok()); if (!result.ok()) return;
  LSE_EXPECT(*result == std::vector<std::uint32_t>{9});
  LSE_EXPECT_EQ(next.stats().prompt_tokens, 2);
  expect_mtp_resident(fixture, session, 2);
}

LSE_TEST(mtp_target_tail_is_owned_and_module_changes_invalidate_reuse) {
  auto fixture = build_mtp_fixture();
  LSE_EXPECT(fixture.ok); if (!fixture.ok) return;
  Session session("owned-tail", fixture.lm->state_slots());
  auto source = graph::Array::full(Shape{1, 1, fixture.config.hidden_size}, DType::kF32, 3.0f);
  LSE_EXPECT_OK(source.materialize());
  const std::uint32_t token = 2;
  auto proposal = fixture.mtp->draft(source, std::span(&token, 1), 0);
  LSE_EXPECT(proposal.ok()); if (!proposal.ok()) return;
  session.advance(1);
  LSE_EXPECT_OK(session.retain_mtp_tail(source, 1, *fixture.mtp));
  LSE_EXPECT(session.mtp_context_matches(*fixture.mtp));
  for (std::size_t i = 0; i < source.shape().elem_count(); ++i)
    graph::interpreter::store_element(*source.node(), i, 9.0f);
  const auto held = array_to_host(session.mtp_tail());
  for (float value : held) LSE_EXPECT_EQ(value, 3.0f);
  const auto revision = fixture.mtp->revision();
  LSE_EXPECT(!fixture.mtp->truncate(2).ok());
  LSE_EXPECT(!fixture.mtp->truncate(-1).ok());
  LSE_EXPECT_EQ(fixture.mtp->revision(), revision);
  LSE_EXPECT(session.mtp_context_matches(*fixture.mtp));
  LSE_EXPECT_OK(fixture.mtp->truncate(1));
  LSE_EXPECT(!session.mtp_context_matches(*fixture.mtp));
  LSE_EXPECT_OK(session.restart());
  LSE_EXPECT(!session.mtp_tail().valid());
}

LSE_TEST(mtp_continued_prefill_carries_the_previous_target_hidden) {
  auto fixture = build_mtp_fixture(true, 16, false, 256, true, true);
  LSE_EXPECT(fixture.ok); if (!fixture.ok) return;
  Session session("shifted-tail", fixture.lm->state_slots());
  GenerationLimits limits;
  limits.max_tokens = 0;
  {
    Generator first(*fixture.lm, greedy_params());
    first.use_mtp(*fixture.mtp);
    LSE_EXPECT(first.generate(session, {2, 11, 13}, limits).ok());
  }
  expect_mtp_resident(fixture, session, 3);
  // The independently seeded module consumes the same shifted boundary.
  const std::uint32_t token = 19;
  auto expected = fixture.mtp->draft(session.mtp_tail(), std::span(&token, 1), 3);
  LSE_EXPECT(expected.ok()); if (!expected.ok()) return;
  LSE_EXPECT_EQ(*expected, 14u);
  // A different pass at the same cursor must not reuse this session's tail.
  LSE_EXPECT_OK(fixture.mtp->truncate(3));
  Generator changed(*fixture.lm, greedy_params());
  changed.use_mtp(*fixture.mtp);
  auto reset = changed.generate(session, {2, 11, 13, 19}, limits);
  LSE_EXPECT(reset.ok()); if (!reset.ok()) return;
  LSE_EXPECT_EQ(changed.stats().prompt_tokens, 4);
  Generator next(*fixture.lm, greedy_params());
  next.use_mtp(*fixture.mtp);
  auto output = next.generate(session, {2, 11, 13, 19, 23}, limits);
  LSE_EXPECT(output.ok()); if (!output.ok()) return;
  LSE_EXPECT_EQ(next.stats().prompt_tokens, 1);
  LSE_EXPECT_EQ(fixture.mtp->position(), 5);
  expect_mtp_resident(fixture, session, 5);
  const std::uint32_t next_token = 27;
  auto warm = fixture.mtp->draft_chain(session.mtp_tail(), std::span(&next_token, 1), 5, 3);
  LSE_EXPECT(warm.ok()); if (!warm.ok()) return;
  auto cold = build_mtp_fixture(true, 16, false, 256, true, true);
  LSE_EXPECT(cold.ok); if (!cold.ok) return;
  Session fresh("cold-shifted-tail", cold.lm->state_slots());
  Generator full(*cold.lm, greedy_params());
  full.use_mtp(*cold.mtp);
  auto prefill = full.generate(fresh, {2, 11, 13, 19, 23}, limits);
  LSE_EXPECT(prefill.ok()); if (!prefill.ok()) return;
  auto expected_chain = cold.mtp->draft_chain(fresh.mtp_tail(), std::span(&next_token, 1), 5, 3);
  LSE_EXPECT(expected_chain.ok()); if (!expected_chain.ok()) return;
  LSE_EXPECT(*warm == std::vector<std::uint32_t>({24, 24, 24}));
  LSE_EXPECT(*warm == *expected_chain);
}

LSE_TEST(mtp_continued_prefill_preserves_tail_across_kv_pool_growth) {
  if (std::getenv("LSE_KV_PREALLOC")) LSE_SKIP("requires growing KV storage");
  auto fixture = build_mtp_fixture(true, 16, false, 256, true, true);
  LSE_EXPECT(fixture.ok); if (!fixture.ok) return;
  Session session("mtp-tail-growth", fixture.lm->state_slots());
  GenerationLimits limits;
  limits.max_tokens = 0;
  std::vector<std::uint32_t> prompt(128, 2);
  Generator first(*fixture.lm, greedy_params());
  first.use_mtp(*fixture.mtp);
  auto primed = first.generate(session, prompt, limits);
  LSE_EXPECT(primed.ok()); if (!primed.ok()) return;
  expect_mtp_resident(fixture, session, 128);
  LSE_EXPECT(session.mtp_tail().node()->inputs.empty());
  const auto old_pool = session.states().front().paged.keys.node();
  prompt.push_back(11);
  Generator next(*fixture.lm, greedy_params());
  next.use_mtp(*fixture.mtp);
  auto continued = next.generate(session, prompt, limits);
  LSE_EXPECT(continued.ok()); if (!continued.ok()) return;
  LSE_EXPECT_EQ(next.stats().prompt_tokens, 1);
  LSE_EXPECT(session.states().front().paged.keys.node() != old_pool);
  expect_mtp_resident(fixture, session, 129);
  LSE_EXPECT(session.mtp_tail().node()->inputs.empty());
  const std::uint32_t token = 17;
  auto warm = fixture.mtp->draft_chain(session.mtp_tail(), std::span(&token, 1), 129, 3);
  LSE_EXPECT(warm.ok()); if (!warm.ok()) return;
  auto cold = build_mtp_fixture(true, 16, false, 256, true, true);
  LSE_EXPECT(cold.ok); if (!cold.ok) return;
  Session fresh("mtp-tail-growth-cold", cold.lm->state_slots());
  Generator full(*cold.lm, greedy_params());
  full.use_mtp(*cold.mtp);
  auto prefill = full.generate(fresh, prompt, limits);
  LSE_EXPECT(prefill.ok()); if (!prefill.ok()) return;
  auto expected = cold.mtp->draft_chain(fresh.mtp_tail(), std::span(&token, 1), 129, 3);
  LSE_EXPECT(expected.ok()); if (!expected.ok()) return;
  LSE_EXPECT(*warm == std::vector<std::uint32_t>({12, 12, 12}));
  LSE_EXPECT(*warm == *expected);
}

LSE_TEST(resident_restart_cursors_distinguish_single_token_prefill_from_decode) {
  for (bool recurrent_only : {false, true}) {
    MtpFixture fx = build_mtp_fixture(false, 16, recurrent_only);
    LSE_EXPECT(fx.ok);
    if (!fx.ok) return;
    Session session("resident-cursors", fx.lm->state_slots());
    auto first = fx.lm->hidden(ids_array({2}), &session.states(), nullptr);
    LSE_EXPECT(first.ok());
    if (!first.ok()) return;
    for (const auto& state : session.states()) LSE_EXPECT_EQ(state.position, 1);
    auto next = fx.lm->hidden(ids_array({11, 33}), &session.states(), nullptr);
    LSE_EXPECT(next.ok());
    if (!next.ok()) return;
    for (const auto& state : session.states()) LSE_EXPECT_EQ(state.position, 3);
    fx.lm->rewind(session.states(), 1);
    for (const auto& state : session.states()) LSE_EXPECT_EQ(state.position, 1);
    LSE_EXPECT_OK(session.restart());
    for (const auto& state : session.states()) LSE_EXPECT_EQ(state.position, 0);
    auto restarted = fx.lm->hidden(ids_array({2}), &session.states(), nullptr);
    LSE_EXPECT(restarted.ok());
    if (!restarted.ok()) return;
    for (const auto& state : session.states()) LSE_EXPECT_EQ(state.position, 1);
  }
}

LSE_TEST(recurrent_carries_own_mutable_buffers_and_restart_severs_old_graphs) {
  MtpFixture fx = build_mtp_fixture(false, 16, true);
  LSE_EXPECT(fx.ok);
  if (!fx.ok) return;
  Session session("resident-ownership", fx.lm->state_slots());
  auto first = fx.lm->hidden(ids_array({2}), &session.states(), nullptr);
  LSE_EXPECT(first.ok());
  if (!first.ok()) return;
  const auto first_carries = fx.lm->retained_program().carries();
  LSE_EXPECT(first_carries.size() > 1);
  auto* sched = graph::default_scheduler();
  LSE_EXPECT(sched != nullptr);
  if (!sched) return;
  auto& devices = sched->devices();
  for (std::size_t member = 0; member < devices.size(); ++member) {
    const auto status = devices.device(member).synchronize();
    LSE_EXPECT_OK(status);
    if (!status.ok()) return;
  }
  auto same_start = [](const backend::DeviceBuffer& a, const backend::DeviceBuffer& b) {
    if (a.residency != b.residency || a.offset != b.offset) return false;
    // A device-local allocation need not have a CPU pointer. Pool views may
    // share an opaque handle, but their offsets still distinguish the windows.
    if (a.handle || b.handle) return a.handle == b.handle;
    return a.ptr == b.ptr;
  };
  std::vector<graph::NodePtr> old_outputs;
  std::vector<backend::DeviceBuffer> inputs;
  for (const auto& c : first_carries) {
    LSE_EXPECT(c.in != nullptr && c.out != nullptr);
    if (!c.in || !c.out) return;
    LSE_EXPECT(c.in->kind == graph::OpKind::kBuffer);
    LSE_EXPECT(c.in->inputs.empty());
    LSE_EXPECT(c.in->buffer.valid());
    LSE_EXPECT(c.out->buffer.valid());
    if (!c.in->buffer.valid() || !c.out->buffer.valid()) return;
    for (const auto& prior : inputs) {
      LSE_EXPECT(!same_start(prior, c.in->buffer));
    }
    LSE_EXPECT(!same_start(c.in->buffer, c.out->buffer));
    inputs.push_back(c.in->buffer);
    old_outputs.push_back(c.out);
  }
  // A fold makes each consumed input writable. Exercise the actual backing
  // allocation: interpreter writes alone would only change a host mirror on
  // HIP, and Array::to_host after folding could re-evaluate the old graph.
  LSE_EXPECT(!first_carries.empty());
  if (first_carries.empty()) return;
  graph::Array ordinary_zero = graph::Array::zeros(
      first_carries[0].in->shape, DType::kF32);
  const auto zero_status = ordinary_zero.eval();
  LSE_EXPECT_OK(zero_status);
  if (!zero_status.ok()) return;
  auto copy_values = [&](graph::Node& node, std::vector<float>& values,
                         bool upload) {
    const auto member = devices.member_of(node.buffer.residency);
    LSE_EXPECT(member < devices.size());
    if (member >= devices.size()) return false;
    auto& backend = devices.device(member);
    const auto status = upload
        ? backend.copy_h2d(values.data(), node.buffer,
                           values.size() * sizeof(float), 0)
        : backend.copy_d2h(node.buffer, values.data(),
                           values.size() * sizeof(float), 0);
    LSE_EXPECT_OK(status);
    // Retire even a partially failed transfer before host storage goes away.
    const auto retired = backend.synchronize();
    LSE_EXPECT_OK(retired);
    return status.ok() && retired.ok();
  };
  auto expect_values = [&](graph::Node& node, float expected) {
    std::vector<float> values(node.shape.elem_count(), -1.0f);
    if (!copy_values(node, values, false)) return false;
    for (float value : values) LSE_EXPECT_EQ(value, expected);
    return true;
  };
  graph::Program folded;
  folded.set_carries(first_carries);
  folded.fold_carries();
  for (std::size_t changed = 0; changed < first_carries.size(); ++changed) {
    auto& output = *first_carries[changed].out;
    std::vector<float> written(output.shape.elem_count(), 13.0f);
    if (!copy_values(output, written, true)) return;
    for (std::size_t i = 0; i < first_carries.size(); ++i) {
      if (!expect_values(*first_carries[i].out, i == changed ? 13.0f : 0.0f)) {
        return;
      }
    }
    if (!expect_values(*ordinary_zero.node(), 0.0f)) return;
    std::fill(written.begin(), written.end(), 0.0f);
    if (!copy_values(output, written, true)) return;
  }
  LSE_EXPECT_OK(session.restart());
  auto restarted = fx.lm->hidden(ids_array({2}), &session.states(), nullptr);
  LSE_EXPECT(restarted.ok());
  if (!restarted.ok()) return;
  for (const auto& c : fx.lm->retained_program().carries()) {
    LSE_EXPECT(c.in != nullptr);
    if (!c.in) continue;
    LSE_EXPECT(c.in->kind == graph::OpKind::kBuffer);
    LSE_EXPECT(c.in->inputs.empty());
    for (const auto& old : old_outputs) LSE_EXPECT(c.in.get() != old.get());
  }
}

LSE_TEST(kv_growth_releases_obsolete_recurrent_graphs) {
  if (std::getenv("LSE_KV_PREALLOC")) LSE_SKIP("requires growing KV storage");
  MtpFixture fx = build_mtp_fixture(false, 16, false, 512);
  LSE_EXPECT(fx.ok);
  if (!fx.ok) return;
  Session session("growth-ownership", fx.lm->state_slots());
  std::vector<std::uint32_t> prompt(128, 2);
  std::weak_ptr<graph::Node> previous;
  for (int pass = 0; pass < 3; ++pass) {
    auto hidden = fx.lm->hidden(ids_array(prompt), &session.states(), nullptr);
    LSE_EXPECT_OK(hidden.status());
    if (!hidden.ok()) return;
    // Each pass exceeds the previous pool rung. The old state computation
    // must no longer be owned through retained programs or the new state.
    if (pass > 0) LSE_EXPECT(previous.expired());
    previous = session.states().front().gdn_state.node();
    for (const auto& state : session.states())
      LSE_EXPECT_EQ(state.position, (pass + 1) * 128);
  }
  auto decode = fx.lm->hidden(ids_array({3}), &session.states(), nullptr);
  LSE_EXPECT_OK(decode.status());
  LSE_EXPECT_OK(session.restart());
  auto restarted = fx.lm->hidden(ids_array({2}), &session.states(), nullptr);
  LSE_EXPECT_OK(restarted.status());
}
