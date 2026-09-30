#include "harness.hpp"
#include "lse/backends/cpu/cpu_backend.hpp"
#include "lse/backends/hrx/hipc/hip_emitter.hpp"
#include "lse/backends/hrx/loomc/loom_emitter.hpp"
#include "lse/dispatch/attention.hpp"
#include "lse/graph/interpreter.hpp"
#include "lse/graph/kernel_env.hpp"
#include "lse/graph/ops.hpp"
#include "lse/kv/cache_codec.hpp"
#include "lse/kv/memory.hpp"
#include "lse/model/config.hpp"
#include "lse/ops/attention.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <utility>

using namespace lse;
using namespace lse::graph;
namespace {
constexpr std::array formats{kv::CacheDType::kF32, kv::CacheDType::kF16,
                             kv::CacheDType::kBF16, kv::CacheDType::kFP8,
                             kv::CacheDType::kBF8};
Array leaf(Shape shape, DType dtype = DType::kF32) {
  auto n = std::make_shared<Node>();
  n->shape = std::move(shape);
  n->dtype = dtype;
  n->materialized = true;
  return Array(n);
}
Array upload(backend::IBackend &cpu, Shape shape,
             const std::vector<float> &values) {
  auto buffer = cpu.allocate(values.size() * sizeof(float),
                             backend::MemoryClass::kDevice);
  LSE_EXPECT(buffer.ok());
  if (!buffer.ok())
    return {};
  auto owned = buffer.release();
  LSE_EXPECT_OK(cpu.copy(owned, values.data(), values.size() * sizeof(float)));
  return Array::from_buffer(std::move(owned), std::move(shape), DType::kF32);
}
Array pool(backend::IBackend &cpu, kv::CacheDType format) {
  Shape shape{2, 1, 16, kv::storage_width(format, 8)};
  const auto bytes =
      dtype_storage_bytes(kv::storage_dtype(format), shape.elem_count());
  auto allocated = cpu.allocate(bytes, backend::MemoryClass::kDevice);
  LSE_EXPECT(allocated.ok());
  if (!allocated.ok())
    return {};
  auto buffer = allocated.release();
  std::vector<std::byte> zero(bytes);
  LSE_EXPECT_OK(cpu.copy(buffer, zero.data(), bytes));
  return Array::from_buffer(std::move(buffer), shape,
                            kv::storage_dtype(format));
}
} // namespace


LSE_TEST(fragmented_host_writes_refresh_other_aliases) {
  backend::BackendAdapter<backend::CpuBackend> cpu;
  LSE_EXPECT_OK(cpu.init(0));
  auto manager = kv::MemoryManager::create();
  auto storage = std::make_shared<kv::FragmentStorage>(manager, cpu,
      backend::kDefaultStream);
  const Shape shape{2, 1, 16, 8};
  const auto bytes = shape.elem_count() * sizeof(float);
  LSE_EXPECT_OK(storage->reserve(bytes));
  LSE_EXPECT_OK(storage->grow(bytes));
  auto binding = storage->binding();
  LSE_EXPECT_OK(binding.status());
  if (!binding.ok()) return;
  auto first = Array::from_buffer(*binding, shape, DType::kF32);
  auto alias = Array::from_buffer(*binding, shape, DType::kF32);
  first.node()->kv_fragments = storage;
  alias.node()->kv_fragments = storage;
  LSE_EXPECT_OK(interpreter::sync_from_device(*alias.node(), cpu));
  auto table = upload(cpu, {1, 2}, {1, 0});
  auto src = upload(cpu, {1, 1, 1, 8}, std::vector<float>(8, 3.0f));
  auto write = kv_page_write(first, src, upload(cpu, {5}, {0, 1, 1, 0, 1}),
                             table, 16, kv::CacheDType::kF32);
  LSE_EXPECT_OK(interpreter::evaluate(write.node(), cpu));
  LSE_EXPECT_OK(interpreter::sync_from_device(*alias.node(), cpu));
  LSE_EXPECT_EQ(interpreter::load_element(*alias.node(), 16 * 8), 3.0f);
  auto next = kv_page_write(alias,
      upload(cpu, {1, 1, 1, 8}, std::vector<float>(8, 7.0f)),
      upload(cpu, {5}, {0, 2, 1, 1, 2}), table, 16, kv::CacheDType::kF32);
  LSE_EXPECT_OK(interpreter::evaluate(next.node(), cpu));
  LSE_EXPECT_OK(interpreter::sync_from_device(*first.node(), cpu));
  LSE_EXPECT_EQ(interpreter::load_element(*first.node(), 16 * 8), 3.0f);
  LSE_EXPECT_EQ(interpreter::load_element(*first.node(), 17 * 8), 7.0f);
}

LSE_TEST(kv_formats_parse_and_keep_explicit_overrides) {
  model::Config cfg;
  LSE_EXPECT(cfg.kv_cache_dtype == kv::CacheDType::kF16);
  for (auto f : formats) {
    cfg.kv_cache_dtype = f;
    auto roundtrip = model::Config::from_json_string(cfg.to_json());
    LSE_EXPECT(roundtrip.ok());
    if (roundtrip.ok())
      LSE_EXPECT(roundtrip->kv_cache_dtype == f);
    auto parsed = kv::cache_dtype_from_string(kv::to_string(f));
    LSE_EXPECT(parsed.ok());
    if (parsed.ok())
      LSE_EXPECT(parsed.release() == f);
    LSE_EXPECT_EQ(kv::logical_width(f, kv::storage_width(f, 256)), 256);
  }
  LSE_EXPECT(!kv::cache_dtype_from_string("int8").ok());
  LSE_EXPECT_EQ(kv::storage_width(kv::CacheDType::kFP8, 257), 0);
  LSE_EXPECT_EQ(dtype_storage_bytes(DType::kU32, 65), 260u);
}

LSE_TEST(kv_storage_defaults_follow_declared_model_dtype) {
  const std::array cases{
      std::pair{R"({})", kv::CacheDType::kF16},
      std::pair{R"({"dtype":"bfloat16"})", kv::CacheDType::kBF16},
      std::pair{R"({"dtype":"bf16"})", kv::CacheDType::kBF16},
      std::pair{R"({"torch_dtype":"bfloat16"})", kv::CacheDType::kBF16},
      std::pair{R"({"torch_dtype":"torch.bfloat16"})", kv::CacheDType::kBF16},
      std::pair{R"({"dtype":"float16"})", kv::CacheDType::kF16},
      std::pair{R"({"dtype":"float32"})", kv::CacheDType::kF16},
      std::pair{R"({"dtype":"float16","torch_dtype":"bfloat16"})", kv::CacheDType::kF16},
      std::pair{R"({"dtype":null,"torch_dtype":"bfloat16"})", kv::CacheDType::kBF16},
      std::pair{R"({"torch_dtype":null})", kv::CacheDType::kF16}};
  for (const auto& [json, expected] : cases) {
    const auto config = model::Config::from_json_string(json);
    LSE_EXPECT(config.ok());
    if (config.ok()) LSE_EXPECT(config->kv_cache_dtype == expected);
  }
  for (const auto format : formats) {
    const auto json = std::string(R"({"dtype":"bfloat16","kv_cache_dtype":")") +
        std::string(kv::to_string(format)) + R"("})";
    const auto config = model::Config::from_json_string(json);
    LSE_EXPECT(config.ok());
    if (config.ok()) LSE_EXPECT(config->kv_cache_dtype == format);
  }
  // Low-level attention references still require an explicit FP32 default.
  const ops::GatedAttentionSpec reference;
  LSE_EXPECT(reference.kv_cache_dtype == kv::CacheDType::kF32);
}

LSE_TEST(kv_storage_metadata_rejects_invalid_explicit_types) {
  for (const auto json : {R"({"dtype":42})", R"({"torch_dtype":false})",
                         R"({"kv_cache_dtype":null})", R"({"kv_cache_dtype":7})",
                         R"({"kv_cache_dtype":"int8"})"}) {
    const auto config = model::Config::from_json_string(json);
    LSE_EXPECT(!config.ok());
    if (!config.ok())
      LSE_EXPECT(config.status().code() == StatusCode::kInvalidArgument);
  }
}

LSE_TEST(scaled_fp8_zero_extremes_and_independent_head_scales) {
  for (auto format : {kv::CacheDType::kFP8, kv::CacheDType::kBF8}) {
    for (float magnitude : {0.0f, 0x1p-120f, .71325f, 1e30f}) {
      std::array<float, 8> values{};
      for (std::size_t i = 0; i < 8; ++i)
        values[i] = (float(i) - 3.0f) * magnitude;
      std::array<std::uint32_t, 3> packed{};
      if (format == kv::CacheDType::kFP8)
        kv::pack_cache_vector<math::MatrixElem::kFp8>(values, packed);
      else
        kv::pack_cache_vector<math::MatrixElem::kBf8>(values, packed);
      const float scale = std::bit_cast<float>(packed[2]);
      LSE_EXPECT(std::isfinite(scale) && scale > 0);
      for (std::size_t i = 0; i < 8; ++i) {
        const float result =
            kv::unpack_cache_element(format, packed.data(), 8, i);
        LSE_EXPECT(std::isfinite(result));
        const auto byte =
            static_cast<std::uint8_t>(packed[i / 4] >> (8 * (i % 4)));
        const auto expected =
            format == kv::CacheDType::kFP8
                ? math::fp8_bits<math::MatrixElem::kFp8>(values[i] / scale)
                : math::fp8_bits<math::MatrixElem::kBf8>(values[i] / scale);
        LSE_EXPECT_EQ(byte, expected);
      }
    }
  }
}

LSE_TEST(paged_storage_cpu_write_and_attention_read_use_logical_pitch) {
  backend::BackendAdapter<backend::CpuBackend> cpu;
  LSE_EXPECT_OK(cpu.init(0));
  const std::vector<float> key{.13f, -.21f, .73f,  .43f, .91f, -.18f,
                               .02f, .66f,  -.23f, .07f, .48f, -.32f,
                               .14f, .55f,  -.8f,  .12f};
  const std::vector<float> value{.24f,  .72f, -.13f, .34f, .05f, -.92f,
                                 .18f,  .82f, -.14f, .06f, .37f, .15f,
                                 -.32f, .42f, .11f,  .54f};
  for (auto format : formats) {
    auto ksrc = upload(cpu, {1, 1, 2, 8}, key),
         vsrc = upload(cpu, {1, 1, 2, 8}, value);
    auto meta = upload(cpu, {5}, {0, 2, 1, 0, 2}),
         table = upload(cpu, {1, 1}, {1});
    auto kp = pool(cpu, format), vp = pool(cpu, format);
    auto k = kv_page_write(kp, ksrc, meta, table, 16, format),
         v = kv_page_write(vp, vsrc, meta, table, 16, format);
    LSE_EXPECT_OK(interpreter::evaluate(k.node(), cpu));
    LSE_EXPECT_OK(interpreter::evaluate(v.node(), cpu));
    auto q = upload(cpu, {1, 1, 2, 8}, key);
    auto output = sdpa_paged(q, k, v, .35355339f, MaskKind::kCausal, 0, meta,
                             table, 16, nullptr, format);
    LSE_EXPECT(output.shape() == Shape({1, 1, 2, 8}));
    LSE_EXPECT_OK(interpreter::evaluate(output.node(), cpu));
    std::array<std::array<float, 8>, 2> decoded_k{}, decoded_v{};
    for (std::size_t row = 0; row < 2; ++row)
      for (std::size_t d = 0; d < 8; ++d) {
        if (kv::packed_cache(format)) {
          const auto pitch =
              static_cast<std::size_t>(kv::storage_width(format, 8));
          const auto *kk = static_cast<const std::uint32_t *>(
              interpreter::host_bytes(*k.node()));
          const auto *vv = static_cast<const std::uint32_t *>(
              interpreter::host_bytes(*v.node()));
          decoded_k[row][d] =
              kv::unpack_cache_element(format, kk + (16 + row) * pitch, 8, d);
          decoded_v[row][d] =
              kv::unpack_cache_element(format, vv + (16 + row) * pitch, 8, d);
        } else {
          decoded_k[row][d] =
              interpreter::load_element(*k.node(), (16 + row) * 8 + d);
          decoded_v[row][d] =
              interpreter::load_element(*v.node(), (16 + row) * 8 + d);
        }
      }
    for (std::size_t row = 0; row < 2; ++row) {
      double scores[2]{}, denom = 0;
      double maximum = -std::numeric_limits<double>::infinity();
      for (std::size_t j = 0; j <= row; ++j) {
        for (std::size_t d = 0; d < 8; ++d)
          scores[j] += double(key[row * 8 + d]) * double(decoded_k[j][d]);
        scores[j] = double(float(scores[j]) * .35355339f);
        maximum = std::max(maximum, scores[j]);
      }
      for (std::size_t j = 0; j <= row; ++j) {
        scores[j] = double(float(std::exp(float(scores[j] - maximum))));
        denom += scores[j];
      }
      for (std::size_t d = 0; d < 8; ++d) {
        double expected = 0;
        for (std::size_t j = 0; j <= row; ++j)
          expected += scores[j] * double(decoded_v[j][d]);
        LSE_EXPECT_NEAR(interpreter::load_element(*output.node(), row * 8 + d),
                        expected / denom, 2e-6);
      }
    }
  }
}

LSE_TEST(
    all_storage_formats_keep_native_attention_routes_and_separate_cache_keys) {
  backend::DeviceInfo gpu;
  gpu.arch = "gfx1201";
  gpu.wavefront_size = 32;
  gpu.max_threads_per_workgroup = 256;
  gpu.lds_bytes_per_workgroup = 65536;
  backend::LoomEmitter loom;
  backend::HipEmitter hip;
  for (int rows : {1, 4, 12}) {
    std::uint64_t previous = 0;
    for (auto format : formats) {
      auto q = leaf({1, 24, rows, 256});
      auto k = leaf({65, 4, 16, kv::storage_width(format, 256)},
                    kv::storage_dtype(format));
      auto v = leaf(k.shape(), k.dtype()), meta = leaf({5}),
           table = leaf({1, 64});
      auto out = sdpa_paged(q, k, v, .0625f, MaskKind::kCausal, 0, meta, table,
                            16, &gpu, format);
      const NodePtr roots[]{out.node()};
      auto groups = Partitioner::partition(roots, &gpu);
      LSE_EXPECT_EQ(groups.size(), rows == 12 ? 1u : 2u);
      for (const auto &group : groups) {
        auto le = loom.emit(group, gpu), he = hip.emit(group, gpu);
        LSE_EXPECT(le.ok());
        LSE_EXPECT(he.ok());
      }
      const auto key = loom.cache_key(groups[0], gpu);
      if (previous)
        LSE_EXPECT(key != previous);
      previous = key;
      auto src = leaf({1, 4, rows, 256});
      auto written = kv_page_write(k, src, meta, table, 16, format);
      const NodePtr write_roots[]{written.node()};
      auto write_groups = Partitioner::partition(write_roots, &gpu);
      LSE_EXPECT_EQ(write_groups.size(), 1u);
      auto emitted = loom.emit(write_groups[0], gpu);
      LSE_EXPECT(emitted.ok());
      if (!emitted.ok())
        std::fprintf(stderr, "WRITE %s Loom: %s\n",
                     std::string(kv::to_string(format)).c_str(),
                     std::string(emitted.status().message()).c_str());
      auto hip_emitted = hip.emit(write_groups[0], gpu);
      LSE_EXPECT(hip_emitted.ok());
      if (!hip_emitted.ok())
        std::fprintf(stderr, "WRITE %s HIP: %s\n",
                     std::string(kv::to_string(format)).c_str(),
                     std::string(hip_emitted.status().message()).c_str());
    }
  }
}

namespace {
template <bool WrongBinding>
struct InplaceWordKernel final
    : KernelPrimitive<InplaceWordKernel<WrongBinding>> {
  static constexpr std::string_view kName =
      WrongBinding ? "fixture.kv_words.bad" : "fixture.kv_words";
  static constexpr std::string_view kEntry = "fixture_kv_words";
  static constexpr std::string_view kSource = {};
  std::size_t arity() const noexcept override { return 1; }
  bool owns_indexing() const noexcept override { return true; }
  bool supports_epilogue() const noexcept override { return false; }
  int inplace_input() const noexcept override { return 0; }
  Result<Shape> infer_shape(std::span<const Shape> in) const override {
    return in[0];
  }
  DType infer_dtype(std::span<const DType> in) const override { return in[0]; }
  std::string emit_kernel(const KernelShapes &s) const override {
    if (!s.intrinsics)
      return {};
    kir::KernelBody body(s.types, *s.intrinsics);
    kir::Buffer<kir::u32> input(&body, &s.types, "in0"),
        output(&body, &s.types, "out");
    ir::env::Emit e{&body};
    const auto i = e.thread_id();
    if (auto live = e.when(i < 4u)) {
      if constexpr (WrongBinding)
        input[i] = e.u32(7);
      else
        output[i] = e.u32(7);
    }
    return body.str();
  }
  static ThreadPlan plan_impl(const KernelShapes &) { return {}; }
};
} // namespace
LSE_TEST(
    inplace_typed_terminal_stores_remain_restricted_to_the_output_binding) {
  backend::DeviceInfo gpu;
  gpu.arch = "gfx1201";
  gpu.wavefront_size = 32;
  gpu.max_threads_per_workgroup = 256;
  gpu.lds_bytes_per_workgroup = 65536;
  InplaceWordKernel<false> good;
  InplaceWordKernel<true> bad;
  for (const Primitive *primitive : {static_cast<const Primitive *>(&good),
                                     static_cast<const Primitive *>(&bad)}) {
    auto input = leaf({4}, DType::kU32);
    auto output = leaf({4}, DType::kU32);
    output.node()->set_kind(OpKind::kCustom);
    output.node()->materialized = false;
    output.node()->inputs = {input.node()};
    output.node()->prim = primitive;
    FusionGroup group;
    group.nodes = group.outputs = {output.node()};
    group.inputs = {input.node()};
    group.anchor = OpKind::kCustom;
    group.anchor_class = output.node()->fclass;
    LSE_EXPECT(backend::LoomEmitter{}.emit(group, gpu).ok() ==
               (primitive == &good));
    LSE_EXPECT(backend::HipEmitter{}.emit(group, gpu).ok() ==
               (primitive == &good));
  }
}
LSE_TEST(packed_attention_rejects_mismatched_storage_tags_and_invalid_pitch) {
  backend::DeviceInfo gpu;
  gpu.arch = "gfx1201";
  gpu.wavefront_size = 32;
  gpu.max_threads_per_workgroup = 256;
  gpu.lds_bytes_per_workgroup = 65536;
  for (int variant = 0; variant < 5; ++variant) {
    auto q = leaf({1, 24, 4, 256});
    auto k = leaf({65, 4, 16, 65}, DType::kU32), v = leaf(k.shape(), k.dtype());
    auto meta = leaf({5}), table = leaf({1, 64});
    if (variant == 1)
      k = v = leaf({65, 4, 16, 64}, DType::kU32);
    if (variant == 2)
      v = leaf(v.shape(), DType::kF32);
    if (variant == 3)
      meta = leaf({2});
    if (variant == 4)
      k = v = leaf({UINT32_MAX / (4 * 16 * 65), 4, 16, 65}, DType::kU32);
    auto out = sdpa_paged(q, k, v, .0625f, MaskKind::kCausal, 0, meta, table,
                          16, nullptr, kv::CacheDType::kFP8);
    if (variant == 0)
      out.node()->attrs[1] = 0.0f;
    const NodePtr roots[]{out.node()};
    auto groups = Partitioner::partition(roots, &gpu);
    LSE_EXPECT(!backend::LoomEmitter{}.emit(groups.back(), gpu).ok());
    LSE_EXPECT(!backend::HipEmitter{}.emit(groups.back(), gpu).ok());
  }
}
LSE_TEST(packed_repeated_writes_preserve_padding_and_overwrite_the_scale_word) {
  backend::BackendAdapter<backend::CpuBackend> cpu;
  LSE_EXPECT_OK(cpu.init(0));
  for (auto format : formats) {
    auto dst = pool(cpu, format), table = upload(cpu, {1, 1}, {1});
    auto input = upload(cpu, {1, 1, 1, 8}, {1, -2, 3, -4, 5, -6, 7, -8});
    auto empty = kv_page_write(dst, input, upload(cpu, {5}, {0, 0, 0, 0, 0}),
                               table, 16, format);
    LSE_EXPECT_OK(interpreter::evaluate(empty.node(), cpu));
    const auto bytes =
        dtype_storage_bytes(dst.dtype(), dst.shape().elem_count());
    std::vector<std::byte> zero(bytes), first(bytes), second(bytes);
    LSE_EXPECT(std::memcmp(interpreter::host_bytes(*empty.node()), zero.data(),
                           bytes) == 0);
    auto written = kv_page_write(dst, input, upload(cpu, {5}, {0, 1, 1, 0, 1}),
                                 table, 16, format);
    LSE_EXPECT_OK(interpreter::evaluate(written.node(), cpu));
    std::memcpy(first.data(), interpreter::host_bytes(*written.node()), bytes);
    auto next = upload(cpu, {1, 1, 1, 8},
                       {.01f, -.02f, .03f, -.04f, .05f, -.06f, .07f, -.08f});
    auto overwritten = kv_page_write(
        written, next, upload(cpu, {5}, {0, 1, 1, 0, 1}), table, 16, format);
    LSE_EXPECT_OK(interpreter::evaluate(overwritten.node(), cpu));
    std::memcpy(second.data(), interpreter::host_bytes(*overwritten.node()),
                bytes);
    const auto vector_bytes = dtype_storage_bytes(
        dst.dtype(), static_cast<std::size_t>(dst.shape().dim(3)));
    const auto begin = 16 * vector_bytes;
    LSE_EXPECT(std::memcmp(first.data(), second.data(), begin) == 0);
    LSE_EXPECT(std::memcmp(first.data() + begin + vector_bytes,
                           second.data() + begin + vector_bytes,
                           bytes - begin - vector_bytes) == 0);
    LSE_EXPECT(std::memcmp(first.data() + begin, second.data() + begin,
                           vector_bytes) != 0);
    if (kv::packed_cache(format)) {
      std::uint32_t old_scale, new_scale;
      std::memcpy(&old_scale, first.data() + begin + vector_bytes - 4, 4);
      std::memcpy(&new_scale, second.data() + begin + vector_bytes - 4, 4);
      LSE_EXPECT(old_scale != new_scale);
    }
  }
}

LSE_TEST(pool_growth_copies_physical_storage_and_refuses_live_format_changes) {
  auto *scheduler = default_scheduler();
  LSE_EXPECT(scheduler != nullptr);
  if (!scheduler)
    return;
  auto &backend = scheduler->backend();
  scheduler->set_mode(Scheduler::Mode::kHostOnly);
  const auto evaluate = [&](const Array &output) {
    const NodePtr roots[]{output.node()};
    LSE_EXPECT_OK(scheduler->eval(roots, true));
  };
  std::vector<float> identity(64, 0.0f);
  for (std::size_t d = 0; d < 8; ++d)
    identity[d * 8 + d] = 1.0f;
  ops::GatedAttentionWeights weights;
  weights.q_proj = weights.k_proj = weights.v_proj = weights.o_proj =
      upload(backend, {8, 8}, identity);
  weights.g_proj = upload(backend, {8, 8}, std::vector<float>(64));
  weights.q_norm = weights.k_norm = upload(backend, {8}, std::vector<float>(8));
  auto rope = ops::build_rope(8, 256, 10000.0f);
  LSE_EXPECT(rope.ok());
  if (!rope.ok())
    return;
  for (auto format : formats) {
    ops::GatedAttentionSpec spec;
    spec.q_heads = spec.kv_heads = 1;
    spec.head_dim = 8;
    spec.kv_length = 2048;
    spec.kv_cache_dtype = format;
    ops::PagedKvLayer layer;
    ops::AttentionCache cache;
    cache.paged = &layer;
    cache.capacity = 2048;
    std::vector<float> inputs(127 * 8);
    for (std::size_t i = 0; i < inputs.size(); ++i)
      inputs[i] = std::sin(float(i) * .13f) * .4f;
    cache.meta = upload(backend, {5}, {0, 127, 1, 0, 127});
    auto first = ops::gated_attention(upload(backend, {1, 127, 8}, inputs),
                                      weights, spec, *rope, 0, &cache);
    LSE_EXPECT(first.ok());
    if (!first.ok())
      return;
    evaluate(*first);
    LSE_EXPECT(layer.keys.dtype() == kv::storage_dtype(format));
    LSE_EXPECT_EQ(layer.keys.shape().dim(3), kv::storage_width(format, 8));
    const auto before_shape = layer.keys.shape();
    const auto before_bytes =
        dtype_storage_bytes(layer.keys.dtype(), before_shape.elem_count());
    std::vector<std::byte> before(before_bytes);
    LSE_EXPECT_OK(interpreter::read_raw(*layer.keys.node(), before.data(),
                                        before.size()));
    const auto blocks = std::vector<kv::BlockId>(
        layer.tables[0].blocks().begin(), layer.tables[0].blocks().end());
    cache.used = 127;
    cache.meta = upload(backend, {5}, {127, 129, 1, 127, 129});
    auto second = ops::gated_attention(
        upload(backend, {1, 2, 8}, std::vector<float>(16, .37f)), weights, spec,
        *rope, 127, &cache);
    LSE_EXPECT(second.ok());
    if (!second.ok())
      return;
    evaluate(*second);
    LSE_EXPECT_EQ(layer.keys.shape().dim(0), 16);
    LSE_EXPECT(layer.storage == format);
    const auto after_bytes = dtype_storage_bytes(
        layer.keys.dtype(), layer.keys.shape().elem_count());
    std::vector<std::byte> after(after_bytes);
    LSE_EXPECT_OK(
        interpreter::read_raw(*layer.keys.node(), after.data(), after.size()));
    const auto vector_bytes = dtype_storage_bytes(
        layer.keys.dtype(),
        static_cast<std::size_t>(layer.keys.shape().dim(3)));
    for (std::size_t token = 0; token < 127; ++token) {
      const auto at =
          (static_cast<std::size_t>(blocks[token / 16]) * 16 + token % 16) *
          vector_bytes;
      LSE_EXPECT(std::memcmp(before.data() + at, after.data() + at,
                             vector_bytes) == 0);
    }
    spec.kv_cache_dtype = format == kv::CacheDType::kF32 ? kv::CacheDType::kBF16
                                                         : kv::CacheDType::kF32;
    LSE_EXPECT(!ops::gated_attention(
                    upload(backend, {1, 1, 8}, std::vector<float>(8, .21f)),
                    weights, spec, *rope, 129, &cache)
                    .ok());
  }
}
LSE_TEST_MAIN()
