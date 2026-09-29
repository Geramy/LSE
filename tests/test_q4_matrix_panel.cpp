#include "harness.hpp"
#include "lse/backends/hrx/arch_database.hpp"
#include "lse/backends/hrx/hipc/hip_emitter.hpp"
#include "lse/backends/hrx/hipc/hip_types.hpp"
#include "lse/backends/hrx/loomc/loom_emitter.hpp"
#include "lse/backends/hrx/loomc/loom_types.hpp"
#include "lse/dispatch/quant.hpp"
#include "lse/graph/kernel_primitive.hpp"
#include "lse/graph/ops.hpp"
#include "lse/graph/program.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

using namespace lse;
using namespace lse::graph;
namespace {
constexpr auto kProducer = "quant_activation.q4_matrix_panel.v1";
constexpr auto kConsumer = "quant_linear.q4_matrix_panel.v1";
constexpr std::size_t kRows = 8, kColumns = 5120, kWidth = 17408, kPeriod = 16;

Array leaf(Shape shape, DType type) {
  auto node = std::make_shared<Node>();
  node->shape = shape;
  node->dtype = type;
  node->materialized = true;
  return Array(node);
}
template <class T>
Array filled(Shape shape, DType type, const std::vector<T> &values) {
  auto *scheduler = default_scheduler();
  if (!scheduler)
    return {};
  auto storage = scheduler->backend().allocate(values.size() * sizeof(T),
                                               backend::MemoryClass::kDevice);
  LSE_EXPECT(storage.ok());
  if (!storage.ok())
    return {};
  auto buffer = storage.release();
  LSE_EXPECT_OK(scheduler->backend().copy(buffer, values.data(),
                                          values.size() * sizeof(T)));
  return Array::from_buffer(std::move(buffer), shape, type);
}
template <class T> std::vector<T> read(Array value) {
  std::vector<T> data(value.shape().elem_count());
  LSE_EXPECT_OK(value.to_host(data.data(), data.size() * sizeof(T)));
  return data;
}
Array contraction(const Array &x, std::int64_t n, std::int64_t k) {
  return quant_linear(x, leaf({n, k / 8}, DType::kU32),
                      leaf({n, k / 64}, DType::kBF16),
                      leaf({n, k / 64}, DType::kBF16), 4, 64);
}
float activation(std::size_t row, std::size_t column) {
  if (row == 0 || column / 64 % 11 == 2 || column / 8 % 13 == 2)
    return 0;
  const auto a =
      std::sin(static_cast<double>((row + 1) * 19 + column * 7) * .071);
  const auto b =
      std::cos(static_cast<double>((row + 3) * 11 + column * 3) * .037);
  return static_cast<float>((a * .6137 + b * .1923) *
                            (.25 + static_cast<double>(column % 7) * .1131));
}
std::vector<float> activations(std::size_t k) {
  std::vector<float> x(kRows * k);
  for (std::size_t row = 0; row < kRows; ++row)
    for (std::size_t column = 0; column < k; ++column)
      x[row * k + column] = activation(row, column);
  return x;
}
struct EncodedGroup {
  std::array<std::int32_t, 64> codes{};
  float step = 0, sum = 0;
};
EncodedGroup encode_group(const float *x) {
  EncodedGroup result;
  std::array<float, 4> maximum{}, sum{};
  for (std::size_t slice = 0; slice < 4; ++slice) {
    maximum[slice] = std::abs(x[slice * 16]);
    sum[slice] = x[slice * 16];
    for (std::size_t t = 1; t < 16; ++t) {
      maximum[slice] = std::max(maximum[slice], std::abs(x[slice * 16 + t]));
      sum[slice] += x[slice * 16 + t];
    }
  }
  float amax = maximum[0];
  result.sum = sum[0];
  for (std::size_t slice = 1; slice < 4; ++slice) {
    amax = std::max(amax, maximum[slice]);
    result.sum += sum[slice];
  }
  result.step = amax * (1.0f / 127.0f);
  const float inverse = 127.0f / std::max(amax, 1e-30f);
  for (std::size_t t = 0; t < 64; ++t)
    result.codes[t] = static_cast<std::int32_t>(std::nearbyint(x[t] * inverse));
  return result;
}
std::vector<std::uint32_t> codec(const std::vector<float> &x, std::size_t k) {
  std::vector<std::uint32_t> words(k / 64 * 288);
  for (std::size_t group = 0; group < k / 64; ++group)
    for (std::size_t row = 0; row < kRows; ++row) {
      const auto encoded = encode_group(x.data() + row * k + group * 64);
      for (std::size_t word = 0; word < 16; ++word) {
        std::uint32_t packed = 0;
        for (std::size_t byte = 0; byte < 4; ++byte)
          packed |=
              (static_cast<std::uint32_t>(encoded.codes[word * 4 + byte]) &
               255u)
              << (byte * 8);
        const auto at = group * 288 + (word / 4) * 64 + (word % 4 / 2) * 32 +
                        row * 2 + word % 2;
        words[at] = packed;
      }
      words[group * 288 + 256 + row * 2] =
          std::bit_cast<std::uint32_t>(encoded.step);
      words[group * 288 + 257 + row * 2] =
          std::bit_cast<std::uint32_t>(encoded.sum);
    }
  return words;
}
struct Fixture {
  std::array<Shape, 5> inputs{Shape{1, 8, 17408}, Shape{5120, 2176},
                              Shape{5120, 272}, Shape{5120, 272},
                              Shape{272, 288}};
  std::array<DType, 5> dtypes{DType::kF32, DType::kU32, DType::kBF16,
                              DType::kBF16, DType::kU32};
  backend::AmdDeviceInfo amd;
  backend::DeviceInfo device;
  DialectSourceTable sources = backend::loom_sources();
  KernelShapes shapes;
  Fixture() {
    device.arch = "gfx1201";
    backend::apply_arch_defaults(device, amd);
    device.extension_id = backend::AmdDeviceInfo::kExtensionId;
    device.extension = &amd;
    shapes.inputs = inputs;
    shapes.input_dtypes = dtypes;
    shapes.output = Shape{1, 8, 5120};
    shapes.iattrs = {4, 64, 0, 0};
    shapes.device = &device;
    shapes.intrinsics = &sources;
    shapes.types = backend::loom_types();
    shapes.store = [](std::string_view index, std::string_view value) {
      return "out[" + std::string(index) + "] = " + std::string(value) + ";";
    };
  }
  KernelShapes original() const {
    auto result = shapes;
    result.inputs = shapes.inputs.first(4);
    result.input_dtypes = shapes.input_dtypes.first(4);
    return result;
  }
};
FusionGroup group(const Array &output) {
  FusionGroup result;
  result.nodes = {output.node()};
  result.outputs = result.nodes;
  result.inputs = output.node()->inputs;
  result.anchor = output.node()->kind;
  result.anchor_class = output.node()->fclass;
  return result;
}
} // namespace

LSE_TEST(q4_matrix_panel_selects_only_measured_down_and_shares_same_input) {
  auto x = leaf({1, 8, 17408}, DType::kF32);
  auto a = contraction(x, 5120, 17408), b = contraction(x, 5120, 17408);
  LSE_EXPECT(a.node()->prim && a.node()->prim->name() == kConsumer);
  LSE_EXPECT_EQ(a.node()->inputs.size(), 5u);
  LSE_EXPECT(a.node()->inputs[4] == b.node()->inputs[4]);
  LSE_EXPECT(a.node()->inputs[4]->shape == Shape{272, 288});
  LSE_EXPECT(a.node()->inputs[4]->prim->name() == kProducer);
  auto cloned = std::make_shared<Node>(*x.node());
  auto c = contraction(Array(cloned), 5120, 17408);
  LSE_EXPECT(c.node()->inputs[4] != a.node()->inputs[4]);
  const auto weak = x.node()->quant_activation_panel;
  a = {};
  b = {};
  LSE_EXPECT(weak.expired());
  for (const auto m : {1, 4, 6, 7, 9, 16, 512}) {
    auto other = contraction(leaf({1, m, 17408}, DType::kF32), 5120, 17408);
    LSE_EXPECT(other.node()->prim->name() != kConsumer);
  }
  auto up = contraction(leaf({1, 8, 5120}, DType::kF32), 17408, 5120);
  LSE_EXPECT(up.node()->prim->name() == "quant_linear.q4_global_panel.v1");
}
LSE_TEST(q4_matrix_panel_dispatch_validates_geometry_capabilities_and_staging) {
  Fixture f;
  LSE_EXPECT(dispatch::q4_matrix_panel_shape(f.original()));
  LSE_EXPECT(dispatch::q4_matrix_panel_row(f.original()) != nullptr);
  for (int fault = 0; fault < 10; ++fault) {
    Fixture bad;
    if (fault == 0)
      bad.inputs[0] = Shape{1, 7, 17408};
    if (fault == 1)
      bad.inputs[1] = Shape{5121, 2176};
    if (fault == 2)
      bad.inputs[2] = Shape{5120, 271};
    if (fault == 3)
      bad.dtypes[2] = DType::kF32;
    if (fault == 4)
      bad.shapes.output = Shape{8, 5120};
    if (fault == 5)
      bad.device.arch = "gfx1200";
    if (fault == 6)
      bad.device.wavefront_size = 64;
    if (fault == 7)
      bad.device.max_threads_per_workgroup = 128;
    if (fault == 8)
      bad.amd.matrix_core = backend::MatrixCore::kNone;
    if (fault == 9)
      bad.shapes.staged.name = "external";
    LSE_EXPECT(dispatch::q4_matrix_panel_row(bad.original()) == nullptr);
  }
  f.inputs[0] = Shape{INT64_MAX, 8, 17408};
  LSE_EXPECT(!dispatch::q4_matrix_panel_shape(f.original()));
  Fixture staged;
  staged.shapes.staged_quant.codes = "external";
  LSE_EXPECT(dispatch::q4_matrix_panel_row(staged.original()) == nullptr);
  Fixture unknown;
  unknown.shapes.intrinsics = nullptr;
  LSE_EXPECT(dispatch::q4_matrix_panel_row(unknown.original()) == nullptr);
}
LSE_TEST(q4_matrix_panel_typed_codec_pads_sixteen_rows_and_refreshes_replay) {
  auto *scheduler = default_scheduler();
  LSE_EXPECT(scheduler != nullptr);
  if (!scheduler)
    return;
  scheduler->set_mode(Scheduler::Mode::kHostOnly);
  const auto *primitive = find_primitive(kProducer);
  LSE_EXPECT(primitive && primitive->has_typed_host_impl());
  if (!primitive)
    return;
  for (const auto shape : {Shape{7, 64}, Shape{16, 64}, Shape{8, 65},
                           Shape{8, INT64_MAX}, Shape{INT64_MAX, 8, 64}}) {
    const std::array inputs{shape};
    LSE_EXPECT(!primitive->infer_shape(inputs).ok());
  }
  auto data = activations(128);
  auto x = filled({1, 8, 128}, DType::kF32, data);
  auto result = custom(kProducer, {x});
  LSE_EXPECT(result.ok());
  if (!result.ok())
    return;
  auto output = result.release();
  const NodePtr roots[]{output.node()};
  Program program;
  LSE_EXPECT_OK(scheduler->eval(roots, false, &program));
  LSE_EXPECT(read<std::uint32_t>(output) == codec(data, 128));
  for (auto &value : data)
    value = value * -.617f + .031f;
  LSE_EXPECT_OK(scheduler->backend().copy(x.node()->buffer, data.data(),
                                          data.size() * 4));
  x.node()->host_dirty = false;
  x.node()->device_dirty = true;
  program.reset_compute();
  LSE_EXPECT_OK(scheduler->eval(roots, false, &program));
  LSE_EXPECT(read<std::uint32_t>(output) == codec(data, 128));
}
LSE_TEST(
    q4_matrix_panel_emission_cache_and_declined_device_keep_original_body) {
  auto output = contraction(leaf({1, 8, 17408}, DType::kF32), 5120, 17408);
  const auto g = group(output);
  Fixture f;
  const auto *kernel =
      dynamic_cast<const KernelPrimitiveBase *>(output.node()->prim);
  const auto *base =
      dynamic_cast<const KernelPrimitiveBase *>(find_primitive("quant_linear"));
  LSE_EXPECT(kernel && base);
  if (!kernel || !base)
    return;
  backend::LoomEmitter loom;
  backend::HipEmitter hip;
  for (auto *emitter : {static_cast<IKernelEmitter *>(&loom),
                        static_cast<IKernelEmitter *>(&hip)}) {
    f.sources = emitter->sources();
    f.shapes.types = emitter->dialect() == Dialect::kLoom
                         ? backend::loom_types()
                         : backend::hip_types();
    const auto admitted = emitter->emit(g, f.device);
    LSE_EXPECT(admitted.ok());
    if (!admitted.ok())
      continue;
    LSE_EXPECT_EQ(admitted->lds_bytes, 0u);
    LSE_EXPECT_EQ(admitted->dims.workgroup_size[0], 256u);
    LSE_EXPECT_EQ(admitted->dims.workgroup_count[0], 40u);
    const auto admitted_key = emitter->cache_key(g, f.device);
    f.amd.matrix_core = backend::MatrixCore::kNone;
    const auto declined_key = emitter->cache_key(g, f.device);
    LSE_EXPECT(admitted_key != declined_key);
    auto original = f.original();
    const auto *legacy = base->specialize(original);
    LSE_EXPECT(kernel->emit_kernel(f.shapes) == legacy->emit_kernel(original));
    const auto actual = kernel->plan(f.shapes),
               expected = legacy->plan(original);
    for (std::size_t axis = 0; axis < 3; ++axis) {
      LSE_EXPECT_EQ(actual.workgroup_size[axis], expected.workgroup_size[axis]);
      LSE_EXPECT_EQ(actual.workgroup_count[axis],
                    expected.workgroup_count[axis]);
    }
    LSE_EXPECT_EQ(actual.lds_bytes, expected.lds_bytes);
    const auto declined = emitter->emit(g, f.device);
    LSE_EXPECT(declined.ok());
    f.amd.matrix_core = backend::MatrixCore::kWMMA;
    LSE_EXPECT_EQ(emitter->cache_key(g, f.device), admitted_key);
    f.inputs[4] = Shape{272, 287};
    LSE_EXPECT(kernel->emit_kernel(f.shapes).empty());
    f.inputs[4] = Shape{272, 288};
  }
}
LSE_TEST(q4_matrix_panel_host_fallback_reads_original_input_not_panel_bits) {
  auto *scheduler = default_scheduler();
  LSE_EXPECT(scheduler != nullptr);
  if (!scheduler)
    return;
  scheduler->set_mode(Scheduler::Mode::kHostOnly);
  const auto data = activations(64);
  auto x = filled({8, 64}, DType::kF32, data);
  auto base = quant_linear(
      x,
      filled({3, 8}, DType::kU32, std::vector<std::uint32_t>(24, 0x13579bdf)),
      filled({3, 1}, DType::kBF16,
             std::vector<bfloat16_t>(3, bfloat16_t(.0237f))),
      filled({3, 1}, DType::kBF16,
             std::vector<bfloat16_t>(3, bfloat16_t(-.131f))),
      4, 64);
  auto modified = std::make_shared<Node>(*base.node());
  modified->prim = find_primitive(kConsumer);
  modified->inputs.push_back(
      filled({1, 288}, DType::kU32,
             std::vector<std::uint32_t>(288, 0xffffffffu))
          .node());
  const auto expected = read<float>(base),
             actual = read<float>(Array(modified));
  LSE_EXPECT(std::memcmp(expected.data(), actual.data(), actual.size() * 4) ==
             0);
}

namespace {
std::int32_t weight_code(std::size_t col, std::size_t group, std::size_t at) {
  return static_cast<std::int32_t>(
      (col % kPeriod * 43 + group % 17 * 67 + at * 17) % 16);
}
float weight_scale(std::size_t col, std::size_t group) {
  return bfloat16_t(.002173f + static_cast<float>(col % kPeriod) * .000149f +
                    static_cast<float>(group % 7) * .000037f)
      .to_float();
}
float weight_bias(std::size_t col, std::size_t group) {
  return bfloat16_t(
             static_cast<float>(static_cast<int>(col % kPeriod % 3) - 1) *
                 .000973f +
             static_cast<float>(static_cast<int>(group % 5) - 2) * .000137f)
      .to_float();
}
std::vector<float> reference(const std::vector<float> &x) {
  std::vector<float> result(kRows * kPeriod);
  for (std::size_t row = 0; row < kRows; ++row)
    for (std::size_t group = 0; group < kWidth / 64; ++group) {
      const auto encoded = encode_group(x.data() + row * kWidth + group * 64);
      for (std::size_t col = 0; col < kPeriod; ++col) {
        std::int32_t dot = 0;
        for (std::size_t t = 0; t < 64; ++t)
          dot += encoded.codes[t] * weight_code(col, group, t);
        auto &out = result[row * kPeriod + col];
        out = std::fma(weight_scale(col, group) * encoded.step,
                       static_cast<float>(dot), out) +
              weight_bias(col, group) * encoded.sum;
      }
    }
  return result;
}
int gpu_down() {
  auto *scheduler = default_scheduler();
  if (!scheduler)
    return 1;
  scheduler->set_mode(Scheduler::Mode::kDeviceFirst);
  scheduler->set_dialect(Dialect::kLoom);
  auto data = activations(kWidth);
  std::vector<std::uint32_t> weights(kColumns * kWidth / 8);
  std::vector<bfloat16_t> scales(kColumns * kWidth / 64), biases(scales.size());
  for (std::size_t col = 0; col < kColumns; ++col) {
    for (std::size_t at = 0; at < kWidth; at += 8) {
      std::uint32_t word = 0;
      for (std::size_t nibble = 0; nibble < 8; ++nibble)
        word |= static_cast<std::uint32_t>(
                    weight_code(col, at / 64, at % 64 + nibble))
                << (nibble * 4);
      weights[col * (kWidth / 8) + at / 8] = word;
    }
    for (std::size_t group = 0; group < kWidth / 64; ++group) {
      scales[col * (kWidth / 64) + group] =
          bfloat16_t(weight_scale(col, group));
      biases[col * (kWidth / 64) + group] = bfloat16_t(weight_bias(col, group));
    }
  }
  auto x = leaf({1, 8, 17408}, DType::kF32);
  auto w = leaf({5120, 2176}, DType::kU32);
  auto s = leaf({5120, 272}, DType::kBF16);
  auto b = leaf({5120, 272}, DType::kBF16);
  auto output = quant_linear(x, w, s, b, 4, 64);
  LSE_EXPECT(output.node()->prim->name() == kConsumer);
  const auto groups = Partitioner::partition(std::array{output.node()});
  LSE_EXPECT_EQ(groups.size(), 2u);
  LSE_EXPECT(groups.front().outputs.front()->prim->name() == kProducer);
  struct Allocation {
    backend::DeviceBuffer buffer;
    std::vector<std::byte> initial;
    std::size_t payload;
    bool readonly;
  };
  std::vector<Allocation> allocations;
  const auto guard = [&](const NodePtr &node, const void *input,
                         bool readonly) {
    const auto bytes = dtype_storage_bytes(node->dtype, node->element_count());
    // Extra eight output rows prove that the padded fragment never writes them.
    const auto extra = !readonly && node->dtype == DType::kF32 ? bytes : 0;
    Allocation a{{},
                 std::vector<std::byte>(bytes + extra + 128, std::byte{0xa5}),
                 bytes,
                 readonly};
    if (input)
      std::memcpy(a.initial.data() + 64, input, bytes);
    else if (node->dtype == DType::kF32) {
      const float nan = std::numeric_limits<float>::quiet_NaN();
      for (std::size_t i = 0; i < node->element_count(); ++i)
        std::memcpy(a.initial.data() + 64 + i * 4, &nan, 4);
    }
    auto storage = scheduler->backend().allocate(a.initial.size(),
                                                 backend::MemoryClass::kDevice);
    LSE_EXPECT(storage.ok());
    if (!storage.ok())
      return;
    a.buffer = storage.release();
    LSE_EXPECT_OK(scheduler->backend().copy(a.buffer, a.initial.data(),
                                            a.initial.size()));
    node->buffer = a.buffer;
    node->buffer.offset += 64;
    node->buffer.size_bytes = bytes;
    node->host_mirror.clear();
    node->host_dirty = false;
    node->device_dirty = input != nullptr;
    allocations.push_back(std::move(a));
  };
  guard(x.node(), data.data(), true);
  guard(w.node(), weights.data(), true);
  guard(s.node(), scales.data(), true);
  guard(b.node(), biases.data(), true);
  guard(output.node()->inputs[4], nullptr, false);
  guard(output.node(), nullptr, false);
  const auto expected = reference(data);
  const auto check = [&](const std::vector<float> &actual,
                         const std::vector<float> *residual) {
    double maximum = 0;
    for (std::size_t row = 0; row < kRows; ++row)
      for (std::size_t col = 0; col < kColumns; ++col) {
        const auto at = row * kColumns + col;
        const auto want = expected[row * kPeriod + col % kPeriod] +
                          (residual ? (*residual)[at] : 0.0f);
        LSE_EXPECT(std::isfinite(actual[at]));
        const auto difference =
            std::abs(static_cast<double>(actual[at]) - static_cast<double>(want));
        maximum = std::max(maximum, difference);
        LSE_EXPECT(difference <= 2e-5 + 5e-6 * std::abs(static_cast<double>(want)));
      }
    std::printf("M8 matrix panel maximum component error %.9g\n", maximum);
  };
  const NodePtr roots[]{output.node(), output.node()->inputs[4]};
  Program program;
  scheduler->reset_accumulated_trace();
  LSE_EXPECT_OK(scheduler->eval(roots, false, &program));
  LSE_EXPECT_OK(scheduler->drain());
  auto trace = scheduler->last_trace();
  LSE_EXPECT_EQ(trace.device_groups, 2u);
  LSE_EXPECT_EQ(trace.host_groups, 0u);
  LSE_EXPECT_EQ(trace.host_fallbacks, 0u);
  const auto raw = read<float>(output);
  check(raw, nullptr);
  LSE_EXPECT(read<std::uint32_t>(Array(output.node()->inputs[4])) ==
             codec(data, kWidth));
  std::vector<float> residual(kRows * kColumns);
  for (std::size_t i = 0; i < residual.size(); ++i)
    residual[i] = std::cos(static_cast<float>(i) * .091f) * .113f;
  auto r = leaf({1, 8, 5120}, DType::kF32);
  guard(r.node(), residual.data(), true);
  program.reset_compute();
  auto epilogue = add(output, r);
  guard(epilogue.node(), nullptr, false);
  const NodePtr epilogue_roots[]{epilogue.node(), output.node()->inputs[4]};
  Program fused_program;
  scheduler->reset_accumulated_trace();
  LSE_EXPECT_OK(scheduler->eval(epilogue_roots, false, &fused_program));
  LSE_EXPECT_OK(scheduler->drain());
  trace = scheduler->last_trace();
  LSE_EXPECT_EQ(trace.device_groups, 2u);
  LSE_EXPECT_EQ(trace.host_groups, 0u);
  LSE_EXPECT_EQ(trace.host_fallbacks, 0u);
  const auto actual = read<float>(epilogue);
  check(actual, &residual);
  for (std::size_t i = 0; i < actual.size(); ++i)
    LSE_EXPECT(std::bit_cast<std::uint32_t>(actual[i]) ==
               std::bit_cast<std::uint32_t>(raw[i] + residual[i]));
  std::fill(data.begin(), data.end(), 0.0f);
  LSE_EXPECT_OK(scheduler->backend().copy(x.node()->buffer, data.data(),
                                          data.size() * 4));
  x.node()->host_dirty = false;
  x.node()->device_dirty = true;
  fused_program.reset_compute();
  scheduler->reset_accumulated_trace();
  LSE_EXPECT_OK(scheduler->eval(epilogue_roots, false, &fused_program));
  LSE_EXPECT_OK(scheduler->drain());
  trace = scheduler->last_trace();
  LSE_EXPECT_EQ(trace.device_groups, 2u);
  LSE_EXPECT_EQ(trace.host_groups, 0u);
  LSE_EXPECT_EQ(trace.host_fallbacks, 0u);
  LSE_EXPECT(read<float>(epilogue) == residual);
  LSE_EXPECT(read<std::uint32_t>(Array(output.node()->inputs[4])) ==
             std::vector<std::uint32_t>(272 * 288));
  for (std::size_t i = 0; i < allocations.size(); ++i) {
    const auto &allocation = allocations[i];
    std::vector<std::byte> bytes(allocation.initial.size());
    LSE_EXPECT_OK(scheduler->backend().copy(bytes.data(), allocation.buffer,
                                            bytes.size()));
    LSE_EXPECT(std::equal(bytes.begin(), bytes.begin() + 64,
                          allocation.initial.begin()));
    LSE_EXPECT(std::equal(
        bytes.begin() + static_cast<std::ptrdiff_t>(64 + allocation.payload),
        bytes.end(),
        allocation.initial.begin() +
            static_cast<std::ptrdiff_t>(64 + allocation.payload)));
    if (allocation.readonly && i != 0)
      LSE_EXPECT(bytes == allocation.initial);
    if (i == 0)
      LSE_EXPECT(std::all_of(
          bytes.begin() + 64,
          bytes.begin() + static_cast<std::ptrdiff_t>(64 + allocation.payload),
          [](std::byte byte) { return byte == std::byte{0}; }));
  }
  std::printf(
      "M8 matrix panel native raw/residual/replay: complete ordered group64 "
      "oracle, exact codec, padded16 zeros and guards; zero host/fallback\n");
  return lse::test::Registry::get().failures ? 1 : 0;
}
} // namespace
int main(int argc, char **argv) {
  if (argc == 2 && std::string_view(argv[1]) == "--gpu-down")
    return gpu_down();
  return lse::test::run_all();
}
