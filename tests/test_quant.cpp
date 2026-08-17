// Quantization scheme tests: layout ABI, round-trip fidelity, and the
// error ordering Q8 < Q6 < Q4 that the whole scheme family depends on.
#include "lse/quant/traits.hpp"

#include <numeric>
#include <random>
#include <vector>

#include "harness.hpp"
#include "lse/quant/group_affine.hpp"

using namespace lse;
using namespace lse::quant;

namespace {

// A weight-like distribution: roughly normal, which is what the block-scaled
// symmetric schemes are tuned for.
std::vector<float> make_weights(std::size_t n, unsigned seed = 42) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> dist(0.0f, 0.1f);
  std::vector<float> out(n);
  for (auto& v : out) v = dist(rng);
  return out;
}

template <typename Scheme>
double rmse_for(const std::vector<float>& src) {
  std::vector<typename Scheme::Block> blocks(src.size() / kBlockElems);
  std::vector<float> scratch(src.size());
  return Scheme::round_trip_rmse(src.data(), src.size(), scratch.data(),
                                 blocks.data());
}

}  // namespace

LSE_TEST(block_layouts_are_abi_stable) {
  // These sizes are baked into model files; a change here silently corrupts
  // every previously quantized checkpoint.
  LSE_EXPECT_EQ(sizeof(BlockQ8), 34u);
  LSE_EXPECT_EQ(sizeof(BlockQ6), 26u);
  LSE_EXPECT_EQ(sizeof(BlockQ4), 18u);

  LSE_EXPECT_NEAR(Q8::bits_per_weight(), 8.5, 1e-9);
  LSE_EXPECT_NEAR(Q6::bits_per_weight(), 6.5, 1e-9);
  LSE_EXPECT_NEAR(Q4::bits_per_weight(), 4.5, 1e-9);
}

LSE_TEST(storage_bytes_matches_block_count) {
  LSE_EXPECT_EQ(Q8::storage_bytes(1024), (1024u / 32u) * 34u);
  LSE_EXPECT_EQ(Q6::storage_bytes(1024), (1024u / 32u) * 26u);
  LSE_EXPECT_EQ(Q4::storage_bytes(1024), (1024u / 32u) * 18u);

  // The runtime table must agree with the compile-time traits.
  LSE_EXPECT_EQ(dtype_storage_bytes(DType::kQ8, 1024), Q8::storage_bytes(1024));
  LSE_EXPECT_EQ(dtype_storage_bytes(DType::kQ6, 1024), Q6::storage_bytes(1024));
  LSE_EXPECT_EQ(dtype_storage_bytes(DType::kQ4, 1024), Q4::storage_bytes(1024));

  // Non-multiples of the block size are rejected rather than silently rounded.
  LSE_EXPECT_EQ(dtype_storage_bytes(DType::kQ4, 33), 0u);
}

LSE_TEST(round_trip_error_decreases_with_bit_width) {
  const auto w = make_weights(4096);
  const double e8 = rmse_for<Q8>(w);
  const double e6 = rmse_for<Q6>(w);
  const double e4 = rmse_for<Q4>(w);

  LSE_EXPECT(e8 < e6);
  LSE_EXPECT(e6 < e4);

  // Sanity bounds. For a block-scaled symmetric quantizer the error is
  // ~ (absmax / maxq) / sqrt(12); with sigma=0.1 and 32-element blocks the
  // block absmax lands around 0.25, giving roughly these magnitudes.
  LSE_EXPECT(e8 < 1e-3);
  LSE_EXPECT(e6 < 5e-3);
  LSE_EXPECT(e4 < 2e-2);
}

LSE_TEST(exact_representation_of_scale_endpoints) {
  // A block whose values are exactly the quantization grid must round-trip
  // with no error at all — this catches off-by-one packing and bias bugs.
  std::vector<float> block(kBlockElems);
  for (std::size_t i = 0; i < kBlockElems; ++i) {
    const int q = static_cast<int>(i % 15) - 7;  // the full Q4 grid [-7, 7]
    block[i] = static_cast<float>(q) * 0.25f;    // scale exactly representable
  }

  std::vector<BlockQ4> packed(1);
  std::vector<float> out(kBlockElems);
  Q4::quantize_row(block.data(), packed.data(), kBlockElems);
  Q4::dequantize_row(packed.data(), out.data(), kBlockElems);

  for (std::size_t i = 0; i < kBlockElems; ++i) {
    LSE_EXPECT_NEAR(out[i], block[i], 1e-6);
  }
}

LSE_TEST(all_zero_block_stays_zero) {
  // Guards the scale==0 division path.
  std::vector<float> zeros(kBlockElems, 0.0f);
  std::vector<BlockQ8> packed(1);
  std::vector<float> out(kBlockElems, 1.0f);
  Q8::quantize_row(zeros.data(), packed.data(), kBlockElems);
  Q8::dequantize_row(packed.data(), out.data(), kBlockElems);
  for (float v : out) LSE_EXPECT_NEAR(v, 0.0f, 0.0);
}

LSE_TEST(dispatch_scheme_selects_the_right_traits) {
  auto bpw = [](DType dt) {
    return dispatch_scheme(dt, []<typename S>() { return S::bits_per_weight(); });
  };
  LSE_EXPECT_NEAR(bpw(DType::kQ8), 8.5, 1e-9);
  LSE_EXPECT_NEAR(bpw(DType::kQ6), 6.5, 1e-9);
  LSE_EXPECT_NEAR(bpw(DType::kQ4), 4.5, 1e-9);
}

// ---------------------------------------------------------------------------
// Group-affine (MLX): a sibling of the block schemes, not a subclass of one.
// ---------------------------------------------------------------------------

namespace {

GroupAffine spec_of(int bits, int group_size) {
  auto s = GroupAffine::make(bits, group_size);
  return s.ok() ? *s : GroupAffine{};
}

}  // namespace

LSE_TEST(group_affine_chunk_geometry_covers_every_legal_width) {
  // A chunk is the shortest run of whole 32-bit lanes holding a whole number
  // of codes. Only the power-of-two widths collapse to "codes never straddle".
  struct Case { int bits, values, words; };
  const Case cases[] = {{2, 16, 1}, {3, 32, 3}, {4, 8, 1},
                        {5, 32, 5}, {6, 16, 3}, {8, 4, 1}};
  for (const Case& c : cases) {
    const GroupAffine q = spec_of(c.bits, 64);
    LSE_EXPECT_EQ(q.values_per_chunk(), c.values);
    LSE_EXPECT_EQ(q.words_per_chunk(), c.words);
    LSE_EXPECT_EQ(q.values_per_chunk() * c.bits, q.words_per_chunk() * 32);
    // Every legal group size is a whole number of chunks, which is what lets
    // one scale/bias pair cover a chunk.
    for (int g : kGroupAffineGroupSizes) {
      LSE_EXPECT_EQ(g % c.values, 0);
    }
    // A 1024-wide row: the plane sizes MLX writes.
    LSE_EXPECT_EQ(q.packed_words(1024), std::size_t(1024 * c.bits / 32));
    LSE_EXPECT_EQ(q.group_count(1024), std::size_t(16));
    LSE_EXPECT_OK(q.check_row(1024));
  }
  // Not a legal width, and not silently rounded to one.
  LSE_EXPECT(!GroupAffine::make(7, 64).ok());
  LSE_EXPECT(!GroupAffine::make(4, 48).ok());
}

LSE_TEST(group_affine_bit_stream_round_trips_at_every_width) {
  // The plane is a dense little-endian bit stream. Writing every code and
  // reading it back catches an off-by-one in the straddle path, which only
  // fires at 3, 5 and 6 bits.
  for (int bits : kGroupAffineBits) {
    const GroupAffine q = spec_of(bits, 64);
    const std::size_t k = 128;
    std::vector<std::uint32_t> row(q.packed_words(k), 0u);
    for (std::size_t c = 0; c < k; ++c) {
      q.set_code(row.data(), c, static_cast<std::uint32_t>(c) & q.max_code());
    }
    for (std::size_t c = 0; c < k; ++c) {
      LSE_EXPECT_EQ(q.code_at(row.data(), c),
                    static_cast<std::uint32_t>(c) & q.max_code());
    }
  }
}

LSE_TEST(group_affine_dequant_matches_hand_arithmetic) {
  // A cell read straight out of the Qwen3.5-0.8B-4bit checkpoint, worked by
  // hand:
  //
  //   column 133 of a 4-bit row, group_size 64
  //     lane        = 133 / 8               = 16
  //     bit offset  = (133 % 8) * 4         = 20
  //     group       = 133 / 64              = 2
  //     raw lane    = 0x76a8cbbb
  //     code        = (0x76a8cbbb >> 20) & 0xF
  //                 = 0x76a & 0xF           = 10
  //     scale       = 1.609375 * 2^-8       =  0.00628662109375
  //     bias        = -1.8125  * 2^-5       = -0.056640625
  //     w           = 10 * scale + bias
  //                 = 0.0628662109375 - 0.056640625
  //                 = 0.0062255859375
  //
  // Both scale and bias are exact in bf16 (6 and 4 fractional bits against
  // bf16's 7), so the expected value is exact, not approximate.
  const GroupAffine q = spec_of(4, 64);
  const std::size_t k = 192;
  std::vector<std::uint32_t> packed(q.packed_words(k), 0u);
  std::vector<bfloat16_t> scales(q.group_count(k));
  std::vector<bfloat16_t> biases(q.group_count(k));
  packed[16] = 0x76a8cbbbu;
  const float scale = 1.609375f / 256.0f;
  const float bias = -1.8125f / 32.0f;
  for (std::size_t g = 0; g < scales.size(); ++g) {
    scales[g] = bfloat16_t(scale);
    biases[g] = bfloat16_t(bias);
  }
  // The two constants survive the narrowing the checkpoint applies.
  LSE_EXPECT_NEAR(scales[2].to_float(), 0.00628662109375, 0.0);
  LSE_EXPECT_NEAR(biases[2].to_float(), -0.056640625, 0.0);

  LSE_EXPECT_EQ(q.code_at(packed.data(), 133), 10u);

  std::vector<float> out(k);
  q.dequantize_row<bfloat16_t>(packed.data(), scales.data(), biases.data(), k,
                               out.data());
  LSE_EXPECT_NEAR(out[133], 0.0062255859375, 0.0);
  // Every other code in that lane is a real code too, not padding.
  LSE_EXPECT_EQ(q.code_at(packed.data(), 128), 0xbu);   // 0x76a8cbbb & 0xF
  LSE_EXPECT_EQ(q.code_at(packed.data(), 135), 0x7u);   // >> 28
}

LSE_TEST(group_affine_dequant_matches_hand_arithmetic_across_a_lane) {
  // The case the 4-bit target never exercises and the 6-bit MoE sibling does:
  // a code split over two lanes.
  //
  //   6-bit code 5 starts at bit 5*6 = 30, so it takes
  //     the top 2 bits of lane 0 and the low 4 bits of lane 1
  //   lane 0 = 0xC0000000 -> bits 30,31 = 0b11        = 3
  //   lane 1 = 0x0000000A -> low 4 bits = 0b1010      = 10
  //   code   = 3 + 10 * 2^2                           = 43
  //   w      = 43 * 0.5 + (-1.0)                      = 20.5
  const GroupAffine q = spec_of(6, 64);
  LSE_EXPECT_EQ(q.chunk_word(5), 0);
  LSE_EXPECT_EQ(q.chunk_bit(5), 30);
  LSE_EXPECT_EQ(q.chunk_carry(5), 4);

  const std::size_t k = 64;
  std::vector<std::uint32_t> packed(q.packed_words(k), 0u);
  std::vector<bfloat16_t> scales(q.group_count(k), bfloat16_t(0.5f));
  std::vector<bfloat16_t> biases(q.group_count(k), bfloat16_t(-1.0f));
  packed[0] = 0xC0000000u;
  packed[1] = 0x0000000Au;

  LSE_EXPECT_EQ(q.code_at(packed.data(), 5), 43u);
  std::vector<float> out(k);
  q.dequantize_row<bfloat16_t>(packed.data(), scales.data(), biases.data(), k,
                               out.data());
  LSE_EXPECT_NEAR(out[5], 20.5, 0.0);
}

LSE_TEST(group_affine_round_trip_error_grows_as_bits_are_removed) {
  const auto w = make_weights(4096);
  double prev = 0.0;
  // Measured on this fixture (sigma=0.1, group 64, bf16 scale and bias):
  // 5.48e-4, 2.21e-3, 4.46e-3, 9.35e-3, 1.94e-2, 4.01e-2.
  for (int bits : {8, 6, 5, 4, 3, 2}) {
    const GroupAffine q = spec_of(bits, 64);
    const double e = q.round_trip_rmse<bfloat16_t>(w.data(), w.size());
    LSE_EXPECT(e > prev);
    prev = e;
  }
  LSE_EXPECT(prev < 5e-2);
  // Affine at 4 bits over 64-element groups lands slightly under the
  // symmetric Q4 above, which spends codes on a range its groups do not use.
  const GroupAffine q4 = spec_of(4, 64);
  const double e4 = q4.round_trip_rmse<bfloat16_t>(w.data(), w.size());
  LSE_EXPECT(e4 < rmse_for<Q4>(w));
}

LSE_TEST(group_affine_scale_is_signed_and_reproduces_its_anchor) {
  // MLX anchors whichever end of the group has the larger magnitude and flips
  // the scale's sign to do it, so which end code 0 sits at depends on the
  // data. Roughly half of a real checkpoint's scales are negative; an
  // implementation that assumes scale > 0 sign-flips those groups silently.
  const GroupAffine q = spec_of(4, 64);
  std::vector<std::uint32_t> packed(q.packed_words(64));
  std::vector<bfloat16_t> scales(1), biases(1);
  std::vector<float> back(64);

  // |min| > |max|: positive scale, bias at the minimum, code 0 there.
  std::vector<float> low(64);
  for (std::size_t i = 0; i < low.size(); ++i) {
    low[i] = -1.0f + 0.01f * static_cast<float>(i);  // [-1.0, -0.37]
  }
  q.quantize_row<bfloat16_t>(low.data(), 64, packed.data(), scales.data(),
                             biases.data());
  LSE_EXPECT(scales[0].to_float() > 0.0f);
  LSE_EXPECT_NEAR(biases[0].to_float(), -1.0, 0.0);
  LSE_EXPECT_EQ(q.code_at(packed.data(), 0), 0u);
  q.dequantize_row<bfloat16_t>(packed.data(), scales.data(), biases.data(), 64,
                               back.data());
  LSE_EXPECT_NEAR(back[0], -1.0, 0.0);
  for (std::size_t i = 0; i < low.size(); ++i) {
    LSE_EXPECT_NEAR(back[i], low[i], 3e-2);
  }

  // |max| >= |min|: the same group mirrored, and the scale goes negative.
  std::vector<float> high(64);
  for (std::size_t i = 0; i < high.size(); ++i) {
    high[i] = 0.37f + 0.01f * static_cast<float>(i);  // [0.37, 1.0]
  }
  q.quantize_row<bfloat16_t>(high.data(), 64, packed.data(), scales.data(),
                             biases.data());
  LSE_EXPECT(scales[0].to_float() < 0.0f);
  LSE_EXPECT_NEAR(biases[0].to_float(), 1.0, 0.0);
  LSE_EXPECT_EQ(q.code_at(packed.data(), 63), 0u);
  q.dequantize_row<bfloat16_t>(packed.data(), scales.data(), biases.data(), 64,
                               back.data());
  LSE_EXPECT_NEAR(back[63], 1.0, 0.0);
  for (std::size_t i = 0; i < high.size(); ++i) {
    LSE_EXPECT_NEAR(back[i], high[i], 3e-2);
  }
}

LSE_TEST(bits_from_shapes_solves_the_mlx_identity) {
  // w.shape(-1) * 32 / bits == scales.shape(-1) * group_size, run backwards.
  // K=1024 at group 64 is 16 groups whatever the width.
  const struct { std::int64_t lanes; int bits; } cases[] = {
      {64, 2}, {96, 3}, {128, 4}, {160, 5}, {192, 6}, {256, 8}};
  for (const auto& c : cases) {
    auto got = GroupAffine::bits_from_shapes(c.lanes, 16, 64);
    LSE_EXPECT(got.ok());
    if (got.ok()) LSE_EXPECT_EQ(*got, c.bits);
  }
  // 7 bits is not a width MLX emits, so shapes that solve to it are refused
  // rather than accepted and mis-decoded.
  LSE_EXPECT(!GroupAffine::bits_from_shapes(224, 16, 64).ok());
  // Shapes that do not solve to a whole width at all.
  LSE_EXPECT(!GroupAffine::bits_from_shapes(100, 16, 64).ok());
  LSE_EXPECT(!GroupAffine::bits_from_shapes(0, 16, 64).ok());
}

namespace {

// The shape of a mixed-precision mlx-lm config: a global pair, per-module
// overrides written into the same object, and one module skipped outright.
constexpr const char* kMixedConfig = R"({
  "model_type": "qwen3_5_moe",
  "quantization": {
    "group_size": 64,
    "bits": 6,
    "mode": "affine",
    "language_model.model.layers.0.mlp.gate": {"group_size": 64, "bits": 8},
    "language_model.model.layers.0.mlp.shared_expert_gate": {"bits": 8},
    "language_model.model.layers.0.self_attn.q_norm": false
  }
})";

constexpr const char* kUniformConfig = R"({
  "quantization": {"group_size": 64, "bits": 4, "mode": "affine"},
  "quantization_config": {"group_size": 64, "bits": 4, "mode": "affine"}
})";

}  // namespace

LSE_TEST(per_tensor_override_selects_by_module_path) {
  auto map = GroupAffineMap::from_config_json(kMixedConfig);
  LSE_EXPECT(map.ok());
  if (!map.ok()) return;
  LSE_EXPECT(map->has_global());
  LSE_EXPECT_EQ(map->global().bits, 6);
  LSE_EXPECT_EQ(map->override_count(), std::size_t(2));

  // The override is keyed by the module, so every leaf of the triple resolves
  // to it — the packed plane, the scales and the biases alike.
  for (const char* leaf : {".weight", ".scales", ".biases"}) {
    const std::string name =
        std::string("language_model.model.layers.0.mlp.gate") + leaf;
    auto got = map->resolve(name);
    LSE_EXPECT(got.ok());
    if (got.ok()) LSE_EXPECT_EQ(got->bits, 8);
  }
  // An override that names only `bits` inherits the global group size.
  auto shared = map->resolve(
      "language_model.model.layers.0.mlp.shared_expert_gate.weight");
  LSE_EXPECT(shared.ok());
  if (shared.ok()) {
    LSE_EXPECT_EQ(shared->bits, 8);
    LSE_EXPECT_EQ(shared->group_size, 64);
  }
  // Everything else takes the global.
  auto other = map->resolve("language_model.model.layers.0.mlp.down_proj.weight");
  LSE_EXPECT(other.ok());
  if (other.ok()) LSE_EXPECT_EQ(other->bits, 6);

  // A module the checkpoint left in full precision is refused, not guessed.
  LSE_EXPECT(map->is_skipped(
      "language_model.model.layers.0.self_attn.q_norm.weight"));
  LSE_EXPECT(!map->resolve(
      "language_model.model.layers.0.self_attn.q_norm.weight").ok());
}

LSE_TEST(override_disagreeing_with_the_shapes_is_an_error) {
  // The failure the override exists to prevent: an 8-bit router read at the
  // global 6 bits still produces plausible weights. Its plane is 256 lanes for
  // K=1024, and 6 bits would need 192, so the cross-check catches it.
  auto mixed = GroupAffineMap::from_config_json(kMixedConfig);
  LSE_EXPECT(mixed.ok());
  if (!mixed.ok()) return;
  const std::string router = "language_model.model.layers.0.mlp.gate.weight";
  LSE_EXPECT(mixed->resolve_checked(router, 256, 16).ok());
  LSE_EXPECT(!mixed->resolve_checked(router, 192, 16).ok());

  // Same checkpoint read by a loader that ignored the overrides: the global
  // says 6 bits, the router's own shapes say 8, and the load fails loudly.
  auto uniform = GroupAffineMap::from_config_json(
      R"({"quantization": {"group_size": 64, "bits": 6}})");
  LSE_EXPECT(uniform.ok());
  if (uniform.ok()) LSE_EXPECT(!uniform->resolve_checked(router, 256, 16).ok());
}

LSE_TEST(quantization_block_is_read_strictly) {
  auto uniform = GroupAffineMap::from_config_json(kUniformConfig);
  LSE_EXPECT(uniform.ok());
  if (uniform.ok()) {
    LSE_EXPECT_EQ(uniform->global().bits, 4);
    LSE_EXPECT_EQ(uniform->override_count(), std::size_t(0));
    LSE_EXPECT(uniform->resolve_checked("any.tensor.weight", 128, 16).ok());
  }

  // mxfp4/nvfp4/mxfp8 encode their scales differently; reading one as affine
  // would produce plausible garbage, so it is refused by name.
  LSE_EXPECT(!GroupAffineMap::from_config_json(
                  R"({"quantization": {"group_size": 32, "bits": 4,
                                       "mode": "mxfp4"}})")
                  .ok());
  // Half a global pair is a broken config, not a default.
  LSE_EXPECT(!GroupAffineMap::from_config_json(
                  R"({"quantization": {"bits": 4}})").ok());
  // An override with nothing to inherit from.
  LSE_EXPECT(!GroupAffineMap::from_config_json(
                  R"({"quantization": {"a.b": {"bits": 4}}})").ok());

  // No quantization block at all: nothing is invented, every lookup fails
  // naming the tensor.
  auto none = GroupAffineMap::from_config_json(R"({"model_type": "qwen3_5"})");
  LSE_EXPECT(none.ok());
  if (none.ok()) {
    LSE_EXPECT(!none->has_global());
    LSE_EXPECT(!none->resolve("some.tensor.weight").ok());
  }
}

LSE_TEST(group_affine_storage_is_not_a_dtype_property) {
  // The reason this is not a QuantScheme and kU32 is not is_quantized(): the
  // tag sizes the plane, the geometry sizes the weight, and only the pair
  // together describe the tensor.
  LSE_EXPECT(!is_quantized(DType::kU32));
  LSE_EXPECT_EQ(dtype_storage_bytes(DType::kU32, 128), std::size_t(512));
  LSE_EXPECT(to_string(DType::kU32) == "u32");
  const GroupAffine q4 = spec_of(4, 64);
  const GroupAffine q8 = spec_of(8, 64);
  LSE_EXPECT_EQ(dtype_storage_bytes(DType::kU32, q4.packed_words(1024)),
                std::size_t(512));
  LSE_EXPECT_EQ(dtype_storage_bytes(DType::kU32, q8.packed_words(1024)),
                std::size_t(1024));
}

LSE_TEST_MAIN()
