// Quantization scheme tests: layout ABI, round-trip fidelity, and the
// error ordering Q8 < Q6 < Q4 that the whole scheme family depends on.
#include "lse/quant/traits.hpp"

#include <numeric>
#include <random>
#include <vector>

#include "harness.hpp"

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

LSE_TEST_MAIN()
