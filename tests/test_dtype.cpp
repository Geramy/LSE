// dtype table, and the bf16/fp16 host conversions the weight loader relies on.
#include "lse/core/dtype.hpp"

#include <cmath>
#include <limits>

#include "harness.hpp"

using namespace lse;

LSE_TEST(dtype_table_is_self_consistent) {
  for (int i = 0; i < static_cast<int>(DType::kCount); ++i) {
    const auto dt = static_cast<DType>(i);
    const DTypeInfo& info = dtype_info(dt);
    LSE_EXPECT(info.dtype == dt);
    LSE_EXPECT(!info.name.empty());
    LSE_EXPECT_EQ(is_quantized(dt), info.is_quantized);
  }
}

LSE_TEST(dtype_from_string_round_trips) {
  for (int i = 0; i < static_cast<int>(DType::kCount); ++i) {
    const auto dt = static_cast<DType>(i);
    LSE_EXPECT(dtype_from_string(to_string(dt)) == dt);
  }
  // Checkpoint spellings.
  LSE_EXPECT(dtype_from_string("bfloat16") == DType::kBF16);
  LSE_EXPECT(dtype_from_string("float32") == DType::kF32);
  LSE_EXPECT(dtype_from_string("nonsense") == DType::kCount);
}

LSE_TEST(bfloat16_round_trips_exactly_representable_values) {
  // bf16 keeps fp32's exponent and the top 7 mantissa bits, so any float whose
  // low 16 bits are already zero must survive exactly.
  const float values[] = {0.0f, 1.0f, -1.0f, 2.0f, 0.5f, -256.0f, 3.140625f};
  for (float v : values) {
    LSE_EXPECT_NEAR(bfloat16_t(v).to_float(), v, 0.0);
  }
}

LSE_TEST(bfloat16_keeps_fp32_exponent_range) {
  // This is the whole reason the model trains in bf16 rather than fp16: values
  // that would overflow fp16 stay finite.
  const float big = 1e30f;
  LSE_EXPECT(std::isfinite(bfloat16_t(big).to_float()));
  const float small = 1e-30f;
  LSE_EXPECT(bfloat16_t(small).to_float() != 0.0f);
}

LSE_TEST(bfloat16_rounds_to_nearest_even) {
  // 1.0 + one ulp of bf16 is 1 + 2^-7; halfway values must round to even.
  const float one_ulp = 1.0f + std::ldexp(1.0f, -7);
  LSE_EXPECT_NEAR(bfloat16_t(one_ulp).to_float(), one_ulp, 0.0);
  // A value just below the midpoint rounds down to 1.0.
  const float below = 1.0f + std::ldexp(1.0f, -9);
  LSE_EXPECT_NEAR(bfloat16_t(below).to_float(), 1.0f, 0.0);
}

LSE_TEST(bfloat16_preserves_nan) {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  LSE_EXPECT(std::isnan(bfloat16_t(nan).to_float()));
}

LSE_TEST(float16_round_trips_exact_values) {
  const float values[] = {0.0f, 1.0f, -1.0f, 0.5f, 2048.0f, -0.125f};
  for (float v : values) {
    LSE_EXPECT_NEAR(float16_t(v).to_float(), v, 0.0);
  }
}

LSE_TEST(float16_saturates_and_underflows) {
  LSE_EXPECT(std::isinf(float16_t(1e30f).to_float()));      // overflow -> inf
  LSE_EXPECT_NEAR(float16_t(1e-30f).to_float(), 0.0f, 0.0);  // underflow -> 0
  // Largest finite fp16 is 65504.
  LSE_EXPECT_NEAR(float16_t(65504.0f).to_float(), 65504.0f, 0.0);
}

LSE_TEST(float16_subnormals_survive) {
  // The smallest fp16 subnormal is 2^-24; the quant block scales can land here
  // for very small weights, so this path must not flush to zero.
  const float tiny = std::ldexp(1.0f, -24);
  LSE_EXPECT_NEAR(float16_t(tiny).to_float(), tiny, 0.0);
}

LSE_TEST(unquantized_storage_bytes) {
  LSE_EXPECT_EQ(dtype_storage_bytes(DType::kF32, 10), 40u);
  LSE_EXPECT_EQ(dtype_storage_bytes(DType::kBF16, 10), 20u);
  LSE_EXPECT_EQ(dtype_storage_bytes(DType::kI8, 10), 10u);
}

LSE_TEST_MAIN()
