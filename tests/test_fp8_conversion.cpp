#include "harness.hpp"
#include "lse/math/fp8.hpp"
#include <bit>
#include <limits>

using namespace lse;
namespace {
template <math::MatrixElem E> void check_format() {
  using F = math::Fp8Format<E>;
  for (unsigned code = 0; code <= F::max_finite_code; ++code) {
    for (unsigned sign : {0u, 0x80u}) {
      const float x = math::fp8_value<E>(code | sign);
      LSE_EXPECT(math::fp8_bits<E>(x) == (code | sign));
      if (code == 0) LSE_EXPECT(std::signbit(x) == (sign != 0));
    }
    if (code == F::max_finite_code) break;
    const float lo = math::fp8_value<E>(code), hi = math::fp8_value<E>(code + 1);
    const float midpoint = (lo + hi) * 0.5f;
    const unsigned tie = code + (code & 1);
    LSE_EXPECT(math::fp8_bits<E>(midpoint) == tie);
    LSE_EXPECT(math::fp8_bits<E>(-midpoint) == (tie | 0x80));
    LSE_EXPECT(math::fp8_bits<E>(std::nextafter(midpoint, lo)) == code);
    LSE_EXPECT(math::fp8_bits<E>(std::nextafter(midpoint, hi)) == code + 1);
  }
  LSE_EXPECT(math::fp8_bits<E>(std::numeric_limits<float>::max()) == F::max_finite_code);
  LSE_EXPECT(math::fp8_bits<E>(-std::numeric_limits<float>::max()) == (F::max_finite_code | 0x80));
  LSE_EXPECT(math::fp8_bits<E>(std::numeric_limits<float>::infinity()) == (F::e4m3 ? 0x7f : 0x7c));
  LSE_EXPECT(math::fp8_bits<E>(-std::numeric_limits<float>::infinity()) == (F::e4m3 ? 0xff : 0xfc));
  for (unsigned raw : {0x7fc00001u, 0x7f800001u, 0xffc12345u})
    LSE_EXPECT(math::fp8_bits<E>(std::bit_cast<float>(raw)) == ((raw >> 24 & 0x80) | 0x7f));
  auto packed = math::pack_fp8<E>(0.0f, -0.0f, F::max_finite, -F::min_subnormal);
  LSE_EXPECT((packed & 0xffff) == 0x8000);
  LSE_EXPECT(math::unpack_fp8<E, 2>(packed) == F::max_finite);
  LSE_EXPECT(math::unpack_fp8<E, 3>(packed) == -F::min_subnormal);
}
}
LSE_TEST(ocp_e4m3_all_representable_values_and_rounding_boundaries) {
  check_format<math::MatrixElem::kFp8>();
  LSE_EXPECT(math::fp8_value<math::MatrixElem::kFp8>(0x38) == 1.0f);
  LSE_EXPECT(math::fp8_value<math::MatrixElem::kFp8>(0x7e) == 448.0f);
  LSE_EXPECT(math::fp8_value<math::MatrixElem::kFp8>(1) == 0x1p-9f);
}
LSE_TEST(ocp_e5m2_all_representable_values_and_rounding_boundaries) {
  check_format<math::MatrixElem::kBf8>();
  LSE_EXPECT(math::fp8_value<math::MatrixElem::kBf8>(0x3c) == 1.0f);
  LSE_EXPECT(math::fp8_value<math::MatrixElem::kBf8>(0x7b) == 57344.0f);
  LSE_EXPECT(math::fp8_value<math::MatrixElem::kBf8>(1) == 0x1p-16f);
}
LSE_TEST_MAIN()
