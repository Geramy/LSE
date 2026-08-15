#include "lse/core/shape.hpp"

#include "harness.hpp"

using namespace lse;

LSE_TEST(shape_basics) {
  Shape s{2, 3, 4};
  LSE_EXPECT_EQ(s.rank(), 3u);
  LSE_EXPECT_EQ(s.elem_count(), 24u);
  LSE_EXPECT_EQ(s[0], 2);
  LSE_EXPECT_EQ(s[2], 4);
  LSE_EXPECT(s.to_string() == "(2, 3, 4)");
}

LSE_TEST(empty_shape_has_no_elements) {
  Shape s;
  LSE_EXPECT_EQ(s.rank(), 0u);
  LSE_EXPECT_EQ(s.elem_count(), 0u);
}

LSE_TEST(row_major_strides) {
  Shape s{2, 3, 4};
  const auto st = s.strides();
  LSE_EXPECT_EQ(st[0], 12);
  LSE_EXPECT_EQ(st[1], 4);
  LSE_EXPECT_EQ(st[2], 1);
}

LSE_TEST(broadcast_aligns_trailing_dims) {
  // [B,T,D] with [D] — the RMSNorm weight case.
  const Shape a{4, 128, 1024};
  const Shape b{1024};
  const Shape out = Shape::broadcast(a, b);
  LSE_EXPECT_EQ(out.rank(), 3u);
  LSE_EXPECT_EQ(out[0], 4);
  LSE_EXPECT_EQ(out[1], 128);
  LSE_EXPECT_EQ(out[2], 1024);
}

LSE_TEST(broadcast_expands_size_one_dims) {
  // [B,1,D] with [1,T,D] — the attention-bias case.
  const Shape out = Shape::broadcast(Shape{4, 1, 64}, Shape{1, 128, 64});
  LSE_EXPECT_EQ(out[0], 4);
  LSE_EXPECT_EQ(out[1], 128);
  LSE_EXPECT_EQ(out[2], 64);
}

LSE_TEST(incompatible_broadcast_reports_failure) {
  const Shape out = Shape::broadcast(Shape{4, 3}, Shape{4, 5});
  LSE_EXPECT_EQ(out.rank(), 0u);  // rank 0 is the failure signal
}

LSE_TEST(broadcastable_to_is_directional) {
  LSE_EXPECT(Shape{1024}.is_broadcastable_to(Shape{4, 128, 1024}));
  LSE_EXPECT(Shape{1, 64}.is_broadcastable_to(Shape{128, 64}));
  // A larger trailing dim cannot broadcast down.
  LSE_EXPECT(!Shape{2048}.is_broadcastable_to(Shape{4, 128, 1024}));
  LSE_EXPECT(!Shape{4, 128, 1024}.is_broadcastable_to(Shape{1024}));
}

LSE_TEST_MAIN()
