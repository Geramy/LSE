// Compile-time checks that only run on a compiler with P2996.
#include "lse/core/reflect.hpp"
#include "lse/core/status.hpp"
#include "lse/graph/dialect_source.hpp"
#include "lse/graph/graph.hpp"
#include "lse/graph/kernel_ir.hpp"

#include "harness.hpp"

#if defined(__cpp_impl_reflection) && __cpp_impl_reflection >= 202506L

LSE_TEST(reflection_counts_match_declared_enums) {
  LSE_EXPECT_EQ(lse::reflected_enum_count<lse::StatusCode>(),
                lse::enum_count(lse::StatusCode{}));
  // Found by ADL: Dialect belongs to the kernel IR, which the graph names.
  LSE_EXPECT_EQ(lse::reflected_enum_count<lse::graph::Dialect>(),
                enum_count(lse::graph::Dialect{}));
  LSE_EXPECT_EQ(lse::reflected_enum_count<lse::graph::OpKind>(),
                lse::graph::enum_count(lse::graph::OpKind{}));
  LSE_EXPECT_EQ(lse::reflected_enum_count<lse::graph::kir::Scalar>(), 12u);
}

LSE_TEST(reflection_names_the_enumerator_not_the_display_string) {
  LSE_EXPECT(lse::reflected_enumerator_name(lse::StatusCode::kOk) == "kOk");
  LSE_EXPECT(lse::to_string(lse::StatusCode::kOk) == "ok");
  LSE_EXPECT(lse::reflected_enumerator_name(lse::graph::OpKind::kRMS) ==
             "kRMS");
  LSE_EXPECT(lse::graph::to_string(lse::graph::OpKind::kRMS) == "rms_norm");
}

#else

LSE_TEST(reflection_requires_gcc16_freflection) {
  LSE_EXPECT(false && "host compiler has no P2996; expected g++-16 -freflection");
}

#endif

LSE_TEST_MAIN()
