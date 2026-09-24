#include "lse/kernels/int8_policy.hpp"
#include "harness.hpp"
#include "loom_dot_fixture.hpp"
LSE_TEST(mixed_dot_signed_result_and_same_width_casts_keep_signed_conversion) {
 auto b=dot_fixture::body(0);LSE_EXPECT(b.ok());if(!b.ok())return;
 LSE_EXPECT(b->text.find("vector.dot4i<s8u8>")!=std::string::npos);
 LSE_EXPECT(b->text.find("scalar.sitofp")!=std::string::npos);
 LSE_EXPECT(b->text.find("scalar.uitofp")==std::string::npos);
 LSE_EXPECT(b->text.find("i32 to i32")==std::string::npos);
 LSE_EXPECT(b->result_type == "f32");
}
LSE_TEST(activation_rint_uses_ties_to_even_not_away_from_zero) {
 auto b=dot_fixture::body(1);LSE_EXPECT(b.ok());if(!b.ok())return;
 LSE_EXPECT(b->text.find("scalar.roundevenf")!=std::string::npos);
 LSE_EXPECT(b->text.find("scalar.roundf")==std::string::npos);
}
LSE_TEST(uniform_unsigned_min_max_keep_index_casts_and_compare_select) {
 auto b=dot_fixture::body(2);LSE_EXPECT(b.ok());if(!b.ok())return;
 LSE_EXPECT(b->text.find("index to i32")!=std::string::npos);
 LSE_EXPECT(b->text.find("scalar.cmpi ult")!=std::string::npos);
 LSE_EXPECT(b->text.find("scalar.cmpi ugt")!=std::string::npos);
 LSE_EXPECT(b->text.find("scf.select")!=std::string::npos);
 LSE_EXPECT(b->text.find("scalar.minui")==std::string::npos);
 LSE_EXPECT(b->text.find("scalar.maxui")==std::string::npos);
 LSE_EXPECT(b->text.find("scalar.uitofp")!=std::string::npos);
}
LSE_TEST(q4_mixed_dot_is_gated_by_actual_arch_capability) {
 for(const auto* arch:{"gfx1030","gfx90a","gfx942","gfx950","gfx1100","gfx1201"}) {
  auto out=dot_fixture::projection(arch,4);LSE_EXPECT(out.ok());if(!out.ok()){std::fprintf(stderr,"%s: %s\n",arch,out.status().to_string().c_str());continue;}
  const bool expected=lse::kernels::activation_int8_enabled() &&
      (std::string_view(arch)=="gfx1100" || std::string_view(arch)=="gfx1201");
  LSE_EXPECT((out->source.find("vector.dot4i<s8u8>")!=std::string::npos)==expected);
 }
 auto disabled=dot_fixture::projection("gfx1201",4,true);LSE_EXPECT(disabled.ok());
 if(disabled.ok())LSE_EXPECT(disabled->source.find("vector.dot4i")==std::string::npos);
 for(int bits:{6,8}){auto out=dot_fixture::projection("gfx1201",bits);LSE_EXPECT(out.ok());if(out.ok())LSE_EXPECT(out->source.find("vector.dot4i")==std::string::npos);}
}
LSE_TEST_MAIN()
