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
  const bool expected=std::string_view(arch)=="gfx1100" ||
      std::string_view(arch)=="gfx1201";
  LSE_EXPECT((out->source.find("vector.dot4i<s8u8>")!=std::string::npos)==expected);
 }
 auto disabled=dot_fixture::projection("gfx1201",4,true);LSE_EXPECT(disabled.ok());
 if(disabled.ok())LSE_EXPECT(disabled->source.find("vector.dot4i")==std::string::npos);
 for(int bits:{6,8}){auto out=dot_fixture::projection("gfx1201",bits);LSE_EXPECT(out.ok());if(out.ok())LSE_EXPECT(out->source.find("vector.dot4i")==std::string::npos);}
}
namespace {
std::uint32_t nibble_plane(std::uint32_t word, int p) {
  std::uint32_t result=0;
  for(int b=0;b<4;++b)
    result |= ((word >> (4*(2*b+p))) & 15u) << (8*b);
  return result;
}
}
LSE_TEST(q4_code_planes_preserve_every_byte_and_full_word_patterns) {
 lse::graph::env::Cpu e;
 auto check=[&](std::uint32_t word) {
  for(int p=0;p<2;++p)
   LSE_EXPECT_EQ(lse::quant::dot4_code_plane(e,word,p),nibble_plane(word,p));
 };
 // Each output byte depends only on the corresponding input byte. Enumerating
 // every byte in every position proves that independent mapping; the other
 // bytes carry nonzero poison to catch cross-byte leakage.
 for(unsigned b=0;b<4;++b)for(std::uint32_t byte=0;byte<256;++byte) {
  const auto mask=255u << (8*b);
  check((0xa5963cf0u & ~mask) | (byte << (8*b)));
  check(byte * 0x01010101u);
 }
 for(auto word:{0u,0xffffffffu,0x80000000u,0x01234567u,0x89abcdefu,
                0x0f0f0f0fu,0xf0f0f0f0u})check(word);
 std::uint32_t word=0x6d2b79f5u;
 for(unsigned i=0;i<65536;++i) {
  word ^= word << 13;word ^= word >> 17;word ^= word << 5;
  check(word);
 }
}
LSE_TEST(q4_code_planes_emit_one_unsigned_mask_per_plane) {
 auto body=dot_fixture::body(3);LSE_EXPECT(body.ok());if(!body.ok())return;
 std::size_t count=0,at=0;
 while((at=body->text.find("scalar.andi",at))!=std::string::npos){++count;++at;}
 LSE_EXPECT_EQ(count,2u);
 LSE_EXPECT(body->text.find("scalar.divui")!=std::string::npos);
 LSE_EXPECT(body->text.find("scalar.divsi")==std::string::npos);
 LSE_EXPECT(body->text.find("scalar.addi")==std::string::npos);
 LSE_EXPECT(body->text.find("constant 252645135")!=std::string::npos);
 LSE_EXPECT(body->text.find("scalar.remui")==std::string::npos);
 auto uniform=dot_fixture::body(4);LSE_EXPECT(uniform.ok());if(!uniform.ok())return;
 LSE_EXPECT(uniform->text.find("index to i32")!=std::string::npos);
 LSE_EXPECT(uniform->text.find("scalar.andi")!=std::string::npos);
}
LSE_TEST_MAIN()
