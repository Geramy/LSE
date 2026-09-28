#include "harness.hpp"
#include "lse/core/sha256.hpp"

#include <algorithm>
#include <array>
#include <string>
#include <utility>

using namespace lse;
namespace {
std::string incremental(std::string_view input, std::size_t stride) {
  Sha256 h;
  for (std::size_t i = 0; i < input.size(); i += stride)
    h.update(std::as_bytes(std::span(input.data() + i, std::min(stride, input.size() - i))));
  auto digest = h.finish();
  std::string out(64, '0');
  constexpr char hex[] = "0123456789abcdef";
  for (std::size_t i = 0; i < 32; ++i) {
    unsigned v = std::uint8_t(digest[i]);
    out[2 * i] = hex[v >> 4];
    out[2 * i + 1] = hex[v & 15];
  }
  return out;
}
}

LSE_TEST(sha256_known_vectors_padding_streaming_and_mutation) {
  const std::array<std::pair<std::string, std::string>, 4> vectors{
      {{"", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
       {"abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"},
       {"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"},
       {std::string(1000000, 'a'),
        "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"}}};
  for (const auto& [input, expected] : vectors) {
    LSE_EXPECT(sha256(input) == expected);
    for (auto stride : {1u, 7u, 64u, 4096u}) LSE_EXPECT(incremental(input, stride) == expected);
  }
  for (auto n : {55u, 56u, 63u, 64u, 65u, 119u, 120u, 128u}) {
    auto input = std::string(n, 'x');
    LSE_EXPECT(sha256(input) == incremental(input, 13));
    auto original = sha256(input);
    input[n / 2] = 'y';
    LSE_EXPECT(sha256(input) != original);
  }
}
LSE_TEST_MAIN()
