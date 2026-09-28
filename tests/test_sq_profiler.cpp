#include "harness.hpp"
#include "../src/backends/hrx/sq_profiler.hpp"

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

namespace {

using Bytes = std::vector<std::uint8_t>;
using lse::hrx::SqProfileParseState;
using lse::hrx::SqProfileWindow;
using lse::hrx::parse_sq_profile_bytes;

void put_u32(Bytes& bytes, std::size_t at, std::uint32_t value) {
  for (unsigned i = 0; i < 4; ++i) bytes[at + i] = (value >> (i * 8)) & 0xff;
}

void put_u64(Bytes& bytes, std::size_t at, std::uint64_t value) {
  put_u32(bytes, at, static_cast<std::uint32_t>(value));
  put_u32(bytes, at + 4, static_cast<std::uint32_t>(value >> 32));
}

void append_text(Bytes& bytes, std::string_view text) {
  bytes.insert(bytes.end(), text.begin(), text.end());
}

Bytes file_header() {
  Bytes bytes(24);
  put_u32(bytes, 0, 0x46505249u);
  put_u32(bytes, 8, 24);
  return bytes;
}

Bytes chunk(std::string_view content_type, const Bytes& payload,
            std::string_view name = {}) {
  Bytes bytes(104);
  put_u64(bytes, 0, 104 + content_type.size() + name.size() + payload.size());
  put_u64(bytes, 8, payload.size());
  put_u32(bytes, 56, 104);
  put_u32(bytes, 60, static_cast<std::uint32_t>(content_type.size()));
  put_u32(bytes, 64, static_cast<std::uint32_t>(name.size()));
  bytes[100] = 2;  // IRPF CHUNK
  append_text(bytes, content_type);
  append_text(bytes, name);
  bytes.insert(bytes.end(), payload.begin(), payload.end());
  return bytes;
}

Bytes counter_chunk() {
  constexpr std::string_view block = "SQ";
  constexpr std::string_view counter_name = "SQ_BUSY_CYCLES";
  constexpr std::string_view description = "cycles";
  Bytes payload(48);
  put_u32(payload, 0, 48 + block.size() + counter_name.size() +
                          description.size());
  put_u32(payload, 28, 2);  // A nonzero value offset catches name misparsing.
  put_u32(payload, 32, 1);
  put_u32(payload, 36, block.size());
  put_u32(payload, 40, counter_name.size());
  put_u32(payload, 44, description.size());
  append_text(payload, block);
  append_text(payload, counter_name);
  append_text(payload, description);
  return chunk("application/vnd.iree.hal.profile.counters", payload,
               "counter-metadata");
}

Bytes sample_chunk(std::uint64_t busy, bool with_ticks = true) {
  Bytes payload(112 + 3 * 8);
  put_u32(payload, 0, static_cast<std::uint32_t>(payload.size()));
  put_u32(payload, 4, with_ticks ? 4 : 0);
  put_u64(payload, 72, 100);
  put_u64(payload, 80, 150);
  put_u32(payload, 104, 3);
  put_u64(payload, 112, 11);
  put_u64(payload, 120, 22);
  put_u64(payload, 128, busy);
  return chunk("application/vnd.iree.hal.profile.counter-samples", payload,
               "dispatch-samples");
}

void append(Bytes& into, const Bytes& from) {
  into.insert(into.end(), from.begin(), from.end());
}

}  // namespace

LSE_TEST(sq_profile_parses_metadata_and_sample_payload_at_wire_offsets) {
  Bytes bytes = file_header();
  append(bytes, counter_chunk());
  append(bytes, sample_chunk(333));
  SqProfileParseState state;
  SqProfileWindow window;
  std::size_t consumed = 0;
  LSE_EXPECT(parse_sq_profile_bytes(bytes.data(), bytes.size(), true, state,
                                    window, consumed));
  LSE_EXPECT_EQ(consumed, bytes.size());
  LSE_EXPECT(state.have_busy_counter);
  LSE_EXPECT_EQ(state.busy_value_offset, 2u);
  LSE_EXPECT_EQ(window.busy, 333u);
  LSE_EXPECT_EQ(window.reference_ticks, 50u);
  LSE_EXPECT_EQ(window.sample_count, 1u);
}

LSE_TEST(sq_profile_only_publishes_new_complete_samples) {
  Bytes first = file_header();
  append(first, counter_chunk());
  SqProfileParseState state;
  SqProfileWindow window;
  std::size_t consumed = 0;
  LSE_EXPECT(parse_sq_profile_bytes(first.data(), first.size(), true, state,
                                    window, consumed));
  LSE_EXPECT_EQ(consumed, first.size());
  LSE_EXPECT_EQ(window.sample_count, 0u);

  Bytes second = sample_chunk(7, false);
  const std::size_t partial_size = second.size() / 2;
  LSE_EXPECT(parse_sq_profile_bytes(second.data(), partial_size, false, state,
                                    window, consumed));
  LSE_EXPECT_EQ(consumed, 0u);
  LSE_EXPECT_EQ(window.sample_count, 0u);
  LSE_EXPECT(parse_sq_profile_bytes(second.data(), second.size(), false,
                                    state, window, consumed));
  LSE_EXPECT_EQ(consumed, second.size());
  LSE_EXPECT_EQ(window.busy, 7u);
  LSE_EXPECT_EQ(window.reference_ticks, 0u);
  LSE_EXPECT_EQ(window.sample_count, 1u);

  LSE_EXPECT(parse_sq_profile_bytes(nullptr, 0, false, state, window, consumed));
  LSE_EXPECT_EQ(consumed, 0u);
  LSE_EXPECT_EQ(window.busy, 0u);
  LSE_EXPECT_EQ(window.sample_count, 0u);
}

LSE_TEST(sq_profile_does_not_guess_a_counter_without_metadata) {
  Bytes bytes = file_header();
  append(bytes, sample_chunk(99));
  SqProfileParseState state;
  SqProfileWindow window;
  std::size_t consumed = 0;
  LSE_EXPECT(parse_sq_profile_bytes(bytes.data(), bytes.size(), true, state,
                                    window, consumed));
  LSE_EXPECT_EQ(consumed, bytes.size());
  LSE_EXPECT_EQ(window.busy, 0u);
  LSE_EXPECT_EQ(window.reference_ticks, 0u);
  LSE_EXPECT_EQ(window.sample_count, 0u);
}

LSE_TEST(sq_profile_rejects_malformed_counter_and_sample_lengths) {
  Bytes bytes = file_header();
  Bytes counter = counter_chunk();
  const std::size_t counter_payload = 104 +
      std::string_view("application/vnd.iree.hal.profile.counters").size() +
      std::string_view("counter-metadata").size();
  put_u32(counter, counter_payload + 40, 0xffffffffu);
  append(bytes, counter);
  SqProfileParseState state;
  SqProfileWindow window;
  std::size_t consumed = 0;
  LSE_EXPECT(!parse_sq_profile_bytes(bytes.data(), bytes.size(), true, state,
                                     window, consumed));
  LSE_EXPECT_EQ(consumed, 0u);
  LSE_EXPECT(!state.have_busy_counter);

  bytes = file_header();
  append(bytes, counter_chunk());
  Bytes sample = sample_chunk(9);
  const std::size_t sample_payload = 104 +
      std::string_view("application/vnd.iree.hal.profile.counter-samples").size() +
      std::string_view("dispatch-samples").size();
  put_u32(sample, sample_payload + 104, 4);  // Only three values are present.
  append(bytes, sample);
  LSE_EXPECT(!parse_sq_profile_bytes(bytes.data(), bytes.size(), true, state,
                                     window, consumed));
  LSE_EXPECT_EQ(consumed, 0u);
  LSE_EXPECT(!state.have_busy_counter);
}

LSE_TEST_MAIN()
