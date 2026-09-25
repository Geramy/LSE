#include "harness.hpp"
#include "lse/backends/hrx/probe_measurement.hpp"
#include "lse/backends/hrx/probe_matrix_measurement.hpp"
#include <limits>
#include <memory>
#include "lse/backends/hrx/probe_retirement.hpp"
#include <vector>

using namespace lse::backend::hrx_kernels;
LSE_TEST(probe_working_set_requires_and_respects_a_memory_budget) {
  LSE_EXPECT_EQ(streaming_probe_bytes(0, std::nullopt), 0u);
  LSE_EXPECT_EQ(streaming_probe_bytes(0, 16u << 20), 0u);
  LSE_EXPECT_EQ(streaming_probe_bytes(std::size_t{32} << 30, 0), 0u);
  LSE_EXPECT_EQ(streaming_probe_bytes(0, 128u << 20), 32u << 20);
  LSE_EXPECT_EQ(streaming_probe_bytes(std::size_t{32} << 30, std::nullopt), 512u << 20);
  LSE_EXPECT_EQ(streaming_probe_bytes(std::size_t{32} << 30, 64u << 20), 16u << 20);
}
LSE_TEST(probe_checks_every_result_and_every_guard) {
  std::vector<float> values(1024 + 16, -1234.0f);
  std::fill_n(values.begin(), 1024, 4096.0f);
  LSE_EXPECT_OK(check_probe_output(values, 1024, 4096.0f));
  for (std::size_t i : {std::size_t{0}, std::size_t{777}, std::size_t{1023},
                         std::size_t{1024}, values.size() - 1}) {
    const float old = values[i];
    values[i] = std::numeric_limits<float>::quiet_NaN();
    LSE_EXPECT(!check_probe_output(values, 1024, 4096.0f).ok());
    values[i] = old;
  }
  LSE_EXPECT(!check_probe_output({}, 0, 1).ok());
  LSE_EXPECT(!check_probe_output(std::span(values).first(1024), 1024, 4096).ok());
}
template <int Tag> struct OwnedProbeBuffer { std::shared_ptr<int> storage; };
LSE_TEST(probe_partial_submission_always_drains_before_release) {
  using Buffer = OwnedProbeBuffer<0>;
  ProbeRetirement<Buffer, 2> pending;
  LSE_EXPECT(pending.available());
  Buffer a{std::make_shared<int>(1)}, b{std::make_shared<int>(2)};
  std::weak_ptr<int> first = a.storage, second = b.storage;
  LSE_EXPECT(pending.track(a));
  LSE_EXPECT(pending.track(b));
  a = {}; b = {};
  bool drained = false;
  int releases = 0;
  // Simulates one accepted launch followed by a rejected launch: regardless
  // of the operation's failure, finalization must drain the earlier work.
  LSE_EXPECT(pending.finish([&] { drained = true; return true; }, [&](Buffer& buffer) {
    LSE_EXPECT(drained);
    ++releases;
    buffer = {};
  }));
  LSE_EXPECT_EQ(releases, 2);
  LSE_EXPECT(first.expired() && second.expired());
  ProbeRetirement<Buffer, 2> next;
  LSE_EXPECT(next.available());
}
LSE_TEST(probe_failed_drain_retains_storage_and_blocks_later_probes) {
  using Buffer = OwnedProbeBuffer<1>;
  std::weak_ptr<int> owner;
  int releases = 0, drains = 0;
  {
    ProbeRetirement<Buffer, 2> pending;
    Buffer buffer{std::make_shared<int>(7)};
    owner = buffer.storage;
    LSE_EXPECT(pending.track(buffer));
    buffer = {};
    LSE_EXPECT(!pending.finish([&] { ++drains; return false; }, [&](Buffer&) { ++releases; }));
  }
  LSE_EXPECT_EQ(drains, 1);
  LSE_EXPECT_EQ(releases, 0);
  LSE_EXPECT(!owner.expired());
  ProbeRetirement<Buffer, 2> next;
  LSE_EXPECT(!next.available());
}
LSE_TEST(probe_quarantine_capacity_is_reserved_before_allocation) {
  using Buffer = OwnedProbeBuffer<2>;
  ProbeRetirement<Buffer, 2> first, second, exhausted;
  LSE_EXPECT(first.available() && second.available());
  LSE_EXPECT(!exhausted.available());
  LSE_EXPECT(first.finish([] { return true; }, [](Buffer&) {}));
  ProbeRetirement<Buffer, 2> reused;
  LSE_EXPECT(reused.available());
}
LSE_TEST(probe_unwinding_without_drain_preserves_buffer_owner) {
  using Buffer = OwnedProbeBuffer<3>;
  std::weak_ptr<int> owner;
  {
    ProbeRetirement<Buffer, 2> pending;
    Buffer buffer{std::make_shared<int>(9)};
    owner = buffer.storage;
    LSE_EXPECT(pending.track(buffer));
  }
  LSE_EXPECT(!owner.expired());
  ProbeRetirement<Buffer, 2> next;
  LSE_EXPECT(!next.available());
}
LSE_TEST(matrix_probe_seeds_exact_typed_ones) {
  for(auto storage:{lse::DType::kF16,lse::DType::kBF16,lse::DType::kI32})for(bool x:{false,true}) {
    auto bytes=matrix_probe_ones(storage,x,64);LSE_EXPECT(bytes.ok());if(!bytes.ok())continue;
    const unsigned width=storage==lse::DType::kI32||x?4:2;
    const std::uint32_t expected=storage==lse::DType::kI32?0x01010101u:x?0x3f800000u:storage==lse::DType::kF16?0x3c00u:0x3f80u;
    for(std::size_t i=0;i<bytes->size();i+=width){std::uint32_t bits=0;std::memcpy(&bits,bytes->data()+i,width);LSE_EXPECT_EQ(bits,expected);}
  }
  LSE_EXPECT(!matrix_probe_ones(lse::DType::kU8,false,64).ok());
  LSE_EXPECT(!matrix_probe_ones(lse::DType::kF16,false,3).ok());
}
LSE_TEST(matrix_probe_checks_prefix_payload_and_suffix_exactly) {
  constexpr std::size_t guard=16,payload=256*256;
  std::vector<float> values(payload+2*guard,-1234.0f);std::fill_n(values.begin()+guard,payload,1024.0f);
  LSE_EXPECT_OK(check_matrix_probe_output(values,guard,payload,1024));
  for(std::size_t i:{std::size_t{0},guard-1,guard,guard+12345,guard+payload-1,guard+payload,values.size()-1}) {
    const auto old=values[i];values[i]=std::numeric_limits<float>::quiet_NaN();
    LSE_EXPECT(!check_matrix_probe_output(values,guard,payload,1024).ok());values[i]=old;
  }
  LSE_EXPECT(!check_matrix_probe_output(values,0,payload,1024).ok());
}
LSE_TEST(matrix_probe_three_buffer_retirement_retains_all_on_failure) {
  using Buffer=OwnedProbeBuffer<4>;
  ProbeRetirement<Buffer,2> pending;std::array<std::weak_ptr<int>,3> owners;
  for(unsigned i=0;i<3;++i){Buffer b{std::make_shared<int>(int(i))};owners[i]=b.storage;LSE_EXPECT(pending.track(b));}
  Buffer extra{std::make_shared<int>(9)};LSE_EXPECT(!pending.track(extra));
  unsigned releases=0;LSE_EXPECT(!pending.finish([]{return false;},[&](Buffer&){++releases;}));
  LSE_EXPECT_EQ(releases,0u);for(auto&owner:owners)LSE_EXPECT(!owner.expired());
  ProbeRetirement<Buffer,2> later;LSE_EXPECT(!later.available());
}
LSE_TEST(matrix_probe_three_buffers_release_only_after_drain) {
  using Buffer=OwnedProbeBuffer<5>;
  ProbeRetirement<Buffer,2> pending;std::array<std::weak_ptr<int>,3> owners;
  for(unsigned i=0;i<3;++i){Buffer b{std::make_shared<int>(int(i))};owners[i]=b.storage;LSE_EXPECT(pending.track(b));}
  bool drained=false;unsigned releases=0;
  LSE_EXPECT(pending.finish([&]{drained=true;return true;},[&](Buffer&b){LSE_EXPECT(drained);b={};++releases;}));
  LSE_EXPECT_EQ(releases,3u);for(auto&owner:owners)LSE_EXPECT(owner.expired());
}
LSE_TEST_MAIN()
