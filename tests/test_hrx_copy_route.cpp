#include <functional>
#include <vector>

#include "harness.hpp"
#include "lse/backends/hrx/copy_route.hpp"
using namespace lse::backend;
LSE_TEST(hrx_copy_route_requires_this_backend_owner_for_both_buffers) {
  DeviceBuffer src, dst;
  src.handle = 1;
  dst.handle = 2;
  src.member = dst.member = 0;
  src.residency = dst.residency = DeviceIndex{4};
  LSE_EXPECT(detail::own_single_device_copy(DeviceIndex{4}, 1, 2, src, dst));
  // Equal tokens from another backend are still foreign to this queue.
  LSE_EXPECT(!detail::own_single_device_copy(DeviceIndex{5}, 1, 2, src, dst));
  src.residency = DeviceIndex{5};
  LSE_EXPECT(!detail::own_single_device_copy(DeviceIndex{4}, 1, 2, src, dst));
  src.residency = DeviceIndex{4};
  dst.residency = DeviceIndex{5};
  LSE_EXPECT(!detail::own_single_device_copy(DeviceIndex{4}, 1, 2, src, dst));
  src.residency = dst.residency = kNoDevice;
  LSE_EXPECT(!detail::own_single_device_copy(kNoDevice, 1, 2, src, dst));
}
LSE_TEST(hrx_copy_route_rejects_spanning_and_unknown_member_placements) {
  DeviceBuffer src, dst;
  src.residency = dst.residency = DeviceIndex{1};
  src.member = 0;
  dst.member = 1;
  LSE_EXPECT(detail::own_single_device_copy(DeviceIndex{1}, 1, 2, src, dst));
  LSE_EXPECT(!detail::own_single_device_copy(DeviceIndex{1}, 2, 2, src, dst));
  LSE_EXPECT(!detail::own_single_device_copy(DeviceIndex{1}, 1, 1, src, dst));
  dst.member = DeviceBuffer::kAnyMember;
  LSE_EXPECT(!detail::own_single_device_copy(DeviceIndex{1}, 1, 2, src, dst));
  src.member = DeviceBuffer::kAnyMember;
  LSE_EXPECT(!detail::own_single_device_copy(DeviceIndex{1}, 1, 2, src, dst));
}
LSE_TEST(hrx_copy_route_does_not_require_cpu_mapped_gpu_memory) {
  DeviceBuffer src, dst;
  src.residency = dst.residency = DeviceIndex{1};
  src.member = dst.member = 0;
  src.handle = 7;
  dst.handle = 8;
  LSE_EXPECT(src.ptr == nullptr && dst.ptr == nullptr);
  LSE_EXPECT(detail::own_single_device_copy(DeviceIndex{1}, 1, 1, src, dst));
}

namespace {

struct CopyPoint {
  std::uint32_t stream;
  std::uint64_t value;
};

struct DeferredCopyQueues {
  struct Submission {
    std::vector<CopyPoint> waits;
    CopyPoint signal;
    std::function<void()> work;
    bool done = false;
  };
  std::array<std::uint64_t, 2> timeline{}, completed{};
  std::array<std::vector<std::function<void()>>, 2> pending;
  std::vector<Submission> submitted;
  std::size_t flush_calls = 0, copy_waits = 0, reservations = 0;
  int fail_flush = -1;

  lse::Status flush(std::uint32_t stream) {
    ++flush_calls;
    if (static_cast<int>(stream) == fail_flush) {
      return LSE_ERROR(kInternal, "submit failed");
    }
    if (pending[stream].empty()) return lse::OkStatus();
    const CopyPoint before{stream, timeline[stream]};
    const CopyPoint after{stream, ++timeline[stream]};
    auto work = std::move(pending[stream]);
    pending[stream].clear();
    submitted.push_back({{before}, after,
                         [work = std::move(work)] {
                           for (const auto& operation : work) operation();
                         }});
    return lse::OkStatus();
  }

  lse::Status copy(std::uint32_t producer, std::uint32_t consumer,
                   int& source, int& destination) {
    return detail::submit_ordered_copy<CopyPoint>(
        producer, consumer, [&](std::uint32_t s) { return flush(s); },
        [&](std::uint32_t s) -> lse::Result<CopyPoint> {
          return CopyPoint{s, timeline[s]};
        },
        [&](std::uint32_t s) -> lse::Result<std::uint64_t> {
          ++reservations;
          return ++timeline[s];
        },
        [&](std::span<const CopyPoint> waits, CopyPoint on, std::uint64_t value) {
          copy_waits = waits.size();
          submitted.push_back({{waits.begin(), waits.end()}, {on.stream, value},
                               [&] { destination = source; }});
          return lse::OkStatus();
        });
  }

  bool drain() {
    std::size_t remaining = submitted.size();
    while (remaining != 0) {
      bool progress = false;
      // Prefer the newest ready submission: FIFO accident cannot hide a
      // missing producer or destination dependency.
      for (auto it = submitted.rbegin(); it != submitted.rend(); ++it) {
        if (it->done) continue;
        bool ready = true;
        for (const CopyPoint wait : it->waits) {
          ready &= completed[wait.stream] >= wait.value;
        }
        if (!ready) continue;
        it->work();
        completed[it->signal.stream] = it->signal.value;
        it->done = true;
        --remaining;
        progress = true;
      }
      if (!progress) return false;
    }
    return true;
  }
};

}  // namespace

LSE_TEST(hrx_ordered_copy_submits_pending_same_stream_producer) {
  DeferredCopyQueues queue;
  int source = 0, destination = -1, observed = -1;
  queue.pending[0].push_back([&] { source = 37; });
  LSE_EXPECT_OK(queue.copy(0, 0, source, destination));
  LSE_EXPECT_EQ(queue.flush_calls, 1u);
  LSE_EXPECT_EQ(queue.copy_waits, 1u);
  // Submission is asynchronous; neither copy nor producer has run.
  LSE_EXPECT_EQ(source, 0);
  LSE_EXPECT_EQ(destination, -1);
  queue.pending[0].push_back([&] { observed = destination; });
  LSE_EXPECT_OK(queue.flush(0));
  LSE_EXPECT(queue.drain());
  LSE_EXPECT_EQ(destination, 37);
  LSE_EXPECT_EQ(observed, 37);
}

LSE_TEST(hrx_ordered_copy_follows_pending_destination_writes) {
  DeferredCopyQueues queue;
  int source = 0, destination = 3, before = -1, after = -1;
  queue.pending[0].push_back([&] { source = 55; });
  queue.pending[1].push_back([&] {
    before = destination;
    destination = 99;
  });
  LSE_EXPECT_OK(queue.copy(0, 1, source, destination));
  LSE_EXPECT_EQ(queue.flush_calls, 2u);
  LSE_EXPECT_EQ(queue.copy_waits, 2u);
  queue.pending[1].push_back([&] { after = destination; });
  LSE_EXPECT_OK(queue.flush(1));
  LSE_EXPECT(queue.drain());
  LSE_EXPECT_EQ(before, 3);
  LSE_EXPECT_EQ(destination, 55);
  LSE_EXPECT_EQ(after, 55);
}

LSE_TEST(hrx_ordered_copy_does_not_reserve_after_submit_failure) {
  for (int failed : {0, 1}) {
    DeferredCopyQueues queue;
    queue.fail_flush = failed;
    int source = 4, destination = 2;
    const auto status = queue.copy(0, 1, source, destination);
    LSE_EXPECT(!status.ok());
    LSE_EXPECT_EQ(queue.reservations, 0u);
    LSE_EXPECT_EQ(queue.copy_waits, 0u);
    LSE_EXPECT_EQ(destination, 2);
  }
}

LSE_TEST_MAIN()
