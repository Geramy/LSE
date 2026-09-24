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
LSE_TEST_MAIN()
