#include "harness.hpp"
#include "lse/backends/cpu/cpu_backend.hpp"
#include "lse/backends/hrx/arch_database.hpp"
#include "lse/backends/hrx/probe_emit.hpp"
#include "lse/probe/device_probe.hpp"
#include <algorithm>
#include <cstring>
#include <limits>
#include <map>
using namespace lse;
namespace {
// Runs the actual HRX calibration state machine using host allocations and
// simulated kernel results. It never opens an HRX or HSA device.
struct RecordingDevice : backend::CpuBackend {
  static constexpr std::string_view kName = "hrx";
  backend::DeviceInfo info;
  backend::AmdDeviceInfo amd;
  enum class Failure { None, Result, TimedResult, Guard, Retirement } failure = Failure::None;
  bool retirement_failed = false;
  unsigned matrix_launches = 0, releases = 0;
  std::vector<std::weak_ptr<void>> owners;
  RecordingDevice() {
    info.arch="gfx1201"; info.compute_units=64; info.wavefront_size=32;
    info.max_threads_per_workgroup=1024; info.lds_bytes_per_workgroup=65536;
    backend::apply_arch_defaults(info,amd);
    info.extension_id=backend::AmdDeviceInfo::kExtensionId; info.extension=&amd;
  }
  const backend::DeviceInfo& device_info() const noexcept { return info; }
  Result<std::size_t> sample_free_memory() const { return 32u << 20; }
  Result<backend::DeviceBuffer> allocate(std::size_t bytes,backend::MemoryClass cls,backend::Stream stream) {
    auto result=backend::CpuBackend::allocate(bytes,cls,stream);
    if (result.ok()) {
      // CpuBackend already owns the allocation. Creating a second owner from
      // ptr would release the first owner here and leave both a dangling pointer
      // and a second free pending.
      LSE_EXPECT(result->storage != nullptr);
      LSE_EXPECT(result->storage.get() == result->ptr);
      owners.push_back(result->storage);
    }
    return result;
  }
  void deallocate(backend::DeviceBuffer& buffer) noexcept { ++releases; buffer={}; }
  Result<backend::KernelHandle> load_executable(std::string_view name,std::span<const std::byte>) {
    return backend::KernelHandle{1,0,std::string(name)};
  }
  Status synchronize() {
    return retirement_failed?LSE_ERROR(kDeviceError,"injected unfinished dispatch"):OkStatus();
  }
  Status launch(const backend::KernelHandle&,const backend::LaunchDims&,const backend::DispatchArgs& args,const backend::DispatchTarget&) {
    const auto& ref=args.bindings.back();
    auto* out=reinterpret_cast<float*>(static_cast<std::byte*>(ref.buffer->ptr)+ref.buffer->offset+ref.offset);
    const auto count=ref.length/sizeof(float);
    if(args.bindings.size()==3) {
      ++matrix_launches;
      std::fill_n(out,count,1024.0f);
      if(failure==Failure::Result || (failure==Failure::TimedResult && matrix_launches>1))out[count/2]=std::numeric_limits<float>::quiet_NaN();
      if(failure==Failure::Guard)out[-1]=0;
      if(failure==Failure::Retirement)retirement_failed=true;
    } else {
      const float expected=args.bindings.size()==2?float(args.bindings.front().length/ref.length):1.0f;
      std::fill_n(out,count,expected);
    }
    return OkStatus();
  }
};
unsigned measured(const probe::DeviceProfile& profile) {
  unsigned result=0; for(auto&row:profile.rows)if(row.support==probe::RowSupport::kMeasured)++result;return result;
}
void run(RecordingDevice::Failure failure) {
  backend::BackendAdapter<RecordingDevice> be;LSE_EXPECT_OK(be.init(0));be.impl().failure=failure;
  auto probe=probe::create_device_probe(be);LSE_EXPECT(probe!=nullptr);if(!probe)return;
  probe::DeviceProfile profile;LSE_EXPECT_OK(probe->run(be,profile));
  if(failure==RecordingDevice::Failure::None) {
    LSE_EXPECT_EQ(measured(profile),3u);
    for(auto&owner:be.impl().owners)LSE_EXPECT(owner.expired());
  } else {
    LSE_EXPECT_EQ(measured(profile),0u);
    for(auto&row:profile.rows)LSE_EXPECT(!row.flops.positive());
    if(failure==RecordingDevice::Failure::Retirement) {
      LSE_EXPECT_EQ(be.impl().matrix_launches,1u);
      unsigned alive=0;for(auto&owner:be.impl().owners)if(!owner.expired())++alive;
      LSE_EXPECT_EQ(alive,3u);
      const auto before=be.impl().owners.size();
      auto later=probe::create_device_probe(be);probe::DeviceProfile next;LSE_EXPECT_OK(later->run(be,next));
      LSE_EXPECT_EQ(be.impl().owners.size(),before);
    } else {
      if(failure==RecordingDevice::Failure::TimedResult)LSE_EXPECT(be.impl().matrix_launches>3u);
      else LSE_EXPECT_EQ(be.impl().matrix_launches,3u);
      for(auto&owner:be.impl().owners)LSE_EXPECT(owner.expired());
    }
  }
}
}
LSE_TEST(matrix_probe_records_only_verified_retired_rates) {run(RecordingDevice::Failure::None);}
LSE_TEST(matrix_probe_corrupt_result_is_not_a_measurement) {run(RecordingDevice::Failure::Result);}
LSE_TEST(matrix_probe_later_corruption_is_not_a_measurement) {run(RecordingDevice::Failure::TimedResult);}
LSE_TEST(matrix_probe_corrupt_guard_is_not_a_measurement) {run(RecordingDevice::Failure::Guard);}
LSE_TEST(matrix_probe_retirement_failure_stops_remaining_rows_and_keeps_owners) {run(RecordingDevice::Failure::Retirement);}
LSE_TEST_MAIN()
