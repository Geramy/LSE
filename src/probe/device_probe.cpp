#include "lse/probe/device_probe.hpp"

#include <chrono>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "lse/probe/probe_kernels.hpp"

namespace lse::probe {

namespace {

using Clock = std::chrono::steady_clock;

double ns_since(Clock::time_point t0) {
  return static_cast<double>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - t0)
          .count());
}

struct Registry {
  std::mutex mu;
  std::map<std::string, DeviceProbeFactory, std::less<>> factories;
};

Registry& registry() {
  static Registry r;
  return r;
}

// 8 MB is past every L2 on the parts this runs on, so a transfer rate here is
// the link's and not a cache's.
constexpr std::size_t kTransferBytes = 8u << 20;
constexpr int kTransferReps = 8;

// A backend with neither an emitter nor a compiler cannot run a generated
// kernel at all: its work is host work, and the probe bodies executed under
// env::Cpu are exactly what it would run.
bool host_executed(const backend::IBackend& be) noexcept {
  return be.emitter() == nullptr && be.compiler() == nullptr;
}

class PortableProbe final : public IDeviceProbe {
 public:
  explicit PortableProbe(backend::IBackend& be)
      : declined_(host_executed(be)
                      ? "host-executed backend: memory and dispatch are the "
                        "host's, and there is no matrix core to rate"
                      : "this backend registered no device probe, so its "
                        "kernel-side rates (streaming bandwidth, dispatch "
                        "cost, matrix-core throughput) are unknown here "
                        "rather than guessed") {
    // Asked here rather than reported after the fact: declined() names what
    // this probe cannot reach on this backend, and whether the backend answers
    // a memory query is a property of the backend, not of the run.
    if (!be.sample_free_memory().ok()) {
      declined_ += "; this backend's runtime answers no memory query, so free "
                   "memory stays unknown rather than being taken from the "
                   "declared total";
    }
  }

  Status run(backend::IBackend& be, DeviceProfile& out) override {
    const Status transfer = measure_transfers(be, out);
    if (const Measured sampled = sample_free_memory(be); sampled.known()) {
      out.free_memory = sampled;
    }
    if (host_executed(be)) measure_host_execution(out);
    return transfer;
  }

  std::string_view name() const noexcept override { return "portable"; }
  std::string_view declined() const noexcept override { return declined_; }

 private:
  static Status measure_transfers(backend::IBackend& be, DeviceProfile& out) {
    auto alloc = be.allocate(kTransferBytes, backend::MemoryClass::kDevice);
    if (!alloc.ok()) return alloc.status();
    backend::DeviceBuffer buf = alloc.release();
    std::vector<std::byte> host(kTransferBytes, std::byte{7});

    Status status = be.copy_h2d(host.data(), buf, kTransferBytes, 0);
    if (status.ok()) status = be.synchronize();
    if (status.ok()) {
      const auto t0 = Clock::now();
      for (int r = 0; r < kTransferReps && status.ok(); ++r) {
        status = be.copy_h2d(host.data(), buf, kTransferBytes, 0);
      }
      if (status.ok()) status = be.synchronize();
      const double ns = ns_since(t0);
      if (status.ok() && ns > 0.0) {
        out.h2d_bytes_per_s = Measured::measured(
            static_cast<double>(kTransferBytes) * kTransferReps * 1e9 / ns);
      }
    }
    if (status.ok()) {
      const auto t0 = Clock::now();
      for (int r = 0; r < kTransferReps && status.ok(); ++r) {
        status = be.copy_d2h(buf, host.data(), kTransferBytes, 0);
      }
      if (status.ok()) status = be.synchronize();
      const double ns = ns_since(t0);
      if (status.ok() && ns > 0.0) {
        out.d2h_bytes_per_s = Measured::measured(
            static_cast<double>(kTransferBytes) * kTransferReps * 1e9 / ns);
      }
    }
    be.deallocate(buf);
    return status;
  }

  // The same bodies a GPU backend records, executed instead of recorded.
  static void measure_host_execution(DeviceProfile& out) {
    constexpr std::uint32_t kElems = kTransferBytes / sizeof(float);
    std::vector<float> src(kElems, 1.0f);
    std::vector<float> dst(1, 0.0f);
    StreamArgs<env::Cpu> args{{src.data()}, {dst.data()}};

    const auto warm = [&] {
      env::run_flat(1, [&](env::Cpu& e, StreamArgs<env::Cpu>& a) {
        stream_read(e, a, kElems, 1u, 16u);
      }, args);
    };
    warm();
    constexpr int kReps = 4;
    const auto t0 = Clock::now();
    for (int r = 0; r < kReps; ++r) warm();
    const double ns = ns_since(t0);
    if (ns > 0.0) {
      out.dram_bytes_per_s = Measured::measured(
          static_cast<double>(kElems) * sizeof(float) * kReps * 1e9 / ns);
    }

    // The slot is volatile so the dispatch loop below cannot be deleted; a
    // measurement of code the compiler removed is not a measurement.
    static volatile float slot = 0.0f;
    TouchArgs<env::Cpu> touch{{const_cast<float*>(&slot)}};
    constexpr int kLaunches = 100000;
    const auto t1 = Clock::now();
    for (int r = 0; r < kLaunches; ++r) {
      env::run_flat(1, [](env::Cpu& e, TouchArgs<env::Cpu>& a) {
        touch_one(e, a);
      }, touch);
    }
    const double launch_ns = ns_since(t1);
    if (launch_ns > 0.0) {
      out.launch_overhead_ns =
          Measured::measured(launch_ns / static_cast<double>(kLaunches));
    }
  }

  std::string declined_;
};

}  // namespace

void register_device_probe(std::string_view backend_name,
                           DeviceProbeFactory factory) {
  Registry& r = registry();
  std::lock_guard lock(r.mu);
  r.factories[std::string(backend_name)] = factory;
}

std::unique_ptr<IDeviceProbe> make_portable_device_probe(
    backend::IBackend& backend) {
  return std::make_unique<PortableProbe>(backend);
}

Measured sample_free_memory(const backend::IBackend& backend) {
  auto free_bytes = backend.sample_free_memory();
  if (!free_bytes.ok()) return Measured::unknown();
  // Zero is a real answer here — a device with nothing left — so unlike the
  // rates below it is not screened out. What it must not be is a stand-in for
  // "did not ask".
  return Measured::measured(static_cast<double>(*free_bytes));
}

std::unique_ptr<IDeviceProbe> create_device_probe(backend::IBackend& backend) {
  Registry& r = registry();
  DeviceProbeFactory factory = nullptr;
  {
    std::lock_guard lock(r.mu);
    const auto it = r.factories.find(backend.name());
    if (it != r.factories.end()) factory = it->second;
  }
  if (factory == nullptr) return make_portable_device_probe(backend);
  auto probe = factory(backend);
  return probe != nullptr ? std::move(probe)
                          : make_portable_device_probe(backend);
}

DeviceProfile device_identity(const backend::IBackend& backend) {
  const backend::DeviceInfo& info = backend.device_info();
  DeviceProfile p;
  p.id.backend = std::string(backend.name());
  p.id.ordinal = static_cast<int>(info.ordinal);
  p.arch = info.arch;
  p.name = info.name;
  // total_memory is the declared part size and deliberately does not become
  // free_memory: a device already holding a model has the same total and no
  // room. free_memory is a measurement and is left to the probe, so that a
  // profile read from disk can be checked against this without either of them
  // carrying a figure nobody sampled this run.
  p.total_memory = info.total_memory;
  p.compute_units = info.compute_units;
  p.unified_memory = info.unified_memory;
  return p;
}

Result<DeviceProfile> probe_device(backend::IBackend& backend) {
  DeviceProfile profile = device_identity(backend);
  auto probe = create_device_probe(backend);
  // The portable probe measures transfers for everyone, including a backend
  // whose own probe covers the kernel side; running it first means a device
  // probe never has to restate the parts that are the same everywhere.
  if (probe->name() != "portable") {
    auto portable = make_portable_device_probe(backend);
    const Status s = portable->run(backend, profile);
    if (!s.ok()) return s;
  }
  LSE_RETURN_IF_ERROR(probe->run(backend, profile));
  return profile;
}

}  // namespace lse::probe
