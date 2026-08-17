// rdma:// and tb5:// — the fabric seam, and what this machine actually has.
//
// Neither of these is stubbed out at compile time and neither is switched off:
// both TUs are always built, both schemes are always registered, and each one
// ASKS THE MACHINE at open whether it can serve the endpoint. A build with a
// NIC answers differently from a build without one, with no edit anywhere,
// which is the whole point of declining rather than #ifdef-ing.
//
// What they decline for today, measured on this box rather than assumed:
//   rdma:// — libibverbs loads and enumerates ZERO devices, /sys/class/infiniband
//             is empty, and librdmacm (the half that supplies listen/connect/
//             accept) is not installed at all. There is nothing to open.
//   tb5://  — libodl_tb5 is installed but odl_tb5.ko is not inserted, so no
//             /dev/odl_tb5_* node exists and no Thunderbolt peer is cabled.
//
// Writing either data path now would be writing against hardware that is not
// here and could not be run once. When a device appears, the probe below
// changes its answer on its own and the transport is the piece to fill in.
#include <dirent.h>
#include <dlfcn.h>

#include <cstring>
#include <mutex>
#include <string>
#include <string_view>

#include "lse/communication/adapter.hpp"
#include "lse/communication/transport.hpp"

namespace lse::comm {

namespace {

// Only the enumeration entry points, and only as opaque pointers: counting
// devices needs no struct layout, so nothing of a third party's ABI is
// hand-copied into this tree where it could drift.
using IbvGetDeviceList = void** (*)(int*);
using IbvFreeDeviceList = void (*)(void**);

struct FabricProbe {
  bool usable = false;
  std::string reason;
};

void* try_dlopen(const char* soname) {
  return ::dlopen(soname, RTLD_LAZY | RTLD_LOCAL);
}

FabricProbe probe_verbs() {
  FabricProbe out;
  void* verbs = try_dlopen("libibverbs.so.1");
  if (verbs == nullptr) {
    out.reason =
        "libibverbs.so.1 is not loadable on this machine, so there is no verbs "
        "provider to open";
    return out;
  }
  auto get_list =
      reinterpret_cast<IbvGetDeviceList>(::dlsym(verbs, "ibv_get_device_list"));
  auto free_list = reinterpret_cast<IbvFreeDeviceList>(
      ::dlsym(verbs, "ibv_free_device_list"));
  if (get_list == nullptr || free_list == nullptr) {
    out.reason =
        "libibverbs.so.1 loaded but does not export ibv_get_device_list; this "
        "is not a usable verbs provider";
    return out;
  }
  int count = 0;
  void** list = get_list(&count);
  if (list != nullptr) free_list(list);
  if (count <= 0) {
    out.reason =
        "libibverbs enumerates 0 RDMA devices on this machine (no InfiniBand, "
        "RoCE or iWARP provider is bound; /sys/class/infiniband is empty)";
    return out;
  }
  // The connection manager is the half that supplies listen/connect/accept.
  // Verbs without it can move bytes between queue pairs that were already
  // introduced out of band, which is not what this seam promises.
  void* cm = try_dlopen("librdmacm.so.1");
  if (cm == nullptr) {
    out.reason =
        "librdmacm.so.1 is not installed, so a verbs device cannot be reached "
        "by endpoint: there is no connection manager to listen or connect with";
    return out;
  }
  out.usable = true;
  return out;
}

bool has_device_node(const char* dir, std::string_view prefix) {
  DIR* d = ::opendir(dir);
  if (d == nullptr) return false;
  bool found = false;
  while (const dirent* e = ::readdir(d)) {
    if (std::string_view(e->d_name).starts_with(prefix)) {
      found = true;
      break;
    }
  }
  ::closedir(d);
  return found;
}

FabricProbe probe_tb5() {
  FabricProbe out;
  void* lib = try_dlopen("libodl_tb5.so.0");
  if (lib == nullptr) {
    out.reason =
        "libodl_tb5.so.0 is not loadable on this machine, so the OdinLink ring "
        "cannot be opened";
    return out;
  }
  if (::dlsym(lib, "odl_tb5_open_path") == nullptr) {
    out.reason =
        "libodl_tb5.so.0 loaded but does not export odl_tb5_open_path; this is "
        "not a usable OdinLink library";
    return out;
  }
  if (!has_device_node("/dev", "odl_tb5")) {
    out.reason =
        "no /dev/odl_tb5_* device node exists: the odl_tb5 module is not "
        "inserted, so there is no ring to map";
    return out;
  }
  out.usable = true;
  return out;
}

// Probed once and remembered: a fabric does not appear between two calls in one
// process, and repeating a dlopen per endpoint would leak a handle per call.
const FabricProbe& verbs_probe() {
  static const FabricProbe probed = probe_verbs();
  return probed;
}
const FabricProbe& tb5_probe() {
  static const FabricProbe probed = probe_tb5();
  return probed;
}

// Shared by both fabrics: everything here is "what this machine has", and the
// answer differs only in which probe is consulted.
template <typename Derived>
class FabricTransport {
 public:
  Status open_impl(const Endpoint& ep, Poller&, EventSink&) {
    const FabricProbe& p = Derived::probe();
    if (!p.usable) {
      return LSE_ERROR(kDeviceError, std::string(Derived::kScheme), ": ",
                       p.reason);
    }
    return LSE_ERROR(kUnimplemented, std::string(Derived::kScheme),
                     " found a usable fabric for '", ep.str(),
                     "' but this build has no data path for it yet");
  }

  void close_impl() noexcept {}

  [[nodiscard]] Capabilities capabilities_impl(const Endpoint&) const noexcept {
    // Every number a fabric link has comes from the device, and there is no
    // device. Zero here means "do not choose on this", which is exactly true.
    Capabilities caps;
    caps.reliable = true;
    caps.ordered = true;
    caps.full_duplex = true;
    caps.registers_memory = true;
    return caps;
  }

  [[nodiscard]] std::string_view declined_impl(const Endpoint&) const noexcept {
    const FabricProbe& p = Derived::probe();
    if (p.usable) {
      return "the fabric is present but this build has no data path for it yet";
    }
    return p.reason;
  }

  Result<ILink*> connect_impl(const Endpoint& ep, std::uint64_t) {
    return LSE_ERROR(kDeviceError, "cannot reach '", ep.str(), "': ",
                     std::string(declined_impl(ep)));
  }

  Result<IListenerImpl*> listen_impl(const Endpoint& ep, std::uint64_t) {
    return LSE_ERROR(kDeviceError, "cannot listen on '", ep.str(), "': ",
                     std::string(declined_impl(ep)));
  }
};

class RdmaTransport final : public FabricTransport<RdmaTransport>,
                            public Transport<RdmaTransport> {
 public:
  static constexpr std::string_view kScheme = "rdma";
  static const FabricProbe& probe() { return verbs_probe(); }
  // A verbs peer is reached by address and port, like any other host.
  [[nodiscard]] bool authority_is_host_impl() const noexcept { return true; }
};

class Tb5Transport final : public FabricTransport<Tb5Transport>,
                           public Transport<Tb5Transport> {
 public:
  static constexpr std::string_view kScheme = "tb5";
  static const FabricProbe& probe() { return tb5_probe(); }
};

}  // namespace

LSE_REGISTER_COMM_TRANSPORT("rdma", RdmaTransport)
LSE_REGISTER_COMM_TRANSPORT("tb5", Tb5Transport)

}  // namespace lse::comm
