#include "lse/backends/hrx/hrx_backend.hpp"

#include <dlfcn.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_set>
#include <optional>
#include <vector>

#include "lse/backends/hrx/arch_database.hpp"
#include "lse/backends/hrx/code_object.hpp"

extern "C" {
#include "hrx_runtime.h"
}

namespace lse::backend {

namespace {

// Residencies stamped by live hrx backends. copy_peer receives whatever
// buffer the link probe offers — including another backend family's — and an
// hrx_buffer_t reinterpreted from a foreign handle walks libhrx off a cliff.
// A buffer reaches copy_peer only after its backend allocated it, so
// registering at allocate and unregistering at shutdown brackets every
// legitimate residency.
std::mutex g_hrx_residency_mu;
std::unordered_set<std::uint16_t> g_hrx_residencies;

void register_hrx_residency(DeviceIndex d) {
  if (!d.bound()) return;
  const std::lock_guard lock(g_hrx_residency_mu);
  g_hrx_residencies.insert(d.value);
}

void unregister_hrx_residency(DeviceIndex d) {
  if (!d.bound()) return;
  const std::lock_guard lock(g_hrx_residency_mu);
  g_hrx_residencies.erase(d.value);
}

bool hrx_owns_residency(DeviceIndex d) {
  if (!d.bound()) return false;
  const std::lock_guard lock(g_hrx_residency_mu);
  return g_hrx_residencies.count(d.value) != 0;
}

// LSE_TRACE_SYNC=1: one unbuffered stderr line before and after every call
// that can block the host on the device, so a hang's last line names the
// blocking call. Diagnostic only; costs a getenv once.
bool trace_sync() noexcept {
  static const bool on = std::getenv("LSE_TRACE_SYNC") != nullptr;
  return on;
}
#define LSE_SYNC_TRACE(...)                        \
  do {                                             \
    if (trace_sync()) {                            \
      std::fprintf(stderr, "[sync] " __VA_ARGS__); \
      std::fprintf(stderr, "\n");                  \
      std::fflush(stderr);                         \
    }                                              \
  } while (0)

// Translates an hrx_status_t into our Status, taking ownership of the message.
[[maybe_unused]] Status from_hrx(hrx_status_t status, const char* context) {
  if (hrx_status_is_ok(status)) return OkStatus();

  std::string detail;
  char* message = nullptr;
  std::size_t length = 0;
  hrx_status_t fmt = hrx_status_to_string(status, &message, &length);
  if (hrx_status_is_ok(fmt) && message != nullptr) {
    detail.assign(message, length);
  }
  if (message != nullptr) hrx_status_free_message(message);
  hrx_status_ignore(fmt);

  const hrx_status_code_t code = hrx_status_code(status);
  StatusCode mapped = StatusCode::kDeviceError;
  switch (code) {
    case HRX_STATUS_INVALID_ARGUMENT: mapped = StatusCode::kInvalidArgument; break;
    case HRX_STATUS_OUT_OF_MEMORY:    mapped = StatusCode::kOutOfMemory; break;
    case HRX_STATUS_NOT_FOUND:        mapped = StatusCode::kNotFound; break;
    case HRX_STATUS_UNIMPLEMENTED:    mapped = StatusCode::kUnimplemented; break;
    default:                          mapped = StatusCode::kDeviceError; break;
  }
  return Status(mapped, std::string(context) + ": " +
                            (detail.empty() ? "hrx error" : detail));
}

template <typename T>
[[maybe_unused]] Status query_property(hrx_device_t device, hrx_device_property_t prop, T* out) {
  return from_hrx(hrx_device_get_property(device, prop, out, sizeof(T)),
                  "hrx_device_get_property");
}

[[maybe_unused]] std::string query_string_property(hrx_device_t device,
                                  hrx_device_property_t prop) {
  char buffer[256] = {};
  hrx_status_t s = hrx_device_get_property(device, prop, buffer, sizeof(buffer));
  if (!hrx_status_is_ok(s)) {
    hrx_status_ignore(s);
    return {};
  }
  buffer[sizeof(buffer) - 1] = '\0';
  return std::string(buffer);
}

// The GPU agents of the copy of libhsa-runtime64 that hrx has *already* loaded
// — dlopen by soname with RTLD_NOLOAD returns that same handle, never a second
// runtime, and a null handle means no device has been brought up yet rather
// than that the machine has none. The entry points below have been ABI-stable
// since HSA 1.0 (the hsa_amd_* pool calls since ROCm 2.x) and are declared here
// rather than by including <hsa/hsa.h>, which is not a build input of this
// backend. hsa_agent_get_info writes exactly the attribute's own width, so a
// caller's buffer must be at least that wide and there is no size to pass.
//
// Every query here can fail, and each caller has its own answer for not
// knowing; none of them invents one.
using HsaStatus = int;
constexpr HsaStatus kHsaSuccess = 0;
constexpr int kInfoDevice = 17;
constexpr int kDeviceTypeGpu = 1;
constexpr int kDeviceTypeCpu = 0;

struct HsaSignal {
  std::uint64_t handle = 0;
};

struct HsaAgent {
  std::uint64_t handle = 0;
};

// A global memory pool of one agent. Peer reach is a question about a pool, not
// about a device: an agent either may address another agent's coarse-grained
// global pool or may not.
struct HsaPool {
  std::uint64_t handle = 0;
};

class HsaRuntime {
 public:
  // `load_if_missing` is for a caller that needs the runtime before hrx has
  // brought it up (the pre-init stable-ref resolver): RTLD_NOLOAD alone finds
  // nothing then, and a plain dlopen by soname returns the same handle
  // preload_gpu_runtime already loaded rather than a second runtime. The hot
  // path keeps load_if_missing=false, so it behaves exactly as before.
  explicit HsaRuntime(bool load_if_missing = false) noexcept {
    lib_ = dlopen("libhsa-runtime64.so.1", RTLD_LAZY | RTLD_NOLOAD);
    if (lib_ == nullptr && load_if_missing) {
      lib_ = dlopen("libhsa-runtime64.so.1", RTLD_LAZY);
    }
    if (lib_ == nullptr) return;
    init_ = reinterpret_cast<InitFn>(dlsym(lib_, "hsa_init"));
    iterate_ = reinterpret_cast<IterateFn>(dlsym(lib_, "hsa_iterate_agents"));
    get_info_ = reinterpret_cast<GetInfoFn>(dlsym(lib_, "hsa_agent_get_info"));
    iterate_pools_ = reinterpret_cast<IteratePoolsFn>(
        dlsym(lib_, "hsa_amd_agent_iterate_memory_pools"));
    pool_info_ = reinterpret_cast<PoolInfoFn>(
        dlsym(lib_, "hsa_amd_memory_pool_get_info"));
    agent_pool_info_ = reinterpret_cast<AgentPoolInfoFn>(
        dlsym(lib_, "hsa_amd_agent_memory_pool_get_info"));
    allow_access_ = reinterpret_cast<AllowAccessFn>(
        dlsym(lib_, "hsa_amd_agents_allow_access"));
    async_copy_ = reinterpret_cast<AsyncCopyFn>(
        dlsym(lib_, "hsa_amd_memory_async_copy"));
    lock_ = reinterpret_cast<LockFn>(dlsym(lib_, "hsa_amd_memory_lock"));
    unlock_ = reinterpret_cast<UnlockFn>(dlsym(lib_, "hsa_amd_memory_unlock"));
    signal_create_ = reinterpret_cast<SignalCreateFn>(
        dlsym(lib_, "hsa_signal_create"));
    signal_destroy_ = reinterpret_cast<SignalDestroyFn>(
        dlsym(lib_, "hsa_signal_destroy"));
    signal_wait_ = reinterpret_cast<SignalWaitFn>(
        dlsym(lib_, "hsa_signal_wait_scacquire"));
    signal_store_ = reinterpret_cast<SignalStoreFn>(
        dlsym(lib_, "hsa_signal_store_screlease"));
  }
  ~HsaRuntime() {
    if (shared_signal_ready_ && signal_destroy_ != nullptr) {
      signal_destroy_(shared_signal_);
    }
    if (lib_ != nullptr) dlclose(lib_);
  }
  HsaRuntime(const HsaRuntime&) = delete;
  HsaRuntime& operator=(const HsaRuntime&) = delete;

  [[nodiscard]] bool reachable() const noexcept {
    return iterate_ != nullptr && get_info_ != nullptr;
  }

  // hsa_iterate_agents enumerates nothing until hsa_init has run. The hot path
  // never needs this because hrx_gpu_initialize inits HSA first, but a caller
  // that enumerates BEFORE the accelerator is up (the pre-init stable-ref
  // resolver) must init it itself. hsa_init is refcounted and idempotent — a
  // second call just increments the count — and IREE's own init joins the same
  // instance, so initialising here and letting hrx init again later is safe.
  // Only hsa_shutdown is one-way, and this never calls it.
  [[nodiscard]] bool ensure_initialized() const noexcept {
    if (init_ == nullptr) return false;
    // Once per process, not once per call: a pool names several stable refs and
    // each one resolves through here, and the count hrx's own init and shutdown
    // keep has to stay exact.
    std::call_once(once_, [&] { initialized_ = init_() == kHsaSuccess; });
    return initialized_;
  }

  [[nodiscard]] bool attribute(HsaAgent agent, int attr, void* out) const noexcept {
    return reachable() && get_info_(agent, attr, out) == kHsaSuccess;
  }

  // Hand every GPU on the box access to one allocation.
  //
  // A device pool answers DISALLOWED_BY_DEFAULT -- kOnRequest -- which the
  // probe reports faithfully and nothing ever acted on, so a peer read fell
  // back to a bounce through host memory: measured on this box at 0.71 GB/s
  // against about a thousand local. The grant is per allocation and there is
  // no way to ask for it once, which is why it lives on the allocation path.
  //
  // Granting to every GPU rather than to a pool's members keeps this from
  // depending on a device set that does not exist yet when a buffer is made.
  // An agent that never touches the buffer costs a page table entry.
  [[nodiscard]] bool allow_peers(const std::vector<HsaAgent>& agents,
                                 void* ptr) const noexcept {
    if (allow_access_ == nullptr || ptr == nullptr || agents.size() < 2) {
      return false;
    }
    return allow_access_(static_cast<std::uint32_t>(agents.size()),
                         agents.data(), nullptr, ptr) == kHsaSuccess;
  }

  // The first CPU agent, which is the one host memory belongs to when a
  // transfer names where its source lives.
  [[nodiscard]] HsaAgent cpu_agent() const noexcept {
    HsaAgent found{};
    if (!reachable()) return found;
    struct Visit {
      const HsaRuntime* self;
      HsaAgent* out;
    } visit{this, &found};
    iterate_(
        [](HsaAgent agent, void* data) -> HsaStatus {
          auto* v = static_cast<Visit*>(data);
          if (v->out->handle != 0) return kHsaSuccess;
          int type = 0;
          if (!v->self->attribute(agent, kInfoDevice, &type)) return kHsaSuccess;
          if (type == kDeviceTypeCpu) *v->out = agent;
          return kHsaSuccess;
        },
        &visit);
    return found;
  }

  [[nodiscard]] bool can_dma() const noexcept {
    return async_copy_ != nullptr && signal_create_ != nullptr &&
           signal_destroy_ != nullptr && signal_wait_ != nullptr &&
           signal_store_ != nullptr;
  }

  // One DMA, waited on. This is the copy engine: the blit kernels the HAL uses
  // for buffer copies move the bytes with the shader cores instead, which on
  // this link measures about 8 GB/s against the DMA's forty-odd.
  [[nodiscard]] bool dma_copy(void* dst, HsaAgent dst_agent, const void* src,
                              HsaAgent src_agent, std::size_t bytes) const noexcept {
    if (!can_dma()) return false;
    if (!shared_signal_ready_) {
      if (signal_create_(1, 0, nullptr, &shared_signal_) != kHsaSuccess) {
        return false;
      }
      shared_signal_ready_ = true;
    } else {
      // Arm the signal again rather than making a new one.
      signal_store_(shared_signal_, 1);
    }
    const bool issued =
        async_copy_(dst, dst_agent, src, src_agent, bytes, 0, nullptr,
                    shared_signal_) == kHsaSuccess;
    if (issued) {
      // Spin rather than sleep: these completions are tens of microseconds and
      // an interrupt round trip is the same order as the transfer.
      constexpr int kConditionLt = 2;
      constexpr int kWaitActive = 1;
      (void)signal_wait_(shared_signal_, kConditionLt, 1, UINT64_MAX,
                         kWaitActive);
    }
    return issued;
  }

  // Pins a host range so the copy engine can reach it, returning the address
  // the agent should be given. Plain heap memory is pageable and no DMA engine
  // may touch it.
  [[nodiscard]] void* lock_host(void* ptr, std::size_t bytes,
                                HsaAgent* agents, int count) const noexcept {
    if (lock_ == nullptr) return nullptr;
    void* locked = nullptr;
    if (lock_(ptr, bytes, agents, count, &locked) != kHsaSuccess) return nullptr;
    return locked;
  }
  void unlock_host(void* ptr) const noexcept {
    if (unlock_ != nullptr) (void)unlock_(ptr);
  }

  // GPU agents only, in hsa_iterate_agents order. The CPU and any accelerator
  // that is not a GPU (this box also enumerates an NPU) are skipped, so the
  // index is a GPU index — which is what an hrx ordinal is meant to be too.
  [[nodiscard]] std::vector<HsaAgent> gpu_agents() const noexcept {
    std::vector<HsaAgent> agents;
    if (!reachable()) return agents;
    struct Visit {
      const HsaRuntime* self;
      std::vector<HsaAgent>* out;
    } visit{this, &agents};
    iterate_(
        [](HsaAgent agent, void* data) -> HsaStatus {
          auto* v = static_cast<Visit*>(data);
          int type = 0;
          if (!v->self->attribute(agent, kInfoDevice, &type)) return kHsaSuccess;
          if (type == kDeviceTypeGpu) v->out->push_back(agent);
          return kHsaSuccess;
        },
        &visit);
    return agents;
  }

  // This agent's coarse-grained global pool — the one device-local allocations
  // come from, and the one a peer would have to reach. Null handle when the
  // agent publishes none.
  [[nodiscard]] HsaPool coarse_global_pool(HsaAgent agent) const noexcept {
    constexpr int kPoolInfoSegment = 0;
    constexpr int kPoolInfoGlobalFlags = 1;
    constexpr std::uint32_t kSegmentGlobal = 0;
    constexpr std::uint32_t kGlobalFlagCoarseGrained = 4;
    if (iterate_pools_ == nullptr || pool_info_ == nullptr) return {};
    struct Visit {
      const HsaRuntime* self;
      HsaPool found;
    } visit{this, {}};
    iterate_pools_(
        agent,
        [](HsaPool pool, void* data) -> HsaStatus {
          auto* v = static_cast<Visit*>(data);
          if (v->found.handle != 0) return kHsaSuccess;
          std::uint32_t segment = 0;
          std::uint32_t flags = 0;
          if (v->self->pool_info_(pool, kPoolInfoSegment, &segment) != kHsaSuccess ||
              v->self->pool_info_(pool, kPoolInfoGlobalFlags, &flags) != kHsaSuccess) {
            return kHsaSuccess;
          }
          if (segment == kSegmentGlobal &&
              (flags & kGlobalFlagCoarseGrained) != 0) {
            v->found = pool;
          }
          return kHsaSuccess;
        },
        &visit);
    return visit.found;
  }

  // Whether `agent` may address `pool`, as the runtime that owns both answers
  // it (HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS). This is a real topology query:
  // ROCr resolves it through the KFD's peer table, unlike
  // hrx_device_can_access_peer, which compares the two devices' *types* and
  // therefore answers "yes" for any two GPUs whether a path exists or not.
  [[nodiscard]] PeerAccess pool_access(HsaAgent agent, HsaPool pool) const noexcept {
    constexpr int kAgentPoolInfoAccess = 0;
    if (agent_pool_info_ == nullptr || pool.handle == 0) {
      return PeerAccess::kUnknown;
    }
    int access = -1;
    if (agent_pool_info_(agent, pool, kAgentPoolInfoAccess, &access) !=
        kHsaSuccess) {
      return PeerAccess::kUnknown;
    }
    switch (access) {
      case 0: return PeerAccess::kNo;         // NEVER_ALLOWED
      case 1: return PeerAccess::kYes;        // ALLOWED_BY_DEFAULT
      case 2: return PeerAccess::kOnRequest;  // DISALLOWED_BY_DEFAULT
      default: return PeerAccess::kUnknown;
    }
  }

 private:
  using InitFn = HsaStatus (*)();
  using IterateFn = HsaStatus (*)(HsaStatus (*)(HsaAgent, void*), void*);
  using GetInfoFn = HsaStatus (*)(HsaAgent, int, void*);
  using IteratePoolsFn = HsaStatus (*)(HsaAgent,
                                       HsaStatus (*)(HsaPool, void*), void*);
  using PoolInfoFn = HsaStatus (*)(HsaPool, int, void*);
  using AgentPoolInfoFn = HsaStatus (*)(HsaAgent, HsaPool, int, void*);
  using AllowAccessFn = HsaStatus (*)(std::uint32_t, const HsaAgent*,
                                      const std::uint32_t*, const void*);
  using AsyncCopyFn = HsaStatus (*)(void*, HsaAgent, const void*, HsaAgent,
                                    std::size_t, std::uint32_t,
                                    const HsaSignal*, HsaSignal);
  using LockFn = HsaStatus (*)(void*, std::size_t, HsaAgent*, int, void**);
  using UnlockFn = HsaStatus (*)(void*);
  using SignalCreateFn = HsaStatus (*)(std::int64_t, std::uint32_t,
                                       const HsaAgent*, HsaSignal*);
  using SignalDestroyFn = HsaStatus (*)(HsaSignal);
  using SignalStoreFn = void (*)(HsaSignal, std::int64_t);
  using SignalWaitFn = std::int64_t (*)(HsaSignal, int, std::int64_t,
                                        std::uint64_t, int);

  void* lib_ = nullptr;
  InitFn init_ = nullptr;
  mutable std::once_flag once_;
  mutable bool initialized_ = false;
  IterateFn iterate_ = nullptr;
  GetInfoFn get_info_ = nullptr;
  IteratePoolsFn iterate_pools_ = nullptr;
  PoolInfoFn pool_info_ = nullptr;
  AgentPoolInfoFn agent_pool_info_ = nullptr;
  AllowAccessFn allow_access_ = nullptr;
  AsyncCopyFn async_copy_ = nullptr;
  LockFn lock_ = nullptr;
  UnlockFn unlock_ = nullptr;
  SignalCreateFn signal_create_ = nullptr;
  SignalDestroyFn signal_destroy_ = nullptr;
  SignalStoreFn signal_store_ = nullptr;
  // One completion signal, reused. Creating one is a driver object and costs
  // about 40 us -- at 1 MB that was two thirds of the transfer, and it is paid
  // per copy, so a model load pays it per tensor.
  mutable HsaSignal shared_signal_{};
  mutable bool shared_signal_ready_ = false;
  SignalWaitFn signal_wait_ = nullptr;
};

// The runtime and its GPU agents, resolved once. Every device allocation grants
// its memory to the peers, and neither the runtime handle nor the agent list
// changes after the process starts, so paying dlopen and an agent walk per
// buffer would be waste.
const HsaRuntime& shared_hsa() noexcept {
  static const HsaRuntime hsa;
  return hsa;
}

const std::vector<HsaAgent>& peer_agents() noexcept {
  static const std::vector<HsaAgent> agents = shared_hsa().gpu_agents();
  return agents;
}

// One attribute of the `ordinal`-th GPU agent. False when the runtime is not
// reachable, the index is past the agents it lists, or the agent will not
// answer.
bool agent_attribute(std::uint8_t ordinal, int attribute, void* out) noexcept {
  const HsaRuntime hsa;
  const std::vector<HsaAgent> agents = hsa.gpu_agents();
  if (ordinal >= agents.size()) return false;
  return hsa.attribute(agents[ordinal], attribute, out);
}

// A string attribute. hsa_agent_get_info writes the attribute's whole declared
// width, so the buffer is the widest string attribute read here (64) rather
// than the length of what comes back; empty when the agent will not answer.
[[maybe_unused]] std::string agent_string(const HsaRuntime& hsa, HsaAgent agent,
                                         int attribute) {
  char buffer[64] = {};
  if (!hsa.attribute(agent, attribute, buffer)) return {};
  buffer[sizeof(buffer) - 1] = '\0';
  return std::string(buffer);
}

// The KFD node id an hrx device name carries. IREE's AMDGPU driver builds the
// name as "<product> (Node <HSA_AGENT_INFO_NODE>)" (drivers/amdgpu/driver.c:188)
// and that node is the one exact identity both hrx and the HSA runtime can see:
// it pins an hrx ordinal to an agent even among boards of the same part, where
// matching the architecture cannot. -1 when the name carries none.
[[maybe_unused]] int node_in_name(std::string_view name) noexcept {
  constexpr std::string_view kOpen = "(Node ";
  const std::size_t open = name.rfind(kOpen);
  if (open == std::string_view::npos) return -1;
  const std::size_t start = open + kOpen.size();
  const std::size_t close = name.find(')', start);
  if (close == std::string_view::npos) return -1;
  int node = -1;
  const auto [ptr, ec] = std::from_chars(name.data() + start,
                                         name.data() + close, node);
  if (ec != std::errc{} || ptr != name.data() + close) return -1;
  return node;
}

// How well the `index`-th GPU agent can be shown to be the device hrx calls
// ordinal `index`.
//
// The two are independent numbering schemes over the same GPUs: hrx skips
// IREE's pseudo-device (one logical device standing for all of them) and
// renumbers what is left, while hsa_iterate_agents has its own order. Nothing
// in hrx_runtime.h publishes a bus id or a uuid to join them on, so the join is
// the node in the name, with the architecture as a weaker fallback — and an
// agent fact is only attributed to an ordinal when one of them holds.
enum class AgentMatch : std::uint8_t {
  kNo,        // not the same device, or nothing could be compared
  kArchOnly,  // same architecture, which identical boards also share
  kNode,      // same KFD node: this is that device
};

[[maybe_unused]] AgentMatch match_agent(const HsaRuntime& hsa, HsaAgent agent,
                                        std::string_view hrx_name,
                                        std::string_view hrx_arch) {
  constexpr int kInfoName = 0;
  constexpr int kInfoNode = 16;
  // For a GPU agent HSA's own name is the ISA name ("gfx1151"), which is what
  // hrx reports as the architecture.
  if (hrx_arch.empty() || agent_string(hsa, agent, kInfoName) != hrx_arch) {
    return AgentMatch::kNo;
  }
  const int named_node = node_in_name(hrx_name);
  if (named_node < 0) return AgentMatch::kArchOnly;
  std::uint32_t node = 0;
  if (!hsa.attribute(agent, kInfoNode, &node)) return AgentMatch::kArchOnly;
  return static_cast<int>(node) == named_node ? AgentMatch::kNode
                                             : AgentMatch::kNo;
}

// The agent's own answer to "how many queues may exist on you at once"
// (HSA_AGENT_INFO_QUEUES_MAX; 128 on gfx1151). hrx has no property for it.
// Returns 0 when the agent cannot be asked, which is not an error: the caller
// then bounds the stream count by the compute units alone.
std::uint32_t agent_queue_maximum(std::uint8_t ordinal) noexcept {
  constexpr int kInfoQueuesMax = 12;
  std::uint32_t queues = 0;
  if (!agent_attribute(ordinal, kInfoQueuesMax, &queues)) return 0;
  return queues;
}

#if LSE_HRX_LINKED
// Bytes free across every global pool this agent owns
// (HSA_AMD_AGENT_INFO_MEMORY_AVAIL, hsa_ext_amd.h). A live figure: it falls as
// any process on the box allocates and rises as they free. This is the same
// query hipMemGetInfo answers with, one layer further out than hrx.
bool agent_memory_available(std::uint8_t ordinal, std::uint64_t* out) noexcept {
  constexpr int kInfoMemoryAvail = 0xA015;
  return agent_attribute(ordinal, kInfoMemoryAvail, out);
}

// Ticks per second of this agent's own counter
// (HSA_AMD_AGENT_INFO_TIMESTAMP_FREQUENCY, hsa_ext_amd.h). Measured 99810000
// on gfx1151 — not the 100 MHz the number invites you to round to, and an
// order of magnitude off the 1000000000 the same runtime reports for the
// host-scope HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY. Dividing agent ticks by the
// system rate inflates every duration by 10.019x and still looks plausible,
// which is why the rate is asked for rather than written down.
//
// This is the domain the device's PM4 timestamp packets and its shader-visible
// steady counter both sample, so it is the denominator for any tick the device
// itself writes.
bool agent_timestamp_frequency(std::uint8_t ordinal, std::uint64_t* out) noexcept {
  constexpr int kInfoTimestampFrequency = 0xA016;
  return agent_attribute(ordinal, kInfoTimestampFrequency, out);
}
#endif

// How many logical queues this device's submission path can address, asked of
// the device rather than read off a header.
//
// The AMDGPU HAL resolves a queue affinity by ANDing the request with the
// queues the logical device actually created and taking the first set bit
// (queue_affinity.c); a bit past the end normalizes to empty and the call
// fails. So walking the bits with a barrier that signals a scratch timeline
// asks exactly the question that matters — "does a submission carrying this
// bit reach a queue" — and the answer is the count of bits that do. Each such
// queue is its own AQL ring (one hsa_queue_create per host queue).
//
// Cheap: N empty barriers once, at init. Returns 1 if the probe cannot run,
// which is the answer that costs nothing to be wrong about.
#if LSE_HRX_LINKED
std::uint32_t probe_queue_count(hrx_device_t device) noexcept {
  hrx_semaphore_t semaphore = nullptr;
  if (!hrx_status_is_ok(hrx_semaphore_create(device, 0, &semaphore))) return 1;

  std::uint64_t value = 0;
  std::uint32_t count = 0;
  // 64 is the width of the affinity mask; the loop stops at the first bit the
  // device refuses, so this is bounded by the queue count in practice.
  for (std::uint32_t bit = 0; bit < 64; ++bit) {
    std::uint64_t next = value + 1;
    hrx_semaphore_list_t signals = {};
    signals.semaphores = &semaphore;
    signals.values = &next;
    signals.count = 1;
    const hrx_status_t status = hrx_queue_barrier(
        device, static_cast<hrx_queue_affinity_t>(1) << bit,
        /*wait_semaphores=*/nullptr, &signals);
    if (!hrx_status_is_ok(status)) {
      hrx_status_ignore(status);
      break;
    }
    value = next;
    ++count;
  }
  if (value > 0) {
    hrx_status_ignore(hrx_semaphore_wait(semaphore, value, UINT64_MAX));
  }
  hrx_semaphore_release(semaphore);
  return count != 0 ? count : 1u;
}
#endif

// How many execution streams this device is worth handing out.
//
// One per addressable hardware queue, and no more. A stream past that shares a
// ring with another, so it cannot overlap with it — it would only cost the
// events the scheduler spends to spread onto it. `queues` is what the probe
// found; the agent's queue maximum and the compute units still bound it,
// because a queue with no CU to run on is a submission that returns nothing.
StreamCapabilities derive_stream_capabilities(const DeviceInfo& info,
                                              std::uint32_t queues) noexcept {
  StreamCapabilities caps;
  const std::uint32_t agent_max = agent_queue_maximum(info.ordinal);
  std::uint32_t n = queues != 0 ? queues : 1u;
  if (agent_max != 0) n = std::min(n, agent_max);
  if (info.compute_units != 0) {
    n = std::min(n, static_cast<std::uint32_t>(info.compute_units));
  }
  caps.stream_count = std::max(n, 1u);
  // Every stream this backend hands out has its own AQL ring, so every one of
  // them can run at the same time as the others. That is the probe's answer,
  // not a promise from the header, and rocprofv3 agrees: spreading a decode
  // step across two rings produced 1.73 ms of genuinely concurrent kernel time
  // per token on two distinct Queue_Ids, where the single-ring path produced
  // 0.000 ms.
  caps.concurrent_streams = caps.stream_count;
  caps.needs_explicit_events = true;
  // Still false, but for a new reason. The original reason is gone: since
  // hrx 2082d042 a stream's command buffer is recorded and submitted with the
  // stream's own queue affinity, so the batched path reaches every ring and a
  // launch costs the same wherever it goes. What remains is that the spread
  // path itself is not safe to enable: with spreading on, a single gfx1201
  // either emits corrupted logits (mojibake, a stop token as the first
  // prediction) or deadlocks after the first token — one correct token, then
  // the host parked on a stream drain forever — while the same build with
  // spreading off decodes correctly at 20.5 tok/s. The hole is in the
  // cross-stream ordering somewhere between plan_streams' dependency edges
  // and the event machinery, and until it is found and tested, spreading is a
  // correctness bug and not a performance question. (It also never paid on
  // memory-bound decode: 93.7 -> 80.9 tok/s measured on gfx1151.)
  //
  // Pinned plans — a spanning device placing by member — do not read this
  // flag; their ordering is emitted regardless (see plan_streams).
  // LSE_SPREAD=1 is the experiment gate for hunting the ordering hole; it is
  // not a supported mode until the hole is found.
  caps.uniform_launch_cost = std::getenv("LSE_SPREAD") != nullptr;
  // A dispatch is a grid, and a grid cannot be cut. Flips when a kernel
  // declares work items instead; nothing else about the seam changes.
  caps.splittable_work = false;
  return caps;
}

#if LSE_HRX_LINKED
// Brings the GPU accelerator up once for the whole process, and leaves it up.
//
// It cannot belong to a backend instance. hrx_gpu_initialize is global, is not
// refcounted, and returns ALREADY_EXISTS on a second call — which is why two
// HrxBackend objects could not both init before this existed — while
// hrx_gpu_shutdown is global too and tears every device down for every holder
// with no owner check. Measured on this box: after one instance's shutdown ran
// the global teardown, a surviving instance still answered property and
// free-memory queries with plausible numbers and then failed inside the
// allocator ("shared proactor pool must be initialized"), which is the worst
// failure shape available — late, partial, and downstream of the report.
//
// ALREADY_EXISTS therefore counts as up, and taking it down is deliberately
// left to process exit: hrx_gpu_shutdown is one-way, hsa_init fails
// RESOURCE_EXHAUSTED on the way back up, so an early teardown would buy a tidy
// exit at the price of a process that can never bind a device again — which is
// exactly what enumerating devices before binding one would otherwise cause.
std::atomic<bool> g_accelerator_up{false};

// Registered during this translation unit's dynamic initialization — before
// main, and so before any object that can own a backend exists. Exit handlers
// run in reverse registration order, which is what puts the global teardown
// AFTER the last HrxBackend has released its streams and device.
//
// Registering it lazily on first bring-up does NOT achieve that, however early
// the bring-up is: the first one happens from inside whatever owns the backends
// (place::Devices, a Scheduler, a test fixture), whose own handler is therefore
// registered first and runs last — leaving hrx_gpu_shutdown to run while those
// backends are still holding streams and a device reference.
[[maybe_unused]] const bool kTeardownRegistered = [] {
  std::atexit([] {
    if (g_accelerator_up.load(std::memory_order_acquire)) {
      hrx_status_ignore(hrx_gpu_shutdown());
    }
  });
  return true;
}();

Status accelerator_up() {
  static std::mutex mu;
  const std::lock_guard lock(mu);
  if (g_accelerator_up.load(std::memory_order_relaxed)) return OkStatus();
  // A pool that named several GPUs gets one device spanning them, which is
  // what gives them a single allocator and therefore one address space: a
  // buffer any of them allocates is bindable by a dispatch on any other. Asked
  // for here because bringing the accelerator up is once per process and the
  // topology is fixed at that moment.
  const std::string group = backend::requested_device_group("hrx");
  const hrx_status_t status =
      group.empty() ? hrx_gpu_initialize(0)
                    : hrx_gpu_initialize_over(group.c_str(), 0);
  if (hrx_status_is_ok(status)) {
    g_accelerator_up.store(true, std::memory_order_release);
    return OkStatus();
  }
  if (hrx_status_code(status) == HRX_STATUS_ALREADY_EXISTS) {
    hrx_status_ignore(status);
    g_accelerator_up.store(true, std::memory_order_release);
    return OkStatus();
  }
  return from_hrx(status, "hrx_gpu_initialize");
}
#endif

}  // namespace

bool HrxBackend::available() noexcept {
#if LSE_HRX_LINKED
  return true;
#else
  return false;
#endif
}

HrxBackend::~HrxBackend() { shutdown_impl(); }

Result<std::vector<DeviceDescriptor>> HrxBackend::enumerate_devices() {
#if !LSE_HRX_LINKED
  return LSE_ERROR(kUnimplemented,
                   "this build was configured with HRX headers but libhrx was "
                   "not linked; build hrx-system and reconfigure with "
                   "-DLSE_HRX_ROOT=<install prefix>");
#else
  LSE_RETURN_IF_ERROR(accelerator_up());

  int count = 0;
  LSE_RETURN_IF_ERROR(
      from_hrx(hrx_gpu_device_count(&count), "hrx_gpu_device_count"));

  // HSA attribute ids, from hsa.h / hsa_ext_amd.h. Declared here for the same
  // reason the entry points are: those headers are not a build input.
  constexpr int kInfoWavefrontSize = 6;
  constexpr int kInfoWorkgroupMaxSize = 8;
  constexpr int kInfoComputeUnitCount = 0xA002;
  constexpr int kInfoBdfId = 0xA006;
  constexpr int kInfoDomain = 0xA00F;
  constexpr int kInfoUuid = 0xA011;
  constexpr int kInfoMemoryAvail = 0xA015;

  const HsaRuntime hsa;
  const std::vector<HsaAgent> agents = hsa.gpu_agents();
  std::vector<HsaPool> pools(agents.size());
  for (std::size_t i = 0; i < agents.size(); ++i) {
    pools[i] = hsa.coarse_global_pool(agents[i]);
  }

  std::vector<DeviceDescriptor> out;
  out.reserve(static_cast<std::size_t>(count));
  for (int i = 0; i < count; ++i) {
    hrx_device_t device = nullptr;
    LSE_RETURN_IF_ERROR(
        from_hrx(hrx_gpu_device_get(i, &device), "hrx_gpu_device_get"));

    DeviceDescriptor d;
    d.backend = std::string(kName);
    d.ordinal = i;
    const auto decline = [&d](std::string_view text) {
      if (!d.declined.empty()) d.declined += "; ";
      d.declined += text;
    };

    const std::string name = query_string_property(device, HRX_DEVICE_PROPERTY_NAME);
    if (!name.empty()) d.product = DeviceFact<std::string>::queried(name);
    const std::string arch =
        query_string_property(device, HRX_DEVICE_PROPERTY_ARCHITECTURE);
    if (!arch.empty()) d.arch = DeviceFact<std::string>::queried(arch);

    std::uint64_t total = 0;
    if (query_property(device, HRX_DEVICE_PROPERTY_TOTAL_MEMORY, &total).ok() &&
        total != 0) {
      d.total_memory =
          DeviceFact<std::size_t>::queried(static_cast<std::size_t>(total));
    }
    std::uint32_t queues = 0;
    if (query_property(device, HRX_DEVICE_PROPERTY_QUEUE_COUNT, &queues).ok() &&
        queues != 0) {
      d.queue_count = DeviceFact<std::uint32_t>::queried(queues);
    }
    // hrx answers these two with a literal 0 ("not available from local-task
    // driver"), which is an absent answer and not a device with no CUs.
    std::uint32_t cus = 0;
    if (query_property(device, HRX_DEVICE_PROPERTY_COMPUTE_UNITS, &cus).ok() &&
        cus != 0) {
      d.compute_units = DeviceFact<std::uint32_t>::queried(cus);
    }
    std::uint32_t threads = 0;
    if (query_property(device, HRX_DEVICE_PROPERTY_MAX_WORKGROUP_SIZE, &threads)
            .ok() &&
        threads != 0) {
      d.max_threads_per_workgroup = DeviceFact<std::uint32_t>::queried(threads);
    }

    // Free memory: the HAL first, so that the day it publishes an availability
    // observation that is the figure used. The AMDGPU HAL populates its
    // observation from the device spec and so answers UNAVAILABLE today.
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    const hrx_status_t mem =
        hrx_device_memory_info(device, &free_bytes, &total_bytes);
    if (hrx_status_is_ok(mem)) {
      d.free_memory = DeviceFact<std::size_t>::queried(free_bytes);
    } else {
      hrx_status_ignore(mem);
    }

    // One row per device of this backend, kUnknown until something answers.
    // The diagonal falls back to the one fact about reach that needs no query.
    const std::size_t index = static_cast<std::size_t>(i);
    d.peers.assign(static_cast<std::size_t>(count), PeerAccess::kUnknown);
    d.peers[index] = PeerAccess::kSelf;

    const AgentMatch matched = index < agents.size()
                                   ? match_agent(hsa, agents[index], name, arch)
                                   : AgentMatch::kNo;
    if (matched == AgentMatch::kNo) {
      decline("this device could not be matched to one of the " +
              std::to_string(agents.size()) +
              " GPU agent(s) the HSA runtime lists, so its uuid, pci path, "
              "free memory, occupancy limits and peer reach are unknown here "
              "rather than read off whichever agent shares its index");
    } else if (matched == AgentMatch::kArchOnly) {
      decline("this device's name carries no KFD node, so its HSA agent was "
              "matched on architecture alone: two boards of the same part "
              "could be permuted between hrx ordinals and agent order");
    }

    if (matched != AgentMatch::kNo) {
      const HsaAgent agent = agents[index];

      if (!d.free_memory.known()) {
        std::uint64_t available = 0;
        if (hsa.attribute(agent, kInfoMemoryAvail, &available)) {
          d.free_memory = DeviceFact<std::size_t>::queried(
              static_cast<std::size_t>(available));
        } else {
          decline("neither this device's HAL nor its agent publishes an "
                  "available-memory figure");
        }
      }
      if (!d.compute_units.known() &&
          hsa.attribute(agent, kInfoComputeUnitCount, &cus) && cus != 0) {
        d.compute_units = DeviceFact<std::uint32_t>::queried(cus);
      }
      if (!d.max_threads_per_workgroup.known() &&
          hsa.attribute(agent, kInfoWorkgroupMaxSize, &threads) &&
          threads != 0) {
        d.max_threads_per_workgroup = DeviceFact<std::uint32_t>::queried(threads);
      }
      std::uint32_t wavefront = 0;
      if (hsa.attribute(agent, kInfoWavefrontSize, &wavefront) &&
          wavefront != 0) {
        d.wavefront_size = DeviceFact<std::uint32_t>::queried(wavefront);
      }

      // "GPU-XX" is what ROCr returns for an agent with no uuid, documented as
      // such in hsa_ext_amd.h. Printing it would be printing a placeholder as
      // an identity.
      const std::string uuid = agent_string(hsa, agent, kInfoUuid);
      if (!uuid.empty() && !uuid.ends_with("-XX")) {
        d.uuid = DeviceFact<std::string>::queried(uuid);
      } else {
        decline("this agent publishes no uuid (ROCr's \"" +
                (uuid.empty() ? std::string("(no answer)") : uuid) +
                "\" placeholder), and hrx exposes no device uuid of its own "
                "even though IREE reads one to build the device path");
      }

      std::uint32_t bdf = 0;
      std::uint32_t domain = 0;
      if (hsa.attribute(agent, kInfoBdfId, &bdf) && bdf != 0 &&
          hsa.attribute(agent, kInfoDomain, &domain)) {
        // BDFID packs bus:device.function; the domain is a separate attribute
        // and the two together are the whole physical location.
        char path[32] = {};
        std::snprintf(path, sizeof(path), "%04x:%02x:%02x.%u", domain,
                      (bdf >> 8) & 0xff, (bdf >> 3) & 0x1f, bdf & 0x7);
        d.pci_path = DeviceFact<std::string>::queried(std::string(path));
      } else {
        decline("this agent publishes no pci location");
      }

      for (int j = 0; j < count; ++j) {
        const std::size_t peer = static_cast<std::size_t>(j);
        const PeerAccess reach = peer < pools.size()
                                     ? hsa.pool_access(agent, pools[peer])
                                     : PeerAccess::kUnknown;
        if (reach != PeerAccess::kUnknown) d.peers[peer] = reach;
      }
      if (std::find(d.peers.begin(), d.peers.end(), PeerAccess::kUnknown) !=
          d.peers.end()) {
        decline("peer reach is read from the HSA agents' pool access, because "
                "hrx_device_can_access_peer compares the two devices' types "
                "instead of querying a path and would answer yes for any two "
                "GPUs; it is unknown for a peer whose agent or pool this could "
                "not reach");
      }
    }

    // The tables fill what no runtime here answers. They are declared, not
    // measured, and say so: an ISA row is fixed by the architecture, a board
    // row is a spec sheet.
    const FamilyIsa* isa = family_isa(arch_family(arch));
    const BoardFallback* board = board_fallback(arch);
    if (!d.compute_units.known() && board != nullptr) {
      d.compute_units = DeviceFact<std::uint32_t>::declared(board->compute_units);
    }
    if (!d.max_threads_per_workgroup.known() && isa != nullptr) {
      d.max_threads_per_workgroup =
          DeviceFact<std::uint32_t>::declared(isa->max_threads_per_workgroup);
    }
    if (!d.wavefront_size.known() && isa != nullptr) {
      d.wavefront_size = DeviceFact<std::uint32_t>::declared(isa->wavefront_size);
    }
    // hrx declares HRX_DEVICE_PROPERTY_MAX_SHARED_MEMORY and implements no case
    // for it, and HSA publishes no per-workgroup LDS attribute, so the ISA row
    // is the only source there is.
    if (isa != nullptr) {
      d.lds_bytes_per_workgroup =
          DeviceFact<std::uint32_t>::declared(isa->lds_bytes_per_workgroup);
    }
    // Nothing queryable states whether host and device share one physical pool,
    // and the difference is a behavioural branch, so it comes from the board
    // row or it is unknown.
    if (board != nullptr) {
      d.unified_memory = DeviceFact<bool>::declared(board->unified_memory);
    }

    out.push_back(std::move(d));
  }
  return out;
#endif
}

#if LSE_HRX_LINKED
// The GPU agent's own PCI location, "dddd:bb:dd.f" — the same join enumerate
// devices makes (BDFID packs bus:device.function, the domain is a separate
// attribute). Empty when the agent publishes no location.
namespace {
std::string agent_pci_path(const HsaRuntime& hsa, HsaAgent agent) {
  constexpr int kBdfId = 0xA006;  // HSA_AMD_AGENT_INFO_BDFID
  constexpr int kDomain = 0xA00F;  // HSA_AMD_AGENT_INFO_DOMAIN
  std::uint32_t bdf = 0;
  std::uint32_t domain = 0;
  if (!hsa.attribute(agent, kBdfId, &bdf) || bdf == 0) return {};
  if (!hsa.attribute(agent, kDomain, &domain)) return {};
  char path[32] = {};
  std::snprintf(path, sizeof(path), "%04x:%02x:%02x.%u", domain,
                (bdf >> 8) & 0xff, (bdf >> 3) & 0x1f, bdf & 0x7);
  return path;
}

std::string agent_uuid(const HsaRuntime& hsa, HsaAgent agent) {
  constexpr int kUuid = 0xA011;  // HSA_AMD_AGENT_INFO_UUID
  return agent_string(hsa, agent, kUuid);
}

bool ieq(std::string_view a, std::string_view b) {
  if (a.size() != b.size()) return false;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const char la = static_cast<char>(std::tolower(static_cast<unsigned char>(a[i])));
    const char lb = static_cast<char>(std::tolower(static_cast<unsigned char>(b[i])));
    if (la != lb) return false;
  }
  return true;
}
// The hex of a UUID, no decoration: "GPU-abc", "0xabc" and "abc" are the same
// identity and the comparison is on the digits, not the frame around them.
std::string uuid_digits(std::string_view text) {
  std::string out;
  for (const char c : text) {
    if (std::isxdigit(static_cast<unsigned char>(c)) != 0) {
      out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
  }
  return out;
}
}  // namespace

std::optional<int> HrxBackend::resolve_stable_ref(std::string_view pci,
                                                  std::string_view uuid) {
  if (pci.empty() && uuid.empty()) return std::nullopt;
  // A fresh runtime, forced to load if hrx has not brought one up yet — this
  // is the point of the resolver, it runs before hrx_gpu_initialize — and
  // hsa_init so the enumeration is not empty. It reads agent attributes only:
  // no accelerator, no allocation, nothing that could fix a topology.
  const HsaRuntime hsa(true);
  if (!hsa.reachable() || !hsa.ensure_initialized()) return std::nullopt;
  const std::vector<HsaAgent> agents = hsa.gpu_agents();
  for (std::size_t i = 0; i < agents.size(); ++i) {
    if (!pci.empty() && ieq(agent_pci_path(hsa, agents[i]), pci)) {
      return static_cast<int>(i);
    }
    if (!uuid.empty()) {
      const std::string u = agent_uuid(hsa, agents[i]);
      if (!u.empty() && !u.ends_with("-XX") &&
          uuid_digits(u) == uuid_digits(uuid)) {
        return static_cast<int>(i);
      }
    }
  }
  return std::nullopt;
}
#endif  // LSE_HRX_LINKED

Status HrxBackend::init_impl(int device_ordinal) {
#if !LSE_HRX_LINKED
  (void)device_ordinal;
  return LSE_ERROR(kUnimplemented,
                   "this build was configured with HRX headers but libhrx was "
                   "not linked; build hrx-system and reconfigure with "
                   "-DLSE_HRX_ROOT=<install prefix>");
#else
  // Which GPU this is in hsa_iterate_agents order, which is the order
  // peer_agents() uses and therefore how the copy engine is told which device
  // it is talking to.
  gpu_ordinal_ = device_ordinal;
  LSE_RETURN_IF_ERROR(accelerator_up());

  // A pool that asked for a spanning device gets ONE hrx device holding every
  // GPU it named. The ordinal arriving here names a member of that group; it
  // stays as this instance's HSA agent -- the copy engine must speak for a GPU
  // the pool actually holds, and aiming it at whatever card enumerates first
  // hung every weight upload on a fence a foreign engine never signalled --
  // while the hrx device index collapses to zero, because the group IS device
  // zero once the accelerator comes up spanning.
  const std::string group = backend::requested_device_group("hrx");
  if (!group.empty()) {
    physical_count_ = 1;
    for (char c : group) physical_count_ += (c == ',') ? 1 : 0;
    device_ordinal = 0;
  }

  int count = 0;
  LSE_RETURN_IF_ERROR(from_hrx(hrx_gpu_device_count(&count), "hrx_gpu_device_count"));
  if (device_ordinal < 0 || device_ordinal >= count) {
    return LSE_ERROR(kInvalidArgument, "device ordinal ",
                     std::to_string(device_ordinal), " out of range; ",
                     std::to_string(count), " HRX device(s) present");
  }

  hrx_device_t device = nullptr;
  LSE_RETURN_IF_ERROR(
      from_hrx(hrx_gpu_device_get(device_ordinal, &device), "hrx_gpu_device_get"));
  // device_get hands back a borrowed pointer into hrx's static device array and
  // does not retain, while shutdown_impl releases — so the reference this
  // instance is about to drop has to be taken here. Without it N instances on
  // one ordinal are N decrements against a device none of them retained, and
  // the only thing hiding that today is that streams and buffers retain it too.
  hrx_device_retain(device);
  device_ = device;

  // Borrowed reference, valid for the device's lifetime — do not release.
  allocator_ = hrx_device_allocator(device);

  // --- device info ---
  info_ = DeviceInfo{};
  info_.ordinal = static_cast<std::uint8_t>(device_ordinal);
  info_.name = query_string_property(device, HRX_DEVICE_PROPERTY_NAME);
  info_.arch = query_string_property(device, HRX_DEVICE_PROPERTY_ARCHITECTURE);

  std::uint64_t total_memory = 0;
  if (query_property(device, HRX_DEVICE_PROPERTY_TOTAL_MEMORY, &total_memory).ok()) {
    info_.total_memory = static_cast<std::size_t>(total_memory);
  }
  std::uint32_t u32 = 0;
  if (query_property(device, HRX_DEVICE_PROPERTY_COMPUTE_UNITS, &u32).ok()) {
    info_.compute_units = static_cast<std::uint16_t>(u32);
  }
  if (query_property(device, HRX_DEVICE_PROPERTY_MAX_WORKGROUP_SIZE, &u32).ok()) {
    info_.max_threads_per_workgroup = static_cast<std::uint16_t>(u32);
  }
  if (query_property(device, HRX_DEVICE_PROPERTY_WARP_SIZE, &u32).ok()) {
    info_.wavefront_size = static_cast<std::uint16_t>(u32);
  }
  if (query_property(device, HRX_DEVICE_PROPERTY_MAX_SHARED_MEMORY, &u32).ok()) {
    info_.lds_bytes_per_workgroup = u32;
    // LSE_LDS_BUDGET=<bytes>: cap the budget the emitters plan against, so a
    // smaller part's phase declines (and their host fallbacks) reproduce on a
    // bigger one. Diagnostic; never raises the real limit.
    if (const char* cap = std::getenv("LSE_LDS_BUDGET")) {
      const auto want = static_cast<std::uint32_t>(std::atoi(cap));
      if (want != 0 && want < info_.lds_bytes_per_workgroup) {
        info_.lds_bytes_per_workgroup = want;
      }
    }
  }
  if (query_property(device, HRX_DEVICE_PROPERTY_CLOCK_RATE, &u32).ok()) {
    amd_.clock_khz = u32;
  }
  info_.ordinal = static_cast<std::uint8_t>(device_ordinal);

  // HRX values already on info_/amd_ stay. Tables fill only zeros.
  apply_arch_defaults(info_, amd_);
  // Capacity facts, after the runtime's own answers are in place so a live
  // query can stand in for a table row. HRX has no kernel-resource or
  // register-file query of its own — confirmed against its public runtime
  // header — so the compiler's target table is the only source for those.
  info_.arch_facts = arch_facts_for(info_);
  info_.extension_id = AmdDeviceInfo::kExtensionId;
  info_.extension = &amd_;

  flush_interval_ = 16;
  if (const char* env = std::getenv("LSE_FLUSH_INTERVAL");
      env != nullptr && env[0] != '\0') {
    char* end = nullptr;
    const long v = std::strtol(env, &end, 10);
    if (end != env && *end == '\0' && v >= 0) {
      flush_interval_ = static_cast<std::uint32_t>(v);
    }
  }

  queue_count_ = probe_queue_count(device);
  stream_caps_ = derive_stream_capabilities(info_, queue_count_);
  if (physical_count_ > 1 && stream_caps_.stream_count < physical_count_) {
    // Placement is by stream on a spanning device, so there has to be one
    // per GPU.
    stream_caps_.stream_count = physical_count_;
  }
  streams_.assign(stream_caps_.stream_count, nullptr);
  unflushed_launches_.assign(stream_caps_.stream_count, 0);
  stream_affinity_.resize(stream_caps_.stream_count);
  // The logical device's queues are flattened over its physical devices in
  // order, so slot s owns [s*per, (s+1)*per). Outside a spanning device there
  // is one slot and this is the whole queue set, as before.
  const std::uint32_t per =
      physical_count_ > 1 ? std::max(1u, queue_count_ / physical_count_)
                          : queue_count_;
  for (std::uint32_t i = 0; i < stream_caps_.stream_count; ++i) {
    const std::uint32_t slot =
        physical_count_ > 1 ? (i % physical_count_) : 0;
    const std::uint32_t within = physical_count_ > 1 ? 0 : (i % per);
    stream_affinity_[i] = static_cast<std::uint64_t>(1) << (slot * per + within);
  }
  // Stream 0 exists from the start: every path that does not name a stream
  // uses it, including the stream-ordered allocator.
  LSE_RETURN_IF_ERROR(stream_at(0).status());

  initialized_ = true;
  return OkStatus();
#endif
}

Result<void*> HrxBackend::stream_at(std::uint32_t index) {
#if !LSE_HRX_LINKED
  (void)index;
  return LSE_ERROR(kUnimplemented, "libhrx not linked");
#else
  if (index >= streams_.size()) {
    return LSE_ERROR(kOutOfRange, "stream ", std::to_string(index),
                     " past the ", std::to_string(streams_.size()),
                     " this device offers");
  }
  if (streams_[index] != nullptr) return streams_[index];
  hrx_stream_t stream = nullptr;
  LSE_RETURN_IF_ERROR(from_hrx(
      hrx_stream_create_on_queue(static_cast<hrx_device_t>(device_), 0,
                                 stream_affinity_[index], &stream),
      "hrx_stream_create_on_queue"));
  streams_[index] = stream;
  return streams_[index];
#endif
}

#if LSE_HRX_LINKED
Status HrxBackend::flush_stream(std::uint32_t index) {
  if (index >= streams_.size() || streams_[index] == nullptr) return OkStatus();
  unflushed_launches_[index] = 0;
  return from_hrx(hrx_stream_flush(static_cast<hrx_stream_t>(streams_[index])),
                  "hrx_stream_flush");
}

#else
Status HrxBackend::flush_stream(std::uint32_t) {
  return LSE_ERROR(kUnimplemented, "libhrx not linked");
}
#endif

void HrxBackend::adopt(DeviceBuffer& buf, std::uint64_t handle,
                       std::size_t bytes) {
  buf.handle = handle;
  buf.size_bytes = bytes;
  buf.storage = std::shared_ptr<void>(
      reinterpret_cast<void*>(handle), [this](void* p) {
        release_buffer(reinterpret_cast<std::uint64_t>(p));
      });
}

void HrxBackend::release_buffer(std::uint64_t handle) noexcept {
#if LSE_HRX_LINKED
  if (handle == 0) return;
  // Safe against in-flight AND still-unflushed dispatches: hrx command
  // buffers are ONE_SHOT (not UNRETAINED), so recording a dispatch inserts
  // its buffers into the CB's resource set, which keeps the hal buffer (and
  // its backing pool) alive until the CB itself retires. This release only
  // drops our reference.
  hrx_buffer_release(reinterpret_cast<hrx_buffer_t>(handle));
#else
  (void)handle;
#endif
}

void HrxBackend::shutdown_impl() noexcept {
#if LSE_HRX_LINKED
  unregister_hrx_residency(device_index());
  {
    // Whatever retired before now is freed here; whatever is still held by a
    // live StreamEvent frees itself when its last copy drops (backend_alive
    // false routes the deleter straight to hrx_event_release).
    const std::lock_guard lock(graveyard_->mu);
    graveyard_->backend_alive = false;
    for (void* e : graveyard_->retired) {
      hrx_event_release(static_cast<hrx_event_t>(e));
    }
    graveyard_->retired.clear();
  }
  // Before the device: each executable retains it.
  for (void* e : loaded_executables_) {
    hrx_executable_release(static_cast<hrx_executable_t>(e));
  }
  loaded_executables_.clear();
  for (void*& s : streams_) {
    if (s == nullptr) continue;
    hrx_stream_release(static_cast<hrx_stream_t>(s));
    s = nullptr;
  }
  streams_.clear();
  unflushed_launches_.clear();
  if (device_ != nullptr) {
    hrx_device_release(static_cast<hrx_device_t>(device_));
    device_ = nullptr;
  }
  allocator_ = nullptr;
  // The accelerator is not this instance's to shut down: it is process-scoped
  // and every other instance is still using it (see accelerator_up).
  initialized_ = false;
#endif
}

Result<DeviceBuffer> HrxBackend::allocate_impl(std::size_t bytes,
                                               MemoryClass cls, Stream stream) {
#if !LSE_HRX_LINKED
  (void)bytes;
  (void)cls;
  return LSE_ERROR(kUnimplemented, "libhrx not linked");
#else
  if (!initialized_) return LSE_ERROR(kInternal, "hrx backend not initialized");
  if (bytes == 0) return LSE_ERROR(kInvalidArgument, "zero-size allocation");
  register_hrx_residency(device_index());

  const bool staging = cls == MemoryClass::kStaging;
  const hrx_buffer_usage_t usage =
      HRX_BUFFER_USAGE_DISPATCH_STORAGE | HRX_BUFFER_USAGE_TRANSFER |
      (staging ? (HRX_BUFFER_USAGE_MAPPING_SCOPED |
                  HRX_BUFFER_USAGE_MAPPING_PERSISTENT)
               : 0);

  hrx_buffer_t buffer = nullptr;
  if (!staging && physical_count_ > 1) {
    // Placement, through the allocator that takes one. The stream names the
    // member whose card these bytes belong on; the stream-ordered path below
    // carries no affinity at all, and a spanning device resolves "anywhere"
    // to host memory every GPU can reach -- a 27B "loaded" into system RAM
    // with every card's VRAM untouched, which amdgpu_top showed and the KFD
    // per-process counter did not.
    if (stream.index >= stream_affinity_.size()) {
      return LSE_ERROR(kOutOfRange, "allocation names stream ",
                       std::to_string(stream.index), " of ",
                       std::to_string(stream_affinity_.size()));
    }
    hrx_buffer_params_t params = {};
    params.type = HRX_MEMORY_TYPE_DEVICE_LOCAL;
    params.access = HRX_MEMORY_ACCESS_ALL;
    params.usage = usage;
    params.queue_affinity =
        static_cast<hrx_queue_affinity_t>(stream_affinity_[stream.index]);
    LSE_RETURN_IF_ERROR(
        from_hrx(hrx_allocator_allocate_buffer(
                     static_cast<hrx_allocator_t>(allocator_), params, bytes,
                     &buffer),
                 "hrx_allocator_allocate_buffer (device)"));
  } else if (!staging) {
    // Stream-ordered like hipMallocAsync, but stronger: hrx_buffer_allocate
    // flushes the stream and then BLOCKS until the backing is committed
    // (libhrx buffer.c waits the alloca's semaphore), so an allocation
    // mid-batch submits the open command buffer for us, and the buffer is
    // safe to touch from any stream the moment this returns.
    //
    // The other streams are deliberately not flushed. A one-shot command
    // buffer inserts every buffer it dispatches against into its own resource
    // set at *record* time, so memory another stream still references cannot
    // be handed back to the pool here, flushed or not.
    auto stream = stream_at(0);
    if (!stream.ok()) return stream.status();
    LSE_RETURN_IF_ERROR(from_hrx(
        hrx_buffer_allocate(static_cast<hrx_stream_t>(*stream), bytes,
                            HRX_MEMORY_TYPE_DEVICE_LOCAL, usage, &buffer),
        "hrx_buffer_allocate"));
    unflushed_launches_[0] = 0;
  } else {
    // Staging has to stay mapped; the stream-ordered path is device-local.
    hrx_buffer_params_t params = {};
    params.type = HRX_MEMORY_TYPE_HOST_VISIBLE | HRX_MEMORY_TYPE_HOST_COHERENT |
                  HRX_MEMORY_TYPE_DEVICE_VISIBLE;
    params.access = HRX_MEMORY_ACCESS_ALL;
    params.usage = usage;
    params.queue_affinity = 0;
    LSE_RETURN_IF_ERROR(
        from_hrx(hrx_allocator_allocate_buffer(
                     static_cast<hrx_allocator_t>(allocator_), params, bytes,
                     &buffer),
                 "hrx_allocator_allocate_buffer"));
  }

  DeviceBuffer out;
  const auto handle = reinterpret_cast<std::uint64_t>(buffer);
  out.size_bytes = bytes;

  // Device-local memory is granted to the other GPUs now, while the pointer is
  // in hand. Without it a peer read is not refused, it is quietly serviced
  // through host memory, and the only sign is a transfer rate three orders
  // below the local one.
  if (!staging) {
    void* device_ptr = nullptr;
    if (hrx_status_is_ok(hrx_buffer_get_device_ptr(buffer, &device_ptr))) {
      (void)shared_hsa().allow_peers(peer_agents(), device_ptr);
    }
  }

  // A staging buffer stays mapped for its whole lifetime; a device-local one
  // has no host address at all, and reaching it means copy_h2d/copy_d2h.
  if (staging) {
    void* mapped = nullptr;
    const hrx_status_t mapped_status = hrx_buffer_map(
        buffer, HRX_MAP_READ | HRX_MAP_WRITE, 0, bytes, &mapped);
    if (!hrx_status_is_ok(mapped_status)) {
      hrx_buffer_release(buffer);
      return from_hrx(mapped_status, "hrx_buffer_map");
    }
    out.ptr = mapped;
    out.handle = handle;
    // Staging is not pooled: the mapping has to die with the buffer.
    out.storage = std::shared_ptr<void>(
        reinterpret_cast<void*>(handle), [](void* p) {
#if LSE_HRX_LINKED
          auto* b = reinterpret_cast<hrx_buffer_t>(p);
          hrx_status_ignore(hrx_buffer_unmap(b));
          hrx_buffer_release(b);
#else
          (void)p;
#endif
        });
    return out;
  }

  adopt(out, handle, bytes);
  return out;
#endif
}

void HrxBackend::deallocate_impl(DeviceBuffer& buf) noexcept {
  buf.storage.reset();
  buf.handle = 0;
  buf.ptr = nullptr;
  buf.size_bytes = 0;
}

Result<std::size_t> HrxBackend::sample_free_memory_impl() const {
#if !LSE_HRX_LINKED
  return LSE_ERROR(kUnimplemented, "libhrx not linked");
#else
  if (!initialized_) return LSE_ERROR(kInternal, "hrx backend not initialized");
  // hrx_device_memory_info rejects a null total_bytes, so the declared total
  // is read and dropped — info_.total_memory already carries it, and it is a
  // different quantity from the one being sampled here.
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  const hrx_status_t status = hrx_device_memory_info(
      static_cast<hrx_device_t>(device_), &free_bytes, &total_bytes);
  if (hrx_status_is_ok(status)) return free_bytes;
  // UNAVAILABLE is this HAL declining to publish an availability observation,
  // which is an absent query and not a broken device — from_hrx would fold it
  // into kDeviceError and a caller that stops on device errors would turn an
  // unanswerable question into a failed run. It is also the answer on every
  // device LSE runs on today: the AMDGPU HAL populates its memory observation
  // from the device spec and therefore sets the total only
  // (iree/hal/drivers/amdgpu/logical_device.c). The agent underneath still
  // knows, so ask it before giving up; hrx stays first so that the day the HAL
  // publishes the figure it is the one that is used.
  if (hrx_status_code(status) != HRX_STATUS_UNAVAILABLE) {
    return from_hrx(status, "hrx_device_memory_info");
  }
  hrx_status_ignore(status);
  std::uint64_t available = 0;
  if (!agent_memory_available(info_.ordinal, &available)) {
    return LSE_ERROR(kUnimplemented,
                     "neither this device's HAL nor its agent publishes an "
                     "available-memory figure");
  }
  return static_cast<std::size_t>(available);
#endif
}

Result<DeviceClock> HrxBackend::device_clock_impl() const {
#if !LSE_HRX_LINKED
  return LSE_ERROR(kUnimplemented, "libhrx not linked");
#else
  if (!initialized_) return LSE_ERROR(kInternal, "hrx backend not initialized");
  // hrx has no property for this. HRX_DEVICE_PROPERTY_CLOCK_RATE is declared in
  // hrx_runtime.h but no case implements it in device.c, and it would be the
  // engine clock in MHz rather than a tick rate even if it did. The agent
  // underneath answers, through the runtime hrx has already loaded.
  std::uint64_t hz = 0;
  if (!agent_timestamp_frequency(info_.ordinal, &hz) || hz == 0) {
    return LSE_ERROR(kUnimplemented,
                     "neither this device's HAL nor its agent publishes a "
                     "timestamp tick rate");
  }
  DeviceClock clock;
  clock.domain = ClockDomain::kDeviceAgent;
  clock.ticks_per_second = hz;
  // The AMDGPU HAL stamps all 64 bits as counting, per queue family and per
  // device (iree/hal/drivers/amdgpu/device_spec_builder.c). Stated here rather
  // than assumed because a narrower counter is normal elsewhere and a wrap read
  // as elapsed time is a duration of several thousand years.
  clock.valid_bits = 64;
  clock.ordinal = info_.ordinal;
  return clock;
#endif
}

Result<DeviceTimestamp> HrxBackend::sample_device_time_impl() const {
#if !LSE_HRX_LINKED
  return LSE_ERROR(kUnimplemented, "libhrx not linked");
#else
  if (!initialized_) return LSE_ERROR(kInternal, "hrx backend not initialized");
  // Declining on purpose, and the alternatives were checked rather than assumed:
  //
  //   - hrx exposes exactly one timing call, hrx_event_elapsed_time, and it is
  //     a host clock. libhrx/src/libhrx/event.c takes iree_time_now() when the
  //     *host* calls record — before the flush, so it orders host calls, not
  //     device work — and returns the difference as a device duration. Calling
  //     it would report submission jitter, which on a batched command buffer is
  //     most of what it would report.
  //   - hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP) is likewise a host clock
  //     despite the name: measured here it reads in 22.8 ns, which is vDSO
  //     speed and admits no device access, and it drifts -19.2 ppm against
  //     CLOCK_MONOTONIC_RAW, which is NTP slewing a host counter. Two host
  //     clocks, no device.
  //   - ROCr exports no call returning a current agent-domain tick, and
  //     hsaKmtGetClockCounters, which would, is static inside libhsakmt.a and
  //     not an exported symbol of libhsa-runtime64.so.1.
  //
  // What remains is asking the device: a dispatch that writes its own counter,
  // which costs a submission and a completion wait. That is the ~3 us round
  // trip already measured and rejected for this engine's dispatch path, so it
  // is not sold here as a clock read.
  return LSE_ERROR(kUnimplemented,
                   "hrx cannot read this device's counter: it exposes no "
                   "timestamp call, and its one timing call "
                   "(hrx_event_elapsed_time) is a host clock. Needs "
                   "hrx_stream_timestamp() forwarding "
                   "iree_hal_device_queue_timestamp(), which the AMDGPU HAL "
                   "already implements");
#endif
}

// Grows the transfer staging buffer to at least `bytes`, in powers of two so a
// run of increasing transfers reallocates a handful of times rather than every
// call.
Status HrxBackend::ensure_staging(std::size_t bytes) {
#if !LSE_HRX_LINKED
  (void)bytes;
  return LSE_ERROR(kUnimplemented, "libhrx not linked");
#else
  if (staging_bytes_ >= bytes && staging_buffer_ != nullptr) return OkStatus();
  std::size_t want = staging_bytes_ != 0 ? staging_bytes_ : (1u << 20);
  while (want < bytes) want *= 2;

  hrx_buffer_params_t params = {};
  params.type = HRX_MEMORY_TYPE_HOST_VISIBLE | HRX_MEMORY_TYPE_HOST_COHERENT |
                HRX_MEMORY_TYPE_DEVICE_VISIBLE;
  params.access = HRX_MEMORY_ACCESS_ALL;
  params.usage = HRX_BUFFER_USAGE_TRANSFER | HRX_BUFFER_USAGE_MAPPING_SCOPED |
                 HRX_BUFFER_USAGE_MAPPING_PERSISTENT;
  params.queue_affinity = 0;
  hrx_buffer_t fresh = nullptr;
  LSE_RETURN_IF_ERROR(
      from_hrx(hrx_allocator_allocate_buffer(
                   static_cast<hrx_allocator_t>(allocator_), params, want,
                   &fresh),
               "hrx_allocator_allocate_buffer (staging)"));
  void* mapped = nullptr;
  const Status mapped_ok = from_hrx(
      hrx_buffer_map(fresh, HRX_MAP_READ | HRX_MAP_WRITE, 0, want,
                     &mapped),
      "hrx_buffer_map (staging)");
  if (!mapped_ok.ok() || mapped == nullptr) {
    hrx_buffer_release(fresh);
    return mapped_ok.ok() ? LSE_ERROR(kInternal, "staging buffer did not map")
                          : mapped_ok;
  }
  if (staging_buffer_ != nullptr) {
    hrx_buffer_release(static_cast<hrx_buffer_t>(staging_buffer_));
  }
  staging_buffer_ = fresh;
  staging_host_ = mapped;
  staging_bytes_ = want;
  return OkStatus();
#endif
}

// A host->device copy with no host-side pass over the data: the caller's range
// becomes a buffer this device can read, and the copy engine does the rest.
// Declines when the runtime will not take the range, which is the caller's cue
// to stage instead.
// A host<->device transfer on the DMA engine.
//
// Three things have to line up: the host range must be pinned, both ends must
// be named by the agent that owns them, and the device address has to come
// from the buffer rather than being assumed. Any of them failing is a decline,
// and the caller falls back to the HAL's own path.
Status HrxBackend::dma_host_transfer(void* host, const DeviceBuffer& device,
                                     std::size_t bytes,
                                     std::size_t device_offset,
                                     bool to_device) {
  // Raw HSA against memory a spanning device manages is off the runtime's
  // supported path: the copy waits on a fence its bookkeeping never signals,
  // and a 27B load hung on the third gigabyte. One physical device is where
  // this path was measured and proven; a spanning device takes the runtime's
  // own stream transfers below instead.
  if (physical_count_ > 1) {
    return LSE_ERROR(kUnimplemented,
                     "raw DMA declines on a device spanning several GPUs");
  }

#if !LSE_HRX_LINKED
  (void)host; (void)device; (void)bytes; (void)device_offset; (void)to_device;
  return LSE_ERROR(kUnimplemented, "libhrx not linked");
#else
  const HsaRuntime& hsa = shared_hsa();
  if (!hsa.can_dma()) return LSE_ERROR(kUnimplemented, "no DMA entry points");
  const std::vector<HsaAgent>& agents = peer_agents();
  const auto here = static_cast<std::size_t>(gpu_ordinal_);
  if (here >= agents.size()) {
    return LSE_ERROR(kUnimplemented, "no agent for this device");
  }
  const HsaAgent gpu = agents[here];
  const HsaAgent cpu = hsa.cpu_agent();
  if (cpu.handle == 0) return LSE_ERROR(kUnimplemented, "no host agent");

  void* device_ptr = nullptr;
  if (!hrx_status_is_ok(hrx_buffer_get_device_ptr(
          reinterpret_cast<hrx_buffer_t>(device.handle), &device_ptr)) ||
      device_ptr == nullptr) {
    return LSE_ERROR(kUnimplemented, "this buffer has no device address");
  }
  auto* dev = static_cast<std::byte*>(device_ptr) + device.offset + device_offset;

  HsaAgent lock_agents[1] = {gpu};
  void* locked = hsa.lock_host(host, bytes, lock_agents, 1);
  if (locked == nullptr) return LSE_ERROR(kUnimplemented, "cannot pin the host range");

  // The transfer must not overtake work already queued on this device.
  const Status ordered = synchronize_impl();
  bool moved = false;
  if (ordered.ok()) {
    moved = to_device ? hsa.dma_copy(dev, gpu, locked, cpu, bytes)
                      : hsa.dma_copy(locked, cpu, dev, gpu, bytes);
  }
  hsa.unlock_host(host);
  if (!ordered.ok()) return ordered;
  if (!moved) return LSE_ERROR(kUnimplemented, "the DMA engine declined");
  return OkStatus();
#endif
}

Status HrxBackend::copy_h2d_imported(const void* src, DeviceBuffer& dst,
                                     std::size_t bytes,
                                     std::size_t dst_offset) {
#if !LSE_HRX_LINKED
  (void)src; (void)dst; (void)bytes; (void)dst_offset;
  return LSE_ERROR(kUnimplemented, "libhrx not linked");
#else
  hrx_buffer_params_t params = {};
  params.type = HRX_MEMORY_TYPE_HOST_VISIBLE | HRX_MEMORY_TYPE_DEVICE_VISIBLE;
  params.access = HRX_MEMORY_ACCESS_ALL;
  params.usage = HRX_BUFFER_USAGE_TRANSFER;
  params.queue_affinity = 0;
  hrx_buffer_t imported = nullptr;
  if (!hrx_status_is_ok(hrx_allocator_import_buffer(
          static_cast<hrx_allocator_t>(allocator_), params,
          const_cast<void*>(src), bytes, &imported)) ||
      imported == nullptr) {
    return LSE_ERROR(kUnimplemented, "this range cannot be imported");
  }
  auto stream = stream_at(0);
  if (!stream.ok()) {
    hrx_buffer_release(imported);
    return stream.status();
  }
  const Status moved =
      from_hrx(hrx_stream_copy_buffer(
                   static_cast<hrx_stream_t>(*stream), imported, 0,
                   reinterpret_cast<hrx_buffer_t>(dst.handle),
                   dst.offset + dst_offset, bytes),
               "hrx_stream_copy_buffer (h2d imported)");
  unflushed_launches_[0] = 0;
  const Status drained = moved.ok() ? synchronize_stream_impl(Stream{0}) : moved;
  hrx_buffer_release(imported);
  return drained;
#endif
}

Status HrxBackend::copy_h2d_impl(const void* src, DeviceBuffer& dst,
                                 std::size_t bytes, std::size_t dst_offset) {
#if !LSE_HRX_LINKED
  (void)src; (void)dst; (void)bytes; (void)dst_offset;
  return LSE_ERROR(kUnimplemented, "libhrx not linked");
#else
  if (src == nullptr || dst.handle == 0) {
    return LSE_ERROR(kInvalidArgument, "null buffer in copy_h2d");
  }
  if (dst_offset + bytes > dst.size_bytes) {
    return LSE_ERROR(kOutOfRange, "copy_h2d writes past the end of the buffer");
  }
  LSE_SYNC_TRACE("copy_h2d %zu bytes", bytes);
  // hrx_synchronous_* is a separate device submission carrying no semaphore
  // dependency on any stream, and a HAL queue does not order entries by
  // submission — only by semaphore edges. Flushing is therefore not enough:
  // the streams have to be *waited* for, or the transfer races whatever they
  // still hold on this buffer. With one stream that race was invisible; with
  // two it is not, and it is the same race either way.
  LSE_RETURN_IF_ERROR(synchronize_impl());
  // On the stream, not through hrx_synchronous_*. The blocking form moves the
  // bytes with the CPU across the PCIe aperture: measured on gfx1201 at
  // 0.64 GB/s for the part of the cost that scales with size, against 48 GB/s
  // for the same link when the copy engine does it. The stream form enqueues
  // the transfer for the device, which is what makes it a DMA rather than a
  // memcpy through a window.
  // Two halves, both fast: a host memcpy into memory the device already owns,
  // then the copy engine. IREE's helper would allocate a staging buffer for
  // this transfer and free it again, which is where the time went.
  auto stream = stream_at(0);
  if (!stream.ok()) return stream.status();
  // Import the caller's memory and let the copy engine read it where it lies.
  // Staging costs a second pass over the data -- at 4 MB the host memcpy was
  // about 200 us against 85 for the DMA itself -- and the import exists
  // precisely so that pass is not needed. It can decline, and then the staging
  // path below still gets the transfer done.
  // The copy engine first, then the HAL's blit path if it will not take it.
  // A spanning device takes the runtime's own blocking transfer. It moves the
  // bytes with the CPU across the aperture at about 0.6 GB/s -- a 14 GB load
  // in half a minute -- where the batched stream path, with its per-call drain
  // of every stream, decayed toward a standstill and never finished a load at
  // all. The raw DMA paths below are single-device machinery.
  if (physical_count_ > 1) {
    LSE_SYNC_TRACE("synchronous_h2d %zu bytes enter", bytes);
    const Status moved = from_hrx(
        hrx_synchronous_h2d(static_cast<hrx_device_t>(device_), src,
                            reinterpret_cast<hrx_buffer_t>(dst.handle),
                            dst.offset + dst_offset, bytes),
        "hrx_synchronous_h2d");
    LSE_SYNC_TRACE("synchronous_h2d leave");
    return moved;
  }
  if (const Status dma = dma_host_transfer(const_cast<void*>(src), dst, bytes,
                                           dst_offset, /*to_device=*/true);
      dma.ok()) {
    return dma;
  }
  if (const Status direct = copy_h2d_imported(src, dst, bytes, dst_offset);
      direct.ok()) {
    return direct;
  }

  constexpr std::size_t kChunk = 32u << 20;
  const auto* in = static_cast<const std::byte*>(src);
  for (std::size_t done = 0; done < bytes;) {
    const std::size_t n = std::min(kChunk, bytes - done);
    LSE_RETURN_IF_ERROR(ensure_staging(n));
    std::memcpy(staging_host_, in + done, n);
    LSE_RETURN_IF_ERROR(from_hrx(
        hrx_stream_copy_buffer(static_cast<hrx_stream_t>(*stream),
                               static_cast<hrx_buffer_t>(staging_buffer_), 0,
                               reinterpret_cast<hrx_buffer_t>(dst.handle),
                               dst.offset + dst_offset + done, n),
        "hrx_stream_copy_buffer (h2d)"));
    unflushed_launches_[0] = 0;
    LSE_RETURN_IF_ERROR(synchronize_stream_impl(Stream{0}));
    done += n;
  }
  return OkStatus();
#endif
}

Status HrxBackend::copy_peer_impl(const DeviceBuffer& src, DeviceBuffer& dst,
                                  std::size_t bytes, std::size_t src_offset,
                                  std::size_t dst_offset) {
#if !LSE_HRX_LINKED
  (void)src; (void)dst; (void)bytes; (void)src_offset; (void)dst_offset;
  return LSE_ERROR(kUnimplemented, "libhrx not linked");
#else
  if (src.handle == 0 || dst.handle == 0) {
    return LSE_ERROR(kInvalidArgument, "null buffer in copy_peer");
  }
  if (src_offset + bytes > src.size_bytes ||
      dst_offset + bytes > dst.size_bytes) {
    return LSE_ERROR(kOutOfRange, "copy_peer runs past the end of a buffer");
  }
  if (bytes == 0) return OkStatus();

  // Straight to the copy engine, around the HAL rather than through it.
  //
  // Importing the peer's allocation into this device's allocator is refused --
  // one hrx_device_t per GPU means one logical device per GPU, and the AMDGPU
  // HAL will not take an allocation owned by an agent outside its own
  // topology. HSA has no such notion: given two addresses, two agents and an
  // access grant, it moves the bytes. The grant is done when the memory is
  // allocated, which is why every device-local allocation offers itself to the
  // other GPUs.
  //
  // Both ends must be hrx allocations before either handle is dereferenced:
  // the link probe's contract is one honest try per pair, and a foreign
  // backend's buffer must come back "unimplemented", not crash inside libhrx.
  if (!hrx_owns_residency(src.residency) ||
      !hrx_owns_residency(dst.residency)) {
    return LSE_ERROR(kUnimplemented, "copy_peer needs two hrx-owned buffers");
  }
  const HsaRuntime& hsa = shared_hsa();
  if (!hsa.can_dma()) return LSE_ERROR(kUnimplemented, "no DMA entry points");
  const std::vector<HsaAgent>& agents = peer_agents();
  const auto here = static_cast<std::size_t>(gpu_ordinal_);
  if (here >= agents.size()) {
    return LSE_ERROR(kUnimplemented, "no agent for this device");
  }

  void* src_ptr = nullptr;
  void* dst_ptr = nullptr;
  if (!hrx_status_is_ok(hrx_buffer_get_device_ptr(
          reinterpret_cast<hrx_buffer_t>(src.handle), &src_ptr)) ||
      !hrx_status_is_ok(hrx_buffer_get_device_ptr(
          reinterpret_cast<hrx_buffer_t>(dst.handle), &dst_ptr)) ||
      src_ptr == nullptr || dst_ptr == nullptr) {
    return LSE_ERROR(kUnimplemented, "a buffer in this pair has no address");
  }
  auto* from = static_cast<std::byte*>(src_ptr) + src.offset + src_offset;
  auto* to = static_cast<std::byte*>(dst_ptr) + dst.offset + dst_offset;

  // Both ends are named by the agent doing the copy. The source belongs to
  // another GPU, and reaching it is what the grant at allocation bought.
  //
  // This drains THIS device only — the destination's streams are not this
  // instance's to drain, and nothing here can reach them. The other half of
  // the ordering is the caller's: the scheduler dispatches consumers of the
  // destination buffer host-ordered after this returns (make_local), so no
  // in-flight destination work can be reading the bytes the DMA overwrites.
  // If peer traffic ever runs concurrent with destination work, the
  // destination needs its own drain before the copy, not after.
  LSE_RETURN_IF_ERROR(synchronize_impl());
  if (!hsa.dma_copy(to, agents[here], from, agents[here], bytes)) {
    return LSE_ERROR(kUnimplemented, "the peer copy was refused");
  }
  return OkStatus();
#endif
}

Status HrxBackend::copy_d2h_impl(const DeviceBuffer& src, void* dst,
                                 std::size_t bytes, std::size_t src_offset) {
#if !LSE_HRX_LINKED
  (void)src; (void)dst; (void)bytes; (void)src_offset;
  return LSE_ERROR(kUnimplemented, "libhrx not linked");
#else
  if (dst == nullptr || src.handle == 0) {
    return LSE_ERROR(kInvalidArgument, "null buffer in copy_d2h");
  }
  if (src_offset + bytes > src.size_bytes) {
    return LSE_ERROR(kOutOfRange, "copy_d2h reads past the end of the buffer");
  }
  // Same hazard as copy_h2d: the transfer is not ordered against any stream
  // by anything but this wait.
  LSE_SYNC_TRACE("copy_d2h %zu bytes", bytes);
  LSE_RETURN_IF_ERROR(synchronize_impl());
  // Same choice as copy_h2d, same reason: the runtime's own blocking transfer
  // on a spanning device, the raw single-device machinery otherwise.
  if (physical_count_ > 1) {
    LSE_SYNC_TRACE("synchronous_d2h %zu bytes enter", bytes);
    const Status moved = from_hrx(
        hrx_synchronous_d2h(static_cast<hrx_device_t>(device_),
                            reinterpret_cast<hrx_buffer_t>(src.handle),
                            src.offset + src_offset, dst, bytes),
        "hrx_synchronous_d2h");
    LSE_SYNC_TRACE("synchronous_d2h leave");
    return moved;
  }
  if (const Status dma = dma_host_transfer(dst, src, bytes, src_offset,
                                           /*to_device=*/false);
      dma.ok()) {
    return dma;
  }

  // The mirror of copy_h2d: the copy engine into staging, then a host memcpy.
  auto stream = stream_at(0);
  if (!stream.ok()) return stream.status();
  constexpr std::size_t kChunk = 32u << 20;
  auto* out = static_cast<std::byte*>(dst);
  for (std::size_t done = 0; done < bytes;) {
    const std::size_t n = std::min(kChunk, bytes - done);
    LSE_RETURN_IF_ERROR(ensure_staging(n));
    LSE_RETURN_IF_ERROR(from_hrx(
        hrx_stream_copy_buffer(static_cast<hrx_stream_t>(*stream),
                               reinterpret_cast<hrx_buffer_t>(src.handle),
                               src.offset + src_offset + done,
                               static_cast<hrx_buffer_t>(staging_buffer_), 0, n),
        "hrx_stream_copy_buffer (d2h)"));
    unflushed_launches_[0] = 0;
    LSE_RETURN_IF_ERROR(synchronize_stream_impl(Stream{0}));
    std::memcpy(out + done, staging_host_, n);
    done += n;
  }
  return OkStatus();
#endif
}

Result<KernelHandle> HrxBackend::load_executable_impl(
    std::string_view name, std::span<const std::byte> code_object) {
#if !LSE_HRX_LINKED
  (void)name; (void)code_object;
  return LSE_ERROR(kUnimplemented, "libhrx not linked");
#else
  if (code_object.empty()) {
    return LSE_ERROR(kInvalidArgument, "empty code object");
  }

  hrx_executable_t executable = nullptr;
  // target_family/target_key tell HRX which advertised device target this code
  // object was built for; the arch string is exactly that key.
  LSE_RETURN_IF_ERROR(from_hrx(
      hrx_executable_load_data(static_cast<hrx_device_t>(device_),
                               code_object.data(), code_object.size(), "amdgpu",
                               info_.arch.c_str(), &executable),
      "hrx_executable_load_data"));

  std::size_t export_count = 0;
  LSE_RETURN_IF_ERROR(from_hrx(
      hrx_executable_export_count(executable, &export_count),
      "hrx_executable_export_count"));
  if (export_count == 0) {
    hrx_executable_release(executable);
    return LSE_ERROR(kCompileError, "code object for '", std::string(name),
                     "' advertises no exports");
  }

  // A code object may hold many kernels (one source file can define several
  // primitives plus shared helpers), so resolve by name rather than assuming
  // ordinal 0. Fall back to the sole export when there is exactly one.
  const std::string entry(name);
  std::uint32_t ordinal = 0;
  hrx_status_t looked_up =
      hrx_executable_lookup_export_by_name(executable, entry.c_str(), &ordinal);
  if (!hrx_status_is_ok(looked_up)) {
    hrx_status_ignore(looked_up);
    if (export_count != 1) {
      std::string available;
      for (std::uint32_t i = 0; i < export_count; ++i) {
        hrx_executable_export_info_t info = {};
        if (hrx_status_is_ok(hrx_executable_export_info(executable, i, &info)) &&
            info.name != nullptr) {
          if (!available.empty()) available += ", ";
          available += info.name;
        }
      }
      hrx_executable_release(executable);
      return LSE_ERROR(kNotFound, "no export '", entry, "' in code object; has: ",
                       available);
    }
  }

  // Owned here, released at shutdown: the KernelHandle going out is a raw
  // ordinal pair with no unload call on the seam, and each executable retains
  // the device, so dropping the reference would leak both past every model
  // that used them.
  loaded_executables_.push_back(executable);

  KernelHandle handle;
  handle.executable = reinterpret_cast<std::uint64_t>(executable);
  handle.export_ordinal = ordinal;
  handle.name = entry;
  return handle;
#endif
}

Status HrxBackend::launch_impl(const KernelHandle& kernel, const LaunchDims& dims,
                               const DispatchArgs& args,
                               const DispatchTarget& target) {
#if !LSE_HRX_LINKED
  (void)kernel; (void)dims; (void)args; (void)target;
  return LSE_ERROR(kUnimplemented, "libhrx not linked");
#else
  if (!kernel.valid()) return LSE_ERROR(kInvalidArgument, "invalid kernel handle");
  if (!target.work.whole()) {
    // A dispatch is still a grid here, and a grid has no subrange. When a
    // kernel declares work items the range lands in the dispatch config and
    // this refusal goes away; nothing else on this path changes.
    return LSE_ERROR(kUnimplemented,
                     "hrx dispatch covers a whole grid; work ranges arrive "
                     "with work-item kernels");
  }
  auto stream = stream_at(target.stream.index);
  if (!stream.ok()) return stream.status();

  hrx_dispatch_config_t config = {};
  for (int i = 0; i < 3; ++i) {
    config.workgroup_count[i] = dims.workgroup_count[i];
    config.workgroup_size[i] = dims.workgroup_size[i];
  }
  config.subgroup_size = dims.subgroup_size;

  std::vector<hrx_buffer_ref_t> bindings;
  bindings.reserve(args.bindings.size());
  for (const BufferRef& ref : args.bindings) {
    if (ref.buffer == nullptr || ref.buffer->handle == 0) {
      return LSE_ERROR(kInvalidArgument, "null buffer binding in dispatch");
    }
    hrx_buffer_ref_t out = {};
    out.buffer = reinterpret_cast<hrx_buffer_t>(ref.buffer->handle);
    out.offset = ref.buffer->offset + ref.offset;
    out.length = ref.length != 0 ? ref.length : ref.buffer->size_bytes;
    bindings.push_back(out);
  }

  const std::uint32_t index = target.stream.index;
  auto* s = static_cast<hrx_stream_t>(*stream);

  // Every stream on one physical device takes the batched path. It used to be
  // stream 0 alone: a command buffer was submitted with
  // IREE_HAL_QUEUE_AFFINITY_ANY, which this HAL resolves by first-set-bit to
  // ring 0, so a batched stream could not name a queue and every other stream
  // paid hrx_queue_dispatch's completion-signal round trip per kernel
  // (measured ~3 us on gfx1151). Since hrx 2082d042 a stream's command buffer
  // is recorded AND submitted with the stream's own affinity (libhrx stream.c
  // begin_cb/flush), and every stream here is created on its own affinity
  // bit, so the cheap path reaches every ring — verified end-to-end on a
  // single gfx1201 (22.3 tok/s, 30k dispatches, planner spreading enabled).
  //
  // A SPANNING device's non-first members stay on the queue path: dispatch
  // command buffers executing on a second physical device hang mid-prefill
  // (~400 dispatches in, host parked in kfd_wait_on_events, reproduced on two
  // GPU pairs and at LSE_FLUSH_INTERVAL=1, while a single CB and a
  // cross-GPU event edge in isolation both pass), and the queue path is the
  // configuration the spanning device was built against. Lift this guard when
  // the runtime's multi-device CB path is proven under load.
  // LSE_BATCH_ALL=1 batches every member's stream. The hang that forced the
  // spanning guard was the cross-GPU device-wait deadlock, which the host-join
  // policy has since removed; this gate is for measuring the difference and
  // becomes the default once the batched spanning path survives soak.
  static const bool batch_all = std::getenv("LSE_BATCH_ALL") != nullptr;
  const bool batched =
      physical_count_ <= 1 || batch_all || index % physical_count_ == 0;
  if (batched) {
    LSE_RETURN_IF_ERROR(from_hrx(
        hrx_stream_dispatch(s,
                            reinterpret_cast<hrx_executable_t>(kernel.executable),
                            kernel.export_ordinal, &config,
                            args.constants.empty() ? nullptr : args.constants.data(),
                            args.constants.size(),
                            bindings.empty() ? nullptr : bindings.data(),
                            bindings.size(), args.flags),
        "hrx_stream_dispatch"));
    // The periodic flush keeps the GPU fed while the host records the rest of
    // the token's launches. Do not reach for hrx_stream_begin_capture or
    // hrx_graph_exec_update to go further: both are UNIMPLEMENTED stubs in
    // libhrx (graph.c) as of this writing.
    if (flush_interval_ != 0 &&
        ++unflushed_launches_[index] >= flush_interval_) {
      return flush_stream(index);
    }
    return OkStatus();
  }

  // Order inside the stream is the timeline: wait on the position this stream
  // is at, signal the next. That is the same guarantee the command buffer's
  // dispatch->dispatch barrier gives, spelled with a semaphore instead, and it
  // is required rather than implied — this HAL states outright that queue
  // entries are not ordered by submission (host_queue_waits.c).
  LSE_SYNC_TRACE("queue_dispatch(%u)", index);
  hrx_timeline_point_t self = {};
  LSE_RETURN_IF_ERROR(from_hrx(hrx_stream_get_timeline_position(s, &self),
                               "hrx_stream_get_timeline_position"));
  std::uint64_t next = 0;
  LSE_RETURN_IF_ERROR(from_hrx(hrx_stream_advance_timeline(s, &next),
                               "hrx_stream_advance_timeline"));

  hrx_semaphore_list_t waits = {};
  waits.semaphores = &self.semaphore;
  waits.values = &self.value;
  waits.count = self.value > 0 ? 1 : 0;
  hrx_semaphore_list_t signals = {};
  signals.semaphores = &self.semaphore;
  signals.values = &next;
  signals.count = 1;

  LSE_RETURN_IF_ERROR(from_hrx(
      hrx_queue_dispatch(static_cast<hrx_device_t>(device_),
                         stream_affinity_[index], &waits, &signals,
                         reinterpret_cast<hrx_executable_t>(kernel.executable),
                         kernel.export_ordinal, &config,
                         args.constants.empty() ? nullptr : args.constants.data(),
                         args.constants.size(),
                         bindings.empty() ? nullptr : bindings.data(),
                         bindings.size(), args.flags),
      "hrx_queue_dispatch"));
  unflushed_launches_[index] = 0;
  return OkStatus();
#endif
}

Result<StreamEvent> HrxBackend::record_event_impl(Stream stream) {
#if !LSE_HRX_LINKED
  (void)stream;
  return LSE_ERROR(kUnimplemented, "libhrx not linked");
#else
  if (stream.index >= streams_.size()) {
    return LSE_ERROR(kOutOfRange, "stream ", std::to_string(stream.index),
                     " past the ", std::to_string(streams_.size()),
                     " this device offers");
  }
  // A stream nobody has used has nothing to wait for. The invalid event says
  // exactly that, and wait_event treats it as satisfied.
  if (streams_[stream.index] == nullptr) {
    StreamEvent none;
    none.stream = stream;
    none.device = device_index();
    return none;
  }

  // The runtime's own synchronization point, not one assembled out of queue
  // barriers and timeline arithmetic here. hrx_event_record carries its own
  // semaphore signalled when the stream reaches this point, so a consumer waits
  // on THAT rather than on the producing stream's latest reservation -- which
  // may itself be a barrier waiting back, and with two streams ordering against
  // each other is a deadlock. hrx_stream_advance_timeline, which the previous
  // implementation used here, is documented for out-of-band work that will
  // signal the returned point later, which is not this.
  LSE_SYNC_TRACE("record_event(%u)", stream.index);
  hrx_event_t event = nullptr;
  LSE_RETURN_IF_ERROR(from_hrx(
      hrx_event_create(static_cast<hrx_device_t>(device_),
                       HRX_EVENT_FLAG_DISABLE_TIMING, &event),
      "hrx_event_create"));
  const hrx_status_t recorded = hrx_event_record(
      event, static_cast<hrx_stream_t>(streams_[stream.index]));
  if (!hrx_status_is_ok(recorded)) {
    hrx_event_release(event);
    return from_hrx(recorded, "hrx_event_record");
  }
  // hrx_event_record flushed the stream (libhrx event.c), so the open command
  // buffer is empty again and the batching counter says so.
  unflushed_launches_[stream.index] = 0;

  StreamEvent ev;
  ev.stream = stream;
  ev.timeline = ++event_serial_;
  ev.handle = event;
  ev.device = device_index();
  // The event lives while any copy of `ev` does — the host may still queue a
  // wait on it — and then goes to the graveyard rather than being freed: a
  // wait already queued on the device still names the object, and only a
  // full-device synchronize proves those have retired. If the backend is
  // gone by the time the last copy drops, there is no later synchronize to
  // wait for and the deleter releases the event itself.
  ev.keepalive = std::shared_ptr<void>(
      static_cast<void*>(event), [g = graveyard_](void* p) {
        const std::lock_guard lock(g->mu);
        if (g->backend_alive) {
          g->retired.push_back(p);
        } else {
          hrx_event_release(static_cast<hrx_event_t>(p));
        }
      });
  return ev;
#endif
}

Status HrxBackend::wait_event_impl(Stream stream, const StreamEvent& event) {
#if !LSE_HRX_LINKED
  (void)stream; (void)event;
  return LSE_ERROR(kUnimplemented, "libhrx not linked");
#else
  if (!event.valid() || event.handle == nullptr) return OkStatus();
  LSE_SYNC_TRACE("wait_event(on=%u, from=%u)", stream.index,
                 event.stream.index);
  // A cross-GPU edge on a spanning device joins on the HOST, not on the
  // device. A queued wait on an event recorded by another physical GPU
  // deadlocks once such edges run in both directions — reproduced
  // deterministically at the same graph position on two GPU pairs, with both
  // dispatch modes and at LSE_FLUSH_INTERVAL=1, while a single such edge in
  // isolation passes (scratch_probe/cb_on_gpu1). The host join costs one
  // round trip per cross edge and keeps every cross-device dependency out of
  // the device's semaphore graph, which is what the runtime cannot currently
  // resolve under load. Remove when the HAL's cross-device semaphore path is
  // proven under a bidirectional pattern.
  // Every cross-stream edge joins on the HOST, in deliberate violation of the
  // seam's "does not block the host". The device-side wait
  // (hrx_stream_wait_event) is not reliable on this runtime: cross-GPU edges
  // deadlock once they run in both directions, and same-GPU cross-ring edges
  // produce racy reads — a single-GPU spread decode emitted corrupted logits
  // with device-side waits and correct text with host joins, everything else
  // equal. Both fit one story: the runtime's queued waits do not order what
  // they promise, and its conformance suite has never tested them (CTS
  // README: dispatch, multidevice, multithread all "Not Yet Tested"). A
  // same-stream event is already ordered by its own timeline and needs
  // nothing. Revisit when hrx conformance covers queued cross-stream waits.
  if (event.stream.index == stream.index) return OkStatus();
  return from_hrx(
      hrx_event_synchronize(static_cast<hrx_event_t>(event.handle)),
      "hrx_event_synchronize");
#endif
}

Status HrxBackend::synchronize_stream_impl(Stream stream) {
#if !LSE_HRX_LINKED
  (void)stream;
  return LSE_ERROR(kUnimplemented, "libhrx not linked");
#else
  if (!initialized_) return LSE_ERROR(kInternal, "hrx backend not initialized");
  if (stream.index >= streams_.size()) {
    return LSE_ERROR(kOutOfRange, "stream ", std::to_string(stream.index),
                     " past the ", std::to_string(streams_.size()),
                     " this device offers");
  }
  if (streams_[stream.index] == nullptr) return OkStatus();
  unflushed_launches_[stream.index] = 0;
  LSE_SYNC_TRACE("stream_synchronize(%u) enter", stream.index);
  const Status synced = from_hrx(
      hrx_stream_synchronize(static_cast<hrx_stream_t>(streams_[stream.index])),
      "hrx_stream_synchronize");
  LSE_SYNC_TRACE("stream_synchronize(%u) leave", stream.index);
  return synced;
#endif
}

Result<void*> HrxBackend::device_pointer_impl(const DeviceBuffer& buf) const {
#if !LSE_HRX_LINKED
  (void)buf;
  return LSE_ERROR(kUnimplemented, "libhrx not linked");
#else
  if (buf.handle == 0) {
    return LSE_ERROR(kInvalidArgument, "null buffer in device_pointer");
  }
  void* p = nullptr;
  LSE_RETURN_IF_ERROR(from_hrx(
      hrx_buffer_get_device_ptr(reinterpret_cast<hrx_buffer_t>(buf.handle), &p),
      "hrx_buffer_get_device_ptr"));
  return static_cast<void*>(static_cast<std::byte*>(p) + buf.offset);
#endif
}

Status HrxBackend::synchronize_impl() {
#if !LSE_HRX_LINKED
  return LSE_ERROR(kUnimplemented, "libhrx not linked");
#else
  if (!initialized_) return LSE_ERROR(kInternal, "hrx backend not initialized");
  // Every stream, because "the device is idle" is the only thing a caller can
  // mean by synchronize() with no stream named. hrx_stream_synchronize flushes
  // the open command buffer itself before waiting (libhrx stream.c), and a
  // stream that was never used is skipped inside synchronize_impl(Stream).
  for (std::uint32_t i = 0; i < streams_.size(); ++i) {
    LSE_RETURN_IF_ERROR(synchronize_stream_impl(Stream{i}));
  }
  // The device is idle, so every queued wait has retired: events whose last
  // host handle has already been dropped can finally be freed.
  std::vector<void*> retired;
  {
    const std::lock_guard lock(graveyard_->mu);
    retired.swap(graveyard_->retired);
  }
  for (void* e : retired) hrx_event_release(static_cast<hrx_event_t>(e));
  return OkStatus();
#endif
}

}  // namespace lse::backend

LSE_REGISTER_BACKEND("hrx", ::lse::backend::HrxBackend)
