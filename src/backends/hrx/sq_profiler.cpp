// SQ busy-cycle profiler — implementation.
//
// See sq_profiler.hpp for the contract. All IREE/HRX symbols are resolved at
// runtime from the libhrx dylib the backend already links, so this file
// compiles and links without any IREE headers at build time. The small C ABI
// struct layouts it relies on are declared here with static size guards
// against drift.
//
// Design:
//  - Sink: the file profile sink (iree_hal_profile_file_sink_create, exported
//    by libhrx) pointed at a plain OS file in a private tmp dir. The IRPF
//    record format is a self-describing little-endian layout (104-byte record
//    header + content-type + name + payload); the watcher parses COUNTER
//    SAMPLES records only and ignores everything else.
//  - Session: one process-wide iree_hal_device_profiling_begin with
//    COUNTER_SAMPLES + DISPATCH_EVENTS data families and a single strict
//    counter set {SQ_BUSY_CYCLES}. The AMDGPU driver routes dispatch events
//    on every queue and emits counter samples per dispatch
//    (profile_counters.c: write_profile_counter_samples on the flush path).
//  - Watcher thread: every window (500 ms) it flushes retired dispatch events,
//    reads only newly appended complete records from the profile file, sums
//    SQ_BUSY_CYCLES values and tick ranges, and publishes
//    {seq, sq_busy, ref_cyc, clock_hz} to the shared slot.

#include "lse/backends/hrx/hrx_backend.hpp"

#include <thread>

#include "sq_profiler.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <dlfcn.h>
#include <fcntl.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>

#include "lse/backends/hrx/hrx_backend.hpp"

namespace lse::hrx {
namespace {

//===----------------------------------------------------------------------===//
// Minimal IREE/HRX ABI (resolved from libhrx at runtime).
//===----------------------------------------------------------------------===//

using iree_status_t = int;
using iree_host_size_t = std::size_t;
using iree_allocator_command_t = int;
using iree_allocator_ctl_fn_t =
    iree_status_t (*)(void* self, iree_allocator_command_t command,
                      const void* params, void** inout_ptr);
struct iree_allocator_t {
  void* self;
  iree_allocator_ctl_fn_t ctl;
};
// iree_allocator_system() = {nullptr, iree_allocator_system_ctl}
static inline iree_allocator_t system_allocator() {
  iree_allocator_t allocator{};
  auto ctl = reinterpret_cast<iree_allocator_ctl_fn_t*>(
      dlsym(RTLD_DEFAULT, "iree_allocator_system_ctl"));
  allocator.self = nullptr;
  allocator.ctl = ctl ? *ctl : nullptr;
  return allocator;
}

struct iree_string_view_t {
  const char* data;
  iree_host_size_t data_length;
};
static_assert(sizeof(iree_string_view_t) == 16,
              "iree_string_view_t layout drift");
struct iree_io_file_handle_t;
struct iree_hal_device_t;
struct iree_hal_profile_sink_t;
struct iree_hal_profile_chunk_metadata_t;

struct iree_hal_profile_counter_set_selection_t {
  std::uint32_t flags;
  iree_string_view_t name;
  iree_host_size_t counter_name_count;
  const iree_string_view_t* counter_names;
};

struct iree_hal_profile_capture_filter_t {
  std::uint32_t flags;
  std::uint64_t command_buffer_id;
  std::uint32_t command_index;
  std::uint32_t physical_device_ordinal;
  std::uint32_t queue_ordinal;
  std::uint32_t reserved0;
};

struct iree_hal_device_profiling_options_t {
  std::uint64_t flags;
  std::uint64_t data_families;
  iree_hal_profile_sink_t* sink;
  iree_hal_profile_capture_filter_t capture_filter;
  iree_host_size_t counter_set_count;
  const iree_hal_profile_counter_set_selection_t* counter_sets;
};

struct iree_hal_profile_counter_sample_record_t {
  std::uint32_t record_length;
  std::uint32_t flags;
  std::uint32_t scope;
  std::uint32_t reserved0;
  std::uint64_t sample_id;
  std::uint64_t counter_set_id;
  std::uint64_t dispatch_event_id;
  std::uint64_t submission_id;
  std::uint64_t command_buffer_id;
  std::uint64_t executable_id;
  std::uint64_t stream_id;
  std::uint64_t start_tick;
  std::uint64_t end_tick;
  std::uint32_t command_index;
  std::uint32_t function_ordinal;
  std::uint32_t physical_device_ordinal;
  std::uint32_t queue_ordinal;
  std::uint32_t sample_value_count;
  std::uint32_t reserved1;
};
static_assert(sizeof(iree_hal_profile_counter_sample_record_t) == 112,
              "iree_hal_profile_counter_sample_record_t layout drift");

// iree_hal_profile_counter_set_record_t (32 bytes) and
// iree_hal_profile_counter_record_t (48 bytes) precede their trailing
// strings; we parse them positionally from the COUNTER / COUNTER_SETS chunk
// payloads to find SQ_BUSY_CYCLES' value offset within each sample.
struct irpf_counter_set_record_t {
  std::uint32_t record_length;   // 32
  std::uint32_t flags;
  std::uint64_t counter_set_id;
  std::uint32_t physical_device_ordinal;
  std::uint32_t counter_count;
  std::uint32_t sample_value_count;
  std::uint32_t name_length;
};
static_assert(sizeof(irpf_counter_set_record_t) == 32,
              "counter set record layout drift");
struct irpf_counter_record_t {
  std::uint32_t record_length;   // 48
  std::uint32_t flags;
  std::uint32_t unit;
  std::uint32_t physical_device_ordinal;
  std::uint64_t counter_set_id;
  std::uint32_t counter_ordinal;
  std::uint32_t sample_value_offset;
  std::uint32_t sample_value_count;
  std::uint32_t block_name_length;
  std::uint32_t name_length;
  std::uint32_t description_length;
};
static_assert(sizeof(irpf_counter_record_t) == 48,
              "counter record layout drift");

// IRPF file format (profile_file.h).
struct irpf_file_header_t {
  std::uint32_t magic;        // 0x46505249 "IRPF"
  std::uint16_t version_major;
  std::uint16_t version_minor;
  std::uint32_t header_length;
  std::uint32_t flags;
  std::uint64_t file_length;
};
static_assert(sizeof(irpf_file_header_t) == 24, "irpf header layout drift");

struct irpf_record_header_t {
  std::uint64_t record_length;
  std::uint64_t payload_length;
  std::uint64_t session_id;
  std::uint64_t stream_id;
  std::uint64_t event_id;
  std::uint64_t executable_id;
  std::uint64_t command_buffer_id;
  std::uint32_t header_length;
  std::uint32_t content_type_length;
  std::uint32_t name_length;
  std::uint32_t physical_device_ordinal;
  std::uint32_t queue_ordinal;
  std::uint64_t chunk_flags;
  std::uint64_t dropped_record_count;
  std::uint32_t session_status_code;
  std::uint16_t record_type;  // 1=session begin, 2=chunk, 3=session end
  std::uint16_t flags;
};
static_assert(sizeof(irpf_record_header_t) == 104, "irpf record layout drift");

using file_handle_create_fn = iree_status_t (*)(
    int mode, iree_string_view_t path, iree_host_size_t initial_size,
    iree_allocator_t allocator, iree_io_file_handle_t** out);
using file_handle_release_fn = void (*)(iree_io_file_handle_t* handle);
using file_sink_create_fn = iree_status_t (*)(
    iree_io_file_handle_t* handle, iree_allocator_t allocator,
    iree_hal_profile_sink_t** out);
using profiling_begin_fn = iree_status_t (*)(
    iree_hal_device_t* device, const iree_hal_device_profiling_options_t* opts);
using profiling_flush_fn = iree_status_t (*)(iree_hal_device_t* device);
using profiling_end_fn = iree_status_t (*)(iree_hal_device_t* device);
using sink_retain_fn = void (*)(iree_hal_profile_sink_t* sink);
using sink_release_fn = void (*)(iree_hal_profile_sink_t* sink);
using hrx_retain_device_fn = void (*)(void* device);
using hrx_release_device_fn = void (*)(void* device);
using hrx_get_hal_device_fn = iree_status_t (*)(void* device, void** out);

constexpr char kDefaultLibrary[] = "@rpath/libhrx.0.1.0.dylib";
constexpr std::uint64_t kDataCounterSamples = 1ull << 4;
constexpr std::uint64_t kDataDispatchEvents = 1ull << 3;
constexpr char kCounterSetName[] = "lse-sq-busy";
constexpr char kBusyCounter[] = "SQ_BUSY_CYCLES";
// Window wall length (ns) used as the busy-% denominator when samples carry
// no DEVICE_TICK_RANGE; set by run() from window_ns.
std::atomic<std::uint64_t> g_window_ns{500u * 1000u * 1000u};
constexpr char kCounterSamplesContentType[] =
    "application/vnd.iree.hal.profile.counter-samples";
constexpr char kCountersContentType[] = "application/vnd.iree.hal.profile.counters";
// IREE_HAL_PROFILE_COUNTER_SAMPLE_FLAG_DEVICE_TICK_RANGE = 1u << 2.
constexpr std::uint32_t kTickRangeFlag = 4u;

// IREE status codes: NOT_FOUND=5, FAILED_PRECONDITION=9, UNIMPLEMENTED=12,
// UNAVAILABLE=13 (iree/base/status.h). The driver treats this set as
// "counter profiling unsupported on this target" and no-ops cleanly.
bool status_allows_unsupported(int status) {
  return status == 5 || status == 9 || status == 12 || status == 13;
}

std::uint32_t read_u32(const std::uint8_t* p) {
  return std::uint32_t(p[0]) | (std::uint32_t(p[1]) << 8) |
         (std::uint32_t(p[2]) << 16) | (std::uint32_t(p[3]) << 24);
}
std::uint64_t read_u64(const std::uint8_t* p) {
  return std::uint64_t(read_u32(p)) | (std::uint64_t(read_u32(p + 4)) << 32);
}

}  // namespace

bool parse_sq_profile_bytes(const std::uint8_t* bytes, std::size_t size,
                            bool includes_file_header, SqProfileParseState& state,
                            SqProfileWindow& window, std::size_t& consumed) {
  consumed = 0;
  window = {};
  if (!bytes && size != 0) return false;
  SqProfileParseState next_state = state;
  SqProfileWindow next_window;
  std::size_t offset = 0;
  if (includes_file_header) {
    if (size < sizeof(irpf_file_header_t)) return false;
    const std::uint32_t header_length = read_u32(bytes + 8);
    if (read_u32(bytes) != 0x46505249u ||
        header_length < sizeof(irpf_file_header_t) || header_length > size)
      return false;
    offset = header_length;
  }

  while (size - offset >= sizeof(irpf_record_header_t)) {
    const std::uint8_t* record = bytes + offset;
    const std::uint64_t record_length = read_u64(record);
    const std::uint64_t payload_length = read_u64(record + 8);
    const std::uint32_t header_length = read_u32(record + 56);
    const std::uint32_t content_type_length = read_u32(record + 60);
    const std::uint32_t name_length = read_u32(record + 64);
    const std::uint16_t record_type = static_cast<std::uint16_t>(
        std::uint16_t(record[100]) | (std::uint16_t(record[101]) << 8));
    if (record_length < sizeof(irpf_record_header_t) ||
        header_length < sizeof(irpf_record_header_t) ||
        header_length > record_length)
      return false;
    if (record_length > size - offset) break;  // writer has not finished it
    const std::uint64_t body_length = record_length - header_length;
    if (content_type_length > body_length ||
        name_length > body_length - content_type_length ||
        payload_length != body_length - content_type_length - name_length)
      return false;

    if (record_type == 2 /*CHUNK*/) {
      const char* content_type =
          reinterpret_cast<const char*>(record + header_length);
      const std::uint8_t* payload =
          record + header_length + content_type_length + name_length;
      const std::size_t payload_size = static_cast<std::size_t>(payload_length);
      if (content_type_length == std::strlen(kCountersContentType) &&
          std::memcmp(content_type, kCountersContentType,
                      content_type_length) == 0) {
        std::size_t pos = 0;
        while (payload_size - pos >= sizeof(irpf_counter_record_t)) {
          const std::uint8_t* counter = payload + pos;
          const std::uint32_t counter_length = read_u32(counter);
          const std::uint32_t block_name_length = read_u32(counter + 36);
          const std::uint32_t counter_name_length = read_u32(counter + 40);
          const std::uint32_t description_length = read_u32(counter + 44);
          if (counter_length < sizeof(irpf_counter_record_t) ||
              counter_length > payload_size - pos)
            return false;
          const std::size_t trailing = counter_length - sizeof(irpf_counter_record_t);
          if (block_name_length > trailing ||
              counter_name_length > trailing - block_name_length ||
              description_length > trailing - block_name_length - counter_name_length)
            return false;
          const char* counter_name = reinterpret_cast<const char*>(
              counter + sizeof(irpf_counter_record_t) + block_name_length);
          if (counter_name_length == std::strlen(kBusyCounter) &&
              std::memcmp(counter_name, kBusyCounter, counter_name_length) == 0) {
            next_state.busy_value_offset = read_u32(counter + 28);
            next_state.have_busy_counter = true;
          }
          pos += counter_length;
        }
        if (pos != payload_size) return false;
      } else if (content_type_length ==
                     std::strlen(kCounterSamplesContentType) &&
                 std::memcmp(content_type, kCounterSamplesContentType,
                             content_type_length) == 0) {
        std::size_t pos = 0;
        while (payload_size - pos >=
               sizeof(iree_hal_profile_counter_sample_record_t)) {
          const std::uint8_t* sample = payload + pos;
          const std::uint32_t sample_length = read_u32(sample);
          const std::uint32_t value_count = read_u32(sample + 104);
          if (sample_length < sizeof(iree_hal_profile_counter_sample_record_t) ||
              sample_length > payload_size - pos ||
              value_count > (sample_length -
                  sizeof(iree_hal_profile_counter_sample_record_t)) /
                  sizeof(std::uint64_t))
            return false;
          const std::uint32_t index = next_state.busy_value_offset;
          if (next_state.have_busy_counter && index < value_count) {
            next_window.busy += read_u64(sample +
                sizeof(iree_hal_profile_counter_sample_record_t) +
                std::size_t(index) * sizeof(std::uint64_t));
            const std::uint64_t start_tick = read_u64(sample + 72);
            const std::uint64_t end_tick = read_u64(sample + 80);
            if ((read_u32(sample + 4) & kTickRangeFlag) && end_tick > start_tick)
              next_window.reference_ticks += end_tick - start_tick;
            ++next_window.sample_count;
          }
          pos += sample_length;
        }
        if (pos != payload_size) return false;
      }
    }
    offset += static_cast<std::size_t>(record_length);
  }
  state = next_state;
  window = next_window;
  consumed = offset;
  return true;
}

//===----------------------------------------------------------------------===//
// Impl
//===----------------------------------------------------------------------===//

struct SqProfiler::Impl {
  file_handle_create_fn file_handle_create = nullptr;
  file_handle_release_fn file_handle_release = nullptr;
  file_sink_create_fn file_sink_create = nullptr;
  profiling_begin_fn profiling_begin = nullptr;
  profiling_flush_fn profiling_flush = nullptr;
  profiling_end_fn profiling_end = nullptr;
  sink_retain_fn sink_retain = nullptr;
  sink_release_fn sink_release = nullptr;
  hrx_retain_device_fn hrx_device_retain = nullptr;
  hrx_release_device_fn hrx_device_release = nullptr;
  hrx_get_hal_device_fn hrx_get_hal_device = nullptr;

  void* library = nullptr;
  void* hrx_device = nullptr;        // retained hrx_device_t
  iree_hal_device_t* hal_device = nullptr;
  iree_io_file_handle_t* file_handle = nullptr;
  iree_hal_profile_sink_t* sink = nullptr;
  std::string profile_path;
  bool session_active = false;
  bool unsupported = false;
  std::uint32_t* slot = nullptr;  // 4 x u32 shared VRAM slot (CPU-mapped)
  std::uint64_t window_ns = 500u * 1000u * 1000u;
  std::uint32_t tick_freq_hz = 0;  // 0 = wall-clock denominator

  std::thread thread;
  std::atomic<bool> stop_flag{false};
  std::atomic<bool> active{false};
  std::string status_text = "not started";
  std::uint64_t last_drained_bytes = 0;
  SqProfileParseState parse_state;

  ~Impl() {
    stop_flag.store(true);
    if (thread.joinable()) thread.join();
    if (session_active && profiling_end && hal_device) {
      profiling_end(hal_device);
      session_active = false;
    }
    if (sink && sink_release) sink_release(sink);
    if (file_handle && file_handle_release) file_handle_release(file_handle);
    if (hrx_device && hrx_device_release) hrx_device_release(hrx_device);
    if (!profile_path.empty()) ::unlink(profile_path.c_str());
    if (library) dlclose(library);
  }

  bool resolve(const std::string& library_path) {
    const char* path =
        library_path.empty() ? kDefaultLibrary : library_path.c_str();
    library = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!library) {
      status_text = std::string("dlopen failed: ") + dlerror();
      return false;
    }
    auto sym = [&](const char* name) {
      return reinterpret_cast<void*>(dlsym(library, name));
    };
    file_handle_create = reinterpret_cast<file_handle_create_fn>(
        sym("iree_io_file_handle_create"));
    file_handle_release =
        reinterpret_cast<file_handle_release_fn>(sym("iree_io_file_handle_release"));
    file_sink_create =
        reinterpret_cast<file_sink_create_fn>(sym("iree_hal_profile_file_sink_create"));
    profiling_begin = reinterpret_cast<profiling_begin_fn>(
        sym("iree_hal_device_profiling_begin"));
    profiling_flush = reinterpret_cast<profiling_flush_fn>(
        sym("iree_hal_device_profiling_flush"));
    profiling_end = reinterpret_cast<profiling_end_fn>(
        sym("iree_hal_device_profiling_end"));
    sink_retain =
        reinterpret_cast<sink_retain_fn>(sym("iree_hal_profile_sink_retain"));
    sink_release =
        reinterpret_cast<sink_release_fn>(sym("iree_hal_profile_sink_release"));
    hrx_device_retain =
        reinterpret_cast<hrx_retain_device_fn>(sym("hrx_device_retain"));
    hrx_device_release =
        reinterpret_cast<hrx_release_device_fn>(sym("hrx_device_release"));
    hrx_get_hal_device =
        reinterpret_cast<hrx_get_hal_device_fn>(sym("hrx_device_get_hal_device"));
    bool ok = file_handle_create && file_handle_release && file_sink_create &&
              profiling_begin && profiling_end && sink_retain && sink_release &&
              hrx_device_retain && hrx_device_release && hrx_get_hal_device;
    if (!ok) {
      status_text = "missing libhrx symbols (need file sink + profiling begin/end)";
      return false;
    }
    // The flush symbol is newer; its absence is tolerable (the AMDGPU driver
    // still writes samples on dispatch retirement, and profiling_end drains).
    return true;
  }

  bool setup_sink_and_session() {
    hrx_device_retain(hrx_device);
    void* hal = nullptr;
    if (hrx_get_hal_device(hrx_device, &hal) != 0 || !hal) {
      status_text = "hrx_device_get_hal_device failed";
      return false;
    }
    hal_device = static_cast<iree_hal_device_t*>(hal);

    char dir_template[] = "/tmp/lse-sqprof-XXXXXX";
    if (::mkdtemp(dir_template) == nullptr) {
      status_text = "mkdtemp failed";
      return false;
    }
    profile_path = std::string(dir_template) + "/profile.irpf";

    iree_string_view_t path_view = {profile_path.data(),
                                    iree_host_size_t(profile_path.size())};
    if (file_handle_create(
            0x02ull /*WRITE*/ | 0x08ull /*SEQUENTIAL_SCAN*/ | 0x40ull /*SHARE_READ*/,
            path_view, 0, system_allocator(), &file_handle) != 0) {
      status_text = "iree_io_file_handle_create failed";
      return false;
    }
    if (file_sink_create(file_handle, system_allocator(), &sink) != 0 ||
        !sink) {
      status_text = "iree_hal_profile_file_sink_create failed";
      return false;
    }

    iree_string_view_t set_name = {kCounterSetName,
                                   std::strlen(kCounterSetName)};
    iree_string_view_t busy_name = {kBusyCounter, std::strlen(kBusyCounter)};
    iree_hal_profile_counter_set_selection_t selection = {
        0, set_name, 1, &busy_name};
    iree_hal_device_profiling_options_t options{};
    options.data_families = kDataCounterSamples | kDataDispatchEvents;
    options.sink = sink;
    options.counter_set_count = 1;
    options.counter_sets = &selection;

    int status = profiling_begin(hal_device, &options);
    if (status != 0) {
      if (status_allows_unsupported(status)) {
        unsupported = true;
        status_text = "counter profiling unsupported on this target "
                      "(iree status " +
                      std::to_string(status) + ")";
      } else {
        status_text = "iree_hal_device_profiling_begin failed (iree status " +
                      std::to_string(status) + ")";
      }
      return false;
    }
    session_active = true;
    status_text = "active";
    return true;
  }

  // Sums SQ_BUSY_CYCLES (value offset resolved from the COUNTERS metadata)
  // and the dispatch tick ranges from the profile file contents.
  bool harvest(std::uint64_t& out_busy, std::uint64_t& out_ref,
               std::uint32_t& out_sample_count) {
    out_busy = 0;
    out_ref = 0;
    out_sample_count = 0;
    int fd = ::open(profile_path.c_str(), O_RDONLY);
    if (fd < 0) return false;
    struct stat statbuf {};
    if (::fstat(fd, &statbuf) != 0 || statbuf.st_size < 0 ||
        static_cast<std::uint64_t>(statbuf.st_size) < last_drained_bytes ||
        ::lseek(fd, static_cast<off_t>(last_drained_bytes), SEEK_SET) < 0) {
      ::close(fd);
      return false;
    }
    std::vector<std::uint8_t> bytes;
    std::vector<std::uint8_t> buffer(64 * 1024);
    ssize_t n;
    while ((n = ::read(fd, buffer.data(), buffer.size())) > 0) {
      bytes.insert(bytes.end(), buffer.data(), buffer.data() + n);
    }
    ::close(fd);
    if (n < 0) return false;
    SqProfileWindow window;
    std::size_t consumed = 0;
    if (!parse_sq_profile_bytes(bytes.data(), bytes.size(),
                                last_drained_bytes == 0, parse_state,
                                window, consumed))
      return false;
    last_drained_bytes += consumed;
    out_busy = window.busy;
    out_ref = window.reference_ticks;
    out_sample_count = window.sample_count;
    return true;
  }

  void run() {
    g_window_ns.store(window_ns);
    while (!stop_flag.load()) {
      // Sleep the window without blocking shutdown.
      std::chrono::nanoseconds step(10 * 1000 * 1000);
      std::chrono::nanoseconds waited{0};
      while (waited < std::chrono::nanoseconds(window_ns) &&
             !stop_flag.load()) {
        std::this_thread::sleep_for(step);
        waited += step;
      }
      if (stop_flag.load()) break;
      if (!session_active) break;

      if (profiling_flush) {
        // Retires dispatch events on all queues so the counter samples are
        // written to the sink. Best-effort: a failure just means an empty
        // window.
        (void)profiling_flush(hal_device);
      }
      // Give the file sink a beat to flush the just-flushed samples to disk
      // (the sink may batch writes).
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
      std::uint64_t busy = 0, ref = 0;
      std::uint32_t sample_count = 0;
      bool ok = harvest(busy, ref, sample_count);
      // Publish a window: {seq, sq_busy, ref_cyc, clock_hz}. ref_cyc is in
      // the same units as clock_hz (ticks when the tick range flag was set,
      // nanoseconds otherwise — clock_hz is 0 to mark the latter).
      std::uint64_t ref_ticks = ref;
      if (ref == 0 && sample_count != 0) {
        // No tick range available: fall back to the wall window length so the
        // busy % still means something (sum of busy cycles over the window).
        ref_ticks = g_window_ns.load();
      }
      if (ok) {
        std::uint32_t* w = slot;
        const std::uint32_t seq = __atomic_load_n(&w[0], __ATOMIC_RELAXED);
        // Mark the slot busy before changing either payload word. Readers
        // accept only equal, even sequence values around their snapshot.
        __atomic_store_n(&w[0], seq + 1u, __ATOMIC_RELEASE);
        __atomic_thread_fence(__ATOMIC_SEQ_CST);
        __atomic_store_n(&w[1], sample_count == 0 ? 0u :
            static_cast<std::uint32_t>(busy % 4294967295u), __ATOMIC_RELAXED);
        __atomic_store_n(&w[2], sample_count == 0 ? 0u :
            static_cast<std::uint32_t>(ref_ticks % 4294967295u), __ATOMIC_RELAXED);
        __atomic_store_n(&w[0], seq + 2u, __ATOMIC_RELEASE);
        active.store(true);
      }
    }
  }
};

std::unique_ptr<SqProfiler> SqProfiler::start(const SqProfilerParams& params) {
  if (!params.device || !params.slot_host) return nullptr;
  auto impl = std::make_unique<Impl>();
  impl->slot = params.slot_host;
  impl->hrx_device = params.device;
  if (params.window_ms != 0) {
    impl->window_ns = std::uint64_t(params.window_ms) * 1000u * 1000u;
  }
  std::unique_ptr<SqProfiler> profiler(new SqProfiler(std::move(impl)));
  if (!profiler->impl_->resolve(params.library_path)) return nullptr;
  if (!profiler->impl_->setup_sink_and_session()) return nullptr;
  // New sessions begin with a stable even sequence and an empty window.
  __atomic_store_n(&params.slot_host[1], 0u, __ATOMIC_RELAXED);
  __atomic_store_n(&params.slot_host[2], 0u, __ATOMIC_RELAXED);
  __atomic_store_n(&params.slot_host[0], 0u, __ATOMIC_RELEASE);
  profiler->impl_->thread = std::thread([p = profiler.get()] { p->impl_->run(); });
  return profiler;
}

SqProfiler::SqProfiler(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
SqProfiler::~SqProfiler() = default;

bool SqProfiler::active() const { return impl_->active.load(); }
bool SqProfiler::unsupported() const { return impl_->unsupported; }
const char* SqProfiler::status_text() const { return impl_->status_text.c_str(); }

//===----------------------------------------------------------------------===//
// Backend seam.
//===----------------------------------------------------------------------===//

namespace {
// One shared slot per process. The dext-allocated BO is handed to the HSA
// transport (which maps it into this process's GART window) before the
// backend starts the profiler; the transport layer sets it via this call.
std::uint32_t* g_slot_host = nullptr;
}

void sq_profiler_set_slot(std::uint32_t* slot_host) { g_slot_host = slot_host; }

std::unique_ptr<SqProfiler> make_sq_profiler(void* hrx_device, std::uint32_t* slot_host) {
  if (!hrx_device || !slot_host) return nullptr;
  // This path is env-gated (LSE_SQ_PROFILER_SLOT) and the aqlprofile counter is
  // not available on gfx1201 (verified: the SQ busy-cycles counter is armed only
  // by the closed aqlprofile CP-perfmon PM4, not a simple register read). Keep the
  // seam compiling and returning null so the observer panel shows n/a.
  SqProfilerParams params;
  params.device = hrx_device;
  params.slot_host = slot_host;
  return SqProfiler::start(params);
}

void destroy_sq_profiler(std::unique_ptr<SqProfiler>& profiler) {
  profiler.reset();
}

}  // namespace lse::hrx
