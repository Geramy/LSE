// SQ busy-cycle profiler.
//
// Opens a device profiling session on the HRX HAL (aqlprofile SQ busy-cycle
// counter over the dispatch events) and publishes a rolling "CU busy %" into a
// shared VRAM slot that the MacAMDGPU dext can expose to observer tools.
//
// Honest provenance: the number is a real hardware SQ performance counter
// (SQ_BUSY_CYCLES via aqlprofile over AQL PM4-IB, compiled into libhrx),
// summed across all CUs over a fixed window. It is NOT CU occupancy (a
// resident-wavefront metric) and NOT the driver's dispatch-rate proxy.
//
// The whole thing is best-effort and degrades to "unavailable": the runtime
// may refuse counter profiling (no AQL PM4-IB support, missing aqlprofile
// event on this gfxip), in which case the slot's sequence never advances and
// consumers show n/a. No LSE functionality depends on this working.

#ifndef LSE_HRX_SQ_PROFILER_HPP
#define LSE_HRX_SQ_PROFILER_HPP

#include <cstddef>
#include <cstdint>
#include <memory>

namespace lse {
namespace hrx {

struct SqProfilerParams {
  // libhrx device handle (borrowed; retained internally while active).
  void* device = nullptr;      // hrx_device_t
  void* stream = nullptr;      // hrx_stream_t used for the slot allocation
  // Shared VRAM slot (16 bytes, 4 little-endian uint32):
  //   [0] seq       monotonic windows published (wraps)
  //   [1] sq_busy   SQ busy cycles accumulated in the window
  //   [2] ref_cyc   reference cycles (dispatch tick range) in the window
  //   [3] clock_hz  timestamp tick frequency
  // Must be mapped into this process's CPU address space (GART host window).
  std::uint32_t* slot_host = nullptr;
  // libhrx path for dlopen; empty = platform default.
  std::string library_path;
  // Poll cadence; 0 = default (500 ms).
  std::uint64_t window_ms = 0;
};

class SqProfiler {
 public:
  static std::unique_ptr<SqProfiler> start(const SqProfilerParams& params);
  ~SqProfiler();
  SqProfiler(const SqProfiler&) = delete;
  SqProfiler& operator=(const SqProfiler&) = delete;

  // True when the profiling session is active and windows are advancing.
  bool active() const;
  // True when the runtime reported the counter set is not supported here
  // (as opposed to "not started yet").
  bool unsupported() const;
  const char* status_text() const;  // diagnostic string, stable for lifetime

 private:
  struct Impl;
  explicit SqProfiler(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

// Complete-type seam so the backend can own the profiler without its full
// definition (the hpp keeps the backend header free of thread machinery).
// Backend seam so the backend header does not need the full type.
std::unique_ptr<SqProfiler> make_sq_profiler(void* hrx_device, std::uint32_t* slot_host);
void destroy_sq_profiler(std::unique_ptr<SqProfiler>& profiler);
void sq_profiler_set_slot(std::uint32_t* slot_host);

}
}

#endif  // LSE_HRX_SQ_PROFILER_HPP
