// What a device measurably does, as dispatch reads it.
//
// The engine measures each device once (probe::qualify_pool) at load and keeps
// the result in the kernel cache directory, keyed by the device and the engine
// build, so a later start reads it. publish_roofline hands the numbers to
// dispatch; a dispatch decision that trades bytes for compute asks
// predict_ns() what a kernel costs on this device instead of reading a table
// written for one part. Nothing here is a default: a device that was never
// measured has no roofline, and a decision that needs one says so.
#pragma once

#include <cstdint>
#include <optional>

#include "lse/backend/backend.hpp"

namespace lse::dispatch {

struct Roofline {
  // Achieved streaming read rate out of device memory (probe stream kernel).
  double stream_bytes_per_s = 0.0;
  // One dispatch on the engine's submission path, with nothing in it.
  double launch_ns = 0.0;
  // FP32 vector multiply-add rate, flop/s.
  double vector_flops = 0.0;
  // BF16 matrix rate with an FP32 accumulator, flop/s; 0 without a matrix core.
  double matrix_flops = 0.0;

  [[nodiscard]] bool valid() const noexcept {
    return stream_bytes_per_s > 0.0 && launch_ns > 0.0 && vector_flops > 0.0;
  }
};

// What one kernel (or a short chain of them) moves and computes.
struct KernelWork {
  double bytes = 0.0;
  double vector_flops = 0.0;
  double matrix_flops = 0.0;
  std::uint32_t launches = 1;
};

// The roofline of a device (by arch and ordinal), replacing any earlier one.
void publish_roofline(const backend::DeviceInfo& device, const Roofline& roofline);
// Its roofline, or none when it was never measured.
[[nodiscard]] std::optional<Roofline> roofline(const backend::DeviceInfo* device);

// Streaming and compute overlap, launches do not:
// max(bytes / stream, vector / vector rate + matrix / matrix rate) + launches.
[[nodiscard]] double predict_ns(const Roofline& roofline, const KernelWork& work) noexcept;

}  // namespace lse::dispatch
