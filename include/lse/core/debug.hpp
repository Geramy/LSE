#pragma once

namespace lse {

// Process-wide debug. `--debug` or LSE_DEBUG=1. When on, each generated HIP
// translation unit is written under the build directory for review.
void set_debug(bool on) noexcept;
[[nodiscard]] bool debug() noexcept;

}  // namespace lse
