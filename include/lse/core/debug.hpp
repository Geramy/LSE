#pragma once

namespace lse {

// Process-wide debug. `--debug` or LSE_DEBUG=1. HIP dumps are always written;
// this flag only adds path/count prints.
void set_debug(bool on) noexcept;
[[nodiscard]] bool debug() noexcept;

}  // namespace lse
