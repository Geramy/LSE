#pragma once

#include <cstdint>
#include <span>
#include <string_view>
namespace lse::model {
struct Config;
[[nodiscard]] constexpr bool qualified_q4_load_scope(std::size_t device_count,
                                                     bool split) noexcept {
  return device_count == 1 && !split;
}
class SafeTensors;
[[nodiscard]] bool qualified_q4_digests(std::string_view config,
                                        std::span<const std::string_view> shards) noexcept;
[[nodiscard]] std::int32_t qualified_q4_compute(const Config&, const SafeTensors&);
}
