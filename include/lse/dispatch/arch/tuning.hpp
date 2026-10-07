// Which architecture's tuning a device gets.
//
// arch::tuning(arch) is the one place a processor name selects dispatch rows:
// the device's own header when there is one, the generic rows otherwise.
// arch::tunings() lists every part's rows for the decisions the graph makes
// before it knows the device (a shape that some part has a kernel for); the
// device-time check that follows reads tuning(arch) alone.
#pragma once

#include <array>
#include <span>
#include <string_view>
#include <type_traits>
#include <utility>

#include "lse/dispatch/arch/generic.hpp"
#include "lse/dispatch/arch/gfx1151.hpp"
#include "lse/dispatch/arch/gfx1201.hpp"
#include "lse/dispatch/arch/tuning_types.hpp"

namespace lse::dispatch::arch {

inline constexpr std::array<const Tuning*, 2> kTunings{
    &gfx1201::kTuning,
    &gfx1151::kTuning,
};

[[nodiscard]] constexpr const Tuning& tuning(std::string_view arch) noexcept {
  for (const Tuning* t : kTunings)
    if (t->arch == arch) return *t;
  return generic::kTuning;
}

[[nodiscard]] constexpr std::span<const Tuning* const> tunings() noexcept {
  return kTunings;
}

// The first row of one rule that `match` accepts: the device's own rows when
// `arch` names a part, then every part's in kTunings order. For rules the
// graph selects by shape before it knows the device; whatever selected the
// row checks its device fields afterwards.
template <auto Member, class Match>
[[nodiscard]] constexpr auto first_rule(std::string_view arch, Match&& match) {
  using Row = typename std::remove_cvref_t<
      decltype(std::declval<const Tuning&>().*Member)>::element_type;
  const Row* found = nullptr;
  if (!arch.empty())
    for (const Row& row : tuning(arch).*Member)
      if (match(row)) return &row;
  for (const Tuning* t : kTunings)
    for (const Row& row : t->*Member)
      if (match(row)) return &row;
  return found;
}

// Whether any part's rows of one rule accept `match`.
template <auto Member, class Match>
[[nodiscard]] constexpr bool any_rule(Match&& match) {
  for (const Tuning* t : kTunings)
    for (const auto& row : t->*Member)
      if (match(row)) return true;
  return false;
}

}  // namespace lse::dispatch::arch

namespace lse::dispatch::attention_shapes {

[[nodiscard]] constexpr bool flash_retain_far_cache(std::string_view arch,
    std::uint32_t rows, std::uint64_t capacity) noexcept {
  for (const auto& rule : arch::tuning(arch).flash_cache)
    if (arch == rule.arch && rows >= rule.min_rows && capacity >= rule.min_capacity)
      return true;
  return false;
}

}  // namespace lse::dispatch::attention_shapes
