// Persisting a measured profile, so a normal start reads it rather than
// re-measuring.
//
// Same shape as the kernel cache: a directory of entries keyed by a stable
// identity, written through a temporary and renamed, and validated on read
// rather than trusted. Invalidation is by fingerprint, never by age — a
// profile does not become wrong because time passed, it becomes wrong because
// the pool or the topology changed.
//
// The text form is also the wire form: a rank serializes its own device profile
// with these functions and the pool exchanges the bytes, so there is one
// encoder rather than one for disk and another for the fabric.
#pragma once

#include <string>
#include <string_view>

#include "lse/core/status.hpp"
#include "lse/probe/profile.hpp"

namespace lse::probe {

// $LSE_PROFILE_DIR, else $XDG_CACHE_HOME/lse/profiles, else ~/.cache/lse/profiles.
[[nodiscard]] std::string default_profile_dir();

[[nodiscard]] std::string profile_path(std::string_view dir,
                                       std::string_view fingerprint);

[[nodiscard]] std::string serialize_pool_profile(const PoolProfile& pool);
[[nodiscard]] Result<PoolProfile> parse_pool_profile(std::string_view text);

[[nodiscard]] std::string serialize_device_profile(const DeviceProfile& device);
[[nodiscard]] Result<DeviceProfile> parse_device_profile(std::string_view text);

// Reads the entry for `fingerprint`. A missing, unreadable or mismatched entry
// is kNotFound, not an error worth propagating — the caller measures instead.
[[nodiscard]] Result<PoolProfile> load_pool_profile(std::string_view fingerprint,
                                                    std::string_view dir);
Status save_pool_profile(const PoolProfile& pool, std::string_view dir);

}  // namespace lse::probe
