// Where a peer is, as a string the engine never interprets.
//
// An endpoint names a *place to reach*, never a technology. The scheme selects
// which registered transport parses the rest; everything after it is that
// transport's own vocabulary — an address and a port, a device path, a pipe
// name, a tuning knob. Nothing above this header may branch on the scheme, and
// no method here answers "is this RDMA": that question is asked of
// Capabilities.
#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "lse/core/status.hpp"

namespace lse::comm {

class Endpoint {
 public:
  Endpoint() = default;

  // "tcp://10.10.13.8:29500?ctrl_ring=65536", "unix:///run/lse.sock?ctrl_ring=4096",
  // "tb5:///dev/odl_tb5_0?stream=7". A bare name with no "://" is taken as the
  // scheme, so a config may say `tcp` as well as `tcp://` — the rule
  // src/dist/transport.cpp already follows.
  //
  // Neither a key nor a value may contain '?', '&' or '='; there is no escape
  // syntax, because a transport knob that needs one is a path, not a knob.
  [[nodiscard]] static Result<Endpoint> parse(std::string_view text);

  // Options given out of band win over options in the query string, so a
  // launch flag overrides a stored endpoint without rewriting it. The pair
  // vector is deliberately dist::TransportConfig::options' own type.
  [[nodiscard]] static Result<Endpoint> parse(
      std::string_view text,
      const std::vector<std::pair<std::string, std::string>>& options);

  [[nodiscard]] std::string_view scheme() const noexcept { return scheme_; }
  [[nodiscard]] std::string_view authority() const noexcept {
    return authority_;
  }
  [[nodiscard]] std::string_view path() const noexcept { return path_; }

  [[nodiscard]] std::string_view option(
      std::string_view key, std::string_view fallback = {}) const noexcept;
  [[nodiscard]] Result<std::uint64_t> option_u64(std::string_view key,
                                                 std::uint64_t fallback) const;

  // Every key a caller passed, so a transport can refuse one it does not know.
  [[nodiscard]] std::span<const std::pair<std::string, std::string>> options()
      const noexcept {
    return options_;
  }

  // Round-trips through parse(). The form that is stored, logged, and folded
  // into a fingerprint.
  [[nodiscard]] const std::string& str() const noexcept { return text_; }

 private:
  std::string text_;
  std::string scheme_;
  std::string authority_;
  std::string path_;
  std::vector<std::pair<std::string, std::string>> options_;
};

// Read by the common layer of every transport: ctrl_ring, max_inflight,
// deadline_ms, offer_ms, frame_max, bind.
[[nodiscard]] std::span<const std::string_view> common_option_keys() noexcept;

// A typo'd tuning knob that silently does nothing is the worst kind of
// configuration, so an unknown key is an error at open, never a no-op. `extra`
// is what this transport reads beyond the common set.
[[nodiscard]] Status check_options(const Endpoint& ep,
                                  std::span<const std::string_view> extra);

// getaddrinfo blocks and nothing else in this module does, so resolution is an
// explicit call the caller makes before entering its loop. An Endpoint handed
// to listen()/connect() must already carry a literal address.
[[nodiscard]] Result<std::vector<Endpoint>> resolve(const Endpoint& ep,
                                                    std::uint32_t timeout_ms);

}  // namespace lse::comm
