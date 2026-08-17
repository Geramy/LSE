#include "lse/communication/endpoint.hpp"

#include <netdb.h>
#include <sys/socket.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <ctime>
#include <memory>
#include <string>

#include "lse/communication/transport.hpp"

namespace lse::comm {

namespace {

constexpr std::array<std::string_view, 6> kCommonKeys{
    "ctrl_ring", "max_inflight", "deadline_ms", "offer_ms", "frame_max", "bind"};

bool scheme_char(char c) noexcept {
  return (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '-' ||
         c == '.' || c == '+';
}

Status parse_query(std::string_view query,
                   std::vector<std::pair<std::string, std::string>>& out) {
  while (!query.empty()) {
    const std::size_t amp = query.find('&');
    const std::string_view item =
        amp == std::string_view::npos ? query : query.substr(0, amp);
    query = amp == std::string_view::npos ? std::string_view{}
                                          : query.substr(amp + 1);
    if (item.empty()) continue;
    const std::size_t eq = item.find('=');
    if (eq == std::string_view::npos || eq == 0) {
      return LSE_ERROR(kInvalidArgument, "endpoint option '", std::string(item),
                       "' is not key=value");
    }
    const std::string key(item.substr(0, eq));
    const std::string value(item.substr(eq + 1));
    if (value.find_first_of("?&=") != std::string::npos) {
      return LSE_ERROR(kInvalidArgument, "endpoint option '", key,
                       "' has a value containing ? & or =, which this "
                       "vocabulary has no escape for");
    }
    const auto dup = std::find_if(
        out.begin(), out.end(),
        [&](const std::pair<std::string, std::string>& kv) {
          return kv.first == key;
        });
    if (dup != out.end()) {
      return LSE_ERROR(kInvalidArgument, "endpoint option '", key,
                       "' is given twice");
    }
    out.emplace_back(key, value);
  }
  return OkStatus();
}

// "10.0.0.1:29500" -> {"10.0.0.1", "29500"}; "[::1]:29500" -> {"::1", "29500"}.
Status split_host_port(std::string_view authority, std::string& host,
                       std::string& port) {
  if (!authority.empty() && authority.front() == '[') {
    const std::size_t end = authority.find(']');
    if (end == std::string_view::npos) {
      return LSE_ERROR(kInvalidArgument, "endpoint authority '",
                       std::string(authority), "' opens '[' and never closes");
    }
    host = std::string(authority.substr(1, end - 1));
    std::string_view rest = authority.substr(end + 1);
    if (!rest.empty() && rest.front() == ':') rest.remove_prefix(1);
    port = std::string(rest);
    return OkStatus();
  }
  const std::size_t colon = authority.rfind(':');
  if (colon == std::string_view::npos) {
    host = std::string(authority);
    port.clear();
    return OkStatus();
  }
  host = std::string(authority.substr(0, colon));
  port = std::string(authority.substr(colon + 1));
  return OkStatus();
}

std::string canonical(std::string_view scheme, std::string_view authority,
                      std::string_view path,
                      std::span<const std::pair<std::string, std::string>> opt) {
  std::string out(scheme);
  out += "://";
  out += authority;
  out += path;
  bool first = true;
  for (const auto& [k, v] : opt) {
    out += first ? '?' : '&';
    first = false;
    out += k;
    out += '=';
    out += v;
  }
  return out;
}

}  // namespace

Result<Endpoint> Endpoint::parse(std::string_view text) {
  return parse(text, {});
}

Result<Endpoint> Endpoint::parse(
    std::string_view text,
    const std::vector<std::pair<std::string, std::string>>& options) {
  if (text.empty()) {
    return LSE_ERROR(kInvalidArgument, "an endpoint may not be empty");
  }
  std::string_view head = text;
  std::string_view query;
  if (const std::size_t q = text.find('?'); q != std::string_view::npos) {
    head = text.substr(0, q);
    query = text.substr(q + 1);
  }

  Endpoint ep;
  const std::size_t sep = head.find("://");
  // A bare name with no "://" is taken as the scheme so a config can say `tcp`
  // as well as `tcp://`.
  std::string_view rest;
  if (sep == std::string_view::npos) {
    ep.scheme_ = std::string(head);
  } else {
    ep.scheme_ = std::string(head.substr(0, sep));
    rest = head.substr(sep + 3);
  }
  if (ep.scheme_.empty()) {
    return LSE_ERROR(kInvalidArgument, "endpoint '", std::string(text),
                     "' has no scheme");
  }
  for (const char c : ep.scheme_) {
    if (!scheme_char(c)) {
      return LSE_ERROR(kInvalidArgument, "endpoint scheme '", ep.scheme_,
                       "' is not lower-case [a-z0-9.+-]");
    }
  }

  const std::size_t slash = rest.find('/');
  if (slash == std::string_view::npos) {
    ep.authority_ = std::string(rest);
  } else {
    ep.authority_ = std::string(rest.substr(0, slash));
    ep.path_ = std::string(rest.substr(slash));
  }

  LSE_RETURN_IF_ERROR(parse_query(query, ep.options_));

  // Options given out of band win over options in the query string, so a launch
  // flag overrides a stored endpoint without rewriting it.
  for (const auto& [k, v] : options) {
    if (k.empty()) {
      return LSE_ERROR(kInvalidArgument, "an endpoint option key may not be empty");
    }
    if (k.find_first_of("?&=") != std::string::npos ||
        v.find_first_of("?&=") != std::string::npos) {
      return LSE_ERROR(kInvalidArgument, "endpoint option '", k,
                       "' contains ? & or =, which this vocabulary has no "
                       "escape for");
    }
    const auto it = std::find_if(
        ep.options_.begin(), ep.options_.end(),
        [&](const std::pair<std::string, std::string>& kv) {
          return kv.first == k;
        });
    if (it != ep.options_.end()) {
      it->second = v;
    } else {
      ep.options_.emplace_back(k, v);
    }
  }

  ep.text_ = canonical(ep.scheme_, ep.authority_, ep.path_, ep.options_);
  return ep;
}

std::string_view Endpoint::option(std::string_view key,
                                  std::string_view fallback) const noexcept {
  for (const auto& [k, v] : options_) {
    if (k == key) return v;
  }
  return fallback;
}

Result<std::uint64_t> Endpoint::option_u64(std::string_view key,
                                           std::uint64_t fallback) const {
  const std::string_view raw = option(key);
  if (raw.empty()) return fallback;
  std::uint64_t value = 0;
  for (const char c : raw) {
    if (c < '0' || c > '9') {
      return LSE_ERROR(kInvalidArgument, "endpoint option '", std::string(key),
                       "=", std::string(raw), "' is not a non-negative integer");
    }
    const auto digit = static_cast<std::uint64_t>(c - '0');
    if (value > (~std::uint64_t{0} - digit) / 10u) {
      return LSE_ERROR(kOutOfRange, "endpoint option '", std::string(key), "=",
                       std::string(raw), "' does not fit in 64 bits");
    }
    value = value * 10u + digit;
  }
  return value;
}

std::span<const std::string_view> common_option_keys() noexcept {
  return kCommonKeys;
}

Status check_options(const Endpoint& ep,
                     std::span<const std::string_view> extra) {
  for (const auto& [k, v] : ep.options()) {
    const bool known =
        std::find(kCommonKeys.begin(), kCommonKeys.end(), k) !=
            kCommonKeys.end() ||
        std::find(extra.begin(), extra.end(), k) != extra.end();
    if (known) continue;
    std::string listed;
    for (const std::string_view name : kCommonKeys) {
      if (!listed.empty()) listed += ", ";
      listed += name;
    }
    for (const std::string_view name : extra) {
      listed += ", ";
      listed += name;
    }
    return LSE_ERROR(kInvalidArgument, "endpoint '", ep.str(),
                     "' sets unknown option '", k, "'; ",
                     std::string(ep.scheme()), " reads: ", listed);
  }
  return OkStatus();
}

Result<std::vector<Endpoint>> resolve(const Endpoint& ep,
                                      std::uint32_t timeout_ms) {
  if (ep.authority().empty()) return std::vector<Endpoint>{ep};

  // Whether the authority is a host at all is the transport's to say. Guessing
  // it from the string would make one caller flow — resolve, select, connect —
  // correct for an address and wrong for a pipe name, which is the caller
  // branching on transport that this seam exists to remove.
  LSE_ASSIGN_OR(std::unique_ptr<ITransport> transport,
                create_transport(ep.str()));
  if (!transport->authority_is_host()) return std::vector<Endpoint>{ep};

  std::string host;
  std::string port;
  LSE_RETURN_IF_ERROR(split_host_port(ep.authority(), host, port));
  if (host.empty()) return std::vector<Endpoint>{ep};

  addrinfo hints{};
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;

  // getaddrinfo has no timeout of its own; this bounds only how long a
  // temporary failure is retried, which is the part a caller can control.
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
  addrinfo* res = nullptr;
  int rc = 0;
  for (;;) {
    rc = ::getaddrinfo(host.c_str(), port.empty() ? nullptr : port.c_str(),
                       &hints, &res);
    if (rc != EAI_AGAIN || std::chrono::steady_clock::now() >= deadline) break;
    const timespec nap{0, 50'000'000};
    ::nanosleep(&nap, nullptr);
  }
  if (rc != 0) {
    return LSE_ERROR(kNotFound, "cannot resolve '", host,
                     "': ", ::gai_strerror(rc));
  }

  std::vector<Endpoint> out;
  for (const addrinfo* a = res; a != nullptr; a = a->ai_next) {
    std::array<char, NI_MAXHOST> numeric{};
    std::array<char, NI_MAXSERV> service{};
    if (::getnameinfo(a->ai_addr, a->ai_addrlen, numeric.data(), numeric.size(),
                      service.data(), service.size(),
                      NI_NUMERICHOST | NI_NUMERICSERV) != 0) {
      continue;
    }
    std::string authority;
    if (a->ai_family == AF_INET6) {
      authority = "[" + std::string(numeric.data()) + "]";
    } else {
      authority = numeric.data();
    }
    if (!port.empty()) {
      authority += ":";
      authority += service.data();
    }
    auto one = Endpoint::parse(canonical(ep.scheme(), authority, ep.path(),
                                         ep.options()));
    if (!one.ok()) continue;
    out.push_back(one.release());
  }
  ::freeaddrinfo(res);

  if (out.empty()) {
    return LSE_ERROR(kNotFound, "'", host,
                     "' resolved to no address this machine can use");
  }
  return out;
}

}  // namespace lse::comm
