#include "lse/dist/transport.hpp"

#include <algorithm>
#include <map>
#include <mutex>
#include <string>

namespace lse::dist {

namespace {

struct Registry {
  std::mutex mu;
  std::map<std::string, TransportFactory, std::less<>> factories;
};

Registry& registry() {
  static Registry r;
  return r;
}

// "loopback://group/8" -> "loopback". A bare name with no "://" is taken as
// the scheme so a config can say `loopback` as well as `loopback://`.
std::string_view scheme_of(std::string_view endpoint) noexcept {
  const std::size_t sep = endpoint.find("://");
  return sep == std::string_view::npos ? endpoint : endpoint.substr(0, sep);
}

}  // namespace

void register_transport(std::string_view scheme, TransportFactory factory) {
  Registry& r = registry();
  std::lock_guard lock(r.mu);
  r.factories.emplace(std::string(scheme), factory);
}

Result<std::unique_ptr<ITransport>> create_transport(
    std::string_view endpoint) {
  Registry& r = registry();
  std::lock_guard lock(r.mu);
  auto it = r.factories.find(scheme_of(endpoint));
  if (it == r.factories.end()) {
    std::string known;
    for (const auto& [key, _] : r.factories) {
      if (!known.empty()) known += ", ";
      known += key;
    }
    return LSE_ERROR(kNotFound, "no transport for '", std::string(endpoint),
                     "'; registered: ", known.empty() ? "(none)" : known);
  }
  return it->second();
}

std::vector<std::string> available_transports() {
  Registry& r = registry();
  std::lock_guard lock(r.mu);
  std::vector<std::string> out;
  out.reserve(r.factories.size());
  for (const auto& [key, _] : r.factories) out.push_back(key);
  return out;
}

}  // namespace lse::dist
