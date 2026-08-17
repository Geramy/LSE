#include "lse/communication/transport.hpp"

#include <map>
#include <mutex>
#include <string>

namespace lse::comm {

namespace {

struct Registry {
  std::mutex mu;
  std::map<std::string, TransportFactory, std::less<>> factories;
};

Registry& registry() {
  static Registry r;
  return r;
}

// "tcp://10.0.0.1:29500" -> "tcp". A bare name with no "://" is taken as the
// scheme so a config can say `tcp` as well as `tcp://`.
std::string_view scheme_of(std::string_view endpoint) noexcept {
  const std::size_t query = endpoint.find('?');
  if (query != std::string_view::npos) endpoint = endpoint.substr(0, query);
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
    for (const auto& entry : r.factories) {
      if (!known.empty()) known += ", ";
      known += entry.first;
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
  for (const auto& entry : r.factories) out.push_back(entry.first);
  return out;
}

}  // namespace lse::comm
