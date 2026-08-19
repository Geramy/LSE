#include "lse/opt/measurements.hpp"

#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>

namespace lse::opt {

namespace {

struct Table {
  mutable std::shared_mutex mu;
  std::unordered_map<std::string, backend::KernelResources> rows;
};

Table& table() {
  static Table t;
  return t;
}

}  // namespace

KernelMeasurements& KernelMeasurements::instance() noexcept {
  static KernelMeasurements k;
  return k;
}

void KernelMeasurements::record(std::string_view entry,
                                const backend::KernelResources& r) {
  if (entry.empty() || !r.any()) return;
  Table& t = table();
  const std::unique_lock lock(t.mu);
  t.rows[std::string(entry)] = r;
}

backend::KernelResources KernelMeasurements::lookup(
    std::string_view entry) const {
  Table& t = table();
  const std::shared_lock lock(t.mu);
  const auto it = t.rows.find(std::string(entry));
  return it == t.rows.end() ? backend::KernelResources{} : it->second;
}

bool KernelMeasurements::known(std::string_view entry) const {
  Table& t = table();
  const std::shared_lock lock(t.mu);
  return t.rows.count(std::string(entry)) != 0;
}

std::size_t KernelMeasurements::size() const noexcept {
  Table& t = table();
  const std::shared_lock lock(t.mu);
  return t.rows.size();
}

void KernelMeasurements::clear() {
  Table& t = table();
  const std::unique_lock lock(t.mu);
  t.rows.clear();
}

}  // namespace lse::opt
