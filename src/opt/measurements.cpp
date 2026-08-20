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
  std::unordered_map<std::string, backend::KernelCensus> counts;
  std::unordered_map<std::string, TrafficModel> intents;
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

void KernelMeasurements::record(std::string_view entry,
                                const backend::KernelCensus& c) {
  if (entry.empty() || !c.any()) return;
  Table& t = table();
  const std::unique_lock lock(t.mu);
  t.counts[std::string(entry)] = c;
}

void KernelMeasurements::record(std::string_view entry,
                                const TrafficModel& t) {
  if (entry.empty() || !t.stated) return;
  Table& tab = table();
  const std::unique_lock lock(tab.mu);
  tab.intents[std::string(entry)] = t;
}

TrafficModel KernelMeasurements::traffic(std::string_view entry) const {
  Table& t = table();
  const std::shared_lock lock(t.mu);
  const auto it = t.intents.find(std::string(entry));
  return it == t.intents.end() ? TrafficModel{} : it->second;
}

backend::KernelCensus KernelMeasurements::census(
    std::string_view entry) const {
  Table& t = table();
  const std::shared_lock lock(t.mu);
  const auto it = t.counts.find(std::string(entry));
  return it == t.counts.end() ? backend::KernelCensus{} : it->second;
}

bool KernelMeasurements::census_known(std::string_view entry) const {
  Table& t = table();
  const std::shared_lock lock(t.mu);
  return t.counts.count(std::string(entry)) != 0;
}

std::size_t KernelMeasurements::census_size() const noexcept {
  Table& t = table();
  const std::shared_lock lock(t.mu);
  return t.counts.size();
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
  t.counts.clear();
  t.intents.clear();
}

}  // namespace lse::opt
