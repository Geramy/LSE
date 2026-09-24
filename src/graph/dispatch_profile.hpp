#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <map>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace lse::graph::detail {

enum class DispatchProfileMode { kOff, kSubmit, kSerial, kInvalid };

inline DispatchProfileMode dispatch_profile_mode(const char* value) {
  if (value == nullptr || std::string_view(value).empty() ||
      std::string_view(value) == "off") return DispatchProfileMode::kOff;
  if (std::string_view(value) == "submit") return DispatchProfileMode::kSubmit;
  if (std::string_view(value) == "serial") return DispatchProfileMode::kSerial;
  return DispatchProfileMode::kInvalid;
}

struct DispatchProfileSample {
  std::uint64_t count = 0;
  std::uint64_t failures = 0;
  std::uint64_t submit_ns = 0;
  std::uint64_t completion_ns = 0;
  std::uint64_t predrain_ns = 0;
  std::uint64_t max_submit_ns = 0;
  std::uint64_t max_completion_ns = 0;
};

// Diagnostic-only aggregation. Bound both the number and length of keys, so a
// long-running server with changing shapes cannot grow its profile indefinitely.
class DispatchProfile {
 public:
  static constexpr std::size_t kMaxShapes = 512;
  static constexpr std::size_t kMaxKeyBytes = 4096;
  using Row = std::pair<std::string, DispatchProfileSample>;

  void record(std::string key, std::uint64_t submit_ns,
              std::uint64_t completion_ns, std::uint64_t predrain_ns,
              bool succeeded) {
    const std::lock_guard lock(mutex_);
    if (key.size() > kMaxKeyBytes ||
        (!samples_.contains(key) && samples_.size() >= kMaxShapes)) {
      key = "<other-shapes>";
    }
    auto& s = samples_[std::move(key)];
    ++s.count;
    if (!succeeded) ++s.failures;
    s.submit_ns += submit_ns;
    s.completion_ns += completion_ns;
    s.predrain_ns += predrain_ns;
    s.max_submit_ns = std::max(s.max_submit_ns, submit_ns);
    s.max_completion_ns = std::max(s.max_completion_ns, completion_ns);
  }

  std::vector<Row> rows(bool serial) const {
    const std::lock_guard lock(mutex_);
    std::vector<Row> out(samples_.begin(), samples_.end());
    std::sort(out.begin(), out.end(), [serial](const Row& a, const Row& b) {
      const auto x = serial ? a.second.completion_ns : a.second.submit_ns;
      const auto y = serial ? b.second.completion_ns : b.second.submit_ns;
      return x != y ? x > y : a.first < b.first;
    });
    return out;
  }

  void print(FILE* file, bool serial) const {
    const auto snapshot = rows(serial);
    if (snapshot.empty()) return;
    std::fprintf(file,
        "[dispatch-profile] mode=%s clock=host-steady gpu-timestamps=no "
        "forced-serialization=%s shapes=%zu max-shapes=%zu "
        "completion-measured=%s "
        "completion-includes-submit-flush-execution-and-wait=yes\n",
        serial ? "serial" : "submit", serial ? "yes" : "no",
        snapshot.size(), kMaxShapes, serial ? "yes" : "no");
    for (const auto& [key, s] : snapshot) {
      const double count = static_cast<double>(s.count);
      std::fprintf(file,
          "[dispatch-profile] n=%llu failed=%llu submit_ms=%.6f "
          "submit_avg_us=%.3f submit_max_us=%.3f completion_ms=%.6f "
          "completion_avg_us=%.3f completion_max_us=%.3f "
          "excluded_predrain_ms=%.6f | %s\n",
          static_cast<unsigned long long>(s.count),
          static_cast<unsigned long long>(s.failures),
          static_cast<double>(s.submit_ns) / 1e6,
          static_cast<double>(s.submit_ns) / count / 1e3,
          static_cast<double>(s.max_submit_ns) / 1e3,
          static_cast<double>(s.completion_ns) / 1e6,
          static_cast<double>(s.completion_ns) / count / 1e3,
          static_cast<double>(s.max_completion_ns) / 1e3,
          static_cast<double>(s.predrain_ns) / 1e6, key.c_str());
    }
  }

 private:
  mutable std::mutex mutex_;
  std::map<std::string, DispatchProfileSample> samples_;
};

}  // namespace lse::graph::detail
