#include "variant_book.hpp"

#include <cerrno>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "lse/graph/jit.hpp"

namespace lse::graph::detail {
namespace {

std::uint64_t mix(std::uint64_t h, std::uint64_t v) noexcept {
  h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
  return h * 0xff51afd7ed558ccdull;
}

}  // namespace

VariantBook::VariantBook() {
  const char* mode = std::getenv("LSE_AUTOTUNE");
  enabled_ = mode == nullptr || (std::strcmp(mode, "off") != 0 &&
                                 std::strcmp(mode, "0") != 0);
  std::uint64_t budget_ms = 20000;
  if (const char* b = std::getenv("LSE_AUTOTUNE_BUDGET_MS")) {
    budget_ms = std::strtoull(b, nullptr, 10);
  }
  budget_ns_ = budget_ms * 1000000ull;
  path_ = default_cache_dir() + "/variants-" +
          std::string(kernel_cache_version()) + ".txt";
}

std::uint64_t VariantBook::key(std::uint64_t base_identity, std::uint32_t offered,
                               std::string_view compiler_identity) {
  std::uint64_t h = mix(0x76617269616e7473ull, base_identity);
  h = mix(h, offered);
  for (const char c : compiler_identity) h = mix(h, static_cast<unsigned char>(c));
  return mix(h, compiler_identity.size());
}

void VariantBook::load_locked() {
  if (loaded_) return;
  loaded_ = true;
  std::FILE* f = std::fopen(path_.c_str(), "r");
  if (f == nullptr) return;
  char line[512];
  while (std::fgets(line, sizeof(line), f) != nullptr) {
    unsigned long long k = 0;
    unsigned v = 0;
    // Later lines win: a decision measured again is appended, not rewritten.
    if (std::sscanf(line, "%llx %u", &k, &v) == 2) decided_[k] = v;
  }
  std::fclose(f);
}

std::optional<std::uint32_t> VariantBook::decided(std::uint64_t key) {
  const std::lock_guard lock(mu_);
  load_locked();
  if (const auto it = decided_.find(key); it != decided_.end()) return it->second;
  return std::nullopt;
}

void VariantBook::record(std::uint64_t key, std::uint32_t variant,
                         std::string_view note) {
  const std::lock_guard lock(mu_);
  load_locked();
  decided_[key] = variant;
  std::FILE* f = std::fopen(path_.c_str(), "a");
  if (f == nullptr) {
    std::fprintf(stderr, "lse: cannot write variant decisions to %s: %s\n",
                 path_.c_str(), std::strerror(errno));
    return;
  }
  // One write per line, so processes sharing the cache interleave whole lines.
  std::string text = std::to_string(variant);
  char head[32];
  std::snprintf(head, sizeof(head), "%016" PRIx64 " ", key);
  text = std::string(head) + text + " " + std::string(note) + "\n";
  std::fwrite(text.data(), 1, text.size(), f);
  std::fclose(f);
}

void VariantBook::hold(std::uint64_t key, std::uint32_t variant) {
  const std::lock_guard lock(mu_);
  load_locked();
  decided_[key] = variant;
}

bool VariantBook::has_budget() const noexcept { return spent_ns_ < budget_ns_; }

void VariantBook::spend(std::uint64_t ns) noexcept { spent_ns_ += ns; }

}  // namespace lse::graph::detail
