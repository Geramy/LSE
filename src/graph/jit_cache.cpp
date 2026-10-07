#include "lse/graph/jit.hpp"
#include "lse/opt/measurements.hpp"
#include "dispatch_profile.hpp"
#include "lse/core/debug.hpp"
#include "resource_profile.hpp"

#include <unistd.h>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <charconv>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <fstream>
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace lse::graph {

namespace fs = std::filesystem;

namespace {

std::vector<std::byte> read_file(const fs::path& path) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) return {};
  const auto size = static_cast<std::streamsize>(in.tellg());
  if (size <= 0) return {};
  std::vector<std::byte> out(static_cast<std::size_t>(size));
  in.seekg(0);
  in.read(reinterpret_cast<char*>(out.data()), size);
  if (in.gcount() != size) return {};
  return out;
}

bool source_matches(const fs::path& path, std::string_view source) {
  const auto bytes = read_file(path);
  return bytes.size() == source.size() &&
         std::memcmp(bytes.data(), source.data(), source.size()) == 0;
}

std::uint64_t fnv(std::string_view s) noexcept {
  std::uint64_t h = 1469598103934665603ull;
  for (char ch : s) {
    h ^= static_cast<unsigned char>(ch);
    h *= 1099511628211ull;
  }
  return h;
}

// A second fingerprint of a source, independent of fnv(): with the length and
// fnv() it stands in for the complete text when a loaded kernel is matched
// again, so a resident kernel does not keep its source in memory.
std::uint64_t fingerprint(std::string_view s) noexcept {
  std::uint64_t h = 0x9e3779b97f4a7c15ull ^ s.size();
  for (char ch : s) {
    h = (h ^ static_cast<unsigned char>(ch)) * 0xff51afd7ed558ccdull;
    h ^= h >> 32;
  }
  return h;
}

std::uint64_t mix(std::uint64_t h, std::uint64_t v) noexcept {
  h ^= v;
  h *= 1099511628211ull;
  return h;
}

struct DiskMeta {
  std::string arch;
  std::uint64_t source_hash = 0;
  std::string entry;
  std::vector<backend::KernelResources> resources;
  std::vector<backend::KernelCensus> census;
  std::uint32_t resource_version = 0;
  std::string engine_version;
};

// A fact renders as its number or as "-", never as a 0 standing in for
// "the toolchain declined". Reading a "-" back as kUnknown is what keeps a
// warm start from inventing a spill count the compiler never reported.
void write_fact(std::ostream& out, const backend::DeviceFact<std::uint32_t>& f) {
  if (f.known()) {
    out << ' ' << f.value;
  } else {
    out << " -";
  }
}

backend::DeviceFact<std::uint32_t> read_fact(std::istream& in) {
  std::string token;
  if (!(in >> token) || token == "-") return {};
  try {
    return backend::DeviceFact<std::uint32_t>::queried(
        static_cast<std::uint32_t>(std::stoul(token)));
  } catch (...) {
    return {};
  }
}

// One line per kernel the object defines, tagged so the three fixed header
// lines stay where they were and a meta file written before this existed still
// reads — it simply carries no resources, which is the correct answer for it.
void write_resources(std::ostream& out,
                     const std::vector<backend::KernelResources>& all) {
  for (const backend::KernelResources& r : all) {
    out << "res " << (r.entry.empty() ? "-" : r.entry);
    write_fact(out, r.vector_registers);
    write_fact(out, r.scalar_registers);
    write_fact(out, r.accum_registers);
    write_fact(out, r.workgroup_segment_bytes);
    write_fact(out, r.private_segment_bytes);
    write_fact(out, r.vector_spills);
    write_fact(out, r.scalar_spills);
    write_fact(out, r.kernarg_segment_bytes);
    write_fact(out, r.max_flat_workgroup_size);
    write_fact(out, r.wavefront_size);
    if (r.required_workgroup_size.known()) {
      out << ' ' << r.required_workgroup_size.value[0] << ' '
          << r.required_workgroup_size.value[1] << ' '
          << r.required_workgroup_size.value[2];
    } else {
      out << " - - -";
    }
    out << '\n';
  }
}

// The census as two tags: one scalar line per kernel and one line per access
// width, so a width the next ISA adds costs a line and not a format. Both carry
// the entry name, so the lines are order-independent and a reader that meets an
// `acc` before its `cen` still files it correctly.
void write_census(std::ostream& out,
                  const std::vector<backend::KernelCensus>& all) {
  for (const backend::KernelCensus& c : all) {
    const std::string name = c.entry.empty() ? "-" : c.entry;
    out << "cen " << name;
    write_fact(out, c.instructions);
    write_fact(out, c.vector_alu);
    write_fact(out, c.scalar_alu);
    write_fact(out, c.dot_products);
    write_fact(out, c.fused_multiply_adds);
    write_fact(out, c.matrix_ops);
    write_fact(out, c.lane_exchanges);
    write_fact(out, c.branches);
    write_fact(out, c.backward_branches);
    write_fact(out, c.memory_waits);
    write_fact(out, c.deepest_load_batch);
    write_fact(out, c.serializing_waits);
    write_fact(out, c.unclassified);
    // Appended, not inserted: a note written before this field existed still
    // reads, and its missing tail comes back unknown rather than zero.
    write_fact(out, c.multiply_accumulates);
    out << '\n';
    const std::pair<const char*, const backend::AccessCensus*> spaces[] = {
        {"gl", &c.global_loads},   {"gs", &c.global_stores},
        {"sl", &c.shared_loads},   {"ss", &c.shared_stores},
        {"pl", &c.private_loads},  {"ps", &c.private_stores},
        {"kl", &c.scalar_loads},
    };
    for (const auto& [tag, access] : spaces) {
      for (const backend::AccessCensus::Width& w : access->widths) {
        out << "acc " << name << ' ' << tag << ' ' << w.bytes << ' ' << w.count
            << ' ' << w.looped << '\n';
      }
    }
  }
}

backend::KernelCensus* census_for(std::vector<backend::KernelCensus>* all,
                                  const std::string& entry) {
  for (backend::KernelCensus& c : *all) {
    if (c.entry == entry) return &c;
  }
  all->push_back(backend::KernelCensus{});
  all->back().entry = entry;
  return &all->back();
}

bool read_census_line(const std::string& line,
                      std::vector<backend::KernelCensus>* all) {
  std::istringstream in(line);
  std::string tag;
  if (!(in >> tag) || tag != "cen") return false;
  std::string entry;
  if (!(in >> entry)) return false;
  if (entry == "-") entry.clear();
  backend::KernelCensus* c = census_for(all, entry);
  c->instructions = read_fact(in);
  c->vector_alu = read_fact(in);
  c->scalar_alu = read_fact(in);
  c->dot_products = read_fact(in);
  c->fused_multiply_adds = read_fact(in);
  c->matrix_ops = read_fact(in);
  c->lane_exchanges = read_fact(in);
  c->branches = read_fact(in);
  c->backward_branches = read_fact(in);
  c->memory_waits = read_fact(in);
  c->deepest_load_batch = read_fact(in);
  c->serializing_waits = read_fact(in);
  c->unclassified = read_fact(in);
  c->multiply_accumulates = read_fact(in);
  return true;
}

bool read_access_line(const std::string& line,
                      std::vector<backend::KernelCensus>* all) {
  std::istringstream in(line);
  std::string tag;
  if (!(in >> tag) || tag != "acc") return false;
  std::string entry;
  std::string space;
  std::uint32_t bytes = 0;
  std::uint32_t count = 0;
  std::uint32_t looped = 0;
  if (!(in >> entry >> space >> bytes >> count >> looped)) return false;
  if (entry == "-") entry.clear();
  if (bytes == 0 || count == 0) return false;
  backend::KernelCensus* c = census_for(all, entry);
  backend::AccessCensus* access = nullptr;
  if (space == "gl") {
    access = &c->global_loads;
  } else if (space == "gs") {
    access = &c->global_stores;
  } else if (space == "sl") {
    access = &c->shared_loads;
  } else if (space == "ss") {
    access = &c->shared_stores;
  } else if (space == "pl") {
    access = &c->private_loads;
  } else if (space == "ps") {
    access = &c->private_stores;
  } else if (space == "kl") {
    access = &c->scalar_loads;
  }
  if (access == nullptr) return false;
  access->widths.push_back(
      backend::AccessCensus::Width{bytes, count, looped});
  return true;
}

bool read_resource_line(const std::string& line,
                        backend::KernelResources* out) {
  std::istringstream in(line);
  std::string tag;
  if (!(in >> tag) || tag != "res") return false;
  if (!(in >> out->entry)) return false;
  if (out->entry == "-") out->entry.clear();
  out->vector_registers = read_fact(in);
  out->scalar_registers = read_fact(in);
  out->accum_registers = read_fact(in);
  out->workgroup_segment_bytes = read_fact(in);
  out->private_segment_bytes = read_fact(in);
  out->vector_spills = read_fact(in);
  out->scalar_spills = read_fact(in);
  out->kernarg_segment_bytes = read_fact(in);
  out->max_flat_workgroup_size = read_fact(in);
  out->wavefront_size = read_fact(in);
  const auto x = read_fact(in);
  const auto y = read_fact(in);
  const auto z = read_fact(in);
  if (x.known() && y.known() && z.known()) {
    out->required_workgroup_size =
        backend::DeviceFact<std::array<std::uint32_t, 3>>::queried(
            {x.value, y.value, z.value});
  }
  return true;
}

constexpr std::string_view kMetaOwner = "lse-kernel-cache-v1";
constexpr std::string_view kAliasOwner = "lse-kernel-alias-v1";

bool unsigned_number(std::string_view text, std::uint64_t* out, int base = 10) {
  if (text.empty()) return false;
  const auto [end, ec] = std::from_chars(text.data(), text.data() + text.size(),
                                        *out, base);
  return ec == std::errc{} && end == text.data() + text.size();
}

using ReleaseVersion = std::array<std::uint32_t, 3>;
bool release_version(std::string_view text, ReleaseVersion* out) {
  for (std::size_t i = 0; i < out->size(); ++i) {
    const auto dot = text.find('.');
    if ((i + 1 == out->size()) != (dot == std::string_view::npos)) return false;
    const auto part = text.substr(0, dot);
    std::uint64_t value = 0;
    if (!unsigned_number(part, &value) || value > UINT32_MAX) return false;
    (*out)[i] = static_cast<std::uint32_t>(value);
    if (dot != std::string_view::npos) text.remove_prefix(dot + 1);
  }
  return true;
}

std::string artifact_stem(std::string_view version, std::uint64_t key) {
  return "lse-" + std::string(version) + "-" + std::to_string(key);
}

bool versioned_stem(std::string_view stem, std::string_view version) {
  const auto prefix = "lse-" + std::string(version) + "-";
  std::uint64_t key = 0;
  return stem.starts_with(prefix) &&
      unsigned_number(stem.substr(prefix.size()), &key);
}

bool regular_file(const fs::path& path) {
  std::error_code ec;
  return fs::is_regular_file(fs::symlink_status(path, ec)) && !ec;
}

bool complete_file(const fs::path& path) {
  std::error_code ec;
  return regular_file(path) && fs::file_size(path, ec) > 0 && !ec;
}

// Serializes cleanup and publication across cooperating processes, not JIT
// compilation. Each release has distinct filenames, including temporary files.
class CacheDiskLock {
 public:
  explicit CacheDiskLock(const fs::path& dir) {
    std::error_code ec;
    fs::create_directories(dir, ec);
    if (ec) return;
    const auto path = dir / ".lse-cache.lock";
    fd_ = ::open(path.c_str(), O_CREAT | O_RDWR | O_CLOEXEC | O_NOFOLLOW, 0600);
    if (fd_ < 0) return;
    struct stat info{};
    if (::fstat(fd_, &info) != 0 || !S_ISREG(info.st_mode)) {
      ::close(fd_);
      fd_ = -1;
      return;
    }
    int result;
    do { result = ::flock(fd_, LOCK_EX); } while (result != 0 && errno == EINTR);
    if (result != 0) { ::close(fd_); fd_ = -1; }
  }
  ~CacheDiskLock() {
    if (fd_ >= 0) { ::flock(fd_, LOCK_UN); ::close(fd_); }
  }
  CacheDiskLock(const CacheDiskLock&) = delete;
  CacheDiskLock& operator=(const CacheDiskLock&) = delete;
  explicit operator bool() const noexcept { return fd_ >= 0; }
 private:
  int fd_ = -1;
};

bool read_meta(const fs::path& path, DiskMeta* out) {
  if (!regular_file(path)) return false;
  std::ifstream in(path);
  if (!in) return false;
  *out = {};
  std::string hash;
  if (!std::getline(in, out->arch)) return false;
  if (out->arch == kMetaOwner) {
    ReleaseVersion version{};
    if (!std::getline(in, out->engine_version) ||
        !release_version(out->engine_version, &version) ||
        !std::getline(in, out->arch)) return false;
  }
  if (!std::getline(in, hash) || !std::getline(in, out->entry)) return false;
  if (out->arch.empty() || out->entry.empty() ||
      !unsigned_number(hash, &out->source_hash, 16)) return false;
  for (std::string line; std::getline(in, line);) {
    if (line.starts_with("resource_version ")) {
      std::istringstream version(line.substr(17));
      std::uint32_t value = 0;
      if (version >> value) out->resource_version = value;
      continue;
    }
    backend::KernelResources r;
    if (read_resource_line(line, &r)) {
      out->resources.push_back(std::move(r));
      continue;
    }
    if (read_census_line(line, &out->census)) continue;
    read_access_line(line, &out->census);
  }
  return true;
}

struct DiskAlias {
  std::string entry;
  std::string artifact;
  std::uint64_t hash = 0;
  std::string engine_version;
};

bool read_alias(const fs::path& path, DiskAlias* out) {
  if (!regular_file(path)) return false;
  std::ifstream in(path);
  if (!in || !std::getline(in, out->entry)) return false;
  if (out->entry == kAliasOwner) {
    ReleaseVersion version{};
    if (!std::getline(in, out->engine_version) ||
        !release_version(out->engine_version, &version) ||
        !std::getline(in, out->entry)) return false;
  }
  std::string hash, extra;
  if (out->entry.empty() || !std::getline(in, out->artifact) ||
      !std::getline(in, hash) || !unsigned_number(hash, &out->hash) ||
      (std::getline(in, extra) && !extra.empty())) return false;
  std::uint64_t key = 0;
  return out->engine_version.empty()
      ? unsigned_number(out->artifact, &key)
      : versioned_stem(out->artifact, out->engine_version);
}

bool legacy_entry(std::string_view entry) {
  for (const auto prefix : {"lse_loom_", "lse_fused_", "lse_phase_", "lse_body_"}) {
    std::uint64_t key = 0;
    if (entry.starts_with(prefix) &&
        unsigned_number(entry.substr(std::strlen(prefix)), &key)) return true;
  }
  return false;
}

void remove_older_cache_entries(const fs::path& dir) {
  ReleaseVersion current{};
  if (!release_version(kernel_cache_version(), &current)) return;
  CacheDiskLock lock(dir);
  if (!lock) return;
  struct Family { fs::path meta; DiskMeta info; std::vector<fs::path> files; };
  std::unordered_map<std::string, Family> older;
  std::error_code ec;
  for (fs::directory_iterator it(dir, ec), end; !ec && it != end; ++it) {
    const auto& path = it->path();
    if (path.extension() != ".meta" || !regular_file(path)) continue;
    DiskMeta meta;
    if (!read_meta(path, &meta)) continue;
    const auto stem = path.stem().string();
    if (meta.engine_version.empty()) {
      std::uint64_t key = 0;
      // Unmarked files are legacy only with the old numeric key, generated LSE
      // entry name, AMD architecture, and matching complete code object.
      if (!unsigned_number(stem, &key) || !legacy_entry(meta.entry) ||
          !meta.arch.starts_with("gfx") || meta.arch.size() <= 3 ||
          !std::all_of(meta.arch.begin() + 3, meta.arch.end(), [](char c) {
            return std::isxdigit(static_cast<unsigned char>(c));
          })) continue;
      const fs::path code = dir / (stem + "." +
          std::to_string(meta.source_hash) + ".co");
      if (!complete_file(code)) continue;
    } else {
      ReleaseVersion version{};
      if (!versioned_stem(stem, meta.engine_version) ||
          !release_version(meta.engine_version, &version) || !(version < current))
        continue;
      const fs::path code = dir / (stem + "." +
          std::to_string(meta.source_hash) + ".co");
      if (!complete_file(code)) continue;
    }
    older.emplace(stem, Family{path, std::move(meta), {path}});
  }
  if (older.empty()) return;
  for (fs::directory_iterator it(dir, ec), end; !ec && it != end; ++it) {
    const auto& path = it->path();
    if (!regular_file(path)) continue;
    if (path.extension() == ".alias") {
      DiskAlias alias;
      if (!read_alias(path, &alias)) continue;
      const auto family = older.find(alias.artifact);
      if (family == older.end() || alias.hash != family->second.info.source_hash ||
          alias.engine_version != family->second.info.engine_version) continue;
      std::uint64_t key = 0;
      const auto stem = path.stem().string();
      if (alias.engine_version.empty()
              ? (unsigned_number(stem, &key) && legacy_entry(alias.entry))
              : versioned_stem(stem, alias.engine_version))
        family->second.files.push_back(path);
      continue;
    }
    if (path.extension() == ".source") {
      if (auto family = older.find(path.stem().string()); family != older.end())
        family->second.files.push_back(path);
      continue;
    }
    if (path.extension() != ".co" || !complete_file(path)) continue;
    const auto name = path.stem().string();
    const auto dot = name.rfind('.');
    if (dot == std::string::npos) continue;
    std::uint64_t hash = 0;
    if (!unsigned_number(std::string_view(name).substr(dot + 1), &hash)) continue;
    if (auto family = older.find(name.substr(0, dot)); family != older.end())
      family->second.files.push_back(path);
  }
  for (const auto& item : older) {
    const auto& family = item.second;
    // Recheck the ownership record before deleting any family member. Partial
    // metadata and temporary files never authorize a cleanup.
    DiskMeta now;
    if (!read_meta(family.meta, &now) ||
        now.engine_version != family.info.engine_version ||
        now.arch != family.info.arch || now.entry != family.info.entry ||
        now.source_hash != family.info.source_hash) continue;
    for (const auto& path : family.files) {
      if (regular_file(path)) { std::error_code rm; fs::remove(path, rm); }
    }
  }
}

void record_structural_measurements(std::string_view alias,
                                    std::string_view canonical,
                                    std::span<const backend::KernelResources> resources,
                                    std::span<const backend::KernelCensus> census) {
  if (alias.empty() || alias == canonical) return;
  auto& measured = opt::KernelMeasurements::instance();
  for (auto r : resources) {
    if (r.entry != canonical) continue;
    r.entry = alias;
    measured.record(alias, r);
  }
  for (auto c : census) {
    if (c.entry != canonical) continue;
    c.entry = alias;
    measured.record(alias, c);
  }
}

// Hand every measurement already on disk to the optimizer, once, before the
// first kernel is emitted. Without this a decision made at emit time can only
// see kernels this process has already compiled, so the first emit of a run
// always falls back to the estimate — and the answer would then depend on how
// long the process had been running, which is exactly what must not happen.
void preload_measurements(const std::string& dir,
    const std::unordered_map<std::string, std::uint32_t>& resource_versions) {
  std::vector<DiskAlias> aliases;
  std::unordered_map<std::string, DiskMeta> metadata;
  std::error_code ec;
  for (fs::directory_iterator it(dir, ec), end; !ec && it != end; ++it) {
    if (it->path().extension() == ".alias") {
      DiskAlias alias;
      if (read_alias(it->path(), &alias) &&
          alias.engine_version == kernel_cache_version() &&
          versioned_stem(it->path().stem().string(), alias.engine_version))
        aliases.push_back(std::move(alias));
      continue;
    }
    if (it->path().extension() != ".meta") continue;
    DiskMeta meta;
    if (!read_meta(it->path(), &meta) ||
        meta.engine_version != kernel_cache_version() ||
        !versioned_stem(it->path().stem().string(), meta.engine_version)) continue;
    const auto version = resource_versions.find(meta.arch);
    if (version == resource_versions.end()) continue;
    // Stale resource-reader facts are not evidence, even within this release.
    if (meta.resource_version != version->second) meta.resources.clear();
    for (const backend::KernelResources& r : meta.resources) {
      opt::KernelMeasurements::instance().record(r.entry, r);
    }
    for (const backend::KernelCensus& c : meta.census) {
      opt::KernelMeasurements::instance().record(c.entry, c);
    }
    metadata.emplace(it->path().stem().string(), std::move(meta));
  }
  for (const auto& alias : aliases) {
    const auto it = metadata.find(alias.artifact);
    if (it == metadata.end() || it->second.source_hash != alias.hash) continue;
    const auto& canonical = it->second;
    record_structural_measurements(alias.entry, canonical.entry,
                                   canonical.resources, canonical.census);
  }
}

void write_atomic(const fs::path& path, std::span<const std::byte> bytes) {
  auto temporary = path.string() + ".tmpXXXXXX";
  const int fd = ::mkstemp(temporary.data());
  if (fd < 0) return;
  std::size_t written = 0;
  while (written < bytes.size()) {
    const auto result = ::write(fd, bytes.data() + written, bytes.size() - written);
    if (result < 0 && errno == EINTR) continue;
    if (result <= 0) break;
    written += static_cast<std::size_t>(result);
  }
  const bool complete = ::close(fd) == 0 && written == bytes.size();
  std::error_code ec;
  if (complete) fs::rename(temporary, path, ec);
  if (!complete || ec) fs::remove(temporary, ec);
}

void write_text(const fs::path& path, std::string_view text) {
  write_atomic(path, std::as_bytes(std::span(text.data(), text.size())));
}

void write_meta(const fs::path& path, const DiskMeta& meta) {
  std::ostringstream out;
  out << kMetaOwner << '\n' << meta.engine_version << '\n' << meta.arch << '\n';
  char buf[17];
  std::snprintf(buf, sizeof(buf), "%016llx",
                static_cast<unsigned long long>(meta.source_hash));
  out << buf << '\n' << meta.entry << '\n';
  out << "resource_version " << meta.resource_version << '\n';
  write_resources(out, meta.resources);
  write_census(out, meta.census);
  write_text(path, out.str());
}

void write_code(const fs::path& path, const std::vector<std::byte>& code) {
  write_atomic(path, code);
}

// ---- launch index --------------------------------------------------------
//
// One file per prepared set: what preload() was given (the kernels' cache
// keys), what each kernel launches with, and which bundle object holds them.
// Binary, little-endian, checksummed; see JitCache::restore for the policy.

constexpr char kIndexMagic[8] = {'L', 'S', 'E', 'L', 'N', 'C', 'H', 'X'};
constexpr std::uint32_t kIndexFormat = 1;
// Generous bounds a well-formed entry never approaches; a length past them is
// damage, not a large kernel.
constexpr std::uint64_t kIndexMaxText = 1u << 16;
constexpr std::uint64_t kIndexMaxCount = 1u << 20;

// Word at a time: an object's bytes are checked on every warm start, and a
// byte-wise FNV over every bundle would cost more than loading them.
std::uint64_t checksum64(std::span<const std::byte> bytes) noexcept {
  std::uint64_t h = 0x9ae16a3b2f90404full ^ static_cast<std::uint64_t>(bytes.size());
  auto step = [&h](std::uint64_t word) {
    h = std::rotl(h ^ (word * 0x87c37b91114253d5ull), 31) * 0x4cf5ad432745937full +
        0x52dce729ull;
  };
  std::size_t at = 0;
  for (; at + 8 <= bytes.size(); at += 8) {
    std::uint64_t word = 0;
    std::memcpy(&word, bytes.data() + at, 8);
    step(word);
  }
  if (at < bytes.size()) {
    std::uint64_t word = 0;
    std::memcpy(&word, bytes.data() + at, bytes.size() - at);
    step(word);
  }
  h ^= h >> 33;
  h *= 0xff51afd7ed558ccdull;
  h ^= h >> 33;
  h *= 0xc4ceb9fe1a85ec53ull;
  h ^= h >> 33;
  return h;
}

std::uint64_t index_mix(std::uint64_t h, std::uint64_t v) noexcept {
  return std::rotl(h ^ (v * 0x87c37b91114253d5ull), 31) * 0x4cf5ad432745937full +
         0x52dce729ull;
}

class IndexWriter {
 public:
  void u8(std::uint8_t v) { bytes_.push_back(static_cast<char>(v)); }
  void u16(std::uint16_t v) { raw(&v, sizeof v); }
  void u32(std::uint32_t v) { raw(&v, sizeof v); }
  void u64(std::uint64_t v) { raw(&v, sizeof v); }
  void text(std::string_view v) {
    u64(v.size());
    bytes_.append(v);
  }
  void launch(const EmittedKernel& e) {
    // Everything EmittedKernel carries but its text and its bindings, which
    // the emitter takes from the group it is launched for.
    text(e.entry_name);
    text(e.structural_entry_name);
    u8(e.content_addressed);
    u8(static_cast<std::uint8_t>(e.dialect));
    u32(e.constants.total_bytes);
    u32(static_cast<std::uint32_t>(e.constants.fields.size()));
    for (const ConstantsLayout::Field& f : e.constants.fields) {
      text(f.name);
      u16(f.offset);
      u8(f.size);
    }
    for (int d = 0; d < 3; ++d) u32(e.dims.workgroup_count[d]);
    for (int d = 0; d < 3; ++d) u32(e.dims.workgroup_size[d]);
    u32(e.dims.subgroup_size);
    u32(e.lds_bytes);
    u64(e.scratch_bytes);
    u8(e.pointer_table);
    u8(e.persist_grid);
    u32(static_cast<std::uint32_t>(opt::kOperandClasses));
    for (std::uint64_t v : e.traffic.read) u64(v);
    for (std::uint64_t v : e.traffic.written) u64(v);
    u64(e.traffic.work);
    u32(e.traffic.workgroups);
    u32(e.traffic.workgroup_threads);
    u8(e.traffic.stated);
  }
  // Appends the checksum of everything before it and returns the file.
  std::string finish() {
    u64(checksum64(std::as_bytes(std::span(bytes_.data(), bytes_.size()))));
    return std::move(bytes_);
  }

 private:
  void raw(const void* p, std::size_t n) {
    bytes_.append(static_cast<const char*>(p), n);
  }
  std::string bytes_;
};

class IndexReader {
 public:
  explicit IndexReader(std::span<const std::byte> bytes) : bytes_(bytes) {}
  [[nodiscard]] bool ok() const noexcept { return ok_; }
  [[nodiscard]] bool done() const noexcept { return at_ == bytes_.size(); }
  std::uint8_t u8() { std::uint8_t v = 0; raw(&v, sizeof v); return v; }
  std::uint16_t u16() { std::uint16_t v = 0; raw(&v, sizeof v); return v; }
  std::uint32_t u32() { std::uint32_t v = 0; raw(&v, sizeof v); return v; }
  std::uint64_t u64() { std::uint64_t v = 0; raw(&v, sizeof v); return v; }
  std::uint64_t count() {
    const std::uint64_t n = u64();
    if (n > kIndexMaxCount) ok_ = false;
    return ok_ ? n : 0;
  }
  std::string text() {
    const std::uint64_t n = u64();
    if (!ok_ || n > kIndexMaxText || n > bytes_.size() - at_) {
      ok_ = false;
      return {};
    }
    std::string out(reinterpret_cast<const char*>(bytes_.data() + at_), n);
    at_ += n;
    return out;
  }
  bool flag() {
    const std::uint8_t v = u8();
    if (v > 1) ok_ = false;
    return v == 1;
  }
  void launch(EmittedKernel* e) {
    e->entry_name = text();
    e->structural_entry_name = text();
    e->content_addressed = flag();
    const std::uint8_t dialect = u8();
    if (dialect >= kDialectCount) ok_ = false;
    e->dialect = static_cast<Dialect>(dialect);
    e->constants.total_bytes = u32();
    const std::uint32_t fields = u32();
    if (fields > kIndexMaxCount) ok_ = false;
    for (std::uint32_t i = 0; ok_ && i < fields; ++i) {
      ConstantsLayout::Field f;
      f.name = text();
      f.offset = u16();
      f.size = u8();
      e->constants.fields.push_back(std::move(f));
    }
    for (int d = 0; d < 3; ++d) e->dims.workgroup_count[d] = u32();
    for (int d = 0; d < 3; ++d) e->dims.workgroup_size[d] = u32();
    e->dims.subgroup_size = u32();
    e->lds_bytes = u32();
    e->scratch_bytes = static_cast<std::size_t>(u64());
    e->pointer_table = flag();
    e->persist_grid = flag();
    if (u32() != opt::kOperandClasses) ok_ = false;
    for (std::uint64_t& v : e->traffic.read) v = u64();
    for (std::uint64_t& v : e->traffic.written) v = u64();
    e->traffic.work = u64();
    e->traffic.workgroups = u32();
    e->traffic.workgroup_threads = u32();
    e->traffic.stated = flag();
    if (e->entry_name.empty()) ok_ = false;
  }

 private:
  void raw(void* p, std::size_t n) {
    if (!ok_ || n > bytes_.size() - at_) {
      ok_ = false;
      return;
    }
    std::memcpy(p, bytes_.data() + at_, n);
    at_ += n;
  }
  std::span<const std::byte> bytes_;
  std::size_t at_ = 0;
  bool ok_ = true;
};

struct IndexRecord {
  std::uint64_t signature = 0;
  std::uint64_t source_hash = 0;
  std::uint64_t source_size = 0;
  std::uint64_t source_print = 0;
  EmittedKernel launch;  // no source, no bindings
};

struct IndexEntry {
  std::uint64_t key = 0;
  std::string build;
  std::string version;
  std::string arch;
  std::uint8_t dialect = 0;
  std::vector<IndexRecord> records;
  // The bundle: its exports in load order, the hash and tag its note was
  // written under, and the checksum of its bytes when this entry was written.
  std::vector<std::string> names;
  std::uint64_t bundle_hash = 0;
  std::string tag;
  std::uint64_t code_checksum = 0;
};

// Empty when `bytes` is a complete, intact entry; otherwise what is wrong.
std::string decode_index(std::span<const std::byte> bytes, IndexEntry* out) {
  if (bytes.size() < sizeof kIndexMagic + 8) return "truncated";
  const std::size_t body = bytes.size() - 8;
  std::uint64_t stored = 0;
  std::memcpy(&stored, bytes.data() + body, 8);
  if (std::memcmp(bytes.data(), kIndexMagic, sizeof kIndexMagic) != 0)
    return "not a launch index entry";
  if (checksum64(bytes.first(body)) != stored) return "checksum mismatch";
  IndexReader in(bytes.first(body).subspan(sizeof kIndexMagic));
  if (in.u32() != kIndexFormat) return "unknown format";
  out->key = in.u64();
  out->build = in.text();
  out->version = in.text();
  out->arch = in.text();
  out->dialect = in.u8();
  const std::uint64_t records = in.count();
  out->records.resize(static_cast<std::size_t>(records));
  for (IndexRecord& r : out->records) {
    if (!in.ok()) break;
    r.signature = in.u64();
    r.source_hash = in.u64();
    r.source_size = in.u64();
    r.source_print = in.u64();
    in.launch(&r.launch);
  }
  const std::uint64_t names = in.count();
  for (std::uint64_t i = 0; in.ok() && i < names; ++i) out->names.push_back(in.text());
  out->bundle_hash = in.u64();
  out->tag = in.text();
  out->code_checksum = in.u64();
  if (!in.ok()) return "malformed";
  if (!in.done()) return "trailing bytes";
  return {};
}

// Everything that is not this build's own index directory, and is plainly
// one (named launch-<release>-<16 hex>, holding nothing but entries), is
// removed when it belongs to an older release; of the other builds of this
// release the most recently used few are kept, so switching between two
// builds does not cost a cold preparation each way.
void prune_launch_indexes(const fs::path& dir, const fs::path& mine) {
  constexpr std::size_t kKeepOtherBuilds = 3;
  ReleaseVersion current{};
  if (!release_version(kernel_cache_version(), &current)) return;
  CacheDiskLock lock(dir);
  if (!lock) return;
  struct Candidate { fs::path path; fs::file_time_type used; bool older = false; };
  std::vector<Candidate> found;
  std::error_code ec;
  for (fs::directory_iterator it(dir, ec), end; !ec && it != end; ++it) {
    const fs::path& path = it->path();
    if (path == mine) continue;
    std::error_code sec;
    if (!fs::is_directory(fs::symlink_status(path, sec)) || sec) continue;
    const std::string name = path.filename().string();
    if (!name.starts_with("launch-")) continue;
    const auto dash = name.rfind('-');
    if (dash == std::string::npos || dash <= 7 || name.size() - dash - 1 != 16) continue;
    std::uint64_t build = 0;
    if (!unsigned_number(std::string_view(name).substr(dash + 1), &build, 16)) continue;
    ReleaseVersion version{};
    if (!release_version(std::string_view(name).substr(7, dash - 7), &version)) continue;
    if (current < version) continue;
    Candidate c{path, fs::last_write_time(path, sec), version < current};
    if (sec) continue;
    found.push_back(std::move(c));
  }
  std::sort(found.begin(), found.end(), [](const Candidate& a, const Candidate& b) {
    return a.used > b.used;
  });
  std::size_t kept = 0;
  for (const Candidate& c : found) {
    if (!c.older && kept < kKeepOtherBuilds) {
      ++kept;
      continue;
    }
    // Only entries and their temporaries; anything else keeps the directory.
    for (fs::directory_iterator it(c.path, ec), end; !ec && it != end; ++it) {
      const std::string name = it->path().filename().string();
      if (regular_file(it->path()) && name.find(".launch") != std::string::npos) {
        std::error_code rm;
        fs::remove(it->path(), rm);
      }
    }
    std::error_code rm;
    fs::remove(c.path, rm);  // fails, harmlessly, if anything else is inside
  }
}

// A bundle object and its note, read back. Empty `code` when either is
// missing or the note names another set; the note's facts are refreshed when
// the compiler's resource reader has moved on, as get_or_compile does.
std::vector<std::byte> read_bundle(const fs::path& stem, std::uint64_t bundle_hash,
                                   const std::string& tag, const std::string& arch,
                                   const IKernelCompiler& compiler,
                                   const std::string& cache_dir,
                                   std::vector<backend::KernelResources>* resources,
                                   std::vector<backend::KernelCensus>* census) {
  const fs::path meta_path = stem.string() + ".meta";
  const fs::path co_path = stem.string() + "." + std::to_string(bundle_hash) + ".co";
  DiskMeta meta;
  if (!read_meta(meta_path, &meta) || meta.engine_version != kernel_cache_version() ||
      meta.arch != arch || meta.source_hash != bundle_hash || meta.entry != tag)
    return {};
  std::vector<std::byte> code = read_file(co_path);
  if (code.empty()) return {};
  bool metadata_changed = false;
  if (meta.resource_version != compiler.resource_metadata_version()) {
    meta.resources = compiler.resources(code);
    meta.resource_version = compiler.resource_metadata_version();
    metadata_changed = true;
  }
  if (meta.census.empty()) {
    meta.census = compiler.census(code);
    metadata_changed = !meta.census.empty() || metadata_changed;
  }
  if (metadata_changed) {
    CacheDiskLock lock(cache_dir);
    if (lock) write_meta(meta_path, meta);
  }
  *resources = std::move(meta.resources);
  *census = std::move(meta.census);
  return code;
}

}  // namespace

void dump_hip_source(const EmittedKernel& emitted, std::uint64_t key) {
  if (emitted.source.empty()) return;
  // DECIDE CHEAPLY, THEN DO THE WORK. This runs once per dispatch -- ~1857 a
  // decode token, 9343 in a 201-token prefill -- and every one of those calls
  // used to getenv for the directory, build a fs::path and a std::string from
  // it, copy and sanitise the entry name, take a mutex and hash the string,
  // only to find it had already written that file. The dedup now happens
  // first, on the key it is already given, so a repeat costs one integer hash.
  // Dedup FIRST, on the key we are already given: a repeat then costs one
  // integer hash instead of a getenv, a path, a string copy and a string hash.
  // The directory is resolved only on the write path, because it comes from
  // the environment and a caller may point it somewhere new between calls --
  // caching it broke `debug_writes_generated_hip_for_review`, which does
  // exactly that.
  {
    static std::mutex seen_mu;
    static std::unordered_set<std::uint64_t> seen;
    const std::lock_guard<std::mutex> lock(seen_mu);
    if (!seen.insert(key).second) return;
  }
  const fs::path dir = hip_dump_directory();
  if (dir.empty()) return;
  std::string name = emitted.entry_name.empty()
                         ? ("kernel_" + std::to_string(key))
                         : emitted.entry_name;
  for (char& c : name) {
    if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '_' ||
          c == '-' || c == '.')) {
      c = '_';
    }
  }

  std::error_code ec;
  fs::create_directories(dir, ec);
  // Extension is the dialect's own name, so a dump directory holding both
  // languages says which is which. kHip spells "hip", so the HIP path's files
  // are named exactly as before.
  write_text(dir / (name + "." + std::string(to_string(emitted.dialect))),
             emitted.source);
}

// Generated source is written for review only when asked: LSE_HIP_DUMP names
// the directory, and debug mode (LSE_DEBUG, --debug) uses the build tree's.
// A server writing every new kernel's text on its first dispatch spent most
// of a cold prefill's kernel lookup on file writes.
std::string hip_dump_directory() {
  if (const char* env = std::getenv("LSE_HIP_DUMP"); env != nullptr && env[0] != '\0') {
    return env;
  }
#ifdef LSE_BUILD_DIR
  if (::lse::debug()) return std::string(LSE_BUILD_DIR) + "/hip";
#endif
  return {};
}

std::string_view kernel_cache_version() noexcept { return LSE_ENGINE_VERSION; }

void purge_kernel_artifacts(std::string_view cache_dir) {
  remove_older_cache_entries(cache_dir.empty() ? default_cache_dir()
                                              : std::string(cache_dir));
  static std::once_flag once;
  std::call_once(once, [] {
    const fs::path dir = hip_dump_directory();
    if (dir.empty()) return;
    // A dump directory can coincide with the code cache. Delete only generated
    // LSE source files, never the directory or arbitrary contents.
    std::error_code ec;
    for (fs::directory_iterator it(dir, ec), end; !ec && it != end; ++it) {
      const auto& path = it->path();
      if ((path.extension() == ".hip" || path.extension() == ".loom") &&
          legacy_entry(path.stem().string()) && regular_file(path)) {
        std::error_code rm;
        fs::remove(path, rm);
      }
    }
  });
}

std::string default_cache_dir() {
  if (const char* env = std::getenv("LSE_CACHE_DIR"); env && *env) return env;
  if (const char* home = std::getenv("HOME"); home && *home) {
    return std::string(home) + "/.lse/cache";
  }
  return ".lse/cache";
}

Status prepare_cache_dir(std::string_view requested) {
  const std::string path = requested.empty() ? default_cache_dir()
                                              : std::string(requested);
  std::error_code ec;
  fs::create_directories(path, ec);
  if (ec || ::access(path.c_str(), W_OK | X_OK) != 0) {
    return LSE_ERROR(kInvalidArgument, "cannot create or write kernel cache '",
                     path, "'", ec ? ": " + ec.message() : "");
  }
  if (::setenv("LSE_CACHE_DIR", path.c_str(), 1) != 0)
    return LSE_ERROR(kInternal, "cannot configure kernel cache '", path, "'");
  return OkStatus();
}

struct JitCache::Impl {
  struct Slot {
    backend::KernelHandle handle;
    std::uint64_t source_hash = 0;
    std::string arch;
    // Content-addressed kernels are matched on their complete source: its
    // length and fingerprint() here, fnv() in source_hash.
    std::size_t source_size = 0;
    std::uint64_t source_print = 0;
    [[nodiscard]] bool same_source(std::string_view text) const noexcept {
      return text.size() == source_size && fingerprint(text) == source_print;
    }
    // What the toolchain said about the object behind this handle. Carried
    // whether the object was compiled here or read back from disk: a warm
    // start that lost the numbers would make them a property of process age.
    std::vector<backend::KernelResources> resources;
    std::vector<backend::KernelCensus> census;
  };
  // One map per (member, dialect). A KernelHandle is an executable loaded on
  // ONE device; handing member B the handle member A loaded is a wrong-device
  // dispatch that no runtime here reports, and two same-arch devices would
  // otherwise share the entry because the disk key is deliberately the same for
  // them. Splitting by dialect as well is what makes try_get — the one lookup
  // that answers without seeing source — unable to return the other language's
  // object even if two dialects ever collided on a key.
  struct Alias {
    std::shared_ptr<const Slot> slot;
    std::string structural_entry;
  };
  std::vector<std::unordered_map<std::uint64_t, Alias>> memory;
  // Hash buckets compare complete source; each member owns its loaded handle.
  std::vector<std::unordered_multimap<std::uint64_t,
                                    std::shared_ptr<const Slot>>> artifacts;

  // One kernel of a prepared set that is not resident yet: where it is filed
  // and the source it stands for. `launch` supplies its entry names, whether
  // it is content-addressed, and its traffic; its text is never read.
  struct Placement {
    std::uint64_t key = 0;
    std::uint64_t artifact_key = 0;
    std::uint64_t source_hash = 0;
    std::uint64_t source_size = 0;
    std::uint64_t source_print = 0;
    const EmittedKernel* launch = nullptr;
  };

  static void record(const Slot& slot, const EmittedKernel& launch) {
    record_structural_measurements(launch.structural_entry_name, launch.entry_name,
                                   slot.resources, slot.census);
    const auto& traffic_entry = launch.structural_entry_name.empty()
        ? launch.entry_name : launch.structural_entry_name;
    opt::KernelMeasurements::instance().record(traffic_entry, launch.traffic);
  }

  // A kernel whose content-addressed object is already loaded here, under
  // another key. Filed under its own key as that object.
  [[nodiscard]] std::shared_ptr<const Slot> loaded_artifact(
      std::size_t table, const Placement& p, const std::string& arch) const {
    if (!p.launch->content_addressed) return nullptr;
    const auto [first, last] = artifacts[table].equal_range(p.artifact_key);
    for (auto it = first; it != last; ++it) {
      if (it->second->arch == arch && it->second->source_size == p.source_size &&
          it->second->source_print == p.source_print)
        return it->second;
    }
    return nullptr;
  }

  // Already resident under `p.key`, from the same source.
  [[nodiscard]] bool resident(std::size_t table, const Placement& p,
                              const std::string& arch) const {
    const auto it = memory[table].find(p.key);
    if (it == memory[table].end()) return false;
    const Slot& slot = *it->second.slot;
    return slot.arch == arch && slot.source_hash == p.source_hash &&
           (!p.launch->content_addressed ||
            (slot.source_size == p.source_size && slot.source_print == p.source_print));
  }

  void alias(std::size_t table, const Placement& p,
             const std::shared_ptr<const Slot>& slot) {
    memory[table].insert_or_assign(p.key, Alias{slot, p.launch->structural_entry_name});
    record(*slot, *p.launch);
  }

  // Loads `names` from one bundle object and files every pending kernel under
  // its key: the shared tail of preload() and restore().
  Status install(backend::IBackend& be, std::size_t table, const std::string& arch,
                 std::span<const std::string> names, std::span<const std::byte> code,
                 const std::vector<backend::KernelResources>& resources,
                 const std::vector<backend::KernelCensus>& census,
                 std::span<const Placement> pending) {
    LSE_ASSIGN_OR(std::vector<backend::KernelHandle> handles,
                  be.load_executables(names, code));
    if (handles.size() != names.size())
      return LSE_ERROR(kInternal, "loading ", std::to_string(names.size()),
                       " kernels returned ", std::to_string(handles.size()), " handles");
    for (const backend::KernelResources& r : resources)
      opt::KernelMeasurements::instance().record(r.entry, r);
    for (const backend::KernelCensus& c : census)
      opt::KernelMeasurements::instance().record(c.entry, c);
    std::map<std::string_view, const Placement*> first;
    for (const Placement& p : pending) first.emplace(p.launch->entry_name, &p);
    std::map<std::string_view, std::shared_ptr<const Slot>> loaded;
    for (std::size_t i = 0; i < names.size(); ++i) {
      const auto at = first.find(names[i]);
      if (at == first.end())
        return LSE_ERROR(kInternal, "bundle export ", names[i], " has no kernel");
      const Placement& p = *at->second;
      std::vector<backend::KernelResources> own_resources;
      for (const auto& r : resources) if (r.entry == names[i]) own_resources.push_back(r);
      std::vector<backend::KernelCensus> own_census;
      for (const auto& c : census) if (c.entry == names[i]) own_census.push_back(c);
      const bool content = p.launch->content_addressed;
      loaded.emplace(p.launch->entry_name, std::make_shared<const Slot>(Slot{
          handles[i], p.source_hash, arch,
          content ? static_cast<std::size_t>(p.source_size) : 0,
          content ? p.source_print : 0,
          std::move(own_resources), std::move(own_census)}));
    }
    for (const Placement& p : pending) {
      const auto& slot = loaded.at(p.launch->entry_name);
      alias(table, p, slot);
      if (p.launch->content_addressed) artifacts[table].emplace(p.artifact_key, slot);
    }
    return OkStatus();
  }
};

JitCache::JitCache(backend::IDeviceSet& devices, std::string cache_dir)
    : devices_(devices),
      cache_dir_(std::move(cache_dir)),
      impl_(std::make_unique<Impl>()) {
  impl_->memory.resize(devices_.size() * kDialectCount);
  impl_->artifacts.resize(impl_->memory.size());
  compiler_id_.resize(devices_.size() * kDialectCount, 0);
  std::unordered_map<std::string, std::uint32_t> resource_versions;
  for (std::size_t i = 0; i < devices_.size(); ++i) {
    // Every dialect the member declares, not just its front one: an object
    // built by the Loom compiler must not sit in a slot keyed by comgr's
    // identity, and the two are told apart here or nowhere.
    for (const KernelToolchain& tc : devices_.device(i).toolchains()) {
      if (tc.compiler == nullptr) continue;
      compiler_id_[toolchain_slot(i, tc.dialect)] = fnv(tc.compiler->identity());
      auto& version = resource_versions[devices_.device(i).device_info().arch];
      version = std::max(version, tc.compiler->resource_metadata_version());
    }
  }
  purge_kernel_artifacts(cache_dir_);
  preload_measurements(cache_dir_, resource_versions);
  open_launch_index();
}

JitCache::JitCache(backend::IBackend& backend, const IKernelCompiler& compiler,
                   std::string cache_dir)
    : own_set_(std::make_unique<backend::SingleDevice>(backend)),
      devices_(*own_set_),
      named_compiler_(&compiler),
      cache_dir_(std::move(cache_dir)),
      impl_(std::make_unique<Impl>()) {
  // The caller named one compiler for every dialect it will ask for, so every
  // slot carries that one identity and two dialects share a key. They still do
  // not share an entry: the memory table is one map per dialect, and on disk an
  // object is named by the hash of the source it was built from.
  compiler_id_.assign(kDialectCount, fnv(compiler.identity()));
  impl_->memory.resize(kDialectCount);
  impl_->artifacts.resize(impl_->memory.size());
  purge_kernel_artifacts(cache_dir_);
  preload_measurements(cache_dir_,
      {{backend.device_info().arch, compiler.resource_metadata_version()}});
  open_launch_index();
}

void JitCache::open_launch_index() {
  const std::string_view build = engine_build_identity();
  if (build.size() < 16 || cache_dir_.empty()) return;
  const fs::path dir = fs::path(cache_dir_) /
      ("launch-" + std::string(kernel_cache_version()) + "-" + std::string(build.substr(0, 16)));
  std::error_code ec;
  fs::create_directories(dir, ec);
  if (ec || !fs::is_directory(fs::symlink_status(dir, ec)) || ec) return;
  // Marks this build's entries as the most recently used, for the pruning
  // other builds do.
  fs::last_write_time(dir, fs::file_time_type::clock::now(), ec);
  prune_launch_indexes(cache_dir_, dir);
  index_dir_ = dir.string();
}

JitCache::~JitCache() = default;

const IKernelCompiler* JitCache::compiler_for(std::size_t member,
                                              Dialect dialect) const noexcept {
  if (named_compiler_ != nullptr) return named_compiler_;
  const KernelToolchain* tc = devices_.device(member).toolchain_for(dialect);
  return tc != nullptr ? tc->compiler : nullptr;
}

std::uint64_t JitCache::slot_key(std::size_t member, Dialect dialect,
                                 std::uint64_t signature) const noexcept {
  // A cached object is only valid for the toolchain that built it, and
  // source_hash cannot tell two toolchains apart. The compiler reports its own
  // identity (version + option lists) rather than a human bumping a revision
  // constant here, which was one forgotten increment away from serving an
  // object built by a different pipeline. Read at the (member, dialect) slot:
  // one entry per member hashed the front dialect's compiler into every
  // dialect's key, which is the same failure with the toolchains swapped.
  // Arch in the key so a device change cannot reuse another target's object.
  const backend::DeviceInfo& info = devices_.device(member).device_info();
  std::uint64_t h = mix(mix(signature, compiler_id_[toolchain_slot(member, dialect)]),
                        fnv(info.arch));
  // Arch is NOT enough between two devices of the same ISA. The emitter chooses
  // workgroup dimensions, an LDS budget and a persistent-grid decision from the
  // CU count, the LDS pool and the workgroup ceiling, so two gfx1151 parts with
  // different geometry are handed different source under one arch string. Left
  // out, they collide on one key: the source-hash guard then forces a recompile
  // per alternation AND each device deletes the other's object as dead — a
  // ~350 ms stall per kernel per switch, on a key that looked like a hit.
  //
  // Deliberately NOT the device's identity. Two members with the same geometry
  // emit byte-identical source and must share the object; keying on which
  // device asked would compile it once per device for nothing.
  h = mix(h, static_cast<std::uint64_t>(info.compute_units));
  h = mix(h, static_cast<std::uint64_t>(info.lds_bytes_per_workgroup));
  h = mix(h, static_cast<std::uint64_t>(info.max_threads_per_workgroup));
  h = mix(h, static_cast<std::uint64_t>(info.wavefront_size));
  h = mix(h, static_cast<std::uint64_t>(info.cus_per_lds_pool));
  return mix(h, fnv(kernel_cache_version()));
}

const backend::KernelHandle* JitCache::try_get(std::size_t member,
                                               std::uint64_t signature,
                                               Dialect dialect) noexcept {
  const std::size_t slot = toolchain_slot(member, dialect);
  if (slot >= impl_->memory.size()) return nullptr;
  auto& slots = impl_->memory[slot];
  const auto it = slots.find(slot_key(member, dialect, signature));
  if (it == slots.end()) return nullptr;
  if (it->second.slot->arch != devices_.device(member).device_info().arch) {
    return nullptr;
  }
  ++stats_.memory_hits;
  return &it->second.slot->handle;
}

Result<backend::KernelHandle> JitCache::get_or_compile(
    std::size_t member, std::uint64_t signature, const EmittedKernel& emitted) {
  const std::size_t table = toolchain_slot(member, emitted.dialect);
  if (table >= impl_->memory.size()) {
    return LSE_ERROR(kOutOfRange, "device set has ",
                     std::to_string(impl_->memory.size() / kDialectCount),
                     " members; there is no member ", std::to_string(member));
  }
  backend::IBackend& be = devices_.device(member);
  // The compiler declared beside the emitter that wrote this text, never the
  // device's front one: the two halves of a dialect belong together, and a
  // caller cannot hand the wrong pair here because the text names its language.
  const IKernelCompiler* compiler = compiler_for(member, emitted.dialect);
  if (compiler == nullptr) {
    return LSE_ERROR(kUnimplemented, "backend '", std::string(be.name()),
                     "' has no ", std::string(to_string(emitted.dialect)),
                     " kernel compiler");
  }
  auto& slots = impl_->memory[table];
  const std::string arch(be.device_info().arch);
  const std::uint64_t key = slot_key(member, emitted.dialect, signature);
  const std::uint64_t src_hash =
      emitted.source.empty() ? 0 : fnv(emitted.source);
  if (emitted.content_addressed && emitted.source.empty())
    return LSE_ERROR(kInvalidArgument, "content-addressed kernel has no source");
  const std::uint64_t artifact_key = emitted.content_addressed
      ? slot_key(member, emitted.dialect,
          mix(mix(fnv("jit.exact-source.v1"),
                  static_cast<std::uint64_t>(emitted.dialect)), src_hash))
      : key;

  dump_hip_source(emitted, key);

  auto publish_alias = [&](const std::shared_ptr<const Impl::Slot>& slot) {
    slots.insert_or_assign(key, Impl::Alias{slot, emitted.structural_entry_name});
    record_structural_measurements(emitted.structural_entry_name,
                                   emitted.entry_name, slot->resources, slot->census);
    // Traffic is an invocation fact, kept under its structural identity.
    const auto& traffic_entry = emitted.structural_entry_name.empty()
        ? emitted.entry_name : emitted.structural_entry_name;
    opt::KernelMeasurements::instance().record(traffic_entry, emitted.traffic);
    if (emitted.content_addressed && !emitted.structural_entry_name.empty()) {
      const fs::path alias_path = fs::path(cache_dir_) /
          (artifact_stem(kernel_cache_version(), key) + ".alias");
      const std::string alias = std::string(kAliasOwner) + "\n" +
          std::string(kernel_cache_version()) + "\n" + emitted.structural_entry_name +
          "\n" + artifact_stem(kernel_cache_version(), artifact_key) + "\n" +
          std::to_string(slot->source_hash) + "\n";
      // A warm start finds the alias it would write already there; rewriting
      // it is a rename per loaded kernel and changes nothing.
      if (!source_matches(alias_path, alias)) {
        CacheDiskLock lock(cache_dir_);
        if (lock) write_text(alias_path, alias);
      }
    }
  };

  if (const auto it = slots.find(key); it != slots.end()) {
    const Impl::Slot& slot = *it->second.slot;
    if (slot.arch == arch &&
        (src_hash == 0 || src_hash == slot.source_hash) &&
        (!emitted.content_addressed || slot.same_source(emitted.source))) {
      ++stats_.memory_hits;
      return slot.handle;
    }
  }

  if (emitted.content_addressed) {
    const auto [first, last] = impl_->artifacts[table].equal_range(artifact_key);
    for (auto it = first; it != last; ++it) {
      const auto& slot = it->second;
      if (slot->arch == arch && slot->same_source(emitted.source)) {
        publish_alias(slot);
        ++stats_.memory_hits;
        return slot->handle;
      }
    }
  }

  const fs::path stem = fs::path(cache_dir_) /
      artifact_stem(kernel_cache_version(), artifact_key);
  const fs::path meta_path = stem.string() + ".meta";

  DiskMeta meta;
  const bool meta_ok = read_meta(meta_path, &meta) &&
      meta.engine_version == kernel_cache_version();
  const fs::path source_path = stem.string() + ".source";
  const bool source_ok =
      (src_hash == 0 || (meta_ok && src_hash == meta.source_hash)) &&
      (!emitted.content_addressed ||
       (meta_ok && meta.entry == emitted.entry_name &&
        source_matches(source_path, emitted.source)));
  const bool device_matches = meta_ok && meta.arch == arch;

  // The object is named by the source hash it was built from, so a load can
  // never pair one source's meta with another source's code: the two renames
  // below are individually atomic but not atomic as a pair, and a crash
  // between them must not be able to poison the key forever.
  const std::uint64_t co_hash =
      src_hash != 0 ? src_hash : (meta_ok ? meta.source_hash : 0);
  const fs::path co_path =
      stem.string() + "." + std::to_string(co_hash) + ".co";

  std::vector<std::byte> code;
  std::vector<backend::KernelResources> resources;
  std::vector<backend::KernelCensus> census;
  if (device_matches && source_ok) {
    code = read_file(co_path);
    if (!code.empty()) {
      ++stats_.disk_hits;
      bool metadata_changed = false;
      if (meta.resource_version != compiler->resource_metadata_version()) {
        meta.resources = compiler->resources(code);
        meta.resource_version = compiler->resource_metadata_version();
        metadata_changed = true;
      }
      resources = meta.resources;
      census = meta.census;
      // An object cached before anything counted instructions still has them:
      // they are in the bytes. Counting them now and rewriting the note keeps a
      // warm start as informed as a cold one, and costs one disassembly once.
      if (census.empty()) {
        census = compiler->census(code);
        if (!census.empty()) {
          meta.census = census;
          metadata_changed = true;
        }
      }
      if (metadata_changed) {
        CacheDiskLock lock(cache_dir_);
        if (lock) write_meta(meta_path, meta);
      }
    }
  }

  if (code.empty()) {
    if (emitted.source.empty()) {
      return LSE_ERROR(kCompileError,
                       "no cached kernel for this device and no source to compile");
    }
    const auto begin = std::chrono::steady_clock::now();
    auto compiled = compiler->compile(emitted.source, arch);
    if (!compiled.ok()) return compiled.status();
    CompiledKernel built = compiled.release();
    code = std::move(built.code);
    resources = std::move(built.resources);
    census = std::move(built.census);
    stats_.compile_ns += static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - begin)
            .count());
    ++stats_.compiles;

    CacheDiskLock lock(cache_dir_);
    std::error_code ec;
    // Best-effort reclaim of objects this key no longer references (older
    // source revisions); the hash-suffixed name makes them dead, not wrong.
    for (fs::directory_iterator it(cache_dir_, ec), end; lock && !ec && it != end;
         ++it) {
      const std::string name = it->path().filename().string();
      const std::string prefix = stem.filename().string() + ".";
      std::uint64_t old_hash = 0;
      if (name.starts_with(prefix) && name.ends_with(".co") &&
          unsigned_number(std::string_view(name).substr(
              prefix.size(), name.size() - prefix.size() - 3), &old_hash) &&
          complete_file(it->path()) && it->path() != co_path) {
        std::error_code rm;
        fs::remove(it->path(), rm);
      }
    }
    if (lock) {
      write_code(co_path, code);
      if (emitted.content_addressed) write_text(source_path, emitted.source);
      write_meta(meta_path,
                 DiskMeta{arch, src_hash, emitted.entry_name, resources, census,
                          compiler->resource_metadata_version(),
                          std::string(kernel_cache_version())});
    }
  }

  auto handle = be.load_executable(
      emitted.entry_name.empty() && meta_ok ? meta.entry : emitted.entry_name,
      code);
  if (!handle.ok()) return handle.status();
  backend::KernelHandle kernel = handle.release();
  const std::uint64_t stored_hash =
      src_hash != 0 ? src_hash : (meta_ok ? meta.source_hash : 0);
  static const auto resource_profile_mode =
      detail::dispatch_profile_mode(std::getenv("LSE_PROFILE_DISPATCH"));
  if (resource_profile_mode == detail::DispatchProfileMode::kSubmit ||
      resource_profile_mode == detail::DispatchProfileMode::kSerial) {
    for (const auto& r : resources) {
      if (r.entry != (emitted.entry_name.empty() ? meta.entry : emitted.entry_name))
        continue;
      const auto diagnostic =
          detail::private_memory_diagnostic(r, arch, artifact_key, stored_hash);
      if (!diagnostic.empty()) std::fputs(diagnostic.c_str(), stderr);
    }
  }
  // Publish what the toolchain said, so a decision made BEFORE the next
  // compile of the same kernel can consult a measurement instead of a
  // prediction. Keyed on the entry name, which a decision site can spell
  // before any text exists.
  for (const backend::KernelResources& r : resources) {
    opt::KernelMeasurements::instance().record(r.entry, r);
  }
  for (const backend::KernelCensus& c : census) {
    opt::KernelMeasurements::instance().record(c.entry, c);
  }
  auto slot = std::make_shared<const Impl::Slot>(Impl::Slot{
      kernel, stored_hash, arch,
      emitted.content_addressed ? emitted.source.size() : 0,
      emitted.content_addressed ? fingerprint(emitted.source) : 0,
      std::move(resources), std::move(census)});
  publish_alias(slot);
  if (emitted.content_addressed)
    impl_->artifacts[table].emplace(artifact_key, slot);
  return kernel;
}

std::uint64_t JitCache::index_key(std::size_t member, Dialect dialect,
                                  std::span<const std::uint64_t> signatures) const noexcept {
  // slot_key already holds the release, the compiler, the arch and the
  // geometry; the cache key under it holds the group's structure, its
  // specializations and the device's facts. The build identity is what is
  // left: the text a template writes, which no key can see until it is
  // written.
  std::uint64_t h = index_mix(fnv("jit.launch-index.v1"), fnv(engine_build_identity()));
  h = index_mix(h, static_cast<std::uint64_t>(dialect));
  h = index_mix(h, signatures.size());
  for (const std::uint64_t signature : signatures)
    h = index_mix(h, slot_key(member, dialect, signature));
  return h;
}

Status JitCache::preload(std::size_t member, std::span<const Preload> kernels,
                         bool index) {
  if (kernels.empty()) return OkStatus();
  const Dialect dialect = kernels.front().emitted->dialect;
  const std::size_t table = toolchain_slot(member, dialect);
  if (table >= impl_->memory.size()) {
    return LSE_ERROR(kOutOfRange, "device set has ",
                     std::to_string(impl_->memory.size() / kDialectCount),
                     " members; there is no member ", std::to_string(member));
  }
  backend::IBackend& be = devices_.device(member);
  const IKernelCompiler* compiler = compiler_for(member, dialect);
  if (compiler == nullptr) {
    return LSE_ERROR(kUnimplemented, "backend '", std::string(be.name()), "' has no ",
                     std::string(to_string(dialect)), " kernel compiler");
  }
  const std::string arch(be.device_info().arch);

  // Every kernel's placement, resident or not: the index records all of them.
  std::vector<Impl::Placement> placed;
  placed.reserve(kernels.size());
  std::vector<Impl::Placement> pending;
  for (const Preload& k : kernels) {
    const EmittedKernel& e = *k.emitted;
    if (e.dialect != dialect)
      return LSE_ERROR(kInvalidArgument, "kernels preloaded together must share a dialect");
    if (e.source.empty())
      return LSE_ERROR(kInvalidArgument, "preloading ", e.entry_name, " needs its source");
    Impl::Placement p;
    p.key = slot_key(member, dialect, k.signature);
    p.source_hash = fnv(e.source);
    p.source_size = e.source.size();
    p.source_print = fingerprint(e.source);
    p.launch = &e;
    p.artifact_key = e.content_addressed
        ? slot_key(member, dialect,
                   mix(mix(fnv("jit.exact-source.v1"), static_cast<std::uint64_t>(dialect)),
                       p.source_hash))
        : p.key;
    placed.push_back(p);
    if (impl_->resident(table, p, arch)) continue;
    if (const auto slot = impl_->loaded_artifact(table, p, arch)) {
      impl_->alias(table, p, slot);
      continue;
    }
    pending.push_back(p);
  }

  // Each distinct entry once, in a fixed order, so the same set always makes
  // the same object.
  std::vector<std::string> names;
  std::uint64_t bundle_hash = 0;
  std::string tag;
  std::uint64_t code_checksum = 0;
  if (!pending.empty()) {
    std::map<std::string_view, const EmittedKernel*> entries;
    for (const Impl::Placement& p : pending) {
      const auto [it, inserted] = entries.emplace(p.launch->entry_name, p.launch);
      if (!inserted && it->second->source != p.launch->source)
        return LSE_ERROR(kInternal, "two kernels named ", p.launch->entry_name,
                         " differ in source");
    }
    std::string bundle;
    names.reserve(entries.size());
    for (const auto& [entry, emitted] : entries) {
      bundle += emitted->source;
      bundle += '\n';
      names.emplace_back(entry);
    }
    bundle_hash = fnv(bundle);
    const std::uint64_t bundle_key =
        slot_key(member, dialect, mix(fnv("jit.bundle.v1"), bundle_hash));
    const fs::path stem =
        fs::path(cache_dir_) / artifact_stem(kernel_cache_version(), bundle_key);
    // No .source beside the object: its length and fingerprint, in the meta's
    // entry line, stand in for the text when it is read back.
    tag = "bundle " + std::to_string(bundle.size()) + " " +
          std::to_string(fingerprint(bundle));

    std::vector<backend::KernelResources> resources;
    std::vector<backend::KernelCensus> census;
    std::vector<std::byte> code =
        read_bundle(stem, bundle_hash, tag, arch, *compiler, cache_dir_, &resources, &census);
    if (!code.empty()) {
      ++stats_.disk_hits;
    } else {
      const auto begin = std::chrono::steady_clock::now();
      auto compiled = compiler->compile(bundle, arch);
      if (!compiled.ok()) return compiled.status();
      CompiledKernel built = compiled.release();
      stats_.compile_ns += static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now() - begin).count());
      ++stats_.compiles;
      code = std::move(built.code);
      resources = std::move(built.resources);
      census = std::move(built.census);
      CacheDiskLock lock(cache_dir_);
      if (lock) {
        write_code(stem.string() + "." + std::to_string(bundle_hash) + ".co", code);
        write_meta(stem.string() + ".meta",
                   DiskMeta{arch, bundle_hash, tag, resources, census,
                            compiler->resource_metadata_version(),
                            std::string(kernel_cache_version())});
      }
    }
    code_checksum = checksum64(code);
    LSE_RETURN_IF_ERROR(
        impl_->install(be, table, arch, names, code, resources, census, pending));
  }

  if (index && !index_dir_.empty()) {
    std::vector<std::uint64_t> signatures;
    signatures.reserve(kernels.size());
    for (const Preload& k : kernels) signatures.push_back(k.signature);
    const std::uint64_t key = index_key(member, dialect, signatures);
    IndexWriter out;
    for (char c : kIndexMagic) out.u8(static_cast<std::uint8_t>(c));
    out.u32(kIndexFormat);
    out.u64(key);
    out.text(engine_build_identity());
    out.text(kernel_cache_version());
    out.text(arch);
    out.u8(static_cast<std::uint8_t>(dialect));
    out.u64(kernels.size());
    for (std::size_t i = 0; i < kernels.size(); ++i) {
      out.u64(kernels[i].signature);
      out.u64(placed[i].source_hash);
      out.u64(placed[i].source_size);
      out.u64(placed[i].source_print);
      out.launch(*kernels[i].emitted);
    }
    out.u64(names.size());
    for (const std::string& name : names) out.text(name);
    out.u64(bundle_hash);
    out.text(tag);
    out.u64(code_checksum);
    const std::string bytes = out.finish();
    CacheDiskLock lock(cache_dir_);
    if (lock)
      write_text(fs::path(index_dir_) / (std::to_string(key) + ".launch"), bytes);
  }
  return OkStatus();
}

Result<bool> JitCache::restore(std::size_t member, Dialect dialect,
                               std::span<const std::uint64_t> signatures,
                               const IKernelEmitter& emitter) {
  if (signatures.empty()) return true;
  const std::size_t table = toolchain_slot(member, dialect);
  if (table >= impl_->memory.size()) {
    return LSE_ERROR(kOutOfRange, "device set has ",
                     std::to_string(impl_->memory.size() / kDialectCount),
                     " members; there is no member ", std::to_string(member));
  }
  backend::IBackend& be = devices_.device(member);
  const IKernelCompiler* compiler = compiler_for(member, dialect);
  if (compiler == nullptr) {
    return LSE_ERROR(kUnimplemented, "backend '", std::string(be.name()), "' has no ",
                     std::string(to_string(dialect)), " kernel compiler");
  }
  if (index_dir_.empty() || !emitter.keeps_launches() || emitter.dialect() != dialect) {
    ++stats_.index_misses;
    return false;
  }
  const std::string arch(be.device_info().arch);
  const std::uint64_t key = index_key(member, dialect, signatures);
  const fs::path path = fs::path(index_dir_) / (std::to_string(key) + ".launch");
  const std::vector<std::byte> bytes = read_file(path);
  if (bytes.empty()) {
    ++stats_.index_misses;
    return false;
  }
  auto reject = [&](const std::vector<fs::path>& files, std::string_view why) {
    std::fprintf(stderr,
                 "lse: launch index: discarding %s: %s; writing and compiling these "
                 "%zu kernels from source\n",
                 files.front().c_str(), std::string(why).c_str(), signatures.size());
    CacheDiskLock lock(cache_dir_);
    for (const fs::path& file : files) {
      std::error_code ec;
      if (regular_file(file)) fs::remove(file, ec);
    }
    ++stats_.index_rejects;
    ++stats_.index_misses;
    return false;
  };

  IndexEntry entry;
  if (const std::string damage = decode_index(bytes, &entry); !damage.empty())
    return reject({path}, damage);
  // The entry names itself; an entry that disagrees with where it was found
  // is damaged or misplaced, and neither is evidence.
  if (entry.key != key) return reject({path}, "its key is not its name");
  if (entry.build != engine_build_identity() || entry.version != kernel_cache_version())
    return reject({path}, "written by another build");
  if (entry.arch != arch || entry.dialect != static_cast<std::uint8_t>(dialect))
    return reject({path}, "written for another device or dialect");
  if (entry.records.size() != signatures.size())
    return reject({path}, "it holds another set");
  for (std::size_t i = 0; i < signatures.size(); ++i) {
    const IndexRecord& r = entry.records[i];
    if (r.signature != signatures[i] || r.launch.dialect != dialect ||
        !r.launch.source.empty())
      return reject({path}, "it holds another set");
  }

  // Placed against what THIS process has resident, exactly as preload would
  // place the same set; the bundle must hold precisely what is left.
  std::vector<Impl::Placement> pending;
  std::vector<std::pair<Impl::Placement, std::shared_ptr<const Impl::Slot>>> aliases;
  std::map<std::string_view, const IndexRecord*> entries;
  for (std::size_t i = 0; i < signatures.size(); ++i) {
    const IndexRecord& r = entry.records[i];
    Impl::Placement p;
    p.key = slot_key(member, dialect, r.signature);
    p.source_hash = r.source_hash;
    p.source_size = r.source_size;
    p.source_print = r.source_print;
    p.launch = &r.launch;
    p.artifact_key = r.launch.content_addressed
        ? slot_key(member, dialect,
                   mix(mix(fnv("jit.exact-source.v1"), static_cast<std::uint64_t>(dialect)),
                       r.source_hash))
        : p.key;
    if (impl_->resident(table, p, arch)) continue;
    if (auto slot = impl_->loaded_artifact(table, p, arch)) {
      aliases.emplace_back(p, std::move(slot));
      continue;
    }
    const auto [it, inserted] = entries.emplace(r.launch.entry_name, &r);
    if (!inserted && (it->second->source_hash != r.source_hash ||
                      it->second->source_size != r.source_size ||
                      it->second->source_print != r.source_print))
      return reject({path}, "two kernels share an entry name but not a source");
    pending.push_back(p);
  }
  if (entries.size() != entry.names.size() ||
      !std::equal(entries.begin(), entries.end(), entry.names.begin(),
                  [](const auto& e, const std::string& name) { return e.first == name; })) {
    ++stats_.index_misses;
    return false;
  }

  if (!pending.empty()) {
    const std::uint64_t bundle_key =
        slot_key(member, dialect, mix(fnv("jit.bundle.v1"), entry.bundle_hash));
    const fs::path stem =
        fs::path(cache_dir_) / artifact_stem(kernel_cache_version(), bundle_key);
    std::vector<backend::KernelResources> resources;
    std::vector<backend::KernelCensus> census;
    const std::vector<std::byte> code = read_bundle(
        stem, entry.bundle_hash, entry.tag, arch, *compiler, cache_dir_, &resources, &census);
    if (code.empty()) {
      ++stats_.index_misses;
      return false;
    }
    if (checksum64(code) != entry.code_checksum) {
      return reject({stem.string() + "." + std::to_string(entry.bundle_hash) + ".co",
                     stem.string() + ".meta", path},
                    "its bundle object's bytes do not match their checksum");
    }
    ++stats_.disk_hits;
    LSE_RETURN_IF_ERROR(
        impl_->install(be, table, arch, entry.names, code, resources, census, pending));
  }
  for (const auto& [p, slot] : aliases) impl_->alias(table, p, slot);
  // Every description, resident or not: one this process wrote itself is
  // kept by the emitter, the rest stand in for the text it never wrote.
  for (const IndexRecord& r : entry.records) (void)emitter.adopt_launch(r.signature, r.launch);
  ++stats_.index_hits;
  return true;
}

const backend::KernelCensus* JitCache::census(
    std::size_t member, std::uint64_t signature, Dialect dialect,
    std::string_view entry) const noexcept {
  const std::size_t slot = toolchain_slot(member, dialect);
  if (slot >= impl_->memory.size()) return nullptr;
  const auto& slots = impl_->memory[slot];
  const auto it = slots.find(slot_key(member, dialect, signature));
  if (it == slots.end()) return nullptr;
  const std::vector<backend::KernelCensus>& all = it->second.slot->census;
  if (all.empty()) return nullptr;
  if (entry.empty() || entry == it->second.structural_entry)
    return all.size() == 1 ? &all.front() : nullptr;
  for (const backend::KernelCensus& c : all) {
    if (c.entry == entry) return &c;
  }
  return nullptr;
}

const backend::KernelResources* JitCache::resources(
    std::size_t member, std::uint64_t signature, Dialect dialect,
    std::string_view entry) const noexcept {
  const std::size_t slot = toolchain_slot(member, dialect);
  if (slot >= impl_->memory.size()) return nullptr;
  const auto& slots = impl_->memory[slot];
  const auto it = slots.find(slot_key(member, dialect, signature));
  if (it == slots.end()) return nullptr;
  const std::vector<backend::KernelResources>& all = it->second.slot->resources;
  if (all.empty()) return nullptr;
  if (entry.empty() || entry == it->second.structural_entry)
    return all.size() == 1 ? &all.front() : nullptr;
  for (const backend::KernelResources& r : all) {
    if (r.entry == entry) return &r;
  }
  return nullptr;
}

}  // namespace lse::graph
