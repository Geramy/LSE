#include "lse/probe/profile_store.hpp"

#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace lse::probe {

namespace fs = std::filesystem;

namespace {

// 2 added DeviceProfile::free_memory to the device record. A stale entry then
// fails the version check, which is the re-measure path rather than an error.
constexpr int kFormatVersion = 2;
// A field with no text. Spaces separate fields, so an empty one needs a mark.
constexpr std::string_view kNone = "-";

std::string_view token_or_empty(std::string_view s) {
  return s == kNone ? std::string_view{} : s;
}

std::string text_or_none(std::string_view s) {
  return s.empty() ? std::string(kNone) : std::string(s);
}

template <typename Enum, std::size_t N>
bool enum_from_string(const std::array<std::pair<Enum, std::string_view>, N>& table,
                      std::string_view text, Enum* out) {
  for (const auto& entry : table) {
    if (entry.second == text) {
      *out = entry.first;
      return true;
    }
  }
  return false;
}

bool provenance_from_string(std::string_view s, Provenance* out) {
  return enum_from_string(kEnumEntries_Provenance, s, out);
}
bool row_support_from_string(std::string_view s, RowSupport* out) {
  return enum_from_string(kEnumEntries_RowSupport, s, out);
}
bool path_kind_from_string(std::string_view s, PathKind* out) {
  return enum_from_string(kEnumEntries_PathKind, s, out);
}

bool matrix_elem_from_string(std::string_view s, math::MatrixElem* out) {
  using math::MatrixElem;
  constexpr MatrixElem kAll[] = {MatrixElem::kF32,  MatrixElem::kF16,
                                 MatrixElem::kBF16, MatrixElem::kI32,
                                 MatrixElem::kI8,   MatrixElem::kI4,
                                 MatrixElem::kFp8,  MatrixElem::kBf8};
  for (MatrixElem e : kAll) {
    if (to_string(e) == s) {
      *out = e;
      return true;
    }
  }
  return false;
}

void append_measured(std::ostringstream& out, const Measured& m) {
  out << ' ' << to_string(m.provenance) << ' ' << m.value;
}

bool read_measured(std::istringstream& in, Measured* out) {
  std::string prov;
  if (!(in >> prov)) return false;
  if (!provenance_from_string(prov, &out->provenance)) return false;
  return static_cast<bool>(in >> out->value);
}

void write_device(std::ostringstream& out, const DeviceProfile& d) {
  out << "device " << d.id.backend << ' ' << d.id.ordinal << ' '
      << text_or_none(d.arch) << ' ' << d.total_memory << ' '
      << d.compute_units << ' ' << (d.unified_memory ? 1 : 0);
  append_measured(out, d.dram_bytes_per_s);
  append_measured(out, d.launch_overhead_ns);
  append_measured(out, d.h2d_bytes_per_s);
  append_measured(out, d.d2h_bytes_per_s);
  append_measured(out, d.free_memory);
  // Free text, so it is last and takes the rest of the line.
  out << ' ' << text_or_none(d.name) << '\n';
  for (const MatrixRowRate& r : d.rows) {
    out << "row " << text_or_none(r.key) << ' ' << to_string(r.acc) << ' '
        << to_string(r.operand) << ' ' << r.m << ' ' << r.n << ' ' << r.k_step
        << ' ' << r.relative << ' ' << to_string(r.support);
    append_measured(out, r.flops);
    out << '\n';
  }
  for (const ComputePath& p : d.paths) {
    out << "path " << to_string(p.operand) << ' ' << to_string(p.executed_as)
        << ' ' << (p.native ? 1 : 0) << ' ' << text_or_none(p.row_key);
    append_measured(out, p.flops);
    out << '\n';
  }
}

// Returns false on a malformed line rather than throwing: a corrupt cache entry
// is a cache miss, not a crash.
bool read_device_line(std::istringstream& in, DeviceProfile* d) {
  int unified = 0;
  std::string arch;
  if (!(in >> d->id.backend >> d->id.ordinal >> arch >> d->total_memory)) {
    return false;
  }
  unsigned cus = 0;
  if (!(in >> cus >> unified)) return false;
  d->compute_units = static_cast<std::uint16_t>(cus);
  d->unified_memory = unified != 0;
  d->arch = std::string(token_or_empty(arch));
  if (!read_measured(in, &d->dram_bytes_per_s)) return false;
  if (!read_measured(in, &d->launch_overhead_ns)) return false;
  if (!read_measured(in, &d->h2d_bytes_per_s)) return false;
  if (!read_measured(in, &d->d2h_bytes_per_s)) return false;
  if (!read_measured(in, &d->free_memory)) return false;
  std::string rest;
  std::getline(in, rest);
  const std::size_t first = rest.find_first_not_of(' ');
  if (first != std::string::npos) {
    d->name = std::string(token_or_empty(rest.substr(first)));
  }
  return true;
}

bool read_row_line(std::istringstream& in, MatrixRowRate* r) {
  std::string key, acc, operand, support;
  if (!(in >> key >> acc >> operand >> r->m >> r->n >> r->k_step >>
        r->relative >> support)) {
    return false;
  }
  r->key = std::string(token_or_empty(key));
  if (!matrix_elem_from_string(acc, &r->acc)) return false;
  if (!matrix_elem_from_string(operand, &r->operand)) return false;
  if (!row_support_from_string(support, &r->support)) return false;
  return read_measured(in, &r->flops);
}

bool read_path_line(std::istringstream& in, ComputePath* p) {
  std::string operand, executed, key;
  int native = 0;
  if (!(in >> operand >> executed >> native >> key)) return false;
  if (!matrix_elem_from_string(operand, &p->operand)) return false;
  if (!matrix_elem_from_string(executed, &p->executed_as)) return false;
  p->native = native != 0;
  p->row_key = std::string(token_or_empty(key));
  return read_measured(in, &p->flops);
}

void write_text_atomic(const fs::path& path, std::string_view text) {
  const fs::path tmp = path.string() + ".tmp" + std::to_string(::getpid());
  {
    std::ofstream out(tmp);
    if (!out) return;
    out.write(text.data(), static_cast<std::streamsize>(text.size()));
  }
  std::error_code ec;
  fs::rename(tmp, path, ec);
  if (ec) fs::remove(tmp, ec);
}

}  // namespace

std::string default_profile_dir() {
  if (const char* env = std::getenv("LSE_PROFILE_DIR");
      env != nullptr && env[0] != '\0') {
    return env;
  }
  if (const char* xdg = std::getenv("XDG_CACHE_HOME")) {
    return std::string(xdg) + "/lse/profiles";
  }
  if (const char* home = std::getenv("HOME")) {
    return std::string(home) + "/.cache/lse/profiles";
  }
  return ".lse-profiles";
}

std::string profile_path(std::string_view dir, std::string_view fingerprint) {
  return std::string(dir) + "/" + std::string(fingerprint) + ".profile";
}

std::string serialize_device_profile(const DeviceProfile& device) {
  std::ostringstream out;
  out.precision(17);
  write_device(out, device);
  return out.str();
}

Result<DeviceProfile> parse_device_profile(std::string_view text) {
  DeviceProfile d;
  bool seen = false;
  std::istringstream lines{std::string(text)};
  std::string line;
  while (std::getline(lines, line)) {
    if (line.empty()) continue;
    std::istringstream in(line);
    std::string kind;
    in >> kind;
    if (kind == "device") {
      if (!read_device_line(in, &d)) {
        return LSE_ERROR(kInvalidArgument, "malformed device record");
      }
      seen = true;
    } else if (kind == "row") {
      MatrixRowRate r;
      if (!read_row_line(in, &r)) {
        return LSE_ERROR(kInvalidArgument, "malformed matrix row record");
      }
      d.rows.push_back(std::move(r));
    } else if (kind == "path") {
      ComputePath p;
      if (!read_path_line(in, &p)) {
        return LSE_ERROR(kInvalidArgument, "malformed compute path record");
      }
      d.paths.push_back(std::move(p));
    }
  }
  if (!seen) return LSE_ERROR(kInvalidArgument, "no device record");
  return d;
}

std::string serialize_pool_profile(const PoolProfile& pool) {
  std::ostringstream out;
  out.precision(17);
  out << "lse-pool-profile " << kFormatVersion << '\n';
  out << "fingerprint " << text_or_none(pool.fingerprint) << '\n';
  for (const DeviceProfile& d : pool.devices) write_device(out, d);
  const std::size_t n = pool.devices.size();
  for (std::size_t i = 0; i < n && pool.links.size() == n * n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      const LinkProfile& l = pool.links[i * n + j];
      out << "link " << i << ' ' << j << ' ' << to_string(l.path);
      append_measured(out, l.latency_ns);
      append_measured(out, l.bandwidth_bytes_per_s);
      out << ' ' << l.fit_error << ' ' << l.points.size();
      for (const TransferPoint& p : l.points) {
        out << ' ' << p.bytes << ' ' << p.ns;
      }
      out << '\n';
    }
  }
  return out.str();
}

Result<PoolProfile> parse_pool_profile(std::string_view text) {
  PoolProfile pool;
  bool header = false;
  std::istringstream lines{std::string(text)};
  std::string line;
  while (std::getline(lines, line)) {
    if (line.empty()) continue;
    std::istringstream in(line);
    std::string kind;
    in >> kind;
    if (kind == "lse-pool-profile") {
      int version = 0;
      if (!(in >> version) || version != kFormatVersion) {
        return LSE_ERROR(kInvalidArgument, "profile format version mismatch");
      }
      header = true;
    } else if (kind == "fingerprint") {
      std::string fp;
      in >> fp;
      pool.fingerprint = std::string(token_or_empty(fp));
    } else if (kind == "device") {
      DeviceProfile d;
      if (!read_device_line(in, &d)) {
        return LSE_ERROR(kInvalidArgument, "malformed device record");
      }
      pool.devices.push_back(std::move(d));
    } else if (kind == "row") {
      if (pool.devices.empty()) {
        return LSE_ERROR(kInvalidArgument, "matrix row before any device");
      }
      MatrixRowRate r;
      if (!read_row_line(in, &r)) {
        return LSE_ERROR(kInvalidArgument, "malformed matrix row record");
      }
      pool.devices.back().rows.push_back(std::move(r));
    } else if (kind == "path") {
      if (pool.devices.empty()) {
        return LSE_ERROR(kInvalidArgument, "compute path before any device");
      }
      ComputePath p;
      if (!read_path_line(in, &p)) {
        return LSE_ERROR(kInvalidArgument, "malformed compute path record");
      }
      pool.devices.back().paths.push_back(std::move(p));
    } else if (kind == "link") {
      std::size_t i = 0, j = 0, npoints = 0;
      std::string path;
      if (!(in >> i >> j >> path)) {
        return LSE_ERROR(kInvalidArgument, "malformed link record");
      }
      const std::size_t n = pool.devices.size();
      if (i >= n || j >= n) {
        return LSE_ERROR(kInvalidArgument, "link names a device that is not "
                                           "in this profile");
      }
      if (pool.links.size() != n * n) pool.links.resize(n * n);
      LinkProfile& l = pool.links[i * n + j];
      l.src = pool.devices[i].id;
      l.dst = pool.devices[j].id;
      if (!path_kind_from_string(path, &l.path)) {
        return LSE_ERROR(kInvalidArgument, "unknown link path kind");
      }
      if (!read_measured(in, &l.latency_ns)) {
        return LSE_ERROR(kInvalidArgument, "malformed link latency");
      }
      if (!read_measured(in, &l.bandwidth_bytes_per_s)) {
        return LSE_ERROR(kInvalidArgument, "malformed link bandwidth");
      }
      if (!(in >> l.fit_error >> npoints)) {
        return LSE_ERROR(kInvalidArgument, "malformed link fit");
      }
      for (std::size_t p = 0; p < npoints; ++p) {
        TransferPoint tp;
        if (!(in >> tp.bytes >> tp.ns)) {
          return LSE_ERROR(kInvalidArgument, "truncated link points");
        }
        l.points.push_back(tp);
      }
    }
  }
  if (!header) return LSE_ERROR(kInvalidArgument, "not a pool profile");
  return pool;
}

Result<PoolProfile> load_pool_profile(std::string_view fingerprint,
                                      std::string_view dir) {
  const fs::path path = profile_path(dir, fingerprint);
  std::ifstream in(path);
  if (!in) {
    return LSE_ERROR(kNotFound, "no profile at ", path.string());
  }
  std::ostringstream buf;
  buf << in.rdbuf();
  auto parsed = parse_pool_profile(buf.str());
  if (!parsed.ok()) {
    return LSE_ERROR(kNotFound, "profile at ", path.string(),
                     " is unreadable: ", parsed.status().message());
  }
  PoolProfile pool = parsed.release();
  // The name on disk is not proof of the contents; a truncated or hand-edited
  // entry must not be served as if it described this pool.
  if (pool.fingerprint != fingerprint) {
    return LSE_ERROR(kNotFound, "profile at ", path.string(),
                     " was measured for a different pool");
  }
  return pool;
}

Status save_pool_profile(const PoolProfile& pool, std::string_view dir) {
  if (pool.fingerprint.empty()) {
    return LSE_ERROR(kInvalidArgument,
                     "a profile with no fingerprint has no valid key");
  }
  std::error_code ec;
  fs::create_directories(dir, ec);
  if (ec) {
    return LSE_ERROR(kIoError, "cannot create ", std::string(dir), ": ",
                     ec.message());
  }
  write_text_atomic(profile_path(dir, pool.fingerprint),
                    serialize_pool_profile(pool));
  return OkStatus();
}

}  // namespace lse::probe
