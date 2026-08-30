#include "lse/graph/jit.hpp"
#include "lse/opt/measurements.hpp"

#include <unistd.h>

#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
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

std::uint64_t fnv(std::string_view s) noexcept {
  std::uint64_t h = 1469598103934665603ull;
  for (char ch : s) {
    h ^= static_cast<unsigned char>(ch);
    h *= 1099511628211ull;
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

bool read_meta(const fs::path& path, DiskMeta* out) {
  std::ifstream in(path);
  if (!in) return false;
  std::string hash;
  if (!std::getline(in, out->arch)) return false;
  if (!std::getline(in, hash)) return false;
  if (!std::getline(in, out->entry)) return false;
  if (out->arch.empty() || hash.empty()) return false;
  try {
    out->source_hash = std::stoull(hash, nullptr, 16);
  } catch (...) {
    return false;
  }
  for (std::string line; std::getline(in, line);) {
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

// Hand every measurement already on disk to the optimizer, once, before the
// first kernel is emitted. Without this a decision made at emit time can only
// see kernels this process has already compiled, so the first emit of a run
// always falls back to the estimate — and the answer would then depend on how
// long the process had been running, which is exactly what must not happen.
void preload_measurements(const std::string& dir) {
  std::error_code ec;
  for (fs::directory_iterator it(dir, ec), end; !ec && it != end; ++it) {
    if (it->path().extension() != ".meta") continue;
    DiskMeta meta;
    if (!read_meta(it->path(), &meta)) continue;
    for (const backend::KernelResources& r : meta.resources) {
      opt::KernelMeasurements::instance().record(r.entry, r);
    }
    for (const backend::KernelCensus& c : meta.census) {
      opt::KernelMeasurements::instance().record(c.entry, c);
    }
  }
}

void write_meta(const fs::path& path, const DiskMeta& meta) {
  const fs::path tmp =
      path.string() + ".tmp" + std::to_string(::getpid());
  std::ofstream out(tmp);
  if (!out) return;
  out << meta.arch << '\n';
  char buf[17];
  std::snprintf(buf, sizeof(buf), "%016llx",
                static_cast<unsigned long long>(meta.source_hash));
  out << buf << '\n' << meta.entry << '\n';
  write_resources(out, meta.resources);
  write_census(out, meta.census);
  out.close();
  std::error_code ec;
  fs::rename(tmp, path, ec);
  if (ec) fs::remove(tmp, ec);
}

void write_code(const fs::path& path, const std::vector<std::byte>& code) {
  const fs::path tmp =
      path.string() + ".tmp" + std::to_string(::getpid());
  std::ofstream out(tmp, std::ios::binary);
  if (!out) return;
  out.write(reinterpret_cast<const char*>(code.data()),
            static_cast<std::streamsize>(code.size()));
  out.close();
  std::error_code ec;
  fs::rename(tmp, path, ec);
  if (ec) fs::remove(tmp, ec);
}

void write_text(const fs::path& path, std::string_view text) {
  const fs::path tmp =
      path.string() + ".tmp" + std::to_string(::getpid());
  std::ofstream out(tmp);
  if (!out) return;
  out.write(text.data(), static_cast<std::streamsize>(text.size()));
  out.close();
  std::error_code ec;
  fs::rename(tmp, path, ec);
  if (ec) fs::remove(tmp, ec);
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

std::string hip_dump_directory() {
  if (const char* env = std::getenv("LSE_HIP_DUMP"); env != nullptr && env[0] != '\0') {
    return env;
  }
#ifdef LSE_BUILD_DIR
  return std::string(LSE_BUILD_DIR) + "/hip";
#else
  return {};
#endif
}

void purge_kernel_artifacts() {
  static std::once_flag once;
  std::call_once(once, [] {
    const auto wipe = [](const fs::path& dir) {
      if (dir.empty()) return;
      const fs::path n = dir.lexically_normal();
      if (n.empty() || n == n.root_path()) return;
      const std::string s = n.generic_string();
      if (s.find("lse") == std::string::npos &&
          s.find("hip") == std::string::npos &&
          s.find("kernel") == std::string::npos) {
        return;
      }
      std::error_code ec;
      fs::remove_all(n, ec);
    };
    // The dump dir is debug output; clearing it keeps build/hip in step with
    // this process's emission. The .co cache is NOT wiped: entries carry an
    // arch + source-hash check on read, so stale ones self-invalidate, and
    // wiping here forced every process to recompile every kernel.
    wipe(hip_dump_directory());
    if (const char* env = std::getenv("LSE_JIT_PURGE");
        env != nullptr && env[0] == '1') {
      wipe(default_cache_dir());
    }
  });
}

std::string default_cache_dir() {
  if (const char* env = std::getenv("LSE_CACHE_DIR")) return env;
  if (const char* xdg = std::getenv("XDG_CACHE_HOME")) {
    return std::string(xdg) + "/lse/kernels";
  }
  if (const char* home = std::getenv("HOME")) {
    return std::string(home) + "/.cache/lse/kernels";
  }
  return ".lse-cache";
}

struct JitCache::Impl {
  struct Slot {
    backend::KernelHandle handle;
    std::uint64_t source_hash = 0;
    std::string arch;
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
  std::vector<std::unordered_map<std::uint64_t, Slot>> memory;
};

JitCache::JitCache(backend::IDeviceSet& devices, std::string cache_dir)
    : devices_(devices),
      cache_dir_(std::move(cache_dir)),
      impl_(std::make_unique<Impl>()) {
  impl_->memory.resize(devices_.size() * kDialectCount);
  compiler_id_.resize(devices_.size() * kDialectCount, 0);
  for (std::size_t i = 0; i < devices_.size(); ++i) {
    // Every dialect the member declares, not just its front one: an object
    // built by the Loom compiler must not sit in a slot keyed by comgr's
    // identity, and the two are told apart here or nowhere.
    for (const KernelToolchain& tc : devices_.device(i).toolchains()) {
      if (tc.compiler == nullptr) continue;
      compiler_id_[toolchain_slot(i, tc.dialect)] = fnv(tc.compiler->identity());
    }
  }
  purge_kernel_artifacts();
  preload_measurements(cache_dir_);
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
  purge_kernel_artifacts();
  preload_measurements(cache_dir_);
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
  return h;
}

const backend::KernelHandle* JitCache::try_get(std::size_t member,
                                               std::uint64_t signature,
                                               Dialect dialect) noexcept {
  const std::size_t slot = toolchain_slot(member, dialect);
  if (slot >= impl_->memory.size()) return nullptr;
  auto& slots = impl_->memory[slot];
  const auto it = slots.find(slot_key(member, dialect, signature));
  if (it == slots.end()) return nullptr;
  if (it->second.arch != devices_.device(member).device_info().arch) {
    return nullptr;
  }
  ++stats_.memory_hits;
  return &it->second.handle;
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

  dump_hip_source(emitted, key);

  if (const auto it = slots.find(key); it != slots.end()) {
    const Impl::Slot& slot = it->second;
    if (slot.arch == arch &&
        (src_hash == 0 || src_hash == slot.source_hash)) {
      ++stats_.memory_hits;
      return slot.handle;
    }
  }

  const fs::path stem = fs::path(cache_dir_) / std::to_string(key);
  const fs::path meta_path = stem.string() + ".meta";

  DiskMeta meta;
  const bool meta_ok = read_meta(meta_path, &meta);
  const bool source_matches =
      src_hash == 0 || (meta_ok && src_hash == meta.source_hash);
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
  if (device_matches && source_matches) {
    code = read_file(co_path);
    if (!code.empty()) {
      ++stats_.disk_hits;
      resources = meta.resources;
      census = meta.census;
      // An object cached before anything counted instructions still has them:
      // they are in the bytes. Counting them now and rewriting the note keeps a
      // warm start as informed as a cold one, and costs one disassembly once.
      if (census.empty()) {
        census = compiler->census(code);
        if (!census.empty()) {
          meta.census = census;
          write_meta(meta_path, meta);
        }
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

    std::error_code ec;
    fs::create_directories(cache_dir_, ec);
    // Best-effort reclaim of objects this key no longer references (older
    // source revisions); the hash-suffixed name makes them dead, not wrong.
    for (fs::directory_iterator it(cache_dir_, ec), end; !ec && it != end;
         ++it) {
      const std::string name = it->path().filename().string();
      const std::string prefix = std::to_string(key) + ".";
      if (name.rfind(prefix, 0) == 0 && name.size() > 3 &&
          name.compare(name.size() - 3, 3, ".co") == 0 &&
          it->path() != co_path) {
        std::error_code rm;
        fs::remove(it->path(), rm);
      }
    }
    write_code(co_path, code);
    write_meta(meta_path,
               DiskMeta{arch, src_hash, emitted.entry_name, resources, census});
  }

  auto handle = be.load_executable(
      emitted.entry_name.empty() && meta_ok ? meta.entry : emitted.entry_name,
      code);
  if (!handle.ok()) return handle.status();
  backend::KernelHandle kernel = handle.release();
  const std::uint64_t stored_hash =
      src_hash != 0 ? src_hash : (meta_ok ? meta.source_hash : 0);
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
  // The intent beside the count, under the same identity, so the two are
  // comparable without anyone holding on to the EmittedKernel that produced
  // them.
  opt::KernelMeasurements::instance().record(emitted.entry_name,
                                             emitted.traffic);
  slots[key] = Impl::Slot{kernel, stored_hash, arch, std::move(resources),
                          std::move(census)};
  return kernel;
}

const backend::KernelCensus* JitCache::census(
    std::size_t member, std::uint64_t signature, Dialect dialect,
    std::string_view entry) const noexcept {
  const std::size_t slot = toolchain_slot(member, dialect);
  if (slot >= impl_->memory.size()) return nullptr;
  const auto& slots = impl_->memory[slot];
  const auto it = slots.find(slot_key(member, dialect, signature));
  if (it == slots.end()) return nullptr;
  const std::vector<backend::KernelCensus>& all = it->second.census;
  if (all.empty()) return nullptr;
  if (entry.empty()) return all.size() == 1 ? &all.front() : nullptr;
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
  const std::vector<backend::KernelResources>& all = it->second.resources;
  if (all.empty()) return nullptr;
  if (entry.empty()) return all.size() == 1 ? &all.front() : nullptr;
  for (const backend::KernelResources& r : all) {
    if (r.entry == entry) return &r;
  }
  return nullptr;
}

}  // namespace lse::graph
