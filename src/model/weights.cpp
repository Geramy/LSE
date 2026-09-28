#include "lse/model/weights.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <optional>
#include <set>
#include <sstream>
#include <thread>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "lse/model/config.hpp"
#include "lse/model/registry.hpp"

namespace lse::model {

namespace fs = std::filesystem;

namespace {

// MLX names a quantized weight's planes off the MODULE, not off the tensor:
// `<module>.weight` is the packed plane and its scales are `<module>.scales`,
// never `<module>.weight.scales`. Returns the module path, or nullopt when the
// name cannot be a packed plane.
//
// LANDMINE: this is WeightBinder::quant_planes' rule restated. The two must
// agree — deriving the name differently here makes every packed plane invisible,
// which silently undercounts parameters and reports checkpoints as loadable
// whose quantized tensors were never examined.
std::optional<std::string> quant_plane_base(std::string_view name) {
  constexpr std::string_view kWeight = ".weight";
  if (name.size() <= kWeight.size() || !name.ends_with(kWeight)) {
    return std::nullopt;
  }
  return std::string(name.substr(0, name.size() - kWeight.size()));
}

DType dtype_from_safetensors(std::string_view s) noexcept {
  if (s == "BF16") return DType::kBF16;
  if (s == "F16") return DType::kF16;
  if (s == "F32") return DType::kF32;
  if (s == "I32") return DType::kI32;
  if (s == "I8") return DType::kI8;
  if (s == "U8") return DType::kU8;
  // MLX writes the packed plane of a group-affine weight as U32; the codes
  // inside a lane are narrower than any dtype names.
  if (s == "U32") return DType::kU32;
  return DType::kCount;
}

// Widening the checkpoint is ~1.4 billion scalar converts and 5.8 GB of stores
// for a 2.9 GB bf16 file — single-threaded that was most of process start. The
// work is memory-bound, so threads past a handful buy nothing; small tensors
// run inline rather than pay for a spawn.
template <typename Fn>
void parallel_chunks(std::size_t count, Fn&& fn) {
  constexpr std::size_t kMinPerThread = 1u << 20;
  unsigned threads = std::thread::hardware_concurrency();
  if (threads == 0) threads = 1;
  threads = std::min(threads, 16u);
  const std::size_t affordable = count / kMinPerThread;
  if (affordable < threads) {
    threads = static_cast<unsigned>(affordable);
  }
  if (threads <= 1) {
    fn(std::size_t{0}, count);
    return;
  }
  const std::size_t chunk = (count + threads - 1) / threads;
  std::vector<std::thread> pool;
  pool.reserve(threads - 1);
  for (unsigned t = 1; t < threads; ++t) {
    const std::size_t lo = t * chunk;
    if (lo >= count) break;
    const std::size_t hi = std::min(count, lo + chunk);
    pool.emplace_back([&fn, lo, hi] { fn(lo, hi); });
  }
  fn(std::size_t{0}, std::min(count, chunk));
  for (std::thread& t : pool) t.join();
}

}  // namespace

Status TensorView::read_native(void* dst, std::size_t bytes) const {
  if (bytes > data.size()) {
    return LSE_ERROR(kOutOfRange, "read_native asked for ",
                     std::to_string(bytes), " of ",
                     std::to_string(data.size()), " bytes in '", name, "'");
  }
  std::memcpy(dst, data.data(), bytes);
  return OkStatus();
}

Status TensorView::read_f32(float* dst, std::size_t count) const {
  if (count > element_count()) {
    return LSE_ERROR(kOutOfRange, "read_f32 asked for ", std::to_string(count),
                     " of ", std::to_string(element_count()), " elements in '",
                     name, "'");
  }
  const void* p = data.data();
  switch (dtype) {
    case DType::kF32:
      std::memcpy(dst, p, count * sizeof(float));
      return OkStatus();
    case DType::kBF16: {
      const auto* src = static_cast<const std::uint16_t*>(p);
      parallel_chunks(count, [dst, src](std::size_t lo, std::size_t hi) {
        for (std::size_t i = lo; i < hi; ++i) {
          bfloat16_t h;
          h.bits = src[i];
          dst[i] = h.to_float();
        }
      });
      return OkStatus();
    }
    case DType::kF16: {
      const auto* src = static_cast<const std::uint16_t*>(p);
      parallel_chunks(count, [dst, src](std::size_t lo, std::size_t hi) {
        for (std::size_t i = lo; i < hi; ++i) {
          float16_t h;
          h.bits = src[i];
          dst[i] = h.to_float();
        }
      });
      return OkStatus();
    }
    case DType::kI32: {
      const auto* src = static_cast<const std::int32_t*>(p);
      for (std::size_t i = 0; i < count; ++i) dst[i] = static_cast<float>(src[i]);
      return OkStatus();
    }
    case DType::kI8: {
      const auto* src = static_cast<const std::int8_t*>(p);
      for (std::size_t i = 0; i < count; ++i) dst[i] = static_cast<float>(src[i]);
      return OkStatus();
    }
    case DType::kU8: {
      const auto* src = static_cast<const std::uint8_t*>(p);
      for (std::size_t i = 0; i < count; ++i) dst[i] = static_cast<float>(src[i]);
      return OkStatus();
    }
    default:
      return LSE_ERROR(kUnimplemented, "read_f32 from ",
                       std::string(to_string(dtype)));
  }
}

void SafeTensors::unmap_all() noexcept {
  for (const Mapping& m : mappings_) {
    if (m.ptr != nullptr) ::munmap(m.ptr, m.size);
  }
  mappings_.clear();
}

SafeTensors::~SafeTensors() { unmap_all(); }

SafeTensors::SafeTensors(SafeTensors&& other) noexcept
    : path_(std::move(other.path_)),
      mappings_(std::move(other.mappings_)),
      tensors_(std::move(other.tensors_)) {
  other.mappings_.clear();
}

SafeTensors& SafeTensors::operator=(SafeTensors&& other) noexcept {
  if (this != &other) {
    unmap_all();
    path_ = std::move(other.path_);
    mappings_ = std::move(other.mappings_);
    tensors_ = std::move(other.tensors_);
    other.mappings_.clear();
  }
  return *this;
}

Status SafeTensors::map_file(const std::string& path) {
  const int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) return LSE_ERROR(kIoError, "cannot open '", path, "'");

  struct stat st {};
  if (::fstat(fd, &st) != 0) {
    ::close(fd);
    return LSE_ERROR(kIoError, "cannot stat '", path, "'");
  }
  const auto file_size = static_cast<std::size_t>(st.st_size);
  if (file_size < 8) {
    ::close(fd);
    return LSE_ERROR(kIoError, "'", path, "' is too small to be safetensors");
  }

  void* map = ::mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
  ::close(fd);
  if (map == MAP_FAILED) return LSE_ERROR(kIoError, "mmap failed for '", path, "'");

  const auto* base = static_cast<const std::byte*>(map);
  std::uint64_t header_len = 0;
  std::memcpy(&header_len, base, sizeof(header_len));
  if (header_len == 0 || header_len + 8 > file_size) {
    ::munmap(map, file_size);
    return LSE_ERROR(kIoError, "safetensors header length is out of range");
  }

  mappings_.push_back(Mapping{map, file_size});

  const std::byte* payload = base + 8 + header_len;
  const std::size_t payload_size = file_size - 8 - header_len;

  nlohmann::json header;
  try {
    header = nlohmann::json::parse(reinterpret_cast<const char*>(base + 8),
                                   reinterpret_cast<const char*>(base + 8 + header_len));
  } catch (const std::exception& e) {
    return LSE_ERROR(kIoError, "safetensors header is not valid JSON: ", e.what());
  }

  for (const auto& [name, entry] : header.items()) {
    if (name == "__metadata__") continue;
    if (!entry.contains("dtype") || !entry.contains("shape") ||
        !entry.contains("data_offsets")) {
      return LSE_ERROR(kIoError, "tensor '", name, "' has an incomplete entry");
    }

    TensorView v;
    v.name = name;
    v.dtype = dtype_from_safetensors(entry["dtype"].get<std::string>());
    if (v.dtype == DType::kCount) {
      return LSE_ERROR(kUnimplemented, "tensor '", name, "' has unsupported dtype ",
                       entry["dtype"].get<std::string>());
    }
    for (const auto& d : entry["shape"]) v.shape.push_back(d.get<std::int64_t>());

    const auto begin = entry["data_offsets"][0].get<std::uint64_t>();
    const auto end = entry["data_offsets"][1].get<std::uint64_t>();
    if (end < begin || end > payload_size) {
      return LSE_ERROR(kIoError, "tensor '", name, "' offsets fall outside the file");
    }
    const std::size_t expected = dtype_storage_bytes(v.dtype, v.element_count());
    if (end - begin != expected) {
      return LSE_ERROR(kIoError, "tensor '", name, "' spans ",
                       std::to_string(end - begin), " bytes, shape implies ",
                       std::to_string(expected));
    }
    v.data = std::span<const std::byte>(payload + begin, end - begin);
    // A name repeated across shards would silently shadow; the index is
    // supposed to make that impossible, so treat it as a corrupt checkpoint.
    if (!tensors_.emplace(name, std::move(v)).second) {
      return LSE_ERROR(kIoError, "tensor '", name, "' appears in more than one shard");
    }
  }

  return OkStatus();
}

Result<SafeTensors> SafeTensors::open(const std::string& path) {
  SafeTensors out;
  out.path_ = path;
  LSE_RETURN_IF_ERROR(out.map_file(path));
  return out;
}

Result<ShardIndex> read_shard_index(const std::string& index_path) {
  std::ifstream in(index_path);
  if (!in) return LSE_ERROR(kIoError, "cannot open '", index_path, "'");
  nlohmann::json index;
  try {
    in >> index;
  } catch (const std::exception& e) {
    return LSE_ERROR(kIoError, "'", index_path, "' is not valid JSON: ", e.what());
  }
  if (!index.contains("weight_map") || !index["weight_map"].is_object()) {
    return LSE_ERROR(kIoError, "'", index_path, "' has no weight_map");
  }

  // Ordered so the tensor namespace does not depend on hash iteration order,
  // and so an error names the same shard on every run.
  std::set<std::string> shards;
  for (const auto& [tensor, file] : index["weight_map"].items()) {
    (void)tensor;
    shards.insert(file.get<std::string>());
  }
  if (shards.empty()) return LSE_ERROR(kIoError, "'", index_path, "' names no shards");

  const fs::path dir = fs::path(index_path).parent_path();

  ShardIndex out;
  out.named_tensors = index["weight_map"].size();
  out.shards.assign(shards.begin(), shards.end());
  std::error_code ec;
  for (const std::string& shard : out.shards) {
    // Follows symlinks, so a snapshot entry left dangling by an interrupted
    // download reads as absent — which is exactly what it is.
    if (!fs::exists(dir / shard, ec)) out.missing.push_back(shard);
  }
  return out;
}

Result<SafeTensors> SafeTensors::open_sharded(const std::string& index_path) {
  LSE_ASSIGN_OR(const ShardIndex index, read_shard_index(index_path));

  // A repo directory can exist with the download unfinished, so check the whole
  // set before mapping any of it rather than failing halfway through.
  if (!index.complete()) {
    return LSE_ERROR(kIoError, "shard '", index.missing.front(), "' named by '",
                     index_path, "' is missing; the checkpoint is incomplete");
  }

  const fs::path dir = fs::path(index_path).parent_path();
  SafeTensors out;
  out.path_ = index_path;
  for (const std::string& shard : index.shards) {
    LSE_RETURN_IF_ERROR(out.map_file((dir / shard).string()));
  }

  if (out.tensors_.size() != index.named_tensors) {
    return LSE_ERROR(kIoError, "'", index_path, "' names ",
                     std::to_string(index.named_tensors),
                     " tensors but the shards hold ",
                     std::to_string(out.tensors_.size()));
  }
  return out;
}

const TensorView* SafeTensors::find(std::string_view name) const {
  auto it = tensors_.find(std::string(name));
  return it == tensors_.end() ? nullptr : &it->second;
}

std::size_t SafeTensors::total_parameters() const noexcept {
  std::size_t n = 0;
  for (const auto& [_, v] : tensors_) n += v.element_count();
  return n;
}

std::size_t SafeTensors::logical_parameters(
    const quant::GroupAffineMap* quant) const noexcept {
  std::size_t n = 0;
  for (const auto& [name, v] : tensors_) {
    if (name.ends_with(".scales") || name.ends_with(".biases")) continue;
    std::size_t count = v.element_count();
    if (v.dtype == DType::kU32 && quant != nullptr) {
      // Only a plane with its own scales is a packed weight; a u32 tensor
      // without them is something else and is counted as it is stored.
      const std::optional<std::string> base = quant_plane_base(name);
      const bool packed =
          base.has_value() && tensors_.count(*base + ".scales") != 0;
      if (auto spec = quant->resolve(name); packed && spec.ok()) {
        count = count * 32u / static_cast<std::size_t>(spec->bits);
      }
    }
    n += count;
  }
  return n;
}

Result<LegacyExpertKey> migrate_legacy_expert_name(std::string_view name) {
  const std::size_t marker = name.find(".experts.");
  if (marker == std::string_view::npos) {
    return LSE_ERROR(kNotFound, "not a legacy expert key");
  }
  const std::size_t idx_begin = marker + 9;
  std::size_t idx_end = idx_begin;
  while (idx_end < name.size() && name[idx_end] >= '0' && name[idx_end] <= '9') {
    ++idx_end;
  }
  if (idx_end == idx_begin || idx_end >= name.size() || name[idx_end] != '.') {
    return LSE_ERROR(kInvalidArgument, "malformed legacy expert key '",
                     std::string(name), "'");
  }

  const std::string_view suffix = name.substr(idx_end + 1);
  std::string_view stacked;
  if (suffix == "w1.weight") stacked = "w_gate";
  else if (suffix == "w3.weight") stacked = "w_up";
  else if (suffix == "w2.weight") stacked = "w_down";
  else {
    return LSE_ERROR(kInvalidArgument, "unknown legacy expert weight '",
                     std::string(suffix), "'");
  }

  LegacyExpertKey out;
  out.stacked_name = std::string(name.substr(0, marker)) + "." + std::string(stacked);
  out.expert_index = std::atoi(std::string(name.substr(idx_begin, idx_end - idx_begin)).c_str());
  return out;
}

namespace {

// An empty environment variable cannot name a directory, so it reads as unset
// rather than as a cache root of "".
const char* env_value(const char* name) {
  const char* v = std::getenv(name);
  return (v != nullptr && v[0] != '\0') ? v : nullptr;
}

// huggingface_hub applies expanduser+expandvars to the cache path it reads, so
// HF_HUB_CACHE=~/hub and HF_HUB_CACHE=$HOME/hub name the same directory there.
// Without this a leading '~' becomes a literal directory name and the engine
// looks somewhere the Python library never would. An undefined variable is left
// as written, which is also what expandvars does.
std::string expand_path(std::string_view in) {
  std::string out;
  out.reserve(in.size());
  std::size_t i = 0;
  if (in.starts_with("~") && (in.size() == 1 || in[1] == '/')) {
    if (const char* home = env_value("HOME")) {
      out += home;
      i = 1;
    }
  }
  for (; i < in.size(); ++i) {
    if (in[i] != '$') {
      out += in[i];
      continue;
    }
    std::size_t begin = i + 1;
    const bool braced = begin < in.size() && in[begin] == '{';
    if (braced) ++begin;
    std::size_t end = begin;
    while (end < in.size() &&
           (std::isalnum(static_cast<unsigned char>(in[end])) != 0 ||
            in[end] == '_')) {
      ++end;
    }
    const bool closed = !braced || (end < in.size() && in[end] == '}');
    const char* value = nullptr;
    if (end > begin && closed) {
      value = std::getenv(std::string(in.substr(begin, end - begin)).c_str());
    }
    if (value == nullptr) {
      out += in[i];
      continue;
    }
    out += value;
    i = braced ? end : end - 1;
  }
  return out;
}

}  // namespace

std::string hf_home() {
  if (const char* v = env_value("HF_HOME")) return expand_path(v);
  if (const char* xdg = env_value("XDG_CACHE_HOME")) {
    return expand_path(xdg) + "/huggingface";
  }
  if (const char* home = env_value("HOME")) {
    return std::string(home) + "/.cache/huggingface";
  }
  return ".cache/huggingface";
}

std::string hf_cache_root() {
  // constants.py reads HF_HUB_CACHE with HUGGINGFACE_HUB_CACHE as its default,
  // and that one with $HF_HOME/hub as its default. So the legacy name keeps
  // working but can never outrank the current one, and HF_HOME is consulted
  // only when neither cache variable is set.
  if (const char* v = env_value("HF_HUB_CACHE")) return expand_path(v);
  if (const char* v = env_value("HUGGINGFACE_HUB_CACHE")) return expand_path(v);
  return hf_home() + "/hub";
}

namespace {

// A directory holding either one .safetensors or a set of shards named by a
// safetensors index, plus a config json.
Result<ModelPaths> from_directory(const fs::path& dir) {
  std::error_code ec;
  fs::path weights;
  // The index wins: picking the first shard off the directory iterator gives a
  // partial tensor set, and the failure surfaces far away as an architecture
  // that nothing recognizes.
  if (const fs::path index = dir / "model.safetensors.index.json";
      fs::exists(index, ec)) {
    weights = index;
  } else {
    for (const auto& entry : fs::directory_iterator(dir, ec)) {
      if (entry.path().extension() == ".safetensors") {
        weights = entry.path();
        break;
      }
    }
  }
  if (weights.empty()) {
    return LSE_ERROR(kNotFound, "no .safetensors in '", dir.string(), "'");
  }

  ModelPaths out;
  out.weights = weights.string();
  // A shard index is already a .json, so the sidecar rule would name the index
  // as its own config and every field would silently take its default.
  fs::path sidecar = out.weights.ends_with(".index.json") ? fs::path{} : weights;
  if (!sidecar.empty()) sidecar.replace_extension(".json");
  if (!sidecar.empty() && fs::exists(sidecar, ec)) {
    out.config = sidecar.string();
  } else if (fs::exists(dir / "config.json", ec)) {
    out.config = (dir / "config.json").string();
  } else {
    return LSE_ERROR(kNotFound, "no config json beside '", out.weights, "'");
  }
  return out;
}

constexpr std::string_view kRepoPrefix = "models--";

// hub/models--<org>--<name>: the separator is DOUBLED. A single dash here meant
// naming a model by its repo id never resolved.
std::string repo_dir_name(std::string_view repo_id) {
  std::string out(kRepoPrefix);
  for (char c : repo_id) {
    if (c == '/') {
      out += "--";
    } else {
      out += c;
    }
  }
  return out;
}

// The inverse, for display. Splits on the first "--" after the prefix: an
// organization name cannot contain one, a model name can.
std::string repo_id_from_dir(std::string_view dir_name) {
  std::string_view rest = dir_name.substr(kRepoPrefix.size());
  const std::size_t sep = rest.find("--");
  if (sep == std::string_view::npos) return std::string(rest);
  return std::string(rest.substr(0, sep)) + "/" +
         std::string(rest.substr(sep + 2));
}

std::string join(const std::vector<std::string>& items) {
  std::string out;
  for (const std::string& s : items) {
    if (!out.empty()) out += ", ";
    out += s;
  }
  return out;
}

// Every models--* directory in the cache, sorted, with the repo id each holds.
std::vector<std::pair<std::string, fs::path>> cache_repos() {
  std::vector<std::pair<std::string, fs::path>> out;
  std::error_code ec;
  const fs::path root(hf_cache_root());
  for (const auto& e : fs::directory_iterator(root, ec)) {
    const std::string name = e.path().filename().string();
    if (!name.starts_with(kRepoPrefix)) continue;
    if (!fs::is_directory(e.path(), ec)) continue;
    out.emplace_back(repo_id_from_dir(name), e.path());
  }
  std::sort(out.begin(), out.end());
  return out;
}

// The revision a repo directory resolves to. refs/main names the commit the
// last download landed on; with one snapshot the name is unambiguous without
// it, with several and no ref it is not.
Result<fs::path> repo_snapshot(const fs::path& repo) {
  std::error_code ec;
  const fs::path snapshots = repo / "snapshots";
  if (!fs::is_directory(snapshots, ec)) {
    return LSE_ERROR(kNotFound, "'", repo.filename().string(),
                     "' has no snapshots directory; the download did not "
                     "finish");
  }
  std::vector<fs::path> found;
  for (const auto& e : fs::directory_iterator(snapshots, ec)) {
    if (fs::is_directory(e.path(), ec)) found.push_back(e.path());
  }
  std::sort(found.begin(), found.end());
  if (found.empty()) {
    return LSE_ERROR(kNotFound, "'", repo.filename().string(),
                     "' has no revision in snapshots/; the download did not "
                     "finish");
  }
  if (found.size() == 1) return found.front();

  std::ifstream ref(repo / "refs" / "main");
  std::string commit;
  if (ref && std::getline(ref, commit) && !commit.empty()) {
    for (const fs::path& p : found) {
      if (p.filename() == commit) return p;
    }
  }
  std::vector<std::string> names;
  names.reserve(found.size());
  for (const fs::path& p : found) names.push_back(p.filename().string());
  return LSE_ERROR(kInvalidArgument, "'", repo_id_from_dir(repo.filename().string()),
                   "' has ", std::to_string(found.size()),
                   " revisions in the cache and no refs/main to choose between "
                   "them; name a snapshot path instead (tried: ", join(names),
                   ")");
}

}  // namespace

Result<ModelPaths> resolve_model(const std::string& name_or_path) {
  std::error_code ec;
  const fs::path p(name_or_path);

  if (fs::is_directory(p, ec)) return from_directory(p);

  if (fs::is_regular_file(p, ec)) {
    ModelPaths out;
    out.weights = name_or_path;
    fs::path sidecar = p;
    sidecar.replace_extension(".json");
    if (!fs::exists(sidecar, ec)) {
      return LSE_ERROR(kNotFound, "no config json beside '", name_or_path, "'");
    }
    out.config = sidecar.string();
    return out;
  }

  // Treat it as an HF repo id and look in the local cache. A bare model name
  // resolves too, but only when it picks out exactly one repo: silently
  // choosing between two organizations' builds of the same model would load
  // something other than what was asked for.
  const std::string root = hf_cache_root();
  const auto repos = cache_repos();
  std::vector<std::string> ids;
  ids.reserve(repos.size());
  const fs::path* exact = nullptr;
  std::vector<const std::pair<std::string, fs::path>*> partial;
  for (const auto& entry : repos) {
    ids.push_back(entry.first);
    if (entry.first == name_or_path) {
      exact = &entry.second;
      continue;
    }
    const std::size_t slash = entry.first.find('/');
    if (slash != std::string::npos &&
        entry.first.substr(slash + 1) == name_or_path) {
      partial.push_back(&entry);
    }
  }

  if (exact != nullptr) {
    LSE_ASSIGN_OR(const fs::path snap, repo_snapshot(*exact));
    return from_directory(snap);
  }
  if (partial.size() == 1) {
    LSE_ASSIGN_OR(const fs::path snap, repo_snapshot(partial.front()->second));
    return from_directory(snap);
  }
  if (partial.size() > 1) {
    std::vector<std::string> names;
    names.reserve(partial.size());
    for (const auto* e : partial) names.push_back(e->first);
    return LSE_ERROR(kInvalidArgument, "'", name_or_path, "' names ",
                     std::to_string(partial.size()),
                     " models in the HF cache at ", root,
                     "; give the full repo id (tried: ", join(names), ")");
  }

  return LSE_ERROR(kNotFound, "'", name_or_path,
                   "' is not a path, and the HF cache at ", root, " has no ",
                   repo_dir_name(name_or_path),
                   ids.empty() ? " (tried: the cache holds no models)"
                               : " (tried: " + join(ids) + ")");
}

std::string_view to_string(Loadable l) noexcept {
  switch (l) {
    case Loadable::kYes: return "yes";
    case Loadable::kNo: return "no";
    case Loadable::kIncomplete: return "incomplete";
    case Loadable::kUnknown: return "unknown";
  }
  return "unknown";
}

namespace {

// Snapshot entries are symlinks into blobs/, so walking a repo and summing
// st_size counts every byte twice. Dedup by (device, inode), which also gets a
// cache downloaded with HF_HUB_DISABLE_SYMLINKS right, where the bytes are in
// the snapshot and blobs/ may not exist.
std::uintmax_t directory_bytes(const fs::path& root) {
  std::set<std::pair<dev_t, ino_t>> seen;
  std::uintmax_t total = 0;
  std::error_code ec;
  fs::recursive_directory_iterator it(
      root, fs::directory_options::skip_permission_denied, ec);
  if (ec) return 0;
  const fs::recursive_directory_iterator end;
  for (; it != end; it.increment(ec)) {
    if (ec) break;
    struct stat st {};
    // stat, not lstat: a snapshot symlink should be charged its blob's size,
    // and a dangling one fails here and is charged nothing.
    if (::stat(it->path().c_str(), &st) != 0) continue;
    if (!S_ISREG(st.st_mode)) continue;
    if (!seen.emplace(st.st_dev, st.st_ino).second) continue;
    total += static_cast<std::uintmax_t>(st.st_size);
  }
  return total;
}

std::string read_file(const fs::path& p) {
  std::ifstream in(p, std::ios::binary);
  if (!in) return {};
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

bool dir_holds_extension(const fs::path& dir, std::string_view ext) {
  std::error_code ec;
  for (const auto& e : fs::directory_iterator(dir, ec)) {
    if (e.path().extension() == ext) return true;
  }
  return false;
}

std::int64_t last_dim(const Shape& s) {
  return s.rank() == 0 ? 0 : s.dim(s.rank() - 1);
}

// The group-affine preconditions WeightBinder::bind_quantized enforces, checked
// against tensor metadata instead of a device upload so a listing can answer
// without a backend.
//
// LANDMINE: these rules are bind_quantized's, restated. If that function gains
// or loses a precondition this must follow it — a repo that fails to bind while
// listing as loadable is the one output worse than saying nothing.
Status group_affine_bindable(const SafeTensors& weights,
                             const quant::GroupAffineMap& quant) {
  for (const auto& [name, v] : weights.tensors()) {
    const std::optional<std::string> base = quant_plane_base(name);
    if (!base.has_value()) continue;
    const TensorView* scales = weights.find(*base + ".scales");
    const TensorView* biases = weights.find(*base + ".biases");
    if (scales == nullptr && biases == nullptr) continue;
    if (scales == nullptr || biases == nullptr) {
      return LSE_ERROR(kInvalidArgument, "'", name, "' has a '",
                       scales != nullptr ? ".scales" : ".biases",
                       "' plane but no '",
                       scales != nullptr ? ".biases" : ".scales", "' plane");
    }
    if (v.dtype != DType::kU32) {
      return LSE_ERROR(kInvalidArgument, "'", name,
                       "' has .scales/.biases planes but is stored as ",
                       std::string(to_string(v.dtype)),
                       "; a group-affine plane is packed into u32 lanes");
    }
    if (v.shape.rank() != 2 && v.shape.rank() != 3) {
      return LSE_ERROR(kUnimplemented, "'", name, "' is ", v.shape.to_string(),
                       "; a group-affine weight is read as an [out, in] matrix "
                       "or an [expert, out, in] stack of them");
    }
    const Result<quant::GroupAffine> spec = quant.resolve_checked(
        name, last_dim(v.shape), last_dim(scales->shape));
    LSE_RETURN_IF_ERROR(spec.status());
  }
  return OkStatus();
}

// mlx-lm omits `mode` for affine, which is also what check_mode treats as the
// default, so an absent key is reported as affine rather than as unknown.
std::string describe_quantization(const std::string& config_text,
                                  const quant::GroupAffineMap& quant) {
  if (!quant.has_global() && quant.override_count() == 0) return "none";
  std::string mode = "affine";
  try {
    const nlohmann::json j = nlohmann::json::parse(config_text);
    for (const char* key : {"quantization", "quantization_config"}) {
      if (j.contains(key) && j[key].is_object() && j[key].contains("mode") &&
          j[key]["mode"].is_string()) {
        mode = j[key]["mode"].get<std::string>();
        break;
      }
    }
  } catch (const std::exception&) {
    return {};
  }
  if (!quant.has_global()) {
    return mode + " per-module only (" + std::to_string(quant.override_count()) +
           " override" + (quant.override_count() == 1 ? "" : "s") + ")";
  }
  std::string out = mode + " " + std::to_string(quant.global().bits) + "-bit g" +
                    std::to_string(quant.global().group_size);
  if (quant.override_count() != 0) {
    out += " +" + std::to_string(quant.override_count()) + " override" +
           (quant.override_count() == 1 ? "" : "s");
  }
  return out;
}

void read_config_facts(const std::string& config_text, CacheModel* m) {
  if (config_text.empty()) return;
  try {
    const nlohmann::json j = nlohmann::json::parse(config_text);
    // model_type in preference to architectures[0]: both are the config's own
    // answer, and the class name is three times as wide in a listing.
    if (j.contains("model_type") && j["model_type"].is_string()) {
      m->architecture = j["model_type"].get<std::string>();
    } else if (j.contains("architectures") && j["architectures"].is_array() &&
               !j["architectures"].empty() && j["architectures"][0].is_string()) {
      m->architecture = j["architectures"][0].get<std::string>();
    }
    // The tower's own config, which is what makes the checkpoint multimodal.
    // Its absence is as much a fact as its presence, hence the second flag.
    m->multimodal = j.contains("vision_config");
    m->multimodal_known = true;
  } catch (const std::exception&) {
    // Leave the fields unset; an unparsable config is reported by the verdict.
  }
}

}  // namespace

CacheModel inspect_model_dir(const std::string& dir, std::string_view repo_id) {
  CacheModel m;
  m.repo_id = std::string(repo_id);
  m.path = dir;

  std::error_code ec;
  if (!fs::is_directory(dir, ec)) {
    m.reason = "'" + dir + "' is not a directory";
    return m;
  }
  m.bytes = directory_bytes(dir);

  const Result<ModelPaths> paths = from_directory(dir);
  if (!paths.ok()) {
    if (dir_holds_extension(dir, ".gguf")) {
      m.loadable = Loadable::kNo;
      m.reason =
          "GGUF container; this engine reads safetensors and has no GGUF "
          "reader";
      read_config_facts(read_file(fs::path(dir) / "config.json"), &m);
      return m;
    }
    m.loadable = Loadable::kIncomplete;
    m.reason = fs::exists(fs::path(dir) / "config.json", ec)
                   ? "no .safetensors in the snapshot; the download did not "
                     "finish"
                   : "no config.json and no .safetensors in the snapshot; the "
                     "download did not finish";
    return m;
  }

  const std::string config_text = read_file(paths->config);
  read_config_facts(config_text, &m);

  // Completeness first, and before mapping: the index is the authority on what
  // a finished download is, a repo can hold a valid index next to missing
  // shards, and an interrupted download can leave a truncated config that would
  // otherwise be reported as a bad model rather than as a partial one.
  if (paths->weights.ends_with(".index.json")) {
    const Result<ShardIndex> index = read_shard_index(paths->weights);
    if (!index.ok()) {
      m.loadable = Loadable::kNo;
      m.reason = index.status().to_string();
      return m;
    }
    if (!index->complete()) {
      m.loadable = Loadable::kIncomplete;
      m.reason = std::to_string(index->missing.size()) + " of " +
                 std::to_string(index->shards.size()) +
                 " shards named by the index are missing (" +
                 index->missing.front() + " first); the download did not finish";
      return m;
    }
  }

  const Result<Config> cfg = Config::from_json_file(paths->config);
  if (!cfg.ok()) {
    m.loadable = Loadable::kNo;
    m.reason = cfg.status().to_string();
    return m;
  }
  m.quantization = describe_quantization(config_text, cfg->quantization);

  // Headers only in effect: the mapping is lazy and nothing below reads a
  // tensor's payload, so this costs address space rather than 36 GB of reads.
  Result<SafeTensors> weights = paths->weights.ends_with(".index.json")
                                    ? SafeTensors::open_sharded(paths->weights)
                                    : SafeTensors::open(paths->weights);
  if (!weights.ok()) {
    m.loadable = Loadable::kNo;
    m.reason = weights.status().to_string();
    return m;
  }
  m.parameters = weights->logical_parameters(&cfg->quantization);

  // The engine's own detection, not a second opinion about it: whatever the
  // loader would match here is what the verdict reports.
  const Result<const ModelArch*> arch = detect_architecture(*cfg, *weights);
  if (!arch.ok()) {
    m.loadable = Loadable::kNo;
    m.reason = arch.status().to_string();
    return m;
  }
  m.engine_arch = std::string((*arch)->name);

  const Status bindable = group_affine_bindable(*weights, cfg->quantization);
  if (!bindable.ok()) {
    m.loadable = Loadable::kNo;
    m.reason = bindable.to_string();
    return m;
  }

  m.loadable = Loadable::kYes;
  return m;
}

Result<std::vector<CacheModel>> list_cached_models() {
  const std::string root = hf_cache_root();
  std::error_code ec;
  if (!fs::is_directory(root, ec)) {
    return LSE_ERROR(kNotFound, "the HF cache directory '", root,
                     "' does not exist");
  }

  std::vector<CacheModel> out;
  for (const auto& [repo_id, repo] : cache_repos()) {
    const Result<fs::path> snap = repo_snapshot(repo);
    if (!snap.ok()) {
      CacheModel m;
      m.repo_id = repo_id;
      // A missing snapshot is an unfinished download; anything else (several
      // revisions with no ref to choose between them) is a repo this cannot
      // speak for, which is not the same claim.
      m.loadable = snap.status().code() == StatusCode::kNotFound
                       ? Loadable::kIncomplete
                       : Loadable::kUnknown;
      m.reason = snap.status().to_string();
      m.bytes = directory_bytes(repo);
      out.push_back(std::move(m));
      continue;
    }
    CacheModel m = inspect_model_dir(snap->string(), repo_id);
    // Charged against the whole repo, not the revision: an unfinished download
    // holds blobs that no snapshot points at, and they are still on the disk.
    m.bytes = directory_bytes(repo);
    out.push_back(std::move(m));
  }
  return out;
}

}  // namespace lse::model
