#include "lse/model/weights.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

namespace lse::model {

namespace fs = std::filesystem;

namespace {

DType dtype_from_safetensors(std::string_view s) noexcept {
  if (s == "BF16") return DType::kBF16;
  if (s == "F16") return DType::kF16;
  if (s == "F32") return DType::kF32;
  if (s == "I32") return DType::kI32;
  if (s == "I8") return DType::kI8;
  if (s == "U8") return DType::kU8;
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

SafeTensors::~SafeTensors() {
  if (mapping_ != nullptr) ::munmap(mapping_, mapping_size_);
}

SafeTensors::SafeTensors(SafeTensors&& other) noexcept
    : path_(std::move(other.path_)),
      mapping_(other.mapping_),
      mapping_size_(other.mapping_size_),
      tensors_(std::move(other.tensors_)) {
  other.mapping_ = nullptr;
  other.mapping_size_ = 0;
}

SafeTensors& SafeTensors::operator=(SafeTensors&& other) noexcept {
  if (this != &other) {
    if (mapping_ != nullptr) ::munmap(mapping_, mapping_size_);
    path_ = std::move(other.path_);
    mapping_ = other.mapping_;
    mapping_size_ = other.mapping_size_;
    tensors_ = std::move(other.tensors_);
    other.mapping_ = nullptr;
    other.mapping_size_ = 0;
  }
  return *this;
}

Result<SafeTensors> SafeTensors::open(const std::string& path) {
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

  SafeTensors out;
  out.path_ = path;
  out.mapping_ = map;
  out.mapping_size_ = file_size;

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
    out.tensors_.emplace(name, std::move(v));
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

std::string hf_cache_root() {
  if (const char* hf_home = std::getenv("HF_HOME")) {
    return std::string(hf_home) + "/hub";
  }
  if (const char* xdg = std::getenv("XDG_CACHE_HOME")) {
    return std::string(xdg) + "/huggingface/hub";
  }
  if (const char* home = std::getenv("HOME")) {
    return std::string(home) + "/.cache/huggingface/hub";
  }
  return ".cache/huggingface/hub";
}

namespace {

// A directory holding one .safetensors plus its sidecar .json.
Result<ModelPaths> from_directory(const fs::path& dir) {
  std::error_code ec;
  fs::path weights;
  for (const auto& entry : fs::directory_iterator(dir, ec)) {
    if (entry.path().extension() == ".safetensors") {
      weights = entry.path();
      break;
    }
  }
  if (weights.empty()) {
    return LSE_ERROR(kNotFound, "no .safetensors in '", dir.string(), "'");
  }

  ModelPaths out;
  out.weights = weights.string();
  fs::path sidecar = weights;
  sidecar.replace_extension(".json");
  if (fs::exists(sidecar, ec)) {
    out.config = sidecar.string();
  } else if (fs::exists(dir / "config.json", ec)) {
    out.config = (dir / "config.json").string();
  } else {
    return LSE_ERROR(kNotFound, "no config json beside '", out.weights, "'");
  }
  return out;
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

  // Treat it as an HF repo id and look in the local cache.
  std::string flat = name_or_path;
  for (char& c : flat) {
    if (c == '/') c = '-';
  }
  const fs::path repo = fs::path(hf_cache_root()) / ("models--" + flat);
  const fs::path snapshots = repo / "snapshots";
  if (fs::is_directory(snapshots, ec)) {
    for (const auto& snap : fs::directory_iterator(snapshots, ec)) {
      auto found = from_directory(snap.path());
      if (found.ok()) return found;
    }
  }

  return LSE_ERROR(kNotFound, "'", name_or_path,
                   "' is not a path and was not found in the HF cache at ",
                   hf_cache_root());
}

}  // namespace lse::model
