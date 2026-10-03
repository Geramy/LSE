#include "lse/model/dflash2_convert.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>

#include <nlohmann/json.hpp>

#include "lse/core/sha256.hpp"

#if defined(__APPLE__)
#include <CommonCrypto/CommonDigest.h>
#endif

// Every float operation below must round exactly as NumPy's float32 ufuncs do:
// one IEEE operation per step, never fused into an FMA.
#if defined(__clang__)
#pragma clang fp contract(off)
#endif

namespace lse::model {
namespace fs = std::filesystem;
using Json = nlohmann::ordered_json;

static_assert(std::endian::native == std::endian::little,
              "safetensors payloads are little endian");

namespace {

constexpr std::int64_t kGroupSize = 64;
constexpr std::int64_t kRowBlock = 128;
constexpr std::size_t kIoChunk = 8u << 20;
constexpr std::string_view kReferenceUrl =
    "https://github.com/z-lab/dflash/blob/"
    "07ebd93db9f472af339b644bb70221ad8428328a/dflash/model_mlx.py";
constexpr std::string_view kFormat = "MLX affine Q8 group64";
constexpr std::string_view kConversion =
    "FP32 min/max; BF16 scale and bias; codes selected against stored scale/bias";
constexpr std::string_view kCacheName = "lse-q8g64";

// ---------------------------------------------------------------------------
// Python json.dumps, byte for byte.

void python_string(std::string& out, std::string_view s) {
  // ensure_ascii: everything outside ' '..'~' is escaped, as are '"' and '\'.
  static constexpr char hex[] = "0123456789abcdef";
  const auto unit = [&](std::uint32_t u) {
    out += "\\u";
    for (int shift = 12; shift >= 0; shift -= 4) out += hex[(u >> shift) & 15u];
  };
  out += '"';
  for (std::size_t i = 0; i < s.size();) {
    const auto c = static_cast<unsigned char>(s[i]);
    if (c < 0x80) {
      ++i;
      switch (c) {
        case '"': out += "\\\""; continue;
        case '\\': out += "\\\\"; continue;
        case '\n': out += "\\n"; continue;
        case '\r': out += "\\r"; continue;
        case '\t': out += "\\t"; continue;
        case '\b': out += "\\b"; continue;
        case '\f': out += "\\f"; continue;
        default: break;
      }
      if (c >= 0x20 && c <= 0x7e) out += static_cast<char>(c);
      else unit(c);
      continue;
    }
    // nlohmann has already rejected malformed UTF-8 at parse time.
    const int extra = c >= 0xf0 ? 3 : c >= 0xe0 ? 2 : 1;
    std::uint32_t cp = c & (0x3fu >> extra);
    for (int k = 1; k <= extra && i + static_cast<std::size_t>(k) < s.size(); ++k)
      cp = (cp << 6) | (static_cast<unsigned char>(s[i + static_cast<std::size_t>(k)]) & 0x3fu);
    i += static_cast<std::size_t>(extra) + 1;
    if (cp >= 0x10000) {
      cp -= 0x10000;
      unit(0xd800u | (cp >> 10));
      unit(0xdc00u | (cp & 0x3ffu));
    } else {
      unit(cp);
    }
  }
  out += '"';
}

// float.__repr__: the shortest round-trip digits, positional for exponents in
// [-4, 16), scientific with a signed two-digit exponent otherwise.
void python_float(std::string& out, double v) {
  if (std::isnan(v)) { out += "NaN"; return; }
  if (std::isinf(v)) { out += v > 0 ? "Infinity" : "-Infinity"; return; }
  char buf[64];
  const auto r = std::to_chars(buf, buf + sizeof buf, v, std::chars_format::scientific);
  std::string_view text(buf, static_cast<std::size_t>(r.ptr - buf));
  if (text.front() == '-') { out += '-'; text.remove_prefix(1); }
  const auto e = text.find('e');
  std::string digits;
  for (char ch : text.substr(0, e)) if (ch != '.') digits += ch;
  int exponent = 0;
  const auto exp_text = text.substr(e + 1);
  std::from_chars(exp_text.data() + (exp_text.front() == '+' ? 1 : 0),
                  exp_text.data() + exp_text.size(), exponent);
  const int decpt = exponent + 1;
  const int n = static_cast<int>(digits.size());
  if (decpt <= -4 || decpt > 16) {
    out += digits[0];
    if (n > 1) { out += '.'; out += digits.substr(1); }
    out += exponent < 0 ? "e-" : "e+";
    const int a = std::abs(exponent);
    if (a < 10) out += '0';
    out += std::to_string(a);
  } else if (decpt <= 0) {
    out += "0.";
    out.append(static_cast<std::size_t>(-decpt), '0');
    out += digits;
  } else if (decpt >= n) {
    out += digits;
    out.append(static_cast<std::size_t>(decpt - n), '0');
    out += ".0";
  } else {
    out += digits.substr(0, static_cast<std::size_t>(decpt));
    out += '.';
    out += digits.substr(static_cast<std::size_t>(decpt));
  }
}

void python_dump(std::string& out, const Json& v, bool indent, int level) {
  const auto newline = [&](int depth) {
    out += '\n';
    out.append(static_cast<std::size_t>(depth) * 2, ' ');
  };
  switch (v.type()) {
    case Json::value_t::null: out += "null"; return;
    case Json::value_t::boolean: out += v.get<bool>() ? "true" : "false"; return;
    case Json::value_t::number_integer: out += std::to_string(v.get<std::int64_t>()); return;
    case Json::value_t::number_unsigned: out += std::to_string(v.get<std::uint64_t>()); return;
    case Json::value_t::number_float: python_float(out, v.get<double>()); return;
    case Json::value_t::string: python_string(out, v.get_ref<const std::string&>()); return;
    case Json::value_t::array: {
      if (v.empty()) { out += "[]"; return; }
      out += '[';
      bool first = true;
      for (const auto& item : v) {
        if (!first) out += ',';
        first = false;
        if (indent) newline(level + 1);
        python_dump(out, item, indent, level + 1);
      }
      if (indent) newline(level);
      out += ']';
      return;
    }
    case Json::value_t::object: {
      if (v.empty()) { out += "{}"; return; }
      out += '{';
      bool first = true;
      for (const auto& [key, item] : v.items()) {
        if (!first) out += ',';
        first = false;
        if (indent) newline(level + 1);
        python_string(out, key);
        out += indent ? ": " : ":";
        python_dump(out, item, indent, level + 1);
      }
      if (indent) newline(level);
      out += '}';
      return;
    }
    default: out += "null"; return;  // binary/discarded never come from text
  }
}

std::string python_json(const Json& v, bool indent) {
  std::string out;
  python_dump(out, v, indent, 0);
  return out;
}

// Python truthiness of a parsed JSON value.
bool truthy(const Json* v) {
  if (v == nullptr) return false;
  switch (v->type()) {
    case Json::value_t::null: return false;
    case Json::value_t::boolean: return v->get<bool>();
    case Json::value_t::number_integer: return v->get<std::int64_t>() != 0;
    case Json::value_t::number_unsigned: return v->get<std::uint64_t>() != 0;
    case Json::value_t::number_float: return v->get<double>() != 0.0;
    case Json::value_t::string:
    case Json::value_t::array:
    case Json::value_t::object: return !v->empty();
    default: return false;
  }
}

const Json* member(const Json& object, std::string_view key) {
  if (!object.is_object()) return nullptr;
  const auto it = object.find(std::string(key));
  return it == object.end() ? nullptr : &*it;
}

// ---------------------------------------------------------------------------
// Files.

class Fd {
 public:
  Fd() = default;
  explicit Fd(int fd) : fd_(fd) {}
  Fd(const Fd&) = delete;
  Fd& operator=(const Fd&) = delete;
  Fd(Fd&& o) noexcept : fd_(std::exchange(o.fd_, -1)) {}
  ~Fd() { if (fd_ >= 0) ::close(fd_); }
  [[nodiscard]] int get() const { return fd_; }
  [[nodiscard]] bool valid() const { return fd_ >= 0; }

 private:
  int fd_ = -1;
};

std::string errno_text() { return std::strerror(errno); }

Result<Fd> open_read(const fs::path& path) {
  Fd fd(::open(path.c_str(), O_RDONLY | O_CLOEXEC));
  if (!fd.valid()) return LSE_ERROR(kIoError, "cannot open ", path.string(), ": ", errno_text());
  return fd;
}

Status pread_exact(int fd, void* data, std::size_t size, std::uint64_t offset) {
  auto* p = static_cast<char*>(data);
  while (size > 0) {
    const auto got = ::pread(fd, p, size, static_cast<off_t>(offset));
    if (got < 0 && errno == EINTR) continue;
    if (got <= 0) return LSE_ERROR(kIoError, "truncated safetensors payload");
    p += got;
    size -= static_cast<std::size_t>(got);
    offset += static_cast<std::uint64_t>(got);
  }
  return OkStatus();
}

Result<std::string> read_text(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return LSE_ERROR(kNotFound, "cannot read ", path.string());
  std::ostringstream text;
  text << in.rdbuf();
  return text.str();
}

Result<Json> read_json(const fs::path& path) {
  LSE_ASSIGN_OR(const std::string text, read_text(path));
  Json j = Json::parse(text, nullptr, /*allow_exceptions=*/false);
  if (j.is_discarded()) return LSE_ERROR(kInvalidArgument, "invalid JSON in ", path.string());
  return j;
}

class Hasher {
 public:
  Hasher() {
#if defined(__APPLE__)
    CC_SHA256_Init(&ctx_);
#endif
  }
  void update(const void* data, std::size_t size) {
#if defined(__APPLE__)
    const auto* p = static_cast<const unsigned char*>(data);
    while (size > 0) {
      const auto n = std::min<std::size_t>(size, 1u << 30);
      CC_SHA256_Update(&ctx_, p, static_cast<CC_LONG>(n));
      p += n;
      size -= n;
    }
#else
    hash_.update(std::span(static_cast<const std::byte*>(data), size));
#endif
  }
  std::string hex() {
    std::array<std::byte, 32> digest{};
#if defined(__APPLE__)
    CC_SHA256_Final(reinterpret_cast<unsigned char*>(digest.data()), &ctx_);
#else
    digest = hash_.finish();
#endif
    static constexpr char hexd[] = "0123456789abcdef";
    std::string out(64, '0');
    for (std::size_t i = 0; i < digest.size(); ++i) {
      const auto b = static_cast<unsigned>(digest[i]);
      out[2 * i] = hexd[b >> 4];
      out[2 * i + 1] = hexd[b & 15u];
    }
    return out;
  }

 private:
#if defined(__APPLE__)
  CC_SHA256_CTX ctx_{};
#else
  Sha256 hash_;
#endif
};

// Buffered sequential writer that hashes what it writes.
class Writer {
 public:
  explicit Writer(Fd fd) : fd_(std::move(fd)) { buffer_.reserve(kIoChunk); }
  Status write(const void* data, std::size_t size) {
    hash_.update(data, size);
    written_ += size;
    const auto* p = static_cast<const char*>(data);
    while (size > 0) {
      const auto n = std::min(size, kIoChunk - buffer_.size());
      buffer_.insert(buffer_.end(), p, p + n);
      p += n;
      size -= n;
      if (buffer_.size() == kIoChunk) LSE_RETURN_IF_ERROR(flush());
    }
    return OkStatus();
  }
  Status finish() {
    LSE_RETURN_IF_ERROR(flush());
    if (::fsync(fd_.get()) != 0) return LSE_ERROR(kIoError, "fsync failed: ", errno_text());
    return OkStatus();
  }
  [[nodiscard]] std::uint64_t written() const { return written_; }
  std::string sha256() { return hash_.hex(); }

 private:
  Status flush() {
    const char* p = buffer_.data();
    std::size_t left = buffer_.size();
    while (left > 0) {
      const auto n = ::write(fd_.get(), p, left);
      if (n < 0 && errno == EINTR) continue;
      if (n <= 0) return LSE_ERROR(kIoError, "write failed: ", errno_text());
      p += n;
      left -= static_cast<std::size_t>(n);
    }
    buffer_.clear();
    return OkStatus();
  }
  Fd fd_;
  std::vector<char> buffer_;
  Hasher hash_;
  std::uint64_t written_ = 0;
};

Status write_file(const fs::path& path, std::string_view text) {
  Fd fd(::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644));
  if (!fd.valid()) return LSE_ERROR(kIoError, "cannot create ", path.string(), ": ", errno_text());
  Writer w(std::move(fd));
  LSE_RETURN_IF_ERROR(w.write(text.data(), text.size()));
  return w.finish();
}

void fsync_dir(const fs::path& dir) {
  Fd fd(::open(dir.c_str(), O_RDONLY | O_CLOEXEC));
  if (fd.valid()) ::fsync(fd.get());
}

// ---------------------------------------------------------------------------
// Progress.

class Progress {
 public:
  explicit Progress(const DFlash2ProgressFn& fn) : fn_(fn) {}
  void report(DFlash2ConvertProgress::Phase phase, std::uint64_t done, std::uint64_t total,
              std::string_view tensor = {}) {
    if (fn_) {
      fn_(DFlash2ConvertProgress{phase, done, total, tensor});
      return;
    }
    // One line per whole percent, plus each phase's first and last line.
    const int percent = total ? static_cast<int>(done * 100 / total) : 100;
    if (phase == last_phase_ && percent == last_percent_ && done != total) return;
    last_phase_ = phase;
    last_percent_ = percent;
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count();
    std::fprintf(stderr,
                 "[dflash2-convert] phase=%.*s done=%llu total=%llu percent=%d elapsed_s=%.1f%s%.*s\n",
                 static_cast<int>(to_string(phase).size()), to_string(phase).data(),
                 static_cast<unsigned long long>(done), static_cast<unsigned long long>(total),
                 percent, seconds, tensor.empty() ? "" : " tensor=",
                 static_cast<int>(tensor.size()), tensor.data());
  }

 private:
  const DFlash2ProgressFn& fn_;
  DFlash2ConvertProgress::Phase last_phase_ = DFlash2ConvertProgress::Phase::kFinish;
  int last_percent_ = -1;
  std::chrono::steady_clock::time_point start_ = std::chrono::steady_clock::now();
};

Result<std::string> hash_file(const fs::path& path, Progress& progress) {
  LSE_ASSIGN_OR(Fd fd, open_read(path));
  struct stat st {};
  if (::fstat(fd.get(), &st) != 0) return LSE_ERROR(kIoError, "cannot stat ", path.string());
  const auto total = static_cast<std::uint64_t>(st.st_size);
  std::vector<char> buffer(kIoChunk);
  Hasher hash;
  std::uint64_t done = 0;
  progress.report(DFlash2ConvertProgress::Phase::kHashSource, 0, total);
  while (done < total) {
    const auto n = static_cast<std::size_t>(std::min<std::uint64_t>(kIoChunk, total - done));
    LSE_RETURN_IF_ERROR(pread_exact(fd.get(), buffer.data(), n, done));
    hash.update(buffer.data(), n);
    done += n;
    progress.report(DFlash2ConvertProgress::Phase::kHashSource, done, total);
  }
  return hash.hex();
}

// ---------------------------------------------------------------------------
// Quantization, mirroring quantize_matrix() step for step in float32.

float widen(std::uint16_t bits) { return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16); }

std::uint16_t bf16(float value) {
  const auto bits = std::bit_cast<std::uint32_t>(value);
  const std::uint32_t rounded = bits + 0x7fffu + ((bits >> 16) & 1u);  // wraps like np.uint32
  return static_cast<std::uint16_t>(rounded >> 16);
}

float nearest(float x) { return std::copysign(std::floor(std::fabs(x) + 0.5f), x); }

// np.maximum propagates NaN from either side.
float np_maximum(float a, float b) {
  if (std::isnan(a)) return a;
  if (std::isnan(b)) return b;
  return a >= b ? a : b;
}

void quantize_group(const std::uint16_t* raw, std::uint8_t* codes, std::uint16_t* scale_out,
                    std::uint16_t* bias_out) {
  float values[kGroupSize];
  for (std::int64_t i = 0; i < kGroupSize; ++i) values[i] = widen(raw[i]);
  float lo = values[0], hi = values[0];
  bool nan = std::isnan(values[0]);
  for (std::int64_t i = 1; i < kGroupSize; ++i) {
    lo = values[i] < lo ? values[i] : lo;
    hi = values[i] > hi ? values[i] : hi;
    nan |= std::isnan(values[i]);
  }
  if (nan) lo = hi = std::numeric_limits<float>::quiet_NaN();  // np.min/np.max propagate NaN
  const bool low_edge = std::fabs(lo) > std::fabs(hi);
  float scale = np_maximum((hi - lo) / 255.0f, 1e-7f);
  scale = low_edge ? scale : -scale;
  const float edge = low_edge ? lo : hi;
  const float code_at_edge = nearest(edge / scale);
  float bias = 0.0f;
  if (code_at_edge != 0.0f) {
    scale = edge / code_at_edge;
    bias = edge;
  }
  const std::uint16_t stored_scale = bf16(scale), stored_bias = bf16(bias);
  scale = widen(stored_scale);
  bias = widen(stored_bias);
  const float inverse = scale != 0.0f ? 1.0f / scale : 0.0f;
  for (std::int64_t i = 0; i < kGroupSize; ++i) {
    float c = nearest((values[i] - bias) * inverse);
    c = c > 0.0f ? c : 0.0f;  // np.clip(c, 0, 255); finite inputs never give NaN
    c = c < 255.0f ? c : 255.0f;
    codes[i] = static_cast<std::uint8_t>(c);
  }
  *scale_out = stored_scale;
  *bias_out = stored_bias;
}

// ---------------------------------------------------------------------------
// Conversion plan: every output tensor and where its bytes come from.

struct Step {
  bool quantize = false;
  std::string name;  // output weight name
  std::uint64_t source_offset = 0, source_bytes = 0;
  std::int64_t rows = 0, columns = 0;
};

struct Plan {
  Json config;           // source config plus the quantization block
  Json tensors = Json::object();  // the output safetensors header
  std::vector<Step> steps;
  std::string header;    // padded output header
  std::uint64_t output_bytes = 0;
  std::uint64_t quantize_bytes = 0;
};

Result<std::int64_t> json_int(const Json& v, const std::string& what) {
  if (v.is_number_integer()) return v.get<std::int64_t>();
  return LSE_ERROR(kInvalidArgument, "expected an integer in ", what);
}

Result<Json> dflash2_config(const ModelPaths& source, bool require_convertible) {
  LSE_ASSIGN_OR(Json config, read_json(source.config));
  const Json* arch = member(config, "architectures");
  bool dflash2 = false;
  if (arch && arch->is_array()) {
    for (const auto& a : *arch) dflash2 |= a.is_string() && a.get<std::string>() == "DFlash2DraftModel";
  } else if (arch && arch->is_string()) {
    dflash2 = arch->get<std::string>().find("DFlash2DraftModel") != std::string::npos;
  }
  if (!dflash2) return LSE_ERROR(kInvalidArgument, "source config does not describe a DFlash2DraftModel");
  if (require_convertible &&
      (truthy(member(config, "quantization")) || truthy(member(config, "quantization_config"))))
    return LSE_ERROR(kInvalidArgument, "source must be an unquantized BF16 checkpoint");
  return config;
}

Result<std::pair<Json, std::uint64_t>> read_header(int fd, std::uint64_t file_size) {
  std::uint64_t header_size = 0;
  if (file_size < 8) return LSE_ERROR(kInvalidArgument, "truncated safetensors payload");
  LSE_RETURN_IF_ERROR(pread_exact(fd, &header_size, 8, 0));
  if (header_size > file_size - 8) return LSE_ERROR(kInvalidArgument, "invalid safetensors header size");
  std::string text(static_cast<std::size_t>(header_size), '\0');
  LSE_RETURN_IF_ERROR(pread_exact(fd, text.data(), text.size(), 8));
  Json header = Json::parse(text, nullptr, false);
  if (header.is_discarded() || !header.is_object())
    return LSE_ERROR(kInvalidArgument, "invalid safetensors header");
  return std::pair{std::move(header), 8 + header_size};
}

std::string output_name(const std::string& original) {
  if (original == "candidate_selector.predecessor_codebook" ||
      original == "candidate_selector.successor_codebook")
    return original + ".weight";
  return original;
}

Result<Plan> make_plan(const ModelPaths& source, int fd, std::uint64_t file_size) {
  Plan plan;
  LSE_ASSIGN_OR(plan.config, dflash2_config(source, true));
  LSE_ASSIGN_OR(auto parsed, read_header(fd, file_size));
  const auto& [header, data_start] = parsed;
  std::uint64_t offset = 0;
  const auto add = [&](const std::string& name, const Json& dtype, Json shape, std::uint64_t size) {
    Json entry = Json::object();
    entry["dtype"] = dtype;
    entry["shape"] = std::move(shape);
    entry["data_offsets"] = Json::array({offset, offset + size});
    plan.tensors[name] = std::move(entry);  // like dict assignment: last write wins in place
    offset += size;
  };
  for (const auto& [original, entry] : header.items()) {
    if (original == "__metadata__") continue;
    const Json* shape = member(entry, "shape");
    const Json* offsets = member(entry, "data_offsets");
    const Json* dtype = member(entry, "dtype");
    if (!shape || !shape->is_array() || !offsets || !offsets->is_array() || offsets->size() != 2 || !dtype)
      return LSE_ERROR(kInvalidArgument, "invalid tensor entry: ", original);
    LSE_ASSIGN_OR(const std::int64_t begin, json_int((*offsets)[0], original));
    LSE_ASSIGN_OR(const std::int64_t end, json_int((*offsets)[1], original));
    if (begin < 0 || end < begin || data_start + static_cast<std::uint64_t>(end) > file_size)
      return LSE_ERROR(kInvalidArgument, "invalid tensor offsets: ", original);
    const std::string name = output_name(original);
    Step step;
    step.name = name;
    step.source_offset = data_start + static_cast<std::uint64_t>(begin);
    step.source_bytes = static_cast<std::uint64_t>(end - begin);
    if (shape->size() != 2 || !name.ends_with(".weight")) {
      add(name, *dtype, *shape, step.source_bytes);
      plan.steps.push_back(std::move(step));
      continue;
    }
    LSE_ASSIGN_OR(step.rows, json_int((*shape)[0], original));
    LSE_ASSIGN_OR(step.columns, json_int((*shape)[1], original));
    if (!dtype->is_string() || dtype->get<std::string>() != "BF16" ||
        static_cast<std::int64_t>(step.source_bytes) != step.rows * step.columns * 2)
      return LSE_ERROR(kInvalidArgument, "expected a BF16 matrix: ", original);
    if (step.rows <= 0 || step.columns <= 0 || step.columns % kGroupSize)
      return LSE_ERROR(kInvalidArgument, "matrix shape is not group64 compatible: [",
                       std::to_string(step.rows), ", ", std::to_string(step.columns), "]");
    step.quantize = true;
    const auto rows = static_cast<std::uint64_t>(step.rows);
    const auto columns = static_cast<std::uint64_t>(step.columns);
    const std::string stem = name.substr(0, name.size() - 7);
    add(name, "U32", Json::array({step.rows, step.columns / 4}), rows * columns);
    const Json groups = Json::array({step.rows, step.columns / kGroupSize});
    add(stem + ".scales", "BF16", groups, rows * columns / kGroupSize * 2);
    add(stem + ".biases", "BF16", groups, rows * columns / kGroupSize * 2);
    plan.quantize_bytes += step.source_bytes;
    plan.steps.push_back(std::move(step));
  }
  plan.header = python_json(plan.tensors, false);
  plan.header.append((8 - plan.header.size() % 8) % 8, ' ');
  plan.output_bytes = 8 + plan.header.size() + offset;
  plan.config["quantization"] = Json{{"bits", 8}, {"group_size", kGroupSize}, {"mode", "affine"}};
  return plan;
}

std::string config_text(const Json& config) { return python_json(config, true) + "\n"; }

Status write_weights(const Plan& plan, int source, const fs::path& path, Progress& progress,
                     std::string* sha) {
  Fd fd(::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644));
  if (!fd.valid()) return LSE_ERROR(kIoError, "cannot create ", path.string(), ": ", errno_text());
  Writer out(std::move(fd));
  const std::uint64_t header_size = plan.header.size();
  LSE_RETURN_IF_ERROR(out.write(&header_size, 8));
  LSE_RETURN_IF_ERROR(out.write(plan.header.data(), plan.header.size()));
  std::vector<char> copy;
  std::vector<std::uint16_t> raw;
  std::vector<std::uint8_t> codes;
  std::vector<std::uint16_t> scales, biases;
  std::uint64_t done = 0;
  const auto phase = DFlash2ConvertProgress::Phase::kQuantize;
  progress.report(phase, 0, plan.quantize_bytes);
  for (const Step& step : plan.steps) {
    if (!step.quantize) {
      copy.resize(static_cast<std::size_t>(std::min<std::uint64_t>(step.source_bytes, kIoChunk)));
      for (std::uint64_t at = 0; at < step.source_bytes;) {
        const auto n = static_cast<std::size_t>(std::min<std::uint64_t>(copy.size(), step.source_bytes - at));
        LSE_RETURN_IF_ERROR(pread_exact(source, copy.data(), n, step.source_offset + at));
        LSE_RETURN_IF_ERROR(out.write(copy.data(), n));
        at += n;
      }
      continue;
    }
    const auto columns = static_cast<std::size_t>(step.columns);
    const auto groups = columns / kGroupSize;
    scales.assign(static_cast<std::size_t>(step.rows) * groups, 0);
    biases.assign(scales.size(), 0);
    for (std::int64_t first = 0; first < step.rows; first += kRowBlock) {
      const auto count = static_cast<std::size_t>(std::min(kRowBlock, step.rows - first));
      raw.resize(count * columns);
      codes.resize(count * columns);
      const auto block_offset = static_cast<std::uint64_t>(first) * columns * 2;
      LSE_RETURN_IF_ERROR(pread_exact(source, raw.data(), raw.size() * 2, step.source_offset + block_offset));
      const std::size_t base = static_cast<std::size_t>(first) * groups;
      for (std::size_t g = 0; g < count * groups; ++g)
        quantize_group(raw.data() + g * kGroupSize, codes.data() + g * kGroupSize,
                       scales.data() + base + g, biases.data() + base + g);
      // Little-endian u32 packing of four codes is the codes in byte order.
      LSE_RETURN_IF_ERROR(out.write(codes.data(), codes.size()));
      done += raw.size() * 2;
      progress.report(phase, done, plan.quantize_bytes, step.name);
    }
    LSE_RETURN_IF_ERROR(out.write(scales.data(), scales.size() * 2));
    LSE_RETURN_IF_ERROR(out.write(biases.data(), biases.size() * 2));
  }
  LSE_RETURN_IF_ERROR(out.finish());
  if (out.written() != plan.output_bytes)
    return LSE_ERROR(kInternal, "DFlash2 conversion wrote an unexpected size");
  *sha = out.sha256();
  return OkStatus();
}

struct TempDir {
  fs::path path;
  ~TempDir() {
    if (path.empty()) return;
    std::error_code ec;
    fs::remove_all(path, ec);
  }
};

// Converts into a fresh temporary directory and renames it to `destination`.
// kAlreadyExists means another writer created `destination` first.
Status convert_into(const ModelPaths& source, const fs::path& destination,
                    const DFlash2ConvertOptions& options, Progress& progress) {
  LSE_ASSIGN_OR(Fd fd, open_read(source.weights));
  struct stat st {};
  if (::fstat(fd.get(), &st) != 0) return LSE_ERROR(kIoError, "cannot stat ", source.weights);
  const auto file_size = static_cast<std::uint64_t>(st.st_size);
  LSE_ASSIGN_OR(const Plan plan, make_plan(source, fd.get(), file_size));

  std::error_code ec;
  const fs::path parent = destination.parent_path();
  fs::create_directories(parent, ec);
  if (ec) return LSE_ERROR(kIoError, "cannot create ", parent.string(), ": ", ec.message());
  struct statvfs vfs {};
  if (::statvfs(parent.c_str(), &vfs) == 0) {
    const auto free_bytes = static_cast<std::uint64_t>(vfs.f_bavail) * vfs.f_frsize;
    const std::uint64_t need = plan.output_bytes + (16u << 20);
    if (free_bytes < need)
      return LSE_ERROR(kIoError, "DFlash2 conversion needs ", std::to_string(need >> 20),
                       " MiB free in ", parent.string(), " and only ",
                       std::to_string(free_bytes >> 20), " MiB are available");
  }

  LSE_ASSIGN_OR(const std::string source_hash, hash_file(source.weights, progress));

  std::string pattern = (parent / (destination.filename().string() + ".XXXXXXXX")).string();
  if (::mkdtemp(pattern.data()) == nullptr)
    return LSE_ERROR(kIoError, "cannot create a temporary directory in ", parent.string(), ": ", errno_text());
  TempDir work{pattern};

  std::string output_sha;
  LSE_RETURN_IF_ERROR(write_weights(plan, fd.get(), work.path / "model.safetensors", progress, &output_sha));
  LSE_RETURN_IF_ERROR(write_file(work.path / "config.json", config_text(plan.config)));
  Json manifest = Json::object();
  manifest["source_repository"] = options.source_repository;
  manifest["source_revision"] = options.source_revision;
  manifest["source_sha256"] = source_hash;
  manifest["reference_implementation"] = kReferenceUrl;
  manifest["format"] = kFormat;
  manifest["conversion"] = kConversion;
  manifest["bytes"] = plan.output_bytes;
  manifest["sha256"] = output_sha;
  manifest["tensors"] = plan.tensors.size();
  LSE_RETURN_IF_ERROR(write_file(work.path / "source-repository.json", python_json(manifest, true) + "\n"));
  fsync_dir(work.path);
  if (::rename(work.path.c_str(), destination.c_str()) != 0) {
    if (errno == EEXIST || errno == ENOTEMPTY)
      return LSE_ERROR(kAlreadyExists, destination.string(), " already exists");
    return LSE_ERROR(kIoError, "cannot rename into ", destination.string(), ": ", errno_text());
  }
  work.path.clear();
  fsync_dir(parent);
  progress.report(DFlash2ConvertProgress::Phase::kFinish, plan.output_bytes, plan.output_bytes);
  return OkStatus();
}

// ---------------------------------------------------------------------------
// Cache.

struct Origin {
  std::string repository, revision;
  bool known = false;
};

Origin source_origin(const ModelPaths& source, const DFlash2PrepareOptions& options) {
  Origin o;
  const fs::path dir = fs::absolute(fs::path(source.weights)).lexically_normal().parent_path();
  const fs::path snapshots = dir.parent_path();
  const std::string repo_dir = snapshots.parent_path().filename().string();
  if (snapshots.filename() == "snapshots" && repo_dir.starts_with("models--")) {
    std::string id = repo_dir.substr(8);
    for (std::size_t at; (at = id.find("--")) != std::string::npos;) id.replace(at, 2, "/");
    o = {id, dir.filename().string(), true};
  } else if (auto origin = read_json(dir / "hf-origin.json"); origin.ok()) {
    const Json* repo = member(*origin, "repository");
    const Json* rev = member(*origin, "revision");
    if (repo && repo->is_string() && rev && rev->is_string())
      o = {repo->get<std::string>(), rev->get<std::string>(), true};
  }
  if (!o.known) o = {dir.filename().string(), "unknown", false};
  if (!options.source_repository.empty()) o.repository = options.source_repository;
  if (!options.source_revision.empty()) o.revision = options.source_revision;
  if (!options.source_repository.empty() && !options.source_revision.empty()) o.known = true;
  return o;
}

std::uint64_t fnv1a(std::string_view s) {
  std::uint64_t h = 1469598103934665603ull;
  for (unsigned char c : s) h = (h ^ c) * 1099511628211ull;
  return h;
}

std::string path_safe(std::string s) {
  for (char& c : s)
    if (c == '/' || c == '\\' || c == ':') c = '-';
  return s;
}

std::string stat_stamp(const struct stat& st) {
#if defined(__APPLE__)
  const auto mtime_ns = static_cast<long long>(st.st_mtimespec.tv_sec) * 1000000000LL + st.st_mtimespec.tv_nsec;
#else
  const auto mtime_ns = static_cast<long long>(st.st_mtim.tv_sec) * 1000000000LL + st.st_mtim.tv_nsec;
#endif
  return std::to_string(static_cast<unsigned long long>(st.st_dev)) + ":" +
         std::to_string(static_cast<unsigned long long>(st.st_ino)) + ":" +
         std::to_string(static_cast<long long>(st.st_size)) + ":" + std::to_string(mtime_ns);
}

fs::path stamp_path(const fs::path& destination) {
  return destination.parent_path() / (destination.filename().string() + ".source-stat");
}

// The source's sha256 without reading it when that is safe: a stamp left by
// an earlier run for this exact file, or the blob name in an HF cache.
std::optional<std::string> known_source_sha(const ModelPaths& source, const fs::path& destination) {
  struct stat st {};
  if (::stat(source.weights.c_str(), &st) != 0) return std::nullopt;
  if (auto text = read_text(stamp_path(destination)); text.ok()) {
    const auto space = text->find(' ');
    if (space != std::string::npos && text->substr(0, space) == stat_stamp(st)) {
      std::string sha = text->substr(space + 1);
      while (!sha.empty() && (sha.back() == '\n' || sha.back() == ' ')) sha.pop_back();
      if (sha.size() == 64) return sha;
    }
  }
  std::error_code ec;
  if (fs::is_symlink(source.weights, ec)) {
    const std::string blob = fs::read_symlink(source.weights, ec).filename().string();
    if (!ec && blob.size() == 64 &&
        std::all_of(blob.begin(), blob.end(), [](char c) { return std::isxdigit(static_cast<unsigned char>(c)) && !std::isupper(static_cast<unsigned char>(c)); }))
      return blob;
  }
  return std::nullopt;
}

void write_stamp(const ModelPaths& source, const fs::path& destination, const std::string& sha) {
  struct stat st {};
  if (::stat(source.weights.c_str(), &st) != 0) return;
  (void)write_file(stamp_path(destination), stat_stamp(st) + " " + sha + "\n");
}

// Whether `destination` holds the conversion of `source`. Hashes the source
// only when no stamp or blob name vouches for it.
bool cache_valid(const ModelPaths& source, const fs::path& destination, Progress& progress,
                 std::string* why) {
  std::error_code ec;
  if (!fs::is_directory(destination, ec)) { *why = "no cached conversion"; return false; }
  auto manifest = read_json(destination / "source-repository.json");
  if (!manifest.ok()) { *why = "cached conversion has no readable source-repository.json"; return false; }
  const Json* bytes = member(*manifest, "bytes");
  const Json* format = member(*manifest, "format");
  const Json* source_sha = member(*manifest, "source_sha256");
  const auto size = fs::file_size(destination / "model.safetensors", ec);
  if (ec || !bytes || !bytes->is_number_unsigned() || bytes->get<std::uint64_t>() != size ||
      !format || *format != kFormat || !source_sha || !source_sha->is_string()) {
    *why = "cached conversion is incomplete";
    return false;
  }
  auto config = dflash2_config(source, true);
  auto cached_config = read_text(destination / "config.json");
  if (!config.ok() || !cached_config.ok()) { *why = "cached config is unreadable"; return false; }
  Json expected = std::move(*config);
  expected["quantization"] = Json{{"bits", 8}, {"group_size", kGroupSize}, {"mode", "affine"}};
  if (*cached_config != config_text(expected)) { *why = "source config changed"; return false; }
  std::string sha;
  if (auto known = known_source_sha(source, destination)) {
    sha = *known;
  } else {
    std::fprintf(stderr, "lse: dflash2: checking %s against the cached conversion\n", source.weights.c_str());
    auto hashed = hash_file(source.weights, progress);
    if (!hashed.ok()) { *why = hashed.status().message(); return false; }
    sha = *hashed;
  }
  if (sha != source_sha->get<std::string>()) { *why = "source weights changed"; return false; }
  write_stamp(source, destination, sha);
  return true;
}

}  // namespace

std::string_view to_string(DFlash2CheckpointKind kind) noexcept {
  switch (kind) {
    case DFlash2CheckpointKind::kNotDFlash2: return "not-dflash2";
    case DFlash2CheckpointKind::kQuantized: return "quantized";
    case DFlash2CheckpointKind::kBF16Source: return "bf16-source";
    case DFlash2CheckpointKind::kUnsupported: return "unsupported";
  }
  return "unknown";
}

std::string_view to_string(DFlash2ConvertProgress::Phase phase) noexcept {
  switch (phase) {
    case DFlash2ConvertProgress::Phase::kHashSource: return "hash-source";
    case DFlash2ConvertProgress::Phase::kQuantize: return "quantize";
    case DFlash2ConvertProgress::Phase::kFinish: return "finish";
  }
  return "unknown";
}

Result<std::string> dflash2_python_json(std::string_view json_text, bool indent) {
  Json j = Json::parse(json_text, nullptr, false);
  if (j.is_discarded()) return LSE_ERROR(kInvalidArgument, "invalid JSON");
  return python_json(j, indent);
}

Result<DFlash2CheckpointKind> inspect_dflash2_checkpoint(const ModelPaths& paths) {
  auto config = dflash2_config(paths, false);
  if (!config.ok()) {
    if (config.status().code() == StatusCode::kInvalidArgument) return DFlash2CheckpointKind::kNotDFlash2;
    return config.status();
  }
  if (truthy(member(*config, "quantization")) || truthy(member(*config, "quantization_config")))
    return DFlash2CheckpointKind::kQuantized;
  if (!paths.weights.ends_with(".safetensors")) return DFlash2CheckpointKind::kUnsupported;
  LSE_ASSIGN_OR(Fd fd, open_read(paths.weights));
  struct stat st {};
  if (::fstat(fd.get(), &st) != 0) return LSE_ERROR(kIoError, "cannot stat ", paths.weights);
  auto plan = make_plan(paths, fd.get(), static_cast<std::uint64_t>(st.st_size));
  if (!plan.ok() || plan->quantize_bytes == 0) return DFlash2CheckpointKind::kUnsupported;
  return DFlash2CheckpointKind::kBF16Source;
}

Result<std::vector<DFlash2PlannedTensor>> dflash2_q8_layout(const ModelPaths& source) {
  LSE_ASSIGN_OR(Fd fd, open_read(source.weights));
  struct stat st {};
  if (::fstat(fd.get(), &st) != 0) return LSE_ERROR(kIoError, "cannot stat ", source.weights);
  LSE_ASSIGN_OR(const Plan plan, make_plan(source, fd.get(), static_cast<std::uint64_t>(st.st_size)));
  std::vector<DFlash2PlannedTensor> out;
  out.reserve(plan.tensors.size());
  for (const auto& [name, entry] : plan.tensors.items()) {
    DFlash2PlannedTensor t;
    t.name = name;
    t.dtype = entry.at("dtype").get<std::string>();
    for (const auto& d : entry.at("shape")) t.shape.push_back(d.get<std::int64_t>());
    out.push_back(std::move(t));
  }
  return out;
}

Status convert_dflash2_q8(const ModelPaths& source, const fs::path& destination,
                          const DFlash2ConvertOptions& options) {
  std::error_code ec;
  const fs::path dest = fs::absolute(destination).lexically_normal();
  const fs::path src = fs::absolute(fs::path(source.weights)).lexically_normal().parent_path();
  if (dest == src || fs::exists(dest, ec))
    return LSE_ERROR(kInvalidArgument, "destination must be a new directory separate from the source");
  Progress progress(options.progress);
  return convert_into(source, dest, options, progress);
}

Result<fs::path> dflash2_cache_path(const ModelPaths& source, const DFlash2PrepareOptions& options) {
  fs::path root = options.cache_dir;
  if (root.empty()) {
    if (const char* env = std::getenv("LSE_DFLASH2_CACHE_DIR"); env && *env) root = env;
  }
  const fs::path weights = fs::absolute(fs::path(source.weights)).lexically_normal();
  if (root.empty()) return weights.parent_path() / kCacheName;
  const Origin origin = source_origin(source, options);
  std::string key;
  if (origin.known) {
    key = path_safe(origin.repository) + "@" + path_safe(origin.revision);
  } else {
    char hex[17];
    std::snprintf(hex, sizeof hex, "%016llx", static_cast<unsigned long long>(fnv1a(weights.string())));
    key = path_safe(weights.parent_path().filename().string()) + "-" + hex;
  }
  return fs::absolute(root).lexically_normal() / (key + "-q8g64");
}

Result<ModelPaths> prepare_dflash2_checkpoint(const std::string& name_or_path,
                                              const DFlash2PrepareOptions& options) {
  LSE_ASSIGN_OR(const ModelPaths paths, resolve_model(name_or_path));
  if (const char* env = std::getenv("LSE_DFLASH2_AUTOCONVERT"); env && std::string_view(env) == "0")
    return paths;
  // Anything that is not clearly a BF16 DFlash2 source loads as before, and
  // the loader reports what is wrong with it.
  auto kind = inspect_dflash2_checkpoint(paths);
  if (!kind.ok() || *kind != DFlash2CheckpointKind::kBF16Source) return paths;

  LSE_ASSIGN_OR(const fs::path destination, dflash2_cache_path(paths, options));
  const ModelPaths converted{(destination / "model.safetensors").string(),
                             (destination / "config.json").string()};
  Progress progress(options.progress);
  std::string why;
  if (cache_valid(paths, destination, progress, &why)) {
    std::fprintf(stderr, "lse: dflash2: using cached Q8 conversion %s\n", destination.c_str());
    return converted;
  }
  std::error_code ec;
  if (fs::exists(destination, ec)) {
    std::fprintf(stderr, "lse: dflash2: discarding cached conversion (%s)\n", why.c_str());
    fs::remove_all(destination, ec);
    if (ec) return LSE_ERROR(kIoError, "cannot remove stale ", destination.string(), ": ", ec.message());
  }
  const Origin origin = source_origin(paths, options);
  std::fprintf(stderr, "lse: dflash2: converting BF16 %s (%s@%s) to Q8 at %s; this runs once\n",
               paths.weights.c_str(), origin.repository.c_str(), origin.revision.c_str(),
               destination.c_str());
  DFlash2ConvertOptions convert;
  convert.source_repository = origin.repository;
  convert.source_revision = origin.revision;
  convert.progress = options.progress;
  const Status status = convert_into(paths, destination, convert, progress);
  if (status.code() == StatusCode::kAlreadyExists) {
    // Another process finished first; use its result if it is sound.
    if (!cache_valid(paths, destination, progress, &why))
      return LSE_ERROR(kIoError, "concurrent DFlash2 conversion left an unusable cache: ", why);
    return converted;
  }
  LSE_RETURN_IF_ERROR(status);
  if (auto manifest = read_json(destination / "source-repository.json"); manifest.ok())
    if (const Json* sha = member(*manifest, "source_sha256"); sha && sha->is_string())
      write_stamp(paths, destination, sha->get<std::string>());
  std::fprintf(stderr, "lse: dflash2: conversion complete: %s\n", destination.c_str());
  return converted;
}

}  // namespace lse::model
