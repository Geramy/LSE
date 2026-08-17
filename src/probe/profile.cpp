#include "lse/probe/profile.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cstdio>

namespace lse::probe {

namespace {

// unknown < unsupported < declared < measured. "Weaker" is the one a combined
// number has to inherit.
int rank_of(Provenance p) noexcept {
  switch (p) {
    case Provenance::kUnknown: return 0;
    case Provenance::kUnsupported: return 1;
    case Provenance::kDeclared: return 2;
    case Provenance::kMeasured: return 3;
  }
  return 0;
}

// Four decades, so the fit has real leverage on both terms, and capped at a
// size the whole pool pass can afford to repeat in both directions.
constexpr std::array<std::size_t, 5> kTransferSizes{
    64, 4u << 10, 64u << 10, 1u << 20, 4u << 20};

constexpr std::array<math::MatrixElem, 6> kOperands{
    math::MatrixElem::kF32,  math::MatrixElem::kF16, math::MatrixElem::kBF16,
    math::MatrixElem::kI8,   math::MatrixElem::kI4,  math::MatrixElem::kFp8};

}  // namespace

Provenance weaker(Provenance a, Provenance b) noexcept {
  return rank_of(a) <= rank_of(b) ? a : b;
}

std::string DeviceId::str() const {
  return backend + ":" + std::to_string(ordinal);
}

Result<DeviceId> parse_device_id(std::string_view text) {
  const std::size_t colon = text.rfind(':');
  if (colon == std::string_view::npos || colon == 0 ||
      colon + 1 >= text.size()) {
    return LSE_ERROR(kInvalidArgument, "device id '", std::string(text),
                     "' is not <backend>:<ordinal>");
  }
  DeviceId id;
  id.backend = std::string(text.substr(0, colon));
  const std::string_view digits = text.substr(colon + 1);
  const auto [ptr, ec] =
      std::from_chars(digits.data(), digits.data() + digits.size(), id.ordinal);
  if (ec != std::errc{} || ptr != digits.data() + digits.size()) {
    return LSE_ERROR(kInvalidArgument, "device id '", std::string(text),
                     "' has no ordinal");
  }
  return id;
}

std::string_view to_string(math::MatrixElem e) noexcept {
  switch (e) {
    case math::MatrixElem::kF32: return "f32";
    case math::MatrixElem::kF16: return "f16";
    case math::MatrixElem::kBF16: return "bf16";
    case math::MatrixElem::kI32: return "i32";
    case math::MatrixElem::kI8: return "i8";
    case math::MatrixElem::kI4: return "i4";
    case math::MatrixElem::kFp8: return "fp8";
    case math::MatrixElem::kBf8: return "bf8";
  }
  return "unknown";
}

std::span<const std::size_t> default_transfer_sizes() noexcept {
  return kTransferSizes;
}

std::span<const math::MatrixElem> profiled_operands() noexcept {
  return kOperands;
}

math::MatrixElem operand_of_storage(DType storage) noexcept {
  switch (storage) {
    case DType::kF16: return math::MatrixElem::kF16;
    case DType::kBF16: return math::MatrixElem::kBF16;
    case DType::kI8: return math::MatrixElem::kI8;
    case DType::kQ4: return math::MatrixElem::kI4;
    // f32 has no operand form on any target in the table; the f16 one is the
    // single narrowing the engine permits, and it is the row an f32 weight has
    // always taken. Q8/Q6 dequantize to the accumulate type inside the kernel,
    // and so does a group-affine packed plane (kU32) — its codes reach the
    // accumulator as floats, never as a matrix-core operand.
    default: return math::MatrixElem::kF16;
  }
}

const ComputePath* DeviceProfile::path_for(
    math::MatrixElem operand) const noexcept {
  for (const ComputePath& p : paths) {
    if (p.operand == operand) return &p;
  }
  return nullptr;
}

const MatrixRowRate* DeviceProfile::row(std::string_view key) const noexcept {
  for (const MatrixRowRate& r : rows) {
    if (r.key == key) return &r;
  }
  return nullptr;
}

Measured DeviceProfile::stream_ns(std::size_t bytes) const noexcept {
  if (!dram_bytes_per_s.positive()) return Measured::unknown();
  return {static_cast<double>(bytes) * 1e9 / dram_bytes_per_s.value,
          dram_bytes_per_s.provenance};
}

Measured LinkProfile::cost_ns(std::size_t bytes) const noexcept {
  if (path == PathKind::kSameDevice) {
    return Measured::measured(0.0);
  }
  if (!latency_ns.known()) return Measured::unknown();
  if (bytes == 0) return latency_ns;
  if (!bandwidth_bytes_per_s.positive()) return Measured::unknown();
  const double ns = latency_ns.value +
                    static_cast<double>(bytes) * 1e9 /
                        bandwidth_bytes_per_s.value;
  return {ns, weaker(latency_ns.provenance, bandwidth_bytes_per_s.provenance)};
}

void fit_link(LinkProfile& link) {
  link.fit_error = 0.0;
  if (link.points.empty()) {
    link.latency_ns = Measured::unknown();
    link.bandwidth_bytes_per_s = Measured::unknown();
    return;
  }
  // Distinct sizes are what separate intercept from slope. One size can only
  // report a rate, and calling that rate a bandwidth would fold the latency
  // into it — which is the error this whole two-term model exists to avoid.
  std::size_t lo = link.points.front().bytes;
  std::size_t hi = lo;
  for (const TransferPoint& p : link.points) {
    lo = std::min(lo, p.bytes);
    hi = std::max(hi, p.bytes);
  }
  if (lo == hi) {
    link.latency_ns = Measured::unknown();
    link.bandwidth_bytes_per_s = Measured::unknown();
    return;
  }

  // Weighted by 1/ns^2, i.e. least *relative* error. The sizes span four
  // decades, so an unweighted fit is decided entirely by the largest point and
  // reports an intercept that has nothing to do with what a small transfer
  // costs — which is the term the placement decision turns on.
  double n = 0.0, sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0;
  for (const TransferPoint& p : link.points) {
    if (p.ns <= 0.0) continue;
    const double x = static_cast<double>(p.bytes);
    const double w = 1.0 / (p.ns * p.ns);
    n += w;
    sx += w * x;
    sy += w * p.ns;
    sxx += w * x * x;
    sxy += w * x * p.ns;
  }
  const double denom = n * sxx - sx * sx;
  if (denom <= 0.0) {
    link.latency_ns = Measured::unknown();
    link.bandwidth_bytes_per_s = Measured::unknown();
    return;
  }
  double slope = (n * sxy - sx * sy) / denom;
  double intercept = (sy - slope * sx) / n;

  // A negative term is not a link property, it is the fit overshooting on a
  // range where the model does not hold. Fall back to the two endpoints, which
  // reproduce them exactly and cannot go negative for a monotone link.
  if (slope <= 0.0 || intercept < 0.0) {
    double lo_ns = 0.0, hi_ns = 0.0;
    for (const TransferPoint& p : link.points) {
      if (p.bytes == lo) lo_ns = p.ns;
      if (p.bytes == hi) hi_ns = p.ns;
    }
    slope = (hi_ns - lo_ns) / static_cast<double>(hi - lo);
    intercept = lo_ns - slope * static_cast<double>(lo);
    if (slope <= 0.0) {
      link.latency_ns = Measured::unknown();
      link.bandwidth_bytes_per_s = Measured::unknown();
      return;
    }
    if (intercept < 0.0) intercept = 0.0;
  }

  link.latency_ns = Measured::measured(intercept);
  link.bandwidth_bytes_per_s = Measured::measured(1e9 / slope);

  for (const TransferPoint& p : link.points) {
    const double model = intercept + slope * static_cast<double>(p.bytes);
    if (p.ns > 0.0) {
      link.fit_error = std::max(link.fit_error, std::abs(model - p.ns) / p.ns);
    }
  }
}

std::size_t PoolProfile::index_of(const DeviceId& id) const noexcept {
  for (std::size_t i = 0; i < devices.size(); ++i) {
    if (devices[i].id == id) return i;
  }
  return devices.size();
}

const DeviceProfile* PoolProfile::device(const DeviceId& id) const noexcept {
  const std::size_t i = index_of(id);
  return i < devices.size() ? &devices[i] : nullptr;
}

const LinkProfile* PoolProfile::link(const DeviceId& src,
                                     const DeviceId& dst) const noexcept {
  const std::size_t a = index_of(src);
  const std::size_t b = index_of(dst);
  const std::size_t n = devices.size();
  if (a >= n || b >= n || links.size() != n * n) return nullptr;
  return &links[a * n + b];
}

bool PoolProfile::complete() const noexcept {
  if (devices.empty()) return false;
  if (links.size() != devices.size() * devices.size()) return false;
  for (const DeviceProfile& d : devices) {
    if (!d.dram_bytes_per_s.known()) return false;
  }
  for (const LinkProfile& l : links) {
    if (!l.cost_ns(1u << 20).known()) return false;
  }
  return true;
}

std::string PoolProfile::describe() const {
  std::string out = "pool " + fingerprint + "\n";
  char line[256];
  for (const DeviceProfile& d : devices) {
    std::snprintf(line, sizeof(line),
                  "  %-8s %-10s  dram %8.1f GB/s (%s)  launch %7.2f us (%s)"
                  "  free %7.1f GB (%s)\n",
                  d.id.str().c_str(), d.arch.c_str(),
                  d.dram_bytes_per_s.value / 1e9,
                  std::string(to_string(d.dram_bytes_per_s.provenance)).c_str(),
                  d.launch_overhead_ns.value / 1e3,
                  std::string(to_string(d.launch_overhead_ns.provenance)).c_str(),
                  d.free_memory.value / 1e9,
                  std::string(to_string(d.free_memory.provenance)).c_str());
    out += line;
    for (const ComputePath& p : d.paths) {
      std::snprintf(line, sizeof(line),
                    "      %-5s -> %-5s %-9s %8.2f TFLOP/s (%s)\n",
                    std::string(to_string(p.operand)).c_str(),
                    std::string(to_string(p.executed_as)).c_str(),
                    p.native ? "native" : "fallback", p.flops.value / 1e12,
                    std::string(to_string(p.flops.provenance)).c_str());
      out += line;
    }
  }
  for (const LinkProfile& l : links) {
    if (l.path == PathKind::kSameDevice) continue;
    std::snprintf(line, sizeof(line),
                  "  %-8s -> %-8s %-12s lat %8.2f us (%s)  bw %7.2f GB/s (%s)\n",
                  l.src.str().c_str(), l.dst.str().c_str(),
                  std::string(to_string(l.path)).c_str(),
                  l.latency_ns.value / 1e3,
                  std::string(to_string(l.latency_ns.provenance)).c_str(),
                  l.bandwidth_bytes_per_s.value / 1e9,
                  std::string(to_string(l.bandwidth_bytes_per_s.provenance)).c_str());
    out += line;
  }
  return out;
}

}  // namespace lse::probe
