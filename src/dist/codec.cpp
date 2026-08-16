#include "lse/dist/codec.hpp"

#include <array>
#include <chrono>
#include <cmath>
#include <map>
#include <mutex>
#include <string>
#include <type_traits>
#include <vector>

#include "lse/quant/block_codec.hpp"

namespace lse::dist {

namespace {

namespace env = graph::env;

// fp16 has an 11-bit significand, so a value rounded to it carries at most this
// much relative error. It is both the f16 wire's whole error and a term in
// every block scheme's, because the codes are formed against the f32 block
// scale but reconstructed against the fp16 one that went on the wire.
constexpr double kF16Eps = 1.0 / 2048.0;

// Derived from the scheme's own code range, not typed in: a new scheme brings
// its kMaxQ and this needs no edit.
template <class Scheme>
constexpr double scheme_tolerance() {
  if constexpr (std::is_same_v<Scheme, lse::f16>) {
    return kF16Eps;
  } else {
    return 1.0 / (2.0 * static_cast<double>(Scheme::kMaxQ)) + kF16Eps;
  }
}

// The one place a runtime dtype becomes a codec body for the host engine; the
// device engine does the same dispatch at emit time.
Status run_pack(DType format, const float* src, std::uint8_t* packed,
                std::uint32_t nblocks) {
  quant::PackArgs<env::Cpu> a{{src}, {packed}};
  for (std::uint32_t b = 0; b < nblocks; ++b) {
    env::Cpu e{b};
    if (!quant::dispatch_block_codec(
            format, [&]<class S> { quant::pack_blocks<S>(e, a, nblocks); })) {
      return LSE_ERROR(kUnimplemented, "no block codec for wire format ",
                       std::string(to_string(format)));
    }
  }
  return OkStatus();
}

Status run_unpack(DType format, const std::uint8_t* packed, float* dst,
                  std::uint32_t nblocks) {
  quant::UnpackArgs<env::Cpu> a{{packed}, {dst}};
  for (std::uint32_t b = 0; b < nblocks; ++b) {
    env::Cpu e{b};
    if (!quant::dispatch_block_codec(
            format, [&]<class S> { quant::unpack_blocks<S>(e, a, nblocks); })) {
      return LSE_ERROR(kUnimplemented, "no block codec for wire format ",
                       std::string(to_string(format)));
    }
  }
  return OkStatus();
}

// CommBuffer::offset is bytes, matching backend::DeviceBuffer.
template <typename T>
T* host_at(const CommBuffer& b) noexcept {
  auto* base = static_cast<std::byte*>(b.host);
  return reinterpret_cast<T*>(base + b.offset);
}

Result<std::uint32_t> block_count(DType format, std::size_t elems,
                                  std::size_t wire_bytes) {
  if (elems % kCodecBlockElems != 0) {
    return LSE_ERROR(kInvalidArgument, "codec element count ",
                     std::to_string(elems), " is not a multiple of ",
                     std::to_string(kCodecBlockElems));
  }
  const std::size_t need = codec_wire_bytes(format, elems);
  if (wire_bytes < need) {
    return LSE_ERROR(kOutOfRange, "wire buffer holds ",
                     std::to_string(wire_bytes), " B, format ",
                     std::string(to_string(format)), " needs ",
                     std::to_string(need));
  }
  return static_cast<std::uint32_t>(elems / kCodecBlockElems);
}

// Measured once per process, not typed in: the selector's compute term has to
// be this machine's, the same way the transport's bandwidth term is. Small
// enough (64 Ki values) to stay inside cache so it measures the codec rather
// than DRAM.
const std::array<std::uint64_t, static_cast<std::size_t>(DType::kCount)>&
host_codec_rates() {
  static const auto rates = [] {
    constexpr std::size_t kElems = 64u << 10;
    std::array<std::uint64_t, static_cast<std::size_t>(DType::kCount)> out{};
    std::vector<float> src(kElems);
    std::vector<float> dst(kElems);
    for (std::size_t i = 0; i < kElems; ++i) {
      src[i] = std::sin(static_cast<float>(i) * 0.017f) * 3.0f;
    }
    for (std::size_t i = 0; i < out.size(); ++i) {
      const auto format = static_cast<DType>(i);
      if (!quant::has_block_codec(format)) continue;
      const auto nblocks =
          static_cast<std::uint32_t>(kElems / kCodecBlockElems);
      std::vector<std::uint8_t> wire(codec_wire_bytes(format, kElems));
      if (!run_pack(format, src.data(), wire.data(), nblocks).ok()) continue;
      const auto t0 = std::chrono::steady_clock::now();
      constexpr int kReps = 4;
      for (int r = 0; r < kReps; ++r) {
        (void)run_pack(format, src.data(), wire.data(), nblocks);
        (void)run_unpack(format, wire.data(), dst.data(), nblocks);
      }
      const double ns = static_cast<double>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now() - t0)
              .count());
      if (ns <= 0.0) continue;
      // Quoted in payload bytes per second for one encode+decode pair, which is
      // what the selector's cost term consumes.
      const double bytes = static_cast<double>(kElems * sizeof(float)) * kReps;
      out[i] = static_cast<std::uint64_t>(bytes * 1e9 / ns);
    }
    return out;
  }();
  return rates;
}

class HostCodecEngine final : public CodecEngine {
 public:
  Status encode(DType format, const CommBuffer& src, const CommBuffer& wire,
                std::size_t elems) override {
    if (src.on_device() || wire.on_device()) {
      return LSE_ERROR(kInvalidArgument,
                       "host codec engine cannot reach device memory");
    }
    if (!supports(format)) {
      return LSE_ERROR(kUnimplemented, std::string(declined(format)));
    }
    LSE_ASSIGN_OR(const std::uint32_t nblocks,
                  block_count(format, elems, wire.bytes));
    return run_pack(format, host_at<const float>(src),
                    host_at<std::uint8_t>(wire), nblocks);
  }

  Status decode(DType format, const CommBuffer& wire, const CommBuffer& dst,
                std::size_t elems) override {
    if (dst.on_device() || wire.on_device()) {
      return LSE_ERROR(kInvalidArgument,
                       "host codec engine cannot reach device memory");
    }
    if (!supports(format)) {
      return LSE_ERROR(kUnimplemented, std::string(declined(format)));
    }
    LSE_ASSIGN_OR(const std::uint32_t nblocks,
                  block_count(format, elems, wire.bytes));
    return run_unpack(format, host_at<const std::uint8_t>(wire),
                      host_at<float>(dst), nblocks);
  }

  bool supports(DType format) const noexcept override {
    return quant::has_block_codec(format);
  }

  std::string_view declined(DType format) const noexcept override {
    if (supports(format)) return {};
    return "no quant::BlockCodec specialization for this wire format; adding "
           "one is that specialization plus its dialect rows, never an "
           "approximation of a format nothing can express";
  }

  std::string_view name() const noexcept override { return "host"; }

  std::uint64_t throughput_bytes_per_s(DType format) const noexcept override {
    const auto i = static_cast<std::size_t>(format);
    const auto& r = host_codec_rates();
    return i < r.size() ? r[i] : 0;
  }

  Status calibrate(DType, std::size_t) override {
    (void)host_codec_rates();  // measured once at first ask, then cached
    return OkStatus();
  }
};

struct EngineRegistry {
  std::mutex mu;
  std::map<std::string, CodecEngineFactory, std::less<>> factories;
};

EngineRegistry& engines() {
  static EngineRegistry r;
  return r;
}

CodecEngine& shared_host_engine() {
  static HostCodecEngine engine;
  return engine;
}

}  // namespace

double codec_block_tolerance(DType wire) noexcept {
  double out = 0.0;
  (void)quant::dispatch_block_codec(
      wire, [&]<class S> { out = scheme_tolerance<S>(); });
  return out;
}

std::unique_ptr<CodecEngine> make_host_codec_engine() {
  return std::make_unique<HostCodecEngine>();
}

void register_codec_engine(std::string_view backend_name,
                           CodecEngineFactory factory) {
  EngineRegistry& r = engines();
  std::lock_guard lock(r.mu);
  r.factories.emplace(std::string(backend_name), factory);
}

std::unique_ptr<CodecEngine> create_codec_engine(backend::IBackend& backend) {
  EngineRegistry& r = engines();
  CodecEngineFactory factory = nullptr;
  {
    std::lock_guard lock(r.mu);
    auto it = r.factories.find(backend.name());
    if (it != r.factories.end()) factory = it->second;
  }
  if (factory == nullptr) return make_host_codec_engine();
  std::unique_ptr<CodecEngine> engine = factory(backend);
  return engine != nullptr ? std::move(engine) : make_host_codec_engine();
}

Status host_encode(DType format, std::span<const float> src,
                   std::span<std::uint8_t> wire) {
  CommBuffer s;
  s.host = const_cast<float*>(src.data());
  s.bytes = src.size_bytes();
  CommBuffer w;
  w.host = wire.data();
  w.bytes = wire.size_bytes();
  w.dtype = DType::kU8;
  return shared_host_engine().encode(format, s, w, src.size());
}

Status host_decode(DType format, std::span<const std::uint8_t> wire,
                   std::span<float> dst) {
  CommBuffer w;
  w.host = const_cast<std::uint8_t*>(wire.data());
  w.bytes = wire.size_bytes();
  w.dtype = DType::kU8;
  CommBuffer d;
  d.host = dst.data();
  d.bytes = dst.size_bytes();
  return shared_host_engine().decode(format, w, d, dst.size());
}

}  // namespace lse::dist
