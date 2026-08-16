// Wire codecs: the compression seam a collective is parameterized by.
//
// The wire format IS a dtype — f16 uncompressed, or one of the block-scaled
// schemes in quant/traits.hpp — so there is no codec enum here. `DType` is the
// tag, `dtype_storage_bytes` is the wire length, and `quant::BlockCodec` is the
// one generic body every engine instantiates. Adding a format is a BlockCodec
// specialization plus a table row; nothing in this header changes.
//
// A CodecEngine is an implementation of that seam for one place the work can
// run. The host engine executes the codec under `env::Cpu` and is the
// reference; a backend registers a device engine that JITs the same body. An
// engine that cannot spell a format on the live device DECLINES — it never
// substitutes an approximation, exactly as the WMMA kernel declines on gfx12.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>

#include "lse/core/dtype.hpp"
#include "lse/core/status.hpp"
#include "lse/dist/transport.hpp"
#include "lse/quant/traits.hpp"

namespace lse::dist {

// Every wire format blocks at the same width as the on-disk quant schemes,
// which is also QuickReduce's. One source: quant::kBlockElems.
inline constexpr std::uint32_t kCodecBlockElems =
    static_cast<std::uint32_t>(quant::kBlockElems);

// Wire size of `elems` values in `wire`. Zero when `elems` is not a whole
// number of blocks — a codec never pads, because the wire length has to be
// derivable from the element count on both ends.
[[nodiscard]] inline std::size_t codec_wire_bytes(DType wire,
                                                  std::size_t elems) noexcept {
  if (elems % kCodecBlockElems != 0) return 0;
  return dtype_storage_bytes(wire, elems);
}

// Worst-case relative error of one encode/decode of a block, against the
// block's own magnitude. Provenance: step = absmax/kMaxQ and the worst rounding
// error is step/2, giving 1/(2*kMaxQ); the block carries its scale as fp16
// while the codes were formed against the f32 scale, which adds one 11-bit
// significand's worth on top. These are the tolerances the collective tests
// hold to. Zero for a dtype with no codec.
[[nodiscard]] double codec_block_tolerance(DType wire) noexcept;

class CodecEngine {
 public:
  virtual ~CodecEngine() = default;

  // `elems` must be a whole number of blocks. `wire` must be at least
  // codec_wire_bytes(format, elems).
  virtual Status encode(DType format, const CommBuffer& src,
                        const CommBuffer& wire, std::size_t elems) = 0;
  virtual Status decode(DType format, const CommBuffer& wire,
                        const CommBuffer& dst, std::size_t elems) = 0;

  [[nodiscard]] virtual bool supports(DType format) const noexcept = 0;
  // Empty when the format is supported. Names the missing capability, so a
  // decline is diagnosable rather than a silent fallback.
  [[nodiscard]] virtual std::string_view declined(
      DType format) const noexcept = 0;
  [[nodiscard]] virtual std::string_view name() const noexcept = 0;

  // Cost side of the selection model: payload bytes/s this engine sustains for
  // one encode+decode pair. Zero means "not measured", which the selector reads
  // as "do not trade bandwidth for compute".
  [[nodiscard]] virtual std::uint64_t throughput_bytes_per_s(
      DType format) const noexcept = 0;

  // Measures that rate on the live hardware. Separate from construction because
  // a device engine has to compile and warm a kernel first, which a constructor
  // should not do.
  virtual Status calibrate(DType format, std::size_t elems) = 0;
};

// Always available, needs no device, and is what the device engines are
// checked against byte for byte.
[[nodiscard]] std::unique_ptr<CodecEngine> make_host_codec_engine();

// Backends register their device engine by backend name ("hrx"), the same way
// they register themselves. Returns the host engine when the named backend has
// no device codec.
using CodecEngineFactory =
    std::unique_ptr<CodecEngine> (*)(backend::IBackend&);
void register_codec_engine(std::string_view backend_name,
                           CodecEngineFactory factory);
[[nodiscard]] std::unique_ptr<CodecEngine> create_codec_engine(
    backend::IBackend& backend);

struct CodecEngineRegistrar {
  CodecEngineRegistrar(std::string_view backend_name,
                       CodecEngineFactory factory) {
    register_codec_engine(backend_name, factory);
  }
};

// Host-side convenience over the same body, for callers that already hold plain
// arrays. Same bytes as the CommBuffer form.
Status host_encode(DType format, std::span<const float> src,
                   std::span<std::uint8_t> wire);
Status host_decode(DType format, std::span<const std::uint8_t> wire,
                   std::span<float> dst);

}  // namespace lse::dist
