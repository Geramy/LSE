// HRX's implementation of the dist codec seam: quant::BlockCodec, JIT'd.
//
// The body comes from include/lse/quant/block_codec.hpp unchanged — this file
// contributes only the entry-point wrapper (the one place HIP text is allowed)
// and the compile/launch plumbing.
//
// A wire format is offered only when every dialect row its body needs exists in
// this backend's table. That is what makes the decline structural rather than a
// list of arch names: give fp8 a BlockCodec that asks for a packed f32->e4m3
// conversion and it will decline here until a target spells one, the same way
// wmma_linear declines on gfx12 rather than emitting a layout nothing has run.
#include <array>
#include <chrono>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "lse/backends/hrx/comgr_compiler.hpp"
#include "lse/backends/hrx/device_info.hpp"
#include "lse/backends/hrx/hip_emitter.hpp"
#include "lse/backends/hrx/hip_sources.hpp"
#include "lse/backends/hrx/hip_types.hpp"
#include "lse/dist/codec.hpp"
#include "lse/graph/kernel_args.hpp"
#include "lse/graph/kernel_env.hpp"
#include "lse/quant/block_codec.hpp"

namespace lse::backend::hrx_kernels {

namespace {

namespace env = graph::env;
namespace kir = graph::kir;
namespace quant = lse::quant;

bool table_can_spell(const graph::DialectSourceTable& table) {
  for (std::string_view sym : quant::kBlockCodecSymbols) {
    if (table.find(sym).empty()) return false;
  }
  return true;
}

std::string entry_name(DType format, bool pack) {
  return std::string("lse_block_") + (pack ? "pack_" : "unpack_") +
         std::string(to_string(format));
}

// The wrapper. Parameter names match env::bind's convention (`in0`, `out`) and
// the flat thread index is named `i`, which is what KernelBody::thread_id()
// records — the same contract the graph emitter writes.
std::string codec_source(DType format, bool pack, std::string_view entry) {
  const kir::TypeTable types = hip_types();
  const graph::DialectSourceTable table = hip_sources();
  if (!table_can_spell(table) || !quant::has_block_codec(format)) return {};

  kir::KernelBody k(types, table);
  // The launch's block count arrives in the push-constant block, so one
  // compiled kernel covers every payload length.
  const kir::Val<kir::u32> nblocks(&k.types(), "k.count");
  constexpr DType kPackIn[] = {DType::kF32};
  constexpr DType kUnpackIn[] = {DType::kU8};
  std::string body;
  // The runtime format picks which instantiation of the one generic body to
  // record. Nothing branches on it inside the kernel.
  if (pack) {
    quant::PackArgs<env::Emit> a;
    if (!env::bind(k, a, kPackIn, DType::kU8)) return {};
    env::Emit e{&k};
    if (!quant::dispatch_block_codec(
            format, [&]<class S> { quant::pack_blocks<S>(e, a, nblocks); })) {
      return {};
    }
    body = k.str();
  } else {
    quant::UnpackArgs<env::Emit> a;
    if (!env::bind(k, a, kUnpackIn, DType::kF32)) return {};
    env::Emit e{&k};
    if (!quant::dispatch_block_codec(
            format, [&]<class S> { quant::unpack_blocks<S>(e, a, nblocks); })) {
      return {};
    }
    body = k.str();
  }
  if (body.empty()) return {};

  const std::string_view f32 = types.scalar(kir::Scalar::kF32);
  const std::string_view u8 = types.scalar(kir::Scalar::kU8);
  const std::string_view in_ty = pack ? f32 : u8;
  const std::string_view out_ty = pack ? u8 : f32;

  const HipEmitter emitter;
  std::string src(emitter.prelude());
  src += "struct LseConstants { unsigned int count; };\n\n";
  src += "extern \"C\" __global__ void ";
  src += entry;
  src += "(\n    const ";
  src += in_ty;
  src += "* __restrict__ in0,\n    ";
  src += out_ty;
  src += "* __restrict__ out,\n    LseConstants k) {\n";
  src += "  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;\n";
  src += body;
  src += "\n}\n";
  return src;
}

class HrxCodecEngine final : public dist::CodecEngine {
 public:
  explicit HrxCodecEngine(IBackend& be) : be_(be) {
    const DeviceInfo& info = be_.device_info();
    usable_ = compiler_.available() &&
              device_extension<AmdDeviceInfo>(info) != nullptr &&
              table_can_spell(hip_sources());
  }

  Status encode(DType format, const dist::CommBuffer& src,
                const dist::CommBuffer& wire, std::size_t elems) override {
    return run(format, true, src, wire, elems);
  }

  Status decode(DType format, const dist::CommBuffer& wire,
                const dist::CommBuffer& dst, std::size_t elems) override {
    return run(format, false, wire, dst, elems);
  }

  bool supports(DType format) const noexcept override {
    return usable_ && quant::has_block_codec(format);
  }

  std::string_view declined(DType format) const noexcept override {
    if (supports(format)) return {};
    if (!compiler_.available()) return "no JIT compiler in this build";
    if (device_extension<AmdDeviceInfo>(be_.device_info()) == nullptr) {
      return "device is not AMD; this engine spells only the HRX table";
    }
    if (!table_can_spell(hip_sources())) {
      return "the HIP table is missing a spelling this codec needs";
    }
    return "no quant::BlockCodec specialization for this wire format, and no "
           "dialect row that would spell it on this target; declining rather "
           "than emitting an unverified conversion";
  }

  std::string_view name() const noexcept override { return "hrx"; }

  std::uint64_t throughput_bytes_per_s(DType format) const noexcept override {
    const auto i = static_cast<std::size_t>(format);
    return i < rate_.size() ? rate_[i] : 0;
  }

  // Times the kernels on device-resident buffers. Staging through the host is
  // deliberately excluded: what the selector compares against a link bandwidth
  // is the codec's own rate, and a staged measurement would be reporting the
  // copy engine instead.
  Status calibrate(DType format, std::size_t elems) override {
    if (!supports(format)) return OkStatus();
    if (elems % quant::kBlockElemsU != 0) {
      return LSE_ERROR(kInvalidArgument, "calibration size is not a whole "
                                         "number of codec blocks");
    }
    const auto nblocks =
        static_cast<std::uint32_t>(elems / quant::kBlockElemsU);
    const std::size_t payload = elems * sizeof(float);
    const std::size_t wire_bytes = dist::codec_wire_bytes(format, elems);

    auto a = be_.allocate(payload, MemoryClass::kDevice);
    if (!a.ok()) return a.status();
    DeviceBuffer src = a.release();
    auto b = be_.allocate(wire_bytes, MemoryClass::kDevice);
    if (!b.ok()) {
      be_.deallocate(src);
      return b.status();
    }
    DeviceBuffer wire = b.release();
    auto c = be_.allocate(payload, MemoryClass::kDevice);
    if (!c.ok()) {
      be_.deallocate(src);
      be_.deallocate(wire);
      return c.status();
    }
    DeviceBuffer dst = c.release();

    std::vector<float> seed(elems);
    for (std::size_t i = 0; i < elems; ++i) {
      seed[i] = static_cast<float>(i % 251) * 0.013f - 1.6f;
    }
    Status status = be_.copy_h2d(seed.data(), src, payload, 0);
    if (status.ok()) status = launch(format, true, src, wire, nblocks);
    if (status.ok()) status = launch(format, false, wire, dst, nblocks);
    if (status.ok()) status = be_.synchronize();  // warm: compile and first run

    if (status.ok()) {
      constexpr int kReps = 16;
      const auto t0 = std::chrono::steady_clock::now();
      for (int r = 0; r < kReps && status.ok(); ++r) {
        status = launch(format, true, src, wire, nblocks);
        if (status.ok()) status = launch(format, false, wire, dst, nblocks);
      }
      if (status.ok()) status = be_.synchronize();
      const double ns = static_cast<double>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now() - t0)
              .count());
      if (status.ok() && ns > 0.0) {
        const double bytes = static_cast<double>(payload) * kReps;
        rate_[static_cast<std::size_t>(format)] =
            static_cast<std::uint64_t>(bytes * 1e9 / ns);
      }
    }
    be_.deallocate(src);
    be_.deallocate(wire);
    be_.deallocate(dst);
    return status;
  }

 private:
  struct Owned {
    DeviceBuffer buf;
    bool owned = false;
  };

  Result<KernelHandle> kernel_for(DType format, bool pack) {
    const std::uint32_t key =
        static_cast<std::uint32_t>(format) * 2u + (pack ? 1u : 0u);
    if (auto it = kernels_.find(key); it != kernels_.end()) return it->second;
    const std::string entry = entry_name(format, pack);
    const std::string src = codec_source(format, pack, entry);
    if (src.empty()) {
      return LSE_ERROR(kUnimplemented, "wire format ",
                       std::string(to_string(format)),
                       " declined to emit on this target");
    }
    auto code = compiler_.compile(src, be_.device_info().arch);
    if (!code.ok()) return code.status();
    auto handle = be_.load_executable(entry, *code);
    if (!handle.ok()) return handle.status();
    kernels_.emplace(key, *handle);
    return *handle;
  }

  // Device-resident buffers are used in place; a host payload is staged, which
  // is what lets one engine serve a collective whose buffers have no residency
  // model yet.
  Result<Owned> resident(const dist::CommBuffer& b, bool upload) {
    if (b.on_device()) {
      Owned o;
      o.buf = *b.device;
      o.buf.offset += b.offset;
      return o;
    }
    auto alloc = be_.allocate(b.bytes, MemoryClass::kDevice);
    if (!alloc.ok()) return alloc.status();
    Owned o;
    o.buf = alloc.release();
    o.owned = true;
    if (upload) {
      Status s = be_.copy_h2d(static_cast<const std::byte*>(b.host) + b.offset,
                              o.buf, b.bytes, 0);
      if (!s.ok()) {
        be_.deallocate(o.buf);
        return s;
      }
    }
    return o;
  }

  Status run(DType format, bool pack, const dist::CommBuffer& in,
             const dist::CommBuffer& out, std::size_t elems) {
    if (!supports(format)) {
      return LSE_ERROR(kUnimplemented, "hrx codec engine declined ",
                       std::string(to_string(format)), ": ",
                       std::string(declined(format)));
    }
    if (elems % quant::kBlockElemsU != 0) {
      return LSE_ERROR(kInvalidArgument, "codec element count ",
                       std::to_string(elems), " is not a multiple of ",
                       std::to_string(quant::kBlockElemsU));
    }
    const auto nblocks =
        static_cast<std::uint32_t>(elems / quant::kBlockElemsU);
    const std::size_t wire_bytes = dist::codec_wire_bytes(format, elems);
    const std::size_t need = pack ? wire_bytes : elems * sizeof(float);
    if (out.bytes < need) {
      return LSE_ERROR(kOutOfRange, "codec destination holds ",
                       std::to_string(out.bytes), " B, needs ",
                       std::to_string(need));
    }

    LSE_ASSIGN_OR(Owned src, resident(in, true));
    auto dst = resident(out, false);
    if (!dst.ok()) {
      if (src.owned) be_.deallocate(src.buf);
      return dst.status();
    }
    Owned dest = dst.release();

    Status status = launch(format, pack, src.buf, dest.buf, nblocks);
    if (status.ok() && !out.on_device()) {
      status = be_.synchronize();
      if (status.ok()) {
        status = be_.copy_d2h(dest.buf,
                              static_cast<std::byte*>(out.host) + out.offset,
                              need, 0);
      }
    } else if (status.ok()) {
      status = be_.synchronize();
    }
    if (src.owned) be_.deallocate(src.buf);
    if (dest.owned) be_.deallocate(dest.buf);
    return status;
  }

  Status launch(DType format, bool pack, const DeviceBuffer& in,
                const DeviceBuffer& out, std::uint32_t nblocks) {
    LSE_ASSIGN_OR(const KernelHandle kernel, kernel_for(format, pack));
    const AmdDeviceInfo* amd =
        device_extension<AmdDeviceInfo>(be_.device_info());
    const std::uint32_t threads =
        be_.device_info().max_threads_per_workgroup >= 256 ? 256u : 64u;

    LaunchDims dims;
    dims.workgroup_size[0] = threads;
    dims.workgroup_count[0] = (nblocks + threads - 1) / threads;
    dims.subgroup_size = amd != nullptr ? amd->wavefront_size : 0;

    const BufferRef refs[] = {{&in, 0, in.size_bytes - in.offset},
                              {&out, 0, out.size_bytes - out.offset}};
    std::array<std::byte, sizeof(std::uint32_t)> constants{};
    std::memcpy(constants.data(), &nblocks, sizeof(nblocks));

    DispatchArgs args;
    args.bindings = refs;
    args.constants = constants;
    return be_.launch(kernel, dims, args);
  }

  IBackend& be_;
  ComgrCompiler compiler_;
  std::map<std::uint32_t, KernelHandle> kernels_;
  std::array<std::uint64_t, static_cast<std::size_t>(DType::kCount)> rate_{};
  bool usable_ = false;
};

std::unique_ptr<dist::CodecEngine> make_hrx_codec_engine(IBackend& be) {
  return std::make_unique<HrxCodecEngine>(be);
}

const dist::CodecEngineRegistrar kRegistrar{"hrx", &make_hrx_codec_engine};

}  // namespace

}  // namespace lse::backend::hrx_kernels
