// HRX's implementation of the device-qualification seam.
//
// Three measurements, and every one of them runs a kernel this file did not
// write: the streaming and dispatch bodies come from probe_kernels.hpp on the
// ordinary authoring surface, and the matrix-core body is the same MatrixTile
// the linear kernel uses, so what is timed is the code the engine would
// actually run rather than a microbenchmark that resembles it.
//
// The only HIP text here is the entry-point wrapper, which is the one place it
// is allowed — the same arrangement as the device codec engine.
//
// What is NOT here is any number this device did not produce. A row the device
// lacks is recorded absent, a row whose lane layout has never been measured on
// hardware is recorded unverified and is not emitted, and an operand form with
// no row at all becomes a fallback path with the rate of the row that would
// actually execute it. A device without fp8 is still a pool member; it is a
// slower one for fp8 work, and saying so is the entire point.
#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "lse/backends/hrx/comgr_compiler.hpp"
#include "lse/backends/hrx/device_info.hpp"
#include "lse/backends/hrx/hip_emitter.hpp"
#include "lse/backends/hrx/hip_sources.hpp"
#include "lse/backends/hrx/hip_types.hpp"
#include "lse/backends/hrx/kernels/vec_mem.hpp"
#include "lse/backends/hrx/kernels/wmma.hpp"
#include "lse/core/hash.hpp"
#include "lse/graph/jit.hpp"
#include "lse/graph/kernel_args.hpp"
#include "lse/graph/kernel_env.hpp"
#include "lse/math.hpp"
#include "lse/probe/device_probe.hpp"
#include "lse/probe/probe_kernels.hpp"

namespace lse::backend::hrx_kernels {

namespace {

using namespace lse::graph;
namespace math = lse::math;
using Clock = std::chrono::steady_clock;

double ns_since(Clock::time_point t0) {
  return static_cast<double>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - t0)
          .count());
}

// The shape the matrix rows are rated at.
//
// Deliberately small enough that both operands stay resident in L2: the tile
// re-reads its A row and B column for every output tile, so a shape that spills
// to DRAM would measure the same roofline the streaming probe already reports
// and tell the cost model nothing new about the matrix core. K is long so the
// instruction loop, not the tile setup, dominates.
constexpr std::uint32_t kRateM = 256;
constexpr std::uint32_t kRateN = 256;
constexpr std::uint32_t kRateK = 1024;
constexpr int kRateReps = 8;

constexpr int kTileM = 16;
constexpr int kTileN = 16;
constexpr int kTileK = 16;

// 64 MB is far past this class of part's L2, so the streaming number is DRAM's.
constexpr std::size_t kStreamBytes = 64u << 20;
constexpr int kStreamReps = 4;
constexpr std::uint32_t kBlock = 256;
constexpr int kLaunchReps = 512;

std::string wrapper(std::string_view entry,
                    std::span<const std::string> params,
                    const std::string& body) {
  const HipEmitter emitter;
  std::string src(emitter.prelude());
  src += "extern \"C\" __global__ void ";
  src += entry;
  src += "(";
  for (std::size_t i = 0; i < params.size(); ++i) {
    src += i == 0 ? "\n    " : ",\n    ";
    src += params[i];
  }
  src += ") {\n";
  // The flat thread index under the name KernelBody::thread_id() records, which
  // is the same contract the graph emitter writes.
  src += "  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;\n";
  src += body;
  src += "\n}\n";
  return src;
}

std::string param(std::string_view type, std::string_view name, bool is_const) {
  std::string p;
  if (is_const) p += "const ";
  p += type;
  p += "* __restrict__ ";
  p += name;
  return p;
}

template <class T>
std::string_view spell(const kir::TypeTable& types) {
  return types.scalar(kir::scalar_of<T>::value);
}

std::string stream_source(std::string_view entry, std::uint32_t elems,
                          std::uint32_t threads, std::uint32_t load_bytes) {
  // Both tables outlive the body: KernelBody keeps references to them.
  const kir::TypeTable types = hip_types();
  const DialectSourceTable table = hip_sources();
  kir::KernelBody k(types, table);
  probe::StreamArgs<env::Emit> a;
  constexpr DType kIn[] = {DType::kF32};
  if (!env::bind(k, a, kIn, DType::kF32)) return {};
  env::Emit e{&k};
  probe::stream_read(e, a, elems, threads, load_bytes);
  const std::string body = k.str();
  if (body.empty()) return {};
  const std::string params[] = {
      param(spell<kir::f32>(types), "in0", true),
      param(spell<kir::f32>(types), "out", false)};
  return wrapper(entry, params, body);
}

std::string touch_source(std::string_view entry) {
  const kir::TypeTable types = hip_types();
  const DialectSourceTable table = hip_sources();
  kir::KernelBody k(types, table);
  probe::TouchArgs<env::Emit> a;
  if (!env::bind(k, a, {}, DType::kF32)) return {};
  env::Emit e{&k};
  probe::touch_one(e, a);
  const std::string body = k.str();
  if (body.empty()) return {};
  const std::string params[] = {param(spell<kir::f32>(types), "out", false)};
  return wrapper(entry, params, body);
}

template <class E, class X = kir::f32, class W = kir::f32>
struct RateArgs {
  env::In<X, E> x;
  env::In<W, E> w;
  env::Out<kir::f32, E> out;
};

// The one thing this adds to the shared tile: the accumulator goes straight to
// memory. There is no fused epilogue to run and no emitter store hook outside
// the graph, so the Out slot the binding contract already names is the target.
template <math::MatrixTarget G, math::MatrixElem A, math::MatrixElem T>
struct RateTile : MatrixTile<RateTile<G, A, T>, G, A, T, kTileM, kTileN, kTileK> {
  const env::Out<kir::f32, env::Emit>* out = nullptr;
  std::uint32_t cols = 0;

  void emit_element(env::Emit&, const kir::Val<kir::u32>& row,
                    const kir::Val<kir::u32>& col,
                    const kir::Val<kir::f32>& v) const {
    (*out)[row * cols + col] = v;
  }
};

// The generations whose lane layouts have been measured here. Same pack the
// linear kernel carries, for the same reason: an unmeasured layout is a silent
// wrong answer, and a probe that emitted one would be timing nonsense.
template <math::MatrixTarget... Live, class F>
[[nodiscard]] std::string for_live_target(math::MatrixTarget t, F&& fn) {
  std::string out;
  const auto one = [&]<math::MatrixTarget L>() {
    if (t == L) out = fn.template operator()<L>();
  };
  (one.template operator()<Live>(), ...);
  return out;
}

// The storage format whose operand form a row consumes, or none when this probe
// has no body that can feed it. `with_matrix_operand` is the table of arms, so
// this is the same set the linear kernel can emit and not a second list.
// Bytes per buffer element on each side. The activation and the weight do not
// have to be the same width — a bf16 checkpoint keeps an f32 activation beside
// it — so a buffer sized from one of them under-allocates the other.
struct OperandWidths {
  std::uint32_t x = 0;
  std::uint32_t w = 0;
};

OperandWidths operand_widths(DType storage) {
  return with_matrix_operand<OperandWidths>(
      storage, []<class X, class W, math::MatrixElem, math::MatrixElem>() {
        return OperandWidths{kir::pack_elem_bytes<X>(),
                             kir::pack_elem_bytes<W>()};
      });
}

std::optional<DType> storage_for_operand(math::MatrixElem operand) noexcept {
  switch (operand) {
    case math::MatrixElem::kF16: return DType::kF16;
    case math::MatrixElem::kBF16: return DType::kBF16;
    case math::MatrixElem::kI8: return DType::kI32;
    default: return std::nullopt;
  }
}

std::string rate_source(std::string_view entry, const DeviceInfo& info,
                        const math::MatrixCoreRow& row, DType storage) {
  const kir::TypeTable types = hip_types();
  const DialectSourceTable table = hip_sources();
  const std::uint32_t kb = static_cast<std::uint32_t>(
      kRateK / static_cast<std::uint32_t>(row.pack));

  const std::string body = for_live_target<math::MatrixTarget::kRdna3>(
      row.target, [&]<math::MatrixTarget G>() -> std::string {
        return with_matrix_operand<std::string>(
            storage,
            [&]<class X, class W, math::MatrixElem A, math::MatrixElem T>()
                -> std::string {
              if (A != row.acc || T != row.operand) return {};
              kir::KernelBody k(types, table);
              RateArgs<env::Emit, X, W> a;
              const DType ins[] = {env::elem_dtype<X>::value,
                                   env::elem_dtype<W>::value};
              if (!env::bind(k, a, ins, DType::kF32)) return {};
              env::Emit e{&k};
              RateTile<G, A, T> tile;
              tile.out = &a.out;
              tile.cols = kRateN;
              tile.run(e, a.x, a.w, kRateM, kRateN, kb,
                       device_load_bytes(&info));
              return k.str();
            });
      });
  if (body.empty()) return {};

  const std::string params[] = {
      param(with_matrix_operand<std::string_view>(
                storage,
                [&]<class X, class, math::MatrixElem, math::MatrixElem>() {
                  return spell<X>(types);
                }),
            "in0", true),
      param(with_matrix_operand<std::string_view>(
                storage,
                [&]<class, class W, math::MatrixElem, math::MatrixElem>() {
                  return spell<W>(types);
                }),
            "in1", true),
      param(spell<kir::f32>(types), "out", false)};
  return wrapper(entry, params, body);
}

class HrxDeviceProbe final : public probe::IDeviceProbe {
 public:
  explicit HrxDeviceProbe(IBackend& be)
      : be_(be), cache_(be, compiler_) {
    const DeviceInfo& info = be_.device_info();
    if (!compiler_.available()) {
      declined_ = "no JIT compiler in this build, so no probe kernel can be "
                  "compiled and every device-side rate stays unknown";
    } else if (device_extension<AmdDeviceInfo>(info) == nullptr) {
      declined_ = "device is not AMD; this probe records only the HRX table";
    } else {
      usable_ = true;
    }
  }

  std::string_view name() const noexcept override { return "hrx"; }
  std::string_view declined() const noexcept override { return declined_; }

  Status run(IBackend& be, probe::DeviceProfile& out) override {
    if (&be != &be_) {
      return LSE_ERROR(kInvalidArgument,
                       "this probe was built for a different backend");
    }
    fill_matrix_rows(out);
    if (!usable_) return OkStatus();
    // A measurement that fails leaves its field unknown; it never falls back to
    // a plausible number, and it does not stop the others from being taken.
    if (const Status s = measure_stream(out); !s.ok()) note(s);
    if (const Status s = measure_launch(out); !s.ok()) note(s);
    if (const Status s = measure_matrix(out); !s.ok()) note(s);
    fill_paths(out);
    return OkStatus();
  }

 private:
  void note(const Status& s) {
    if (!declined_.empty()) declined_ += "; ";
    declined_ += s.message();
  }

  Result<KernelHandle> compiled(std::string_view entry,
                                const std::string& source) {
    if (source.empty()) {
      return LSE_ERROR(kUnimplemented, "probe kernel '", std::string(entry),
                       "' declined to emit on this target");
    }
    graph::EmittedKernel emitted;
    emitted.source = source;
    emitted.entry_name = std::string(entry);
    // Member 0: this probe holds one backend, so its cache is the set that
    // backend is.
    return cache_.get_or_compile(0, hash_bytes(entry), emitted);
  }

  // --- roofline ----------------------------------------------------------
  Status measure_stream(probe::DeviceProfile& out) {
    const DeviceInfo& info = be_.device_info();
    const std::uint32_t load_bytes = device_load_bytes(&info);
    const std::uint32_t width =
        kir::pack_n(load_bytes, kir::pack_elem_bytes<kir::f32>());
    const std::uint32_t groups =
        (info.compute_units != 0 ? info.compute_units : 8u) * 8u;
    const std::uint32_t threads = groups * kBlock;
    const std::uint32_t span = threads * width;
    // A whole number of strides, so the last pack of every thread lands inside
    // the buffer and no load needs a tail guard.
    std::uint32_t elems =
        static_cast<std::uint32_t>(kStreamBytes / sizeof(float));
    elems = (elems / span) * span;
    if (elems == 0) {
      return LSE_ERROR(kInvalidArgument, "stream probe has no work to do");
    }

    LSE_ASSIGN_OR(const KernelHandle kernel,
                  compiled("lse_probe_stream",
                           stream_source("lse_probe_stream", elems, threads,
                                         load_bytes)));

    auto in = be_.allocate(static_cast<std::size_t>(elems) * sizeof(float),
                           MemoryClass::kDevice);
    if (!in.ok()) return in.status();
    DeviceBuffer src = in.release();
    auto sink = be_.allocate(static_cast<std::size_t>(threads) * sizeof(float),
                             MemoryClass::kDevice);
    if (!sink.ok()) {
      be_.deallocate(src);
      return sink.status();
    }
    DeviceBuffer dst = sink.release();

    // Written before it is read: an allocation nothing has touched is not
    // backed by anything, and streaming it would time the page fault path
    // rather than the memory.
    {
      std::vector<float> seed(elems, 1.0f);
      const Status s = be_.copy_h2d(seed.data(), src,
                                    static_cast<std::size_t>(elems) *
                                        sizeof(float), 0);
      if (!s.ok()) {
        be_.deallocate(src);
        be_.deallocate(dst);
        return s;
      }
    }

    LaunchDims dims;
    dims.workgroup_size[0] = kBlock;
    dims.workgroup_count[0] = groups;
    dims.subgroup_size = wavefront();
    const BufferRef refs[] = {{&src, 0, src.size_bytes},
                              {&dst, 0, dst.size_bytes}};
    DispatchArgs args;
    args.bindings = refs;

    Status status = be_.launch(kernel, dims, args);
    if (status.ok()) status = be_.synchronize();
    if (status.ok()) {
      const auto t0 = Clock::now();
      for (int r = 0; r < kStreamReps && status.ok(); ++r) {
        status = be_.launch(kernel, dims, args);
      }
      if (status.ok()) status = be_.synchronize();
      const double ns = ns_since(t0);
      if (status.ok() && ns > 0.0) {
        out.dram_bytes_per_s = probe::Measured::measured(
            static_cast<double>(elems) * sizeof(float) * kStreamReps * 1e9 / ns);
      }
    }
    be_.deallocate(src);
    be_.deallocate(dst);
    return status;
  }

  // --- dispatch cost -----------------------------------------------------
  Status measure_launch(probe::DeviceProfile& out) {
    LSE_ASSIGN_OR(const KernelHandle kernel,
                  compiled("lse_probe_touch", touch_source("lse_probe_touch")));
    auto sink = be_.allocate(sizeof(float), MemoryClass::kDevice);
    if (!sink.ok()) return sink.status();
    DeviceBuffer dst = sink.release();

    LaunchDims dims;
    dims.workgroup_size[0] = wavefront() != 0 ? wavefront() : 32u;
    dims.workgroup_count[0] = 1;
    dims.subgroup_size = wavefront();
    const BufferRef refs[] = {{&dst, 0, dst.size_bytes}};
    DispatchArgs args;
    args.bindings = refs;

    Status status = be_.launch(kernel, dims, args);
    if (status.ok()) status = be_.synchronize();
    if (status.ok()) {
      const auto t0 = Clock::now();
      for (int r = 0; r < kLaunchReps && status.ok(); ++r) {
        status = be_.launch(kernel, dims, args);
      }
      if (status.ok()) status = be_.synchronize();
      const double ns = ns_since(t0);
      if (status.ok() && ns > 0.0) {
        // One workgroup of one wave writing one float: what is left after
        // dividing is the submission path, which is what a placement decision
        // is charged for every group it moves.
        out.launch_overhead_ns =
            probe::Measured::measured(ns / static_cast<double>(kLaunchReps));
      }
    }
    be_.deallocate(dst);
    return status;
  }

  // --- matrix core, per row ----------------------------------------------
  void fill_matrix_rows(probe::DeviceProfile& out) {
    const DeviceInfo& info = be_.device_info();
    const std::optional<math::MatrixTarget> target = matrix_target(info);
    if (!target.has_value()) return;
    const std::uint32_t caps = device_matrix_caps(info);
    const DialectSourceTable table = hip_sources();

    for (const math::MatrixCoreRow& r : math::matrix_core_table()) {
      if (r.target != *target) continue;
      probe::MatrixRowRate rate;
      rate.key = std::string(r.key);
      rate.acc = r.acc;
      rate.operand = r.operand;
      rate.m = r.m;
      rate.n = r.n;
      rate.k_step = r.k_step;
      rate.relative = r.throughput;
      if (!math::has_cap(caps, r.cap)) {
        rate.support = probe::RowSupport::kAbsent;
        rate.flops = probe::Measured::unsupported();
      } else if (!r.emittable() || table.find(r.key).empty()) {
        rate.support = probe::RowSupport::kUnverified;
      } else {
        // The device has it and it is emittable; whether this probe has a body
        // that can feed it is decided when the measurement is attempted.
        rate.support = probe::RowSupport::kDeclared;
      }
      out.rows.push_back(std::move(rate));
    }
  }

  Status measure_matrix(probe::DeviceProfile& out) {
    const DeviceInfo& info = be_.device_info();
    Status first;
    for (probe::MatrixRowRate& rate : out.rows) {
      if (rate.support != probe::RowSupport::kDeclared) continue;
      if (rate.m != kTileM || rate.n != kTileN || rate.k_step != kTileK) {
        continue;  // no tile here for that shape; the row stays declared
      }
      const std::optional<DType> storage = storage_for_operand(rate.operand);
      if (!storage.has_value()) continue;
      const math::MatrixCoreRow* row = nullptr;
      for (const math::MatrixCoreRow& r : math::matrix_core_table()) {
        if (r.key == rate.key && r.acc == rate.acc &&
            r.operand == rate.operand && r.k_step == rate.k_step) {
          row = &r;
          break;
        }
      }
      if (row == nullptr) continue;

      const Status s = measure_one_row(info, *row, *storage, rate);
      if (!s.ok() && first.ok()) first = s;
    }
    return first;
  }

  Status measure_one_row(const DeviceInfo& info,
                         const math::MatrixCoreRow& row, DType storage,
                         probe::MatrixRowRate& rate) {
    const std::string entry = "lse_probe_mma_" + std::string(to_string(storage));
    LSE_ASSIGN_OR(const KernelHandle kernel,
                  compiled(entry, rate_source(entry, info, row, storage)));

    const std::size_t operand_elems =
        static_cast<std::size_t>(kRateK / static_cast<std::uint32_t>(row.pack));
    const OperandWidths widths = operand_widths(storage);
    if (widths.x == 0 || widths.w == 0) {
      return LSE_ERROR(kUnimplemented, "no operand widths for ",
                       std::string(to_string(storage)));
    }
    const std::size_t a_bytes = kRateM * operand_elems * widths.x;
    const std::size_t b_bytes = kRateN * operand_elems * widths.w;
    auto ba = be_.allocate(a_bytes, MemoryClass::kDevice);
    if (!ba.ok()) return ba.status();
    DeviceBuffer x = ba.release();
    auto bb = be_.allocate(b_bytes, MemoryClass::kDevice);
    if (!bb.ok()) {
      be_.deallocate(x);
      return bb.status();
    }
    DeviceBuffer w = bb.release();
    auto bo = be_.allocate(
        static_cast<std::size_t>(kRateM) * kRateN * sizeof(float),
        MemoryClass::kDevice);
    if (!bo.ok()) {
      be_.deallocate(x);
      be_.deallocate(w);
      return bo.status();
    }
    DeviceBuffer o = bo.release();

    // Defined operands: an uninitialized buffer is a timing measurement over
    // whatever the allocator last left there.
    std::vector<std::byte> seed(std::max(a_bytes, b_bytes), std::byte{0x11});
    Status seeded = be_.copy_h2d(seed.data(), x, a_bytes, 0);
    if (seeded.ok()) seeded = be_.copy_h2d(seed.data(), w, b_bytes, 0);
    if (!seeded.ok()) {
      be_.deallocate(x);
      be_.deallocate(w);
      be_.deallocate(o);
      return seeded;
    }

    const auto lanes = static_cast<std::uint32_t>(row.wave);
    const std::uint32_t tiles_n = (kRateN + kTileN - 1) / kTileN;
    const std::uint32_t tiles = ((kRateM + kTileM - 1) / kTileM) * tiles_n;
    const std::uint32_t cap =
        info.max_threads_per_workgroup >= 256 ? 256u : 64u;
    const std::uint32_t waves = cap / lanes;
    LaunchDims dims;
    dims.workgroup_size[0] = waves * lanes;
    dims.workgroup_count[0] = (tiles + waves - 1) / waves;
    dims.subgroup_size = lanes;

    const BufferRef refs[] = {{&x, 0, x.size_bytes},
                              {&w, 0, w.size_bytes},
                              {&o, 0, o.size_bytes}};
    DispatchArgs args;
    args.bindings = refs;

    Status status = be_.launch(kernel, dims, args);
    if (status.ok()) status = be_.synchronize();
    if (status.ok()) {
      const auto t0 = Clock::now();
      for (int r = 0; r < kRateReps && status.ok(); ++r) {
        status = be_.launch(kernel, dims, args);
      }
      if (status.ok()) status = be_.synchronize();
      const double ns = ns_since(t0);
      if (status.ok() && ns > 0.0) {
        const double flops = 2.0 * kRateM * kRateN * kRateK * kRateReps;
        rate.flops = probe::Measured::measured(flops * 1e9 / ns);
        rate.support = probe::RowSupport::kMeasured;
      }
    }
    be_.deallocate(x);
    be_.deallocate(w);
    be_.deallocate(o);
    return status;
  }

  // --- per-operand answer ------------------------------------------------
  void fill_paths(probe::DeviceProfile& out) {
    // The row that would execute an operand this device has no form for. The
    // engine does not convert the tensor — the checkpoint's format is what
    // moves and what is stored — so the fallback is a kernel that dequantizes
    // at the register boundary and multiplies in whatever this device does
    // have. That is the fastest measured row, and the f32 accumulator is
    // preferred because it is the one an activation's range survives.
    const probe::MatrixRowRate* fallback = nullptr;
    for (const probe::MatrixRowRate& r : out.rows) {
      if (r.support != probe::RowSupport::kMeasured || !r.flops.positive()) {
        continue;
      }
      if (r.acc != math::MatrixElem::kF32) continue;
      if (fallback == nullptr || r.flops.value > fallback->flops.value) {
        fallback = &r;
      }
    }

    for (math::MatrixElem operand : probe::profiled_operands()) {
      probe::ComputePath path;
      path.operand = operand;
      path.executed_as = operand;

      const probe::MatrixRowRate* native = nullptr;
      for (const probe::MatrixRowRate& r : out.rows) {
        // Any accumulator: an integer operand accumulates in i32 and that is
        // still its own native form, not a fallback.
        if (r.operand != operand) continue;
        if (r.support != probe::RowSupport::kMeasured || !r.flops.positive()) {
          continue;
        }
        if (native == nullptr || r.flops.value > native->flops.value) {
          native = &r;
        }
      }
      if (native != nullptr) {
        path.native = true;
        path.row_key = native->key;
        path.flops = native->flops;
      } else if (fallback != nullptr) {
        path.native = false;
        path.executed_as = fallback->operand;
        path.row_key = fallback->key;
        // Declared, not measured: this is the rate of the instruction that
        // would run, and nobody has timed the dequantization in front of it.
        // The distinction is the difference between a number and a guess.
        path.flops = probe::Measured::declared(fallback->flops.value);
      }
      out.paths.push_back(std::move(path));
    }
  }

  std::uint32_t wavefront() const noexcept {
    return be_.device_info().wavefront_size;
  }

  IBackend& be_;
  ComgrCompiler compiler_;
  graph::JitCache cache_;
  bool usable_ = false;
  std::string declined_;
};

std::unique_ptr<probe::IDeviceProbe> make_hrx_device_probe(IBackend& be) {
  return std::make_unique<HrxDeviceProbe>(be);
}

const probe::DeviceProbeRegistrar kRegistrar{"hrx", &make_hrx_device_probe};

}  // namespace

}  // namespace lse::backend::hrx_kernels
