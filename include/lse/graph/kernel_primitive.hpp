// Primitives that compute an output element with a whole loop of their own.
//
// An elementwise primitive contributes one expression to a fused body. A kernel
// primitive needs a loop — matmul, attention, a chunk scan — so it supplies the
// *body of a device function* returning output element `i`, and the emitter
// wraps it in the entry point. The author never writes the signature, which is
// what lets the emitter append epilogue inputs and fuse trailing elementwise
// work into the same kernel instead of launching it separately.
//
//   struct MyGemm final : KernelPrimitive<MyGemm> {
//     static constexpr std::string_view kName = "my.gemm";
//     static constexpr std::string_view kEntry = "my_gemm";
//     // In scope: `i` (flat output index), `in0`..`inN` (input pointers, in
//     // storage dtype). Must return the value for element i.
//     static constexpr std::string_view kSource = "  return in0[i] * in1[i];";
//     static ThreadPlan plan_impl(const KernelShapes& s) { ... }
//   };
//
// Shapes are part of the JIT cache key, so a primitive is free to bake extents
// in as literals via emit_kernel() rather than reading them from a constant.
#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <string_view>

#include "lse/backend/backend.hpp"
#include "lse/core/dtype.hpp"
#include "lse/core/shape.hpp"
#include "lse/core/status.hpp"
#include "lse/graph/kernel_ir.hpp"
#include "lse/graph/primitive.hpp"
#include "lse/opt/traffic.hpp"

namespace lse::graph {

// An activation panel in workgroup scratch: `count` f32 elements holding row
// `workgroup_id_y` of one operand, spelled `name` in the body being built.
//
// The row offset lives in the panel, not in the subscript — element j of the
// panel is element `workgroup_id_y * count + j` of the operand. Reading it with
// a row term added back is the width-invariance defect: every row of the pass
// would then read some other row's activation.
struct StagedPanel {
  std::string_view name;
  std::uint32_t count = 0;
};

// The int8 form of a StagedPanel, hoisted once for a run whose stages all
// quantize the same row under the same spec. Three siblings over one row
// otherwise amax-reduce, round and pack it three times into three private
// tiles, which is both the work and the scratch tripled.
//
// `count`, `group_size` and `bits` are the spec it was staged under, not a
// request: a stage whose own spec disagrees must stage for itself rather than
// read codes quantized to a granularity it does not expect.
struct StagedQuantPanel {
  std::string_view codes;
  std::string_view scale;
  std::string_view sum;
  std::uint32_t count = 0;
  std::int32_t group_size = 0;
  std::int32_t bits = 0;

  [[nodiscard]] bool matches(std::uint32_t k, std::int32_t gs,
                             std::int32_t b) const noexcept {
    return !codes.empty() && !scale.empty() && !sum.empty() && count == k &&
           group_size == gs && bits == b;
  }
};

// What a kernel primitive is told about the invocation it must cover.
struct KernelShapes {
  std::span<const Shape> inputs;
  Shape output;
  // Storage dtype of each input buffer, parallel to `inputs`, and of the
  // output. `env::bind` refuses a slot whose declared element type disagrees,
  // which is the only thing standing between a stale `env::In<kir::f32>` over
  // a bf16 buffer and a kernel that reinterprets memory without a diagnostic.
  // Empty means the caller did not describe them and no check is possible.
  std::span<const DType> input_dtypes;
  DType output_dtype = DType::kF32;
  std::array<float, 4> attrs{};
  std::array<std::int32_t, 4> iattrs{};
  const backend::DeviceInfo* device = nullptr;

  // The emitting backend's spellings. A primitive written against kir uses
  // these instead of naming a target's types or intrinsics itself.
  kir::TypeTable types{};
  const DialectSourceTable* intrinsics = nullptr;

  // How a self-indexing primitive must write a result. The emitter expands this
  // into any fused epilogue followed by the store, so trailing elementwise work
  // runs on the value in register rather than in a second launch. A primitive
  // that writes the output buffer itself cannot be given an epilogue.
  std::function<std::string(std::string_view index, std::string_view value)>
      store;

  // The panel the emitter has already staged for this body, matching what
  // `staged_row()` asked for. Empty name means it staged nothing and the
  // primitive owns the question.
  //
  // A named panel is a completed fill: the declaration, the fill loop and the
  // barrier behind it are all already in the body, under a workgroup-uniform
  // guard that dominates everything spliced after it. Allocate nothing, fill
  // nothing, barrier nothing — just read it.
  StagedPanel staged{};

  // The quantized form of `staged`, when the run hoisted one. Same contract:
  // a named set is a completed fill, already behind its barrier.
  StagedQuantPanel staged_quant{};
};

struct ThreadPlan {
  std::uint32_t workgroup_size[3] = {256, 1, 1};
  std::uint32_t workgroup_count[3] = {1, 1, 1};
  std::uint32_t lds_bytes = 0;
};

class KernelPrimitiveBase : public Primitive {
 public:
  FusionClass fusion_class() const noexcept override { return FusionClass::kBarrier; }

  // Body of the device function the emitter wraps. See the file comment for
  // what is in scope.
  virtual std::string emit_kernel(const KernelShapes& shapes) const = 0;
  virtual std::string_view entry_name() const noexcept = 0;
  virtual ThreadPlan plan(const KernelShapes& shapes) const = 0;

  // Whether trailing elementwise work may be fused into this kernel's launch.
  // True by default: the emitter owns the entry point, so it appends the
  // epilogue after the call rather than relying on the body to run it.
  virtual bool supports_epilogue() const noexcept { return true; }

  // Whether the primitive maps threads to work itself instead of computing one
  // output element per thread.
  //
  // A wave-cooperative instruction — WMMA on RDNA, MFMA on CDNA — has all lanes
  // of a wave build one output fragment together, so "return the value for
  // element i" cannot express it. A primitive that owns its indexing instead
  // emits the whole body and supplies its own ThreadPlan; the emitter still
  // writes the signature and binds the buffers.
  //
  // It does not write the output buffer itself — it hands each result to
  // `store`, which is what keeps an epilogue possible. Its operands are bound
  // first, so they are always `in0..inN` in the primitive's own order however
  // many extra inputs the epilogue brings.
  //
  // In scope for such a body: `i` (the flat thread id) and `k.count`.
  virtual bool owns_indexing() const noexcept { return false; }

  // The activation panel this primitive puts in workgroup scratch when it owns
  // the launch: `count` f32 elements of operand `input`, holding row
  // `workgroup_id_y` of it, for a launch covering `rows` rows. `count == 0`
  // means it stages nothing.
  //
  // Independent siblings sharing one launch also share that row, so the emitter
  // stages it once and hands every one of them the same panel. That makes this
  // a contract rather than a hint. Answering with a panel says: the bytes the
  // body reads from `input` are exactly row `workgroup_id_y` of it and nothing
  // else, and KernelShapes::staged is accepted in place of staging it here.
  //
  // The emitter also sums these to price the run's scratch, so a primitive that
  // understates its panel understates the launch's occupancy cost.
  struct StagedRow {
    std::size_t input = 0;
    std::uint32_t count = 0;
    std::uint32_t rows = 0;
  };
  virtual StagedRow staged_row(const KernelShapes&) const { return {}; }

  // What ONE WORKGROUP of this primitive's launch reads and writes, by operand
  // class. Unstated by default, and unstated is not zero: a primitive that has
  // not said what it moves must never read as moving nothing.
  //
  // It belongs here rather than in the emitter because the split into weight
  // and activation is a property of the contraction, not of the text: only the
  // primitive knows that one operand is re-read once per output tile and the
  // other once per weight tile, and that difference is the whole reason the two
  // are counted apart. The emitter sums these over the stages of a run.
  //
  // Per workgroup because that is the granularity a tile changes; the grid is a
  // consequence of the tile and is reported alongside rather than multiplied in.
  virtual opt::TrafficModel traffic(const KernelShapes&) const { return {}; }

  // The primitive that should actually run for this invocation.
  //
  // A device-specific form — a matrix-core GEMM standing in for the scalar
  // loop — cannot be chosen when the graph is built, because that is before a
  // device is known. The emitter asks here instead, once it has both. The
  // returned primitive must compute the same function; only the schedule may
  // differ, and it must outlive the emit call.
  virtual const KernelPrimitiveBase* specialize(const KernelShapes&) const {
    return this;
  }

  // A barrier contributes no expression to a fused loop.
  std::string emit_device(const EmitContext&) const override { return {}; }
};

template <typename Derived>
class KernelPrimitive : public KernelPrimitiveBase {
 public:
  std::string_view name() const noexcept override { return Derived::kName; }
  std::string_view entry_name() const noexcept override { return Derived::kEntry; }

  // Per-element kernels (slice, rope, …) override `kClass` so the generator
  // can fold them into a neighbour instead of launching them alone.
  FusionClass fusion_class() const noexcept override {
    if constexpr (requires { Derived::kClass; }) return Derived::kClass;
    return FusionClass::kBarrier;
  }

  std::string emit_kernel(const KernelShapes& shapes) const override {
    (void)shapes;
    return std::string(Derived::kSource);
  }

  ThreadPlan plan(const KernelShapes& shapes) const override {
    return Derived::plan_impl(shapes);
  }

  bool has_host_impl() const noexcept override { return false; }
  bool has_device_impl() const noexcept override { return true; }

  void eval_cpu(std::span<const float* const>, float*, std::size_t,
                const std::array<float, 4>&) const override {}
};

}  // namespace lse::graph
