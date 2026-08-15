// User-defined primitives that participate in fusion and JIT codegen.
//
// A primitive supplies two things the engine cannot infer: how to compute one
// element on the host (the oracle), and the device source expression for that
// same computation (spliced into a fused kernel). Because the source is an
// *expression* rather than a kernel, a custom primitive fuses with built-in ops
// into a single kernel instead of forcing a launch boundary.
//
// Minimal elementwise primitive:
//
//   struct Swish final : ElementwisePrimitive<Swish, 1> {
//     static constexpr std::string_view kName = "swish";
//     static float apply(float x) { return x / (1.0f + std::exp(-x)); }
//   };
//   LSE_REGISTER_PRIMITIVE(Swish);
//
// Then `custom("swish", {x})` records a node that fuses with everything around
// it. Device text comes from the emitting backend's source table, keyed by
// kName — not from this file.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "lse/core/dtype.hpp"
#include "lse/core/shape.hpp"
#include "lse/core/status.hpp"
#include "lse/graph/codegen.hpp"
#include "lse/graph/dialect_source.hpp"

namespace lse::graph {

// How the partitioner may fuse a node. Determined by the op or, for custom
// nodes, by the primitive — so a user primitive is not automatically a barrier.
enum class FusionClass : std::uint8_t {
  kLeaf,
  kElementwise,
  kReduction,
  kStructural,
  kBarrier,
  kCollective,
};

struct EmitContext {
  // Expressions naming each input's value in the generated scope.
  std::span<const std::string> inputs;
  // Variable the primitive must assign its result to.
  std::string_view out;
  const backend::DeviceInfo* device = nullptr;
  // Source language the caller wants back. A primitive that cannot produce it
  // must say so rather than emit source for another target.
  Dialect dialect = Dialect::kHip;
  // The emitting backend's spellings for primitives that do not carry their
  // own. Null when the caller is not emitting for a backend (host paths).
  const DialectSourceTable* sources = nullptr;
  // Per-node immediates, mirroring Node::attrs/iattrs.
  std::array<float, 4> attrs{};
  std::array<std::int32_t, 4> iattrs{};
};

class Primitive {
 public:
  virtual ~Primitive() = default;

  virtual std::string_view name() const noexcept = 0;
  virtual std::size_t arity() const noexcept = 0;
  virtual FusionClass fusion_class() const noexcept = 0;

  virtual Result<Shape> infer_shape(std::span<const Shape> inputs) const = 0;
  virtual DType infer_dtype(std::span<const DType> inputs) const = 0;

  // Statements in ctx.dialect assigning to ctx.out. Must be side-effect free
  // and depend only on ctx.inputs, so the emitter is free to inline it
  // anywhere in a fused loop body. Empty means "not in this dialect" — the
  // caller falls back rather than compiling source meant for another target.
  virtual std::string emit_device(const EmitContext& ctx) const = 0;

  // Dialects emit_device can produce. Device source is written by hand against
  // one target's intrinsics, so claiming more than one is an explicit act.
  virtual bool supports(Dialect d) const noexcept { return d == Dialect::kHip; }

  // Device-side declarations this primitive's body calls into (__device__
  // helpers, constants, structs). Emitted once per kernel ahead of the loop.
  virtual std::string_view device_preamble() const noexcept { return {}; }

  // Identity of that preamble. Primitives sharing an id emit it once, which is
  // what lets several primitives defined in one file share helper functions.
  virtual std::string_view preamble_id() const noexcept { return name(); }

  // Which targets this primitive can actually run on. A primitive may supply
  // only one: device-only source has no host body, and a host-only primitive
  // makes a GPU backend fall back to the CPU for that node rather than fail.
  virtual bool has_host_impl() const noexcept { return true; }
  virtual bool has_device_impl() const noexcept { return true; }

  // When >= 0 the output is the same allocation as inputs[i]. The kernel
  // writes a subset; the rest of the buffer stays as the input left it.
  virtual int inplace_input() const noexcept { return -1; }

  // Host reference. `inputs[i][e]` is element e of input i; write `count`
  // results into `out`. This is what the CPU interpreter runs and what the
  // JIT's generated kernel is diffed against.
  virtual void eval_cpu(std::span<const float* const> inputs, float* out,
                        std::size_t count,
                        const std::array<float, 4>& attrs) const = 0;
};

// Substitutes $0, $1, ... in a source template with the caller's expressions.
std::string substitute(std::string_view tmpl, std::span<const std::string> args);
std::string substitute(std::string_view tmpl, std::span<const std::string> args,
                       std::span<const float> attrs);

// CRTP base for the common case. Derived supplies kName and apply(). Device
// source comes from the emitting backend's table by default; a primitive
// written against one target's intrinsics may instead declare its own
// `static constexpr std::array<DialectExpr, N> kSources`.
template <typename Derived, std::size_t Arity>
class ElementwisePrimitive : public Primitive {
 public:
  std::string_view name() const noexcept override { return Derived::kName; }
  std::size_t arity() const noexcept override { return Arity; }
  FusionClass fusion_class() const noexcept override {
    return FusionClass::kElementwise;
  }

  Result<Shape> infer_shape(std::span<const Shape> inputs) const override {
    if (inputs.size() != Arity) {
      return LSE_ERROR(kInvalidArgument, std::string(Derived::kName),
                       " takes ", std::to_string(Arity), " inputs, got ",
                       std::to_string(inputs.size()));
    }
    Shape out = inputs[0];
    for (std::size_t i = 1; i < inputs.size(); ++i) {
      out = Shape::broadcast(out, inputs[i]);
      if (out.rank() == 0) {
        return LSE_ERROR(kInvalidArgument, std::string(Derived::kName),
                         ": input shapes do not broadcast");
      }
    }
    return out;
  }

  DType infer_dtype(std::span<const DType> inputs) const override {
    DType out = inputs.empty() ? DType::kF32 : inputs[0];
    for (DType d : inputs) {
      if (d == DType::kF32) out = DType::kF32;
    }
    return out;
  }

  std::string emit_device(const EmitContext& ctx) const override {
    const std::string_view expr = source_expr(ctx);
    if (expr.empty()) return {};
    std::string body(ctx.out);
    body += " = ";
    body += substitute(expr, ctx.inputs, ctx.attrs);
    body += ";";
    return body;
  }

  bool supports(Dialect d) const noexcept override {
    if constexpr (kSelfDescribed) {
      if (!expr_for(Derived::kSources, d).empty()) return true;
    }
    // Otherwise it is the backend's table that decides, and that is not
    // reachable from here — assume yes and let emit_device report the gap.
    return true;
  }

  void eval_cpu(std::span<const float* const> inputs, float* out,
                std::size_t count,
                const std::array<float, 4>& attrs) const override {
    (void)attrs;
    for (std::size_t i = 0; i < count; ++i) {
      out[i] = invoke(inputs, i, std::make_index_sequence<Arity>{});
    }
  }

 protected:
  static constexpr bool kSelfDescribed = requires { Derived::kSources; };

  // A primitive's own source wins over the backend table: it was written for
  // an intrinsic the generic spelling cannot express.
  [[nodiscard]] static std::string_view source_expr(
      const EmitContext& ctx) noexcept {
    if constexpr (kSelfDescribed) {
      const std::string_view own = expr_for(Derived::kSources, ctx.dialect);
      if (!own.empty()) return own;
    }
    return ctx.sources != nullptr ? ctx.sources->find(Derived::kName)
                                  : std::string_view{};
  }

 private:
  template <std::size_t... I>
  static float invoke(std::span<const float* const> inputs, std::size_t e,
                      std::index_sequence<I...>) {
    return Derived::apply(inputs[I][e]...);
  }
};

// Non-owning: `prim` must outlive the process (the usual case for a static
// instance registered by LSE_REGISTER_PRIMITIVE).
Status register_primitive(const Primitive* prim);

// Owning overload for dynamically created primitives. `keepalive` pins whatever
// owns `prim` — without it the registry would dangle when the owner dies.
Status register_primitive(const Primitive* prim, std::shared_ptr<void> keepalive);

Status unregister_primitive(std::string_view name);
const Primitive* find_primitive(std::string_view name);
std::vector<std::string> registered_primitives();

struct PrimitiveRegistrar {
  explicit PrimitiveRegistrar(const Primitive* p) {
    (void)register_primitive(p);
  }
};

#define LSE_REGISTER_PRIMITIVE(Type)                                  \
  namespace {                                                         \
  const Type _lse_prim_instance_##Type{};                             \
  const ::lse::graph::PrimitiveRegistrar _lse_prim_reg_##Type{        \
      &_lse_prim_instance_##Type};                                    \
  }  // namespace

}  // namespace lse::graph
