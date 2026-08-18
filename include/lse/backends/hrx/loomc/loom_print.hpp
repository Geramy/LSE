// Kernel IR -> Loom source text.
//
// The second printer, beside the C-family one in src/ir/lower.cpp. It is not a
// variant of that walk: Loom is SSA, so every op is a statement that names its
// result and there is no expression to inline at a use. What the two share is
// what is genuinely shared — the region walk order, the op set, and the value
// naming, which lives in ir::ssa_name so a store hook's operand text and the
// statement that defined it cannot drift.
//
// The printer emits the INSIDE of a `kernel.def`'s launch region. The kernel
// header, the `launch(...)` parameter list and the `buffer.view` for each
// binding belong to the emitter, because only the emitter knows a buffer's
// extent — the IR types a kernel parameter as "some elements of f32" and Loom
// needs the number.
//
// Anything the printer cannot spell makes it FAIL, with the op named. Nothing
// is approximated: a body that reaches Loom is one Loom can express, and one
// that cannot declines so the group falls back to the HIP toolchain visibly.
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>

#include "lse/core/status.hpp"
#include "lse/graph/kernel_ir.hpp"

namespace lse::backend {

// What the emitter knows about a buffer parameter and the IR does not.
struct LoomBufferView {
  graph::kir::Scalar elem = graph::kir::Scalar::kF32;
  // Real element count. Loom views are bounded and the bound reaches the ISA,
  // so a sentinel here is a wrong kernel rather than a loose one.
  std::uint64_t elements = 0;
  // SSA name of the `buffer.view` the emitter already emitted, with the `%`.
  std::string view;
};

struct LoomPrintOptions {
  // Keyed by the IR symbol name the recorder interned for the buffer — `in0`,
  // `b3`, `out`. Every global symbol the body names must be present.
  std::unordered_map<std::string, LoomBufferView> buffers;
  // The symbol the recorder binds as the flat thread id. The EMITTER defines
  // it — it has to, because the launch guard is written against the same
  // value — so the printer only records its type. Every other scalar symbol is
  // a name this dialect cannot produce, and declines.
  std::string thread_id = "i";
  // Indentation of the outermost statement, in two-space levels.
  int indent = 1;
  // Prefix for the names the printer itself mints.
  std::string name_prefix = "p";
  // Prefix applied to EVERY value name. Value ids restart at zero in each
  // recorded body, so two bodies spliced into one kernel would both mint
  // `%v3`. Must stay empty wherever a store hook is installed: a hook's
  // operand text comes from ir::ssa_name, which knows nothing of this.
  std::string value_prefix;
};

struct LoomBody {
  std::string text;
  // Set when the body's last top-level op is `return <value>` — the
  // per-element form, whose value the caller splices where HIP would call a
  // `__device__` helper.
  std::string result;
  std::string result_type;
};

[[nodiscard]] Result<LoomBody> loom_print(const graph::kir::Body& body,
                                          const LoomPrintOptions& opts);

}  // namespace lse::backend
