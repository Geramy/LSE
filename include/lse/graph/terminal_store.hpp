#pragma once

#include <string_view>

#include "lse/core/dtype.hpp"
#include "lse/core/status.hpp"
#include "lse/ir/body.hpp"

namespace lse::graph {

// Direct output stores are allowed only for a standalone, terminal kernel.
inline Status validate_terminal_stores(const ir::Body& body,
                                      std::string_view output_name, DType dtype) {
  ir::Scalar elem;
  switch (dtype) {
    case DType::kF32: elem = ir::Scalar::kF32; break;
    case DType::kF16: elem = ir::Scalar::kF16; break;
    case DType::kBF16: elem = ir::Scalar::kBF16; break;
    case DType::kI32: elem = ir::Scalar::kI32; break;
    case DType::kI8: elem = ir::Scalar::kI8; break;
    case DType::kU8: elem = ir::Scalar::kU8; break;
    case DType::kU32: elem = ir::Scalar::kU32; break;
    default: return LSE_ERROR(kUnimplemented, "terminal output is not a scalar storage type");
  }
  ir::ValueId output = ir::kNoValue;
  for (ir::ValueId id = 0; id < body.value_count(); ++id) {
    const auto& value = body.value(id);
    if (value.name != output_name || value.def == ir::kNoOp ||
        body.op(value.def).kind != ir::OpKind::kSymbol) continue;
    if (output != ir::kNoValue || value.type.space != ir::Space::kGlobal ||
        value.type.elem != elem)
      return LSE_ERROR(kInvalidArgument, "terminal output binding has the wrong storage type");
    output = id;
  }
  if (output == ir::kNoValue)
    return LSE_ERROR(kInvalidArgument, "terminal output binding is missing");
  Status status;
  bool stored = false;
  auto check = [&](ir::ValueId target, ir::ValueId rhs, bool vector) {
    const auto& buffer = body.value(target);
    if (buffer.type.space != ir::Space::kGlobal) return;
    const auto& value = body.value(rhs);
    if (target != output || buffer.type.elem != elem || value.type.elem != elem ||
        value.type.is_memory() || (!vector && value.type.lanes != 1)) {
      status = LSE_ERROR(kInvalidArgument, "terminal store targets another binding or storage type");
      return;
    }
    stored = true;
  };
  body.walk([&](ir::OpId id) {
    if (!status.ok()) return;
    const auto& op = body.op(id);
    if (op.kind == ir::OpKind::kRawStmt) {
      status = LSE_ERROR(kUnimplemented, "terminal stores require structured kernel IR");
    } else if (op.kind == ir::OpKind::kAssign && op.operands.size() == 2) {
      const auto& lhs = body.value(op.operands[0]);
      if (lhs.def == ir::kNoOp) return;
      const auto& at = body.op(lhs.def);
      if (at.kind == ir::OpKind::kSubscript && at.operands.size() == 2) {
        if (body.value(at.operands[0]).type.space == ir::Space::kGlobal &&
            (lhs.type.elem != elem || lhs.type.is_memory() || lhs.type.lanes != 1)) {
          status = LSE_ERROR(kInvalidArgument, "terminal lvalue has the wrong storage type");
          return;
        }
        check(at.operands[0], op.operands[1], false);
      } else if (lhs.type.space == ir::Space::kGlobal) {
        status = LSE_ERROR(kInvalidArgument, "terminal assignment must index its output");
      }
    } else if (op.kind == ir::OpKind::kStoreVec && op.operands.size() == 3) {
      if (body.value(op.operands[0]).type.space == ir::Space::kGlobal &&
          (op.imm <= 0 || static_cast<std::uint64_t>(op.imm) >
                              body.value(op.operands[2]).type.lanes)) {
        status = LSE_ERROR(kInvalidArgument, "terminal vector store has the wrong width");
        return;
      }
      check(op.operands[0], op.operands[2], true);
    }
  });
  if (!status.ok()) return status;
  if (!stored) return LSE_ERROR(kInvalidArgument, "terminal kernel never stored its output");
  return OkStatus();
}

}  // namespace lse::graph
