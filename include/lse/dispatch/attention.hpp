#pragma once

#include <cstdint>

#include "lse/graph/kernel_primitive.hpp"

namespace lse::dispatch {

enum class AttentionPlan : std::uint8_t { kScalar = 0, kFlashWmma = 4 };

struct FlashDims {
  std::uint32_t bsz = 0, qh = 0, tq = 0, dh = 0, kvh = 0, ts = 0, dv = 0;
  std::uint32_t group = 0, stride = 0, window = 0;
  float scale = 0.0f;
  int mask = 0;
  bool valid = false;
};

[[nodiscard]] bool paged_attention_inputs_valid(const graph::KernelShapes&);
[[nodiscard]] AttentionPlan attention_plan(const graph::KernelShapes&);
[[nodiscard]] FlashDims flash_dimensions(const graph::KernelShapes&);
[[nodiscard]] bool flash_wmma_supported(const graph::KernelShapes&);
// A tree pass the flash split serves, by shape and device part (the kernel
// checks the rest when it emits).
[[nodiscard]] bool flash_split_scope(const graph::KernelShapes&);
[[nodiscard]] bool split_decode_scope(const Shape& query, std::int64_t offset,
                                       std::int64_t capacity);
[[nodiscard]] bool split_decode_supported(const graph::KernelShapes&);
[[nodiscard]] bool split_decode_merge_supported(const graph::KernelShapes&);
[[nodiscard]] bool split_short_scope(const Shape& query, std::int64_t offset,
                                    std::int64_t capacity);
[[nodiscard]] bool split_short_default_supported(const graph::KernelShapes&);
[[nodiscard]] bool split_short_supported(const graph::KernelShapes&);
[[nodiscard]] bool split_short_merge_supported(const graph::KernelShapes&);
[[nodiscard]] bool reduction_row_supported(const graph::KernelShapes&);
[[nodiscard]] bool cooperative_rms_supported(const graph::KernelShapes&);
[[nodiscard]] bool phase_cooperative_rms_supported(const graph::KernelShapes&);
[[nodiscard]] bool wave32_l2_supported(const graph::KernelShapes&);
[[nodiscard]] std::uint32_t scalar_threads(const graph::KernelShapes&);

}  // namespace lse::dispatch
