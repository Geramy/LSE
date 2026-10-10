// Whether a decode or verify pass reads its context through the page
// summaries (graph::sparse_decode_attention) or exactly.
//
// The choice is a prediction from the device's measured roofline
// (dispatch::Roofline) and its measured sparse-decode kernel times
// (SparseDecodeCalibration): exact attention streams the live K/V; sparse
// streams the far pages' summaries, scores them, selects, and streams the
// budget's K/V. Sparse is taken only where it saves at least
// sparse_decode::kMinSavedShare of the exact time at the smallest context the
// KV rung holds, so short contexts stay exact and a slower-memory device
// switches earlier.
#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include "lse/backend/backend.hpp"

namespace lse::dispatch {

struct SparseDecodeShape {
  std::uint32_t query_heads = 0, key_heads = 0, head_dim = 0;
  // Query rows of the pass: 1 for decode, the chain or tree width for verify.
  std::uint32_t rows = 1;
  // Tokens the pass's block table covers (the KV rung).
  std::uint64_t capacity = 0;
  std::uint32_t budget = 0;
  // Bytes per K or V element.
  std::uint32_t element_bytes = 2;
};

// Measured on the device at load (see api: calibrate_sparse_decode): what the
// kernels achieve against the roofline's prediction.
struct SparseDecodeCalibration {
  // Achieved share of the streaming rate: exact split/flash attention over
  // live K/V, and the score kernel over summaries.
  double exact_efficiency = 0.0;
  // Exact attention's cost beyond its stream (launches, merge, tails).
  double exact_fixed_ns = 0.0;
  // Score kernel time per (page, key head, query row of the group).
  double score_ns_per_row_page = 0.0;
  double score_fixed_ns = 0.0;
  // Selection and descriptor, per key head workgroup set.
  double select_fixed_ns = 0.0;
  double select_ns_per_page = 0.0;
  [[nodiscard]] bool valid() const noexcept {
    return exact_efficiency > 0.0 && score_ns_per_row_page > 0.0 && select_fixed_ns > 0.0;
  }
};
void publish_sparse_decode_calibration(const backend::DeviceInfo& device,
                                       const SparseDecodeCalibration& calibration);
[[nodiscard]] std::optional<SparseDecodeCalibration> sparse_decode_calibration(
    const backend::DeviceInfo* device);

struct SparseDecodePlan {
  bool sparse = false;
  // Both predictions were made from measured numbers.
  bool priced = false;
  double exact_ns = 0.0, sparse_ns = 0.0;
  std::uint64_t live = 0;  // the context the prediction is made at
};

// Predicted per-layer attention time both ways at `live` tokens.
[[nodiscard]] SparseDecodePlan sparse_decode_predict(const backend::DeviceInfo* device,
                                                     const SparseDecodeShape& shape,
                                                     std::uint64_t live);
// The plan for a pass over this rung; `force` takes sparse whenever the
// context can exceed the budget.
[[nodiscard]] SparseDecodePlan sparse_decode_plan(const backend::DeviceInfo* device,
                                                  const SparseDecodeShape& shape, bool force);
// Smallest context at which `rows`-row passes take sparse on this device, or
// 0 when no context up to `limit` does (or nothing is measured).
[[nodiscard]] std::uint64_t sparse_decode_crossover(const backend::DeviceInfo* device,
                                                    SparseDecodeShape shape, std::uint64_t limit);

}  // namespace lse::dispatch
