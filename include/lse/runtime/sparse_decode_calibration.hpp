// Measuring what the sparse decode kernels cost on this device, so the
// optimizer's choice (dispatch::sparse_decode_plan) predicts from measured
// kernel times rather than from the probe's compute rates.
//
// At load, with the model's attention geometry, the engine times on synthetic
// paged K/V: exact split attention at two contexts (its achieved share of the
// streaming rate), the score kernel at one and eight rows (its cost per page,
// key head and query row), and the selection at two contexts. The result is
// kept beside the device roofline, keyed by device, engine build and geometry.
#pragma once

#include <cstdint>
#include <string>

#include "lse/backend/backend.hpp"
#include "lse/core/status.hpp"
#include "lse/dispatch/sparse_decode.hpp"

namespace lse::runtime {

struct SparseDecodeGeometry {
  std::int32_t query_heads = 0, key_heads = 0, head_dim = 0;
  std::int32_t budget = 0;
};

// Reads `path` when it holds this geometry's numbers, measures and writes it
// otherwise, then publishes the calibration to dispatch for `device`.
Result<dispatch::SparseDecodeCalibration> calibrate_sparse_decode(
    const backend::DeviceInfo& device, const SparseDecodeGeometry& geometry,
    const std::string& path);

}  // namespace lse::runtime
