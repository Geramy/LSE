// Normalization variants.
#pragma once

#include "lse/graph/graph.hpp"

namespace lse::ops {

using graph::Array;

// Zero-centered RMSNorm: scale = 1 + weight, so a weight initialized at 0
// starts as identity. lemonseed uses this everywhere; check before assuming it
// for another model.
Array rms_norm_zero_centered(const Array& x, const Array& weight, float eps);

// Plain RMSNorm: scale = weight.
Array rms_norm(const Array& x, const Array& weight, float eps);

}  // namespace lse::ops
