#include "lse/ops/norm.hpp"

#include "lse/graph/ops.hpp"

namespace lse::ops {

Array rms_norm_zero_centered(const Array& x, const Array& weight, float eps) {
  return graph::rms_norm(x, weight, eps, /*zero_centered=*/true);
}

Array rms_norm(const Array& x, const Array& weight, float eps) {
  return graph::rms_norm(x, weight, eps, /*zero_centered=*/false);
}

}  // namespace lse::ops
