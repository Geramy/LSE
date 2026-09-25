#pragma once

namespace lse::kernels {

// Q6 lanes consume 16 consecutive floats. Rotate each 32-word block so the
// lanes do not concentrate those reads on the same LDS banks. Staging and
// contraction must use this identical permutation; it adds no padding.
// Every supported group-affine row contains whole 32-word blocks.
template <class Index>
auto q6_panel_index(const Index& index) {
  const auto block = index / 32u;
  return block * 32u + (index + block) % 32u;
}

}  // namespace lse::kernels
