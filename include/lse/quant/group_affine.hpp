// MLX "affine" group quantization: w = code * scale + bias, code unsigned.
//
// This is not a QuantScheme and cannot be made into one. That family is a
// single packed struct per 32 weights carrying one fp16 scale and no zero
// point, and the three sizeof() asserts in traits.hpp are the on-disk ABI of
// the collective wire path. Group-affine is three *separate* tensors — a
// packed bit stream, one scale per group, one bias per group — over a bit
// width and a group size that are properties of the tensor rather than of the
// format. There is no Block whose sizeof describes that and no single pointer
// QuantScheme::dequantize_row could take, so this is a sibling abstraction.
//
// Three properties the code cannot state:
//   * The scale is SIGNED and the bias is not the group minimum. MLX anchors
//     whichever end of the range has the larger magnitude and flips the sign
//     of the scale to do it, so roughly half of a checkpoint's scales are
//     negative. Assuming scale > 0, or bias == min, silently sign-flips
//     weights instead of failing.
//   * Codes are UNSIGNED, [0, 2^bits). QuantScheme's Q4/Q6 store a biased code
//     and subtract a constant; reusing that here shifts every weight by 8.
//   * The packed plane is a dense little-endian bit stream, not a nibble
//     array. It only degenerates to "8 nibbles per lane" because the target's
//     bit width happens to divide 32.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "lse/core/dtype.hpp"
#include "lse/core/status.hpp"

namespace lse::quant {

// Bit widths MLX will emit. 7 is absent upstream, so it is absent here.
inline constexpr int kGroupAffineBits[] = {2, 3, 4, 5, 6, 8};
inline constexpr int kGroupAffineGroupSizes[] = {32, 64, 128};

// Geometry of one quantized tensor. Held beside the plane, never inside a
// DType tag: a mixed-precision checkpoint gives two tensors in the same file
// different bit widths.
struct GroupAffine {
  int bits = 4;
  int group_size = 64;

  static Result<GroupAffine> make(int bits, int group_size);

  [[nodiscard]] constexpr std::uint32_t max_code() const noexcept {
    return (1u << bits) - 1u;
  }

  // The repeat unit of the bit stream: the smallest run of whole lanes that
  // holds a whole number of codes. Within one chunk every code's word and bit
  // offset is a constant, which is what lets the kernel emit straight-line
  // shifts instead of computing them per element.
  [[nodiscard]] constexpr int values_per_chunk() const noexcept {
    return 32 / gcd32(bits);
  }
  [[nodiscard]] constexpr int words_per_chunk() const noexcept {
    return bits / gcd32(bits);
  }

  // Lane and bit offset of code `c` within its chunk. Both are constants the
  // caller resolves at emit time.
  [[nodiscard]] constexpr int chunk_word(int c) const noexcept {
    return (c * bits) / 32;
  }
  [[nodiscard]] constexpr int chunk_bit(int c) const noexcept {
    return (c * bits) % 32;
  }
  // Bits of code `c` that live in the next lane; 0 when it does not straddle.
  [[nodiscard]] constexpr int chunk_carry(int c) const noexcept {
    const int avail = 32 - chunk_bit(c);
    return bits > avail ? bits - avail : 0;
  }

  // Storage a row of `k` weights occupies. Both are exact: k is always a
  // multiple of group_size, and group_size is always a multiple of
  // values_per_chunk (32 | group_size and values_per_chunk | 32).
  [[nodiscard]] std::size_t packed_words(std::size_t k) const noexcept {
    return k * static_cast<std::size_t>(bits) / 32;
  }
  [[nodiscard]] std::size_t group_count(std::size_t k) const noexcept {
    return k / static_cast<std::size_t>(group_size);
  }

  // Whether a row of `k` weights lands on whole lanes and whole groups.
  [[nodiscard]] Status check_row(std::size_t k) const;

  // MLX's own shape identity — w.shape(-1) * 32 / bits == scales.shape(-1) *
  // group_size — run backwards.
  //
  // The identity pins only the RATIO bits/group_size, so this answers with the
  // width implied by the group size it is given and cannot check that group
  // size. A wrong `bits` at the right group size is caught. A (bits,
  // group_size) pair that is wrong in the same proportion — reading a 4-bit
  // group-32 plane as 2-bit group-64, say — satisfies every check and decodes
  // the row at twice its width. So the group_size half of a per-tensor
  // override is load-bearing and unverifiable; only `bits` has a second
  // opinion.
  static Result<int> bits_from_shapes(std::int64_t packed_last,
                                      std::int64_t scales_last,
                                      int group_size);

  // Code `c` of a packed row. `row` is the tensor's own U32 plane.
  [[nodiscard]] std::uint32_t code_at(const std::uint32_t* row,
                                      std::size_t c) const noexcept;
  void set_code(std::uint32_t* row, std::size_t c, std::uint32_t q) const noexcept;

  // scales/biases are the checkpoint's own narrow float; S is bfloat16_t for
  // every MLX build seen so far, float16_t for a producer that chose f16. The
  // round trip goes through that rounding because the file does.
  template <typename S = bfloat16_t>
  void quantize_row(const float* src, std::size_t k, std::uint32_t* packed,
                    S* scales, S* biases) const;

  template <typename S = bfloat16_t>
  void dequantize_row(const std::uint32_t* packed, const S* scales,
                      const S* biases, std::size_t k, float* dst) const;

  template <typename S = bfloat16_t>
  double round_trip_rmse(const float* src, std::size_t k) const;

 private:
  static constexpr int gcd32(int b) noexcept {
    int a = 32;
    while (b != 0) {
      const int t = a % b;
      a = b;
      b = t;
    }
    return a;
  }
};

// Which geometry applies to which tensor.
//
// mlx-lm writes the global {group_size, bits, mode} into the config's
// `quantization` object, and for a mixed-precision checkpoint adds one entry
// per module *to that same object*: `"<module.path>": {"bits": b, ...}` for an
// override, `"<module.path>": false` for a module left in full precision. The
// 35B MoE sibling uses exactly this to make every router 8-bit inside a 6-bit
// model. A router decoded at the wrong width wrecks routing while the model
// still emits plausible text, so the override is not optional and a global
// bits/group_size is not enough.
class GroupAffineMap {
 public:
  // Reads the whole config.json text. `quantization` is preferred over
  // `quantization_config` when both are present and they disagree, matching
  // what mlx-lm treats as authoritative; identical copies are the common case.
  static Result<GroupAffineMap> from_config_json(std::string_view config_json);

  [[nodiscard]] bool has_global() const noexcept { return has_global_; }
  [[nodiscard]] const GroupAffine& global() const noexcept { return global_; }
  [[nodiscard]] std::size_t override_count() const noexcept {
    return overrides_.size();
  }

  // The module a checkpoint tensor belongs to: its name without the
  // `.weight` / `.scales` / `.biases` leaf. mlx-lm keys overrides by exactly
  // this, because it walks the module tree rather than the tensor list.
  static std::string_view module_path_of(std::string_view tensor_name) noexcept;

  // A module the checkpoint explicitly left in full precision.
  [[nodiscard]] bool is_skipped(std::string_view tensor_name) const noexcept;

  // Geometry from the config alone.
  [[nodiscard]] Result<GroupAffine> resolve(std::string_view tensor_name) const;

  // The same, then cross-checked against the tensor's own shapes: the group
  // size comes from the config (the shapes alone cannot separate it from the
  // bit width) and the bit width is re-derived and required to agree. A
  // disagreement names the tensor and both numbers rather than picking one.
  [[nodiscard]] Result<GroupAffine> resolve_checked(
      std::string_view tensor_name, std::int64_t packed_last,
      std::int64_t scales_last) const;

 private:
  GroupAffine global_;
  bool has_global_ = false;
  // Small and read once per tensor at load; a map would cost more than it
  // saves at this size, and insertion order is worth keeping for diagnostics.
  std::vector<std::pair<std::string, GroupAffine>> overrides_;
  std::vector<std::string> skipped_;
};

// ---------------------------------------------------------------------------

inline std::uint32_t GroupAffine::code_at(const std::uint32_t* row,
                                          std::size_t c) const noexcept {
  const std::size_t bit = c * static_cast<std::size_t>(bits);
  const std::size_t word = bit / 32;
  const int off = static_cast<int>(bit % 32);
  const int avail = 32 - off;
  std::uint32_t q = row[word] >> off;
  if (bits > avail) {
    q |= row[word + 1] << avail;
  }
  return q & max_code();
}

inline void GroupAffine::set_code(std::uint32_t* row, std::size_t c,
                                  std::uint32_t q) const noexcept {
  const std::size_t bit = c * static_cast<std::size_t>(bits);
  const std::size_t word = bit / 32;
  const int off = static_cast<int>(bit % 32);
  const int avail = 32 - off;
  const std::uint32_t m = max_code();
  row[word] = (row[word] & ~(m << off)) | ((q & m) << off);
  if (bits > avail) {
    const std::uint32_t hi = m >> avail;
    row[word + 1] = (row[word + 1] & ~hi) | ((q & m) >> avail);
  }
}

template <typename S>
void GroupAffine::quantize_row(const float* src, std::size_t k,
                               std::uint32_t* packed, S* scales,
                               S* biases) const {
  // MLX's affine_quantize, term for term. The edge with the larger magnitude
  // is reproduced exactly by dividing it by its own rounded code, which is
  // where the negative scales come from.
  //
  // One deliberate departure: the codes are chosen against the scale and bias
  // as *stored*, where MLX chooses them against the unrounded f32 pair and
  // narrows afterwards. The file format is defined by the dequant rule, not by
  // the search, so this writes a file MLX reads unchanged — with strictly less
  // error, because the encoder and the decoder now agree on the scale.
  constexpr float kEps = 1e-7f;
  const auto n_bins = static_cast<float>(max_code());
  const std::size_t groups = group_count(k);
  const auto g = static_cast<std::size_t>(group_size);

  for (std::size_t i = 0; i < packed_words(k); ++i) packed[i] = 0;

  for (std::size_t b = 0; b < groups; ++b) {
    const float* s = src + b * g;
    float w_min = s[0];
    float w_max = s[0];
    for (std::size_t i = 1; i < g; ++i) {
      if (s[i] < w_min) w_min = s[i];
      if (s[i] > w_max) w_max = s[i];
    }
    const bool low_edge = std::fabs(w_min) > std::fabs(w_max);
    float scale = (w_max - w_min) / n_bins;
    if (scale < kEps) scale = kEps;
    if (!low_edge) scale = -scale;
    const float edge = low_edge ? w_min : w_max;
    const float q0 = std::round(edge / scale);
    if (q0 != 0.0f) scale = edge / q0;
    const float bias = (q0 == 0.0f) ? 0.0f : edge;

    scales[b].bits = S::from_float(scale);
    biases[b].bits = S::from_float(bias);
    const float stored_scale = scales[b].to_float();
    const float stored_bias = biases[b].to_float();
    const float inv = stored_scale != 0.0f ? 1.0f / stored_scale : 0.0f;

    for (std::size_t i = 0; i < g; ++i) {
      float q = std::round((s[i] - stored_bias) * inv);
      if (q < 0.0f) q = 0.0f;
      if (q > n_bins) q = n_bins;
      set_code(packed, b * g + i, static_cast<std::uint32_t>(q));
    }
  }
}

template <typename S>
void GroupAffine::dequantize_row(const std::uint32_t* packed, const S* scales,
                                 const S* biases, std::size_t k,
                                 float* dst) const {
  const auto g = static_cast<std::size_t>(group_size);
  for (std::size_t i = 0; i < k; ++i) {
    const std::size_t b = i / g;
    // Accumulated in f32 unrounded. MLX rounds q*scale to the storage format
    // before adding the bias; matching that would be bit-identical to MLX and
    // strictly less accurate, so it is not matched.
    dst[i] = static_cast<float>(code_at(packed, i)) * scales[b].to_float() +
             biases[b].to_float();
  }
}

template <typename S>
double GroupAffine::round_trip_rmse(const float* src, std::size_t k) const {
  std::vector<std::uint32_t> packed(packed_words(k));
  std::vector<S> scales(group_count(k));
  std::vector<S> biases(group_count(k));
  std::vector<float> back(k);
  quantize_row<S>(src, k, packed.data(), scales.data(), biases.data());
  dequantize_row<S>(packed.data(), scales.data(), biases.data(), k, back.data());
  double acc = 0.0;
  for (std::size_t i = 0; i < k; ++i) {
    const double d = static_cast<double>(src[i]) - static_cast<double>(back[i]);
    acc += d * d;
  }
  return std::sqrt(acc / static_cast<double>(k));
}

}  // namespace lse::quant
