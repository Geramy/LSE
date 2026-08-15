// Inline dims, never allocates: every graph node carries one by value.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>

namespace lse {

inline constexpr std::size_t kMaxRank = 6;

class Shape {
 public:
  Shape() = default;

  Shape(std::initializer_list<std::int64_t> dims) {
    for (std::int64_t d : dims) {
      if (rank_ < kMaxRank) dims_[rank_++] = d;
    }
    recompute();
  }

  [[nodiscard]] std::size_t rank() const noexcept { return rank_; }
  [[nodiscard]] std::int64_t dim(std::size_t i) const noexcept { return dims_[i]; }
  [[nodiscard]] std::int64_t operator[](std::size_t i) const noexcept { return dims_[i]; }

  // Cached: the interpreter used to recompute this per element.
  [[nodiscard]] std::size_t elem_count() const noexcept { return elems_; }

  // Row-major, in elements.
  [[nodiscard]] std::array<std::int64_t, kMaxRank> strides() const noexcept {
    std::array<std::int64_t, kMaxRank> s{};
    std::int64_t acc = 1;
    for (std::size_t i = rank_; i-- > 0;) {
      s[i] = acc;
      acc *= dims_[i];
    }
    return s;
  }

  [[nodiscard]] bool operator==(const Shape& other) const noexcept {
    if (rank_ != other.rank_) return false;
    for (std::size_t i = 0; i < rank_; ++i) {
      if (dims_[i] != other.dims_[i]) return false;
    }
    return true;
  }

  void push_back(std::int64_t d) {
    if (rank_ < kMaxRank) {
      dims_[rank_++] = d;
      recompute();
    }
  }

  [[nodiscard]] std::string to_string() const;

  // Returns rank 0 when the shapes are incompatible.
  [[nodiscard]] static Shape broadcast(const Shape& a, const Shape& b) noexcept;

  [[nodiscard]] bool is_broadcastable_to(const Shape& target) const noexcept;

 private:
  void recompute() noexcept {
    if (rank_ == 0) {
      elems_ = 0;
      return;
    }
    std::size_t n = 1;
    for (std::size_t i = 0; i < rank_; ++i) n *= static_cast<std::size_t>(dims_[i]);
    elems_ = n;
  }

  std::array<std::int64_t, kMaxRank> dims_{};
  std::size_t rank_ = 0;
  std::size_t elems_ = 0;
};

// Maps a flat output index to a flat source index under NumPy trailing-dim
// broadcast. Built once and shared by the host interpreter and the kernel
// emitter so the two can never disagree — they did: the emitter was missing the
// rank-underflow guard and silently read element 0 for every position.
struct BroadcastMap {
  std::array<std::int64_t, kMaxRank> out_stride{};
  std::array<std::int64_t, kMaxRank> src_stride{};
  std::array<std::int64_t, kMaxRank> out_dim{};
  std::size_t rank = 0;
  std::size_t gap = 0;
  // src maps 1:1 onto out; no index arithmetic needed.
  bool identity = false;
  // src is a single element broadcast everywhere.
  bool scalar = false;

  [[nodiscard]] static BroadcastMap build(const Shape& src, const Shape& out) noexcept;

  [[nodiscard]] std::size_t apply(std::size_t flat) const noexcept {
    if (identity) return flat;
    if (scalar) return 0;
    std::size_t src_index = 0;
    for (std::size_t i = gap; i < rank; ++i) {
      const std::size_t si = i - gap;
      if (src_stride[si] == 0) continue;
      const std::int64_t coord =
          static_cast<std::int64_t>(flat) / out_stride[i] % out_dim[i];
      src_index += static_cast<std::size_t>(coord * src_stride[si]);
    }
    return src_index;
  }
};

}  // namespace lse
