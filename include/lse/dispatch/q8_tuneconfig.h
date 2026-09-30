#pragma once

#include <array>
#include <cstdint>
#include <string_view>

namespace lse::dispatch::q8_shapes {

struct MatrixRule {
  std::string_view arch;
  std::uint32_t wave, bits, rows, lds, threads, max_round_groups;
};
inline constexpr std::array kMatrixRules{
    MatrixRule{"gfx1201", 32, 8, 16, 1664, 256, 4},
    MatrixRule{"gfx1201", 32, 8, 64, 6656, 256, 1},
};

struct ShapeRule { std::uint32_t min_m, max_m, n, k, rows; };
inline constexpr std::array kShapes{
    ShapeRule{64, UINT32_MAX, 0, 0, 64},
    ShapeRule{1, 8, 5120, 10240, 16},
    ShapeRule{1, 8, 17408, 5120, 16},
    ShapeRule{3, 8, 5120, 17408, 16},
    ShapeRule{8, 8, 4096, 5120, 16},
    ShapeRule{8, 8, 5120, 4096, 16},
    ShapeRule{8, 8, 1024, 5120, 16},
    ShapeRule{8, 8, 1280, 5120, 16},
    ShapeRule{3, 7, 5120, 25600, 16},
};

struct PackedShape { std::uint32_t n, k; };
inline constexpr std::array kPackedShapes{
    PackedShape{17408, 5120}, PackedShape{5120, 10240},
};

}  // namespace lse::dispatch::q8_shapes
