// The matrix core, above the descriptor.
//
// `lse::math` owns the table: one row per (target, accumulator, operand, tile),
// carrying the fragment widths, the wave, the K step, the capability and the
// measured lane mappings. This header is the layer between that row and a
// kernel — the device gate that decides whether the live device may run a row,
// and the CRTP tile that holds the algorithm every row shares.
//
// Nothing here spells an instruction, a width or a lane index. A new variant is
// a row in the table plus, if it is a new storage format, an arm in
// `with_matrix_operand`. It is never an edit to a kernel body.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "lse/backends/hrx/device_info.hpp"
#include "lse/kernels/vec_mem.hpp"
#include "lse/core/dtype.hpp"
#include "lse/graph/kernel_env.hpp"
#include "lse/graph/kernel_ir.hpp"
#include "lse/graph/kernel_primitive.hpp"
#include "lse/math.hpp"
#include "lse/quant/group_affine_codec.hpp"

namespace lse::kernels {

// These name device facts, which the backend supplies.
using backend::MatrixCore;
using backend::AmdDeviceInfo;
using backend::DeviceInfo;
using backend::arch_family;
using backend::device_extension;
using backend::ArchFamily;

// The matrix-core form of `linear` for this invocation, or null when the
// device, the shapes or the opt-in switch rule it out and the scalar loop
// should stand.
const graph::KernelPrimitiveBase* wmma_linear_for(const graph::KernelShapes& s);

// The same, for the group-affine quantized contraction.
const graph::KernelPrimitiveBase* wmma_quant_linear_for(
    const graph::KernelShapes& s);

// Which family of matrix instructions this device speaks. Not "does it have a
// matrix core" — the families are different instructions with different
// operand layouts, so a row belongs to exactly one of them.
[[nodiscard]] inline std::optional<lse::math::MatrixTarget> matrix_target(
    const DeviceInfo& info) noexcept {
  const AmdDeviceInfo* amd = device_extension<AmdDeviceInfo>(info);
  if (amd == nullptr) return std::nullopt;
  switch (arch_family(info.arch)) {
    case ArchFamily::kRdna3:
    case ArchFamily::kRdna35:
      if (amd->matrix_core == MatrixCore::kWMMA && info.wavefront_size == 32) {
        return lse::math::MatrixTarget::kRdna3;
      }
      return std::nullopt;
    case ArchFamily::kRdna4:
      if (amd->matrix_core == MatrixCore::kWMMA && info.wavefront_size == 32) {
        return lse::math::MatrixTarget::kRdna4;
      }
      return std::nullopt;
    case ArchFamily::kCdna3:
    case ArchFamily::kCdna4:
      if (amd->matrix_core == MatrixCore::kMFMA && info.wavefront_size == 64) {
        return lse::math::MatrixTarget::kCdna3;
      }
      return std::nullopt;
    default:
      return std::nullopt;
  }
}

// The operand forms this device's matrix core has. Derived from the arch family
// and the core generation HRX already reports, so a new device needs no new
// probe — but it is a separate answer from how a target *spells* the
// instruction, and a row is legal only when both agree.
[[nodiscard]] inline std::uint32_t device_matrix_caps(
    const DeviceInfo& info) noexcept {
  using lse::math::MatrixCap;
  const AmdDeviceInfo* amd = device_extension<AmdDeviceInfo>(info);
  const std::optional<lse::math::MatrixTarget> target = matrix_target(info);
  if (amd == nullptr || !target.has_value()) return 0;
  std::uint32_t caps = 0;
  switch (*target) {
    case lse::math::MatrixTarget::kRdna3:
      caps = cap_bits(MatrixCap::kWmmaF16) | cap_bits(MatrixCap::kWmmaInt8) |
             cap_bits(MatrixCap::kWmmaInt4);
      if (amd->matrix_core_bf16) caps |= cap_bits(MatrixCap::kWmmaBf16);
      break;
    case lse::math::MatrixTarget::kRdna4:
      caps = cap_bits(MatrixCap::kWmma12F16) |
             cap_bits(MatrixCap::kWmma12Int8) |
             cap_bits(MatrixCap::kWmma12Int4) | cap_bits(MatrixCap::kWmma12Fp8);
      if (amd->matrix_core_bf16) caps |= cap_bits(MatrixCap::kWmma12Bf16);
      break;
    case lse::math::MatrixTarget::kCdna3:
      caps = cap_bits(MatrixCap::kMfmaF16) | cap_bits(MatrixCap::kMfmaInt8) |
             cap_bits(MatrixCap::kMfmaFp8);
      if (amd->matrix_core_bf16) caps |= cap_bits(MatrixCap::kMfmaBf16);
      break;
  }
  return caps;
}

// Which row a stored weight selects, and what the activation buffer beside it
// looks like. Both are table arms, not branches: a checkpoint format joins the
// matrix core by adding a line here and its row, and no kernel changes.
//
// f32 is the one narrowing and it is unavoidable — the matrix core has no f32
// operand form, so an f32 weight has always taken the f16 one. There is no
// lateral conversion between float formats: a bf16 checkpoint feeds the bf16
// instruction, which is the arithmetic it was trained in.
template <class R, class F>
[[nodiscard]] R with_matrix_operand(DType weight, F&& fn) {
  using lse::math::MatrixElem;
  namespace kir = graph::kir;
  switch (weight) {
    case DType::kF32:
      return fn.template operator()<kir::f32, kir::f32, MatrixElem::kF32,
                                    MatrixElem::kF16>();
    case DType::kBF16:
      return fn.template operator()<kir::f32, lse::bf16, MatrixElem::kF32,
                                    MatrixElem::kBF16>();
    case DType::kF16:
      return fn.template operator()<kir::f32, lse::f16, MatrixElem::kF32,
                                    MatrixElem::kF16>();
    // An integer matrix core takes its operands already packed — four int8 to
    // an i32 lane — so both buffers are i32 and the fill converts nothing.
    case DType::kI32:
      return fn.template operator()<std::int32_t, std::int32_t,
                                    MatrixElem::kI32, MatrixElem::kI8>();
    default:
      return R{};
  }
}

// Which generation's instructions this invocation gets, as a compile-time
// parameter. The target is a device fact discovered at run time, and every
// width the tile needs hangs off the row that target selects, so a kernel that
// wants to serve more than one generation has to cross from one to the other
// exactly once. This is that crossing; nothing downstream names a generation.
template <class R, class F>
[[nodiscard]] R with_matrix_target(lse::math::MatrixTarget target, F&& fn) {
  using lse::math::MatrixTarget;
  switch (target) {
    case MatrixTarget::kRdna3:
      return fn.template operator()<MatrixTarget::kRdna3>();
    case MatrixTarget::kRdna4:
      return fn.template operator()<MatrixTarget::kRdna4>();
    case MatrixTarget::kCdna3:
      return fn.template operator()<MatrixTarget::kCdna3>();
  }
  return R{};
}

// What a row's measured layouts mean in indices, worked out once here so that
// no kernel repeats it and none of them names a generation.
//
// A tile written against this asks where its lane's operand slice begins and
// where an accumulator slot lands; it never asks which layout the row carries.
// Adding a generation is then a row plus, at most, an arm below -- not an edit
// to every kernel that multiplies.
struct TileGeometry {
  std::uint32_t wave = 0;      // lanes cooperating on one tile
  std::uint32_t slots = 0;     // accumulator registers a lane holds
  std::uint32_t frag = 0;      // operand registers one instruction takes
  std::uint32_t halves = 0;    // lanes per column: wave / n
  // Operand values a lane holds of the instruction's k step, and whether the
  // halves of the wave divide that step between them.
  std::uint32_t lane_k = 0;
  bool split_k = false;
  // Rows between one accumulator slot and the next. The pair form interleaves
  // the halves of the wave and steps by `halves`; the block form gives each
  // half a contiguous run and steps by one.
  std::uint32_t slot_step = 0;
  // Rows this lane's slot zero sits above the tile's origin.
  std::uint32_t half_rows = 0;
};

[[nodiscard]] consteval TileGeometry geometry_of(
    const lse::math::MatrixCoreRow& r) {
  TileGeometry g;
  g.wave = static_cast<std::uint32_t>(r.wave);
  g.slots = static_cast<std::uint32_t>(r.c_len);
  g.frag = static_cast<std::uint32_t>(r.a_len / r.chained);
  g.halves = g.wave / static_cast<std::uint32_t>(r.n);
  g.split_k = r.operands == lse::math::OperandLayout::kLaneRowSplitK;
  g.lane_k = g.split_k ? static_cast<std::uint32_t>(r.k) / g.halves
                       : static_cast<std::uint32_t>(r.k);
  const bool pair = r.acc_layout == lse::math::AccLayout::kPairRowHalfWave;
  g.slot_step = pair ? g.halves : 1u;
  g.half_rows = pair ? 1u : g.slots;
  return g;
}

// A stored weight that is narrower than the matrix core's operand.
//
// The core has no 4-bit operand a float activation can meet, so a 4-bit
// checkpoint feeds the 8-bit row and the codes are widened on the way into the
// fragment. That widening belongs here, beside the arm that chose the row, and
// not in a kernel: a kernel that spells `(word >> 4 * nib) & 15` has hardcoded
// one checkpoint's storage into the one place this header exists to keep it
// out of.
//
// `Bits` is the stored width. 8 is the identity — the buffer already holds
// what the fragment wants — and is written as the same loop so the two cannot
// drift.
template <int Bits>
struct PackedCodes {
  static_assert(Bits == 4 || Bits == 8, "no widening arm for this width");

  // Stored codes in one 32-bit buffer element.
  static constexpr std::uint32_t kPerWord = 32u / static_cast<std::uint32_t>(Bits);
  // Codes in one fragment register, which the 8-bit row fixes at four.
  static constexpr std::uint32_t kPerFrag = 4u;
  // Buffer elements one fragment register is built from.
  static constexpr std::uint32_t kWordsPerFrag = kPerFrag / kPerWord ? kPerFrag / kPerWord : 1u;

  // Fragment register `f` of a row starting at buffer element `base`, for a
  // K slice `k0` counted in codes.
  static graph::kir::Val<graph::kir::u32> word(
      graph::env::Emit& e,
      const graph::env::In<std::uint32_t, graph::env::Emit>& buf,
      const graph::kir::Val<graph::kir::u32>& base,
      std::uint32_t f) {
    namespace kir = graph::kir;
    if constexpr (Bits == 8) {
      return e.let(buf[e.let(base + f)]);
    } else {
      // Two fragment registers come out of one stored word, and a nibble's
      // position in that word is its own code index. This is NOT the plane
      // split quant::dot4_operand_slot spells: dot4 pairs byte b of a plane
      // with code 2b+p because it consumes a chunk as two interleaved
      // operands, whereas this row is kLaneRowContiguousK -- register f holds
      // codes 4f..4f+3 in order, against sixteen contiguous activations.
      // Borrowing the plane order here pairs every code with the wrong
      // activation; quant_linear_matches_the_weights_it_encodes catches it.
      const auto src = e.let(buf[e.let(base + f / 2u)]);
      const std::uint32_t first = (f % 2u) * kPerFrag;
      auto out = e.let(e.u32(0));
      for (std::uint32_t b = 0; b < kPerFrag; ++b) {
        const auto code = e.let((src / (1u << (4 * (first + b)))) % 16u);
        out = e.let(out + code * (1u << (8 * b)));
      }
      return out;
    }
  }
};

// ---------------------------------------------------------------------------
// The shared algorithm
// ---------------------------------------------------------------------------
//
// One wave owns one output tile of the row's shape and walks K in the row's
// own step. Every width, every lane index and every instruction count comes
// from `kRow`; `Derived` supplies only what is specific to the operation it is
// building — where an operand row starts, and what happens to a finished
// accumulator slot.
//
// The same shape `quant::QuantScheme` uses: the base owns everything identical
// across rows, the derived class owns the two things that are not.
template <class Derived, lse::math::MatrixTarget G, lse::math::MatrixElem Acc,
          lse::math::MatrixElem T, int Mi, int Ni, int Ki>
struct MatrixTile {
  using Op = lse::math::op::Mma<G, Acc, T, Mi, Ni, Ki>;
  static constexpr lse::math::MatrixCoreRow kRow = Op::kRow;

  using AFrag = lse::math::matrix_scalar_t<kRow.a_elem>;
  using BFrag = lse::math::matrix_scalar_t<kRow.b_elem>;
  using CFrag = lse::math::matrix_scalar_t<kRow.c_elem>;

  static_assert(kRow.a_len == kRow.b_len,
                "the dense tile assumes symmetric operands; a sparse row "
                "(swmmac, A at half width) needs its own tile");
  static_assert(kRow.k_step == kRow.k * kRow.chained,
                "k_step must be what chaining actually consumes");
  static_assert(kRow.chained >= 1 && kRow.a_len % kRow.chained == 0,
                "a chained row splits its fragment evenly per instruction");
  static_assert(kRow.wave % kRow.n == 0,
                "the accumulator mapping folds the wave over the tile width");

  // Fragment registers, and operand values, per chained instruction.
  static constexpr std::uint32_t kAPer =
      static_cast<std::uint32_t>(kRow.a_len / kRow.chained);
  // How many buffer elements one instruction's fragment spans. Equal to kAPer
  // whenever a lane holds its whole K slice contiguously, which is what
  // kLaneRowContiguousK means; a layout that splits K across half-waves does
  // not, and is why such a row is not emittable until it is measured.
  static constexpr std::uint32_t kHalves =
      static_cast<std::uint32_t>(kRow.wave / kRow.n);

  template <int Len>
  using AReg = graph::kir::Local<AFrag, Len>;

  // Widest load of `elems` buffer elements that stays inside one fragment and
  // on a natural boundary for every row of a `kb`-element buffer.
  template <class Src>
  static std::uint32_t operand_step(std::uint32_t kb,
                                    std::uint32_t load_bytes) noexcept {
    std::uint32_t n =
        row_pack(kb, load_bytes, graph::kir::pack_elem_bytes<Src>());
    while (n > kAPer) n >>= 1;
    return n;
  }

  // One operand fragment, filled from a row-major buffer.
  //
  // The buffer is indexed in its own elements, so a packed operand simply has
  // K/pack of them per row and the loop is the same one. `narrow` is the
  // identity when the register already holds the buffer's element type, which
  // is the integer case.
  template <int Len, class Frag, class Src>
  static void fill(graph::env::Emit& e,
                   const graph::env::In<Src, graph::env::Emit>& buf,
                   const graph::kir::Val<graph::kir::u32>& base,
                   const graph::kir::Val<graph::kir::u32>& k0,
                   const graph::kir::Val<graph::kir::boolean>& in_range,
                   std::uint32_t kb, const graph::kir::Local<Frag, Len>& frag,
                   std::uint32_t step) {
    constexpr std::uint32_t kSrcBytes = graph::kir::pack_elem_bytes<Src>();
    for (auto c : e.unroll(static_cast<std::uint32_t>(Len) / step)) {
      const auto kk = e.let(k0 + c * step);
      if (auto g = e.when(in_range && kk + step <= kb)) {
        const auto v = e.load(buf, base * kb + kk, step * kSrcBytes);
        for (auto j : e.unroll(step)) {
          frag[c * step + j] = lse::math::narrow<Frag>(v[j]);
        }
      }
    }
  }

  // The tile. `m`, `n` are the matrix's own extents and `kb` its row length in
  // buffer elements; the instruction's shape is the row's.
  template <class XSrc, class WSrc>
  void run(graph::env::Emit& e,
           const graph::env::In<XSrc, graph::env::Emit>& x,
           const graph::env::In<WSrc, graph::env::Emit>& w, std::uint32_t m,
           std::uint32_t n, std::uint32_t kb, std::uint32_t load_bytes) const {
    namespace kir = graph::kir;
    namespace math = lse::math;

    constexpr auto kWave = static_cast<std::uint32_t>(kRow.wave);
    constexpr auto kM = static_cast<std::uint32_t>(kRow.m);
    constexpr auto kN = static_cast<std::uint32_t>(kRow.n);
    constexpr auto kChain = static_cast<std::uint32_t>(kRow.chained);
    // The K loop steps in buffer elements: the row's K step divided by how many
    // operand values ride in one of them.
    constexpr std::uint32_t kStepBuf =
        static_cast<std::uint32_t>(kRow.k_step / kRow.pack);

    const std::uint32_t tiles_n = (n + kN - 1u) / kN;
    const std::uint32_t tiles = ((m + kM - 1u) / kM) * tiles_n;

    const auto wave = e.let(e.thread_id() / kWave);
    // Predicated, never an early return: sibling fusion concatenates several
    // of these bodies into one kernel behind one thread index, so a `return`
    // here retires waves a later sibling still needs.
    if (auto in_tiles = e.when(wave < tiles)) {
      const auto lane = e.let(e.thread_id() % kWave);
      const auto lane_lo = e.let(lane % kN);
      const auto lane_hi = e.let(lane / kN);
      const auto m0 = e.let((wave / tiles_n) * kM);
      const auto n0 = e.let((wave % tiles_n) * kN);

      // Lane L supplies row L % n of x and of w. On the contiguous layout the
      // two halves of the wave carry the same rows over the whole k step; on
      // the split layout each half carries its own half of the k step instead,
      // which is the RDNA4 fill (see OperandLayout::kLaneRowSplitK). Either
      // way the fragment load is the checkpoint's own layout — what moves is
      // where a lane's k window starts, below.
      const auto arow = e.let(m0 + lane_lo);
      const auto bcol = e.let(n0 + lane_lo);
      // This lane's k origin within one instruction's k step, in buffer
      // elements. Zero on the contiguous layout; the half-wave's share on the
      // split one.
      constexpr auto kG = geometry_of(kRow);
      static_assert(!kG.split_k || (kG.lane_k % kRow.pack) == 0,
                    "a lane's k share must be whole buffer elements");
      constexpr std::uint32_t kLaneKBuf =
          kG.split_k ? kG.lane_k / static_cast<std::uint32_t>(kRow.pack) : 0u;

      const auto acc = e.local<CFrag, kRow.c_len>();
      for (auto z : e.unroll(static_cast<std::uint32_t>(kRow.c_len))) {
        acc[z] = math::narrow<CFrag>(e.f32(0.0f));
      }

      // Each operand takes the widest load its own element size and row length
      // allow, so a 2-byte weight moves 8 per instruction where the 4-byte
      // activation moves 4.
      const std::uint32_t xstep = operand_step<XSrc>(kb, load_bytes);
      const std::uint32_t wstep = operand_step<WSrc>(kb, load_bytes);
      for (auto k0 : e.range(0u, kb, kStepBuf)) {
        // One pair of fragments per chained instruction. A row that fuses two
        // steps to fill a 128-bit load holds two.
        std::vector<AReg<static_cast<int>(kAPer)>> af;
        std::vector<graph::kir::Local<BFrag, static_cast<int>(kAPer)>> bf;
        af.reserve(kChain);
        bf.reserve(kChain);
        for (std::uint32_t c = 0; c < kChain; ++c) {
          af.push_back(e.local<AFrag, static_cast<int>(kAPer)>());
          bf.push_back(e.local<BFrag, static_cast<int>(kAPer)>());
        }
        for (std::uint32_t c = 0; c < kChain; ++c) {
          const auto azero = math::narrow<AFrag>(e.f32(0.0f));
          const auto bzero = math::narrow<BFrag>(e.f32(0.0f));
          for (auto z : e.unroll(kAPer)) {
            af[c][z] = azero;
            bf[c][z] = bzero;
          }
        }
        for (std::uint32_t c = 0; c < kChain; ++c) {
          const auto kc0 = c == 0 ? k0 : e.let(k0 + c * kAPer);
          // The lane's own window: on the split layout the upper half-wave
          // reads the second half of the k step.
          const auto kc =
              kLaneKBuf == 0 ? kc0 : e.let(kc0 + lane_hi * kLaneKBuf);
          fill(e, x, arow, kc, arow < m, kb, af[c], xstep);
          fill(e, w, bcol, kc, bcol < n, kb, bf[c], wstep);
          acc = math::mma<Op>(af[c].value(), bf[c].value(), acc.value());
        }
      }

      // Where lane L's accumulator element z lands. Measured on the device,
      // carried by the row through its geometry, not rediscovered here: the
      // pair form interleaves the wave halves (row = z*halves + lane_hi), the
      // block form gives each half a contiguous run (row = z + c_len*lane_hi)
      // — both are z*slot_step + lane_hi*half_rows.
      static_assert(kG.slot_step * 1u + kG.half_rows > 0,
                    "the accumulator geometry must be measured to emit");
      const auto col = e.let(n0 + lane_lo);
      for (auto z : e.unroll(static_cast<std::uint32_t>(kRow.c_len))) {
        const auto row =
            e.let(m0 + z * kG.slot_step + lane_hi * kG.half_rows);
        if (auto gr = e.when(row < m && col < n)) {
          static_cast<const Derived*>(this)->emit_element(
              e, row, col, math::widen(acc[z].read()));
        }
      }
    }
  }
};

}  // namespace lse::kernels
