// Bytes a kernel moves, by what it is moving, and whether the object agrees.
//
// WHY THE SPLIT BY CLASS IS THE POINT. A contraction reads two things with
// completely different reuse: the weight is read once per output tile, the
// activation once per weight tile. A tile that divides one term does nothing to
// the other, so a model that reports a single byte total cannot say which
// direction to tile — and pricing only the weight is exactly the failure this
// exists to prevent. Measured on this tree's own 4-bit contraction at 8 rows
// and 8 columns per workgroup, sixteen of eighteen wide loads were activation
// staging and two were weights.
//
// TWO SIDES, DELIBERATELY. `TrafficModel` is what the emitter means to move,
// derived from shapes and the tile it chose. `EmittedTraffic` is what the
// object's instructions add up to. `check_traffic` compares them, and a gap
// says the emission did not do what the emitter meant. Neither side is a
// measurement of a run: one is arithmetic over shapes, the other a static count
// over a body, so the comparison catches a wrong intention and not a slow
// machine.
#pragma once

#include <array>
#include <cstdint>
#include <string>

#include "lse/backend/census.hpp"
#include "lse/core/enum_names.hpp"

namespace lse::opt {

// What a kernel is reading, in engine vocabulary. The distinction that matters
// is reuse, not dtype: a weight is read once per output tile and an activation
// once per weight tile, which is why the two never share a term.
#define LSE_OPERAND_CLASS_LIST(X)                                            \
  X(kWeight, "weight")         /* a parameter of the model */                \
  X(kActivation, "activation") /* a value this step produced */              \
  X(kScale, "scale")           /* quantization metadata beside a weight */   \
  X(kIndex, "index")           /* a selector, read per row */                \
  X(kOutput, "output")         /* what the kernel writes */                  \
  X(kOther, "other")

LSE_DECLARE_ENUM(OperandClass, std::uint8_t, LSE_OPERAND_CLASS_LIST)

inline constexpr std::size_t kOperandClasses =
    kEnumEntries_OperandClass.size();

// Bytes ONE WORKGROUP moves, split by class. Per workgroup rather than per
// launch because that is the granularity a tile choice changes: the grid is a
// consequence of the tile, not an input to it.
struct TrafficModel {
  std::array<std::uint64_t, kOperandClasses> read{};
  std::array<std::uint64_t, kOperandClasses> written{};

  // Multiply-accumulates one workgroup performs. The denominator of
  // bytes_per_work(), and zero for a kernel that contracts nothing — in which
  // case bytes per unit of work is not a quantity, not a zero.
  std::uint64_t work = 0;
  // Workgroups the launch dispatches, so a caller can scale to the whole grid
  // without re-deriving the geometry.
  std::uint32_t workgroups = 0;
  // Threads in one workgroup. Carried here because the census counts per lane
  // and this is the multiplier that puts the two sides in the same unit;
  // keeping it beside the model is what stops a caller supplying its own.
  std::uint32_t workgroup_threads = 0;

  // Whether anything filled this in. An unstated model is not an empty one:
  // a primitive that has not said what it moves must never read as moving
  // nothing.
  bool stated = false;

  void add_read(OperandClass what, std::uint64_t bytes);
  void add_written(OperandClass what, std::uint64_t bytes);

  [[nodiscard]] std::uint64_t bytes_read(OperandClass what) const;
  [[nodiscard]] std::uint64_t bytes_read() const;
  [[nodiscard]] std::uint64_t bytes_written() const;
  [[nodiscard]] std::uint64_t total_bytes() const;

  // Bytes moved per multiply-accumulate. The reciprocal of arithmetic
  // intensity, and what the ridge point is compared against.
  [[nodiscard]] double bytes_per_work() const;
  // This class's share of what the workgroup reads, 0 when it reads nothing.
  [[nodiscard]] double read_share(OperandClass what) const;

  // Summing stages of one launch. A run is unstated unless every stage stated
  // itself: a partial sum understates traffic, which is the direction that
  // looks like an improvement.
  TrafficModel& operator+=(const TrafficModel& other);

  [[nodiscard]] std::string describe() const;
};

// One workgroup's traffic through a contraction of `rows` activation rows
// against `cols` weight columns over a reduction depth of `depth`.
//
// This is the arithmetic from the design record, written once: the weight term
// scales with the columns a workgroup owns and the activation term with the
// rows, and neither divides the other. `weight_bits` is per element because a
// packed 4-bit code is half a byte and rounding it to one doubles the term.
[[nodiscard]] TrafficModel contraction_traffic(
    std::uint64_t rows, std::uint64_t cols, std::uint64_t depth,
    std::uint32_t weight_bits, std::uint32_t activation_bytes,
    std::uint64_t scale_bytes, std::uint64_t output_bytes);

// What the object's instructions add up to for one workgroup: per-lane bytes
// times the threads a workgroup launches.
//
// `exact` is false when any counted access sits under a backward branch, in
// which case these are the bytes of ONE traversal and the launch moves at least
// this much. There is no class split here, and there deliberately is none: the
// object names address spaces, not roles, and inventing the mapping would put a
// guess where the whole point is a fact.
struct EmittedTraffic {
  std::uint64_t global_read = 0;
  std::uint64_t global_written = 0;
  std::uint64_t shared_read = 0;
  std::uint64_t shared_written = 0;
  // Global bytes from accesses that cannot repeat. A floor on the launch's
  // global traffic whatever the loops do.
  std::uint64_t global_floor = 0;
  // Multiply-accumulates the workgroup performs over the same traversal the
  // bytes above cover — an upper bound, since arithmetic spent on dequantizing
  // or addressing is a multiply-add too.
  std::uint64_t work = 0;
  bool exact = false;
  bool known = false;

  [[nodiscard]] std::uint64_t global_bytes() const {
    return global_read + global_written;
  }

  // WHY THIS IS THE COMPARABLE QUANTITY, and the absolute bytes are not.
  //
  // Bytes survive a loop badly: a static count of a looping body is one
  // traversal and the launch does many. Intensity survives it better, because
  // a loop that repeats bytes and arithmetic together drops out of the ratio.
  //
  // IT IS NOT TRIP-COUNT INVARIANT HERE, and the earlier claim that it was is
  // wrong. The two are not in the same loop on these kernels: the activation
  // bytes come from the staging loops and the arithmetic from the contraction
  // loop, whose trip counts differ by construction — a factor of two at one
  // column per wave, more at other tiles. Measured over 44 emitted shapes the
  // object's bytes-per-mac ran from 0.51x to 1.87x the launch's true figure,
  // and 26 of them sat above it, so there is no bound in either direction.
  //
  // What survives is that the ratio is bounded, and measurably so. That is why
  // the band below is set from that spread and not from a power of two.
  [[nodiscard]] double bytes_per_work() const {
    if (work == 0) return 0.0;
    return static_cast<double>(global_bytes()) / static_cast<double>(work);
  }
};

[[nodiscard]] EmittedTraffic emitted_traffic(const backend::KernelCensus& c,
                                             std::uint32_t workgroup_threads);

#define LSE_TRAFFIC_VERDICT_LIST(X)                                           \
  X(kUnknown, "unknown")        /* one side said nothing */                   \
  X(kUndecided, "undecided")    /* the body loops and neither the intensity   \
                                   nor the floor settles it */                \
  X(kAgrees, "agrees")                                                        \
  X(kEmitsMore, "emits-more")   /* the object moves more than was intended */ \
  X(kEmitsLess, "emits-less")   /* the object moves less than was intended */

LSE_DECLARE_ENUM(TrafficVerdict, std::uint8_t, LSE_TRAFFIC_VERDICT_LIST)

// WHAT IS DECIDABLE, AND WHY THAT IS NOT SYMMETRIC.
//
// The emitted side is a static count. On a body with no loop it is the whole
// launch's traffic and the comparison runs both ways. On a body with a loop it
// is one traversal, and three readings survive that, in order of strength:
//
//   1. INTENSITY. A loop that repeats bytes and arithmetic together drops out
//      of their ratio, which is what makes a looping body comparable at all.
//      It does not drop out cleanly here — the bytes and the macs sit in
//      different loops — so the ratio carries a measured instrument spread of
//      0.51x to 1.87x rather than a bound. Only a difference well outside that
//      spread is a disagreement.
//   2. THE FLOOR. What sits outside every loop cannot be repeated away, so it
//      bounds the launch from below whatever the trip counts are.
//   3. THE WHOLE COUNT, which is exact only where nothing loops.
//
// "The object moves less than intended" is therefore not decidable on a looping
// body, and is not reported there. Reporting it anyway would fire on every
// contraction in the tree, since a K loop traversed once looks like a kernel
// that reads almost nothing. A looping body under its intent comes back
// undecided rather than agreeing: an agreement nobody can check is worse than
// an admission.
//
// THE BAND IS A FACTOR OF TWO, and it is a rule rather than a number tuned
// until a case passed.
//
// The two sides count different things — the model counts operand bytes, the
// object counts issued instructions — so they can never be equal. What they
// cannot differ by is a whole term. Every way either side miscounts moves in
// powers of two: a width class read as b64 instead of b128, a row of the tile
// counted once instead of twice, one operand of a pair forgotten. Below 2x the
// difference is address setup, loop tails and the epilogue's own operands; at
// or above it an entire term is missing or duplicated. The failure this exists
// to catch was 9x.
// Set from the instrument's own measured spread, not from a power of two. The
// static ratio ran 0.51x to 1.87x the true figure across 44 emitted shapes, so
// a band of two is 7% from calling a correct model wrong. Three clears that
// spread with margin and still catches what this check exists for: pricing the
// contraction by its weights alone, the defect that motivated it, reads 13.9x
// at K=5120 and 4.04x at K=17408.
inline constexpr double kTrafficDisagreementFactor = 3.0;

struct TrafficCheck {
  TrafficVerdict verdict = TrafficVerdict::kUnknown;
  // All per workgroup, all counting global traffic only: shared scratch is an
  // internal staging of bytes already counted once at the global level, and
  // adding it would charge the same byte twice.
  std::uint64_t intended_bytes = 0;
  // Every counted access once. On a body with no loop that is the launch's
  // traffic; on one with a loop it is a traversal's, and the launch moves more.
  std::uint64_t emitted_bytes = 0;
  // Only the accesses that cannot repeat. This one is a floor on what the
  // launch moves whatever the trip counts are, which is what makes kEmitsMore
  // decidable on a looping body at all.
  std::uint64_t emitted_floor = 0;
  // False when part of the body repeats, so `emitted_bytes` under-counts by an
  // unknown factor and the verdict on it is an indication rather than a proof.
  bool emitted_exact = false;

  // Bytes per multiply-accumulate, from each side. Trip-count invariant, so
  // this is the comparison that decides a looping body; zero where either side
  // could not say how much work it does.
  double intended_bytes_per_work = 0.0;
  double emitted_bytes_per_work = 0.0;

  [[nodiscard]] double ratio() const;
  // emitted / intended, in bytes per unit of work. Zero when one side is silent.
  [[nodiscard]] double intensity_ratio() const;
  [[nodiscard]] bool disagrees() const {
    return verdict == TrafficVerdict::kEmitsMore ||
           verdict == TrafficVerdict::kEmitsLess;
  }
  [[nodiscard]] std::string describe() const;
};

[[nodiscard]] TrafficCheck check_traffic(const TrafficModel& intended,
                                         const backend::KernelCensus& emitted,
                                         std::uint32_t workgroup_threads);

}  // namespace lse::opt
