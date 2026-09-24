#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <string_view>

namespace lse::kernels {
// Storage quantization and compute rounding are separate contracts. This
// profile is a qualification target, not a claim that Q6 implies FP8 accuracy.
enum class QuantOperand : std::uint8_t {
  kExisting,
  kE4M3,
  kE5M2,
  kFP32,
  kBF16
};
enum class QuantOperandStrategy : std::uint8_t {
  kSingleProduct,
  kResidualThreeProduct,
  kNative
};
enum class OperandQualification : std::uint8_t { kCandidate, kAccepted };
enum class OperandReason : std::uint8_t {
  kSelected,
  kStorage,
  kArchitecture,
  kShape,
  kResources,
  kUnavailable,
  kUnsafeConversion,
  kAccuracy,
  kUnqualified,
  kCostEvidence
};
struct QuantOperandRequest {
  std::string_view arch;
  std::uint32_t wave = 0, bits = 0, group = 0;
  std::uint64_t m = 0, n = 0, k = 0;
  std::uint32_t lds_bytes = 0;
  bool f32_activation_output = false, bf16_scale_bias = false;
  bool indexed = false, caller_staged = false;
};
[[nodiscard]] constexpr std::uint64_t
quant_operand_implementation_id(std::string_view name) noexcept {
  std::uint64_t hash = 1469598103934665603ull;
  for (char c : name) {
    hash ^= static_cast<unsigned char>(c);
    hash *= 1099511628211ull;
  }
  return hash;
}
struct QuantOperandKernel {
  QuantOperand operand = QuantOperand::kExisting;
  QuantOperandStrategy strategy = QuantOperandStrategy::kResidualThreeProduct;
  std::uint64_t implementation_id = 0;
  std::uint32_t implementation_revision = 1;
  bool implemented = false, matrix_intrinsic = false,
       conversion_intrinsic = false;
  // Both are mandatory. Finite saturation alone would conceal invalid input.
  bool block_absmax_scaling = false, fp32_exceptional_block_fallback = false;
  std::uint32_t lds_bytes = 0;
};
struct QuantOperandProfile {
  std::uint32_t revision = 1;
  OperandQualification qualification = OperandQualification::kCandidate;
  QuantOperand preferred = QuantOperand::kE4M3;
  QuantOperandStrategy strategy = QuantOperandStrategy::kResidualThreeProduct;
  std::uint32_t min_m = 16, min_n = 16, k_multiple = 64;
  std::uint32_t max_m = 2048, max_n = 32768, max_k = 18432;
  // Five parts per thousand relative L2. Absolute/cancellation and nonfinite
  // checks are independent mandatory acceptance conditions in the fixture.
  std::uint32_t relative_l2_limit_ppm = 5000;
  // Zero means no measured result yet, not zero error. Only accepted profiles
  // carry a nonzero measurement count and all acceptance checks.
  std::uint32_t measured_cases = 0, measured_relative_l2_ppm = 0;
  bool absolute_error_pass = false, nonfinite_pass = false;
  bool model_quality_pass = false, performance_pass = false;
  // Production never runs an unvalidated profile. Standalone qualification
  // tests may explicitly enable a local profile; no environment override exists.
  bool run_qualification_candidate = false;
};
inline constexpr QuantOperandProfile kQuantOperandProfile{};
struct QuantOperandDecision {
  QuantOperand operand = QuantOperand::kExisting;
  QuantOperandStrategy strategy = QuantOperandStrategy::kResidualThreeProduct;
  OperandReason reason = OperandReason::kUnavailable;
  OperandQualification qualification = OperandQualification::kCandidate;
  std::uint32_t relative_l2_limit_ppm = 0;
  std::uint32_t implementation_revision = 0;
  std::uint64_t implementation_id = 0;
};

[[nodiscard]] constexpr QuantOperandDecision select_quant_operand(
    const QuantOperandRequest &r, const QuantOperandKernel &kernel,
    const QuantOperandProfile &p = kQuantOperandProfile) noexcept {
  QuantOperandDecision result{
      QuantOperand::kExisting,     p.strategy,
      OperandReason::kUnavailable, p.qualification,
      p.relative_l2_limit_ppm,     kernel.implementation_revision,
      kernel.implementation_id};
  auto decline = [&](OperandReason why) {
    result.reason = why;
    return result;
  };
  if (r.bits != 6 || r.group != 64 || !r.f32_activation_output ||
      !r.bf16_scale_bias || r.indexed || r.caller_staged)
    return decline(OperandReason::kStorage);
  if (r.arch != "gfx1201" || r.wave != 32)
    return decline(OperandReason::kArchitecture);
  const bool scalar = kernel.operand == QuantOperand::kFP32;
  const bool fp8 = kernel.operand == QuantOperand::kE4M3 ||
                   kernel.operand == QuantOperand::kE5M2;
  if (!r.m || !r.n || !r.k || (!scalar && (r.m < p.min_m || r.n < p.min_n)) ||
      !p.k_multiple || r.k % p.k_multiple || r.m > p.max_m || r.n > p.max_n ||
      r.k > p.max_k)
    return decline(OperandReason::kShape);
  if ((!scalar && !kernel.lds_bytes) || kernel.lds_bytes > r.lds_bytes)
    return decline(OperandReason::kResources);
  if (kernel.operand == QuantOperand::kExisting ||
      kernel.operand != p.preferred || kernel.strategy != p.strategy ||
      !kernel.implemented || !kernel.implementation_id ||
      (!scalar && (!kernel.matrix_intrinsic || !kernel.conversion_intrinsic)))
    return decline(OperandReason::kUnavailable);
  if (fp8 &&
      (!kernel.block_absmax_scaling || !kernel.fp32_exceptional_block_fallback))
    return decline(OperandReason::kUnsafeConversion);
  if (!p.relative_l2_limit_ppm || p.relative_l2_limit_ppm > 5000)
    return decline(OperandReason::kAccuracy);
  if (p.qualification == OperandQualification::kAccepted) {
    if (!p.measured_cases ||
        p.measured_relative_l2_ppm > p.relative_l2_limit_ppm ||
        !p.absolute_error_pass || !p.nonfinite_pass || !p.model_quality_pass ||
        !p.performance_pass)
      return decline(OperandReason::kAccuracy);
  } else if (!p.run_qualification_candidate) {
    return decline(OperandReason::kUnqualified);
  }
  result.operand = kernel.operand;
  result.reason = OperandReason::kSelected;
  return result;
}

[[nodiscard]] constexpr std::string_view
quant_operand_name(QuantOperand value) {
  switch (value) {
  case QuantOperand::kE4M3:
    return "fp8-e4m3";
  case QuantOperand::kE5M2:
    return "bf8-e5m2";
  case QuantOperand::kFP32:
    return "fp32";
  case QuantOperand::kBF16:
    return "bf16";
  default:
    return "existing-floating-point";
  }
}
[[nodiscard]] constexpr std::string_view
quant_operand_reason(OperandReason value) {
  switch (value) {
  case OperandReason::kSelected:
    return "selected";
  case OperandReason::kStorage:
    return "storage-contract";
  case OperandReason::kArchitecture:
    return "device-capability";
  case OperandReason::kShape:
    return "matrix-shape";
  case OperandReason::kResources:
    return "lds-budget";
  case OperandReason::kUnavailable:
    return "kernel-or-intrinsic-unavailable";
  case OperandReason::kUnsafeConversion:
    return "conversion-safety";
  case OperandReason::kAccuracy:
    return "accuracy-or-promotion-evidence";
  case OperandReason::kUnqualified:
    return "unqualified-profile";
  case OperandReason::kCostEvidence:
    return "no-comparable-cost-evidence";
  }
  return "unknown";
}
[[nodiscard]] inline std::string
quant_operand_diagnostic(const QuantOperandRequest &r,
                         const QuantOperandDecision &d) {
  return "operand=" + std::string(quant_operand_name(d.operand)) +
         " strategy=" +
         (d.operand == QuantOperand::kExisting          ? "unchanged"
          : d.strategy == QuantOperandStrategy::kNative ? "native"
          : d.strategy == QuantOperandStrategy::kResidualThreeProduct
              ? "residual-three-product"
              : "single-product") +
         " conversion=" +
         (d.operand == QuantOperand::kExisting ? "unchanged"
          : d.strategy == QuantOperandStrategy::kNative
              ? "native"
              : "block-absmax-rne-fp32-exceptional-fallback") +
         " shape=" + std::to_string(r.m) + "x" + std::to_string(r.n) + "x" +
         std::to_string(r.k) +
         " reason=" + std::string(quant_operand_reason(d.reason)) +
         " qualification=" +
         (d.qualification == OperandQualification::kAccepted
              ? "accepted"
              : "candidate-unvalidated") +
         " target_relative_l2_ppm=" + std::to_string(d.relative_l2_limit_ppm);
}

// Profile changes invalidate both emission and persistent JIT identities,
// including failed selection. Chosen primitive names also distinguish formats.
[[nodiscard]] constexpr std::uint64_t quant_operand_cache_key(
    std::uint64_t key,
    const QuantOperandProfile &p = kQuantOperandProfile) noexcept {
  auto mix = [&](std::uint64_t value) {
    key ^= value;
    key *= 1099511628211ull;
  };
  mix(0x716f706572616e31ull);
  mix(p.revision);
  mix(static_cast<unsigned>(p.qualification));
  mix(static_cast<unsigned>(p.preferred));
  mix(static_cast<unsigned>(p.strategy));
  mix(p.min_m);
  mix(p.min_n);
  mix(p.k_multiple);
  mix(p.max_m);
  mix(p.max_n);
  mix(p.max_k);
  mix(p.relative_l2_limit_ppm);
  mix(p.measured_cases);
  mix(p.measured_relative_l2_ppm);
  mix(p.absolute_error_pass);
  mix(p.nonfinite_pass);
  mix(p.model_quality_pass);
  mix(p.performance_pass);
  mix(p.run_qualification_candidate);
  return key;
}

// Accepted cost records describe a matched experiment. Host intervals and
// device queue intervals are different clocks and must never be ranked
// together.
enum class OperandCostClock : std::uint8_t {
  kHostEvalRetire,
  kGpuQueueInterval
};
enum class OperandCostStatistic : std::uint8_t { kMean, kMedian };
struct QuantOperandCost {
  std::string_view arch;
  std::uint64_t m = 0, n = 0, k = 0;
  std::uint64_t cohort = 0, cost_ns = 0;
  std::uint32_t wave = 0, bits = 0, group = 0, samples = 0;
  std::uint32_t accepted_profile_revision = 0, implementation_revision = 0;
  OperandCostClock clock = OperandCostClock::kHostEvalRetire;
  std::uint64_t implementation_id = 0;
  OperandCostStatistic statistic = OperandCostStatistic::kMean;
};
struct QuantOperandOption {
  QuantOperandKernel kernel;
  QuantOperandProfile profile;
  QuantOperandCost cost;
};
struct QuantOperandRanking {
  QuantOperandDecision decision{};
  std::size_t option = static_cast<std::size_t>(-1);
  std::uint64_t cost_ns = 0;
};
[[nodiscard]] constexpr QuantOperandRanking rank_quant_operands(
    const QuantOperandRequest &r, std::span<const QuantOperandOption> options,
    std::uint64_t cohort, OperandCostClock clock,
    OperandCostStatistic statistic = OperandCostStatistic::kMean) noexcept {
  QuantOperandRanking best;
  best.decision.reason = OperandReason::kCostEvidence;
  for (std::size_t i = 0; i < options.size(); ++i) {
    const auto &o = options[i];
    const auto d = select_quant_operand(r, o.kernel, o.profile);
    if (d.reason != OperandReason::kSelected ||
        o.profile.qualification != OperandQualification::kAccepted)
      continue;
    const auto &c = o.cost;
    if (!cohort || c.cohort != cohort || c.clock != clock ||
        c.statistic != statistic || !c.cost_ns || c.samples < 3 ||
        c.accepted_profile_revision != o.profile.revision ||
        !c.implementation_revision ||
        c.implementation_revision != o.kernel.implementation_revision ||
        c.implementation_id != o.kernel.implementation_id || c.arch != r.arch ||
        c.wave != r.wave || c.bits != r.bits || c.group != r.group ||
        c.m != r.m || c.n != r.n || c.k != r.k)
      continue;
    if (best.option == static_cast<std::size_t>(-1) ||
        c.cost_ns < best.cost_ns) {
      best = {d, i, c.cost_ns};
    }
  }
  return best;
}
// Timing observations do not belong in the JIT identity. A change in the
// selected strategy/profile does; repeated measurements of the same accepted
// strategy need not invalidate code. Shape/device are in the emitter key.
[[nodiscard]] constexpr std::uint64_t
quant_operand_choice_key(std::uint64_t key, const QuantOperandDecision &d,
                         std::uint32_t accepted_profile_revision) noexcept {
  for (auto value : {static_cast<std::uint64_t>(d.operand),
                     static_cast<std::uint64_t>(d.strategy),
                     static_cast<std::uint64_t>(d.implementation_revision),
                     d.implementation_id,
                     static_cast<std::uint64_t>(accepted_profile_revision)}) {
    key ^= value;
    key *= 1099511628211ull;
  }
  return key;
}
} // namespace lse::kernels
