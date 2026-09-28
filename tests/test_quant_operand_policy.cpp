#include "lse/kernels/quant_operand_policy.hpp"
#include <array>
#include <cstdio>
#include <stdexcept>
using namespace lse::kernels;
namespace {
void require(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}
QuantOperandRequest request() {
  return {"gfx1201", 32,    6,    64,   512,   17408,
          5120,      65536, true, true, false, false};
}
QuantOperandKernel kernel() {
  QuantOperandKernel k;
  k.operand = QuantOperand::kE4M3;
  k.implementation_id = quant_operand_implementation_id("mock-residual");
  k.strategy = QuantOperandStrategy::kResidualThreeProduct;
  k.implemented = k.matrix_intrinsic = k.conversion_intrinsic = true;
  k.block_absmax_scaling = k.fp32_exceptional_block_fallback = true;
  k.lds_bytes = 16384;
  return k;
}
void rejected(QuantOperandRequest r, QuantOperandKernel k,
              OperandReason expected, QuantOperandProfile p = {}) {
  auto d = select_quant_operand(r, k, p);
  require(d.operand == QuantOperand::kExisting && d.reason == expected,
          "rejected candidate selected or wrong fallback reason");
}
} // namespace
int main() {
  try {
    auto r = request();
    auto k = kernel();
    auto p = kQuantOperandProfile;
    require(!p.run_qualification_candidate, "production qualification default enabled");
    rejected(r,k,OperandReason::kUnqualified,p);
    p.run_qualification_candidate=true;
    auto d = select_quant_operand(r, k, p);
    require(
        d.operand == QuantOperand::kE4M3 &&
            d.reason == OperandReason::kSelected,
        "explicit local qualification did not select available residual E4M3");
    require(d.qualification == OperandQualification::kCandidate,
            "unvalidated candidate presented as accepted");
    for (int fault = 0; fault < 6; ++fault) {
      auto x = r;
      if (fault == 0)
        x.bits = 8;
      if (fault == 1)
        x.group = 32;
      if (fault == 2)
        x.f32_activation_output = false;
      if (fault == 3)
        x.bf16_scale_bias = false;
      if (fault == 4)
        x.indexed = true;
      if (fault == 5)
        x.caller_staged = true;
      rejected(x, k, OperandReason::kStorage);
    }
    auto x = r;
    x.arch = "gfx1100";
    rejected(x, k, OperandReason::kArchitecture);
    x = r;
    x.wave = 64;
    rejected(x, k, OperandReason::kArchitecture);
    for (auto m : {0u, 1u, 15u}) {
      x = r;
      x.m = m;
      rejected(x, k, OperandReason::kShape);
    }
    for (auto n : {0u, 1u, 15u}) {
      x = r;
      x.n = n;
      rejected(x, k, OperandReason::kShape);
    }
    for (auto n : {0u, 63u, 65u}) {
      x = r;
      x.k = n;
      rejected(x, k, OperandReason::kShape);
    }
    x = r;
    x.m = 2049;
    rejected(x, k, OperandReason::kShape);
    x = r;
    x.n = 32769;
    rejected(x, k, OperandReason::kShape);
    x = r;
    x.k = 18496;
    rejected(x, k, OperandReason::kShape);
    x = r;
    x.lds_bytes = 16383;
    rejected(x, k, OperandReason::kResources);
    for (int fault = 0; fault < 6; ++fault) {
      auto y = k;
      if (fault == 0)
        y.implemented = false;
      if (fault == 1)
        y.matrix_intrinsic = false;
      if (fault == 2)
        y.conversion_intrinsic = false;
      if (fault == 3)
        y.operand = QuantOperand::kE5M2;
      if (fault == 4)
        y.operand = QuantOperand::kExisting;
      if (fault == 5)
        y.strategy = QuantOperandStrategy::kSingleProduct;
      rejected(r, y, OperandReason::kUnavailable);
    }
    auto y = k;
    y.block_absmax_scaling = false;
    rejected(r, y, OperandReason::kUnsafeConversion);
    y = k;
    y.fp32_exceptional_block_fallback = false;
    rejected(r, y, OperandReason::kUnsafeConversion);
    p.run_qualification_candidate = false;
    rejected(r, k, OperandReason::kUnqualified, p);
    p = {};
    p.qualification = OperandQualification::kAccepted;
    rejected(r, k, OperandReason::kAccuracy, p);
    p.measured_cases = 60;
    p.ppl_scored_tokens = 1024;
    p.absolute_error_pass = p.nonfinite_pass = p.perplexity_pass =
        p.performance_pass = true;
    require(select_quant_operand(r, k, p).operand == QuantOperand::kE4M3,
            "accepted evidence rejected");
    for (int missing = 0; missing < 6; ++missing) {
      auto q = p;
      if (missing == 0)
        q.measured_cases = 0;
      if (missing == 1)
        q.ppl_scored_tokens = 1023;
      if (missing == 2)
        q.absolute_error_pass = false;
      if (missing == 3)
        q.nonfinite_pass = false;
      if (missing == 4)
        q.perplexity_pass = false;
      if (missing == 5)
        q.performance_pass = false;
      rejected(r, k, OperandReason::kAccuracy, q);
    }
    auto too_long = p;
    too_long.ppl_scored_tokens = 2049;
    rejected(r, k, OperandReason::kAccuracy, too_long);
    auto upper_bound = p;
    upper_bound.ppl_scored_tokens = 2048;
    require(select_quant_operand(r, k, upper_bound).operand == QuantOperand::kE4M3,
            "2048-token perplexity evidence rejected");
    auto historical = p;
    historical.preferred = QuantOperand::kBF16;
    historical.strategy = QuantOperandStrategy::kNative;
    historical.min_m = historical.max_m = 512;
    historical.ppl_scored_tokens = 1022;
    historical.historical_bf16_ppl_1022_accepted = true;
    auto hk = k;
    hk.operand = QuantOperand::kBF16;
    hk.strategy = QuantOperandStrategy::kNative;
    require(select_quant_operand(r, hk, historical).operand == QuantOperand::kBF16,
            "accepted historical BF16 record rejected");
    auto hr = r;
    hr.n = 5120;
    hr.k = 17408;
    require(select_quant_operand(hr, hk, historical).operand == QuantOperand::kBF16,
            "accepted historical BF16 down projection rejected");
    hr.n = 5120;
    hr.k = 5120;
    rejected(hr, hk, OperandReason::kAccuracy, historical);
    historical.ppl_scored_tokens = 1023;
    rejected(r, hk, OperandReason::kAccuracy, historical);
    historical.ppl_scored_tokens = 1022;
    historical.historical_bf16_ppl_1022_accepted = false;
    rejected(r, hk, OperandReason::kAccuracy, historical);
    // BF8 is an independently qualified strategy, not a fallback inferred from
    // exponent range or Q6's storage bit count.
    auto bf = p;
    bf.preferred = QuantOperand::kE5M2;
    y = k;
    y.operand = QuantOperand::kE5M2;
    require(select_quant_operand(r, y, bf).operand == QuantOperand::kE5M2,
            "explicit BF8 profile rejected");
    auto id = quant_operand_cache_key(42);
    require(id != quant_operand_cache_key(43), "base identity lost");
    for (int field = 0; field < 18; ++field) {
      QuantOperandProfile q;
      switch (field) {
      case 0:
        ++q.revision;
        break;
      case 1:
        q.qualification = OperandQualification::kAccepted;
        break;
      case 2:
        q.preferred = QuantOperand::kE5M2;
        break;
      case 3:
        q.strategy = QuantOperandStrategy::kSingleProduct;
        break;
      case 4:
        ++q.min_m;
        break;
      case 5:
        ++q.min_n;
        break;
      case 6:
        ++q.k_multiple;
        break;
      case 7:
        ++q.ppl_scored_tokens;
        break;
      case 8:
        ++q.measured_cases;
        break;
      case 9:
        q.absolute_error_pass = true;
        break;
      case 10:
        q.nonfinite_pass = true;
        break;
      case 11:
        q.perplexity_pass = true;
        break;
      case 12:
        q.performance_pass = true;
        break;
      case 13:
        q.run_qualification_candidate = true;
        break;
      case 14:
        ++q.max_m;
        break;
      case 15:
        ++q.max_n;
        break;
      case 16:
        ++q.max_k;
        break;
      case 17:
        q.historical_bf16_ppl_1022_accepted = true;
        break;
      }
      require(quant_operand_cache_key(42, q) != id,
              "changed profile reused cache identity");
    }
    // Costs here are mock observations: prove ranking, not hardware speed.
    std::array<QuantOperandOption, 3> options;
    for (auto &o : options) {
      o.kernel = k;
      o.profile = p;
      o.cost = {r.arch,
                r.m,
                r.n,
                r.k,
                73,
                200,
                32,
                6,
                64,
                8,
                p.revision,
                1,
                OperandCostClock::kHostEvalRetire,
                k.implementation_id};
    }
    options[0].kernel.operand = options[0].profile.preferred =
        QuantOperand::kFP32;
    options[0].kernel.strategy = options[0].profile.strategy =
        QuantOperandStrategy::kNative;
    options[0].kernel.matrix_intrinsic =
        options[0].kernel.conversion_intrinsic = false;
    options[0].kernel.lds_bytes = 0;
    options[0].cost.cost_ns = 500;
    options[1].kernel.operand = options[1].profile.preferred =
        QuantOperand::kBF16;
    options[1].kernel.strategy = options[1].profile.strategy =
        QuantOperandStrategy::kNative;
    options[1].cost.cost_ns = 100;
    auto ranked =
        rank_quant_operands(r, options, 73, OperandCostClock::kHostEvalRetire);
    require(ranked.option == 1 &&
                ranked.decision.operand == QuantOperand::kBF16,
            "slower FP8 displaced faster accepted BF16");
    const auto oldkey =
        quant_operand_choice_key(42, ranked.decision, p.revision);
    options[1].cost.cost_ns = 90;
    ranked =
        rank_quant_operands(r, options, 73, OperandCostClock::kHostEvalRetire);
    require(oldkey == quant_operand_choice_key(42, ranked.decision, p.revision),
            "new timing of unchanged strategy invalidated JIT");
    options[2].cost.cost_ns = 50;
    ranked =
        rank_quant_operands(r, options, 73, OperandCostClock::kHostEvalRetire);
    require(ranked.option == 2 &&
                ranked.decision.operand == QuantOperand::kE4M3,
            "faster accepted FP8 not selected");
    require(oldkey != quant_operand_choice_key(42, ranked.decision, p.revision),
            "changed strategy reused JIT key");
    for (int stale = 0; stale < 9; ++stale) {
      auto row = options[2];
      if (stale == 0)
        ++row.cost.m;
      if (stale == 1)
        ++row.cost.cohort;
      if (stale == 2)
        row.cost.clock = OperandCostClock::kGpuQueueInterval;
      if (stale == 3)
        ++row.cost.accepted_profile_revision;
      if (stale == 4)
        ++row.cost.implementation_revision;
      if (stale == 5)
        row.cost.samples = 2;
      if (stale == 6)
        row.profile.qualification = OperandQualification::kCandidate;
      if (stale == 7)
        ++row.cost.implementation_id;
      if (stale == 8)
        row.cost.statistic = OperandCostStatistic::kMedian;
      auto no = rank_quant_operands(r, std::span(&row, 1), 73,
                                    OperandCostClock::kHostEvalRetire);
      require(no.option == static_cast<std::size_t>(-1) &&
                  no.decision.reason == OperandReason::kCostEvidence,
              "incomparable or unaccepted performance record selected");
    }
    auto text = quant_operand_diagnostic(r, d);
    for (auto part :
         {"operand=fp8-e4m3", "residual-three-product", "512x17408x5120",
          "candidate-unvalidated", "ppl_scored_tokens=0"})
      require(text.find(part) != std::string::npos,
              "automatic diagnostic incomplete");
    std::puts(
        "PASS automatic operand policy, residual strategy, "
        "capability/shape/safety gates, evidence and cache identity; no GPU");
  } catch (const std::exception &e) {
    std::fprintf(stderr, "FAIL %s\n", e.what());
    return 1;
  }
}
