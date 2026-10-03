#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "lse/core/status.hpp"
#include "lse/graph/graph.hpp"
#include "lse/model/config.hpp"
#include "lse/model/hybrid_lm.hpp"
#include "lse/runtime/sampler.hpp"

namespace lse::model {

struct DFlash2Config {
  std::int32_t hidden_size = 0, vocab_size = 0, num_layers = 0;
  std::int32_t target_num_layers = 0, q_heads = 0, kv_heads = 0, head_dim = 0;
  std::int32_t intermediate_size = 0, sliding_window = 0;
  std::int32_t conv_group_size = 0, selector_rank = 0, selector_top_k = 0;
  std::uint32_t block_size = 0, mask_token = 0;
  float rms_eps = 0, rope_theta = 0;
  std::vector<std::int32_t> target_layers;
  quant::GroupAffineMap quantization;

  static Result<DFlash2Config> from_json_string(const std::string& text);
  Status validate(const Config& target) const;
};

// The first score matrix is conditioned on the anchor. Later matrices have
// one predecessor row per candidate of the previous position.
Result<std::vector<std::uint32_t>> dflash2_select_path(
    std::span<const float> scores, std::span<const std::uint32_t> candidates,
    std::uint32_t positions, std::uint32_t top_k);

struct DFlash2Proposal {
  std::vector<std::uint32_t> tokens;
  std::vector<runtime::DiscreteDistribution> conditionals;
};

Result<DFlash2Proposal> dflash2_sample_path(
    std::span<const float> scores, std::span<const std::uint32_t> candidates,
    std::uint32_t positions, std::uint32_t top_k, std::uint32_t vocab_size,
    float temperature, runtime::SpeculativeSampler& sampler);

// hidden [B,T,D], dynamic [B,T,2,D/group], base [2,D].
graph::Array dflash2_convolve(const graph::Array& hidden,
                              const graph::Array& dynamic,
                              const graph::Array& base,
                              std::int32_t group_size);

class DFlash2Module {
 public:
  static Result<std::unique_ptr<DFlash2Module>> open(
      const std::string& path, const Config& parent, HybridLM& model);
  ~DFlash2Module();

  [[nodiscard]] std::span<const std::int32_t> target_layers() const noexcept;
  [[nodiscard]] std::uint32_t block_size() const noexcept;
  [[nodiscard]] std::int32_t context_position() const noexcept;

  // Features concatenate the selected target block outputs in target_layers()
  // order: [1,S,taps*D], F32. Only verified context belongs in this cache.
  Status append_context(const graph::Array& features, std::int32_t first);
  Status rewind(std::int32_t position);
  void reset();
  // reset(), and the draft programs too: the workspace the module holds
  // between requests, given back when no session needs it.
  void release_programs();
  Status retire_prefill();

  Result<std::vector<std::uint32_t>> draft(std::uint32_t anchor,
                                          std::int32_t first,
                                          std::uint32_t proposals);

  Result<DFlash2Proposal> draft_sampled(std::uint32_t anchor,
      std::int32_t first, std::uint32_t proposals, float temperature,
      runtime::SpeculativeSampler& sampler);

 private:
  struct Impl;
  explicit DFlash2Module(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace lse::model
