// Field names match the JSON keys lemonseed's save_model() writes beside the
// .safetensors, so its checkpoints load without conversion.
// Source of truth: reference/lemonseed/lemonseed/config.py @ 6fb97de
//
// A HuggingFace config.json shares none of those names and nests the decoder
// under "text_config", so from_json_string dispatches on that key and reads
// the HF spelling instead. Every field the HF path needs is required there:
// silently keeping a default would hand the builder a model of the wrong
// shape that still passes validate().
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "lse/core/dtype.hpp"
#include "lse/core/status.hpp"

namespace lse::model {

struct Config {
  std::int32_t vocab_size = 4096;
  std::int32_t hidden_size = 256;
  std::int32_t num_layers = 8;

  std::int32_t full_attention_interval = 4;
  std::vector<std::int32_t> global_attention_layers{7};
  std::int32_t sliding_window = 64;

  // Attention (GQA + RoPE)
  std::int32_t attn_q_heads = 4;
  std::int32_t attn_kv_heads = 1;
  std::int32_t attn_head_dim = 64;
  std::int32_t rope_dim = 64;
  float rope_theta = 10000.0f;

  // Gated DeltaNet
  std::int32_t gdn_qk_heads = 4;
  std::int32_t gdn_v_heads = 2;
  std::int32_t gdn_head_dim = 32;
  std::int32_t gdn_conv_kernel = 4;
  std::int32_t gdn_chunk_size = 64;

  // Dense feed-forward width. 0 means every layer's FFN is routed.
  std::int32_t mlp_intermediate = 0;

  // MoE
  std::int32_t num_experts = 8;
  std::int32_t num_active_experts = 2;  // upper cap, see expert_score_band
  std::int32_t num_shared_experts = 1;
  std::int32_t expert_intermediate = 256;
  // 0 means the shared expert is as wide as a routed one, which is lemonseed's
  // layout; Qwen sizes the two independently.
  std::int32_t shared_expert_intermediate = 0;
  float router_aux_loss_coef = 0.01f;
  float router_z_loss_coef = 0.001f;
  // Keep experts within this fraction of the top expert's probability.
  // 1.0 == plain top-k.
  float expert_score_band = 0.15f;

  bool use_mod = true;             // checkpoint compat only; MoD is always on
  std::int32_t mod_top_k = 3;      // capacity floor
  float mod_threshold = 0.15f;     // capacity ratio
  float mod_aux_loss_coef = 0.01f;

  // Mirrors LEMONSEED_MOE_BIAS_BALANCE: the expert_bias in the checkpoint only
  // steers routing when this is on. Default off, matching the reference.
  bool moe_bias_balance = false;

  // False means the checkpoint carries a separate lm_head.
  bool tie_word_embeddings = true;

  // Training-side
  std::int32_t train_seq_len = 128;
  float rms_eps = 1e-6f;
  std::string dtype = "bfloat16";
  std::int32_t grad_checkpoint_segment = 1;

  // Engine KV capacity in tokens. 0 means kv_capacity() picks a default from
  // train_seq_len. Attention allocates [B, kv_heads, capacity, head_dim] once
  // and advances a write position; the tensor never grows.
  std::int32_t kv_length = 0;

  [[nodiscard]] std::int32_t kv_capacity() const noexcept {
    if (kv_length > 0) return kv_length;
    return train_seq_len * 2 > 2048 ? train_seq_len * 2 : 2048;
  }

  [[nodiscard]] bool is_attention_layer(std::int32_t layer_idx) const noexcept {
    return (layer_idx + 1) % full_attention_interval == 0;
  }

  [[nodiscard]] bool is_global_attention(std::int32_t layer_idx) const noexcept {
    for (std::int32_t l : global_attention_layers) {
      if (l == layer_idx) return true;
    }
    return false;
  }

  [[nodiscard]] std::int64_t param_count() const noexcept;

  static Result<Config> from_json_file(const std::string& path);
  static Result<Config> from_json_string(const std::string& json);
  [[nodiscard]] std::string to_json() const;

  [[nodiscard]] Status validate() const;
};

Config preset_tiny();
Config preset_b1_5();   // ~1.5B pretrain target
Config preset_scaled();
Result<Config> preset_by_name(const std::string& name);

}  // namespace lse::model
