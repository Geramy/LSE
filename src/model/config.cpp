#include "lse/model/config.hpp"

#include <fstream>
#include <sstream>

#include <nlohmann/json.hpp>

// The padded vocab is a property of the tokenizer, so the constant lives there
// and the presets follow it.
#include "lse/tokenizer/tokenizer.hpp"

namespace lse::model {

namespace {

template <typename T>
void read(const nlohmann::json& j, const char* key, T& out) {
  if (j.contains(key) && !j[key].is_null()) out = j[key].get<T>();
}

}  // namespace

Result<Config> Config::from_json_string(const std::string& text) {
  nlohmann::json j;
  try {
    j = nlohmann::json::parse(text);
  } catch (const std::exception& e) {
    return LSE_ERROR(kInvalidArgument, "model config is not valid JSON: ", e.what());
  }

  Config c;
  read(j, "vocab_size", c.vocab_size);
  read(j, "hidden_size", c.hidden_size);
  read(j, "num_layers", c.num_layers);
  read(j, "full_attention_interval", c.full_attention_interval);
  read(j, "global_attention_layers", c.global_attention_layers);
  read(j, "sliding_window", c.sliding_window);

  read(j, "attn_q_heads", c.attn_q_heads);
  read(j, "attn_kv_heads", c.attn_kv_heads);
  read(j, "attn_head_dim", c.attn_head_dim);
  read(j, "rope_dim", c.rope_dim);
  read(j, "rope_theta", c.rope_theta);

  read(j, "gdn_qk_heads", c.gdn_qk_heads);
  read(j, "gdn_v_heads", c.gdn_v_heads);
  read(j, "gdn_head_dim", c.gdn_head_dim);
  read(j, "gdn_conv_kernel", c.gdn_conv_kernel);
  read(j, "gdn_chunk_size", c.gdn_chunk_size);

  read(j, "num_experts", c.num_experts);
  read(j, "num_active_experts", c.num_active_experts);
  read(j, "num_shared_experts", c.num_shared_experts);
  read(j, "expert_intermediate", c.expert_intermediate);
  read(j, "router_aux_loss_coef", c.router_aux_loss_coef);
  read(j, "router_z_loss_coef", c.router_z_loss_coef);
  read(j, "expert_score_band", c.expert_score_band);

  read(j, "use_mod", c.use_mod);
  read(j, "mod_top_k", c.mod_top_k);
  read(j, "mod_threshold", c.mod_threshold);
  read(j, "mod_aux_loss_coef", c.mod_aux_loss_coef);

  read(j, "train_seq_len", c.train_seq_len);
  read(j, "kv_length", c.kv_length);
  read(j, "rms_eps", c.rms_eps);
  read(j, "dtype", c.dtype);
  read(j, "grad_checkpoint_segment", c.grad_checkpoint_segment);

  LSE_RETURN_IF_ERROR(c.validate());
  return c;
}

Result<Config> Config::from_json_file(const std::string& path) {
  std::ifstream in(path);
  if (!in) return LSE_ERROR(kIoError, "cannot open model config '", path, "'");
  std::ostringstream buf;
  buf << in.rdbuf();
  return from_json_string(buf.str());
}

std::string Config::to_json() const {
  nlohmann::json j;
  j["vocab_size"] = vocab_size;
  j["hidden_size"] = hidden_size;
  j["num_layers"] = num_layers;
  j["full_attention_interval"] = full_attention_interval;
  j["global_attention_layers"] = global_attention_layers;
  j["sliding_window"] = sliding_window;
  j["attn_q_heads"] = attn_q_heads;
  j["attn_kv_heads"] = attn_kv_heads;
  j["attn_head_dim"] = attn_head_dim;
  j["rope_dim"] = rope_dim;
  j["rope_theta"] = rope_theta;
  j["gdn_qk_heads"] = gdn_qk_heads;
  j["gdn_v_heads"] = gdn_v_heads;
  j["gdn_head_dim"] = gdn_head_dim;
  j["gdn_conv_kernel"] = gdn_conv_kernel;
  j["gdn_chunk_size"] = gdn_chunk_size;
  j["num_experts"] = num_experts;
  j["num_active_experts"] = num_active_experts;
  j["num_shared_experts"] = num_shared_experts;
  j["expert_intermediate"] = expert_intermediate;
  j["router_aux_loss_coef"] = router_aux_loss_coef;
  j["router_z_loss_coef"] = router_z_loss_coef;
  j["expert_score_band"] = expert_score_band;
  j["use_mod"] = use_mod;
  j["mod_top_k"] = mod_top_k;
  j["mod_threshold"] = mod_threshold;
  j["mod_aux_loss_coef"] = mod_aux_loss_coef;
  j["train_seq_len"] = train_seq_len;
  j["kv_length"] = kv_length;
  j["rms_eps"] = rms_eps;
  j["dtype"] = dtype;
  j["grad_checkpoint_segment"] = grad_checkpoint_segment;
  return j.dump(2);
}

Status Config::validate() const {
  if (vocab_size <= 0 || hidden_size <= 0 || num_layers <= 0) {
    return LSE_ERROR(kInvalidArgument, "vocab_size, hidden_size and num_layers "
                                       "must all be positive");
  }
  if (full_attention_interval <= 0) {
    return LSE_ERROR(kInvalidArgument, "full_attention_interval must be positive");
  }
  if (attn_q_heads % attn_kv_heads != 0) {
    return LSE_ERROR(kInvalidArgument, "GQA requires attn_q_heads (",
                     std::to_string(attn_q_heads), ") to be a multiple of "
                     "attn_kv_heads (", std::to_string(attn_kv_heads), ")");
  }
  if (gdn_qk_heads % gdn_v_heads != 0) {
    return LSE_ERROR(kInvalidArgument, "gdn_qk_heads must be a multiple of gdn_v_heads");
  }
  if (num_active_experts > num_experts) {
    return LSE_ERROR(kInvalidArgument, "num_active_experts exceeds num_experts");
  }
  if (dtype_from_string(dtype) == DType::kCount) {
    return LSE_ERROR(kInvalidArgument, "unknown dtype '", dtype, "'");
  }
  if (kv_length < 0) {
    return LSE_ERROR(kInvalidArgument, "kv_length must be >= 0");
  }
  for (std::int32_t l : global_attention_layers) {
    if (l < 0 || l >= num_layers) {
      return LSE_ERROR(kOutOfRange, "global_attention_layers contains ",
                       std::to_string(l), " but num_layers is ",
                       std::to_string(num_layers));
    }
  }
  return OkStatus();
}

std::int64_t Config::param_count() const noexcept {
  const std::int64_t d = hidden_size;
  const std::int64_t tied_embed = static_cast<std::int64_t>(vocab_size) * d;

  std::int64_t per_attn = 0;
  {
    const std::int64_t q = static_cast<std::int64_t>(attn_q_heads) * attn_head_dim;
    const std::int64_t kv = static_cast<std::int64_t>(attn_kv_heads) * attn_head_dim;
    per_attn = d * q + 2 * d * kv + q * d;  // q, k, v, out
    per_attn += d * q;                      // output gate
  }

  std::int64_t per_gdn = 0;
  {
    const std::int64_t qk = static_cast<std::int64_t>(gdn_qk_heads) * gdn_head_dim;
    const std::int64_t v = static_cast<std::int64_t>(gdn_v_heads) * gdn_head_dim;
    per_gdn = d * qk * 2 + d * v + v * d;              // in_proj q/k/v, out_proj
    per_gdn += d * v;                                   // out gate
    per_gdn += 2 * d * gdn_qk_heads;                    // in_proj a/b
    per_gdn += (qk * 2 + v) * (gdn_conv_kernel + 1);    // depthwise conv + bias
    per_gdn += 2 * gdn_qk_heads + gdn_head_dim;         // A_log, dt_bias, norm
  }

  const std::int64_t routed =
      static_cast<std::int64_t>(num_experts) * 3 * d * expert_intermediate;
  const std::int64_t shared =
      static_cast<std::int64_t>(num_shared_experts) * 3 * d * expert_intermediate;
  const std::int64_t per_moe = routed + shared + d * num_experts + num_experts;
  const std::int64_t per_mod = d + 1;
  const std::int64_t per_norm = 2 * d;

  std::int64_t total = tied_embed + d;  // embedding + final norm
  for (std::int32_t i = 0; i < num_layers; ++i) {
    total += (is_attention_layer(i) ? per_attn : per_gdn) + per_moe + per_mod + per_norm;
  }
  return total;
}

Config preset_tiny() { return Config{}; }

Config preset_b1_5() {
  Config c;
  c.vocab_size = tokenizer::kQwen36PaddedVocabSize;
  c.hidden_size = 1024;
  c.num_layers = 20;
  c.full_attention_interval = 4;
  c.global_attention_layers = {3, 7, 11, 15, 19};
  c.sliding_window = 256;
  c.attn_q_heads = 16;
  c.attn_kv_heads = 2;
  c.attn_head_dim = 64;
  c.rope_dim = 64;
  c.rope_theta = 500000.0f;
  c.gdn_qk_heads = 8;
  c.gdn_v_heads = 4;
  c.gdn_head_dim = 64;
  c.gdn_conv_kernel = 4;
  // 32, not 64: the delta rule's (I - beta K K^T)^-1 is numerically delicate and
  // 64 NaNs in bf16 despite being ~4% faster.
  c.gdn_chunk_size = 32;
  c.num_experts = 8;
  c.num_active_experts = 2;
  c.num_shared_experts = 1;
  c.expert_intermediate = 2176;
  c.train_seq_len = 2048;
  return c;
}

Config preset_scaled() {
  Config c;
  c.vocab_size = tokenizer::kQwen36PaddedVocabSize;
  c.hidden_size = 2048;
  c.num_layers = 28;
  c.full_attention_interval = 4;
  c.global_attention_layers = {7, 15, 23, 27};
  c.sliding_window = 512;
  c.attn_q_heads = 16;
  c.attn_kv_heads = 4;
  c.attn_head_dim = 128;
  c.rope_dim = 128;
  c.rope_theta = 500000.0f;
  c.gdn_qk_heads = 16;
  c.gdn_v_heads = 8;
  c.gdn_head_dim = 128;
  c.gdn_conv_kernel = 4;
  c.num_experts = 64;
  c.num_active_experts = 4;
  c.num_shared_experts = 1;
  c.expert_intermediate = 1024;
  c.train_seq_len = 4096;
  return c;
}

Result<Config> preset_by_name(const std::string& name) {
  if (name == "tiny") return preset_tiny();
  if (name == "b1.5" || name == "1.5b" || name == "b1_5") return preset_b1_5();
  if (name == "scaled" || name == "large") return preset_scaled();
  return LSE_ERROR(kNotFound, "unknown preset '", name,
                   "'; choose from: tiny, b1.5, scaled");
}

}  // namespace lse::model
