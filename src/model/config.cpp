#include "lse/model/config.hpp"

#include <fstream>
#include <sstream>
#include <utility>

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

// The HF path reads through this instead: a missing field there is an error,
// not a default, because the defaults describe a 256-wide toy model and would
// let a 4B checkpoint load into the wrong shape without a word.
template <typename T>
Status need(const nlohmann::json& j, const char* key, T& out) {
  if (!j.contains(key) || j[key].is_null()) {
    return LSE_ERROR(kNotFound, "model config is missing '", key, "'");
  }
  try {
    out = j[key].get<T>();
  } catch (const std::exception& e) {
    return LSE_ERROR(kInvalidArgument, "model config field '", key,
                     "' has the wrong type: ", e.what());
  }
  return OkStatus();
}

// Qwen3.5 (dense and MoE), read off Qwen/Qwen3.5-0.8B, Qwen/Qwen3.5-4B and
// Qwen/Qwen3.5-35B-A3B and cross-checked against
// transformers/models/qwen3_5/configuration_qwen3_5.py. Both are gated
// DeltaNet hybrids, not plain transformer stacks.
Result<Config> from_hf_json(const nlohmann::json& root) {
  const nlohmann::json& t = root["text_config"];
  Config c;

  LSE_RETURN_IF_ERROR(need(t, "vocab_size", c.vocab_size));
  LSE_RETURN_IF_ERROR(need(t, "hidden_size", c.hidden_size));
  LSE_RETURN_IF_ERROR(need(t, "num_hidden_layers", c.num_layers));
  LSE_RETURN_IF_ERROR(need(t, "rms_norm_eps", c.rms_eps));
  LSE_RETURN_IF_ERROR(need(t, "full_attention_interval", c.full_attention_interval));

  LSE_RETURN_IF_ERROR(need(t, "num_attention_heads", c.attn_q_heads));
  LSE_RETURN_IF_ERROR(need(t, "num_key_value_heads", c.attn_kv_heads));
  LSE_RETURN_IF_ERROR(need(t, "head_dim", c.attn_head_dim));

  LSE_RETURN_IF_ERROR(need(t, "linear_num_key_heads", c.gdn_qk_heads));
  LSE_RETURN_IF_ERROR(need(t, "linear_num_value_heads", c.gdn_v_heads));
  LSE_RETURN_IF_ERROR(need(t, "linear_conv_kernel_dim", c.gdn_conv_kernel));
  std::int32_t key_head_dim = 0;
  std::int32_t value_head_dim = 0;
  LSE_RETURN_IF_ERROR(need(t, "linear_key_head_dim", key_head_dim));
  LSE_RETURN_IF_ERROR(need(t, "linear_value_head_dim", value_head_dim));
  // The delta rule carries a square state, so the engine has one head dim.
  if (key_head_dim != value_head_dim) {
    return LSE_ERROR(kUnimplemented, "linear_key_head_dim (",
                     std::to_string(key_head_dim), ") and linear_value_head_dim (",
                     std::to_string(value_head_dim), ") differ; the delta rule "
                     "this engine implements carries a square state");
  }
  c.gdn_head_dim = key_head_dim;

  if (!t.contains("rope_parameters") || !t["rope_parameters"].is_object()) {
    return LSE_ERROR(kNotFound, "model config is missing 'rope_parameters'");
  }
  const nlohmann::json& rope = t["rope_parameters"];
  LSE_RETURN_IF_ERROR(need(rope, "rope_theta", c.rope_theta));
  float partial = 0.0f;
  LSE_RETURN_IF_ERROR(need(rope, "partial_rotary_factor", partial));
  const auto rotary = static_cast<std::int32_t>(
      static_cast<float>(c.attn_head_dim) * partial + 0.5f);
  if (rotary <= 0 || rotary % 2 != 0 || rotary > c.attn_head_dim) {
    return LSE_ERROR(kInvalidArgument, "partial_rotary_factor ",
                     std::to_string(partial), " of head_dim ",
                     std::to_string(c.attn_head_dim), " gives a rope width of ",
                     std::to_string(rotary), ", which must be even and positive");
  }
  c.rope_dim = rotary;

  // mRoPE splits the rotary channels across three position axes so an image
  // patch can carry a row and a column alongside its time index. Every axis of
  // a text-only prompt holds the same position, so the three sections collapse
  // onto one and plain RoPE is exactly equivalent — but only while they cover
  // the rotary half exactly, which is the part worth checking rather than
  // assuming.
  if (rope.contains("mrope_section") && rope["mrope_section"].is_array()) {
    const auto sections = rope["mrope_section"].get<std::vector<std::int32_t>>();
    std::int32_t covered = 0;
    for (std::int32_t s : sections) covered += s;
    if (covered * 2 != c.rope_dim) {
      return LSE_ERROR(kUnimplemented, "mrope_section covers ",
                       std::to_string(covered), " rotary pairs but rope_dim ",
                       std::to_string(c.rope_dim), " has ",
                       std::to_string(c.rope_dim / 2),
                       "; text-only decode is only equivalent to plain RoPE "
                       "when the sections tile the rotary half");
    }
  }

  // The stack builds one mixer per layer from full_attention_interval. A
  // checkpoint that also names layers whose mixer is replaced by a plain MLP
  // would come out with the wrong layer at every named index.
  if (t.contains("mlp_only_layers") && t["mlp_only_layers"].is_array() &&
      !t["mlp_only_layers"].empty()) {
    return LSE_ERROR(kUnimplemented, "mlp_only_layers names ",
                     std::to_string(t["mlp_only_layers"].size()),
                     " layer(s) with no mixer; the block factory builds a "
                     "mixer for every layer");
  }

  // Qwen has no sliding window; every full-attention layer sees the whole
  // prefix, which is what a zero window selects in GatedAttention::load.
  c.sliding_window = 0;
  c.global_attention_layers.clear();

  if (t.contains("num_experts")) {
    LSE_RETURN_IF_ERROR(need(t, "num_experts", c.num_experts));
    LSE_RETURN_IF_ERROR(need(t, "num_experts_per_tok", c.num_active_experts));
    LSE_RETURN_IF_ERROR(need(t, "moe_intermediate_size", c.expert_intermediate));
    LSE_RETURN_IF_ERROR(
        need(t, "shared_expert_intermediate_size", c.shared_expert_intermediate));
    c.num_shared_experts = 1;
    c.mlp_intermediate = 0;
    // Plain top-k with renormalization; there is no band and no router bias.
    c.expert_score_band = 1.0f;
  } else {
    LSE_RETURN_IF_ERROR(need(t, "intermediate_size", c.mlp_intermediate));
    c.num_experts = 0;
    c.num_active_experts = 0;
    c.num_shared_experts = 0;
    c.expert_intermediate = 0;
  }

  read(root, "tie_word_embeddings", c.tie_word_embeddings);
  read(t, "tie_word_embeddings", c.tie_word_embeddings);
  read(t, "mtp_num_hidden_layers", c.mtp_layers);
  read(t, "mtp_use_dedicated_embeddings", c.mtp_dedicated_embeddings);
  read(t, "dtype", c.dtype);
  read(t, "max_position_embeddings", c.train_seq_len);

  // layer_types is the checkpoint's own statement of which layers attend. The
  // engine derives that from full_attention_interval, so a disagreement means
  // the interval rule does not describe this model and every mixer after the
  // first divergence would be built wrong.
  if (t.contains("layer_types") && t["layer_types"].is_array()) {
    const auto types = t["layer_types"].get<std::vector<std::string>>();
    if (static_cast<std::int32_t>(types.size()) != c.num_layers) {
      return LSE_ERROR(kInvalidArgument, "layer_types has ",
                       std::to_string(types.size()), " entries but "
                       "num_hidden_layers is ", std::to_string(c.num_layers));
    }
    for (std::size_t i = 0; i < types.size(); ++i) {
      const bool attends = types[i] == "full_attention";
      if (attends != c.is_attention_layer(static_cast<std::int32_t>(i))) {
        return LSE_ERROR(kUnimplemented, "layer_types says layer ",
                         std::to_string(i), " is '", types[i],
                         "' but full_attention_interval ",
                         std::to_string(c.full_attention_interval),
                         " says otherwise");
      }
    }
  }

  LSE_RETURN_IF_ERROR(c.validate());
  return c;
}

}  // namespace

Result<Config> Config::from_json_string(const std::string& text) {
  nlohmann::json j;
  try {
    j = nlohmann::json::parse(text);
  } catch (const std::exception& e) {
    return LSE_ERROR(kInvalidArgument, "model config is not valid JSON: ", e.what());
  }

  // Read before the dispatch: the quantization block sits at the top level in
  // both spellings, and everything the two paths disagree about is shape
  // rather than storage.
  LSE_ASSIGN_OR(quant::GroupAffineMap quantization,
                quant::GroupAffineMap::from_config_json(text));

  if (j.contains("text_config") && j["text_config"].is_object()) {
    LSE_ASSIGN_OR(Config hf, from_hf_json(j));
    hf.quantization = std::move(quantization);
    return hf;
  }

  Config c;
  c.quantization = std::move(quantization);
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

  read(j, "mlp_intermediate", c.mlp_intermediate);
  read(j, "num_experts", c.num_experts);
  read(j, "num_active_experts", c.num_active_experts);
  read(j, "num_shared_experts", c.num_shared_experts);
  read(j, "expert_intermediate", c.expert_intermediate);
  read(j, "shared_expert_intermediate", c.shared_expert_intermediate);
  read(j, "tie_word_embeddings", c.tie_word_embeddings);
  read(j, "mtp_layers", c.mtp_layers);
  read(j, "mtp_dedicated_embeddings", c.mtp_dedicated_embeddings);
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
  j["mlp_intermediate"] = mlp_intermediate;
  j["num_experts"] = num_experts;
  j["num_active_experts"] = num_active_experts;
  j["num_shared_experts"] = num_shared_experts;
  j["expert_intermediate"] = expert_intermediate;
  j["shared_expert_intermediate"] = shared_expert_intermediate;
  j["tie_word_embeddings"] = tie_word_embeddings;
  j["mtp_layers"] = mtp_layers;
  j["mtp_dedicated_embeddings"] = mtp_dedicated_embeddings;
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
  if (gdn_qk_heads <= 0 || gdn_v_heads <= 0 ||
      (gdn_qk_heads % gdn_v_heads != 0 && gdn_v_heads % gdn_qk_heads != 0)) {
    // Either direction: lemonseed's layer runs v at the key width and reads
    // gdn_v_heads as a divisor of it, Qwen shares 16 key heads across 32 value
    // heads. Both are GQA over the same state, counted from opposite ends.
    return LSE_ERROR(kInvalidArgument, "gdn_qk_heads (",
                     std::to_string(gdn_qk_heads), ") and gdn_v_heads (",
                     std::to_string(gdn_v_heads),
                     ") must be positive and one must divide the other");
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
