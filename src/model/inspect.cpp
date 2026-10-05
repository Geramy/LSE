// Model inspection and memory planning without loading a model. See
// include/lse/model/inspect.hpp for the contract.
//
// Every number here follows a rule the loader or the allocator applies, and
// the comment beside it names which. When one of those rules changes, the
// change belongs here as well; tests/test_model_info.cpp compares the two.
#include "lse/model/inspect.hpp"
#include "lse/models/thinking_controls.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <span>
#include <sstream>
#include <string_view>
#include <utility>
#include <vector>

#include "lse/core/dtype.hpp"
#include "lse/dispatch/q8_matrix.hpp"
#include "lse/kv/sizing.hpp"
#include "lse/model/config.hpp"
#include "lse/model/dflash2.hpp"
#include "lse/model/dflash2_convert.hpp"
#include "lse/model/hybrid_lm.hpp"
#include "lse/model/layer.hpp"
#include "lse/model/mtp.hpp"
#include "lse/model/qwen3_5_common.hpp"
#include "lse/model/registry.hpp"
#include "lse/model/weights.hpp"
#include "lse/runtime/prefill_batch.hpp"

namespace lse::model {

namespace {

namespace fs = std::filesystem;
using json = nlohmann::json;

// --- the checkpoint, as headers ------------------------------------------------

struct Tensor {
  std::string name;
  DType dtype = DType::kCount;
  std::vector<std::int64_t> shape;

  [[nodiscard]] std::uint64_t elements() const noexcept {
    std::uint64_t n = 1;
    for (const auto d : shape) n *= static_cast<std::uint64_t>(d);
    return n;
  }
  [[nodiscard]] std::int64_t last() const noexcept {
    return shape.empty() ? 0 : shape.back();
  }
};

enum class Kind { kModel, kMtp, kDFlash2 };

const char* kind_name(Kind k) {
  switch (k) {
    case Kind::kMtp: return "mtp";
    case Kind::kDFlash2: return "dflash2_draft";
    default: return "model";
  }
}

struct Checkpoint {
  std::string input;
  ModelPaths paths;
  std::string config_text;
  json raw;
  Kind kind = Kind::kModel;
  std::map<std::string, Tensor> tensors;  // as the loader will bind them
  std::optional<SafeTensors> mapped;       // the headers, when read from disk
  std::uint64_t disk_bytes = 0;           // the weight files on disk
  // DFlash2 only.
  std::string dflash2_source;  // "quantized", "bf16" or "unsupported"
  bool dflash2_converts = false;
  std::string dflash2_cache;
  bool dflash2_cached = false;

  [[nodiscard]] const Tensor* find(const std::string& name) const {
    const auto it = tensors.find(name);
    return it == tensors.end() ? nullptr : &it->second;
  }
};

DType dtype_from_safetensors(std::string_view s) noexcept {
  if (s == "BF16") return DType::kBF16;
  if (s == "F16") return DType::kF16;
  if (s == "F32") return DType::kF32;
  if (s == "I32") return DType::kI32;
  if (s == "I8") return DType::kI8;
  if (s == "U8") return DType::kU8;
  if (s == "U32") return DType::kU32;
  return DType::kCount;
}

std::string read_text(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return {};
  std::ostringstream s;
  s << in.rdbuf();
  return s.str();
}

bool has_architecture(const json& raw, std::string_view want) {
  const auto it = raw.find("architectures");
  if (it == raw.end() || !it->is_array()) return false;
  for (const auto& a : *it)
    if (a.is_string() && a.get<std::string>() == want) return true;
  return false;
}

bool dflash2_autoconvert() {
  const char* env = std::getenv("LSE_DFLASH2_AUTOCONVERT");
  return env == nullptr || std::string_view(env) != "0";
}

Status read_headers(const ModelPaths& paths, Checkpoint& ck) {
  LSE_ASSIGN_OR(SafeTensors st, paths.weights.ends_with(".index.json")
                                    ? SafeTensors::open_sharded(paths.weights)
                                    : SafeTensors::open(paths.weights));
  for (const auto& [name, view] : st.tensors()) {
    Tensor t;
    t.name = name;
    t.dtype = view.dtype;
    for (std::size_t i = 0; i < view.shape.rank(); ++i) t.shape.push_back(view.shape.dim(i));
    ck.tensors.emplace(name, std::move(t));
  }
  ck.mapped.emplace(std::move(st));
  return OkStatus();
}

std::uint64_t weight_file_bytes(const ModelPaths& paths) {
  std::error_code ec;
  if (!paths.weights.ends_with(".index.json")) {
    const auto n = fs::file_size(paths.weights, ec);
    return ec ? 0 : n;
  }
  const auto index = read_shard_index(paths.weights);
  if (!index.ok()) return 0;
  std::uint64_t total = 0;
  const fs::path dir = fs::path(paths.weights).parent_path();
  for (const auto& shard : index->shards) {
    const auto n = fs::file_size(dir / shard, ec);
    if (!ec) total += n;
  }
  return total;
}

Result<Checkpoint> open_checkpoint(const std::string& name_or_path) {
  Checkpoint ck;
  ck.input = name_or_path;
  LSE_ASSIGN_OR(ck.paths, resolve_model(name_or_path));
  ck.config_text = read_text(ck.paths.config);
  if (ck.config_text.empty())
    return LSE_ERROR(kIoError, "cannot read the model config '", ck.paths.config, "'");
  try {
    ck.raw = json::parse(ck.config_text);
  } catch (const std::exception& e) {
    return LSE_ERROR(kInvalidArgument, "'", ck.paths.config, "' is not valid JSON: ", e.what());
  }
  ck.disk_bytes = weight_file_bytes(ck.paths);

  if (has_architecture(ck.raw, "DFlash2DraftModel")) {
    ck.kind = Kind::kDFlash2;
    const auto kind = inspect_dflash2_checkpoint(ck.paths);
    ck.dflash2_source = !kind.ok()                                  ? "unsupported"
                        : *kind == DFlash2CheckpointKind::kQuantized ? "quantized"
                        : *kind == DFlash2CheckpointKind::kBF16Source ? "bf16"
                                                                     : "unsupported";
    // prepare_dflash2_checkpoint's decision: a BF16 source is converted to Q8
    // before it is loaded, so the converted layout is what reaches the device.
    if (kind.ok() && *kind == DFlash2CheckpointKind::kBF16Source && dflash2_autoconvert()) {
      ck.dflash2_converts = true;
      if (const auto cache = dflash2_cache_path(ck.paths); cache.ok()) {
        ck.dflash2_cache = cache->string();
        std::error_code ec;
        ck.dflash2_cached = fs::exists(*cache / "model.safetensors", ec);
      }
      LSE_ASSIGN_OR(const auto layout, dflash2_q8_layout(ck.paths));
      for (const auto& p : layout) {
        Tensor t;
        t.name = p.name;
        t.dtype = dtype_from_safetensors(p.dtype);
        t.shape = p.shape;
        ck.tensors.emplace(p.name, std::move(t));
      }
      return ck;
    }
    LSE_RETURN_IF_ERROR(read_headers(ck.paths, ck));
    return ck;
  }

  LSE_RETURN_IF_ERROR(read_headers(ck.paths, ck));
  // MtpModule's own tensors: a fusion projection and the two pre-fusion norms
  // beside one decoder layer. The registry refuses such a checkpoint standalone.
  if (ck.find("fc.weight") != nullptr && ck.find("pre_fc_norm_hidden.weight") != nullptr &&
      ck.find("pre_fc_norm_embedding.weight") != nullptr)
    ck.kind = Kind::kMtp;
  return ck;
}

// --- what the loader uploads ---------------------------------------------------

// WeightBinder's device_storage: float formats and packed planes stay as
// stored, anything else is widened to f32.
DType device_storage(DType stored) noexcept {
  switch (stored) {
    case DType::kF32:
    case DType::kF16:
    case DType::kBF16:
    case DType::kU32:
      return stored;
    default:
      return DType::kF32;
  }
}

std::uint64_t device_bytes(const Tensor& t) {
  return dtype_storage_bytes(device_storage(t.dtype), static_cast<std::size_t>(t.elements()));
}

// Splits `a.b.weight` into the module and its plane, the way the binder pairs
// a packed plane with its `.scales` and `.biases`.
std::pair<std::string_view, int> module_and_plane(std::string_view name) {
  for (const auto& [suffix, plane] : {std::pair<std::string_view, int>{".weight", 0},
                                      {".scales", 1},
                                      {".biases", 2}}) {
    if (name.size() > suffix.size() && name.ends_with(suffix))
      return {name.substr(0, name.size() - suffix.size()), plane};
  }
  return {name, 3};
}

// The order a layer's loader asks for its tensors in. The slab packing below
// is first-fit, so order decides only which hole a small tensor lands in; it
// is kept faithful anyway so the slab count matches the loader's.
constexpr std::string_view kLayerOrder[] = {
    "input_layernorm", "post_attention_layernorm", "norm1", "norm2",
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.q_norm", "self_attn.k_norm",
    "self_attn.v_proj", "self_attn.o_proj",
    "linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.in_proj_a",
    "linear_attn.in_proj_b", "linear_attn.conv1d", "linear_attn.A_log",
    "linear_attn.dt_bias", "linear_attn.norm", "linear_attn.out_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "mlp.gate",
    "mlp.switch_mlp.gate_proj", "mlp.switch_mlp.up_proj", "mlp.switch_mlp.down_proj",
    "mlp.shared_expert.gate_proj", "mlp.shared_expert.up_proj",
    "mlp.shared_expert.down_proj", "mlp.shared_expert_gate",
    "attention_conv.base_kernel", "attention_conv.kernel_projection",
    "mlp_conv.base_kernel", "mlp_conv.kernel_projection"};

struct OrderKey {
  int group = 2;
  std::int64_t layer = 0;
  int rank = 0;
  std::string module;
  int plane = 0;
  auto operator<=>(const OrderKey&) const = default;
};

OrderKey order_key(const std::string& name, std::span<const std::string_view> head,
                   std::string_view block_prefix) {
  OrderKey k;
  const auto [module, plane] = module_and_plane(name);
  k.module = std::string(module);
  k.plane = plane;
  for (std::size_t i = 0; i < head.size(); ++i) {
    if (module == head[i] || name == head[i]) {
      k.group = 0;
      k.rank = static_cast<int>(i);
      return k;
    }
  }
  const std::string prefix = std::string(block_prefix) + ".";
  if (!name.starts_with(prefix)) return k;
  std::string_view rest = std::string_view(name).substr(prefix.size());
  std::int64_t layer = 0;
  const char* const stop = rest.data() + rest.size();
  const auto [end, ec] = std::from_chars(rest.data(), stop, layer);
  if (ec != std::errc{} || end == rest.data() || end == stop || *end != '.') return k;
  k.group = 1;
  k.layer = layer;
  const std::size_t at = prefix.size() + static_cast<std::size_t>(end - rest.data()) + 1;
  const std::string_view local = at <= module.size() ? module.substr(at) : std::string_view{};
  k.rank = static_cast<int>(std::size(kLayerOrder));
  for (std::size_t i = 0; i < std::size(kLayerOrder); ++i)
    if (local == kLayerOrder[i]) k.rank = static_cast<int>(i);
  return k;
}

// A tensor the loader reads with require_rows or require_columns. Those never
// take the packed Q8 copy, which bind_quantized makes only for a whole matrix.
bool row_gathered(std::string_view name) {
  for (std::string_view s : {".self_attn.q_proj.weight", ".self_attn.k_proj.weight",
                             ".self_attn.v_proj.weight", ".linear_attn.in_proj_qkv.weight",
                             ".linear_attn.in_proj_z.weight", ".linear_attn.in_proj_a.weight",
                             ".linear_attn.in_proj_b.weight"})
    if (name.ends_with(s)) return true;
  return false;
}

// slab_window, replayed: a tensor takes its size rounded up to the alignment
// out of the first slab with room, and a tensor larger than a slab gets a
// slab of exactly its own rounded size.
struct SlabPlan {
  std::vector<std::pair<std::uint64_t, std::uint64_t>> slabs;  // capacity, used

  void place(std::uint64_t bytes) {
    const std::uint64_t need = (bytes + kWeightAlignment - 1) & ~std::uint64_t{kWeightAlignment - 1};
    for (auto& [capacity, used] : slabs) {
      if (used + need <= capacity) {
        used += need;
        return;
      }
    }
    slabs.emplace_back(std::max<std::uint64_t>(need, kWeightSlabBytes), need);
  }
  [[nodiscard]] std::uint64_t reserved() const noexcept {
    std::uint64_t total = 0;
    for (const auto& [capacity, used] : slabs) total += capacity;
    return total;
  }
};

struct WeightLoad {
  std::uint64_t tensor_bytes = 0;     // sum of uploaded tensors
  std::uint64_t packed_q8_bytes = 0;  // the packed copies, when the device takes them
  std::uint64_t packed_q8_matrices = 0;
  std::uint64_t derived_bytes = 0;    // small buffers built from weights at load
  std::uint64_t refused_bytes = 0;
  std::vector<json> refused;
};

// Uploads one checkpoint's tensors into `slabs` in load order. `refused` are
// prefixes the architecture deliberately leaves on disk (HybridLMSpec::refused).
WeightLoad plan_weights(const Checkpoint& ck, const quant::GroupAffineMap* quant,
                        std::span<const std::string_view> head,
                        std::string_view block_prefix,
                        const std::vector<HybridLMSpec::Refusal>& refused,
                        bool pack_q8, SlabPlan& slabs) {
  WeightLoad out;
  std::vector<std::pair<OrderKey, const Tensor*>> order;
  std::vector<std::uint64_t> refused_count(refused.size(), 0), refused_bytes(refused.size(), 0);
  for (const auto& [name, t] : ck.tensors) {
    bool skip = false;
    for (std::size_t i = 0; i < refused.size(); ++i) {
      if (!name.starts_with(refused[i].prefix)) continue;
      ++refused_count[i];
      refused_bytes[i] += dtype_storage_bytes(t.dtype, static_cast<std::size_t>(t.elements()));
      skip = true;
      break;
    }
    if (!skip) order.emplace_back(order_key(name, head, block_prefix), &t);
  }
  for (std::size_t i = 0; i < refused.size(); ++i) {
    if (refused_count[i] == 0) continue;
    out.refused_bytes += refused_bytes[i];
    out.refused.push_back({{"prefix", refused[i].prefix},
                           {"tensors", refused_count[i]},
                           {"bytes", refused_bytes[i]},
                           {"reason", refused[i].reason}});
  }
  std::sort(order.begin(), order.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });
  for (const auto& [key, t] : order) {
    const std::uint64_t bytes = device_bytes(*t);
    out.tensor_bytes += bytes;
    slabs.place(bytes);
    // bind_quantized's packed copy: allocated beside the slabs, per matrix.
    if (!pack_q8 || quant == nullptr || key.plane != 0 || t->dtype != DType::kU32 ||
        t->shape.size() != 2 || row_gathered(t->name))
      continue;
    const Tensor* scales = ck.find(key.module + ".scales");
    const Tensor* biases = ck.find(key.module + ".biases");
    if (scales == nullptr || biases == nullptr) continue;
    const auto spec = quant->resolve_checked(t->name, t->last(), scales->last());
    if (!spec.ok()) continue;
    const std::int64_t rows = t->shape[0];
    const std::int64_t features = t->last() * 32 / spec->bits;
    if (!packs_q8_matrix(*spec, 2, scales->dtype, rows, features)) continue;
    out.packed_q8_bytes += packed_q8_matrix_bytes(rows, features);
    ++out.packed_q8_matrices;
  }
  return out;
}

// --- model facts ----------------------------------------------------------------

struct Arch {
  std::string name;  // the registry's name, empty when nothing matched
  std::string reason;
  bool qwen = false;
};

Arch detect(const Checkpoint& ck, const Config& config) {
  Arch a;
  if (!ck.mapped) {
    a.reason = "the checkpoint's tensors were not read";
    return a;
  }
  const auto arch = detect_architecture(config, *ck.mapped);
  if (!arch.ok()) {
    a.reason = arch.status().to_string();
    return a;
  }
  a.name = std::string((*arch)->name);
  a.qwen = a.name.starts_with("qwen3.5");
  return a;
}

HybridLMSpec spec_for(const Arch& arch, const Config& config) {
  if (arch.qwen) return qwen3_5::lm_spec(config);
  return HybridLMSpec{};  // lemonseed: default names, tied head
}

std::vector<std::string_view> head_order(const HybridLMSpec& spec) {
  std::vector<std::string_view> head{spec.embed_name, spec.final_norm_name};
  if (!spec.lm_head_name.empty()) head.push_back(spec.lm_head_name);
  return head;
}

// GDN state geometry, the way HybridLM::hidden allocates it.
struct Recurrent {
  std::int32_t layers = 0;
  std::int64_t heads = 0, dim = 0, tail = 0, conv_width = 0;
  [[nodiscard]] std::uint64_t bytes_per_sequence() const noexcept {
    // [B, heads, dim, dim] f32 plus the conv tails [B, kernel-1, width] f32.
    return static_cast<std::uint64_t>(layers) * 4u *
           static_cast<std::uint64_t>(heads * dim * dim + tail * conv_width);
  }
};

Recurrent recurrent_of(const Config& c, const HybridLMSpec& spec) {
  Recurrent r;
  for (std::int32_t i = 0; i < c.num_layers; ++i)
    if (!c.is_attention_layer(i)) ++r.layers;
  r.heads = spec.gdn_state_heads > 0 ? spec.gdn_state_heads : c.gdn_qk_heads;
  r.dim = spec.gdn_state_dim > 0 ? spec.gdn_state_dim : c.gdn_head_dim;
  r.tail = c.gdn_conv_kernel > 1 ? c.gdn_conv_kernel - 1 : 0;
  // One fused q|k|v tail, or three separate ones at the state width.
  r.conv_width = spec.gdn_conv_width > 0 ? spec.gdn_conv_width : 3 * r.heads * r.dim;
  return r;
}

std::int32_t kv_layer_count(const Config& c) {
  std::int32_t n = 0;
  for (std::int32_t i = 0; i < c.num_layers; ++i)
    if (c.is_attention_layer(i)) ++n;
  return n;
}

bool is_moe(const Config& c) { return c.num_experts > 0 && c.mlp_intermediate == 0; }

json quantization_json(const Checkpoint& ck, const quant::GroupAffineMap& q) {
  if (!q.has_global() && q.override_count() == 0) return nullptr;
  std::string mode = "affine";
  for (const char* key : {"quantization", "quantization_config"}) {
    const auto it = ck.raw.find(key);
    if (it != ck.raw.end() && it->is_object() && it->contains("mode") && (*it)["mode"].is_string()) {
      mode = (*it)["mode"].get<std::string>();
      break;
    }
  }
  json j{{"mode", mode}, {"overrides", q.override_count()}};
  if (q.has_global()) {
    j["bits"] = q.global().bits;
    j["group_size"] = q.global().group_size;
  } else {
    j["bits"] = nullptr;
    j["group_size"] = nullptr;
  }
  return j;
}

constexpr std::array kCacheFormats{kv::CacheDType::kF32, kv::CacheDType::kF16,
                                   kv::CacheDType::kBF16, kv::CacheDType::kFP8,
                                   kv::CacheDType::kBF8};

json kv_formats_json(std::int32_t layers, std::int64_t kv_heads, std::int64_t head_dim,
                     bool mtp) {
  json list = json::array();
  for (const auto f : kCacheFormats) {
    const std::uint64_t per_layer = kv::token_bytes(f, kv_heads, head_dim);
    json e{{"name", std::string(kv::to_string(f))},
           {"supported", per_layer > 0},
           {"bytes_per_token", per_layer * static_cast<std::uint64_t>(layers)},
           {"bytes_per_block",
            2 * kv::block_plane_bytes(f, kv_heads, head_dim) * static_cast<std::uint64_t>(layers)}};
    // The MTP module's one attention layer shares the target's format and length.
    e["mtp_bytes_per_token"] = mtp ? json(per_layer) : json(nullptr);
    list.push_back(std::move(e));
  }
  return list;
}

json kv_allocation_json() {
  return {{"block_tokens", kv::kBlockSize},
          {"contiguous",
           "per layer, K and V pools of whole blocks; a pool starts at 8 blocks, doubles to "
           "2048 blocks, then grows 256 blocks at a time, and never exceeds kv_len"},
          {"fragmented",
           "per layer, K and V backed by 256 KiB fragments covering the blocks in use, drawn "
           "from 256 MiB arenas shared by every layer"},
          {"fragment_bytes", kv::kFragmentBytes},
          {"arena_bytes", kv::kArenaBytes}};
}

// --- memory plan ---------------------------------------------------------------

// Elements of f32 activation one token keeps live inside one block during a
// wide pass: the residual stream and its norms, the widest mixer's
// projections, and the feed-forward. Slots are recycled across layers, so one
// block's worth is what a pass holds, not one per layer.
std::uint64_t block_floats(const Config& c, std::int32_t kv_layers, const Recurrent& r,
                           bool qwen) {
  const std::uint64_t d = static_cast<std::uint64_t>(c.hidden_size);
  const std::uint64_t qh = static_cast<std::uint64_t>(c.attn_q_heads);
  const std::uint64_t kvh = static_cast<std::uint64_t>(c.attn_kv_heads);
  const std::uint64_t hd = static_cast<std::uint64_t>(c.attn_head_dim);
  std::uint64_t attention = 0;
  if (kv_layers > 0) {
    // fused q|gate, k, v, the normed and rotated q/k, attention out, gated, o_proj
    attention = 2 * qh * hd + 2 * kvh * hd + qh * hd + kvh * hd + 2 * qh * hd + d;
  }
  std::uint64_t gdn = 0;
  if (r.layers > 0) {
    const std::uint64_t kh = static_cast<std::uint64_t>(c.gdn_qk_heads);
    const std::uint64_t vh = static_cast<std::uint64_t>(qwen ? c.gdn_v_heads : c.gdn_qk_heads);
    const std::uint64_t ghd = static_cast<std::uint64_t>(c.gdn_head_dim);
    const std::uint64_t conv = 2 * kh * ghd + vh * ghd;
    const std::uint64_t chunk = static_cast<std::uint64_t>(c.gdn_chunk_size > 0 ? c.gdn_chunk_size : 64);
    // projection, conv output, gate, decay inputs, delta-rule output and its
    // gated norm, out_proj, and the per-chunk [C, C] and [C, D] intermediates.
    gdn = 2 * conv + 3 * vh * ghd + 2 * vh + d + vh * (chunk + 2 * ghd);
  }
  std::uint64_t ffn = 0;
  if (is_moe(c)) {
    const std::uint64_t e = static_cast<std::uint64_t>(c.num_experts);
    const std::uint64_t active = static_cast<std::uint64_t>(c.num_active_experts);
    const std::uint64_t inter = static_cast<std::uint64_t>(c.expert_intermediate);
    const std::uint64_t shared = static_cast<std::uint64_t>(
        c.shared_expert_intermediate > 0 ? c.shared_expert_intermediate : c.expert_intermediate);
    ffn = e + active * 3 * inter +
          static_cast<std::uint64_t>(c.num_shared_experts > 0 ? 1 : 0) * 3 * shared + d;
  } else {
    ffn = 3 * static_cast<std::uint64_t>(c.mlp_intermediate) + d;
  }
  return 4 * d + std::max(attention, gdn) + ffn;
}

// Retained programs keep their slots for replay, and a slot is recycled only
// by a later cut that wants exactly its byte count, so a pass holds more than
// one block's activation set. The factor is calibrated, in halves, against
// live backend allocations recorded on an R9700 (Qwen3.8-27B Q4 with a Q8
// DFlash2 draft, ubatch 1024, a 14,000-token request): 2.26 GB beyond weights
// and KV, against 2.28 GB estimated here with a factor of 2.5.
constexpr std::uint64_t kActivationHalves = 5;

std::uint64_t scaled(std::uint64_t bytes) { return bytes * kActivationHalves / 2; }

// Device memory the runtime and its allocators reserve beyond the engine's own
// allocations: kernel code, command buffers and pool slack. Driver counters on
// the same R9700 runs read 1.8-2.2 GB above the engine's allocations, so 2 GiB
// is held back before deciding what fits.
constexpr std::uint64_t kDeviceReserveBytes = std::uint64_t{2} << 30;

struct DraftPlan {
  DraftKind kind = DraftKind::kNone;
  std::string path;
  std::optional<Checkpoint> ck;
  std::optional<Config> mtp_config;        // the module's declared config
  std::optional<DFlash2Config> dflash2;
  std::string note;
};

struct Static {
  // Fixed by the checkpoints and the device, not by the context length.
  Config config;
  Arch arch;
  HybridLMSpec spec;
  Recurrent recurrent;
  std::int32_t kv_layers = 0;
  WeightLoad target, draft;
  std::uint64_t target_slab_bytes = 0, total_slab_bytes = 0, slabs = 0;
  std::uint64_t draft_derived = 0;
  bool packing = false;
  std::string packing_label;
};

json estimate_at(const MemoryPlanRequest& req, const Static& s, const DraftPlan& draft,
                 kv::CacheDType format, std::int32_t kv_len, std::int32_t context) {
  const Config& c = s.config;
  const auto rows_bucket = batch_bucket(std::max(1, req.sequences));
  const std::int32_t rows = rows_bucket.ok() ? *rows_bucket : 1;
  const std::int64_t kvh = c.attn_kv_heads, hd = c.attn_head_dim;
  const bool mtp = draft.kind == DraftKind::kMtp && draft.ck.has_value();
  const bool dflash = draft.kind == DraftKind::kDFlash2 && draft.dflash2.has_value();

  // KV: the paged allocator's rules (ops::ensure_paged_kv), per layer, both planes.
  std::uint64_t kv_target = 0, kv_mtp = 0, kv_index = 0, growth = 0, fragments = 0;
  const std::int32_t mtp_layers = mtp ? 1 : 0;
  for (std::int32_t layer = 0; layer < s.kv_layers + mtp_layers; ++layer) {
    std::uint64_t bytes = 0;
    if (req.fragmented_kv) {
      const auto n = kv::fragmented_layer_fragments(format, kvh, hd, context, rows);
      fragments += n;
      bytes = n * kv::kFragmentBytes;
      kv_index += 2 * kv::fragment_table_bytes(format, kvh, hd, kv_len, rows);
    } else {
      bytes = kv::contiguous_layer_bytes(format, kvh, hd, context, kv_len, rows);
      growth = std::max<std::uint64_t>(
          growth, kv::contiguous_growth_bytes(format, kvh, hd, context, kv_len, rows));
    }
    kv_index += kv::block_table_bytes(context, kv_len, rows);
    (layer < s.kv_layers ? kv_target : kv_mtp) += bytes;
  }
  std::uint64_t arena_slack = 0;
  if (req.fragmented_kv) {
    const std::uint64_t used = fragments * kv::kFragmentBytes;
    arena_slack = kv::arena_bytes(fragments) - used;
  }
  const std::uint64_t per_token =
      kv::token_bytes(format, kvh, hd) * static_cast<std::uint64_t>(s.kv_layers);

  // RoPE tables, cos and sin [positions, rope_dim] f32 (ops::build_rope). The
  // MTP module shares the target's; DFlash2 builds its own over its head width.
  const std::uint64_t rope_target =
      2ull * static_cast<std::uint64_t>(kv_len) * static_cast<std::uint64_t>(c.rope_dim) * 4ull;

  // Recurrent state: one live copy, plus the carry outputs of the retained
  // decode program and, with a draft, of the verifier.
  const bool speculative = mtp || dflash;
  const std::uint64_t recurrent_copy =
      s.recurrent.bytes_per_sequence() * static_cast<std::uint64_t>(rows);
  const std::uint64_t recurrent_copies = 2 + (speculative ? 1 : 0);
  const std::uint64_t recurrent = recurrent_copy * recurrent_copies;

  const std::uint64_t vocab = static_cast<std::uint64_t>(c.vocab_size);
  const std::uint64_t d = static_cast<std::uint64_t>(c.hidden_size);
  const std::uint64_t per_token_block = block_floats(c, s.kv_layers, s.recurrent, s.arch.qwen) * 4;
  const std::uint64_t ubatch = req.ubatch_size;
  const std::uint64_t verify_rows =
      mtp ? req.mtp_depth + 1 : dflash ? draft.dflash2->block_size : 0;

  // Prefill: one chunk of ubatch tokens at a time.
  std::uint64_t activation = scaled(ubatch * per_token_block) + vocab * 4;
  if (dflash) {
    const auto& dc = *draft.dflash2;
    const std::uint64_t taps = dc.target_layers.size();
    // The captured target features, the draft's input slot for them, its fc
    // output and norm, and the K/V it projects for each draft layer.
    activation += ubatch * 4 *
                  (2 * taps * d + 2 * d +
                   static_cast<std::uint64_t>(dc.num_layers) * 2 *
                       static_cast<std::uint64_t>(dc.kv_heads) * static_cast<std::uint64_t>(dc.head_dim));
  }
  if (mtp) {
    // The module's catch-up pass over the chunk: the fused input and one block.
    activation += ubatch * 4 * (3 * d) + ubatch * per_token_block;
  }

  // Resident programs: decode, the verifier and the draft passes, with the
  // logits each reads back.
  std::uint64_t programs = scaled((1 + verify_rows) * per_token_block) +
                           (1 + verify_rows) * vocab * 4;
  if (dflash) {
    const auto& dc = *draft.dflash2;
    programs += static_cast<std::uint64_t>(dc.block_size) * vocab * 4 +
                scaled(static_cast<std::uint64_t>(dc.block_size) * 4 *
                       (4 * d + 3 * static_cast<std::uint64_t>(dc.intermediate_size))) +
                verify_rows * dc.target_layers.size() * d * 4;
  }
  if (mtp) programs += 2 * vocab * 4 + scaled((verify_rows + 1) * per_token_block);
  const std::uint64_t workspace = recurrent + rope_target + programs;

  // Draft: weights placed after the target's, its own KV or ring, its RoPE.
  std::uint64_t draft_weights = 0, draft_cache = 0, draft_rope = 0;
  if (speculative) {
    draft_weights = (s.total_slab_bytes - s.target_slab_bytes) +
                    (s.packing ? s.draft.packed_q8_bytes : 0) + s.draft_derived;
  }
  if (mtp) draft_cache = kv_mtp;
  if (dflash) {
    const auto& dc = *draft.dflash2;
    // DFlash2Module::Impl::load: K and V rings [1, kv_heads, window-1+block,
    // head_dim] f32 per draft layer, and build_rope over head_dim.
    const std::uint64_t capacity =
        static_cast<std::uint64_t>(dc.sliding_window) - 1 + dc.block_size;
    draft_cache = static_cast<std::uint64_t>(dc.num_layers) * 2 *
                  static_cast<std::uint64_t>(dc.kv_heads) * capacity *
                  static_cast<std::uint64_t>(dc.head_dim) * 4;
    draft_rope = 2ull * (static_cast<std::uint64_t>(kv_len) + dc.block_size - 1) *
                 static_cast<std::uint64_t>(dc.head_dim) * 4ull;
  }
  const std::uint64_t draft_total = draft_weights + draft_cache + draft_rope;

  const std::uint64_t weights_total = s.target_slab_bytes +
                                      (s.packing ? s.target.packed_q8_bytes : 0) +
                                      s.target.derived_bytes;
  const std::uint64_t kv_total = kv_target + kv_index + arena_slack;
  const std::uint64_t resident = weights_total + kv_total + workspace + draft_total;
  const std::uint64_t total = resident + activation + growth;

  json w{{"tensor_bytes", s.target.tensor_bytes},
         {"slab_bytes", s.target_slab_bytes},
         {"packed_q8_bytes", s.packing ? s.target.packed_q8_bytes : 0},
         {"derived_bytes", s.target.derived_bytes},
         {"unloaded_bytes", s.target.refused_bytes},
         {"total_bytes", weights_total}};
  json kv{{"format", std::string(kv::to_string(format))},
          {"storage", req.fragmented_kv ? "fragmented" : "contiguous"},
          {"layers", s.kv_layers},
          {"sequences", rows},
          {"tokens", context},
          {"capacity_tokens", kv_len},
          {"bytes_per_token", per_token},
          {"pool_bytes", kv_target},
          {"index_bytes", kv_index},
          {"arena_slack_bytes", arena_slack},
          {"total_bytes", kv_total}};
  if (!req.fragmented_kv) kv["growth_transient_bytes"] = growth;
  json dj{{"kind", draft.kind == DraftKind::kMtp ? "mtp"
                   : draft.kind == DraftKind::kDFlash2 ? "dflash2" : "none"},
          {"path", draft.path.empty() ? json(nullptr) : json(draft.path)},
          {"weights_bytes", draft_weights},
          {"total_bytes", draft_total}};
  if (mtp) {
    dj["kv_bytes"] = draft_cache;
    dj["depth"] = req.mtp_depth;
  }
  if (dflash) {
    dj["ring_bytes"] = draft_cache;
    dj["rope_bytes"] = draft_rope;
    dj["block_size"] = draft.dflash2->block_size;
  }
  if (!draft.note.empty()) dj["note"] = draft.note;
  return json{
      {"weights", std::move(w)},
      {"kv", std::move(kv)},
      {"workspace", {{"recurrent_state_bytes", recurrent},
                     {"recurrent_state_copies", recurrent_copies},
                     {"rope_bytes", rope_target},
                     {"programs_bytes", programs},
                     {"total_bytes", workspace}}},
      {"activation", {{"prefill_bytes", activation},
                      {"ubatch_tokens", ubatch},
                      {"approximate", true}}},
      {"draft", std::move(dj)},
      {"weights_bytes", weights_total},
      {"kv_bytes", kv_total},
      {"workspace_bytes", workspace},
      {"activation_bytes", activation},
      {"draft_bytes", draft_total},
      {"resident_bytes", resident},
      {"total_bytes", total},
      {"device_reserve_bytes", kDeviceReserveBytes},
      {"device_total_bytes", total + kDeviceReserveBytes}};
}

Result<Config> target_config(const Checkpoint& ck) {
  if (ck.kind != Kind::kModel)
    return LSE_ERROR(kInvalidArgument, "'", ck.input, "' is ",
                     ck.kind == Kind::kMtp ? "an MTP module" : "a DFlash2 draft",
                     ", not a model to serve");
  return Config::from_json_file(ck.paths.config);
}

}  // namespace

// --- model_info ----------------------------------------------------------------

Result<nlohmann::json> model_info(const std::string& name_or_path) {
  LSE_ASSIGN_OR(Checkpoint ck, open_checkpoint(name_or_path));
  json out{{"path", name_or_path},
           {"config_path", ck.paths.config},
           {"weights_path", ck.paths.weights},
           {"kind", kind_name(ck.kind)},
           {"weights_disk_bytes", ck.disk_bytes}};
  if (const auto it = ck.raw.find("model_type"); it != ck.raw.end() && it->is_string())
    out["model_type"] = *it;
  else
    out["model_type"] = nullptr;
  out["architectures"] = ck.raw.contains("architectures") ? ck.raw["architectures"] : json::array();

  const bool pack = true;  // a planning default: the packed copies a gfx1201 adds

  if (ck.kind == Kind::kDFlash2) {
    auto parsed = DFlash2Config::from_json_string(
        ck.dflash2_converts
            ? [&] {
                json converted = ck.raw;
                converted["quantization"] = {{"bits", 8}, {"group_size", 64}, {"mode", "affine"}};
                return converted.dump();
              }()
            : ck.config_text);
    out["loadable"] = parsed.ok() && ck.dflash2_source != "unsupported";
    out["reason"] = parsed.ok() ? (ck.dflash2_source == "unsupported"
                                       ? json("an unquantized DFlash2 checkpoint that is not a "
                                              "single BF16 file cannot be converted")
                                       : json(nullptr))
                                : json(parsed.status().to_string());
    out["architecture"] = "dflash2";
    out["family"] = "dflash2";
    if (!parsed.ok()) return out;
    const DFlash2Config& dc = *parsed;
    out["moe"] = false;
    out["layers"] = dc.num_layers;
    out["hidden_size"] = dc.hidden_size;
    out["attention_heads"] = dc.q_heads;
    out["kv_heads"] = dc.kv_heads;
    out["head_dim"] = dc.head_dim;
    out["vocab_size"] = dc.vocab_size;
    out["intermediate_size"] = dc.intermediate_size;
    out["max_position_embeddings"] =
        ck.raw.value("max_position_embeddings", json(nullptr));
    out["quantization"] = quantization_json(ck, dc.quantization);
    if (ck.dflash2_converts) out["quantization"] = {{"mode", "affine"}, {"bits", 8}, {"group_size", 64}, {"overrides", 0}};
    SlabPlan slabs;
    const std::array<std::string_view, 6> head{"fc", "hidden_norm", "norm",
                                               "candidate_selector.hidden_projection",
                                               "candidate_selector.predecessor_codebook",
                                               "candidate_selector.successor_codebook"};
    const auto load = plan_weights(ck, &dc.quantization, head, "layers", {}, pack, slabs);
    out["weights_bytes"] = load.tensor_bytes;
    out["weights_vram_bytes"] = slabs.reserved() + load.packed_q8_bytes;
    out["packed_q8_bytes"] = load.packed_q8_bytes;
    const std::uint64_t capacity = static_cast<std::uint64_t>(dc.sliding_window) - 1 + dc.block_size;
    out["dflash2"] = {
        {"draft", true},
        {"hidden_size", dc.hidden_size},
        {"vocab_size", dc.vocab_size},
        {"num_target_layers", dc.target_num_layers},
        {"target_layer_ids", dc.target_layers},
        {"block_size", dc.block_size},
        {"sliding_window", dc.sliding_window},
        {"ring_bytes", static_cast<std::uint64_t>(dc.num_layers) * 2 *
                           static_cast<std::uint64_t>(dc.kv_heads) * capacity *
                           static_cast<std::uint64_t>(dc.head_dim) * 4},
        {"source", ck.dflash2_source},
        {"conversion", {{"needed", ck.dflash2_converts},
                        {"cache_path", ck.dflash2_cache.empty() ? json(nullptr) : json(ck.dflash2_cache)},
                        {"cached", ck.dflash2_cached}}}};
    out["kv_cache_dtypes"] = json::array();
    out["kv_note"] = "a DFlash2 draft keeps a fixed FP32 ring instead of a paged KV cache";
    return out;
  }

  LSE_ASSIGN_OR(const Config config, Config::from_json_file(ck.paths.config));
  const std::int32_t kv_layers = kv_layer_count(config);
  out["max_position_embeddings"] = config.train_seq_len;
  out["layers"] = config.num_layers;
  out["hidden_size"] = config.hidden_size;
  out["attention_heads"] = config.attn_q_heads;
  out["kv_heads"] = config.attn_kv_heads;
  out["head_dim"] = config.attn_head_dim;
  out["vocab_size"] = config.vocab_size;
  out["dtype"] = config.dtype;
  out["quantization"] = quantization_json(ck, config.quantization);
  out["moe"] = is_moe(config);
  out["intermediate_size"] = is_moe(config) ? json(nullptr) : json(config.mlp_intermediate);
  if (is_moe(config)) {
    out["experts"] = {{"count", config.num_experts},
                      {"active", config.num_active_experts},
                      {"intermediate_size", config.expert_intermediate},
                      {"shared_intermediate_size", config.shared_expert_intermediate > 0
                                                       ? config.shared_expert_intermediate
                                                       : config.expert_intermediate}};
  }

  if (ck.kind == Kind::kMtp) {
    // An MTP module: one full-attention layer hung off its target's embedding
    // and head. Its config repeats the target's, so the geometry above is the
    // target's and the module itself is one layer.
    out["loadable"] = true;
    out["reason"] = nullptr;
    out["architecture"] = "mtp";
    out["family"] = "mtp";
    out["layers"] = 1;
    out["kv_layers"] = json::array({0});
    out["kv_layer_count"] = 1;
    out["mtp"] = {{"layers", 1},
                  {"default_depth", ck.raw.value("block_size", 3)},
                  {"max_depth", 7}};
    SlabPlan slabs;
    const std::array<std::string_view, 4> head{"fc", "pre_fc_norm_hidden",
                                               "pre_fc_norm_embedding", "norm"};
    const auto load = plan_weights(ck, &config.quantization, head, "layers", {}, pack, slabs);
    out["weights_bytes"] = load.tensor_bytes;
    out["weights_vram_bytes"] = slabs.reserved() + load.packed_q8_bytes;
    out["packed_q8_bytes"] = load.packed_q8_bytes;
    out["kv_cache_dtype_default"] = std::string(kv::to_string(config.kv_cache_dtype));
    out["kv_block_tokens"] = kv::kBlockSize;
    out["kv_cache_dtypes"] = kv_formats_json(1, config.attn_kv_heads, config.attn_head_dim, false);
    return out;
  }

  const Arch arch = detect(ck, config);
  const CacheModel verdict = inspect_model_dir(
      fs::path(ck.paths.weights).parent_path().string(), name_or_path);
  out["architecture"] = arch.name.empty() ? json(nullptr) : json(arch.name);
  out["family"] = arch.qwen ? "qwen3.5" : arch.name.empty() ? json(nullptr) : json(arch.name);
  out["loadable"] = !arch.name.empty() && verdict.loadable == Loadable::kYes;
  out["reason"] = !arch.name.empty() ? (verdict.loadable == Loadable::kYes ? json(nullptr)
                                                                           : json(verdict.reason))
                                     : json(arch.reason);
  out["parameters"] = verdict.parameters;

  json kinds = json::array(), kv_list = json::array();
  for (std::int32_t i = 0; i < config.num_layers; ++i) {
    const bool attends = config.is_attention_layer(i);
    kinds.push_back(attends ? "full_attention" : "linear_attention");
    if (attends) kv_list.push_back(i);
  }
  out["layer_kinds"] = std::move(kinds);
  out["kv_layers"] = std::move(kv_list);
  out["kv_layer_count"] = kv_layers;
  const HybridLMSpec spec = spec_for(arch, config);
  const Recurrent rec = recurrent_of(config, spec);
  out["linear_attention"] = {{"layers", rec.layers},
                             {"key_heads", config.gdn_qk_heads},
                             {"value_heads", arch.qwen ? config.gdn_v_heads : config.gdn_qk_heads},
                             {"head_dim", config.gdn_head_dim},
                             {"conv_kernel", config.gdn_conv_kernel},
                             {"state_bytes_per_sequence", rec.bytes_per_sequence()}};
  out["default_kv_len"] = config.kv_capacity();
  out["max_context_length"] = config.train_seq_len;

  SlabPlan slabs;
  const auto head = head_order(spec);
  const auto load = plan_weights(ck, &config.quantization, head, spec.block_prefix,
                                 spec.refused, pack, slabs);
  out["weights_bytes"] = load.tensor_bytes;
  out["weights_vram_bytes"] = slabs.reserved() + load.packed_q8_bytes;
  out["packed_q8_bytes"] = load.packed_q8_bytes;
  out["unloaded_bytes"] = load.refused_bytes;
  out["unloaded"] = load.refused;

  const std::string beside = MtpModule::find_beside(name_or_path);
  out["mtp"] = {{"layers", config.mtp_layers},
                {"present", !beside.empty()},
                {"module_path", beside.empty() ? json(nullptr) : json(beside)},
                {"default_depth", 3},
                {"max_depth", 7}};
  out["dflash2"] = {{"draft", false},
                    {"compatibility", {{"hidden_size", config.hidden_size},
                                       {"vocab_size", config.vocab_size},
                                       {"num_target_layers", config.num_layers}}}};
  out["kv_cache_dtype_default"] = std::string(kv::to_string(config.kv_cache_dtype));
  out["kv_block_tokens"] = kv::kBlockSize;
  out["kv_cache_dtypes"] = kv_formats_json(kv_layers, config.attn_kv_heads,
                                           config.attn_head_dim, config.mtp_layers > 0);
  out["kv_allocation"] = kv_allocation_json();
  // What the checkpoint's own files say about generating: sampling defaults
  // and any output limit (generation_config.json, config.json), and the
  // thinking levels its chat template defines.
  out["generation_defaults"] = config.sampling_defaults.to_json();
  {
    auto thinking = models::load_thinking_controls(fs::path(ck.paths.config).parent_path().string());
    out["thinking"] = thinking.ok() ? thinking->to_json()
                                    : json{{"supported", false}, {"error", std::string(thinking.status().message())}};
  }
  return out;
}

// --- estimate_memory --------------------------------------------------------------

Result<nlohmann::json> estimate_memory(const MemoryPlanRequest& req) {
  if (req.model.empty()) return LSE_ERROR(kInvalidArgument, "no model to estimate");
  if (req.kv_len < 0 || req.context_tokens < 0 || req.sequences < 1)
    return LSE_ERROR(kInvalidArgument, "kv_len and context_tokens must be >= 0 and sequences >= 1");
  if (!runtime::PrefillBatch::valid_size(req.batch_size) ||
      !runtime::PrefillBatch::valid_size(req.ubatch_size) || req.ubatch_size > req.batch_size)
    return LSE_ERROR(kInvalidArgument,
                     "batch sizes must be powers of two from 128 to 4096 with ubatch <= batch");
  if (req.mtp_depth < 1 || req.mtp_depth > 7)
    return LSE_ERROR(kInvalidArgument, "MTP depth must be from 1 to 7");
  if (!batch_bucket(req.sequences).ok())
    return LSE_ERROR(kInvalidArgument, "at most 32 sequences share one pass");

  LSE_ASSIGN_OR(Checkpoint ck, open_checkpoint(req.model));
  Static s;
  LSE_ASSIGN_OR(s.config, target_config(ck));
  s.arch = detect(ck, s.config);
  if (s.arch.name.empty())
    return LSE_ERROR(kInvalidArgument, "no architecture in this build loads '", req.model,
                     "': ", s.arch.reason);
  s.spec = spec_for(s.arch, s.config);
  s.recurrent = recurrent_of(s.config, s.spec);
  s.kv_layers = kv_layer_count(s.config);

  const kv::CacheDType format = req.kv_cache_dtype.value_or(s.config.kv_cache_dtype);
  if (kv::storage_width(format, s.config.attn_head_dim) <= 0)
    return LSE_ERROR(kInvalidArgument, "KV format ", std::string(kv::to_string(format)),
                     " cannot store a head width of ", std::to_string(s.config.attn_head_dim));
  const std::int32_t kv_len = req.kv_len > 0 ? req.kv_len : s.config.kv_capacity();
  const std::int32_t context =
      req.context_tokens > 0 ? std::min(req.context_tokens, kv_len) : kv_len;

  // The draft, resolved the way lse_open resolves it.
  DraftPlan draft;
  draft.kind = req.draft;
  if (req.draft == DraftKind::kMtp) {
    draft.path = req.draft_path.empty() ? MtpModule::find_beside(req.model) : req.draft_path;
    if (draft.path.empty()) {
      draft.kind = DraftKind::kNone;
      draft.note = "no MTP module was found beside the model; none is loaded";
    } else {
      auto opened = open_checkpoint(draft.path);
      Status why = opened.ok() ? OkStatus() : opened.status();
      if (opened.ok() && opened->kind != Kind::kMtp)
        why = LSE_ERROR(kInvalidArgument, "'", draft.path, "' is not an MTP module");
      std::optional<Config> declared;
      if (why.ok()) {
        auto parsed = Config::from_json_string(opened->config_text);
        if (!parsed.ok()) why = parsed.status();
        else if (parsed->hidden_size != s.config.hidden_size ||
                 parsed->vocab_size != s.config.vocab_size ||
                 parsed->attn_kv_heads != s.config.attn_kv_heads ||
                 parsed->attn_head_dim != s.config.attn_head_dim)
          why = LSE_ERROR(kInvalidArgument, "the MTP module's geometry differs from the model's");
        else declared = parsed.release();
      }
      if (!why.ok()) {
        // A module named explicitly must load; one merely found beside the
        // model is skipped, as lse_open skips it.
        if (!req.draft_path.empty()) return why;
        draft.kind = DraftKind::kNone;
        draft.note = "the MTP module beside the model would not load: " + why.to_string();
      } else {
        draft.ck.emplace(opened.release());
        draft.mtp_config = std::move(declared);
      }
    }
  } else if (req.draft == DraftKind::kDFlash2) {
    if (req.draft_path.empty())
      return LSE_ERROR(kInvalidArgument, "a DFlash2 estimate needs the draft checkpoint");
    draft.path = req.draft_path;
    LSE_ASSIGN_OR(Checkpoint dck, open_checkpoint(req.draft_path));
    if (dck.kind != Kind::kDFlash2)
      return LSE_ERROR(kInvalidArgument, "'", req.draft_path, "' is not a DFlash2 draft");
    json cfg = dck.raw;
    if (dck.dflash2_converts) cfg["quantization"] = {{"bits", 8}, {"group_size", 64}, {"mode", "affine"}};
    LSE_ASSIGN_OR(DFlash2Config dc, DFlash2Config::from_json_string(cfg.dump()));
    LSE_RETURN_IF_ERROR(dc.validate(s.config));
    draft.dflash2 = std::move(dc);
    draft.ck.emplace(std::move(dck));
  }

  // Weights: target first, then the draft, into one set of slabs.
  s.packing = req.device_arch.empty() || dispatch::q8_packed_weight_arch(req.device_arch);
  s.packing_label = req.device_arch.empty() ? "assumed" : s.packing ? "on" : "off";
  SlabPlan slabs;
  const auto head = head_order(s.spec);
  s.target = plan_weights(ck, &s.config.quantization, head, s.spec.block_prefix,
                          s.spec.refused, s.packing, slabs);
  // Built from weights at load, outside the slabs: each GDN layer's prepared
  // decay rate (ops::prepare_gated_delta_rate), [value heads] f32.
  if (s.arch.qwen) s.target.derived_bytes += static_cast<std::uint64_t>(s.recurrent.layers) *
                                             static_cast<std::uint64_t>(s.config.gdn_v_heads) * 4;
  s.target_slab_bytes = slabs.reserved();
  if (draft.kind == DraftKind::kMtp && draft.ck) {
    const std::array<std::string_view, 4> mtp_head{"fc", "pre_fc_norm_hidden",
                                                   "pre_fc_norm_embedding", "norm"};
    s.draft = plan_weights(*draft.ck, &draft.mtp_config->quantization, mtp_head, "layers", {},
                           s.packing, slabs);
  } else if (draft.kind == DraftKind::kDFlash2 && draft.ck) {
    const std::array<std::string_view, 6> dhead{"fc", "hidden_norm", "norm",
                                                "candidate_selector.hidden_projection",
                                                "candidate_selector.predecessor_codebook",
                                                "candidate_selector.successor_codebook"};
    s.draft = plan_weights(*draft.ck, &draft.dflash2->quantization, dhead, "layers", {},
                           s.packing, slabs);
    // The two-tap conv base kernels are widened to f32 once at load.
    s.draft_derived = static_cast<std::uint64_t>(draft.dflash2->num_layers) * 2 * 4 *
                      static_cast<std::uint64_t>(draft.dflash2->hidden_size) * 4;
  }
  s.total_slab_bytes = slabs.reserved();
  s.slabs = slabs.slabs.size();

  json out = estimate_at(req, s, draft, format, kv_len, context);
  out["model"] = req.model;
  out["architecture"] = s.arch.name;
  out["settings"] = {{"kv_cache_dtype", std::string(kv::to_string(format))},
                     {"kv_len", kv_len},
                     {"context_tokens", context},
                     {"batch_size", req.batch_size},
                     {"ubatch_size", req.ubatch_size},
                     {"sequences", req.sequences},
                     {"kv_storage", req.fragmented_kv ? "fragmented" : "contiguous"},
                     {"device_arch", req.device_arch.empty() ? json(nullptr) : json(req.device_arch)},
                     {"q8_packing", s.packing_label}};
  out["weights"]["slabs"] = s.slabs;
  out["weights"]["packed_q8_matrices"] = s.target.packed_q8_matrices;
  out["method"] =
      "weights replay the loader's slab packing; KV, recurrent state, RoPE tables and the "
      "DFlash2 ring follow the allocator's rules exactly; activation and program workspace "
      "are modelled from the layer shapes and calibrated against measured R9700 runs";

  if (req.device_memory_bytes > 0) {
    // The largest kv_len whose full pool fits, by bisection: every component
    // grows with kv_len or holds still.
    const auto fits = [&](std::int32_t len) {
      const json e = estimate_at(req, s, draft, format, len, len);
      return e["device_total_bytes"].get<std::uint64_t>() <= req.device_memory_bytes;
    };
    std::int32_t lo = 0;
    std::int32_t hi = std::max<std::int32_t>(s.config.train_seq_len, kv_len);
    hi = std::min<std::int32_t>(hi, 1 << 22);
    if (fits(hi)) {
      lo = hi;
    } else {
      while (hi - lo > kv::kBlockSize) {
        const std::int32_t mid = lo + (hi - lo) / 2;
        if (mid > 0 && fits(mid)) lo = mid;
        else hi = mid;
      }
      lo = lo / kv::kBlockSize * kv::kBlockSize;
    }
    out["device_memory_bytes"] = req.device_memory_bytes;
    out["fits"] = out["device_total_bytes"].get<std::uint64_t>() <= req.device_memory_bytes;
    out["max_kv_len"] = lo;
  }
  return out;
}

}  // namespace lse::model
