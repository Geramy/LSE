// Model inspection and memory planning (lse/model/inspect.hpp, lse_model_info
// and lse_estimate) against small fixture checkpoints written here: a dense
// Qwen3.5, a MoE Qwen3.5 and a BF16 DFlash2 draft. Beyond the reported facts,
// the plan is checked against what the engine actually allocates on the host
// backend: the weights a real load binds, the KV pools the paged allocator
// grows, and the fragments the Loom K/V manager hands out.
#include "harness.hpp"

#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "lse/backend/backend.hpp"
#include "lse/backends/cpu/cpu_backend.hpp"
#include "lse/graph/graph.hpp"
#include "lse/kv/memory.hpp"
#include "lse/kv/sizing.hpp"
#include "lse/lse.h"
#include "lse/model/config.hpp"
#include "lse/model/dflash2.hpp"
#include "lse/model/inspect.hpp"
#include "lse/model/registry.hpp"
#include "lse/model/weights.hpp"
#include "lse/ops/attention.hpp"
#include "lse/place/devices.hpp"

using namespace lse;

#define LSE_EXPECT_STR(a, b)                                                   \
  do {                                                                         \
    const std::string _sa = (a), _sb = (b);                                   \
    if (_sa != _sb) ::lse::test::fail(__FILE__, __LINE__, _sa + " vs " + _sb); \
  } while (0)
using json = nlohmann::json;
namespace fs = std::filesystem;

namespace {

struct TensorSpec {
  std::string name;
  std::string dtype;  // safetensors spelling
  std::vector<std::int64_t> shape;
};

std::size_t elem_bytes(const std::string& dtype) {
  return dtype == "BF16" || dtype == "F16" ? 2 : 4;
}

std::uint64_t bytes_of(const TensorSpec& t) {
  std::uint64_t n = elem_bytes(t.dtype);
  for (auto d : t.shape) n *= static_cast<std::uint64_t>(d);
  return n;
}

// Zero-filled safetensors; zeros are a valid value for every tensor here.
void write_safetensors(const fs::path& path, const std::vector<TensorSpec>& tensors) {
  json header = json::object();
  std::uint64_t offset = 0;
  for (const auto& t : tensors) {
    const auto n = bytes_of(t);
    header[t.name] = {{"dtype", t.dtype}, {"shape", t.shape}, {"data_offsets", {offset, offset + n}}};
    offset += n;
  }
  std::string text = header.dump();
  text.append((8 - text.size() % 8) % 8, ' ');
  std::ofstream out(path, std::ios::binary);
  const std::uint64_t len = text.size();
  out.write(reinterpret_cast<const char*>(&len), 8);
  out.write(text.data(), static_cast<std::streamsize>(text.size()));
  const std::vector<char> zeros(1 << 16, 0);
  for (std::uint64_t left = offset; left > 0;) {
    const auto n = static_cast<std::streamsize>(std::min<std::uint64_t>(left, zeros.size()));
    out.write(zeros.data(), n);
    left -= static_cast<std::uint64_t>(n);
  }
}

void write_text(const fs::path& path, const std::string& text) {
  std::ofstream out(path);
  out << text;
}

struct TempDir {
  fs::path path;
  explicit TempDir(const char* tag) {
    path = fs::temp_directory_path() /
           (std::string("lse-model-info-") + tag + "-" + std::to_string(getpid()));
    fs::remove_all(path);
    fs::create_directories(path);
  }
  ~TempDir() {
    std::error_code ec;
    fs::remove_all(path, ec);
  }
};

// Shared widths of the Qwen fixtures: hidden 64, 4 layers (3 GDN + 1
// attention), 2 query heads and 1 KV head of 64, GDN 2 key / 4 value heads of
// 32, conv kernel 4, vocab 64.
constexpr std::int64_t kD = 64, kVocab = 64, kInter = 128;
constexpr std::int64_t kQH = 2, kKVH = 1, kHD = 64;
constexpr std::int64_t kGK = 2, kGV = 4, kGHD = 32, kConv = 2 * kGK * kGHD + kGV * kGHD;

json qwen_text_config(bool moe) {
  json t{{"vocab_size", kVocab},
         {"hidden_size", kD},
         {"num_hidden_layers", 4},
         {"rms_norm_eps", 1e-6},
         {"full_attention_interval", 4},
         {"num_attention_heads", kQH},
         {"num_key_value_heads", kKVH},
         {"head_dim", kHD},
         {"linear_num_key_heads", kGK},
         {"linear_num_value_heads", kGV},
         {"linear_conv_kernel_dim", 4},
         {"linear_key_head_dim", kGHD},
         {"linear_value_head_dim", kGHD},
         {"rope_parameters", {{"rope_theta", 1e6}, {"partial_rotary_factor", 0.25}}},
         {"max_position_embeddings", 2048},
         {"tie_word_embeddings", false},
         {"mtp_num_hidden_layers", 1},
         {"dtype", "bfloat16"},
         {"layer_types", {"linear_attention", "linear_attention", "linear_attention",
                          "full_attention"}}};
  if (moe) {
    t["num_experts"] = 4;
    t["num_experts_per_tok"] = 2;
    t["moe_intermediate_size"] = 64;
    t["shared_expert_intermediate_size"] = 64;
  } else {
    t["intermediate_size"] = kInter;
  }
  return t;
}

// A Q4 g64 plane triple for an [rows, cols] matrix.
void q4(std::vector<TensorSpec>& out, const std::string& module, std::int64_t rows,
        std::int64_t cols, std::vector<std::int64_t> lead = {}) {
  auto shape = [&](std::int64_t last) {
    auto s = lead;
    s.push_back(rows);
    s.push_back(last);
    return s;
  };
  out.push_back({module + ".weight", "U32", shape(cols * 4 / 32)});
  out.push_back({module + ".scales", "BF16", shape(cols / 64)});
  out.push_back({module + ".biases", "BF16", shape(cols / 64)});
}

std::vector<TensorSpec> qwen_tensors(bool moe) {
  std::vector<TensorSpec> t;
  const std::string m = "language_model.model.";
  t.push_back({m + "embed_tokens.weight", "BF16", {kVocab, kD}});
  t.push_back({m + "norm.weight", "BF16", {kD}});
  t.push_back({"language_model.lm_head.weight", "BF16", {kVocab, kD}});
  for (int i = 0; i < 4; ++i) {
    const std::string l = m + "layers." + std::to_string(i) + ".";
    t.push_back({l + "input_layernorm.weight", "BF16", {kD}});
    t.push_back({l + "post_attention_layernorm.weight", "BF16", {kD}});
    if (i == 3) {
      t.push_back({l + "self_attn.q_proj.weight", "BF16", {2 * kQH * kHD, kD}});
      t.push_back({l + "self_attn.k_proj.weight", "BF16", {kKVH * kHD, kD}});
      t.push_back({l + "self_attn.v_proj.weight", "BF16", {kKVH * kHD, kD}});
      t.push_back({l + "self_attn.o_proj.weight", "BF16", {kD, kQH * kHD}});
      t.push_back({l + "self_attn.q_norm.weight", "BF16", {kHD}});
      t.push_back({l + "self_attn.k_norm.weight", "BF16", {kHD}});
    } else {
      t.push_back({l + "linear_attn.in_proj_qkv.weight", "BF16", {kConv, kD}});
      t.push_back({l + "linear_attn.in_proj_z.weight", "BF16", {kGV * kGHD, kD}});
      t.push_back({l + "linear_attn.in_proj_a.weight", "BF16", {kGV, kD}});
      t.push_back({l + "linear_attn.in_proj_b.weight", "BF16", {kGV, kD}});
      t.push_back({l + "linear_attn.conv1d.weight", "BF16", {kConv, 4, 1}});
      t.push_back({l + "linear_attn.A_log", "F32", {kGV}});
      t.push_back({l + "linear_attn.dt_bias", "BF16", {kGV}});
      t.push_back({l + "linear_attn.norm.weight", "F32", {kGHD}});
      t.push_back({l + "linear_attn.out_proj.weight", "BF16", {kD, kGV * kGHD}});
    }
    if (moe) {
      t.push_back({l + "mlp.gate.weight", "BF16", {4, kD}});
      q4(t, l + "mlp.switch_mlp.gate_proj", 64, kD, {4});
      q4(t, l + "mlp.switch_mlp.up_proj", 64, kD, {4});
      q4(t, l + "mlp.switch_mlp.down_proj", kD, 64, {4});
      t.push_back({l + "mlp.shared_expert.gate_proj.weight", "BF16", {64, kD}});
      t.push_back({l + "mlp.shared_expert.up_proj.weight", "BF16", {64, kD}});
      t.push_back({l + "mlp.shared_expert.down_proj.weight", "BF16", {kD, 64}});
      t.push_back({l + "mlp.shared_expert_gate.weight", "BF16", {1, kD}});
    } else {
      q4(t, l + "mlp.gate_proj", kInter, kD);
      q4(t, l + "mlp.up_proj", kInter, kD);
      q4(t, l + "mlp.down_proj", kD, kInter);
    }
  }
  // The vision tower the text-only build leaves on disk.
  t.push_back({"vision_tower.patch_embed.proj.weight", "BF16", {16, 48}});
  return t;
}

void write_qwen(const fs::path& dir, bool moe) {
  json config{{"architectures", {moe ? "Qwen3_5MoeForConditionalGeneration"
                                     : "Qwen3_5ForConditionalGeneration"}},
              {"model_type", moe ? "qwen3_5_moe" : "qwen3_5"},
              {"text_config", qwen_text_config(moe)},
              {"vision_config", json::object()},
              {"quantization", {{"group_size", 64}, {"bits", 4}, {"mode", "affine"}}}};
  write_text(dir / "config.json", config.dump(2));
  write_safetensors(dir / "model.safetensors", qwen_tensors(moe));
}

// A BF16 DFlash2 draft for the dense fixture: 1 layer, 2 query heads and 1 KV
// head of 32, two feature taps, block 4, selector rank 64.
std::vector<TensorSpec> dflash2_tensors() {
  std::vector<TensorSpec> t{
      {"fc.weight", "BF16", {kD, 2 * kD}},
      {"hidden_norm.weight", "BF16", {kD}},
      {"norm.weight", "BF16", {kD}},
      {"candidate_selector.hidden_projection.weight", "BF16", {64, kD}},
      {"candidate_selector.predecessor_codebook", "BF16", {kVocab, 64}},
      {"candidate_selector.successor_codebook", "BF16", {kVocab, 64}},
      {"layers.0.input_layernorm.weight", "BF16", {kD}},
      {"layers.0.post_attention_layernorm.weight", "BF16", {kD}},
      {"layers.0.self_attn.q_proj.weight", "BF16", {64, kD}},
      {"layers.0.self_attn.k_proj.weight", "BF16", {32, kD}},
      {"layers.0.self_attn.v_proj.weight", "BF16", {32, kD}},
      {"layers.0.self_attn.o_proj.weight", "BF16", {kD, 64}},
      {"layers.0.self_attn.q_norm.weight", "BF16", {32}},
      {"layers.0.self_attn.k_norm.weight", "BF16", {32}},
      {"layers.0.mlp.gate_proj.weight", "BF16", {kInter, kD}},
      {"layers.0.mlp.up_proj.weight", "BF16", {kInter, kD}},
      {"layers.0.mlp.down_proj.weight", "BF16", {kD, kInter}},
      {"layers.0.attention_conv.base_kernel", "BF16", {2, 2, kD}},
      {"layers.0.attention_conv.kernel_projection.weight", "BF16", {16, kD}},
      {"layers.0.mlp_conv.base_kernel", "BF16", {2, 2, kD}},
      {"layers.0.mlp_conv.kernel_projection.weight", "BF16", {16, kD}},
  };
  return t;
}

void write_dflash2(const fs::path& dir) {
  json config{{"architectures", {"DFlash2DraftModel"}},
              {"is_causal", false},
              {"hidden_size", kD},
              {"vocab_size", kVocab},
              {"num_hidden_layers", 1},
              {"num_target_layers", 4},
              {"num_attention_heads", 2},
              {"num_key_value_heads", 1},
              {"head_dim", 32},
              {"intermediate_size", kInter},
              {"sliding_window", 16},
              {"rms_norm_eps", 1e-6},
              {"rope_parameters", {{"rope_type", "default"}, {"rope_theta", 1e6}}},
              {"layer_types", {"sliding_attention"}},
              {"max_position_embeddings", 2048},
              {"dtype", "bfloat16"},
              {"dflash_config", {{"block_size", 4},
                                 {"mask_token_id", 63},
                                 {"target_layer_ids", {1, 3}},
                                 {"conv_group_size", 16},
                                 {"conv_kernel_size", 2},
                                 {"selector_rank", 64},
                                 {"selector_top_k", 4}}}};
  write_text(dir / "config.json", config.dump(2));
  write_safetensors(dir / "model.safetensors", dflash2_tensors());
}

// What the converter writes for a BF16 [rows, cols] matrix: U32 codes plus
// BF16 scales and biases per 64 columns.
std::uint64_t q8_bytes(std::int64_t rows, std::int64_t cols) {
  return static_cast<std::uint64_t>(rows * cols) + 2 * static_cast<std::uint64_t>(rows * cols / 64) * 2;
}

// lse_place is linked (through libLSE), so the scheduler serves whatever
// device set was opened: open the host one, as lse_open opens its pool.
graph::Scheduler* host_scheduler() {
  if (place::default_devices() == nullptr || place::default_devices()->size() == 0) {
    const Status opened = place::open_default_devices("cpu:0");
    LSE_EXPECT_OK(opened);
  }
  return graph::default_scheduler();
}

std::uint64_t device_live() {
  return backend::allocation_totals(backend::MemoryClass::kDevice).live;
}

json estimate(const model::MemoryPlanRequest& r) {
  auto e = model::estimate_memory(r);
  LSE_EXPECT_OK(e.status());
  return e.ok() ? e.release() : json::object();
}

}  // namespace

LSE_TEST(dense_qwen_fixture_reports_the_loader_facts) {
  TempDir dir("dense");
  write_qwen(dir.path, false);
  auto got = model::model_info(dir.path.string());
  LSE_EXPECT_OK(got.status());
  if (!got.ok()) return;
  const json& i = *got;
  LSE_EXPECT_STR(i["kind"].get<std::string>(), std::string("model"));
  LSE_EXPECT_STR(i["architecture"].get<std::string>(), std::string("qwen3.5"));
  LSE_EXPECT(i["loadable"].get<bool>());
  LSE_EXPECT(!i["moe"].get<bool>());
  LSE_EXPECT_EQ(i["layers"].get<int>(), 4);
  LSE_EXPECT_EQ(i["hidden_size"].get<int>(), kD);
  LSE_EXPECT_EQ(i["attention_heads"].get<int>(), kQH);
  LSE_EXPECT_EQ(i["kv_heads"].get<int>(), kKVH);
  LSE_EXPECT_EQ(i["head_dim"].get<int>(), kHD);
  LSE_EXPECT_EQ(i["vocab_size"].get<int>(), kVocab);
  LSE_EXPECT_EQ(i["kv_layer_count"].get<int>(), 1);
  LSE_EXPECT(i["kv_layers"] == json::array({3}));
  LSE_EXPECT_STR(i["layer_kinds"][0].get<std::string>(), std::string("linear_attention"));
  LSE_EXPECT_STR(i["layer_kinds"][3].get<std::string>(), std::string("full_attention"));
  LSE_EXPECT_EQ(i["linear_attention"]["layers"].get<int>(), 3);
  // [value heads, dim, dim] f32 plus the fused conv tail [kernel-1, width] f32.
  LSE_EXPECT_EQ(i["linear_attention"]["state_bytes_per_sequence"].get<std::uint64_t>(),
                3u * 4u * static_cast<std::uint64_t>(kGV * kGHD * kGHD + 3 * kConv));
  LSE_EXPECT_EQ(i["max_position_embeddings"].get<int>(), 2048);
  LSE_EXPECT_EQ(i["default_kv_len"].get<int>(), 4096);
  LSE_EXPECT_EQ(i["quantization"]["bits"].get<int>(), 4);
  LSE_EXPECT_EQ(i["quantization"]["group_size"].get<int>(), 64);
  LSE_EXPECT_STR(i["kv_cache_dtype_default"].get<std::string>(), std::string("bf16"));
  LSE_EXPECT_EQ(i["mtp"]["layers"].get<int>(), 1);
  LSE_EXPECT(!i["mtp"]["present"].get<bool>());
  LSE_EXPECT_EQ(i["dflash2"]["compatibility"]["num_target_layers"].get<int>(), 4);

  // Everything but the refused vision tower is uploaded, at its stored width.
  std::uint64_t loaded = 0, refused = 0;
  for (const auto& t : qwen_tensors(false))
    (t.name.starts_with("vision_tower.") ? refused : loaded) += bytes_of(t);
  LSE_EXPECT_EQ(i["weights_bytes"].get<std::uint64_t>(), loaded);
  LSE_EXPECT_EQ(i["unloaded_bytes"].get<std::uint64_t>(), refused);
  LSE_EXPECT_EQ(i["weights_vram_bytes"].get<std::uint64_t>(), model::kWeightSlabBytes);

  // KV per token: one attention layer, K and V, one head of 64.
  for (const auto& f : i["kv_cache_dtypes"]) {
    const auto name = f["name"].get<std::string>();
    const std::uint64_t want = name == "fp32" ? 2 * 64 * 4
                               : name == "fp16" || name == "bf16" ? 2 * 64 * 2
                                                                  : 2 * (64 / 4 + 1) * 4;
    LSE_EXPECT_EQ(f["bytes_per_token"].get<std::uint64_t>(), want);
    LSE_EXPECT_EQ(f["bytes_per_block"].get<std::uint64_t>(), want * 16);
    LSE_EXPECT_EQ(f["mtp_bytes_per_token"].get<std::uint64_t>(), want);
  }
}

LSE_TEST(moe_qwen_fixture_is_detected_from_its_tensors) {
  TempDir dir("moe");
  write_qwen(dir.path, true);
  auto got = model::model_info(dir.path.string());
  LSE_EXPECT_OK(got.status());
  if (!got.ok()) return;
  const json& i = *got;
  LSE_EXPECT_STR(i["architecture"].get<std::string>(), std::string("qwen3.5-moe"));
  LSE_EXPECT(i["moe"].get<bool>());
  LSE_EXPECT_EQ(i["experts"]["count"].get<int>(), 4);
  LSE_EXPECT_EQ(i["experts"]["active"].get<int>(), 2);
  LSE_EXPECT_EQ(i["experts"]["intermediate_size"].get<int>(), 64);
  LSE_EXPECT(i["intermediate_size"].is_null());
  LSE_EXPECT(i["loadable"].get<bool>());

  // The tensors decide, not model_type: the same MoE tensors under a dense
  // model_type are still the MoE architecture.
  json config = json::parse(std::ifstream(dir.path / "config.json"));
  config["model_type"] = "qwen3_5";
  write_text(dir.path / "config.json", config.dump());
  auto again = model::model_info(dir.path.string());
  LSE_EXPECT_OK(again.status());
  if (again.ok())
    LSE_EXPECT_STR((*again)["architecture"].get<std::string>(), std::string("qwen3.5-moe"));
}

LSE_TEST(bf16_dflash2_fixture_reports_its_converted_layout) {
  TempDir dir("dflash2");
  write_dflash2(dir.path);
  auto got = model::model_info(dir.path.string());
  LSE_EXPECT_OK(got.status());
  if (!got.ok()) return;
  const json& i = *got;
  LSE_EXPECT_STR(i["kind"].get<std::string>(), std::string("dflash2_draft"));
  LSE_EXPECT(i["loadable"].get<bool>());
  const json& d = i["dflash2"];
  LSE_EXPECT(d["draft"].get<bool>());
  LSE_EXPECT_EQ(d["hidden_size"].get<int>(), kD);
  LSE_EXPECT_EQ(d["vocab_size"].get<int>(), kVocab);
  LSE_EXPECT_EQ(d["num_target_layers"].get<int>(), 4);
  LSE_EXPECT_EQ(d["block_size"].get<int>(), 4);
  LSE_EXPECT_STR(d["source"].get<std::string>(), std::string("bf16"));
  LSE_EXPECT(d["conversion"]["needed"].get<bool>());
  LSE_EXPECT(!d["conversion"]["cached"].get<bool>());
  // Ring: K and V [1, kv_heads, window - 1 + block, head_dim] f32 per layer.
  LSE_EXPECT_EQ(d["ring_bytes"].get<std::uint64_t>(), 2u * 1 * (16 - 1 + 4) * 32 * 4);
  LSE_EXPECT_EQ(i["quantization"]["bits"].get<int>(), 8);

  // Every BF16 matrix is converted to Q8 before it reaches the device.
  std::uint64_t want = 0;
  for (const auto& t : dflash2_tensors()) {
    const bool matrix = t.shape.size() == 2 &&
                        (t.name.ends_with(".weight") || t.name.ends_with("_codebook"));
    want += matrix ? q8_bytes(t.shape[0], t.shape[1]) : bytes_of(t);
  }
  LSE_EXPECT_EQ(i["weights_bytes"].get<std::uint64_t>(), want);
  // Nothing was converted by asking.
  LSE_EXPECT(!fs::exists(dir.path / "lse-q8g64"));
}

LSE_TEST(kv_sizing_matches_the_paged_allocator) {
  auto* sched = host_scheduler();
  LSE_EXPECT(sched != nullptr);
  if (sched == nullptr) return;
  // The real model's geometry: four KV heads of 256, every format, through
  // the pool rungs and past the 2048-block switch to 256-block steps.
  for (const auto format : {kv::CacheDType::kF32, kv::CacheDType::kF16, kv::CacheDType::kBF16,
                            kv::CacheDType::kFP8, kv::CacheDType::kBF8}) {
    const std::int32_t capacity = 40000;
    ops::PagedKvLayer layer;
    const std::uint64_t before = device_live();
    for (const std::int32_t tokens : {1, 17, 130, 1000, 4096, 32769, 33000, 40000}) {
      LSE_EXPECT_OK(ops::ensure_paged_kv(layer, 1, tokens, capacity, 4, 256, format));
      const std::uint64_t pools = kv::contiguous_layer_bytes(format, 4, 256, tokens, capacity, 1);
      LSE_EXPECT_EQ(static_cast<std::uint64_t>(layer.pool_bytes()), pools);
      LSE_EXPECT_EQ(device_live() - before, pools + kv::block_table_bytes(tokens, capacity, 1));
    }
  }
}

LSE_TEST(fragment_sizing_matches_the_kv_memory_manager) {
  backend::BackendAdapter<backend::CpuBackend> cpu;
  LSE_EXPECT_OK(cpu.init(0));
  auto manager = kv::MemoryManager::create();
  const auto format = kv::CacheDType::kBF16;
  const std::int32_t capacity = 262144;
  std::vector<std::shared_ptr<kv::FragmentStorage>> planes;
  std::size_t fragments = 0;
  const std::uint64_t before = device_live();
  std::uint64_t tables = 0;
  // One layer's K and V at 1000 tokens, then grown to 70000, which takes the
  // two planes past one arena.
  for (const std::int32_t tokens : {1000, 70000}) {
    for (int p = 0; p < 2; ++p) {
      if (planes.size() <= static_cast<std::size_t>(p)) {
        planes.push_back(std::make_shared<kv::FragmentStorage>(manager, cpu, backend::kDefaultStream));
        LSE_EXPECT_OK(planes.back()->reserve(
            static_cast<std::size_t>(kv::blocks_for(capacity, kv::kBlockSize)) *
            kv::block_plane_bytes(format, 4, 256)));
        tables += kv::fragment_table_bytes(format, 4, 256, capacity, 1);
      }
      const auto blocks = kv::blocks_for(tokens, kv::kBlockSize);
      LSE_EXPECT_OK(planes[static_cast<std::size_t>(p)]->grow(
          static_cast<std::size_t>(blocks) * kv::block_plane_bytes(format, 4, 256)));
      LSE_EXPECT_EQ(planes[static_cast<std::size_t>(p)]->fragment_count(),
                    kv::plane_fragments(format, 4, 256, blocks));
    }
    fragments = kv::fragmented_layer_fragments(format, 4, 256, tokens, 1);
    LSE_EXPECT_EQ(static_cast<std::uint64_t>(manager->stats().reserved_bytes),
                  static_cast<std::uint64_t>(kv::arena_bytes(fragments)));
    LSE_EXPECT_EQ(device_live() - before, kv::arena_bytes(fragments) + tables);
  }
}

LSE_TEST(estimate_matches_a_real_load_on_the_host_backend) {
  auto* sched = host_scheduler();
  LSE_EXPECT(sched != nullptr);
  if (sched == nullptr) return;
  TempDir target("load-target"), draft("load-draft");
  write_qwen(target.path, false);
  write_dflash2(draft.path);

  model::MemoryPlanRequest r;
  r.model = target.path.string();
  r.draft = model::DraftKind::kDFlash2;
  r.draft_path = draft.path.string();
  r.kv_len = 4096;
  r.device_arch = "cpu";
  r.fragmented_kv = false;
  const json plan = estimate(r);
  if (plan.empty()) return;

  auto paths = model::resolve_model(r.model);
  LSE_EXPECT_OK(paths.status());
  if (!paths.ok()) return;
  auto config = model::Config::from_json_file(paths->config);
  LSE_EXPECT_OK(config.status());
  auto weights = model::SafeTensors::open(paths->weights);
  LSE_EXPECT_OK(weights.status());
  if (!config.ok() || !weights.ok()) return;
  config->kv_length = r.kv_len;
  auto lm = model::build_model(*config, *weights);
  LSE_EXPECT_OK(lm.status());
  if (!lm.ok()) return;

  const std::uint64_t before = device_live();
  model::WeightBinder binder(*weights, &config->quantization);
  LSE_EXPECT_OK((*lm)->load(binder));
  // The scheduler keeps the last program it ran, with that program's
  // outputs, until the next one replaces it. That is not load memory, and
  // whether a load ends with one depends on what it evaluated last (the RoPE
  // tables are no longer evaluated as a graph), so drop it before measuring.
  LSE_EXPECT_OK(sched->release_program());
  const std::uint64_t target_bytes = device_live() - before;
  // Slabs, the GDN rates prepared at load, and the RoPE tables.
  LSE_EXPECT_EQ(target_bytes, plan["weights_bytes"].get<std::uint64_t>() +
                                  plan["workspace"]["rope_bytes"].get<std::uint64_t>());

  const std::uint64_t middle = device_live();
  auto opened = model::DFlash2Module::open(r.draft_path, *config, **lm);
  LSE_EXPECT_OK(opened.status());
  if (!opened.ok()) return;
  // Its slab share, the widened conv kernels, the ring and its RoPE.
  LSE_EXPECT_EQ(device_live() - middle, plan["draft_bytes"].get<std::uint64_t>());
}

LSE_TEST(estimate_follows_the_settings) {
  TempDir dir("settings");
  write_qwen(dir.path, false);
  model::MemoryPlanRequest r;
  r.model = dir.path.string();
  r.kv_len = 8192;
  r.device_arch = "cpu";
  const json bf16 = estimate(r);
  r.kv_cache_dtype = kv::CacheDType::kFP8;
  const json fp8 = estimate(r);
  r.kv_cache_dtype = kv::CacheDType::kF32;
  const json fp32 = estimate(r);
  if (bf16.empty() || fp8.empty() || fp32.empty()) return;
  LSE_EXPECT(fp8["kv"]["pool_bytes"].get<std::uint64_t>() <
             bf16["kv"]["pool_bytes"].get<std::uint64_t>());
  LSE_EXPECT_EQ(fp32["kv"]["pool_bytes"].get<std::uint64_t>(),
                2 * bf16["kv"]["pool_bytes"].get<std::uint64_t>());
  LSE_EXPECT_EQ(bf16["kv"]["pool_bytes"].get<std::uint64_t>(),
                static_cast<std::uint64_t>(kv::fragmented_layer_fragments(
                    kv::CacheDType::kBF16, kKVH, kHD, 8192, 1)) * kv::kFragmentBytes);
  LSE_EXPECT_EQ(bf16["settings"]["kv_len"].get<int>(), 8192);
  LSE_EXPECT_STR(bf16["draft"]["kind"].get<std::string>(), std::string("none"));

  // A smaller context holds less KV; a wider ubatch needs more activation.
  r.kv_cache_dtype.reset();
  r.context_tokens = 1000;
  const json small = estimate(r);
  r.context_tokens = 0;
  r.ubatch_size = 512;
  const json narrow = estimate(r);
  if (small.empty() || narrow.empty()) return;
  LSE_EXPECT(small["kv_bytes"].get<std::uint64_t>() < bf16["kv_bytes"].get<std::uint64_t>());
  LSE_EXPECT(narrow["activation_bytes"].get<std::uint64_t>() <
             bf16["activation_bytes"].get<std::uint64_t>());

  // The largest kv_len that fits is the one whose estimate fits: the RoPE
  // tables grow with every position, so a budget of exactly the estimate at
  // 1536 tokens answers 1536.
  r.ubatch_size = 1024;
  r.kv_len = 1536;
  const json at1536 = estimate(r);
  if (at1536.empty()) return;
  r.device_memory_bytes = at1536["device_total_bytes"].get<std::uint64_t>();
  const json sized = estimate(r);
  if (sized.empty()) return;
  LSE_EXPECT(sized["fits"].get<bool>());
  LSE_EXPECT_EQ(sized["max_kv_len"].get<std::int32_t>(), 1536);
  r.device_memory_bytes = 1;
  LSE_EXPECT_EQ(estimate(r)["max_kv_len"].get<std::int32_t>(), 0);
  r.device_memory_bytes = 0;

  // Bad settings are refused, not estimated.
  model::MemoryPlanRequest bad = r;
  bad.ubatch_size = 2048;
  bad.batch_size = 1024;
  LSE_EXPECT(!model::estimate_memory(bad).ok());
  bad = r;
  bad.draft = model::DraftKind::kDFlash2;
  bad.draft_path = dir.path.string();  // a model, not a draft
  LSE_EXPECT(!model::estimate_memory(bad).ok());
}

LSE_TEST(c_api_answers_and_refuses) {
  TempDir dir("capi");
  write_qwen(dir.path, false);
  char* out = nullptr;
  char* err = nullptr;
  LSE_EXPECT_EQ(lse_model_info(dir.path.c_str(), &out, &err), LSE_OK);
  LSE_EXPECT(out != nullptr && err == nullptr);
  if (out != nullptr) {
    const json info = json::parse(out);
    LSE_EXPECT_STR(info["architecture"].get<std::string>(), std::string("qwen3.5"));
    // No generation_config.json and no chat template: LSE's neutral defaults,
    // said to be so, and no thinking controls.
    LSE_EXPECT(info["generation_defaults"]["sources"]["temperature"] == "lse_default");
    LSE_EXPECT(info["generation_defaults"]["max_new_tokens"].is_null());
    LSE_EXPECT(!info["thinking"]["supported"].get<bool>());
    LSE_EXPECT(info["thinking"]["source"].is_null());
  }
  lse_free(out);

  // The checkpoint's own files, copied in: its generation_config and its
  // chat template's thinking levels are what model info reports.
  for (const char* file : {"chat_template.jinja", "generation_config.json"})
    std::filesystem::copy_file(std::filesystem::path("tests/fixtures/chat_templates/qwen3.8") / file,
                               dir.path / file, std::filesystem::copy_options::overwrite_existing);
  out = nullptr;
  LSE_EXPECT_EQ(lse_model_info(dir.path.c_str(), &out, &err), LSE_OK);
  if (out != nullptr) {
    const json info = json::parse(out);
    LSE_EXPECT_EQ(info["generation_defaults"]["top_k"].get<int>(), 20);
    LSE_EXPECT(info["generation_defaults"]["sources"]["top_k"] == "generation_config.json");
    LSE_EXPECT(info["thinking"]["supported"].get<bool>());
    LSE_EXPECT_STR(info["thinking"]["default_level"].get<std::string>(), std::string("xhigh"));
    LSE_EXPECT_EQ(info["thinking"]["levels"].size(), std::size_t{4});
  }
  lse_free(out);

  out = nullptr;
  LSE_EXPECT_EQ(lse_model_info("/no/such/model/dir", &out, &err), LSE_ERR_INVALID_ARGUMENT);
  LSE_EXPECT(out == nullptr && err != nullptr);
  lse_free(err);
  err = nullptr;

  lse_config cfg;
  lse_config_init(&cfg);
  cfg.model = dir.path.c_str();
  cfg.kv_len = 2048;
  cfg.kv_cache_dtype = "fp8";
  cfg.dialect = "hip";
  LSE_EXPECT_EQ(lse_estimate(&cfg, R"({"device_arch":"cpu","sequences":2})", &out, &err), LSE_OK);
  if (out != nullptr) {
    const json e = json::parse(out);
    LSE_EXPECT_STR(e["settings"]["kv_cache_dtype"].get<std::string>(), std::string("fp8"));
    LSE_EXPECT_STR(e["settings"]["kv_storage"].get<std::string>(), std::string("contiguous"));
    LSE_EXPECT_EQ(e["kv"]["sequences"].get<int>(), 2);
    LSE_EXPECT(e["total_bytes"].get<std::uint64_t>() > e["weights_bytes"].get<std::uint64_t>());
  }
  lse_free(out);
  // No dialect named is Loom, on every platform, and Loom stores K/V in
  // fragments.
  out = nullptr;
  cfg.dialect = nullptr;
  LSE_EXPECT_EQ(lse_estimate(&cfg, R"({"device_arch":"cpu","sequences":2})", &out, &err), LSE_OK);
  if (out != nullptr) {
    const json e = json::parse(out);
    LSE_EXPECT_STR(e["settings"]["kv_storage"].get<std::string>(), std::string("fragmented"));
  }
  lse_free(out);
  cfg.dialect = "hip";
  out = nullptr;
  LSE_EXPECT_EQ(lse_estimate(&cfg, "[1, 2]", &out, &err), LSE_ERR_INVALID_ARGUMENT);
  lse_free(err);
  err = nullptr;
  cfg.kv_cache_dtype = "int3";
  LSE_EXPECT_EQ(lse_estimate(&cfg, nullptr, &out, &err), LSE_ERR_INVALID_ARGUMENT);
  lse_free(err);
  err = nullptr;
  lse_config stale;
  std::memset(&stale, 0, sizeof stale);
  LSE_EXPECT_EQ(lse_estimate(&stale, nullptr, &out, &err), LSE_ERR_INVALID_ARGUMENT);
  lse_free(err);
}

LSE_TEST(allocation_ledger_returns_every_charge) {
  backend::BackendAdapter<backend::CpuBackend> cpu;
  LSE_EXPECT_OK(cpu.init(0));
  const auto before = backend::allocation_totals(backend::MemoryClass::kDevice);
  {
    auto a = cpu.allocate(1 << 20, backend::MemoryClass::kDevice, backend::kDefaultStream);
    LSE_EXPECT_OK(a.status());
    if (!a.ok()) return;
    backend::DeviceBuffer view = *a;  // a view shares the charge
    auto mid = backend::allocation_totals(backend::MemoryClass::kDevice);
    LSE_EXPECT_EQ(mid.live - before.live, std::uint64_t{1} << 20);
    LSE_EXPECT(mid.peak >= mid.live);
    cpu.deallocate(*a);
    mid = backend::allocation_totals(backend::MemoryClass::kDevice);
    LSE_EXPECT_EQ(mid.live - before.live, std::uint64_t{1} << 20);
    (void)view;
  }
  LSE_EXPECT_EQ(backend::allocation_totals(backend::MemoryClass::kDevice).live, before.live);
}

// lse_open and lse_close in one process, three times over: everything an
// engine allocated goes back to the device when it closes, so each open starts
// from what the process held before the first one, and a second engine is
// refused while one is open.
LSE_TEST(closing_an_engine_returns_every_device_byte) {
  TempDir target("reopen-target"), draft("reopen-draft");
  write_qwen(target.path, false);
  write_dflash2(draft.path);
  {
    // Character-level, ids below the fixture's vocabulary.
    json vocab = json::object();
    int next = 0;
    for (char c = 'a'; c <= 'z'; ++c) vocab[std::string(1, c)] = next++;
    vocab[" "] = next++;
    json added = json::array();
    for (const char* special : {"<|im_start|>", "<|im_end|>", "<|endoftext|>"})
      added.push_back({{"id", next++}, {"content", special}, {"special", true},
                       {"single_word", false}, {"lstrip", false}, {"rstrip", false},
                       {"normalized", false}});
    write_text(target.path / "tokenizer.json",
               json{{"model", {{"type", "BPE"}, {"vocab", vocab}, {"merges", json::array()}}},
                    {"added_tokens", added}}.dump());
  }
  const std::string target_dir = target.path.string(), draft_dir = draft.path.string();
  lse_config cfg;
  lse_config_init(&cfg);
  cfg.model = target_dir.c_str();
  cfg.dflash2 = 1;
  cfg.dflash2_model = draft_dir.c_str();
  cfg.kv_len = 2048;
  // The ctest environment selects cpu:0 (LSE_POOL), and an earlier test in
  // this process may already have opened that set.
  cfg.has_temperature = 1;
  cfg.temperature = 0.0f;
  // Earlier tests in this process may still hold arrays (and so slabs) they
  // made; closing the first engine releases the slabs nothing uses, so the
  // floor is what remains after that close.
  const std::uint64_t before = device_live();
  std::uint64_t loaded = 0, floor = 0;
  for (int cycle = 0; cycle < 3; ++cycle) {
    char* err = nullptr;
    lse_engine* engine = lse_open(&cfg, &err);
    if (engine == nullptr) {
      std::printf("       open failed: %s\n", err ? err : "?");
      lse_free(err);
      LSE_EXPECT(engine != nullptr);
      return;
    }
    const std::uint64_t open_bytes = device_live();
    if (cycle == 0) loaded = open_bytes;
    // Every open allocates what the first one did, on top of the floor.
    LSE_EXPECT_EQ(open_bytes, loaded);
    // A second engine is refused while this one is open.
    char* refused = nullptr;
    LSE_EXPECT(lse_open(&cfg, &refused) == nullptr);
    LSE_EXPECT_EQ(lse_last_error(), LSE_ERR_STATE);
    lse_free(refused);
    // The peak is this engine's own.
    char* status = nullptr;
    LSE_EXPECT_EQ(lse_status(engine, &status), LSE_OK);
    if (status != nullptr) {
      const json s = json::parse(status);
      LSE_EXPECT(s["memory"]["device_peak_bytes"].get<std::uint64_t>() < 2 * loaded);
    }
    lse_free(status);
    lse_close(engine);
    std::printf("       cycle %d: open %llu, after close %llu (before %llu)\n", cycle,
                static_cast<unsigned long long>(open_bytes),
                static_cast<unsigned long long>(device_live()),
                static_cast<unsigned long long>(before));
    LSE_EXPECT(device_live() <= before);
    if (cycle == 0) floor = device_live();
    LSE_EXPECT_EQ(device_live(), floor);
    // The model is gone, not merely unreferenced by the engine.
    LSE_EXPECT(loaded - floor > (std::uint64_t{1} << 20));
  }
}

LSE_TEST_MAIN()
