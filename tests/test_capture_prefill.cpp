// The device-first scheduler path on a host device whose kernels are
// recorded and never run ("capture": Loom emission, no execution), so a wide
// prefill's real allocations can be checked without a GPU: the phase path
// recycles its workspace and gives the device back after the request; and
// when a phase group cannot run, the pass falls back to the per-group path,
// which says so and fails cleanly when its intermediates cannot fit instead
// of running out of memory part way.
#include "harness.hpp"

#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "lse/backend/backend.hpp"
#include "lse/backends/cpu/cpu_backend.hpp"
#include "lse/backends/hrx/arch_database.hpp"
#include "lse/backends/hrx/code_object.hpp"
#include "lse/backends/hrx/device_info.hpp"
#include "lse/backends/hrx/loomc/loom_emitter.hpp"
#include "lse/graph/codegen.hpp"
#include "lse/graph/graph.hpp"
#include "lse/model/config.hpp"
#include "lse/model/hybrid_lm.hpp"
#include "lse/model/registry.hpp"
#include "lse/model/weights.hpp"
#include "lse/place/devices.hpp"
#include "lse/server/router.hpp"
#include "lse/tokenizer/tokenizer.hpp"

using namespace lse;
using namespace lse::graph;
using json = nlohmann::json;

namespace lse::backend {
struct NullCompiler final : IKernelCompiler {
  Result<CompiledKernel> compile(std::string_view, std::string_view) const override {
    CompiledKernel k; k.code.push_back(std::byte{1}); return k;
  }
  bool available() const override { return true; }
  std::string identity() const override { return "capture-no-execution-v1"; }
};
struct CaptureEmitter final : IKernelEmitter {
  mutable LoomEmitter real;
  mutable unsigned failures = 0;
  Result<EmittedKernel> emit(const FusionGroup& g, const DeviceInfo& d) const override {
    auto r = real.emit(g, d);
    if (!r.ok() && failures++ < 20)
      std::fprintf(stderr, "capture: emit refused (%zu nodes, phase=%d): %s\n", g.nodes.size(),
                   g.is_phase, r.status().to_string().c_str());
    return r;
  }
  Dialect dialect() const noexcept override { return Dialect::kLoom; }
  std::string_view prelude() const noexcept override { return {}; }
  DialectSourceTable sources() const noexcept override { return real.sources(); }
  std::uint64_t cache_key(const FusionGroup& g, const DeviceInfo& d) const override { return real.cache_key(g, d); }
};
struct CaptureBackend : Backend<CaptureBackend> {
  static constexpr std::string_view kName = "capture";
  CpuBackend cpu;
  mutable CaptureEmitter emitter;
  mutable NullCompiler compiler;
  mutable KernelToolchain chain{Dialect::kLoom, &emitter, &compiler};
  DeviceInfo info;
  AmdDeviceInfo amd;
  std::uint64_t launches = 0;
  // Test knobs, process-wide: refuse every launch after this many (-1:
  // never), and report this much device memory (0: no figure).
  static inline int refuse_launches_after = -1;
  static inline int launched = 0;
  static inline std::size_t free_bytes = 0;
  Status init_impl(int ordinal) {
    LSE_RETURN_IF_ERROR(cpu.init_impl(ordinal));
    info = cpu.device_info_impl();
    info.arch = "gfx1201"; info.compute_units = 64; info.wavefront_size = 32;
    info.max_threads_per_workgroup = 1024; info.lds_bytes_per_workgroup = 65536;
    info.total_memory = 34208743424ull;
    info.name = "capture gfx1201";
    apply_arch_defaults(info, amd);
    info.arch_facts = arch_facts_for(info);
    info.extension_id = AmdDeviceInfo::kExtensionId;
    info.extension = &amd;
    return OkStatus();
  }
  void shutdown_impl() noexcept { cpu.shutdown_impl(); }
  static Result<std::vector<DeviceDescriptor>> enumerate_devices() { return CpuBackend::enumerate_devices(); }
  const DeviceInfo& device_info_impl() const noexcept { return info; }
  Result<DeviceBuffer> allocate_impl(std::size_t b, MemoryClass c, Stream s) { return cpu.allocate_impl(b, c, s); }
  void deallocate_impl(DeviceBuffer& b) noexcept { cpu.deallocate_impl(b); }
  Status copy_h2d_impl(const void* src, DeviceBuffer& dst, std::size_t n, std::size_t off) { return cpu.copy_h2d_impl(src, dst, n, off); }
  Status copy_d2h_impl(const DeviceBuffer& src, void* dst, std::size_t n, std::size_t off) { return cpu.copy_d2h_impl(src, dst, n, off); }
  Status copy_peer_impl(const DeviceBuffer& src, DeviceBuffer& dst, std::size_t n, std::size_t so, std::size_t d) {
    if (!src.ptr || !dst.ptr) return LSE_ERROR(kInvalidArgument, "capture copy without host storage");
    std::memmove(static_cast<std::byte*>(dst.ptr) + dst.offset + d,
                 static_cast<const std::byte*>(src.ptr) + src.offset + so, n);
    return OkStatus();
  }
  Status synchronize_impl() { return OkStatus(); }
  std::span<const KernelToolchain> toolchains_impl() const noexcept { return {&chain, 1}; }
  Result<KernelHandle> load_executable_impl(std::string_view name, std::span<const std::byte>) {
    return KernelHandle{1, 0, std::string(name)};
  }
  Result<std::size_t> sample_free_memory_impl() const {
    if (free_bytes == 0) return LSE_ERROR(kUnimplemented, "capture has no free-memory figure");
    const auto live = allocation_totals(MemoryClass::kDevice).live;
    return live < free_bytes ? free_bytes - live : 0;
  }
  Status launch_impl(const KernelHandle&, const LaunchDims&, const DispatchArgs&) {
    if (refuse_launches_after >= 0 && launched++ >= refuse_launches_after)
      return LSE_ERROR(kDeviceError, "capture device refuses this kernel");
    ++launches;
    return OkStatus();
  }
};
}  // namespace lse::backend
LSE_REGISTER_BACKEND("capture", ::lse::backend::CaptureBackend)

namespace {

struct NamedTensor {
  std::string name;
  std::vector<std::int64_t> dims;
};

std::size_t tensor_elems(const NamedTensor& t) {
  std::size_t n = 1;
  for (std::int64_t d : t.dims) n *= static_cast<std::size_t>(d);
  return n;
}

std::string safetensors_header(const std::vector<NamedTensor>& tensors,
                               std::size_t* total) {
  std::string header = "{";
  std::size_t offset = 0;
  for (const NamedTensor& t : tensors) {
    std::string dims;
    for (std::int64_t d : t.dims) {
      if (!dims.empty()) dims += ",";
      dims += std::to_string(d);
    }
    const std::size_t bytes = tensor_elems(t) * 4;
    if (header.size() > 1) header += ",";
    header += "\"" + t.name + "\":{\"dtype\":\"F32\",\"shape\":[" + dims +
              "],\"data_offsets\":[" + std::to_string(offset) + "," +
              std::to_string(offset + bytes) + "]}";
    offset += bytes;
  }
  header += "}";
  while (header.size() % 8 != 0) header += " ";
  *total = offset;
  return header;
}

// `value(name, index)` fills each tensor, so a fixture can hand one tensor a
// structure and let the rest take filler.
template <typename Fill>
void write_shaped(const std::filesystem::path& path,
                  const std::vector<NamedTensor>& tensors, const Fill& value) {
  std::size_t total = 0;
  const std::string header = safetensors_header(tensors, &total);
  std::ofstream out(path, std::ios::binary);
  const std::uint64_t n = header.size();
  out.write(reinterpret_cast<const char*>(&n), sizeof(n));
  out.write(header.data(), static_cast<std::streamsize>(header.size()));
  std::vector<float> data;
  for (const NamedTensor& t : tensors) {
    data.resize(tensor_elems(t));
    for (std::size_t i = 0; i < data.size(); ++i) data[i] = value(t.name, i);
    out.write(reinterpret_cast<const char*>(data.data()),
              static_cast<std::streamsize>(data.size() * 4));
  }
}

// Deterministic and not symmetric: an all-zero module drafts token 0 every
// step, which would make a rejection test pass for the wrong reason.
float filler(std::size_t i) {
  return 0.03f * static_cast<float>(static_cast<int>((i * 7) % 23) - 11);
}

model::Config mtp_test_config(std::int32_t gdn_head_dim = 16) {
  model::Config c;
  c.vocab_size = 64;
  c.hidden_size = 32;
  c.num_layers = 4;
  c.full_attention_interval = 4;
  c.global_attention_layers.clear();
  c.sliding_window = 0;
  c.attn_q_heads = 2;
  c.attn_kv_heads = 1;
  c.attn_head_dim = 16;
  c.rope_dim = 4;
  c.rope_theta = 10000000.0f;
  c.gdn_qk_heads = 2;
  c.gdn_v_heads = 2;
  c.gdn_head_dim = gdn_head_dim;
  c.gdn_conv_kernel = 4;
  c.mlp_intermediate = 64;
  c.num_experts = 0;
  c.num_active_experts = 0;
  c.num_shared_experts = 0;
  c.expert_intermediate = 0;
  c.tie_word_embeddings = true;
  c.dtype = "float32";
  c.kv_length = 64;
  c.mtp_layers = 1;
  return c;
}

std::vector<NamedTensor> qwen_dense_tensors(const model::Config& c) {
  const std::int64_t h = c.hidden_size;
  const std::int64_t qh = c.attn_q_heads, kvh = c.attn_kv_heads;
  const std::int64_t ahd = c.attn_head_dim;
  const std::int64_t kh = c.gdn_qk_heads, vh = c.gdn_v_heads;
  const std::int64_t ghd = c.gdn_head_dim;
  const std::int64_t conv_dim = 2 * kh * ghd + vh * ghd;

  std::vector<NamedTensor> t;
  t.push_back({"language_model.model.embed_tokens.weight", {c.vocab_size, h}});
  t.push_back({"language_model.model.norm.weight", {h}});
  for (std::int32_t i = 0; i < c.num_layers; ++i) {
    const std::string p =
        "language_model.model.layers." + std::to_string(i) + ".";
    t.push_back({p + "input_layernorm.weight", {h}});
    t.push_back({p + "post_attention_layernorm.weight", {h}});
    if (c.is_attention_layer(i)) {
      const std::string a = p + "self_attn.";
      t.push_back({a + "q_proj.weight", {2 * qh * ahd, h}});
      t.push_back({a + "k_proj.weight", {kvh * ahd, h}});
      t.push_back({a + "v_proj.weight", {kvh * ahd, h}});
      t.push_back({a + "o_proj.weight", {h, qh * ahd}});
      t.push_back({a + "q_norm.weight", {ahd}});
      t.push_back({a + "k_norm.weight", {ahd}});
    } else {
      const std::string g = p + "linear_attn.";
      t.push_back({g + "in_proj_qkv.weight", {conv_dim, h}});
      t.push_back({g + "in_proj_z.weight", {vh * ghd, h}});
      t.push_back({g + "in_proj_a.weight", {vh, h}});
      t.push_back({g + "in_proj_b.weight", {vh, h}});
      t.push_back({g + "conv1d.weight", {conv_dim, c.gdn_conv_kernel, 1}});
      t.push_back({g + "A_log", {vh}});
      t.push_back({g + "dt_bias", {vh}});
      t.push_back({g + "norm.weight", {ghd}});
      t.push_back({g + "out_proj.weight", {h, vh * ghd}});
    }
    const std::string m = p + "mlp.";
    t.push_back({m + "gate_proj.weight", {c.mlp_intermediate, h}});
    t.push_back({m + "up_proj.weight", {c.mlp_intermediate, h}});
    t.push_back({m + "down_proj.weight", {h, c.mlp_intermediate}});
  }
  return t;
}



// The module's own file, named exactly as mlx-community/Qwen3.8-27B-MTP-4bit
// names it: no `language_model.model` prefix and no layer index above zero.
std::vector<NamedTensor> mtp_tensors(const model::Config& c) {
  const std::int64_t h = c.hidden_size;
  const std::int64_t qh = c.attn_q_heads, kvh = c.attn_kv_heads;
  const std::int64_t ahd = c.attn_head_dim;
  return {
      {"fc.weight", {h, 2 * h}},
      {"pre_fc_norm_hidden.weight", {h}},
      {"pre_fc_norm_embedding.weight", {h}},
      {"norm.weight", {h}},
      {"layers.0.input_layernorm.weight", {h}},
      {"layers.0.post_attention_layernorm.weight", {h}},
      {"layers.0.self_attn.q_proj.weight", {2 * qh * ahd, h}},
      {"layers.0.self_attn.k_proj.weight", {kvh * ahd, h}},
      {"layers.0.self_attn.v_proj.weight", {kvh * ahd, h}},
      {"layers.0.self_attn.o_proj.weight", {h, qh * ahd}},
      {"layers.0.self_attn.q_norm.weight", {ahd}},
      {"layers.0.self_attn.k_norm.weight", {ahd}},
      {"layers.0.mlp.gate_proj.weight", {c.mlp_intermediate, h}},
      {"layers.0.mlp.up_proj.weight", {c.mlp_intermediate, h}},
      {"layers.0.mlp.down_proj.weight", {h, c.mlp_intermediate}},
  };
}

std::string mtp_config_json(const model::Config& c) {
  return std::string("{\"tie_word_embeddings\": true, \"text_config\": {") +
         "\"vocab_size\": " + std::to_string(c.vocab_size) +
         ", \"hidden_size\": " + std::to_string(c.hidden_size) +
         ", \"num_hidden_layers\": " + std::to_string(c.num_layers) +
         ", \"rms_norm_eps\": 1e-06" +
         ", \"full_attention_interval\": " +
         std::to_string(c.full_attention_interval) +
         ", \"num_attention_heads\": " + std::to_string(c.attn_q_heads) +
         ", \"num_key_value_heads\": " + std::to_string(c.attn_kv_heads) +
         ", \"head_dim\": " + std::to_string(c.attn_head_dim) +
         ", \"linear_num_key_heads\": " + std::to_string(c.gdn_qk_heads) +
         ", \"linear_num_value_heads\": " + std::to_string(c.gdn_v_heads) +
         ", \"linear_conv_kernel_dim\": " +
         std::to_string(c.gdn_conv_kernel) +
         ", \"linear_key_head_dim\": " + std::to_string(c.gdn_head_dim) +
         ", \"linear_value_head_dim\": " + std::to_string(c.gdn_head_dim) +
         ", \"intermediate_size\": " + std::to_string(c.mlp_intermediate) +
         ", \"rope_parameters\": {\"rope_theta\": 10000000.0,"
         " \"partial_rotary_factor\": 0.25}" +
         ", \"max_position_embeddings\": 128, \"dtype\": \"float32\"" +
         ", \"mtp_num_hidden_layers\": 1" +
         ", \"mtp_use_dedicated_embeddings\": false}}";
}

std::uint64_t live() { return backend::allocation_totals(backend::MemoryClass::kDevice).live; }

// A character-level BPE vocabulary: every lowercase letter and a space, plus
// the ChatML specials, so any prompt below tokenizes one token per character.
struct Engine {
  std::filesystem::path dir;
  model::Config config;
  std::optional<model::SafeTensors> weights;
  std::unique_ptr<model::HybridLM> lm;
  std::unique_ptr<model::MtpModule> mtp;
  std::optional<tokenizer::Tokenizer> tok;
  std::unique_ptr<server::Router> router;
  std::uint64_t baseline = 0;

  ~Engine() {
    router.reset();
    mtp.reset();
    lm.reset();
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
  }

  Status open(std::size_t max_sessions, std::size_t budget, bool with_mtp = false) {
    dir = std::filesystem::temp_directory_path() / ("lse-memory-lifecycle-" + std::to_string(getpid()));
    std::filesystem::create_directories(dir);
    json vocab = json::object();
    std::uint32_t next = 0;
    for (char c = 'a'; c <= 'z'; ++c) vocab[std::string(1, c)] = next++;
    vocab[" "] = next++;
    json added = json::array();
    for (const char* special : {"<|im_start|>", "<|im_end|>", "<|endoftext|>"})
      added.push_back({{"id", next++}, {"content", special}, {"special", true},
                       {"single_word", false}, {"lstrip", false}, {"rstrip", false},
                       {"normalized", false}});
    {
      std::ofstream out(dir / "tokenizer.json");
      out << json{{"model", {{"type", "BPE"}, {"vocab", vocab}, {"merges", json::array()}}},
                  {"added_tokens", added}}.dump();
    }
    auto t = tokenizer::Tokenizer::from_file((dir / "tokenizer.json").string());
    if (!t.ok()) return t.status();
    tok.emplace(t.release());

    config = mtp_test_config(16);
    config.vocab_size = 64;
    config.kv_length = 4096;
    config.mtp_layers = with_mtp ? 1 : 0;
    config.sampling_defaults.temperature = 0.0f;
    write_shaped(dir / "model.safetensors", qwen_dense_tensors(config),
                 [](const std::string&, std::size_t i) { return filler(i); });
    auto st = model::SafeTensors::open((dir / "model.safetensors").string());
    if (!st.ok()) return st.status();
    weights.emplace(st.release());
    auto built = model::build_model(config, *weights, "");
    if (!built.ok()) return built.status();
    lm = built.release();
    model::WeightBinder binder(*weights);
    LSE_RETURN_IF_ERROR(lm->load(binder));
    if (with_mtp) {
      std::filesystem::create_directories(dir / "mtp");
      write_shaped(dir / "mtp" / "model.safetensors", mtp_tensors(config),
                   [](const std::string&, std::size_t i) { return filler(i); });
      std::ofstream(dir / "mtp" / "config.json") << mtp_config_json(config);
      auto opened = model::MtpModule::open((dir / "mtp").string(), config, *lm);
      if (!opened.ok()) return opened.status();
      mtp = opened.release();
    }
    server::ServerOptions options;
    options.model_id = "memory-fixture";
    options.max_sessions = max_sessions;
    options.session_memory_budget = budget;
    router = std::make_unique<server::Router>(*lm, *tok, options);
    if (mtp) router->use_mtp(*mtp);
    // What the engine holds once loaded. Caches the scheduler keeps for the
    // process (interned constants) may hold a little more from an earlier
    // engine; closing the last session lets those go too, so "back to
    // baseline" is checked as "no more than baseline".
    baseline = live();
    return OkStatus();
  }

  // One completion; returns the HTTP status.
  int complete(const std::string& session, const std::string& prompt, int max_tokens = 8) {
    json body{{"model", "memory-fixture"}, {"prompt", prompt}, {"max_tokens", max_tokens},
              {"temperature", 0}};
    if (!session.empty()) body["session_id"] = session;
    const auto reply = router->handle("POST", "/v1/completions", body.dump());
    if (reply.status != 200) std::printf("       %d: %s\n", reply.status, reply.body.c_str());
    return reply.status;
  }

  json sessions() {
    return json::parse(router->handle("GET", "/v1/lse/sessions", "").body)["data"];
  }

  int close(const std::string& session) {
    return router->handle("DELETE", "/v1/lse/sessions/" + session, "").status;
  }
};

// A prompt of `n` characters (one token each), distinct per seed.
std::string prompt(std::size_t n, unsigned seed) {
  std::string s;
  for (std::size_t i = 0; i < n; ++i) s += static_cast<char>('a' + (i * 7 + seed * 13) % 26);
  return s;
}


std::uint64_t site_peak(backend::AllocationSite site) {
  for (const auto& s : backend::allocation_sites())
    if (s.site == site) return s.peak;
  return 0;
}

}  // namespace

LSE_TEST(a_wide_prefill_on_the_device_path_recycles_its_workspace) {
  LSE_EXPECT_OK(place::open_default_devices("capture:0"));
  Engine e;
  LSE_EXPECT_OK(e.open(8, 0));
  if (!e.router) return;
  backend::reset_allocation_peaks();
  for (unsigned r = 0; r < 3; ++r) LSE_EXPECT_EQ(e.complete("", prompt(2500, r)), 200);
  std::printf("       device path: workspace peak %llu, fallback peak %llu, live %llu (baseline %llu)\n",
              static_cast<unsigned long long>(site_peak(backend::AllocationSite::kWorkspace)),
              static_cast<unsigned long long>(site_peak(backend::AllocationSite::kFallback)),
              static_cast<unsigned long long>(live()), static_cast<unsigned long long>(e.baseline));
  LSE_EXPECT(site_peak(backend::AllocationSite::kWorkspace) > 0);
  LSE_EXPECT_EQ(site_peak(backend::AllocationSite::kFallback), 0u);
  LSE_EXPECT(live() <= e.baseline);
}

LSE_TEST(a_pass_that_leaves_the_device_path_fails_cleanly_when_it_cannot_fit) {
  LSE_EXPECT_OK(place::open_default_devices("capture:0"));
  Engine e;
  LSE_EXPECT_OK(e.open(8, 0));
  if (!e.router) return;
  // The device refuses every launch after the first few, and reports only a
  // little more free memory than the engine already holds.
  backend::CaptureBackend::launched = 0;
  backend::CaptureBackend::refuse_launches_after = 8;
  backend::CaptureBackend::free_bytes = live() + (1u << 20);
  json body{{"model", "memory-fixture"}, {"prompt", prompt(2500, 9)}, {"max_tokens", 4},
            {"temperature", 0}};
  const auto reply = e.router->handle("POST", "/v1/completions", body.dump());
  backend::CaptureBackend::refuse_launches_after = -1;
  backend::CaptureBackend::free_bytes = 0;
  std::printf("       %d: %s\n", reply.status, reply.body.substr(0, 400).c_str());
  LSE_EXPECT(reply.status != 200);
  LSE_EXPECT(reply.body.find("out of GPU memory") != std::string::npos);
  LSE_EXPECT(reply.body.find("per-group fallback") != std::string::npos);
  LSE_EXPECT(reply.body.find("capture device refuses this kernel") != std::string::npos);
  // The engine recovers: the next request runs on the device path again.
  LSE_EXPECT_EQ(e.complete("", prompt(600, 3)), 200);
  LSE_EXPECT(live() <= e.baseline);
}

LSE_TEST_MAIN()
