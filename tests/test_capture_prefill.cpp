// The device-first scheduler path on a host device whose kernels are
// recorded and never run ("capture": Loom emission, no execution), so a wide
// prefill's real allocations can be checked without a GPU: the phase path
// recycles its workspace and gives the device back after the request; and
// when a phase group cannot run, the pass falls back to the per-group path,
// which says so and fails cleanly when its intermediates cannot fit instead
// of running out of memory part way.
#include "harness.hpp"

#include <mach/mach.h>
#include <malloc/malloc.h>

#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <mutex>
#include <condition_variable>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "lse/backend/backend.hpp"
#include "lse/backends/cpu/cpu_backend.hpp"
#include "lse/backends/hrx/arch_database.hpp"
#include "lse/backends/hrx/code_object.hpp"
#include "lse/backends/hrx/device_info.hpp"
#include "lse/backends/hrx/loomc/loom_emitter.hpp"
#include "lse/backends/hrx/loomc/loomc_compiler.hpp"
#include <sys/resource.h>
#include "lse/graph/codegen.hpp"
#include "lse/graph/graph.hpp"
#include "lse/model/config.hpp"
#include "lse/model/hybrid_lm.hpp"
#include "lse/model/registry.hpp"
#include "lse/model/weights.hpp"
#include "lse/place/devices.hpp"
#include "lse/server/router.hpp"
#include "lse/tokenizer/tokenizer.hpp"
#include "lse/lse.h"

using namespace lse;
using namespace lse::graph;
using json = nlohmann::json;

namespace lse::backend {
// LSE_TEST_REAL_COMPILE=1: compile with loomc on the host and report each
// compile's footprint growth and the process peak, to size what an on-device
// compile needs.
struct MeasuredLoomc final : IKernelCompiler {
  mutable LoomcCompiler real;
  Result<CompiledKernel> compile(std::string_view source, std::string_view arch) const override {
    const auto before = footprint_now();
    rusage ru{};
    getrusage(RUSAGE_SELF, &ru);
    const auto peak_before = static_cast<std::uint64_t>(ru.ru_maxrss);
    auto r = real.compile(source, arch);
    getrusage(RUSAGE_SELF, &ru);
    const auto after = footprint_now();
    std::fprintf(stderr, "capture: loomc compile %zu bytes of source: footprint %+.1f MiB, peak RSS %.1f -> %.1f MiB%s%s\n",
                 source.size(), (double(after) - double(before)) / 1048576.0, peak_before / 1048576.0,
                 static_cast<double>(ru.ru_maxrss) / 1048576.0, r.ok() ? "" : ": ",
                 r.ok() ? "" : r.status().to_string().c_str());
    return r;
  }
  bool available() const override { return real.available(); }
  std::string identity() const override { return real.identity(); }
  static std::uint64_t footprint_now() {
    task_vm_info_data_t info{};
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&info), &count);
    return info.phys_footprint;
  }
};
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
  bool joins_run(std::span<const NodePtr> run, const DeviceInfo& d) const override { return real.joins_run(run, d); }
  RunScratch run_scratch(std::span<const NodePtr> run, const DeviceInfo& d) const override { return real.run_scratch(run, d); }
};
struct CaptureBackend : Backend<CaptureBackend> {
  static constexpr std::string_view kName = "capture";
  CpuBackend cpu;
  mutable CaptureEmitter emitter;
  mutable NullCompiler compiler;
  mutable MeasuredLoomc loomc;
  mutable KernelToolchain chain{Dialect::kLoom, &emitter,
      std::getenv("LSE_TEST_REAL_COMPILE") ? static_cast<const IKernelCompiler*>(&loomc) : &compiler};
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
  // Opaque, as device memory is: the host reaches it only through copies
  // (handle is the backing address), so host mirrors, staging and uploads
  // take the paths they take on a GPU.
  static std::byte* at(const DeviceBuffer& b, std::size_t off) {
    return reinterpret_cast<std::byte*>(b.handle) + b.offset + off;
  }
  Result<DeviceBuffer> allocate_impl(std::size_t b, MemoryClass c, Stream s) {
    auto buf = cpu.allocate_impl(b, c, s);
    if (buf.ok() && c == MemoryClass::kDevice) buf->ptr = nullptr;
    return buf;
  }
  void deallocate_impl(DeviceBuffer& b) noexcept { b.storage.reset(); b.handle = 0; b.ptr = nullptr; }
  Result<void*> device_pointer_impl(const DeviceBuffer& b) const { return static_cast<void*>(at(b, 0)); }
  Status copy_h2d_impl(const void* src, DeviceBuffer& dst, std::size_t n, std::size_t off) {
    if (!src || !dst.handle || off + n > dst.size_bytes) return LSE_ERROR(kInvalidArgument, "capture copy_h2d");
    std::memcpy(at(dst, off), src, n);
    return OkStatus();
  }
  Status copy_d2h_impl(const DeviceBuffer& src, void* dst, std::size_t n, std::size_t off) {
    if (!dst || !src.handle || off + n > src.size_bytes) return LSE_ERROR(kInvalidArgument, "capture copy_d2h");
    std::memcpy(dst, at(src, off), n);
    return OkStatus();
  }
  Status copy_peer_impl(const DeviceBuffer& src, DeviceBuffer& dst, std::size_t n, std::size_t so, std::size_t d) {
    if (!src.handle || !dst.handle) return LSE_ERROR(kInvalidArgument, "capture copy without storage");
    std::memmove(at(dst, d), at(src, so), n);
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
  std::printf("       device path: workspace peak %llu, cpu fallbacks %zu, live %llu (baseline %llu)\n",
              static_cast<unsigned long long>(site_peak(backend::AllocationSite::kWorkspace)),
              graph::cpu_fallback_events().size(),
              static_cast<unsigned long long>(live()), static_cast<unsigned long long>(e.baseline));
  LSE_EXPECT(site_peak(backend::AllocationSite::kWorkspace) > 0);
  LSE_EXPECT(graph::cpu_fallback_events().empty());
  LSE_EXPECT(live() <= e.baseline);
}

LSE_TEST(a_refused_device_dispatch_fails_the_request_and_the_engine_recovers) {
  LSE_EXPECT_OK(place::open_default_devices("capture:0"));
  Engine e;
  LSE_EXPECT_OK(e.open(8, 0));
  if (!e.router) return;
  backend::reset_allocation_peaks();
  const auto before = backend::allocation_totals(backend::MemoryClass::kDevice);
  // The device refuses every launch after the first few.
  backend::CaptureBackend::launched = 0;
  backend::CaptureBackend::refuse_launches_after = 8;
  json body{{"model", "memory-fixture"}, {"prompt", prompt(2500, 9)}, {"max_tokens", 4},
            {"temperature", 0}};
  const auto reply = e.router->handle("POST", "/v1/completions", body.dump());
  backend::CaptureBackend::refuse_launches_after = -1;
  std::printf("       %d: %s\n", reply.status, reply.body.substr(0, 500).c_str());
  LSE_EXPECT_EQ(reply.status, 500);
  // The error names the dispatch and carries the device's own error.
  LSE_EXPECT(reply.body.find("device dispatch failed") != std::string::npos);
  LSE_EXPECT(reply.body.find(" group ") != std::string::npos);
  LSE_EXPECT(reply.body.find("nodes:") != std::string::npos);
  LSE_EXPECT(reply.body.find("capture device refuses this kernel") != std::string::npos);
  // No fallback ran: nothing beyond the pass's own workspace was allocated,
  // and no CPU fallback was recorded.
  const auto peak = backend::allocation_totals(backend::MemoryClass::kDevice).peak;
  std::printf("       peak over the request: %llu bytes above load\n",
              static_cast<unsigned long long>(peak - before.live));
  // The first pass allocates the KV pool (256 MiB for this fixture's
  // capacity) and its workspace; a per-node fallback would add every
  // intermediate of the pass on top.
  LSE_EXPECT(peak - before.live < (std::uint64_t{288} << 20));
  LSE_EXPECT(graph::cpu_fallback_events().empty());
  // The engine recovers: the next request runs, and the device holds what it
  // held after load.
  LSE_EXPECT_EQ(e.complete("", prompt(600, 3)), 200);
  LSE_EXPECT(live() <= e.baseline);
}


namespace {
std::uint64_t phys_footprint() {
  task_vm_info_data_t info{};
  mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
  if (task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&info), &count) != KERN_SUCCESS)
    return 0;
  return info.phys_footprint;
}
std::uint64_t malloc_in_use() {
  malloc_statistics_t st{};
  malloc_zone_statistics(nullptr, &st);
  return st.size_in_use;
}
}  // namespace

// Host memory a loaded engine keeps, on a real checkpoint: the process
// footprint less the device memory (which on the capture device is host
// memory too). Gated: it loads a whole model. Point it at checkpoints:
//   LSE_TEST_MODEL_TARGET=.../qwen38-27b-q4 [LSE_TEST_MODEL_DFLASH2=...]
//   LSE_TEST_HOST_BUDGET_MB=256 (default: the host heap a loaded engine may keep)
LSE_TEST(a_loaded_engine_keeps_little_host_memory) {
  const char* target = std::getenv("LSE_TEST_MODEL_TARGET");
  if (target == nullptr) {
    lse::test::skip("set LSE_TEST_MODEL_TARGET (and LSE_TEST_MODEL_DFLASH2) to run");
    return;
  }
  const char* draft = std::getenv("LSE_TEST_MODEL_DFLASH2");
  const char* budget_text = std::getenv("LSE_TEST_HOST_BUDGET_MB");
  const std::uint64_t budget = (budget_text ? std::strtoull(budget_text, nullptr, 10) : 256) << 20;
  const auto footprint_before = phys_footprint();
  const auto heap_before = malloc_in_use();
  const auto device_before = live();
  lse_config cfg;
  lse_config_init(&cfg);
  cfg.model = target;
  if (draft != nullptr) { cfg.dflash2 = 1; cfg.dflash2_model = draft; }
  cfg.kv_len = 32768;
  cfg.kv_cache_dtype = "bf16";
  cfg.pool = "capture:0";
  cfg.dialect = "loom";
  char* err = nullptr;
  lse_engine* engine = lse_open(&cfg, &err);
  if (engine == nullptr) {
    std::printf("       open failed: %s\n", err ? err : "?");
    lse_free(err);
    LSE_EXPECT(engine != nullptr);
    return;
  }
  const auto device = live() - device_before;
  const auto footprint = phys_footprint() - footprint_before;
  const auto host = footprint > device ? footprint - device : 0;
  // On this device the device memory is malloc memory too; what the heap
  // holds beyond it is what the engine keeps on the host.
  const auto heap = malloc_in_use() - heap_before;
  const auto host_heap = heap > device ? heap - device : 0;
  char* status = nullptr;
  (void)lse_status(engine, &status);
  std::string host_report;
  if (status != nullptr) {
    const json s = json::parse(status);
    if (s.contains("host_memory")) host_report = s["host_memory"].dump();
  }
  lse_free(status);
  std::printf("       loaded: footprint +%.0f MiB, device +%.0f MiB; host heap kept %.0f MiB (budget %.0f MiB), "
              "host footprint beyond device %.0f MiB (includes pages the allocator caches)\n"
              "       host_memory: %s\n",
              footprint / 1048576.0, device / 1048576.0, host_heap / 1048576.0, budget / 1048576.0,
              host / 1048576.0, host_report.c_str());
  LSE_EXPECT(host_heap <= budget);
  if (std::getenv("LSE_TEST_REAL_COMPILE")) {
    std::string text;
    for (int i = 0; i < 3500; ++i) text += (i % 7 ? " lemon" : " tree");
    const json body{{"prompt", text}, {"max_tokens", 4}, {"temperature", 0}};
    const std::string payload = body.dump();
    struct Wait { std::mutex m; std::condition_variable cv; bool done = false; } wait;
    (void)lse_request(engine, "POST", "/v1/completions", payload.c_str(), payload.size(),
        [](void* u, lse_request_id, lse_event ev, int status, const char* data, size_t len) {
          if (ev == LSE_EVENT_CHUNK) return;
          std::printf("       request %d: %.*s\n", status, int(std::min<size_t>(len, 300)), data ? data : "");
          auto* w = static_cast<Wait*>(u);
          std::lock_guard l(w->m); w->done = true; w->cv.notify_all();
        }, &wait, nullptr);
    std::unique_lock l(wait.m);
    wait.cv.wait(l, [&] { return wait.done; });
  }
  if (const char* pause = std::getenv("LSE_TEST_PAUSE_SECONDS")) {
    std::printf("       pid %d paused\n", getpid());
    std::fflush(stdout);
    sleep(static_cast<unsigned>(std::atoi(pause)));
  }
  lse_close(engine);
}

LSE_TEST_MAIN()
