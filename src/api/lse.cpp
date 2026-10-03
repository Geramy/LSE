// The in-process engine behind include/lse/lse.h. lse_open is what lse-server
// does between parsing its command line and listening; lse_request hands a
// request to the same Router the HTTP server forwards to.
#include "lse/lse.h"

#include <unistd.h>
#if defined(__APPLE__)
#include <TargetConditionals.h>
#endif

#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "lse/backend/backend.hpp"
#include "lse/core/progress.hpp"
#include "lse/graph/graph.hpp"
#include "lse/graph/jit.hpp"
#include "lse/model/config.hpp"
#include "lse/model/dflash2.hpp"
#include "lse/model/inspect.hpp"
#include "lse/model/mtp.hpp"
#include "lse/model/registry.hpp"
#include "lse/model/weights.hpp"
#include "lse/place/devices.hpp"
#include "lse/runtime/generator.hpp"
#include "lse/server/http_server.hpp"
#include "lse/server/in_process.hpp"
#include "lse/server/router.hpp"
#include "lse/tokenizer/tokenizer.hpp"

#ifndef LSE_ENGINE_VERSION
#define LSE_ENGINE_VERSION "unknown"
#endif

using json = nlohmann::json;

namespace {

using namespace lse;

thread_local lse_result t_last_error = LSE_OK;

char* dup_string(const std::string& s) {
  char* out = static_cast<char*>(std::malloc(s.size() + 1));
  if (out != nullptr) std::memcpy(out, s.c_str(), s.size() + 1);
  return out;
}

std::string str(const char* s) { return s != nullptr ? std::string(s) : std::string(); }

// ---------------------------------------------------------------------------
// Log capture. The engine reports through stderr; while a callback is set,
// fd 2 is a pipe whose reader hands each line to the callback and still copies
// it to the stderr the process started with.
// ---------------------------------------------------------------------------
struct LogCapture {
  std::mutex lock;
  // Held while a callback runs and while the callback changes, so once
  // lse_set_log_callback returns no call into the previous one is running.
  std::recursive_mutex calling;
  lse_log_cb cb = nullptr;
  void* user = nullptr;
  int saved_stderr = -1;
  int read_end = -1;
  std::thread reader;

  void deliver(const std::string& line, lse_log_level level = LSE_LOG_INFO) {
    std::lock_guard call(calling);
    lse_log_cb f;
    void* u;
    {
      std::lock_guard held(lock);
      f = cb;
      u = user;
    }
    if (f != nullptr) f(u, level, line.c_str());
  }

  void read_loop(int fd, int echo) {
    std::string pending;
    char buf[4096];
    for (;;) {
      const ssize_t n = ::read(fd, buf, sizeof buf);
      if (n <= 0) break;
      if (echo >= 0) (void)!::write(echo, buf, static_cast<std::size_t>(n));
      pending.append(buf, static_cast<std::size_t>(n));
      std::size_t at;
      while ((at = pending.find('\n')) != std::string::npos) {
        deliver(pending.substr(0, at));
        pending.erase(0, at + 1);
      }
    }
    if (!pending.empty()) deliver(pending);
    ::close(fd);
  }

  void set(lse_log_cb f, void* u) {
    {
      std::lock_guard call(calling);
      std::lock_guard held(lock);
      cb = f;
      user = u;
    }
    std::unique_lock held(lock);
    if (f != nullptr && saved_stderr < 0) {
      int fds[2];
      if (::pipe(fds) != 0) return;
      std::fflush(stderr);
      saved_stderr = ::dup(STDERR_FILENO);
      ::dup2(fds[1], STDERR_FILENO);
      ::close(fds[1]);
      read_end = fds[0];
      reader = std::thread([this, fd = read_end, echo = saved_stderr] { read_loop(fd, echo); });
    } else if (f == nullptr && saved_stderr >= 0) {
      std::fflush(stderr);
      // Restoring fd 2 drops the pipe's last writer, so the reader drains
      // what is left and ends.
      ::dup2(saved_stderr, STDERR_FILENO);
      const int saved = saved_stderr;
      saved_stderr = -1;
      read_end = -1;
      std::thread finishing = std::move(reader);
      held.unlock();
      if (finishing.joinable()) finishing.join();
      ::close(saved);
    }
  }
};

LogCapture& log_capture() {
  static LogCapture* capture = new LogCapture;  // outlives static destructors
  return *capture;
}

void log_error(const std::string& line) { log_capture().deliver(line, LSE_LOG_ERROR); }

// The load state reported before an engine exists.
std::atomic<int> g_open_state{0};  // 0 idle, 1 loading, 2 ready, 3 failed, 4 closing
std::mutex g_open_error_lock;
std::string g_open_error;
std::atomic<bool> g_engine_open{false};

const char* state_name(int s) {
  switch (s) {
    case 1: return "loading";
    case 2: return "ready";
    case 3: return "failed";
    case 4: return "closing";
    default: return "idle";
  }
}

struct OpenError {
  lse_result code;
  std::string message;
};

}  // namespace

// ---------------------------------------------------------------------------
// The engine: everything lse-server holds between loading and exiting, in the
// order it built them, so they are destroyed in the reverse order it did.
// ---------------------------------------------------------------------------
struct lse_engine {
  std::string model_id;
  std::string host = "127.0.0.1";
  int port = 8080;

  std::optional<model::SafeTensors> weights;
  std::optional<model::Config> cfg;
  std::unique_ptr<model::HybridLM> lm;
  std::optional<model::WeightBinder> binder;
  std::optional<tokenizer::Tokenizer> tok;
  std::unique_ptr<server::Router> router;
  std::unique_ptr<model::MtpModule> mtp;
  std::unique_ptr<model::DFlash2Module> dflash2;

  // HTTP adapter.
  std::mutex http_lock;
  std::unique_ptr<server::HttpServer> http;
  std::thread http_thread;
  Status http_status = OkStatus();

  // In-process requests: the same router, without a socket.
  std::unique_ptr<server::InProcess> requests;
  std::atomic<bool> closing{false};
};

namespace {

// Everything lse-server checks and loads after its command line is parsed.
// Messages are the ones lse-server prints after "lse-server: ".
std::optional<OpenError> open_engine(const lse_config& c, lse_engine& e) {
  using lse::Status;
  const auto invalid = [](std::string m) {
    return std::optional<OpenError>(OpenError{LSE_ERR_INVALID_ARGUMENT, std::move(m)});
  };
  const auto fail = [](const Status& s, const char* what) {
    return std::optional<OpenError>(
        OpenError{LSE_ERR_FAILED, std::string(what) + ": " + std::string(s.message())});
  };

  server::ServerOptions opt;
  opt.max_tokens_cap = c.max_tokens;
  if (c.mtp_depth != 0) {
    opt.mtp_depth = c.mtp_depth;
    if (!runtime::valid_mtp_depth(opt.mtp_depth))
      return invalid("MTP depth must be an integer from 1 to 7");
  }
  if (c.batch_size != 0) opt.prefill.batch_size = c.batch_size;
  if (c.ubatch_size != 0) opt.prefill.ubatch_size = c.ubatch_size;
  if (!runtime::PrefillBatch::valid_size(opt.prefill.batch_size) ||
      !runtime::PrefillBatch::valid_size(opt.prefill.ubatch_size))
    return invalid("prefill batch size must be a power of two from 128 to 4096");
  if (c.has_temperature &&
      (!std::isfinite(c.temperature) || c.temperature < 0 || c.temperature > 2))
    return invalid("temperature must be a finite number from 0 to 2");
  if (c.shutdown_grace_seconds < 1 || c.shutdown_grace_seconds > 600)
    return invalid("shutdown grace must be an integer from 1 to 600 seconds");
  const std::string dialect = str(c.dialect);
  if (!dialect.empty() && !graph::dialect_from_name(dialect).has_value())
    return invalid("no dialect is spelled '" + dialect + "'");
  const std::string kv_cache_dtype = str(c.kv_cache_dtype);
  if (!kv_cache_dtype.empty()) {
    const auto parsed = kv::cache_dtype_from_string(kv_cache_dtype);
    if (!parsed.ok()) return invalid(std::string(parsed.status().message()));
  }
  if (c.flashprefill_v2 < -1 || c.flashprefill_v2 > 1)
    return invalid("--FlashPrefillV2 must be on or off");

  const std::string model = str(c.model);
  const bool no_mtp = c.no_mtp != 0;
  const bool dflash2_on = c.dflash2 != 0;
  const std::string mtp_path = str(c.mtp_path);
  const std::string dflash2_model =
      c.dflash2_model != nullptr ? std::string(c.dflash2_model) : "incoai/Qwen3.8-27B-DFlash2";
  const std::string tokenizer_repo = c.tokenizer_repo != nullptr
                                         ? std::string(c.tokenizer_repo)
                                         : std::string(tokenizer::kQwen36TokenizerRepo);
  const std::string attention_calibration = str(c.attention_calibration);
  const std::optional<float> temperature_override =
      c.has_temperature ? std::optional<float>(c.temperature) : std::nullopt;
  const std::optional<bool> flashprefill_toggle =
      c.flashprefill_v2 < 0 ? std::nullopt : std::optional<bool>(c.flashprefill_v2 == 1);
  ops::SparseAttentionOptions sparse_attention;
  bool prefill_explicit = false;
  for (const auto& [flag, value] : {std::pair{"--attention-prefill", c.attention_prefill},
                                    std::pair{"--attention-decode", c.attention_decode}}) {
    if (value == nullptr) continue;
    const std::string a = flag;
    const std::string mode = value;
    if (mode != "dense" && mode != "blasst" &&
        !(a == "--attention-prefill" && mode == "flashprefill-v2"))
      return invalid("attention mode must be dense or blasst; flashprefill-v2 is prefill only");
    auto& phase = a == "--attention-prefill" ? sparse_attention.prefill : sparse_attention.decode;
    phase.blasst = mode == "blasst";
    phase.flashprefill = mode == "flashprefill-v2";
    if (a == "--attention-prefill") prefill_explicit = true;
  }

  if (!opt.prefill.valid()) return invalid("--ubatch-size must not exceed --batch-size");
  std::fprintf(stderr, "lse-server: prefill batch=%u ubatch=%u\n",
               opt.prefill.batch_size, opt.prefill.ubatch_size);

  if (prefill_explicit && flashprefill_toggle.has_value() &&
      sparse_attention.prefill.flashprefill != *flashprefill_toggle)
    return invalid("conflicting --FlashPrefillV2 and --attention-prefill settings");
  if (flashprefill_toggle.value_or(false)) sparse_attention.prefill.flashprefill = true;
  if (sparse_attention.prefill.flashprefill)
    sparse_attention.prefill.scale = ops::kFlashPrefillDefaultAlpha;
  const bool automatic_flashprefill = !prefill_explicit && !flashprefill_toggle.has_value();
  const bool sparse_requested = sparse_attention.prefill.enabled() || sparse_attention.decode.enabled();
  if (sparse_requested) {
    if ((!no_mtp || dflash2_on) &&
        (sparse_attention.prefill.blasst || sparse_attention.decode.enabled()))
      return invalid("BLASST requires --no-mtp and DFlash2 off; FlashPrefill V2 supports prompt prefill with dense speculative verification");
    if (attention_calibration.empty() &&
        (sparse_attention.prefill.blasst || sparse_attention.decode.blasst))
      return invalid("BLASST requires --attention-calibration");
    if (dialect != "loom")
      return invalid("experimental sparse attention requires --dialect loom");
    if (!attention_calibration.empty()) try {
      std::ifstream input(attention_calibration);
      const auto calibration = nlohmann::json::parse(input);
      if (calibration.at("version").get<int>() != 1)
        throw std::runtime_error("unsupported calibration version");
      if (calibration.at("model").get<std::string>() != model)
        throw std::runtime_error("calibration model must match --model exactly");
      auto load_phase = [&](const char* name, ops::SparseAttentionPhase& phase) {
        if (!phase.enabled()) return;
        const auto& value = calibration.at(name).at("scale");
        if (!value.is_number()) throw std::runtime_error("phase scale must be numeric");
        phase.scale = value.get<float>();
        if (!std::isfinite(phase.scale) || phase.scale < 0 || (phase.flashprefill && phase.scale > 1))
          throw std::runtime_error("phase scale must be finite, nonnegative, and at most 1 for flashprefill-v2");
      };
      load_phase("prefill", sparse_attention.prefill);
      load_phase("decode", sparse_attention.decode);
    } catch (const std::exception& error) {
      return invalid(std::string("invalid attention calibration: ") + error.what());
    }
    std::fprintf(stderr, "lse-server: experimental sparse attention prefill=%s scale=%g decode=%s scale=%g; approximate attention, full KV retained\n",
        sparse_attention.prefill.flashprefill ? "flashprefill-v2" : sparse_attention.prefill.blasst ? "blasst" : "dense", double(sparse_attention.prefill.scale),
        sparse_attention.decode.blasst ? "blasst" : "dense", double(sparse_attention.decode.scale));
  }
  if (model.empty()) return invalid("no model. Pass --model or set $LSE_MODEL.");
  opt.model_id = c.served_name != nullptr && *c.served_name ? std::string(c.served_name) : model;
  opt.api_key = str(c.api_key);
  if (c.host != nullptr && *c.host) opt.host = c.host;
  opt.port = c.port;
  e.model_id = opt.model_id;
  e.host = opt.host;
  e.port = opt.port;

  // Bring the devices up BEFORE the weights are bound. Without this the
  // model loads against no backend: nothing reaches the GPU, the forward
  // pass runs through the host interpreter, and from outside it looks like
  // the server never loaded a model at all.
  progress::begin("opening_devices", str(c.pool));
  if (const Status opened = place::open_default_devices(str(c.pool)); !opened.ok()) {
    return fail(opened, "opening the device set");
  }
  place::Devices* devices = place::default_devices();
  if (devices == nullptr || devices->size() == 0) {
    return fail(LSE_ERROR(kDeviceError, "no device came up"), "opening the device set");
  }
  if (sparse_requested) {
    for (std::size_t i = 0; i < devices->size(); ++i) {
      const auto& info = devices->device(i).device_info();
      if (info.arch != "gfx1201" || info.wavefront_size != 32)
        return invalid("experimental sparse attention is qualified only for gfx1201 Wave32");
    }
  }
  backend::IBackend& first_device = devices->device(devices->primary());
  if (first_device.emitter() == nullptr) {
    // Say which backend declined and why. Without the reason this reads as the
    // server choosing the host interpreter, when what happened is that every
    // code-generating backend refused and only the fallback was left.
    std::fprintf(stderr,
                 "lse-server: no code-generating backend came up (%s); running on '%s' through the host interpreter, which is far slower\n",
                 std::string(devices->declined()).c_str(),
                 std::string(first_device.name()).c_str());
  } else {
    std::fprintf(stderr, "lse-server: device %s\n", std::string(first_device.name()).c_str());
  }

  graph::Scheduler* sched = graph::default_scheduler();
  if (sched == nullptr) {
    return fail(LSE_ERROR(kDeviceError, "no scheduler could be built"),
                "selecting the kernel dialect");
  }
  if (!dialect.empty()) {
    const graph::Dialect want = *graph::dialect_from_name(dialect);
    sched->set_dialect(want);
    for (std::size_t i = 0; i < devices->size(); ++i) {
      const graph::KernelToolchain* tc = sched->toolchain(i);
      if (tc == nullptr) continue;
      std::fprintf(stderr, "lse-server: %s generates %s%s\n",
                   std::string(devices->device(i).name()).c_str(),
                   std::string(to_string(tc->dialect)).c_str(),
                   tc->dialect == want ? "" : " -- it does not declare the one asked for");
    }
  }

  std::fprintf(stderr, "lse-server: loading %s\n", model.c_str());
  progress::begin("resolving_model", model);
  std::string cache_dir = str(c.cache_dir);
#if defined(__APPLE__) && TARGET_OS_IOS
  // An app can write only inside its container, and $HOME is the container's
  // root, so the dot-directories the engine defaults to elsewhere are not
  // writable. Kernel, profile and download caches go to Library/Caches unless
  // the caller chose otherwise.
  if (const char* home = std::getenv("HOME"); home != nullptr && *home) {
    const std::string caches = std::string(home) + "/Library/Caches";
    if (std::getenv("XDG_CACHE_HOME") == nullptr) ::setenv("XDG_CACHE_HOME", caches.c_str(), 0);
    if (cache_dir.empty() && std::getenv("LSE_CACHE_DIR") == nullptr) cache_dir = caches + "/lse/kernels";
  }
#endif
  const Status cache_status = graph::prepare_cache_dir(cache_dir);
  if (!cache_status.ok()) return fail(cache_status, "kernel cache");
  std::fprintf(stderr, "lse-server: kernel cache %s\n", graph::default_cache_dir().c_str());
  auto paths = model::resolve_model(model);
  if (!paths.ok()) return fail(paths.status(), "resolving the model");

  auto cfg = model::Config::from_json_file(paths->config);
  if (!cfg.ok()) return fail(cfg.status(), "reading the config");
  cfg->sparse_attention = sparse_attention;
  if (sparse_requested && cfg->attn_head_dim != 256)
    return invalid("experimental sparse attention requires head dimension 256");
  if (temperature_override) cfg->sampling_defaults.temperature = *temperature_override;
  std::fprintf(stderr, "sampling defaults: temperature=%.3g top_k=%d top_p=%.3g\n",
               static_cast<double>(cfg->sampling_defaults.temperature),
               cfg->sampling_defaults.top_k,
               static_cast<double>(cfg->sampling_defaults.top_p));
  if (c.kv_len > 0) cfg->kv_length = c.kv_len;
  if (!kv_cache_dtype.empty())
    cfg->kv_cache_dtype = kv::cache_dtype_from_string(kv_cache_dtype).release();
  std::fprintf(stderr, "KV cache: %s; attention accumulation fp32 (target/MTP paged cache)\n",
               std::string(kv::to_string(cfg->kv_cache_dtype)).c_str());

  progress::begin("mapping_weights", paths->weights);
  auto weights = paths->weights.ends_with(".index.json")
                     ? model::SafeTensors::open_sharded(paths->weights)
                     : model::SafeTensors::open(paths->weights);
  if (!weights.ok()) return fail(weights.status(), "opening the weights");
  e.weights.emplace(weights.release());
  e.cfg.emplace(cfg.release());
  model::Config& config = *e.cfg;

  const std::string mtp_where =
      (no_mtp || dflash2_on) ? std::string()
             : (mtp_path.empty() ? model::MtpModule::find_beside(model) : mtp_path);
  if (automatic_flashprefill) {
    auto arch = model::detect_architecture(config, *e.weights);
    bool supported = arch.ok() && (*arch)->name == "qwen3.5" &&
                     config.attn_head_dim == 256 &&
                     (config.kv_cache_dtype == kv::CacheDType::kBF16 ||
                      config.kv_cache_dtype == kv::CacheDType::kF32);
    for (std::size_t i = 0; i < devices->size(); ++i) {
      const auto& info = devices->device(i).device_info();
      const auto* tc = sched->toolchain(i);
      supported = supported && info.arch == "gfx1201" && info.wavefront_size == 32 &&
                  tc != nullptr && tc->dialect == graph::Dialect::kLoom;
    }
    if (supported) {
      config.sparse_attention.prefill = {false, ops::kFlashPrefillDefaultAlpha, true};
      if (!attention_calibration.empty()) {
        try {
          std::ifstream input(attention_calibration);
          const auto calibration = nlohmann::json::parse(input);
          if (calibration.at("version").get<int>() != 1 ||
              calibration.at("model").get<std::string>() != model)
            throw std::runtime_error("calibration version/model mismatch");
          const auto& value = calibration.at("prefill").at("scale");
          if (!value.is_number()) throw std::runtime_error("phase scale must be numeric");
          const float alpha = value.get<float>();
          if (!std::isfinite(alpha) || alpha < 0 || alpha > 1)
            throw std::runtime_error("FlashPrefill V2 alpha must be in [0, 1]");
          config.sparse_attention.prefill.scale = alpha;
        } catch (const std::exception& error) {
          return invalid(std::string("invalid attention calibration: ") + error.what());
        }
      }
      std::fprintf(stderr, "lse-server: FlashPrefillV2=on alpha=%g (default); disable with --FlashPrefillV2=off\n",
                   double(config.sparse_attention.prefill.scale));
    } else {
      std::fputs("lse-server: FlashPrefillV2 inactive for this configuration; prefill=dense\n", stderr);
    }
  } else if (flashprefill_toggle == false && !sparse_attention.prefill.enabled()) {
    std::fputs("lse-server: FlashPrefillV2=off; prefill=dense\n", stderr);
  }

  progress::begin("building_model");
  auto built = model::build_model(config, *e.weights, "");
  if (!built.ok()) return fail(built.status(), "building the model");
  e.lm = built.release();

  progress::begin("loading_weights", paths->weights);
  e.binder.emplace(*e.weights, &config.quantization);
  if (const Status s = e.lm->load(*e.binder); !s.ok()) {
    return fail(s, "binding the weights");
  }

  progress::begin("loading_tokenizer");
  const std::string tok_dir = paths->weights.substr(0, paths->weights.find_last_of('/'));
  auto tok = tokenizer::Tokenizer::for_model_dir(tok_dir, tokenizer_repo);
  if (!tok.ok()) return fail(tok.status(), "loading the tokenizer");
  e.tok.emplace(tok.release());

  // Speculative decoding when the checkpoint ships a module, exactly as the
  // CLI resolves it.
  if (!mtp_where.empty()) {
    progress::begin("loading_mtp", mtp_where);
    auto opened = model::MtpModule::open(mtp_where, config, *e.lm);
    if (opened.ok()) {
      e.mtp = opened.release();
      opt.draft_path = mtp_where;
      std::fprintf(stderr, "lse-server: MTP depth %u from %s\n", opt.mtp_depth, mtp_where.c_str());
    } else if (!mtp_path.empty()) {
      // Named explicitly and it did not load: that is an error, where a
      // module merely found beside the model is not.
      return fail(opened.status(), "loading the MTP module");
    } else {
      std::fprintf(stderr, "lse-server: MTP unavailable: %s\n", opened.status().to_string().c_str());
    }
  }

  if (dflash2_on) {
    progress::begin("loading_dflash2", dflash2_model);
    auto opened = model::DFlash2Module::open(dflash2_model, config, *e.lm);
    if (!opened.ok()) return fail(opened.status(), "loading DFlash2");
    e.dflash2 = opened.release();
    opt.draft_path = dflash2_model;
    std::fprintf(stderr, "lse-server: DFlash2 block %u from %s\n", e.dflash2->block_size(),
                 dflash2_model.c_str());
  }

  // What /v1/lse/model_info and /v1/lse/estimate describe: this model, where
  // it was loaded from, and the device it runs on.
  opt.model_path = model;
  opt.device_arch = first_device.device_info().arch;
  {
    const graph::KernelToolchain* tc = sched->toolchain(devices->primary());
    opt.fragmented_kv = tc != nullptr && tc->dialect == graph::Dialect::kLoom;
  }
  e.router = std::make_unique<server::Router>(*e.lm, *e.tok, opt);
  if (e.mtp) e.router->use_mtp(*e.mtp);
  if (e.dflash2) e.router->use_dflash2(*e.dflash2);
  e.requests = std::make_unique<server::InProcess>(*e.router);
  progress::begin("ready", opt.model_id);
  return std::nullopt;
}

}  // namespace

namespace {

lse_result answer(Result<json> got, char** json_out, char** err) {
  if (!got.ok()) {
    if (err != nullptr) *err = dup_string(std::string(got.status().message()));
    const auto code = got.status().code();
    return code == StatusCode::kInvalidArgument || code == StatusCode::kOutOfRange ||
                   code == StatusCode::kNotFound
               ? LSE_ERR_INVALID_ARGUMENT
               : LSE_ERR_FAILED;
  }
  *json_out = dup_string(got->dump());
  return *json_out != nullptr ? LSE_OK : LSE_ERR_FAILED;
}

// lse_estimate's reading of an lse_config: the same defaults lse_open applies.
Result<model::MemoryPlanRequest> plan_request(const lse_config& c, const char* options) {
  model::MemoryPlanRequest r;
  r.model = str(c.model);
  if (r.model.empty()) return LSE_ERROR(kInvalidArgument, "no model. Set lse_config.model.");
  if (c.mtp_depth != 0) r.mtp_depth = c.mtp_depth;
  if (c.batch_size != 0) r.batch_size = c.batch_size;
  if (c.ubatch_size != 0) r.ubatch_size = c.ubatch_size;
  if (c.kv_len > 0) r.kv_len = c.kv_len;
  if (c.kv_cache_dtype != nullptr && *c.kv_cache_dtype) {
    LSE_ASSIGN_OR(r.kv_cache_dtype, kv::cache_dtype_from_string(c.kv_cache_dtype));
  }
  if (c.dflash2 != 0) {
    r.draft = model::DraftKind::kDFlash2;
    r.draft_path = c.dflash2_model != nullptr ? std::string(c.dflash2_model)
                                              : "incoai/Qwen3.8-27B-DFlash2";
  } else if (c.no_mtp == 0) {
    r.draft = model::DraftKind::kMtp;
    r.draft_path = str(c.mtp_path);
  }
  // Loom keeps K/V in fragments; HIP keeps contiguous pools. With no dialect
  // named, the device's first toolchain decides, which is HIP only where this
  // build has COMGR.
  const std::string dialect = str(c.dialect);
#if defined(LSE_HAVE_COMGR) && LSE_HAVE_COMGR
  r.fragmented_kv = dialect == "loom";
#else
  r.fragmented_kv = dialect != "hip";
#endif
  if (options != nullptr && *options) {
    const json o = json::parse(options, nullptr, false);
    if (o.is_discarded() || !o.is_object())
      return LSE_ERROR(kInvalidArgument, "estimate options must be a JSON object");
    try {
      if (o.contains("context_tokens")) r.context_tokens = o["context_tokens"].get<std::int32_t>();
      if (o.contains("sequences")) r.sequences = o["sequences"].get<std::int32_t>();
      if (o.contains("device_arch")) r.device_arch = o["device_arch"].get<std::string>();
      if (o.contains("device_memory_bytes"))
        r.device_memory_bytes = o["device_memory_bytes"].get<std::uint64_t>();
      if (o.contains("kv_storage")) {
        const auto storage = o["kv_storage"].get<std::string>();
        if (storage != "fragmented" && storage != "contiguous")
          return LSE_ERROR(kInvalidArgument, "kv_storage must be fragmented or contiguous");
        r.fragmented_kv = storage == "fragmented";
      }
    } catch (const json::exception& e) {
      return LSE_ERROR(kInvalidArgument, "invalid estimate option: ", e.what());
    }
  }
  return r;
}

}  // namespace

extern "C" {

const char* lse_version(void) { return LSE_ENGINE_VERSION; }
uint32_t lse_abi_version(void) { return LSE_ABI_VERSION; }

void lse_config_init(lse_config* cfg) {
  if (cfg == nullptr) return;
  std::memset(cfg, 0, sizeof *cfg);
  cfg->abi_version = LSE_ABI_VERSION;
  cfg->struct_size = sizeof *cfg;
  cfg->mtp_depth = 3;
  cfg->flashprefill_v2 = -1;
  cfg->batch_size = 1024;
  cfg->ubatch_size = 1024;
  cfg->max_tokens = 4096;
  cfg->port = 8080;
  cfg->shutdown_grace_seconds = 30;
}

void lse_set_log_callback(lse_log_cb cb, void* user) { log_capture().set(cb, user); }

lse_result lse_last_error(void) { return t_last_error; }

lse_engine* lse_open(const lse_config* cfg, char** err) {
  if (err != nullptr) *err = nullptr;
  const auto refuse = [&](lse_result code, const std::string& message) -> lse_engine* {
    t_last_error = code;
    if (err != nullptr) *err = dup_string(message);
    return nullptr;
  };
  if (cfg == nullptr || cfg->abi_version != LSE_ABI_VERSION ||
      cfg->struct_size != sizeof(lse_config))
    return refuse(LSE_ERR_INVALID_ARGUMENT,
                  "lse_config was not initialized by lse_config_init for this ABI");
  bool expected = false;
  if (!g_engine_open.compare_exchange_strong(expected, true))
    return refuse(LSE_ERR_STATE, "an engine is already open in this process");

  g_open_state.store(1);
  auto engine = std::make_unique<lse_engine>();
  std::optional<OpenError> error;
  try {
    error = open_engine(*cfg, *engine);
  } catch (const std::exception& ex) {
    error = OpenError{LSE_ERR_FAILED, std::string("loading: ") + ex.what()};
  }
  if (error) {
    {
      std::lock_guard held(g_open_error_lock);
      g_open_error = error->message;
    }
    g_open_state.store(3);
    progress::begin("failed", error->message);
    log_error("lse: " + error->message);
    engine.reset();
    g_engine_open.store(false);
    return refuse(error->code, error->message);
  }
  g_open_state.store(2);
  t_last_error = LSE_OK;
  return engine.release();
}

void lse_close(lse_engine* e) {
  if (e == nullptr) return;
  g_open_state.store(4);
  e->closing.store(true);
  lse_http_stop(e);
  if (e->requests) e->requests->shutdown();
  (void)lse_http_wait(e, nullptr);
  delete e;
  progress::begin("idle");
  g_open_state.store(0);
  g_engine_open.store(false);
}

lse_result lse_request(lse_engine* e, const char* method, const char* path,
                       const char* json_body, size_t len, lse_response_cb cb,
                       void* user, lse_request_id* out_id) {
  if (e == nullptr || method == nullptr || path == nullptr) return LSE_ERR_INVALID_ARGUMENT;
  if (e->closing.load() || !e->requests) return LSE_ERR_STATE;
  std::string body = json_body != nullptr ? std::string(json_body, len) : std::string();
  const auto id = e->requests->submit(
      method, path, std::move(body),
      [cb, user](std::uint64_t id, server::InProcess::Event event, int status,
                 const std::string* data) {
        if (cb == nullptr) return;
        lse_event kind = LSE_EVENT_ERROR;
        switch (event) {
          case server::InProcess::Event::kResponse: kind = LSE_EVENT_RESPONSE; break;
          case server::InProcess::Event::kChunk: kind = LSE_EVENT_CHUNK; break;
          case server::InProcess::Event::kDone: kind = LSE_EVENT_DONE; break;
          case server::InProcess::Event::kError: kind = LSE_EVENT_ERROR; break;
        }
        cb(user, id, kind, status, data ? data->c_str() : nullptr, data ? data->size() : 0);
      });
  if (id == 0) return LSE_ERR_STATE;
  if (out_id != nullptr) *out_id = id;
  return LSE_OK;
}

lse_result lse_cancel(lse_engine* e, lse_request_id id) {
  if (e == nullptr) return LSE_ERR_INVALID_ARGUMENT;
  if (!e->requests || !e->requests->cancel(id)) return LSE_ERR_STATE;
  return LSE_OK;
}

lse_result lse_status(const lse_engine* e, char** json_out) {
  if (json_out == nullptr) return LSE_ERR_INVALID_ARGUMENT;
  const progress::Snapshot p = progress::current();
  json s{{"version", LSE_ENGINE_VERSION},
         {"abi", LSE_ABI_VERSION},
         {"state", state_name(g_open_state.load())},
         {"phase", p.phase},
         {"detail", p.detail},
         {"progress", p.fraction < 0 ? json(nullptr) : json(p.fraction)}};
  {
    std::lock_guard held(g_open_error_lock);
    if (!g_open_error.empty() && g_open_state.load() == 3) s["error"] = g_open_error;
  }
  {
    const auto device = backend::allocation_totals(backend::MemoryClass::kDevice);
    const auto staging = backend::allocation_totals(backend::MemoryClass::kStaging);
    s["memory"] = {{"device_bytes", device.live},
                   {"device_peak_bytes", device.peak},
                   {"device_allocations", device.allocations},
                   {"staging_bytes", staging.live}};
  }
  if (e != nullptr) {
    auto& engine = const_cast<lse_engine&>(*e);
    s["model"] = engine.model_id;
    if (engine.cfg) {
      s["kv_cache_dtype"] = std::string(kv::to_string(engine.cfg->kv_cache_dtype));
      s["kv_len"] = engine.cfg->kv_capacity();
    }
    if (engine.requests) {
      const auto c = engine.requests->counters();
      s["requests"] = {{"started", c.started},
                       {"active", c.active},
                       {"completed", c.completed},
                       {"failed", c.failed},
                       {"cancelled", c.cancelled}};
    }
    if (engine.router) s["engine"] = json::parse(engine.router->metrics_json());
    std::lock_guard held(engine.http_lock);
    s["http"] = {{"listening", engine.http != nullptr},
                 {"host", engine.host},
                 {"port", engine.port}};
  }
  *json_out = dup_string(s.dump());
  return *json_out != nullptr ? LSE_OK : LSE_ERR_FAILED;
}

lse_result lse_model_info(const char* model, char** json_out, char** err) {
  if (err != nullptr) *err = nullptr;
  if (json_out == nullptr) return LSE_ERR_INVALID_ARGUMENT;
  *json_out = nullptr;
  if (model == nullptr || *model == '\0') {
    if (err != nullptr) *err = dup_string("no model path");
    return LSE_ERR_INVALID_ARGUMENT;
  }
  try {
    return answer(model::model_info(model), json_out, err);
  } catch (const std::exception& ex) {
    if (err != nullptr) *err = dup_string(std::string("inspecting the model: ") + ex.what());
    return LSE_ERR_FAILED;
  }
}

lse_result lse_estimate(const lse_config* cfg, const char* options_json, char** json_out,
                        char** err) {
  if (err != nullptr) *err = nullptr;
  if (json_out == nullptr) return LSE_ERR_INVALID_ARGUMENT;
  *json_out = nullptr;
  if (cfg == nullptr || cfg->abi_version != LSE_ABI_VERSION ||
      cfg->struct_size != sizeof(lse_config)) {
    if (err != nullptr)
      *err = dup_string("lse_config was not initialized by lse_config_init for this ABI");
    return LSE_ERR_INVALID_ARGUMENT;
  }
  try {
    auto request = plan_request(*cfg, options_json);
    if (!request.ok()) return answer(request.status(), json_out, err);
    return answer(model::estimate_memory(*request), json_out, err);
  } catch (const std::exception& ex) {
    if (err != nullptr) *err = dup_string(std::string("estimating: ") + ex.what());
    return LSE_ERR_FAILED;
  }
}

lse_result lse_http_start(lse_engine* e, const char* host, int32_t port, char** err) {
  if (err != nullptr) *err = nullptr;
  if (e == nullptr) return LSE_ERR_INVALID_ARGUMENT;
  std::lock_guard held(e->http_lock);
  if (e->http != nullptr || e->closing.load()) return LSE_ERR_STATE;
  if (host != nullptr && *host) e->host = host;
  if (port > 0) e->port = port;
  auto http = std::make_unique<server::HttpServer>(*e->router, e->host, e->port);
  if (const Status bound = http->bind(); !bound.ok()) {
    if (err != nullptr) *err = dup_string(std::string(bound.message()));
    return LSE_ERR_FAILED;
  }
  e->http = std::move(http);
  e->http_status = OkStatus();
  e->http_thread = std::thread([e, server = e->http.get()] {
    Status served = server->listen();
    std::lock_guard held(e->http_lock);
    e->http_status = served;
  });
  return LSE_OK;
}

void lse_http_stop(lse_engine* e) {
  if (e == nullptr) return;
  std::lock_guard held(e->http_lock);
  if (e->http != nullptr) e->http->stop();
}

lse_result lse_http_wait(lse_engine* e, char** err) {
  if (err != nullptr) *err = nullptr;
  if (e == nullptr) return LSE_ERR_INVALID_ARGUMENT;
  std::thread listener;
  {
    std::lock_guard held(e->http_lock);
    listener = std::move(e->http_thread);
  }
  // httplib joins every request worker before listen returns. Only then may
  // the server, and after it the model, be destroyed.
  if (listener.joinable()) listener.join();
  std::lock_guard held(e->http_lock);
  e->http.reset();
  if (!e->http_status.ok()) {
    if (err != nullptr) *err = dup_string(std::string(e->http_status.message()));
    return LSE_ERR_FAILED;
  }
  return LSE_OK;
}

void lse_free(void* p) { std::free(p); }

}  // extern "C"
