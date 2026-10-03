// lse-server — the /v1 HTTP surface over one loaded model: parse the command
// line into an lse_config, open the engine, serve it over HTTP until a signal.
#include <atomic>
#include <charconv>
#include <cmath>
#include <optional>
#include <locale>
#include <sstream>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <csignal>
#include <string>
#include <thread>

#include "lse/lse.h"
#include "lse/graph/jit.hpp"
#include "lse/graph/graph.hpp"
#include "lse/kv/cache_dtype.hpp"
#include "lse/runtime/generator.hpp"
#include "lse/runtime/prefill_batch.hpp"
#include "lse/server/shutdown.hpp"

namespace {

using namespace lse;

// Lock-free atomic access is signal-safe and also synchronizes with the watcher.
static_assert(std::atomic<bool>::is_always_lock_free);
std::atomic<bool> g_stopping{false};
void on_signal(int) { g_stopping.store(true, std::memory_order_relaxed); }

void usage() {
  std::puts(
      "usage: lse-server [options]\n"
      "\n"
      "  -m, --model NAME     checkpoint directory, .safetensors, or an HF repo\n"
      "                       id (default: $LSE_MODEL)\n"
      "      --host ADDR      address to bind (default 127.0.0.1)\n"
      "      --port N         port to bind (default 8080)\n"
      "      --api-key KEY    require Authorization: Bearer KEY\n"
      "      --served-name ID model id reported by /v1/models (default: the\n"
      "                       model argument)\n"
      "      --temperature F default request temperature, 0..2 (default: model)\n"
      "      --max-tokens N   refuse requests asking for more (default 4096)\n"
      "      --shutdown-grace-seconds N  drain requests before failing (1..600, default 30)\n"
      "      --mtp PATH       multi-token-prediction module (default: the one\n"
      "                       beside the model, when the checkpoint has one)\n"
      "      --mtp-depth N    draft proposals per verifier pass (1..7, default 3)\n"
      "      --dflash2=on     use the DFlash2 block drafter (default off)\n"
      "      --dflash2-model PATH  DFlash2 checkpoint directory or HF repo id\n"
      "      --no-mtp         decode one token per pass, ignoring any\n"
      "                       multi-token-prediction module\n"
      "      --tokenizer REPO HF repo for tokenizer.json when the model\n"
      "                       directory has none\n"
      "      --FlashPrefillV2=off  disable default FlashPrefill V2 prefill (on/off)\n"
      "                       default on for supported HRX/LOOM prompt prefill, including MTP/DFlash2\n"
      "      --attention-prefill MODE  dense, blasst, flashprefill-v2 (experimental)\n"
      "      --attention-decode MODE   dense (default) or blasst (experimental)\n"
      "      --attention-calibration FILE  JSON with version=1 and phase scale values\n"
      "      --kv-cache-dtype TYPE  fp32, fp16, bf16, fp8, bf8\n"
      "                           default: bf16 for BF16 models, fp16 otherwise\n"
      "      --batch-size N   prompt token batch limit (default 1024)\n"
      "      --ubatch-size N  tokens per physical prefill pass (default 1024)\n"
      "                       powers of two from 128 to 4096; ubatch <= batch\n"
      "      --kv-len N       allocate the KV cache for N tokens\n"
      "      --cache-dir PATH kernel cache directory (default ~/.lse/cache)\n"
      "      --pool LIST      device pool, for example hrx:0 or cpu:0\n"
      "      --dialect NAME   source dialect: hip or loom\n"
      "      --model-info     print what the model is (JSON) and exit; reads\n"
      "                       config.json and tensor headers only\n"
      "      --estimate[=JSON]  print the device memory the other options would\n"
      "                       allocate (JSON) and exit, without opening a device;\n"
      "                       JSON may set context_tokens, sequences, device_arch,\n"
      "                       kv_storage and device_memory_bytes\n"
      "  -h, --help           this message\n"
      "\n"
      "Endpoints: GET /health, GET /v1/models, POST /v1/chat/completions,\n"
      "POST /v1/completions, GET /v1/lse/model_info, GET|POST /v1/lse/estimate.\n"
      "Both completion routes stream when the request sets \"stream\": true.");
}

}  // namespace

int main(int argc, char** argv) {
  lse_config cfg;
  lse_config_init(&cfg);
  std::string model = std::getenv("LSE_MODEL") ? std::getenv("LSE_MODEL") : "";
  std::string mtp_path;
  bool no_mtp = false;
  bool dflash2_on = false;
  std::string dflash2_model = "incoai/Qwen3.8-27B-DFlash2";
  std::string tokenizer_repo;
  std::string served_name;
  std::string host;
  std::string api_key;
  std::string pool;
  std::string dialect;
  std::string cache_dir;
  std::int32_t kv_len = 0;
  std::optional<float> temperature_override;
  std::string kv_cache_dtype;
  std::string attention_prefill, attention_decode;
  std::string attention_calibration;
  std::optional<bool> flashprefill_toggle;
  int shutdown_grace_seconds = 30;
  bool model_info = false;
  std::optional<std::string> estimate;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto value = [&](const char* flag) -> std::string {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "lse-server: %s needs a value\n", flag);
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "-h" || a == "--help") { usage(); return 0; }
    else if (a == "-m" || a == "--model") model = value("--model");
    else if (a == "--host") { host = value("--host"); cfg.host = host.c_str(); }
    else if (a == "--port") cfg.port = std::atoi(value("--port").c_str());
    else if (a == "--api-key") api_key = value("--api-key");
    else if (a == "--served-name") served_name = value("--served-name");
    else if (a == "--max-tokens") cfg.max_tokens = std::atoi(value("--max-tokens").c_str());
    else if (a == "--shutdown-grace-seconds") {
      const auto text = value("--shutdown-grace-seconds");
      const auto parsed = std::from_chars(text.data(), text.data() + text.size(),
                                          shutdown_grace_seconds);
      if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() ||
          shutdown_grace_seconds < 1 || shutdown_grace_seconds > 600) {
        std::fputs("lse-server: shutdown grace must be an integer from 1 to 600 seconds\n", stderr);
        return 2;
      }
    }
    else if (a == "--mtp") mtp_path = value("--mtp");
    else if (a == "--mtp-depth") {
      const auto text = value("--mtp-depth");
      const auto parsed = std::from_chars(text.data(), text.data() + text.size(),
                                          cfg.mtp_depth);
      if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() ||
          !runtime::valid_mtp_depth(cfg.mtp_depth)) {
        std::fputs("lse-server: MTP depth must be an integer from 1 to 7\n", stderr);
        return 2;
      }
    }
    else if (a == "--dflash2" || a.starts_with("--dflash2=")) {
      const std::string text = a == "--dflash2"
                                  ? value("--dflash2")
                                  : a.substr(std::string("--dflash2=").size());
      if (text != "on" && text != "off") {
        std::fputs("lse-server: --dflash2 must be on or off\n", stderr);
        return 2;
      }
      dflash2_on = text == "on";
    }
    else if (a == "--dflash2-model") dflash2_model = value("--dflash2-model");
    else if (a == "--no-mtp") no_mtp = true;
    else if (a == "--tokenizer") tokenizer_repo = value("--tokenizer");
    else if (a == "--FlashPrefillV2" || a.starts_with("--FlashPrefillV2=")) {
      const auto text = a == "--FlashPrefillV2"
                            ? value("--FlashPrefillV2")
                            : a.substr(std::string("--FlashPrefillV2=").size());
      if (text != "on" && text != "off") {
        std::fputs("lse-server: --FlashPrefillV2 must be on or off\n", stderr);
        return 2;
      }
      flashprefill_toggle = text == "on";
    }
    else if (a == "--attention-prefill" || a == "--attention-decode") {
      const auto mode = value(a.c_str());
      if (mode != "dense" && mode != "blasst" &&
          !(a == "--attention-prefill" && mode == "flashprefill-v2")) {
        std::fputs("lse-server: attention mode must be dense or blasst; flashprefill-v2 is prefill only\n", stderr);
        return 2;
      }
      (a == "--attention-prefill" ? attention_prefill : attention_decode) = mode;
    }
    else if (a == "--attention-calibration") attention_calibration = value(a.c_str());
    else if (a == "--kv-cache-dtype") {
      kv_cache_dtype = value("--kv-cache-dtype");
      const auto parsed = kv::cache_dtype_from_string(kv_cache_dtype);
      if (!parsed.ok()) { std::fprintf(stderr, "lse-server: %s\n", std::string(parsed.status().message()).c_str()); return 2; }
    }
    else if (a == "--temperature") {
      const auto text = value("--temperature");
      float temperature = 0;
      std::istringstream parsed(text);
      parsed.imbue(std::locale::classic());
      parsed >> std::noskipws >> temperature;
      if (text.empty() || text.front() == '+' || parsed.fail() || !parsed.eof() ||
          !std::isfinite(temperature) || temperature < 0 || temperature > 2) {
        std::fputs("lse-server: temperature must be a finite number from 0 to 2\n", stderr);
        return 2;
      }
      temperature_override = temperature;
    }
    else if (a == "--batch-size" || a == "--ubatch-size") {
      const auto text = value(a.c_str());
      auto& size = a == "--batch-size" ? cfg.batch_size : cfg.ubatch_size;
      const auto parsed = std::from_chars(text.data(), text.data() + text.size(), size);
      if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() ||
          !runtime::PrefillBatch::valid_size(size)) {
        std::fputs("lse-server: prefill batch size must be a power of two from 128 to 4096\n", stderr);
        return 2;
      }
    }
    else if (a == "--kv-len") kv_len = std::atoi(value("--kv-len").c_str());
    else if (a == "--cache-dir" || a.starts_with("--cache-dir=")) {
      cache_dir = a == "--cache-dir" ? value("--cache-dir") : a.substr(12);
      if (cache_dir.empty()) {
        std::fputs("lse-server: --cache-dir needs a nonempty path\n", stderr);
        return 2;
      }
    }
    else if (a == "--model-info") model_info = true;
    else if (a == "--estimate") estimate = std::string();
    else if (a.starts_with("--estimate=")) estimate = a.substr(std::string("--estimate=").size());
    else if (a == "--pool") pool = value("--pool");
    else if (a == "--dialect") {
      dialect = value("--dialect");
      if (!graph::dialect_from_name(dialect).has_value()) {
        std::fprintf(stderr, "lse-server: no dialect is spelled '%s'\n",
                     dialect.c_str());
        return 2;
      }
    }
    else {
      std::fprintf(stderr, "lse-server: unknown option '%s'\n", a.c_str());
      return 2;
    }
  }

  // Everything after the command line — the checks that combine options,
  // the devices, the model — is the engine's, shared with every embedder.
  cfg.model = model.c_str();
  if (!served_name.empty()) cfg.served_name = served_name.c_str();
  if (!api_key.empty()) cfg.api_key = api_key.c_str();
  if (!tokenizer_repo.empty()) cfg.tokenizer_repo = tokenizer_repo.c_str();
  if (!mtp_path.empty()) cfg.mtp_path = mtp_path.c_str();
  cfg.no_mtp = no_mtp;
  cfg.dflash2 = dflash2_on;
  cfg.dflash2_model = dflash2_model.c_str();
  cfg.flashprefill_v2 = flashprefill_toggle ? (*flashprefill_toggle ? 1 : 0) : -1;
  if (!attention_prefill.empty()) cfg.attention_prefill = attention_prefill.c_str();
  if (!attention_decode.empty()) cfg.attention_decode = attention_decode.c_str();
  if (!attention_calibration.empty()) cfg.attention_calibration = attention_calibration.c_str();
  if (!kv_cache_dtype.empty()) cfg.kv_cache_dtype = kv_cache_dtype.c_str();
  cfg.kv_len = kv_len;
  cfg.has_temperature = temperature_override.has_value();
  cfg.temperature = temperature_override.value_or(0.0f);
  if (!pool.empty()) cfg.pool = pool.c_str();
  if (!dialect.empty()) cfg.dialect = dialect.c_str();
  if (!cache_dir.empty()) cfg.cache_dir = cache_dir.c_str();
  cfg.shutdown_grace_seconds = shutdown_grace_seconds;

  char* err = nullptr;
  if (model_info || estimate) {
    // Answered from config.json and the tensor headers: no device is opened.
    char* json = nullptr;
    const lse_result r = model_info
        ? lse_model_info(model.c_str(), &json, &err)
        : lse_estimate(&cfg, estimate->empty() ? nullptr : estimate->c_str(), &json, &err);
    if (r != LSE_OK) {
      std::fprintf(stderr, "lse-server: %s\n", err != nullptr ? err : "failed");
      lse_free(err);
      return r == LSE_ERR_INVALID_ARGUMENT ? 2 : 1;
    }
    std::puts(json);
    lse_free(json);
    return 0;
  }
  lse_engine* engine = lse_open(&cfg, &err);
  if (engine == nullptr) {
    std::fprintf(stderr, "lse-server: %s\n", err != nullptr ? err : "could not open the engine");
    lse_free(err);
    return lse_last_error() == LSE_ERR_INVALID_ARGUMENT ? 2 : 1;
  }

  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);
  server::detail::ShutdownWatch shutdown(
      g_stopping, std::chrono::seconds(shutdown_grace_seconds), [&] { lse_http_stop(engine); });

  std::fprintf(stderr, "lse-server: %s on http://%s:%d\n",
               served_name.empty() ? model.c_str() : served_name.c_str(),
               host.empty() ? "127.0.0.1" : host.c_str(), cfg.port);
  lse_result served = lse_http_start(engine, nullptr, 0, &err);
  // httplib joins every request worker before the listener returns. Only now
  // may the model, cached executables, device allocations and runtime be
  // destroyed.
  if (served == LSE_OK) served = lse_http_wait(engine, &err);
  shutdown.finish();
  if (served != LSE_OK) {
    std::fprintf(stderr, "lse-server: listening: %s\n", err != nullptr ? err : "failed");
    lse_free(err);
    lse_close(engine);
    return 1;
  }
  std::fprintf(stderr, "lse-server: requests drained; releasing model and runtime\n");
  lse_close(engine);
  return 0;
}
