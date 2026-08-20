// lse-server — the /v1 HTTP surface over one loaded model.
#include <cstdio>
#include <cstdlib>
#include <csignal>
#include <cstdlib>
#include <string>
#include <thread>

#include "lse/model/config.hpp"
#include "lse/model/mtp.hpp"
#include "lse/model/registry.hpp"
#include "lse/place/devices.hpp"
#include "lse/model/weights.hpp"
#include "lse/server/http_server.hpp"
#include "lse/tokenizer/tokenizer.hpp"

namespace {

using namespace lse;

server::HttpServer* g_server = nullptr;

void on_signal(int) {
  if (g_server != nullptr) g_server->stop();
}

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
      "      --max-tokens N   refuse requests asking for more (default 4096)\n"
      "      --mtp PATH       multi-token-prediction module (default: the one\n"
      "                       beside the model, when the checkpoint has one)\n"
      "      --tokenizer REPO HF repo for tokenizer.json when the model\n"
      "                       directory has none\n"
      "      --kv-len N       allocate the KV cache for N tokens\n"
      "  -h, --help           this message\n"
      "\n"
      "Endpoints: GET /health, GET /v1/models, POST /v1/chat/completions,\n"
      "POST /v1/completions. Both completion routes stream when the request\n"
      "sets \"stream\": true.");
}

int fail(const Status& s, const char* what) {
  std::fprintf(stderr, "lse-server: %s: %s\n", what, std::string(s.message()).c_str());
  return 1;
}

}  // namespace

int main(int argc, char** argv) {
  server::ServerOptions opt;
  std::string model = std::getenv("LSE_MODEL") ? std::getenv("LSE_MODEL") : "";
  std::string mtp_path;
  std::string tokenizer_repo{tokenizer::kQwen36TokenizerRepo};
  std::string served_name;
  std::string pool;
  std::int32_t kv_len = 0;

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
    else if (a == "--host") opt.host = value("--host");
    else if (a == "--port") opt.port = std::atoi(value("--port").c_str());
    else if (a == "--api-key") opt.api_key = value("--api-key");
    else if (a == "--served-name") served_name = value("--served-name");
    else if (a == "--max-tokens") opt.max_tokens_cap = std::atoi(value("--max-tokens").c_str());
    else if (a == "--mtp") mtp_path = value("--mtp");
    else if (a == "--tokenizer") tokenizer_repo = value("--tokenizer");
    else if (a == "--kv-len") kv_len = std::atoi(value("--kv-len").c_str());
    else if (a == "--pool") pool = value("--pool");
    else {
      std::fprintf(stderr, "lse-server: unknown option '%s'\n", a.c_str());
      return 2;
    }
  }

  if (model.empty()) {
    std::fputs("lse-server: no model. Pass --model or set $LSE_MODEL.\n", stderr);
    return 2;
  }
  opt.model_id = served_name.empty() ? model : served_name;

  // Bring the devices up BEFORE the weights are bound. Without this the
  // model loads against no backend: nothing reaches the GPU, the forward
  // pass runs through the host interpreter, and from outside it looks like
  // the server never loaded a model at all.
  if (const Status opened = place::open_default_devices(pool); !opened.ok()) {
    return fail(opened, "opening the device set");
  }
  place::Devices* devices = place::default_devices();
  if (devices == nullptr || devices->size() == 0) {
    return fail(LSE_ERROR(kDeviceError, "no device came up"),
                "opening the device set");
  }
  backend::IBackend& first_device = devices->device(devices->primary());
  if (first_device.emitter() == nullptr) {
    std::fprintf(stderr,
                 "lse-server: no code-generating backend came up; running on '%s' through the host interpreter, which is far slower\n",
                 std::string(first_device.name()).c_str());
  } else {
    std::fprintf(stderr, "lse-server: device %s\n",
                 std::string(first_device.name()).c_str());
  }

  std::fprintf(stderr, "lse-server: loading %s\n", model.c_str());
  auto paths = model::resolve_model(model);
  if (!paths.ok()) return fail(paths.status(), "resolving the model");

  auto cfg = model::Config::from_json_file(paths->config);
  if (!cfg.ok()) return fail(cfg.status(), "reading the config");
  if (kv_len > 0) cfg->kv_length = kv_len;

  auto weights = paths->weights.ends_with(".index.json")
                     ? model::SafeTensors::open_sharded(paths->weights)
                     : model::SafeTensors::open(paths->weights);
  if (!weights.ok()) return fail(weights.status(), "opening the weights");

  auto built = model::build_model(*cfg, *weights, "");
  if (!built.ok()) return fail(built.status(), "building the model");
  std::unique_ptr<model::HybridLM> lm = built.release();

  model::WeightBinder binder(*weights, &cfg->quantization);
  if (const Status s = lm->load(binder); !s.ok()) {
    return fail(s, "binding the weights");
  }

  const std::string tok_dir = paths->weights.substr(0, paths->weights.find_last_of('/'));
  auto tok = tokenizer::Tokenizer::for_model_dir(tok_dir, tokenizer_repo);
  if (!tok.ok()) return fail(tok.status(), "loading the tokenizer");

  server::HttpServer http(*lm, *tok, opt);

  // Speculative decoding when the checkpoint ships a module, exactly as the
  // CLI resolves it.
  std::unique_ptr<model::MtpModule> mtp;
  const std::string mtp_where =
      mtp_path.empty() ? model::MtpModule::find_beside(model) : mtp_path;
  if (!mtp_where.empty()) {
    auto opened = model::MtpModule::open(mtp_where, *cfg, *lm);
    if (opened.ok()) {
      mtp = opened.release();
      http.use_mtp(*mtp);
      std::fprintf(stderr, "lse-server: speculative decoding from %s\n",
                   mtp_where.c_str());
    } else if (!mtp_path.empty()) {
      // Named explicitly and it did not load: that is an error, where a
      // module merely found beside the model is not.
      return fail(opened.status(), "loading the MTP module");
    }
  }

  g_server = &http;
  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);

  std::fprintf(stderr, "lse-server: %s on http://%s:%d\n", opt.model_id.c_str(),
               opt.host.c_str(), opt.port);
  const Status served = http.listen();
  if (!served.ok()) return fail(served, "listening");

  // Leave without running the static destructors. Tearing the device down
  // while the runtime still holds threads has been seen to wedge: the device
  // closes, one worker spins and the main thread waits on a futex that never
  // fires, leaving a process that answers nothing and still holds the model's
  // memory. Nothing here owns state that outlives the process, so the kernel
  // reclaiming it is both correct and the only exit that cannot hang.
  std::fflush(nullptr);
  std::fprintf(stderr, "lse-server: stopped\n");
  std::_Exit(0);
}
