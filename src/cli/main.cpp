// lse — load a checkpoint, tokenize a prompt, stream the continuation.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "lse/backend/backend.hpp"
#include "lse/core/debug.hpp"
#include "lse/graph/graph.hpp"
#include "lse/graph/jit.hpp"
#include "lse/ir/pass/pass.hpp"
#include "lse/model/config.hpp"
#include "lse/model/registry.hpp"
#include "lse/model/weights.hpp"
#include "lse/place/devices.hpp"
#include "lse/probe/pool.hpp"
#include "lse/probe/profile_store.hpp"
#include "lse/runtime/batch.hpp"
#include "lse/runtime/generator.hpp"
#include "lse/tokenizer/tokenizer.hpp"

namespace {

using namespace lse;

struct Options {
  std::string model;
  std::string prompt = "Hello";
  runtime::SamplingParams sampling;
  runtime::GenerationLimits limits;
  std::string tokenizer_repo{tokenizer::kQwen36TokenizerRepo};
  // Empty means detect from the checkpoint.
  std::string arch;
  // Empty means $LSE_POOL, and empty again means one device.
  std::string pool;
  // Which source dialect this run would rather its kernels were written in.
  // Empty means the device's own first choice, which is what every run that
  // does not say gets.
  std::string dialect;
  bool show_stats = false;
  bool list_devices = false;
  bool list_cache = false;
  bool debug = false;
  std::int32_t kv_len = 0;
  // Sequences to decode in one engine. 1 keeps the single-session Generator,
  // which is the path every baseline was taken on.
  std::int32_t batch = 1;
  // Blocks one attention layer's KV pool may hold. 0 sizes it so nothing is
  // ever preempted.
  std::int32_t kv_blocks = 0;
  // Extra prompts, one sequence each. Their lengths differ, so the rows sit at
  // different positions and the batch is ragged — which is the case a shared
  // position gets wrong.
  std::vector<std::string> prompts;
};

void usage() {
  std::puts(
      "usage: lse [options] [prompt]\n"
      "\n"
      "  -m, --model NAME       checkpoint directory, .safetensors, or an HF\n"
      "                         repo id such as mlx-community/Qwen3.5-4B-4bit;\n"
      "                         a bare model name resolves when it is unique\n"
      "                         (default: $LSE_MODEL)\n"
      "  -n, --max-tokens N     tokens to generate (default 256)\n"
      "  -t, --temperature F    0 or less is greedy (default 0.8)\n"
      "      --top-k N          keep the N most likely tokens (default off)\n"
      "      --top-p F          nucleus threshold (default 1.0, off)\n"
      "      --repeat-penalty F penalize repeats, >1 discourages (default 1.0)\n"
      "  -s, --seed N           sampler seed (default 0)\n"
      "      --tokenizer REPO   HF repo for tokenizer.json, used only when the\n"
      "                         model directory has none (default Qwen/Qwen3.6-27B)\n"
      "      --arch NAME        force a model kernel instead of detecting one\n"
      "      --list-models      print the registered model kernels and exit\n"
      "      --list-cache       list the models in the HF cache and whether\n"
      "                         this build can load each one, and exit\n"
      "      --kv-len N         allocate the KV cache for N tokens and keep\n"
      "                         that shape (default: max(2*train_seq, 2048))\n"
      "  -b, --batch N          decode N copies of the prompt as one batch,\n"
      "                         reporting aggregate throughput and per-session\n"
      "                         latency (default 1: the single-session path)\n"
      "  -p, --prompt TEXT      one more sequence for the batch; repeatable,\n"
      "                         and prompts of different lengths put the rows\n"
      "                         at different positions\n"
      "      --kv-blocks N      blocks one attention layer's pool may hold;\n"
      "                         below what the batch needs, sequences are\n"
      "                         preempted and resume (default: no limit)\n"
      "      --stats            print timings when done\n"
      "      --debug            print the HIP dump path and file count\n"
      "      --devices          report every device this build can see, with\n"
      "                         its identity and what it will not answer\n"
      "      --pool LIST         devices this run may use, backend-qualified\n"
      "                         and best first: hrx:0,cpu:0 (default $LSE_POOL,\n"
      "                         else one device -- the first backend that comes\n"
      "                         up, at $LSE_DEVICE's ordinal)\n"
      "      --dialect NAME     source dialect to generate kernels in: hip or\n"
      "                         loom. A device that does not declare it is\n"
      "                         given its own first choice instead (default:\n"
      "                         every device's first choice)\n"
      "  -h, --help             this message");
}

// Returns false when the flag needed a value it did not get.
bool take_value(int argc, char** argv, int& i, const char* flag,
                std::string* out) {
  if (i + 1 >= argc) {
    std::fprintf(stderr, "lse: %s needs a value\n", flag);
    return false;
  }
  *out = argv[++i];
  return true;
}

bool parse(int argc, char** argv, Options* opt) {
  std::vector<std::string> positional;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    std::string v;
    if (a == "-h" || a == "--help") {
      usage();
      std::exit(0);
    } else if (a == "--devices") {
      opt->list_devices = true;
    } else if (a == "--list-cache") {
      opt->list_cache = true;
    } else if (a == "--pool") {
      if (!take_value(argc, argv, i, "--pool", &opt->pool)) return false;
    } else if (a == "--dialect") {
      if (!take_value(argc, argv, i, "--dialect", &opt->dialect)) return false;
      if (!graph::dialect_from_name(opt->dialect).has_value()) {
        std::fprintf(stderr, "lse: no dialect is spelled '%s'\n",
                     opt->dialect.c_str());
        return false;
      }
    } else if (a == "--stats") {
      opt->show_stats = true;
    } else if (a == "--debug") {
      opt->debug = true;
    } else if (a == "-b" || a == "--batch") {
      if (!take_value(argc, argv, i, "--batch", &v)) return false;
      opt->batch = std::atoi(v.c_str());
    } else if (a == "-p" || a == "--prompt") {
      if (!take_value(argc, argv, i, "--prompt", &v)) return false;
      opt->prompts.push_back(v);
    } else if (a == "--kv-blocks") {
      if (!take_value(argc, argv, i, "--kv-blocks", &v)) return false;
      opt->kv_blocks = std::atoi(v.c_str());
    } else if (a == "-m" || a == "--model") {
      if (!take_value(argc, argv, i, "--model", &opt->model)) return false;
    } else if (a == "-n" || a == "--max-tokens") {
      if (!take_value(argc, argv, i, "--max-tokens", &v)) return false;
      opt->limits.max_tokens = std::atoi(v.c_str());
    } else if (a == "-t" || a == "--temperature") {
      if (!take_value(argc, argv, i, "--temperature", &v)) return false;
      opt->sampling.temperature = std::strtof(v.c_str(), nullptr);
    } else if (a == "--top-k") {
      if (!take_value(argc, argv, i, "--top-k", &v)) return false;
      opt->sampling.top_k = std::atoi(v.c_str());
    } else if (a == "--top-p") {
      if (!take_value(argc, argv, i, "--top-p", &v)) return false;
      opt->sampling.top_p = std::strtof(v.c_str(), nullptr);
    } else if (a == "--repeat-penalty") {
      if (!take_value(argc, argv, i, "--repeat-penalty", &v)) return false;
      opt->sampling.repetition_penalty = std::strtof(v.c_str(), nullptr);
    } else if (a == "--arch") {
      if (!take_value(argc, argv, i, "--arch", &opt->arch)) return false;
    } else if (a == "--list-models") {
      for (std::string_view name : model::registered_architectures()) {
        std::printf("%s\n", std::string(name).c_str());
      }
      std::exit(0);
    } else if (a == "--tokenizer") {
      if (!take_value(argc, argv, i, "--tokenizer", &opt->tokenizer_repo)) return false;
    } else if (a == "--kv-len") {
      if (!take_value(argc, argv, i, "--kv-len", &v)) return false;
      opt->kv_len = std::atoi(v.c_str());
    } else if (a == "-s" || a == "--seed") {
      if (!take_value(argc, argv, i, "--seed", &v)) return false;
      opt->sampling.seed = std::strtoull(v.c_str(), nullptr, 10);
    } else if (!a.empty() && a[0] == '-' && a != "-") {
      std::fprintf(stderr, "lse: unknown option '%s'\n", a.c_str());
      return false;
    } else {
      positional.push_back(a);
    }
  }

  if (!positional.empty()) {
    opt->prompt.clear();
    for (std::size_t i = 0; i < positional.size(); ++i) {
      if (i != 0) opt->prompt += ' ';
      opt->prompt += positional[i];
    }
  }
  return true;
}

// "iB" throughout so a byte count is never read as a parameter count: the
// SCALE column's B means billion and this column's B would mean bytes.
std::string human_bytes(std::uintmax_t n) {
  const char* unit[] = {"B", "KiB", "MiB", "GiB", "TiB"};
  double v = static_cast<double>(n);
  int u = 0;
  while (v >= 1024.0 && u < 4) {
    v /= 1024.0;
    ++u;
  }
  char buf[32];
  std::snprintf(buf, sizeof(buf), v < 10.0 && u > 0 ? "%.1f%s" : "%.0f%s", v,
                unit[u]);
  return buf;
}

std::string human_scale(std::size_t params) {
  if (params == 0) return "-";
  char buf[32];
  if (params >= 1000000000ull) {
    std::snprintf(buf, sizeof(buf), "%.1fB",
                  static_cast<double>(params) / 1e9);
  } else {
    std::snprintf(buf, sizeof(buf), "%.0fM", static_cast<double>(params) / 1e6);
  }
  return buf;
}

// One line per model, and the reason on a continuation line whenever the
// verdict is anything but yes — the same shape list_devices uses for a backend
// it cannot bring up.
int list_cache() {
  auto models = model::list_cached_models();
  if (!models.ok()) {
    std::fprintf(stderr, "lse: %s\n", models.status().to_string().c_str());
    return 1;
  }
  std::printf("cache %s\n", model::hf_cache_root().c_str());
  if (models->empty()) {
    std::puts("no models in the cache");
    return 0;
  }

  std::size_t repo_w = 4;
  std::size_t arch_w = 4;
  std::size_t quant_w = 5;
  for (const model::CacheModel& m : *models) {
    repo_w = std::max(repo_w, m.repo_id.size());
    arch_w = std::max(arch_w, m.architecture.size());
    quant_w = std::max(quant_w, m.quantization.size());
  }

  std::printf("\n%-*s  %-*s  %5s  %-*s  %-4s  %6s  %s\n",
              static_cast<int>(repo_w), "REPO", static_cast<int>(arch_w),
              "ARCH", "SCALE", static_cast<int>(quant_w), "QUANT", "MM",
              "ONDISK", "LOAD");
  for (const model::CacheModel& m : *models) {
    std::string load(model::to_string(m.loadable));
    if (!m.engine_arch.empty()) load += " (" + m.engine_arch + ")";
    std::printf("%-*s  %-*s  %5s  %-*s  %-4s  %6s  %s\n",
                static_cast<int>(repo_w), m.repo_id.c_str(),
                static_cast<int>(arch_w),
                m.architecture.empty() ? "-" : m.architecture.c_str(),
                human_scale(m.parameters).c_str(), static_cast<int>(quant_w),
                m.quantization.empty() ? "-" : m.quantization.c_str(),
                m.multimodal_known ? (m.multimodal ? "yes" : "no") : "-",
                human_bytes(m.bytes).c_str(), load.c_str());
    if (!m.reason.empty()) std::printf("    %s\n", m.reason.c_str());
  }
  // The tower is in every one of these checkpoints and none of it is read, so
  // saying "multimodal" without saying that would overstate what a load gets.
  std::puts("\nMM marks a checkpoint carrying a vision tower; this build decodes text only.");
  return 0;
}

// One block per DEVICE, not per backend: on a machine with several GPUs the
// old shape printed one line for the whole hrx backend and the properties of
// whichever device it happened to bind.
//
// Nothing here binds a device. Enumeration reads identities, so a report costs
// no stream, no allocation and no queue probe, and the ordinal a run would
// select is not privileged over the others.
int list_devices() {
  const auto names = backend::available_backends();
  if (names.empty()) {
    std::puts("no backends were compiled into this build");
    return 1;
  }
  std::size_t devices = 0;
  for (std::string_view name : names) {
    auto found = backend::enumerate_devices(name);
    if (!found.ok()) {
      std::printf("%-6s unavailable: %s\n", std::string(name).c_str(),
                  found.status().to_string().c_str());
#ifdef LSE_HSA_RUNTIME_DIR
      // The loader captured LD_LIBRARY_PATH before main ran, so this cannot be
      // fixed from here — say what to set rather than fail silently.
      if (found.status().to_string().find("hsa_") != std::string::npos) {
        std::printf("       the distro HSA runtime is too old; run with\n"
                    "         LD_LIBRARY_PATH=%s\n", LSE_HSA_RUNTIME_DIR);
      }
#endif
      continue;
    }
    if (found->empty()) {
      std::printf("%-6s no devices attached\n", std::string(name).c_str());
      continue;
    }
    for (const backend::DeviceDescriptor& d : *found) {
      std::printf("%-6s %s", d.id().c_str(), d.describe().c_str());
      ++devices;
    }
  }
  if (devices == 0) return 1;
  // What a run would pick, in the order it would try — the report names
  // devices because a human is reading it, and this is the one line that says
  // which of them the engine would choose. Placement upstream reads
  // capabilities, never these names.
  std::string order;
  for (const std::string& candidate : backend::default_backend_order()) {
    if (!order.empty()) order += ", ";
    order += candidate;
  }
  int selected = 0;
  if (const char* env = std::getenv("LSE_DEVICE")) selected = std::atoi(env);
  std::printf(
      "\n%zu device%s. A run binds ordinal %d of the first backend that comes "
      "up, trying %s (LSE_BACKEND and LSE_DEVICE override); --pool or "
      "$LSE_POOL names a set instead, best first, e.g. --pool hrx:0,cpu:0.\n"
      "Values are the device's own answers unless marked (declared), which "
      "means an architecture table answered; a property nothing here can "
      "answer reads unknown, and n/a means the device has no such property.\n",
      devices, devices == 1 ? "" : "s", selected, order.c_str());
  return 0;
}

int fail(const Status& s, const char* what) {
  std::fprintf(stderr, "lse: %s: %s\n", what, s.to_string().c_str());
  return 1;
}

// Enough of a store entry to tell a rewrite from an untouched read.
struct StoreEntry {
  bool present = false;
  std::uintmax_t size = 0;
  std::filesystem::file_time_type written{};
  friend bool operator==(const StoreEntry&, const StoreEntry&) = default;
};

StoreEntry stat_entry(const std::string& path) {
  StoreEntry e;
  std::error_code ec;
  if (path.empty() || !std::filesystem::is_regular_file(path, ec)) return e;
  e.size = std::filesystem::file_size(path, ec);
  if (ec) return e;
  e.written = std::filesystem::last_write_time(path, ec);
  if (ec) return e;
  e.present = true;
  return e;
}

struct Qualification {
  probe::PoolProfile pool;
  // Whether the device set opened at all. Distinct from `status`: a device
  // that merely could not be MEASURED is a pool with unknown numbers and a
  // run that proceeds, while a device that was asked for and never came up is
  // the end of the run — and saying so here rather than letting the weights
  // fail two seconds later is the difference between naming the device and
  // reporting that a tensor had nowhere to go.
  bool devices_up = false;
  // Non-ok when nothing could be qualified at all. A device that merely could
  // not be measured is not a failure: it comes back with kUnknown numbers.
  Status status;
  bool from_cache = false;
  double ms = 0.0;
};

// What this process will run on, measured before anything is placed on it.
//
// Called before the weights are bound, for two reasons that both matter: free
// memory has to mean "what could hold a shard", not "what the model left over",
// and the probe's own transfers must not be timed against the checkpoint
// streaming to the same device.
Qualification qualify_startup_pool(const Options& opt) {
  Qualification q;

  // Untimed on purpose: this brings the devices up, which the engine pays for
  // either way — building the model was the first caller before this one. The
  // clock below measures qualification, not device init, or a warm start would
  // report a couple of hundred milliseconds it did not add.
  if (const Status opened = place::open_default_devices(opt.pool);
      !opened.ok()) {
    q.status = opened;
    return q;
  }
  place::Devices* devices = place::default_devices();
  if (devices == nullptr || devices->size() == 0) {
    q.status = LSE_ERROR(kDeviceError, "no device came up to qualify");
    return q;
  }
  q.devices_up = true;
  // Falling back to a backend with no code generator runs the whole model
  // through the host interpreter. That is a two-order-of-magnitude cliff and it
  // used to be silent: the `--stats` line reported zero device groups next to a
  // five-figure launch count and read like broken instrumentation.
  backend::IBackend& first = devices->device(devices->primary());
  if (first.emitter() == nullptr && !devices->declined().empty()) {
    std::fprintf(stderr,
                 "lse: no code-generating backend came up (%s); running on "
                 "'%s' through the host interpreter\n",
                 std::string(devices->declined()).c_str(),
                 std::string(first.name()).c_str());
  }
  graph::Scheduler* sched = graph::default_scheduler();
  if (sched == nullptr) {
    q.status = LSE_ERROR(kDeviceError, "no scheduler could be built");
    return q;
  }
  if (!opt.dialect.empty()) {
    // parse() refuses a name no dialect is spelled with, so the only remaining
    // question is what each member declares.
    const graph::Dialect want = *graph::dialect_from_name(opt.dialect);
    sched->set_dialect(want);
    for (std::size_t i = 0; i < devices->size(); ++i) {
      const graph::KernelToolchain* tc = sched->toolchain(i);
      if (tc == nullptr) continue;
      // Said out loud either way. A run that asked for a dialect and quietly
      // got the other one is exactly what this selection path exists to make
      // visible, and a degrade that prints nothing is that failure.
      std::fprintf(stderr, "lse: %s generates %s%s\n",
                   std::string(devices->device(i).name()).c_str(),
                   std::string(to_string(tc->dialect)).c_str(),
                   tc->dialect == want
                       ? ""
                       : " -- it does not declare the one asked for");
    }
  }

  const auto started = std::chrono::steady_clock::now();
  const auto finish = [&](Qualification&& out) {
    out.ms = std::chrono::duration<double, std::milli>(
                 std::chrono::steady_clock::now() - started)
                 .count();
    return std::move(out);
  };

  // Whether this start read a profile or paid for one is not the same question
  // as whether a file was there: a truncated, foreign or unreadable entry is
  // present and still re-measured, and a store that cannot be written back is
  // re-measured on every start. Only a measurement rewrites the entry, so the
  // entry is its own witness — and it stays one however qualify_pool decides
  // what it will serve.
  const std::vector<probe::PoolMember> members = devices->pool_members();
  std::string entry;
  if (auto fingerprint = probe::pool_fingerprint(members, nullptr);
      fingerprint.ok()) {
    entry = probe::profile_path(probe::default_profile_dir(), *fingerprint);
  }
  const StoreEntry before = stat_entry(entry);

  const Status qualified = devices->qualify();
  q.from_cache = before.present && stat_entry(entry) == before;
  if (!qualified.ok()) {
    q.status = qualified;
    return finish(std::move(q));
  }
  q.pool = devices->profile();
  return finish(std::move(q));
}

void report_pool(const Qualification& q) {
  if (!q.status.ok()) {
    std::fprintf(stderr, "probe declined in %.2f ms: %s\n", q.ms,
                 q.status.to_string().c_str());
    return;
  }
  std::fprintf(stderr, "probe %s in %.2f ms | %zu device%s | cost model %s\n",
               q.from_cache ? "cached" : "measured", q.ms,
               q.pool.devices.size(), q.pool.devices.size() == 1 ? "" : "s",
               q.pool.complete() ? "complete" : "incomplete");
  std::fputs(q.pool.describe().c_str(), stderr);
}

}  // namespace

int main(int argc, char** argv) {
  Options opt;
  if (!parse(argc, argv, &opt)) return 2;
  if (opt.list_devices) return list_devices();
  if (opt.list_cache) return list_cache();
  if (opt.debug) {
    lse::set_debug(true);
    const std::string dir = lse::graph::hip_dump_directory();
    if (!dir.empty()) {
      std::fprintf(stderr, "debug: writing generated HIP to %s\n", dir.c_str());
    }
  }

  if (opt.model.empty()) {
    if (const char* env = std::getenv("LSE_MODEL")) opt.model = env;
  }
  if (opt.model.empty()) {
    std::fputs("lse: no model given; pass --model or set LSE_MODEL\n", stderr);
    return 2;
  }

  auto paths = model::resolve_model(opt.model);
  if (!paths.ok()) return fail(paths.status(), "resolving the model");

  auto cfg = model::Config::from_json_file(paths->config);
  if (!cfg.ok()) return fail(cfg.status(), "reading the config");
  if (opt.kv_len > 0) cfg->kv_length = opt.kv_len;

  auto weights = paths->weights.ends_with(".index.json")
                     ? model::SafeTensors::open_sharded(paths->weights)
                     : model::SafeTensors::open(paths->weights);
  if (!weights.ok()) return fail(weights.status(), "opening the weights");

  auto arch = model::detect_architecture(*cfg, *weights);
  if (opt.arch.empty() && !arch.ok()) {
    return fail(arch.status(), "identifying the architecture");
  }
  std::fprintf(stderr, "model: %s (%d layers, hidden %d)\n",
               opt.arch.empty() ? std::string((*arch)->name).c_str()
                                : opt.arch.c_str(),
               cfg->num_layers, cfg->hidden_size);

  const Qualification qual = qualify_startup_pool(opt);
  if (!qual.devices_up) return fail(qual.status, "opening the device set");

  // Building the tokenizer is a 248k-entry automaton and binding the weights
  // streams the whole checkpoint; neither needs the other, so the smaller
  // hides behind the larger. jthread so an early return below still joins.
  // Started after qualification, not before: the automaton saturates a core
  // for over a second, and a bandwidth or dispatch figure timed against it is
  // stamped kMeasured and memoized under a fingerprint that cannot see load.
  const std::string tok_dir =
      paths->weights.substr(0, paths->weights.find_last_of('/'));
  std::optional<Result<tokenizer::Tokenizer>> tok_slot;
  std::jthread tok_load([&] {
    tok_slot.emplace(
        tokenizer::Tokenizer::for_model_dir(tok_dir, opt.tokenizer_repo));
  });

  auto built = model::build_model(*cfg, *weights, opt.arch);
  if (!built.ok()) return fail(built.status(), "building the model");
  std::unique_ptr<model::HybridLM> lm = built.release();

  model::WeightBinder binder(*weights, &cfg->quantization);
  const Status loaded = lm->load(binder);
  if (!loaded.ok()) return fail(loaded, "binding the weights");

  tok_load.join();
  if (!tok_slot->ok()) return fail(tok_slot->status(), "loading the tokenizer");
  Result<tokenizer::Tokenizer>& tok = *tok_slot;

  auto prompt = tok->encode(opt.prompt);
  if (!prompt.ok()) return fail(prompt.status(), "encoding the prompt");
  if (prompt->empty()) {
    std::fputs("lse: the prompt encoded to no tokens\n", stderr);
    return 2;
  }

  if (opt.limits.stop_tokens.empty()) {
    opt.limits.stop_tokens.push_back(tokenizer::kQwen36Eos);
  }

  // More than one sequence: the batch driver, decoding them all in one step.
  // Aggregate throughput and per-session latency both get reported because they
  // move in opposite directions — a wider batch raises the first and lowers the
  // second, and printing only the first would hide the trade.
  if (opt.batch > 1 || !opt.prompts.empty()) {
    std::vector<std::vector<std::uint32_t>> encoded;
    for (const std::string& text : opt.prompts) {
      auto ids = tok->encode(text);
      if (!ids.ok()) return fail(ids.status(), "encoding a batch prompt");
      encoded.push_back(ids.release());
    }
    for (std::int32_t i = static_cast<std::int32_t>(encoded.size());
         i < opt.batch; ++i) {
      encoded.push_back(*prompt);
    }

    runtime::BatchLimits limits;
    limits.max_batch = std::max<std::int32_t>(
        opt.batch, static_cast<std::int32_t>(encoded.size()));
    limits.max_tokens = opt.limits.max_tokens;
    limits.stop_tokens = opt.limits.stop_tokens;
    limits.kv_blocks = opt.kv_blocks;

    runtime::BatchScheduler batch(*lm, opt.sampling, limits);
    for (std::size_t i = 0; i < encoded.size(); ++i) {
      const Status s = batch.submit({"s" + std::to_string(i), encoded[i],
                                     opt.limits.max_tokens});
      if (!s.ok()) return fail(s, "submitting a sequence");
    }
    auto done = batch.run();
    if (!done.ok()) return fail(done.status(), "batched generation");

    std::sort(done->begin(), done->end(),
              [](const runtime::SequenceResult& a,
                 const runtime::SequenceResult& b) { return a.id < b.id; });
    for (const runtime::SequenceResult& r : *done) {
      auto text = tok->decode(r.generated);
      std::fprintf(stdout, "[%s] %s\n", r.id.c_str(),
                   text.ok() ? text->c_str() : "<undecodable>");
    }
    std::fflush(stdout);

    const runtime::BatchStats& bs = batch.stats();
    std::fprintf(stderr,
                 "batch %d (bucket %d): %.2f tok/s aggregate | %d token(s) in "
                 "%.3f s over %d step(s) | occupancy %.2f\n",
                 static_cast<int>(encoded.size()), bs.bucket,
                 bs.aggregate_tokens_per_second(), bs.generated_tokens,
                 static_cast<double>(bs.wall_ns) / 1e9, bs.steps,
                 bs.occupancy());
    std::vector<double> per_session;
    for (const runtime::SequenceResult& r : *done) {
      per_session.push_back(r.tokens_per_second());
      std::fprintf(stderr,
                   "  %-6s %4zu token(s)  ttft %8.2f ms  %7.2f tok/s"
                   "  preempted %d\n",
                   r.id.c_str(), r.generated.size(),
                   static_cast<double>(r.ttft_ns) / 1e6, r.tokens_per_second(),
                   r.preemptions);
    }
    std::sort(per_session.begin(), per_session.end());
    const double median =
        per_session.empty()
            ? 0.0
            : per_session[per_session.size() / 2];
    std::fprintf(stderr,
                 "per-session median %.2f tok/s | admissions %d | preemptions "
                 "%d | launches %u | host groups %u | jit compiles %llu\n",
                 median, bs.admissions, bs.preemptions, bs.kernels_launched,
                 bs.host_groups,
                 static_cast<unsigned long long>(bs.jit_compiles));
    if (opt.show_stats) report_pool(qual);
    return 0;
  }

  runtime::Generator gen(*lm, opt.sampling);
  auto stream = tok->stream();

  // Streamed through DecodeStream so a multi-byte character is never cut in
  // half between two tokens.
  auto emit = [&stream](std::uint32_t id) {
    auto text = stream.push(id);
    if (text.ok() && !text->empty()) {
      std::fputs(text->c_str(), stdout);
      std::fflush(stdout);
    }
    return true;
  };

  auto out = gen.generate(*prompt, opt.limits, emit);
  if (!out.ok()) return fail(out.status(), "generating");
  std::putchar('\n');

  if (opt.show_stats) {
    const runtime::GenerationStats& s = gen.stats();
    std::fprintf(stderr,
                 "prompt %d tokens, prefill %.2f s | generated %d tokens, "
                 "%.2f tok/s\n"
                 "launches %u | phases %u (ideal %u launch%s) | groups "
                 "device=%u host=%u views=%u fallbacks=%u\n"
                 "streams %u of %u | cross-stream waits %u | chain %u of %u "
                 "groups (%.0f%% spread)\n"
                 "sched partition=%.3f s emit=%.3f s launch=%.3f s sync=%.3f s\n"
                 "jit mem=%llu disk=%llu compile=%llu (%.3f s)\n",
                 s.prompt_tokens, static_cast<double>(s.prefill_ns) / 1e9,
                 s.generated_tokens, s.decode_tokens_per_second(),
                 s.kernels_launched, s.phase_groups, s.phase_ideal_launches,
                 s.phase_ideal_launches == 1 ? "" : "es",
                 s.device_groups, s.host_groups, s.views_aliased,
                 s.host_fallbacks,
                 s.streams_used, s.streams_available, s.stream_waits,
                 s.stream_chain, s.device_groups, s.spread() * 100.0,
                 static_cast<double>(s.partition_ns) / 1e9,
                 static_cast<double>(s.emit_ns) / 1e9,
                 static_cast<double>(s.launch_ns) / 1e9,
                 static_cast<double>(s.sync_ns) / 1e9,
                 static_cast<unsigned long long>(s.jit_memory_hits),
                 static_cast<unsigned long long>(s.jit_disk_hits),
                 static_cast<unsigned long long>(s.jit_compiles),
                 static_cast<double>(s.jit_compile_ns) / 1e9);
    report_pool(qual);
    // What the kernel IR's middle end actually did, per pass. A pass that
    // reports zero here fired on nothing in this model and is dead weight.
    const std::vector<lse::ir::PassStat> passes = lse::ir::pass_totals();
    if (!passes.empty()) {
      std::fprintf(stderr, "ir passes");
      for (const lse::ir::PassStat& ps : passes) {
        std::fprintf(stderr, " %.*s=%llu", static_cast<int>(ps.name.size()),
                     ps.name.data(),
                     static_cast<unsigned long long>(ps.fired));
      }
      std::fputc('\n', stderr);
    }
    std::unordered_map<std::string, unsigned> reasons;
    for (const std::string& r : gen.host_group_reasons()) ++reasons[r];
    unsigned shown = 0;
    for (const auto& [why, n] : reasons) {
      std::fprintf(stderr, "  host x%u  %s\n", n, why.c_str());
      if (++shown == 12) break;
    }
  }
  {
    const std::string dir = lse::graph::hip_dump_directory();
    std::error_code ec;
    std::size_t n = 0;
    if (!dir.empty() && std::filesystem::exists(dir, ec)) {
      for (const auto& e : std::filesystem::directory_iterator(dir, ec)) {
        if (!ec && e.path().extension() == ".hip") ++n;
      }
    }
    if (n > 0 || opt.debug) {
      std::fprintf(stderr, "hip: %zu file%s in %s\n", n, n == 1 ? "" : "s",
                   dir.c_str());
    }
  }
  return 0;
}
