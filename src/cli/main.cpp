// lse — load a checkpoint, tokenize a prompt, stream the continuation.
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
#include "lse/probe/pool.hpp"
#include "lse/probe/profile_store.hpp"
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
  bool show_stats = false;
  bool list_devices = false;
  bool debug = false;
  std::int32_t kv_len = 0;
};

void usage() {
  std::puts(
      "usage: lse [options] [prompt]\n"
      "\n"
      "  -m, --model PATH       checkpoint directory or HF repo id\n"
      "                         (default: $LSE_MODEL, else the HF cache)\n"
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
      "      --kv-len N         allocate the KV cache for N tokens and keep\n"
      "                         that shape (default: max(2*train_seq, 2048))\n"
      "      --stats            print timings when done\n"
      "      --debug            print the HIP dump path and file count\n"
      "      --devices          list available backends and exit\n"
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
    } else if (a == "--stats") {
      opt->show_stats = true;
    } else if (a == "--debug") {
      opt->debug = true;
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

int list_devices() {
  const auto names = backend::available_backends();
  if (names.empty()) {
    std::puts("no backends were compiled into this build");
    return 1;
  }
  for (std::string_view name : names) {
    auto b = backend::create_backend(name);
    if (!b.ok()) {
      std::printf("%-6s unavailable: %s\n", std::string(name).c_str(),
                  b.status().to_string().c_str());
      continue;
    }
    int ordinal = 0;
    if (const char* env = std::getenv("LSE_DEVICE")) ordinal = std::atoi(env);
    const Status init = (*b)->init(ordinal);
    if (!init.ok()) {
      std::printf("%-6s unavailable: %s\n", std::string(name).c_str(),
                  init.to_string().c_str());
#ifdef LSE_HSA_RUNTIME_DIR
      // The loader captured LD_LIBRARY_PATH before main ran, so this cannot be
      // fixed from here — say what to set rather than fail silently.
      if (init.to_string().find("hsa_") != std::string::npos) {
        std::printf("       the distro HSA runtime is too old; run with\n"
                    "         LD_LIBRARY_PATH=%s\n", LSE_HSA_RUNTIME_DIR);
      }
#endif
      continue;
    }
    std::printf("%-6s %s", std::string(name).c_str(),
                (*b)->device_info().describe().c_str());
  }
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
Qualification qualify_startup_pool() {
  Qualification q;

  // Untimed on purpose: this brings the backend up, which the engine pays for
  // either way — building the model was the first caller before this one. The
  // clock below measures qualification, not device init, or a warm start would
  // report a couple of hundred milliseconds it did not add.
  graph::Scheduler* sched = graph::default_scheduler();
  if (sched == nullptr) {
    q.status = LSE_ERROR(kDeviceError, "no backend came up to qualify");
    return q;
  }

  const auto started = std::chrono::steady_clock::now();
  const auto finish = [&](Qualification&& out) {
    out.ms = std::chrono::duration<double, std::milli>(
                 std::chrono::steady_clock::now() - started)
                 .count();
    return std::move(out);
  };

  backend::IBackend& device = sched->backend();
  probe::PoolMember member;
  member.id.backend = std::string(device.name());
  member.id.ordinal = static_cast<int>(device.device_info().ordinal);
  member.backend = &device;
  const probe::PoolMember members[] = {member};

  // Whether this start read a profile or paid for one is not the same question
  // as whether a file was there: a truncated, foreign or unreadable entry is
  // present and still re-measured, and a store that cannot be written back is
  // re-measured on every start. Only a measurement rewrites the entry, so the
  // entry is its own witness — and it stays one however qualify_pool decides
  // what it will serve.
  std::string entry;
  if (auto fingerprint = probe::pool_fingerprint(members, nullptr);
      fingerprint.ok()) {
    entry = probe::profile_path(probe::default_profile_dir(), *fingerprint);
  }
  const StoreEntry before = stat_entry(entry);

  auto pool = probe::qualify_pool(members, nullptr);
  q.from_cache = before.present && stat_entry(entry) == before;
  if (!pool.ok()) {
    q.status = pool.status();
    return finish(std::move(q));
  }
  q.pool = pool.release();
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

  auto weights = model::SafeTensors::open(paths->weights);
  if (!weights.ok()) return fail(weights.status(), "opening the weights");

  auto arch = model::detect_architecture(*cfg, *weights);
  if (opt.arch.empty() && !arch.ok()) {
    return fail(arch.status(), "identifying the architecture");
  }
  std::fprintf(stderr, "model: %s (%d layers, hidden %d)\n",
               opt.arch.empty() ? std::string((*arch)->name).c_str()
                                : opt.arch.c_str(),
               cfg->num_layers, cfg->hidden_size);

  const Qualification qual = qualify_startup_pool();

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

  model::WeightBinder binder(*weights);
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
