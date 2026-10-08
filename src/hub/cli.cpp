// `models` and `pull`: the model-management subcommands lse and lse-server
// share. Listing is offline; only pull touches the network.
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "lse/hub/hub.hpp"
#include "lse/model/weights.hpp"

namespace lse::hub {

namespace {

using json = nlohmann::json;

std::string human(std::uint64_t n) {
  const char* unit[] = {"B", "KiB", "MiB", "GiB", "TiB"};
  double v = static_cast<double>(n);
  int u = 0;
  while (v >= 1024.0 && u < 4) {
    v /= 1024.0;
    ++u;
  }
  char buf[32];
  std::snprintf(buf, sizeof buf, u == 0 ? "%.0f %s" : "%.1f %s", v, unit[u]);
  return buf;
}

std::string scale(std::uint64_t params) {
  if (params == 0) return "-";
  char buf[32];
  if (params >= 1000000000ull) std::snprintf(buf, sizeof buf, "%.1fB", static_cast<double>(params) / 1e9);
  else std::snprintf(buf, sizeof buf, "%.0fM", static_cast<double>(params) / 1e6);
  return buf;
}

std::string revision_text(const Checkpoint& c) {
  if (c.commit.empty()) return "-";
  std::string out = c.commit.substr(0, 12);
  if (!c.refs.empty()) {
    out += " (";
    for (std::size_t i = 0; i < c.refs.size(); ++i) out += (i ? "," : "") + c.refs[i];
    out += ")";
  }
  return out;
}

// The command that fetches a target's missing companion.
std::string pull_command(std::string_view prog, const Checkpoint& target, const CompanionState& st) {
  if (st.recommended == nullptr) return {};
  if (!target.repo_id.empty())
    return std::string(prog) + " pull " + target.name() + (st.kind == Kind::kMtp ? " --with-mtp" : " --with-dflash2");
  return std::string(prog) + " pull " + st.recommended->repo_id + "@" + st.recommended->revision;
}

// The registry's answer, or else a local checkpoint (--dir, $LSE_MODEL_DIRS)
// of the same kind trained against the same widths. A local one is shown, not
// chosen: a run that wants it names it with --mtp or --dflash2-model.
CompanionState listed_state(const Listing& listing, const Checkpoint& target, Kind kind) {
  CompanionState st = companion_state(target, kind);
  if (!st.present.empty() || (kind == Kind::kMtp && target.mtp_layers == 0)) return st;
  for (const Checkpoint& c : listing.companions) {
    if (c.kind != kind || !c.repo_id.empty() || c.hidden_size != target.hidden_size) continue;
    if (c.vocab_size != 0 && target.vocab_size != 0 && c.vocab_size != target.vocab_size) continue;
    st.present = c.path;
    st.where = "local";
    st.converted = c.bits == 8 || c.dflash2_converted;
    return st;
  }
  return st;
}

std::string state_text(const Checkpoint& target, const CompanionState& st) {
  if (st.kind == Kind::kMtp && target.mtp_layers == 0) return "-";
  if (!st.present.empty()) {
    if (st.kind == Kind::kDFlash2) return st.where + (st.converted ? " (Q8)" : "");
    return st.where;
  }
  return st.recommended != nullptr ? "missing" : (st.kind == Kind::kMtp ? "none known" : "-");
}

json checkpoint_json(const Checkpoint& c) {
  return json{{"name", c.name()},
              {"kind", std::string(to_string(c.kind))},
              {"repo_id", c.repo_id.empty() ? json(nullptr) : json(c.repo_id)},
              {"revision", c.commit.empty() ? json(nullptr) : json(c.commit)},
              {"refs", c.refs},
              {"path", c.path},
              {"architecture", c.architecture},
              {"quant", c.quant},
              {"bits", c.bits},
              {"parameters", c.parameters},
              {"bytes", c.bytes},
              {"hidden_size", c.hidden_size},
              {"layers", c.layers}};
}

json companion_json(std::string_view prog, const Checkpoint& target, const CompanionState& st) {
  json j{{"status", st.present.empty() ? (st.recommended ? "missing" : "none") : "present"},
         {"present", st.present.empty() ? json(nullptr) : json(st.present)}};
  if (!st.where.empty()) j["where"] = st.where;
  if (st.kind == Kind::kDFlash2 && !st.present.empty()) j["q8_converted"] = st.converted;
  if (st.recommended != nullptr) {
    j["recommended"] = {{"repo_id", st.recommended->repo_id},
                        {"revision", st.recommended->revision},
                        {"bytes", st.recommended->weights_bytes}};
    if (st.present.empty()) j["pull"] = pull_command(prog, target, st);
  }
  if (st.kind == Kind::kMtp && target.mtp_layers == 0) j["status"] = "not_applicable";
  return j;
}

int fail(std::string_view prog, const Status& s) {
  std::fprintf(stderr, "%.*s: %s\n", static_cast<int>(prog.size()), prog.data(), s.message().c_str());
  return s.code() == StatusCode::kInvalidArgument ? 2 : 1;
}

void models_usage(std::string_view prog) {
  const std::string p(prog);
  std::printf(
      "usage: %s models [list] [--json] [--dir PATH]...\n"
      "       %s models rm ORG/NAME[@REVISION] [--yes]\n"
      "\n"
      "Lists the MLX models in the Hugging Face cache that this build loads, with\n"
      "their MTP heads and DFlash2 drafts. Reads config.json and safetensors\n"
      "headers only; never touches the network.\n"
      "\n"
      "  --json          print the listing as JSON\n"
      "  --dir PATH      also list checkpoints in PATH (a checkpoint directory or a\n"
      "                  directory of them); $LSE_MODEL_DIRS adds more, ':'-separated\n"
      "  rm              remove a repo (or one cached revision of it) from the cache;\n"
      "                  prints what would go unless --yes is given\n"
      "\n"
      "The cache is $HF_HUB_CACHE, else $HF_HOME/hub, else ~/.cache/huggingface/hub.\n",
      p.c_str(), p.c_str());
}

void pull_usage(std::string_view prog) {
  const std::string p(prog);
  std::printf(
      "usage: %s pull ORG/NAME[@REVISION] [options]     (alias: download)\n"
      "\n"
      "Downloads a model into the Hugging Face cache, in the layout huggingface-cli,\n"
      "transformers and mlx-lm use. Checks the repo is an MLX model this build loads\n"
      "(or an MTP head or DFlash2 draft) from its config and safetensors headers\n"
      "before fetching any weights. Resumes an interrupted download and verifies\n"
      "every file's sha256 (or git blob hash).\n"
      "\n"
      "  --include GLOB    only files matching GLOB (repeatable)\n"
      "  --exclude GLOB    skip files matching GLOB (repeatable)\n"
      "  --with-dflash2    also pull the target's recommended DFlash2 draft\n"
      "  --with-mtp        also pull the target's recommended MTP head\n"
      "\n"
      "Gated repos need a token: HF_TOKEN, or the file huggingface-cli login writes\n"
      "($HF_HOME/token). HF_ENDPOINT selects a mirror; HF_HUB_OFFLINE=1 refuses.\n",
      p.c_str());
}

int list_models(std::string_view prog, int argc, char** argv) {
  bool as_json = false;
  std::vector<std::string> dirs;
  for (int i = 0; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--json") as_json = true;
    else if (a == "--dir" && i + 1 < argc) dirs.push_back(argv[++i]);
    else if (a.starts_with("--dir=")) dirs.push_back(a.substr(6));
    else if (a == "-h" || a == "--help") {
      models_usage(prog);
      return 0;
    } else {
      std::fprintf(stderr, "%.*s models: unknown argument '%s'\n", static_cast<int>(prog.size()), prog.data(),
                   a.c_str());
      return 2;
    }
  }
  const Listing listing = scan(dirs);
  if (as_json) {
    std::puts(listing_json(listing, prog).c_str());
    return 0;
  }
  std::printf("HF cache: %s\n", listing.cache_root.c_str());
  if (listing.models.empty()) {
    std::printf("\nNo MLX model this build loads is in the cache. Fetch one with, for example:\n"
                "  %.*s pull mlx-community/Qwen3.5-0.8B-4bit\n",
                static_cast<int>(prog.size()), prog.data());
  } else {
    struct Row {
      std::vector<std::string> cells;
      const Checkpoint* c;
      CompanionState mtp, dflash2;
    };
    std::vector<Row> rows;
    std::vector<std::string> head{"MODEL", "REVISION", "ARCH", "QUANT", "PARAMS", "SIZE", "MTP", "DFLASH2"};
    for (const Checkpoint& c : listing.models) {
      Row r{{}, &c, listed_state(listing, c, Kind::kMtp), listed_state(listing, c, Kind::kDFlash2)};
      r.cells = {c.name(), revision_text(c), c.architecture, c.quant, scale(c.parameters), human(c.bytes),
                 state_text(c, r.mtp), state_text(c, r.dflash2)};
      rows.push_back(std::move(r));
    }
    std::vector<std::size_t> width(head.size());
    for (std::size_t i = 0; i < head.size(); ++i) width[i] = head[i].size();
    for (const Row& r : rows)
      for (std::size_t i = 0; i < r.cells.size(); ++i) width[i] = std::max(width[i], r.cells[i].size());
    const auto line = [&](const std::vector<std::string>& cells) {
      std::string out;
      for (std::size_t i = 0; i < cells.size(); ++i) {
        out += cells[i];
        if (i + 1 < cells.size()) out += std::string(width[i] - cells[i].size() + 2, ' ');
      }
      std::printf("%s\n", out.c_str());
    };
    std::printf("\n");
    line(head);
    std::vector<std::string> hints;
    for (const Row& r : rows) {
      line(r.cells);
      if (r.c->path != r.c->name()) std::printf("    %s\n", r.c->path.c_str());
      for (const CompanionState* st : {&r.mtp, &r.dflash2}) {
        if (!st->present.empty() || st->recommended == nullptr) continue;
        if (st->kind == Kind::kMtp && r.c->mtp_layers == 0) continue;
        char buf[512];
        std::snprintf(buf, sizeof buf, "  %s\n      %s %s, %s of weights", pull_command(prog, *r.c, *st).c_str(),
                      st->kind == Kind::kMtp ? "MTP head" : "DFlash2 draft", st->recommended->repo_id.c_str(),
                      human(st->recommended->weights_bytes).c_str());
        hints.emplace_back(buf);
      }
    }
    if (!hints.empty()) {
      std::printf("\nMissing companions (fetched automatically when a run enables them, unless --offline):\n");
      for (const std::string& h : hints) std::printf("%s\n", h.c_str());
    }
  }
  if (!listing.companions.empty()) {
    std::printf("\nCompanions:\n");
    std::size_t w = 0;
    for (const Checkpoint& c : listing.companions) w = std::max(w, c.name().size());
    for (const Checkpoint& c : listing.companions) {
      std::printf("  %-*s  %-13s  %-6s  %9s  %s%s\n", static_cast<int>(w), c.name().c_str(),
                  c.kind == Kind::kMtp ? "MTP head" : "DFlash2 draft", c.quant.c_str(), human(c.bytes).c_str(),
                  revision_text(c).c_str(),
                  c.kind == Kind::kDFlash2 && c.quant == "BF16"
                      ? (c.dflash2_converted ? ", Q8 conversion cached" : ", converted to Q8 on first use")
                      : "");
    }
  }
  std::printf("\n%zu model%s", listing.models.size(), listing.models.size() == 1 ? "" : "s");
  if (!listing.companions.empty())
    std::printf(", %zu companion%s", listing.companions.size(), listing.companions.size() == 1 ? "" : "s");
  std::printf(". Load one with: %.*s --model NAME\n", static_cast<int>(prog.size()), prog.data());
  return 0;
}

int remove_model(std::string_view prog, int argc, char** argv) {
  std::string target;
  bool yes = false;
  for (int i = 0; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "-y" || a == "--yes") yes = true;
    else if (a == "-h" || a == "--help") {
      models_usage(prog);
      return 0;
    } else if (!a.starts_with("-") && target.empty()) target = a;
    else {
      std::fprintf(stderr, "%.*s models rm: unexpected argument '%s'\n", static_cast<int>(prog.size()),
                   prog.data(), a.c_str());
      return 2;
    }
  }
  if (target.empty()) {
    std::fprintf(stderr, "%.*s models rm: name a repo, ORG/NAME[@REVISION]\n", static_cast<int>(prog.size()),
                 prog.data());
    return 2;
  }
  auto spec = parse_repo_spec(target);
  if (!spec.ok()) return fail(prog, spec.status());
  auto plan = plan_removal(*spec);
  if (!plan.ok()) return fail(prog, plan.status());
  std::printf("%s %s, freeing %s:\n", yes ? "removing" : "would remove", plan->description.c_str(),
              human(plan->bytes).c_str());
  for (const auto& p : plan->paths) std::printf("  %s\n", p.c_str());
  if (!yes) {
    std::printf("nothing was removed; add --yes to delete\n");
    return 1;
  }
  if (const Status s = remove(*plan); !s.ok()) return fail(prog, s);
  std::printf("removed\n");
  return 0;
}

int pull_cli(std::string_view prog, int argc, char** argv) {
  PullOptions options;
  std::string target;
  bool with_dflash2 = false, with_mtp = false;
  for (int i = 0; i < argc; ++i) {
    const std::string a = argv[i];
    const auto value = [&](std::vector<std::string>& into, const char* flag) {
      if (a.starts_with(std::string(flag) + "=")) {
        into.push_back(a.substr(std::string(flag).size() + 1));
        return true;
      }
      if (a == flag && i + 1 < argc) {
        into.push_back(argv[++i]);
        return true;
      }
      return false;
    };
    if (a == "-h" || a == "--help") {
      pull_usage(prog);
      return 0;
    }
    if (value(options.include, "--include") || value(options.exclude, "--exclude")) continue;
    if (a == "--with-dflash2") with_dflash2 = true;
    else if (a == "--with-mtp") with_mtp = true;
    else if (a == "--offline") set_offline();
    else if (!a.starts_with("-") && target.empty()) target = a;
    else {
      std::fprintf(stderr, "%.*s pull: unexpected argument '%s' (see %.*s pull --help)\n",
                   static_cast<int>(prog.size()), prog.data(), a.c_str(), static_cast<int>(prog.size()),
                   prog.data());
      return 2;
    }
  }
  if (target.empty()) {
    pull_usage(prog);
    return 2;
  }
  auto spec = parse_repo_spec(target);
  if (!spec.ok()) return fail(prog, spec.status());
  std::fprintf(stderr, "pulling %s from %s into %s\n", spec->to_string().c_str(), endpoint().c_str(),
               model::hf_cache_root().c_str());
  options.progress = stderr_progress("  ");
  options.on_checked = [](const Checkpoint& c) {
    std::fprintf(stderr, "  checked before downloading: %s %s%s, %s of weights\n",
                 c.kind == Kind::kModel ? "MLX model" : c.kind == Kind::kMtp ? "MTP head" : "DFlash2 draft",
                 c.kind == Kind::kModel ? (c.architecture + ", ").c_str() : "", c.quant.c_str(),
                 human(c.bytes).c_str());
  };
  const auto started = std::chrono::steady_clock::now();
  auto got = pull(*spec, options);
  if (!got.ok()) return fail(prog, got.status());
  const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  std::printf("%s@%s: %zu file%s, %s (%s downloaded, %zu already cached) in %.1f s\n", got->repo_id.c_str(),
              got->commit.substr(0, 12).c_str(), got->files, got->files == 1 ? "" : "s", human(got->bytes).c_str(),
              human(got->downloaded).c_str(), got->reused, secs);
  std::printf("  %s\n", got->snapshot.c_str());
  int status = 0;
  if (got->checkpoint.kind == Kind::kModel) {
    for (const auto& [wanted, kind] : {std::pair{with_mtp, Kind::kMtp}, std::pair{with_dflash2, Kind::kDFlash2}}) {
      if (!wanted) continue;
      const CompanionState st = companion_state(got->checkpoint, kind);
      if (!st.present.empty()) {
        std::printf("  %s already cached: %s\n", kind == Kind::kMtp ? "MTP head" : "DFlash2 draft", st.present.c_str());
        continue;
      }
      EnsureOptions e;
      e.progress = stderr_progress("  ");
      e.prog = std::string(prog);
      auto c = ensure_companion(got->checkpoint.name(), kind, e);
      if (!c.ok()) {
        status = fail(prog, c.status());
      } else if (c->empty()) {
        std::fprintf(stderr, "%.*s: no %s is known for %s\n", static_cast<int>(prog.size()), prog.data(),
                     kind == Kind::kMtp ? "MTP head" : "DFlash2 draft", got->checkpoint.name().c_str());
        status = 1;
      } else {
        std::printf("  %s: %s\n", kind == Kind::kMtp ? "MTP head" : "DFlash2 draft", c->c_str());
      }
    }
    std::printf("run it: %.*s --model %s%s\n", static_cast<int>(prog.size()), prog.data(),
                got->checkpoint.name().c_str(), with_dflash2 ? " --dflash2=on" : "");
  } else if (with_dflash2 || with_mtp) {
    std::fprintf(stderr, "%.*s: --with-dflash2/--with-mtp apply to a target model; %s is a %s\n",
                 static_cast<int>(prog.size()), prog.data(), got->repo_id.c_str(),
                 std::string(to_string(got->checkpoint.kind)).c_str());
    status = 2;
  }
  return status;
}

}  // namespace

std::string listing_json(const Listing& listing, std::string_view prog) {
  json models = json::array(), companions = json::array();
  for (const Checkpoint& c : listing.models) {
    json j = checkpoint_json(c);
    j["mtp"] = companion_json(prog, c, listed_state(listing, c, Kind::kMtp));
    j["dflash2"] = companion_json(prog, c, listed_state(listing, c, Kind::kDFlash2));
    models.push_back(std::move(j));
  }
  for (const Checkpoint& c : listing.companions) {
    json j = checkpoint_json(c);
    if (c.kind == Kind::kDFlash2) j["q8_converted"] = c.dflash2_converted;
    companions.push_back(std::move(j));
  }
  return json{{"cache", listing.cache_root}, {"models", models}, {"companions", companions}}.dump();
}

bool is_subcommand(std::string_view arg) {
  return arg == "models" || arg == "pull" || arg == "download";
}

int run_cli(std::string_view prog, int argc, char** argv) {
  const std::string sub = argv[0];
  if (sub == "pull" || sub == "download") return pull_cli(prog, argc - 1, argv + 1);
  if (argc > 1 && (std::string(argv[1]) == "rm" || std::string(argv[1]) == "remove"))
    return remove_model(prog, argc - 2, argv + 2);
  if (argc > 1 && std::string(argv[1]) == "list") return list_models(prog, argc - 2, argv + 2);
  return list_models(prog, argc - 1, argv + 1);
}

}  // namespace lse::hub
