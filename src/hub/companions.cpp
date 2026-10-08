// Which DFlash2 draft and MTP head a target pairs with, whether they are on
// disk, and fetching them when a run asks for one without naming it.
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <filesystem>

#include "lse/hub/hub.hpp"
#include "lse/model/dflash2_convert.hpp"
#include "lse/model/mtp.hpp"
#include "lse/model/weights.hpp"

namespace lse::hub {

namespace fs = std::filesystem;

namespace {

// Pins verified against the hub on 2026-10-07. The DFlash2 revision is the one
// docs/DFLASH2.md validated (its weights hash 67fc76d6...); the MTP heads are
// mlx-community's conversions, the Q8 one first because it is what the
// published MTP measurements used.
const std::vector<CompanionRule>& rules_table() {
  static const std::vector<CompanionRule> rules{
      CompanionRule{
          "Qwen3.8-27B",
          {"qwen3827b"},
          "qwen3.5",
          5120,
          64,
          248320,
          CompanionRepo{"incoai/Qwen3.8-27B-DFlash2", "dedf8df68adfb1afeaf7b7480c0a0243108177b4",
                        "model.safetensors",
                        "67fc76d68dc5a9415511a4f394ef744d67510cd20e93b37cc2cc7d28e4bab65c",
                        3848817896ull},
          {
              CompanionRepo{"mlx-community/Qwen3.8-27B-MTP-8bit", "e88e48d055732ad75d9435f3059139d5279f2064",
                            "model.safetensors",
                            "20b7873a0a9b0b6618fe7f7d20c324ea9f87bba833173e37e26fdde936ac1f2c",
                            451270785ull},
              CompanionRepo{"mlx-community/Qwen3.8-27B-MTP-4bit", "b643c01b6d3b094e325edb6ebd832e16c486c575",
                            "model.safetensors",
                            "76663c101e7e8ea9c0ae17bcb95183cd7f733ce424c912b8b264a7b1c48e4cc6",
                            238934137ull},
          },
      },
  };
  return rules;
}

std::string normalized(std::string_view s) {
  std::string out;
  for (char c : s) {
    if (std::isalnum(static_cast<unsigned char>(c)) != 0)
      out += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return out;
}

// The name a rule is matched against: the repo's model name, or the
// checkpoint directory's own name.
std::string match_name(const Checkpoint& c) {
  if (!c.repo_id.empty()) return c.repo_id.substr(c.repo_id.find('/') + 1);
  return fs::path(c.path).lexically_normal().filename().string();
}

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

// A cached snapshot of `r` whose weights are the pinned file: a hub snapshot
// links each LFS file to blobs/<sha256>, so the link names its hash. A file
// stored in the snapshot itself (a symlink-free download) is taken as is.
// Returns the name that loads it.
std::string cached(const CompanionRepo& r) {
  const fs::path repo = repo_dir(r.repo_id);
  std::error_code ec;
  if (!fs::is_directory(repo / "snapshots", ec)) return {};
  const auto def = model::cached_snapshot(r.repo_id, "");
  std::vector<fs::path> snaps;
  for (const auto& e : fs::directory_iterator(repo / "snapshots", ec))
    if (e.is_directory(ec)) snaps.push_back(e.path());
  // The pinned revision first, then the default one, then the rest.
  std::sort(snaps.begin(), snaps.end(), [&](const fs::path& a, const fs::path& b) {
    const auto rank = [&](const fs::path& p) {
      if (p.filename() == r.revision) return 0;
      if (def.ok() && fs::path(*def).filename() == p.filename()) return 1;
      return 2;
    };
    return std::pair(rank(a), a.string()) < std::pair(rank(b), b.string());
  });
  for (const fs::path& s : snaps) {
    const fs::path f = s / r.weights_file;
    if (!fs::exists(f, ec)) continue;  // follows the link: a dangling one is absent
    if (fs::is_symlink(f, ec)) {
      if (fs::read_symlink(f, ec).filename() != r.weights_sha256) continue;
    } else if (fs::file_size(f, ec) != r.weights_bytes) {
      continue;
    }
    if (!fs::exists(s / "config.json", ec)) continue;
    const bool is_default = def.ok() && fs::path(*def).filename() == s.filename();
    return is_default ? r.repo_id : r.repo_id + "@" + s.filename().string();
  }
  return {};
}

std::string kind_label(Kind k) { return k == Kind::kMtp ? "MTP head" : "DFlash2 draft"; }

}  // namespace

std::span<const CompanionRule> builtin_rules() { return rules_table(); }

const CompanionRule* match_rule(const Checkpoint& target, std::span<const CompanionRule> rules) {
  if (target.kind != Kind::kModel) return nullptr;
  const std::string name = normalized(match_name(target));
  for (const CompanionRule& r : rules) {
    const bool named = std::any_of(r.name_keys.begin(), r.name_keys.end(), [&](const std::string& k) {
      return name.find(k) != std::string::npos;
    });
    if (!named) continue;
    if (r.architecture != target.architecture || r.hidden_size != target.hidden_size ||
        r.layers != target.layers || (r.vocab_size != 0 && r.vocab_size != target.vocab_size))
      continue;
    return &r;
  }
  return nullptr;
}

CompanionState companion_state(const Checkpoint& target, Kind kind,
                               std::span<const CompanionRule> rules) {
  CompanionState st;
  st.kind = kind;
  const CompanionRule* rule = match_rule(target, rules);
  if (kind == Kind::kMtp) {
    if (target.mtp_layers == 0) return st;
    if (rule != nullptr && !rule->mtp.empty()) st.recommended = &rule->mtp.front();
    const std::string beside = model::MtpModule::find_beside(target.repo_id.empty() ? target.path : target.name());
    if (!beside.empty()) {
      st.present = beside;
      std::error_code ec;
      st.where = fs::is_directory(beside, ec) ? "beside" : "cached";
      return st;
    }
    if (rule != nullptr) {
      for (const CompanionRepo& r : rule->mtp) {
        if (std::string name = cached(r); !name.empty()) {
          st.present = name;
          st.where = "cached";
          return st;
        }
      }
    }
    return st;
  }
  if (rule == nullptr || !rule->dflash2.has_value()) return st;
  st.recommended = &*rule->dflash2;
  st.present = cached(*rule->dflash2);
  if (!st.present.empty()) {
    st.where = "cached";
    if (auto paths = model::resolve_model(st.present); paths.ok()) {
      if (auto cache = model::dflash2_cache_path(*paths); cache.ok()) {
        std::error_code ec;
        st.converted = fs::exists(*cache / "model.safetensors", ec);
      }
    }
  }
  return st;
}

std::optional<CompanionRepo> recommended_companion(const std::string& target, Kind kind,
                                                   std::span<const CompanionRule> rules) {
  auto paths = model::resolve_model(target);
  if (!paths.ok()) return std::nullopt;
  auto t = classify(fs::path(paths->config).parent_path().string());
  if (!t.ok()) return std::nullopt;
  const CompanionRule* rule = match_rule(*t, rules);
  if (rule == nullptr) return std::nullopt;
  if (kind == Kind::kDFlash2) return rule->dflash2;
  if (rule->mtp.empty() || t->mtp_layers == 0) return std::nullopt;
  return rule->mtp.front();
}

Result<std::string> ensure_companion(const std::string& target, Kind kind, const EnsureOptions& options) {
  const char* flag = kind == Kind::kMtp ? "--mtp" : "--dflash2-model";
  LSE_ASSIGN_OR(const model::ModelPaths paths, model::resolve_model(target));
  auto t = classify(fs::path(paths.config).parent_path().string());
  if (!t.ok()) {
    if (kind == Kind::kMtp) return std::string();
    return LSE_ERROR(kInvalidArgument, "no DFlash2 draft can be chosen for '", target, "': ",
                     t.status().message(), "; name one with --dflash2-model");
  }
  if (t->repo_id.empty()) t->path = fs::path(paths.config).parent_path().string();
  const CompanionState st = companion_state(*t, kind, options.rules);
  if (!st.present.empty()) return st.present;
  if (st.recommended == nullptr) {
    if (kind == Kind::kMtp) return std::string();
    std::string known;
    for (const CompanionRule& r : options.rules) {
      if (!r.dflash2) continue;
      if (!known.empty()) known += ", ";
      known += r.family + " -> " + r.dflash2->repo_id;
    }
    return LSE_ERROR(kInvalidArgument, "no DFlash2 draft is known for '", t->name(), "' (",
                     t->architecture, ", hidden ", std::to_string(t->hidden_size), ", ",
                     std::to_string(t->layers), " layers; known: ", known.empty() ? "none" : known,
                     "); name one with --dflash2-model");
  }
  const CompanionRepo& r = *st.recommended;
  std::string how = "`" + options.prog + " pull " + r.repo_id + "@" + r.revision + "`";
  if (!t->repo_id.empty())
    how = "`" + options.prog + " pull " + t->name() + (kind == Kind::kMtp ? " --with-mtp" : " --with-dflash2") +
          "` (or " + how + ")";
  if (!options.allow_network || offline()) {
    return LSE_ERROR(kNotFound, "the ", kind_label(kind), " for ", t->name(), " (", r.repo_id, ", ",
                     human(r.weights_bytes), ") is not in the HF cache at ", model::hf_cache_root(),
                     ", and ", offline() ? "downloads are off (--offline or HF_HUB_OFFLINE)" : "this operation never downloads",
                     "; run ", how, kind == Kind::kMtp ? ", or pass --no-mtp" : "",
                     ", or name a local copy with ", flag);
  }
  std::fprintf(stderr, "lse: %s for %s: downloading %s@%.12s (%s of weights) into %s\n",
               kind_label(kind).c_str(), t->name().c_str(), r.repo_id.c_str(), r.revision.c_str(),
               human(r.weights_bytes).c_str(), model::hf_cache_root().c_str());
  PullOptions pull_options;
  pull_options.accept = {kind};
  pull_options.pinned_file = r.weights_file;
  pull_options.pinned_sha256 = r.weights_sha256;
  pull_options.progress = options.progress;
  auto pulled = pull(RepoSpec{r.repo_id, r.revision}, pull_options);
  if (!pulled.ok()) {
    return Status(pulled.status().code(),
                  "downloading the " + kind_label(kind) + " " + r.repo_id + ": " + pulled.status().message() +
                      (kind == Kind::kMtp ? " (pass --no-mtp to run without it)" : ""));
  }
  return pulled->checkpoint.name();
}

}  // namespace lse::hub
