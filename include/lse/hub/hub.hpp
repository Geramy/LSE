// Hugging Face model management: what the local hub cache holds that LSE can
// load, the companions (MTP heads, DFlash2 drafts) each target pairs with, and
// downloads into the same cache layout huggingface_hub writes, so
// huggingface-cli, transformers and mlx-lm see the same files.
//
// Offline unless a caller asks for a download (pull, or ensure_companion with
// network allowed), and never when HF_HUB_OFFLINE is set.
#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "lse/core/status.hpp"

namespace lse::hub {

// ---------------------------------------------------------------------------
// Environment, read the way huggingface_hub reads it.

// $HF_ENDPOINT without a trailing slash, else https://huggingface.co.
std::string endpoint();

// HF_TOKEN, then HUGGING_FACE_HUB_TOKEN, then the file $HF_TOKEN_PATH names
// (default $HF_HOME/token). nullopt when none is set.
struct Token {
  std::string value;
  std::string source;  // "HF_TOKEN", "/Users/x/.cache/huggingface/token", ...
};
std::optional<Token> token();

// HF_HUB_OFFLINE set to 1/true/yes/on (any case).
bool offline();
// Sets HF_HUB_OFFLINE=1 for this process: what --offline does.
void set_offline();

// ---------------------------------------------------------------------------
// Repo specs.

// "org/name" or "org/name@revision". revision is empty when none is named.
struct RepoSpec {
  std::string repo_id;
  std::string revision;
  [[nodiscard]] std::string to_string() const {
    return revision.empty() ? repo_id : repo_id + "@" + revision;
  }
};
// Validates the shape the hub accepts: one '/', and [A-Za-z0-9._-] around it.
Result<RepoSpec> parse_repo_spec(std::string_view text);
// True when `text` parses as a repo spec and is not an existing path.
bool names_a_repo(std::string_view text);

// <cache root>/models--org--name.
std::filesystem::path repo_dir(std::string_view repo_id);

// ---------------------------------------------------------------------------
// Checkpoints LSE loads.

enum class Kind : std::uint8_t {
  kModel,    // an MLX target model of an architecture this build loads
  kMtp,      // a multi-token-prediction head
  kDFlash2,  // a DFlash2 block draft
};
std::string_view to_string(Kind k) noexcept;

struct Checkpoint {
  Kind kind = Kind::kModel;
  std::string repo_id;   // empty for a directory outside the hub cache
  std::string commit;    // the snapshot directory's name (hub cache only)
  std::vector<std::string> refs;  // refs naming that commit, e.g. "main"
  std::string path;      // the checkpoint directory
  std::string architecture;  // engine architecture ("qwen3.5"), "mtp", "dflash2"
  std::string quant;     // "Q4 g64", "Q8 g64", "BF16", ...
  int bits = 0;          // 0 when unquantized
  std::uint64_t parameters = 0;
  std::uint64_t bytes = 0;   // weights on disk
  int hidden_size = 0;
  int layers = 0;
  int vocab_size = 0;
  int mtp_layers = 0;        // what a target declares
  bool dflash2_converted = false;  // a DFlash2 BF16 source with its Q8 copy cached
  bool default_revision = true;    // the snapshot a bare repo id resolves to

  // What a person types to name it: "org/name", "org/name@rev" when the
  // snapshot is not the one refs/main names, or the path.
  [[nodiscard]] std::string name() const;
};

// What a checkpoint directory is, decided by the engine's own inspection
// (lse::model::model_info): a target must be loadable and written by MLX
// (safetensors metadata format=mlx, or an mlx-lm quantization block). An MTP
// head or a DFlash2 draft must be loadable as one. Anything else is an error
// whose message says why LSE does not load it.
Result<Checkpoint> classify(const std::string& dir);

struct Listing {
  std::string cache_root;
  std::vector<Checkpoint> models;      // targets, sorted by name
  std::vector<Checkpoint> companions;  // MTP heads and DFlash2 drafts
};
// Every supported checkpoint in the hub cache (each cached revision) and in
// `dirs` (each a checkpoint directory or a directory of them), plus the
// directories $LSE_MODEL_DIRS names (':'-separated). Unsupported repos, GGUF
// files, datasets and unfinished downloads are left out.
Listing scan(std::span<const std::string> dirs = {});

// ---------------------------------------------------------------------------
// Companions: the DFlash2 draft and MTP head a target pairs with.

struct CompanionRepo {
  std::string repo_id;
  std::string revision;       // the commit a download pins
  std::string weights_file;   // "model.safetensors"
  std::string weights_sha256; // what that file must hash to
  std::uint64_t weights_bytes = 0;
};

// One family of targets: matched by name (a normalized key such as
// "qwen3827b" found in the repo id or directory name) and confirmed by the
// geometry the companion was trained against.
struct CompanionRule {
  std::string family;                 // "Qwen3.8-27B", for messages
  std::vector<std::string> name_keys; // lowercase, [a-z0-9] only
  std::string architecture;           // engine architecture
  int hidden_size = 0;
  int layers = 0;
  int vocab_size = 0;
  std::optional<CompanionRepo> dflash2;
  std::vector<CompanionRepo> mtp;     // first is recommended; all are accepted
};

std::span<const CompanionRule> builtin_rules();

// The rule a target matches, or nullptr.
const CompanionRule* match_rule(const Checkpoint& target,
                                std::span<const CompanionRule> rules = builtin_rules());

struct CompanionState {
  Kind kind = Kind::kMtp;
  const CompanionRepo* recommended = nullptr;  // nullptr: no known companion
  std::string present;  // what loads: a repo id, "org/name@rev" or a path
  std::string where;    // "beside" (an mtp/ directory in the snapshot), "cached"
  bool converted = false;  // DFlash2: the Q8 conversion is cached
};
// Whether a target's companion of `kind` is on disk, offline. For MTP an mtp/
// directory beside the target and the <name>-MTP repos the loader looks for
// count, as do every repo the rule accepts.
CompanionState companion_state(const Checkpoint& target, Kind kind,
                               std::span<const CompanionRule> rules = builtin_rules());

// ---------------------------------------------------------------------------
// Downloads.

struct Progress {
  std::string repo_id;
  std::string file;
  std::uint64_t file_done = 0, file_total = 0;
  std::uint64_t done = 0, total = 0;  // across the files being downloaded
  double bytes_per_second = 0;
  bool file_finished = false;
};
using ProgressFn = std::function<void(const Progress&)>;
// Throttled lines on stderr (a rewriting line on a terminal), what the CLI
// and lse_open print.
ProgressFn stderr_progress(std::string prefix);

struct PullOptions {
  std::vector<std::string> include;  // fnmatch globs on repo paths
  std::vector<std::string> exclude;
  // Refuse unless the repo classifies as one of these (checked from config
  // and safetensors headers before any weights are fetched).
  std::vector<Kind> accept{Kind::kModel, Kind::kMtp, Kind::kDFlash2};
  // The file whose sha256 a companion pin requires, checked against the hub's
  // metadata before the download starts.
  std::string pinned_file, pinned_sha256;
  ProgressFn progress;
  // Called once the repo has been classified, before any weights move.
  std::function<void(const Checkpoint&)> on_checked;
};

struct PullResult {
  std::string repo_id, commit, snapshot;
  Checkpoint checkpoint;
  std::size_t files = 0, reused = 0;
  std::uint64_t bytes = 0, downloaded = 0;
};

// Downloads a repo at a revision (default main) into the hub cache:
// blobs/<etag> with snapshots/<commit>/<file> symlinks and refs/<revision>,
// resuming blobs/<etag>.incomplete, holding .locks/<repo>/<etag>.lock while
// writing, and verifying each file's sha256 (LFS) or git blob hash. Refuses a
// repo LSE does not load before fetching its weights, and a download the
// filesystem has no room for.
Result<PullResult> pull(const RepoSpec& spec, const PullOptions& options);

// What --mtp/--dflash2-model default to. `target` is what --model named.
// Returns the name the loader should open (a cached repo or path), or "" when
// no companion is known for the target (MTP only; a missing DFlash2 draft is
// an error). A known companion that is not cached is downloaded when
// `allow_network` and HF_HUB_OFFLINE is unset; otherwise the error carries the
// exact pull command.
struct EnsureOptions {
  bool allow_network = true;
  ProgressFn progress;
  std::span<const CompanionRule> rules = builtin_rules();
  std::string prog = "lse-server";  // the command named in messages
};
Result<std::string> ensure_companion(const std::string& target, Kind kind,
                                     const EnsureOptions& options);

// The recommended companion for a target, cached or not; nullopt when none is
// known. Offline.
std::optional<CompanionRepo> recommended_companion(
    const std::string& target, Kind kind,
    std::span<const CompanionRule> rules = builtin_rules());

// If `name` is a repo spec that is not in the cache, pulls it (what --pull
// does for --model, --mtp and --dflash2-model). A path or a cached repo is
// left alone.
Status pull_if_missing(const std::string& name, std::span<const Kind> accept,
                       const ProgressFn& progress);

// ---------------------------------------------------------------------------
// Removal.

struct Removal {
  std::vector<std::filesystem::path> paths;  // what goes, in order
  std::uint64_t bytes = 0;                   // freed
  std::string description;
};
// Plans removing a repo (no revision: the whole repo directory and its locks)
// or one revision (its snapshot, the refs naming it, and the blobs no other
// snapshot uses). Nothing is deleted.
Result<Removal> plan_removal(const RepoSpec& spec);
Status remove(const Removal& plan);

// ---------------------------------------------------------------------------
// Command line: `models`, `pull`/`download`, shared by lse and lse-server.

// The listing as JSON (what `models --json` prints and lse_models_list
// returns): every target with its companions' state and the pull command
// that fetches a missing one.
std::string listing_json(const Listing& listing, std::string_view prog = "lse-server");

bool is_subcommand(std::string_view arg);
// argv[0] is the subcommand. Returns the process exit status.
int run_cli(std::string_view prog, int argc, char** argv);

}  // namespace lse::hub
