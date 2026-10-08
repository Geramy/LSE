// The local side of the hub: the environment, repo specs, what a checkpoint
// directory is, what the cache holds, companions on disk, and removal.
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <sstream>

#include <nlohmann/json.hpp>

#include "lse/hub/hub.hpp"
#include "lse/model/dflash2_convert.hpp"
#include "lse/model/inspect.hpp"
#include "lse/model/mtp.hpp"
#include "lse/model/weights.hpp"

namespace lse::hub {

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

const char* env_value(const char* name) {
  const char* v = std::getenv(name);
  return (v != nullptr && v[0] != '\0') ? v : nullptr;
}

std::string trim(std::string s) {
  while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back())) != 0) s.pop_back();
  std::size_t i = 0;
  while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i])) != 0) ++i;
  return s.substr(i);
}

std::string read_text(const fs::path& p) {
  std::ifstream in(p, std::ios::binary);
  if (!in) return {};
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

bool repo_char(char c) {
  return std::isalnum(static_cast<unsigned char>(c)) != 0 || c == '-' || c == '_' || c == '.';
}

// The safetensors header's __metadata__.format, which mlx-lm sets to "mlx".
std::string safetensors_format(const fs::path& file) {
  std::ifstream in(file, std::ios::binary);
  std::uint64_t n = 0;
  if (!in.read(reinterpret_cast<char*>(&n), 8) || n == 0 || n > (100u << 20)) return {};
  std::string header(static_cast<std::size_t>(n), '\0');
  if (!in.read(header.data(), static_cast<std::streamsize>(n))) return {};
  try {
    const json j = json::parse(header);
    const auto it = j.find("__metadata__");
    if (it != j.end() && it->is_object() && it->contains("format") && (*it)["format"].is_string())
      return (*it)["format"].get<std::string>();
  } catch (const std::exception&) {
  }
  return {};
}

// Written by MLX: mlx-lm stamps format=mlx into every safetensors file it
// saves, and its quantized configs carry a top-level "quantization" block
// (transformers writes only quantization_config).
bool written_by_mlx(const fs::path& dir) {
  std::error_code ec;
  std::vector<fs::path> files;
  for (const auto& e : fs::directory_iterator(dir, ec)) {
    if (e.path().extension() == ".safetensors") files.push_back(e.path());
  }
  std::sort(files.begin(), files.end());
  if (!files.empty() && safetensors_format(files.front()) == "mlx") return true;
  try {
    const json c = json::parse(read_text(dir / "config.json"));
    const auto it = c.find("quantization");
    return it != c.end() && it->is_object() && it->contains("bits");
  } catch (const std::exception&) {
    return false;
  }
}

std::string upper_dtype(const std::string& dtype) {
  if (dtype == "bfloat16") return "BF16";
  if (dtype == "float16") return "FP16";
  if (dtype == "float32") return "FP32";
  std::string out = dtype;
  for (char& c : out) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
  return out;
}

int int_or(const json& j, const char* key, int fallback = 0) {
  const auto it = j.find(key);
  return it != j.end() && it->is_number_integer() ? it->get<int>() : fallback;
}

std::uint64_t u64_or(const json& j, const char* key) {
  const auto it = j.find(key);
  return it != j.end() && it->is_number_unsigned() ? it->get<std::uint64_t>()
         : it != j.end() && it->is_number_integer() ? static_cast<std::uint64_t>(std::max<std::int64_t>(0, it->get<std::int64_t>()))
                                                    : 0;
}

// models--org--name/snapshots/<commit> -> {org/name, commit}.
std::pair<std::string, std::string> hub_origin(const fs::path& dir) {
  const fs::path abs = fs::absolute(dir).lexically_normal();
  const fs::path snapshots = abs.parent_path();
  const std::string repo = snapshots.parent_path().filename().string();
  if (snapshots.filename() != "snapshots" || !repo.starts_with("models--")) return {};
  std::string rest = repo.substr(8);
  const std::size_t sep = rest.find("--");
  if (sep == std::string::npos) return {};
  return {rest.substr(0, sep) + "/" + rest.substr(sep + 2), abs.filename().string()};
}

// refs/<name> files of a repo directory, by the commit each names.
std::map<std::string, std::vector<std::string>> repo_refs(const fs::path& repo) {
  std::map<std::string, std::vector<std::string>> out;
  std::error_code ec;
  const fs::path refs = repo / "refs";
  for (fs::recursive_directory_iterator it(refs, ec), end; !ec && it != end; it.increment(ec)) {
    if (!it->is_regular_file(ec)) continue;
    const std::string commit = trim(read_text(it->path()));
    if (!commit.empty()) out[commit].push_back(it->path().lexically_relative(refs).string());
  }
  for (auto& [commit, names] : out) std::sort(names.begin(), names.end());
  return out;
}

std::uintmax_t tree_bytes(const fs::path& root) {
  std::set<std::pair<dev_t, ino_t>> seen;
  std::uintmax_t total = 0;
  std::error_code ec;
  struct stat st {};
  if (::stat(root.c_str(), &st) == 0 && S_ISREG(st.st_mode)) return static_cast<std::uintmax_t>(st.st_size);
  for (fs::recursive_directory_iterator it(root, fs::directory_options::skip_permission_denied, ec), end;
       !ec && it != end; it.increment(ec)) {
    if (::stat(it->path().c_str(), &st) != 0 || !S_ISREG(st.st_mode)) continue;
    if (!seen.emplace(st.st_dev, st.st_ino).second) continue;
    total += static_cast<std::uintmax_t>(st.st_size);
  }
  return total;
}

void add_scanned(Listing& out, Checkpoint c) {
  (c.kind == Kind::kModel ? out.models : out.companions).push_back(std::move(c));
}

void scan_dir(Listing& out, const fs::path& dir, std::set<std::string>& seen) {
  std::error_code ec;
  const auto take = [&](const fs::path& d) {
    const std::string key = fs::weakly_canonical(d, ec).string();
    if (!seen.insert(key).second) return;
    if (auto c = classify(d.string()); c.ok()) add_scanned(out, c.release());
  };
  if (fs::exists(dir / "config.json", ec)) {
    take(dir);
    return;
  }
  std::vector<fs::path> subdirs;
  for (const auto& e : fs::directory_iterator(dir, ec)) {
    if (e.is_directory(ec) && fs::exists(e.path() / "config.json", ec)) subdirs.push_back(e.path());
  }
  std::sort(subdirs.begin(), subdirs.end());
  for (const fs::path& d : subdirs) take(d);
}

}  // namespace

// ---------------------------------------------------------------------------

std::string endpoint() {
  std::string e = env_value("HF_ENDPOINT") ? env_value("HF_ENDPOINT") : "https://huggingface.co";
  while (!e.empty() && e.back() == '/') e.pop_back();
  return e;
}

std::optional<Token> token() {
  for (const char* name : {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"}) {
    if (const char* v = env_value(name)) {
      const std::string t = trim(v);
      if (!t.empty()) return Token{t, name};
    }
  }
  const std::string path = env_value("HF_TOKEN_PATH") ? std::string(env_value("HF_TOKEN_PATH"))
                                                      : model::hf_home() + "/token";
  const std::string t = trim(read_text(path));
  if (!t.empty()) return Token{t, path};
  return std::nullopt;
}

bool offline() {
  const char* v = env_value("HF_HUB_OFFLINE");
  if (v == nullptr) return false;
  std::string s = v;
  for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return s == "1" || s == "true" || s == "yes" || s == "on";
}

void set_offline() { ::setenv("HF_HUB_OFFLINE", "1", 1); }

Result<RepoSpec> parse_repo_spec(std::string_view text) {
  RepoSpec spec;
  std::string_view id = text;
  if (const std::size_t at = text.rfind('@'); at != std::string_view::npos) {
    id = text.substr(0, at);
    spec.revision = std::string(text.substr(at + 1));
    if (spec.revision.empty() || spec.revision.find("..") != std::string::npos ||
        spec.revision.starts_with("/")) {
      return LSE_ERROR(kInvalidArgument, "'", std::string(text), "' names no usable revision after '@'");
    }
  }
  const std::size_t slash = id.find('/');
  const bool shaped = slash != std::string_view::npos && slash > 0 && slash + 1 < id.size() &&
                      id.find('/', slash + 1) == std::string_view::npos &&
                      std::all_of(id.begin(), id.end(), [](char c) { return c == '/' || repo_char(c); }) &&
                      !id.starts_with(".") && id.find("--") == std::string_view::npos &&
                      id.find("..") == std::string_view::npos;
  if (!shaped) {
    return LSE_ERROR(kInvalidArgument, "'", std::string(text),
                     "' is not a Hugging Face repo id; expected ORG/NAME or ORG/NAME@REVISION");
  }
  spec.repo_id = std::string(id);
  return spec;
}

bool names_a_repo(std::string_view text) {
  std::error_code ec;
  if (fs::exists(fs::path(std::string(text)), ec)) return false;
  return parse_repo_spec(text).ok();
}

fs::path repo_dir(std::string_view repo_id) {
  return fs::path(model::hf_cache_root()) / model::repo_cache_dir_name(repo_id);
}

std::string_view to_string(Kind k) noexcept {
  switch (k) {
    case Kind::kModel: return "model";
    case Kind::kMtp: return "mtp";
    case Kind::kDFlash2: return "dflash2";
  }
  return "model";
}

std::string Checkpoint::name() const {
  if (repo_id.empty()) return path;
  if (default_revision || commit.empty()) return repo_id;
  return repo_id + "@" + commit.substr(0, 12);
}

Result<Checkpoint> classify(const std::string& dir) {
  auto info = model::model_info(dir);
  if (!info.ok()) return info.status();
  const json& i = *info;
  const std::string kind = i.value("kind", std::string());
  const bool loadable = i.value("loadable", false);
  if (!loadable) {
    const auto r = i.find("reason");
    std::string why = r != i.end() && r->is_string() ? r->get<std::string>() : std::string("this build cannot load it");
    // The reason is a Status rendered as "<code>: <message>"; the code is noise here.
    for (const char* code : {"not_found: ", "invalid_argument: ", "unimplemented: ", "out_of_range: "})
      if (why.starts_with(code)) why.erase(0, std::strlen(code));
    return LSE_ERROR(kUnimplemented, "LSE does not load it: ", why);
  }
  Checkpoint c;
  c.path = dir;
  c.kind = kind == "mtp" ? Kind::kMtp : kind == "dflash2_draft" ? Kind::kDFlash2 : Kind::kModel;
  if (c.kind == Kind::kModel && !written_by_mlx(dir)) {
    return LSE_ERROR(kUnimplemented,
                     "not an MLX checkpoint: its safetensors carry no format=mlx metadata and its "
                     "config no mlx-lm quantization block (LSE lists and pulls MLX conversions, "
                     "e.g. from mlx-community)");
  }
  if (const auto a = i.find("architecture"); a != i.end() && a->is_string()) c.architecture = a->get<std::string>();
  c.hidden_size = int_or(i, "hidden_size");
  c.layers = int_or(i, "layers");
  c.vocab_size = int_or(i, "vocab_size");
  c.parameters = u64_or(i, "parameters");
  c.bytes = u64_or(i, "weights_disk_bytes");
  if (const auto m = i.find("mtp"); c.kind == Kind::kModel && m != i.end() && m->is_object())
    c.mtp_layers = int_or(*m, "layers");
  const auto q = i.find("quantization");
  if (q != i.end() && q->is_object() && int_or(*q, "bits") > 0) {
    c.bits = int_or(*q, "bits");
    c.quant = "Q" + std::to_string(c.bits);
    if (int_or(*q, "group_size") > 0) c.quant += " g" + std::to_string(int_or(*q, "group_size"));
  } else if (const auto d = i.find("dtype"); d != i.end() && d->is_string()) {
    c.quant = upper_dtype(d->get<std::string>());
  }
  if (c.kind == Kind::kDFlash2) {
    const json& d = i.value("dflash2", json::object());
    if (d.value("source", std::string()) == "bf16") {
      c.bits = 0;
      c.quant = "BF16";
      if (d.contains("conversion") && d["conversion"].is_object())
        c.dflash2_converted = d["conversion"].value("cached", false);
    }
  }
  if (c.quant.empty()) c.quant = "-";
  const auto [repo_id, commit] = hub_origin(dir);
  c.repo_id = repo_id;
  c.commit = commit;
  if (!repo_id.empty()) {
    const auto refs = repo_refs(repo_dir(repo_id));
    if (const auto it = refs.find(commit); it != refs.end()) c.refs = it->second;
    const auto def = model::cached_snapshot(repo_id, "");
    c.default_revision = def.ok() && fs::path(*def).filename() == commit;
  }
  return c;
}

Listing scan(std::span<const std::string> dirs) {
  Listing out;
  out.cache_root = model::hf_cache_root();
  std::error_code ec;
  std::vector<fs::path> repos;
  for (const auto& e : fs::directory_iterator(out.cache_root, ec)) {
    if (e.path().filename().string().starts_with("models--") && e.is_directory(ec)) repos.push_back(e.path());
  }
  std::sort(repos.begin(), repos.end());
  std::set<std::string> seen;
  for (const fs::path& repo : repos) {
    std::vector<fs::path> snaps;
    for (const auto& e : fs::directory_iterator(repo / "snapshots", ec)) {
      if (e.is_directory(ec)) snaps.push_back(e.path());
    }
    std::sort(snaps.begin(), snaps.end());
    for (const fs::path& s : snaps) {
      seen.insert(fs::weakly_canonical(s, ec).string());
      if (auto c = classify(s.string()); c.ok()) add_scanned(out, c.release());
    }
  }
  std::vector<std::string> extra(dirs.begin(), dirs.end());
  if (const char* env = env_value("LSE_MODEL_DIRS")) {
    std::stringstream ss(env);
    for (std::string d; std::getline(ss, d, ':');)
      if (!d.empty()) extra.push_back(d);
  }
  for (const std::string& d : extra) scan_dir(out, d, seen);
  const auto by_name = [](const Checkpoint& a, const Checkpoint& b) { return a.name() < b.name(); };
  std::sort(out.models.begin(), out.models.end(), by_name);
  std::sort(out.companions.begin(), out.companions.end(), by_name);
  return out;
}

// ---------------------------------------------------------------------------
// Removal.

Result<Removal> plan_removal(const RepoSpec& spec) {
  const fs::path repo = repo_dir(spec.repo_id);
  std::error_code ec;
  if (!fs::is_directory(repo, ec)) {
    return LSE_ERROR(kNotFound, "the HF cache at ", model::hf_cache_root(), " has no ",
                     model::repo_cache_dir_name(spec.repo_id));
  }
  Removal plan;
  const fs::path locks = fs::path(model::hf_cache_root()) / ".locks" / repo.filename();
  std::vector<fs::path> snaps;
  for (const auto& e : fs::directory_iterator(repo / "snapshots", ec))
    if (e.is_directory(ec)) snaps.push_back(e.path());

  const auto whole_repo = [&]() {
    plan.paths.push_back(repo);
    if (fs::exists(locks, ec)) plan.paths.push_back(locks);
    plan.bytes = tree_bytes(repo);
    plan.description = spec.repo_id + " (" + std::to_string(snaps.size()) + " revision" +
                       (snaps.size() == 1 ? "" : "s") + ")";
  };
  if (spec.revision.empty()) {
    whole_repo();
    return plan;
  }
  LSE_ASSIGN_OR(const std::string snap_text, model::cached_snapshot(spec.repo_id, spec.revision));
  const fs::path snap(snap_text);
  if (snaps.size() <= 1) {
    whole_repo();
    plan.description = spec.repo_id + "@" + snap.filename().string() + " (its only revision: the whole repo)";
    return plan;
  }
  // Blobs each snapshot points at, by file name.
  const auto blobs_of = [&](const fs::path& s) {
    std::set<std::string> out;
    for (fs::recursive_directory_iterator it(s, ec), end; !ec && it != end; it.increment(ec)) {
      if (!it->is_symlink(ec)) continue;
      const fs::path target = fs::read_symlink(it->path(), ec);
      if (!ec) out.insert(target.filename().string());
    }
    return out;
  };
  const std::set<std::string> mine = blobs_of(snap);
  std::set<std::string> others;
  for (const fs::path& s : snaps) {
    if (s.filename() == snap.filename()) continue;
    const auto b = blobs_of(s);
    others.insert(b.begin(), b.end());
  }
  plan.paths.push_back(snap);
  plan.bytes = 0;
  // Files stored in the snapshot itself (an LSE conversion cache, a
  // symlink-free download) go with it.
  {
    std::set<std::pair<dev_t, ino_t>> counted;
    struct stat st {};
    for (fs::recursive_directory_iterator it(snap, ec), end; !ec && it != end; it.increment(ec)) {
      if (it->is_symlink(ec) || ::lstat(it->path().c_str(), &st) != 0 || !S_ISREG(st.st_mode)) continue;
      if (counted.emplace(st.st_dev, st.st_ino).second) plan.bytes += static_cast<std::uint64_t>(st.st_size);
    }
  }
  for (const std::string& b : mine) {
    if (others.contains(b)) continue;
    const fs::path blob = repo / "blobs" / b;
    if (!fs::exists(blob, ec)) continue;
    plan.paths.push_back(blob);
    plan.bytes += fs::file_size(blob, ec);
  }
  for (const auto& [commit, names] : repo_refs(repo)) {
    if (commit != snap.filename().string()) continue;
    for (const std::string& n : names) plan.paths.push_back(repo / "refs" / n);
  }
  plan.description = spec.repo_id + "@" + snap.filename().string();
  return plan;
}

Status remove(const Removal& plan) {
  for (const fs::path& p : plan.paths) {
    std::error_code ec;
    fs::remove_all(p, ec);
    if (ec) return LSE_ERROR(kIoError, "cannot remove ", p.string(), ": ", ec.message());
  }
  return OkStatus();
}

}  // namespace lse::hub
