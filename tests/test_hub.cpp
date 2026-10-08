// Hugging Face model management (lse/hub/hub.hpp) against a fake hub cache
// written here — symlinked blobs, several revisions, refs — and a mock hub
// HTTP server on loopback that speaks the parts of the Hub API a pull uses:
// the revision's file list, resolve with a CDN redirect, ranges, ETags, gated
// and missing repos. Nothing here reaches the network or a device.
#include "harness.hpp"

#include <unistd.h>

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <regex>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

#define CPPHTTPLIB_USE_POLL
#include "httplib.h"

#include "lse/core/sha256.hpp"
#include "lse/hub/hub.hpp"
#include "lse/model/weights.hpp"
#include "sha1.hpp"

using namespace lse;
using json = nlohmann::json;
namespace fs = std::filesystem;

#define LSE_EXPECT_HAS(text, part)                                                         \
  do {                                                                                     \
    const std::string _t = (text), _p = (part);                                            \
    if (_t.find(_p) == std::string::npos)                                                  \
      ::lse::test::fail(__FILE__, __LINE__, "'" + _p + "' not in: " + _t);                 \
  } while (0)

namespace {

// --- fixture checkpoints --------------------------------------------------

struct TensorSpec {
  std::string name;
  std::string dtype;
  std::vector<std::int64_t> shape;
};

std::string safetensors(const std::vector<TensorSpec>& tensors, bool mlx) {
  json header = json::object();
  std::uint64_t offset = 0;
  for (const auto& t : tensors) {
    std::uint64_t n = t.dtype == "BF16" || t.dtype == "F16" ? 2 : 4;
    for (auto d : t.shape) n *= static_cast<std::uint64_t>(d);
    header[t.name] = {{"dtype", t.dtype}, {"shape", t.shape}, {"data_offsets", {offset, offset + n}}};
    offset += n;
  }
  if (mlx) header["__metadata__"] = {{"format", "mlx"}};
  std::string text = header.dump();
  text.append((8 - text.size() % 8) % 8, ' ');
  std::string out(8, '\0');
  const std::uint64_t len = text.size();
  std::memcpy(out.data(), &len, 8);
  out += text;
  // Not zeros: a payload that differs byte to byte makes a corrupted or
  // misplaced range visible to the hash.
  for (std::uint64_t i = 0; i < offset; ++i) out += static_cast<char>((i * 7 + 3) & 0x3f);
  return out;
}

constexpr std::int64_t kD = 64, kVocab = 64, kInter = 128;
constexpr std::int64_t kQH = 2, kKVH = 1, kHD = 64;
constexpr std::int64_t kGK = 2, kGV = 4, kGHD = 32, kConv = 2 * kGK * kGHD + kGV * kGHD;

json qwen_config(bool mlx_quant_block) {
  json t{{"vocab_size", kVocab},
         {"hidden_size", kD},
         {"num_hidden_layers", 4},
         {"rms_norm_eps", 1e-6},
         {"full_attention_interval", 4},
         {"num_attention_heads", kQH},
         {"num_key_value_heads", kKVH},
         {"head_dim", kHD},
         {"linear_num_key_heads", kGK},
         {"linear_num_value_heads", kGV},
         {"linear_conv_kernel_dim", 4},
         {"linear_key_head_dim", kGHD},
         {"linear_value_head_dim", kGHD},
         {"rope_parameters", {{"rope_theta", 1e6}, {"partial_rotary_factor", 0.25}}},
         {"max_position_embeddings", 2048},
         {"tie_word_embeddings", false},
         {"mtp_num_hidden_layers", 1},
         {"dtype", "bfloat16"},
         {"intermediate_size", kInter},
         {"layer_types", {"linear_attention", "linear_attention", "linear_attention", "full_attention"}}};
  json c{{"architectures", {"Qwen3_5ForConditionalGeneration"}}, {"model_type", "qwen3_5"}, {"text_config", t}};
  if (mlx_quant_block) c["quantization"] = {{"group_size", 64}, {"bits", 4}, {"mode", "affine"}};
  c["quantization_config"] = {{"group_size", 64}, {"bits", 4}, {"mode", "affine"}};
  return c;
}

void q4(std::vector<TensorSpec>& out, const std::string& m, std::int64_t rows, std::int64_t cols) {
  out.push_back({m + ".weight", "U32", {rows, cols * 4 / 32}});
  out.push_back({m + ".scales", "BF16", {rows, cols / 64}});
  out.push_back({m + ".biases", "BF16", {rows, cols / 64}});
}

std::vector<TensorSpec> qwen_tensors() {
  std::vector<TensorSpec> t;
  const std::string m = "language_model.model.";
  t.push_back({m + "embed_tokens.weight", "BF16", {kVocab, kD}});
  t.push_back({m + "norm.weight", "BF16", {kD}});
  t.push_back({"language_model.lm_head.weight", "BF16", {kVocab, kD}});
  for (int i = 0; i < 4; ++i) {
    const std::string l = m + "layers." + std::to_string(i) + ".";
    t.push_back({l + "input_layernorm.weight", "BF16", {kD}});
    t.push_back({l + "post_attention_layernorm.weight", "BF16", {kD}});
    if (i == 3) {
      t.push_back({l + "self_attn.q_proj.weight", "BF16", {2 * kQH * kHD, kD}});
      t.push_back({l + "self_attn.k_proj.weight", "BF16", {kKVH * kHD, kD}});
      t.push_back({l + "self_attn.v_proj.weight", "BF16", {kKVH * kHD, kD}});
      t.push_back({l + "self_attn.o_proj.weight", "BF16", {kD, kQH * kHD}});
      t.push_back({l + "self_attn.q_norm.weight", "BF16", {kHD}});
      t.push_back({l + "self_attn.k_norm.weight", "BF16", {kHD}});
    } else {
      t.push_back({l + "linear_attn.in_proj_qkv.weight", "BF16", {kConv, kD}});
      t.push_back({l + "linear_attn.in_proj_z.weight", "BF16", {kGV * kGHD, kD}});
      t.push_back({l + "linear_attn.in_proj_a.weight", "BF16", {kGV, kD}});
      t.push_back({l + "linear_attn.in_proj_b.weight", "BF16", {kGV, kD}});
      t.push_back({l + "linear_attn.conv1d.weight", "BF16", {kConv, 4, 1}});
      t.push_back({l + "linear_attn.A_log", "F32", {kGV}});
      t.push_back({l + "linear_attn.dt_bias", "BF16", {kGV}});
      t.push_back({l + "linear_attn.norm.weight", "F32", {kGHD}});
      t.push_back({l + "linear_attn.out_proj.weight", "BF16", {kD, kGV * kGHD}});
    }
    q4(t, l + "mlp.gate_proj", kInter, kD);
    q4(t, l + "mlp.up_proj", kInter, kD);
    q4(t, l + "mlp.down_proj", kD, kInter);
  }
  return t;
}

std::vector<TensorSpec> mtp_tensors() {
  return {{"fc.weight", "BF16", {kD, 2 * kD}},
          {"pre_fc_norm_hidden.weight", "BF16", {kD}},
          {"pre_fc_norm_embedding.weight", "BF16", {kD}},
          {"norm.weight", "BF16", {kD}},
          {"layers.0.input_layernorm.weight", "BF16", {kD}},
          {"layers.0.post_attention_layernorm.weight", "BF16", {kD}},
          {"layers.0.self_attn.q_proj.weight", "BF16", {2 * kQH * kHD, kD}},
          {"layers.0.self_attn.k_proj.weight", "BF16", {kKVH * kHD, kD}},
          {"layers.0.self_attn.v_proj.weight", "BF16", {kKVH * kHD, kD}},
          {"layers.0.self_attn.o_proj.weight", "BF16", {kD, kQH * kHD}},
          {"layers.0.self_attn.q_norm.weight", "BF16", {kHD}},
          {"layers.0.self_attn.k_norm.weight", "BF16", {kHD}},
          {"layers.0.mlp.gate_proj.weight", "BF16", {kInter, kD}},
          {"layers.0.mlp.up_proj.weight", "BF16", {kInter, kD}},
          {"layers.0.mlp.down_proj.weight", "BF16", {kD, kInter}}};
}

std::vector<TensorSpec> dflash2_tensors() {
  return {{"fc.weight", "BF16", {kD, 2 * kD}},
          {"hidden_norm.weight", "BF16", {kD}},
          {"norm.weight", "BF16", {kD}},
          {"candidate_selector.hidden_projection.weight", "BF16", {64, kD}},
          {"candidate_selector.predecessor_codebook", "BF16", {kVocab, 64}},
          {"candidate_selector.successor_codebook", "BF16", {kVocab, 64}},
          {"layers.0.input_layernorm.weight", "BF16", {kD}},
          {"layers.0.post_attention_layernorm.weight", "BF16", {kD}},
          {"layers.0.self_attn.q_proj.weight", "BF16", {64, kD}},
          {"layers.0.self_attn.k_proj.weight", "BF16", {32, kD}},
          {"layers.0.self_attn.v_proj.weight", "BF16", {32, kD}},
          {"layers.0.self_attn.o_proj.weight", "BF16", {kD, 64}},
          {"layers.0.self_attn.q_norm.weight", "BF16", {32}},
          {"layers.0.self_attn.k_norm.weight", "BF16", {32}},
          {"layers.0.mlp.gate_proj.weight", "BF16", {kInter, kD}},
          {"layers.0.mlp.up_proj.weight", "BF16", {kInter, kD}},
          {"layers.0.mlp.down_proj.weight", "BF16", {kD, kInter}},
          {"layers.0.attention_conv.base_kernel", "BF16", {2, 2, kD}},
          {"layers.0.attention_conv.kernel_projection.weight", "BF16", {16, kD}},
          {"layers.0.mlp_conv.base_kernel", "BF16", {2, 2, kD}},
          {"layers.0.mlp_conv.kernel_projection.weight", "BF16", {16, kD}}};
}

json dflash2_config() {
  return json{{"architectures", {"DFlash2DraftModel"}},
              {"model_type", "qwen3"},
              {"is_causal", false},
              {"hidden_size", kD},
              {"vocab_size", kVocab},
              {"num_hidden_layers", 1},
              {"num_target_layers", 4},
              {"num_attention_heads", 2},
              {"num_key_value_heads", 1},
              {"head_dim", 32},
              {"intermediate_size", kInter},
              {"sliding_window", 16},
              {"rms_norm_eps", 1e-6},
              {"rope_parameters", {{"rope_type", "default"}, {"rope_theta", 1e6}}},
              {"layer_types", {"sliding_attention"}},
              {"max_position_embeddings", 2048},
              {"dtype", "bfloat16"},
              {"dflash_config", {{"block_size", 4},
                                 {"mask_token_id", 63},
                                 {"target_layer_ids", {1, 3}},
                                 {"conv_group_size", 16},
                                 {"conv_kernel_size", 2},
                                 {"selector_rank", 64},
                                 {"selector_top_k", 4}}}};
}

using Files = std::map<std::string, std::string>;  // repo path -> content

Files target_files(bool mlx = true) {
  return {{"config.json", qwen_config(mlx).dump(2)},
          {"model.safetensors", safetensors(qwen_tensors(), mlx)},
          {"README.md", "# tiny qwen\n"},
          {"assets/note.txt", "nested file\n"}};
}
Files mtp_files() {
  return {{"config.json", qwen_config(true).dump(2)}, {"model.safetensors", safetensors(mtp_tensors(), true)}};
}
Files dflash2_files() {
  return {{"config.json", dflash2_config().dump(2)}, {"model.safetensors", safetensors(dflash2_tensors(), false)}};
}

bool is_lfs(const std::string& path) { return path.ends_with(".safetensors"); }
std::string etag_of(const std::string& path, const std::string& content) {
  return is_lfs(path) ? sha256(content) : hub::git_blob_sha1(content);
}

// --- environment ----------------------------------------------------------

class ScopedEnv {
 public:
  ScopedEnv(const char* name, const char* value) : name_(name) {
    if (const char* old = std::getenv(name)) {
      had_ = true;
      old_ = old;
    }
    if (value) ::setenv(name, value, 1);
    else ::unsetenv(name);
  }
  ~ScopedEnv() {
    if (had_) ::setenv(name_.c_str(), old_.c_str(), 1);
    else ::unsetenv(name_.c_str());
  }

 private:
  std::string name_, old_;
  bool had_ = false;
};

// A hub cache under a temp directory, laid out the way huggingface_hub lays
// it out: content in blobs/<etag>, relative symlinks from snapshots/<commit>.
class FakeCache {
 public:
  explicit FakeCache(const std::string& tag)
      : root_(fs::temp_directory_path() / ("lse-test-hub-" + tag + "-" + std::to_string(::getpid()))),
        hub_cache_("HF_HUB_CACHE", (root_ / "hub").c_str()),
        legacy_("HUGGINGFACE_HUB_CACHE", nullptr),
        home_("HF_HOME", root_.c_str()),
        offline_("HF_HUB_OFFLINE", nullptr),
        token_("HF_TOKEN", nullptr),
        token2_("HUGGING_FACE_HUB_TOKEN", nullptr),
        dirs_("LSE_MODEL_DIRS", nullptr) {
    std::error_code ec;
    fs::remove_all(root_, ec);
    fs::create_directories(root_ / "hub", ec);
  }
  ~FakeCache() {
    std::error_code ec;
    fs::remove_all(root_, ec);
  }
  [[nodiscard]] fs::path hub() const { return root_ / "hub"; }
  [[nodiscard]] const fs::path& root() const { return root_; }

  fs::path add(const std::string& repo, const std::string& commit, const Files& files,
               const std::vector<std::string>& refs = {"main"}) {
    const fs::path dir = hub() / model::repo_cache_dir_name(repo);
    const fs::path snap = dir / "snapshots" / commit;
    for (const auto& [path, content] : files) {
      const std::string etag = etag_of(path, content);
      const fs::path blob = dir / "blobs" / etag;
      fs::create_directories(blob.parent_path());
      std::ofstream(blob, std::ios::binary) << content;
      const fs::path pointer = snap / path;
      fs::create_directories(pointer.parent_path());
      std::error_code ec;
      fs::remove(pointer, ec);
      fs::create_symlink(blob.lexically_relative(pointer.parent_path()), pointer);
    }
    for (const std::string& r : refs) {
      fs::create_directories(dir / "refs");
      std::ofstream(dir / "refs" / r) << commit;
    }
    return snap;
  }

 private:
  fs::path root_;
  ScopedEnv hub_cache_, legacy_, home_, offline_, token_, token2_, dirs_;
};

std::string commit_id(char c) { return std::string(40, c); }

// --- mock hub server ------------------------------------------------------

struct MockRepo {
  std::string id;
  std::string commit;
  Files files;
  std::string gated;           // "" or "manual"
  std::string required_token;  // resolve answers 401 GatedRepo without it
};

class MockHub {
 public:
  MockHub() {
    server_.Get(R"(/api/models/([^/]+)/([^/]+)/revision/([^/?]+))",
                [this](const httplib::Request& req, httplib::Response& res) { info(req, res); });
    server_.Get(R"(/([^/]+)/([^/]+)/resolve/([^/]+)/(.+))",
                [this](const httplib::Request& req, httplib::Response& res) { resolve(req, res); });
    server_.Get(R"(/cdn/([0-9a-f]+))",
                [this](const httplib::Request& req, httplib::Response& res) { cdn(req, res); });
    port_ = server_.bind_to_any_port("127.0.0.1");
    thread_ = std::thread([this] { server_.listen_after_bind(); });
    server_.wait_until_ready();
    endpoint_ = std::make_unique<ScopedEnv>("HF_ENDPOINT", url().c_str());
  }
  ~MockHub() {
    server_.stop();
    thread_.join();
  }
  [[nodiscard]] std::string url() const { return "http://127.0.0.1:" + std::to_string(port_); }

  void add(std::string id, std::string commit, Files files, std::string gated = {},
           std::string required_token = {}) {
    std::lock_guard g(lock_);
    MockRepo r;
    r.id = id;
    r.commit = std::move(commit);
    r.files = std::move(files);
    r.gated = std::move(gated);
    r.required_token = std::move(required_token);
    repos_[id] = std::move(r);
  }

  // Knobs.
  std::string wrong_etag_for;  // advertise a different etag for this path
  std::string corrupt;         // serve this path with one byte changed
  std::size_t truncate_after = 0;  // close a weights body after this many bytes

  struct Seen {
    std::string path, range, auth;
  };
  std::vector<Seen> seen() {
    std::lock_guard g(lock_);
    return seen_;
  }
  // Bytes of .safetensors payload served without a header-sized Range: what a
  // refused repo must never cost.
  std::size_t weight_bodies() {
    std::lock_guard g(lock_);
    return weight_bodies_;
  }

 private:
  void record(const httplib::Request& req) {
    std::lock_guard g(lock_);
    seen_.push_back({req.path, req.get_header_value("Range"), req.get_header_value("Authorization")});
  }

  const MockRepo* find(const std::string& id) {
    std::lock_guard g(lock_);
    const auto it = repos_.find(id);
    return it == repos_.end() ? nullptr : &it->second;
  }

  void info(const httplib::Request& req, httplib::Response& res) {
    record(req);
    const std::string id = req.matches[1].str() + "/" + req.matches[2].str();
    const std::string rev = req.matches[3].str();
    const MockRepo* r = find(id);
    if (r == nullptr) {
      res.status = 401;
      res.set_header("X-Error-Message", "Invalid username or password.");
      return;
    }
    if (rev != "main" && rev != r->commit) {
      res.status = 404;
      res.set_header("X-Error-Code", "RevisionNotFound");
      res.set_header("X-Error-Message", "Invalid rev id: " + rev);
      return;
    }
    json siblings = json::array();
    for (const auto& [path, content] : r->files) {
      json s{{"rfilename", path}, {"size", content.size()}, {"blobId", hub::git_blob_sha1(content)}};
      if (is_lfs(path)) s["lfs"] = {{"sha256", sha256(content)}, {"size", content.size()}, {"pointerSize", 134}};
      siblings.push_back(s);
    }
    json body{{"id", r->id}, {"sha", r->commit}, {"private", false}, {"siblings", siblings}};
    body["gated"] = r->gated.empty() ? json(false) : json(r->gated);
    res.set_content(body.dump(), "application/json");
  }

  void resolve(const httplib::Request& req, httplib::Response& res) {
    record(req);
    const std::string id = req.matches[1].str() + "/" + req.matches[2].str();
    const std::string path = req.matches[4].str();
    const MockRepo* r = find(id);
    if (r == nullptr) {
      res.status = 401;
      return;
    }
    if (!r->required_token.empty() && req.get_header_value("Authorization") != "Bearer " + r->required_token) {
      res.status = 401;
      res.set_header("X-Error-Code", "GatedRepo");
      res.set_header("X-Error-Message", "Access to model " + id + " is restricted.");
      return;
    }
    const auto f = r->files.find(path);
    if (f == r->files.end() || req.matches[3].str() != r->commit) {
      res.status = 404;
      res.set_header("X-Error-Code", "EntryNotFound");
      return;
    }
    std::string etag = etag_of(path, f->second);
    if (path == wrong_etag_for) etag = std::string(etag.size(), 'e');
    res.set_header("X-Repo-Commit", r->commit);
    if (is_lfs(path)) {
      // The hub's answer for an LFS file: the identity here, the bytes on a CDN.
      res.status = 302;
      res.set_header("X-Linked-Etag", "\"" + etag + "\"");
      res.set_header("X-Linked-Size", std::to_string(f->second.size()));
      res.set_header("Location", "/cdn/" + sha256(f->second));
      return;
    }
    res.set_header("ETag", "\"" + etag + "\"");
    serve(req, res, path, f->second);
  }

  void cdn(const httplib::Request& req, httplib::Response& res) {
    record(req);
    const std::string sha = req.matches[1].str();
    std::string path, content;
    bool found = false;
    {
      std::lock_guard g(lock_);
      for (const auto& [id, r] : repos_) {
        for (const auto& [p, c] : r.files) {
          if (!found && is_lfs(p) && sha256(c) == sha) {
            path = p;
            content = c;
            found = true;
          }
        }
      }
    }
    if (!found) {
      res.status = 404;
      return;
    }
    serve(req, res, path, content);
  }

  // httplib answers a Range request itself (206 and Content-Range, or 416)
  // from the full body, so only the whole content is handed over here.
  void serve(const httplib::Request& req, httplib::Response& res, const std::string& path, std::string content) {
    if (path == corrupt && !content.empty()) content[content.size() / 2] ^= 0x01;
    const std::string range = req.get_header_value("Range");
    std::size_t span = content.size();
    std::smatch m;
    if (std::regex_match(range, m, std::regex(R"(bytes=(\d+)-(\d*))"))) {
      const std::size_t from = std::stoull(m[1].str());
      const std::size_t to = m[2].str().empty() ? content.size() - 1
                                                : std::min<std::size_t>(std::stoull(m[2].str()), content.size() - 1);
      span = from <= to ? to - from + 1 : 0;
    }
    if (is_lfs(path)) {
      std::lock_guard g(lock_);
      if (range.empty() || span > (1u << 20)) weight_bodies_ += span;
    }
    if (is_lfs(path) && range.empty() && truncate_after != 0 && content.size() > truncate_after) {
      // A connection that drops: announce the full length, send part of it.
      const std::size_t limit = truncate_after;
      res.set_content_provider(content.size(), "application/octet-stream",
                               [content, limit](std::size_t offset, std::size_t, httplib::DataSink& sink) {
                                 if (offset >= limit) return false;
                                 sink.write(content.data() + offset, limit - offset);
                                 return true;
                               });
      return;
    }
    res.set_content(content, "application/octet-stream");
  }

  httplib::Server server_;
  int port_ = 0;
  std::thread thread_;
  std::unique_ptr<ScopedEnv> endpoint_;
  std::mutex lock_;
  std::map<std::string, MockRepo> repos_;
  std::vector<Seen> seen_;
  std::size_t weight_bodies_ = 0;
};

std::string read(const fs::path& p) {
  std::ifstream in(p, std::ios::binary);
  return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

// Companion rules for the fixtures: a family "TinyQwen" matched by name and
// confirmed by the fixture's widths.
std::vector<hub::CompanionRule> tiny_rules(const Files& dflash2, const Files& mtp) {
  hub::CompanionRule r;
  r.family = "TinyQwen";
  r.name_keys = {"tinyqwen"};
  r.architecture = "qwen3.5";
  r.hidden_size = kD;
  r.layers = 4;
  r.vocab_size = kVocab;
  r.dflash2 = hub::CompanionRepo{"test/TinyQwen-DFlash2", commit_id('d'), "model.safetensors",
                                 sha256(dflash2.at("model.safetensors")),
                                 dflash2.at("model.safetensors").size()};
  r.mtp = {hub::CompanionRepo{"test/TinyQwen-MTP-8bit", commit_id('8'), "model.safetensors",
                              sha256(mtp.at("model.safetensors")), mtp.at("model.safetensors").size()}};
  return {r};
}

}  // namespace

// --- repo specs and the environment ------------------------------------------

LSE_TEST(repo_specs_parse_and_refuse_what_the_hub_would) {
  auto a = hub::parse_repo_spec("mlx-community/Qwen3.5-0.8B-4bit");
  LSE_EXPECT(a.ok() && a->repo_id == "mlx-community/Qwen3.5-0.8B-4bit" && a->revision.empty());
  auto b = hub::parse_repo_spec("org/name@refs/pr/1");
  LSE_EXPECT(b.ok() && b->repo_id == "org/name" && b->revision == "refs/pr/1");
  for (const char* bad : {"name", "/abs/path", "a/b/c", "org/", "org/name@", "a b/c", "org--x/y", "../x/y"}) {
    LSE_EXPECT(!hub::parse_repo_spec(bad).ok());
  }
}

LSE_TEST(token_endpoint_and_offline_follow_huggingface_hub) {
  FakeCache cache("env");
  LSE_EXPECT(!hub::token().has_value());
  std::ofstream(cache.root() / "token") << "hf_from_file\n";
  LSE_EXPECT(hub::token().has_value() && hub::token()->value == "hf_from_file");
  {
    ScopedEnv t("HF_TOKEN", "hf_from_env");
    LSE_EXPECT(hub::token()->value == "hf_from_env" && hub::token()->source == "HF_TOKEN");
  }
  {
    ScopedEnv e("HF_ENDPOINT", "https://mirror.example/");
    LSE_EXPECT(hub::endpoint() == "https://mirror.example");
  }
  LSE_EXPECT(!hub::offline());
  ScopedEnv o("HF_HUB_OFFLINE", "TRUE");
  LSE_EXPECT(hub::offline());
}

// --- classification ----------------------------------------------------------

LSE_TEST(classify_accepts_mlx_targets_and_companions_and_refuses_the_rest) {
  FakeCache cache("classify");
  const fs::path t = cache.add("mlx-community/TinyQwen-4bit", commit_id('a'), target_files());
  auto c = hub::classify(t.string());
  LSE_EXPECT_OK(c.status());
  if (c.ok()) {
    LSE_EXPECT(c->kind == hub::Kind::kModel);
    LSE_EXPECT(c->architecture == "qwen3.5");
    LSE_EXPECT(c->quant == "Q4 g64");
    LSE_EXPECT(c->repo_id == "mlx-community/TinyQwen-4bit");
    LSE_EXPECT(c->commit == commit_id('a'));
    LSE_EXPECT(c->refs == std::vector<std::string>{"main"});
    LSE_EXPECT(c->mtp_layers == 1);
    LSE_EXPECT(c->name() == "mlx-community/TinyQwen-4bit");
  }
  // The same weights written by transformers: no format=mlx, no mlx-lm block.
  const fs::path hf = cache.add("Qwen/TinyQwen", commit_id('b'), target_files(false));
  auto refused = hub::classify(hf.string());
  LSE_EXPECT(!refused.ok());
  LSE_EXPECT_HAS(refused.status().message(), "not an MLX checkpoint");

  auto m = hub::classify(cache.add("test/TinyQwen-MTP-8bit", commit_id('8'), mtp_files()).string());
  LSE_EXPECT(m.ok() && m->kind == hub::Kind::kMtp);
  auto d = hub::classify(cache.add("test/TinyQwen-DFlash2", commit_id('d'), dflash2_files()).string());
  LSE_EXPECT(d.ok() && d->kind == hub::Kind::kDFlash2 && d->quant == "BF16");

  const fs::path g = cache.add("someone/Tiny-GGUF", commit_id('c'), {{"tiny-Q4_K_M.gguf", "GGUF...."}});
  LSE_EXPECT(!hub::classify(g.string()).ok());
  const fs::path u = cache.add("someone/Encoder", commit_id('e'),
                               {{"config.json", "{\"model_type\":\"bert\"}"},
                                {"model.safetensors", safetensors({{"encoder.layer.0.weight", "F32", {2, 2}}}, true)}});
  auto unknown = hub::classify(u.string());
  LSE_EXPECT(!unknown.ok());
  LSE_EXPECT_HAS(unknown.status().message(), "LSE does not load it");
}

// --- the cache listing ---------------------------------------------------------

LSE_TEST(scan_lists_only_supported_mlx_models_with_every_cached_revision) {
  FakeCache cache("scan");
  cache.add("mlx-community/TinyQwen-4bit", commit_id('a'), target_files(), {"main"});
  cache.add("mlx-community/TinyQwen-4bit", commit_id('f'), target_files(), {"v1"});
  cache.add("Qwen/TinyQwen", commit_id('b'), target_files(false));
  cache.add("someone/Tiny-GGUF", commit_id('c'), {{"tiny.gguf", "GGUF"}});
  cache.add("test/TinyQwen-DFlash2", commit_id('d'), dflash2_files());
  // An unfinished download: the index names a shard that is not there.
  {
    Files partial = target_files();
    partial.erase("model.safetensors");
    partial["model.safetensors.index.json"] =
        json{{"weight_map", {{"language_model.model.norm.weight", "model-00001-of-00001.safetensors"}}}}.dump();
    cache.add("mlx-community/Partial-4bit", commit_id('9'), partial);
  }
  fs::create_directories(cache.hub() / "datasets--someone--data" / "snapshots" / commit_id('1'));

  const hub::Listing l = hub::scan();
  LSE_EXPECT_EQ(l.models.size(), 2u);
  if (l.models.size() == 2) {
    LSE_EXPECT(l.models[0].name() == "mlx-community/TinyQwen-4bit");
    LSE_EXPECT(l.models[0].default_revision);
    // The revision refs/main does not name is spelled with its commit.
    LSE_EXPECT(l.models[1].name() == "mlx-community/TinyQwen-4bit@" + commit_id('f').substr(0, 12));
    LSE_EXPECT(l.models[1].refs == std::vector<std::string>{"v1"});
  }
  LSE_EXPECT_EQ(l.companions.size(), 1u);
  const json j = json::parse(hub::listing_json(l));
  LSE_EXPECT_EQ(j["models"].size(), 2u);
  LSE_EXPECT(j["models"][0]["mtp"]["status"] == "none");

  // A checkpoint outside the cache, through --dir / $LSE_MODEL_DIRS.
  const fs::path local = cache.root() / "models" / "tinyqwen-q4";
  fs::create_directories(local);
  for (const auto& [p, content] : target_files()) {
    fs::create_directories((local / p).parent_path());
    std::ofstream(local / p, std::ios::binary) << content;
  }
  ScopedEnv dirs("LSE_MODEL_DIRS", (cache.root() / "models").c_str());
  const hub::Listing with_local = hub::scan();
  LSE_EXPECT_EQ(with_local.models.size(), 3u);
}

LSE_TEST(a_revision_resolves_by_ref_commit_or_unique_prefix) {
  FakeCache cache("revisions");
  const fs::path main_snap = cache.add("org/TinyQwen-4bit", "aaaa1111" + std::string(32, '0'), target_files(), {"main"});
  const fs::path v1_snap = cache.add("org/TinyQwen-4bit", "aaaa2222" + std::string(32, '0'), target_files(), {"v1"});
  auto by_default = model::resolve_model("org/TinyQwen-4bit");
  LSE_EXPECT(by_default.ok() && fs::path(by_default->config).parent_path() == main_snap);
  auto by_ref = model::resolve_model("org/TinyQwen-4bit@v1");
  LSE_EXPECT(by_ref.ok() && fs::path(by_ref->config).parent_path() == v1_snap);
  auto by_commit = model::resolve_model("org/TinyQwen-4bit@aaaa2222" + std::string(32, '0'));
  LSE_EXPECT(by_commit.ok() && fs::path(by_commit->config).parent_path() == v1_snap);
  auto by_prefix = model::resolve_model("org/TinyQwen-4bit@aaaa111");
  LSE_EXPECT(by_prefix.ok() && fs::path(by_prefix->config).parent_path() == main_snap);
  auto ambiguous = model::resolve_model("org/TinyQwen-4bit@aaaa");
  LSE_EXPECT(!ambiguous.ok());
  auto two = model::resolve_model("org/TinyQwen-4bit@aaaa000");
  LSE_EXPECT(!two.ok());
  LSE_EXPECT_HAS(two.status().message(), "lse-server pull org/TinyQwen-4bit@aaaa000");
  auto missing = model::resolve_model("org/Absent-4bit");
  LSE_EXPECT(!missing.ok());
  LSE_EXPECT_HAS(missing.status().message(), "lse-server pull org/Absent-4bit");
  cache.add("someone/Tiny-GGUF", commit_id('c'), {{"tiny.gguf", "GGUF"}, {"config.json", "{}"}});
  auto gguf = model::resolve_model("someone/Tiny-GGUF");
  LSE_EXPECT(!gguf.ok());
  LSE_EXPECT_HAS(gguf.status().message(), "GGUF repository");
}

LSE_TEST(removing_a_revision_keeps_blobs_another_revision_uses) {
  FakeCache cache("rm");
  Files second = target_files();
  second["README.md"] = "# changed\n";
  const fs::path a = cache.add("org/TinyQwen-4bit", commit_id('a'), target_files(), {"main"});
  const fs::path b = cache.add("org/TinyQwen-4bit", commit_id('b'), second, {"v2"});
  auto plan = hub::plan_removal({"org/TinyQwen-4bit", "v2"});
  LSE_EXPECT_OK(plan.status());
  if (!plan.ok()) return;
  LSE_EXPECT_OK(hub::remove(*plan));
  const fs::path repo = cache.hub() / "models--org--TinyQwen-4bit";
  LSE_EXPECT(!fs::exists(b));
  LSE_EXPECT(!fs::exists(repo / "refs" / "v2"));
  LSE_EXPECT(!fs::exists(repo / "blobs" / hub::git_blob_sha1("# changed\n")));
  // The weights both revisions share stay, and the remaining one still loads.
  LSE_EXPECT(fs::exists(repo / "blobs" / sha256(target_files().at("model.safetensors"))));
  LSE_EXPECT(hub::classify(a.string()).ok());
  auto whole = hub::plan_removal({"org/TinyQwen-4bit", ""});
  LSE_EXPECT(whole.ok() && hub::remove(*whole).ok());
  LSE_EXPECT(!fs::exists(repo));
}

// --- companions ---------------------------------------------------------------

LSE_TEST(companion_state_knows_present_missing_and_unknown) {
  FakeCache cache("companions");
  const Files d = dflash2_files(), m = mtp_files();
  const auto rules = tiny_rules(d, m);
  auto target = hub::classify(cache.add("mlx-community/TinyQwen-4bit", commit_id('a'), target_files()).string());
  LSE_EXPECT_OK(target.status());
  if (!target.ok()) return;
  LSE_EXPECT(hub::match_rule(*target, rules) != nullptr);
  auto st = hub::companion_state(*target, hub::Kind::kDFlash2, rules);
  LSE_EXPECT(st.present.empty() && st.recommended != nullptr);
  // Cached at another revision with the pinned weights: present.
  cache.add("test/TinyQwen-DFlash2", commit_id('e'), d);
  st = hub::companion_state(*target, hub::Kind::kDFlash2, rules);
  LSE_EXPECT(st.present == "test/TinyQwen-DFlash2");
  // Weights that are not the pinned ones do not count.
  Files other = m;
  other["model.safetensors"] += std::string(8, '\0');
  cache.add("test/TinyQwen-MTP-8bit", commit_id('8'), other);
  LSE_EXPECT(hub::companion_state(*target, hub::Kind::kMtp, rules).present.empty());
  // Neither name nor widths match: no companion is known.
  auto unknown = hub::classify(cache.add("mlx-community/Other-4bit", commit_id('b'), target_files()).string());
  LSE_EXPECT(unknown.ok() && hub::match_rule(*unknown, rules) == nullptr);
  LSE_EXPECT(hub::companion_state(*unknown, hub::Kind::kDFlash2, rules).recommended == nullptr);
}

LSE_TEST(the_builtin_registry_pairs_qwen38_27b_with_its_validated_draft) {
  hub::Checkpoint c;
  c.kind = hub::Kind::kModel;
  c.repo_id = "mlx-community/Qwen3.8-27B-4bit";
  c.architecture = "qwen3.5";
  c.hidden_size = 5120;
  c.layers = 64;
  c.vocab_size = 248320;
  const hub::CompanionRule* r = hub::match_rule(c);
  LSE_EXPECT(r != nullptr);
  if (r == nullptr) return;
  LSE_EXPECT(r->dflash2 && r->dflash2->repo_id == "incoai/Qwen3.8-27B-DFlash2");
  LSE_EXPECT(r->dflash2->weights_sha256.starts_with("67fc76d68dc5a941"));
  LSE_EXPECT(!r->mtp.empty() && r->mtp.front().repo_id == "mlx-community/Qwen3.8-27B-MTP-8bit");
  // A local directory named for the model matches too; another size does not.
  c.repo_id.clear();
  c.path = "/models/qwen38-27b-q4-mlx";
  LSE_EXPECT(hub::match_rule(c) != nullptr);
  c.hidden_size = 4096;
  LSE_EXPECT(hub::match_rule(c) == nullptr);
}

// --- downloads against the mock hub --------------------------------------------

LSE_TEST(pull_writes_the_huggingface_hub_layout_and_reuses_it) {
  FakeCache cache("pull");
  MockHub mock;
  const Files files = target_files();
  mock.add("mlx-community/TinyQwen-4bit", commit_id('a'), files);
  bool checked = false;
  hub::PullOptions o;
  o.on_checked = [&](const hub::Checkpoint& c) { checked = c.kind == hub::Kind::kModel; };
  auto got = hub::pull({"mlx-community/TinyQwen-4bit", ""}, o);
  LSE_EXPECT_OK(got.status());
  if (!got.ok()) return;
  LSE_EXPECT(checked);
  LSE_EXPECT_EQ(got->files, files.size());
  LSE_EXPECT_EQ(got->reused, 0u);
  const fs::path repo = cache.hub() / "models--mlx-community--TinyQwen-4bit";
  LSE_EXPECT(read(repo / "refs" / "main") == commit_id('a'));
  for (const auto& [path, content] : files) {
    const fs::path pointer = repo / "snapshots" / commit_id('a') / path;
    LSE_EXPECT(fs::is_symlink(pointer));
    LSE_EXPECT(fs::read_symlink(pointer).is_relative());
    LSE_EXPECT(fs::read_symlink(pointer).filename() == etag_of(path, content));
    LSE_EXPECT(read(pointer) == content);
  }
  LSE_EXPECT(fs::exists(cache.hub() / ".locks" / "models--mlx-community--TinyQwen-4bit"));
  // The headers were read before the weights, by Range.
  bool ranged = false;
  for (const auto& s : mock.seen()) ranged |= s.path.starts_with("/cdn/") && s.range == "bytes=0-" + std::to_string(std::min<std::size_t>(files.at("model.safetensors").size(), 1u << 20) - 1);
  LSE_EXPECT(ranged);
  LSE_EXPECT(model::resolve_model("mlx-community/TinyQwen-4bit").ok());
  auto again = hub::pull({"mlx-community/TinyQwen-4bit", ""}, {});
  LSE_EXPECT(again.ok() && again->reused == files.size() && again->downloaded == 0);
}

LSE_TEST(pull_resumes_an_incomplete_blob_with_a_range_request) {
  FakeCache cache("resume");
  MockHub mock;
  const Files files = target_files();
  mock.add("org/TinyQwen-4bit", commit_id('a'), files);
  const std::string weights = files.at("model.safetensors");
  const fs::path blobs = cache.hub() / "models--org--TinyQwen-4bit" / "blobs";
  fs::create_directories(blobs);
  const std::size_t half = weights.size() / 2;
  std::ofstream(blobs / (sha256(weights) + ".incomplete"), std::ios::binary) << weights.substr(0, half);
  auto got = hub::pull({"org/TinyQwen-4bit", ""}, {});
  LSE_EXPECT_OK(got.status());
  bool resumed = false;
  for (const auto& s : mock.seen()) resumed |= s.range == "bytes=" + std::to_string(half) + "-";
  LSE_EXPECT(resumed);
  LSE_EXPECT(read(blobs / sha256(weights)) == weights);
  LSE_EXPECT(!fs::exists(blobs / (sha256(weights) + ".incomplete")));
  // Only what was missing crossed the wire.
  const std::size_t rest = (weights.size() - half) + files.at("config.json").size() +
                           files.at("README.md").size() + files.at("assets/note.txt").size();
  LSE_EXPECT(got.ok() && got->downloaded == rest);
}

LSE_TEST(a_dropped_connection_keeps_the_partial_blob_for_the_next_pull) {
  FakeCache cache("drop");
  MockHub mock;
  const Files files = target_files();
  mock.add("org/TinyQwen-4bit", commit_id('a'), files);
  const std::string weights = files.at("model.safetensors");
  mock.truncate_after = weights.size() / 3;
  auto first = hub::pull({"org/TinyQwen-4bit", ""}, {});
  LSE_EXPECT(!first.ok());
  LSE_EXPECT_HAS(first.status().message(), "run the pull again to resume");
  const fs::path part = cache.hub() / "models--org--TinyQwen-4bit" / "blobs" / (sha256(weights) + ".incomplete");
  LSE_EXPECT(fs::exists(part) && fs::file_size(part) > 0);
  mock.truncate_after = 0;
  auto second = hub::pull({"org/TinyQwen-4bit", ""}, {});
  LSE_EXPECT_OK(second.status());
  LSE_EXPECT(model::resolve_model("org/TinyQwen-4bit").ok());
}

LSE_TEST(an_etag_mismatch_or_a_corrupt_body_fails_loudly) {
  FakeCache cache("verify");
  MockHub mock;
  mock.add("org/TinyQwen-4bit", commit_id('a'), target_files());
  mock.wrong_etag_for = "README.md";
  auto etag = hub::pull({"org/TinyQwen-4bit", ""}, {});
  LSE_EXPECT(!etag.ok());
  LSE_EXPECT_HAS(etag.status().message(), "etag mismatch for README.md");
  mock.wrong_etag_for.clear();
  mock.corrupt = "model.safetensors";
  auto corrupt = hub::pull({"org/TinyQwen-4bit", ""}, {});
  LSE_EXPECT(!corrupt.ok());
  LSE_EXPECT_HAS(corrupt.status().message(), "checksum mismatch for model.safetensors");
  const std::string weights = target_files().at("model.safetensors");
  const fs::path blobs = cache.hub() / "models--org--TinyQwen-4bit" / "blobs";
  LSE_EXPECT(!fs::exists(blobs / sha256(weights)));
  LSE_EXPECT(!fs::exists(blobs / (sha256(weights) + ".incomplete")));
  // Nothing claims the download finished.
  LSE_EXPECT(!fs::exists(cache.hub() / "models--org--TinyQwen-4bit" / "refs" / "main"));
}

LSE_TEST(gated_unknown_and_unsupported_repos_are_refused_precisely) {
  FakeCache cache("refuse");
  MockHub mock;
  mock.add("meta/Gated-4bit", commit_id('a'), target_files(), "manual", "hf_good");
  auto no_token = hub::pull({"meta/Gated-4bit", ""}, {});
  LSE_EXPECT(!no_token.ok());
  LSE_EXPECT_HAS(no_token.status().message(), "is gated: accept its terms at " + mock.url() + "/meta/Gated-4bit");
  {
    ScopedEnv t("HF_TOKEN", "hf_wrong");
    auto wrong = hub::pull({"meta/Gated-4bit", ""}, {});
    LSE_EXPECT(!wrong.ok());
    LSE_EXPECT_HAS(wrong.status().message(), "the token from HF_TOKEN has no access");
  }
  {
    // The token file huggingface-cli login writes, sent as a bearer token.
    std::ofstream(cache.root() / "token") << "hf_good\n";
    auto ok = hub::pull({"meta/Gated-4bit", ""}, {});
    LSE_EXPECT_OK(ok.status());
    bool sent = false;
    for (const auto& s : mock.seen()) sent |= s.auth == "Bearer hf_good";
    LSE_EXPECT(sent);
    fs::remove(cache.root() / "token");
  }
  auto unknown = hub::pull({"nobody/Nothing", ""}, {});
  LSE_EXPECT(!unknown.ok());
  LSE_EXPECT_HAS(unknown.status().message(), "no repository nobody/Nothing on " + mock.url());
  auto revision = hub::pull({"meta/Gated-4bit", "nope"}, {});
  LSE_EXPECT(!revision.ok());

  const std::size_t before = mock.weight_bodies();
  mock.add("Qwen/TinyQwen", commit_id('b'), target_files(false));
  auto not_mlx = hub::pull({"Qwen/TinyQwen", ""}, {});
  LSE_EXPECT(!not_mlx.ok());
  LSE_EXPECT_HAS(not_mlx.status().message(), "not an MLX checkpoint");
  mock.add("someone/Tiny-GGUF", commit_id('c'), {{"tiny-Q4_K_M.gguf", std::string(4096, 'g')}});
  auto gguf = hub::pull({"someone/Tiny-GGUF", ""}, {});
  LSE_EXPECT(!gguf.ok());
  LSE_EXPECT_HAS(gguf.status().message(), "GGUF repository");
  // Refused from headers alone: no weights were fetched, nothing was cached.
  LSE_EXPECT_EQ(mock.weight_bodies(), before);
  LSE_EXPECT(!fs::exists(cache.hub() / "models--Qwen--TinyQwen"));
  // A companion where a target was asked for.
  mock.add("test/TinyQwen-MTP-8bit", commit_id('8'), mtp_files());
  hub::PullOptions only_models;
  only_models.accept = {hub::Kind::kModel};
  auto wrong_kind = hub::pull({"test/TinyQwen-MTP-8bit", ""}, only_models);
  LSE_EXPECT(!wrong_kind.ok());
  {
    ScopedEnv off("HF_HUB_OFFLINE", "1");
    auto offline = hub::pull({"meta/Gated-4bit", ""}, {});
    LSE_EXPECT(!offline.ok());
    LSE_EXPECT_HAS(offline.status().message(), "HF_HUB_OFFLINE");
  }
}

LSE_TEST(include_and_exclude_select_files) {
  FakeCache cache("filters");
  MockHub mock;
  mock.add("org/TinyQwen-4bit", commit_id('a'), target_files());
  hub::PullOptions o;
  o.exclude = {"assets/", "*.md"};
  auto got = hub::pull({"org/TinyQwen-4bit", ""}, o);
  LSE_EXPECT_OK(got.status());
  const fs::path snap = cache.hub() / "models--org--TinyQwen-4bit" / "snapshots" / commit_id('a');
  LSE_EXPECT(fs::exists(snap / "model.safetensors") && fs::exists(snap / "config.json"));
  LSE_EXPECT(!fs::exists(snap / "README.md") && !fs::exists(snap / "assets" / "note.txt"));
  hub::PullOptions none;
  none.include = {"*.bin"};
  LSE_EXPECT(!hub::pull({"org/TinyQwen-4bit", ""}, none).ok());
}

LSE_TEST(ensure_companion_uses_the_cache_downloads_when_allowed_and_names_the_pull_offline) {
  FakeCache cache("ensure");
  MockHub mock;
  const Files d = dflash2_files(), m = mtp_files();
  const auto rules = tiny_rules(d, m);
  mock.add("test/TinyQwen-DFlash2", commit_id('d'), d);
  mock.add("test/TinyQwen-MTP-8bit", commit_id('8'), m);
  cache.add("mlx-community/TinyQwen-4bit", commit_id('a'), target_files());
  hub::EnsureOptions e;
  e.rules = rules;

  {
    // Offline: the exact command, and no request leaves the process.
    ScopedEnv off("HF_HUB_OFFLINE", "1");
    auto r = hub::ensure_companion("mlx-community/TinyQwen-4bit", hub::Kind::kDFlash2, e);
    LSE_EXPECT(!r.ok());
    LSE_EXPECT_HAS(r.status().message(), "lse-server pull mlx-community/TinyQwen-4bit --with-dflash2");
    LSE_EXPECT_HAS(r.status().message(), "lse-server pull test/TinyQwen-DFlash2@" + commit_id('d'));
    auto mtp = hub::ensure_companion("mlx-community/TinyQwen-4bit", hub::Kind::kMtp, e);
    LSE_EXPECT(!mtp.ok());
    LSE_EXPECT_HAS(mtp.status().message(), "--no-mtp");
    LSE_EXPECT(mock.seen().empty());
  }
  // Allowed: downloaded at the pinned commit, then named for the loader.
  auto got = hub::ensure_companion("mlx-community/TinyQwen-4bit", hub::Kind::kDFlash2, e);
  LSE_EXPECT_OK(got.status());
  if (got.ok()) {
    LSE_EXPECT(*got == "test/TinyQwen-DFlash2");
    auto paths = model::resolve_model(*got);
    LSE_EXPECT(paths.ok() && read(paths->weights) == d.at("model.safetensors"));
  }
  auto mtp = hub::ensure_companion("mlx-community/TinyQwen-4bit", hub::Kind::kMtp, e);
  LSE_EXPECT(mtp.ok() && *mtp == "test/TinyQwen-MTP-8bit");
  // Cached now: answered without the network.
  const std::size_t requests = mock.seen().size();
  {
    ScopedEnv off("HF_HUB_OFFLINE", "1");
    auto cached = hub::ensure_companion("mlx-community/TinyQwen-4bit", hub::Kind::kDFlash2, e);
    LSE_EXPECT(cached.ok() && *cached == "test/TinyQwen-DFlash2");
  }
  LSE_EXPECT_EQ(mock.seen().size(), requests);
  // A target with no known companion: MTP is simply absent, DFlash2 an error.
  cache.add("mlx-community/Other-4bit", commit_id('b'), target_files());
  auto none = hub::ensure_companion("mlx-community/Other-4bit", hub::Kind::kMtp, e);
  LSE_EXPECT(none.ok() && none->empty());
  auto no_draft = hub::ensure_companion("mlx-community/Other-4bit", hub::Kind::kDFlash2, e);
  LSE_EXPECT(!no_draft.ok());
  LSE_EXPECT_HAS(no_draft.status().message(), "no DFlash2 draft is known");
}

LSE_TEST(a_companion_whose_upstream_weights_changed_is_refused) {
  FakeCache cache("pin");
  MockHub mock;
  const Files d = dflash2_files(), m = mtp_files();
  auto rules = tiny_rules(d, m);
  Files changed = d;
  changed["model.safetensors"] = safetensors(dflash2_tensors(), true);  // other bytes, same shapes
  mock.add("test/TinyQwen-DFlash2", commit_id('d'), changed);
  cache.add("mlx-community/TinyQwen-4bit", commit_id('a'), target_files());
  hub::EnsureOptions e;
  e.rules = rules;
  auto r = hub::ensure_companion("mlx-community/TinyQwen-4bit", hub::Kind::kDFlash2, e);
  LSE_EXPECT(!r.ok());
  LSE_EXPECT_HAS(r.status().message(), "not the " + sha256(d.at("model.safetensors")));
}

LSE_TEST(pull_if_missing_leaves_paths_and_cached_repos_alone) {
  FakeCache cache("pull-if-missing");
  MockHub mock;
  mock.add("org/TinyQwen-4bit", commit_id('a'), target_files());
  const hub::Kind accept[] = {hub::Kind::kModel};
  LSE_EXPECT_OK(hub::pull_if_missing(cache.root().string(), accept, {}));
  LSE_EXPECT(mock.seen().empty());
  LSE_EXPECT_OK(hub::pull_if_missing("org/TinyQwen-4bit", accept, {}));
  LSE_EXPECT(model::resolve_model("org/TinyQwen-4bit").ok());
  const std::size_t n = mock.seen().size();
  LSE_EXPECT_OK(hub::pull_if_missing("org/TinyQwen-4bit", accept, {}));
  LSE_EXPECT_EQ(mock.seen().size(), n);
}

LSE_TEST_MAIN()
