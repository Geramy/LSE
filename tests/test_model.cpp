// Loads the real lemonseed b1.5 checkpoint when it is present; the pure-config
// cases run everywhere.
#include "lse/model/config.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "harness.hpp"
#include "lse/graph/interpreter.hpp"
#include "lse/model/hybrid_lm.hpp"
#include "lse/model/layer.hpp"
#include "lse/model/registry.hpp"
#include "lse/model/weights.hpp"
#include "lse/ops/rope.hpp"
#include "lse/quant/group_affine.hpp"

using namespace lse;
using namespace lse::model;

namespace {

// Override with LSE_TEST_MODEL to point at another checkpoint.
std::string model_dir() {
  if (const char* p = std::getenv("LSE_TEST_MODEL")) return p;
  const char* home = std::getenv("HOME");
  return home ? std::string(home) + "/Documents/Dev/LDE/model" : "";
}

bool have_model() {
  std::error_code ec;
  return !model_dir().empty() && std::filesystem::is_directory(model_dir(), ec);
}

}  // namespace

LSE_TEST(b1_5_preset_matches_the_shipped_config) {
  const Config c = preset_b1_5();
  LSE_EXPECT_EQ(c.vocab_size, 248320);
  LSE_EXPECT_EQ(c.hidden_size, 1024);
  LSE_EXPECT_EQ(c.num_layers, 20);
  LSE_EXPECT_EQ(c.attn_q_heads, 16);
  LSE_EXPECT_EQ(c.attn_kv_heads, 2);
  LSE_EXPECT_EQ(c.gdn_chunk_size, 32);
  LSE_EXPECT_EQ(c.expert_intermediate, 2176);
}

LSE_TEST(layer_layout_follows_the_interval_rule) {
  const Config c = preset_b1_5();
  // interval 4 -> layers 3, 7, 11, 15, 19 are attention, the rest GDN.
  const int attn[] = {3, 7, 11, 15, 19};
  for (int i = 0; i < c.num_layers; ++i) {
    bool want = false;
    for (int a : attn) want = want || (a == i);
    LSE_EXPECT_EQ(c.is_attention_layer(i), want);
  }
  LSE_EXPECT(c.is_global_attention(19));
  LSE_EXPECT(!c.is_global_attention(2));
}

LSE_TEST(config_round_trips_through_json) {
  const Config a = preset_b1_5();
  auto b = Config::from_json_string(a.to_json());
  LSE_EXPECT(b.ok());
  LSE_EXPECT_EQ(b->vocab_size, a.vocab_size);
  LSE_EXPECT_EQ(b->num_layers, a.num_layers);
  LSE_EXPECT_EQ(b->global_attention_layers.size(), a.global_attention_layers.size());
  LSE_EXPECT_NEAR(b->rope_theta, a.rope_theta, 1e-3);
  LSE_EXPECT(b->dtype == a.dtype);
}

LSE_TEST(validate_rejects_incoherent_gqa) {
  Config c = preset_b1_5();
  c.attn_kv_heads = 5;  // 16 is not a multiple of 5
  LSE_EXPECT(!c.validate().ok());
}

LSE_TEST(validate_rejects_out_of_range_global_layers) {
  Config c = preset_b1_5();
  c.global_attention_layers = {99};
  auto s = c.validate();
  LSE_EXPECT(!s.ok());
  LSE_EXPECT(s.code() == StatusCode::kOutOfRange);
}

LSE_TEST(legacy_expert_names_migrate_to_stacked) {
  auto g = migrate_legacy_expert_name("blocks.3.moe.experts.5.w1.weight");
  LSE_EXPECT(g.ok());
  LSE_EXPECT(g->stacked_name == "blocks.3.moe.w_gate");
  LSE_EXPECT_EQ(g->expert_index, 5);

  auto u = migrate_legacy_expert_name("blocks.0.moe.experts.0.w3.weight");
  LSE_EXPECT(u.ok());
  LSE_EXPECT(u->stacked_name == "blocks.0.moe.w_up");

  auto d = migrate_legacy_expert_name("blocks.0.moe.experts.7.w2.weight");
  LSE_EXPECT(d.ok());
  LSE_EXPECT(d->stacked_name == "blocks.0.moe.w_down");

  // Already-current names are not legacy.
  LSE_EXPECT(!migrate_legacy_expert_name("blocks.0.moe.w_gate").ok());
}

LSE_TEST(hf_cache_root_honours_env) {
  LSE_EXPECT(hf_cache_root().find("huggingface") != std::string::npos ||
             hf_cache_root().find("hub") != std::string::npos);
}

LSE_TEST(resolve_reports_missing_models_clearly) {
  auto r = resolve_model("definitely/not-a-real-model");
  LSE_EXPECT(!r.ok());
  LSE_EXPECT(r.status().code() == StatusCode::kNotFound);
}

namespace {

// Every cache test rewrites the variables that decide where the cache is, so
// each one is restored before the next test reads them.
class ScopedEnv {
 public:
  ScopedEnv(const char* name, const char* value) : name_(name) {
    if (const char* old = std::getenv(name)) {
      had_ = true;
      old_ = old;
    }
    if (value == nullptr) {
      ::unsetenv(name);
    } else {
      ::setenv(name, value, 1);
    }
  }
  ~ScopedEnv() {
    if (had_) {
      ::setenv(name_.c_str(), old_.c_str(), 1);
    } else {
      ::unsetenv(name_.c_str());
    }
  }
  ScopedEnv(const ScopedEnv&) = delete;
  ScopedEnv& operator=(const ScopedEnv&) = delete;

 private:
  std::string name_;
  std::string old_;
  bool had_ = false;
};

// A hub cache tree under a temp directory, with HF_HUB_CACHE pointed at it so
// resolve_model and list_cached_models look here and not at the real cache.
class FakeHub {
 public:
  explicit FakeHub(const std::string& tag)
      : root_(std::filesystem::temp_directory_path() / ("lse_hub_" + tag)) {
    std::error_code ec;
    std::filesystem::remove_all(root_, ec);
    std::filesystem::create_directories(root_, ec);
    pin_ = std::make_unique<ScopedEnv>("HF_HUB_CACHE", root_.string().c_str());
    legacy_ = std::make_unique<ScopedEnv>("HUGGINGFACE_HUB_CACHE", nullptr);
  }
  ~FakeHub() {
    std::error_code ec;
    std::filesystem::remove_all(root_, ec);
  }
  FakeHub(const FakeHub&) = delete;
  FakeHub& operator=(const FakeHub&) = delete;

  [[nodiscard]] const std::filesystem::path& root() const { return root_; }

  // The snapshot directory for a repo id, created. Doubles the separator the
  // way the hub does.
  std::filesystem::path snapshot(const std::string& repo_id,
                                 const std::string& sha = "cafe0001") {
    std::string dir = "models--";
    for (char c : repo_id) {
      if (c == '/') {
        dir += "--";
      } else {
        dir += c;
      }
    }
    const std::filesystem::path p = root_ / dir / "snapshots" / sha;
    std::error_code ec;
    std::filesystem::create_directories(p, ec);
    return p;
  }

 private:
  std::filesystem::path root_;
  std::unique_ptr<ScopedEnv> pin_;
  std::unique_ptr<ScopedEnv> legacy_;
};

void write_text(const std::filesystem::path& p, const std::string& text) {
  std::ofstream out(p, std::ios::binary);
  out << text;
}

// A config the native (lemonseed) path accepts, so a cache test exercises
// discovery rather than config validation.
std::string minimal_config() {
  return Config{}.to_json();
}

// A valid one-tensor safetensors file at an arbitrary path.
void write_safetensors_at(const std::filesystem::path& p,
                          const std::vector<std::string>& names) {
  std::string header = "{";
  std::size_t offset = 0;
  for (const std::string& name : names) {
    if (header.size() > 1) header += ",";
    header += "\"" + name + "\":{\"dtype\":\"F32\",\"shape\":[2,2]," +
              "\"data_offsets\":[" + std::to_string(offset) + "," +
              std::to_string(offset + 16) + "]}";
    offset += 16;
  }
  header += "}";
  while (header.size() % 8 != 0) header += " ";

  std::ofstream out(p, std::ios::binary);
  const std::uint64_t n = header.size();
  out.write(reinterpret_cast<const char*>(&n), sizeof(n));
  out.write(header.data(), static_cast<std::streamsize>(header.size()));
  const std::vector<float> data(offset / 4, 0.5f);
  out.write(reinterpret_cast<const char*>(data.data()),
            static_cast<std::streamsize>(offset));
}

const CacheModel* find_repo(const std::vector<CacheModel>& models,
                            std::string_view repo_id) {
  for (const CacheModel& m : models) {
    if (m.repo_id == repo_id) return &m;
  }
  return nullptr;
}

}  // namespace

// The order huggingface_hub uses. Getting this wrong sends the engine looking
// somewhere the Python library never would, so each rung is pinned.
LSE_TEST(the_cache_root_follows_the_huggingface_precedence) {
  const std::string base =
      (std::filesystem::temp_directory_path() / "lse_hf_env").string();

  {  // HF_HUB_CACHE outranks every other variable.
    const ScopedEnv a("HF_HUB_CACHE", (base + "/direct").c_str());
    const ScopedEnv b("HUGGINGFACE_HUB_CACHE", (base + "/legacy").c_str());
    const ScopedEnv c("HF_HOME", (base + "/home").c_str());
    const ScopedEnv d("XDG_CACHE_HOME", (base + "/xdg").c_str());
    LSE_EXPECT(hf_cache_root() == base + "/direct");
  }
  {  // The legacy alias is still honoured, one rung down.
    const ScopedEnv a("HF_HUB_CACHE", nullptr);
    const ScopedEnv b("HUGGINGFACE_HUB_CACHE", (base + "/legacy").c_str());
    const ScopedEnv c("HF_HOME", (base + "/home").c_str());
    LSE_EXPECT(hf_cache_root() == base + "/legacy");
  }
  {  // Then $HF_HOME/hub.
    const ScopedEnv a("HF_HUB_CACHE", nullptr);
    const ScopedEnv b("HUGGINGFACE_HUB_CACHE", nullptr);
    const ScopedEnv c("HF_HOME", (base + "/home").c_str());
    const ScopedEnv d("XDG_CACHE_HOME", (base + "/xdg").c_str());
    LSE_EXPECT(hf_cache_root() == base + "/home/hub");
  }
  {  // XDG_CACHE_HOME feeds HF_HOME's default only, and is not a fallback for
     // either cache variable.
    const ScopedEnv a("HF_HUB_CACHE", nullptr);
    const ScopedEnv b("HUGGINGFACE_HUB_CACHE", nullptr);
    const ScopedEnv c("HF_HOME", nullptr);
    const ScopedEnv d("XDG_CACHE_HOME", (base + "/xdg").c_str());
    LSE_EXPECT(hf_cache_root() == base + "/xdg/huggingface/hub");
  }
  {  // The documented default.
    const ScopedEnv a("HF_HUB_CACHE", nullptr);
    const ScopedEnv b("HUGGINGFACE_HUB_CACHE", nullptr);
    const ScopedEnv c("HF_HOME", nullptr);
    const ScopedEnv d("XDG_CACHE_HOME", nullptr);
    const ScopedEnv e("HOME", base.c_str());
    LSE_EXPECT(hf_cache_root() == base + "/.cache/huggingface/hub");
  }
  {  // TRANSFORMERS_CACHE is absent from huggingface_hub and must not be read.
    const ScopedEnv a("HF_HUB_CACHE", nullptr);
    const ScopedEnv b("HUGGINGFACE_HUB_CACHE", nullptr);
    const ScopedEnv c("HF_HOME", nullptr);
    const ScopedEnv d("XDG_CACHE_HOME", nullptr);
    const ScopedEnv e("HOME", base.c_str());
    const ScopedEnv f("TRANSFORMERS_CACHE", (base + "/transformers").c_str());
    LSE_EXPECT(hf_cache_root().find("transformers") == std::string::npos);
  }
  {  // An empty value cannot name a directory, so it reads as unset.
    const ScopedEnv a("HF_HUB_CACHE", "");
    const ScopedEnv b("HUGGINGFACE_HUB_CACHE", nullptr);
    const ScopedEnv c("HF_HOME", (base + "/home").c_str());
    LSE_EXPECT(hf_cache_root() == base + "/home/hub");
  }
}

// The Python library expanduser+expandvars the value it reads; without this a
// '~' becomes a literal directory name.
LSE_TEST(a_cache_path_expands_a_tilde_and_a_variable) {
  const ScopedEnv home("HOME", "/tmp/lse-fake-home");
  const ScopedEnv legacy("HUGGINGFACE_HUB_CACHE", nullptr);
  {
    const ScopedEnv c("HF_HUB_CACHE", "~/hub");
    LSE_EXPECT(hf_cache_root() == "/tmp/lse-fake-home/hub");
  }
  {
    const ScopedEnv c("HF_HUB_CACHE", "$HOME/hub");
    LSE_EXPECT(hf_cache_root() == "/tmp/lse-fake-home/hub");
  }
  {
    const ScopedEnv c("HF_HUB_CACHE", "${HOME}/hub");
    LSE_EXPECT(hf_cache_root() == "/tmp/lse-fake-home/hub");
  }
  {  // An undefined variable is left as written, as expandvars leaves it.
    const ScopedEnv c("HF_HUB_CACHE", "/tmp/$LSE_NO_SUCH_VAR_HERE/hub");
    LSE_EXPECT(hf_cache_root() == "/tmp/$LSE_NO_SUCH_VAR_HERE/hub");
  }
}

// The cache doubles the separator: models--<org>--<name>. A single dash meant
// naming a model by its repo id never resolved.
LSE_TEST(a_repo_id_resolves_to_its_snapshot) {
  FakeHub hub("resolve");
  const std::filesystem::path snap = hub.snapshot("mlx-community/Tiny-4bit");
  write_text(snap / "config.json", minimal_config());
  write_safetensors_at(snap / "model.safetensors", {"w"});

  auto r = resolve_model("mlx-community/Tiny-4bit");
  LSE_EXPECT_OK(r.status());
  LSE_EXPECT(r->weights == (snap / "model.safetensors").string());
  LSE_EXPECT(r->config == (snap / "config.json").string());
}

LSE_TEST(a_bare_model_name_resolves_only_when_it_is_unique) {
  FakeHub hub("bare");
  const std::filesystem::path a = hub.snapshot("mlx-community/Solo-4bit");
  write_text(a / "config.json", minimal_config());
  write_safetensors_at(a / "model.safetensors", {"w"});

  auto solo = resolve_model("Solo-4bit");
  LSE_EXPECT_OK(solo.status());
  LSE_EXPECT(solo->weights == (a / "model.safetensors").string());

  // A second organization's build of the same name makes the bare name
  // ambiguous, and picking one silently would load the wrong checkpoint.
  const std::filesystem::path b = hub.snapshot("other-org/Solo-4bit");
  write_text(b / "config.json", minimal_config());
  write_safetensors_at(b / "model.safetensors", {"w"});

  auto both = resolve_model("Solo-4bit");
  LSE_EXPECT(!both.ok());
  LSE_EXPECT(both.status().code() == StatusCode::kInvalidArgument);
  const std::string msg = both.status().to_string();
  LSE_EXPECT(msg.find("mlx-community/Solo-4bit") != std::string::npos);
  LSE_EXPECT(msg.find("other-org/Solo-4bit") != std::string::npos);

  // The full repo id still resolves either of them.
  auto exact = resolve_model("other-org/Solo-4bit");
  LSE_EXPECT_OK(exact.status());
  LSE_EXPECT(exact->weights == (b / "model.safetensors").string());
}

LSE_TEST(an_absent_name_names_what_was_tried) {
  FakeHub hub("absent");
  const std::filesystem::path a = hub.snapshot("mlx-community/Present-4bit");
  write_text(a / "config.json", minimal_config());
  write_safetensors_at(a / "model.safetensors", {"w"});

  auto r = resolve_model("nobody/Missing-8bit");
  LSE_EXPECT(!r.ok());
  LSE_EXPECT(r.status().code() == StatusCode::kNotFound);
  const std::string msg = r.status().to_string();
  // What was looked for, where, and what was there instead.
  LSE_EXPECT(msg.find("models--nobody--Missing-8bit") != std::string::npos);
  LSE_EXPECT(msg.find(hub.root().string()) != std::string::npos);
  LSE_EXPECT(msg.find("tried: mlx-community/Present-4bit") != std::string::npos);
}

// The two stubs in the real cache would otherwise look like models: a repo
// directory exists whether or not the download finished.
LSE_TEST(an_incomplete_repo_is_never_offered_as_loadable) {
  FakeHub hub("incomplete");

  // Nothing but a tokenizer, which is what an interrupted download leaves.
  const std::filesystem::path bare = hub.snapshot("org/Interrupted");
  write_text(bare / "tokenizer.json", "{}");

  // A repo with no snapshots directory at all.
  std::error_code ec;
  std::filesystem::create_directories(
      hub.root() / "models--org--NoSnapshot" / "refs", ec);

  // A valid index naming two shards, one of which never arrived. This is the
  // case a config.json check alone would pass.
  const std::filesystem::path partial = hub.snapshot("org/HalfSharded");
  write_text(partial / "config.json", minimal_config());
  write_text(partial / "model.safetensors.index.json",
             R"({"weight_map":{"a":"model-00001-of-00002.safetensors",)"
             R"("b":"model-00002-of-00002.safetensors"}})");
  write_safetensors_at(partial / "model-00001-of-00002.safetensors", {"a"});

  auto models = list_cached_models();
  LSE_EXPECT_OK(models.status());

  for (const char* repo : {"org/Interrupted", "org/NoSnapshot",
                           "org/HalfSharded"}) {
    const CacheModel* m = find_repo(*models, repo);
    LSE_EXPECT(m != nullptr);
    if (m == nullptr) continue;
    LSE_EXPECT(m->loadable == Loadable::kIncomplete);
    LSE_EXPECT(!m->reason.empty());
  }

  // The missing shard has to be named, not merely counted.
  const CacheModel* half = find_repo(*models, "org/HalfSharded");
  LSE_EXPECT(half != nullptr);
  if (half != nullptr) {
    LSE_EXPECT(half->reason.find("model-00002-of-00002.safetensors") !=
               std::string::npos);
  }

  // And opening one still fails rather than half-loading.
  LSE_EXPECT(!SafeTensors::open_sharded(
                  (partial / "model.safetensors.index.json").string())
                  .ok());
}

// Two downloaded revisions and no refs/main: the name does not identify one, so
// neither resolution nor the listing may pick for the user.
LSE_TEST(several_revisions_with_no_ref_are_ambiguous_not_incomplete) {
  FakeHub hub("revisions");
  for (const char* sha : {"aaaa1111", "bbbb2222"}) {
    const std::filesystem::path snap = hub.snapshot("org/TwoRevs", sha);
    write_text(snap / "config.json", minimal_config());
    write_safetensors_at(snap / "model.safetensors", {"w"});
  }

  auto r = resolve_model("org/TwoRevs");
  LSE_EXPECT(!r.ok());
  LSE_EXPECT(r.status().code() == StatusCode::kInvalidArgument);
  LSE_EXPECT(r.status().to_string().find("aaaa1111") != std::string::npos);
  LSE_EXPECT(r.status().to_string().find("bbbb2222") != std::string::npos);

  auto models = list_cached_models();
  LSE_EXPECT_OK(models.status());
  const CacheModel* m = find_repo(*models, "org/TwoRevs");
  LSE_EXPECT(m != nullptr);
  if (m == nullptr) return;
  // Unknown, not incomplete: both downloads finished.
  LSE_EXPECT(m->loadable == Loadable::kUnknown);

  // refs/main settles it, and then the repo resolves.
  std::error_code ec;
  std::filesystem::create_directories(
      hub.root() / "models--org--TwoRevs" / "refs", ec);
  write_text(hub.root() / "models--org--TwoRevs" / "refs" / "main", "bbbb2222");
  auto pinned = resolve_model("org/TwoRevs");
  LSE_EXPECT_OK(pinned.status());
  LSE_EXPECT(pinned->weights.find("bbbb2222") != std::string::npos);
}

LSE_TEST(a_gguf_repo_is_reported_unloadable_not_incomplete) {
  FakeHub hub("gguf");
  const std::filesystem::path snap = hub.snapshot("org/Gguf-Only");
  write_text(snap / "config.json", minimal_config());
  write_text(snap / "model.gguf", "GGUF not really");

  auto models = list_cached_models();
  LSE_EXPECT_OK(models.status());
  const CacheModel* m = find_repo(*models, "org/Gguf-Only");
  LSE_EXPECT(m != nullptr);
  if (m == nullptr) return;
  LSE_EXPECT(m->loadable == Loadable::kNo);
  LSE_EXPECT(m->reason.find("GGUF") != std::string::npos);
}

// The verdict comes from the engine's own detection, so a checkpoint no
// architecture claims is reported as unloadable naming what was tried — never
// guessed at from the repo's name.
LSE_TEST(an_unrecognized_checkpoint_lists_as_unloadable_naming_what_was_tried) {
  FakeHub hub("stranger");
  const std::filesystem::path snap = hub.snapshot("org/Stranger");
  write_text(snap / "config.json", minimal_config());
  write_safetensors_at(snap / "model.safetensors", {"encoder.layer.0.weight"});

  auto models = list_cached_models();
  LSE_EXPECT_OK(models.status());
  const CacheModel* m = find_repo(*models, "org/Stranger");
  LSE_EXPECT(m != nullptr);
  if (m == nullptr) return;
  LSE_EXPECT(m->loadable == Loadable::kNo);
  LSE_EXPECT(m->reason.find("tried:") != std::string::npos);
  LSE_EXPECT(m->engine_arch.empty());
}

// Resolution and listing must read the same variables hf_cache_root does, or
// the engine looks in one place and reports on another.
LSE_TEST(discovery_follows_the_cache_variables_end_to_end) {
  const std::filesystem::path base =
      std::filesystem::temp_directory_path() / "lse_hub_endtoend";
  std::error_code ec;
  std::filesystem::remove_all(base, ec);
  const std::filesystem::path snap =
      base / "hub" / "models--org--ViaHfHome" / "snapshots" / "d00d";
  std::filesystem::create_directories(snap, ec);
  write_text(snap / "config.json", minimal_config());
  write_safetensors_at(snap / "model.safetensors", {"w"});

  {  // Reached through $HF_HOME/hub, with no cache variable set.
    const ScopedEnv a("HF_HUB_CACHE", nullptr);
    const ScopedEnv b("HUGGINGFACE_HUB_CACHE", nullptr);
    const ScopedEnv c("HF_HOME", base.string().c_str());
    auto r = resolve_model("org/ViaHfHome");
    LSE_EXPECT_OK(r.status());
    auto models = list_cached_models();
    LSE_EXPECT_OK(models.status());
    LSE_EXPECT(find_repo(*models, "org/ViaHfHome") != nullptr);
  }
  {  // The legacy variable points somewhere else entirely, and wins over
     // HF_HOME.
    const ScopedEnv a("HF_HUB_CACHE", nullptr);
    const ScopedEnv b("HUGGINGFACE_HUB_CACHE",
                      (base / "elsewhere").string().c_str());
    const ScopedEnv c("HF_HOME", base.string().c_str());
    LSE_EXPECT(!resolve_model("org/ViaHfHome").ok());
  }
  std::filesystem::remove_all(base, ec);
}

// A listing is sorted and complete: a repo missing from it cannot be chosen.
LSE_TEST(the_listing_reports_every_repo_in_the_cache) {
  FakeHub hub("listing");
  for (const char* repo : {"z-org/Last", "a-org/First", "m-org/Middle"}) {
    const std::filesystem::path snap = hub.snapshot(repo);
    write_text(snap / "config.json", minimal_config());
    write_safetensors_at(snap / "model.safetensors", {"w"});
  }
  auto models = list_cached_models();
  LSE_EXPECT_OK(models.status());
  LSE_EXPECT_EQ(models->size(), 3u);
  LSE_EXPECT((*models)[0].repo_id == "a-org/First");
  LSE_EXPECT((*models)[1].repo_id == "m-org/Middle");
  LSE_EXPECT((*models)[2].repo_id == "z-org/Last");
  for (const CacheModel& m : *models) LSE_EXPECT(m.bytes > 0);
}


LSE_TEST(loads_the_real_b1_5_checkpoint) {
  if (!have_model()) {
    std::printf("       (skipped: no checkpoint at %s)\n", model_dir().c_str());
    return;
  }

  auto paths = resolve_model(model_dir());
  LSE_EXPECT(paths.ok());
  if (!paths.ok()) return;

  auto cfg = Config::from_json_file(paths->config);
  LSE_EXPECT(cfg.ok());
  if (!cfg.ok()) return;
  LSE_EXPECT_EQ(cfg->vocab_size, 248320);
  LSE_EXPECT_EQ(cfg->num_layers, 20);

  auto st = SafeTensors::open(paths->weights);
  LSE_EXPECT(st.ok());
  if (!st.ok()) return;

  LSE_EXPECT_EQ(st->tensors().size(), 517u);

  const double params = static_cast<double>(st->total_parameters()) / 1e9;
  LSE_EXPECT_NEAR(params, 1.5145, 0.001);

  // The stacked-MoE layout this checkpoint uses.
  const TensorView* w_gate = st->find("blocks.0.moe.w_gate");
  LSE_EXPECT(w_gate != nullptr);
  if (w_gate) {
    LSE_EXPECT_EQ(w_gate->shape.rank(), 3u);
    LSE_EXPECT_EQ(w_gate->shape[0], cfg->num_experts);
    LSE_EXPECT_EQ(w_gate->shape[1], cfg->expert_intermediate);
    LSE_EXPECT_EQ(w_gate->shape[2], cfg->hidden_size);
    LSE_EXPECT(w_gate->dtype == DType::kBF16);
  }

  const TensorView* embed = st->find("embed.weight");
  LSE_EXPECT(embed != nullptr);
  if (embed) {
    LSE_EXPECT_EQ(embed->shape[0], cfg->vocab_size);
    LSE_EXPECT_EQ(embed->shape[1], cfg->hidden_size);
  }

  // Every layer must have the mixer its index implies.
  for (int i = 0; i < cfg->num_layers; ++i) {
    const std::string prefix = "blocks." + std::to_string(i) + ".mixer.";
    const bool is_attn = cfg->is_attention_layer(i);
    const TensorView* gdn_marker = st->find(prefix + "A_log");
    LSE_EXPECT_EQ(gdn_marker == nullptr, is_attn);
  }

  // Widening a real bf16 tensor must produce finite values.
  const TensorView* norm = st->find("blocks.0.norm1.weight");
  LSE_EXPECT(norm != nullptr);
  if (norm) {
    std::vector<float> v(static_cast<std::size_t>(norm->element_count()));
    LSE_EXPECT_OK(norm->read_f32(v.data(), v.size()));
    bool all_finite = true;
    for (float x : v) all_finite = all_finite && std::isfinite(x);
    LSE_EXPECT(all_finite);
  }
}

// --- architecture detection -------------------------------------------------
//
// The predicates only read tensor names, so a handful of empty tensors with the
// right names is a faithful stand-in for a checkpoint and the whole matrix runs
// without one. What is being tested is that each architecture claims its own
// layout and refuses every other, because a wrong claim loads a checkpoint into
// the wrong model kernel and produces plausible garbage rather than an error.

namespace {

std::string write_fixture(const std::string& stem,
                          const std::vector<std::string>& names) {
  std::string header = "{";
  std::size_t offset = 0;
  for (const std::string& n : names) {
    if (header.size() > 1) header += ",";
    header += "\"" + n + "\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[" +
              std::to_string(offset) + "," + std::to_string(offset + 4) + "]}";
    offset += 4;
  }
  header += "}";
  // The header is padded to 8 bytes because safetensors requires the data
  // section to start aligned.
  while (header.size() % 8 != 0) header += " ";

  const std::string path =
      (std::filesystem::temp_directory_path() / (stem + ".safetensors")).string();
  std::ofstream out(path, std::ios::binary);
  const std::uint64_t n = header.size();
  out.write(reinterpret_cast<const char*>(&n), sizeof(n));
  out.write(header.data(), static_cast<std::streamsize>(header.size()));
  const std::vector<char> zeros(offset, 0);
  out.write(zeros.data(), static_cast<std::streamsize>(zeros.size()));
  return path;
}

const std::vector<std::string>& lemonseed_names() {
  static const std::vector<std::string> v{"blocks.0.mod.router.weight",
                                          "blocks.0.moe.w_gate",
                                          "blocks.0.mixer.in_proj_q.weight"};
  return v;
}

// Names taken from the tensor index of mlx-community/Qwen3.5-0.8B-4bit.
const std::vector<std::string>& qwen_dense_names() {
  static const std::vector<std::string> v{
      "language_model.model.layers.0.linear_attn.in_proj_qkv.weight",
      "language_model.model.layers.0.mlp.gate_proj.weight"};
  return v;
}

// ...and of mlx-community/Qwen3.5-35B-A3B-8bit, whose routed experts are three
// separate switch_mlp stacks rather than HF's fused gate_up_proj.
const std::vector<std::string>& qwen_moe_names() {
  static const std::vector<std::string> v{
      "language_model.model.layers.0.linear_attn.in_proj_qkv.weight",
      "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight",
      "language_model.model.layers.0.mlp.gate.weight"};
  return v;
}

// Detected name, or "" when nothing claims the fixture.
std::string detect(const std::string& stem, const std::vector<std::string>& names) {
  auto st = SafeTensors::open(write_fixture(stem, names));
  if (!st.ok()) return "<open failed>";
  const Config cfg;
  auto arch = detect_architecture(cfg, *st);
  if (!arch.ok()) return "";
  return std::string((*arch)->name);
}

}  // namespace

LSE_TEST(every_architecture_claims_only_its_own_checkpoint) {
  LSE_EXPECT(detect("lemonseed", lemonseed_names()) == "lemonseed");
  LSE_EXPECT(detect("q35", qwen_dense_names()) == "qwen3.5");
  LSE_EXPECT(detect("q35moe", qwen_moe_names()) == "qwen3.5-moe");
}

LSE_TEST(qwen_predicates_cannot_both_claim_a_checkpoint) {
  // The dense test requires the stacked-expert tensor absent and the MoE test
  // requires it present, so no checkpoint satisfies both. A checkpoint that
  // carries the MoE marker plus the dense MLP names is still MoE only.
  std::vector<std::string> both = qwen_moe_names();
  for (const std::string& n : qwen_dense_names()) both.push_back(n);
  LSE_EXPECT(detect("q35both", both) == "qwen3.5-moe");
}

LSE_TEST(qwen_predicates_refuse_a_lemonseed_checkpoint) {
  // Both are GDN + MoE hybrids, so this is the collision that matters: what
  // separates them is the tensor prefix and the routed-expert layout, never a
  // config name field.
  LSE_EXPECT(detect("lemonseed2", lemonseed_names()) == "lemonseed");

  // ...and lemonseed refuses theirs: it needs the Mixture-of-Depths router,
  // which no Qwen checkpoint has.
  std::vector<std::string> mixed = qwen_moe_names();
  mixed.push_back("blocks.0.moe.w_gate");
  LSE_EXPECT(detect("q35moe2", mixed) == "qwen3.5-moe");
}

LSE_TEST(an_unrecognized_checkpoint_names_what_was_tried) {
  auto st = SafeTensors::open(write_fixture("stranger", {"encoder.layer.0.weight"}));
  LSE_EXPECT(st.ok());
  if (!st.ok()) return;
  const Config cfg;
  auto arch = detect_architecture(cfg, *st);
  LSE_EXPECT(!arch.ok());
  if (arch.ok()) return;
  LSE_EXPECT(arch.status().code() == StatusCode::kNotFound);
  const std::string msg = arch.status().message();
  LSE_EXPECT(msg.find("lemonseed") != std::string::npos);
  LSE_EXPECT(msg.find("qwen3.5") != std::string::npos);
  LSE_EXPECT(msg.find("qwen3.5-moe") != std::string::npos);
}

// --- the two weight reorderings the Qwen kernels rely on ---------------------

namespace {

// Same mapping qwen3_5_common.cpp uses to reconcile HF's rotate_half RoPE with
// graph::rope's adjacent-pair rotation.
std::int64_t rope_source(std::int64_t d, std::int64_t rope_dim) {
  if (d >= rope_dim) return d;
  const std::int64_t j = d / 2;
  return d % 2 == 0 ? j : j + rope_dim / 2;
}

std::string write_f32_fixture(const std::string& stem, const std::string& name,
                              const std::vector<std::int64_t>& shape,
                              const std::vector<float>& data) {
  std::string dims;
  for (std::int64_t d : shape) {
    if (!dims.empty()) dims += ",";
    dims += std::to_string(d);
  }
  std::string header = "{\"" + name + "\":{\"dtype\":\"F32\",\"shape\":[" + dims +
                       "],\"data_offsets\":[0," +
                       std::to_string(data.size() * 4) + "]}}";
  while (header.size() % 8 != 0) header += " ";

  const std::string path =
      (std::filesystem::temp_directory_path() / (stem + ".safetensors")).string();
  std::ofstream out(path, std::ios::binary);
  const std::uint64_t n = header.size();
  out.write(reinterpret_cast<const char*>(&n), sizeof(n));
  out.write(header.data(), static_cast<std::streamsize>(header.size()));
  out.write(reinterpret_cast<const char*>(data.data()),
            static_cast<std::streamsize>(data.size() * 4));
  return path;
}

}  // namespace

LSE_TEST(rope_reordering_turns_rotate_half_into_adjacent_pairs) {
  // Qwen's weights are reordered at load so graph::rope reproduces HF's
  // rotate_half. Everything else in the attention path depends on that being
  // exactly right, and it is invisible in the tensor names, so check the
  // identity directly: interleaved-rope(reorder(x)) == reorder(rotate_half(x)).
  constexpr std::int64_t kRopeDim = 8;
  constexpr std::int64_t kHeadDim = 12;  // 4 channels past the rotation
  constexpr float kTheta = 10000.0f;
  constexpr std::int32_t kPos = 3;

  std::vector<float> x(kHeadDim);
  for (std::int64_t i = 0; i < kHeadDim; ++i) {
    x[static_cast<std::size_t>(i)] = 0.5f + 0.25f * static_cast<float>(i);
  }

  // HF: pairs (i, i + rope_dim/2) share the angle for i in [0, rope_dim/2).
  std::vector<float> want(kHeadDim);
  const std::int64_t half = kRopeDim / 2;
  for (std::int64_t i = 0; i < half; ++i) {
    const auto freq = static_cast<float>(
        1.0 / std::pow(static_cast<double>(kTheta),
                       static_cast<double>(i) / static_cast<double>(half)));
    const auto ang = static_cast<double>(kPos) * static_cast<double>(freq);
    const auto c = static_cast<float>(std::cos(ang));
    const auto s = static_cast<float>(std::sin(ang));
    const float a = x[static_cast<std::size_t>(i)];
    const float b = x[static_cast<std::size_t>(i + half)];
    want[static_cast<std::size_t>(i)] = a * c - b * s;
    want[static_cast<std::size_t>(i + half)] = b * c + a * s;
  }
  for (std::int64_t d = kRopeDim; d < kHeadDim; ++d) {
    want[static_cast<std::size_t>(d)] = x[static_cast<std::size_t>(d)];
  }

  auto tables = ops::build_rope(kRopeDim, 16, kTheta);
  LSE_EXPECT(tables.ok());
  if (!tables.ok()) return;

  graph::Array v = graph::Array::zeros(Shape{1, 1, 1, kHeadDim}, DType::kF32);
  LSE_EXPECT_OK(v.eval());
  for (std::int64_t d = 0; d < kHeadDim; ++d) {
    graph::interpreter::store_element(
        *v.node(), static_cast<std::size_t>(d),
        x[static_cast<std::size_t>(rope_source(d, kRopeDim))]);
  }
  auto rotated = ops::apply_rope(v, *tables, kPos);
  LSE_EXPECT(rotated.ok());
  if (!rotated.ok()) return;

  std::vector<float> got(kHeadDim);
  LSE_EXPECT_OK(rotated->to_host(got.data(), got.size() * sizeof(float)));
  for (std::int64_t d = 0; d < kHeadDim; ++d) {
    LSE_EXPECT_NEAR(got[static_cast<std::size_t>(d)],
                    want[static_cast<std::size_t>(rope_source(d, kRopeDim))],
                    1e-5);
  }
}

LSE_TEST(require_rows_splits_a_fused_gate_up_expert_tensor) {
  // Qwen3.5 MoE stores gate and up as one [E, 2*inter, hidden] tensor whose
  // per-expert rows are gate then up. Getting the halves the wrong way round
  // swaps SiLU's argument with its multiplicand, which stays finite and stays
  // plausible, so it is worth pinning.
  constexpr std::int64_t kE = 2, kInter = 2, kHidden = 3;
  std::vector<float> data;
  for (std::int64_t e = 0; e < kE; ++e) {
    for (std::int64_t j = 0; j < 2 * kInter; ++j) {
      for (std::int64_t h = 0; h < kHidden; ++h) {
        data.push_back(static_cast<float>(e * 100 + j * 10 + h));
      }
    }
  }
  const std::string path = write_f32_fixture(
      "fused_experts", "gu", {kE, 2 * kInter, kHidden}, data);
  auto st = SafeTensors::open(path);
  LSE_EXPECT(st.ok());
  if (!st.ok()) return;

  std::vector<std::int64_t> gate_rows, up_rows;
  for (std::int64_t e = 0; e < kE; ++e) {
    for (std::int64_t j = 0; j < kInter; ++j) {
      gate_rows.push_back(e * 2 * kInter + j);
      up_rows.push_back(e * 2 * kInter + kInter + j);
    }
  }

  WeightBinder binder(*st);
  auto gate = binder.require_rows("gu", gate_rows, Shape{kE, kInter, kHidden});
  auto up = binder.require_rows("gu", up_rows, Shape{kE, kInter, kHidden});
  LSE_EXPECT(gate.ok());
  LSE_EXPECT(up.ok());
  if (!gate.ok() || !up.ok()) return;

  std::vector<float> g(static_cast<std::size_t>(kE * kInter * kHidden));
  std::vector<float> u(g.size());
  LSE_EXPECT_OK(gate->to_host(g.data(), g.size() * sizeof(float)));
  LSE_EXPECT_OK(up->to_host(u.data(), u.size() * sizeof(float)));
  for (std::int64_t e = 0; e < kE; ++e) {
    for (std::int64_t j = 0; j < kInter; ++j) {
      for (std::int64_t h = 0; h < kHidden; ++h) {
        const auto at = static_cast<std::size_t>((e * kInter + j) * kHidden + h);
        LSE_EXPECT_NEAR(g[at], static_cast<float>(e * 100 + j * 10 + h), 1e-6);
        LSE_EXPECT_NEAR(u[at],
                        static_cast<float>(e * 100 + (kInter + j) * 10 + h), 1e-6);
      }
    }
  }
}

LSE_TEST(require_rows_rejects_an_order_that_cannot_fill_the_shape) {
  const std::string path =
      write_f32_fixture("small_rows", "w", {4, 2}, {0, 1, 2, 3, 4, 5, 6, 7});
  auto st = SafeTensors::open(path);
  LSE_EXPECT(st.ok());
  if (!st.ok()) return;
  WeightBinder binder(*st);
  LSE_EXPECT(!binder.require_rows("w", {0, 1}, Shape{4, 2}).ok());
  LSE_EXPECT(!binder.require_rows("w", {0, 9}, Shape{2, 2}).ok());
}

// --- HuggingFace config parsing ---------------------------------------------

namespace {

// The fields Qwen3.5-4B's config.json carries, trimmed to what the engine reads.
std::string hf_config(const std::string& extra = {}) {
  return R"({"tie_word_embeddings": true, "text_config": {
    "vocab_size": 248320, "hidden_size": 2560, "num_hidden_layers": 32,
    "rms_norm_eps": 1e-06, "full_attention_interval": 4,
    "num_attention_heads": 16, "num_key_value_heads": 4, "head_dim": 256,
    "linear_num_key_heads": 16, "linear_num_value_heads": 32,
    "linear_key_head_dim": 128, "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4, "intermediate_size": 9216,
    "rope_parameters": {"rope_theta": 10000000, "partial_rotary_factor": 0.25}
    )" + extra + "}}";
}

}  // namespace

LSE_TEST(hf_config_is_read_with_hf_field_names) {
  auto c = Config::from_json_string(hf_config());
  LSE_EXPECT(c.ok());
  if (!c.ok()) return;
  LSE_EXPECT_EQ(c->vocab_size, 248320);
  LSE_EXPECT_EQ(c->hidden_size, 2560);
  LSE_EXPECT_EQ(c->num_layers, 32);
  LSE_EXPECT_EQ(c->attn_q_heads, 16);
  LSE_EXPECT_EQ(c->attn_kv_heads, 4);
  LSE_EXPECT_EQ(c->attn_head_dim, 256);
  // 0.25 of a 256-wide head.
  LSE_EXPECT_EQ(c->rope_dim, 64);
  LSE_EXPECT_EQ(c->gdn_qk_heads, 16);
  LSE_EXPECT_EQ(c->gdn_v_heads, 32);
  LSE_EXPECT_EQ(c->gdn_head_dim, 128);
  LSE_EXPECT_EQ(c->mlp_intermediate, 9216);
  // No sliding window, so every attention layer sees the whole prefix.
  LSE_EXPECT_EQ(c->sliding_window, 0);
  LSE_EXPECT(c->tie_word_embeddings);
  LSE_EXPECT_EQ(c->num_experts, 0);
}

LSE_TEST(the_quantization_block_travels_with_the_config) {
  // An MLX checkpoint carries its group geometry beside the shape fields, and
  // the loader needs both from the same read. The block is top level in both
  // spellings, so it is not inside text_config.
  const std::string quantized =
      R"({"quantization": {"group_size": 64, "bits": 4, "mode": "affine",
                           "language_model.model.layers.0.mlp.gate": {"bits": 8}}, )" +
      hf_config().substr(1);
  auto c = Config::from_json_string(quantized);
  LSE_EXPECT(c.ok());
  if (!c.ok()) return;
  LSE_EXPECT(c->quantization.has_global());
  LSE_EXPECT_EQ(c->quantization.global().bits, 4);
  LSE_EXPECT_EQ(c->quantization.override_count(), std::size_t(1));
  auto router =
      c->quantization.resolve("language_model.model.layers.0.mlp.gate.weight");
  LSE_EXPECT(router.ok());
  if (router.ok()) LSE_EXPECT_EQ(router->bits, 8);

  // An unquantized checkpoint carries no block, and nothing is invented for it.
  auto plain = Config::from_json_string(hf_config());
  LSE_EXPECT(plain.ok());
  if (plain.ok()) {
    LSE_EXPECT(!plain->quantization.has_global());
    LSE_EXPECT(!plain->quantization.resolve("any.tensor.weight").ok());
  }
}

LSE_TEST(hf_moe_config_carries_the_routing_shape) {
  auto c = Config::from_json_string(hf_config(
      R"(, "num_experts": 256, "num_experts_per_tok": 8,
          "moe_intermediate_size": 512, "shared_expert_intermediate_size": 512)"));
  LSE_EXPECT(c.ok());
  if (!c.ok()) return;
  LSE_EXPECT_EQ(c->num_experts, 256);
  LSE_EXPECT_EQ(c->num_active_experts, 8);
  LSE_EXPECT_EQ(c->expert_intermediate, 512);
  LSE_EXPECT_EQ(c->shared_expert_intermediate, 512);
  LSE_EXPECT_EQ(c->num_shared_experts, 1);
  // Plain top-k, not lemonseed's band.
  LSE_EXPECT_NEAR(c->expert_score_band, 1.0, 1e-9);
}

LSE_TEST(a_missing_hf_field_is_an_error_not_a_default) {
  // Dropping head_dim must not leave the 64 the Config defaults to: that would
  // build a model of the wrong shape that still passes validate().
  std::string text = hf_config();
  const std::string key = R"("head_dim": 256,)";
  const std::size_t at = text.find(key);
  LSE_EXPECT(at != std::string::npos);
  if (at == std::string::npos) return;
  text.erase(at, key.size());

  auto c = Config::from_json_string(text);
  LSE_EXPECT(!c.ok());
  if (c.ok()) return;
  LSE_EXPECT(c.status().code() == StatusCode::kNotFound);
  LSE_EXPECT(c.status().message().find("head_dim") != std::string::npos);
}

LSE_TEST(layer_types_must_agree_with_the_interval_rule) {
  // The engine derives the mixer schedule from full_attention_interval. If the
  // checkpoint's own layer_types disagrees, every layer past the divergence
  // would be built as the wrong mixer.
  std::string types = R"(, "num_hidden_layers": 4, "layer_types":
      ["linear_attention", "full_attention", "linear_attention", "full_attention"])";
  auto bad = Config::from_json_string(hf_config(types));
  LSE_EXPECT(!bad.ok());
  if (!bad.ok()) {
    LSE_EXPECT(bad.status().message().find("layer_types") != std::string::npos);
  }

  std::string good = R"(, "num_hidden_layers": 4, "layer_types":
      ["linear_attention", "linear_attention", "linear_attention", "full_attention"])";
  LSE_EXPECT(Config::from_json_string(hf_config(good)).ok());
}

LSE_TEST(a_lemonseed_config_still_takes_the_native_path) {
  // No text_config, so none of the HF requirements apply.
  const Config a = preset_b1_5();
  auto b = Config::from_json_string(a.to_json());
  LSE_EXPECT(b.ok());
  if (!b.ok()) return;
  LSE_EXPECT_EQ(b->gdn_qk_heads, 8);
  LSE_EXPECT_EQ(b->gdn_v_heads, 4);
  LSE_EXPECT_EQ(b->sliding_window, 256);
}

LSE_TEST(param_count_estimate_is_in_the_right_ballpark) {
  if (!have_model()) return;
  auto paths = resolve_model(model_dir());
  if (!paths.ok()) return;
  auto st = SafeTensors::open(paths->weights);
  if (!st.ok()) return;

  const double actual = static_cast<double>(st->total_parameters());
  const double estimate = static_cast<double>(preset_b1_5().param_count());
  // The analytic estimate is a budgeting aid, not a checksum: 5% is enough to
  // catch a structural mistake without pinning every bias term.
  LSE_EXPECT(std::fabs(estimate - actual) / actual < 0.05);
}

// --- the builder must account for every tensor -------------------------------
//
// layer.hpp names an unclaimed tensor as the likeliest silent failure once a
// second architecture exists, so the guarantee is checked from both ends on a
// checkpoint small enough to write by hand: a tensor the builder never reads
// must fail the load, and a tensor the builder needs but the file lacks must
// fail it too. A whole checkpoint is generated from the Config so the two
// negative cases differ from the positive one by exactly one tensor.

namespace {

struct NamedShape {
  std::string name;
  std::vector<std::int64_t> dims;
};

std::string write_shaped_fixture(const std::string& stem,
                                 const std::vector<NamedShape>& tensors) {
  std::string header = "{";
  std::size_t offset = 0;
  for (const NamedShape& t : tensors) {
    std::size_t count = 1;
    std::string dims;
    for (std::int64_t d : t.dims) {
      if (!dims.empty()) dims += ",";
      dims += std::to_string(d);
      count *= static_cast<std::size_t>(d);
    }
    if (header.size() > 1) header += ",";
    header += "\"" + t.name + "\":{\"dtype\":\"F32\",\"shape\":[" + dims +
              "],\"data_offsets\":[" + std::to_string(offset) + "," +
              std::to_string(offset + count * 4) + "]}";
    offset += count * 4;
  }
  header += "}";
  while (header.size() % 8 != 0) header += " ";

  const std::string path =
      (std::filesystem::temp_directory_path() / (stem + ".safetensors")).string();
  std::ofstream out(path, std::ios::binary);
  const std::uint64_t n = header.size();
  out.write(reinterpret_cast<const char*>(&n), sizeof(n));
  out.write(header.data(), static_cast<std::streamsize>(header.size()));
  // Small non-zero values: a load must survive real numbers, and an all-zero
  // file would hide a shape error that only shows up as a NaN.
  std::vector<float> data(offset / 4);
  for (std::size_t i = 0; i < data.size(); ++i) {
    data[i] = 0.01f * static_cast<float>(static_cast<int>(i % 17) - 8);
  }
  out.write(reinterpret_cast<const char*>(data.data()),
            static_cast<std::streamsize>(offset));
  return path;
}

Config tiny_qwen_config(bool moe) {
  Config c;
  c.vocab_size = 64;
  c.hidden_size = 32;
  c.num_layers = 4;
  c.full_attention_interval = 4;
  c.global_attention_layers.clear();
  c.sliding_window = 0;
  c.attn_q_heads = 2;
  c.attn_kv_heads = 1;
  c.attn_head_dim = 16;
  c.rope_dim = 4;
  c.rope_theta = 10000000.0f;
  c.gdn_qk_heads = 2;
  c.gdn_v_heads = 2;
  c.gdn_head_dim = 8;
  c.gdn_conv_kernel = 4;
  c.dtype = "float32";
  if (moe) {
    c.mlp_intermediate = 0;
    c.num_experts = 4;
    c.num_active_experts = 2;
    c.num_shared_experts = 1;
    c.expert_intermediate = 16;
    c.shared_expert_intermediate = 16;
    c.expert_score_band = 1.0f;
    c.tie_word_embeddings = false;
  } else {
    c.mlp_intermediate = 64;
    c.num_experts = 0;
    c.num_active_experts = 0;
    c.num_shared_experts = 0;
    c.expert_intermediate = 0;
    c.tie_word_embeddings = true;
  }
  return c;
}

// Every tensor Qwen3.5 puts in the file, derived from the config the same way
// the builder derives what it asks for.
std::vector<NamedShape> qwen_checkpoint(const Config& c) {
  const std::int64_t h = c.hidden_size;
  const std::int64_t qh = c.attn_q_heads, kvh = c.attn_kv_heads;
  const std::int64_t ahd = c.attn_head_dim;
  const std::int64_t kh = c.gdn_qk_heads, vh = c.gdn_v_heads;
  const std::int64_t ghd = c.gdn_head_dim;
  const std::int64_t conv_dim = 2 * kh * ghd + vh * ghd;

  std::vector<NamedShape> t;
  t.push_back({"language_model.model.embed_tokens.weight", {c.vocab_size, h}});
  t.push_back({"language_model.model.norm.weight", {h}});
  if (!c.tie_word_embeddings) {
    t.push_back({"language_model.lm_head.weight", {c.vocab_size, h}});
  }

  for (std::int32_t i = 0; i < c.num_layers; ++i) {
    const std::string p =
        "language_model.model.layers." + std::to_string(i) + ".";
    t.push_back({p + "input_layernorm.weight", {h}});
    t.push_back({p + "post_attention_layernorm.weight", {h}});

    if (c.is_attention_layer(i)) {
      const std::string a = p + "self_attn.";
      t.push_back({a + "q_proj.weight", {2 * qh * ahd, h}});
      t.push_back({a + "k_proj.weight", {kvh * ahd, h}});
      t.push_back({a + "v_proj.weight", {kvh * ahd, h}});
      t.push_back({a + "o_proj.weight", {h, qh * ahd}});
      t.push_back({a + "q_norm.weight", {ahd}});
      t.push_back({a + "k_norm.weight", {ahd}});
    } else {
      const std::string g = p + "linear_attn.";
      t.push_back({g + "in_proj_qkv.weight", {conv_dim, h}});
      t.push_back({g + "in_proj_z.weight", {vh * ghd, h}});
      t.push_back({g + "in_proj_a.weight", {vh, h}});
      t.push_back({g + "in_proj_b.weight", {vh, h}});
      t.push_back({g + "conv1d.weight", {conv_dim, c.gdn_conv_kernel, 1}});
      t.push_back({g + "A_log", {vh}});
      t.push_back({g + "dt_bias", {vh}});
      t.push_back({g + "norm.weight", {ghd}});
      t.push_back({g + "out_proj.weight", {h, vh * ghd}});
    }

    const std::string m = p + "mlp.";
    if (c.num_experts > 0) {
      const std::int64_t e = c.num_experts, in = c.expert_intermediate;
      const std::int64_t si = c.shared_expert_intermediate;
      t.push_back({m + "gate.weight", {e, h}});
      t.push_back({m + "switch_mlp.gate_proj.weight", {e, in, h}});
      t.push_back({m + "switch_mlp.up_proj.weight", {e, in, h}});
      t.push_back({m + "switch_mlp.down_proj.weight", {e, h, in}});
      t.push_back({m + "shared_expert.gate_proj.weight", {si, h}});
      t.push_back({m + "shared_expert.up_proj.weight", {si, h}});
      t.push_back({m + "shared_expert.down_proj.weight", {h, si}});
      t.push_back({m + "shared_expert_gate.weight", {1, h}});
    } else {
      const std::int64_t in = c.mlp_intermediate;
      t.push_back({m + "gate_proj.weight", {in, h}});
      t.push_back({m + "up_proj.weight", {in, h}});
      t.push_back({m + "down_proj.weight", {h, in}});
    }
  }
  return t;
}

// Builds and loads, returning the load status so a caller can assert on the
// message rather than just on failure.
Status load_fixture(const std::string& stem, const Config& c,
                    const std::vector<NamedShape>& tensors) {
  auto st = SafeTensors::open(write_shaped_fixture(stem, tensors));
  if (!st.ok()) return st.status();
  auto model = build_model(c, *st);
  if (!model.ok()) return model.status();
  WeightBinder binder(*st);
  return (*model)->load(binder);
}

}  // namespace

LSE_TEST(qwen_dense_load_claims_every_tensor_in_the_checkpoint) {
  const Config c = tiny_qwen_config(/*moe=*/false);
  const std::vector<NamedShape> full = qwen_checkpoint(c);
  LSE_EXPECT_OK(load_fixture("q35_exact", c, full));

  // One tensor no layer reads. Nothing in the forward pass would notice, which
  // is exactly why the audit has to.
  std::vector<NamedShape> extra = full;
  extra.push_back(
      {"language_model.model.layers.0.linear_attn.in_proj_g.weight", {8, 32}});
  const Status stray = load_fixture("q35_extra", c, extra);
  LSE_EXPECT(!stray.ok());
  if (!stray.ok()) {
    LSE_EXPECT(stray.message().find("in_proj_g") != std::string::npos);
  }

  // One tensor a layer needs and the file lacks.
  std::vector<NamedShape> absent;
  for (const NamedShape& t : full) {
    if (t.name != "language_model.model.layers.1.mlp.up_proj.weight") {
      absent.push_back(t);
    }
  }
  LSE_EXPECT_EQ(absent.size() + 1, full.size());
  const Status missing = load_fixture("q35_absent", c, absent);
  LSE_EXPECT(!missing.ok());
  if (!missing.ok()) {
    LSE_EXPECT(missing.code() == StatusCode::kNotFound);
    LSE_EXPECT(missing.message().find("layers.1.mlp.up_proj") !=
               std::string::npos);
  }
}

LSE_TEST(qwen_moe_load_claims_every_tensor_in_the_checkpoint) {
  // The MoE builder has no end-to-end run behind it, so the tensor contract is
  // the only thing pinning it: expert stack, shared expert, shared gate and an
  // untied head all have to be claimed, and nothing else may be.
  const Config c = tiny_qwen_config(/*moe=*/true);
  const std::vector<NamedShape> full = qwen_checkpoint(c);
  LSE_EXPECT_OK(load_fixture("q35moe_exact", c, full));

  std::vector<NamedShape> extra = full;
  extra.push_back({"language_model.model.layers.2.mlp.experts.bias", {4, 16}});
  const Status stray = load_fixture("q35moe_extra", c, extra);
  LSE_EXPECT(!stray.ok());
  if (!stray.ok()) {
    LSE_EXPECT(stray.message().find("experts.bias") != std::string::npos);
  }

  // The untied head is part of the contract: dropping it must fail, not
  // silently fall back to the embedding table.
  std::vector<NamedShape> headless;
  for (const NamedShape& t : full) {
    if (t.name != "language_model.lm_head.weight") headless.push_back(t);
  }
  const Status no_head = load_fixture("q35moe_headless", c, headless);
  LSE_EXPECT(!no_head.ok());
}

LSE_TEST(qwen_refuses_the_vision_tower_and_nothing_else) {
  // One prefix is declared not-part-of-the-decoder, and it is the one the
  // checkpoint actually uses: `vision_tower.`, not HF's `model.visual.`. It has
  // to actually be skipped, and the declaration must not widen into "ignore
  // what we missed".
  const Config c = tiny_qwen_config(/*moe=*/false);
  std::vector<NamedShape> t = qwen_checkpoint(c);
  t.push_back({"vision_tower.blocks.0.attn.qkv.weight", {8, 8}});
  t.push_back({"vision_tower.merger.linear_fc2.weight", {8, 8}});
  LSE_EXPECT_OK(load_fixture("q35_refused", c, t));

  // A near-miss on the refused prefix is still an unread tensor.
  t.push_back({"vision_towerx.weight", {4}});
  LSE_EXPECT(!load_fixture("q35_nearmiss", c, t).ok());
}

namespace {

// A checkpoint in MLX's shape: every rank-2 `.weight` becomes a packed U32
// plane plus BF16 `.scales` and `.biases`, and everything else stays F32. That
// is the layout of every Qwen3.5 release, so a load that only works on an
// unquantized fixture has not been tested against anything real.
struct QuantFixture {
  std::string path;
  std::string config_json;
};

bool is_quantized_weight(const NamedShape& t) {
  return t.dims.size() == 2 && t.name.size() > 7 &&
         t.name.compare(t.name.size() - 7, 7, ".weight") == 0;
}

QuantFixture write_quantized_fixture(const std::string& stem,
                                     const std::vector<NamedShape>& tensors,
                                     int bits, int group_size,
                                     bool drop_biases_of_first) {
  auto spec = quant::GroupAffine::make(bits, group_size);
  std::string header = "{";
  std::size_t offset = 0;
  bool dropped = false;

  auto entry = [&](const std::string& name, const char* dtype,
                   const std::vector<std::int64_t>& dims,
                   std::size_t elem_bytes) {
    std::size_t count = 1;
    std::string d;
    for (std::int64_t v : dims) {
      if (!d.empty()) d += ",";
      d += std::to_string(v);
      count *= static_cast<std::size_t>(v);
    }
    if (header.size() > 1) header += ",";
    header += "\"" + name + "\":{\"dtype\":\"" + dtype + "\",\"shape\":[" + d +
              "],\"data_offsets\":[" + std::to_string(offset) + "," +
              std::to_string(offset + count * elem_bytes) + "]}";
    offset += count * elem_bytes;
  };

  for (const NamedShape& t : tensors) {
    if (!is_quantized_weight(t)) {
      entry(t.name, "F32", t.dims, 4);
      continue;
    }
    const std::int64_t rows = t.dims[0], k = t.dims[1];
    const auto lanes = static_cast<std::int64_t>(
        spec.ok() ? spec->packed_words(static_cast<std::size_t>(k)) : 0);
    const auto groups = static_cast<std::int64_t>(
        spec.ok() ? spec->group_count(static_cast<std::size_t>(k)) : 0);
    const std::string base = t.name.substr(0, t.name.size() - 7);
    entry(t.name, "U32", {rows, lanes}, 4);
    entry(base + ".scales", "BF16", {rows, groups}, 2);
    if (drop_biases_of_first && !dropped) {
      dropped = true;
    } else {
      entry(base + ".biases", "BF16", {rows, groups}, 2);
    }
  }
  header += "}";
  while (header.size() % 8 != 0) header += " ";

  const std::string path =
      (std::filesystem::temp_directory_path() / (stem + ".safetensors")).string();
  std::ofstream out(path, std::ios::binary);
  const std::uint64_t n = header.size();
  out.write(reinterpret_cast<const char*>(&n), sizeof(n));
  out.write(header.data(), static_cast<std::streamsize>(header.size()));
  // Byte-patterned rather than zeroed: an all-zero scale plane would make every
  // dequantized weight zero and hide a wiring error behind a finite result.
  std::vector<std::uint8_t> data(offset);
  for (std::size_t i = 0; i < data.size(); ++i) {
    data[i] = static_cast<std::uint8_t>((i * 37u + 11u) % 61u);
  }
  out.write(reinterpret_cast<const char*>(data.data()),
            static_cast<std::streamsize>(offset));

  QuantFixture f;
  f.path = path;
  f.config_json = "{\"quantization\":{\"group_size\":" +
                  std::to_string(group_size) + ",\"bits\":" +
                  std::to_string(bits) + ",\"mode\":\"affine\"}}";
  return f;
}

// Every last axis a Qwen3.5 weight contracts over has to be a whole number of
// groups, which the 32-wide toy config does not manage for the GDN output
// projection until the value heads are as wide as one group.
Config tiny_quantized_qwen_config() {
  Config c = tiny_qwen_config(/*moe=*/false);
  c.gdn_head_dim = 16;
  return c;
}

}  // namespace

// A packed plane holds several weights per u32 lane, so summing stored elements
// reports an 0.8B checkpoint as 0.2B. The scale column has to expand them.
LSE_TEST(parameter_scale_expands_packed_planes) {
  constexpr std::int64_t kOut = 8;
  constexpr std::int64_t kIn = 64;
  const QuantFixture fx = write_quantized_fixture(
      "scale_expand", {{"m.weight", {kOut, kIn}}}, 4, 32, /*drop=*/false);

  auto st = SafeTensors::open(fx.path);
  LSE_EXPECT_OK(st.status());
  auto quant = quant::GroupAffineMap::from_config_json(fx.config_json);
  LSE_EXPECT_OK(quant.status());

  // Logical: exactly the weights the matrix holds. Stored: a quarter of the
  // lanes plus the scale and bias planes, which are not parameters at all.
  LSE_EXPECT_EQ(st->logical_parameters(&*quant),
                static_cast<std::size_t>(kOut * kIn));
  LSE_EXPECT(st->total_parameters() < static_cast<std::size_t>(kOut * kIn));
}

namespace {

// A safetensors file with per-tensor dtypes, which write_shaped_fixture (all
// F32) and write_quantized_fixture (2-D weights only) cannot express: a stacked
// expert plane is 3-D and u32.
struct RawTensor {
  std::string name;
  std::string dtype;
  std::vector<std::int64_t> dims;
  int elem_bytes;
};

void write_raw_safetensors(const std::filesystem::path& path,
                           const std::vector<RawTensor>& tensors) {
  std::string header = "{";
  std::size_t offset = 0;
  for (const RawTensor& t : tensors) {
    std::size_t count = 1;
    std::string dims;
    for (std::int64_t d : t.dims) {
      if (!dims.empty()) dims += ",";
      dims += std::to_string(d);
      count *= static_cast<std::size_t>(d);
    }
    if (header.size() > 1) header += ",";
    const std::size_t bytes = count * static_cast<std::size_t>(t.elem_bytes);
    header += "\"" + t.name + "\":{\"dtype\":\"" + t.dtype + "\",\"shape\":[" +
              dims + "],\"data_offsets\":[" + std::to_string(offset) + "," +
              std::to_string(offset + bytes) + "]}";
    offset += bytes;
  }
  header += "}";
  while (header.size() % 8 != 0) header += " ";

  std::ofstream out(path, std::ios::binary);
  const std::uint64_t n = header.size();
  out.write(reinterpret_cast<const char*>(&n), sizeof(n));
  out.write(header.data(), static_cast<std::streamsize>(header.size()));
  std::vector<std::uint8_t> data(offset);
  for (std::size_t i = 0; i < data.size(); ++i) {
    data[i] = static_cast<std::uint8_t>((i * 31u + 7u) % 59u);
  }
  out.write(reinterpret_cast<const char*>(data.data()),
            static_cast<std::streamsize>(offset));
}

}  // namespace

// A stacked 3-D group-affine expert tensor binds: quant_linear_indexed reads
// one expert out of the stack in place. The verdict this listing prints is
// pinned to the same precondition the binder enforces, and the two rules live in
// separate files (weights.cpp restates layer.cpp's), so a checkpoint the binder
// accepts and the listing calls unloadable — or the reverse — is the failure
// this catches.
LSE_TEST(a_stacked_expert_checkpoint_is_reported_loadable) {
  constexpr int kBits = 8;
  constexpr int kGroup = 32;
  const Config c = tiny_qwen_config(/*moe=*/true);

  // Every tensor F32, except the two stacked expert planes that contract over
  // hidden_size, which are written the way MLX writes them: a u32 packed plane
  // with bf16 scales and biases beside it.
  std::vector<RawTensor> raw;
  for (const NamedShape& t : qwen_checkpoint(c)) {
    const bool stacked =
        t.name.find("switch_mlp.gate_proj.weight") != std::string::npos ||
        t.name.find("switch_mlp.up_proj.weight") != std::string::npos;
    if (!stacked) {
      raw.push_back({t.name, "F32", t.dims, 4});
      continue;
    }
    const std::int64_t k = t.dims.back();
    const std::string base = t.name.substr(0, t.name.size() - 7);
    raw.push_back({t.name, "U32", {t.dims[0], t.dims[1], k * kBits / 32}, 4});
    raw.push_back({base + ".scales", "BF16", {t.dims[0], t.dims[1], k / kGroup}, 2});
    raw.push_back({base + ".biases", "BF16", {t.dims[0], t.dims[1], k / kGroup}, 2});
  }

  FakeHub hub("stacked");
  const std::filesystem::path snap = hub.snapshot("org/Stacked-8bit");
  write_raw_safetensors(snap / "model.safetensors", raw);
  // The config declares the geometry, so the refusal is about rank and not
  // about a missing quantization block.
  std::string cfg = c.to_json();
  cfg.insert(1, "\"quantization\":{\"group_size\":" + std::to_string(kGroup) +
                    ",\"bits\":" + std::to_string(kBits) +
                    ",\"mode\":\"affine\"},");
  write_text(snap / "config.json", cfg);

  const CacheModel m = inspect_model_dir(snap.string(), "org/Stacked-8bit");
  LSE_EXPECT(m.engine_arch == "qwen3.5-moe");
  LSE_EXPECT(m.loadable == Loadable::kYes);
  LSE_EXPECT(m.reason.empty());
  // The geometry is reported from the config, and the stack does not change it:
  // the expert axis is not a quantization axis.
  LSE_EXPECT(m.quantization.find("8-bit") != std::string::npos);

  // The rank the stack is not: a 4-D plane has no expert-and-row addressing, and
  // it must be refused rather than folded into one of the two legal ranks.
  std::vector<RawTensor> rank4;
  for (const RawTensor& t : raw) {
    RawTensor mod = t;
    if (mod.name.find("switch_mlp.gate_proj") != std::string::npos) {
      mod.dims.push_back(1);
    }
    rank4.push_back(mod);
  }
  const std::filesystem::path bad = hub.snapshot("org/Stacked-rank4");
  write_raw_safetensors(bad / "model.safetensors", rank4);
  write_text(bad / "config.json", cfg);
  const CacheModel r4 = inspect_model_dir(bad.string(), "org/Stacked-rank4");
  LSE_EXPECT(r4.loadable == Loadable::kNo);
  LSE_EXPECT(r4.reason.find("switch_mlp") != std::string::npos);
}

LSE_TEST(a_quantized_qwen_checkpoint_binds_all_three_planes) {
  const Config base = tiny_quantized_qwen_config();
  const std::vector<NamedShape> t = qwen_checkpoint(base);
  const QuantFixture f =
      write_quantized_fixture("q35_quant", t, 4, 32, /*drop=*/false);

  auto cfg = Config::from_json_string(f.config_json);
  LSE_EXPECT(cfg.ok());
  if (!cfg.ok()) return;
  Config c = base;
  c.quantization = cfg->quantization;
  LSE_EXPECT(c.quantization.has_global());

  auto st = SafeTensors::open(f.path);
  LSE_EXPECT(st.ok());
  if (!st.ok()) return;
  auto model = build_model(c, *st);
  LSE_EXPECT(model.ok());
  if (!model.ok()) return;

  WeightBinder binder(*st, &c.quantization);
  LSE_EXPECT_OK((*model)->load(binder));
  // Three planes per quantized weight, all claimed: the audit inside load()
  // fails on anything left over, so reaching here is the proof.
  LSE_EXPECT_EQ(binder.claimed_count(), st->tensors().size());

  const std::int64_t T = 3;
  graph::Array tokens = graph::Array::zeros(Shape{1, T}, DType::kF32);
  LSE_EXPECT_OK(tokens.eval());
  for (std::int64_t i = 0; i < T; ++i) {
    graph::interpreter::store_element(*tokens.node(), (std::size_t)i,
                                      (float)(i + 1));
  }
  auto hid = (*model)->hidden(tokens, nullptr, nullptr);
  LSE_EXPECT(hid.ok());
  if (!hid.ok()) return;
  std::vector<float> v((std::size_t)(T * c.hidden_size));
  LSE_EXPECT_OK(hid->to_host(v.data(), v.size() * sizeof(float)));
  bool finite = true;
  for (float x : v) finite = finite && std::isfinite(x);
  LSE_EXPECT(finite);

  // The tied head reads the same packed table the embedding gathered from.
  auto logits = (*model)->lm_head(*hid);
  LSE_EXPECT(logits.ok());
  if (logits.ok()) LSE_EXPECT_EQ(logits->shape().dim(2), c.vocab_size);
}

LSE_TEST(a_quantized_weight_without_its_geometry_is_an_error) {
  const Config base = tiny_quantized_qwen_config();
  const std::vector<NamedShape> t = qwen_checkpoint(base);

  // No quantization block in the config: the group size is not derivable from
  // the shapes, so guessing one is the failure mode this refuses.
  {
    const QuantFixture f =
        write_quantized_fixture("q35_noquant", t, 4, 32, /*drop=*/false);
    auto st = SafeTensors::open(f.path);
    LSE_EXPECT(st.ok());
    if (!st.ok()) return;
    auto model = build_model(base, *st);
    LSE_EXPECT(model.ok());
    if (!model.ok()) return;
    WeightBinder binder(*st);
    const Status s = (*model)->load(binder);
    LSE_EXPECT(!s.ok());
    if (!s.ok()) {
      LSE_EXPECT(s.message().find("no quantization block") != std::string::npos);
    }
  }

  // Half a triple: a scale plane with no bias plane.
  {
    const QuantFixture f =
        write_quantized_fixture("q35_halftriple", t, 4, 32, /*drop=*/true);
    auto cfg = Config::from_json_string(f.config_json);
    LSE_EXPECT(cfg.ok());
    if (!cfg.ok()) return;
    Config c = base;
    c.quantization = cfg->quantization;
    auto st = SafeTensors::open(f.path);
    LSE_EXPECT(st.ok());
    if (!st.ok()) return;
    auto model = build_model(c, *st);
    LSE_EXPECT(model.ok());
    if (!model.ok()) return;
    WeightBinder binder(*st, &c.quantization);
    const Status s = (*model)->load(binder);
    LSE_EXPECT(!s.ok());
    if (!s.ok()) {
      LSE_EXPECT(s.message().find(".biases") != std::string::npos);
    }
  }
}

LSE_TEST(a_tiny_qwen_of_each_kind_runs_a_forward_pass) {
  // Claiming the right tensors is not the same as computing with them. The MoE
  // path in particular has no checkpoint small enough to run end to end, so the
  // routed experts, the shared expert and its sigmoid gate are exercised here:
  // a wrong shape or a router that indexes past the stack fails or NaNs, and
  // neither shows up in a load-only test.
  for (const bool moe : {false, true}) {
    const Config c = tiny_qwen_config(moe);
    const std::vector<NamedShape> t = qwen_checkpoint(c);
    const std::string stem = moe ? "q35moe_fwd" : "q35_fwd";
    auto st = SafeTensors::open(write_shaped_fixture(stem, t));
    LSE_EXPECT(st.ok());
    if (!st.ok()) continue;
    auto model = build_model(c, *st);
    LSE_EXPECT(model.ok());
    if (!model.ok()) continue;
    WeightBinder binder(*st);
    LSE_EXPECT_OK((*model)->load(binder));

    const std::int64_t T = 3;
    graph::Array tokens = graph::Array::zeros(Shape{1, T}, DType::kF32);
    LSE_EXPECT_OK(tokens.eval());
    for (std::int64_t i = 0; i < T; ++i) {
      graph::interpreter::store_element(*tokens.node(), (std::size_t)i,
                                        (float)(i + 1));
    }
    auto hid = (*model)->hidden(tokens, nullptr, nullptr);
    LSE_EXPECT(hid.ok());
    if (!hid.ok()) continue;
    LSE_EXPECT_EQ(hid->shape().dim(2), c.hidden_size);

    std::vector<float> v((std::size_t)(T * c.hidden_size));
    LSE_EXPECT_OK(hid->to_host(v.data(), v.size() * sizeof(float)));
    bool finite = true;
    for (float x : v) finite = finite && std::isfinite(x);
    LSE_EXPECT(finite);

    auto logits = (*model)->lm_head(*hid);
    LSE_EXPECT(logits.ok());
    if (logits.ok()) LSE_EXPECT_EQ(logits->shape().dim(2), c.vocab_size);
  }
}

// --- detection, from the other side -----------------------------------------

LSE_TEST(no_qwen_predicate_claims_a_lemonseed_shaped_checkpoint) {
  // Asserting that lemonseed wins on a lemonseed file proves nothing about the
  // Qwen predicates: on a tie detect_architecture keeps the first match it
  // finds, so a second claimant would be invisible. Make lemonseed abstain by
  // dropping the one tensor its own test requires, and the detection must then
  // fail outright — if either Qwen predicate could claim this file, it would
  // answer here.
  std::vector<std::string> names;
  for (const std::string& n : lemonseed_names()) {
    if (n != "blocks.0.mod.router.weight") names.push_back(n);
  }
  LSE_EXPECT(detect("lemonseed_no_mod", names).empty());

  // Same argument for the real checkpoint's full tensor list when it is here.
  if (!have_model()) return;
  auto paths = resolve_model(model_dir());
  if (!paths.ok()) return;
  auto st = SafeTensors::open(paths->weights);
  if (!st.ok()) return;
  std::vector<std::string> real;
  for (const auto& [name, _] : st->tensors()) {
    if (name != "blocks.0.mod.router.weight") real.push_back(name);
  }
  LSE_EXPECT(detect("lemonseed_real_no_mod", real).empty());
}

// What paging actually moves, measured per model rather than derived: the pools
// a session holds after a short prompt against what one contiguous span at the
// engine KV length would have reserved.
//
// LSE_TEST_KV_MODELS overrides the list (colon-separated repo ids or paths).
// The default skips checkpoints large enough to make the suite slow; point the
// variable at one to measure it.
LSE_TEST(a_paged_session_reserves_a_rung_not_the_capacity) {
  std::vector<std::string> want{model_dir()};
  if (const char* env = std::getenv("LSE_TEST_KV_MODELS")) {
    want.clear();
    std::string all(env);
    for (std::size_t at = 0; at <= all.size();) {
      const std::size_t sep = all.find(':', at);
      const std::string one = all.substr(at, sep == std::string::npos
                                                ? std::string::npos
                                                : sep - at);
      if (!one.empty()) want.push_back(one);
      if (sep == std::string::npos) break;
      at = sep + 1;
    }
  } else {
    want.push_back("mlx-community/Qwen3.5-0.8B-4bit");
    want.push_back("mlx-community/Qwen3.5-4B-4bit");
  }

  std::size_t measured = 0;
  for (const std::string& name : want) {
    if (name.empty()) continue;
    auto paths = resolve_model(name);
    if (!paths.ok()) continue;
    auto ckpt = paths->weights.ends_with(".index.json")
                    ? SafeTensors::open_sharded(paths->weights)
                    : SafeTensors::open(paths->weights);
    auto cfg = Config::from_json_file(paths->config);
    if (!ckpt.ok() || !cfg.ok()) continue;
    auto lm = build_model(*cfg, *ckpt);
    if (!lm.ok()) continue;
    WeightBinder binder(*ckpt, &cfg->quantization);
    if (!(*lm)->load(binder).ok()) continue;

    std::vector<MixerState> states = (*lm)->make_states();
    graph::Array tokens = graph::Array::zeros(Shape{1, 5}, DType::kF32);
    graph::Scheduler* sched = graph::default_scheduler();
    if (sched == nullptr) return;
    graph::Node& tn = *tokens.node();
    if (!graph::interpreter::ensure_output_buffer(tn, sched->backend()).ok()) {
      continue;
    }
    for (std::size_t i = 0; i < 5; ++i) {
      graph::interpreter::store_element(tn, i, static_cast<float>(3 + i));
    }
    tn.materialized = true;
    if (!graph::interpreter::sync_to_device(tn, sched->backend()).ok()) continue;
    auto h = (*lm)->hidden(tokens, &states, nullptr);
    LSE_EXPECT_OK(h.status());
    if (!h.ok()) continue;

    std::size_t paged = 0;
    std::size_t contiguous = 0;
    std::size_t blocks = 0;
    std::size_t layers = 0;
    for (const MixerState& st : states) {
      if (!st.paged.valid()) continue;
      ++layers;
      paged += st.paged.pool_bytes();
      blocks += static_cast<std::size_t>(st.paged.tables[0].size());
      const Shape& p = st.key_cache.shape();
      contiguous += static_cast<std::size_t>(p.dim(1) * p.dim(3)) *
                    sizeof(float) * 2 *
                    static_cast<std::size_t>(cfg->kv_capacity());
    }
    LSE_EXPECT(layers > 0u);
    if (layers == 0u) continue;
    std::printf(
        "       %-42s %2zu attn layer(s), %2zu block(s) held: "
        "%8.3f MiB paged vs %8.3f MiB contiguous (%.1fx)\n",
        name.c_str(), layers, blocks,
        static_cast<double>(paged) / 1048576.0,
        static_cast<double>(contiguous) / 1048576.0,
        static_cast<double>(contiguous) / static_cast<double>(paged));
    // One block per attention layer covers 5 tokens, and the pool is at the
    // smallest rung.
    LSE_EXPECT_EQ(blocks, layers);
    LSE_EXPECT(paged * 8 <= contiguous);
    ++measured;
  }
  std::printf("       measured %zu checkpoint(s)\n", measured);
}

namespace {

// Token ids for one prefill pass, staged on the device the way the generator
// stages them.
bool stage_tokens(std::int64_t width, std::int64_t first, std::int64_t vocab,
                  graph::Array& out) {
  graph::Scheduler* sched = graph::default_scheduler();
  if (sched == nullptr) return false;
  out = graph::Array::zeros(Shape{1, width}, DType::kF32);
  graph::Node& n = *out.node();
  if (!graph::interpreter::ensure_output_buffer(n, sched->backend()).ok()) {
    return false;
  }
  for (std::int64_t i = 0; i < width; ++i) {
    graph::interpreter::store_element(
        n, static_cast<std::size_t>(i),
        static_cast<float>((first * 7 + i * 13) % (vocab - 2) + 1));
  }
  n.materialized = true;
  return graph::interpreter::sync_to_device(n, sched->backend()).ok();
}

bool buffers_overlap(const graph::Node& a, const graph::Node& b) {
  if (!a.buffer.valid() || !b.buffer.valid()) return false;
  if (a.buffer.handle != b.buffer.handle || a.buffer.ptr != b.buffer.ptr) {
    return false;
  }
  const std::size_t as = a.buffer.offset, ae = as + a.buffer.size_bytes;
  const std::size_t bs = b.buffer.offset, be = bs + b.buffer.size_bytes;
  return as < be && bs < ae;
}

// Nodes this pass writes that share bytes with one of its carry endpoints.
std::size_t carry_aliases(const graph::Program& p) {
  std::size_t bad = 0;
  for (const graph::Program::Carry& c : p.carries()) {
    if (!c.in || !c.out) continue;
    for (const graph::FusionGroup& g : p.groups()) {
      for (const graph::NodePtr& n : g.nodes) {
        if (!n || n.get() == c.in.get() || n.get() == c.out.get()) continue;
        if (buffers_overlap(*n, *c.out) || buffers_overlap(*n, *c.in)) ++bad;
      }
    }
  }
  return bad;
}

}  // namespace

// The same prompt, same chunking, run once with the retained program replayed
// and once with every pass rebuilt. Passing a trace turns the reuse term in
// HybridLM::hidden off without touching the arithmetic, so the rebuild run is
// an oracle for the replay. Nothing compared the two paths before, which is
// how a carried state that shared bytes with its own pass shipped: it needs
// two consecutive same-width passes to show, i.e. a prompt past ~280 tokens.
LSE_TEST(long_prefill_reuse_matches_a_forced_rebuild) {
  if (graph::default_scheduler() == nullptr) return;
  auto paths = resolve_model("mlx-community/Qwen3.5-0.8B-4bit");
  if (!paths.ok()) return;
  auto ckpt = paths->weights.ends_with(".index.json")
                  ? SafeTensors::open_sharded(paths->weights)
                  : SafeTensors::open(paths->weights);
  auto cfg = Config::from_json_file(paths->config);
  if (!ckpt.ok() || !cfg.ok()) return;

  // Chunks of the engine's prefill width. The first builds; once the paged
  // pool stops growing into a bigger rung the rest replay it.
  constexpr std::int64_t kWidth = 128;
  constexpr int kPasses = 8;

  auto run = [&](bool reuse, std::vector<float>& out, int& replays) {
    auto lm = build_model(*cfg, *ckpt);
    LSE_EXPECT_OK(lm.status());
    if (!lm.ok()) return;
    WeightBinder binder(*ckpt, &cfg->quantization);
    LSE_EXPECT_OK((*lm)->load(binder));
    std::vector<MixerState> states = (*lm)->make_states();
    const graph::Node* previous = nullptr;
    for (int p = 0; p < kPasses; ++p) {
      graph::Array tokens;
      LSE_EXPECT(stage_tokens(kWidth, p * kWidth, 1000, tokens));
      std::vector<graph::Array> trace;
      auto h = (*lm)->hidden(tokens, &states, nullptr,
                             reuse ? nullptr : &trace);
      LSE_EXPECT_OK(h.status());
      if (!h.ok()) return;
      // A replayed pass answers with the node the retained program holds; a
      // rebuilt one answers with a fresh node. That is how this test knows it
      // is exercising the path it claims to.
      if (h->node().get() == previous) ++replays;
      previous = h->node().get();
      if (reuse) LSE_EXPECT_EQ(carry_aliases((*lm)->retained_program()), 0u);
      if (p + 1 < kPasses) continue;
      out.resize(static_cast<std::size_t>(h->shape().elem_count()));
      LSE_EXPECT_OK(h->to_host(out.data(), out.size() * sizeof(float)));
    }
  };

  std::vector<float> replayed, rebuilt;
  int replays = 0, rebuilds = 0;
  run(/*reuse=*/true, replayed, replays);
  run(/*reuse=*/false, rebuilt, rebuilds);
  std::printf("       %d of %d passes replayed the retained program\n",
              replays, kPasses);
  LSE_EXPECT(replays >= 2);
  LSE_EXPECT_EQ(rebuilds, 0);
  LSE_EXPECT(!replayed.empty());
  LSE_EXPECT_EQ(replayed.size(), rebuilt.size());
  if (replayed.size() != rebuilt.size() || replayed.empty()) return;

  double worst = 0.0, scale = 0.0;
  for (std::size_t i = 0; i < replayed.size(); ++i) {
    worst = std::max(worst, (double)std::fabs(replayed[i] - rebuilt[i]));
    scale = std::max(scale, (double)std::fabs(rebuilt[i]));
  }
  std::printf("       last-pass hidden: max |replay - rebuild| = %g over a "
              "range of %g\n", worst, scale);
  // Measured at exactly 0 on gfx1151: the replay runs the same kernels on the
  // same bytes. The tolerance is there only so a different reduction split on
  // another device is not a failure — a carried state read out of clobbered
  // bytes moves this by more than the whole range.
  LSE_EXPECT(worst <= 1e-5 * std::max(1.0, scale));
}

LSE_TEST_MAIN()
