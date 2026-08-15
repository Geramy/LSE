// Differential tests against lemonseed's own activations.
//
// These are the M6 gate: each C++ layer must reproduce what the Python model
// produced from the same weights. Both sides run fp32 from the same widened
// bf16 checkpoint, so a mismatch is algorithmic, not precision.
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "harness.hpp"
#include "lse/core/hash.hpp"
#include "lse/graph/interpreter.hpp"
#include "lse/graph/ops.hpp"
#include "lse/model/config.hpp"
#include "lse/model/hybrid_lm.hpp"
#include "lse/model/layer.hpp"
#include "lse/model/lemonseed.hpp"
#include "lse/model/weights.hpp"
#include "reference.hpp"

using namespace lse;
using namespace lse::graph;
using namespace lse::model;
using namespace lse::test;

namespace {

std::string model_dir() {
  if (const char* p = std::getenv("LSE_TEST_MODEL")) return p;
  const char* home = std::getenv("HOME");
  return home ? std::string(home) + "/Documents/Dev/LDE/model" : "";
}

bool have_model() {
  std::error_code ec;
  return !model_dir().empty() && std::filesystem::is_directory(model_dir(), ec);
}

// Builds an Array holding a widened copy of a checkpoint tensor.
Array from_checkpoint(const SafeTensors& st, const std::string& name) {
  const TensorView* v = st.find(name);
  if (v == nullptr) return Array{};
  Array a = Array::zeros(v->shape, DType::kF32);
  if (const Status s = a.eval(); !s.ok()) {
    std::printf("       from_checkpoint(%s) eval: %s\n", name.c_str(),
                s.to_string().c_str());
    return Array{};
  }
  std::vector<float> host(v->element_count());
  if (const Status s = v->read_f32(host.data(), host.size()); !s.ok()) {
    std::printf("       from_checkpoint(%s) read: %s\n", name.c_str(),
                s.to_string().c_str());
    return Array{};
  }
  for (std::size_t i = 0; i < host.size(); ++i) {
    interpreter::store_element(*a.node(), i, host[i]);
  }
  return a;
}

Array from_host(const std::vector<float>& values, Shape shape) {
  Array a = Array::zeros(shape, DType::kF32);
  if (!a.eval().ok()) return Array{};
  for (std::size_t i = 0; i < values.size(); ++i) {
    interpreter::store_element(*a.node(), i, values[i]);
  }
  return a;
}

// Bit-level fingerprint of a computed tensor. A refactor that preserves
// numerics preserves this exactly; anything else is a numerics change.
void report(const char* label, const std::vector<float>& got) {
  std::uint64_t h = kHashSeed;
  for (float v : got) h = hash_float(h, v);
  std::printf("       fnv %-12s %016llx\n", label,
              static_cast<unsigned long long>(h));
}

std::vector<float> drain(Array& a) {
  if (!a.eval().ok()) return {};
  std::vector<float> out(a.shape().elem_count());
  for (std::size_t i = 0; i < out.size(); ++i) {
    out[i] = interpreter::load_element(*a.node(), i);
  }
  return out;
}

}  // namespace

LSE_TEST(reference_data_is_present_and_sane) {
  if (!have_reference()) {
    std::printf("       (skipped: no %s — regenerate with "
                "scripts/dump_reference.py)\n", reference_path().c_str());
    return;
  }
  auto st = SafeTensors::open(reference_path());
  LSE_EXPECT(st.ok());
  if (!st.ok()) return;

  // embed, 20 blocks + 20 aux, final_norm, logits, tokens, and five
  // sub-layer probes per block.
  LSE_EXPECT(st->tensors().size() >= 145u);
  LSE_EXPECT(st->find("probe.b19.mixer") != nullptr);
  LSE_EXPECT(st->find("embed") != nullptr);
  LSE_EXPECT(st->find("block.19") != nullptr);
  LSE_EXPECT(st->find("logits_last") != nullptr);

  const auto embed = tensor_f32(*st, "embed");
  LSE_EXPECT(!embed.empty());
  bool finite = true;
  for (float v : embed) finite = finite && std::isfinite(v);
  LSE_EXPECT(finite);
}

LSE_TEST(embedding_matches_lemonseed) {
  if (!have_reference() || !have_model()) return;

  auto ref = SafeTensors::open(reference_path());
  auto paths = resolve_model(model_dir());
  if (!ref.ok() || !paths.ok()) return;
  auto ckpt = SafeTensors::open(paths->weights);
  if (!ckpt.ok()) return;

  const auto tokens = tensor_f32(*ref, "tokens");
  const auto want = tensor_f32(*ref, "embed");
  LSE_EXPECT(!tokens.empty());
  LSE_EXPECT(!want.empty());
  if (tokens.empty() || want.empty()) return;

  Array table = from_checkpoint(*ckpt, "embed.weight");
  LSE_EXPECT(table.valid());
  if (!table.valid()) return;

  Array ids = from_host(tokens, Shape{1, static_cast<std::int64_t>(tokens.size())});
  Array out = embedding(table, ids);
  const auto got = drain(out);
  report("embedding", got);

  const Deviation d = compare(got, want);
  LSE_EXPECT(d.count == want.size());
  LSE_EXPECT(d.all_finite);
  // A gather is exact: any difference at all means the wrong rows were read.
  LSE_EXPECT_NEAR(d.max_abs, 0.0, 1e-6);
}

LSE_TEST(rms_norm_matches_lemonseed_on_real_weights) {
  if (!have_reference() || !have_model()) return;

  auto ref = SafeTensors::open(reference_path());
  auto paths = resolve_model(model_dir());
  if (!ref.ok() || !paths.ok()) return;
  auto ckpt = SafeTensors::open(paths->weights);
  if (!ckpt.ok()) return;

  const auto x_host = tensor_f32(*ref, "embed");
  const auto want = tensor_f32(*ref, "probe.block0.norm1");
  if (x_host.empty() || want.empty()) return;

  auto cfg = Config::from_json_file(paths->config);
  if (!cfg.ok()) return;

  const std::int64_t seq =
      static_cast<std::int64_t>(x_host.size()) / cfg->hidden_size;
  Array x = from_host(x_host, Shape{1, seq, cfg->hidden_size});
  Array w = from_checkpoint(*ckpt, "blocks.0.norm1.weight");
  LSE_EXPECT(w.valid());
  if (!w.valid()) return;

  Array y = rms_norm(x, w, cfg->rms_eps, /*zero_centered=*/true);
  const auto got = drain(y);
  report("rms_norm", got);

  const Deviation d = compare(got, want);
  LSE_EXPECT(d.count == want.size());
  LSE_EXPECT(d.all_finite);
  if (d.max_rel > 1e-5) {
    std::printf("       rms_norm max_abs=%.3e max_rel=%.3e (ref absmax %.3f)\n",
                d.max_abs, d.max_rel, d.ref_absmax);
  }
  LSE_EXPECT(d.max_rel < 1e-5);
}

LSE_TEST(tied_lm_head_matches_lemonseed_logits) {
  // The head is a plain matmul against the tied embedding, so it exercises the
  // widest tensor in the model (248320 columns) end to end.
  if (!have_reference() || !have_model()) return;

  auto ref = SafeTensors::open(reference_path());
  auto paths = resolve_model(model_dir());
  if (!ref.ok() || !paths.ok()) return;
  auto ckpt = SafeTensors::open(paths->weights);
  if (!ckpt.ok()) return;

  const auto h_host = tensor_f32(*ref, "final_norm");
  const auto want = tensor_f32(*ref, "logits_last");
  if (h_host.empty() || want.empty()) return;

  auto cfg = Config::from_json_file(paths->config);
  if (!cfg.ok()) return;

  const std::int64_t d_model = cfg->hidden_size;
  const std::int64_t seq = static_cast<std::int64_t>(h_host.size()) / d_model;

  // Last position only, matching what the reference dumped.
  std::vector<float> last(h_host.end() - d_model, h_host.end());
  Array h = from_host(last, Shape{1, d_model});
  Array table = from_checkpoint(*ckpt, "embed.weight");
  if (!table.valid()) return;

  Array logits = linear(h, table);
  const auto got = drain(logits);
  report("lm_head", got);

  const Deviation d = compare(got, want);
  LSE_EXPECT(d.count == want.size());
  LSE_EXPECT(d.all_finite);
  std::printf("       logits max_abs=%.3e max_rel=%.3e over %zu values "
              "(seq %lld)\n", d.max_abs, d.max_rel, d.count,
              static_cast<long long>(seq));
  // fp32 dot over 1024 terms: accumulation order differs between the two
  // implementations, so match to fp32 epsilon scale rather than exactly.
  LSE_EXPECT(d.max_rel < 1e-5);
}

LSE_TEST(argmax_token_agrees_with_lemonseed) {
  // Even if the last bits differ, the sampled token must not.
  if (!have_reference()) return;
  auto ref = SafeTensors::open(reference_path());
  if (!ref.ok()) return;
  const auto want = tensor_f32(*ref, "logits_last");
  if (want.empty()) return;

  // Compare the engine's argmax against the reference's, both from the same
  // fixture. A recorded token id would break on every retrain of the
  // checkpoint while proving nothing about the engine — this has already
  // happened twice (710 -> 8).
  if (!have_model()) return;
  auto paths = resolve_model(model_dir());
  if (!paths.ok()) return;
  auto ckpt = SafeTensors::open(paths->weights);
  auto cfg = Config::from_json_file(paths->config);
  if (!ckpt.ok() || !cfg.ok()) return;

  const auto h_host = tensor_f32(*ref, "final_norm");
  if (h_host.empty()) return;
  const std::int64_t d_model = cfg->hidden_size;
  std::vector<float> last(h_host.end() - d_model, h_host.end());
  Array h = from_host(last, Shape{1, d_model});
  Array table = from_checkpoint(*ckpt, "embed.weight");
  if (!table.valid()) return;
  Array logits = linear(h, table);
  const auto got = drain(logits);
  LSE_EXPECT_EQ(got.size(), want.size());
  if (got.size() != want.size()) return;

  std::size_t want_best = 0;
  std::size_t got_best = 0;
  for (std::size_t i = 1; i < want.size(); ++i) {
    if (want[i] > want[want_best]) want_best = i;
    if (got[i] > got[got_best]) got_best = i;
  }
  std::printf("       argmax: engine=%zu reference=%zu\n", got_best, want_best);
  LSE_EXPECT_EQ(got_best, want_best);
}

namespace {

struct LayerFixture {
  SafeTensors ckpt;
  SafeTensors ref;
  Config cfg;
  bool ok = false;
};

bool load_fixture(LayerFixture& f) {
  if (!have_reference() || !have_model()) return false;
  auto paths = resolve_model(model_dir());
  if (!paths.ok()) return false;
  auto c = Config::from_json_file(paths->config);
  if (!c.ok()) return false;
  auto ck = SafeTensors::open(paths->weights);
  auto rf = SafeTensors::open(reference_path());
  if (!ck.ok() || !rf.ok()) return false;
  f.ckpt = ck.release();
  f.ref = rf.release();
  f.cfg = c.release();
  f.ok = true;
  return true;
}

Array ref_array(const SafeTensors& st, const std::string& name, const Config& cfg) {
  const auto host = tensor_f32(st, name);
  if (host.empty()) return Array{};
  const std::int64_t seq =
      static_cast<std::int64_t>(host.size()) / cfg.hidden_size;
  return from_host(host, Shape{1, seq, cfg.hidden_size});
}

}  // namespace

LSE_TEST(moe_shared_expert_matches_lemonseed) {
  LayerFixture f;
  if (!load_fixture(f)) return;

  WeightBinder binder(f.ckpt);
  LayerContext ctx{&f.cfg, 0, false};
  auto moe = make_lemonseed_moe();
  auto s = moe->load(binder, "blocks.19", ctx);
  LSE_EXPECT_OK(s);
  if (!s.ok()) return;

  Array h2 = ref_array(f.ref, "probe.block19.norm2", f.cfg);
  if (!h2.valid()) return;

  auto got = lemonseed_moe_shared(*moe, h2);
  LSE_EXPECT(got.ok());
  if (!got.ok()) return;
  auto vals = drain(*got);
  report("moe_shared", vals);
  const auto want = tensor_f32(f.ref, "probe.block19.moe_shared");

  const Deviation d = compare(vals, want);
  LSE_EXPECT(d.all_finite);
  std::printf("       moe_shared  max_rel=%.3e\n", d.max_rel);
  LSE_EXPECT(d.max_rel < 1e-4);
}

LSE_TEST(moe_routed_experts_match_lemonseed) {
  // Layer 19: the only block where all 8 routed experts are trained. Layers
  // 0/1/9 have ~1e-12 routed weights, so their reference output is legitimately
  // zero and validates nothing.
  LayerFixture f;
  if (!load_fixture(f)) return;

  WeightBinder binder(f.ckpt);
  LayerContext ctx{&f.cfg, 0, false};
  auto moe = make_lemonseed_moe();
  if (!moe->load(binder, "blocks.19", ctx).ok()) return;

  Array h2 = ref_array(f.ref, "probe.block19.norm2", f.cfg);
  if (!h2.valid()) return;

  auto got = lemonseed_moe_routed(*moe, h2);
  LSE_EXPECT(got.ok());
  if (!got.ok()) {
    std::printf("       %s\n", got.status().to_string().c_str());
    return;
  }
  auto vals = drain(*got);
  report("moe_routed", vals);
  const auto want = tensor_f32(f.ref, "probe.block19.moe_routed");

  const Deviation d = compare(vals, want);
  LSE_EXPECT(d.all_finite);
  std::printf("       moe_routed  max_rel=%.3e (ref absmax %.4f)\n",
              d.max_rel, d.ref_absmax);
  LSE_EXPECT(d.max_rel < 1e-4);
}

LSE_TEST(mod_gate_matches_lemonseed_inference_path) {
  // gate_all: per-token sigmoid on EVERY position, no top-k.
  LayerFixture f;
  if (!load_fixture(f)) return;

  WeightBinder binder(f.ckpt);
  auto mod = make_lemonseed_mod();
  auto s = mod->load(binder, "blocks.0");
  LSE_EXPECT_OK(s);
  if (!s.ok()) return;

  Array h2 = ref_array(f.ref, "probe.block0.norm2", f.cfg);
  Array routed = ref_array(f.ref, "probe.block0.moe_routed", f.cfg);
  if (!h2.valid() || !routed.valid()) return;

  auto got = mod->gate_all(h2, routed);
  LSE_EXPECT(got.ok());
  if (!got.ok()) return;
  auto vals = drain(*got);
  report("mod_gated", vals);
  const auto want = tensor_f32(f.ref, "probe.block0.mod_gated");

  const Deviation d = compare(vals, want);
  LSE_EXPECT(d.all_finite);
  std::printf("       mod_gated   max_rel=%.3e\n", d.max_rel);
  LSE_EXPECT(d.max_rel < 1e-4);
}

namespace {

// Both mixers take the block's norm1 output and are compared against the
// mixer probe dumped from the same position in the Python model.
void check_mixer(std::unique_ptr<IMixer> mixer, std::int32_t layer,
                 const char* label, const std::string& probe) {
  LayerFixture f;
  if (!load_fixture(f)) return;

  const std::string block = "blocks." + std::to_string(layer);
  WeightBinder binder(f.ckpt);
  LayerContext ctx{&f.cfg, layer, false};
  auto s = mixer->load(binder, block, ctx);
  LSE_EXPECT_OK(s);
  if (!s.ok()) return;

  Array h = ref_array(f.ref, "probe.block" + std::to_string(layer) + ".norm1",
                      f.cfg);
  if (!h.valid()) return;

  auto got = mixer->forward(h, nullptr, ctx);
  LSE_EXPECT(got.ok());
  if (!got.ok()) {
    std::printf("       %s\n", got.status().to_string().c_str());
    return;
  }
  auto vals = drain(*got);
  report(label, vals);
  const auto want = tensor_f32(f.ref, probe);

  const Deviation d = compare(vals, want);
  LSE_EXPECT(d.count == want.size());
  LSE_EXPECT(d.all_finite);
  std::printf("       %-11s max_abs=%.3e max_rel=%.3e (ref absmax %.4f)\n",
              label, d.max_abs, d.max_rel, d.ref_absmax);
  LSE_EXPECT(d.max_rel < 1e-4);
}

}  // namespace

LSE_TEST(gated_delta_net_matches_lemonseed) {
  // Block 0 is a GDN block; 8 tokens fit in one chunk, so the reference takes
  // the exact per-token recurrence — the same path this implements.
  check_mixer(make_lemonseed_gdn(), 0, "mixer_gdn", "probe.block0.mixer_gdn");
}

LSE_TEST(gated_attention_matches_lemonseed) {
  // Block 3 is a global attention layer: full causal mask, no sliding window.
  check_mixer(make_lemonseed_attention(), 3, "mixer_attn",
              "probe.block3.mixer_attn");
}

LSE_TEST(weight_binder_reports_unclaimed_tensors) {
  LayerFixture f;
  if (!load_fixture(f)) return;
  WeightBinder binder(f.ckpt);
  LayerContext ctx{&f.cfg, 0, false};
  auto moe = make_lemonseed_moe();
  if (!moe->load(binder, "blocks.19", ctx).ok()) return;
  // MoE claims router + 3 stacked + expert_bias + 3 shared = 8.
  LSE_EXPECT(binder.claimed_count() >= 7u);
  LSE_EXPECT(!binder.unclaimed().empty());
}

LSE_TEST(full_stack_matches_lemonseed_at_every_block) {
  // The only test that runs the real stack. Per-block comparison is the point:
  // a single logits check says "something is wrong", this says which layer.
  if (!have_reference() || !have_model()) return;

  auto ref = SafeTensors::open(reference_path());
  auto paths = resolve_model(model_dir());
  if (!ref.ok() || !paths.ok()) return;
  auto ckpt = SafeTensors::open(paths->weights);
  auto cfg = Config::from_json_file(paths->config);
  if (!ckpt.ok() || !cfg.ok()) return;

  const auto tokens = tensor_f32(*ref, "tokens");
  if (tokens.empty()) return;

  auto model = make_lemonseed(*cfg);
  WeightBinder binder(*ckpt);
  const Status loaded = model->load(binder);
  LSE_EXPECT(loaded.ok());
  if (!loaded.ok()) {
    std::printf("       load: %s\n", loaded.to_string().c_str());
    return;
  }

  Array ids = from_host(tokens, Shape{1, static_cast<std::int64_t>(tokens.size())});
  std::vector<Array> trace;
  auto h = model->hidden(ids, nullptr, nullptr, &trace);
  LSE_EXPECT(h.ok());
  if (!h.ok()) {
    std::printf("       hidden: %s\n", h.status().to_string().c_str());
    return;
  }

  LSE_EXPECT(trace.size() == static_cast<std::size_t>(cfg->num_layers));
  double worst = 0.0;
  std::size_t worst_layer = 0;
  for (std::size_t i = 0; i < trace.size(); ++i) {
    const auto want = tensor_f32(*ref, "block." + std::to_string(i));
    if (want.empty()) continue;
    const auto got = drain(trace[i]);
    const Deviation d = compare(got, want);
    LSE_EXPECT(d.count == want.size());
    LSE_EXPECT(d.all_finite);
    if (d.max_rel > worst) {
      worst = d.max_rel;
      worst_layer = i;
    }
    // Reported per layer only when it is the first to break, so a genuine
    // divergence is visible without 20 lines of noise on a passing run.
    if (d.max_rel >= 1e-5 || std::getenv("LSE_TRACE_BLOCKS") != nullptr) {
      std::printf("       block %2zu max_abs=%.3e max_rel=%.3e\n", i, d.max_abs,
                  d.max_rel);
    }
  }
  std::printf("       worst block: %zu max_rel=%.3e\n", worst_layer, worst);
  LSE_EXPECT(worst < 1e-5);

  Array hidden = h.release();
  const auto got_norm = drain(hidden);
  report("stack_norm", got_norm);
  const Deviation dn = compare(got_norm, tensor_f32(*ref, "final_norm"));
  std::printf("       final_norm  max_abs=%.3e max_rel=%.3e\n", dn.max_abs,
              dn.max_rel);
  LSE_EXPECT(dn.all_finite);
  LSE_EXPECT(dn.max_rel < 1e-5);

  // Head on the last position only, matching the fixture.
  const std::int64_t d_model = cfg->hidden_size;
  std::vector<float> last(got_norm.end() - d_model, got_norm.end());
  Array last_h = from_host(last, Shape{1, d_model});
  auto logits = model->lm_head(last_h);
  LSE_EXPECT(logits.ok());
  if (!logits.ok()) return;

  Array lg = logits.release();
  const auto got_logits = drain(lg);
  const auto want_logits = tensor_f32(*ref, "logits_last");
  const Deviation dl = compare(got_logits, want_logits);
  std::printf("       stack logits max_abs=%.3e max_rel=%.3e\n", dl.max_abs,
              dl.max_rel);
  LSE_EXPECT(dl.all_finite);
  LSE_EXPECT(dl.max_rel < 1e-5);

  const auto arg = [](const std::vector<float>& v) {
    return static_cast<std::size_t>(
        std::max_element(v.begin(), v.end()) - v.begin());
  };
  std::printf("       stack argmax: engine=%zu reference=%zu\n",
              arg(got_logits), arg(want_logits));
  LSE_EXPECT(arg(got_logits) == arg(want_logits));
}

LSE_TEST(each_block_matches_from_the_reference_input) {
  // Complements the full-stack test: every block is fed the reference's own
  // input for that block, so a deviation here is that block's own bug rather
  // than drift inherited from the blocks before it.
  if (!have_reference() || !have_model()) return;

  auto ref = SafeTensors::open(reference_path());
  auto paths = resolve_model(model_dir());
  if (!ref.ok() || !paths.ok()) return;
  auto ckpt = SafeTensors::open(paths->weights);
  auto cfg = Config::from_json_file(paths->config);
  if (!ckpt.ok() || !cfg.ok()) return;

  auto model = make_lemonseed(*cfg);
  WeightBinder binder(*ckpt);
  if (!model->load(binder).ok()) return;

  const std::int64_t d_model = cfg->hidden_size;
  double worst = 0.0;
  std::size_t worst_layer = 0;
  for (std::int32_t i = 0; i < cfg->num_layers; ++i) {
    const auto in_host =
        i == 0 ? tensor_f32(*ref, "embed")
               : tensor_f32(*ref, "block." + std::to_string(i - 1));
    const auto want = tensor_f32(*ref, "block." + std::to_string(i));
    if (in_host.empty() || want.empty()) continue;

    const std::int64_t seq = static_cast<std::int64_t>(in_host.size()) / d_model;
    Array x = from_host(in_host, Shape{1, seq, d_model});
    LayerContext ctx{&*cfg, i, false};
    auto y = model->block(static_cast<std::size_t>(i))
                 .forward(x, nullptr, nullptr, ctx);
    LSE_EXPECT(y.ok());
    if (!y.ok()) {
      std::printf("       block %d: %s\n", i, y.status().to_string().c_str());
      continue;
    }
    Array out = y.release();
    const Deviation d = compare(drain(out), want);
    LSE_EXPECT(d.all_finite);
    if (d.max_rel > worst) {
      worst = d.max_rel;
      worst_layer = static_cast<std::size_t>(i);
    }
    if (d.max_rel >= 1e-5) {
      std::printf("       isolated block %2d max_abs=%.3e max_rel=%.3e "
                  "(%s)\n", i, d.max_abs, d.max_rel,
                  cfg->is_attention_layer(i) ? "attn" : "gdn");
    }
  }
  std::printf("       worst isolated block: %zu max_rel=%.3e\n", worst_layer,
              worst);
  LSE_EXPECT(worst < 1e-5);
}

LSE_TEST(every_sub_layer_matches_along_the_real_stack_path) {
  // Finest granularity: mixer, routed experts, shared expert and MoD gate for
  // all 20 blocks, each fed the input the real forward pass produces. When a
  // block-level test fails, this says which of the four pieces moved.
  if (!have_reference() || !have_model()) return;

  auto ref = SafeTensors::open(reference_path());
  auto paths = resolve_model(model_dir());
  if (!ref.ok() || !paths.ok()) return;
  auto ckpt = SafeTensors::open(paths->weights);
  auto cfg = Config::from_json_file(paths->config);
  if (!ckpt.ok() || !cfg.ok()) return;
  if (ref->find("probe.b0.mixer") == nullptr) {
    std::printf("       (skipped: fixture predates per-block probes)\n");
    return;
  }

  auto model = make_lemonseed(*cfg);
  WeightBinder binder(*ckpt);
  if (!model->load(binder).ok()) return;

  const std::int64_t d_model = cfg->hidden_size;
  double worst = 0.0;
  std::string worst_name;
  for (std::int32_t i = 0; i < cfg->num_layers; ++i) {
    const std::string b = "probe.b" + std::to_string(i) + ".";
    const auto in_host = i == 0 ? tensor_f32(*ref, "embed")
                                : tensor_f32(*ref, "block." + std::to_string(i - 1));
    if (in_host.empty()) continue;
    const std::int64_t seq = static_cast<std::int64_t>(in_host.size()) / d_model;

    // The block's own pieces, driven from the reference input for this layer.
    Array n2 = from_host(tensor_f32(*ref, b + "norm2"), Shape{1, seq, d_model});

    HybridBlock& block = model->block(static_cast<std::size_t>(i));
    LayerContext ctx{&*cfg, i, false};
    struct Case { const char* suffix; std::vector<float> got; };
    std::vector<Case> cases;

    Array n1 = from_host(tensor_f32(*ref, b + "norm1"), Shape{1, seq, d_model});
    auto mixed = block.mixer().forward(n1, nullptr, ctx);
    LSE_EXPECT(mixed.ok());
    if (!mixed.ok()) {
      std::printf("       %smixer: %s\n", b.c_str(),
                  mixed.status().to_string().c_str());
      continue;
    }
    Array mixed_a = mixed.release();
    cases.push_back({"mixer", drain(mixed_a)});

    auto routed = lemonseed_moe_routed(block.ffn(), n2);
    auto shared = lemonseed_moe_shared(block.ffn(), n2);
    LSE_EXPECT(routed.ok() && shared.ok());
    if (!routed.ok() || !shared.ok()) continue;
    Array routed_a = routed.release();
    Array shared_a = shared.release();
    cases.push_back({"routed", drain(routed_a)});
    cases.push_back({"shared", drain(shared_a)});

    for (const auto& c : cases) {
      const auto want = tensor_f32(*ref, b + c.suffix);
      if (want.empty()) continue;
      const Deviation d = compare(c.got, want);
      LSE_EXPECT(d.all_finite);
      if (d.max_rel > worst) {
        worst = d.max_rel;
        worst_name = b + c.suffix;
      }
      if (d.max_rel >= 1e-5) {
        std::printf("       %-20s max_abs=%.3e max_rel=%.3e\n",
                    (b + c.suffix).c_str(), d.max_abs, d.max_rel);
      }
    }
  }
  std::printf("       worst sub-layer: %s max_rel=%.3e\n", worst_name.c_str(),
              worst);
  LSE_EXPECT(worst < 1e-5);
}

LSE_TEST(a_whole_block_collapses_into_few_fusion_groups) {
  // The payoff check for lazy evaluation: a block is hundreds of graph nodes,
  // and almost all of them must disappear into a neighbour's kernel. What is
  // left over is the hand-written-kernel budget — every barrier group is a
  // kernel somebody has to supply.
  if (!have_reference() || !have_model()) return;

  auto paths = resolve_model(model_dir());
  if (!paths.ok()) return;
  auto ckpt = SafeTensors::open(paths->weights);
  auto cfg = Config::from_json_file(paths->config);
  if (!ckpt.ok() || !cfg.ok()) return;

  auto model = make_lemonseed(*cfg);
  WeightBinder binder(*ckpt);
  if (!model->load(binder).ok()) return;

  const std::int64_t d_model = cfg->hidden_size;
  Array x = Array::zeros(Shape{1, 8, d_model}, DType::kF32);
  if (!x.eval().ok()) return;

  LayerContext ctx{&*cfg, 0, false};
  auto y = model->block(0).forward(x, nullptr, nullptr, ctx);
  LSE_EXPECT(y.ok());
  if (!y.ok()) return;

  // Partition without evaluating: this is the lazy graph as built.
  Array out = y.release();
  const NodePtr roots[] = {out.node()};
  const auto groups = Partitioner::partition(roots);

  std::size_t nodes = 0;
  std::size_t barriers = 0;
  std::size_t barrier_nodes = 0;
  std::map<std::string, std::size_t> barrier_kinds;
  std::map<std::string, std::size_t> anchors;
  for (const FusionGroup& g : groups) {
    nodes += g.nodes.size();
    anchors[std::string(to_string(g.anchor))]++;
    if (g.anchor_class == FusionClass::kBarrier) {
      ++barriers;
      barrier_nodes += g.nodes.size();
      barrier_kinds[std::string(to_string(g.anchor))]++;
    }
  }
  for (const auto& [kind, n] : anchors) {
    std::printf("         anchor %-16s x%zu\n", kind.c_str(), n);
  }
  std::printf("       generated groups: %zu holding %zu nodes\n",
              groups.size() - barriers, nodes - barrier_nodes);

  std::printf("       block 0: %zu nodes -> %zu groups (%zu barrier)\n", nodes,
              groups.size(), barriers);
  for (const auto& [kind, n] : barrier_kinds) {
    std::printf("         barrier %-14s x%zu\n", kind.c_str(), n);
  }

  // Mixer + on-device MoE top-k. A new name here is a kernel somebody still
  // has to write. test_jit measures epilogue fusion into those barriers.
  std::printf("       distinct barrier kinds: %zu (over %zu barrier groups)\n",
              barrier_kinds.size(), barriers);
  LSE_EXPECT(barrier_kinds.size() <= 5u);
  LSE_EXPECT(nodes > groups.size());

  // Elementwise work must not be one group per node. Barriers legitimately get
  // their own group, so the ratio is measured over what is left.
  const std::size_t generated = groups.size() - barriers;
  const double per_generated =
      generated == 0 ? 0.0
                     : static_cast<double>(nodes - barrier_nodes) /
                           static_cast<double>(generated);
  std::printf("       %.1f nodes per generated kernel\n", per_generated);
  LSE_EXPECT(per_generated > 1.0);
}

LSE_TEST(an_elementwise_chain_becomes_one_kernel) {
  // The direct fusion check, independent of any model: a chain of elementwise
  // ops must collapse to a single group no matter how long it is.
  Array x = Array::full(Shape{2048}, DType::kF32, 0.5f);
  Array y = x;
  Array half = Array::full(Shape{1}, DType::kF32, 0.5f);
  for (int i = 0; i < 12; ++i) y = graph::silu(y * y + x) * half;

  const NodePtr roots[] = {y.node()};
  const auto groups = Partitioner::partition(roots);

  std::size_t nodes = 0;
  for (const FusionGroup& g : groups) nodes += g.nodes.size();
  std::printf("       %zu elementwise nodes -> %zu group(s)\n", nodes,
              groups.size());
  // Two, not one: the first iteration reads `x` both as the squared value and
  // as the addend, and that fan-out splits once before the chain settles.
  // Everything after it stays in a single kernel.
  LSE_EXPECT(groups.size() <= 2u);
  LSE_EXPECT(nodes >= 36u);
  std::size_t largest = 0;
  for (const FusionGroup& g : groups) largest = std::max(largest, g.nodes.size());
  LSE_EXPECT(largest >= nodes - 6);
}

int main() {
  // f32 lemonseed dumps. WMMA multiplies in f16 and would miss 1e-5.
  ::setenv("LSE_WMMA", "0", 1);
  return ::lse::test::run_all();
}
