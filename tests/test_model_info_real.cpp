// The memory plan for real checkpoints against what the engine allocates when
// it loads them on the host backend: the weights the binder places, the draft
// module, and the paged KV pools grown to several contexts. Nothing here runs
// a forward pass, so activation and program workspace are not measured; every
// other component of the plan is.
//
// Gated: it maps and copies a whole model into RAM. Point it at checkpoints:
//
//   LSE_BACKEND=cpu \
//   LSE_TEST_MODEL_TARGET=.../qwen38-27b-q4 \
//   LSE_TEST_MODEL_DFLASH2=.../qwen38-27b-dflash2-q8 \   (or)
//   LSE_TEST_MODEL_MTP=.../qwen38-27b-mtp-q8 \
//   build/tests/test_model_info_real
//
// With both draft variables set it measures the DFlash2 draft; the MTP case
// needs its own process because weight slabs live for the process.
#include "harness.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>

#include <nlohmann/json.hpp>

#include "lse/backend/backend.hpp"
#include "lse/graph/graph.hpp"
#include "lse/kv/sizing.hpp"
#include "lse/model/config.hpp"
#include "lse/model/dflash2.hpp"
#include "lse/model/inspect.hpp"
#include "lse/model/mtp.hpp"
#include "lse/model/registry.hpp"
#include "lse/model/weights.hpp"
#include "lse/ops/attention.hpp"

using namespace lse;
using json = nlohmann::json;

namespace {

std::string env(const char* name) {
  const char* v = std::getenv(name);
  return v != nullptr ? v : "";
}

std::uint64_t device_live() {
  return backend::allocation_totals(backend::MemoryClass::kDevice).live;
}

// Within `percent` of the plan, printed either way so a run shows the margin.
void close_to(const char* what, std::uint64_t measured, std::uint64_t planned,
              double percent) {
  const double diff = planned == 0 ? (measured == 0 ? 0.0 : 100.0)
                                   : 100.0 * (static_cast<double>(measured) -
                                              static_cast<double>(planned)) /
                                         static_cast<double>(planned);
  std::printf("       %-28s measured %14llu  planned %14llu  %+.3f%%\n", what,
              static_cast<unsigned long long>(measured),
              static_cast<unsigned long long>(planned), diff);
  LSE_EXPECT(diff <= percent && diff >= -percent);
}

}  // namespace

LSE_TEST(plan_matches_a_host_load_of_real_checkpoints) {
  const std::string target = env("LSE_TEST_MODEL_TARGET");
  const std::string dflash2 = env("LSE_TEST_MODEL_DFLASH2");
  const std::string mtp = dflash2.empty() ? env("LSE_TEST_MODEL_MTP") : "";
  if (target.empty()) {
    lse::test::skip("set LSE_TEST_MODEL_TARGET (and a draft) to run");
    return;
  }
  auto* sched = graph::default_scheduler();
  LSE_EXPECT(sched != nullptr);
  if (sched == nullptr) return;

  model::MemoryPlanRequest r;
  r.model = target;
  r.kv_len = 262100;
  r.device_arch = "cpu";      // the host backend takes no packed Q8 copies
  r.fragmented_kv = false;    // nor Loom fragments
  if (!dflash2.empty()) {
    r.draft = model::DraftKind::kDFlash2;
    r.draft_path = dflash2;
  } else if (!mtp.empty()) {
    r.draft = model::DraftKind::kMtp;
    r.draft_path = mtp;
  }
  auto planned = model::estimate_memory(r);
  LSE_EXPECT_OK(planned.status());
  if (!planned.ok()) return;
  const json& plan = *planned;

  auto paths = model::resolve_model(target);
  LSE_EXPECT_OK(paths.status());
  if (!paths.ok()) return;
  auto config = model::Config::from_json_file(paths->config);
  auto weights = paths->weights.ends_with(".index.json")
                     ? model::SafeTensors::open_sharded(paths->weights)
                     : model::SafeTensors::open(paths->weights);
  LSE_EXPECT_OK(config.status());
  LSE_EXPECT_OK(weights.status());
  if (!config.ok() || !weights.ok()) return;
  config->kv_length = r.kv_len;
  auto lm = model::build_model(*config, *weights);
  LSE_EXPECT_OK(lm.status());
  if (!lm.ok()) return;

  const std::uint64_t before = device_live();
  model::WeightBinder binder(*weights, &config->quantization);
  LSE_EXPECT_OK((*lm)->load(binder));
  close_to("target weights + RoPE", device_live() - before,
           plan["weights_bytes"].get<std::uint64_t>() +
               plan["workspace"]["rope_bytes"].get<std::uint64_t>(),
           1.0);

  if (!dflash2.empty() || !mtp.empty()) {
    const std::uint64_t middle = device_live();
    if (!dflash2.empty()) {
      auto opened = model::DFlash2Module::open(dflash2, *config, **lm);
      LSE_EXPECT_OK(opened.status());
      if (opened.ok())
        close_to("DFlash2 draft", device_live() - middle,
                 plan["draft_bytes"].get<std::uint64_t>(), 1.0);
    } else {
      auto opened = model::MtpModule::open(mtp, *config, **lm);
      LSE_EXPECT_OK(opened.status());
      // The module's KV grows with use; its weights are what loading takes.
      if (opened.ok())
        close_to("MTP module weights", device_live() - middle,
                 plan["draft"]["weights_bytes"].get<std::uint64_t>(), 1.0);
    }
  }

  // KV: one layer grown through several contexts by the paged allocator, times
  // the model's KV layers, against the plan at each context.
  for (const std::int32_t context : {1321, 14000, 68301}) {
    model::MemoryPlanRequest at = r;
    at.draft = model::DraftKind::kNone;
    at.context_tokens = context;
    auto e = model::estimate_memory(at);
    LSE_EXPECT_OK(e.status());
    if (!e.ok()) return;
    ops::PagedKvLayer layer;
    const std::uint64_t start = device_live();
    LSE_EXPECT_OK(ops::ensure_paged_kv(layer, 1, context, r.kv_len, config->attn_kv_heads,
                                       config->attn_head_dim, config->kv_cache_dtype));
    const auto layers = static_cast<std::uint64_t>((*e)["kv"]["layers"].get<int>());
    char what[64];
    std::snprintf(what, sizeof what, "KV at %d tokens", context);
    close_to(what, (device_live() - start) * layers, (*e)["kv_bytes"].get<std::uint64_t>(), 1.0);
  }
}

LSE_TEST_MAIN()
