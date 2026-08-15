// Which model kernel a checkpoint needs.
//
// A checkpoint does not reliably say what it is: lemonseed's config carries no
// architecture field, and HF configs disagree on the name even between
// releases of one family. So an architecture is identified by what it *has* —
// the tensors it must contain — and each model kernel registers its own test
// next to its implementation. Adding a model is one file; nothing here changes.
#pragma once

#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "lse/core/status.hpp"
#include "lse/model/config.hpp"
#include "lse/model/hybrid_lm.hpp"
#include "lse/model/weights.hpp"

namespace lse::model {

struct ModelArch {
  std::string_view name;
  // Higher wins when several match. Use it only to separate a specialization
  // from the general case it overlaps, not to paper over a weak test.
  int priority = 0;
  // Must be cheap: it runs for every registered architecture on every load.
  std::function<bool(const Config&, const SafeTensors&)> matches;
  std::function<std::unique_ptr<HybridLM>(const Config&)> build;
};

void register_model_arch(ModelArch arch);

// Names of every registered architecture, for diagnostics and --list-models.
[[nodiscard]] std::vector<std::string_view> registered_architectures();

// The architecture whose test the checkpoint passes. kNotFound names what was
// tried, because "unsupported model" with no detail is the worst failure a
// loader can produce.
Result<const ModelArch*> detect_architecture(const Config& config,
                                             const SafeTensors& weights);

// detect_architecture, then build. `override_name` skips detection.
Result<std::unique_ptr<HybridLM>> build_model(const Config& config,
                                              const SafeTensors& weights,
                                              std::string_view override_name = {});

}  // namespace lse::model

#define LSE_REGISTER_MODEL(...)                                    \
  namespace {                                                      \
  const int lse_model_registrar = [] {                             \
    ::lse::model::register_model_arch(__VA_ARGS__);                \
    return 0;                                                      \
  }();                                                             \
  }
