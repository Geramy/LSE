// lemonseed's concrete layers (src/models/lemonseed.cpp).
#pragma once

#include <memory>

#include "lse/core/status.hpp"
#include "lse/graph/graph.hpp"
#include "lse/model/hybrid_lm.hpp"
#include "lse/model/layer.hpp"

namespace lse::model {

std::unique_ptr<IFeedForward> make_lemonseed_moe();
std::unique_ptr<IMixer> make_lemonseed_attention();
std::unique_ptr<IMixer> make_lemonseed_gdn();
std::unique_ptr<IModGate> make_lemonseed_mod();

// The whole model: every layer is 3x GDN then 1x gated attention, MoE FFN
// behind a MoD gate, zero-centered norms, head tied to the embedding.
std::unique_ptr<HybridLM> make_lemonseed(const Config& config);

// The MoE's two halves, exposed separately because MoD gates the routed
// experts and must leave the shared one alone.
Result<Array> lemonseed_moe_routed(const IFeedForward& moe, const Array& x);
Result<Array> lemonseed_moe_shared(const IFeedForward& moe, const Array& x);

}  // namespace lse::model
