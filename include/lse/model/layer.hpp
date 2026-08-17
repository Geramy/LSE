// Layer contract.
//
// Layers are composed, not inherited into a single blessed block. lemonseed and
// Qwen3.6 share a skeleton (3x linear-attention, 1x gated attention, MoE FFN)
// but differ in almost every internal: fused vs split QKV, gate position,
// Mixture-of-Depths present or absent, tied or untied head. A HybridBlock
// therefore holds a mixer and an FFN chosen per model rather than assuming one
// implementation covers both.
#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "lse/core/status.hpp"
#include "lse/graph/graph.hpp"
#include "lse/model/config.hpp"
#include "lse/model/weights.hpp"

namespace lse::model {

using graph::Array;

// Per-sequence mixer state: GDN keeps a fixed recurrent state, attention keeps
// a KV pair allocated at the engine length. Null on a prefill with no cache.
struct MixerState {
  Array gdn_state;
  // The GDN conv tails travel with it; see ops::GatedDeltaNetState.
  Array gdn_conv_q, gdn_conv_k, gdn_conv_v, gdn_conv_qkv;
  Array key_cache;
  Array value_cache;
  // Shared 1-element write cursor the decode program pokes each step.
  Array kv_pos;
  std::int32_t position = 0;

  [[nodiscard]] bool empty() const noexcept {
    return !gdn_state.valid() && !key_cache.valid();
  }
};

struct LayerContext {
  const Config* config = nullptr;
  std::int32_t layer_index = 0;
  bool training = false;
};

// Names bound so far, used to report anything the checkpoint had that no layer
// claimed — the most likely silent failure when adding a second architecture.
class WeightBinder {
 public:
  explicit WeightBinder(const SafeTensors& weights) : weights_(&weights) {}

  Result<Array> require(std::string_view name);
  Result<Array> optional(std::string_view name);

  // Reads `name` keeping only the rows in `order`, in that order, and reports
  // the result as `shape`. A row is one span of the tensor's last axis, so a
  // rank-1 tensor has one element per row.
  //
  // This is where a checkpoint's channel layout is converted to the engine's,
  // once at load: reordering weights costs nothing per token, while the slices
  // and transposes that would express the same permutation in the graph are
  // paid on every forward. Both uses today are Qwen3.5's — de-interleaving the
  // attention gate out of q_proj, and splitting a fused gate_up expert.
  Result<Array> require_rows(std::string_view name,
                             const std::vector<std::int64_t>& order,
                             Shape shape);

  [[nodiscard]] std::vector<std::string> unclaimed() const;
  [[nodiscard]] std::size_t claimed_count() const noexcept { return claimed_.size(); }

 private:
  const SafeTensors* weights_;
  std::vector<std::string> claimed_;
};

template <typename Derived>
class Layer {
 public:
  Status load(WeightBinder& binder, std::string_view prefix,
              const LayerContext& ctx) {
    return derived().load_impl(binder, prefix, ctx);
  }

  Result<Array> forward(const Array& x, MixerState* state,
                        const LayerContext& ctx) {
    return derived().forward_impl(x, state, ctx);
  }

  [[nodiscard]] std::string_view name() const noexcept { return Derived::kName; }

 protected:
  Layer() = default;

 private:
  Derived& derived() noexcept { return static_cast<Derived&>(*this); }
};

// Virtual seam so a block can hold whichever mixer/FFN a model needs.
class IMixer {
 public:
  virtual ~IMixer() = default;
  virtual Status load(WeightBinder&, std::string_view prefix,
                      const LayerContext&) = 0;
  virtual Result<Array> forward(const Array& x, MixerState* state,
                                const LayerContext&) = 0;
  virtual std::string_view name() const noexcept = 0;
};

class IFeedForward {
 public:
  virtual ~IFeedForward() = default;
  virtual Status load(WeightBinder&, std::string_view prefix,
                      const LayerContext&) = 0;
  // aux_loss accumulates router/MoD regularizers; null outside training.
  virtual Result<Array> forward(const Array& x, Array* aux_loss,
                                const LayerContext&) = 0;
  // The part of the output a MoD gate must not touch — lemonseed's shared
  // expert. Invalid Array when the FFN has no such part.
  virtual Result<Array> ungated(const Array& x) const {
    (void)x;
    return Array{};
  }
  virtual std::string_view name() const noexcept = 0;
};

class IModGate {
 public:
  virtual ~IModGate() = default;
  virtual Status load(WeightBinder&, std::string_view prefix) = 0;
  virtual Result<Array> gate_all(const Array& h, const Array& routed) const = 0;
};

template <typename Derived>
class MixerAdapter final : public IMixer {
 public:
  Status load(WeightBinder& b, std::string_view p, const LayerContext& c) override {
    return impl_.load(b, p, c);
  }
  Result<Array> forward(const Array& x, MixerState* s,
                        const LayerContext& c) override {
    return impl_.forward(x, s, c);
  }
  std::string_view name() const noexcept override { return Derived::kName; }
  Derived& impl() noexcept { return impl_; }

 private:
  Derived impl_;
};

// Suffixes of the two block norms, appended to the block prefix. lemonseed
// names them norm1/norm2, Qwen input_layernorm/post_attention_layernorm.
struct HybridBlockSpec {
  std::string norm1_name = ".norm1.weight";
  std::string norm2_name = ".norm2.weight";
};

// norm1 -> mixer -> residual -> norm2 -> ffn -> residual.
// `mod` is optional: lemonseed routes only a token subset through the FFN,
// Qwen3.5 has no MoD at all. It gates IFeedForward::forward only; whatever
// IFeedForward::ungated returns is added afterwards.
class HybridBlock {
 public:
  HybridBlock(std::unique_ptr<IMixer> mixer, std::unique_ptr<IFeedForward> ffn,
              bool zero_centered_norm,
              std::unique_ptr<IModGate> mod = nullptr,
              HybridBlockSpec spec = {})
      : mixer_(std::move(mixer)),
        ffn_(std::move(ffn)),
        mod_(std::move(mod)),
        spec_(std::move(spec)),
        zero_centered_norm_(zero_centered_norm) {}

  Status load(WeightBinder& binder, std::string_view prefix,
              const LayerContext& ctx);
  Result<Array> forward(const Array& x, MixerState* state, Array* aux_loss,
                        const LayerContext& ctx);

  [[nodiscard]] const IMixer& mixer() const noexcept { return *mixer_; }
  [[nodiscard]] IMixer& mixer() noexcept { return *mixer_; }
  [[nodiscard]] const IFeedForward& ffn() const noexcept { return *ffn_; }

 private:
  std::unique_ptr<IMixer> mixer_;
  std::unique_ptr<IFeedForward> ffn_;
  std::unique_ptr<IModGate> mod_;
  HybridBlockSpec spec_;
  Array norm1_weight_;
  Array norm2_weight_;
  bool zero_centered_norm_ = false;
};

}  // namespace lse::model
