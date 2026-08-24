// Layer contract.
//
// Layers are composed, not inherited into a single blessed block. lemonseed and
// Qwen3.6 share a skeleton (3x linear-attention, 1x gated attention, MoE FFN)
// but differ in almost every internal: fused vs split QKV, gate position,
// Mixture-of-Depths present or absent, tied or untied head. A HybridBlock
// therefore holds a mixer and an FFN chosen per model rather than assuming one
// implementation covers both.
#pragma once

#include <array>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "lse/core/status.hpp"
#include "lse/graph/graph.hpp"
#include "lse/model/config.hpp"
#include "lse/model/weights.hpp"
#include "lse/ops/attention.hpp"
#include "lse/quant/group_affine.hpp"

namespace lse::model {

using graph::Array;

// Per-sequence mixer state: GDN keeps a fixed recurrent state, attention keeps
// a paged K/V pool plus the block lists that index it. Null on a prefill with no
// cache.
//
// GDN carries by value replacement and is never indexed by position, so it needs
// no block table: in paged terms it is a one-block-per-sequence pool.
struct MixerState {
  Array gdn_state;
  // The GDN conv tails travel with it; see ops::GatedDeltaNetState.
  Array gdn_conv_q, gdn_conv_k, gdn_conv_v, gdn_conv_qkv;
  // The paged pools and their block lists. `key_cache`/`value_cache` mirror
  // paged.keys/values so the program's carry pairs and the cache-leaf check keep
  // working on plain Arrays.
  ops::PagedKvLayer paged;
  Array key_cache;
  Array value_cache;
  // Shared per-step descriptor the decode program pokes: f32 [3] =
  // {first query position, live KV length, real rows}.
  Array kv_meta;
  std::int32_t position = 0;

  [[nodiscard]] bool empty() const noexcept {
    return !gdn_state.valid() && !key_cache.valid();
  }
};

struct LayerContext {
  const Config* config = nullptr;
  std::int32_t layer_index = 0;
  bool training = false;
  // How many ways this layer's weights are split across the pool, and so how
  // many MixerStates the `state` pointer addresses: one per member, laid out
  // consecutively. One under every scheme but a tensor split.
  std::int32_t shards = 1;
};

// Names bound so far, used to report anything the checkpoint had that no layer
// claimed — the most likely silent failure when adding a second architecture.
// A half-open window over a tensor's last axis, in elements of that axis.
// Namespace scope rather than nested: a nested type's default member
// initializers are not usable in a default argument of its own enclosing class.
struct TensorWindow {
  std::int64_t first = 0;
  std::int64_t count = 0;

  [[nodiscard]] bool empty() const noexcept { return count <= 0; }
};

class WeightBinder {
 public:
  // `quantization` says which tensors are group-affine and at what geometry.
  // A quantized tensor is a triple — `<name>`, `<name minus .weight>.scales`,
  // `.biases` — and the binder claims all three together, returning the packed
  // plane with the other two attached, so a caller still asks for one name and
  // hands the result to graph::linear unchanged. Null means the checkpoint
  // declared no quantization: a tensor that nonetheless has the side planes
  // then fails naming itself rather than being read at a guessed width.
  explicit WeightBinder(const SafeTensors& weights,
                        const quant::GroupAffineMap* quantization = nullptr)
      : weights_(&weights), quantization_(quantization) {}

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
  //
  // On a group-affine tensor a row is an output feature, and its scale and
  // bias rows carry the same index, so all three planes take the permutation
  // and `shape` still names the logical [out, in].
  Result<Array> require_rows(std::string_view name,
                             const std::vector<std::int64_t>& order,
                             Shape shape);

  // Reads `name` keeping only input features [first, first + count) of every
  // row, reporting the result as `shape`.
  //
  // The other half of a tensor split. A weight whose OUTPUT features are split
  // is a row selection -- require_rows above -- and each member then holds a
  // slice of the result. A weight whose INPUT features are split is this: every
  // member holds every output row but only its own span of the contraction, so
  // each computes a partial sum and the members add. Down-projections and
  // attention output projections are the second kind, which is why a tensor
  // split needs both and why one of them cannot be spelled as a permutation.
  //
  // On a group-affine tensor the window has to fall on a group boundary and on
  // a whole number of packed lanes: the codes of one group share a scale and
  // several input features share a u32, so a window that split either would
  // have to re-quantize rather than slice. Both hold when `first` and `count`
  // are multiples of the group size, which is what a pool of any size gives on
  // these models.
  Result<Array> require_columns(std::string_view name, std::int64_t first,
                                std::int64_t count, Shape shape);

  // The same tensor under a different shape of the same elements. MLX writes a
  // depthwise conv weight as [channels, kernel, 1]; the graph wants
  // [channels, kernel]. Nothing moves — only the reported shape differs.
  Result<Array> require_as(std::string_view name, Shape shape);

  [[nodiscard]] const SafeTensors& weights() const noexcept { return *weights_; }
  [[nodiscard]] std::vector<std::string> unclaimed() const;
  [[nodiscard]] std::size_t claimed_count() const noexcept { return claimed_.size(); }

 private:
  // The `.scales` / `.biases` planes beside `name`, or nulls when the
  // checkpoint stored it in full precision. Errors when exactly one is there.
  Result<std::array<const TensorView*, 2>> quant_planes(
      std::string_view name) const;

  // All three planes, claimed together. `order` and `logical` come from
  // require_rows and are empty otherwise.
  Result<Array> bind_quantized(std::string_view name, const TensorView& packed,
                               const TensorView& scales,
                               const TensorView& biases,
                               const std::vector<std::int64_t>* order,
                               Shape logical, TensorWindow window = {});

  const SafeTensors* weights_;
  const quant::GroupAffineMap* quantization_ = nullptr;
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

  // Whether this mixer splits its own weights across the pool.
  //
  // A tensor split is only worth taking when the whole block takes it. Sharding
  // the feed-forward alone leaves every mixer on one member, so the other one
  // holds half of one operation per layer and idles through the rest -- 13.4
  // tok/s measured against 24.4 for a plain layer split. So the scheme is
  // chosen on this, and a mixer that has not been taught to shard keeps the
  // pool on layers rather than quietly making it slower.
  [[nodiscard]] virtual bool shards_across_pool() const noexcept {
    return false;
  }

  // One partial per member, un-summed, given one input per member.
  //
  // The block adds them, because the block is what knows where the sum has to
  // end up: on EVERY member, so the residual and the norms that follow are
  // local work rather than a fetch. Summing here and handing back one array
  // would put the whole residual stream on one device and make every branch of
  // the next layer reach across the link for its input.
  virtual Result<std::vector<Array>> forward_shards(
      const std::vector<Array>& xs, MixerState* state,
      const LayerContext& ctx) {
    LSE_ASSIGN_OR(Array y, forward(xs.front(), state, ctx));
    return std::vector<Array>{std::move(y)};
  }
};

class IFeedForward {
 public:
  virtual ~IFeedForward() = default;
  virtual Status load(WeightBinder&, std::string_view prefix,
                      const LayerContext&) = 0;
  // Whether this feed-forward splits its own weights across the pool. The
  // same question shards_across_pool answers for a mixer, asked of the other
  // half of the block for the same reason: a tensor split chosen on the mixer
  // alone leaves an FFN that cannot shard whole on one member — the dominant
  // cost of the layer running at one device's speed while the split pays its
  // reduces anyway, which is the regression the mixer gate was measured to
  // prevent (13.4 vs 24.4 tok/s).
  [[nodiscard]] virtual bool shards_across_pool() const noexcept {
    return false;
  }
  // One partial per member; see IMixer::forward_shards.
  virtual Result<std::vector<Array>> forward_shards(
      const std::vector<Array>& xs, Array* aux_loss,
      const LayerContext& ctx) {
    LSE_ASSIGN_OR(Array y, forward(xs.front(), aux_loss, ctx));
    return std::vector<Array>{std::move(y)};
  }
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
  // Whether this block's mixer shards across the pool; see IMixer.
  [[nodiscard]] bool mixer_shards() const noexcept {
    return mixer_ != nullptr && mixer_->shards_across_pool();
  }

  // Whether this block's feed-forward does; see IFeedForward. A tensor split
  // is chosen only when both halves of every block take it.
  [[nodiscard]] bool ffn_shards() const noexcept {
    return ffn_ != nullptr && ffn_->shards_across_pool();
  }

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

  // The same block with the activation held on every member at once. Returns
  // one array per member, each holding the whole layer output: the partials
  // are summed once per member so the residual stream never has to be fetched.
  Result<std::vector<Array>> forward_shards(const std::vector<Array>& xs,
                                            MixerState* state, Array* aux_loss,
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
  // A copy per member when the block is split: the norms are elementwise and
  // cheap, so running them redundantly on each device costs less than moving
  // their input across the link.
  std::vector<Array> norm1_shards_, norm2_shards_;
  bool zero_centered_norm_ = false;
};

}  // namespace lse::model
