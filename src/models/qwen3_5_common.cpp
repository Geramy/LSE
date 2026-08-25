// Qwen3.5 mixers: checkpoint names -> weight structs -> op specs.
//
// No math lives here. Every algorithm is in lse::ops and is shared with the
// other model kernels; this file only says which tensor plays which role and
// which parameters Qwen picked.
//
// Checked against the tensors of mlx-community/Qwen3.5-0.8B-4bit. It is not a
// plain transformer: every layer whose index is not 3 mod 4 is a Gated
// DeltaNet, the same skeleton lemonseed uses. The pieces that differ from
// lemonseed and are easily got wrong:
//   - the GDN decay rate is exp(A_log), with no softplus around it
//   - the GDN output gate and the post-conv activation are SiLU, not sigmoid
//   - every norm is plain RMSNorm, weight * x. lemonseed's are zero-centered,
//     (1 + weight) * x. The checkpoint says which: these weights average 1.0,
//     and a zero-centered norm stores the deviation from 1, so they would
//     average 0.
//   - attention has no separate gate weight: q_proj is twice as wide and each
//     head's second head_dim is its gate
//   - RoPE is HF rotate_half over a quarter of a 256-wide head
#include "lse/model/qwen3_5_common.hpp"

#include <cmath>
#include <optional>
#include <cstdint>
#include <vector>

#include "lse/ops/activation.hpp"
#include "lse/ops/attention.hpp"
#include "lse/ops/linear_attention.hpp"
#include "lse/ops/rope.hpp"

namespace lse::model::qwen3_5 {

using graph::Array;

namespace {

// One set of tables PER MEMBER, not one for the pool. They are constants every
// attention layer reads, so a member reading another's copy fetches them over
// the link at every such layer of every token -- a megabyte a time, measured
// at 64 crossings and 64 MB per token on a two-member split, which is more
// traffic than the tensor split's actual partial sums by an order of
// magnitude. Replicating costs one buffer per GPU and removes the crossing.
// Cached so the layers that share a member share its copy.
Result<ops::RopeTables> shared_rope(const Config& c, std::size_t member) {
  const std::int32_t max_seq = c.kv_capacity();
  struct Key {
    std::int32_t dim = 0;
    std::int32_t max_seq = 0;
    float theta = 0.0f;
  };
  static std::vector<Key> have;
  static std::vector<ops::RopeTables> tables;
  if (tables.size() <= member) {
    tables.resize(member + 1);
    have.resize(member + 1);
  }
  if (tables[member].cos.valid() && have[member].dim == c.rope_dim &&
      have[member].max_seq == max_seq && have[member].theta == c.rope_theta) {
    return tables[member];
  }
  const graph::ScopedMember on(member);
  LSE_ASSIGN_OR(tables[member],
                ops::build_rope(c.rope_dim, max_seq, c.rope_theta));
  have[member] = {c.rope_dim, max_seq, c.rope_theta};
  return tables[member];
}

// Where channel `d` of an engine-convention head vector comes from in an HF
// one. HF rotates with rotate_half, pairing channel i with i + rope_dim/2;
// graph::rope rotates adjacent pairs (2j, 2j+1). Reordering the projection
// rows once at load makes the two agree, and since q and k are reordered
// identically their dot product — the only thing attention reads — is
// unchanged. Channels at or past rope_dim are not rotated and stay put.
std::int64_t rope_source(std::int64_t d, std::int64_t rope_dim) {
  if (d >= rope_dim) return d;
  const std::int64_t j = d / 2;
  return d % 2 == 0 ? j : j + rope_dim / 2;
}

// How many ways a weight whose split axis is `extent` should be cut. One --
// meaning "do not split this" -- whenever the pool is single, the scheme is
// not a tensor split, or the axis will not divide evenly into whole
// quantization groups on every member. A weight that cannot be cut cleanly is
// left whole rather than padded: padding changes what the kernel contracts.
std::size_t tensor_shards(const LayerContext& ctx, std::int64_t extent) {
  // Asked of the context, not of the device set. How many ways a layer is cut
  // is a property of the load, which is what makes it testable without the
  // devices: a test can load the same layer whole and split and compare the
  // two, which is the only way to see a sharding bug apart from every other
  // thing a multi-device run does at once.
  const auto n = static_cast<std::int64_t>(ctx.shards);
  if (n <= 1) return 1;
  constexpr std::int64_t kGroup = 64;
  if (extent % n != 0 || (extent / n) % kGroup != 0) return 1;
  return static_cast<std::size_t>(ctx.shards);
}

// Heads split only if every member gets whole heads of both kinds.
std::size_t head_shards(const LayerContext& ctx, std::int64_t q_heads,
                       std::int64_t kv_heads) {
  const auto n = static_cast<std::int64_t>(ctx.shards);
  if (n <= 1) return 1;
  if (q_heads % n != 0 || kv_heads % n != 0) return 1;
  return static_cast<std::size_t>(ctx.shards);
}

class Qwen35Attention final : public IMixer {
 public:
  static constexpr std::string_view kName = "qwen3_5.attention";
  std::string_view name() const noexcept override { return kName; }

  Status load(WeightBinder& b, std::string_view prefix,
              const LayerContext& ctx) override {
    const Config& c = *ctx.config;
    const std::string p = std::string(prefix) + ".self_attn";
    const auto qh = static_cast<std::int64_t>(c.attn_q_heads);
    const auto kvh = static_cast<std::int64_t>(c.attn_kv_heads);
    const auto hd = static_cast<std::int64_t>(c.attn_head_dim);
    const auto rd = static_cast<std::int64_t>(c.rope_dim);
    const auto hidden = static_cast<std::int64_t>(c.hidden_size);

    // q_proj is [q_heads * 2 * head_dim, hidden] and HF splits it per head:
    // head h owns rows [h*2*hd, (h+1)*2*hd), of which the first head_dim is q
    // and the second is the output gate. Gathering all the q rows first and
    // all the gate rows after turns that into the contiguous upper-half gate
    // GateSource::kFusedInQProj already expects, and the q half carries the
    // RoPE reordering at the same time.
    // Heads are the split axis: a head's projection rows, its slice of the
    // output projection's contraction, and its pages of the KV pool all carry
    // the same index, so cutting there needs no communication until the output
    // projections' partial sums are added.
    const std::size_t shards = head_shards(ctx, qh, kvh);
    const auto n = static_cast<std::int64_t>(shards);
    const std::int64_t qh_m = qh / n;
    const std::int64_t kvh_m = kvh / n;
    w_.resize(shards);
    spec_.resize(shards);

    for (std::size_t m = 0; m < shards; ++m) {
      const std::optional<graph::ScopedMember> on =
          shards > 1 ? std::optional<graph::ScopedMember>(std::in_place, m)
                     : std::nullopt;
      const std::int64_t q0 = static_cast<std::int64_t>(m) * qh_m;
      const std::int64_t kv0 = static_cast<std::int64_t>(m) * kvh_m;
      ops::GatedAttentionWeights& w = w_[m];

    std::vector<std::int64_t> q_rows;
    q_rows.reserve(static_cast<std::size_t>(2 * qh_m * hd));
    for (std::int64_t h = q0; h < q0 + qh_m; ++h) {
      for (std::int64_t d = 0; d < hd; ++d) {
        q_rows.push_back(h * 2 * hd + rope_source(d, rd));
      }
    }
    for (std::int64_t h = q0; h < q0 + qh_m; ++h) {
      for (std::int64_t d = 0; d < hd; ++d) q_rows.push_back(h * 2 * hd + hd + d);
    }
    LSE_ASSIGN_OR(w.q_proj, b.require_rows(p + ".q_proj.weight", q_rows,
                                           Shape{2 * qh_m * hd, hidden}));

    std::vector<std::int64_t> k_rows;
    k_rows.reserve(static_cast<std::size_t>(kvh_m * hd));
    for (std::int64_t h = kv0; h < kv0 + kvh_m; ++h) {
      for (std::int64_t d = 0; d < hd; ++d) {
        k_rows.push_back(h * hd + rope_source(d, rd));
      }
    }
    LSE_ASSIGN_OR(w.k_proj, b.require_rows(p + ".k_proj.weight", k_rows,
                                           Shape{kvh_m * hd, hidden}));

    // The head norms are applied before the rotation, so they follow the same
    // reordering; RMSNorm's scaling is permutation-invariant, its weight is not.
    std::vector<std::int64_t> norm_rows;
    norm_rows.reserve(static_cast<std::size_t>(hd));
    for (std::int64_t d = 0; d < hd; ++d) norm_rows.push_back(rope_source(d, rd));
    // Head-dim wide, identical for every head, so each member keeps a copy
    // rather than a slice.
    LSE_ASSIGN_OR(w.q_norm,
                  b.require_rows(p + ".q_norm.weight", norm_rows, Shape{hd}));
    LSE_ASSIGN_OR(w.k_norm,
                  b.require_rows(p + ".k_norm.weight", norm_rows, Shape{hd}));

    std::vector<std::int64_t> v_rows;
    v_rows.reserve(static_cast<std::size_t>(kvh_m * hd));
    for (std::int64_t r = kv0 * hd; r < (kv0 + kvh_m) * hd; ++r) {
      v_rows.push_back(r);
    }
    LSE_ASSIGN_OR(w.v_proj, b.require_rows(p + ".v_proj.weight", v_rows,
                                           Shape{kvh_m * hd, hidden}));
    // The output projection contracts over the heads, so it splits the other
    // way: every member keeps all `hidden` rows but only its heads' columns,
    // and each produces a partial sum of the whole output.
    if (shards == 1) {
      LSE_ASSIGN_OR(w.o_proj, b.require(p + ".o_proj.weight"));
      LSE_RETURN_IF_ERROR(expect_shape(w.o_proj, p + ".o_proj.weight",
                                       Shape{hidden, qh * hd}));
    } else {
      LSE_ASSIGN_OR(w.o_proj,
                    b.require_columns(p + ".o_proj.weight", q0 * hd, qh_m * hd,
                                      Shape{hidden, qh_m * hd}));
    }

      ops::GatedAttentionSpec& sp = spec_[m];
      sp.q_heads = static_cast<std::int32_t>(qh_m);
      sp.kv_heads = static_cast<std::int32_t>(kvh_m);
      sp.head_dim = c.attn_head_dim;
      sp.gate = ops::GateSource::kFusedInQProj;
      // Qwen has no sliding window: every full-attention layer is global.
      sp.mask = graph::MaskKind::kCausal;
      sp.window = 0;
      sp.norm_eps = c.rms_eps;
      sp.zero_centered_norm = false;
      sp.kv_length = c.kv_capacity();
    }
    rope_.resize(spec_.size());
    for (std::size_t m = 0; m < spec_.size(); ++m) {
      LSE_ASSIGN_OR(rope_[m], shared_rope(c, m));
    }
    return OkStatus();
  }

  [[nodiscard]] bool shards_across_pool() const noexcept override {
    return true;
  }

  Result<std::vector<Array>> forward_shards(const std::vector<Array>& xs,
                                            MixerState* state,
                                            const LayerContext& ctx) override {
    (void)ctx;
    std::vector<Array> parts(w_.size());
    for (std::size_t m = 0; m < w_.size(); ++m) {
      const graph::ScopedMember on(m);
      LSE_ASSIGN_OR(parts[m], shard_forward(xs[m], state, m));
    }
    return parts;
  }

  Result<Array> forward(const Array& x, MixerState* state,
                        const LayerContext& ctx) override {
    (void)ctx;
    Array sum;
    for (std::size_t m = 0; m < w_.size(); ++m) {
      const std::optional<graph::ScopedMember> on =
          w_.size() > 1 ? std::optional<graph::ScopedMember>(std::in_place, m)
                        : std::nullopt;
      LSE_ASSIGN_OR(Array part, shard_forward(x, state, m));
      sum = m == 0 ? part : graph::add(sum, part);
    }
    return sum;
  }

 private:
  Result<Array> shard_forward(const Array& x, MixerState* state,
                              std::size_t m) {
    if (state == nullptr) {
      return ops::gated_attention(x, w_[m], spec_[m], rope_[m], 0);
    }
    MixerState& st = state[m];
    ops::AttentionCache cache;
    cache.keys = st.key_cache;
    cache.values = st.value_cache;
    cache.table = st.paged.table;
    cache.meta = st.kv_meta;
    cache.paged = &st.paged;
    cache.capacity = spec_[m].kv_length;
    cache.used = st.position;
    LSE_ASSIGN_OR(Array y, ops::gated_attention(x, w_[m], spec_[m], rope_[m],
                                                st.position, &cache));
    st.key_cache = cache.keys;
    st.value_cache = cache.values;
    return y;
  }

 public:

 private:
  std::vector<ops::GatedAttentionWeights> w_;
  std::vector<ops::GatedAttentionSpec> spec_;
  std::vector<ops::RopeTables> rope_;
};

class Qwen35GatedDeltaNet final : public IMixer {
 public:
  static constexpr std::string_view kName = "qwen3_5.gdn";
  std::string_view name() const noexcept override { return kName; }

  Status load(WeightBinder& b, std::string_view prefix,
              const LayerContext& ctx) override {
    const Config& c = *ctx.config;
    const std::string p = std::string(prefix) + ".linear_attn";
    const auto kh = static_cast<std::int64_t>(c.gdn_qk_heads);
    const auto vh = static_cast<std::int64_t>(c.gdn_v_heads);
    const auto hd = static_cast<std::int64_t>(c.gdn_head_dim);
    const auto hidden = static_cast<std::int64_t>(c.hidden_size);
    const std::int64_t conv_dim = 2 * kh * hd + vh * hd;

    const auto kernel = static_cast<std::int64_t>(c.gdn_conv_kernel);
    const std::size_t shards = head_shards(ctx, kh, vh);
    const auto n = static_cast<std::int64_t>(shards);
    const std::int64_t kh_m = kh / n;
    const std::int64_t vh_m = vh / n;
    w_.resize(shards);
    spec_.resize(shards);

    for (std::size_t m = 0; m < shards; ++m) {
      const std::optional<graph::ScopedMember> on =
          shards > 1 ? std::optional<graph::ScopedMember>(std::in_place, m)
                     : std::nullopt;
      const auto mi = static_cast<std::int64_t>(m);
      ops::GatedDeltaNetWeights& w = w_[m];

      // The fused projection is q, then k, then v, each a run of whole heads,
      // so this member's rows are three separate spans rather than one.
      std::vector<std::int64_t> qkv_rows;
      qkv_rows.reserve(static_cast<std::size_t>(2 * kh_m * hd + vh_m * hd));
      for (std::int64_t r = mi * kh_m * hd; r < (mi + 1) * kh_m * hd; ++r) {
        qkv_rows.push_back(r);
      }
      for (std::int64_t r = kh * hd + mi * kh_m * hd;
           r < kh * hd + (mi + 1) * kh_m * hd; ++r) {
        qkv_rows.push_back(r);
      }
      for (std::int64_t r = 2 * kh * hd + mi * vh_m * hd;
           r < 2 * kh * hd + (mi + 1) * vh_m * hd; ++r) {
        qkv_rows.push_back(r);
      }
      const std::int64_t conv_dim_m = 2 * kh_m * hd + vh_m * hd;
      LSE_ASSIGN_OR(w.in_proj_qkv,
                    b.require_rows(p + ".in_proj_qkv.weight", qkv_rows,
                                   Shape{conv_dim_m, hidden}));

      std::vector<std::int64_t> v_rows;
      v_rows.reserve(static_cast<std::size_t>(vh_m * hd));
      for (std::int64_t r = mi * vh_m * hd; r < (mi + 1) * vh_m * hd; ++r) {
        v_rows.push_back(r);
      }
      LSE_ASSIGN_OR(w.gate_proj, b.require_rows(p + ".in_proj_z.weight", v_rows,
                                                Shape{vh_m * hd, hidden}));

      std::vector<std::int64_t> vh_rows;
      vh_rows.reserve(static_cast<std::size_t>(vh_m));
      for (std::int64_t r = mi * vh_m; r < (mi + 1) * vh_m; ++r) {
        vh_rows.push_back(r);
      }
      LSE_ASSIGN_OR(w.in_proj_a, b.require_rows(p + ".in_proj_a.weight",
                                                vh_rows, Shape{vh_m, hidden}));
      LSE_ASSIGN_OR(w.in_proj_b, b.require_rows(p + ".in_proj_b.weight",
                                                vh_rows, Shape{vh_m, hidden}));

      // MLX writes a depthwise conv weight as [channels, kernel, in/groups]
      // and this conv is fully depthwise, so the trailing axis is 1 and the
      // elements are already in the [conv_dim, kernel] order
      // graph::causal_conv1d wants. With a trailing axis of 1 a "row" is one
      // element, so a channel is `kernel` consecutive rows.
      if (shards == 1) {
        LSE_ASSIGN_OR(w.conv_w, b.require_as(p + ".conv1d.weight",
                                             Shape{conv_dim, kernel}));
      } else {
        std::vector<std::int64_t> conv_rows;
        conv_rows.reserve(static_cast<std::size_t>(conv_dim_m * kernel));
        for (std::int64_t ch : qkv_rows) {
          for (std::int64_t k = 0; k < kernel; ++k) {
            conv_rows.push_back(ch * kernel + k);
          }
        }
        LSE_ASSIGN_OR(w.conv_w,
                      b.require_rows(p + ".conv1d.weight", conv_rows,
                                     Shape{conv_dim_m, kernel}));
      }

      LSE_ASSIGN_OR(w.a_log,
                    b.require_rows(p + ".A_log", vh_rows, Shape{vh_m}));
      LSE_ASSIGN_OR(w.dt_bias,
                    b.require_rows(p + ".dt_bias", vh_rows, Shape{vh_m}));
      // Head-dim wide and identical for every head: copied, not cut.
      LSE_ASSIGN_OR(w.norm, b.require(p + ".norm.weight"));
      LSE_RETURN_IF_ERROR(expect_shape(w.norm, p + ".norm.weight", Shape{hd}));

      if (shards == 1) {
        LSE_ASSIGN_OR(w.out_proj, b.require(p + ".out_proj.weight"));
        LSE_RETURN_IF_ERROR(expect_shape(w.out_proj, p + ".out_proj.weight",
                                         Shape{hidden, vh * hd}));
      } else {
        LSE_ASSIGN_OR(w.out_proj,
                      b.require_columns(p + ".out_proj.weight", mi * vh_m * hd,
                                        vh_m * hd, Shape{hidden, vh_m * hd}));
      }

      ops::GatedDeltaNetSpec& sp = spec_[m];
      sp.key_heads = static_cast<std::int32_t>(kh_m);
      sp.value_heads = static_cast<std::int32_t>(vh_m);
      sp.key_head_dim = sp.value_head_dim = c.gdn_head_dim;
      sp.conv_bias = false;
      sp.decay_per_value_head = true;
      sp.decay = ops::DecayRate::kExpALog;
      sp.gate = ops::GateActivation::kSiLU;
      sp.conv_activation = true;
      sp.query_scale = 1.0f / std::sqrt(static_cast<float>(c.gdn_head_dim));
      sp.layout = ops::ProjLayout::kFusedQKV;
      sp.norm_eps = c.rms_eps;
      sp.zero_centered_norm = false;
    }
    return OkStatus();
  }

  [[nodiscard]] bool shards_across_pool() const noexcept override {
    return true;
  }

  Result<std::vector<Array>> forward_shards(const std::vector<Array>& xs,
                                            MixerState* state,
                                            const LayerContext& ctx) override {
    (void)ctx;
    std::vector<Array> parts(w_.size());
    for (std::size_t m = 0; m < w_.size(); ++m) {
      const graph::ScopedMember on(m);
      LSE_ASSIGN_OR(parts[m], shard_forward(xs[m], state, m));
    }
    return parts;
  }

  Result<Array> forward(const Array& x, MixerState* state,
                        const LayerContext& ctx) override {
    (void)ctx;
    Array sum;
    for (std::size_t m = 0; m < w_.size(); ++m) {
      const std::optional<graph::ScopedMember> on =
          w_.size() > 1 ? std::optional<graph::ScopedMember>(std::in_place, m)
                        : std::nullopt;
      LSE_ASSIGN_OR(Array part, shard_forward(x, state, m));
      sum = m == 0 ? part : graph::add(sum, part);
    }
    return sum;
  }

 private:
  Result<Array> shard_forward(const Array& x, MixerState* state,
                              std::size_t m) {
      ops::GatedDeltaNetState carried;
      MixerState* st = state != nullptr ? &state[m] : nullptr;
      if (st != nullptr) {
        carried.recurrent = st->gdn_state;
        carried.conv_qkv = st->gdn_conv_qkv;
      }
      LSE_ASSIGN_OR(Array part,
                    ops::gated_delta_net(x, w_[m], spec_[m],
                                         st != nullptr ? &carried : nullptr));
      if (st != nullptr) {
        st->gdn_state = carried.recurrent;
        st->gdn_conv_qkv = carried.conv_qkv;
      }
      return part;
  }

 public:

 private:
  std::vector<ops::GatedDeltaNetWeights> w_;
  std::vector<ops::GatedDeltaNetSpec> spec_;
};

class Qwen35MLP final : public IFeedForward {
 public:
  std::string_view name() const noexcept override { return "qwen3_5.mlp"; }

  Status load(WeightBinder& b, std::string_view prefix,
              const LayerContext& ctx) override {
    const Config& c = *ctx.config;
    const std::string p = std::string(prefix) + ".mlp";
    const auto hidden = static_cast<std::int64_t>(c.hidden_size);
    const auto inter = static_cast<std::int64_t>(c.mlp_intermediate);

    const std::size_t shards = tensor_shards(ctx, inter);
    const std::int64_t span = inter / static_cast<std::int64_t>(shards);
    gate_.resize(shards);
    up_.resize(shards);
    down_.resize(shards);

    if (shards == 1) {
      // Unsharded: leave placement to whoever is loading us. Naming member 0
      // here would drag every layer's feed-forward onto the primary and undo a
      // layer split completely -- measured as 24.4 tok/s falling to 10.9.
      LSE_ASSIGN_OR(gate_[0], b.require(p + ".gate_proj.weight"));
      LSE_RETURN_IF_ERROR(expect_shape(gate_[0], p + ".gate_proj.weight",
                                       Shape{inter, hidden}));
      LSE_ASSIGN_OR(up_[0], b.require(p + ".up_proj.weight"));
      LSE_RETURN_IF_ERROR(expect_shape(up_[0], p + ".up_proj.weight",
                                       Shape{inter, hidden}));
      LSE_ASSIGN_OR(down_[0], b.require(p + ".down_proj.weight"));
      LSE_RETURN_IF_ERROR(expect_shape(down_[0], p + ".down_proj.weight",
                                       Shape{hidden, inter}));
      return OkStatus();
    }

    for (std::size_t m = 0; m < shards; ++m) {
      const graph::ScopedMember on(m);
      // gate and up are split down their OUTPUT features, so this member owns
      // rows [m*span, (m+1)*span) and produces that slice of the intermediate.
      // down is split down its INPUT features to match, so the same member
      // contracts exactly the slice it just produced and every member ends up
      // holding a partial sum over the full hidden width.
      std::vector<std::int64_t> rows(static_cast<std::size_t>(span));
      for (std::int64_t r = 0; r < span; ++r) {
        rows[static_cast<std::size_t>(r)] =
            static_cast<std::int64_t>(m) * span + r;
      }
      LSE_ASSIGN_OR(gate_[m], b.require_rows(p + ".gate_proj.weight", rows,
                                             Shape{span, hidden}));
      LSE_ASSIGN_OR(up_[m], b.require_rows(p + ".up_proj.weight", rows,
                                           Shape{span, hidden}));
      LSE_ASSIGN_OR(down_[m],
                    b.require_columns(p + ".down_proj.weight",
                                      static_cast<std::int64_t>(m) * span, span,
                                      Shape{hidden, span}));
    }
    return OkStatus();
  }

  Result<Array> forward(const Array& x, Array* aux_loss,
                        const LayerContext& ctx) override {
    (void)aux_loss;
    (void)ctx;
    if (gate_.size() == 1) return ops::swiglu(x, gate_[0], up_[0], down_[0]);

    // One branch per member, each on its own device, each reading its own
    // shard. The branches are independent until the add, and each add is
    // stamped with the member doing it — an unstamped node that reads two
    // members does not trip joins_another_member, so the join could be buried
    // inside a phase (the mixer's forward above stamps its adds for the same
    // reason).
    Array sum;
    for (std::size_t m = 0; m < gate_.size(); ++m) {
      const graph::ScopedMember on(m);
      Array part = ops::swiglu(x, gate_[m], up_[m], down_[m]);
      sum = m == 0 ? part : graph::add(sum, part);
    }
    return sum;
  }

  // Row-cut gate/up, column-cut down: this MLP genuinely splits, which is
  // half of what a tensor split is chosen on (the mixer is the other half).
  [[nodiscard]] bool shards_across_pool() const noexcept override {
    return true;
  }

  Result<std::vector<Array>> forward_shards(const std::vector<Array>& xs,
                                            Array* aux_loss,
                                            const LayerContext& ctx) override {
    (void)aux_loss;
    (void)ctx;
    std::vector<Array> parts(gate_.size());
    for (std::size_t m = 0; m < gate_.size(); ++m) {
      const graph::ScopedMember on(m);
      parts[m] = ops::swiglu(xs[m], gate_[m], up_[m], down_[m]);
    }
    return parts;
  }

 private:
  std::vector<Array> gate_, up_, down_;
};

}  // namespace

std::unique_ptr<IFeedForward> make_mlp() {
  return std::make_unique<Qwen35MLP>();
}

std::unique_ptr<IMixer> make_attention() {
  return std::make_unique<Qwen35Attention>();
}

std::unique_ptr<IMixer> make_gdn() {
  return std::make_unique<Qwen35GatedDeltaNet>();
}

HybridBlockSpec block_spec() {
  return HybridBlockSpec{".input_layernorm.weight",
                         ".post_attention_layernorm.weight"};
}

HybridLMSpec lm_spec(const Config& config) {
  HybridLMSpec spec;
  spec.embed_name = "language_model.model.embed_tokens.weight";
  spec.final_norm_name = "language_model.model.norm.weight";
  spec.block_prefix = std::string(kBlockPrefix);
  spec.lm_head_name =
      config.tie_word_embeddings ? "" : "language_model.lm_head.weight";
  spec.zero_centered_norm = false;
  // Text-only. Naming the tower here is what makes every *other* unclaimed
  // tensor a load error, and hybrid_lm prints what this one cost.
  spec.refused.push_back(
      {std::string(kVisionPrefix),
       "this build decodes text only; the Qwen3.5 vision tower is not "
       "implemented, so image and video tokens cannot be encoded"});
  spec.gdn_state_heads = config.gdn_v_heads;
  spec.gdn_state_dim = config.gdn_head_dim;
  spec.gdn_conv_width =
      (2 * config.gdn_qk_heads + config.gdn_v_heads) * config.gdn_head_dim;
  return spec;
}

Status expect_shape(const Array& a, std::string_view name, Shape want) {
  // The logical shape, not the stored one: a group-affine weight's plane counts
  // packed lanes on its last axis, and comparing that against the config would
  // reject every quantized checkpoint.
  const Shape have = graph::weight_shape(a);
  if (have == want) return OkStatus();
  return LSE_ERROR(kInvalidArgument, "'", std::string(name), "' is ",
                   have.to_string(), " but the config implies ",
                   want.to_string());
}

}  // namespace lse::model::qwen3_5
