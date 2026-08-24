#include "lse/model/hybrid_lm.hpp"

#include <algorithm>
#include <cstdio>
#include <span>

#include "lse/graph/interpreter.hpp"
#include "lse/graph/ops.hpp"
#include "lse/ops/norm.hpp"

namespace lse::model {

namespace {

// layer.hpp warns that an unclaimed tensor is the likeliest silent failure when
// a second architecture arrives, and nothing checked for one. This is that
// check: it runs on every load, and it names the tensors rather than just their
// count so the report says which part of the checkpoint went unread.
//
// A tensor under a declared refusal prefix is not an error, but it is not
// silent either: each prefix that actually matched something is reported with
// what it cost and why it was refused.
Status audit_unclaimed(const WeightBinder& binder,
                       const std::vector<HybridLMSpec::Refusal>& refused) {
  std::vector<std::string> missed;
  std::vector<std::size_t> counts(refused.size(), 0);
  std::vector<std::size_t> bytes(refused.size(), 0);
  for (const std::string& name : binder.unclaimed()) {
    bool covered = false;
    for (std::size_t i = 0; i < refused.size(); ++i) {
      if (name.rfind(refused[i].prefix, 0) != 0) continue;
      ++counts[i];
      if (const TensorView* v = binder.weights().find(name)) {
        bytes[i] += v->data.size();
      }
      covered = true;
      break;
    }
    if (!covered) missed.push_back(name);
  }
  for (std::size_t i = 0; i < refused.size(); ++i) {
    if (counts[i] == 0) continue;
    std::fprintf(stderr, "lse: refused %zu tensor(s) under '%s' (%.1f MiB): %s\n",
                 counts[i], refused[i].prefix.c_str(),
                 static_cast<double>(bytes[i]) / (1024.0 * 1024.0),
                 refused[i].reason.c_str());
  }
  if (missed.empty()) return OkStatus();

  std::string names;
  const std::size_t shown = missed.size() < 8 ? missed.size() : 8;
  for (std::size_t i = 0; i < shown; ++i) {
    if (i != 0) names += ", ";
    names += missed[i];
  }
  if (missed.size() > shown) {
    names += ", and " + std::to_string(missed.size() - shown) + " more";
  }
  return LSE_ERROR(kInvalidArgument, "no layer claimed ",
                   std::to_string(missed.size()),
                   " checkpoint tensor(s): ", names);
}

}  // namespace

Result<std::int32_t> batch_bucket(std::int32_t rows) {
  if (rows <= 0) {
    return LSE_ERROR(kInvalidArgument, "a pass needs at least one row, got ",
                     std::to_string(rows));
  }
  for (std::int32_t rung : kBatchRungs) {
    if (rows <= rung) return rung;
  }
  return LSE_ERROR(kOutOfRange, "batch of ", std::to_string(rows),
                   " fits no bucket; the ladder tops out at ",
                   std::to_string(kBatchRungs[std::size(kBatchRungs) - 1]),
                   " rows, so split the batch");
}

std::size_t HybridLM::state_shards() const noexcept {
  if (graph::split_scheme() != graph::SplitScheme::kTensor) return 1;
  const graph::Scheduler* s = graph::default_scheduler();
  return s != nullptr && s->devices().size() > 0 ? s->devices().size() : 1;
}

std::size_t HybridLM::member_for_layer(std::int32_t i) const noexcept {
  const graph::Scheduler* s = graph::default_scheduler();
  const std::size_t members =
      s != nullptr && s->devices().size() > 0 ? s->devices().size() : 1;
  const auto layers = static_cast<std::size_t>(config_.num_layers);
  if (members <= 1 || layers == 0 || i < 0) return 0;
  // A tensor split puts a piece of every layer on every member, so no layer
  // belongs anywhere: the shards name their own member and what is left --
  // the norms, the residual -- stays with the primary.
  if (graph::split_scheme() == graph::SplitScheme::kTensor) return 0;
  return std::min(members - 1,
                  static_cast<std::size_t>(i) * members / layers);
}

Status HybridLM::load(WeightBinder& binder) {
  // The scheme is settled before a single weight is read, because it decides
  // how they are read. Tensor split whenever the pool has more than one member
  // and the projections divide into whole quantization groups on each of them;
  // a run of layers each otherwise, which needs nothing of the shapes.
  //
  // Tensor first because it is the only one of the two that makes the members'
  // memory bandwidth add: a layer split hands layer n+1 what layer n produced,
  // so the devices take turns and the pool runs at one device's speed.
  {
    const graph::Scheduler* s = graph::default_scheduler();
    const std::size_t members =
        s != nullptr ? s->devices().size() : std::size_t{1};
    graph::SplitScheme scheme = graph::SplitScheme::kNone;
    if (members > 1) {
      const auto inter = static_cast<std::int64_t>(config_.mlp_intermediate);
      const auto n = static_cast<std::int64_t>(members);
      // inter > 0 is load-bearing: an MoE config stores its widths in
      // expert_intermediate and sets mlp_intermediate to 0, and 0 divides
      // everything — this test used to read a zero as "divides" and pick a
      // tensor split for a model whose FFN cannot take one.
      const bool divides =
          inter > 0 && inter % n == 0 && (inter / n) % 64 == 0;
      // Asked of the blocks themselves, built for the question and thrown
      // away: every layer has to shard — BOTH halves of it. A block whose
      // mixer shards but whose feed-forward does not (an MoE block today)
      // would run the dominant cost of the layer on one member while paying
      // the split's reduces anyway, and on a hybrid model the layer that
      // cannot shard is not the one at index 0.
      bool blocks_shard = config_.num_layers > 0;
      for (std::int32_t i = 0; i < config_.num_layers && blocks_shard; ++i) {
        auto probe = factory_(i);
        if (!probe.ok() || *probe == nullptr || !(*probe)->mixer_shards() ||
            !(*probe)->ffn_shards()) {
          blocks_shard = false;
        }
      }
      scheme = divides && blocks_shard ? graph::SplitScheme::kTensor
                                       : graph::SplitScheme::kLayer;
      // LSE_SPLIT=layer|tensor overrides the choice — diagnostic, for
      // isolating one scheme's bugs from the other's on the same pool.
      if (const char* forced = std::getenv("LSE_SPLIT")) {
        if (std::string_view(forced) == "layer") {
          scheme = graph::SplitScheme::kLayer;
        } else if (std::string_view(forced) == "tensor") {
          scheme = graph::SplitScheme::kTensor;
        }
      }
    }
    graph::set_split_scheme(scheme);
  }
  LSE_ASSIGN_OR(embed_weight_, binder.require(spec_.embed_name));
  LSE_ASSIGN_OR(final_norm_weight_, binder.require(spec_.final_norm_name));
  if (!spec_.lm_head_name.empty()) {
    LSE_ASSIGN_OR(lm_head_weight_, binder.require(spec_.lm_head_name));
  }

  // Which member holds layer `i`. One member is every layer; more than one
  // splits them into contiguous blocks of as equal a size as the count allows.
  blocks_.clear();
  blocks_.reserve(static_cast<std::size_t>(config_.num_layers));
  for (std::int32_t i = 0; i < config_.num_layers; ++i) {
    LSE_ASSIGN_OR(std::unique_ptr<HybridBlock> block, factory_(i));
    if (block == nullptr) {
      return LSE_ERROR(kInternal, "block factory returned null for layer ",
                       std::to_string(i));
    }
    LayerContext ctx;
    ctx.config = &config_;
    ctx.layer_index = i;
    ctx.shards = static_cast<std::int32_t>(state_shards());
    const std::string prefix = spec_.block_prefix + "." + std::to_string(i);
    // Contiguous blocks, not round robin. A layer reads what the one before it
    // wrote, so the boundaries are what cost: blocks of layers cross the link
    // once per member, where interleaving would cross it once per layer.
    {
      const graph::ScopedMember on(member_for_layer(i));
      LSE_RETURN_IF_ERROR(block->load(binder, prefix, ctx));
    }
    blocks_.push_back(std::move(block));
  }
  return audit_unclaimed(binder, spec_.refused);
}

Result<Array> HybridLM::embed(const Array& tokens) const {
  if (!embed_weight_.valid()) {
    return LSE_ERROR(kInternal, "HybridLM::embed before load()");
  }
  return graph::embedding(embed_weight_, tokens);
}

namespace {

struct StateSnap {
  graph::NodePtr gdn, cq, ck, cv, cqkv, keys, values;
};

StateSnap snap_state(const MixerState& s) {
  StateSnap o;
  if (s.gdn_state.valid()) o.gdn = s.gdn_state.node();
  if (s.gdn_conv_q.valid()) o.cq = s.gdn_conv_q.node();
  if (s.gdn_conv_k.valid()) o.ck = s.gdn_conv_k.node();
  if (s.gdn_conv_v.valid()) o.cv = s.gdn_conv_v.node();
  if (s.gdn_conv_qkv.valid()) o.cqkv = s.gdn_conv_qkv.node();
  if (s.key_cache.valid()) o.keys = s.key_cache.node();
  if (s.value_cache.valid()) o.values = s.value_cache.node();
  return o;
}

void add_carry(std::vector<graph::Program::Carry>& c, const graph::NodePtr& in,
               const Array& out) {
  if (!in || !out.valid() || !out.node()) return;
  if (out.node().get() == in.get()) return;
  c.push_back({in, out.node()});
}

Status poke_values(Array& slot, std::span<const float> values) {
  if (!slot.valid()) return LSE_ERROR(kInvalidArgument, "poke on empty Array");
  graph::Node& dst = *slot.node();
  if (dst.element_count() < values.size()) {
    return LSE_ERROR(kInvalidArgument, "poke of ",
                     std::to_string(values.size()), " into a slot of ",
                     std::to_string(dst.element_count()));
  }
  graph::Scheduler* sched = graph::default_scheduler();
  if (sched == nullptr) return LSE_ERROR(kInternal, "no backend for a poke");
  if (!dst.buffer.valid()) {
    LSE_RETURN_IF_ERROR(
        graph::interpreter::ensure_output_buffer(dst, sched->backend()));
  }
  const std::size_t bytes = dtype_storage_bytes(dst.dtype, dst.element_count());
  if (dst.host_mirror.size() < bytes) dst.host_mirror.resize(bytes);
  for (std::size_t i = 0; i < values.size(); ++i) {
    graph::interpreter::store_element(dst, i, values[i]);
  }
  dst.host_dirty = true;
  dst.device_dirty = false;
  dst.materialized = true;
  return graph::interpreter::sync_to_device(dst, sched->backend());
}

std::uint32_t host_groups_so_far() {
  graph::Scheduler* sched = graph::default_scheduler();
  return sched == nullptr ? 0u : sched->accumulated_trace().host_groups;
}

Status poke_tokens(Array& slot, const Array& incoming) {
  if (!slot.valid() || !incoming.valid()) {
    return LSE_ERROR(kInvalidArgument, "token poke on empty Array");
  }
  if (slot.shape().elem_count() != incoming.shape().elem_count()) {
    return LSE_ERROR(kInvalidArgument, "token count changed");
  }
  graph::Node& dst = *slot.node();
  const graph::Node& src = *incoming.node();
  graph::Scheduler* sched = graph::default_scheduler();
  if (sched == nullptr) return LSE_ERROR(kInternal, "no backend for token poke");
  if (!dst.buffer.valid()) {
    LSE_RETURN_IF_ERROR(
        graph::interpreter::ensure_output_buffer(dst, sched->backend()));
  }
  const std::size_t bytes =
      dtype_storage_bytes(dst.dtype, dst.element_count());
  if (dst.host_mirror.size() < bytes) dst.host_mirror.resize(bytes);
  const std::size_t n = dst.element_count();
  for (std::size_t i = 0; i < n; ++i) {
    graph::interpreter::store_element(dst, i,
                                      graph::interpreter::load_element(src, i));
  }
  dst.host_dirty = true;
  dst.device_dirty = false;
  dst.materialized = true;
  return graph::interpreter::sync_to_device(dst, sched->backend());
}

}  // namespace

void HybridLM::rewind(std::vector<MixerState>& states,
                      std::int32_t position) const {
  for (MixerState& s : states) {
    if (s.key_cache.valid()) s.position = position;
  }
}

Result<Array> HybridLM::hidden(const Array& tokens,
                               std::vector<MixerState>* states, Array* aux_loss,
                               std::vector<Array>* trace, const StepRows* rows,
                               bool replaces_previous) {
  if (blocks_.empty()) {
    return LSE_ERROR(kInternal, "HybridLM::hidden before load()");
  }
  if (states != nullptr &&
      states->size() != blocks_.size() * state_shards()) {
    return LSE_ERROR(kInvalidArgument, "expected ",
                     std::to_string(blocks_.size() * state_shards()),
                     " mixer states (", std::to_string(blocks_.size()),
                     " layers x ", std::to_string(state_shards()),
                     " shards), got ", std::to_string(states->size()));
  }

  const std::int64_t t_now =
      tokens.valid() ? tokens.shape().dim(tokens.shape().rank() - 1) : 0;
  // The retention slot for THIS pass shape. Named cache_ so the thirty uses
  // below read as they always did; the member is caches_ now.
  ForwardCache& cache_ = cache_slot(t_now);

  if (states != nullptr && tokens.valid()) {
    const auto batch = tokens.shape().dim(0);
    const auto vh = static_cast<std::int64_t>(
        spec_.gdn_state_heads > 0 ? spec_.gdn_state_heads : config_.gdn_qk_heads);
    const auto vd = static_cast<std::int64_t>(
        spec_.gdn_state_dim > 0 ? spec_.gdn_state_dim : config_.gdn_head_dim);
    const auto tail = config_.gdn_conv_kernel > 1
                          ? static_cast<std::int64_t>(config_.gdn_conv_kernel - 1)
                          : 0;
    const std::size_t shards = state_shards();
    const auto n = static_cast<std::int64_t>(shards);
    // A tensor split cuts the recurrence along its head axis with the weights
    // that drive it, so each member carries only its own heads' state and its
    // own channels of the convolution tail.
    //
    // Sharded on the SAME gate the mixer's load uses (head_shards): every
    // member must get whole heads of both kinds, or the mixer declines and
    // keeps its heads whole. Testing one axis here while the mixer tests two
    // allocated a state n× too small for the scan that reads it whenever the
    // v-heads divided and the qk-heads did not.
    const auto kh = static_cast<std::int64_t>(config_.gdn_qk_heads);
    const auto vheads = static_cast<std::int64_t>(config_.gdn_v_heads);
    const bool heads_divide = shards > 1 && kh > 0 && vheads > 0 &&
                              kh % n == 0 && vheads % n == 0;
    const std::int64_t vh_m = heads_divide && vh % n == 0 ? vh / n : vh;
    const auto fused_full = static_cast<std::int64_t>(spec_.gdn_conv_width);
    const std::int64_t fused =
        heads_divide && fused_full % n == 0 ? fused_full / n : fused_full;
    const auto width = vh_m * vd;
    for (std::size_t i = 0; i < states->size(); ++i) {
      // Flat over (layer, member), so the layer is the quotient -- an
      // attention layer's shards must not be given a recurrent state.
      const auto layer = static_cast<std::int32_t>(i / shards);
      if (config_.is_attention_layer(layer)) {
        continue;
      }
      MixerState& st = (*states)[i];
      // A tensor split stamps the shard's member; a layer split stamps the
      // member the layer actually runs on. Member 0 unconditionally put every
      // layer's state on the primary while the layer ran elsewhere, which
      // cost a peer fetch per state per rebuild and undid the point of
      // splitting by layers.
      const graph::ScopedMember on(shards > 1
                                       ? i % shards
                                       : member_for_layer(layer));
      if (!st.gdn_state.valid()) {
        st.gdn_state = Array::zeros(Shape{batch, vh_m, vd, vd}, DType::kF32);
      }
      if (tail <= 0) continue;
      // The tail is not an optimization: causal_conv1d zero-pads, so a decode
      // step with no tail convolves against zeros where the previous kernel-1
      // tokens belong.
      if (fused > 0) {
        if (!st.gdn_conv_qkv.valid()) {
          st.gdn_conv_qkv = Array::zeros(Shape{batch, tail, fused}, DType::kF32);
        }
      } else if (width > 0) {
        if (!st.gdn_conv_q.valid()) {
          st.gdn_conv_q = Array::zeros(Shape{batch, tail, width}, DType::kF32);
        }
        if (!st.gdn_conv_k.valid()) {
          st.gdn_conv_k = Array::zeros(Shape{batch, tail, width}, DType::kF32);
        }
        if (!st.gdn_conv_v.valid()) {
          st.gdn_conv_v = Array::zeros(Shape{batch, tail, width}, DType::kF32);
        }
      }
    }
  }

  const auto bucket = static_cast<std::int32_t>(
      tokens.valid() ? tokens.shape().dim(0) : 1);
  // The batch axis reaching the graph must already be a rung: it is what the JIT
  // keys on, so an off-ladder width is a shape set nobody budgeted for.
  LSE_ASSIGN_OR(const std::int32_t rung, batch_bucket(bucket));
  if (rung != bucket) {
    return LSE_ERROR(kInvalidArgument, "a pass of ", std::to_string(bucket),
                     " rows is not a batch bucket; pad it to ",
                     std::to_string(rung));
  }

  // Where every row of the pass sits. Without a plan the pass is one sequence
  // and every row shares the state's position, which is what a prefill chunk and
  // a single-session decode step both are.
  std::int32_t shared_pos = 0;
  if (states != nullptr) {
    for (const MixerState& s : *states) {
      if (s.position > shared_pos) shared_pos = s.position;
    }
  }
  std::vector<std::int32_t> first;
  if (rows != nullptr) {
    if (rows->bucket() != bucket) {
      return LSE_ERROR(kInvalidArgument, "a step plan for ",
                       std::to_string(rows->bucket()),
                       " rows against a pass of ", std::to_string(bucket));
    }
    first = rows->first;
  } else {
    first.assign(static_cast<std::size_t>(bucket), shared_pos);
  }
  const std::int32_t live_rows =
      rows != nullptr ? rows->live_prefix() : bucket;

  // The step descriptor the paged kernels read. Every value in it is a dispatch
  // value, not a shape — which is what lets one code object serve every
  // position, every length and every batch up to the bucket. See kv/block.hpp
  // for the layout and for why the per-row half may only reach a guard.
  std::int32_t kv_pos = 0;
  std::int32_t kv_len = 0;
  std::vector<float> meta(
      static_cast<std::size_t>(kv::step_meta_elems(bucket)), 0.0f);
  std::vector<std::int32_t> row_tokens(static_cast<std::size_t>(bucket), 0);
  for (std::size_t r = 0; r < first.size(); ++r) {
    const std::size_t at = static_cast<std::size_t>(kv::kStepMetaHeader) +
                           r * static_cast<std::size_t>(kv::kStepMetaPerRow);
    if (first[r] < 0) continue;  // holds no sequence: position 0, length 0
    const std::int32_t after = first[r] + static_cast<std::int32_t>(t_now);
    row_tokens[r] = after;
    meta[at] = static_cast<float>(first[r]);
    meta[at + 1] = static_cast<float>(after);
    if (first[r] > kv_pos) kv_pos = first[r];
    if (after > kv_len) kv_len = after;
  }
  // Checked here rather than only where the pool is sized: the decode fast path
  // replays a held program and never re-enters the attention layer, so a session
  // running past the engine length would write past the last slot its block table
  // has instead of being refused.
  if (states != nullptr && kv_len > config_.kv_capacity()) {
    return LSE_ERROR(kOutOfRange, "this pass would reach KV position ",
                     std::to_string(kv_len), ", past the engine length ",
                     std::to_string(config_.kv_capacity()));
  }
  meta[0] = static_cast<float>(kv_pos);
  meta[1] = static_cast<float>(kv_len);
  meta[2] = static_cast<float>(live_rows);

  // Blocks first, on both paths: the retained decode program is not re-recorded,
  // yet it crosses a block boundary every kv::kBlockSize tokens and needs the
  // next block in the table before it runs. A pool that has to move to a bigger
  // rung cannot be absorbed by a replay — the buffer changes identity — so that
  // answer forces the rebuild below.
  bool pool_moved = false;
  if (states != nullptr) {
    for (MixerState& st : *states) {
      // Set before the validity check: the first pass through a layer allocates
      // its pool inside the graph record, and it sizes it from exactly this.
      st.paged.row_tokens = row_tokens;
      if (!st.paged.valid()) continue;
      LSE_ASSIGN_OR(const bool moved, ops::extend_paged(st.paged, kv_len));
      pool_moved = pool_moved || moved;
    }
  }

  const std::uint64_t prev_pass_run = last_pass_id_;
  auto kv_leaves_match = [&]() -> bool {
    if (states == nullptr) return cache_.kv_leaves.empty();
    if (cache_.kv_leaves.size() != states->size() * 2) return false;
    for (std::size_t i = 0; i < states->size(); ++i) {
      graph::Node* k = (*states)[i].key_cache.valid()
                           ? (*states)[i].key_cache.node().get()
                           : nullptr;
      graph::Node* v = (*states)[i].value_cache.valid()
                           ? (*states)[i].value_cache.node().get()
                           : nullptr;
      if (cache_.kv_leaves[2 * i] != k || cache_.kv_leaves[2 * i + 1] != v) {
        if (std::getenv("LSE_DEBUG_REUSE") != nullptr) {
          std::fprintf(stderr,
                       "[reuse] kv mismatch i=%zu cachedK=%p curK=%p "
                       "cachedV=%p curV=%p poolK=%p\n",
                       i, (void*)cache_.kv_leaves[2 * i], (void*)k,
                       (void*)cache_.kv_leaves[2 * i + 1], (void*)v,
                       (void*)((*states)[i].paged.keys.valid()
                                   ? (*states)[i].paged.keys.node().get()
                                   : nullptr));
        }
        return false;
      }
    }
    return true;
  };

  // The descriptor is one float array per bucket width, so a batch that changes
  // rung gets a new slot and the program that pointed at the old one cannot be
  // replayed. Sized here, before the reuse test, so that answer is part of it.
  const auto meta_elems = static_cast<std::int64_t>(kv::step_meta_elems(bucket));
  const bool meta_moved =
      !cache_.meta.valid() ||
      static_cast<std::int64_t>(cache_.meta.shape().elem_count()) != meta_elems;

  // LSE_NO_REPLAY=1: rebuild the graph every pass instead of replaying the
  // retained program. Diagnostic only — it converts a wrong-replay bug into a
  // slow-but-correct run, which is what isolates one.
  static const bool no_replay = std::getenv("LSE_NO_REPLAY") != nullptr;
  bool at_sequence_start = true;
  if (states != nullptr) {
    for (const MixerState& st : *states) {
      if (st.position > 0) {
        at_sequence_start = false;
        break;
      }
    }
  }
  // A slot's carry-ins are the out-nodes of the pass that preceded it when it
  // was retained. Replaying it after any OTHER pass reads that stale chain:
  // valid only at a sequence start (the ins are zeroed), when the same chain
  // predecessor just ran, or when the slot repeats itself (decode; the fold
  // advances its state).
  const bool chain_ok = at_sequence_start ||
                        (cache_.prev_pass != 0 &&
                         cache_.prev_pass == prev_pass_run) ||
                        (cache_.pass_id != 0 &&
                         cache_.pass_id == prev_pass_run);
  const bool can_reuse =
      !no_replay && chain_ok &&
      aux_loss == nullptr && trace == nullptr && !pool_moved && !meta_moved &&
      cache_.hidden.valid() && cache_.tokens.valid() && tokens.valid() &&
      tokens.shape().elem_count() == cache_.tokens.shape().elem_count() &&
      cache_.states == states && !cache_.program.empty() &&
      !cache_.program.groups().empty() && kv_leaves_match();
  if (std::getenv("LSE_DEBUG_REUSE") != nullptr) {
    std::fprintf(
        stderr,
        "[reuse] t=%lld can=%d aux=%d tr=%d pool=%d meta=%d hid=%d ctok=%d "
        "tok=%d shp=%d st=%d prog=%d grp=%d kv=%d\n",
        static_cast<long long>(t_now), (int)can_reuse,
        (int)(aux_loss == nullptr), (int)(trace == nullptr),
        (int)!pool_moved, (int)!meta_moved, (int)cache_.hidden.valid(),
        (int)cache_.tokens.valid(), (int)tokens.valid(),
        (int)(cache_.tokens.valid() && tokens.valid() &&
              tokens.shape().elem_count() ==
                  cache_.tokens.shape().elem_count()),
        (int)(cache_.states == states), (int)!cache_.program.empty(),
        (int)!cache_.program.groups().empty(), (int)kv_leaves_match());
  }
  if (replaces_previous && last_pass_host_groups_ != 0) {
    return LSE_ERROR(kUnimplemented, "the pass being replaced put ",
                     std::to_string(last_pass_host_groups_),
                     " group(s) on the host, and a host group does not leave "
                     "the carried state where a replacement pass has to find "
                     "it");
  }

  const std::uint32_t host_before = host_groups_so_far();
  if (can_reuse) {
    cache_.program.reset_compute();
    // A whole-prompt prefill replay is a FRESH sequence: its carried states
    // must start from zero, not from wherever the previous conversation's
    // folds left the in-buffers. Folding here handed request 2 the tail of
    // request 1's state. A batch resumption arrives with row metadata and
    // keeps the fold. Decode (t_now == 1) always folds: it genuinely carries.
    // Only the FIRST chunk of a prefill ladder starts the sequence; the later
    // chunks' carry-ins hold the state the chunk before them just produced,
    // and zeroing those mid-ladder replays the rest of the prompt against a
    // blank recurrence.
    const bool fresh_sequence =
        t_now > 1 && rows == nullptr && !replaces_previous &&
        at_sequence_start;
    if (fresh_sequence) {
      graph::Scheduler* sched = graph::default_scheduler();
      if (sched == nullptr) return LSE_ERROR(kInternal, "no backend");
      for (const graph::Program::Carry& c : cache_.program.carries()) {
        if (!c.in || !c.in->buffer.valid()) continue;
        const std::size_t bytes =
            dtype_storage_bytes(c.in->dtype, c.in->element_count());
        if (bytes == 0) continue;
        const std::vector<std::byte> zeros(bytes, std::byte{0});
        LSE_RETURN_IF_ERROR(
            sched->backend().copy_h2d(zeros.data(), c.in->buffer, bytes, 0));
        c.in->materialized = true;
        c.in->device_dirty = true;
        c.in->host_dirty = false;
      }
    } else if (replaces_previous) {
      cache_.program.hold_carries();
    } else if (cache_.pass_id == prev_pass_run) {
      // Same program again: its outs are the freshest state, move them in.
      cache_.program.fold_carries();
    }
    // Different program ran last: this slot's in-nodes are that program's out
    // nodes and were just written by its eval — nothing to fold.
    LSE_RETURN_IF_ERROR(poke_tokens(cache_.tokens, tokens));
    LSE_RETURN_IF_ERROR(poke_values(cache_.meta, meta));
    if (graph::Scheduler* sched = graph::default_scheduler()) {
      LSE_RETURN_IF_ERROR(
          sched->eval(cache_.program.roots(), false, &cache_.program));
    }
    last_pass_host_groups_ = host_groups_so_far() - host_before;
    if (states != nullptr &&
        cache_.state_stamp.size() == states->size()) {
      for (std::size_t i = 0; i < states->size(); ++i) {
        MixerState& s = (*states)[i];
        const ForwardCache::StateStamp& ss = cache_.state_stamp[i];
        s.gdn_state = ss.gdn;
        s.gdn_conv_q = ss.cq;
        s.gdn_conv_k = ss.ck;
        s.gdn_conv_v = ss.cv;
        s.gdn_conv_qkv = ss.cqkv;
        s.key_cache = ss.keys;
        s.value_cache = ss.values;
        s.kv_meta = ss.meta;
      }
    }
    if (states != nullptr) {
      for (MixerState& s : *states) {
        if (s.key_cache.valid()) {
          s.position = kv_len;
        }
      }
    }
    last_pass_id_ = cache_.pass_id;
    return cache_.hidden;
  }

  if (replaces_previous) {
    // A rebuild records the graph from whatever the state Arrays hold now,
    // which after the pass being replaced is the state that pass produced.
    // There is nothing to roll back to, so this is an error rather than a
    // silently wrong continuation.
    return LSE_ERROR(kInternal,
                     "a pass that replaces the previous one cannot rebuild the "
                     "graph; the state it would start from is the one it is "
                     "meant to discard");
  }

  if (meta_moved) {
    graph::Scheduler* sched = graph::default_scheduler();
    if (sched == nullptr) {
      return LSE_ERROR(kInternal, "no backend for the KV step descriptor");
    }
    const std::size_t bytes =
        dtype_storage_bytes(DType::kF32, static_cast<std::size_t>(meta_elems));
    auto buf = sched->backend().allocate(bytes, backend::MemoryClass::kDevice);
    if (!buf.ok()) return buf.status();
    cache_.meta =
        Array::from_buffer(buf.release(), Shape{meta_elems}, DType::kF32);
  }
  LSE_RETURN_IF_ERROR(poke_values(cache_.meta, meta));
  if (states != nullptr) {
    // Assigned every rebuild, not only the first: a bucket change replaces the
    // slot, and a layer still pointing at the old one would read another
    // width's positions.
    for (MixerState& s : *states) s.kv_meta = cache_.meta;
  }

  std::vector<StateSnap> before;
  if (states != nullptr) {
    before.reserve(states->size());
    for (const MixerState& s : *states) before.push_back(snap_state(s));
  }

  LSE_ASSIGN_OR(Array x, embed(tokens));
  if (trace != nullptr) trace->reserve(blocks_.size());

  const std::size_t shards = state_shards();
  if (shards > 1) {
    // The activation is carried on every member from the embedding to the
    // head, so a layer's branches read their input locally and only the
    // partial sums cross. Collapsing to one member between layers would put
    // the whole residual stream on one device and make the crossing the step.
    // Every member starts from the one embedding; each fetches it once, and
    // from the first layer on each is producing its own.
    std::vector<Array> xs(shards, x);
    for (std::size_t i = 0; i < blocks_.size(); ++i) {
      LayerContext ctx;
      ctx.config = &config_;
      ctx.layer_index = static_cast<std::int32_t>(i);
      ctx.shards = static_cast<std::int32_t>(shards);
      MixerState* state =
          states != nullptr ? &(*states)[i * shards] : nullptr;
      LSE_ASSIGN_OR(xs, blocks_[i]->forward_shards(xs, state, aux_loss, ctx));
      if (trace != nullptr) trace->push_back(xs[0]);
    }
    x = xs[0];
  } else {
  for (std::size_t i = 0; i < blocks_.size(); ++i) {
    LayerContext ctx;
    ctx.config = &config_;
    ctx.layer_index = static_cast<std::int32_t>(i);
    ctx.shards = static_cast<std::int32_t>(state_shards());
    MixerState* state =
        states != nullptr ? &(*states)[i * state_shards()] : nullptr;
    const graph::ScopedMember on(
        member_for_layer(static_cast<std::int32_t>(i)));
    LSE_ASSIGN_OR(x, blocks_[i]->forward(x, state, aux_loss, ctx));
    if (trace != nullptr) trace->push_back(x);
  }
  }

  // One eval for the residual and every mixer cache, so the 20 blocks stay
  // on the device queue instead of draining it after each layer. Sibling
  // state tensors are not ancestors of x; they have to be roots or the
  // next token would recompute them from whatever intermediates survived.
  Array y = spec_.zero_centered_norm
                ? ops::rms_norm_zero_centered(x, final_norm_weight_,
                                              config_.rms_eps)
                : ops::rms_norm(x, final_norm_weight_, config_.rms_eps);

  std::vector<graph::NodePtr> roots;
  roots.push_back(y.node());
  // A traced block output is only read after the whole stack has run, and the
  // slot planner recycles any buffer whose in-graph consumers are done. Held
  // Array handles are not consumers, so without this every trace entry aliases
  // one slot and reads back the last block's activations.
  if (trace != nullptr) {
    for (const Array& t : *trace) {
      if (t.valid() && t.node()) roots.push_back(t.node());
    }
  }
  // The state nodes that become a carry `out` below. Program::fold_carries
  // hands that buffer to the carry `in` node, where it has to survive a whole
  // replayed pass — but `in` is foreign to the retained slot plan, which never
  // promised the bytes to anyone for that long. So a carry endpoint has to own
  // its buffer outright. Workgroup::plan_slots recycles dead activation slots
  // by exact byte size, and a GDN state is batch*v_heads*head_dim^2 f32
  // whatever the pass width is — 1 MiB on the shipped models, which is exactly
  // what a (1, 128, 2048) f32 block activation costs at a prefill width of
  // 128. Without this the planner hands the state a slot six live activations
  // also hold, and the next pass reads its recurrence out of bytes it
  // overwrites before the scan runs.
  std::vector<graph::NodePtr> carried;
  if (states != nullptr) {
    auto add = [&](const Array& a, bool is_carry) {
      if (!a.valid() || !a.node()) return;
      if (!a.node()->materialized) roots.push_back(a.node());
      if (is_carry) carried.push_back(a.node());
    };
    for (MixerState& st : *states) {
      add(st.gdn_state, true);
      add(st.gdn_conv_q, true);
      add(st.gdn_conv_k, true);
      add(st.gdn_conv_v, true);
      add(st.gdn_conv_qkv, true);
      // The same condition the carry list below uses: a paged pool is written
      // in place and is never carried.
      const bool grows = !st.paged.valid();
      add(st.key_cache, grows);
      add(st.value_cache, grows);
    }
  }
  if (graph::Scheduler* sched = graph::default_scheduler()) {
    // Before eval, because plan_slots leaves any node that already owns a
    // buffer out of its free list. The skips mirror the planner's own: a
    // reshape, a leaf and an in-place primitive all take their bytes from
    // somewhere else by design.
    backend::IDeviceSet& set = sched->devices();
    for (const graph::NodePtr& n : carried) {
      if (n->buffer.valid() || n->kind == graph::OpKind::kReshape) continue;
      if (n->fclass == graph::FusionClass::kLeaf) continue;
      if (n->prim != nullptr && n->prim->inplace_input() >= 0) continue;
      // On the member the state was stamped with when it was made, through
      // that member's stream — on a set that places by stream, the stream IS
      // the member. The primary's backend with no stream put every shard's
      // recurrence in the primary's VRAM, and the other members' scans read
      // and wrote it across the link every token.
      const std::size_t member =
          n->member != graph::Node::kAnyMember &&
                  static_cast<std::size_t>(n->member) < set.size()
              ? static_cast<std::size_t>(n->member)
              : set.primary();
      backend::IBackend& on = set.device(member);
      const backend::Stream at =
          set.stream_for(member).value_or(backend::kDefaultStream);
      LSE_RETURN_IF_ERROR(
          graph::interpreter::ensure_output_buffer(*n, on, at));
    }
    LSE_RETURN_IF_ERROR(sched->eval(roots, false, &cache_.program));
  }
  last_pass_host_groups_ = host_groups_so_far() - host_before;

  cache_.tokens = tokens;
  cache_.hidden = y;
  cache_.states = states;
  cache_.seq = t_now;
  cache_.pass_id = ++pass_counter_;
  cache_.prev_pass = prev_pass_run;
  last_pass_id_ = cache_.pass_id;
  cache_.kv_leaves.clear();
  if (states != nullptr) {
    cache_.kv_leaves.reserve(states->size() * 2);
    for (MixerState& st : *states) {
      cache_.kv_leaves.push_back(
          st.key_cache.valid() ? st.key_cache.node().get() : nullptr);
      cache_.kv_leaves.push_back(
          st.value_cache.valid() ? st.value_cache.node().get() : nullptr);
      if (st.key_cache.valid()) {
        st.position = kv_len;
      }
    }
  }
  cache_.state_stamp.clear();
  if (states != nullptr) {
    cache_.state_stamp.reserve(states->size());
    for (const MixerState& st : *states) {
      cache_.state_stamp.push_back({st.gdn_state, st.gdn_conv_q,
                                    st.gdn_conv_k, st.gdn_conv_v,
                                    st.gdn_conv_qkv, st.key_cache,
                                    st.value_cache, st.kv_meta});
    }
  }
  if (states != nullptr && before.size() == states->size()) {
    std::vector<graph::Program::Carry> carries;
    for (std::size_t i = 0; i < states->size(); ++i) {
      add_carry(carries, before[i].gdn, (*states)[i].gdn_state);
      add_carry(carries, before[i].cq, (*states)[i].gdn_conv_q);
      add_carry(carries, before[i].ck, (*states)[i].gdn_conv_k);
      add_carry(carries, before[i].cv, (*states)[i].gdn_conv_v);
      add_carry(carries, before[i].cqkv, (*states)[i].gdn_conv_qkv);
      // Only the growing-concat path, where key_cache IS the concat node and has
      // to become the next step's input. A paged pool is a leaf the write aliases
      // in place, so there is nothing to carry — and carrying it is actively
      // wrong: Program::fold_carries swaps the two nodes' buffers, so a pool that
      // moved to a bigger rung would be handed back the smaller buffer it grew
      // out of. Five consecutive 128-token prefill passes replay the same
      // program, which is how that lands on a real prompt.
      if (!(*states)[i].paged.valid()) {
        add_carry(carries, before[i].keys, (*states)[i].key_cache);
        add_carry(carries, before[i].values, (*states)[i].value_cache);
      }
    }
    cache_.program.set_carries(std::move(carries));
  }
  return y;
}

Result<Array> HybridLM::lm_head(const Array& hidden_states) const {
  const Array& w =
      lm_head_weight_.valid() ? lm_head_weight_ : embed_weight_;
  if (!w.valid()) {
    return LSE_ERROR(kInternal, "HybridLM::lm_head before load()");
  }
  return graph::linear(hidden_states, w);
}

}  // namespace lse::model
