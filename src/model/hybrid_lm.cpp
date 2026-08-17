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

Status HybridLM::load(WeightBinder& binder) {
  LSE_ASSIGN_OR(embed_weight_, binder.require(spec_.embed_name));
  LSE_ASSIGN_OR(final_norm_weight_, binder.require(spec_.final_norm_name));
  if (!spec_.lm_head_name.empty()) {
    LSE_ASSIGN_OR(lm_head_weight_, binder.require(spec_.lm_head_name));
  }

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
    const std::string prefix = spec_.block_prefix + "." + std::to_string(i);
    LSE_RETURN_IF_ERROR(block->load(binder, prefix, ctx));
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

Result<Array> HybridLM::hidden(const Array& tokens,
                               std::vector<MixerState>* states, Array* aux_loss,
                               std::vector<Array>* trace, std::int32_t rows) {
  if (blocks_.empty()) {
    return LSE_ERROR(kInternal, "HybridLM::hidden before load()");
  }
  if (states != nullptr && states->size() != blocks_.size()) {
    return LSE_ERROR(kInvalidArgument, "expected ",
                     std::to_string(blocks_.size()), " mixer states, got ",
                     std::to_string(states->size()));
  }

  const std::int64_t t_now =
      tokens.valid() ? tokens.shape().dim(tokens.shape().rank() - 1) : 0;

  if (states != nullptr && tokens.valid()) {
    const auto batch = tokens.shape().dim(0);
    const auto vh = static_cast<std::int64_t>(
        spec_.gdn_state_heads > 0 ? spec_.gdn_state_heads : config_.gdn_qk_heads);
    const auto vd = static_cast<std::int64_t>(
        spec_.gdn_state_dim > 0 ? spec_.gdn_state_dim : config_.gdn_head_dim);
    const auto tail = config_.gdn_conv_kernel > 1
                          ? static_cast<std::int64_t>(config_.gdn_conv_kernel - 1)
                          : 0;
    const auto fused = static_cast<std::int64_t>(spec_.gdn_conv_width);
    const auto width = vh * vd;
    for (std::size_t i = 0; i < states->size(); ++i) {
      if (config_.is_attention_layer(static_cast<std::int32_t>(i))) continue;
      MixerState& st = (*states)[i];
      if (!st.gdn_state.valid()) {
        st.gdn_state = Array::zeros(Shape{batch, vh, vd, vd}, DType::kF32);
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

  std::int32_t kv_pos = 0;
  if (states != nullptr) {
    for (const MixerState& s : *states) {
      if (s.position > kv_pos) kv_pos = s.position;
    }
  }
  // The step descriptor the paged kernels read: where this pass's first query
  // sits, how much KV is live behind it, and how many of the padded batch rows
  // are real. All three are dispatch values, not shapes — which is what lets one
  // code object serve every position, every length and every batch up to the
  // bucket.
  const std::int32_t kv_len = kv_pos + static_cast<std::int32_t>(t_now);
  // Checked here rather than only where the pool is sized: the decode fast path
  // replays a held program and never re-enters the attention layer, so a session
  // running past the engine length would write past the last slot its block table
  // has instead of being refused.
  if (states != nullptr && kv_len > config_.kv_capacity()) {
    return LSE_ERROR(kOutOfRange, "this pass would reach KV position ",
                     std::to_string(kv_len), ", past the engine length ",
                     std::to_string(config_.kv_capacity()));
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
  const std::int32_t live_rows = rows > 0 ? rows : bucket;
  if (live_rows > bucket) {
    return LSE_ERROR(kInvalidArgument, "asked for ", std::to_string(live_rows),
                     " live rows in a pass of ", std::to_string(bucket));
  }
  const float meta[3] = {static_cast<float>(kv_pos), static_cast<float>(kv_len),
                         static_cast<float>(live_rows)};

  // Blocks first, on both paths: the retained decode program is not re-recorded,
  // yet it crosses a block boundary every kv::kBlockSize tokens and needs the
  // next block in the table before it runs. A pool that has to move to a bigger
  // rung cannot be absorbed by a replay — the buffer changes identity — so that
  // answer forces the rebuild below.
  bool pool_moved = false;
  if (states != nullptr) {
    for (MixerState& st : *states) {
      if (!st.paged.valid()) continue;
      LSE_ASSIGN_OR(const bool moved, ops::extend_paged(st.paged, kv_len));
      pool_moved = pool_moved || moved;
    }
  }

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
        return false;
      }
    }
    return true;
  };

  const bool can_reuse =
      aux_loss == nullptr && trace == nullptr && !pool_moved &&
      cache_.hidden.valid() && cache_.tokens.valid() && tokens.valid() &&
      cache_.meta.valid() &&
      tokens.shape().elem_count() == cache_.tokens.shape().elem_count() &&
      cache_.states == states && !cache_.program.empty() &&
      !cache_.program.groups().empty() && kv_leaves_match();
  if (can_reuse) {
    cache_.program.reset_compute();
    cache_.program.fold_carries();
    LSE_RETURN_IF_ERROR(poke_tokens(cache_.tokens, tokens));
    LSE_RETURN_IF_ERROR(poke_values(cache_.meta, meta));
    if (graph::Scheduler* sched = graph::default_scheduler()) {
      LSE_RETURN_IF_ERROR(
          sched->eval(cache_.program.roots(), false, &cache_.program));
    }
    if (states != nullptr) {
      for (MixerState& s : *states) {
        if (s.key_cache.valid()) {
          s.position = kv_pos + static_cast<std::int32_t>(t_now);
        }
      }
    }
    return cache_.hidden;
  }

  if (!cache_.meta.valid()) {
    graph::Scheduler* sched = graph::default_scheduler();
    if (sched == nullptr) {
      return LSE_ERROR(kInternal, "no backend for the KV step descriptor");
    }
    const std::size_t bytes = dtype_storage_bytes(DType::kF32, 3);
    auto buf = sched->backend().allocate(bytes, backend::MemoryClass::kDevice);
    if (!buf.ok()) return buf.status();
    cache_.meta = Array::from_buffer(buf.release(), Shape{3}, DType::kF32);
  }
  LSE_RETURN_IF_ERROR(poke_values(cache_.meta, meta));
  if (states != nullptr) {
    for (MixerState& s : *states) {
      if (!s.kv_meta.valid()) s.kv_meta = cache_.meta;
    }
  }

  std::vector<StateSnap> before;
  if (states != nullptr) {
    before.reserve(states->size());
    for (const MixerState& s : *states) before.push_back(snap_state(s));
  }

  LSE_ASSIGN_OR(Array x, embed(tokens));
  if (trace != nullptr) trace->reserve(blocks_.size());

  for (std::size_t i = 0; i < blocks_.size(); ++i) {
    LayerContext ctx;
    ctx.config = &config_;
    ctx.layer_index = static_cast<std::int32_t>(i);
    MixerState* state = states != nullptr ? &(*states)[i] : nullptr;
    LSE_ASSIGN_OR(x, blocks_[i]->forward(x, state, aux_loss, ctx));
    if (trace != nullptr) trace->push_back(x);
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
  if (states != nullptr) {
    auto add = [&](const Array& a) {
      if (a.valid() && a.node() && !a.node()->materialized) {
        roots.push_back(a.node());
      }
    };
    for (MixerState& st : *states) {
      add(st.gdn_state);
      add(st.gdn_conv_q);
      add(st.gdn_conv_k);
      add(st.gdn_conv_v);
      add(st.gdn_conv_qkv);
      add(st.key_cache);
      add(st.value_cache);
    }
  }
  if (graph::Scheduler* sched = graph::default_scheduler()) {
    LSE_RETURN_IF_ERROR(sched->eval(roots, false, &cache_.program));
  }

  cache_.tokens = tokens;
  cache_.hidden = y;
  cache_.states = states;
  cache_.seq = t_now;
  cache_.kv_leaves.clear();
  if (states != nullptr) {
    cache_.kv_leaves.reserve(states->size() * 2);
    for (MixerState& st : *states) {
      cache_.kv_leaves.push_back(
          st.key_cache.valid() ? st.key_cache.node().get() : nullptr);
      cache_.kv_leaves.push_back(
          st.value_cache.valid() ? st.value_cache.node().get() : nullptr);
      if (st.key_cache.valid()) {
        st.position = kv_pos + static_cast<std::int32_t>(t_now);
      }
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
