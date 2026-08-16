#include "lse/model/hybrid_lm.hpp"

#include "lse/graph/interpreter.hpp"
#include "lse/graph/ops.hpp"
#include "lse/ops/norm.hpp"

namespace lse::model {

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
  return OkStatus();
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

Status poke_scalar(Array& slot, float value) {
  if (!slot.valid()) return LSE_ERROR(kInvalidArgument, "scalar poke on empty Array");
  graph::Node& dst = *slot.node();
  graph::Scheduler* sched = graph::default_scheduler();
  if (sched == nullptr) return LSE_ERROR(kInternal, "no backend for scalar poke");
  if (!dst.buffer.valid()) {
    LSE_RETURN_IF_ERROR(
        graph::interpreter::ensure_output_buffer(dst, sched->backend()));
  }
  const std::size_t bytes = dtype_storage_bytes(dst.dtype, dst.element_count());
  if (dst.host_mirror.size() < bytes) dst.host_mirror.resize(bytes);
  graph::interpreter::store_element(dst, 0, value);
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
                               std::vector<Array>* trace) {
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
    const auto vh = static_cast<std::int64_t>(config_.gdn_qk_heads);
    const auto vd = static_cast<std::int64_t>(config_.gdn_head_dim);
    const auto tail = config_.gdn_conv_kernel > 1
                          ? static_cast<std::int64_t>(config_.gdn_conv_kernel - 1)
                          : 0;
    const auto width = vh * vd;
    for (std::size_t i = 0; i < states->size(); ++i) {
      if (config_.is_attention_layer(static_cast<std::int32_t>(i))) continue;
      MixerState& st = (*states)[i];
      if (!st.gdn_state.valid()) {
        st.gdn_state = Array::zeros(Shape{batch, vh, vd, vd}, DType::kF32);
      }
      if (tail > 0 && width > 0) {
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
      aux_loss == nullptr && trace == nullptr && cache_.hidden.valid() &&
      cache_.tokens.valid() && tokens.valid() && cache_.pos.valid() &&
      tokens.shape().elem_count() == cache_.tokens.shape().elem_count() &&
      cache_.states == states && !cache_.program.empty() &&
      !cache_.program.groups().empty() && kv_leaves_match();
  if (can_reuse) {
    cache_.program.reset_compute();
    cache_.program.fold_carries();
    LSE_RETURN_IF_ERROR(poke_tokens(cache_.tokens, tokens));
    LSE_RETURN_IF_ERROR(
        poke_scalar(cache_.pos, static_cast<float>(kv_pos)));
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

  if (!cache_.pos.valid()) {
    graph::Scheduler* sched = graph::default_scheduler();
    if (sched == nullptr) {
      return LSE_ERROR(kInternal, "no backend for the KV position slot");
    }
    const std::size_t bytes = dtype_storage_bytes(DType::kF32, 1);
    auto buf = sched->backend().allocate(bytes, backend::MemoryClass::kDevice);
    if (!buf.ok()) return buf.status();
    cache_.pos = Array::from_buffer(buf.release(), Shape{1}, DType::kF32);
  }
  LSE_RETURN_IF_ERROR(poke_scalar(cache_.pos, static_cast<float>(kv_pos)));
  if (states != nullptr) {
    for (MixerState& s : *states) {
      if (!s.kv_pos.valid()) s.kv_pos = cache_.pos;
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
      add_carry(carries, before[i].keys, (*states)[i].key_cache);
      add_carry(carries, before[i].values, (*states)[i].value_cache);
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
