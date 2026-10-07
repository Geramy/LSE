#include "lse/model/mtp.hpp"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <utility>

#include "lse/graph/interpreter.hpp"
#include "lse/graph/ops.hpp"
#include "lse/kv/block.hpp"
#include "lse/model/qwen3_5_common.hpp"
#include "lse/model/weights.hpp"
#include "lse/ops/attention.hpp"
#include "lse/ops/norm.hpp"

namespace lse::model {

namespace {

using graph::Array;

Status poke(Array& slot, std::span<const float> values) {
  if (!slot.valid()) return LSE_ERROR(kInvalidArgument, "poke on empty Array");
  graph::Node& dst = *slot.node();
  if (dst.element_count() != values.size()) {
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
  dst.materialized = true;
  if (dst.dtype == DType::kF32 && !dst.kv_fragments && dst.buffer.ptr == nullptr) {
    // A pass's tokens and step descriptor: queued behind the work that may
    // still read the slot instead of waiting for the device first.
    LSE_RETURN_IF_ERROR(sched->backend().write_ordered(dst.buffer, values.data(),
                                                       values.size_bytes(), 0));
    dst.host_dirty = false;
    dst.device_dirty = false;
    return OkStatus();
  }
  dst.host_dirty = true;
  dst.device_dirty = false;
  return graph::interpreter::sync_to_device(dst, sched->backend());
}

Result<Array> hold_hidden(Array hidden) {
  LSE_RETURN_IF_ERROR(hidden.materialize());
  graph::Scheduler* sched = graph::default_scheduler();
  if (sched == nullptr) return LSE_ERROR(kInternal, "no backend for MTP hidden");
  graph::Node& src = *hidden.node();
  if (!src.buffer.valid()) {
    return LSE_ERROR(kInternal, "MTP hidden has no materialized buffer");
  }
  const auto member = sched->devices().member_of(src.buffer.residency);
  if (src.buffer.residency.bound() && member >= sched->devices().size()) {
    return LSE_ERROR(kInvalidArgument, "MTP hidden belongs to another device set");
  }
  auto& owner = member < sched->devices().size()
                    ? sched->devices().device(member) : sched->backend();
  LSE_RETURN_IF_ERROR(graph::interpreter::sync_to_device(src, owner));
  // A replay can reset the producing node; this leaf holds its finished bytes.
  return Array::from_buffer(src.buffer, src.shape, src.dtype);
}

Status poke(Array& slot, const Array& hidden) {
  graph::Scheduler* sched = graph::default_scheduler();
  if (sched == nullptr) return LSE_ERROR(kInternal, "no backend for MTP hidden");
  graph::Node& dst = *slot.node();
  graph::Node& src = *hidden.node();
  const std::size_t bytes = dtype_storage_bytes(dst.dtype, dst.element_count());
  if (src.dtype != dst.dtype || src.element_count() != dst.element_count() ||
      bytes > src.buffer.size_bytes) {
    return LSE_ERROR(kInvalidArgument, "MTP hidden does not fit its input slot");
  }
  LSE_RETURN_IF_ERROR(
      graph::interpreter::ensure_output_buffer(dst, sched->backend()));
  auto& be = sched->backend();
  Status copied = LSE_ERROR(kUnimplemented, "MTP hidden needs a host transfer");
  if (src.buffer.residency == be.device_index()) {
    if (sched->devices().size() == 1 && src.buffer.member == 0 &&
        dst.buffer.member == 0 && !be.stream_capabilities().may_spread()) {
      copied = be.copy_ordered(src.buffer, dst.buffer, bytes, 0, 0);
    } else {
      copied = be.copy({dst.buffer}, {src.buffer}, bytes);
    }
  }
  if (copied.ok()) {
    dst.host_dirty = false;
    dst.device_dirty = dst.buffer.ptr == nullptr;
    dst.materialized = true;
    return OkStatus();
  }
  if (copied.code() != StatusCode::kUnimplemented) return copied;

  const auto member = sched->devices().member_of(src.buffer.residency);
  auto& owner = member < sched->devices().size()
                    ? sched->devices().device(member) : be;
  LSE_RETURN_IF_ERROR(graph::interpreter::sync_from_device(src, owner));
  std::vector<float> values(src.element_count());
  LSE_RETURN_IF_ERROR(graph::interpreter::read_raw(src, values.data(), bytes));
  return poke(slot, std::span<const float>(values));
}

// A pass's tokens: uploaded into its own slot, or (a chained one-row pass)
// read where the previous pass left its pick. The pick is written after this
// pass's embedding has read the token, in stream order.
template <class PassT>
Status feed_tokens(PassT& pass, std::span<const float> ids, const Array* device_token) {
  graph::Node& slot = *pass.tokens.node();
  if (device_token == nullptr) {
    if (pass.token_slot.valid()) slot.buffer = pass.token_slot;
    return poke(pass.tokens, ids);
  }
  const graph::Node& from = *device_token->node();
  slot.buffer = from.buffer;
  slot.member = from.member;
  slot.materialized = true;
  slot.host_dirty = false;
  slot.device_dirty = true;
  return OkStatus();
}

Result<Array> device_slot(Shape shape) {
  graph::Scheduler* sched = graph::default_scheduler();
  if (sched == nullptr) {
    return LSE_ERROR(kInternal, "no backend to hold an MTP input");
  }
  const std::size_t bytes = dtype_storage_bytes(
      DType::kF32, static_cast<std::size_t>(shape.elem_count()));
  auto buf = sched->backend().allocate(bytes, backend::MemoryClass::kDevice);
  if (!buf.ok()) return buf.status();
  return Array::from_buffer(buf.release(), std::move(shape), DType::kF32);
}

std::string read_file(const std::string& path) {
  std::ifstream in(path);
  if (!in) return {};
  std::ostringstream buf;
  buf << in.rdbuf();
  return buf.str();
}

// The parent's geometry the module has to agree with. It is loaded against the
// parent's Config, so a module built for another width would bind tensors of
// the wrong shape — expect_shape catches most of it, but the head counts reach
// the kernels rather than the binder.
Status same_geometry(const Config& parent, const Config& module) {
  struct Field {
    const char* name;
    std::int32_t want, have;
  };
  const Field fields[] = {
      {"hidden_size", parent.hidden_size, module.hidden_size},
      {"vocab_size", parent.vocab_size, module.vocab_size},
      {"num_attention_heads", parent.attn_q_heads, module.attn_q_heads},
      {"num_key_value_heads", parent.attn_kv_heads, module.attn_kv_heads},
      {"head_dim", parent.attn_head_dim, module.attn_head_dim},
      {"rope_dim", parent.rope_dim, module.rope_dim},
      {"intermediate_size", parent.mlp_intermediate, module.mlp_intermediate},
  };
  for (const Field& f : fields) {
    if (f.want == f.have) continue;
    return LSE_ERROR(kInvalidArgument, "the MTP module's ", f.name, " is ",
                     std::to_string(f.have), " but the decoder's is ",
                     std::to_string(f.want));
  }
  if (parent.rope_theta != module.rope_theta) {
    return LSE_ERROR(kInvalidArgument,
                     "the MTP module and the decoder disagree on rope_theta");
  }
  return OkStatus();
}

// mlx-community/Qwen3.8-27B-4bit -> mlx-community/Qwen3.8-27B-MTP-4bit.
// The quantization suffix is last, so the marker goes before it; a name with
// no such suffix just takes it at the end.
std::vector<std::string> mtp_repo_names(const std::string& model) {
  std::vector<std::string> out;
  const std::size_t dash = model.find_last_of('-');
  if (dash != std::string::npos && dash + 1 < model.size()) {
    out.push_back(model.substr(0, dash) + "-MTP" + model.substr(dash));
  }
  out.push_back(model + "-MTP");
  return out;
}

}  // namespace

std::string MtpModule::find_beside(const std::string& model_name) {
  std::error_code ec;
  if (auto paths = resolve_model(model_name); paths.ok()) {
    const std::filesystem::path dir =
        std::filesystem::path(paths->config).parent_path() / "mtp";
    if (std::filesystem::is_directory(dir, ec)) return dir.string();
  }
  for (const std::string& name : mtp_repo_names(model_name)) {
    if (resolve_model(name).ok()) return name;
  }
  return {};
}

Result<std::unique_ptr<MtpModule>> MtpModule::open(const std::string& path,
                                                   const Config& parent,
                                                   HybridLM& model) {
  if (parent.mtp_dedicated_embeddings) {
    return LSE_ERROR(kUnimplemented,
                     "this checkpoint's MTP module keeps its own embedding "
                     "table; only the shared layout is implemented");
  }
  LSE_ASSIGN_OR(const ModelPaths paths, resolve_model(path));
  const std::string config_text = read_file(paths.config);
  if (config_text.empty()) {
    return LSE_ERROR(kNotFound, "the MTP module at '", path,
                     "' has no readable config beside it");
  }
  LSE_ASSIGN_OR(const Config declared, Config::from_json_string(config_text));
  LSE_RETURN_IF_ERROR(same_geometry(parent, declared));

  auto mtp = std::unique_ptr<MtpModule>(new MtpModule);
  mtp->path_ = paths.weights;
  mtp->model_ = &model;
  mtp->config_ = parent;
  mtp->config_.quantization = declared.quantization;
  // One layer, and it attends: is_attention_layer() is (i + 1) % interval, so
  // an interval of 1 is what makes layer 0 the full-attention kind the module's
  // self_attn tensors describe.
  mtp->config_.num_layers = 1;
  mtp->config_.full_attention_interval = 1;
  mtp->config_.global_attention_layers.clear();

  LSE_ASSIGN_OR(SafeTensors weights,
                paths.weights.ends_with(".index.json")
                    ? SafeTensors::open_sharded(paths.weights)
                    : SafeTensors::open(paths.weights));
  WeightBinder binder(weights, &mtp->config_.quantization);
  LSE_RETURN_IF_ERROR(mtp->build(binder));
  LSE_RETURN_IF_ERROR(binder.finish("MTP"));

  const std::vector<std::string> unclaimed = binder.unclaimed();
  if (!unclaimed.empty()) {
    std::string names;
    for (std::size_t i = 0; i < unclaimed.size() && i < 8; ++i) {
      if (i != 0) names += ", ";
      names += unclaimed[i];
    }
    return LSE_ERROR(kInvalidArgument, "the MTP module left ",
                     std::to_string(unclaimed.size()),
                     " tensor(s) unclaimed: ", names);
  }
  return mtp;
}

Status MtpModule::build(WeightBinder& binder) {
  const auto hidden = static_cast<std::int64_t>(config_.hidden_size);
  LSE_ASSIGN_OR(fc_, binder.require("fc.weight"));
  LSE_RETURN_IF_ERROR(
      qwen3_5::expect_shape(fc_, "fc.weight", Shape{hidden, 2 * hidden}));
  LSE_ASSIGN_OR(pre_norm_hidden_, binder.require("pre_fc_norm_hidden.weight"));
  LSE_ASSIGN_OR(pre_norm_embedding_,
                binder.require("pre_fc_norm_embedding.weight"));
  LSE_ASSIGN_OR(final_norm_, binder.require("norm.weight"));

  block_ = std::make_unique<HybridBlock>(
      qwen3_5::make_attention(), qwen3_5::make_mlp(),
      /*zero_centered_norm=*/false, /*mod=*/nullptr, qwen3_5::block_spec());
  LayerContext ctx;
  ctx.config = &config_;
  ctx.layer_index = 0;
  // The draft head is one block with one mixer state, so it stays whole even
  // when the model it drafts for is split across the pool.
  const graph::ScopedSplitScheme unsplit(graph::SplitScheme::kNone);
  return block_->load(binder, "layers.0", ctx);
}

Status MtpModule::truncate(std::int32_t position) {
  if (position < 0 || position > position_) {
    return LSE_ERROR(kOutOfRange, "MTP truncation is outside the cached prefix");
  }
  state_.position = position;
  state_.paged.row_tokens.assign(1, position);
  position_ = position;
  ++revision_;
  return OkStatus();
}

void MtpModule::reset() {
  if (state_.paged.valid()) {
    const Status s = ops::release_row(state_.paged, 0);
    (void)s;  // A pool with nothing in it is the state reset() is producing.
  }
  state_ = MixerState{};
  pass_ = Pass{};
  passes_.clear();
  chain_picks_ = Array{};
  position_ = 0;
  ++revision_;
}

Result<Array> MtpModule::record(std::int64_t rows) {
  const auto hidden = static_cast<std::int64_t>(config_.hidden_size);
  LSE_ASSIGN_OR(pass_.hidden, device_slot(Shape{1, rows, hidden}));
  LSE_ASSIGN_OR(pass_.tokens, device_slot(Shape{1, rows}));

  LSE_ASSIGN_OR(Array embedded, model_->embed(pass_.tokens));
  // Embedding first. fc is one [hidden, 2 * hidden] weight, so the halves are
  // not interchangeable and swapping them still produces fluent drafts that
  // are almost never the decoder's own token — which reads as speculation not
  // paying rather than as a bug.
  Array x = graph::linear(
      graph::concat({ops::rms_norm(embedded, pre_norm_embedding_,
                                   config_.rms_eps),
                     ops::rms_norm(pass_.hidden, pre_norm_hidden_,
                                   config_.rms_eps)},
                    -1),
      fc_);

  LayerContext ctx;
  ctx.config = &config_;
  ctx.layer_index = 0;
  ctx.attention_phase = ops::AttentionExecutionPhase::kSpeculative;
  LSE_ASSIGN_OR(x, block_->forward(x, &state_, nullptr, ctx));
  x = ops::rms_norm(x, final_norm_, config_.rms_eps);

  // Only the last row proposes: the earlier rows of a pass are there to put
  // their own positions in the module's KV, and running the 248k-wide head over
  // them would cost more than the module itself.
  Array last_rows = graph::slice(x, 1, rows - 1, rows);
  // Kept as a pass output: a CHAIN of drafts feeds this back in as the next
  // row's hidden, because the decoder has not run at the drafted position and
  // the module's own representation is the only hidden that exists there.
  // Keep the slice materialized so replay cannot recycle the chained input.
  pass_.last = last_rows;
  Array last = graph::reshape(last_rows, Shape{1, hidden});
  LSE_ASSIGN_OR(Array logits, model_->lm_head(last));
  Array pick = graph::argmax(logits);
  if (!pick.valid()) return LSE_ERROR(kInternal, "argmax over an empty row");
  if (scored_) {
    // The pick's probability: the row's largest softmax value, reduced on
    // the device (softmax_top) so only one float joins the readback.
    const auto vocab = static_cast<float>(logits.shape().dim(logits.shape().rank() - 1));
    LSE_ASSIGN_OR(Array partial, graph::custom("softmax_top.partial", {logits},
                                               {vocab, 0.0f, 0.0f, 0.0f}));
    LSE_ASSIGN_OR(pass_.top, graph::custom("softmax_top.final", {partial},
        {static_cast<float>(partial.shape().dim(1)), 0.0f, 0.0f, 0.0f}));
  }
  return pick;
}

Result<std::uint32_t> MtpModule::draft_pass(
    std::span<const float> hidden, std::span<const std::uint32_t> tokens,
    std::int32_t first, const Array* device_hidden) {
  LSE_RETURN_IF_ERROR(submit_pass(hidden, tokens, first, device_hidden, nullptr));
  graph::Scheduler* sched = graph::default_scheduler();
  if (sched == nullptr) return LSE_ERROR(kInternal, "no backend to run the MTP module");
  LSE_RETURN_IF_ERROR(graph::interpreter::sync_from_device(
      *pass_.pick.node(), sched->backend()));
  return static_cast<std::uint32_t>(
      graph::interpreter::load_element(*pass_.pick.node(), 0));
}

Status MtpModule::submit_pass(
    std::span<const float> hidden, std::span<const std::uint32_t> tokens,
    std::int32_t first, const Array* device_hidden, const Array* device_token) {
  if (device_token != nullptr &&
      (tokens.size() != 1 || !device_token->valid() ||
       device_token->dtype() != DType::kF32 || device_token->shape().elem_count() != 1 ||
       !device_token->node()->buffer.valid()))
    return LSE_ERROR(kInvalidArgument, "a chained MTP pass takes one device token");
  if (tokens.empty()) {
    return LSE_ERROR(kInvalidArgument, "an MTP pass needs at least one row");
  }
  const auto rows = static_cast<std::int64_t>(tokens.size());
  const auto width = static_cast<std::size_t>(config_.hidden_size);
  const std::size_t hidden_count = device_hidden != nullptr && device_hidden->valid()
                                       ? device_hidden->shape().elem_count()
                                       : hidden.size();
  if ((device_hidden != nullptr &&
       (!device_hidden->valid() || device_hidden->dtype() != DType::kF32)) ||
      hidden_count != tokens.size() * width) {
    return LSE_ERROR(kInvalidArgument, "an MTP pass of ",
                     std::to_string(tokens.size()), " row(s) wants ",
                     std::to_string(tokens.size() * width),
                     " hidden floats, got ", std::to_string(hidden_count));
  }
  if (first < 0) {
    return LSE_ERROR(kInvalidArgument, "an MTP pass cannot start at ",
                     std::to_string(first));
  }
  const auto after = static_cast<std::int32_t>(first + rows);
  if (after > config_.kv_capacity()) {
    return LSE_ERROR(kOutOfRange, "the MTP module would reach KV position ",
                     std::to_string(after), ", past the engine length ",
                     std::to_string(config_.kv_capacity()));
  }

  ++revision_;
  Array held;
  if (device_hidden != nullptr) {
    LSE_ASSIGN_OR(held, hold_hidden(*device_hidden));
  }

  std::vector<float> meta(static_cast<std::size_t>(kv::step_meta_elems(1)),
                          0.0f);
  meta[0] = static_cast<float>(first);
  meta[1] = static_cast<float>(after);
  meta[2] = 1.0f;
  meta[kv::kStepMetaHeader] = static_cast<float>(first);
  meta[kv::kStepMetaHeader + 1] = static_cast<float>(after);

  state_.paged.row_tokens.assign(1, after);
  bool pool_moved = false;
  if (state_.paged.valid()) {
    LSE_ASSIGN_OR(pool_moved, ops::extend_paged(state_.paged, after));
  }

  if (pass_.rows != rows && pass_.rows != 0) {
    Pass parked = std::move(passes_[rows]);
    passes_[pass_.rows] = std::move(pass_);
    pass_ = std::move(parked);
  }
  const bool leaves_match =
      pass_.keys == (state_.key_cache.valid() ? state_.key_cache.node().get()
                                              : nullptr) &&
      pass_.values == (state_.value_cache.valid()
                           ? state_.value_cache.node().get()
                           : nullptr);
  const bool reuse = !pool_moved && pass_.rows == rows &&
                     pass_.pick.valid() && !pass_.program.empty() &&
                     leaves_match;

  graph::Scheduler* sched = graph::default_scheduler();
  if (sched == nullptr) {
    return LSE_ERROR(kInternal, "no backend to run the MTP module");
  }

  std::vector<float> ids(tokens.size());
  for (std::size_t i = 0; i < tokens.size(); ++i) {
    ids[i] = static_cast<float>(tokens[i]);
  }

  state_.position = first;
  if (reuse) {
    pass_.program.reset_compute();
    LSE_RETURN_IF_ERROR(held.valid() ? poke(pass_.hidden, held)
                                      : poke(pass_.hidden, hidden));
    LSE_RETURN_IF_ERROR(feed_tokens(pass_, ids, device_token));
    LSE_RETURN_IF_ERROR(poke(pass_.meta, meta));
    LSE_RETURN_IF_ERROR(
        sched->eval(pass_.program.roots(), false, &pass_.program));
  } else {
    pass_ = Pass{};
    pass_.rows = rows;
    LSE_ASSIGN_OR(pass_.meta,
                  device_slot(Shape{kv::step_meta_elems(1)}));
    state_.kv_meta = pass_.meta;
    LSE_RETURN_IF_ERROR(poke(pass_.meta, meta));
    LSE_ASSIGN_OR(pass_.pick, record(rows));
    LSE_RETURN_IF_ERROR(held.valid() ? poke(pass_.hidden, held)
                                      : poke(pass_.hidden, hidden));
    pass_.token_slot = pass_.tokens.node()->buffer;
    LSE_RETURN_IF_ERROR(feed_tokens(pass_, ids, device_token));

    std::vector<graph::NodePtr> roots{pass_.pick.node()};
    if (pass_.last.valid()) roots.push_back(pass_.last.node());
    if (pass_.top.valid()) roots.push_back(pass_.top.node());
    for (const Array& a : {state_.key_cache, state_.value_cache}) {
      if (a.valid() && a.node() && !a.node()->materialized) {
        roots.push_back(a.node());
      }
    }
    LSE_RETURN_IF_ERROR(sched->eval(roots, false, &pass_.program));
    pass_.keys =
        state_.key_cache.valid() ? state_.key_cache.node().get() : nullptr;
    pass_.values =
        state_.value_cache.valid() ? state_.value_cache.node().get() : nullptr;
  }

  state_.position = after;
  position_ = after;
  return OkStatus();
}

Result<std::uint32_t> MtpModule::draft(std::span<const float> hidden,
                                       std::span<const std::uint32_t> tokens,
                                       std::int32_t first) {
  return draft_pass(hidden, tokens, first);
}

Result<std::uint32_t> MtpModule::draft(
    const Array& hidden, std::span<const std::uint32_t> tokens,
    std::int32_t first) {
  return draft_pass({}, tokens, first, &hidden);
}

Result<std::vector<std::uint32_t>> MtpModule::draft_chain(
    std::span<const float> hidden, std::span<const std::uint32_t> tokens,
    std::int32_t first, std::uint32_t depth) {
  return draft_chain_impl(hidden, nullptr, tokens, first, depth, nullptr);
}

Result<std::vector<std::uint32_t>> MtpModule::draft_chain(
    const Array& hidden, std::span<const std::uint32_t> tokens,
    std::int32_t first, std::uint32_t depth, std::vector<double>* confidence) {
  return draft_chain_impl({}, &hidden, tokens, first, depth, confidence);
}

void MtpModule::set_scored(bool scored) {
  if (scored == scored_) return;
  if (auto* sched = graph::default_scheduler()) (void)sched->drain();
  scored_ = scored;
  pass_ = Pass{};
  passes_.clear();
}

Result<std::vector<std::uint32_t>> MtpModule::draft_chain_impl(
    std::span<const float> hidden, const Array* device_hidden,
    std::span<const std::uint32_t> tokens, std::int32_t first,
    std::uint32_t depth, std::vector<double>* confidence) {
  std::vector<std::uint32_t> out;
  if (depth == 0) return out;
  if (depth > kChainSlots)
    return LSE_ERROR(kInvalidArgument, "an MTP chain holds at most ", std::to_string(kChainSlots),
                     " proposals, asked for ", std::to_string(depth));
  if (confidence != nullptr && !scored_)
    return LSE_ERROR(kInvalidArgument, "MTP confidence needs a scored module (set_scored)");
  graph::Scheduler* sched = graph::default_scheduler();
  if (sched == nullptr) return LSE_ERROR(kInternal, "no backend to run the MTP module");
  auto& be = sched->backend();
  // Every pass feeds the next its proposal and hidden on the device; their
  // picks are copied, in stream order, into one buffer read back once.
  if (!chain_picks_.valid()) {
    LSE_ASSIGN_OR(chain_picks_, device_slot(Shape{2 * kChainSlots}));
  }
  auto gather = [&](std::uint32_t i) -> Status {
    const auto& pick = pass_.pick.node()->buffer;
    LSE_RETURN_IF_ERROR(be.copy_ordered(pick, chain_picks_.node()->buffer, sizeof(float), 0,
                                        i * sizeof(float)));
    if (confidence == nullptr) return OkStatus();
    if (!pass_.top.valid() || !pass_.top.node()->buffer.valid())
      return LSE_ERROR(kInternal, "the MTP pass kept no pick probability");
    return be.copy_ordered(pass_.top.node()->buffer, chain_picks_.node()->buffer, sizeof(float),
                           0, (kChainSlots + i) * sizeof(float));
  };
  LSE_RETURN_IF_ERROR(submit_pass(hidden, tokens, first, device_hidden, nullptr));
  LSE_RETURN_IF_ERROR(gather(0));
  for (std::uint32_t i = 1; i < depth; ++i) {
    if (!pass_.last.valid()) {
      return LSE_ERROR(kInternal, "the draft pass kept no hidden to chain on");
    }
    const Array own = pass_.last;
    const Array previous = pass_.pick;
    const std::uint32_t row = 0;
    LSE_RETURN_IF_ERROR(submit_pass({}, std::span(&row, 1), position_, &own, &previous));
    LSE_RETURN_IF_ERROR(gather(i));
  }
  std::vector<float> picks(confidence != nullptr ? 2 * kChainSlots : depth);
  LSE_RETURN_IF_ERROR(be.copy(picks.data(), backend::MemRef(chain_picks_.node()->buffer, 0),
                              picks.size() * sizeof(float)));
  if (confidence != nullptr) {
    confidence->clear();
    for (std::uint32_t i = 0; i < depth; ++i) {
      const float top = picks[kChainSlots + i];
      if (!(top > 0.0f && top <= 1.0f))
        return LSE_ERROR(kInternal, "MTP pick probability ", std::to_string(top),
                         " is not a probability");
      confidence->push_back(static_cast<double>(top));
    }
    picks.resize(depth);
  }
  out.reserve(depth);
  const auto vocab = static_cast<float>(config_.vocab_size);
  for (const float pick : picks) {
    if (!(pick >= 0.0f && pick < vocab) || pick != static_cast<float>(static_cast<std::uint32_t>(pick)))
      return LSE_ERROR(kInternal, "MTP proposal ", std::to_string(pick), " is not a token");
    out.push_back(static_cast<std::uint32_t>(pick));
  }
  return out;
}

}  // namespace lse::model
