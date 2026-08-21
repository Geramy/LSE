#include "lse/model/mtp.hpp"

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
  dst.host_dirty = true;
  dst.device_dirty = false;
  dst.materialized = true;
  return graph::interpreter::sync_to_device(dst, sched->backend());
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

void MtpModule::reset() {
  if (state_.paged.valid()) {
    const Status s = ops::release_row(state_.paged, 0);
    (void)s;  // A pool with nothing in it is the state reset() is producing.
  }
  state_ = MixerState{};
  pass_ = Pass{};
  position_ = 0;
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
  LSE_ASSIGN_OR(x, block_->forward(x, &state_, nullptr, ctx));
  x = ops::rms_norm(x, final_norm_, config_.rms_eps);

  // Only the last row proposes: the earlier rows of a pass are there to put
  // their own positions in the module's KV, and running the 248k-wide head over
  // them would cost more than the module itself.
  Array last = graph::reshape(graph::slice(x, 1, rows - 1, rows),
                              Shape{1, hidden});
  LSE_ASSIGN_OR(Array logits, model_->lm_head(last));
  Array pick = graph::argmax(logits);
  if (!pick.valid()) return LSE_ERROR(kInternal, "argmax over an empty row");
  return pick;
}

Result<std::uint32_t> MtpModule::draft(std::span<const float> hidden,
                                       std::span<const std::uint32_t> tokens,
                                       std::int32_t first) {
  if (tokens.empty()) {
    return LSE_ERROR(kInvalidArgument, "an MTP pass needs at least one row");
  }
  const auto rows = static_cast<std::int64_t>(tokens.size());
  const auto width = static_cast<std::size_t>(config_.hidden_size);
  if (hidden.size() != tokens.size() * width) {
    return LSE_ERROR(kInvalidArgument, "an MTP pass of ",
                     std::to_string(tokens.size()), " row(s) wants ",
                     std::to_string(tokens.size() * width),
                     " hidden floats, got ", std::to_string(hidden.size()));
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
    LSE_RETURN_IF_ERROR(poke(pass_.hidden, hidden));
    LSE_RETURN_IF_ERROR(poke(pass_.tokens, ids));
    LSE_RETURN_IF_ERROR(poke(pass_.meta, meta));
    LSE_RETURN_IF_ERROR(
        sched->eval(pass_.program.roots(), true, &pass_.program));
  } else {
    pass_ = Pass{};
    pass_.rows = rows;
    LSE_ASSIGN_OR(pass_.meta,
                  device_slot(Shape{kv::step_meta_elems(1)}));
    state_.kv_meta = pass_.meta;
    LSE_RETURN_IF_ERROR(poke(pass_.meta, meta));
    LSE_ASSIGN_OR(pass_.pick, record(rows));
    LSE_RETURN_IF_ERROR(poke(pass_.hidden, hidden));
    LSE_RETURN_IF_ERROR(poke(pass_.tokens, ids));

    std::vector<graph::NodePtr> roots{pass_.pick.node()};
    for (const Array& a : {state_.key_cache, state_.value_cache}) {
      if (a.valid() && a.node() && !a.node()->materialized) {
        roots.push_back(a.node());
      }
    }
    LSE_RETURN_IF_ERROR(sched->eval(roots, true, &pass_.program));
    pass_.keys =
        state_.key_cache.valid() ? state_.key_cache.node().get() : nullptr;
    pass_.values =
        state_.value_cache.valid() ? state_.value_cache.node().get() : nullptr;
  }

  state_.position = after;
  position_ = after;
  return static_cast<std::uint32_t>(
      graph::interpreter::load_element(*pass_.pick.node(), 0));
}

}  // namespace lse::model
