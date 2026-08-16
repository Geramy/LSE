// Model-independent layer machinery: weight binding and the generic block.
#include "lse/model/layer.hpp"

#include <algorithm>
#include <vector>

// host_bytes / sync_to_device: a checkpoint tensor is host data and Array has
// no host-write path, so the buffer is filled behind the graph.
#include "lse/graph/interpreter.hpp"
#include "lse/graph/ops.hpp"
#include "lse/ops/norm.hpp"

namespace lse::model {

Result<Array> WeightBinder::require(std::string_view name) {
  auto got = optional(name);
  if (got.ok()) return got;
  // Only a missing tensor becomes a "no such tensor" report; anything else
  // (no backend, allocation failure) would otherwise be misattributed.
  if (got.status().code() != StatusCode::kNotFound) return got.status();
  return LSE_ERROR(kNotFound, "checkpoint has no tensor '", std::string(name),
                   "'");
}

namespace {

// The dtype a checkpoint tensor is held in on the device. Float formats keep
// the format they were stored in — widening them in memory is a pure bandwidth
// tax on every token, and converting in register costs nothing. Integer and
// block-quantized storage has no device kernel that reads it directly yet, so
// it still widens here.
DType device_storage(DType checkpoint) noexcept {
  switch (checkpoint) {
    case DType::kF32:
    case DType::kF16:
    case DType::kBF16:
      return checkpoint;
    default:
      return DType::kF32;
  }
}

}  // namespace

Result<Array> WeightBinder::optional(std::string_view name) {
  const TensorView* v = weights_->find(name);
  if (v == nullptr) return LSE_ERROR(kNotFound, std::string(name));

  // A weight is data, not a computation. Going through eval() would dispatch a
  // fill kernel across every element and then overwrite the result one element
  // at a time — for the tied head that is a quarter of a billion pointless
  // writes on each side, and it dominated model load. Allocate the buffer and
  // read the tensor straight into it instead.
  graph::Scheduler* sched = graph::default_scheduler();
  if (sched == nullptr) {
    return LSE_ERROR(kInternal, "no usable backend to load '",
                     std::string(name), "' into");
  }
  backend::IBackend& be = sched->backend();

  const DType dt = device_storage(v->dtype);
  Array a = Array::zeros(v->shape, dt);
  graph::Node& n = *a.node();
  LSE_RETURN_IF_ERROR(graph::interpreter::ensure_output_buffer(n, be));
  if (dt == v->dtype) {
    LSE_RETURN_IF_ERROR(
        v->read_native(graph::interpreter::host_bytes(n),
                       dtype_storage_bytes(dt, n.element_count())));
  } else {
    LSE_RETURN_IF_ERROR(v->read_f32(
        static_cast<float*>(graph::interpreter::host_bytes(n)),
        n.element_count()));
  }
  // The mirror now holds the only copy; it is the authority until it is pushed.
  n.host_dirty = true;
  n.device_dirty = false;
  n.materialized = true;
  LSE_RETURN_IF_ERROR(graph::interpreter::sync_to_device(n, be));

  claimed_.emplace_back(name);
  return a;
}

std::vector<std::string> WeightBinder::unclaimed() const {
  std::vector<std::string> out;
  for (const auto& [name, _] : weights_->tensors()) {
    if (std::find(claimed_.begin(), claimed_.end(), name) == claimed_.end()) {
      out.push_back(name);
    }
  }
  return out;
}

Status HybridBlock::load(WeightBinder& binder, std::string_view prefix,
                         const LayerContext& ctx) {
  const std::string p(prefix);
  LSE_ASSIGN_OR(norm1_weight_, binder.require(p + ".norm1.weight"));
  LSE_ASSIGN_OR(norm2_weight_, binder.require(p + ".norm2.weight"));
  LSE_RETURN_IF_ERROR(mixer_->load(binder, prefix, ctx));
  LSE_RETURN_IF_ERROR(ffn_->load(binder, prefix, ctx));
  if (mod_) LSE_RETURN_IF_ERROR(mod_->load(binder, prefix));
  return OkStatus();
}

Result<Array> HybridBlock::forward(const Array& x, MixerState* state,
                                   Array* aux_loss, const LayerContext& ctx) {
  if (ctx.config == nullptr) {
    return LSE_ERROR(kInvalidArgument, "HybridBlock::forward needs a config");
  }
  const float eps = ctx.config->rms_eps;
  auto norm = [&](const Array& v, const Array& w) {
    return zero_centered_norm_ ? ops::rms_norm_zero_centered(v, w, eps)
                               : ops::rms_norm(v, w, eps);
  };

  LSE_ASSIGN_OR(Array mixed,
                mixer_->forward(norm(x, norm1_weight_), state, ctx));
  Array h = x + mixed;

  Array h2 = norm(h, norm2_weight_);
  LSE_ASSIGN_OR(Array ff, ffn_->forward(h2, aux_loss, ctx));
  if (mod_) {
    LSE_ASSIGN_OR(ff, mod_->gate_all(h2, ff));
  }
  LSE_ASSIGN_OR(Array dense, ffn_->ungated(h2));
  if (dense.valid()) ff = ff + dense;
  return h + ff;
}

}  // namespace lse::model
