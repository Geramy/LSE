#include "lse/graph/graph.hpp"

#include <ostream>
#include <unordered_map>

#include "lse/graph/interpreter.hpp"
#include "lse/graph/view.hpp"

namespace lse::graph {

bool is_elementwise(OpKind k) noexcept {
  switch (k) {
    case OpKind::kAdd: case OpKind::kSub: case OpKind::kMul: case OpKind::kDiv:
    case OpKind::kNeg: case OpKind::kExp: case OpKind::kLog: case OpKind::kSqrt:
    case OpKind::kRsqrt: case OpKind::kSiLU: case OpKind::kGELU:
    case OpKind::kSigmoid: case OpKind::kTanh: case OpKind::kReLU:
    case OpKind::kCast: case OpKind::kClamp: case OpKind::kWhere:
    case OpKind::kSoftplus:
      return true;
    default:
      return false;
  }
}

bool is_reduction(OpKind k) noexcept {
  switch (k) {
    case OpKind::kSum: case OpKind::kMax: case OpKind::kMean:
    case OpKind::kRMS: case OpKind::kSoftmax: case OpKind::kLogSumExp:
    case OpKind::kL2Norm:
      return true;
    default:
      return false;
  }
}

bool is_structural(OpKind k) noexcept {
  switch (k) {
    case OpKind::kReshape: case OpKind::kTranspose: case OpKind::kBroadcast:
    case OpKind::kSlice: case OpKind::kConcat: case OpKind::kGather:
    case OpKind::kScatter: case OpKind::kRepeat: case OpKind::kConvTailShift:
      return true;
    default:
      return false;
  }
}

bool is_collective(OpKind k) noexcept {
  switch (k) {
    case OpKind::kAllReduce: case OpKind::kAllGather: case OpKind::kReduceScatter:
    case OpKind::kAllToAll: case OpKind::kBroadcastRank:
      return true;
    default:
      return false;
  }
}

bool is_barrier(OpKind k) noexcept {
  switch (k) {
    case OpKind::kCausalConv1d:
    case OpKind::kMatMul: case OpKind::kLinear: case OpKind::kQuantMatMul:
    case OpKind::kAttention: case OpKind::kGDNChunkScan: case OpKind::kMoEDispatch:
    case OpKind::kMoECombine: case OpKind::kEmbedding:
    case OpKind::kQuantEmbedding: case OpKind::kRoPE:
    case OpKind::kTopK: case OpKind::kArgMax: case OpKind::kOverwriteSlice:
    case OpKind::kKvPageWrite:
      return true;
    default:
      return is_collective(k);
  }
}


FusionClass fusion_class_of(OpKind k) noexcept {
  if (k == OpKind::kBuffer || k == OpKind::kConstant) return FusionClass::kLeaf;
  if (is_elementwise(k)) return FusionClass::kElementwise;
  if (is_reduction(k)) return FusionClass::kReduction;
  if (is_structural(k)) return FusionClass::kStructural;
  if (is_collective(k)) return FusionClass::kCollective;
  return FusionClass::kBarrier;
}

std::uint64_t Node::recompute_cost() const noexcept {
  if (materialized) return 0;
  const std::uint64_t elems = element_count();
  if (kind == OpKind::kConstant || is_structural(kind)) return 0;
  if (is_barrier(kind)) return elems * 64;
  if (is_reduction(kind)) return elems * 4;
  return elems;
}

Array Array::from_buffer(backend::DeviceBuffer buf, Shape shape, DType dtype) {
  auto n = std::make_shared<Node>();
  n->set_kind(OpKind::kBuffer);
  n->shape = shape;
  n->dtype = dtype;
  n->member = stamped_member();
  n->buffer = buf;
  n->device_dirty = buf.ptr == nullptr;
  n->materialized = true;
  return Array(n);
}

Array Array::zeros(Shape shape, DType dtype) { return full(shape, dtype, 0.0f); }

Array Array::full(Shape shape, DType dtype, float value) {
  auto n = std::make_shared<Node>();
  n->set_kind(OpKind::kConstant);
  n->shape = shape;
  n->dtype = dtype;
  n->member = stamped_member();
  n->attrs[0] = value;
  return Array(n);
}

namespace {

Status run_eval(const NodePtr& node, bool pull_host) {
  if (!node) return LSE_ERROR(kInvalidArgument, "eval on an empty Array");
  Scheduler* sched = default_scheduler();
  if (sched == nullptr) {
    return LSE_ERROR(kInternal,
                     "no usable backend: nothing is registered, or init failed");
  }
  const NodePtr roots[] = {node};
  return sched->eval(roots, pull_host);
}

}  // namespace

Status Array::eval() { return run_eval(node_, true); }

Status Array::materialize() { return run_eval(node_, false); }

Status Array::materialize_owned() {
  if (!node_) return LSE_ERROR(kInvalidArgument, "owned materialize on an empty Array");
  LSE_RETURN_IF_ERROR(materialize());
  auto* scheduler = default_scheduler();
  if (!scheduler) return LSE_ERROR(kInternal, "no backend for owned storage");
  auto& devices = scheduler->devices();
  const auto member = node_->member != Node::kAnyMember && node_->member < devices.size()
      ? static_cast<std::size_t>(node_->member) : devices.primary();
  auto& backend = devices.device(member);
  const auto stream = devices.stream_for(member).value_or(backend::kDefaultStream);
  LSE_ASSIGN_OR(auto buffer, interpreter::snapshot_buffer(*node_, backend, stream));
  *this = Array::from_buffer(std::move(buffer), node_->shape, node_->dtype);
  return OkStatus();
}

Result<float> Array::item() {
  LSE_RETURN_IF_ERROR(eval());
  return interpreter::read_scalar(*node_);
}

Status Array::to_host(void* dst, std::size_t bytes) {
  LSE_RETURN_IF_ERROR(eval());
  return interpreter::read_raw(*node_, dst, bytes);
}

std::ostream& operator<<(std::ostream& os, Array& a) {
  Status s = a.eval();
  if (!s.ok()) return os << "<Array error: " << s.to_string() << ">";
  return os << "Array" << a.shape().to_string() << ":" << to_string(a.dtype());
}

std::uint64_t FusionGroup::signature() const noexcept {
  std::uint64_t h = 1469598103934665603ull;
  auto mix = [&h](std::uint64_t v) {
    h ^= v;
    h *= 1099511628211ull;
  };
  for (const NodePtr& n : nodes) {
    mix(n->kv_fragments ? 1 : 0);
    mix(static_cast<std::uint64_t>(n->kind));
    mix(static_cast<std::uint64_t>(n->dtype));
    mix(n->shape.rank());
    for (std::size_t i = 0; i < n->shape.rank(); ++i) {
      mix(static_cast<std::uint64_t>(n->shape.dim(i)));
    }
    for (std::int32_t v : n->iattrs) mix(static_cast<std::uint64_t>(v));
    // attrs carry literal values (a constant's payload, clamp bounds, eps).
    // Omitting them made two constants with different values share a cached
    // kernel, so the second silently computed the first one's value.
    for (float f : n->attrs) {
      std::uint32_t bits;
      __builtin_memcpy(&bits, &f, sizeof(bits));
      mix(bits);
    }
    if (n->prim != nullptr) {
      for (char c : n->prim->name()) mix(static_cast<std::uint64_t>(c));
    }
  }
  // Input shapes, not just how many there are. A kernel primitive bakes its
  // operand extents in as literals, so two groups that differ only in an
  // input's inner dimension — linear with K=2176 vs K=512, same [1,8,1024]
  // output — hashed identically and the second reused the first's kernel,
  // indexing far past its buffer.
  mix(inputs.size());
  for (const NodePtr& n : inputs) {
    mix(n->kv_fragments ? 1 : 0);
    mix(static_cast<std::uint64_t>(n->dtype));
    mix(n->shape.rank());
    for (std::size_t i = 0; i < n->shape.rank(); ++i) {
      mix(static_cast<std::uint64_t>(n->shape.dim(i)));
    }
  }
  mix(outputs.size());
  for (const NodePtr& n : outputs) {
    mix(n->kv_fragments ? 1 : 0);
    mix(static_cast<std::uint64_t>(n->dtype));
    mix(n->shape.rank());
    for (std::size_t i = 0; i < n->shape.rank(); ++i) {
      mix(static_cast<std::uint64_t>(n->shape.dim(i)));
    }
  }
  return h;
}

std::uint64_t FusionGroup::emission_signature() const {
  std::uint64_t hash = 1469598103934665603ull;
  auto mix = [&](std::uint64_t value) { hash ^= value; hash *= 1099511628211ull; };
  mix(0x65646765732e7631ull);  // edges.v1
  mix(static_cast<std::uint64_t>(anchor));
  mix(static_cast<std::uint64_t>(anchor_class));
  mix(is_phase);
  std::unordered_map<const Node*, std::uint64_t> ids;
  std::vector<const Node*> identified;
  auto identify = [&](const NodePtr& node) {
    if (!node) return std::uint64_t{0};
    const auto [it, inserted] = ids.emplace(node.get(), ids.size() + 1);
    if (inserted) identified.push_back(node.get());
    return it->second;
  };
  auto sequence = [&](const auto& list) {
    mix(list.size());
    for (const auto& node : list) mix(identify(node));
  };
  sequence(inputs);
  sequence(nodes);
  sequence(outputs);
  for (const auto& node : nodes) sequence(node->inputs);
  mix(identified.size());
  for (const Node* node : identified) {
    mix(static_cast<std::uint64_t>(node->kind));
    mix(static_cast<std::uint64_t>(node->fclass));
    mix(static_cast<std::uint64_t>(node->dtype));
    mix(node->kv_fragments ? 1 : 0);
    mix(node->shape.rank());
    for (std::size_t i = 0; i < node->shape.rank(); ++i)
      mix(static_cast<std::uint64_t>(node->shape.dim(i)));
    mix(node->iattrs.size());
    for (auto value : node->iattrs) mix(static_cast<std::uint64_t>(value));
    mix(node->attrs.size());
    for (float value : node->attrs) {
      std::uint32_t bits;
      __builtin_memcpy(&bits, &value, sizeof(bits));
      mix(bits);
    }
    const auto name = node->prim ? node->prim->name() : std::string_view{};
    mix(name.size());
    for (unsigned char c : name) mix(c);
  }
  // The producer graph can be outside this kernel, but shared allocations
  // still determine aliasing relationships between its bindings.
  std::unordered_map<const Node*, std::uint64_t> owners;
  std::vector<std::uint64_t> ownership;
  bool shared = false;
  for (const Node* node : identified) {
    const Node* owner = buffer_allocation_owner(node);
    const auto [it, inserted] = owners.emplace(owner, owners.size() + 1);
    ownership.push_back(it->second);
    shared |= !inserted;
  }
  if (shared) {
    mix(0x6f776e6572732e31ull);  // owners.1
    for (auto owner : ownership) mix(owner);
  }
  // Phase argument naming strips reshapes but retains slice identities:
  // two slices of one allocation can have different offsets, and reshapes
  // of those slices must resolve to the correct distinct argument slots.
  std::unordered_map<const Node*, std::uint64_t> view_targets;
  std::vector<std::uint64_t> binding_views;
  for (const Node* node : identified) {
    while (node && node->kind == OpKind::kReshape && node->inputs.size() == 1)
      node = node->inputs[0].get();
    const auto [it, inserted] = view_targets.emplace(node, view_targets.size() + 1);
    (void)inserted;
    binding_views.push_back(it->second);
  }
  if (binding_views != ownership) {
    mix(0x76696577732e7631ull);  // views.v1
    for (auto view : binding_views) mix(view);
  }
  return hash;
}

}  // namespace lse::graph
