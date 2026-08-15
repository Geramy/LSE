#include "lse/graph/fallback.hpp"

#include <algorithm>
#include <mutex>

#include "lse/graph/graph.hpp"
#include "lse/graph/interpreter.hpp"

namespace lse::graph {

namespace {

class HostInterpreterFallback final : public FallbackHandler {
 public:
  std::string_view name() const noexcept override { return "host-interpreter"; }

  bool can_handle(const Node& node, const backend::IBackend&) const override {
    return node.prim == nullptr || node.prim->has_host_impl();
  }

  Status execute(Node& node, backend::IBackend& backend) const override {
    // The interpreter takes a NodePtr but must not own this node; an aliasing
    // shared_ptr with a no-op deleter keeps the signature without transferring
    // ownership.
    NodePtr alias(&node, [](Node*) {});
    return interpreter::evaluate(alias, backend);
  }
};

const HostInterpreterFallback kHostFallback{};

}  // namespace

const FallbackHandler* host_interpreter_fallback() { return &kHostFallback; }

Status FallbackChain::add(const FallbackHandler* handler, int priority) {
  if (handler == nullptr) return LSE_ERROR(kInvalidArgument, "null handler");
  for (const Entry& e : entries_) {
    if (e.handler->name() == handler->name()) {
      return LSE_ERROR(kAlreadyExists, "fallback handler '",
                       std::string(handler->name()), "' is already registered");
    }
  }
  entries_.push_back(Entry{handler, priority, next_seq_++, nullptr});
  std::stable_sort(entries_.begin(), entries_.end(),
                   [](const Entry& a, const Entry& b) {
                     if (a.priority != b.priority) return a.priority > b.priority;
                     return a.seq < b.seq;
                   });
  return OkStatus();
}

Status FallbackChain::add_owned(std::unique_ptr<FallbackHandler> handler,
                                int priority) {
  if (handler == nullptr) return LSE_ERROR(kInvalidArgument, "null handler");
  std::shared_ptr<FallbackHandler> shared{std::move(handler)};
  const FallbackHandler* raw = shared.get();
  LSE_RETURN_IF_ERROR(add(raw, priority));
  for (Entry& e : entries_) {
    if (e.handler == raw) e.owned = std::move(shared);
  }
  return OkStatus();
}

Status FallbackChain::remove(std::string_view name) {
  const auto it = std::find_if(entries_.begin(), entries_.end(),
                               [&](const Entry& e) { return e.handler->name() == name; });
  if (it == entries_.end()) {
    return LSE_ERROR(kNotFound, "no fallback handler '", std::string(name), "'");
  }
  entries_.erase(it);
  return OkStatus();
}

void FallbackChain::clear() { entries_.clear(); }

const FallbackHandler* FallbackChain::resolve(
    const Node& node, const backend::IBackend& backend) const {
  for (const Entry& e : entries_) {
    if (e.handler->can_handle(node, backend)) return e.handler;
  }
  return nullptr;
}

const FallbackHandler* FallbackChain::resolve_intercept(
    const Node& node, const backend::IBackend& backend) const {
  for (const Entry& e : entries_) {
    if (e.handler->intercepts() && e.handler->can_handle(node, backend)) {
      return e.handler;
    }
  }
  return nullptr;
}

std::vector<std::string> FallbackChain::handler_names() const {
  std::vector<std::string> out;
  out.reserve(entries_.size());
  for (const Entry& e : entries_) out.emplace_back(e.handler->name());
  return out;
}

FallbackChain& default_fallback_chain() {
  static FallbackChain chain = [] {
    FallbackChain c;
    (void)c.add(host_interpreter_fallback(), 0);
    return c;
  }();
  return chain;
}

}  // namespace lse::graph
