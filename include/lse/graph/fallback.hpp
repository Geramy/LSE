// Pluggable execution fallback.
//
// When the active backend cannot run a node, the scheduler walks a chain of
// handlers in priority order and hands the node to the first that accepts it.
// A handler can do anything — run on the host, ship to another device, call an
// external library, forward over a transport.
//
//   struct MyFallback final : FallbackHandler {
//     std::string_view name() const noexcept override { return "my"; }
//     bool can_handle(const Node& n, const backend::IBackend&) const override {
//       return n.prim != nullptr && n.prim->name() == "my.op";
//     }
//     Status execute(Node& n, backend::IBackend& be) const override { ... }
//   };
//   default_fallback_chain().add(&my_handler, /*priority=*/10);
#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "lse/backend/backend.hpp"
#include "lse/core/status.hpp"

namespace lse::graph {

class Node;

class FallbackHandler {
 public:
  virtual ~FallbackHandler() = default;

  virtual std::string_view name() const noexcept = 0;

  // Whether this handler can execute `node` given the backend in play.
  virtual bool can_handle(const Node& node,
                          const backend::IBackend& backend) const = 0;

  // False (the default) means "only when the backend cannot run this node".
  // True means the handler is consulted for *every* node, so it can claim work
  // the backend was perfectly capable of — this is what makes per-op device
  // routing switchable at runtime.
  virtual bool intercepts() const noexcept { return false; }

  // The node's output buffer is already allocated; fill it and set
  // node.materialized. Must leave `node` materialized on success.
  virtual Status execute(Node& node, backend::IBackend& backend) const = 0;
};

// Ordered by descending priority; ties keep insertion order. Lookup is by first
// acceptance, so a specific handler at high priority shadows a general one.
class FallbackChain {
 public:
  Status add(const FallbackHandler* handler, int priority = 0);
  Status add_owned(std::unique_ptr<FallbackHandler> handler, int priority = 0);
  Status remove(std::string_view name);
  void clear();

  // First handler that accepts, honouring priority.
  [[nodiscard]] const FallbackHandler* resolve(
      const Node& node, const backend::IBackend& backend) const;

  // First *intercepting* handler that accepts. Consulted before the backend
  // runs a node it is otherwise capable of.
  [[nodiscard]] const FallbackHandler* resolve_intercept(
      const Node& node, const backend::IBackend& backend) const;

  [[nodiscard]] std::vector<std::string> handler_names() const;
  [[nodiscard]] std::size_t size() const noexcept { return entries_.size(); }

 private:
  struct Entry {
    const FallbackHandler* handler = nullptr;
    int priority = 0;
    std::size_t seq = 0;
    std::shared_ptr<FallbackHandler> owned;
  };
  std::vector<Entry> entries_;
  std::size_t next_seq_ = 0;
};

// Seeded with the host-interpreter handler at priority 0, so anything
// registered above it takes precedence.
FallbackChain& default_fallback_chain();

// Runs the node through the CPU reference interpreter. Accepts any node whose
// primitive has a host implementation.
const FallbackHandler* host_interpreter_fallback();

}  // namespace lse::graph
