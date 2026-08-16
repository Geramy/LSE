#include "lse/graph/primitive.hpp"

#include "lse/graph/codegen.hpp"

#include <map>
#include <mutex>

namespace lse::graph {

namespace {

struct Entry {
  const Primitive* prim = nullptr;
  std::shared_ptr<void> keepalive;
};

struct Registry {
  std::mutex mu;
  std::map<std::string, Entry, std::less<>> by_name;
};

Registry& registry() {
  static Registry r;
  return r;
}

}  // namespace

Status register_primitive(const Primitive* prim, std::shared_ptr<void> keepalive) {
  if (prim == nullptr) return LSE_ERROR(kInvalidArgument, "null primitive");
  Registry& r = registry();
  std::lock_guard lock(r.mu);
  auto [it, inserted] =
      r.by_name.emplace(std::string(prim->name()), Entry{prim, std::move(keepalive)});
  if (!inserted) {
    return LSE_ERROR(kAlreadyExists, "primitive '", std::string(prim->name()),
                     "' is already registered");
  }
  return OkStatus();
}

Status register_primitive(const Primitive* prim) {
  return register_primitive(prim, nullptr);
}

Status unregister_primitive(std::string_view name) {
  Registry& r = registry();
  std::lock_guard lock(r.mu);
  return r.by_name.erase(std::string(name)) > 0
             ? OkStatus()
             : LSE_ERROR(kNotFound, "no primitive '", std::string(name), "'");
}

const Primitive* find_primitive(std::string_view name) {
  Registry& r = registry();
  std::lock_guard lock(r.mu);
  auto it = r.by_name.find(name);
  return it == r.by_name.end() ? nullptr : it->second.prim;
}

std::vector<std::string> registered_primitives() {
  Registry& r = registry();
  std::lock_guard lock(r.mu);
  std::vector<std::string> out;
  out.reserve(r.by_name.size());
  for (const auto& [name, _] : r.by_name) out.push_back(name);
  return out;
}

}  // namespace lse::graph
