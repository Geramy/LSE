#include "lse/graph/primitive.hpp"

#include "lse/graph/codegen.hpp"

#include <map>
#include <mutex>

namespace lse::graph {

std::string substitute(std::string_view tmpl, std::span<const std::string> args) {
  return substitute(tmpl, args, {});
}

std::string substitute(std::string_view tmpl, std::span<const std::string> args,
                       std::span<const float> attrs) {
  std::string out;
  out.reserve(tmpl.size() + args.size() * 8);
  for (std::size_t i = 0; i < tmpl.size(); ++i) {
    if (tmpl[i] != '$' || i + 1 >= tmpl.size()) {
      out += tmpl[i];
      continue;
    }
    // "$aN" splices attrs[N] as a literal; "$N" splices input N.
    const bool is_attr = tmpl[i + 1] == 'a';
    std::size_t j = i + 1 + (is_attr ? 1 : 0);
    std::size_t index = 0;
    bool digits = false;
    while (j < tmpl.size() && tmpl[j] >= '0' && tmpl[j] <= '9') {
      index = index * 10 + static_cast<std::size_t>(tmpl[j] - '0');
      ++j;
      digits = true;
    }
    if (!digits) {
      out += tmpl[i];
      continue;
    }
    if (is_attr) {
      if (index < attrs.size()) out += float_literal(attrs[index]);
    } else if (index < args.size()) {
      // Parenthesize: a template like "$0 * $0" must not re-associate when the
      // argument is itself an expression.
      out += '(';
      out += args[index];
      out += ')';
    }
    i = j - 1;
  }
  return out;
}

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
