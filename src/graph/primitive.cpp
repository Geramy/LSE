#include "lse/graph/kernel_primitive.hpp"
#include "lse/graph/primitive.hpp"

#include "lse/graph/codegen.hpp"

#include <algorithm>
#include <cstring>
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

namespace lse::graph {
namespace {
thread_local std::uint32_t g_emission_variant = 0;
}  // namespace
std::uint32_t emission_variant() noexcept { return g_emission_variant; }
EmissionVariantScope::EmissionVariantScope(std::uint32_t variant) noexcept
    : previous_(g_emission_variant) {
  g_emission_variant = variant;
}
EmissionVariantScope::~EmissionVariantScope() { g_emission_variant = previous_; }
namespace {
std::uint64_t class_mix(std::uint64_t h, std::uint64_t v) noexcept {
  h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
  return h * 0xff51afd7ed558ccdull;
}
std::uint64_t ceil_pow2(std::int64_t v) noexcept {
  if (v <= 1) return static_cast<std::uint64_t>(std::max<std::int64_t>(v, 0));
  std::uint64_t p = 1;
  while (p < static_cast<std::uint64_t>(v)) p <<= 1;
  return p;
}
}  // namespace

std::uint64_t KernelPrimitiveBase::variant_class(const KernelShapes& s) const {
  std::uint64_t h = 0x636c617373ull;
  for (const char c : name()) h = class_mix(h, static_cast<unsigned char>(c));
  const auto shape = [&](const Shape& sh) {
    h = class_mix(h, static_cast<std::uint64_t>(sh.rank()));
    for (std::size_t d = 0; d < sh.rank(); ++d) h = class_mix(h, ceil_pow2(sh.dim(d)));
  };
  for (const Shape& sh : s.inputs) shape(sh);
  shape(s.output);
  for (const DType t : s.input_dtypes) h = class_mix(h, static_cast<std::uint64_t>(t));
  h = class_mix(h, static_cast<std::uint64_t>(s.output_dtype));
  for (const float a : s.attrs) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &a, sizeof bits);
    h = class_mix(h, bits);
  }
  for (const auto i : s.iattrs) h = class_mix(h, static_cast<std::uint64_t>(i));
  return h;
}

}  // namespace lse::graph

