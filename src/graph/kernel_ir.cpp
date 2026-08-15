#include "lse/graph/kernel_ir.hpp"

#include <array>

namespace lse::graph::kir {

thread_local KernelBody* KernelBody::current_ = nullptr;

void KernelBody::bind() noexcept {
  prev_ = current_;
  current_ = this;
}

void KernelBody::unbind() noexcept { current_ = prev_; }

KernelBody* KernelBody::try_current() noexcept { return current_; }

namespace detail {

std::string literal_u32(std::uint32_t v) { return std::to_string(v) + "u"; }
std::string literal_i32(std::int32_t v) { return std::to_string(v); }

// Ordered by Scalar, like the backend type tables: a new element type without
// a suffix fails to compile rather than silently becoming "x".
constexpr std::array<std::string_view, 12> kSuffixes{{
    "u8", "i8", "u16", "i16", "u32", "i32", "u64", "i64", "f16", "bf16",
    "f32", "b",
}};
static_assert(kSuffixes.size() == static_cast<std::size_t>(Scalar::kBool) + 1,
              "kSuffixes must have one entry per Scalar, in enumerator order");

std::string_view scalar_suffix(Scalar s) noexcept {
  const auto i = static_cast<std::size_t>(s);
  return i < kSuffixes.size() ? kSuffixes[i] : std::string_view{};
}

}  // namespace detail

void KernelBody::statement(std::string text) {
  lines_.push_back(indent() + std::move(text));
}

void KernelBody::ret() { statement("return;"); }

void KernelBody::ret_if(const Val<boolean>& cond) {
  statement("if (" + cond.text() + ") return;");
}

void KernelBody::store(const Val<u32>& index, const Val<f32>& value) {
  if (!store_) return;  // emit_kernel declines when the emitter set no hook
  // Braced so the epilogue's own locals are scoped to this store: a kernel
  // that stores several results does so in one scope.
  statement("{");
  ++depth_;
  const std::string text = store_(index.text(), value.text());
  std::size_t pos = 0;
  while (pos <= text.size()) {
    const std::size_t nl = text.find('\n', pos);
    const std::string line =
        text.substr(pos, nl == std::string::npos ? text.size() - pos : nl - pos);
    if (!line.empty()) statement(line);
    if (nl == std::string::npos) break;
    pos = nl + 1;
  }
  --depth_;
  statement("}");
}

std::string KernelBody::str() const {
  std::string out;
  // Function-scope typedefs: the kernel stays self-contained rather than
  // requiring the emitter to hoist declarations on its behalf.
  for (const std::string& t : typedefs_) out += "  " + t + "\n";
  for (const std::string& l : lines_) out += l + "\n";
  if (!out.empty()) out.pop_back();
  return out;
}

}  // namespace lse::graph::kir
