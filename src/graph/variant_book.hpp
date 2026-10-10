#pragma once

// Which variant of a primitive each kernel runs, decided on the device.
//
// A primitive may offer several layouts of one computation
// (KernelPrimitiveBase::variants): where its operands wait, how it buffers,
// how it shares work. Which is fastest is a fact about the device, the shape
// and the compiler, not something a table written by hand can know for every
// part. So the first time a kernel with a choice runs, every variant is
// launched on the real operands; a variant whose output differs from variant
// 0 in any bit is refused, the fastest of the rest is kept, and the decision
// is written beside the kernel cache so later processes start with it.
//
// Variant 0 stays unless another is measurably faster: a device where nothing
// wins runs exactly the kernels it ran before variants existed.

#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>

namespace lse::graph::detail {

class VariantBook {
 public:
  // Off (LSE_AUTOTUNE=off) runs variant 0 everywhere and measures nothing.
  [[nodiscard]] bool enabled() const noexcept { return enabled_; }

  // The decision's name: the kernel's shape class on its device
  // (IKernelEmitter::variant_class), how many variants were on offer, the
  // compiler that builds them and the engine build (engine_build_identity).
  // Decisions are appended to variants-<release>.txt in the kernel cache, so
  // the next process starts with every decided class and measures only the
  // undecided ones, in its warm-up. A rebuilt compiler or engine or a
  // changed menu asks again; a shape grown within its class does not.
  [[nodiscard]] static std::uint64_t key(std::uint64_t base_identity,
                                         std::uint32_t offered,
                                         std::string_view compiler_identity);

  [[nodiscard]] std::optional<std::uint32_t> decided(std::uint64_t key);
  // Kept for this process and appended to the book on disk.
  void record(std::uint64_t key, std::uint32_t variant, std::string_view note);
  // Kept for this process only: a kernel this process could not measure is
  // asked about again by the next one.
  void hold(std::uint64_t key, std::uint32_t variant);

  // Startup measuring is bounded: once the trials of this process have taken
  // the budget (LSE_AUTOTUNE_BUDGET_MS, default 20000), kernels not yet
  // decided run variant 0 and are measured by a later process.
  [[nodiscard]] bool has_budget() const noexcept;
  void spend(std::uint64_t ns) noexcept;

  VariantBook();

 private:
  void load_locked();

  bool enabled_ = true;
  bool loaded_ = false;
  std::uint64_t budget_ns_ = 0;
  std::uint64_t spent_ns_ = 0;
  std::string path_;
  std::mutex mu_;
  std::unordered_map<std::uint64_t, std::uint32_t> decided_;
};

}  // namespace lse::graph::detail
