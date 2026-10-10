#include "lse/dispatch/sparse_decode.hpp"

#include <map>
#include <mutex>
#include <utility>

#include "lse/dispatch/roofline.hpp"
#include "lse/dispatch/sparse_decode_tuneconfig.h"

namespace lse::dispatch {
namespace {

namespace sd = sparse_decode;

struct Registry {
  std::mutex mu;
  std::map<std::pair<std::string, int>, SparseDecodeCalibration> entries;
};
Registry& registry() {
  static Registry r;
  return r;
}

// The smallest live context a pass over this table can see: KV rungs double
// up to 32K tokens and grow 4K at a time beyond (kv::pool_rung).
std::uint64_t smallest_live(std::uint64_t capacity) noexcept {
  constexpr std::uint64_t kDoublingEnd = 32768, kStep = 4096;
  if (capacity <= kDoublingEnd) return capacity / 2 + 1;
  return capacity - kStep + 1;
}

}  // namespace

void publish_sparse_decode_calibration(const backend::DeviceInfo& device,
                                       const SparseDecodeCalibration& calibration) {
  Registry& r = registry();
  std::lock_guard held(r.mu);
  r.entries[{device.arch, static_cast<int>(device.ordinal)}] = calibration;
}

std::optional<SparseDecodeCalibration> sparse_decode_calibration(const backend::DeviceInfo* device) {
  if (device == nullptr) return std::nullopt;
  Registry& r = registry();
  std::lock_guard held(r.mu);
  const auto it = r.entries.find({device->arch, static_cast<int>(device->ordinal)});
  if (it == r.entries.end() || !it->second.valid()) return std::nullopt;
  return it->second;
}

SparseDecodePlan sparse_decode_predict(const backend::DeviceInfo* device,
                                       const SparseDecodeShape& shape, std::uint64_t live) {
  SparseDecodePlan plan;
  plan.live = live;
  const auto roof = roofline(device);
  const auto cal = sparse_decode_calibration(device);
  if (!roof || !cal || shape.key_heads == 0 || shape.head_dim == 0 || shape.budget == 0) return plan;
  const double kv_token = 2.0 * shape.key_heads * shape.head_dim * shape.element_bytes;
  const double bw = roof->stream_bytes_per_s * cal->exact_efficiency;
  // Exact: the split/flash partial and the merge over every live token.
  plan.exact_ns = cal->exact_fixed_ns + static_cast<double>(live) * kv_token * 1e9 / bw;
  // Sparse: scores over the far pages, the selection, the descriptor, and
  // the same kernels over the budget.
  const double pages = static_cast<double>((live + sd::kPage - 1) / sd::kPage);
  const double group_rows = static_cast<double>(shape.query_heads / shape.key_heads) * shape.rows;
  const double score = cal->score_fixed_ns + pages * shape.key_heads * group_rows * cal->score_ns_per_row_page;
  const double select = cal->select_fixed_ns + pages * cal->select_ns_per_page;
  const double attend = cal->exact_fixed_ns +
                        static_cast<double>(shape.budget) * kv_token * 1e9 / bw;
  const double tail = cal->tail_fixed_ns + pages * shape.key_heads * group_rows * cal->tail_ns_per_row_page;
  plan.sparse_ns = score + select + tail + attend;
  plan.priced = true;
  plan.sparse = live > shape.budget && plan.sparse_ns <= (1.0 - sd::kMinSavedShare) * plan.exact_ns;
  return plan;
}

SparseDecodePlan sparse_decode_plan(const backend::DeviceInfo* device,
                                    const SparseDecodeShape& shape, bool force) {
  if (shape.capacity <= shape.budget) return {};
  if (shape.capacity / sd::kPage > sd::kMaxPages) return {};
  if (force) {
    SparseDecodePlan plan = sparse_decode_predict(device, shape, smallest_live(shape.capacity));
    plan.sparse = true;
    return plan;
  }
  return sparse_decode_predict(device, shape, smallest_live(shape.capacity));
}

std::uint64_t sparse_decode_crossover(const backend::DeviceInfo* device, SparseDecodeShape shape,
                                      std::uint64_t limit) {
  for (std::uint64_t live = shape.budget + sd::kPage; live <= limit; live += 256) {
    const auto plan = sparse_decode_predict(device, shape, live);
    if (!plan.priced) return 0;
    if (plan.sparse) return live;
  }
  return 0;
}

}  // namespace lse::dispatch
