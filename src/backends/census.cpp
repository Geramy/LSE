#include "lse/backend/census.hpp"

#include <algorithm>
#include <sstream>

namespace lse::backend {

namespace {

void access_line(std::ostringstream& os, const char* label,
                 const AccessCensus& a, const char* unit) {
  if (a.empty()) return;
  os << "  " << label << ": " << a.count() << " (" << a.bytes() << unit << ")";
  if (a.loops()) {
    os << ", " << a.looped_count() << " looped (" << a.straight_line_bytes()
       << unit << " straight-line)";
  }
  os << " widths";
  for (const AccessCensus::Width& w : a.widths) {
    os << ' ' << w.bytes << "B x" << w.count;
  }
  os << '\n';
}

void fact_line(std::ostringstream& os, const char* label,
               const DeviceFact<std::uint32_t>& f) {
  os << "  " << label << ": ";
  if (f.known()) {
    os << f.value << '\n';
  } else {
    os << to_string(f.source) << '\n';
  }
}

}  // namespace

void AccessCensus::add(std::uint32_t width_bytes, bool in_loop) {
  const auto it = std::lower_bound(
      widths.begin(), widths.end(), width_bytes,
      [](const Width& w, std::uint32_t b) { return w.bytes < b; });
  Width* slot = nullptr;
  if (it != widths.end() && it->bytes == width_bytes) {
    slot = &*it;
  } else {
    slot = &*widths.insert(it, Width{width_bytes, 0, 0});
  }
  ++slot->count;
  if (in_loop) ++slot->looped;
}

std::uint32_t AccessCensus::count() const noexcept {
  std::uint32_t n = 0;
  for (const Width& w : widths) n += w.count;
  return n;
}

std::uint32_t AccessCensus::looped_count() const noexcept {
  std::uint32_t n = 0;
  for (const Width& w : widths) n += w.looped;
  return n;
}

std::uint64_t AccessCensus::bytes() const noexcept {
  std::uint64_t n = 0;
  for (const Width& w : widths) {
    n += static_cast<std::uint64_t>(w.bytes) * w.count;
  }
  return n;
}

std::uint64_t AccessCensus::straight_line_bytes() const noexcept {
  std::uint64_t n = 0;
  for (const Width& w : widths) {
    n += static_cast<std::uint64_t>(w.bytes) * (w.count - w.looped);
  }
  return n;
}

std::string KernelCensus::describe() const {
  std::ostringstream os;
  os << (entry.empty() ? "<unnamed kernel>" : entry) << '\n';
  access_line(os, "global loads  ", global_loads, " B/lane");
  access_line(os, "global stores ", global_stores, " B/lane");
  access_line(os, "shared loads  ", shared_loads, " B/lane");
  access_line(os, "shared stores ", shared_stores, " B/lane");
  access_line(os, "private loads ", private_loads, " B/lane");
  access_line(os, "private stores", private_stores, " B/lane");
  access_line(os, "scalar loads  ", scalar_loads, " B/wave");
  fact_line(os, "instructions  ", instructions);
  fact_line(os, "vector alu    ", vector_alu);
  fact_line(os, "scalar alu    ", scalar_alu);
  fact_line(os, "dot products  ", dot_products);
  fact_line(os, "fma           ", fused_multiply_adds);
  fact_line(os, "matrix ops    ", matrix_ops);
  fact_line(os, "macs/lane     ", multiply_accumulates);
  fact_line(os, "lane exchanges", lane_exchanges);
  fact_line(os, "branches      ", branches);
  fact_line(os, "backward br   ", backward_branches);
  fact_line(os, "memory waits  ", memory_waits);
  fact_line(os, "deepest batch ", deepest_load_batch);
  fact_line(os, "serializing   ", serializing_waits);
  fact_line(os, "unclassified  ", unclassified);
  os << "  straight line : " << (straight_line() ? "yes" : "no") << '\n';
  return os.str();
}

}  // namespace lse::backend
