#include "lse/trace/collector.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <map>
#include <mutex>
#include <utility>

namespace lse::trace {

namespace {

using Clock = std::chrono::steady_clock;

std::uint64_t now_ns() noexcept {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          Clock::now().time_since_epoch())
          .count());
}

std::size_t round_up_pow2(std::size_t n) noexcept {
  std::size_t p = 1;
  while (p < n) p <<= 1;
  return p;
}

// One lane per thread, owned by the registry rather than by the thread: records
// a finished thread wrote are still part of the run's timeline.
struct Lane {
  std::vector<DispatchRecord> ring;
  std::size_t mask = 0;
  std::uint64_t written = 0;
  std::uint32_t index = 0;

  std::vector<SpanRecord> spans;
  std::vector<SpanRecord> open;
  std::size_t span_cap = 0;
  std::uint64_t spans_dropped = 0;
};

struct Registry {
  mutable std::mutex mu;
  std::vector<std::unique_ptr<Lane>> lanes;
  std::map<std::uint64_t, GroupRecord> groups;
  std::vector<CounterSet> counters;
  Collector::Capacity cap{};
  // The correlation id. Process-global and monotonic because it is what a
  // vendor profiler carries as an external correlation id, and that has to be
  // one number, not a (lane, index) pair.
  std::atomic<std::uint64_t> sequence{0};
};

Registry& registry() {
  static Registry r;
  return r;
}

// Cached per thread so the hot path touches no function-local static guard.
struct Tls {
  Lane* lane = nullptr;
  std::atomic<std::uint64_t>* sequence = nullptr;
};

thread_local Tls tls;

Lane& make_lane() {
  Registry& r = registry();
  std::lock_guard lock(r.mu);
  auto owned = std::make_unique<Lane>();
  Lane* fresh = owned.get();
  fresh->index = static_cast<std::uint32_t>(r.lanes.size());
  const std::size_t depth = round_up_pow2(std::max<std::size_t>(r.cap.dispatches, 2));
  fresh->ring.resize(depth);
  fresh->mask = depth - 1;
  fresh->span_cap = std::max<std::size_t>(r.cap.spans, 1);
  fresh->spans.reserve(std::min<std::size_t>(fresh->span_cap, 1u << 10));
  r.lanes.push_back(std::move(owned));
  tls.sequence = &r.sequence;
  return *fresh;
}

inline Lane& current_lane() noexcept {
  if (tls.lane == nullptr) tls.lane = &make_lane();
  return *tls.lane;
}

// Oldest first. `written` counts every record ever stored, so anything past one
// wrap is gone and the difference is what was dropped.
void for_each_live(auto& target, auto&& fn) {
  const std::uint64_t depth = target.ring.size();
  const std::uint64_t first = target.written > depth ? target.written - depth : 0;
  for (std::uint64_t i = first; i < target.written; ++i) {
    fn(target.ring[static_cast<std::size_t>(i & target.mask)]);
  }
}

}  // namespace

Collector& Collector::instance() noexcept {
  static Collector c;
  return c;
}

void Collector::reserve(Capacity capacity) noexcept {
  Registry& r = registry();
  std::lock_guard lock(r.mu);
  r.cap = capacity;
}

Collector::Capacity Collector::capacity() const noexcept {
  Registry& r = registry();
  std::lock_guard lock(r.mu);
  return r.cap;
}

void Collector::record_group(GroupRecord group) {
  if (!group.id.valid()) return;
  Registry& r = registry();
  std::lock_guard lock(r.mu);
  const auto it = r.groups.find(group.id.cache_key);
  if (it == r.groups.end()) {
    r.groups.emplace(group.id.cache_key, std::move(group));
    return;
  }
  // Never lose what the table already had: the memory-hit path cannot reach a
  // source hash, so a later record without one must not erase the one that a
  // compile recorded.
  GroupRecord& held = it->second;
  if (!held.source_hash.has_value() && group.source_hash.has_value()) {
    held.source_hash = group.source_hash;
  }
  if (held.nodes.empty() && !group.nodes.empty()) {
    held.nodes = std::move(group.nodes);
  }
  if (held.entry_name.empty()) held.entry_name = std::move(group.entry_name);
  if (held.arch.empty()) held.arch = std::move(group.arch);
}

void Collector::record_counters(CounterSet counters) {
  Registry& r = registry();
  std::lock_guard lock(r.mu);
  if (r.counters.size() >= r.cap.counters) return;
  r.counters.push_back(std::move(counters));
}

TraceData Collector::snapshot() const {
  Registry& r = registry();
  TraceData out;
  // Taken here rather than at export: a MONOTONIC-to-BOOTTIME offset read after
  // a suspend is not the offset that held while the records were written, and
  // this is the closest point to them that still costs nothing.
  out.host_clocks = read_host_clocks();
  {
    std::lock_guard lock(r.mu);
    std::size_t dispatches = 0;
    std::size_t spans = 0;
    for (const auto& held : r.lanes) {
      dispatches += static_cast<std::size_t>(
          std::min<std::uint64_t>(held->written, held->ring.size()));
      spans += held->spans.size();
      out.dispatches_dropped += held->written > held->ring.size()
                                    ? held->written - held->ring.size()
                                    : 0;
      out.spans_dropped += held->spans_dropped;
    }
    out.dispatches.reserve(dispatches);
    out.spans.reserve(spans);
    for (const auto& held : r.lanes) {
      for_each_live(*held, [&](const DispatchRecord& rec) {
        out.dispatches.push_back(rec);
      });
      out.spans.insert(out.spans.end(), held->spans.begin(), held->spans.end());
    }
    out.groups.reserve(r.groups.size());
    for (const auto& [_, group] : r.groups) out.groups.push_back(group);
    out.counters = r.counters;
  }
  std::sort(out.dispatches.begin(), out.dispatches.end(),
            [](const DispatchRecord& a, const DispatchRecord& b) {
              return a.sequence < b.sequence;
            });
  // Nesting order: an enclosing span opens no later and closes no earlier, so
  // (begin asc, end desc) is a valid depth-first walk of one track.
  std::sort(out.spans.begin(), out.spans.end(),
            [](const SpanRecord& a, const SpanRecord& b) {
              if (a.begin_ns != b.begin_ns) return a.begin_ns < b.begin_ns;
              return a.end_ns > b.end_ns;
            });
  return out;
}

void Collector::clear() noexcept {
  Registry& r = registry();
  std::lock_guard lock(r.mu);
  for (auto& held : r.lanes) {
    held->written = 0;
    held->spans.clear();
    held->open.clear();
    held->spans_dropped = 0;
  }
  r.groups.clear();
  r.counters.clear();
  r.sequence.store(0, std::memory_order_relaxed);
}

Status Collector::resolve_device_spans(ITrace& tracer, std::size_t* resolved) {
  Registry& r = registry();
  std::lock_guard lock(r.mu);
  std::size_t filled = 0;
  Status first_decline;
  for (auto& held : r.lanes) {
    for_each_live(*held, [&](DispatchRecord& rec) {
      if (rec.device.known() || !first_decline.ok()) return;
      Status status = tracer.resolve_device_span(rec);
      if (!status.ok()) {
        first_decline = std::move(status);
        return;
      }
      ++filled;
    });
  }
  if (resolved != nullptr) *resolved = filled;
  return first_decline;
}

std::uint64_t Collector::dispatches_recorded() const noexcept {
  Registry& r = registry();
  std::lock_guard lock(r.mu);
  std::uint64_t total = 0;
  for (const auto& held : r.lanes) total += held->written;
  return total;
}

void record_dispatch(const DispatchRecord& record) noexcept {
  Lane& l = current_lane();
  DispatchRecord& slot = l.ring[static_cast<std::size_t>(l.written & l.mask)];
  slot = record;
  slot.sequence = tls.sequence->fetch_add(1, std::memory_order_relaxed);
  slot.thread = l.index;
  ++l.written;
}

void record_span(SpanRecord span) {
  Lane& l = current_lane();
  if (l.spans.size() >= l.span_cap) {
    ++l.spans_dropped;
    return;
  }
  span.thread = l.index;
  l.spans.push_back(std::move(span));
}

Span::Span(SpanKind kind, std::string name) noexcept : Span(kind, std::move(name), GroupId{}) {}

Span::Span(SpanKind kind, std::string name, GroupId group) noexcept {
  Lane& l = current_lane();
  SpanRecord rec;
  rec.kind = kind;
  rec.name = std::move(name);
  rec.group = group;
  rec.depth = static_cast<std::uint32_t>(l.open.size());
  rec.begin_ns = now_ns();
  l.open.push_back(std::move(rec));
}

Span::~Span() {
  Lane& l = current_lane();
  if (l.open.empty()) return;
  SpanRecord rec = std::move(l.open.back());
  l.open.pop_back();
  rec.end_ns = now_ns();
  record_span(std::move(rec));
}

}  // namespace lse::trace
