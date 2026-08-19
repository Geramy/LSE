#include "lse/graph/graph.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <span>
#include <unordered_map>
#include <unordered_set>

#include "lse/core/dtype.hpp"
#include "lse/graph/fallback.hpp"
#include "lse/graph/interpreter.hpp"
#include "lse/graph/jit.hpp"
#include "lse/graph/program.hpp"
#include "lse/graph/stream_plan.hpp"
#include "lse/graph/kernel_primitive.hpp"

namespace lse::graph {
namespace {

using SpanClock = std::chrono::steady_clock;

std::uint64_t elapsed_ns(SpanClock::time_point from,
                         SpanClock::time_point to) noexcept {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(to - from).count());
}

// One span, closed exactly once, whichever way its region is left.
//
// The bind region has eight failure exits and the emit region two; charging
// their time to nothing is why a group the emitter refused used to cost nothing
// at all in the report. close() hands back the tick it read so the next span can
// open on it — the spans are adjacent, so one read serves both ends and the
// instrumentation costs one clock read per boundary rather than two.
class SpanTimer {
 public:
  SpanTimer(trace::HostDuration& into, SpanClock::time_point begin) noexcept
      : into_(&into), begin_(begin) {}
  explicit SpanTimer(trace::HostDuration& into) noexcept
      : SpanTimer(into, SpanClock::now()) {}
  ~SpanTimer() {
    if (into_ != nullptr) into_->add(elapsed_ns(begin_, SpanClock::now()));
  }

  SpanTimer(const SpanTimer&) = delete;
  SpanTimer& operator=(const SpanTimer&) = delete;

  // Idempotent: a region reached down two paths closes on whichever runs.
  SpanClock::time_point close() noexcept {
    const auto now = SpanClock::now();
    if (into_ != nullptr) {
      into_->add(elapsed_ns(begin_, now));
      into_ = nullptr;
    }
    return now;
  }

 private:
  trace::HostDuration* into_;
  SpanClock::time_point begin_;
};

}  // namespace

struct Scheduler::Impl {
  std::unique_ptr<JitCache> jit;
  std::vector<backend::DeviceBuffer> phase_tables;
  // One per stream: a persistent-grid kernel spins on this counter across its
  // whole grid, so two of them running at once must not share one.
  std::vector<backend::DeviceBuffer> grid_bar;

  Program program;
  StreamPlan plan;
  std::vector<backend::StreamEvent> events;
  // Streams still holding work from an earlier step. A plan covers one step,
  // so without this the first group a step puts on a stream would be ordered
  // against nothing at all — the previous step's work on every *other* stream
  // is invisible to it. Cleared whenever the device is known idle.
  std::vector<std::uint8_t> outstanding;
  std::vector<backend::StreamEvent> entry_events;

  Status ensure_jit(backend::IDeviceSet& devices, std::size_t member,
                    const KernelToolchain& tc) {
    if (jit != nullptr) return OkStatus();
    // Asked of the member about to be dispatched to, because a set may hold a
    // device with codegen beside one without: the refusal has to name the
    // device that declined, not the set.
    const backend::IBackend& be = devices.device(member);
    if (tc.compiler == nullptr) {
      return LSE_ERROR(kUnimplemented, "backend '", std::string(be.name()),
                       "' has no ", std::string(to_string(tc.dialect)),
                       " kernel compiler");
    }
    jit = std::make_unique<JitCache>(devices);
    return OkStatus();
  }
};

Scheduler::Scheduler(backend::IDeviceSet& devices)
    : devices_(devices), impl_(std::make_unique<Impl>()) {
  // Without codegen there is no code-object path, so device-first is
  // meaningless. Asked of the front entry: a preference is set after
  // construction and cannot add codegen to a device that has none.
  if (backend().emitter() == nullptr) mode_ = Mode::kHostOnly;
}

Scheduler::Scheduler(backend::IBackend& backend)
    : own_set_(std::make_unique<backend::SingleDevice>(backend)),
      devices_(*own_set_),
      impl_(std::make_unique<Impl>()) {
  if (backend.emitter() == nullptr) mode_ = Mode::kHostOnly;
}

Scheduler::~Scheduler() = default;

Scheduler::JitStats Scheduler::jit_stats() const noexcept {
  if (impl_->jit == nullptr) return {};
  const JitCache::Stats& s = impl_->jit->stats();
  return JitStats{s.memory_hits, s.disk_hits, s.compiles, s.compile_ns};
}

std::string Scheduler::device_gap(const Node& node,
                                  const backend::IBackend& backend) {
  if (node.prim != nullptr && !node.prim->has_device_impl()) {
    return std::string(node.prim->name()) + ": no device implementation";
  }
  // The reference backend runs everything through the interpreter, so nothing
  // is a gap there.
  if (backend.name() == "cpu") return {};
  return {};
}

std::size_t Scheduler::member_for(const FusionGroup& group) const noexcept {
  if (devices_.size() == 1) return devices_.primary();
  // The first input that names a device decides. Operands cannot be read from
  // anywhere else until something moves them, so this is forced rather than
  // chosen — what a Planner decides is whether moving them is worth it, and it
  // decides that before the group reaches here.
  for (const NodePtr& n : group.inputs) {
    if (!n) continue;
    const std::size_t m = devices_.member_of(n->buffer.residency);
    if (m < devices_.size()) return m;
  }
  return devices_.primary();
}

void Scheduler::release(backend::DeviceBuffer& buf) const noexcept {
  // Through whoever allocated it. Freeing a member's allocation on another
  // member's allocator is the mirror of a wrong-device launch, and residency is
  // what makes the owner recoverable from the buffer instead of from wherever
  // the caller happened to be standing.
  const std::size_t owner = devices_.member_of(buf.residency);
  devices_.device(owner < devices_.size() ? owner : devices_.primary())
      .deallocate(buf);
}

Status Scheduler::check_residency(std::span<const backend::BufferRef> bindings,
                                  std::size_t member) const {
  const backend::DeviceIndex mine = devices_.residency(member);
  for (const backend::BufferRef& b : bindings) {
    if (b.buffer == nullptr) continue;
    const backend::DeviceIndex held = b.buffer->residency;
    // The common case is an inline compare that never leaves this loop; the set
    // is only asked about a residency that is not already the target's, which
    // on one device is never.
    if (!held.bound() || held == mine) continue;
    LSE_RETURN_IF_ERROR(devices_.may_read(held, member));
  }
  return OkStatus();
}

Status Scheduler::try_dispatch_group(const FusionGroup& group,
                                     backend::Stream stream,
                                     std::size_t member) {
  backend::IBackend& be = devices_.device(member);
  // The dialect this run asked for if this member declares it, and this
  // member's own first choice otherwise. Both halves come from the one
  // toolchain, so the compiler the cache reaches for below is the one declared
  // beside the emitter that wrote the text.
  const KernelToolchain* tc = be.toolchain(dialect_);
  const IKernelEmitter* emitter = tc != nullptr ? tc->emitter : nullptr;
  if (emitter == nullptr) {
    return LSE_ERROR(kUnimplemented, "backend '", std::string(be.name()),
                     "' has no kernel emitter");
  }
  LSE_RETURN_IF_ERROR(impl_->ensure_jit(devices_, member, *tc));

  const std::uint64_t ident =
      emitter->cache_key(group, be.device_info());

  const auto t_emit = SpanClock::now();
  auto emitted = emitter->emit(group, be.device_info());
  const auto t_emitted = SpanClock::now();
  trace_.spans.emit.add(elapsed_ns(t_emit, t_emitted));
  if (!emitted.ok()) return emitted.status();

  // Compile only when this kernel is not already loaded for this device.
  // Disk miss / source change / arch change still go through get_or_compile.
  //
  // The compiler runs INSIDE get_or_compile, which is inside this span, so its
  // own measurement is subtracted back out rather than counted twice. Without
  // that, a cold prefill reported 3113 ms of "emit" that was 98.5% compile, and
  // adding the report's emit line to its jit line counted the same milliseconds
  // three times.
  const std::uint64_t compile_before = impl_->jit->stats().compile_ns;
  dump_hip_source(*emitted, ident);
  backend::KernelHandle launched;
  Status jit_status = OkStatus();
  if (const backend::KernelHandle* cached =
          impl_->jit->try_get(member, ident, emitted->dialect)) {
    launched = *cached;
  } else {
    auto kernel = impl_->jit->get_or_compile(member, ident, *emitted);
    if (kernel.ok()) {
      launched = kernel.release();
    } else {
      jit_status = kernel.status();
    }
  }
  const auto t_resolved = SpanClock::now();
  const std::uint64_t compiled =
      impl_->jit->stats().compile_ns - compile_before;
  const std::uint64_t looked_up = elapsed_ns(t_emitted, t_resolved);
  trace_.spans.jit_compile.add(compiled);
  trace_.spans.jit_lookup.add(looked_up > compiled ? looked_up - compiled : 0);
  LSE_RETURN_IF_ERROR(jit_status);

  SpanTimer bind_span(trace_.spans.bind, t_resolved);

  for (const NodePtr& n : group.nodes) {
    if (!n || n->prim == nullptr) continue;
    const int a = n->prim->inplace_input();
    if (a < 0 || static_cast<std::size_t>(a) >= n->inputs.size()) continue;
    const NodePtr& src = n->inputs[static_cast<std::size_t>(a)];
    if (!src || !src->buffer.valid()) {
      return LSE_ERROR(kInvalidArgument, "inplace input has no buffer");
    }
    n->buffer = src->buffer;
  }
  for (const NodePtr& n : emitted->binding_order) {
    if (!n || n->prim == nullptr) continue;
    const int a = n->prim->inplace_input();
    if (a < 0 || static_cast<std::size_t>(a) >= n->inputs.size()) continue;
    const NodePtr& src = n->inputs[static_cast<std::size_t>(a)];
    if (!src || !src->buffer.valid()) {
      return LSE_ERROR(kInvalidArgument, "inplace input has no buffer");
    }
    n->buffer = src->buffer;
  }

  // Inputs must already be materialized; outputs need a buffer to write into.
  std::vector<backend::BufferRef> bindings;
  bindings.reserve(emitted->binding_order.size());
  for (const NodePtr& n : emitted->binding_order) {
    // A constant carries its value in attrs and is usually inlined as a
    // literal, which leaves it "materialized" with no buffer at all. When a
    // later group binds it as a buffer instead, that fresh allocation holds
    // nothing — so the value has to be written whenever we allocate here.
    const bool fresh = !n->buffer.valid();
    if (fresh) {
      LSE_RETURN_IF_ERROR(interpreter::ensure_output_buffer(*n, be));
    }
    if (fresh && n->kind == OpKind::kConstant) {
      for (std::size_t e = 0; e < n->element_count(); ++e) {
        interpreter::store_element(*n, e, n->attrs[0]);
      }
      n->materialized = true;
    }
    // Weights and any host-computed input have to reach device memory before
    // the kernel reads them.
    LSE_RETURN_IF_ERROR(interpreter::sync_to_device(*n, be));
    bindings.push_back(backend::BufferRef{&n->buffer, 0, n->buffer.size_bytes});
  }

  std::size_t elements = 0;
  for (const NodePtr& n : group.outputs) {
    elements = std::max(elements, n->element_count());
  }
  if (elements == 0 && !group.nodes.empty()) {
    elements = group.nodes.back()->element_count();
  }

  std::vector<std::byte> constants(emitted->constants.total_bytes);
  if (constants.size() < sizeof(std::uint32_t)) {
    return LSE_ERROR(kInternal, "constants block has no room for the count");
  }
  const auto count = static_cast<std::uint32_t>(elements);
  std::memcpy(constants.data(), &count, sizeof(count));

  backend::DeviceBuffer table_buf;
  std::vector<backend::BufferRef> table_bindings;
  if (emitted->pointer_table) {
    std::vector<void*> ptrs(bindings.size(), nullptr);
    for (std::size_t i = 0; i < bindings.size(); ++i) {
      auto p = be.device_pointer(*bindings[i].buffer);
      if (!p.ok()) return p.status();
      ptrs[i] = *p;
    }
    auto tab = be.allocate(ptrs.size() * sizeof(void*),
                                 backend::MemoryClass::kDevice);
    if (!tab.ok()) return tab.status();
    table_buf = tab.release();
    LSE_RETURN_IF_ERROR(be.copy_h2d(ptrs.data(), table_buf,
                                           ptrs.size() * sizeof(void*), 0));
    table_bindings.push_back(
        backend::BufferRef{&table_buf, 0, table_buf.size_bytes});
    impl_->phase_tables.push_back(std::move(table_buf));
    table_bindings.back().buffer = &impl_->phase_tables.back();
  }

  if (emitted->persist_grid) {
    if (impl_->grid_bar.size() <= stream.index) {
      impl_->grid_bar.resize(stream.index + 1);
    }
    backend::DeviceBuffer& bar_buf = impl_->grid_bar[stream.index];
    if (!bar_buf.valid()) {
      auto bar = be.allocate(sizeof(std::uint32_t),
                                   backend::MemoryClass::kDevice);
      if (!bar.ok()) return bar.status();
      bar_buf = bar.release();
      const std::uint32_t zero = 0;
      LSE_RETURN_IF_ERROR(be.copy_h2d(&zero, bar_buf, sizeof(zero), 0));
    }
    bindings.push_back(backend::BufferRef{&bar_buf, 0, bar_buf.size_bytes});
  }

  backend::DispatchArgs args;
  args.bindings = emitted->pointer_table ? table_bindings : bindings;
  args.constants = constants;

  if (const char* want = std::getenv("LSE_DUMP_ENTRY")) {
    if (emitted->entry_name.find(want) != std::string::npos) {
      std::fprintf(stderr, "--- %s ---\n%s---\n", emitted->entry_name.c_str(),
                   emitted->source.c_str());
    }
  }
  if (std::getenv("LSE_TRACE_DISPATCH") != nullptr) {
    std::fprintf(stderr, "dispatch %s phase=%d count=%u wg=%u x %u |",
                 emitted->entry_name.c_str(), (int)group.is_phase, count,
                 emitted->dims.workgroup_count[0], emitted->dims.workgroup_size[0]);
    for (const NodePtr& n : emitted->binding_order) {
      std::fprintf(stderr, " %s%s", std::string(to_string(n->kind)).c_str(),
                   n->shape.to_string().c_str());
    }
    std::fprintf(stderr, "\n");
  }
  // `bindings`, not args.bindings: with a pointer table the dispatch binds one
  // buffer of device addresses and the operands it points at are exactly the
  // ones a wrong-device launch would read.
  LSE_RETURN_IF_ERROR(check_residency(bindings, member));

  const auto t_launch = bind_span.close();
  const Status submitted = be.launch(
      launched, emitted->dims, args,
      backend::DispatchTarget{stream, devices_.residency(member), {}});
  trace_.spans.submit.add(elapsed_ns(t_launch, SpanClock::now()));
  LSE_RETURN_IF_ERROR(submitted);

  for (const NodePtr& n : group.nodes) {
    n->materialized = true;
    // Everything this group produced now lives in device memory, so any later
    // host read has to pull it back. Marking only group.outputs would miss a
    // root with no consumers — exactly the value a caller is about to read.
    n->device_dirty = true;
    // The kernel overwrote whatever the mirror held, so there is nothing left
    // to push. Leaving this set would upload stale bytes on the next dispatch.
    n->host_dirty = false;
  }
  return OkStatus();
}

FallbackChain& Scheduler::fallback_chain() const noexcept {
  return fallbacks_ != nullptr ? *fallbacks_ : default_fallback_chain();
}

namespace {

backend::DeviceBuffer view_of(const backend::DeviceBuffer& src,
                              std::size_t byte_offset, std::size_t byte_size) {
  backend::DeviceBuffer v = src;
  v.offset = src.offset + byte_offset;
  v.size_bytes = byte_size;
  return v;
}

void alias_onto(Node& dst, const Node& src, backend::DeviceBuffer buf,
                std::size_t host_byte_off) {
  dst.buffer = buf;
  dst.device_dirty = src.device_dirty && !src.host_dirty;
  dst.host_dirty = src.host_dirty;
  dst.host_mirror.clear();
  if (src.host_dirty && host_byte_off + buf.size_bytes <= src.host_mirror.size()) {
    const auto begin = src.host_mirror.begin() +
                       static_cast<std::ptrdiff_t>(host_byte_off);
    dst.host_mirror.assign(begin, begin + static_cast<std::ptrdiff_t>(buf.size_bytes));
  }
  dst.materialized = true;
}

// Reshape is a view of the same bytes. A slice is a real device copy (its own
// kernel) so MixerState never aliases a live producer buffer.
Status try_alias_group(const FusionGroup& group) {
  for (const NodePtr& n : group.nodes) {
    if (n->kind != OpKind::kReshape || n->inputs.size() != 1) {
      return LSE_ERROR(kUnimplemented, "not a reshape view");
    }
    const Node& src = *n->inputs[0];
    if (!src.materialized || !src.buffer.valid()) {
      return LSE_ERROR(kUnimplemented, "view alias needs a materialized input");
    }
    if (src.dtype != n->dtype || src.element_count() != n->element_count()) {
      return LSE_ERROR(kInvalidArgument, "reshape changes dtype or element count");
    }
    const std::size_t bytes =
        dtype_storage_bytes(n->dtype, n->element_count());
    if (bytes == 0) {
      return LSE_ERROR(kUnimplemented, "view alias cannot size the window");
    }
    alias_onto(*n, src, view_of(src.buffer, 0, bytes), 0);
  }
  return OkStatus();
}

void alias_ready_reshapes(const FusionGroup& group) {
  bool progressed = true;
  while (progressed) {
    progressed = false;
    for (const NodePtr& n : group.nodes) {
      if (!n || n->kind != OpKind::kReshape || n->inputs.size() != 1) continue;
      if (n->buffer.valid()) continue;
      const Node& src = *n->inputs[0];
      if (!src.materialized || !src.buffer.valid()) continue;
      if (src.dtype != n->dtype || src.element_count() != n->element_count()) {
        continue;
      }
      const std::size_t bytes =
          dtype_storage_bytes(n->dtype, n->element_count());
      if (bytes == 0) continue;
      alias_onto(*n, src, view_of(src.buffer, 0, bytes), 0);
      progressed = true;
    }
  }
}

bool is_gdn_node(const Node& n) noexcept {
  return n.kind == OpKind::kGDNChunkScan;
}

bool same_gdn_inputs(const Node& a, const Node& b) noexcept {
  if (a.inputs.size() != 6 || b.inputs.size() != 6) return false;
  for (std::size_t i = 0; i < 6; ++i) {
    if (a.inputs[i].get() != b.inputs[i].get()) return false;
  }
  return true;
}

Status accumulate_spans(Scheduler::Trace::Spans& acc,
                        const Scheduler::Trace::Spans& step) {
  LSE_RETURN_IF_ERROR(acc.partition.add(step.partition));
  LSE_RETURN_IF_ERROR(acc.schedule.add(step.schedule));
  LSE_RETURN_IF_ERROR(acc.emit.add(step.emit));
  LSE_RETURN_IF_ERROR(acc.jit_lookup.add(step.jit_lookup));
  LSE_RETURN_IF_ERROR(acc.jit_compile.add(step.jit_compile));
  LSE_RETURN_IF_ERROR(acc.bind.add(step.bind));
  LSE_RETURN_IF_ERROR(acc.submit.add(step.submit));
  LSE_RETURN_IF_ERROR(acc.host_wait.add(step.host_wait));
  LSE_RETURN_IF_ERROR(acc.readback.add(step.readback));
  LSE_RETURN_IF_ERROR(acc.host_exec.add(step.host_exec));
  LSE_RETURN_IF_ERROR(acc.step.add(step.step));
  LSE_RETURN_IF_ERROR(acc.unattributed.add(step.unattributed));
  return OkStatus();
}

// Everything the step's spans account for. Disjoint, so this is a sum and not a
// max, and `step` minus this is the remainder.
std::uint64_t attributed_ns(const Scheduler::Trace::Spans& s) noexcept {
  return s.partition.ns + s.schedule.ns + s.emit.ns + s.jit_lookup.ns +
         s.jit_compile.ns + s.bind.ns + s.submit.ns + s.host_wait.ns +
         s.readback.ns + s.host_exec.ns;
}

Status accumulate(Scheduler::Trace& acc, const Scheduler::Trace& step) {
  acc.device_groups += step.device_groups;
  acc.host_groups += step.host_groups;
  acc.phase_groups += step.phase_groups;
  acc.phase_ideal_launches += step.phase_ideal_launches;
  acc.views_aliased += step.views_aliased;
  acc.slots_reused += step.slots_reused;
  acc.slots_allocated += step.slots_allocated;
  acc.host_fallbacks += step.host_fallbacks;
  acc.kernels_launched += step.kernels_launched;
  acc.nodes_evaluated += step.nodes_evaluated;
  acc.intercepted += step.intercepted;
  acc.collectives_issued += step.collectives_issued;
  acc.streams_used = std::max(acc.streams_used, step.streams_used);
  acc.stream_waits += step.stream_waits;
  acc.stream_chain += step.stream_chain;
  LSE_RETURN_IF_ERROR(accumulate_spans(acc.spans, step.spans));
  acc.partition_ns += step.partition_ns;
  acc.emit_ns += step.emit_ns;
  acc.launch_ns += step.launch_ns;
  acc.sync_ns += step.sync_ns;
  acc.host_group_reasons.insert(acc.host_group_reasons.end(),
                                step.host_group_reasons.begin(),
                                step.host_group_reasons.end());
  acc.replayed = acc.replayed || step.replayed;
  return OkStatus();
}

}  // namespace

Status Scheduler::eval(std::span<const NodePtr> roots, bool pull_host) {
  return eval(roots, pull_host, nullptr);
}

Status Scheduler::eval(std::span<const NodePtr> roots, bool pull_host,
                      Program* plan) {
  const auto t_step = SpanClock::now();
  trace_ = Trace{};
  const Status ran = eval_step(roots, pull_host, plan);

  trace_.spans.step.add(elapsed_ns(t_step, SpanClock::now()));
  const std::uint64_t attributed = attributed_ns(trace_.spans);
  // Saturating, so an overlap surfaces as a zero remainder rather than as a
  // number that would hide it. The spans are disjoint by construction; this is
  // the check on that, not a correction for it.
  trace_.spans.unattributed.add(trace_.spans.step.ns > attributed
                                    ? trace_.spans.step.ns - attributed
                                    : 0);
  // Only the partition span. The old counter never held the per-step setup
  // either — it was written under `if (!replayed)` and so read zero on every
  // steady-state decode token — so `schedule` is new information and reaches a
  // reader through `spans`, not through a field whose name would not cover it.
  trace_.partition_ns = trace_.spans.partition.ns;
  trace_.emit_ns = trace_.spans.emit.ns;
  trace_.launch_ns = trace_.spans.submit.ns;
  trace_.sync_ns = trace_.spans.host_wait.ns;

  const Status accumulated = accumulate(acc_, trace_);
  if (!ran.ok()) return ran;
  return accumulated;
}

Status Scheduler::eval_step(std::span<const NodePtr> roots, bool pull_host,
                            Program* plan) {
  Program& rec = plan != nullptr ? *plan : impl_->program;
  // Asked once. Four sites below need it and it is a virtual call on a
  // device-first step, and a compiler that cannot see through the repeat cannot
  // see that the null check already happened either.
  const KernelToolchain* const primary_tc = toolchain(devices_.primary());
  const IKernelEmitter* const emitter =
      primary_tc != nullptr ? primary_tc->emitter : nullptr;
  const bool device_first = mode_ == Mode::kDeviceFirst && emitter != nullptr;
  // Opened here rather than in eval() so that the trace reset and the step
  // bookkeeping around it land in the remainder instead of in a span.
  SpanTimer setup_span(trace_.spans.schedule);
  for (auto& t : impl_->phase_tables) release(t);
  impl_->phase_tables.clear();

  const std::vector<NodePtr> order = Partitioner::unmaterialized(roots);

  std::vector<FusionGroup> phase_groups;
  bool replayed = false;
  if (device_first && rec.holds(roots) && !rec.groups().empty()) {
    bool ready = !roots.empty();
    for (const NodePtr& r : roots) {
      if (r && !r->materialized) {
        ready = false;
        break;
      }
    }
    if (ready) {
      const auto t_ready = setup_span.close();
      if (pull_host) {
        SpanTimer readback_span(trace_.spans.readback, t_ready);
        for (const NodePtr& r : roots) {
          if (r) LSE_RETURN_IF_ERROR(interpreter::sync_from_device(*r, backend()));
        }
      }
      return OkStatus();
    }
    replayed = true;
    trace_.replayed = true;
    trace_.phase_groups = static_cast<std::uint32_t>(rec.phases().size());
    if (trace_.phase_groups == 0) trace_.phase_groups = 1;
    trace_.phase_ideal_launches = 1;
  }

  // The build: what groups this step is made of. A replay reuses the retained
  // ones and pays none of it, which is why this and `schedule` are two spans
  // and not one.
  SpanTimer build_span(trace_.spans.partition, setup_span.close());
  auto planned = replayed
                     ? std::vector<Workgroup>{}
                     : Partitioner::phases(roots, &backend().device_info());
  if (!replayed) {
    trace_.phase_groups = static_cast<std::uint32_t>(planned.size());
    for (const Workgroup& wg : planned) {
      trace_.phase_ideal_launches += wg.ideal_launches();
    }
  }

  if (device_first && !replayed) {
    // Null when this emitter has no staged phase body: every node then takes
    // the same one-group path a node it refuses to stage already takes.
    const IPhaseStaging* staging = emitter->staging();
    for (Workgroup& wg : planned) {
      FusionGroup g = Partitioner::phase_group(wg, roots);
      if (g.nodes.empty()) continue;
      alias_ready_reshapes(g);
      FusionGroup staged;
      staged.is_phase = true;
      staged.launches = 1;
      staged.anchor_class = FusionClass::kBarrier;
      bool staged_grid = false;
      bool staged_lane = true;
      bool staged_fused = false;
      auto flush_staged = [&] {
        if (staged.nodes.empty()) return;
        auto chunks = Partitioner::phase_chunks(staged, 480);
        for (FusionGroup& c : chunks) phase_groups.push_back(std::move(c));
        staged = FusionGroup{};
        staged.is_phase = true;
        staged.launches = 1;
        staged.anchor_class = FusionClass::kBarrier;
      };
      // Mirrors the emitter's `dependent` test: a chunk whose stages never
      // read each other needs no barrier, which is what lets it keep a grid.
      auto reads_staged = [&](const NodePtr& n) {
        for (const NodePtr& in : n->inputs) {
          const Node* p = in.get();
          while (p != nullptr && p->kind == OpKind::kReshape &&
                 p->inputs.size() == 1) {
            p = p->inputs[0].get();
          }
          for (const NodePtr& m : staged.nodes) {
            if (m.get() == p) return true;
          }
        }
        return false;
      };
      auto linear_n = [](const Node& n) -> std::int64_t {
        if (n.inputs.size() < 2) return 0;
        const auto* kp = dynamic_cast<const KernelPrimitiveBase*>(n.prim);
        if (kp == nullptr) return 0;
        const auto name = kp->name();
        // quant_linear belongs here too: it stages a row and is priced into
        // run_lds_bytes exactly as the dense kernels are, so leaving it out of
        // this list is what made every wide linear in a quantized checkpoint
        // its own launch.
        if (name != "linear" && name != "linear.lds" &&
            name != "linear_indexed" && name != "linear_indexed.lds" &&
            name != "quant_linear") {
          return 0;
        }
        const Shape& w = n.inputs[1]->shape;
        if (w.rank() == 3) return w.dim(1);
        if (w.rank() >= 2) return w.dim(0);
        return 0;
      };
      auto join_gdn_pair = [&](const NodePtr& n) -> bool {
        if (phase_groups.empty() || !is_gdn_node(*n)) return false;
        FusionGroup& prev = phase_groups.back();
        if (prev.is_phase || prev.nodes.size() != 1) return false;
        if (!is_gdn_node(*prev.nodes[0])) return false;
        if (!same_gdn_inputs(*prev.nodes[0], *n)) return false;
        if (prev.nodes[0]->iattrs[0] == n->iattrs[0]) return false;
        prev.nodes.push_back(n);
        prev.outputs.clear();
        for (const NodePtr& m : prev.nodes) {
          if (m->iattrs[0] == 0) prev.outputs.push_back(m);
        }
        if (prev.outputs.empty()) prev.outputs.push_back(prev.nodes[0]);
        return true;
      };
      auto join_wide_linear = [&](const NodePtr& n) -> bool {
        if (phase_groups.empty()) return false;
        FusionGroup& prev = phase_groups.back();
        if (prev.is_phase || prev.nodes.empty()) return false;
        if (n->inputs.empty()) return false;
        const Node* x = n->inputs[0].get();
        for (const NodePtr& m : prev.nodes) {
          if (m->inputs.empty() || m->inputs[0].get() != x) return false;
          for (const NodePtr& in : n->inputs) {
            if (in.get() == m.get()) return false;
          }
        }
        prev.nodes.push_back(n);
        prev.outputs.push_back(n);
        for (const NodePtr& in : n->inputs) {
          bool seen = false;
          for (const NodePtr& e : prev.inputs) {
            if (e.get() == in.get()) {
              seen = true;
              break;
            }
          }
          if (!seen) prev.inputs.push_back(in);
        }
        return true;
      };
      for (const NodePtr& n : g.nodes) {
        // N>=128 matches Workgroup::is_wide_linear. Those GEMVs get a
        // fat grid; stream visibility is the barrier. A software grid
        // sync deadlocks without a cooperative launch.
        const bool wide = linear_n(*n) >= 128;
        if (wide) {
          flush_staged();
          if (join_wide_linear(n)) continue;
          FusionGroup one;
          one.nodes.push_back(n);
          one.outputs.push_back(n);
          one.inputs = n->inputs;
          one.anchor = n->kind;
          one.anchor_class = n->fclass;
          phase_groups.push_back(std::move(one));
          continue;
        }
        if (staging == nullptr || !staging->can_stage(*n)) {
          flush_staged();
          if (join_gdn_pair(n)) continue;
          FusionGroup one;
          one.nodes.push_back(n);
          one.outputs.push_back(n);
          one.inputs = n->inputs;
          one.anchor = n->kind;
          one.anchor_class = n->fclass;
          phase_groups.push_back(std::move(one));
          continue;
        }
        // A stage with thousands of independent items must not share the
        // one-workgroup fallback with the dependent chain around it: the
        // launch boundary is the barrier a grid-wide stage needs, and it
        // costs ~2 us against tens of microseconds of idle CUs. 2048 is
        // eight workgroups — below that the grid is not worth the split.
        constexpr std::uint32_t kGridStage = 2048;
        const bool fat =
            staging->stage_threads(*n, backend().device_info()) >= kGridStage;
        const bool grid_chunk = staged_grid && !staged.nodes.empty();
        // Mirrors the emitter's `lane_chunk`: every stage per-lane over one
        // element count, every read of the chunk at the index the reading
        // thread wrote. The launch boundary a dependence normally buys is a
        // grid-wide barrier, and this chain never leaves the thread — so the
        // chunk keeps it and the pair costs one launch. The predicate has to
        // hold for the whole chunk, not just the edge, or the emitter would
        // decline the grid and the merged stages would land on one workgroup.
        bool lane_fits = staged_lane && staging->lane_stage(*n);
        if (lane_fits) {
          for (const NodePtr& m : staged.nodes) {
            if (!staging->lane_aligned(*m, *n)) {
              lane_fits = false;
              break;
            }
          }
        }
        const bool lane_join = grid_chunk && lane_fits;
        // A chunk holding a dependence only its lane shape can carry has to
        // keep that shape. Admitting a stage that breaks it makes the emitter
        // refuse the grid, and then every stage in the chunk — including the
        // wide ones that were already on a grid — runs on one workgroup.
        const bool breaks_lane_fused = staged_fused && !lane_fits;
        // A grid chunk keeps growing while its stages stay independent, so
        // sibling wide stages cost one launch, not one each.
        if (breaks_lane_fused ||
            (fat ? (!grid_chunk || (reads_staged(n) && !lane_join))
                 : (grid_chunk && reads_staged(n) && !lane_join))) {
          flush_staged();
        }
        if (staged.nodes.empty()) {
          staged_grid = fat;
          staged_lane = staging->lane_stage(*n);
          staged_fused = false;
        } else {
          staged_lane = lane_fits;
          if (lane_join && reads_staged(n)) staged_fused = true;
        }
        staged.nodes.push_back(n);
      }
      flush_staged();
      wg.plan_slots(roots);
      if (wg.bind_slots(backend()).ok()) {
        trace_.slots_reused += wg.reused_slots();
        trace_.slots_allocated += wg.slot_count();
      }
    }
  }

  std::vector<FusionGroup> ran;
  // Back to per-step setup: the stream plan and the entry barriers.
  SpanTimer place_span(trace_.spans.schedule, build_span.close());
  // Replays run the retained groups in place: no copy of the group list, and
  // no re-retain afterwards — retain() re-walks the whole reachable graph and
  // was most of the host-side churn of a decode token.
  const std::vector<FusionGroup>& staged_groups =
      replayed ? rec.groups() : phase_groups;
  if (device_first && !staged_groups.empty()) {
      // Placement for the whole step, decided before any of it is issued.
      impl_->plan = plan_streams(staged_groups, backend().stream_capabilities(),
                                 backend().device_info());
      impl_->events.assign(staged_groups.size(), backend::StreamEvent{});
      trace_.streams_used = impl_->plan.streams_used;
      trace_.stream_waits = impl_->plan.waits_total;
      trace_.stream_chain = impl_->plan.chain;

      // The step inherits whatever earlier steps left running. Snapshot each
      // such stream *before* issuing anything, so a group that later joins a
      // different stream waits for the previous step and not for this one's
      // work as well — and snapshot only the streams this plan can actually
      // race with, which in a single-stream step is none of them.
      const std::uint32_t stream_count =
          backend().stream_capabilities().stream_count;
      if (impl_->outstanding.size() != stream_count) {
        impl_->outstanding.assign(stream_count, 0);
      }
      impl_->entry_events.assign(stream_count, backend::StreamEvent{});
      std::vector<std::uint8_t> plans_on(stream_count, 0);
      for (std::uint32_t i = 0; i < impl_->plan.stream.size(); ++i) {
        if (impl_->plan.stream[i] < stream_count) {
          plans_on[impl_->plan.stream[i]] = 1;
        }
      }
      for (std::uint32_t s = 0; s < stream_count; ++s) {
        if (impl_->outstanding[s] == 0) continue;
        bool raced = false;
        for (std::uint32_t t = 0; t < stream_count && !raced; ++t) {
          raced = t != s && plans_on[t] != 0;
        }
        if (!raced) continue;
        auto ev = backend().record_event(backend::Stream{s});
        if (ev.ok()) impl_->entry_events[s] = ev.release();
      }
      std::vector<std::uint8_t> entered(stream_count, 0);

      // Nothing has been issued yet, so this is where the pre-dispatch region
      // ends. The old partition_ns closed AFTER the loop below and so contained
      // every emit, compile and submit in the step.
      place_span.close();

      bool launched_phase = true;
      std::size_t done = 0;
      for (const FusionGroup& g : staged_groups) {
        const std::uint32_t gi = static_cast<std::uint32_t>(done);
        if (views_only(g)) {
          alias_ready_reshapes(g);
          ++trace_.views_aliased;
          trace_.nodes_evaluated += static_cast<std::uint32_t>(g.nodes.size());
          if (!replayed) ran.push_back(g);
          ++done;
          continue;
        }
        const backend::Stream on{impl_->plan.stream[gi]};
        Status st = OkStatus();
        if (on.index < stream_count && entered[on.index] == 0) {
          entered[on.index] = 1;
          for (std::uint32_t s = 0; s < stream_count && st.ok(); ++s) {
            if (s == on.index) continue;
            st = backend().wait_event(on, impl_->entry_events[s]);
          }
        }
        for (std::uint32_t j : impl_->plan.waits[gi]) {
          if (!st.ok()) break;
          st = backend().wait_event(on, impl_->events[j]);
          if (!st.ok()) break;
        }
        if (st.ok()) st = try_dispatch_group(g, on, member_for(g));
        // A joined run the emitter cannot express is not a reason to abandon
        // the phase: its members are independent by construction, so each one
        // still dispatches alone. Splitting costs one launch per member;
        // giving up costs a full repartition of the step and every group in it
        // re-placed, which measured seconds per wide prefill pass.
        std::vector<FusionGroup> split;
        if (!st.ok() && !g.is_phase && g.nodes.size() > 1) {
          Status each = OkStatus();
          for (const NodePtr& n : g.nodes) {
            FusionGroup one;
            one.nodes.push_back(n);
            one.outputs.push_back(n);
            one.inputs = n->inputs;
            one.anchor = n->kind;
            one.anchor_class = n->fclass;
            each = try_dispatch_group(one, on, member_for(one));
            if (!each.ok()) break;
            ++trace_.device_groups;
            ++trace_.kernels_launched;
            split.push_back(std::move(one));
          }
          if (each.ok()) {
            st = OkStatus();
          } else {
            trace_.device_groups -= static_cast<std::uint32_t>(split.size());
            trace_.kernels_launched -= static_cast<std::uint32_t>(split.size());
            split.clear();
          }
        }
        if (st.ok() && impl_->plan.record_after[gi] != 0) {
          auto ev = backend().record_event(on);
          if (ev.ok()) {
            impl_->events[gi] = ev.release();
          } else {
            st = ev.status();
          }
        }
        if (!st.ok()) {
          if (std::getenv("LSE_DEBUG_PHASES") != nullptr) {
            std::fprintf(stderr, "phase dispatch failed: %s\n",
                         st.to_string().c_str());
          }
          launched_phase = false;
          break;
        }
        alias_ready_reshapes(g);
        if (on.index < stream_count) impl_->outstanding[on.index] = 1;
        if (split.empty()) {
          ++trace_.device_groups;
          ++trace_.kernels_launched;
        }
        trace_.nodes_evaluated += static_cast<std::uint32_t>(g.nodes.size());
        // The replay has to see what actually ran, not the group that was
        // refused, or it re-runs the same refusal every pass.
        if (!replayed) {
          if (split.empty()) {
            ran.push_back(g);
          } else {
            for (FusionGroup& one : split) ran.push_back(std::move(one));
          }
        }
        ++done;
      }
      if (launched_phase) {
        if (!replayed) rec.retain(roots, std::move(planned), ran, order);
        if (pull_host) {
          const auto t_wait = SpanClock::now();
          const Status waited = backend().synchronize();
          trace_.spans.host_wait.add(elapsed_ns(t_wait, SpanClock::now()));
          LSE_RETURN_IF_ERROR(waited);
          // The device is idle: nothing is left for a later step to be
          // ordered against, so the next one starts with no entry barriers.
          // Host bookkeeping, and outside host_wait for that reason.
          std::fill(impl_->outstanding.begin(), impl_->outstanding.end(), 0);
          SpanTimer readback_span(trace_.spans.readback);
          for (const NodePtr& r : roots) {
            if (r) {
              LSE_RETURN_IF_ERROR(interpreter::sync_from_device(*r, backend()));
            }
          }
        }
        for (auto& t : impl_->phase_tables) release(t);
        impl_->phase_tables.clear();
        return OkStatus();
      }
    // Prefix in `ran` already launched. Partition only the rest; retain
    // prefix + suffix so the next reset_compute has a full cover. A failed
    // replay recovers its launched prefix from the retained groups here.
    if (replayed) {
      ran.assign(staged_groups.begin(),
                 staged_groups.begin() + static_cast<std::ptrdiff_t>(done));
    }
  }

  // Already-closed is a no-op, so a failed device-first attempt contributes its
  // dispatch loop to the remainder rather than being folded in here — which is
  // what the old partition_ns did, silently.
  const auto t_repartition = place_span.close();
  std::vector<FusionGroup> groups =
      Partitioner::partition(roots, &backend().device_info());
  trace_.spans.partition.add(elapsed_ns(t_repartition, SpanClock::now()));
  // Every group taken here has to land in `ran`, on whichever arm ran it.
  // retain() below is what the next step replays, and a replay first calls
  // reset_compute() on the whole reachable graph: a cover missing the device
  // and view groups leaves their nodes unmaterialized and unrecomputed, and the
  // step after that reaches them again through the state carry chain and
  // re-runs a stale KV write against a regrown pool.
  bool launched = false;
  for (const FusionGroup& g : groups) {
    std::string desc(to_string(g.anchor));
    desc += " [";
    for (std::size_t i = 0; i < g.nodes.size(); ++i) {
      if (i) desc += "+";
      desc += to_string(g.nodes[i]->kind);
    }
    desc += "]";
    trace_.group_descriptions.push_back(std::move(desc));

    if (mode_ == Mode::kDeviceFirst) {
      if (try_alias_group(g).ok()) {
        ++trace_.views_aliased;
        trace_.nodes_evaluated += static_cast<std::uint32_t>(g.nodes.size());
        ran.push_back(g);
        continue;
      }
      Status dispatched =
          try_dispatch_group(g, backend::kDefaultStream, member_for(g));
      if (dispatched.ok()) {
        ++trace_.device_groups;
        ++trace_.kernels_launched;
        trace_.nodes_evaluated += static_cast<std::uint32_t>(g.nodes.size());
        launched = true;
        ran.push_back(g);
        continue;
      }
      // Not a hard failure: a group the emitter cannot express falls to the
      // host, which is the documented behaviour. Record why.
      ++trace_.host_groups;
      trace_.host_group_reasons.push_back(dispatched.message());
    }

    if (launched) {
      const auto t_wait = SpanClock::now();
      const Status waited = backend().synchronize();
      trace_.spans.host_wait.add(elapsed_ns(t_wait, SpanClock::now()));
      LSE_RETURN_IF_ERROR(waited);
      launched = false;
    }

    // Host-only mode reaches here without passing the device-first arm above,
    // so nothing had counted this group. Leaving it uncounted printed
    // `groups device=0 host=0` beside a five-figure launch count, which reads
    // like broken instrumentation and hid a run that was executing the whole
    // model on the host interpreter. Every group is now in exactly one bucket.
    if (mode_ != Mode::kDeviceFirst) ++trace_.host_groups;

    SpanTimer host_exec_span(trace_.spans.host_exec);
    for (const NodePtr& n : g.nodes) {
      // Routing first: an intercepting handler may claim a node the backend
      // could have run, which is how per-op device placement is switched at
      // runtime.
      if (const FallbackHandler* router =
              fallback_chain().resolve_intercept(*n, backend())) {
        LSE_RETURN_IF_ERROR(interpreter::ensure_output_buffer(*n, backend()));
        LSE_RETURN_IF_ERROR(router->execute(*n, backend()));
        if (!n->materialized) {
          return LSE_ERROR(kInternal, "handler '", std::string(router->name()),
                           "' returned OK without materializing the node");
        }
        ++trace_.intercepted;
        trace_.fallback_handlers.emplace_back(router->name());
        ++trace_.nodes_evaluated;
        continue;
      }

      const std::string gap = device_gap(*n, backend());
      if (gap.empty()) {
        LSE_RETURN_IF_ERROR(interpreter::evaluate(n, backend()));
        ++trace_.nodes_evaluated;
        continue;
      }

      const FallbackHandler* handler = fallback_chain().resolve(*n, backend());
      if (handler == nullptr) {
        return LSE_ERROR(kUnimplemented, "node '", std::string(to_string(n->kind)),
                         "' cannot run on ", std::string(backend().name()),
                         " (", gap, ") and no fallback handler accepted it");
      }
      LSE_RETURN_IF_ERROR(interpreter::ensure_output_buffer(*n, backend()));
      LSE_RETURN_IF_ERROR(handler->execute(*n, backend()));
      if (!n->materialized) {
        return LSE_ERROR(kInternal, "fallback handler '",
                         std::string(handler->name()),
                         "' returned OK without materializing the node");
      }
      ++trace_.host_fallbacks;
      trace_.fallback_reasons.push_back(gap);
      trace_.fallback_handlers.emplace_back(handler->name());
      ++trace_.nodes_evaluated;
    }
    host_exec_span.close();
    if (is_collective(g.anchor)) ++trace_.collectives_issued;
    ++trace_.kernels_launched;
    ran.push_back(g);
  }

  if (launched) {
    const auto t_wait = SpanClock::now();
    const Status waited = backend().synchronize();
    trace_.spans.host_wait.add(elapsed_ns(t_wait, SpanClock::now()));
    LSE_RETURN_IF_ERROR(waited);
  }
  // A host-visible eval pulls the roots back. materialize() leaves them on
  // the device so the next kernel can read them without a round trip.
  if (pull_host) {
    SpanTimer readback_span(trace_.spans.readback);
    for (const NodePtr& root : roots) {
      LSE_RETURN_IF_ERROR(interpreter::sync_from_device(*root, backend()));
    }
  }
  rec.retain(roots, std::move(planned), ran, order);
  return OkStatus();
}

namespace {

// Zero-initialized before any dynamic initialization runs, so the layer that
// owns device lifetimes can register during its own static init and be seen
// here whenever the first scheduler is asked for.
DeviceSetFactory g_device_set_factory = nullptr;

struct DefaultScheduler {
  std::unique_ptr<backend::IBackend> backend;
  std::unique_ptr<Scheduler> scheduler;

  DefaultScheduler() {
    // The set, when the layer that owns one is linked. It has already applied
    // whatever the run asked for; nothing is decided here.
    if (g_device_set_factory != nullptr) {
      if (backend::IDeviceSet* set = g_device_set_factory();
          set != nullptr && set->size() > 0) {
        scheduler = std::make_unique<Scheduler>(*set);
      }
      // Registered and empty means nothing came up. Falling through to the loop
      // below would bind a second instance of the device it just failed on.
      return;
    }

    // No such layer in this binary — a test, or a tool that only needs one
    // device. Bind one the same way, which is the set of size one.
    //
    // LSE_DEVICE picks the ordinal; on a mixed box the discrete card and the
    // integrated one are different devices with different memory models.
    int ordinal = 0;
    if (const char* env = std::getenv("LSE_DEVICE")) ordinal = std::atoi(env);

    // Preference order has to survive init, not just construction: the HRX
    // backend builds fine and then fails to come up when the HSA runtime on the
    // loader path is too old. Treating that as fatal would turn preferring the
    // GPU into a hard stop on any machine without one.
    for (const std::string& name : backend::default_backend_order()) {
      auto be = backend::create_backend(name);
      if (!be.ok()) continue;
      auto candidate = be.release();
      if (const Status init = candidate->init(ordinal); !init.ok()) continue;
      backend = std::move(candidate);
      scheduler = std::make_unique<Scheduler>(*backend);
      return;
    }
  }
};

}  // namespace

void register_device_set_factory(DeviceSetFactory factory) {
  if (g_device_set_factory == nullptr) g_device_set_factory = factory;
}

Scheduler* default_scheduler() {
  static DefaultScheduler d;
  return d.scheduler.get();
}

}  // namespace lse::graph
