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

  Status ensure_jit(backend::IBackend& backend) {
    if (jit != nullptr) return OkStatus();
    const IKernelCompiler* compiler = backend.compiler();
    if (compiler == nullptr) {
      return LSE_ERROR(kUnimplemented, "backend '",
                       std::string(backend.name()),
                       "' has no kernel compiler");
    }
    jit = std::make_unique<JitCache>(backend, *compiler);
    return OkStatus();
  }
};

Scheduler::Scheduler(backend::IBackend& backend)
    : backend_(backend), impl_(std::make_unique<Impl>()) {
  // Without codegen there is no code-object path, so device-first is
  // meaningless.
  if (backend_.emitter() == nullptr) mode_ = Mode::kHostOnly;
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

Status Scheduler::try_dispatch_group(const FusionGroup& group,
                                     backend::Stream stream) {
  const IKernelEmitter* emitter = backend_.emitter();
  if (emitter == nullptr) {
    return LSE_ERROR(kUnimplemented, "backend '", std::string(backend_.name()),
                     "' has no kernel emitter");
  }
  LSE_RETURN_IF_ERROR(impl_->ensure_jit(backend_));

  const std::uint64_t ident =
      emitter->cache_key(group, backend_.device_info());

  const auto t_emit = std::chrono::steady_clock::now();
  auto emitted = emitter->emit(group, backend_.device_info());
  if (!emitted.ok()) return emitted.status();
  dump_hip_source(*emitted, ident);

  // Compile only when this kernel is not already loaded for this device.
  // Disk miss / source change / arch change still go through get_or_compile.
  backend::KernelHandle launched;
  if (const backend::KernelHandle* cached = impl_->jit->try_get(ident)) {
    launched = *cached;
  } else {
    auto kernel = impl_->jit->get_or_compile(ident, *emitted);
    if (!kernel.ok()) return kernel.status();
    launched = kernel.release();
  }
  trace_.emit_ns += static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - t_emit)
          .count());

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
      LSE_RETURN_IF_ERROR(interpreter::ensure_output_buffer(*n, backend_));
    }
    if (fresh && n->kind == OpKind::kConstant) {
      for (std::size_t e = 0; e < n->element_count(); ++e) {
        interpreter::store_element(*n, e, n->attrs[0]);
      }
      n->materialized = true;
    }
    // Weights and any host-computed input have to reach device memory before
    // the kernel reads them.
    LSE_RETURN_IF_ERROR(interpreter::sync_to_device(*n, backend_));
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
      auto p = backend_.device_pointer(*bindings[i].buffer);
      if (!p.ok()) return p.status();
      ptrs[i] = *p;
    }
    auto tab = backend_.allocate(ptrs.size() * sizeof(void*),
                                 backend::MemoryClass::kDevice);
    if (!tab.ok()) return tab.status();
    table_buf = tab.release();
    LSE_RETURN_IF_ERROR(backend_.copy_h2d(ptrs.data(), table_buf,
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
      auto bar = backend_.allocate(sizeof(std::uint32_t),
                                   backend::MemoryClass::kDevice);
      if (!bar.ok()) return bar.status();
      bar_buf = bar.release();
      const std::uint32_t zero = 0;
      LSE_RETURN_IF_ERROR(backend_.copy_h2d(&zero, bar_buf, sizeof(zero), 0));
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
  const auto t_launch = std::chrono::steady_clock::now();
  LSE_RETURN_IF_ERROR(backend_.launch(launched, emitted->dims, args,
                                      backend::DispatchTarget{stream, {}}));
  trace_.launch_ns += static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - t_launch)
          .count());

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

void accumulate(Scheduler::Trace& acc, const Scheduler::Trace& step) {
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
  acc.partition_ns += step.partition_ns;
  acc.emit_ns += step.emit_ns;
  acc.launch_ns += step.launch_ns;
  acc.sync_ns += step.sync_ns;
  acc.host_group_reasons.insert(acc.host_group_reasons.end(),
                                step.host_group_reasons.begin(),
                                step.host_group_reasons.end());
  acc.replayed = acc.replayed || step.replayed;
}

}  // namespace

Status Scheduler::eval(std::span<const NodePtr> roots, bool pull_host) {
  return eval(roots, pull_host, nullptr);
}

Status Scheduler::eval(std::span<const NodePtr> roots, bool pull_host,
                      Program* plan) {
  Program& rec = plan != nullptr ? *plan : impl_->program;
  trace_ = Trace{};
  for (auto& t : impl_->phase_tables) backend_.deallocate(t);
  impl_->phase_tables.clear();

  const auto t_part = std::chrono::steady_clock::now();
  const std::vector<NodePtr> order = Partitioner::unmaterialized(roots);

  std::vector<FusionGroup> phase_groups;
  bool replayed = false;
  if (mode_ == Mode::kDeviceFirst && backend_.emitter() != nullptr &&
      rec.holds(roots) && !rec.groups().empty()) {
    bool ready = !roots.empty();
    for (const NodePtr& r : roots) {
      if (r && !r->materialized) {
        ready = false;
        break;
      }
    }
    if (ready) {
      if (pull_host) {
        for (const NodePtr& r : roots) {
          if (r) LSE_RETURN_IF_ERROR(interpreter::sync_from_device(*r, backend_));
        }
      }
      accumulate(acc_, trace_);
      return OkStatus();
    }
    replayed = true;
    trace_.replayed = true;
    trace_.phase_groups = static_cast<std::uint32_t>(rec.phases().size());
    if (trace_.phase_groups == 0) trace_.phase_groups = 1;
    trace_.phase_ideal_launches = 1;
  }

  auto planned = replayed
                     ? std::vector<Workgroup>{}
                     : Partitioner::phases(roots, &backend_.device_info());
  if (!replayed) {
    trace_.phase_groups = static_cast<std::uint32_t>(planned.size());
    for (const Workgroup& wg : planned) {
      trace_.phase_ideal_launches += wg.ideal_launches();
    }
  }

  if (mode_ == Mode::kDeviceFirst && backend_.emitter() != nullptr &&
      !replayed) {
    // Null when this emitter has no staged phase body: every node then takes
    // the same one-group path a node it refuses to stage already takes.
    const IPhaseStaging* staging = backend_.emitter()->staging();
    for (Workgroup& wg : planned) {
      FusionGroup g = Partitioner::phase_group(wg, roots);
      if (g.nodes.empty()) continue;
      alias_ready_reshapes(g);
      FusionGroup staged;
      staged.is_phase = true;
      staged.launches = 1;
      staged.anchor_class = FusionClass::kBarrier;
      bool staged_grid = false;
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
        if (name != "linear" && name != "linear.lds" &&
            name != "linear_indexed" && name != "linear_indexed.lds") {
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
            staging->stage_threads(*n, backend_.device_info()) >= kGridStage;
        const bool grid_chunk = staged_grid && !staged.nodes.empty();
        // A grid chunk keeps growing while its stages stay independent, so
        // sibling wide stages cost one launch, not one each.
        if (fat ? (!grid_chunk || reads_staged(n))
                : (grid_chunk && reads_staged(n))) {
          flush_staged();
        }
        if (staged.nodes.empty()) staged_grid = fat;
        staged.nodes.push_back(n);
      }
      flush_staged();
      wg.plan_slots(roots);
      if (wg.bind_slots(backend_).ok()) {
        trace_.slots_reused += wg.reused_slots();
        trace_.slots_allocated += wg.slot_count();
      }
    }
  }

  std::vector<FusionGroup> ran;
  // Replays run the retained groups in place: no copy of the group list, and
  // no re-retain afterwards — retain() re-walks the whole reachable graph and
  // was most of the host-side churn of a decode token.
  const std::vector<FusionGroup>& staged_groups =
      replayed ? rec.groups() : phase_groups;
  if (mode_ == Mode::kDeviceFirst && backend_.emitter() != nullptr &&
      !staged_groups.empty()) {
      // Placement for the whole step, decided before any of it is issued.
      impl_->plan = plan_streams(staged_groups, backend_.stream_capabilities(),
                                 backend_.device_info());
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
          backend_.stream_capabilities().stream_count;
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
        auto ev = backend_.record_event(backend::Stream{s});
        if (ev.ok()) impl_->entry_events[s] = ev.release();
      }
      std::vector<std::uint8_t> entered(stream_count, 0);

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
            st = backend_.wait_event(on, impl_->entry_events[s]);
          }
        }
        for (std::uint32_t j : impl_->plan.waits[gi]) {
          if (!st.ok()) break;
          st = backend_.wait_event(on, impl_->events[j]);
          if (!st.ok()) break;
        }
        if (st.ok()) st = try_dispatch_group(g, on);
        if (st.ok() && impl_->plan.record_after[gi] != 0) {
          auto ev = backend_.record_event(on);
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
        ++trace_.device_groups;
        ++trace_.kernels_launched;
        trace_.nodes_evaluated += static_cast<std::uint32_t>(g.nodes.size());
        if (!replayed) ran.push_back(g);
        ++done;
      }
      if (launched_phase) {
        if (!replayed) {
          rec.retain(roots, std::move(planned), ran, order);
          trace_.partition_ns = static_cast<std::uint64_t>(
              std::chrono::duration_cast<std::chrono::nanoseconds>(
                  std::chrono::steady_clock::now() - t_part)
                  .count());
        }
        if (pull_host) {
          const auto t_sync = std::chrono::steady_clock::now();
          LSE_RETURN_IF_ERROR(backend_.synchronize());
          // The device is idle: nothing is left for a later step to be
          // ordered against, so the next one starts with no entry barriers.
          std::fill(impl_->outstanding.begin(), impl_->outstanding.end(), 0);
          trace_.sync_ns += static_cast<std::uint64_t>(
              std::chrono::duration_cast<std::chrono::nanoseconds>(
                  std::chrono::steady_clock::now() - t_sync)
                  .count());
          for (const NodePtr& r : roots) {
            if (r) {
              LSE_RETURN_IF_ERROR(interpreter::sync_from_device(*r, backend_));
            }
          }
        }
        for (auto& t : impl_->phase_tables) backend_.deallocate(t);
        impl_->phase_tables.clear();
        accumulate(acc_, trace_);
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

  std::vector<FusionGroup> groups =
      Partitioner::partition(roots, &backend_.device_info());
  trace_.partition_ns = static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - t_part)
          .count());
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
        continue;
      }
      Status dispatched = try_dispatch_group(g, backend::kDefaultStream);
      if (dispatched.ok()) {
        ++trace_.device_groups;
        ++trace_.kernels_launched;
        trace_.nodes_evaluated += static_cast<std::uint32_t>(g.nodes.size());
        launched = true;
        continue;
      }
      // Not a hard failure: a group the emitter cannot express falls to the
      // host, which is the documented behaviour. Record why.
      ++trace_.host_groups;
      trace_.host_group_reasons.push_back(dispatched.message());
    }

    if (launched) {
      const auto t_sync = std::chrono::steady_clock::now();
      LSE_RETURN_IF_ERROR(backend_.synchronize());
      trace_.sync_ns += static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now() - t_sync)
              .count());
      launched = false;
    }

    // Host-only mode reaches here without passing the device-first arm above,
    // so nothing had counted this group. Leaving it uncounted printed
    // `groups device=0 host=0` beside a five-figure launch count, which reads
    // like broken instrumentation and hid a run that was executing the whole
    // model on the host interpreter. Every group is now in exactly one bucket.
    if (mode_ != Mode::kDeviceFirst) ++trace_.host_groups;

    for (const NodePtr& n : g.nodes) {
      // Routing first: an intercepting handler may claim a node the backend
      // could have run, which is how per-op device placement is switched at
      // runtime.
      if (const FallbackHandler* router =
              fallback_chain().resolve_intercept(*n, backend_)) {
        LSE_RETURN_IF_ERROR(interpreter::ensure_output_buffer(*n, backend_));
        LSE_RETURN_IF_ERROR(router->execute(*n, backend_));
        if (!n->materialized) {
          return LSE_ERROR(kInternal, "handler '", std::string(router->name()),
                           "' returned OK without materializing the node");
        }
        ++trace_.intercepted;
        trace_.fallback_handlers.emplace_back(router->name());
        ++trace_.nodes_evaluated;
        continue;
      }

      const std::string gap = device_gap(*n, backend_);
      if (gap.empty()) {
        LSE_RETURN_IF_ERROR(interpreter::evaluate(n, backend_));
        ++trace_.nodes_evaluated;
        continue;
      }

      const FallbackHandler* handler = fallback_chain().resolve(*n, backend_);
      if (handler == nullptr) {
        return LSE_ERROR(kUnimplemented, "node '", std::string(to_string(n->kind)),
                         "' cannot run on ", std::string(backend_.name()),
                         " (", gap, ") and no fallback handler accepted it");
      }
      LSE_RETURN_IF_ERROR(interpreter::ensure_output_buffer(*n, backend_));
      LSE_RETURN_IF_ERROR(handler->execute(*n, backend_));
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
    if (is_collective(g.anchor)) ++trace_.collectives_issued;
    ++trace_.kernels_launched;
    ran.push_back(g);
  }

  if (launched) {
    const auto t_sync = std::chrono::steady_clock::now();
    LSE_RETURN_IF_ERROR(backend_.synchronize());
    trace_.sync_ns += static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - t_sync)
            .count());
  }
  // A host-visible eval pulls the roots back. materialize() leaves them on
  // the device so the next kernel can read them without a round trip.
  if (pull_host) {
    for (const NodePtr& root : roots) {
      LSE_RETURN_IF_ERROR(interpreter::sync_from_device(*root, backend_));
    }
  }
  rec.retain(roots, std::move(planned), ran, order);
  accumulate(acc_, trace_);
  return OkStatus();
}

namespace {

struct DefaultScheduler {
  std::unique_ptr<backend::IBackend> backend;
  std::unique_ptr<Scheduler> scheduler;

  DefaultScheduler() {
    // LSE_DEVICE picks the ordinal; on a mixed box the discrete card and the
    // integrated one are different devices with different memory models.
    int ordinal = 0;
    if (const char* env = std::getenv("LSE_DEVICE")) ordinal = std::atoi(env);

    // Preference order has to survive init, not just construction: the HRX
    // backend builds fine and then fails to come up when the HSA runtime on the
    // loader path is too old. Treating that as fatal would turn preferring the
    // GPU into a hard stop on any machine without one.
    std::string declined;
    for (const std::string& name : backend::default_backend_order()) {
      auto be = backend::create_backend(name);
      if (!be.ok()) continue;
      auto candidate = be.release();
      if (const Status init = candidate->init(ordinal); !init.ok()) {
        if (declined.empty()) declined = name + ": " + init.to_string();
        continue;
      }
      backend = std::move(candidate);
      scheduler = std::make_unique<Scheduler>(*backend);
      // Falling back to a backend with no code generator runs the whole model
      // through the host interpreter. That is a two-order-of-magnitude cliff
      // and it used to be silent: the `--stats` line reported zero device
      // groups next to a five-figure launch count and read like broken
      // instrumentation. Say it once, where it happens.
      if (backend->emitter() == nullptr && !declined.empty()) {
        std::fprintf(stderr,
                     "lse: no code-generating backend came up (%s); running on "
                     "'%s' through the host interpreter\n",
                     declined.c_str(), std::string(backend->name()).c_str());
      }
      return;
    }
  }
};

}  // namespace

Scheduler* default_scheduler() {
  static DefaultScheduler d;
  return d.scheduler.get();
}

}  // namespace lse::graph
