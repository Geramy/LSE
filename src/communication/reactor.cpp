#include "lse/communication/reactor.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "lse/communication/transport.hpp"
#include "lse/core/debug.hpp"
#include "lse/communication/poller.hpp"

namespace lse::comm {

namespace {

constexpr std::size_t kArenaBlockBytes = 64u << 10;

// A control payload is copied out of the link that produced it, because an
// event may outlive that link: a dangling span is not a diagnosable failure.
// Blocks never move, so a pointer handed out this sweep stays good until the
// next poll releases the whole arena at once.
class Arena {
 public:
  const std::byte* put(std::span<const std::byte> bytes) {
    if (bytes.empty()) return nullptr;
    for (std::size_t i = cursor_; i < blocks_.size(); ++i) {
      Block& b = blocks_[i];
      if (b.capacity - b.used >= bytes.size()) {
        std::byte* at = b.data.get() + b.used;
        std::memcpy(at, bytes.data(), bytes.size());
        b.used += bytes.size();
        cursor_ = i;
        return at;
      }
    }
    const std::size_t capacity = std::max(kArenaBlockBytes, bytes.size());
    Block fresh;
    fresh.data = std::make_unique<std::byte[]>(capacity);
    fresh.capacity = capacity;
    fresh.used = bytes.size();
    std::memcpy(fresh.data.get(), bytes.data(), bytes.size());
    std::byte* at = fresh.data.get();
    cursor_ = blocks_.size();
    blocks_.push_back(std::move(fresh));
    return at;
  }

  void reset() {
    const bool poison = debug();
    for (Block& b : blocks_) {
      if (poison && b.used > 0) std::memset(b.data.get(), 0xDD, b.used);
      b.used = 0;
    }
    cursor_ = 0;
  }

 private:
  struct Block {
    std::unique_ptr<std::byte[]> data;
    std::size_t capacity = 0;
    std::size_t used = 0;
  };
  std::vector<Block> blocks_;
  std::size_t cursor_ = 0;
};

struct OpSlot {
  std::uint32_t generation = 0;
  bool live = false;
};

struct ChannelSlot {
  std::uint32_t generation = 0;
  bool live = false;
  ILink* link = nullptr;
  std::size_t transport = 0;
  std::uint32_t max_inflight = 64;
  std::uint32_t cursor = 0;
  std::vector<OpSlot> ops;
  Endpoint peer;
  Capabilities caps;
  Status error;
};

struct ListenerSlot {
  std::uint32_t generation = 0;
  bool live = false;
  IListenerImpl* impl = nullptr;
  std::size_t transport = 0;
  Endpoint bound;
};

struct Deadline {
  std::uint64_t at_ns = 0;
  std::uint64_t channel = 0;
  std::uint64_t op = 0;
  friend bool operator<(const Deadline& a, const Deadline& b) noexcept {
    return a.at_ns > b.at_ns;  // std::push_heap builds a max-heap
  }
};

const Status& ok_status() noexcept {
  static const Status ok;
  return ok;
}
const Endpoint& nowhere() noexcept {
  static const Endpoint none;
  return none;
}
const Capabilities& nothing() noexcept {
  static const Capabilities none;
  return none;
}

std::uint64_t pack(std::size_t index, std::uint32_t generation) noexcept {
  return static_cast<std::uint64_t>(index + 1) |
         (static_cast<std::uint64_t>(generation) << 32);
}
std::size_t index_of(std::uint64_t id) noexcept {
  return static_cast<std::size_t>((id & 0xFFFF'FFFFull) - 1);
}
std::uint32_t generation_of(std::uint64_t id) noexcept {
  return static_cast<std::uint32_t>(id >> 32);
}

}  // namespace

std::uint64_t steady_now_ns() noexcept {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

class ReactorState {
 public:
  struct Entry;

  class Sink final : public EventSink {
   public:
    Sink(ReactorState& state, std::size_t transport) noexcept
        : state_(&state), transport_(transport) {}

    void emit(const Event& ev) override { state_->file(ev); }
    const std::byte* retain(std::span<const std::byte> bytes) override {
      return state_->arena.put(bytes);
    }
    std::uint64_t reserve_channel() override {
      return state_->open_channel(transport_);
    }
    void adopt(std::uint64_t channel, ILink* link,
               std::uint64_t listener) override {
      state_->adopt(channel, link, listener);
    }

   private:
    ReactorState* state_ = nullptr;
    std::size_t transport_ = 0;
  };

  struct Entry {
    std::string scheme;
    std::unique_ptr<ITransport> transport;
    std::unique_ptr<Sink> sink;
  };

  explicit ReactorState(Poller p) noexcept : poller(std::move(p)) {}
  ~ReactorState() {
    for (ChannelSlot& c : channels) {
      if (c.live && c.link != nullptr) c.link->abort();
    }
    for (ListenerSlot& l : listeners) {
      if (l.live && l.impl != nullptr) (void)l.impl->close();
    }
    for (const auto& e : transports) e->transport->close();
  }
  ReactorState(const ReactorState&) = delete;
  ReactorState& operator=(const ReactorState&) = delete;

  Poller poller;
  Arena arena;
  std::vector<Event> pending;
  std::size_t delivered = 0;
  std::vector<std::unique_ptr<Entry>> transports;
  std::vector<ChannelSlot> channels;
  std::vector<ListenerSlot> listeners;
  std::vector<Deadline> deadlines;
  std::vector<int> wait_fds;
  std::thread::id owner = std::this_thread::get_id();

  [[nodiscard]] ChannelSlot* channel_slot(std::uint64_t id) noexcept {
    if (id == 0) return nullptr;
    const std::size_t i = index_of(id);
    if (i >= channels.size()) return nullptr;
    ChannelSlot& c = channels[i];
    if (!c.live || c.generation != generation_of(id)) return nullptr;
    return &c;
  }

  [[nodiscard]] ListenerSlot* listener_slot(std::uint64_t id) noexcept {
    if (id == 0) return nullptr;
    const std::size_t i = index_of(id);
    if (i >= listeners.size()) return nullptr;
    ListenerSlot& l = listeners[i];
    if (!l.live || l.generation != generation_of(id)) return nullptr;
    return &l;
  }

  void file(const Event& ev) {
    pending.push_back(ev);
    if (ev.kind == EventKind::kSendComplete ||
        ev.kind == EventKind::kRecvComplete) {
      free_op(ev.channel, ev.op);
    } else if (ev.kind == EventKind::kClosed && ev.channel != 0) {
      close_channel(ev.channel);
    }
  }

  std::uint64_t open_channel(std::size_t transport) {
    std::size_t index = channels.size();
    for (std::size_t i = 0; i < channels.size(); ++i) {
      if (!channels[i].live) {
        index = i;
        break;
      }
    }
    if (index == channels.size()) channels.emplace_back();
    ChannelSlot& c = channels[index];
    ++c.generation;
    c.live = true;
    c.link = nullptr;
    c.transport = transport;
    c.cursor = 0;
    c.error = Status();
    return pack(index, c.generation);
  }

  void bind_channel(std::uint64_t id, ILink* link) {
    ChannelSlot* c = channel_slot(id);
    if (c == nullptr) return;
    c->link = link;
    c->peer = link->peer();
    c->caps = link->capabilities();
    c->max_inflight = c->caps.max_inflight != 0 ? c->caps.max_inflight : 64u;
    c->ops.assign(static_cast<std::size_t>(c->max_inflight) * 2u, OpSlot{});
  }

  void adopt(std::uint64_t channel, ILink* link, std::uint64_t listener) {
    bind_channel(channel, link);
    Event ev;
    ev.kind = EventKind::kAccepted;
    ev.channel = channel;
    ev.listener = listener;
    pending.push_back(ev);
  }

  void close_channel(std::uint64_t id) {
    ChannelSlot* c = channel_slot(id);
    if (c == nullptr) return;
    if (c->link != nullptr) c->error = c->link->last_error();
    c->link = nullptr;
    c->live = false;
    for (OpSlot& op : c->ops) op.live = false;
  }

  // op ids live in one table per channel; the lane picks which half.
  [[nodiscard]] Result<std::uint64_t> alloc_op(ChannelSlot& c, Lane lane) {
    const std::size_t half = c.ops.size() / 2;
    const std::size_t base = lane == Lane::kControl ? 0 : half;
    for (std::size_t n = 0; n < half; ++n) {
      const std::size_t i = base + (c.cursor + n) % half;
      if (c.ops[i].live) continue;
      c.cursor = static_cast<std::uint32_t>((c.cursor + n + 1) % half);
      ++c.ops[i].generation;
      c.ops[i].live = true;
      return pack(i, c.ops[i].generation);
    }
    return LSE_ERROR(kOutOfMemory, std::to_string(half),
                     " operations are already outstanding on this lane; a "
                     "deeper queue would turn a slow peer into an OOM kill");
  }

  void free_op(std::uint64_t channel, std::uint64_t op) noexcept {
    ChannelSlot* c = channel_slot(channel);
    if (c == nullptr || op == 0) return;
    const std::size_t i = index_of(op);
    if (i >= c->ops.size()) return;
    if (c->ops[i].generation != generation_of(op)) return;
    c->ops[i].live = false;
  }

  [[nodiscard]] bool op_live(const ChannelSlot& c, std::uint64_t op) const noexcept {
    if (op == 0) return false;
    const std::size_t i = index_of(op);
    if (i >= c.ops.size()) return false;
    return c.ops[i].live && c.ops[i].generation == generation_of(op);
  }

  [[nodiscard]] bool op_known(const ChannelSlot& c,
                              std::uint64_t op) const noexcept {
    if (op == 0) return false;
    const std::size_t i = index_of(op);
    if (i >= c.ops.size()) return false;
    return c.ops[i].generation == generation_of(op);
  }

  void arm_deadline(std::uint64_t channel, std::uint64_t op,
                    std::uint64_t at_ns) {
    if (at_ns == 0) return;
    deadlines.push_back(Deadline{at_ns, channel, op});
    std::push_heap(deadlines.begin(), deadlines.end());
  }

  void expire(std::uint64_t now_ns) {
    while (!deadlines.empty() && deadlines.front().at_ns <= now_ns) {
      const Deadline due = deadlines.front();
      std::pop_heap(deadlines.begin(), deadlines.end());
      deadlines.pop_back();
      ChannelSlot* c = channel_slot(due.channel);
      if (c == nullptr || c->link == nullptr || !op_live(*c, due.op)) continue;
      ILink* link = c->link;
      const Status s = link->cancel(due.op);
      if (s.ok() || s.code() == StatusCode::kNotFound) continue;
      // Partly on the wire: it cannot be un-sent, so the channel goes. A tight
      // deadline on a large transfer is arming a channel kill, and the header
      // says so.
      link->abort();
    }
  }

  [[nodiscard]] std::uint64_t next_wakeup_ns() const noexcept {
    std::uint64_t soonest = deadlines.empty() ? 0 : deadlines.front().at_ns;
    for (const auto& e : transports) {
      const std::uint64_t at = e->transport->next_timer_ns();
      if (at != 0 && (soonest == 0 || at < soonest)) soonest = at;
    }
    return soonest;
  }

  std::size_t drain(std::span<Event> out) noexcept {
    const std::size_t left = pending.size() - delivered;
    const std::size_t n = std::min(left, out.size());
    for (std::size_t i = 0; i < n; ++i) out[i] = pending[delivered + i];
    delivered += n;
    if (delivered == pending.size()) {
      pending.clear();
      delivered = 0;
    }
    return n;
  }

  [[nodiscard]] Status check_thread() const {
    // Off by default because it costs a comparison on the transfer path and the
    // rule it enforces is a design statement, not a runtime hazard the engine
    // recovers from.
    if (!debug() || owner == std::this_thread::get_id()) return OkStatus();
    return LSE_ERROR(kInternal,
                     "this reactor is driven by another thread; every member "
                     "but wake() is owner-thread only");
  }

  Result<ITransport*> transport_for(const Endpoint& ep) {
    for (const auto& e : transports) {
      if (e->scheme == ep.scheme()) return e->transport.get();
    }
    LSE_ASSIGN_OR(std::unique_ptr<ITransport> made, create_transport(ep.str()));
    // A transport that cannot serve this endpoint says why, and the caller sees
    // that sentence. There is no fallback to a different scheme here: silently
    // demoting a plan onto a slower path would produce an engine that reports
    // success. Choosing BETWEEN candidate rails is select_endpoint's job, above
    // the seam, and it carries its reason.
    if (const std::string_view why = made->declined(ep); !why.empty()) {
      return LSE_ERROR(kUnimplemented, "transport '", std::string(made->name()),
                       "' declines '", ep.str(), "': ", std::string(why));
    }
    auto entry = std::make_unique<Entry>();
    entry->scheme = std::string(ep.scheme());
    entry->transport = std::move(made);
    entry->sink = std::make_unique<Sink>(*this, transports.size());
    LSE_RETURN_IF_ERROR(entry->transport->open(ep, poller, *entry->sink));
    ITransport* raw = entry->transport.get();
    transports.push_back(std::move(entry));
    return raw;
  }
};

// --- Reactor ---------------------------------------------------------------

Reactor::Reactor(std::unique_ptr<ReactorState> state) noexcept
    : state_(std::move(state)) {}
Reactor::~Reactor() = default;
Reactor::Reactor(Reactor&&) noexcept = default;
Reactor& Reactor::operator=(Reactor&&) noexcept = default;

Result<Reactor> Reactor::create() {
  LSE_ASSIGN_OR(Poller poller, Poller::create());
  auto state = std::make_unique<ReactorState>(std::move(poller));
  state->wait_fds.push_back(state->poller.fd());
  return Reactor(std::move(state));
}

Result<Listener> Reactor::listen(const Endpoint& ep) {
  LSE_RETURN_IF_ERROR(state_->check_thread());
  LSE_ASSIGN_OR(ITransport* transport, state_->transport_for(ep));

  std::size_t index = state_->listeners.size();
  for (std::size_t i = 0; i < state_->listeners.size(); ++i) {
    if (!state_->listeners[i].live) {
      index = i;
      break;
    }
  }
  if (index == state_->listeners.size()) state_->listeners.emplace_back();
  ListenerSlot& slot = state_->listeners[index];
  ++slot.generation;
  slot.live = true;
  const std::uint64_t id = pack(index, slot.generation);

  auto made = transport->listen(ep, id);
  if (!made.ok()) {
    state_->listeners[index].live = false;
    return made.status();
  }
  ListenerSlot& filled = state_->listeners[index];
  filled.impl = made.release();
  filled.bound = filled.impl->endpoint();
  for (std::size_t i = 0; i < state_->transports.size(); ++i) {
    if (state_->transports[i]->transport.get() == transport) {
      filled.transport = i;
      break;
    }
  }

  Listener out;
  out.state_ = state_.get();
  out.id_ = id;
  return out;
}

Result<Channel> Reactor::connect(const Endpoint& ep) {
  LSE_RETURN_IF_ERROR(state_->check_thread());
  LSE_ASSIGN_OR(ITransport* transport, state_->transport_for(ep));

  std::size_t transport_index = 0;
  for (std::size_t i = 0; i < state_->transports.size(); ++i) {
    if (state_->transports[i]->transport.get() == transport) {
      transport_index = i;
      break;
    }
  }
  const std::uint64_t id = state_->open_channel(transport_index);
  auto made = transport->connect(ep, id);
  if (!made.ok()) {
    state_->close_channel(id);
    return made.status();
  }
  state_->bind_channel(id, made.release());

  Channel out;
  out.state_ = state_.get();
  out.id_ = id;
  return out;
}

Channel Reactor::channel(std::uint64_t id) noexcept {
  Channel out;
  if (state_ != nullptr && state_->channel_slot(id) != nullptr) {
    out.state_ = state_.get();
    out.id_ = id;
  }
  return out;
}

Result<std::size_t> Reactor::poll(std::span<Event> out,
                                  std::uint64_t timeout_ns) {
  LSE_RETURN_IF_ERROR(state_->check_thread());
  ReactorState& st = *state_;

  // A partial batch is handed out before any I/O runs, which is what keeps
  // every kControl payload valid for exactly one poll.
  if (st.delivered < st.pending.size()) return st.drain(out);

  st.arena.reset();
  // Retired links are reaped here, at the top of a sweep: nothing delivered
  // from the previous sweep still names one.
  for (const auto& e : st.transports) {
    LSE_RETURN_IF_ERROR(e->transport->progress());
  }
  st.expire(steady_now_ns());

  std::uint64_t wait_ns = st.pending.empty() ? timeout_ns : 0;
  if (wait_ns > 0) {
    if (const std::uint64_t at = st.next_wakeup_ns(); at != 0) {
      const std::uint64_t now = steady_now_ns();
      wait_ns = std::min(wait_ns, at > now ? at - now : 0);
    }
  }
  LSE_RETURN_IF_ERROR(st.poller.wait(wait_ns));

  // A transport with no descriptor of its own gets its sweep here; an
  // fd-driven one has already been served by the wait's callbacks.
  for (const auto& e : st.transports) {
    if (e->transport->requires_polling()) {
      LSE_RETURN_IF_ERROR(e->transport->progress());
    }
  }
  st.expire(steady_now_ns());
  return st.drain(out);
}

Result<Region> Reactor::register_region(const Channel& on,
                                        const RegionRequest& req) {
  LSE_RETURN_IF_ERROR(state_->check_thread());
  ChannelSlot* c = state_->channel_slot(on.id());
  if (c == nullptr) {
    return LSE_ERROR(kNotFound, "no such channel to register a region on");
  }
  return state_->transports[c->transport]->transport->register_region(req);
}

Status Reactor::deregister_region(const Channel& on, Region r) {
  LSE_RETURN_IF_ERROR(state_->check_thread());
  ChannelSlot* c = state_->channel_slot(on.id());
  if (c == nullptr) {
    return LSE_ERROR(kNotFound, "no such channel to deregister a region on");
  }
  return state_->transports[c->transport]->transport->deregister_region(r);
}

std::span<const int> Reactor::wait_fds() const noexcept {
  return state_->wait_fds;
}

std::uint64_t Reactor::next_deadline_ns() const noexcept {
  return state_->next_wakeup_ns();
}

bool Reactor::requires_polling() const noexcept {
  for (const auto& e : state_->transports) {
    if (e->transport->requires_polling()) return true;
  }
  return false;
}

void Reactor::wake() noexcept { state_->poller.wake(); }

const Status& Reactor::last_error(std::uint64_t channel) const noexcept {
  if (channel == 0) return ok_status();
  const std::size_t i = index_of(channel);
  if (i >= state_->channels.size()) return ok_status();
  const ChannelSlot& c = state_->channels[i];
  if (c.generation != generation_of(channel)) return ok_status();
  if (c.live && c.link != nullptr) return c.link->last_error();
  return c.error;
}

std::size_t Reactor::open_channels() const noexcept {
  std::size_t n = 0;
  for (const ChannelSlot& c : state_->channels) {
    if (c.live) ++n;
  }
  return n;
}

std::size_t Reactor::open_listeners() const noexcept {
  std::size_t n = 0;
  for (const ListenerSlot& l : state_->listeners) {
    if (l.live) ++n;
  }
  return n;
}

// --- Channel ---------------------------------------------------------------

const Capabilities& Channel::capabilities() const noexcept {
  if (state_ == nullptr) return nothing();
  const ChannelSlot* c = state_->channel_slot(id_);
  return c != nullptr ? c->caps : nothing();
}

const Endpoint& Channel::peer() const noexcept {
  if (state_ == nullptr) return nowhere();
  const ChannelSlot* c = state_->channel_slot(id_);
  return c != nullptr ? c->peer : nowhere();
}

namespace {

Result<Ticket> post_one(ReactorState* state, std::uint64_t id, Lane lane,
                        Direction dir, const Transfer& t) {
  if (state == nullptr) {
    return LSE_ERROR(kNotFound, "this channel was never opened");
  }
  LSE_RETURN_IF_ERROR(state->check_thread());
  ChannelSlot* c = state->channel_slot(id);
  if (c == nullptr || c->link == nullptr) {
    return LSE_ERROR(kNotFound, "this channel is closed");
  }
  LSE_ASSIGN_OR(const std::uint64_t op, state->alloc_op(*c, lane));

  ILink* link = c->link;
  const Status posted = link->post(lane, dir, t, op);
  if (!posted.ok()) {
    state->free_op(id, op);
    return posted;
  }
  const std::uint64_t at =
      t.deadline_ns != 0 ? t.deadline_ns : std::uint64_t{0};
  state->arm_deadline(id, op, at);
  return Ticket{id, op};
}

}  // namespace

Result<Ticket> Channel::post_control(std::span<const std::byte> message,
                                     std::uint32_t tag) {
  Transfer t;
  t.region = host_region(const_cast<std::byte*>(message.data()), message.size());
  t.bytes = message.size();
  t.tag = tag;
  return post_one(state_, id_, Lane::kControl, Direction::kSend, t);
}

Result<Ticket> Channel::post_send(const Transfer& t) {
  return post_one(state_, id_, Lane::kData, Direction::kSend, t);
}

Result<Ticket> Channel::post_recv(const Transfer& t) {
  return post_one(state_, id_, Lane::kData, Direction::kRecv, t);
}

Status Channel::reject(std::uint32_t tag) {
  if (state_ == nullptr) {
    return LSE_ERROR(kNotFound, "this channel was never opened");
  }
  ChannelSlot* c = state_->channel_slot(id_);
  if (c == nullptr || c->link == nullptr) {
    return LSE_ERROR(kNotFound, "this channel is closed");
  }
  return c->link->reject(tag);
}

std::size_t Channel::credit(Lane lane) const noexcept {
  if (state_ == nullptr) return 0;
  const ChannelSlot* c = state_->channel_slot(id_);
  if (c == nullptr || c->link == nullptr) return 0;
  if (lane == Lane::kControl) return c->link->credit(Lane::kControl);
  // On the data lane the unit is operations, not bytes: nothing is copied, so
  // the bound is how many transfers may be outstanding at once.
  const std::size_t half = c->ops.size() / 2;
  std::size_t free_slots = 0;
  for (std::size_t i = half; i < c->ops.size(); ++i) {
    if (!c->ops[i].live) ++free_slots;
  }
  return free_slots;
}

bool Channel::done(Ticket t) const noexcept {
  if (state_ == nullptr || t.channel != id_ || t.op == 0) return false;
  const ChannelSlot* c = state_->channel_slot(id_);
  // A channel completes every ticket it accepted before it emits kClosed, so
  // once the channel is gone nothing it took is still outstanding. Answering
  // false here would leave a caller that waits on done() waiting for ever for
  // a completion it has already been handed.
  if (c == nullptr) return true;
  return state_->op_known(*c, t.op) && !state_->op_live(*c, t.op);
}

Status Channel::cancel(Ticket t) {
  if (state_ == nullptr || t.channel != id_) {
    return LSE_ERROR(kNotFound, "that ticket does not name this channel");
  }
  ChannelSlot* c = state_->channel_slot(id_);
  if (c == nullptr || c->link == nullptr) {
    return LSE_ERROR(kNotFound, "this channel is closed");
  }
  if (!state_->op_live(*c, t.op)) {
    return LSE_ERROR(kNotFound,
                     "that ticket has already completed or names a slot that "
                     "has since been reused");
  }
  return c->link->cancel(t.op);
}

Status Channel::close() {
  if (state_ == nullptr) return OkStatus();
  ChannelSlot* c = state_->channel_slot(id_);
  if (c == nullptr || c->link == nullptr) return OkStatus();
  return c->link->close();
}

void Channel::abort() noexcept {
  if (state_ == nullptr) return;
  ChannelSlot* c = state_->channel_slot(id_);
  if (c == nullptr || c->link == nullptr) return;
  c->link->abort();
}

// --- Listener --------------------------------------------------------------

const Endpoint& Listener::endpoint() const noexcept {
  if (state_ == nullptr) return nowhere();
  const ListenerSlot* l = state_->listener_slot(id_);
  return l != nullptr ? l->bound : nowhere();
}

Status Listener::close() {
  if (state_ == nullptr) return OkStatus();
  ListenerSlot* l = state_->listener_slot(id_);
  if (l == nullptr) return OkStatus();
  const Status s = l->impl != nullptr ? l->impl->close() : OkStatus();
  l->live = false;
  l->impl = nullptr;
  Event ev;
  ev.kind = EventKind::kClosed;
  ev.listener = id_;
  state_->pending.push_back(ev);
  return s;
}

}  // namespace lse::comm
