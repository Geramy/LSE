// Collectives synthesized over point-to-point, for transports with no native
// ones. These are the *uncompressed* algorithms; the compressed two-shot in
// quick_reduce.hpp is a peer, not a layer on top of these.
//
// LANDMINE — a compressed collective must never be synthesized over these.
// Every ring step is a hop, and a lossy codec on a ring hop compounds: a
// P-rank ring puts 2(P-1) encode/decode pairs on each element, so the error
// grows with the world size. That is exactly why QuickReduce is two-shot, and
// why `select_all_reduce` never pairs a lossy codec with CollectiveAlgo::kRing.
//
// The bodies are templates over anything exposing rank/world_size/send/recv/
// wait, so the same code serves the CRTP `Transport<Derived>` and `ITransport`.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "lse/dist/transport.hpp"

namespace lse::dist {

namespace detail {

// The synthesized algorithms reduce on the host. A device-resident payload
// needs the placement/residency seam, which does not exist yet; reducing it
// here would mean inventing an ownership rule this layer cannot honour.
[[nodiscard]] inline Status require_host(const CommBuffer& b,
                                         const char* what) {
  if (b.on_device()) {
    return LSE_ERROR(kUnimplemented, what,
                     " reduces on the host; a device-resident payload needs "
                     "the placement seam");
  }
  if (b.host == nullptr) {
    return LSE_ERROR(kInvalidArgument, what, " got a null buffer");
  }
  if (b.dtype != DType::kF32) {
    return LSE_ERROR(kUnimplemented, what, " handles f32 payloads only");
  }
  return OkStatus();
}

inline float* elems_of(const CommBuffer& b) noexcept {
  return reinterpret_cast<float*>(static_cast<std::byte*>(b.host) + b.offset);
}

inline std::size_t elem_count(const CommBuffer& b) noexcept {
  return b.bytes / sizeof(float);
}

inline void reduce_into(float* acc, const float* in, std::size_t n,
                        ReduceOp op) {
  switch (op) {
    case ReduceOp::kProd:
      for (std::size_t i = 0; i < n; ++i) acc[i] *= in[i];
      break;
    case ReduceOp::kMin:
      for (std::size_t i = 0; i < n; ++i) {
        acc[i] = acc[i] < in[i] ? acc[i] : in[i];
      }
      break;
    case ReduceOp::kMax:
      for (std::size_t i = 0; i < n; ++i) {
        acc[i] = acc[i] > in[i] ? acc[i] : in[i];
      }
      break;
    case ReduceOp::kSum:
    case ReduceOp::kAvg:
      for (std::size_t i = 0; i < n; ++i) acc[i] += in[i];
      break;
  }
}

// A view of `count` floats starting at `first` inside `b`.
inline CommBuffer window(const CommBuffer& b, std::size_t first,
                         std::size_t count) noexcept {
  CommBuffer w = b;
  w.offset = b.offset + first * sizeof(float);
  w.bytes = count * sizeof(float);
  return w;
}

// Contiguous split of n into world_size segments; the tail carries the
// remainder so every element is owned exactly once.
struct Segment {
  std::size_t first;
  std::size_t count;
};

inline Segment segment_of(std::size_t n, Rank world, Rank r) noexcept {
  const auto w = static_cast<std::size_t>(world);
  const auto i = static_cast<std::size_t>(r);
  const std::size_t base = n / w;
  const std::size_t extra = n % w;
  const std::size_t first = i * base + (i < extra ? i : extra);
  return {first, base + (i < extra ? 1u : 0u)};
}

template <class T>
Status exchange(T& t, const CommBuffer& out, Rank dst, CommBuffer& in, Rank src,
                int tag) {
  // A transport that declares `asynchronous` buffers its sends, so send-first
  // can never block and no ordering is needed. Without that guarantee both
  // peers blocking in send at once is a deadlock, so the order comes from rank
  // parity — which only separates the pair when the peers are adjacent. That
  // holds for the ring's +-1 neighbour exchange and NOT for two-shot's stride-k
  // one, which is why two_shot_all_reduce requires an asynchronous transport.
  if (t.capabilities().asynchronous || t.rank() % 2 == 0) {
    auto s = t.send(out, dst, tag);
    if (!s.ok()) return s.status();
    LSE_RETURN_IF_ERROR(t.wait(*s));
    auto r = t.recv(in, src, tag);
    if (!r.ok()) return r.status();
    return t.wait(*r);
  }
  auto r = t.recv(in, src, tag);
  if (!r.ok()) return r.status();
  LSE_RETURN_IF_ERROR(t.wait(*r));
  auto s = t.send(out, dst, tag);
  if (!s.ok()) return s.status();
  return t.wait(*s);
}

}  // namespace detail

template <class T>
Status barrier_over(T& t) {
  const Rank world = t.world_size();
  if (world <= 1) return OkStatus();
  std::uint8_t token = 0;
  CommBuffer b;
  b.host = &token;
  b.bytes = sizeof(token);
  b.dtype = DType::kU8;
  const Rank me = t.rank();
  const Rank next = static_cast<Rank>((me + 1) % world);
  const Rank prev = static_cast<Rank>((me + world - 1) % world);
  // Two laps of the ring: the first proves everyone arrived, the second that
  // everyone learned it.
  for (int lap = 0; lap < 2; ++lap) {
    CommBuffer in = b;
    LSE_RETURN_IF_ERROR(detail::exchange(t, b, next, in, prev, 0x7E));
  }
  return OkStatus();
}

template <class T>
Status ring_reduce_scatter_over(T& t, const CommBuffer& src, CommBuffer& dst,
                                ReduceOp op) {
  LSE_RETURN_IF_ERROR(detail::require_host(src, "ring_reduce_scatter"));
  LSE_RETURN_IF_ERROR(detail::require_host(dst, "ring_reduce_scatter"));
  const Rank world = t.world_size();
  const std::size_t n = detail::elem_count(src);
  const detail::Segment mine = detail::segment_of(n, world, t.rank());
  if (detail::elem_count(dst) < mine.count) {
    return LSE_ERROR(kOutOfRange, "reduce_scatter destination is too small");
  }
  // Scratch copy: the ring rewrites the payload as it accumulates, and the
  // caller's source must survive.
  std::vector<float> work(detail::elems_of(src), detail::elems_of(src) + n);
  CommBuffer buf = src;
  buf.host = work.data();
  buf.offset = 0;

  const Rank me = t.rank();
  const Rank next = static_cast<Rank>((me + 1) % world);
  const Rank prev = static_cast<Rank>((me + world - 1) % world);
  std::vector<float> inbox;
  // Rank r owns segment r when the ring stops, which fixes the offsets: the
  // last chunk a rank receives must be its own, and that is `world - 2` steps
  // after chunk `r - 1` leaves it.
  for (Rank step = 0; step + 1 < world; ++step) {
    const Rank send_seg =
        static_cast<Rank>((me - 1 - step + 2 * world) % world);
    const Rank recv_seg =
        static_cast<Rank>((me - 2 - step + 2 * world) % world);
    const detail::Segment so = detail::segment_of(n, world, send_seg);
    const detail::Segment si = detail::segment_of(n, world, recv_seg);
    inbox.assign(si.count, 0.0f);
    CommBuffer in = buf;
    in.host = inbox.data();
    in.offset = 0;
    in.bytes = si.count * sizeof(float);
    LSE_RETURN_IF_ERROR(detail::exchange(
        t, detail::window(buf, so.first, so.count), next, in, prev, 0x10));
    detail::reduce_into(work.data() + si.first, inbox.data(), si.count, op);
  }
  std::copy_n(work.data() + mine.first, mine.count, detail::elems_of(dst));
  if (op == ReduceOp::kAvg) {
    float* out = detail::elems_of(dst);
    for (std::size_t i = 0; i < mine.count; ++i) {
      out[i] /= static_cast<float>(world);
    }
  }
  return OkStatus();
}

template <class T>
Status ring_all_gather_over(T& t, const CommBuffer& src, CommBuffer& dst) {
  LSE_RETURN_IF_ERROR(detail::require_host(src, "ring_all_gather"));
  LSE_RETURN_IF_ERROR(detail::require_host(dst, "ring_all_gather"));
  const Rank world = t.world_size();
  const std::size_t n = detail::elem_count(dst);
  const detail::Segment mine = detail::segment_of(n, world, t.rank());
  if (detail::elem_count(src) < mine.count) {
    return LSE_ERROR(kOutOfRange, "all_gather source is too small");
  }
  float* out = detail::elems_of(dst);
  std::copy_n(detail::elems_of(src), mine.count, out + mine.first);

  const Rank me = t.rank();
  const Rank next = static_cast<Rank>((me + 1) % world);
  const Rank prev = static_cast<Rank>((me + world - 1) % world);
  for (Rank step = 0; step + 1 < world; ++step) {
    const Rank send_seg = static_cast<Rank>((me - step + 2 * world) % world);
    const Rank recv_seg =
        static_cast<Rank>((me - step - 1 + 2 * world) % world);
    const detail::Segment so = detail::segment_of(n, world, send_seg);
    const detail::Segment si = detail::segment_of(n, world, recv_seg);
    CommBuffer in = detail::window(dst, si.first, si.count);
    LSE_RETURN_IF_ERROR(detail::exchange(
        t, detail::window(dst, so.first, so.count), next, in, prev, 0x11));
  }
  return OkStatus();
}

template <class T>
Status ring_all_reduce_over(T& t, CommBuffer& buf, ReduceOp op) {
  if (t.world_size() <= 1) return OkStatus();
  LSE_RETURN_IF_ERROR(detail::require_host(buf, "ring_all_reduce"));
  const Rank world = t.world_size();
  const std::size_t n = detail::elem_count(buf);
  const detail::Segment mine = detail::segment_of(n, world, t.rank());
  std::vector<float> part(mine.count, 0.0f);
  CommBuffer p = buf;
  p.host = part.data();
  p.offset = 0;
  p.bytes = mine.count * sizeof(float);
  LSE_RETURN_IF_ERROR(ring_reduce_scatter_over(t, buf, p, op));
  return ring_all_gather_over(t, p, buf);
}

template <class T>
Status tree_broadcast_over(T& t, CommBuffer& buf, Rank root) {
  const Rank world = t.world_size();
  if (world <= 1) return OkStatus();
  const Rank me = t.rank();
  // Binomial tree in root-relative rank space.
  const Rank v = static_cast<Rank>((me - root + world) % world);
  if (v != 0) {
    Rank mask = 1;
    while (mask <= v) mask <<= 1;
    const Rank parent = static_cast<Rank>(((v - (mask >> 1)) + root) % world);
    auto r = t.recv(buf, parent, 0x12);
    if (!r.ok()) return r.status();
    LSE_RETURN_IF_ERROR(t.wait(*r));
  }
  for (Rank mask = 1; mask < world; mask <<= 1) {
    if (v >= mask) continue;
    const Rank child = static_cast<Rank>(v + mask);
    if (child >= world) break;
    auto s = t.send(buf, static_cast<Rank>((child + root) % world), 0x12);
    if (!s.ok()) return s.status();
    LSE_RETURN_IF_ERROR(t.wait(*s));
  }
  return OkStatus();
}

template <class T>
Status bruck_all_to_all_over(T& t, const CommBuffer& src, CommBuffer& dst) {
  LSE_RETURN_IF_ERROR(detail::require_host(src, "bruck_all_to_all"));
  LSE_RETURN_IF_ERROR(detail::require_host(dst, "bruck_all_to_all"));
  const Rank world = t.world_size();
  const std::size_t n = detail::elem_count(src);
  if (n % static_cast<std::size_t>(world) != 0) {
    return LSE_ERROR(kInvalidArgument,
                     "all_to_all needs an element count divisible by the "
                     "world size");
  }
  const std::size_t per = n / static_cast<std::size_t>(world);
  // Direct exchange: Bruck's log(P) rotation only pays off once the transport
  // charges per message, which no transport here reports yet.
  const Rank me = t.rank();
  std::copy_n(detail::elems_of(src) + static_cast<std::size_t>(me) * per, per,
              detail::elems_of(dst) + static_cast<std::size_t>(me) * per);
  for (Rank step = 1; step < world; ++step) {
    const Rank dstr = static_cast<Rank>((me + step) % world);
    const Rank srcr = static_cast<Rank>((me - step + world) % world);
    CommBuffer in =
        detail::window(dst, static_cast<std::size_t>(srcr) * per, per);
    LSE_RETURN_IF_ERROR(detail::exchange(
        t, detail::window(src, static_cast<std::size_t>(dstr) * per, per), dstr,
        in, srcr, 0x13));
  }
  return OkStatus();
}

// The declarations on Transport<Derived> live in transport.hpp; these are their
// definitions, kept here so the ring algorithms sit beside their peers.
template <typename Derived>
Status Transport<Derived>::sendrecv(const CommBuffer& send_buf, Rank dst,
                                    CommBuffer& recv_buf, Rank src, int tag) {
  return detail::exchange(*this, send_buf, dst, recv_buf, src, tag);
}
template <typename Derived>
Status Transport<Derived>::ring_all_reduce(CommBuffer& buf, ReduceOp op) {
  return ring_all_reduce_over(*this, buf, op);
}
template <typename Derived>
Status Transport<Derived>::tree_broadcast(CommBuffer& buf, Rank root) {
  return tree_broadcast_over(*this, buf, root);
}
template <typename Derived>
Status Transport<Derived>::ring_all_gather(const CommBuffer& src,
                                           CommBuffer& dst) {
  return ring_all_gather_over(*this, src, dst);
}
template <typename Derived>
Status Transport<Derived>::ring_reduce_scatter(const CommBuffer& src,
                                               CommBuffer& dst, ReduceOp op) {
  return ring_reduce_scatter_over(*this, src, dst, op);
}
template <typename Derived>
Status Transport<Derived>::bruck_all_to_all(const CommBuffer& src,
                                            CommBuffer& dst) {
  return bruck_all_to_all_over(*this, src, dst);
}

}  // namespace lse::dist
