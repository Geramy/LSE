// Small point-to-point helpers the probe shares.
//
// Not a transport feature and not a collective: the probe needs to hand a few
// bytes to a peer and get a few back, deadlock-free, without pulling in the
// reduction machinery that `collectives.hpp` is about.
#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "lse/core/status.hpp"
#include "lse/dist/transport.hpp"

namespace lse::probe::wire {

inline dist::CommBuffer bytes_of(void* p, std::size_t n) {
  dist::CommBuffer b;
  b.host = p;
  b.bytes = n;
  b.dtype = DType::kU8;
  return b;
}

inline Status send_bytes(dist::ITransport& t, const dist::CommBuffer& b,
                         dist::Rank dst, int tag) {
  auto s = t.send(b, dst, tag);
  if (!s.ok()) return s.status();
  return t.wait(*s);
}

inline Status recv_bytes(dist::ITransport& t, dist::CommBuffer& b,
                         dist::Rank src, int tag) {
  auto r = t.recv(b, src, tag);
  if (!r.ok()) return r.status();
  return t.wait(*r);
}

// Send-first is only safe when the transport buffers sends; without that
// guarantee both peers blocking in send at once is a deadlock, and rank parity
// is what separates them. Same rule the synthesized collectives use.
inline Status exchange_bytes(dist::ITransport& t, const dist::CommBuffer& out,
                             dist::Rank dst, dist::CommBuffer& in,
                             dist::Rank src, int tag) {
  if (t.capabilities().asynchronous || t.rank() % 2 == 0) {
    LSE_RETURN_IF_ERROR(send_bytes(t, out, dst, tag));
    return recv_bytes(t, in, src, tag);
  }
  LSE_RETURN_IF_ERROR(recv_bytes(t, in, src, tag));
  return send_bytes(t, out, dst, tag);
}

// Every rank ends up with every rank's string, indexed by rank. Lengths differ,
// so each step exchanges the length first.
inline Result<std::vector<std::string>> all_gather_strings(
    dist::ITransport& t, const std::string& mine, int tag) {
  const dist::Rank n = t.world_size();
  const dist::Rank me = t.rank();
  std::vector<std::string> out(static_cast<std::size_t>(n < 1 ? 1 : n));
  if (me >= 0 && me < n) out[static_cast<std::size_t>(me)] = mine;
  for (dist::Rank step = 1; step < n; ++step) {
    const dist::Rank dst = static_cast<dist::Rank>((me + step) % n);
    const dist::Rank src = static_cast<dist::Rank>((me - step + n) % n);
    std::uint64_t len_out = mine.size();
    std::uint64_t len_in = 0;
    dist::CommBuffer lo = bytes_of(&len_out, sizeof(len_out));
    dist::CommBuffer li = bytes_of(&len_in, sizeof(len_in));
    LSE_RETURN_IF_ERROR(exchange_bytes(t, lo, dst, li, src, tag));
    std::string inbox(static_cast<std::size_t>(len_in), '\0');
    dist::CommBuffer po =
        bytes_of(const_cast<char*>(mine.data()), mine.size());
    dist::CommBuffer pi = bytes_of(inbox.data(), inbox.size());
    LSE_RETURN_IF_ERROR(exchange_bytes(t, po, dst, pi, src, tag + 1));
    out[static_cast<std::size_t>(src)] = std::move(inbox);
  }
  return out;
}

}  // namespace lse::probe::wire
