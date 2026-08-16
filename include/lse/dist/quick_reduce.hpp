// Two-shot all-reduce with a compressed wire payload — the QuickReduce shape.
//
// Split the payload into world_size segments; rank i owns segment i. Every
// rank encodes its contribution to segment j and sends it to rank j (shot 1);
// rank j decodes every contribution, accumulates them, encodes the result and
// sends it to everyone (shot 2); every rank decodes its copy.
//
// WHERE THE ERROR ENTERS, AND WHY THIS SHAPE BOUNDS IT
// Each element crosses the codec exactly twice — once inbound to its owner,
// once outbound from it — for ANY world size. A ring all-reduce would put
// 2*(world_size-1) encode/decode pairs on the same element, so its error grows
// with the world size and its quantization noise is re-quantized at every hop.
// That is the whole reason QuickReduce is two-shot rather than ring, and it is
// why a compressed collective must never be synthesized over the ring
// algorithms in collectives.hpp.
//
// The reduction itself accumulates in f32, deliberately unlike QuickReduce,
// which accumulates the decoded contributions in fp16 (v_pk_add_f16). The wire
// is where the bandwidth is saved; the accumulator is not, and at world_size 8
// an fp16 accumulator costs more than the codec does.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "lse/dist/codec.hpp"
#include "lse/dist/collectives.hpp"
#include "lse/dist/transport.hpp"

namespace lse::dist {

// Every rank must agree on the segment split and on the wire length, so the
// payload has to divide evenly into world_size whole codec blocks. The caller
// pads; this layer will not silently reshape a collective's payload.
[[nodiscard]] inline bool quick_reduce_shape_ok(std::size_t elems,
                                                Rank world) noexcept {
  const auto w = static_cast<std::size_t>(world);
  return w > 0 && elems % (w * kCodecBlockElems) == 0;
}

template <class T>
Status two_shot_all_reduce(T& t, CodecEngine& engine, DType wire,
                           CommBuffer& buf, ReduceOp op) {
  const Rank world = t.world_size();
  if (world <= 1) return OkStatus();
  LSE_RETURN_IF_ERROR(detail::require_host(buf, "two_shot_all_reduce"));
  if (!engine.supports(wire)) {
    return LSE_ERROR(kUnimplemented, "wire format ",
                     std::string(to_string(wire)), " declined on engine '",
                     std::string(engine.name()), "': ",
                     std::string(engine.declined(wire)));
  }
  // Shot 1 exchanges with the rank `k` away, not with a neighbour, so a
  // transport that cannot buffer a send has no deadlock-free ordering here.
  if (!t.capabilities().asynchronous) {
    return LSE_ERROR(kUnimplemented,
                     "two-shot all-reduce needs a transport that buffers "
                     "sends; this one declares synchronous send/recv");
  }
  const std::size_t n = detail::elem_count(buf);
  if (!quick_reduce_shape_ok(n, world)) {
    return LSE_ERROR(kInvalidArgument, "two-shot all-reduce needs ",
                     std::to_string(n), " elements to divide into ",
                     std::to_string(world), " whole ",
                     std::to_string(kCodecBlockElems), "-element blocks");
  }

  const auto w = static_cast<std::size_t>(world);
  const std::size_t seg = n / w;
  const std::size_t wire_bytes = codec_wire_bytes(wire, seg);
  const auto me = static_cast<std::size_t>(t.rank());

  std::vector<std::uint8_t> out_bytes(wire_bytes);
  std::vector<std::uint8_t> in_bytes(wire_bytes);
  std::vector<float> staged(seg);
  std::vector<float> acc(seg);

  CommBuffer out_w;
  out_w.host = out_bytes.data();
  out_w.bytes = wire_bytes;
  out_w.dtype = DType::kU8;
  CommBuffer in_w = out_w;
  in_w.host = in_bytes.data();
  CommBuffer staged_b;
  staged_b.host = staged.data();
  staged_b.bytes = seg * sizeof(float);

  const auto segment_buf = [&](std::size_t s) {
    CommBuffer b = buf;
    b.offset = buf.offset + s * seg * sizeof(float);
    b.bytes = seg * sizeof(float);
    return b;
  };

  // Shot 1. The rank's own segment goes through the codec too, so every
  // element of the result has seen exactly the same two codec hops no matter
  // which rank owned it — an asymmetry here would make the result depend on
  // rank, which is the one thing a collective may not do.
  LSE_RETURN_IF_ERROR(engine.encode(wire, segment_buf(me), out_w, seg));
  LSE_RETURN_IF_ERROR(engine.decode(wire, out_w, staged_b, seg));
  std::copy(staged.begin(), staged.end(), acc.begin());

  for (Rank k = 1; k < world; ++k) {
    const auto dst =
        static_cast<std::size_t>((static_cast<Rank>(me) + k) % world);
    const auto src =
        static_cast<std::size_t>((static_cast<Rank>(me) - k + world) % world);
    LSE_RETURN_IF_ERROR(engine.encode(wire, segment_buf(dst), out_w, seg));
    LSE_RETURN_IF_ERROR(detail::exchange(t, out_w, static_cast<Rank>(dst), in_w,
                                         static_cast<Rank>(src), 0x20));
    LSE_RETURN_IF_ERROR(engine.decode(wire, in_w, staged_b, seg));
    detail::reduce_into(acc.data(), staged.data(), seg, op);
  }
  if (op == ReduceOp::kAvg) {
    for (float& v : acc) v /= static_cast<float>(world);
  }

  // Shot 2: the owner's reduced segment goes back out to everyone.
  CommBuffer acc_b;
  acc_b.host = acc.data();
  acc_b.bytes = seg * sizeof(float);
  LSE_RETURN_IF_ERROR(engine.encode(wire, acc_b, out_w, seg));
  CommBuffer mine = segment_buf(me);
  LSE_RETURN_IF_ERROR(engine.decode(wire, out_w, mine, seg));

  for (Rank k = 1; k < world; ++k) {
    const auto dst =
        static_cast<std::size_t>((static_cast<Rank>(me) + k) % world);
    const auto src =
        static_cast<std::size_t>((static_cast<Rank>(me) - k + world) % world);
    LSE_RETURN_IF_ERROR(detail::exchange(t, out_w, static_cast<Rank>(dst), in_w,
                                         static_cast<Rank>(src), 0x21));
    CommBuffer theirs = segment_buf(src);
    LSE_RETURN_IF_ERROR(engine.decode(wire, in_w, theirs, seg));
  }
  return OkStatus();
}

}  // namespace lse::dist
