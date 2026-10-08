#include "lse/ops/rope.hpp"

#include <cmath>
#include <cstring>

#include <algorithm>
#include <thread>
#include <vector>

// host_bytes: Array has no host-write path, and the tables are built on the
// host from cos/sin rather than by any graph op.
#include "lse/core/file_read.hpp"
#include "lse/graph/interpreter.hpp"
#include "lse/graph/ops.hpp"

namespace lse::ops {

namespace {

// Fills rows [first, last) of the pair-interleaved tables; `cos` and `sin`
// hold row `base` at their first element.
void fill_rope_rows(float* cos, float* sin, std::int32_t rope_dim,
                    const std::vector<double>& freq, std::int32_t base,
                    std::int32_t first, std::int32_t last) {
  const std::int32_t half = rope_dim / 2;
  for (std::int32_t p = first; p < last; ++p) {
    const std::size_t at = static_cast<std::size_t>(p - base) * static_cast<std::size_t>(rope_dim);
    float* c_row = cos + at;
    float* s_row = sin + at;
    for (std::int32_t i = 0; i < half; ++i) {
      const double angle = static_cast<double>(p) * freq[static_cast<std::size_t>(i)];
      const auto c = static_cast<float>(std::cos(angle));
      const auto s = static_cast<float>(std::sin(angle));
      // Interleaved so channel pairs (2i, 2i+1) share an angle.
      c_row[2 * i] = c;
      c_row[2 * i + 1] = c;
      s_row[2 * i] = s;
      s_row[2 * i + 1] = s;
    }
  }
}

// Rows [first, last) of the tables, computed across the host's cores, into
// `cos` and `sin` starting at their first element. A draft model sized for a
// 262K context has 2 x 134 MB of them, which one thread took over a second to
// produce element by element.
void fill_rope(float* cos, float* sin, std::int32_t rope_dim,
               std::int32_t first, std::int32_t last, float theta) {
  const std::int32_t half = rope_dim / 2;
  std::vector<double> freq(static_cast<std::size_t>(half));
  for (std::int32_t i = 0; i < half; ++i) {
    freq[static_cast<std::size_t>(i)] =
        1.0 / std::pow(static_cast<double>(theta),
                       static_cast<double>(i) / static_cast<double>(half));
  }
  const std::int32_t rows = last - first;
  constexpr std::int32_t kRowsPerThread = 4096;
  const std::int32_t hw = static_cast<std::int32_t>(
      std::max(1u, std::min(16u, std::thread::hardware_concurrency())));
  const std::int32_t threads =
      std::max(1, std::min(hw, (rows + kRowsPerThread - 1) / kRowsPerThread));
  if (threads == 1) {
    fill_rope_rows(cos, sin, rope_dim, freq, first, first, last);
    return;
  }
  std::vector<std::thread> pool;
  pool.reserve(static_cast<std::size_t>(threads));
  const std::int32_t step = (rows + threads - 1) / threads;
  for (std::int32_t t = 0; t < threads; ++t) {
    const std::int32_t a = first + t * step;
    const std::int32_t b = std::min(last, a + step);
    if (a >= b) break;
    pool.emplace_back(fill_rope_rows, cos, sin, rope_dim, std::cref(freq), first, a, b);
  }
  for (std::thread& t : pool) t.join();
}

// The pair of tables as device-resident constants: the buffers come from the
// device the current placement names (as weight loading picks it) and the
// values are computed a block of rows at a time into scratch memory and
// uploaded, so no host copy of the tables is ever whole -- 268 MB of them for
// a 262K-context draft model. Without a scheduler there is no device and the
// tables are host values, as every value is then.
Status place_rope(RopeTables& t, float theta) {
  const Shape shape{t.max_seq, t.dim};
  t.cos = Array::zeros(shape, DType::kF32);
  t.sin = Array::zeros(shape, DType::kF32);
  graph::Node& cn = *t.cos.node();
  graph::Node& sn = *t.sin.node();
  const std::size_t row_floats = static_cast<std::size_t>(t.dim);
  const std::size_t bytes = static_cast<std::size_t>(t.max_seq) * row_floats * sizeof(float);
  graph::Scheduler* sched = graph::default_scheduler();
  if (sched == nullptr) {
    fill_rope(static_cast<float*>(graph::interpreter::host_bytes(cn)),
              static_cast<float*>(graph::interpreter::host_bytes(sn)), t.dim, 0,
              t.max_seq, theta);
    for (graph::Node* n : {&cn, &sn}) {
      n->materialized = true;
      n->host_dirty = true;
      n->device_dirty = false;
    }
    return OkStatus();
  }
  backend::IDeviceSet& set = sched->devices();
  const std::size_t member = graph::preferred_member();
  backend::IBackend& be =
      member < set.size() ? set.device(member) : sched->backend();
  const backend::Stream at =
      member < set.size()
          ? set.stream_for(member).value_or(backend::kDefaultStream)
          : backend::kDefaultStream;
  LSE_ASSIGN_OR(cn.buffer, be.allocate(bytes, backend::MemoryClass::kDevice, at));
  LSE_ASSIGN_OR(sn.buffer, be.allocate(bytes, backend::MemoryClass::kDevice, at));
  constexpr std::int32_t kBlockRows = 32768;
  const std::int32_t block = std::min(kBlockRows, t.max_seq);
  const std::size_t block_bytes = static_cast<std::size_t>(block) * row_floats * sizeof(float);
  LSE_ASSIGN_OR(HostScratch cos, HostScratch::allocate(block_bytes, "RoPE cos rows"));
  LSE_ASSIGN_OR(HostScratch sin, HostScratch::allocate(block_bytes, "RoPE sin rows"));
  for (std::int32_t first = 0; first < t.max_seq; first += block) {
    const std::int32_t last = std::min(t.max_seq, first + block);
    fill_rope(reinterpret_cast<float*>(cos.data()), reinterpret_cast<float*>(sin.data()),
              t.dim, first, last, theta);
    const std::size_t offset = static_cast<std::size_t>(first) * row_floats * sizeof(float);
    const std::size_t n = static_cast<std::size_t>(last - first) * row_floats * sizeof(float);
    LSE_RETURN_IF_ERROR(be.upload(cos.data(), cn.buffer, n, offset));
    LSE_RETURN_IF_ERROR(be.upload(sin.data(), sn.buffer, n, offset));
  }
  for (graph::Node* n : {&cn, &sn}) {
    n->materialized = true;
    n->host_dirty = false;
    n->device_dirty = n->buffer.ptr == nullptr;
  }
  return OkStatus();
}

}  // namespace

Result<RopeTables> build_rope(std::int32_t rope_dim, std::int32_t max_seq,
                              float theta) {
  if (rope_dim <= 0 || rope_dim % 2 != 0) {
    return LSE_ERROR(kInvalidArgument, "rope_dim must be positive and even");
  }
  RopeTables t;
  t.dim = rope_dim;
  t.max_seq = max_seq;
  if (max_seq <= 0) {
    // Nothing to compute or place.
    t.cos = Array::zeros(Shape{max_seq, rope_dim}, DType::kF32);
    t.sin = Array::zeros(Shape{max_seq, rope_dim}, DType::kF32);
    return t;
  }
  LSE_RETURN_IF_ERROR(place_rope(t, theta));
  return t;
}

Result<Array> apply_rope(const Array& x, const RopeTables& tables,
                         std::int32_t offset) {
  const Shape& s = x.shape();
  const std::int64_t head_dim = s.dim(s.rank() - 1);
  if (tables.dim > head_dim) {
    return LSE_ERROR(kInvalidArgument, "rope dim exceeds head_dim");
  }
  if (tables.dim == head_dim) {
    return graph::rope(x, tables.cos, tables.sin, offset);
  }
  // Partial rotation: rotate the leading channels, pass the rest through, in
  // one launch.
  return graph::rope(x, tables.cos, tables.sin, offset,
                     static_cast<int>(tables.dim));
}

Result<Array> apply_rope(const Array& x, const RopeTables& tables,
                         const Array& offset) {
  const Shape& s = x.shape();
  const std::int64_t head_dim = s.dim(s.rank() - 1);
  if (tables.dim > head_dim) {
    return LSE_ERROR(kInvalidArgument, "rope dim exceeds head_dim");
  }
  if (tables.dim == head_dim) {
    return graph::rope(x, tables.cos, tables.sin, offset);
  }
  return graph::rope(x, tables.cos, tables.sin, offset,
                     static_cast<int>(tables.dim));
}

Result<Array> apply_rope_rows(const Array& x, const RopeTables& tables,
                              const Array& positions) {
  const Shape& s = x.shape();
  const std::int64_t head_dim = s.dim(s.rank() - 1);
  if (tables.dim > head_dim) {
    return LSE_ERROR(kInvalidArgument, "rope dim exceeds head_dim");
  }
  return graph::rope_rows(x, tables.cos, tables.sin, positions,
                          tables.dim == head_dim ? 0 : static_cast<int>(tables.dim));
}

}  // namespace lse::ops
