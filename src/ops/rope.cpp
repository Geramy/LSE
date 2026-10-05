#include "lse/ops/rope.hpp"

#include <cmath>
#include <cstring>

#include <algorithm>
#include <thread>
#include <vector>

// host_bytes: Array has no host-write path, and the tables are built on the
// host from cos/sin rather than by any graph op.
#include "lse/graph/interpreter.hpp"
#include "lse/graph/ops.hpp"

namespace lse::ops {

namespace {

// Fills rows [first, last) of the pair-interleaved tables.
void fill_rope_rows(float* cos, float* sin, std::int32_t rope_dim,
                    const std::vector<double>& freq, std::int32_t first,
                    std::int32_t last) {
  const std::int32_t half = rope_dim / 2;
  for (std::int32_t p = first; p < last; ++p) {
    float* c_row = cos + static_cast<std::size_t>(p) * static_cast<std::size_t>(rope_dim);
    float* s_row = sin + static_cast<std::size_t>(p) * static_cast<std::size_t>(rope_dim);
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

// The table's values, computed across the host's cores. A draft model sized
// for a 262K context has 2 x 134 MB of them, which one thread took over a
// second to produce element by element.
void fill_rope(float* cos, float* sin, std::int32_t rope_dim,
               std::int32_t max_seq, float theta) {
  const std::int32_t half = rope_dim / 2;
  std::vector<double> freq(static_cast<std::size_t>(half));
  for (std::int32_t i = 0; i < half; ++i) {
    freq[static_cast<std::size_t>(i)] =
        1.0 / std::pow(static_cast<double>(theta),
                       static_cast<double>(i) / static_cast<double>(half));
  }
  constexpr std::int32_t kRowsPerThread = 4096;
  const std::int32_t hw = static_cast<std::int32_t>(
      std::max(1u, std::min(16u, std::thread::hardware_concurrency())));
  const std::int32_t threads =
      std::max(1, std::min(hw, (max_seq + kRowsPerThread - 1) / kRowsPerThread));
  if (threads == 1) {
    fill_rope_rows(cos, sin, rope_dim, freq, 0, max_seq);
    return;
  }
  std::vector<std::thread> pool;
  pool.reserve(static_cast<std::size_t>(threads));
  const std::int32_t step = (max_seq + threads - 1) / threads;
  for (std::int32_t t = 0; t < threads; ++t) {
    const std::int32_t first = t * step;
    const std::int32_t last = std::min(max_seq, first + step);
    if (first >= last) break;
    pool.emplace_back(fill_rope_rows, cos, sin, rope_dim, std::cref(freq), first,
                      last);
  }
  for (std::thread& t : pool) t.join();
}

// A table as a device-resident constant: the buffer comes from the device the
// current placement names (as weight loading picks it), the values are
// uploaded, and no host copy is kept. Without a scheduler there is no device
// and the table lives on the host, as every value does then.
Result<Array> rope_table(Shape shape, const std::vector<float>& values) {
  Array a = Array::zeros(shape, DType::kF32);
  graph::Node& n = *a.node();
  const std::size_t bytes = values.size() * sizeof(float);
  graph::Scheduler* sched = graph::default_scheduler();
  if (sched == nullptr) {
    std::memcpy(graph::interpreter::host_bytes(n), values.data(), bytes);
    n.materialized = true;
    n.host_dirty = true;
    n.device_dirty = false;
    return a;
  }
  backend::IDeviceSet& set = sched->devices();
  const std::size_t member = graph::preferred_member();
  backend::IBackend& be =
      member < set.size() ? set.device(member) : sched->backend();
  const backend::Stream at =
      member < set.size()
          ? set.stream_for(member).value_or(backend::kDefaultStream)
          : backend::kDefaultStream;
  LSE_ASSIGN_OR(n.buffer, be.allocate(bytes, backend::MemoryClass::kDevice, at));
  LSE_RETURN_IF_ERROR(be.upload(values.data(), n.buffer, bytes, 0));
  n.materialized = true;
  n.host_dirty = false;
  n.device_dirty = n.buffer.ptr == nullptr;
  return a;
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
  const std::size_t count =
      static_cast<std::size_t>(max_seq) * static_cast<std::size_t>(rope_dim);
  std::vector<float> cos(count), sin(count);
  fill_rope(cos.data(), sin.data(), rope_dim, max_seq, theta);
  const Shape shape{max_seq, rope_dim};
  LSE_ASSIGN_OR(t.cos, rope_table(shape, cos));
  LSE_ASSIGN_OR(t.sin, rope_table(shape, sin));
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
  // Partial rotation: rotate the leading channels, pass the rest through.
  Array rotated = graph::rope(graph::slice(x, -1, 0, tables.dim), tables.cos,
                              tables.sin, offset);
  Array passthrough = graph::slice(x, -1, tables.dim, head_dim);
  return graph::concat({rotated, passthrough}, -1);
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
  Array rotated = graph::rope(graph::slice(x, -1, 0, tables.dim), tables.cos,
                              tables.sin, offset);
  Array passthrough = graph::slice(x, -1, tables.dim, head_dim);
  return graph::concat({rotated, passthrough}, -1);
}

}  // namespace lse::ops
