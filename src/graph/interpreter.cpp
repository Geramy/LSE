#include "lse/graph/interpreter.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

// The step-descriptor layout the paged kernels read. These rules have to agree
// with src/backends/hrx/kernels/sdpa.cpp and rope.cpp element for element: a
// group that misses the device path runs here instead, and a host fallback that
// disagrees is a wrong answer nothing else reports.
#include "lse/kv/block.hpp"
#include "lse/quant/group_affine.hpp"

namespace lse::graph::interpreter {

namespace {

std::size_t broadcast_index(const Shape& src, const Shape& out,
                            std::size_t out_index) noexcept {
  return BroadcastMap::build(src, out).apply(out_index);
}

Status ensure_buffer(Node& n, backend::IBackend& backend) {
  if (n.buffer.valid()) return OkStatus();
  const std::size_t bytes = dtype_storage_bytes(n.dtype, n.element_count());
  if (bytes == 0) {
    return LSE_ERROR(kInvalidArgument, "cannot size a buffer for ",
                     std::string(to_string(n.dtype)), n.shape.to_string());
  }
  // Device-local: only kernels are expected to touch it. The host reaches it
  // through the mirror, and only when a node actually falls back to the host.
  auto buf = backend.allocate(bytes, backend::MemoryClass::kDevice);
  if (!buf.ok()) return buf.status();
  n.buffer = buf.release();
  return OkStatus();
}

// Splits a shape into (outer, axis_len, inner) around the reduced axis so a
// reduction is one flat triple loop regardless of rank.
struct AxisSplit {
  std::size_t outer = 1;
  std::size_t axis_len = 1;
  std::size_t inner = 1;
};

AxisSplit split_axis(const Shape& s, std::size_t axis) noexcept {
  AxisSplit out;
  for (std::size_t i = 0; i < s.rank(); ++i) {
    const auto d = static_cast<std::size_t>(s.dim(i));
    if (i < axis) out.outer *= d;
    else if (i == axis) out.axis_len = d;
    else out.inner *= d;
  }
  return out;
}

Status eval_reduction(Node& n) {
  const Node& src = *n.inputs[0];
  const auto axis = static_cast<std::size_t>(n.iattrs[0]);
  const AxisSplit sp = split_axis(src.shape, axis);

  for (std::size_t o = 0; o < sp.outer; ++o) {
    for (std::size_t i = 0; i < sp.inner; ++i) {
      const std::size_t base = o * sp.axis_len * sp.inner + i;
      float acc = (n.kind == OpKind::kMax) ? -std::numeric_limits<float>::infinity()
                                           : 0.0f;
      for (std::size_t a = 0; a < sp.axis_len; ++a) {
        const float v = load_element(src, base + a * sp.inner);
        if (n.kind == OpKind::kMax) acc = v > acc ? v : acc;
        else acc += v;
      }
      if (n.kind == OpKind::kMean) acc /= static_cast<float>(sp.axis_len);
      store_element(n, o * sp.inner + i, acc);
    }
  }
  return OkStatus();
}

Status eval_softmax(Node& n) {
  const Node& src = *n.inputs[0];
  const auto axis = static_cast<std::size_t>(n.iattrs[0]);
  const AxisSplit sp = split_axis(src.shape, axis);

  for (std::size_t o = 0; o < sp.outer; ++o) {
    for (std::size_t i = 0; i < sp.inner; ++i) {
      const std::size_t base = o * sp.axis_len * sp.inner + i;
      float m = -std::numeric_limits<float>::infinity();
      for (std::size_t a = 0; a < sp.axis_len; ++a) {
        const float v = load_element(src, base + a * sp.inner);
        m = v > m ? v : m;
      }
      float denom = 0.0f;
      for (std::size_t a = 0; a < sp.axis_len; ++a) {
        denom += std::exp(load_element(src, base + a * sp.inner) - m);
      }
      for (std::size_t a = 0; a < sp.axis_len; ++a) {
        const float v = std::exp(load_element(src, base + a * sp.inner) - m) / denom;
        store_element(n, base + a * sp.inner, v);
      }
    }
  }
  return OkStatus();
}

// Rank of e is how many others are strictly larger, then how many equals have
// a smaller index. Slot s is the unique e whose rank is s, so the output is
// descending and ties are deterministic.
Status eval_topk(Node& n) {
  const Node& src = *n.inputs[0];
  const auto axis = static_cast<std::size_t>(n.iattrs[0]);
  const auto k = static_cast<std::size_t>(n.iattrs[1]);
  const bool write_idx = n.iattrs[2] != 0;
  const float score_band = n.attrs[0] == 0.0f ? 1.0f : n.attrs[0];
  const AxisSplit sp = split_axis(src.shape, axis);
  if (k == 0 || k > sp.axis_len) {
    return LSE_ERROR(kInvalidArgument, "topk k=", std::to_string(k),
                     " along axis of ", std::to_string(sp.axis_len));
  }

  std::vector<float> vals(k);
  std::vector<float> ids(k);
  for (std::size_t o = 0; o < sp.outer; ++o) {
    for (std::size_t i = 0; i < sp.inner; ++i) {
      const std::size_t in_base = o * sp.axis_len * sp.inner + i;
      const std::size_t out_base = o * k * sp.inner + i;
      for (std::size_t slot = 0; slot < k; ++slot) {
        bool found = false;
        for (std::size_t e = 0; e < sp.axis_len; ++e) {
          const float v = load_element(src, in_base + e * sp.inner);
          std::size_t rank = 0;
          for (std::size_t j = 0; j < sp.axis_len; ++j) {
            const float u = load_element(src, in_base + j * sp.inner);
            if (u > v || (u == v && j < e)) ++rank;
          }
          if (rank != slot) continue;
          vals[slot] = v;
          ids[slot] = static_cast<float>(e);
          found = true;
          break;
        }
        if (!found) {
          return LSE_ERROR(kInternal, "topk found no element for slot ",
                           std::to_string(slot));
        }
      }
      if (!write_idx && score_band < 1.0f) {
        const float top = vals[0];
        float total = 0.0f;
        for (std::size_t s = 0; s < k; ++s) {
          if (vals[s] < (1.0f - score_band) * top) vals[s] = 0.0f;
          total += vals[s];
        }
        total += 1e-9f;
        for (std::size_t s = 0; s < k; ++s) vals[s] /= total;
      }
      for (std::size_t slot = 0; slot < k; ++slot) {
        store_element(n, out_base + slot * sp.inner,
                      write_idx ? ids[slot] : vals[slot]);
      }
    }
  }
  return OkStatus();
}

// Both argmax stages share one comparator: higher value wins, equal values
// keep the smaller index. This must match runtime::argmax and the device
// kernels exactly or greedy decode diverges between backends.
Status eval_argmax(Node& n) {
  const Node& src = *n.inputs[0];
  if (n.iattrs[0] == 0) {
    const auto d = static_cast<std::size_t>(src.shape.dim(src.shape.rank() - 1));
    const auto chunk = static_cast<std::size_t>(n.iattrs[1]);
    if (d == 0 || chunk == 0) {
      return LSE_ERROR(kInvalidArgument, "argmax over an empty axis");
    }
    const std::size_t nchunks = (d + chunk - 1) / chunk;
    const std::size_t rows = src.element_count() / d;
    for (std::size_t r = 0; r < rows; ++r) {
      for (std::size_t c = 0; c < nchunks; ++c) {
        float best_v = -std::numeric_limits<float>::infinity();
        float best_i = static_cast<float>(d);
        const std::size_t hi = std::min(d, (c + 1) * chunk);
        for (std::size_t e = c * chunk; e < hi; ++e) {
          const float v = load_element(src, r * d + e);
          const auto idx = static_cast<float>(e);
          if (v > best_v || (v == best_v && idx < best_i)) {
            best_v = v;
            best_i = idx;
          }
        }
        store_element(n, (r * nchunks + c) * 2, best_v);
        store_element(n, (r * nchunks + c) * 2 + 1, best_i);
      }
    }
    return OkStatus();
  }
  const auto nchunks =
      static_cast<std::size_t>(src.shape.dim(src.shape.rank() - 2));
  if (nchunks == 0) return LSE_ERROR(kInvalidArgument, "argmax has no partials");
  const std::size_t rows = src.element_count() / (nchunks * 2);
  for (std::size_t r = 0; r < rows; ++r) {
    float best_v = -std::numeric_limits<float>::infinity();
    float best_i = std::numeric_limits<float>::max();
    for (std::size_t c = 0; c < nchunks; ++c) {
      const float v = load_element(src, (r * nchunks + c) * 2);
      const float idx = load_element(src, (r * nchunks + c) * 2 + 1);
      if (v > best_v || (v == best_v && idx < best_i)) {
        best_v = v;
        best_i = idx;
      }
    }
    store_element(n, r, best_i);
  }
  return OkStatus();
}

Status eval_rms_norm(Node& n) {
  const Node& x = *n.inputs[0];
  const Node& w = *n.inputs[1];
  const auto dim = static_cast<std::size_t>(x.shape.dim(x.shape.rank() - 1));
  const std::size_t rows = x.element_count() / dim;
  const float eps = n.attrs[0];
  const float w_bias = n.iattrs[0] != 0 ? 1.0f : 0.0f;

  for (std::size_t r = 0; r < rows; ++r) {
    // fp64 accumulation: the reduction is what makes RMSNorm numerically
    // touchy, and the device kernels accumulate in fp32 for the same reason.
    double acc = 0.0;
    for (std::size_t c = 0; c < dim; ++c) {
      const double v = static_cast<double>(load_element(x, r * dim + c));
      acc += v * v;
    }
    const float scale =
        1.0f / std::sqrt(static_cast<float>(acc / static_cast<double>(dim)) + eps);
    for (std::size_t c = 0; c < dim; ++c) {
      store_element(n, r * dim + c, load_element(x, r * dim + c) * scale *
                                        (w_bias + load_element(w, c)));
    }
  }
  return OkStatus();
}

// log1p(exp(x)) with the standard large-x guard: exp overflows past ~88f.
Status eval_softplus(Node& n) {
  const Node& src = *n.inputs[0];
  for (std::size_t i = 0; i < n.element_count(); ++i) {
    const float v = load_element(src, i);
    store_element(n, i, v > 20.0f ? v : std::log1p(std::exp(v)));
  }
  return OkStatus();
}

Status eval_l2_norm(Node& n) {
  const Node& src = *n.inputs[0];
  const auto dim = static_cast<std::size_t>(n.shape.dim(n.shape.rank() - 1));
  const std::size_t rows = n.element_count() / dim;
  const float eps = n.attrs[0];
  for (std::size_t r = 0; r < rows; ++r) {
    double acc = 0.0;
    for (std::size_t c = 0; c < dim; ++c) {
      const double x = static_cast<double>(load_element(src, r * dim + c));
      acc += x * x;
    }
    // eps floors the norm rather than sitting under the sqrt: it is only a
    // divide-by-zero guard, and must not perturb rows whose norm is already
    // well above it. Head vectors here have norms near 0.05, where folding eps
    // into the radicand shifts the result by ~2e-4.
    const float inv =
        1.0f / std::max(std::sqrt(static_cast<float>(acc)), eps);
    for (std::size_t c = 0; c < dim; ++c) {
      store_element(n, r * dim + c, load_element(src, r * dim + c) * inv);
    }
  }
  return OkStatus();
}

// x [B,T,C], weight [C,K], bias [C]. Left-padded with zeros so out[t] sees only
// inputs <= t. An optional 4th input [B,K-1,C] is the carried tail: window
// positions before x's start read it instead of the zero pad.
Status eval_causal_conv1d(Node& n) {
  const Node& x = *n.inputs[0];
  const Node& w = *n.inputs[1];
  const Node& b = *n.inputs[2];
  const Node* tail = n.inputs.size() > 3 ? n.inputs[3].get() : nullptr;
  const std::size_t rank = x.shape.rank();
  const auto channels = static_cast<std::size_t>(x.shape.dim(rank - 1));
  const auto seq = static_cast<std::size_t>(x.shape.dim(rank - 2));
  const auto kernel = static_cast<std::size_t>(w.shape.dim(1));
  const std::size_t batch = x.element_count() / (seq * channels);

  for (std::size_t bi = 0; bi < batch; ++bi) {
    for (std::size_t t = 0; t < seq; ++t) {
      for (std::size_t c = 0; c < channels; ++c) {
        double acc = static_cast<double>(load_element(b, c));
        for (std::size_t j = 0; j < kernel; ++j) {
          // weight[c][j] pairs with the input j steps before t; positions
          // before the start are the zero pad or the tail.
          const std::size_t back = kernel - 1 - j;
          double in;
          if (t >= back) {
            in = static_cast<double>(load_element(
                x, ((bi * seq) + (t - back)) * channels + c));
          } else if (tail != nullptr) {
            // t < back <= K-1 puts t+j inside the K-1 tail columns.
            in = static_cast<double>(load_element(
                *tail, ((bi * (kernel - 1)) + (t + j)) * channels + c));
          } else {
            continue;
          }
          acc += in * static_cast<double>(load_element(w, c * kernel + j));
        }
        store_element(n, (bi * seq + t) * channels + c, static_cast<float>(acc));
      }
    }
  }
  return OkStatus();
}

// out == last tail_len columns of tail ++ x, copied.
Status eval_conv_tail(Node& n) {
  const Node& tail = *n.inputs[0];
  const Node& x = *n.inputs[1];
  const std::size_t rank = x.shape.rank();
  const auto channels = static_cast<std::size_t>(x.shape.dim(rank - 1));
  const auto seq = static_cast<std::size_t>(x.shape.dim(rank - 2));
  const auto keep = static_cast<std::size_t>(n.shape.dim(n.shape.rank() - 2));
  const std::size_t batch = n.element_count() / (keep * channels);

  for (std::size_t bi = 0; bi < batch; ++bi) {
    for (std::size_t p = 0; p < keep; ++p) {
      // Position seq + p in tail ++ x, counted so the window ends at x's end.
      const std::size_t g = seq + p;
      for (std::size_t c = 0; c < channels; ++c) {
        const float v =
            g >= keep
                ? load_element(x, ((bi * seq) + (g - keep)) * channels + c)
                : load_element(tail, ((bi * keep) + g) * channels + c);
        store_element(n, ((bi * keep) + p) * channels + c, v);
      }
    }
  }
  return OkStatus();
}

// S = S*alpha ; S += ((v - S k) * beta) k^T ; o = S q, stepped over time.
Status eval_gated_delta(Node& n) {
  const Node& q = *n.inputs[0];
  const Node& k = *n.inputs[1];
  const Node& v = *n.inputs[2];
  const Node& alpha = *n.inputs[3];
  const Node& beta = *n.inputs[4];
  const Node& s_in = *n.inputs[5];
  const bool write_state = n.iattrs[0] != 0;

  const auto batch = static_cast<std::size_t>(q.shape.dim(0));
  const auto seq = static_cast<std::size_t>(q.shape.dim(1));
  const auto heads = static_cast<std::size_t>(q.shape.dim(2));
  const auto dim = static_cast<std::size_t>(q.shape.dim(3));

  std::vector<double> state(batch * heads * dim * dim);
  for (std::size_t i = 0; i < state.size(); ++i) state[i] = static_cast<double>(load_element(s_in, i));

  std::vector<double> sk(dim);
  for (std::size_t bi = 0; bi < batch; ++bi) {
    for (std::size_t t = 0; t < seq; ++t) {
      for (std::size_t h = 0; h < heads; ++h) {
        double* S = state.data() + ((bi * heads) + h) * dim * dim;
        const std::size_t vec = (((bi * seq) + t) * heads + h) * dim;
        const std::size_t sc = ((bi * seq) + t) * heads + h;

        const double a = static_cast<double>(load_element(alpha, sc));
        for (std::size_t i = 0; i < dim * dim; ++i) S[i] *= a;

        for (std::size_t i = 0; i < dim; ++i) {
          double acc = 0.0;
          for (std::size_t j = 0; j < dim; ++j) {
            acc += S[i * dim + j] * static_cast<double>(load_element(k, vec + j));
          }
          sk[i] = acc;
        }

        const double bt = static_cast<double>(load_element(beta, sc));
        for (std::size_t i = 0; i < dim; ++i) {
          const double delta =
              (static_cast<double>(load_element(v, vec + i)) - sk[i]) * bt;
          for (std::size_t j = 0; j < dim; ++j) {
            S[i * dim + j] += delta * static_cast<double>(load_element(k, vec + j));
          }
        }

        for (std::size_t i = 0; i < dim; ++i) {
          double acc = 0.0;
          for (std::size_t j = 0; j < dim; ++j) {
            acc += S[i * dim + j] * static_cast<double>(load_element(q, vec + j));
          }
          if (!write_state) {
            store_element(n, vec + i, static_cast<float>(acc));
          }
        }
      }
    }
  }

  if (write_state) {
    for (std::size_t i = 0; i < state.size(); ++i) {
      store_element(n, i, static_cast<float>(state[i]));
    }
  }
  return OkStatus();
}

Status eval_matmul(Node& n) {
  const Node& a = *n.inputs[0];
  const Node& b = *n.inputs[1];
  const auto k = static_cast<std::size_t>(a.shape.dim(a.shape.rank() - 1));
  const auto cols = static_cast<std::size_t>(b.shape.dim(b.shape.rank() - 1));
  const std::size_t rows = a.element_count() / k;

  for (std::size_t r = 0; r < rows; ++r) {
    for (std::size_t c = 0; c < cols; ++c) {
      double acc = 0.0;
      for (std::size_t i = 0; i < k; ++i) {
        acc += static_cast<double>(load_element(a, r * k + i)) *
               static_cast<double>(load_element(b, i * cols + c));
      }
      store_element(n, r * cols + c, static_cast<float>(acc));
    }
  }
  return OkStatus();
}

Status eval_transpose(Node& n) {
  const Node& src = *n.inputs[0];
  const auto src_strides = src.shape.strides();
  const auto out_strides = n.shape.strides();
  const std::size_t rank = n.shape.rank();
  const std::size_t count = n.element_count();

  for (std::size_t i = 0; i < count; ++i) {
    std::size_t src_index = 0;
    for (std::size_t d = 0; d < rank; ++d) {
      const std::int64_t coord =
          static_cast<std::int64_t>(i) / out_strides[d] % n.shape.dim(d);
      src_index += static_cast<std::size_t>(
          coord * src_strides[static_cast<std::size_t>(n.iattrs[d])]);
    }
    store_element(n, i, load_element(src, src_index));
  }
  return OkStatus();
}

Status eval_concat(Node& n) {
  const auto axis = static_cast<std::size_t>(n.iattrs[0]);
  const AxisSplit out_sp = split_axis(n.shape, axis);
  std::size_t written = 0;
  for (const NodePtr& part : n.inputs) {
    const AxisSplit in_sp = split_axis(part->shape, axis);
    for (std::size_t o = 0; o < in_sp.outer; ++o) {
      for (std::size_t a = 0; a < in_sp.axis_len; ++a) {
        for (std::size_t i = 0; i < in_sp.inner; ++i) {
          const std::size_t src = (o * in_sp.axis_len + a) * in_sp.inner + i;
          const std::size_t dst = (o * out_sp.axis_len + written + a) * out_sp.inner + i;
          store_element(n, dst, load_element(*part, src));
        }
      }
    }
    written += in_sp.axis_len;
  }
  return OkStatus();
}

Status eval_slice(Node& n) {
  const Node& src = *n.inputs[0];
  const auto axis = static_cast<std::size_t>(n.iattrs[0]);
  const auto begin = static_cast<std::size_t>(n.iattrs[1]);
  const AxisSplit in_sp = split_axis(src.shape, axis);
  const AxisSplit out_sp = split_axis(n.shape, axis);

  for (std::size_t o = 0; o < out_sp.outer; ++o) {
    for (std::size_t a = 0; a < out_sp.axis_len; ++a) {
      for (std::size_t i = 0; i < out_sp.inner; ++i) {
        const std::size_t s = (o * in_sp.axis_len + begin + a) * in_sp.inner + i;
        const std::size_t d = (o * out_sp.axis_len + a) * out_sp.inner + i;
        store_element(n, d, load_element(src, s));
      }
    }
  }
  return OkStatus();
}

Status eval_repeat(Node& n) {
  const Node& src = *n.inputs[0];
  const auto axis = static_cast<std::size_t>(n.iattrs[0]);
  const auto count = static_cast<std::size_t>(n.iattrs[1]);
  const AxisSplit in_sp = split_axis(src.shape, axis);
  const AxisSplit out_sp = split_axis(n.shape, axis);

  for (std::size_t o = 0; o < in_sp.outer; ++o) {
    for (std::size_t a = 0; a < in_sp.axis_len; ++a) {
      for (std::size_t r = 0; r < count; ++r) {
        for (std::size_t i = 0; i < in_sp.inner; ++i) {
          const std::size_t s = (o * in_sp.axis_len + a) * in_sp.inner + i;
          const std::size_t d = (o * out_sp.axis_len + a * count + r) * out_sp.inner + i;
          store_element(n, d, load_element(src, s));
        }
      }
    }
  }
  return OkStatus();
}

Status eval_linear_indexed(Node& n) {
  const Node& x = *n.inputs[0];
  const Node& w = *n.inputs[1];
  const Node& idx = *n.inputs[2];
  const auto experts = static_cast<std::size_t>(w.shape.dim(0));
  const auto out_dim = static_cast<std::size_t>(w.shape.dim(1));
  const auto in_dim = static_cast<std::size_t>(w.shape.dim(2));
  const auto keep = static_cast<std::size_t>(idx.shape.dim(idx.shape.rank() - 1));
  const auto slot = static_cast<std::size_t>(n.iattrs[0]);
  const std::size_t rows = x.element_count() / in_dim;
  if (keep == 0 || slot >= keep) {
    return LSE_ERROR(kInvalidArgument, "linear_indexed slot out of range");
  }

  for (std::size_t r = 0; r < rows; ++r) {
    const auto e = static_cast<std::size_t>(load_element(idx, r * keep + slot));
    if (e >= experts) {
      return LSE_ERROR(kOutOfRange, "expert ", std::to_string(e),
                       " is outside ", std::to_string(experts));
    }
    for (std::size_t o = 0; o < out_dim; ++o) {
      double acc = 0.0;
      const std::size_t wrow = (e * out_dim + o) * in_dim;
      for (std::size_t i = 0; i < in_dim; ++i) {
        acc += static_cast<double>(load_element(x, r * in_dim + i)) *
               static_cast<double>(load_element(w, wrow + i));
      }
      store_element(n, r * out_dim + o, static_cast<float>(acc));
    }
  }
  return OkStatus();
}

Status eval_linear(Node& n) {
  const Node& x = *n.inputs[0];
  const Node& w = *n.inputs[1];
  const auto in_dim = static_cast<std::size_t>(w.shape.dim(1));
  const auto out_dim = static_cast<std::size_t>(w.shape.dim(0));
  const std::size_t rows = x.element_count() / in_dim;

  for (std::size_t r = 0; r < rows; ++r) {
    for (std::size_t o = 0; o < out_dim; ++o) {
      double acc = 0.0;
      for (std::size_t i = 0; i < in_dim; ++i) {
        acc += static_cast<double>(load_element(x, r * in_dim + i)) *
               static_cast<double>(load_element(w, o * in_dim + i));
      }
      store_element(n, r * out_dim + o, static_cast<float>(acc));
    }
  }
  return OkStatus();
}

// The three planes and the bit width have to describe one and the same row,
// because the loops below turn `i / group_size` into a scale index without
// bounds-checking it. A row that is not a whole number of groups reads the
// *next* row's scales for its tail, and the last row reads past the plane
// entirely — plausible numbers, no complaint. These are the two conditions the
// device kernels' dims_of() already require, so enforcing them here only makes
// the host arm accept exactly what the device one does.
Status check_quant_planes(const quant::GroupAffine& spec, const char* op,
                          const Shape& packed, const Shape& scales,
                          const Shape& biases, std::size_t in_dim) {
  // Rank 2 is one matrix, rank 3 a stack of them. Only the last axis differs
  // between the planes — lanes on one side, groups on the other — so every
  // axis before it has to agree whichever rank this is.
  bool shaped = scales == biases && packed.rank() == scales.rank() &&
                (packed.rank() == 2 || packed.rank() == 3);
  for (std::size_t i = 0; shaped && i + 1 < packed.rank(); ++i) {
    if (packed.dim(i) != scales.dim(i)) shaped = false;
  }
  if (!shaped) {
    return LSE_ERROR(kInvalidArgument, op, " has a ", packed.to_string(),
                     " plane with scales ", scales.to_string(), " and biases ",
                     biases.to_string(),
                     "; the three must share a rank and agree on every axis "
                     "but the last");
  }
  const auto lanes = static_cast<std::size_t>(packed.dim(packed.rank() - 1));
  const auto groups = static_cast<std::size_t>(scales.dim(scales.rank() - 1));
  if (lanes * 32 != in_dim * static_cast<std::size_t>(spec.bits)) {
    return LSE_ERROR(kInvalidArgument, op, " packed plane of ",
                     std::to_string(lanes), " lanes does not hold a whole "
                     "number of ", std::to_string(spec.bits), "-bit codes");
  }
  const std::size_t covered =
      groups * static_cast<std::size_t>(spec.group_size);
  if (covered != in_dim) {
    return LSE_ERROR(kInvalidArgument, op, " scale plane has ",
                     std::to_string(groups), " groups of ",
                     std::to_string(spec.group_size), ", covering ",
                     std::to_string(covered), " of the row's ",
                     std::to_string(in_dim), " weights");
  }
  return OkStatus();
}

// The reference the emitted quant_linear kernel is diffed against. A U32 lane
// holds several codes, so the packed plane is read at its own width rather
// than through load_element, which speaks in floats.
Status eval_quant_matmul(Node& n) {
  const Node& x = *n.inputs[0];
  const Node& packed = *n.inputs[1];
  const Node& scales = *n.inputs[2];
  const Node& biases = *n.inputs[3];
  LSE_ASSIGN_OR(const quant::GroupAffine spec,
                quant::GroupAffine::make(n.iattrs[0], n.iattrs[1]));

  const auto out_dim = static_cast<std::size_t>(packed.shape.dim(0));
  const auto lanes = static_cast<std::size_t>(packed.shape.dim(1));
  const auto groups = static_cast<std::size_t>(scales.shape.dim(1));
  const std::size_t in_dim = lanes * 32 / static_cast<std::size_t>(spec.bits);
  LSE_RETURN_IF_ERROR(check_quant_planes(spec, "quant_linear", packed.shape,
                                         scales.shape, biases.shape, in_dim));
  // The row count below comes from the plane's width, but the output was sized
  // from x's leading axes. Let them disagree and a wider x writes more rows
  // than the output holds — a heap overflow, not a wrong number. The device
  // kernel contracts over x's own last axis and declines the mismatch; match
  // it rather than reinterpreting x's shape.
  const auto x_width =
      x.shape.rank() == 0
          ? 0
          : static_cast<std::size_t>(x.shape.dim(x.shape.rank() - 1));
  if (x_width != in_dim) {
    return LSE_ERROR(kInvalidArgument, "quant_linear contracts a row of ",
                     std::to_string(x_width), " against a weight of ",
                     std::to_string(in_dim), " per output");
  }
  const auto* w = static_cast<const std::uint32_t*>(host_bytes(packed));
  if (w == nullptr) {
    return LSE_ERROR(kInternal, "quant_linear has no host copy of its packed "
                                "plane to read");
  }
  const std::size_t rows = x.element_count() / in_dim;

  for (std::size_t r = 0; r < rows; ++r) {
    for (std::size_t o = 0; o < out_dim; ++o) {
      const std::uint32_t* row = w + o * lanes;
      double acc = 0.0;
      for (std::size_t i = 0; i < in_dim; ++i) {
        const std::size_t g = i / static_cast<std::size_t>(spec.group_size);
        const float wv =
            static_cast<float>(spec.code_at(row, i)) *
                load_element(scales, o * groups + g) +
            load_element(biases, o * groups + g);
        acc += static_cast<double>(load_element(x, r * in_dim + i)) *
               static_cast<double>(wv);
      }
      store_element(n, r * out_dim + o, static_cast<float>(acc));
    }
  }
  return OkStatus();
}

// The reference the emitted quant_linear_indexed kernel is diffed against: the
// loop above with the expert chosen per row. Nothing is gathered — the row of
// the stack is addressed in place, exactly as the device body addresses it, so
// a disagreement between the two is a decode bug and not a layout one.
//
// This is the only oracle a mis-decoded router has. Reading the 6-bit
// checkpoint's 8-bit routers at 6 bits permutes which experts win and the
// model still emits fluent text, so a sample proves nothing and this does.
Status eval_quant_linear_indexed(Node& n) {
  if (n.inputs.size() != 5) {
    return LSE_ERROR(kInvalidArgument,
                     "quant_linear_indexed takes x, packed, scales, biases, "
                     "idx");
  }
  const Node& x = *n.inputs[0];
  const Node& packed = *n.inputs[1];
  const Node& scales = *n.inputs[2];
  const Node& biases = *n.inputs[3];
  const Node& idx = *n.inputs[4];
  LSE_ASSIGN_OR(const quant::GroupAffine spec,
                quant::GroupAffine::make(n.iattrs[1], n.iattrs[2]));
  if (packed.shape.rank() != 3) {
    return LSE_ERROR(kInvalidArgument,
                     "quant_linear_indexed reads a stacked [E, out, lanes] "
                     "plane, not ", packed.shape.to_string());
  }
  const auto experts = static_cast<std::size_t>(packed.shape.dim(0));
  const auto out_dim = static_cast<std::size_t>(packed.shape.dim(1));
  const auto lanes = static_cast<std::size_t>(packed.shape.dim(2));
  const auto groups = static_cast<std::size_t>(scales.shape.dim(2));
  const std::size_t in_dim = lanes * 32 / static_cast<std::size_t>(spec.bits);
  LSE_RETURN_IF_ERROR(check_quant_planes(spec, "quant_linear_indexed",
                                         packed.shape, scales.shape,
                                         biases.shape, in_dim));
  const auto x_width =
      x.shape.rank() == 0
          ? 0
          : static_cast<std::size_t>(x.shape.dim(x.shape.rank() - 1));
  if (x_width != in_dim) {
    return LSE_ERROR(kInvalidArgument,
                     "quant_linear_indexed contracts a row of ",
                     std::to_string(x_width), " against a weight of ",
                     std::to_string(in_dim), " per output");
  }
  const auto keep =
      static_cast<std::size_t>(idx.shape.dim(idx.shape.rank() - 1));
  const auto slot = static_cast<std::size_t>(n.iattrs[0]);
  if (keep == 0 || slot >= keep) {
    return LSE_ERROR(kInvalidArgument, "quant_linear_indexed slot ",
                     std::to_string(slot), " is outside the ",
                     std::to_string(keep), " kept experts");
  }
  const auto* w = static_cast<const std::uint32_t*>(host_bytes(packed));
  if (w == nullptr) {
    return LSE_ERROR(kInternal, "quant_linear_indexed has no host copy of its "
                                "packed plane to read");
  }
  const std::size_t rows = x.element_count() / in_dim;

  for (std::size_t r = 0; r < rows; ++r) {
    const auto e = static_cast<std::size_t>(load_element(idx, r * keep + slot));
    if (e >= experts) {
      return LSE_ERROR(kOutOfRange, "expert ", std::to_string(e),
                       " is outside ", std::to_string(experts));
    }
    for (std::size_t o = 0; o < out_dim; ++o) {
      const std::uint32_t* row = w + (e * out_dim + o) * lanes;
      const std::size_t gbase = (e * out_dim + o) * groups;
      double acc = 0.0;
      for (std::size_t i = 0; i < in_dim; ++i) {
        const std::size_t g = i / static_cast<std::size_t>(spec.group_size);
        const float wv = static_cast<float>(spec.code_at(row, i)) *
                             load_element(scales, gbase + g) +
                         load_element(biases, gbase + g);
        acc += static_cast<double>(load_element(x, r * in_dim + i)) *
               static_cast<double>(wv);
      }
      store_element(n, r * out_dim + o, static_cast<float>(acc));
    }
  }
  return OkStatus();
}

// x [.., width] viewed as [N, width]; rows[i] selects a row.
Status eval_gather_rows(Node& n) {
  const Node& x = *n.inputs[0];
  const Node& rows = *n.inputs[1];
  const auto width = static_cast<std::size_t>(n.shape.dim(n.shape.rank() - 1));
  const std::size_t total = x.element_count() / width;
  const std::size_t count = rows.element_count();

  for (std::size_t i = 0; i < count; ++i) {
    const auto row = static_cast<std::size_t>(load_element(rows, i));
    if (row >= total) {
      return LSE_ERROR(kOutOfRange, "gather row ", std::to_string(row),
                       " is outside ", std::to_string(total), " rows");
    }
    for (std::size_t c = 0; c < width; ++c) {
      store_element(n, i * width + c, load_element(x, row * width + c));
    }
  }
  return OkStatus();
}

// Accumulates rather than overwrites: two routed experts may land on the same
// token, and their contributions must sum.
Status eval_scatter_add_rows(Node& n) {
  const Node& base = *n.inputs[0];
  const Node& rows = *n.inputs[1];
  const Node& values = *n.inputs[2];
  const auto width = static_cast<std::size_t>(n.shape.dim(n.shape.rank() - 1));
  const std::size_t total = n.element_count() / width;
  const std::size_t count = rows.element_count();

  for (std::size_t i = 0; i < n.element_count(); ++i) {
    store_element(n, i, load_element(base, i));
  }
  for (std::size_t i = 0; i < count; ++i) {
    const auto row = static_cast<std::size_t>(load_element(rows, i));
    if (row >= total) {
      return LSE_ERROR(kOutOfRange, "scatter row ", std::to_string(row),
                       " is outside ", std::to_string(total), " rows");
    }
    for (std::size_t c = 0; c < width; ++c) {
      const std::size_t dst = row * width + c;
      store_element(n, dst,
                    load_element(n, dst) + load_element(values, i * width + c));
    }
  }
  return OkStatus();
}

Status eval_embedding(Node& n) {
  const Node& table = *n.inputs[0];
  const Node& ids = *n.inputs[1];
  const auto dim = static_cast<std::size_t>(table.shape.dim(1));
  const auto vocab = static_cast<std::size_t>(table.shape.dim(0));
  const std::size_t count = ids.element_count();

  for (std::size_t t = 0; t < count; ++t) {
    const auto id = static_cast<std::size_t>(load_element(ids, t));
    if (id >= vocab) {
      return LSE_ERROR(kOutOfRange, "token id ", std::to_string(id),
                       " is outside a vocab of ", std::to_string(vocab));
    }
    for (std::size_t d = 0; d < dim; ++d) {
      store_element(n, t * dim + d, load_element(table, id * dim + d));
    }
  }
  return OkStatus();
}

// The same gather against a group-affine table. Like eval_quant_matmul it
// reads the packed plane at its own width, because load_element speaks floats
// and a U32 lane holds several codes.
Status eval_quant_embedding(Node& n) {
  const Node& packed = *n.inputs[0];
  const Node& scales = *n.inputs[1];
  const Node& biases = *n.inputs[2];
  const Node& ids = *n.inputs[3];
  LSE_ASSIGN_OR(const quant::GroupAffine spec,
                quant::GroupAffine::make(n.iattrs[0], n.iattrs[1]));

  const auto vocab = static_cast<std::size_t>(packed.shape.dim(0));
  const auto lanes = static_cast<std::size_t>(packed.shape.dim(1));
  const auto groups = static_cast<std::size_t>(scales.shape.dim(1));
  const std::size_t dim = lanes * 32 / static_cast<std::size_t>(spec.bits);
  LSE_RETURN_IF_ERROR(check_quant_planes(spec, "quant_embedding", packed.shape,
                                         scales.shape, biases.shape, dim));
  const auto* w = static_cast<const std::uint32_t*>(host_bytes(packed));
  if (w == nullptr) {
    return LSE_ERROR(kInternal,
                     "quant_embedding has no host copy of its packed table");
  }
  const std::size_t count = ids.element_count();

  for (std::size_t t = 0; t < count; ++t) {
    const auto id = static_cast<std::size_t>(load_element(ids, t));
    if (id >= vocab) {
      return LSE_ERROR(kOutOfRange, "token id ", std::to_string(id),
                       " is outside a vocab of ", std::to_string(vocab));
    }
    const std::uint32_t* row = w + id * lanes;
    for (std::size_t d = 0; d < dim; ++d) {
      const std::size_t g = d / static_cast<std::size_t>(spec.group_size);
      store_element(n, t * dim + d,
                    static_cast<float>(spec.code_at(row, d)) *
                            load_element(scales, id * groups + g) +
                        load_element(biases, id * groups + g));
    }
  }
  return OkStatus();
}

// Interleaved-pair rotation: (x0,x1) -> (x0*c - x1*s, x1*c + x0*s).
Status eval_rope(Node& n) {
  const Node& x = *n.inputs[0];
  const Node& cos = *n.inputs[1];
  const Node& sin = *n.inputs[2];
  const std::size_t rank = x.shape.rank();
  const auto dim = static_cast<std::size_t>(x.shape.dim(rank - 1));
  const auto seq = static_cast<std::size_t>(x.shape.dim(rank - 2));
  const std::size_t rows = x.element_count() / dim;
  const auto batch = static_cast<std::size_t>(x.shape.dim(0));
  const std::size_t rows_per_batch = batch == 0 ? rows : rows / batch;
  // A step descriptor carries one origin per row; a 1-element input is the
  // single-sequence form. Same rule as RopeKernel.
  const bool ragged =
      n.inputs.size() >= 4 && batch > 0 &&
      n.inputs[3]->element_count() >=
          static_cast<std::size_t>(kv::step_meta_elems(
              static_cast<std::int32_t>(batch)));
  const auto offset = n.inputs.size() >= 4
                          ? static_cast<std::size_t>(load_element(*n.inputs[3], 0))
                          : static_cast<std::size_t>(n.iattrs[0]);

  for (std::size_t r = 0; r < rows; ++r) {
    const std::size_t row_off =
        ragged ? static_cast<std::size_t>(load_element(
                     *n.inputs[3],
                     static_cast<std::size_t>(kv::kStepMetaHeader) +
                         (r / rows_per_batch) *
                             static_cast<std::size_t>(kv::kStepMetaPerRow)))
               : offset;
    const std::size_t t = row_off + (r % seq);
    for (std::size_t d = 0; d + 1 < dim; d += 2) {
      const float c = load_element(cos, t * dim + d);
      const float s = load_element(sin, t * dim + d);
      const float a = load_element(x, r * dim + d);
      const float b = load_element(x, r * dim + d + 1);
      store_element(n, r * dim + d, a * c - b * s);
      store_element(n, r * dim + d + 1, b * c + a * s);
    }
  }
  return OkStatus();
}

// Writes src [rows, kvh, T, width] into the pool dst [blocks, kvh, block_size,
// width] at absolute position meta[0], following the block table. Aliases the
// pool and touches only the positions it covers, exactly as overwrite_slice
// does for a contiguous cache.
Status eval_kv_page_write(Node& n) {
  if (n.inputs.size() != 4) {
    return LSE_ERROR(kInvalidArgument, "kv_page_write takes 4 inputs");
  }
  const Node& dst = *n.inputs[0];
  const Node& src = *n.inputs[1];
  const Node& meta = *n.inputs[2];
  const Node& table = *n.inputs[3];
  if (dst.shape.rank() != 4 || src.shape.rank() != 4) {
    return LSE_ERROR(kInvalidArgument,
                     "kv_page_write needs rank-4 pool and source");
  }
  const auto bs = static_cast<std::size_t>(dst.shape.dim(2));
  const auto kvh = static_cast<std::size_t>(dst.shape.dim(1));
  const auto width = static_cast<std::size_t>(dst.shape.dim(3));
  const auto pool_blocks = static_cast<std::size_t>(dst.shape.dim(0));
  const auto batch = static_cast<std::size_t>(src.shape.dim(0));
  const auto t = static_cast<std::size_t>(src.shape.dim(2));
  if (bs == 0 || kvh == 0 || width == 0 ||
      static_cast<std::size_t>(src.shape.dim(1)) != kvh ||
      static_cast<std::size_t>(src.shape.dim(3)) != width) {
    return LSE_ERROR(kInvalidArgument, "kv_page_write geometry mismatch");
  }
  const auto stride =
      static_cast<std::size_t>(table.shape.dim(table.shape.rank() - 1));
  const auto rows = static_cast<std::size_t>(load_element(meta, 2));
  if (meta.element_count() <
      static_cast<std::size_t>(
          kv::step_meta_elems(static_cast<std::int32_t>(batch)))) {
    return LSE_ERROR(kInvalidArgument, "kv_page_write got a step descriptor of ",
                     std::to_string(meta.element_count()), " for ",
                     std::to_string(batch), " rows");
  }

  const bool aliased =
      n.buffer.valid() && dst.buffer.valid() &&
      n.buffer.handle == dst.buffer.handle && n.buffer.ptr == dst.buffer.ptr;
  if (!aliased) {
    for (std::size_t i = 0; i < n.element_count(); ++i) {
      store_element(n, i, load_element(dst, i));
    }
  }
  for (std::size_t r = 0; r < batch && r < rows; ++r) {
    const std::size_t mb = static_cast<std::size_t>(kv::kStepMetaHeader) +
                           r * static_cast<std::size_t>(kv::kStepMetaPerRow);
    // Zero live length is a row holding no sequence: its table row still names
    // real blocks, and they belong to whoever held the slot last.
    if (static_cast<std::size_t>(load_element(meta, mb + 1)) == 0) continue;
    const auto pos = static_cast<std::size_t>(load_element(meta, mb));
    for (std::size_t j = 0; j < t; ++j) {
      const std::size_t abs = pos + j;
      const std::size_t slot = abs / bs;
      if (slot >= stride) {
        return LSE_ERROR(kOutOfRange, "kv_page_write position ",
                         std::to_string(abs), " needs table slot ",
                         std::to_string(slot), " of ", std::to_string(stride));
      }
      const auto blk =
          static_cast<std::size_t>(load_element(table, r * stride + slot));
      if (blk >= pool_blocks) {
        return LSE_ERROR(kOutOfRange, "kv_page_write block ",
                         std::to_string(blk), " is outside a pool of ",
                         std::to_string(pool_blocks));
      }
      for (std::size_t h = 0; h < kvh; ++h) {
        const std::size_t di =
            ((blk * kvh + h) * bs + (abs % bs)) * width;
        const std::size_t si = ((r * kvh + h) * t + j) * width;
        for (std::size_t w = 0; w < width; ++w) {
          store_element(n, di + w, load_element(src, si + w));
        }
      }
    }
  }
  return OkStatus();
}

Status eval_overwrite_slice(Node& n) {
  if (n.inputs.size() != 3) {
    return LSE_ERROR(kInvalidArgument, "overwrite_slice takes 3 inputs");
  }
  const Node& dst = *n.inputs[0];
  const Node& src = *n.inputs[1];
  const Node& begin_n = *n.inputs[2];
  const auto axis = static_cast<std::size_t>(n.iattrs[0]);
  if (axis >= dst.shape.rank() || axis >= src.shape.rank()) {
    return LSE_ERROR(kInvalidArgument, "overwrite_slice axis out of rank");
  }
  const auto begin = static_cast<std::size_t>(load_element(begin_n, 0));
  const AxisSplit d = split_axis(dst.shape, axis);
  const AxisSplit s = split_axis(src.shape, axis);
  if (s.outer != d.outer || s.inner != d.inner) {
    return LSE_ERROR(kInvalidArgument, "overwrite_slice shape mismatch");
  }
  if (begin + s.axis_len > d.axis_len) {
    return LSE_ERROR(kOutOfRange, "overwrite_slice window [",
                     std::to_string(begin), ", ",
                     std::to_string(begin + s.axis_len), ") exceeds ",
                     std::to_string(d.axis_len));
  }
  const bool aliased =
      n.buffer.valid() && dst.buffer.valid() &&
      n.buffer.handle == dst.buffer.handle && n.buffer.ptr == dst.buffer.ptr;
  if (!aliased) {
    for (std::size_t i = 0; i < n.element_count(); ++i) {
      store_element(n, i, load_element(dst, i));
    }
  }
  for (std::size_t o = 0; o < s.outer; ++o) {
    for (std::size_t a = 0; a < s.axis_len; ++a) {
      for (std::size_t i = 0; i < s.inner; ++i) {
        const std::size_t si = (o * s.axis_len + a) * s.inner + i;
        const std::size_t di = (o * d.axis_len + begin + a) * d.inner + i;
        store_element(n, di, load_element(src, si));
      }
    }
  }
  return OkStatus();
}

Status eval_sdpa(Node& n) {
  const Node& q = *n.inputs[0];
  const Node& k = *n.inputs[1];
  const Node& v = *n.inputs[2];

  const auto batch = static_cast<std::size_t>(q.shape.dim(0));
  const auto qh = static_cast<std::size_t>(q.shape.dim(1));
  const auto tq = static_cast<std::size_t>(q.shape.dim(2));
  const auto dh = static_cast<std::size_t>(q.shape.dim(3));
  const auto kvh = static_cast<std::size_t>(k.shape.dim(1));
  const auto ts = static_cast<std::size_t>(k.shape.dim(2));
  const auto dv = static_cast<std::size_t>(v.shape.dim(3));

  const float scale = n.attrs[0];
  const auto mask = static_cast<int>(n.iattrs[0]);
  const auto window = static_cast<std::size_t>(n.iattrs[1]);
  // Paged: inputs[1]/[2] are pools of blocks and inputs[4] says which block
  // holds each position. The scan order over j is unchanged, so a paged read
  // and a contiguous read of the same logical KV agree bit for bit.
  const bool paged = n.inputs.size() == 5;
  const Node* table = paged ? n.inputs[4].get() : nullptr;
  const std::size_t stride =
      paged ? static_cast<std::size_t>(table->shape.dim(table->shape.rank() - 1))
            : 0;
  const auto pass_off = n.inputs.size() >= 4
                            ? static_cast<std::size_t>(load_element(*n.inputs[3], 0))
                            : static_cast<std::size_t>(n.iattrs[2]);
  const std::size_t rows =
      paged ? static_cast<std::size_t>(load_element(*n.inputs[3], 2)) : batch;
  if (paged && n.inputs[3]->element_count() <
                   static_cast<std::size_t>(kv::step_meta_elems(
                       static_cast<std::int32_t>(batch)))) {
    return LSE_ERROR(kInvalidArgument, "sdpa got a step descriptor of ",
                     std::to_string(n.inputs[3]->element_count()), " for ",
                     std::to_string(batch), " rows");
  }
  const std::size_t group = qh / kvh;  // GQA: several q heads share one kv head

  // Element offset of key/value j of (b, kh). Paged form walks the block table;
  // ts is the block size there, not the sequence length.
  const auto kv_base = [&](std::size_t b, std::size_t kh, std::size_t j,
                           std::size_t width) -> std::size_t {
    if (!paged) return ((b * kvh + kh) * ts + j) * width;
    const auto blk = static_cast<std::size_t>(
        load_element(*table, b * stride + j / ts));
    return ((blk * kvh + kh) * ts + (j % ts)) * width;
  };

  std::vector<float> logits;
  for (std::size_t b = 0; b < batch; ++b) {
    // Rows past the real count are batch padding: they run the same code path
    // on real blocks and their output is zero, so nothing downstream can tell
    // how many rows shared the pass.
    if (b >= rows) {
      for (std::size_t e = 0; e < qh * tq * dv; ++e) {
        store_element(n, ((b * qh) * tq) * dv + e, 0.0f);
      }
      continue;
    }
    // Every row carries its own origin and its own live length, so a batch of
    // sequences at different positions masks and normalizes per row. A row
    // holding no sequence has length 0 and accumulates nothing, which is the
    // zero it must answer.
    const std::size_t mb = static_cast<std::size_t>(kv::kStepMetaHeader) +
                           b * static_cast<std::size_t>(kv::kStepMetaPerRow);
    const std::size_t offset =
        paged ? static_cast<std::size_t>(load_element(*n.inputs[3], mb))
              : pass_off;
    const std::size_t used =
        paged ? static_cast<std::size_t>(load_element(*n.inputs[3], mb + 1))
              : std::min(ts, pass_off + tq);
    logits.assign(used, 0.0f);
    for (std::size_t h = 0; h < qh; ++h) {
      const std::size_t kh = h / group;
      for (std::size_t i = 0; i < tq; ++i) {
        const std::size_t qbase = ((b * qh + h) * tq + i) * dh;
        const std::size_t abs_i = offset + i;

        float m = -std::numeric_limits<float>::infinity();
        for (std::size_t j = 0; j < used; ++j) {
          bool allowed = true;
          if (mask != 0) allowed = j <= abs_i;
          if (allowed && mask == 2 && window > 0) allowed = (abs_i - j) < window;
          if (!allowed) {
            logits[j] = -std::numeric_limits<float>::infinity();
            continue;
          }
          const std::size_t kbase = kv_base(b, kh, j, dh);
          double acc = 0.0;
          for (std::size_t d = 0; d < dh; ++d) {
            acc += static_cast<double>(load_element(q, qbase + d)) *
                   static_cast<double>(load_element(k, kbase + d));
          }
          logits[j] = static_cast<float>(acc) * scale;
          m = logits[j] > m ? logits[j] : m;
        }

        double denom = 0.0;
        for (std::size_t j = 0; j < used; ++j) {
          if (std::isinf(logits[j]) && logits[j] < 0) {
            logits[j] = 0.0f;
            continue;
          }
          logits[j] = std::exp(logits[j] - m);
          denom += static_cast<double>(logits[j]);
        }
        // A fully-masked row would divide by zero; the causal mask always keeps
        // at least the diagonal, so this only guards degenerate windows.
        if (denom == 0.0) denom = 1.0;

        const std::size_t obase = ((b * qh + h) * tq + i) * dv;
        for (std::size_t d = 0; d < dv; ++d) {
          double acc = 0.0;
          for (std::size_t j = 0; j < used; ++j) {
            if (logits[j] == 0.0f) continue;
            acc += static_cast<double>(logits[j]) *
                   static_cast<double>(load_element(v, kv_base(b, kh, j, dv) + d));
          }
          store_element(n, obase + d, static_cast<float>(acc / denom));
        }
      }
    }
  }
  return OkStatus();
}

}  // namespace

void* host_bytes(Node& node) {
  if (node.buffer.ptr != nullptr) {
    return static_cast<std::byte*>(node.buffer.ptr) + node.buffer.offset;
  }
  const std::size_t bytes =
      dtype_storage_bytes(node.dtype, node.element_count());
  if (node.host_mirror.size() < bytes) node.host_mirror.resize(bytes);
  return node.host_mirror.data();
}

const void* host_bytes(const Node& node) noexcept {
  if (node.buffer.ptr != nullptr) {
    return static_cast<const std::byte*>(node.buffer.ptr) + node.buffer.offset;
  }
  return node.host_mirror.empty() ? nullptr : node.host_mirror.data();
}

Status sync_to_device(Node& node, backend::IBackend& backend) {
  if (node.buffer.ptr != nullptr || !node.host_dirty) return OkStatus();
  if (!node.buffer.valid()) return OkStatus();
  const std::size_t bytes =
      dtype_storage_bytes(node.dtype, node.element_count());
  if (node.host_mirror.size() < bytes) return OkStatus();
  LSE_RETURN_IF_ERROR(
      backend.copy_h2d(node.host_mirror.data(), node.buffer, bytes, 0));
  node.host_dirty = false;
  return OkStatus();
}

Status sync_from_device(Node& node, backend::IBackend& backend) {
  if (node.buffer.ptr != nullptr || !node.device_dirty) return OkStatus();
  if (!node.buffer.valid()) return OkStatus();
  const std::size_t bytes =
      dtype_storage_bytes(node.dtype, node.element_count());
  if (node.host_mirror.size() < bytes) node.host_mirror.resize(bytes);
  LSE_RETURN_IF_ERROR(
      backend.copy_d2h(node.buffer, node.host_mirror.data(), bytes, 0));
  node.device_dirty = false;
  return OkStatus();
}

float load_element(const Node& node, std::size_t index) noexcept {
  if (node.kind == OpKind::kConstant && !node.buffer.valid()) {
    return node.attrs[0];
  }
  const void* p = host_bytes(node);
  if (p == nullptr) return 0.0f;
  switch (node.dtype) {
    case DType::kF32: return static_cast<const float*>(p)[index];
    case DType::kBF16: {
      bfloat16_t h;
      h.bits = static_cast<const std::uint16_t*>(p)[index];
      return h.to_float();
    }
    case DType::kF16: {
      float16_t h;
      h.bits = static_cast<const std::uint16_t*>(p)[index];
      return h.to_float();
    }
    case DType::kI32:
      return static_cast<float>(static_cast<const std::int32_t*>(p)[index]);
    case DType::kI8:
      return static_cast<float>(static_cast<const std::int8_t*>(p)[index]);
    case DType::kU8:
      return static_cast<float>(static_cast<const std::uint8_t*>(p)[index]);
    // Exact only below 2^24, same as kI32. A packed group-affine plane routinely
    // exceeds that, which is why eval_quant_matmul reads its lanes directly
    // instead of coming through here.
    case DType::kU32:
      return static_cast<float>(static_cast<const std::uint32_t*>(p)[index]);
    default: return 0.0f;
  }
}

void store_element(Node& node, std::size_t index, float value) noexcept {
  void* p = host_bytes(node);
  if (p == nullptr) return;
  // The host is now authoritative: the scheduler pushes this before it
  // dispatches, and any pending pull would copy stale device bytes back over
  // what was just written. Callers write whole buffers, never a subset of one
  // the device still owns.
  node.host_dirty = true;
  node.device_dirty = false;
  switch (node.dtype) {
    case DType::kF32: static_cast<float*>(p)[index] = value; break;
    case DType::kBF16:
      static_cast<std::uint16_t*>(p)[index] = bfloat16_t::from_float(value);
      break;
    case DType::kF16:
      static_cast<std::uint16_t*>(p)[index] = float16_t::from_float(value);
      break;
    case DType::kI32:
      static_cast<std::int32_t*>(p)[index] = static_cast<std::int32_t>(value);
      break;
    case DType::kI8:
      static_cast<std::int8_t*>(p)[index] = static_cast<std::int8_t>(value);
      break;
    case DType::kU8:
      static_cast<std::uint8_t*>(p)[index] = static_cast<std::uint8_t>(value);
      break;
    case DType::kU32:
      static_cast<std::uint32_t*>(p)[index] = static_cast<std::uint32_t>(value);
      break;
    default: break;
  }
}

Status ensure_output_buffer(Node& node, backend::IBackend& backend) {
  return ensure_buffer(node, backend);
}

Status evaluate(const NodePtr& node, backend::IBackend& backend) {
  Node& n = *node;
  if (n.materialized) return OkStatus();

  if (n.prim != nullptr) {
    const int a = n.prim->inplace_input();
    if (a >= 0 && static_cast<std::size_t>(a) < n.inputs.size() &&
        n.inputs[static_cast<std::size_t>(a)] &&
        n.inputs[static_cast<std::size_t>(a)]->buffer.valid() &&
        !n.buffer.valid()) {
      n.buffer = n.inputs[static_cast<std::size_t>(a)]->buffer;
    }
  }

  LSE_RETURN_IF_ERROR(ensure_buffer(n, backend));
  // Every read below is a host read, so anything a kernel wrote has to come
  // back first. No-op unless a device group actually produced the input.
  for (const NodePtr& in : n.inputs) {
    LSE_RETURN_IF_ERROR(sync_from_device(*in, backend));
  }
  const std::size_t count = n.element_count();

  // Anything with a registered primitive evaluates through it. Built-in
  // elementwise ops attach one too, so there is a single path.
  // A primitive without a host body is not automatically fatal: the
  // interpreter may still have a built-in rule for the op (matmul carries a
  // device kernel primitive but the host path is the oracle). Only a kCustom
  // node has nowhere else to go.
  if (n.prim != nullptr && !n.prim->has_host_impl() && n.kind == OpKind::kCustom) {
    return LSE_ERROR(kUnimplemented, "primitive '",
                     std::string(n.prim->name()),
                     "' is device source only; it needs the JIT backend");
  }
  if (n.prim != nullptr && n.prim->has_host_impl()) {
    std::vector<std::vector<float>> staged(n.inputs.size());
    std::vector<const float*> ptrs(n.inputs.size());
    for (std::size_t i = 0; i < n.inputs.size(); ++i) {
      staged[i].resize(count);
      const Node& src = *n.inputs[i];
      for (std::size_t e = 0; e < count; ++e) {
        staged[i][e] = load_element(src, broadcast_index(src.shape, n.shape, e));
      }
      ptrs[i] = staged[i].data();
    }
    std::vector<float> result(count);
    n.prim->eval_cpu(ptrs, result.data(), count, n.attrs);
    for (std::size_t e = 0; e < count; ++e) store_element(n, e, result[e]);
    n.materialized = true;
    return OkStatus();
  }

  switch (n.kind) {
    case OpKind::kConstant:
      for (std::size_t i = 0; i < count; ++i) store_element(n, i, n.attrs[0]);
      break;

    case OpKind::kSum: case OpKind::kMax: case OpKind::kMean:
      LSE_RETURN_IF_ERROR(eval_reduction(n));
      break;

    case OpKind::kSoftmax:
      LSE_RETURN_IF_ERROR(eval_softmax(n));
      break;

    case OpKind::kRMS:
      LSE_RETURN_IF_ERROR(eval_rms_norm(n));
      break;

    case OpKind::kMatMul:
      LSE_RETURN_IF_ERROR(eval_matmul(n));
      break;

    case OpKind::kSoftplus:
      LSE_RETURN_IF_ERROR(eval_softplus(n));
      break;

    case OpKind::kL2Norm:
      LSE_RETURN_IF_ERROR(eval_l2_norm(n));
      break;

    case OpKind::kCausalConv1d:
      LSE_RETURN_IF_ERROR(eval_causal_conv1d(n));
      break;

    case OpKind::kConvTailShift:
      LSE_RETURN_IF_ERROR(eval_conv_tail(n));
      break;

    case OpKind::kGDNChunkScan:
      LSE_RETURN_IF_ERROR(eval_gated_delta(n));
      break;

    case OpKind::kLinear:
      LSE_RETURN_IF_ERROR(eval_linear(n));
      break;

    case OpKind::kQuantMatMul:
      LSE_RETURN_IF_ERROR(eval_quant_matmul(n));
      break;

    // Both routed contractions are the same dispatch; the fifth input is the
    // group-affine plane's scales and biases, which is what separates them.
    case OpKind::kMoEDispatch:
      LSE_RETURN_IF_ERROR(n.inputs.size() == 5
                              ? eval_quant_linear_indexed(n)
                              : eval_linear_indexed(n));
      break;

    case OpKind::kGather:
      LSE_RETURN_IF_ERROR(eval_gather_rows(n));
      break;

    case OpKind::kScatter:
      LSE_RETURN_IF_ERROR(eval_scatter_add_rows(n));
      break;

    case OpKind::kOverwriteSlice:
      LSE_RETURN_IF_ERROR(eval_overwrite_slice(n));
      break;

    case OpKind::kEmbedding:
      LSE_RETURN_IF_ERROR(eval_embedding(n));
      break;

    case OpKind::kQuantEmbedding:
      LSE_RETURN_IF_ERROR(eval_quant_embedding(n));
      break;

    case OpKind::kTranspose:
      LSE_RETURN_IF_ERROR(eval_transpose(n));
      break;

    case OpKind::kConcat:
      LSE_RETURN_IF_ERROR(eval_concat(n));
      break;

    case OpKind::kSlice:
      LSE_RETURN_IF_ERROR(eval_slice(n));
      break;

    case OpKind::kRepeat:
      LSE_RETURN_IF_ERROR(eval_repeat(n));
      break;

    case OpKind::kRoPE:
      LSE_RETURN_IF_ERROR(eval_rope(n));
      break;

    case OpKind::kAttention:
      LSE_RETURN_IF_ERROR(eval_sdpa(n));
      break;

    case OpKind::kKvPageWrite:
      LSE_RETURN_IF_ERROR(eval_kv_page_write(n));
      break;

    case OpKind::kTopK:
      LSE_RETURN_IF_ERROR(eval_topk(n));
      break;

    case OpKind::kArgMax:
      LSE_RETURN_IF_ERROR(eval_argmax(n));
      break;

    default:
      if (n.kind == OpKind::kCast || n.kind == OpKind::kReshape) {
        const Node& src = *n.inputs[0];
        for (std::size_t i = 0; i < count; ++i) {
          store_element(n, i,
                        load_element(src, broadcast_index(src.shape, n.shape, i)));
        }
        break;
      }
      return LSE_ERROR(kUnimplemented, "cpu interpreter has no rule for ",
                       std::string(to_string(n.kind)));
  }

  n.materialized = true;
  return OkStatus();
}

Result<float> read_scalar(const Node& node) {
  if (node.element_count() == 0) {
    return LSE_ERROR(kOutOfRange, "item() on an empty Array");
  }
  return load_element(node, 0);
}

Status read_raw(const Node& node, void* dst, std::size_t bytes) {
  if (!node.buffer.valid()) return LSE_ERROR(kInternal, "node has no buffer");
  const std::size_t have = dtype_storage_bytes(node.dtype, node.element_count());
  if (bytes > have) {
    return LSE_ERROR(kOutOfRange, "to_host asked for ", std::to_string(bytes),
                     " bytes, tensor holds ", std::to_string(have));
  }
  const void* src = host_bytes(node);
  if (src == nullptr) return LSE_ERROR(kInternal, "node has no host bytes");
  std::memcpy(dst, src, bytes);
  return OkStatus();
}

}  // namespace lse::graph::interpreter
