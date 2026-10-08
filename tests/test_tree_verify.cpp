// Draft-tree verification kernels against the chain they stand in for.
//
// Without arguments: the tree kernels emit for a described gfx1201 and the
// tree-masked attention takes the split and flash paths it should.
// `--gpu`: on the device, every tree node's Gated DeltaNet output and every
// path's state against the chain scan over that node's path (bit for bit),
// the tree conv against the chain conv over each path (bit for bit), the
// conv tail and the path K/V write against host references, and the
// tree-masked attention (flash and split) against a double reference.
#include "harness.hpp"
#include "lse/backends/hrx/loomc/loom_emitter.hpp"
#include "lse/graph/kernel_primitive.hpp"
#include "lse/graph/ops.hpp"
#include "lse/graph/program.hpp"
#include "lse/kv/block.hpp"
#include "lse/kv/cache_dtype.hpp"
#include "lse/runtime/draft_tree.hpp"

#include <chrono>
#include <cmath>
#include <functional>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string_view>
#include <vector>

using namespace lse;
using namespace lse::graph;

namespace {

runtime::DraftTree make_tree(std::uint32_t nodes, std::uint64_t seed) {
  runtime::DraftLattice l;
  l.positions = 7;
  l.top = 4;
  std::mt19937 rng(static_cast<std::uint32_t>(seed));
  std::uniform_real_distribution<float> uni(0.0f, 3.0f);
  for (std::uint32_t p = 0; p < l.positions; ++p)
    for (std::uint32_t k = 0; k < l.top; ++k) l.ids.push_back(100 * p + k);
  for (std::size_t i = 0; i < std::size_t{l.positions} * l.top * l.top; ++i) l.scores.push_back(uni(rng));
  auto e = runtime::expand_tree(l, 1.0, nodes);
  LSE_EXPECT_OK(e.status());
  auto t = runtime::layout_tree(l, 7, *e, nodes);
  LSE_EXPECT_OK(t.status());
  return *t;
}

std::vector<std::uint32_t> path_to(const runtime::DraftTree& t, std::uint32_t row) {
  std::vector<std::uint32_t> path;
  for (auto r = static_cast<std::int32_t>(row); r >= 0; r = t.parent[static_cast<std::size_t>(r)])
    path.insert(path.begin(), static_cast<std::uint32_t>(r));
  return path;
}

Array leaf(Shape shape, DType dtype = DType::kF32) {
  auto node = std::make_shared<Node>();
  node->shape = shape;
  node->dtype = dtype;
  node->materialized = true;
  return Array(node);
}

backend::DeviceInfo described_gfx1201() {
  backend::DeviceInfo gpu;
  gpu.arch = "gfx1201";
  gpu.wavefront_size = 32;
  gpu.max_threads_per_workgroup = 1024;
  gpu.lds_bytes_per_workgroup = 65536;
  gpu.compute_units = 64;
  return gpu;
}

template <class T>
Array filled(Shape shape, DType type, const std::vector<T>& values) {
  auto* scheduler = default_scheduler();
  auto storage = scheduler->backend().allocate(values.size() * sizeof(T),
                                               backend::MemoryClass::kDevice);
  LSE_EXPECT(storage.ok());
  if (!storage.ok()) return {};
  auto buffer = storage.release();
  LSE_EXPECT_OK(scheduler->backend().copy(buffer, values.data(), values.size() * sizeof(T)));
  return Array::from_buffer(std::move(buffer), shape, type);
}

template <class T>
std::vector<T> read(Array value) {
  std::vector<T> data(value.shape().elem_count());
  LSE_EXPECT_OK(value.to_host(data.data(), data.size() * sizeof(T)));
  return data;
}

Status run(std::initializer_list<Array> outs) {
  std::vector<NodePtr> roots;
  for (const Array& a : outs) roots.push_back(a.node());
  auto* scheduler = default_scheduler();
  LSE_RETURN_IF_ERROR(scheduler->eval(roots, false));
  return scheduler->drain();
}

std::uint16_t to_bf16(float x) {
  std::uint32_t bits;
  std::memcpy(&bits, &x, 4);
  bits += 0x7fffu + ((bits >> 16) & 1u);
  return static_cast<std::uint16_t>(bits >> 16);
}
float from_bf16(std::uint16_t b) {
  const std::uint32_t bits = static_cast<std::uint32_t>(b) << 16;
  float x;
  std::memcpy(&x, &bits, 4);
  return x;
}

// Conv ancestors as HybridLM writes them: row b levels up, or -1 - c for
// tail column c past the root.
std::vector<float> conv_ancestors(const runtime::DraftTree& t, std::uint32_t taps) {
  std::vector<float> out;
  for (std::uint32_t r = 0; r < t.rows(); ++r)
    for (std::uint32_t b = 1; b <= taps; ++b) {
      const std::int32_t a = t.ancestor(r, b);
      const auto d = static_cast<std::int32_t>(t.depth[r]);
      out.push_back(a >= 0 ? static_cast<float>(a)
                           : -1.0f - static_cast<float>(static_cast<std::int32_t>(taps) - (static_cast<std::int32_t>(b) - d)));
    }
  return out;
}

std::vector<float> descriptor(const std::vector<std::uint32_t>& path, std::uint32_t first,
                              std::size_t capacity) {
  std::vector<float> d(2 + capacity, 0.0f);
  d[0] = static_cast<float>(path.size());
  d[1] = static_cast<float>(first);
  for (std::size_t j = 0; j < path.size(); ++j) d[2 + j] = static_cast<float>(path[j]);
  return d;
}

}  // namespace

LSE_TEST(tree_kernels_emit_for_a_described_device) {
  const auto gpu = described_gfx1201();
  backend::LoomEmitter emitter;
  const std::int64_t n = 25;
  auto q = leaf({1, n, 16, 128}), k = leaf({1, n, 16, 128}), v = leaf({1, n, 48, 128});
  auto a = leaf({1, n, 48}), b = leaf({1, n, 48}), s0 = leaf({1, 48, 128, 128});
  auto depth = leaf({n}), path = leaf({11});
  auto x = leaf({1, n, 10240}), w = leaf({10240, 4}), bias = leaf({10240}),
       tail = leaf({1, 3, 10240}), anc = leaf({n, 3});
  auto pool = leaf({65, 4, 16, 256}, DType::kBF16), src = leaf({1, 4, n, 256}),
       table = leaf({1, 64});
  const Array outs[] = {
      gated_delta_tree(q, k, v, a, b, s0, depth), gated_delta_path(k, v, a, b, s0, path),
      causal_conv1d_tree(x, w, bias, tail, anc), conv_tail_rows(tail, x, path),
      kv_page_write_rows(pool, src, table, path, 16, kv::CacheDType::kBF16)};
  for (const Array& out : outs) {
    LSE_EXPECT(out.valid() && out.node()->prim != nullptr);
    if (!out.valid() || out.node()->prim == nullptr) continue;
    const NodePtr roots[]{out.node()};
    for (const auto& group : Partitioner::partition(roots, &gpu)) {
      auto emitted = emitter.emit(group, gpu);
      if (!emitted.ok())
        std::fprintf(stderr, "%s: %s\n", std::string(out.node()->prim->name()).c_str(),
                     std::string(emitted.status().message()).c_str());
      LSE_EXPECT(emitted.ok());
    }
  }
}

LSE_TEST(tree_mask_takes_the_split_and_flash_paths) {
  auto gpu = described_gfx1201();
  gpu.max_threads_per_workgroup = 256;
  for (const std::int64_t n : {4, 8, 16, 32}) {
    for (const int capacity : {256, 4096}) {
      auto q = leaf({1, 24, n, 256});
      auto k = leaf({capacity / 16 + 1, 4, 16, 256}, DType::kBF16), v = leaf(k.shape(), DType::kBF16);
      auto meta = leaf({kv::tree_meta_elems(1, static_cast<std::int32_t>(n))});
      auto table = leaf({1, capacity / 16});
      auto o = sdpa_paged(q, k, v, 0.0625f, MaskKind::kTree, 0, meta, table, 16, &gpu,
                          kv::CacheDType::kBF16);
      // A tree takes the split kernel at a context it serves (up to 32 rows).
      const bool split = capacity >= 1024 && n <= 32;
      LSE_EXPECT(o.node()->prim->name() ==
                 (split ? "attention.split_merge128.wg128c2.v1" : "attention"));
      const NodePtr roots[]{o.node()};
      backend::LoomEmitter emitter;
      for (const auto& group : Partitioner::partition(roots, &gpu)) {
        auto emitted = emitter.emit(group, gpu);
        if (!emitted.ok())
          std::fprintf(stderr, "tree attention n%lld cap%d: %s\n", static_cast<long long>(n), capacity,
                       std::string(emitted.status().message()).c_str());
        LSE_EXPECT(emitted.ok());
      }
    }
  }
}

namespace {

double max_diff(const std::vector<float>& a, const std::vector<float>& b) {
  double worst = 0;
  for (std::size_t i = 0; i < a.size() && i < b.size(); ++i)
    worst = std::max(worst, std::abs(static_cast<double>(a[i]) - b[i]));
  return a.size() == b.size() ? worst : INFINITY;
}

int gpu() {
  auto* scheduler = default_scheduler();
  if (!scheduler) return 1;
  scheduler->set_mode(Scheduler::Mode::kDeviceFirst);
  scheduler->set_dialect(Dialect::kLoom);
  std::mt19937 rng(91);
  std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
  const runtime::DraftTree tree = make_tree(24, 5);
  const auto n = static_cast<std::int64_t>(tree.rows());
  std::printf("tree: %lld rows, depth %u, top path %u rows\n", static_cast<long long>(n),
              tree.max_depth(), tree.main_rows);
  std::vector<float> depth;
  for (std::uint32_t r = 0; r < tree.rows(); ++r) depth.push_back(static_cast<float>(tree.depth[r]));

  // ---- Gated DeltaNet: every node against the chain scan over its path.
  const std::int64_t H = 48, KH = 16, D = 128;
  std::vector<float> q(n * KH * D), k(q.size()), v(n * H * D), alpha(n * H), beta(n * H), s0(H * D * D);
  for (auto& x : q) x = uni(rng) * 0.1f;
  for (std::int64_t i = 0; i < n * KH; ++i) {
    double norm = 0;
    for (std::int64_t j = 0; j < D; ++j) norm += (k[i * D + j] = uni(rng)) * k[i * D + j];
    for (std::int64_t j = 0; j < D; ++j) k[i * D + j] /= static_cast<float>(std::sqrt(norm));
  }
  for (auto& x : v) x = uni(rng);
  for (auto& x : alpha) x = 0.9f + 0.1f * std::abs(uni(rng));
  for (auto& x : beta) x = 0.5f * std::abs(uni(rng));
  for (auto& x : s0) x = 0.1f * uni(rng);
  auto aq = filled(Shape{1, n, KH, D}, DType::kF32, q), ak = filled(Shape{1, n, KH, D}, DType::kF32, k);
  auto av = filled(Shape{1, n, H, D}, DType::kF32, v), aa = filled(Shape{1, n, H}, DType::kF32, alpha);
  auto ab = filled(Shape{1, n, H}, DType::kF32, beta), as = filled(Shape{1, H, D, D}, DType::kF32, s0);
  auto ad = filled(Shape{n}, DType::kF32, depth);
  auto tree_out = gated_delta_tree(aq, ak, av, aa, ab, as, ad);
  scheduler->reset_accumulated_trace();
  LSE_EXPECT_OK(run({tree_out}));
  LSE_EXPECT_EQ(scheduler->accumulated_trace().host_groups, 0u);
  const auto got = read<float>(tree_out);
  const auto gather = [&](const std::vector<float>& src, std::int64_t width,
                          const std::vector<std::uint32_t>& rows) {
    std::vector<float> out;
    for (const auto r : rows)
      out.insert(out.end(), src.begin() + static_cast<std::ptrdiff_t>(r * width),
                 src.begin() + static_cast<std::ptrdiff_t>((r + 1) * width));
    return out;
  };
  // A double reference along each row's path: output and final state.
  const auto reference = [&](const std::vector<std::uint32_t>& path, std::vector<double>& out,
                             std::vector<double>& state) {
    out.assign(H * D, 0.0);
    state.assign(H * D * D, 0.0);
    for (std::int64_t h = 0; h < H; ++h)
      for (std::int64_t r = 0; r < D; ++r) {
        double* srow = &state[(h * D + r) * D];
        for (std::int64_t j = 0; j < D; ++j) srow[j] = s0[(h * D + r) * D + j];
        for (const auto t : path) {
          const std::int64_t sc = t * H + h, kv = (t * KH + h / (H / KH)) * D;
          double sk = 0;
          for (std::int64_t j = 0; j < D; ++j) sk += (srow[j] *= alpha[sc]) * k[kv + j];
          const double delta = (v[sc * D + r] - sk) * beta[sc];
          double acc = 0;
          for (std::int64_t j = 0; j < D; ++j) acc += (srow[j] += delta * k[kv + j]) * q[kv + j];
          out[h * D + r] = acc;
        }
      }
  };
  double ref_tree = 0, ref_chain = 0, ref_state = 0;
  double gdn_worst = 0, state_worst = 0;
  for (std::uint32_t row = 0; row < tree.rows(); ++row) {
    const auto path = path_to(tree, row);
    const auto L = static_cast<std::int64_t>(path.size());
    auto cq = filled(Shape{1, L, KH, D}, DType::kF32, gather(q, KH * D, path));
    auto ck = filled(Shape{1, L, KH, D}, DType::kF32, gather(k, KH * D, path));
    auto cv = filled(Shape{1, L, H, D}, DType::kF32, gather(v, H * D, path));
    auto ca = filled(Shape{1, L, H}, DType::kF32, gather(alpha, H, path));
    auto cb = filled(Shape{1, L, H}, DType::kF32, gather(beta, H, path));
    Array chain_state;
    auto chain = gated_delta_step(cq, ck, cv, ca, cb, as, &chain_state);
    auto desc = filled(Shape{static_cast<std::int64_t>(2 + std::min<std::int64_t>(n, 9))}, DType::kF32,
                       descriptor(path, 0, static_cast<std::size_t>(std::min<std::int64_t>(n, 9))));
    auto path_state = gated_delta_path(ak, av, aa, ab, as, desc);
    LSE_EXPECT_OK(run({chain, chain_state, path_state}));
    const auto want = read<float>(chain);
    const std::vector<float> last(want.end() - H * D, want.end());
    const std::vector<float> mine(got.begin() + row * H * D, got.begin() + (row + 1) * H * D);
    gdn_worst = std::max(gdn_worst, max_diff(last, mine));
    const auto chain_s = read<float>(chain_state), path_s = read<float>(path_state);
    state_worst = std::max(state_worst, max_diff(chain_s, path_s));
    if (row < 3 || row + 1 == tree.rows()) {
      std::vector<double> ro, rs;
      reference(path, ro, rs);
      for (std::size_t i = 0; i < ro.size(); ++i) {
        ref_tree = std::max(ref_tree, std::abs(ro[i] - mine[i]));
        ref_chain = std::max(ref_chain, std::abs(ro[i] - last[i]));
      }
      for (std::size_t i = 0; i < rs.size(); ++i) ref_state = std::max(ref_state, std::abs(rs[i] - path_s[i]));
    }
  }
  std::printf("against a double reference: tree %.3g, chain %.3g, path state %.3g\n", ref_tree,
              ref_chain, ref_state);
  std::printf("gdn tree vs chain over each path: worst output diff %.3g, worst state diff %.3g\n",
              gdn_worst, state_worst);
  LSE_EXPECT(gdn_worst == 0.0);
  LSE_EXPECT(state_worst == 0.0);

  // ---- Conv over each path, and the conv tail after a path.
  const std::int64_t C = 512, K = 4;
  std::vector<float> x(n * C), w(C * K), bias(C), tail((K - 1) * C);
  for (auto* vec : {&x, &w, &bias, &tail})
    for (auto& value : *vec) value = uni(rng);
  auto ax = filled(Shape{1, n, C}, DType::kF32, x), aw = filled(Shape{C, K}, DType::kF32, w);
  auto abias = filled(Shape{C}, DType::kF32, bias), atail = filled(Shape{1, K - 1, C}, DType::kF32, tail);
  auto anc = filled(Shape{n, K - 1}, DType::kF32, conv_ancestors(tree, static_cast<std::uint32_t>(K - 1)));
  auto conv = causal_conv1d_tree(ax, aw, abias, atail, anc);
  LSE_EXPECT_OK(run({conv}));
  const auto conv_got = read<float>(conv);
  double conv_worst = 0, tail_worst = 0;
  for (std::uint32_t row = 0; row < tree.rows(); ++row) {
    const auto path = path_to(tree, row);
    const auto L = static_cast<std::int64_t>(path.size());
    auto cx = filled(Shape{1, L, C}, DType::kF32, gather(x, C, path));
    auto chain = causal_conv1d(cx, aw, abias, atail);
    auto chain_tail = conv_tail(atail, cx);
    auto desc = filled(Shape{static_cast<std::int64_t>(2 + std::min<std::int64_t>(n, 9))}, DType::kF32,
                       descriptor(path, 0, static_cast<std::size_t>(std::min<std::int64_t>(n, 9))));
    auto rows_tail = conv_tail_rows(atail, ax, desc);
    LSE_EXPECT_OK(run({chain, chain_tail, rows_tail}));
    const auto want = read<float>(chain);
    const std::vector<float> last(want.end() - C, want.end());
    const std::vector<float> mine(conv_got.begin() + row * C, conv_got.begin() + (row + 1) * C);
    conv_worst = std::max(conv_worst, max_diff(last, mine));
    tail_worst = std::max(tail_worst, max_diff(read<float>(chain_tail), read<float>(rows_tail)));
  }
  std::printf("tree conv vs chain conv: worst diff %.3g; conv tail: %.3g\n", conv_worst, tail_worst);
  LSE_EXPECT(conv_worst == 0.0);
  LSE_EXPECT(tail_worst == 0.0);

  // ---- Attention with the tree mask, at a short (flash) and a long context
  // (split for a tree of up to eight rows, flash for more).
  const runtime::DraftTree small = make_tree(7, 9);
  for (const runtime::DraftTree* shape : {&tree, &small})
  for (const std::int32_t first : {40, 1500}) {
    const runtime::DraftTree& tree = *shape;
    const auto n = static_cast<std::int64_t>(tree.rows());
    const std::int64_t kvh = 4, qh = 24, hd = 256, bs = 16;
    const std::int32_t live = first + static_cast<std::int32_t>(n);
    const std::int64_t capacity = live <= 256 ? 256 : 2048;
    const std::int64_t blocks = capacity / bs;
    std::vector<std::uint16_t> kpool(blocks * kvh * bs * hd), vpool(kpool.size());
    for (auto& e : kpool) e = to_bf16(uni(rng));
    for (auto& e : vpool) e = to_bf16(uni(rng));
    std::vector<float> table(blocks), qv(qh * n * hd);
    for (std::int64_t b = 0; b < blocks; ++b) table[b] = static_cast<float>(blocks - 1 - b);
    for (auto& e : qv) e = uni(rng);
    std::vector<float> meta(kv::tree_meta_elems(1, static_cast<std::int32_t>(n)), 0.0f);
    meta[0] = static_cast<float>(first);
    meta[1] = static_cast<float>(live);
    meta[2] = 1;
    meta[3] = static_cast<float>(first);
    meta[4] = static_cast<float>(live);
    for (std::uint32_t r = 0; r < tree.rows(); ++r)
      for (std::uint32_t c = 0; c < tree.rows(); ++c)
        meta[kv::tree_mask_offset(1) + r * n + c] = tree.sees(r, c) ? 1.0f : 0.0f;
    auto aqq = filled(Shape{1, qh, n, hd}, DType::kF32, qv);
    auto akp = filled(Shape{blocks, kvh, bs, hd}, DType::kBF16, kpool);
    auto avp = filled(Shape{blocks, kvh, bs, hd}, DType::kBF16, vpool);
    auto ameta = filled(Shape{static_cast<std::int64_t>(meta.size())}, DType::kF32, meta);
    auto atable = filled(Shape{1, blocks}, DType::kF32, table);
    const std::size_t member = preferred_member();
    const auto* device = member < scheduler->devices().size()
        ? &scheduler->devices().device(member).device_info() : nullptr;
    auto o = sdpa_paged(aqq, akp, avp, 0.0625f, MaskKind::kTree, 0, ameta, atable,
                        static_cast<int>(bs), device, kv::CacheDType::kBF16);
    LSE_EXPECT_OK(run({o}));
    const auto out = read<float>(o);
    const auto key_at = [&](const std::vector<std::uint16_t>& pool, std::int64_t pos, std::int64_t h,
                            std::int64_t d) {
      const auto blk = static_cast<std::int64_t>(table[pos / bs]);
      return from_bf16(pool[((blk * kvh + h) * bs + pos % bs) * hd + d]);
    };
    double worst = 0, causal_gap = 0;
    for (std::int64_t h = 0; h < qh; ++h)
      for (std::int64_t i = 0; i < n; ++i) {
        for (const bool tree_mask : {true, false}) {
          std::vector<double> score(live, -INFINITY);
          double top = -INFINITY;
          for (std::int64_t j = 0; j < live; ++j) {
            const bool seen = j < first ? true
                : tree_mask ? tree.sees(static_cast<std::uint32_t>(i), static_cast<std::uint32_t>(j - first))
                            : j <= first + i;
            if (!seen) continue;
            double s = 0;
            for (std::int64_t d = 0; d < hd; ++d)
              s += static_cast<double>(qv[(h * n + i) * hd + d]) * key_at(kpool, j, h / (qh / kvh), d);
            score[j] = s * 0.0625;
            top = std::max(top, score[j]);
          }
          double denom = 0;
          for (auto& s : score) denom += (s = std::exp(s - top));
          for (std::int64_t d = 0; d < hd; ++d) {
            double acc = 0;
            for (std::int64_t j = 0; j < live; ++j)
              if (score[j] > 0) acc += score[j] * key_at(vpool, j, h / (qh / kvh), d);
            const double diff = std::abs(acc / denom - out[(h * n + i) * hd + d]);
            if (tree_mask) worst = std::max(worst, diff);
            else causal_gap = std::max(causal_gap, diff);
          }
        }
      }
    std::printf("tree attention %lld rows, first %d (%s): worst diff %.3g vs reference, %.3g vs a causal mask\n",
                static_cast<long long>(n), first, o.node()->prim->name().data(), worst, causal_gap);
    LSE_EXPECT(worst < 2e-3);
    LSE_EXPECT(causal_gap > 4 * worst);
  }

  // ---- The accepted path's keys written into place.
  {
    const std::int64_t kvh = 4, hd = 256, bs = 16, blocks = 8;
    std::vector<std::uint16_t> pool(blocks * kvh * bs * hd);
    for (auto& e : pool) e = to_bf16(uni(rng));
    std::vector<float> src(kvh * n * hd), table(blocks);
    for (auto& e : src) e = uni(rng);
    for (std::int64_t b = 0; b < blocks; ++b) table[b] = static_cast<float>((b * 3) % blocks);
    const auto path = path_to(tree, tree.rows() - 1);
    const std::uint32_t first = 37;
    auto apool = filled(Shape{blocks, kvh, bs, hd}, DType::kBF16, pool);
    auto asrc = filled(Shape{1, kvh, n, hd}, DType::kF32, src);
    auto atable = filled(Shape{1, blocks}, DType::kF32, table);
    auto desc = filled(Shape{static_cast<std::int64_t>(2 + std::min<std::int64_t>(n, 9))}, DType::kF32,
                       descriptor(path, first, static_cast<std::size_t>(std::min<std::int64_t>(n, 9))));
    auto written = kv_page_write_rows(apool, asrc, atable, desc, static_cast<int>(bs),
                                      kv::CacheDType::kBF16);
    LSE_EXPECT_OK(run({written}));
    const auto after = read<std::uint16_t>(apool);
    std::size_t wrong = 0, moved = 0;
    for (std::int64_t pos = 0; pos < blocks * bs; ++pos)
      for (std::int64_t h = 0; h < kvh; ++h)
        for (std::int64_t d = 0; d < hd; ++d) {
          const auto blk = static_cast<std::int64_t>(table[pos / bs]);
          const auto at = ((blk * kvh + h) * bs + pos % bs) * hd + d;
          const std::int64_t j = pos - first;
          std::uint16_t want = pool[at];
          if (j >= 0 && j < static_cast<std::int64_t>(path.size())) {
            want = to_bf16(src[(h * n + path[j]) * hd + d]);
            ++moved;
          }
          wrong += after[at] != want;
        }
    std::printf("path K/V write: %zu values moved, %zu wrong\n", moved, wrong);
    LSE_EXPECT_EQ(wrong, std::size_t{0});
  }
  return lse::test::Registry::get().failures ? 1 : 0;
}

}  // namespace

// `--time`: device time of the Gated DeltaNet kernels a tree pass and its
// commit run, against the chain scan, at the model's geometry.
int time_kernels() {
  auto* scheduler = default_scheduler();
  if (!scheduler) return 1;
  scheduler->set_mode(Scheduler::Mode::kDeviceFirst);
  scheduler->set_dialect(Dialect::kLoom);
  std::mt19937 rng(3);
  std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
  const std::int64_t H = 48, KH = 16, D = 128;
  const auto timed = [&](const char* what, const std::function<Array()>& make) {
    Array warm = make();
    LSE_EXPECT_OK(run({warm}));
    constexpr int kReps = 200;
    std::vector<NodePtr> roots;
    std::vector<Array> keep;
    for (int i = 0; i < kReps; ++i) {
      keep.push_back(make());
      roots.push_back(keep.back().node());
    }
    const auto t0 = std::chrono::steady_clock::now();
    LSE_EXPECT_OK(scheduler->eval(roots, false));
    LSE_EXPECT_OK(scheduler->drain());
    const auto us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
    std::printf("%-40s %8.1f us per launch\n", what, us / kReps);
  };
  for (const std::int64_t n : {8, 16, 32}) {
    const runtime::DraftTree tree = make_tree(static_cast<std::uint32_t>(n - 1), 11);
    std::vector<float> q(n * KH * D), k(q.size()), v(n * H * D), al(n * H), bt(n * H), s0(H * D * D), depth;
    for (auto* x : {&q, &k, &v, &s0}) for (auto& e : *x) e = 0.05f * uni(rng);
    for (auto& e : al) e = 0.95f;
    for (auto& e : bt) e = 0.3f;
    for (std::uint32_t r = 0; r < tree.rows(); ++r) depth.push_back(static_cast<float>(tree.depth[r]));
    auto aq = filled(Shape{1, n, KH, D}, DType::kF32, q), ak = filled(Shape{1, n, KH, D}, DType::kF32, k);
    auto av = filled(Shape{1, n, H, D}, DType::kF32, v), aa = filled(Shape{1, n, H}, DType::kF32, al);
    auto ab = filled(Shape{1, n, H}, DType::kF32, bt), as = filled(Shape{1, H, D, D}, DType::kF32, s0);
    auto ad = filled(Shape{n}, DType::kF32, depth);
    const auto cap = std::min<std::int64_t>(n, 9);
    std::vector<std::uint32_t> main;
    for (std::uint32_t r = 0; r < tree.main_rows; ++r) main.push_back(r);
    auto desc = filled(Shape{2 + cap}, DType::kF32, descriptor(main, 0, static_cast<std::size_t>(cap)));
    char label[64];
    std::snprintf(label, sizeof label, "chain scan, %lld rows", static_cast<long long>(n));
    timed(label, [&] { Array st; auto o = gated_delta_step(aq, ak, av, aa, ab, as, &st); return o; });
    std::snprintf(label, sizeof label, "tree scan, %lld rows", static_cast<long long>(n));
    timed(label, [&] { return gated_delta_tree(aq, ak, av, aa, ab, as, ad); });
    std::snprintf(label, sizeof label, "path state, %u of %lld rows", tree.main_rows, static_cast<long long>(n));
    timed(label, [&] { return gated_delta_path(ak, av, aa, ab, as, desc); });
  }
  return lse::test::Registry::get().failures ? 1 : 0;
}

int main(int argc, char** argv) {
  if (argc == 2 && std::string_view(argv[1]) == "--gpu") return gpu();
  if (argc == 2 && std::string_view(argv[1]) == "--time") return time_kernels();
  return lse::test::run_all();
}
