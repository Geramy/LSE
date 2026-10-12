// Long-context attention at the 27B's full-attention geometry (24 query
// heads, 4 key heads, width 256, bf16 K/V in 16-key blocks).
//
// Without arguments: on a described gfx1151, each pass width takes the
// kernel it should at a long context.
// `--check`: on the device, tree and chain passes at a context the split's
// pair form takes on gfx1151 (FlashSplitPairRule), against a double
// reference.
// `--time L [rows...] [--reps N]`: on the device, the attention a pass of
// `rows` runs at L live keys -- one row (a decode step), 2 to 8 causal rows
// (an MTP or DFlash2 chain) and 8, 15, 31 tree-masked rows (a draft tree; a
// row count may be written tN) -- with the pools as the server lays them out:
// values dimension-major, the block table at the pool rung. Prints device
// time per launch (partial and merge together), the effective rate over the
// K/V bytes one layer reads, and an FNV-1a hash of the output, so builds and
// variants compare bit for bit. Run with an empty LSE_CACHE_DIR to have the
// scheduler measure every variant a kernel offers (its stderr names them).
#include "harness.hpp"
#include "lse/backends/hrx/loomc/loom_emitter.hpp"
#include "lse/backends/hrx/loomc/loomc_compiler.hpp"
#include "lse/backends/hrx/arch_database.hpp"
#include "lse/dispatch/attention.hpp"
#include "lse/dispatch/attention_tuneconfig.h"
#include "lse/graph/kernel_primitive.hpp"
#include "lse/graph/ops.hpp"
#include "lse/graph/program.hpp"
#include "lse/kv/block.hpp"
#include "lse/kv/cache_dtype.hpp"
#include "lse/runtime/draft_tree.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <string_view>
#include <vector>

using namespace lse;
using namespace lse::graph;

namespace {

constexpr std::int64_t kQh = 24, kKvh = 4, kHd = 256, kBs = 16;

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

Array leaf(Shape shape, DType dtype = DType::kF32) {
  auto node = std::make_shared<Node>();
  node->shape = shape;
  node->dtype = dtype;
  node->materialized = true;
  return Array(node);
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

Status run(const std::vector<NodePtr>& roots) {
  auto* scheduler = default_scheduler();
  LSE_RETURN_IF_ERROR(scheduler->eval(roots, false));
  return scheduler->drain();
}

// bf16 bits of a value in [-1, 1) from a xorshift stream: fast enough for
// the half-gigabyte pools of a 128K context.
struct Bits {
  std::uint64_t s;
  std::uint16_t next() {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    const float x = static_cast<float>(static_cast<std::int32_t>(s >> 32)) * (1.0f / 2147483648.0f);
    std::uint32_t b;
    std::memcpy(&b, &x, 4);
    b += 0x7fffu + ((b >> 16) & 1u);
    return static_cast<std::uint16_t>(b >> 16);
  }
};

std::uint64_t fnv(const std::vector<float>& v) {
  std::uint64_t h = 0xcbf29ce484222325ull;
  const auto* p = reinterpret_cast<const unsigned char*>(v.data());
  for (std::size_t i = 0; i < v.size() * sizeof(float); ++i) h = (h ^ p[i]) * 0x100000001b3ull;
  return h;
}

// The pool rung the server's KV reaches at `tokens` (kv::pool_rung).
std::int64_t rung_blocks(std::int64_t tokens) {
  const auto blocks = kv::blocks_for(static_cast<std::int32_t>(tokens), kBs);
  return kv::pool_rung(blocks, kv::blocks_for(262144, kBs));
}

int time_passes(std::int64_t live, const std::vector<std::string>& passes, int reps) {
  auto* scheduler = default_scheduler();
  if (!scheduler) return 1;
  scheduler->set_mode(Scheduler::Mode::kDeviceFirst);
  scheduler->set_dialect(Dialect::kLoom);
  const std::size_t member = preferred_member();
  const auto* device = member < scheduler->devices().size()
      ? &scheduler->devices().device(member).device_info() : nullptr;
  if (!device) return 1;
  const std::int64_t blocks = rung_blocks(live + 64);
  std::printf("device %s, %u CUs; live %lld keys, pool %lld blocks (%lld keys)\n",
              device->arch.c_str(), device->compute_units, static_cast<long long>(live),
              static_cast<long long>(blocks), static_cast<long long>(blocks * kBs));
  Bits bits{0x9e3779b97f4a7c15ull};
  std::vector<std::uint16_t> kpool(static_cast<std::size_t>(blocks * kKvh * kBs * kHd)), vpool(kpool.size());
  for (auto& e : kpool) e = bits.next();
  for (auto& e : vpool) e = bits.next();
  std::vector<float> table(static_cast<std::size_t>(blocks));
  // LONG_ALIAS=N points every table slot at one of the first N blocks: the
  // same work over a cache-resident working set (a diagnostic of whether a
  // kernel waits on memory).
  const char* alias_env = std::getenv("LONG_ALIAS");
  const std::int64_t alias = alias_env ? std::max<std::int64_t>(1, std::atoll(alias_env)) : blocks;
  for (std::int64_t b = 0; b < blocks; ++b) table[static_cast<std::size_t>(b)] = static_cast<float>(b % alias);
  auto akp = filled(Shape{blocks, kKvh, kBs, kHd}, DType::kBF16, kpool);
  // Values dimension-major (kv::dimension_major_values), as the server keeps them.
  auto avp = filled(Shape{blocks, kKvh, kHd, kBs}, DType::kBF16, vpool);
  auto atable = filled(Shape{1, blocks}, DType::kF32, table);
  kpool.clear();
  kpool.shrink_to_fit();
  vpool.clear();
  vpool.shrink_to_fit();
  // K and V of one layer at `live` keys.
  const double layer_bytes = static_cast<double>(live) * kKvh * kHd * 2.0 * 2.0;
  for (const std::string& pass : passes) {
    const bool tree = !pass.empty() && pass[0] == 't';
    const auto n = std::strtoll(pass.c_str() + (tree ? 1 : 0), nullptr, 10);
    if (n < 1 || n > 64) return 2;
    const std::int32_t first = static_cast<std::int32_t>(live - n);
    std::vector<float> qv(static_cast<std::size_t>(kQh * n * kHd));
    std::mt19937 rng(static_cast<std::uint32_t>(n));
    std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
    for (auto& e : qv) e = uni(rng);
    std::vector<float> meta(tree ? kv::tree_meta_elems(1, static_cast<std::int32_t>(n))
                                 : kv::step_meta_elems(1), 0.0f);
    meta[0] = static_cast<float>(first);
    meta[1] = static_cast<float>(live);
    meta[2] = 1;
    meta[3] = static_cast<float>(first);
    meta[4] = static_cast<float>(live);
    meta[static_cast<std::size_t>(kv::step_meta_stride_index(1))] = static_cast<float>(atable.shape().dim(1));
    if (tree) {
      const runtime::DraftTree t = make_tree(static_cast<std::uint32_t>(n - 1), 11);
      if (t.rows() != static_cast<std::uint32_t>(n)) return 3;
      for (std::uint32_t r = 0; r < t.rows(); ++r)
        for (std::uint32_t c = 0; c < t.rows(); ++c)
          meta[static_cast<std::size_t>(kv::tree_mask_offset(1)) + r * n + c] = t.sees(r, c) ? 1.0f : 0.0f;
    }
    auto aqq = filled(Shape{1, kQh, n, kHd}, DType::kF32, qv);
    auto ameta = filled(Shape{static_cast<std::int64_t>(meta.size())}, DType::kF32, meta);
    const auto make = [&] {
      return sdpa_paged(aqq, akp, avp, 0.0625f, tree ? MaskKind::kTree : MaskKind::kCausal, 0,
                        ameta, atable, static_cast<int>(kBs), device, kv::CacheDType::kBF16);
    };
    Array first_out = make();
    std::string kernels(first_out.node()->prim->name());
    for (const auto& in : first_out.node()->inputs)
      if (in->prim) kernels = std::string(in->prim->name()) + in->shape.to_string() + " + " + kernels;
    LSE_EXPECT_OK(run({first_out.node()}));
    std::vector<float> out(first_out.shape().elem_count());
    LSE_EXPECT_OK(first_out.to_host(out.data(), out.size() * sizeof(float)));
    // Warm, then timed.
    double us = 0;
    for (int round = 0; round < 2; ++round) {
      std::vector<Array> keep;
      std::vector<NodePtr> roots;
      const int count = round == 0 ? 3 : reps;
      for (int i = 0; i < count; ++i) {
        keep.push_back(make());
        roots.push_back(keep.back().node());
      }
      const auto t0 = std::chrono::steady_clock::now();
      LSE_EXPECT_OK(run(roots));
      us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / count;
    }
    // The merge alone, over the partial records the first pass left.
    double merge_us = 0;
    if (!first_out.node()->inputs.empty()) {
      const Array records(first_out.node()->inputs[0]);
      for (int round = 0; round < 2; ++round) {
        std::vector<Array> keep;
        std::vector<NodePtr> roots;
        const int count = round == 0 ? 3 : reps;
        for (int i = 0; i < count; ++i) {
          auto merged = custom(first_out.node()->prim->name(), {records, ameta});
          LSE_EXPECT_OK(merged.status());
          if (!merged.ok()) break;
          merged->node()->attrs = first_out.node()->attrs;
          merged->node()->iattrs = first_out.node()->iattrs;
          keep.push_back(*merged);
          roots.push_back(keep.back().node());
        }
        const auto t0 = std::chrono::steady_clock::now();
        LSE_EXPECT_OK(run(roots));
        merge_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / count;
      }
    }
    std::printf("%-5s rows %2lld at %lld: %9.1f us (merge %6.1f)  %6.1f GB/s  hash %016llx  [%s]\n",
                tree ? "tree" : n == 1 ? "dec" : "chain", static_cast<long long>(n),
                static_cast<long long>(live), us, merge_us, layer_bytes / us / 1e3,
                static_cast<unsigned long long>(fnv(out)), kernels.c_str());
    std::fflush(stdout);
  }
  return lse::test::Registry::get().failures ? 1 : 0;
}

float from_bf16(std::uint16_t b) {
  const std::uint32_t bits = static_cast<std::uint32_t>(b) << 16;
  float x;
  std::memcpy(&x, &bits, 4);
  return x;
}

int check_passes() {
  auto* scheduler = default_scheduler();
  if (!scheduler) return 1;
  scheduler->set_mode(Scheduler::Mode::kDeviceFirst);
  scheduler->set_dialect(Dialect::kLoom);
  const std::size_t member = preferred_member();
  const auto* device = member < scheduler->devices().size()
      ? &scheduler->devices().device(member).device_info() : nullptr;
  if (!device) return 1;
  const std::int64_t live = 5000, blocks = rung_blocks(live + 64);
  Bits bits{0x2545f4914f6cdd1dull};
  std::vector<std::uint16_t> kpool(static_cast<std::size_t>(blocks * kKvh * kBs * kHd)), vpool(kpool.size());
  for (auto& e : kpool) e = bits.next();
  for (auto& e : vpool) e = bits.next();
  std::vector<float> table(static_cast<std::size_t>(blocks));
  for (std::int64_t b = 0; b < blocks; ++b) table[static_cast<std::size_t>(b)] = static_cast<float>(blocks - 1 - b);
  auto akp = filled(Shape{blocks, kKvh, kBs, kHd}, DType::kBF16, kpool);
  auto avp = filled(Shape{blocks, kKvh, kHd, kBs}, DType::kBF16, vpool);
  auto atable = filled(Shape{1, blocks}, DType::kF32, table);
  const auto key_at = [&](std::int64_t pos, std::int64_t h, std::int64_t d) {
    const auto blk = static_cast<std::int64_t>(table[static_cast<std::size_t>(pos / kBs)]);
    return from_bf16(kpool[static_cast<std::size_t>(((blk * kKvh + h) * kBs + pos % kBs) * kHd + d)]);
  };
  const auto value_at = [&](std::int64_t pos, std::int64_t h, std::int64_t d) {
    const auto blk = static_cast<std::int64_t>(table[static_cast<std::size_t>(pos / kBs)]);
    return from_bf16(vpool[static_cast<std::size_t>(((blk * kKvh + h) * kHd + d) * kBs + pos % kBs)]);
  };
  for (const std::string pass : {"t15", "t31", "t8", "8", "4"}) {
    const bool tree = pass[0] == 't';
    const auto n = std::strtoll(pass.c_str() + (tree ? 1 : 0), nullptr, 10);
    const std::int64_t first = live - n;
    const runtime::DraftTree t = make_tree(static_cast<std::uint32_t>(n - 1), 13);
    std::vector<float> qv(static_cast<std::size_t>(kQh * n * kHd));
    std::mt19937 rng(static_cast<std::uint32_t>(n + 7));
    std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
    for (auto& e : qv) e = uni(rng);
    std::vector<float> meta(tree ? kv::tree_meta_elems(1, static_cast<std::int32_t>(n))
                                 : kv::step_meta_elems(1), 0.0f);
    meta[0] = static_cast<float>(first);
    meta[1] = static_cast<float>(live);
    meta[2] = 1;
    meta[3] = static_cast<float>(first);
    meta[4] = static_cast<float>(live);
    meta[static_cast<std::size_t>(kv::step_meta_stride_index(1))] = static_cast<float>(atable.shape().dim(1));
    if (tree)
      for (std::uint32_t r = 0; r < t.rows(); ++r)
        for (std::uint32_t c = 0; c < t.rows(); ++c)
          meta[static_cast<std::size_t>(kv::tree_mask_offset(1)) + r * n + c] = t.sees(r, c) ? 1.0f : 0.0f;
    auto aqq = filled(Shape{1, kQh, n, kHd}, DType::kF32, qv);
    auto ameta = filled(Shape{static_cast<std::int64_t>(meta.size())}, DType::kF32, meta);
    auto o = sdpa_paged(aqq, akp, avp, 0.0625f, tree ? MaskKind::kTree : MaskKind::kCausal, 0,
                        ameta, atable, static_cast<int>(kBs), device, kv::CacheDType::kBF16);
    LSE_EXPECT_OK(run({o.node()}));
    std::vector<float> out(o.shape().elem_count());
    LSE_EXPECT_OK(o.to_host(out.data(), out.size() * sizeof(float)));
    double worst = 0;
    for (std::int64_t h = 0; h < kQh; ++h)
      for (std::int64_t i = 0; i < n; ++i) {
        std::vector<double> score(static_cast<std::size_t>(live), -INFINITY);
        double top = -INFINITY;
        for (std::int64_t j = 0; j < live; ++j) {
          const bool seen = j < first ? true
              : tree ? t.sees(static_cast<std::uint32_t>(i), static_cast<std::uint32_t>(j - first))
                     : j <= first + i;
          if (!seen) continue;
          double acc = 0;
          for (std::int64_t d = 0; d < kHd; ++d)
            acc += static_cast<double>(qv[static_cast<std::size_t>((h * n + i) * kHd + d)]) * key_at(j, h / (kQh / kKvh), d);
          score[static_cast<std::size_t>(j)] = acc * 0.0625;
          top = std::max(top, score[static_cast<std::size_t>(j)]);
        }
        double denom = 0;
        for (auto& sc : score) denom += (sc = std::exp(sc - top));
        for (std::int64_t d = 0; d < kHd; ++d) {
          double acc = 0;
          for (std::int64_t j = 0; j < live; ++j)
            if (score[static_cast<std::size_t>(j)] > 0)
              acc += score[static_cast<std::size_t>(j)] * value_at(j, h / (kQh / kKvh), d);
          worst = std::max(worst, std::abs(acc / denom - out[static_cast<std::size_t>((h * n + i) * kHd + d)]));
        }
      }
    std::printf("%s %lld rows at %lld keys (pool %lld): worst diff %.3g vs a double reference\n",
                tree ? "tree" : "chain", static_cast<long long>(n), static_cast<long long>(live),
                static_cast<long long>(blocks * kBs), worst);
    LSE_EXPECT(worst < 2e-3);
  }
  return lse::test::Registry::get().failures ? 1 : 0;
}

backend::DeviceInfo described_gfx1151(backend::AmdDeviceInfo& amd, const char* arch = "gfx1151",
                                      std::uint32_t compute_units = 40) {
  backend::DeviceInfo gpu;
  gpu.arch = arch;
  gpu.wavefront_size = 32;
  gpu.max_threads_per_workgroup = 1024;
  gpu.lds_bytes_per_workgroup = 65536;
  gpu.compute_units = compute_units;
  backend::apply_arch_defaults(gpu, amd);
  gpu.extension_id = backend::AmdDeviceInfo::kExtensionId;
  gpu.extension = &amd;
  return gpu;
}

}  // namespace

LSE_TEST(long_context_passes_take_the_split_kernels_on_gfx1151) {
  backend::AmdDeviceInfo amd;
  const auto gpu = described_gfx1151(amd);
  for (const std::int64_t live : {65536, 131072}) {
    const std::int64_t blocks = rung_blocks(live + 64);
    for (const std::int64_t n : {1, 2, 4, 8, 15, 31}) {
      const auto mask = n > 8 ? MaskKind::kTree : MaskKind::kCausal;
      auto q = leaf({1, kQh, n, kHd});
      auto k = leaf({blocks, kKvh, kBs, kHd}, DType::kBF16), v = leaf({blocks, kKvh, kHd, kBs}, DType::kBF16);
      auto meta = leaf({mask == MaskKind::kTree ? kv::tree_meta_elems(1, static_cast<std::int32_t>(n))
                                                : kv::step_meta_elems(1)});
      auto table = leaf({1, blocks});
      auto o = sdpa_paged(q, k, v, 0.0625f, mask, 0, meta, table, static_cast<int>(kBs), &gpu,
                          kv::CacheDType::kBF16);
      LSE_EXPECT(o.node()->prim->name() == "attention.split_merge128.wg128c2.v1");
      const std::string_view partial = o.node()->inputs[0]->prim->name();
      LSE_EXPECT(partial == (n == 1 ? "attention.split_partial128.wg128c2.v1"
                                    : "attention.flash_split.wmma16.v1"));
    }
  }
}

// The kernels a pass of `n` rows (a tree when `tree`) emits at a cache of
// `keys`, on `gpu`: their sources, by group, with the self-export name
// canonical.
std::vector<std::string> pass_sources(const backend::DeviceInfo& gpu, std::int64_t n, bool tree,
                                      std::int64_t keys, std::vector<std::string>* names = nullptr) {
  const std::int64_t blocks = rung_blocks(keys);
  auto q = leaf({1, kQh, n, kHd});
  auto k = leaf({blocks, kKvh, kBs, kHd}, DType::kBF16), v = leaf({blocks, kKvh, kHd, kBs}, DType::kBF16);
  auto meta = leaf({tree ? kv::tree_meta_elems(1, static_cast<std::int32_t>(n)) : kv::step_meta_elems(1)});
  auto table = leaf({1, blocks});
  // A pass of more than 64 rows is a prompt chunk, here with FlashPrefill's
  // pool and selector when FLASHPREFILL is set.
  ops::SparseAttentionPhase sparse;
  if (n > 64 && std::getenv("FLASHPREFILL")) sparse = {false, 0.1f, true};
  auto o = sdpa_paged(q, k, v, 0.0625f, tree ? MaskKind::kTree : MaskKind::kCausal, 0, meta, table,
                      static_cast<int>(kBs), &gpu, kv::CacheDType::kBF16, sparse);
  const NodePtr roots[]{o.node()};
  backend::LoomEmitter emitter;
  std::vector<std::string> out;
  for (const auto& group : Partitioner::partition(roots, &gpu)) {
    auto emitted = emitter.emit(group, gpu);
    LSE_EXPECT_OK(emitted.status());
    if (!emitted.ok()) continue;
    std::string text = emitted->source;
    for (std::size_t at = text.find(emitted->entry_name); at != std::string::npos;
         at = text.find(emitted->entry_name, at))
      text.replace(at, emitted->entry_name.size(), "self");
    out.push_back(std::move(text));
    if (names) names->push_back(std::string(group.nodes.back()->prim ? group.nodes.back()->prim->name() : "?"));
  }
  return out;
}

int emit_diff(std::int64_t n, bool tree, std::int64_t a, std::int64_t b) {
  backend::AmdDeviceInfo amd;
  const auto gpu = described_gfx1151(amd);
  std::vector<std::string> names;
  const auto x = pass_sources(gpu, n, tree, a, &names), y = pass_sources(gpu, n, tree, b);
  for (std::size_t g = 0; g < x.size() && g < y.size(); ++g) {
    std::printf("group %zu %s: %s\n", g, names[g].c_str(), x[g] == y[g] ? "same" : "differs");
    if (x[g] == y[g]) continue;
    // The first lines that differ.
    std::size_t i = 0, j = 0, shown = 0;
    while (shown < 12 && i < x[g].size() && j < y[g].size()) {
      const auto ei = x[g].find('\n', i), ej = y[g].find('\n', j);
      const std::string li = x[g].substr(i, ei - i), lj = y[g].substr(j, ej - j);
      if (li != lj) {
        std::printf("  - %s\n  + %s\n", li.substr(0, 200).c_str(), lj.substr(0, 200).c_str());
        ++shown;
      }
      if (ei == std::string::npos || ej == std::string::npos) break;
      i = ei + 1;
      j = ej + 1;
    }
  }
  return 0;
}

// The cache's size is a run-time value of every attention kernel: from the
// last code-shape bucket on (dispatch::attention_kv_buckets), a pass at any
// rung emits the same text, so a session that grows compiles nothing.
LSE_TEST(attention_kernel_text_is_the_same_at_every_cache_rung) {
  for (const auto& [arch, units] : {std::pair{"gfx1151", 40u}, std::pair{"gfx1201", 64u}}) {
    backend::AmdDeviceInfo amd;
    const auto gpu = described_gfx1151(amd, arch, units);
    std::int64_t from = 4096;
    for (const std::int64_t keys : dispatch::attention_kv_buckets(gpu)) from = std::max(from, keys);
    for (const char* pass : {"1", "2", "4", "8", "t8", "t15", "t31"}) {
      const bool tree = pass[0] == 't';
      const std::int64_t n = std::atoll(pass + (tree ? 1 : 0));
      std::vector<std::string> names;
      const auto base = pass_sources(gpu, n, tree, from, &names);
      LSE_EXPECT(!base.empty());
      for (const std::int64_t keys : {from * 2, std::int64_t{65536}, std::int64_t{131072}, std::int64_t{262144}}) {
        const auto at = pass_sources(gpu, n, tree, keys);
        LSE_EXPECT_EQ(at.size(), base.size());
        for (std::size_t g = 0; g < at.size() && g < base.size(); ++g) {
          if (at[g] != base[g])
            std::fprintf(stderr, "%s pass %s group %zu (%s) differs at %lld keys from %lld\n", arch, pass,
                         g, names[g].c_str(), static_cast<long long>(keys), static_cast<long long>(from));
          LSE_EXPECT(at[g] == base[g]);
        }
      }
    }
  }
}

namespace {

// `--grow`: on the device, every pass width once at 4K keys and once at each
// code-shape bucket (what a server prepares at load), then at 16K to 128K
// keys with the pools at each rung. Counts the kernels compiled or loaded
// after the first two: a session that grows should need none.
int grow_passes() {
  auto* scheduler = default_scheduler();
  if (!scheduler) return 1;
  scheduler->set_mode(Scheduler::Mode::kDeviceFirst);
  scheduler->set_dialect(Dialect::kLoom);
  const std::size_t member = preferred_member();
  const auto* device = member < scheduler->devices().size()
      ? &scheduler->devices().device(member).device_info() : nullptr;
  if (!device) return 1;
  std::vector<std::int64_t> lives{4096};
  for (const std::int64_t keys : dispatch::attention_kv_buckets(*device)) lives.push_back(keys);
  const std::size_t prepared = lives.size();
  for (const std::int64_t keys : {16384, 32768, 65536, 131072}) lives.push_back(keys);
  Scheduler::JitStats after_load{};
  for (std::size_t li = 0; li < lives.size(); ++li) {
    const std::int64_t live = lives[li];
    const std::int64_t blocks = rung_blocks(live + 64);
    std::vector<std::uint16_t> pool(static_cast<std::size_t>(blocks * kKvh * kBs * kHd), 0x3c00);
    std::vector<float> table(static_cast<std::size_t>(blocks));
    for (std::int64_t b = 0; b < blocks; ++b) table[static_cast<std::size_t>(b)] = static_cast<float>(b);
    auto akp = filled(Shape{blocks, kKvh, kBs, kHd}, DType::kBF16, pool);
    auto avp = filled(Shape{blocks, kKvh, kHd, kBs}, DType::kBF16, pool);
    auto atable = filled(Shape{1, blocks}, DType::kF32, table);
    pool = {};
    for (const char* pass : {"1", "2", "3", "4", "5", "6", "7", "8", "t8", "t15", "t31"}) {
      const bool tree = pass[0] == 't';
      const std::int64_t n = std::atoll(pass + (tree ? 1 : 0));
      std::vector<float> meta(tree ? kv::tree_meta_elems(1, static_cast<std::int32_t>(n))
                                   : kv::step_meta_elems(1), 0.0f);
      meta[0] = meta[3] = static_cast<float>(live - n);
      meta[1] = meta[4] = static_cast<float>(live);
      meta[2] = 1;
      meta[static_cast<std::size_t>(kv::step_meta_stride_index(1))] = static_cast<float>(blocks);
      if (tree)  // a star: each row sees the root and itself
        for (std::int64_t r = 0; r < n; ++r) {
          meta[static_cast<std::size_t>(kv::tree_mask_offset(1) + r * n)] = 1.0f;
          meta[static_cast<std::size_t>(kv::tree_mask_offset(1) + r * n + r)] = 1.0f;
        }
      auto q = filled(Shape{1, kQh, n, kHd}, DType::kF32, std::vector<float>(static_cast<std::size_t>(kQh * n * kHd), 0.5f));
      auto ameta = filled(Shape{static_cast<std::int64_t>(meta.size())}, DType::kF32, meta);
      auto out = sdpa_paged(q, akp, avp, 0.0625f, tree ? MaskKind::kTree : MaskKind::kCausal, 0,
                            ameta, atable, static_cast<int>(kBs), device, kv::CacheDType::kBF16);
      LSE_EXPECT_OK(run({out.node()}));
    }
    const auto jit = scheduler->jit_stats();
    std::printf("at %6lld keys (pool %lld blocks): %llu compiled, %llu loaded, %llu variant trials\n",
                static_cast<long long>(live), static_cast<long long>(blocks),
                static_cast<unsigned long long>(jit.compiles), static_cast<unsigned long long>(jit.disk_hits),
                static_cast<unsigned long long>(jit.variant_trials));
    if (li + 1 == prepared) after_load = jit;
  }
  const auto jit = scheduler->jit_stats();
  const auto grown = (jit.compiles - after_load.compiles) + (jit.disk_hits - after_load.disk_hits);
  std::printf("kernels compiled or loaded after load: %llu\n", static_cast<unsigned long long>(grown));
  LSE_EXPECT_EQ(grown, 0u);
  return lse::test::Registry::get().failures ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 2 && std::string_view(argv[1]) == "--grow") return grow_passes();
  if (argc == 5 && std::string_view(argv[1]) == "--compile") {
    // `--compile ARCH [t]N KEYS`: each kernel of the pass compiled for a
    // described device, without one.
    const bool tree = argv[3][0] == 't';
    backend::AmdDeviceInfo amd;
    const auto gpu = described_gfx1151(amd, argv[2], std::string_view(argv[2]) == "gfx1201" ? 64u : 40u);
    std::vector<std::string> names;
    const auto sources = pass_sources(gpu, std::atoll(argv[3] + (tree ? 1 : 0)), tree, std::atoll(argv[4]), &names);
    backend::LoomcCompiler compiler;
    int bad = 0;
    for (std::size_t g = 0; g < sources.size(); ++g) {
      auto code = compiler.compile(sources[g], gpu.arch);
      std::printf("%s: %s\n", names[g].c_str(), code.ok() ? "compiled" : code.status().to_string().substr(0, 600).c_str());
      if (!code.ok()) {
        ++bad;
        const std::string path = "/tmp/lc-" + std::to_string(g) + ".loom";
        if (FILE* f = std::fopen(path.c_str(), "w")) { std::fputs(sources[g].c_str(), f); std::fclose(f); }
      }
    }
    return bad;
  }
  if (argc == 4 && std::string_view(argv[1]) == "--emit") {
    const bool tree = argv[2][0] == 't';
    backend::AmdDeviceInfo amd;
    const auto gpu = described_gfx1151(amd);
    std::vector<std::string> names;
    const auto sources = pass_sources(gpu, std::atoll(argv[2] + (tree ? 1 : 0)), tree, std::atoll(argv[3]), &names);
    for (std::size_t g = 0; g < sources.size(); ++g)
      std::printf("// group %zu %s\n%s\n", g, names[g].c_str(), sources[g].c_str());
    return 0;
  }
  if (argc == 5 && std::string_view(argv[1]) == "--emit-diff") {
    const bool tree = argv[2][0] == 't';
    return emit_diff(std::atoll(argv[2] + (tree ? 1 : 0)), tree, std::atoll(argv[3]), std::atoll(argv[4]));
  }
  if (argc == 2 && std::string_view(argv[1]) == "--check") return check_passes();
  if (argc >= 3 && std::string_view(argv[1]) == "--time") {
    const auto live = std::strtoll(argv[2], nullptr, 10);
    std::vector<std::string> passes;
    int reps = 10;
    for (int i = 3; i < argc; ++i) {
      if (std::string_view(argv[i]) == "--reps" && i + 1 < argc) reps = std::atoi(argv[++i]);
      else passes.emplace_back(argv[i]);
    }
    if (passes.empty()) passes = {"1", "2", "3", "4", "5", "6", "7", "8", "t8", "t15", "t31"};
    return time_passes(live, passes, reps);
  }
  return lse::test::run_all();
}
