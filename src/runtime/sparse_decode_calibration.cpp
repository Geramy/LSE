#include "lse/runtime/sparse_decode_calibration.hpp"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <vector>

#include "lse/dispatch/roofline.hpp"
#include "lse/dispatch/sparse_decode_tuneconfig.h"
#include "lse/graph/graph.hpp"
#include "lse/graph/ops.hpp"
#include "lse/graph/program.hpp"
#include "lse/kv/block.hpp"

namespace lse::runtime {
namespace {

namespace sd = dispatch::sparse_decode;
using graph::Array;
using graph::NodePtr;
constexpr int kFormat = 3;
// Contexts the exact and selection costs are fitted between (past every
// last-level cache), and the replays each timing averages.
constexpr std::int64_t kLong = 131072, kShort = 65536;
constexpr int kReplays = 16;

std::string key_of(const SparseDecodeGeometry& g) {
  return std::to_string(g.query_heads) + "x" + std::to_string(g.key_heads) + "x" +
         std::to_string(g.head_dim) + "b" + std::to_string(g.budget);
}

bool read_cached(const std::string& path, const std::string& key,
                 dispatch::SparseDecodeCalibration* out) {
  std::ifstream in(path);
  int format = 0;
  std::string k;
  if (!(in >> format >> k) || format != kFormat || k != key) return false;
  return static_cast<bool>(in >> out->exact_efficiency >> out->exact_fixed_ns >> out->score_ns_per_row_page >>
                           out->score_fixed_ns >> out->select_fixed_ns >> out->select_ns_per_page >>
                           out->tail_ns_per_row_page >> out->tail_fixed_ns) &&
         out->valid();
}

void write_cached(const std::string& path, const std::string& key,
                  const dispatch::SparseDecodeCalibration& c) {
  std::error_code ec;
  std::filesystem::create_directories(std::filesystem::path(path).parent_path(), ec);
  const std::string tmp = path + ".tmp";
  {
    std::ofstream out(tmp);
    out.precision(9);
    out << kFormat << ' ' << key << ' ' << c.exact_efficiency << ' ' << c.exact_fixed_ns << ' '
        << c.score_ns_per_row_page << ' '
        << c.score_fixed_ns << ' ' << c.select_fixed_ns << ' ' << c.select_ns_per_page << ' '
        << c.tail_ns_per_row_page << ' ' << c.tail_fixed_ns << '\n';
  }
  std::filesystem::rename(tmp, path, ec);
}

struct Bench {
  graph::Scheduler& sched;
  const backend::DeviceInfo& device;

  Result<Array> zeros(Shape shape, DType dtype) {
    const std::size_t bytes = dtype_storage_bytes(dtype, shape.elem_count());
    LSE_ASSIGN_OR(auto buf, sched.backend().allocate(bytes, backend::MemoryClass::kDevice));
    const std::vector<std::byte> host(bytes, std::byte{0});
    LSE_RETURN_IF_ERROR(sched.backend().copy(buf, host.data(), bytes));
    return Array::from_buffer(std::move(buf), shape, dtype);
  }
  Result<Array> values(Shape shape, const std::vector<float>& data) {
    LSE_ASSIGN_OR(auto buf, sched.backend().allocate(data.size() * 4, backend::MemoryClass::kDevice));
    LSE_RETURN_IF_ERROR(sched.backend().copy(buf, data.data(), data.size() * 4));
    return Array::from_buffer(std::move(buf), shape, DType::kF32);
  }
  // Mean ns of one replay of the program that computes `roots`.
  Result<double> time(const std::vector<NodePtr>& roots) {
    graph::Program program;
    LSE_RETURN_IF_ERROR(sched.eval(roots, false, &program));
    LSE_RETURN_IF_ERROR(sched.drain());
    if (sched.last_trace().host_groups != 0)
      return LSE_ERROR(kUnimplemented, "a sparse decode calibration kernel ran on the host");
    for (int warm = 0; warm < 4; ++warm) {
      program.reset_compute();
      LSE_RETURN_IF_ERROR(sched.eval(roots, false, &program));
    }
    LSE_RETURN_IF_ERROR(sched.drain());
    // The fastest of three batches: what the kernels cost, not what a
    // clock change or another process cost during one batch.
    double fastest = 0.0;
    for (int batch = 0; batch < 3; ++batch) {
      const auto t0 = std::chrono::steady_clock::now();
      for (int r = 0; r < kReplays; ++r) {
        program.reset_compute();
        LSE_RETURN_IF_ERROR(sched.eval(roots, false, &program));
      }
      LSE_RETURN_IF_ERROR(sched.drain());
      const double ns = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count() / kReplays;
      if (batch == 0 || ns < fastest) fastest = ns;
    }
    program.destroy();
    return fastest;
  }
};

}  // namespace

Result<dispatch::SparseDecodeCalibration> calibrate_sparse_decode(
    const backend::DeviceInfo& device, const SparseDecodeGeometry& g, const std::string& path) {
  dispatch::SparseDecodeCalibration cal;
  const std::string key = key_of(g);
  if (read_cached(path, key, &cal)) {
    dispatch::publish_sparse_decode_calibration(device, cal);
    return cal;
  }
  const auto roof = dispatch::roofline(&device);
  if (!roof) return LSE_ERROR(kNotFound, "no measured roofline for ", device.arch);
  graph::Scheduler* sched = graph::default_scheduler();
  if (sched == nullptr) return LSE_ERROR(kInternal, "no scheduler to calibrate sparse decode on");
  Bench bench{*sched, device};
  const std::int64_t kvh = g.key_heads, hd = g.head_dim, heads = g.query_heads;
  const std::int64_t pages = kLong / kv::kBlockSize, blocks = pages + 8;
  const auto storage = kv::CacheDType::kBF16;
  LSE_ASSIGN_OR(Array keys, bench.zeros(Shape{blocks, kvh, kv::kBlockSize, hd}, DType::kBF16));
  // The engine's value pool is dimension-major for 16-bit K/V.
  LSE_ASSIGN_OR(Array vals, bench.zeros(Shape{blocks, kvh, hd, kv::kBlockSize}, DType::kBF16));
  LSE_ASSIGN_OR(Array summary, bench.zeros(Shape{blocks, kvh, sd::kSummaryRows, hd}, DType::kBF16));
  // A block table as long as the context, as the engine's KV rung is.
  auto table_for = [&](std::int64_t live) {
    const std::int64_t n = live / kv::kBlockSize;
    std::vector<float> table(static_cast<std::size_t>(n));
    for (std::int64_t p = 0; p < n; ++p) table[static_cast<std::size_t>(p)] = static_cast<float>(p);
    return bench.values(Shape{1, n}, table);
  };
  const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
  auto meta_for = [&](std::int64_t live, std::int64_t rows) {
    const float first = static_cast<float>(live - rows), l = static_cast<float>(live);
    return bench.values(Shape{5}, {first, l, 1.0f, first, l});
  };
  auto query = [&](std::int64_t rows) {
    return bench.values(Shape{1, heads, rows, hd},
                        std::vector<float>(static_cast<std::size_t>(heads * rows * hd), 0.01f));
  };
  // Exact split attention at both contexts.
  double exact[2] = {0, 0};
  const std::int64_t lives[2] = {kShort, kLong};
  // The long exact pass first, untimed, to bring the clocks up: the load
  // before calibration leaves the device idle.
  for (int i : {1, 0, 1}) {
    LSE_ASSIGN_OR(Array q, query(1));
    LSE_ASSIGN_OR(Array meta, meta_for(lives[i], 1));
    LSE_ASSIGN_OR(Array tab, table_for(lives[i]));
    Array o = graph::sdpa_paged(q, keys, vals, scale, graph::MaskKind::kCausal, 0, meta, tab,
                                kv::kBlockSize, &device, storage);
    LSE_ASSIGN_OR(exact[i], bench.time({o.node()}));
  }
  // Score (1 and 8 rows) and score + selection, at both contexts for 1 row.
  struct Stages { NodePtr scores, selected, tail; };
  auto stages = [&](std::int64_t live, std::int64_t rows) -> Result<Stages> {
    LSE_ASSIGN_OR(Array q, query(rows));
    LSE_ASSIGN_OR(Array meta, meta_for(live, rows));
    LSE_ASSIGN_OR(Array tab, table_for(live));
    LSE_ASSIGN_OR(Array o, graph::sparse_decode_attention(q, keys, vals, scale, graph::MaskKind::kCausal,
                                                          meta, tab, kv::kBlockSize, &device, storage,
                                                          summary, g.budget, false));
    // merge <- concat(partial, tail), or the partial alone without the
    // correction; the partial's fifth input is the selection.
    const NodePtr records = o.node()->inputs[0];
    const bool tail = records->inputs.size() == 2;
    const NodePtr partial = tail ? records->inputs[0] : records;
    // The partial reads the selection's block ids through a slice.
    const NodePtr selected = partial->inputs[4]->inputs[0];
    return Stages{selected->inputs[0], selected, tail ? records->inputs[1] : selected};
  };
  double score1[2] = {0, 0}, both1[2] = {0, 0}, score8 = 0, tail1 = 0, select8 = 0, tail8 = 0;
  for (int i = 0; i < 2; ++i) {
    LSE_ASSIGN_OR(auto s, stages(lives[i], 1));
    LSE_ASSIGN_OR(score1[i], bench.time({s.scores}));
    LSE_ASSIGN_OR(auto t, stages(lives[i], 1));
    LSE_ASSIGN_OR(both1[i], bench.time({t.selected}));
  }
  {
    LSE_ASSIGN_OR(auto s, stages(kLong, 8));
    LSE_ASSIGN_OR(score8, bench.time({s.scores}));
    LSE_ASSIGN_OR(auto t, stages(kLong, 8));
    LSE_ASSIGN_OR(select8, bench.time({t.selected}));
    LSE_ASSIGN_OR(auto u, stages(kLong, 8));
    LSE_ASSIGN_OR(tail8, bench.time({u.tail}));
    tail8 -= select8;
    LSE_ASSIGN_OR(auto v, stages(kLong, 1));
    LSE_ASSIGN_OR(tail1, bench.time({v.tail}));
    tail1 -= both1[1];
  }
  // What one replay costs beyond its kernels: the shifted-descriptor kernel
  // alone, less one launch. Taken out of every fixed term below.
  double overhead = 0.0;
  {
    LSE_ASSIGN_OR(Array meta, meta_for(kLong, 1));
    LSE_ASSIGN_OR(Array tab, table_for(kLong));
    LSE_ASSIGN_OR(Array q, query(1));
    LSE_ASSIGN_OR(Array o, graph::sparse_decode_attention(q, keys, vals, scale, graph::MaskKind::kCausal,
                                                          meta, tab, kv::kBlockSize, &device, storage,
                                                          summary, g.budget, false));
    const NodePtr records = o.node()->inputs[0];
    const NodePtr partial = records->inputs.size() == 2 ? records->inputs[0] : records;
    LSE_ASSIGN_OR(const double t, bench.time({partial->inputs[3]}));
    overhead = std::max(0.0, t - roof->launch_ns);
  }
  const double kv_token = 2.0 * static_cast<double>(kvh * hd) * 2.0;
  const double slope = (exact[1] - exact[0]) / static_cast<double>(kLong - kShort);
  cal.exact_efficiency = slope > 0.0 ? kv_token * 1e9 / (slope * roof->stream_bytes_per_s) : 0.0;
  cal.exact_fixed_ns = std::max(roof->launch_ns, exact[0] - overhead - slope * static_cast<double>(kShort));
  const double group = static_cast<double>(heads / kvh);
  const double far_pages = static_cast<double>(pages);
  cal.score_ns_per_row_page = std::max(0.0, (score8 - score1[1]) / (far_pages * kvh * group * 7.0));
  cal.score_fixed_ns = std::max(0.0, score1[1] - overhead - far_pages * kvh * group * cal.score_ns_per_row_page);
  const double select_long = both1[1] - score1[1], select_short = both1[0] - score1[0];
  cal.select_ns_per_page = std::max(0.0, (select_long - select_short) /
                                             static_cast<double>((kLong - kShort) / kv::kBlockSize));
  cal.select_fixed_ns = std::max(1.0, select_long - cal.select_ns_per_page * far_pages);
  cal.tail_ns_per_row_page = std::max(0.0, (tail8 - tail1) / (far_pages * kvh * group * 7.0));
  cal.tail_fixed_ns = std::max(0.0, tail1 - far_pages * kvh * group * cal.tail_ns_per_row_page);
  std::fprintf(stderr,
               "lse-server: sparse decode calibration %s: exact attention %.1f/%.1f us at %lldK/%lldK "
               "(%.0f%% of stream), score %.1f us (1 row) %.1f us (8 rows) at %lldK, "
               "selection %.1f/%.1f us, mean correction %.1f us (1 row) %.1f us (8 rows), replay overhead %.1f us\n",
               device.arch.c_str(), exact[0] / 1e3, exact[1] / 1e3, static_cast<long long>(kShort / 1024),
               static_cast<long long>(kLong / 1024), cal.exact_efficiency * 100.0, score1[1] / 1e3,
               score8 / 1e3, static_cast<long long>(kLong / 1024), select_short / 1e3, select_long / 1e3,
               tail1 / 1e3, tail8 / 1e3, overhead / 1e3);
  if (!cal.valid())
    return LSE_ERROR(kInternal, "sparse decode calibration measured no usable costs");
  write_cached(path, key, cal);
  dispatch::publish_sparse_decode_calibration(device, cal);
  return cal;
}

}  // namespace lse::runtime
