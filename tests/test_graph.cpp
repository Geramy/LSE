#include "lse/graph/graph.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <sstream>
#include <string>
#include <system_error>
#include <unordered_map>
#include <vector>

#include <unistd.h>

#include "harness.hpp"
#include "lse/graph/interpreter.hpp"
#include "lse/graph/ops.hpp"
#include "lse/graph/program.hpp"
#include "lse/graph/sharding.hpp"
#include "lse/backends/hrx/device_info.hpp"
#include "lse/graph/stream_plan.hpp"
#include "lse/graph/workgroup.hpp"
#include "lse/quant/group_affine.hpp"

using namespace lse;
using namespace lse::graph;

namespace {

Array host_array(std::vector<float> values, Shape shape) {
  Array a = Array::zeros(shape, DType::kF32);
  // Materialize, then write through the buffer the interpreter allocated.
  (void)a.eval();
  for (std::size_t i = 0; i < values.size(); ++i) {
    interpreter::store_element(*a.node(), i, values[i]);
  }
  return a;
}

std::vector<float> drain(Array& a) {
  (void)a.eval();
  std::vector<float> out(a.shape().elem_count());
  for (std::size_t i = 0; i < out.size(); ++i) {
    out[i] = interpreter::load_element(*a.node(), i);
  }
  return out;
}

}  // namespace

LSE_TEST(ops_record_without_computing) {
  Array x = Array::full(Shape{4}, DType::kF32, 2.0f);
  Array y = x * x + x;
  // Nothing has run: the node exists but holds no buffer.
  LSE_EXPECT(!y.node()->materialized);
  LSE_EXPECT(y.node()->kind == OpKind::kAdd);
  LSE_EXPECT(y.node()->inputs.size() == 2);
}

LSE_TEST(eval_computes_elementwise_chain) {
  Array x = Array::full(Shape{4}, DType::kF32, 3.0f);
  Array y = x * x - x;
  const auto out = drain(y);
  LSE_EXPECT_EQ(out.size(), 4u);
  for (float v : out) LSE_EXPECT_NEAR(v, 6.0, 1e-6);
  LSE_EXPECT(y.node()->materialized);
}

LSE_TEST(broadcast_binary_matches_numpy_rules) {
  Array a = host_array({1, 2, 3, 4, 5, 6}, Shape{2, 3});
  Array b = host_array({10, 20, 30}, Shape{3});
  Array c = a + b;
  const auto out = drain(c);
  const float want[] = {11, 22, 33, 14, 25, 36};
  LSE_EXPECT_EQ(out.size(), 6u);
  for (std::size_t i = 0; i < 6; ++i) LSE_EXPECT_NEAR(out[i], want[i], 1e-6);
}

LSE_TEST(reductions_over_last_axis) {
  Array a = host_array({1, 2, 3, 4, 5, 6}, Shape{2, 3});
  Array s = sum(a, -1);
  const auto out = drain(s);
  LSE_EXPECT_EQ(out.size(), 2u);
  LSE_EXPECT_NEAR(out[0], 6.0, 1e-6);
  LSE_EXPECT_NEAR(out[1], 15.0, 1e-6);
}

LSE_TEST(reductions_over_leading_axis) {
  Array a = host_array({1, 2, 3, 4, 5, 6}, Shape{2, 3});
  Array s = sum(a, 0);
  const auto out = drain(s);
  LSE_EXPECT_EQ(out.size(), 3u);
  LSE_EXPECT_NEAR(out[0], 5.0, 1e-6);
  LSE_EXPECT_NEAR(out[1], 7.0, 1e-6);
  LSE_EXPECT_NEAR(out[2], 9.0, 1e-6);
}

LSE_TEST(softmax_rows_sum_to_one) {
  Array a = host_array({1, 2, 3, 1, 1, 1}, Shape{2, 3});
  Array s = softmax(a, -1);
  const auto out = drain(s);
  LSE_EXPECT_NEAR(out[0] + out[1] + out[2], 1.0, 1e-6);
  LSE_EXPECT_NEAR(out[3] + out[4] + out[5], 1.0, 1e-6);
  // Uniform logits give a uniform distribution.
  for (std::size_t i = 3; i < 6; ++i) LSE_EXPECT_NEAR(out[i], 1.0 / 3.0, 1e-6);
}

LSE_TEST(softmax_is_shift_invariant) {
  // The max-subtraction guard: adding a constant must not change the result.
  Array a = host_array({1, 2, 3}, Shape{3});
  Array b = host_array({101, 102, 103}, Shape{3});
  Array sa = softmax(a, -1);
  Array sb = softmax(b, -1);
  const auto oa = drain(sa);
  const auto ob = drain(sb);
  for (std::size_t i = 0; i < 3; ++i) LSE_EXPECT_NEAR(oa[i], ob[i], 1e-6);
}

LSE_TEST(rms_norm_matches_definition) {
  Array x = host_array({1, 2, 3, 4}, Shape{1, 4});
  Array w = host_array({1, 1, 1, 1}, Shape{4});
  Array y = rms_norm(x, w, 1e-6f);
  const auto out = drain(y);

  const double ms = (1.0 + 4.0 + 9.0 + 16.0) / 4.0;
  const double scale = 1.0 / std::sqrt(ms + 1e-6);
  for (std::size_t i = 0; i < 4; ++i) {
    LSE_EXPECT_NEAR(out[i], static_cast<double>(i + 1) * scale, 1e-5);
  }
}

LSE_TEST(matmul_2x3_by_3x2) {
  Array a = host_array({1, 2, 3, 4, 5, 6}, Shape{2, 3});
  Array b = host_array({7, 8, 9, 10, 11, 12}, Shape{3, 2});
  Array c = matmul(a, b);
  const auto out = drain(c);
  LSE_EXPECT_EQ(out.size(), 4u);
  LSE_EXPECT_NEAR(out[0], 58.0, 1e-5);
  LSE_EXPECT_NEAR(out[1], 64.0, 1e-5);
  LSE_EXPECT_NEAR(out[2], 139.0, 1e-5);
  LSE_EXPECT_NEAR(out[3], 154.0, 1e-5);
}

LSE_TEST(silu_matches_reference) {
  Array x = host_array({-1, 0, 1, 2}, Shape{4});
  Array y = silu(x);
  const auto out = drain(y);
  for (std::size_t i = 0; i < 4; ++i) {
    const double v = static_cast<double>(static_cast<int>(i) - 1);
    LSE_EXPECT_NEAR(out[i], v / (1.0 + std::exp(-v)), 1e-6);
  }
}

LSE_TEST(bf16_round_trips_through_the_graph) {
  Array x = Array::full(Shape{8}, DType::kBF16, 1.5f);
  Array y = x + x;
  const auto out = drain(y);
  for (float v : out) LSE_EXPECT_NEAR(v, 3.0, 1e-6);
  LSE_EXPECT(y.dtype() == DType::kBF16);
}

LSE_TEST(reshape_of_a_materialized_buffer_is_a_view) {
  Array a = host_array({1, 2, 3, 4, 5, 6}, Shape{2, 3});
  Array b = reshape(a, Shape{3, 2});
  LSE_EXPECT_OK(b.eval());
  LSE_EXPECT(b.node()->buffer.handle == a.node()->buffer.handle);
  LSE_EXPECT_EQ(b.node()->buffer.offset, a.node()->buffer.offset);
  const auto out = drain(b);
  const float want[] = {1, 2, 3, 4, 5, 6};
  LSE_EXPECT_EQ(out.size(), 6u);
  for (std::size_t i = 0; i < 6; ++i) LSE_EXPECT_NEAR(out[i], want[i], 1e-6);
}

LSE_TEST(slice_copies_a_window) {
  Array a = host_array({1, 2, 3, 4, 5, 6}, Shape{1, 3, 2});
  Array prefix = slice(a, 1, 0, 1);
  const auto p = drain(prefix);
  LSE_EXPECT_EQ(p.size(), 2u);
  LSE_EXPECT_NEAR(p[0], 1.0, 1e-6);
  LSE_EXPECT_NEAR(p[1], 2.0, 1e-6);

  Array tail = slice(a, 1, 2, 3);
  const auto t = drain(tail);
  LSE_EXPECT_EQ(t.size(), 2u);
  LSE_EXPECT_NEAR(t[0], 5.0, 1e-6);
  LSE_EXPECT_NEAR(t[1], 6.0, 1e-6);
}

LSE_TEST(elementwise_chain_fuses_into_one_group) {
  Array x = Array::full(Shape{16}, DType::kF32, 1.0f);
  Array y = silu(x * x + x);
  const NodePtr roots[] = {y.node()};
  const auto groups = Partitioner::partition(roots);
  LSE_EXPECT_EQ(groups.size(), 1u);
  // constant, mul, add, silu
  LSE_EXPECT_EQ(groups[0].nodes.size(), 4u);
}

LSE_TEST(matmul_opens_its_own_group) {
  Array a = Array::full(Shape{4, 4}, DType::kF32, 1.0f);
  Array b = Array::full(Shape{4, 4}, DType::kF32, 2.0f);
  Array c = matmul(a, b);
  const NodePtr roots[] = {c.node()};
  const auto groups = Partitioner::partition(roots);
  // Two constants can fuse with nothing; the matmul is its own barrier group.
  LSE_EXPECT(groups.size() >= 2u);
  bool matmul_isolated = false;
  for (const auto& g : groups) {
    if (g.anchor == OpKind::kMatMul) {
      matmul_isolated = g.nodes.size() == 1;
    }
  }
  LSE_EXPECT(matmul_isolated);
}

LSE_TEST(elementwise_fuses_into_a_reduction_as_prologue) {
  Array x = Array::full(Shape{4, 8}, DType::kF32, 2.0f);
  Array y = sum(x * x, -1);
  const NodePtr roots[] = {y.node()};
  const auto groups = Partitioner::partition(roots);
  LSE_EXPECT_EQ(groups.size(), 1u);
  LSE_EXPECT(groups[0].anchor == OpKind::kSum);
}

LSE_TEST(fusion_rules_are_directly_assertable) {
  Node ew_a, ew_b, red, mm;
  ew_a.set_kind(OpKind::kMul);    ew_a.shape = Shape{8};
  ew_b.set_kind(OpKind::kAdd);    ew_b.shape = Shape{8};
  red.set_kind(OpKind::kSum);     red.shape  = Shape{1};
  mm.set_kind(OpKind::kMatMul);   mm.shape   = Shape{8, 8};

  LSE_EXPECT(Partitioner::can_fuse(ew_a, ew_b));
  LSE_EXPECT(Partitioner::can_fuse(ew_a, red));
  LSE_EXPECT(!Partitioner::can_fuse(ew_a, mm));
  // A barrier absorbs an epilogue only if its kernel advertises a slot for one.
  // This bare Node has no kernel primitive, so it must not.
  LSE_EXPECT(!Partitioner::can_fuse(mm, ew_a));
}

LSE_TEST(materialized_producer_never_fuses) {
  Node produced, consumer;
  produced.set_kind(OpKind::kMul);
  produced.shape = Shape{8};
  produced.materialized = true;
  consumer.set_kind(OpKind::kAdd);
  consumer.shape = Shape{8};
  LSE_EXPECT(!Partitioner::can_fuse(produced, consumer));
}

LSE_TEST(host_read_splits_the_graph_in_two) {
  // The documented "cout breaks a kernel into two" behaviour.
  Array x = Array::full(Shape{16}, DType::kF32, 2.0f);
  Array a = x * x;

  std::ostringstream sink;
  sink << a;  // demand -> flush

  LSE_EXPECT(a.node()->materialized);

  Array b = a + x;
  const NodePtr roots[] = {b.node()};
  const auto groups = Partitioner::partition(roots);
  // `a` is already real, so only `b` remains to schedule.
  LSE_EXPECT_EQ(groups.size(), 1u);
  LSE_EXPECT_EQ(groups[0].nodes.size(), 1u);
}

LSE_TEST(scheduler_reports_what_it_did) {
  Array x = Array::full(Shape{8}, DType::kF32, 1.0f);
  Array y = silu(x + x);
  LSE_EXPECT_OK(y.eval());
  const auto& trace = default_scheduler()->last_trace();
  LSE_EXPECT(trace.kernels_launched >= 1);
  LSE_EXPECT(trace.nodes_evaluated >= 2);
}

LSE_TEST(item_reads_the_first_element) {
  Array a = host_array({42, 1, 2}, Shape{3});
  auto v = a.item();
  LSE_EXPECT(v.ok());
  LSE_EXPECT_NEAR(*v, 42.0, 1e-6);
}

LSE_TEST(eval_is_idempotent) {
  Array x = Array::full(Shape{4}, DType::kF32, 5.0f);
  Array y = x + x;
  LSE_EXPECT_OK(y.eval());
  const auto first = drain(y);
  LSE_EXPECT_OK(y.eval());
  const auto second = drain(y);
  for (std::size_t i = 0; i < first.size(); ++i) {
    LSE_EXPECT_NEAR(first[i], second[i], 0.0);
  }
}

LSE_TEST(linear_uses_out_in_weight_layout) {
  // w is [out, in]; y[o] = sum_i x[i] * w[o, i].
  Array x = host_array({1, 2, 3}, Shape{1, 3});
  Array w = host_array({1, 0, 0, 0, 1, 0}, Shape{2, 3});
  Array y = linear(x, w);
  const auto out = drain(y);
  LSE_EXPECT_EQ(out.size(), 2u);
  LSE_EXPECT_NEAR(out[0], 1.0, 1e-6);
  LSE_EXPECT_NEAR(out[1], 2.0, 1e-6);
}

LSE_TEST(embedding_gathers_rows) {
  Array table = host_array({10, 11, 20, 21, 30, 31}, Shape{3, 2});
  Array ids = host_array({2, 0}, Shape{2});
  Array y = embedding(table, ids);
  const auto out = drain(y);
  LSE_EXPECT_EQ(out.size(), 4u);
  LSE_EXPECT_NEAR(out[0], 30.0, 1e-6);
  LSE_EXPECT_NEAR(out[1], 31.0, 1e-6);
  LSE_EXPECT_NEAR(out[2], 10.0, 1e-6);
}

LSE_TEST(embedding_rejects_out_of_range_ids) {
  // The host oracle reports a bad id. The device kernel does not: a gather
  // past the table is a bad read, not a Status.
  Scheduler* sched = default_scheduler();
  const auto prev = sched != nullptr ? sched->mode() : Scheduler::Mode::kHostOnly;
  if (sched != nullptr) sched->set_mode(Scheduler::Mode::kHostOnly);
  Array table = host_array({1, 2, 3, 4}, Shape{2, 2});
  Array ids = host_array({7}, Shape{1});
  Array y = embedding(table, ids);
  auto s = y.eval();
  if (sched != nullptr) sched->set_mode(prev);
  LSE_EXPECT(!s.ok());
  LSE_EXPECT(s.code() == StatusCode::kOutOfRange);
}

LSE_TEST(gather_rows_picks_by_index) {
  Array x = host_array({10, 11, 20, 21, 30, 31}, Shape{3, 2});
  Array rows = host_array({2, 0}, Shape{2});
  Array y = gather_rows(x, rows);
  const auto out = drain(y);
  LSE_EXPECT_EQ(out.size(), 4u);
  LSE_EXPECT_NEAR(out[0], 30.0, 1e-6);
  LSE_EXPECT_NEAR(out[1], 31.0, 1e-6);
  LSE_EXPECT_NEAR(out[2], 10.0, 1e-6);
}

LSE_TEST(topk_descending_with_indices) {
  Array x = host_array({0.1f, 0.5f, 0.2f, 0.4f, 1.0f, 0.0f, 0.8f, 0.3f},
                       Shape{2, 4});
  Array idx;
  Array vals = topk(x, 2, -1, &idx);
  const auto v = drain(vals);
  const auto i = drain(idx);
  LSE_EXPECT_EQ(v.size(), 4u);
  LSE_EXPECT_NEAR(v[0], 0.5, 1e-6);
  LSE_EXPECT_NEAR(v[1], 0.4, 1e-6);
  LSE_EXPECT_NEAR(v[2], 1.0, 1e-6);
  LSE_EXPECT_NEAR(v[3], 0.8, 1e-6);
  LSE_EXPECT_NEAR(i[0], 1.0, 1e-6);
  LSE_EXPECT_NEAR(i[1], 3.0, 1e-6);
  LSE_EXPECT_NEAR(i[2], 0.0, 1e-6);
  LSE_EXPECT_NEAR(i[3], 2.0, 1e-6);
}

LSE_TEST(eq_and_ge_are_masks) {
  Array a = host_array({1, 2, 3, 2}, Shape{4});
  Array b = host_array({1, 0, 3, 4}, Shape{4});
  Array eq_ab = eq(a, b);
  Array ge_ab = ge(a, b);
  const auto e = drain(eq_ab);
  const auto g = drain(ge_ab);
  LSE_EXPECT_NEAR(e[0], 1.0, 1e-6);
  LSE_EXPECT_NEAR(e[1], 0.0, 1e-6);
  LSE_EXPECT_NEAR(e[2], 1.0, 1e-6);
  LSE_EXPECT_NEAR(e[3], 0.0, 1e-6);
  LSE_EXPECT_NEAR(g[0], 1.0, 1e-6);
  LSE_EXPECT_NEAR(g[1], 1.0, 1e-6);
  LSE_EXPECT_NEAR(g[2], 1.0, 1e-6);
  LSE_EXPECT_NEAR(g[3], 0.0, 1e-6);
}

LSE_TEST(topk_score_band_renormalizes) {
  Array x = host_array({0.80f, 0.15f, 0.04f, 0.01f}, Shape{4});
  Array idx;
  Array vals = topk(x, 2, -1, &idx, 0.15f);
  const auto v = drain(vals);
  const auto i = drain(idx);
  // 0.15 < 0.85*0.80, so the second slot is zeroed; only expert 0 remains.
  LSE_EXPECT_NEAR(i[0], 0.0, 1e-6);
  LSE_EXPECT_NEAR(i[1], 1.0, 1e-6);
  LSE_EXPECT_NEAR(v[0], 1.0, 1e-5);
  LSE_EXPECT_NEAR(v[1], 0.0, 1e-6);
}

LSE_TEST(topk_ties_keep_smaller_index) {
  Array x = host_array({1.0f, 1.0f, 0.0f}, Shape{3});
  Array idx;
  Array vals = topk(x, 2, -1, &idx);
  const auto v = drain(vals);
  const auto i = drain(idx);
  LSE_EXPECT_EQ(v.size(), 2u);
  LSE_EXPECT_NEAR(v[0], 1.0, 1e-6);
  LSE_EXPECT_NEAR(v[1], 1.0, 1e-6);
  LSE_EXPECT_NEAR(i[0], 0.0, 1e-6);
  LSE_EXPECT_NEAR(i[1], 1.0, 1e-6);
}

LSE_TEST(scatter_add_rows_accumulates) {
  Array base = host_array({1, 1, 1, 1}, Shape{2, 2});
  Array rows = host_array({1, 1}, Shape{2});
  Array values = host_array({10, 20, 3, 4}, Shape{2, 2});
  Array y = scatter_add_rows(base, rows, values);
  const auto out = drain(y);
  LSE_EXPECT_EQ(out.size(), 4u);
  LSE_EXPECT_NEAR(out[0], 1.0, 1e-6);
  LSE_EXPECT_NEAR(out[1], 1.0, 1e-6);
  LSE_EXPECT_NEAR(out[2], 14.0, 1e-6);
  LSE_EXPECT_NEAR(out[3], 25.0, 1e-6);
}

LSE_TEST(causal_conv1d_is_left_padded) {
  // K=2, one channel: out[t] = b + w0*x[t-1] + w1*x[t], x[-1]=0.
  Array x = host_array({1, 2, 3}, Shape{1, 3, 1});
  Array w = host_array({10, 1}, Shape{1, 2});
  Array b = host_array({0.5f}, Shape{1});
  Array y = causal_conv1d(x, w, b);
  const auto out = drain(y);
  LSE_EXPECT_EQ(out.size(), 3u);
  LSE_EXPECT_NEAR(out[0], 0.5 + 1.0, 1e-5);
  LSE_EXPECT_NEAR(out[1], 0.5 + 10.0 + 2.0, 1e-5);
  LSE_EXPECT_NEAR(out[2], 0.5 + 20.0 + 3.0, 1e-5);
}

LSE_TEST(transpose_permutes_axes) {
  Array a = host_array({1, 2, 3, 4, 5, 6}, Shape{2, 3});
  Array t = transpose(a, {1, 0});
  LSE_EXPECT_EQ(t.shape()[0], 3);
  LSE_EXPECT_EQ(t.shape()[1], 2);
  const auto out = drain(t);
  const float want[] = {1, 4, 2, 5, 3, 6};
  for (std::size_t i = 0; i < 6; ++i) LSE_EXPECT_NEAR(out[i], want[i], 1e-6);
}

LSE_TEST(a_held_overwrite_keeps_the_first_write) {
  Array dst = host_array({0, 0, 0, 0, 0, 0, 0, 0}, Shape{1, 1, 4, 2});
  Array src = host_array({1, 2}, Shape{1, 1, 1, 2});
  Array pos = host_array({0}, Shape{1});
  Array y = overwrite_slice(dst, src, 2, pos);
  Program p;
  const NodePtr roots[] = {y.node()};
  Scheduler* sched = default_scheduler();
  LSE_EXPECT(sched != nullptr);
  LSE_EXPECT_OK(sched->eval(roots, true, &p));
  LSE_EXPECT(p.holds(roots));
  LSE_EXPECT(!p.groups().empty());

  interpreter::store_element(*src.node(), 0, 3.0f);
  interpreter::store_element(*src.node(), 1, 4.0f);
  interpreter::store_element(*pos.node(), 0, 1.0f);
  src.node()->host_dirty = true;
  pos.node()->host_dirty = true;
  LSE_EXPECT_OK(interpreter::sync_to_device(*src.node(), sched->backend()));
  LSE_EXPECT_OK(interpreter::sync_to_device(*pos.node(), sched->backend()));

  p.reset_compute();
  LSE_EXPECT_OK(sched->eval(roots, true, &p));
  LSE_EXPECT(sched->last_trace().replayed);
  const auto out = drain(y);
  const float want[] = {1, 2, 3, 4, 0, 0, 0, 0};
  for (std::size_t i = 0; i < 8; ++i) LSE_EXPECT_NEAR(out[i], want[i], 1e-6);
}

LSE_TEST(overwrite_slice_keeps_earlier_writes) {
  Array dst = host_array({0, 0, 0, 0, 0, 0, 0, 0}, Shape{1, 1, 4, 2});
  Array a = host_array({1, 2}, Shape{1, 1, 1, 2});
  Array b = host_array({3, 4}, Shape{1, 1, 1, 2});
  Array p0 = host_array({0}, Shape{1});
  Array p1 = host_array({1}, Shape{1});
  Array y0 = overwrite_slice(dst, a, 2, p0);
  (void)drain(y0);
  Array y1 = overwrite_slice(dst, b, 2, p1);
  const auto out = drain(y1);
  const float want[] = {1, 2, 3, 4, 0, 0, 0, 0};
  for (std::size_t i = 0; i < 8; ++i) LSE_EXPECT_NEAR(out[i], want[i], 1e-6);
}

LSE_TEST(overwrite_slice_writes_a_window) {
  Array dst = host_array({1, 1, 2, 2, 3, 3, 4, 4}, Shape{1, 1, 4, 2});
  Array src = host_array({9, 8}, Shape{1, 1, 1, 2});
  Array begin = host_array({2}, Shape{1});
  Array y = overwrite_slice(dst, src, 2, begin);
  const auto out = drain(y);
  const float want[] = {1, 1, 2, 2, 9, 8, 4, 4};
  for (std::size_t i = 0; i < 8; ++i) LSE_EXPECT_NEAR(out[i], want[i], 1e-6);
  LSE_EXPECT_EQ(y.shape().dim(2), 4);
}

LSE_TEST(sdpa_ignores_padding_past_the_offset) {
  Array q = host_array({1, 0, 1, 0}, Shape{1, 1, 2, 2});
  Array k = host_array({1, 0, 1, 0, 0, 0, 0, 0}, Shape{1, 1, 4, 2});
  Array v = host_array({0, 0, 10, 10, 99, 99, 99, 99}, Shape{1, 1, 4, 2});
  Array off = host_array({0}, Shape{1});
  Array o = sdpa(q, k, v, 1.0f, MaskKind::kCausal, 0, off);
  const auto out = drain(o);
  LSE_EXPECT_NEAR(out[0], 0.0, 1e-5);
  LSE_EXPECT_NEAR(out[1], 0.0, 1e-5);
  LSE_EXPECT_NEAR(out[2], 5.0, 1e-5);
  LSE_EXPECT_NEAR(out[3], 5.0, 1e-5);
}

LSE_TEST(concat_and_slice_are_inverse) {
  Array a = host_array({1, 2, 3, 4}, Shape{2, 2});
  Array b = host_array({5, 6, 7, 8}, Shape{2, 2});
  Array c = concat({a, b}, 1);
  LSE_EXPECT_EQ(c.shape()[1], 4);
  const auto joined = drain(c);
  const float want[] = {1, 2, 5, 6, 3, 4, 7, 8};
  for (std::size_t i = 0; i < 8; ++i) LSE_EXPECT_NEAR(joined[i], want[i], 1e-6);

  Array back = slice(c, 1, 2, 4);
  const auto sliced = drain(back);
  const float want_b[] = {5, 6, 7, 8};
  for (std::size_t i = 0; i < 4; ++i) LSE_EXPECT_NEAR(sliced[i], want_b[i], 1e-6);
}

LSE_TEST(repeat_interleaves_for_gqa_expansion) {
  // 2 kv heads -> 4 q heads means each kv head is used by two adjacent q heads.
  Array a = host_array({1, 2}, Shape{2, 1});
  Array r = repeat(a, 2, 0);
  LSE_EXPECT_EQ(r.shape()[0], 4);
  const auto out = drain(r);
  const float want[] = {1, 1, 2, 2};
  for (std::size_t i = 0; i < 4; ++i) LSE_EXPECT_NEAR(out[i], want[i], 1e-6);
}

LSE_TEST(rope_preserves_norm_and_is_identity_at_zero_angle) {
  // cos=1, sin=0 must leave the vector untouched.
  Array x = host_array({1, 2, 3, 4}, Shape{1, 1, 1, 4});
  Array cos = host_array({1, 1, 1, 1}, Shape{1, 4});
  Array sin = host_array({0, 0, 0, 0}, Shape{1, 4});
  Array y = rope(x, cos, sin, 0);
  const auto out = drain(y);
  for (std::size_t i = 0; i < 4; ++i) LSE_EXPECT_NEAR(out[i], static_cast<double>(i + 1), 1e-6);
}

LSE_TEST(rope_rotates_pairs_and_preserves_length) {
  // 90 degrees: (x0,x1) -> (-x1, x0).
  Array x = host_array({1, 0, 0, 1}, Shape{1, 1, 1, 4});
  Array cos = host_array({0, 0, 0, 0}, Shape{1, 4});
  Array sin = host_array({1, 1, 1, 1}, Shape{1, 4});
  Array y = rope(x, cos, sin, 0);
  const auto out = drain(y);
  LSE_EXPECT_NEAR(out[0], 0.0, 1e-6);
  LSE_EXPECT_NEAR(out[1], 1.0, 1e-6);
  LSE_EXPECT_NEAR(out[2], -1.0, 1e-6);
  LSE_EXPECT_NEAR(out[3], 0.0, 1e-6);
}

LSE_TEST(sdpa_with_one_key_returns_that_value) {
  Array q = host_array({1, 0}, Shape{1, 1, 1, 2});
  Array k = host_array({1, 0}, Shape{1, 1, 1, 2});
  Array v = host_array({7, 9}, Shape{1, 1, 1, 2});
  Array o = sdpa(q, k, v, 1.0f, MaskKind::kCausal);
  const auto out = drain(o);
  LSE_EXPECT_NEAR(out[0], 7.0, 1e-5);
  LSE_EXPECT_NEAR(out[1], 9.0, 1e-5);
}

LSE_TEST(sdpa_causal_mask_blocks_the_future) {
  // Two positions, identical q/k so the scores tie. Position 0 may only see
  // key 0, so it returns v[0] exactly; position 1 sees both, so it averages.
  Array q = host_array({1, 0, 1, 0}, Shape{1, 1, 2, 2});
  Array k = host_array({1, 0, 1, 0}, Shape{1, 1, 2, 2});
  Array v = host_array({0, 0, 10, 10}, Shape{1, 1, 2, 2});
  Array o = sdpa(q, k, v, 1.0f, MaskKind::kCausal);
  const auto out = drain(o);
  LSE_EXPECT_NEAR(out[0], 0.0, 1e-5);
  LSE_EXPECT_NEAR(out[1], 0.0, 1e-5);
  LSE_EXPECT_NEAR(out[2], 5.0, 1e-5);
  LSE_EXPECT_NEAR(out[3], 5.0, 1e-5);
}

LSE_TEST(sdpa_shares_kv_heads_across_query_heads) {
  // 2 q heads, 1 kv head: both q heads must read the same k/v.
  Array q = host_array({1, 0, 1, 0}, Shape{1, 2, 1, 2});
  Array k = host_array({1, 0}, Shape{1, 1, 1, 2});
  Array v = host_array({3, 4}, Shape{1, 1, 1, 2});
  Array o = sdpa(q, k, v, 1.0f, MaskKind::kCausal);
  const auto out = drain(o);
  LSE_EXPECT_NEAR(out[0], 3.0, 1e-5);
  LSE_EXPECT_NEAR(out[1], 4.0, 1e-5);
  LSE_EXPECT_NEAR(out[2], 3.0, 1e-5);
  LSE_EXPECT_NEAR(out[3], 4.0, 1e-5);
}

LSE_TEST(sdpa_sliding_window_forgets_distant_keys) {
  // window=1 means each position sees only itself.
  Array q = host_array({1, 0, 1, 0}, Shape{1, 1, 2, 2});
  Array k = host_array({1, 0, 1, 0}, Shape{1, 1, 2, 2});
  Array v = host_array({0, 0, 10, 10}, Shape{1, 1, 2, 2});
  Array o = sdpa(q, k, v, 1.0f, MaskKind::kSlidingWindow, 1);
  const auto out = drain(o);
  LSE_EXPECT_NEAR(out[0], 0.0, 1e-5);
  LSE_EXPECT_NEAR(out[2], 10.0, 1e-5);
}

LSE_TEST(a_node_never_schedules_before_an_input_it_needs) {
  // Regression: a diamond where one arm is a barrier. `mul` could fuse with
  // `silu`, but its other operand comes from a barrier scheduled later, so
  // joining that group would read an unmaterialized buffer and silently yield
  // zeros. The partitioner must refuse the fusion.
  Array x = host_array({1, 2, 3, 4}, Shape{1, 4});
  Array wa = host_array({1, 0, 0, 0, 0, 1, 0, 0}, Shape{2, 4});
  Array wb = host_array({0, 0, 1, 0, 0, 0, 0, 1}, Shape{2, 4});

  Array a = linear(x, wa);        // barrier
  Array b = linear(x, wb);        // barrier, visited after `a`
  Array y = silu(a) * b;          // wants to fuse with silu(a)

  const NodePtr roots[] = {y.node()};
  const auto groups = Partitioner::partition(roots);

  std::unordered_map<const Node*, std::size_t> where;
  for (std::size_t g = 0; g < groups.size(); ++g) {
    for (const auto& n : groups[g].nodes) where[n.get()] = g;
  }
  for (std::size_t g = 0; g < groups.size(); ++g) {
    for (const auto& n : groups[g].nodes) {
      for (const auto& in : n->inputs) {
        auto it = where.find(in.get());
        if (it == where.end()) continue;
        LSE_EXPECT(it->second <= g);
      }
    }
  }

  const auto out = drain(y);
  LSE_EXPECT_EQ(out.size(), 2u);
  // a = [1,2], b = [3,4]; silu(a)*b
  LSE_EXPECT_NEAR(out[0], (1.0 / (1.0 + std::exp(-1.0))) * 3.0, 1e-5);
  LSE_EXPECT_NEAR(out[1], (2.0 / (1.0 + std::exp(-2.0))) * 4.0, 1e-5);
}

LSE_TEST(workgroup_device_defaults_match_gfx1151) {
  const WorkgroupDevice d = WorkgroupDevice::from(nullptr);
  LSE_EXPECT_EQ(d.lds_bytes, 65536u);
  LSE_EXPECT_EQ(d.compute_units, 40u);
  LSE_EXPECT_EQ(d.wavefront, 32u);
  LSE_EXPECT_EQ(d.max_waves_per_cu, 32u);
}

LSE_TEST(sibling_linears_share_a_workgroup) {
  Array x = Array::full(Shape{1, 32}, DType::kF32, 1.0f);
  Array wq = Array::full(Shape{32, 32}, DType::kF32, 0.1f);
  Array wk = Array::full(Shape{16, 32}, DType::kF32, 0.2f);
  Array q = linear(x, wq);
  Array k = linear(x, wk);

  Workgroup wg;
  LSE_EXPECT(wg.try_add(q.node()));
  LSE_EXPECT(wg.try_add(k.node()));
  LSE_EXPECT_EQ(wg.ideal_launches(), 1u);
  LSE_EXPECT_EQ(wg.launches(), 1u);
  LSE_EXPECT(wg.fused());
  LSE_EXPECT(wg.occupancy() > 0);
  LSE_EXPECT(wg.chain() == WorkgroupChain::kIndependent);
}

LSE_TEST(workgroup_staged_chain_is_not_independent) {
  // rms → linear: the linear reads the whole row, so the chain is staged.
  // Decode hidden fits in LDS, so hardware wants one launch; the emitter
  // still cannot write that body.
  Array x = Array::full(Shape{1, 32}, DType::kF32, 1.0f);
  Array rw = Array::full(Shape{32}, DType::kF32, 1.0f);
  Array w = Array::full(Shape{32, 32}, DType::kF32, 0.1f);
  Array h = rms_norm(x, rw, 1e-6f);
  Array y = linear(h, w);

  Workgroup wg;
  LSE_EXPECT(wg.try_add(h.node()));
  LSE_EXPECT(wg.try_add(y.node()));
  LSE_EXPECT(wg.chain() == WorkgroupChain::kStaged);
  LSE_EXPECT(wg.phase() == WorkgroupPhase::kDecode);
  LSE_EXPECT_EQ(wg.ideal_launches(), 1u);
  LSE_EXPECT_EQ(wg.launches(), 2u);
}

LSE_TEST(workgroup_diamond_is_a_fork) {
  // silu(Q)*K reads two kernel members. The join cannot ride in the
  // producers' launch.
  Array x = Array::full(Shape{1, 32}, DType::kF32, 1.0f);
  Array wq = Array::full(Shape{32, 32}, DType::kF32, 0.1f);
  Array wk = Array::full(Shape{32, 32}, DType::kF32, 0.2f);
  Array q = linear(x, wq);
  Array k = linear(x, wk);
  Array s = silu(q);
  Array y = s * k;

  Workgroup wg;
  LSE_EXPECT(wg.try_add(q.node()));
  LSE_EXPECT(wg.try_add(k.node()));
  LSE_EXPECT(wg.chain() == WorkgroupChain::kIndependent);
  LSE_EXPECT(wg.try_add(s.node()));
  LSE_EXPECT(wg.try_add(y.node()));
  LSE_EXPECT(wg.chain() == WorkgroupChain::kFork);
  LSE_EXPECT(wg.phase() == WorkgroupPhase::kDecode);
  LSE_EXPECT_EQ(wg.ideal_launches(), 1u);
  LSE_EXPECT(wg.launches() >= 2u);
}

LSE_TEST(workgroup_swiglu_chain_is_a_fork) {
  Array x = Array::full(Shape{1, 32}, DType::kF32, 1.0f);
  Array wg = Array::full(Shape{64, 32}, DType::kF32, 0.1f);
  Array wu = Array::full(Shape{64, 32}, DType::kF32, 0.2f);
  Array wd = Array::full(Shape{32, 64}, DType::kF32, 0.3f);
  Array gate = linear(x, wg);
  Array up = linear(x, wu);
  Array act = silu(gate);
  Array hid = act * up;
  Array down = linear(hid, wd);

  Workgroup plan;
  LSE_EXPECT(plan.try_add(gate.node()));
  LSE_EXPECT(plan.try_add(up.node()));
  LSE_EXPECT(plan.chain() == WorkgroupChain::kIndependent);
  LSE_EXPECT(plan.try_add(act.node()));
  LSE_EXPECT(plan.try_add(hid.node()));
  LSE_EXPECT(plan.try_add(down.node()));
  LSE_EXPECT(plan.chain() == WorkgroupChain::kFork);
  LSE_EXPECT(plan.phase() == WorkgroupPhase::kDecode);
  LSE_EXPECT_EQ(plan.ideal_launches(), 1u);
  LSE_EXPECT(plan.launches() >= 2u);
}

LSE_TEST(a_held_program_reevals_the_same_nodes) {
  Array x = Array::full(Shape{1, 4}, DType::kF32, 2.0f);
  Array w = Array::full(Shape{4, 4}, DType::kF32, 0.5f);
  Array y = linear(x, w);
  Workgroup wg;
  LSE_EXPECT(wg.try_add(y.node()));
  LSE_EXPECT_OK(y.eval());
  const auto first = drain(y);

  wg.reset_compute();
  LSE_EXPECT(!y.node()->materialized);
  LSE_EXPECT_OK(y.eval());
  const auto again = drain(y);
  LSE_EXPECT_EQ(first.size(), again.size());
  for (std::size_t i = 0; i < first.size(); ++i) {
    LSE_EXPECT_NEAR(first[i], again[i], 1e-5);
  }

  Program p;
  const NodePtr roots[] = {y.node()};
  p.retain(roots, {}, {}, {});
  LSE_EXPECT(p.holds(roots));
  p.reset_compute();
  LSE_EXPECT(!y.node()->materialized);
  LSE_EXPECT_OK(y.eval());
}

LSE_TEST(workgroup_refuses_an_unrelated_or_oversized_member) {
  Array x = Array::full(Shape{1, 32}, DType::kF32, 1.0f);
  Array w = Array::full(Shape{32, 32}, DType::kF32, 0.1f);
  Array q = linear(x, w);

  Array y = Array::full(Shape{1, 8}, DType::kF32, 3.0f);
  Array wy = Array::full(Shape{8, 8}, DType::kF32, 0.1f);
  Array z = linear(y, wy);

  Workgroup wg;
  LSE_EXPECT(wg.try_add(q.node()));
  LSE_EXPECT(!wg.try_add(z.node()));

  WorkgroupDevice tiny;
  tiny.max_waves_per_cu = 0;
  Workgroup tight(tiny);
  LSE_EXPECT(!tight.try_add(q.node()));
}

LSE_TEST(sibling_linears_are_one_fusion_group) {
  Array x = Array::full(Shape{1, 32}, DType::kF32, 1.0f);
  Array wq = Array::full(Shape{32, 32}, DType::kF32, 0.1f);
  Array wk = Array::full(Shape{16, 32}, DType::kF32, 0.2f);
  Array q = linear(x, wq);
  Array k = linear(x, wk);
  Workgroup wg;
  LSE_EXPECT(wg.try_add(q.node()));
  LSE_EXPECT(wg.try_add(k.node()));
  LSE_EXPECT_EQ(wg.ideal_launches(), 1u);
  LSE_EXPECT_EQ(wg.launches(), 1u);
  const NodePtr roots[] = {q.node(), k.node()};
  const auto groups = Partitioner::partition(roots);
  bool together = false;
  for (const auto& g : groups) {
    bool has_q = false;
    bool has_k = false;
    for (const NodePtr& n : g.nodes) {
      if (n.get() == q.node().get()) has_q = true;
      if (n.get() == k.node().get()) has_k = true;
    }
    if (has_q && has_k) {
      together = true;
      LSE_EXPECT_EQ(g.launches, 1u);
    }
  }
  LSE_EXPECT(together);

  const auto qv = drain(q);
  const auto kv = drain(k);
  LSE_EXPECT_EQ(qv.size(), 32u);
  LSE_EXPECT_EQ(kv.size(), 16u);
  for (float v : qv) LSE_EXPECT_NEAR(v, 3.2, 1e-4);
  for (float v : kv) LSE_EXPECT_NEAR(v, 6.4, 1e-4);
}

// What one launch costs in workgroup scratch: every DISTINCT staged row once,
// summed. Two `__shared__` declarations do not overlap even in disjoint block
// scopes, so a max here would say a two-row launch costs one row.
LSE_TEST(a_launch_is_priced_for_every_distinct_row_it_stages) {
  const WorkgroupDevice dev;
  Array x0 = Array::full(Shape{1, 1024}, DType::kF32, 1.0f);
  Array x1 = Array::full(Shape{1, 1024}, DType::kF32, 2.0f);
  Array wa = Array::full(Shape{128, 1024}, DType::kF32, 0.1f);
  Array wb = Array::full(Shape{128, 1024}, DType::kF32, 0.2f);
  Array wc = Array::full(Shape{128, 1024}, DType::kF32, 0.3f);
  Array a = linear(x0, wa);
  Array b = linear(x0, wb);
  Array c = linear(x1, wc);

  LSE_EXPECT_EQ(staged_row_bytes(*a.node(), dev), 4096u);
  // Two siblings off one activation share the panel the emitter hoists.
  const Node* shared[] = {a.node().get(), b.node().get()};
  LSE_EXPECT_EQ(group_lds_bytes(shared, dev), 4096u);
  // A third off a different buffer brings its own.
  const Node* both[] = {a.node().get(), b.node().get(), c.node().get()};
  LSE_EXPECT_EQ(group_lds_bytes(both, dev), 8192u);

  // A row too wide to stage at all costs nothing: the emitter reads that
  // activation from global instead. This is the 27B's down projection, whose
  // 17408-long row would want 69632 bytes.
  Array wide = Array::full(Shape{1, 17408}, DType::kF32, 1.0f);
  Array wd = Array::full(Shape{128, 17408}, DType::kF32, 0.1f);
  Array d = linear(wide, wd);
  LSE_EXPECT_EQ(staged_row_bytes(*d.node(), dev), 0u);
}

// The gate on sibling fusion is a workgroup-scratch budget, and it has to be in
// the unit the hardware charges in.
//
// A solo grid GEMV already stages its whole activation row against the ENTIRE
// device budget — the 4B's K=9216 down projection declares 36864 bytes on its
// own, verified in the emitted dumps — so merging siblings that share that row
// costs no occupancy at all. The rule this replaced held one row to a quarter of
// the budget, 16384 bytes, and so refused the 27B's 5120-wide hidden while the
// emitter was already emitting more than twice that in a single stage.
LSE_TEST(sibling_linears_fuse_at_a_row_the_quarter_budget_rule_refused) {
  const WorkgroupDevice dev;
  // At least one workgroup per CU: on a 64 KiB pool shared by two CUs that is
  // half the budget, not a quarter of it.
  LSE_EXPECT_EQ(dev.resident_lds_bytes(), 32768u);

  // 27B geometry: hidden 5120, so the row gate and up both stage is 20480 B.
  //
  // The MLP shape is what needs the gate: DFS post-order is x, wg, gate, silu,
  // wu, up, and the silu between the two linears is exactly what
  // group_sibling_linears exists to rotate past. Refuse on the budget and the
  // rotation never happens, so gate and up stay in separate groups and separate
  // launches.
  Array x = Array::full(Shape{1, 5120}, DType::kF32, 1.0f);
  Array wgate = Array::full(Shape{128, 5120}, DType::kF32, 0.1f);
  Array wu = Array::full(Shape{128, 5120}, DType::kF32, 0.2f);
  // Weights are materialized buffers in a real checkpoint, so they are not in
  // the traversal and cannot hold the rotation back on their own account.
  (void)x.eval();
  (void)wgate.eval();
  (void)wu.eval();
  Array gate = linear(x, wgate);
  Array up = linear(x, wu);
  Array out = silu(gate) * up;

  const Node* run[] = {gate.node().get(), up.node().get()};
  LSE_EXPECT_EQ(group_lds_bytes(run, dev), 20480u);
  // Over the old quarter rule, under the honest one.
  LSE_EXPECT(20480u > dev.lds_bytes / 4u);
  LSE_EXPECT(20480u <= dev.resident_lds_bytes());

  // group_sibling_linears runs in phases(), which is what pulls `up` past the
  // silu so cuts() can put both in one grid launch.
  const NodePtr roots[] = {out.node()};
  const auto wgs = Partitioner::phases(roots);
  LSE_EXPECT(!wgs.empty());
  if (wgs.empty()) return;
  bool one_launch = false;
  std::size_t cut_count = 0;
  for (const Workgroup& phase : wgs) {
    for (const WorkgroupCut& cut : phase.cuts()) {
      ++cut_count;
      std::size_t hits = 0;
      for (const NodePtr& n : cut.nodes) {
        for (const Node* m : run) {
          if (n.get() == m) ++hits;
        }
      }
      if (hits == 2) {
        one_launch = true;
        LSE_EXPECT(cut.grid_linears);
      }
    }
  }
  LSE_EXPECT(one_launch);
  if (!one_launch) {
    std::printf("       gate and up split across %zu cut(s)\n", cut_count);
  }
  // The phase's worst launch is that one shared row, counted once.
  LSE_EXPECT_EQ(wgs.front().lds_bytes(), 20480u);

  // And the answer is still the answer: silu(512) * 1024, 5120 terms each.
  const auto ov = drain(out);
  LSE_EXPECT_EQ(ov.size(), 128u);
  for (float e : ov) LSE_EXPECT_NEAR(e, 512.0 * 1024.0, 4.0);
}

LSE_TEST(sibling_linears_at_hidden_width) {
  Array x = Array::full(Shape{4, 1024}, DType::kF32, 1.0f);
  Array wq = Array::full(Shape{1024, 1024}, DType::kF32, 0.01f);
  Array wk = Array::full(Shape{256, 1024}, DType::kF32, 0.02f);
  Array wv = Array::full(Shape{256, 1024}, DType::kF32, 0.03f);
  Array wa = Array::full(Shape{32, 1024}, DType::kF32, 0.04f);
  Array wb = Array::full(Shape{32, 1024}, DType::kF32, 0.05f);
  Array wg = Array::full(Shape{1024, 1024}, DType::kF32, 0.01f);
  Array q = linear(x, wq);
  Array k = linear(x, wk);
  Array v = linear(x, wv);
  Array a = linear(x, wa);
  Array b = linear(x, wb);
  Array gate = linear(x, wg);
  Workgroup plan;
  LSE_EXPECT(plan.try_add(q.node()));
  LSE_EXPECT(plan.try_add(k.node()));
  LSE_EXPECT(plan.try_add(v.node()));
  LSE_EXPECT(plan.try_add(a.node()));
  LSE_EXPECT(plan.try_add(b.node()));
  LSE_EXPECT(plan.try_add(gate.node()));
  LSE_EXPECT_EQ(plan.ideal_launches(), 1u);
  LSE_EXPECT_EQ(plan.launches(), 1u);
  const auto qv = drain(q);
  const auto kv = drain(k);
  const auto av = drain(a);
  LSE_EXPECT_EQ(qv.size(), 4096u);
  LSE_EXPECT_EQ(kv.size(), 1024u);
  LSE_EXPECT_EQ(av.size(), 128u);
  // WMMA multiplies f16 fragments; 1024 terms of 0.01 sit about 2e-3 off.
  for (float e : qv) LSE_EXPECT_NEAR(e, 10.24, 5e-3);
  for (float e : kv) LSE_EXPECT_NEAR(e, 20.48, 5e-3);
  for (float e : av) LSE_EXPECT_NEAR(e, 40.96, 5e-3);
}

LSE_TEST(a_small_decode_ffn_is_one_cut) {
  Array x = Array::full(Shape{1, 32}, DType::kF32, 1.0f);
  Array rw = Array::full(Shape{32}, DType::kF32, 1.0f);
  Array w1 = Array::full(Shape{64, 32}, DType::kF32, 0.1f);
  Array w2 = Array::full(Shape{32, 64}, DType::kF32, 0.2f);
  Array nrm = rms_norm(x, rw, 1e-6f);
  Array hid = linear(nrm, w1);
  Array act = silu(hid);
  Array y = linear(act, w2);
  Workgroup wg;
  LSE_EXPECT(wg.try_add(nrm.node()));
  LSE_EXPECT(wg.try_add(hid.node()));
  LSE_EXPECT(wg.try_add(act.node()));
  LSE_EXPECT(wg.try_add(y.node()));
  const auto cuts = wg.cuts();
  LSE_EXPECT_EQ(cuts.size(), 1u);
  LSE_EXPECT(cuts[0].sync_before == WorkgroupSync::kNone);
}

LSE_TEST(workgroup_cuts_wide_linears_and_reuses_a_dead_slot) {
  Array x = Array::full(Shape{1, 128}, DType::kF32, 1.0f);
  Array w1 = Array::full(Shape{128, 128}, DType::kF32, 0.1f);
  Array w2 = Array::full(Shape{128, 128}, DType::kF32, 0.2f);
  Array hid = linear(x, w1);
  Array act = silu(hid);
  Array y = linear(act, w2);
  Workgroup wg;
  LSE_EXPECT(wg.try_add(hid.node()));
  LSE_EXPECT(wg.try_add(act.node()));
  LSE_EXPECT(wg.try_add(y.node()));
  const auto cuts = wg.cuts();
  LSE_EXPECT(cuts.size() >= 2u);
  LSE_EXPECT(cuts.front().grid_linears);
  LSE_EXPECT(cuts[1].sync_before == WorkgroupSync::kStream);
  const NodePtr roots[] = {y.node()};
  wg.plan_slots(roots);
  LSE_EXPECT(wg.reused_slots() >= 1u);
  LSE_EXPECT(wg.slot_count() < 3u);
}

LSE_TEST(a_decode_ffn_is_one_phase_workgroup) {
  Array x = Array::full(Shape{1, 32}, DType::kF32, 1.0f);
  Array rw = Array::full(Shape{32}, DType::kF32, 1.0f);
  Array w1 = Array::full(Shape{64, 32}, DType::kF32, 0.1f);
  Array w2 = Array::full(Shape{32, 64}, DType::kF32, 0.2f);
  Array y = linear(silu(linear(rms_norm(x, rw, 1e-6f), w1)), w2);
  const NodePtr roots[] = {y.node()};
  const auto planned = Partitioner::phases(roots);
  LSE_EXPECT_EQ(planned.size(), 1u);
  LSE_EXPECT(planned[0].phase() == WorkgroupPhase::kDecode);
  LSE_EXPECT_EQ(planned[0].ideal_launches(), 1u);
}

LSE_TEST(prefill_and_decode_are_separate_phases) {
  Array x_dec = Array::full(Shape{1, 32}, DType::kF32, 1.0f);
  Array x_pre = Array::full(Shape{32, 32}, DType::kF32, 1.0f);
  Array w = Array::full(Shape{32, 32}, DType::kF32, 0.1f);
  Array d = linear(x_dec, w);
  Array p = linear(x_pre, w);
  Workgroup dec;
  Workgroup pre;
  LSE_EXPECT(dec.try_add(d.node()));
  LSE_EXPECT(pre.try_add(p.node()));
  LSE_EXPECT(dec.phase() == WorkgroupPhase::kDecode);
  LSE_EXPECT(pre.phase() == WorkgroupPhase::kPrefill);
  LSE_EXPECT(!dec.try_add(p.node()));
  LSE_EXPECT_EQ(dec.ideal_launches(), 1u);
  LSE_EXPECT_EQ(pre.ideal_launches(), 1u);
}

LSE_TEST(decode_ffn_phase_kernel_launches_once) {
  Array x = Array::full(Shape{1, 32}, DType::kF32, 1.0f);
  Array rw = Array::full(Shape{32}, DType::kF32, 1.0f);
  Array w1 = Array::full(Shape{64, 32}, DType::kF32, 0.1f);
  Array w2 = Array::full(Shape{32, 64}, DType::kF32, 0.2f);
  Array y = linear(silu(linear(rms_norm(x, rw, 1e-6f), w1)), w2);
  Scheduler* sched = default_scheduler();
  if (sched != nullptr) sched->reset_accumulated_trace();
  LSE_EXPECT_OK(y.eval());
  if (sched == nullptr) return;
  const auto launches = sched->last_trace().kernels_launched;
  std::printf("       ffn launches=%u device=%u phase=%u ideal=%u\n",
              launches, sched->last_trace().device_groups,
              sched->last_trace().phase_groups,
              sched->last_trace().phase_ideal_launches);
  LSE_EXPECT_EQ(launches, 1u);
  const auto got = drain(y);
  LSE_EXPECT_EQ(got.size(), 32u);
  // x=1 → rms≈1 → linear 32*0.1=3.2 → silu(3.2)≈3.077 → linear 64*0.2*silu
  const double hid = 3.2 / (1.0 + std::exp(-3.2));
  const double want = 64.0 * 0.2 * hid;
  for (float v : got) LSE_EXPECT_NEAR(v, want, 1e-3);
}

LSE_TEST(swiglu_chain_evaluates) {
  Array x = Array::full(Shape{1, 4}, DType::kF32, 1.0f);
  Array wg = Array::full(Shape{8, 4}, DType::kF32, 0.1f);
  Array wu = Array::full(Shape{8, 4}, DType::kF32, 0.2f);
  Array wd = Array::full(Shape{4, 8}, DType::kF32, 0.3f);
  Array y = linear(silu(linear(x, wg)) * linear(x, wu), wd);
  const NodePtr roots[] = {y.node()};
  const auto groups = Partitioner::partition(roots);
  std::size_t lin = 0;
  for (const FusionGroup& g : groups) {
    for (const NodePtr& n : g.nodes) {
      if (n->kind == OpKind::kLinear) ++lin;
    }
  }
  LSE_EXPECT_EQ(lin, 3u);
  const auto got = drain(y);
  LSE_EXPECT_EQ(got.size(), 4u);
}

LSE_TEST(two_slices_same_count_do_not_share_a_body) {
  // Same element count and axis, different begin. A name that only
  // hashed those would emit one function and copy the wrong window.
  Array x = host_array({0, 1, 2, 3, 4, 5, 6, 7}, Shape{8});
  Array a = slice(x, 0, 0, 4);
  Array b = slice(x, 0, 4, 8);
  Array y = a + b;
  Scheduler* sched = default_scheduler();
  if (sched != nullptr) sched->reset_accumulated_trace();
  const auto got = drain(y);
  LSE_EXPECT_EQ(got.size(), 4u);
  LSE_EXPECT_NEAR(got[0], 4.0, 1e-5);
  LSE_EXPECT_NEAR(got[1], 6.0, 1e-5);
  LSE_EXPECT_NEAR(got[2], 8.0, 1e-5);
  LSE_EXPECT_NEAR(got[3], 10.0, 1e-5);
  if (sched == nullptr) return;
  const auto launches = sched->last_trace().kernels_launched;
  std::printf("       slice launches=%u phase=%u ideal=%u\n", launches,
              sched->last_trace().phase_groups,
              sched->last_trace().phase_ideal_launches);
  LSE_EXPECT_EQ(launches, 1u);
}

LSE_TEST(swiglu_phase_kernel_launches_once) {
  Array x = Array::full(Shape{1, 4}, DType::kF32, 1.0f);
  Array wg = Array::full(Shape{8, 4}, DType::kF32, 0.1f);
  Array wu = Array::full(Shape{8, 4}, DType::kF32, 0.2f);
  Array wd = Array::full(Shape{4, 8}, DType::kF32, 0.3f);
  Array y = linear(silu(linear(x, wg)) * linear(x, wu), wd);
  Scheduler* sched = default_scheduler();
  if (sched != nullptr) sched->reset_accumulated_trace();
  LSE_EXPECT_OK(y.eval());
  if (sched != nullptr) {
    const auto launches = sched->last_trace().kernels_launched;
    std::printf("       swiglu launches=%u phase=%u ideal=%u\n", launches,
                sched->last_trace().phase_groups,
                sched->last_trace().phase_ideal_launches);
    LSE_EXPECT_EQ(launches, 1u);
  }
  const auto got = drain(y);
  LSE_EXPECT_EQ(got.size(), 4u);
}

LSE_TEST(gdn_phase_kernel_matches_the_delta_rule) {
  // B=1 T=1 H=1 D=16, q=k=v=1, alpha=1, beta=0.5, S=0.
  // S <- 0.5 * 1; o = 16 * 0.5 = 8; carried S is 0.5 everywhere.
  const std::int64_t D = 16;
  Array q = Array::full(Shape{1, 1, 1, D}, DType::kF32, 1.0f);
  Array k = Array::full(Shape{1, 1, 1, D}, DType::kF32, 1.0f);
  Array v = Array::full(Shape{1, 1, 1, D}, DType::kF32, 1.0f);
  Array alpha = Array::full(Shape{1, 1, 1}, DType::kF32, 1.0f);
  Array beta = Array::full(Shape{1, 1, 1}, DType::kF32, 0.5f);
  Array s_in = Array::zeros(Shape{1, 1, D, D}, DType::kF32);
  Array s_out;
  Array o = gated_delta_step(q, k, v, alpha, beta, s_in, &s_out);
  Scheduler* sched = default_scheduler();
  if (sched != nullptr) sched->reset_accumulated_trace();
  const NodePtr roots[] = {o.node(), s_out.node()};
  if (sched != nullptr) {
    LSE_EXPECT_OK(sched->eval(roots, true));
    const auto launches = sched->last_trace().kernels_launched;
    std::printf("       gdn launches=%u phase=%u ideal=%u\n", launches,
                sched->last_trace().phase_groups,
                sched->last_trace().phase_ideal_launches);
    LSE_EXPECT_EQ(launches, 1u);
  } else {
    LSE_EXPECT_OK(o.eval());
    LSE_EXPECT_OK(s_out.eval());
  }
  const auto ov = drain(o);
  const auto sv = drain(s_out);
  LSE_EXPECT_EQ(ov.size(), static_cast<std::size_t>(D));
  LSE_EXPECT_EQ(sv.size(), static_cast<std::size_t>(D * D));
  for (float e : ov) LSE_EXPECT_NEAR(e, 8.0, 1e-4);
  for (float e : sv) LSE_EXPECT_NEAR(e, 0.5, 1e-4);
}

// --- stream placement -------------------------------------------------------
// The policy runs on every backend; whether a device can cash it is a separate
// question its StreamCapabilities answers. These drive it directly so the
// placement rules are checked on a machine with one queue, or none.
namespace {

// A group that writes `out` and reads `ins`, with buffers standing in for the
// allocations the planner actually compares.
FusionGroup placed(std::uint64_t out_handle, std::size_t out_bytes,
                   std::initializer_list<std::uint64_t> ins) {
  FusionGroup g;
  auto make = [](std::uint64_t handle, std::size_t bytes) {
    auto n = std::make_shared<Node>();
    n->kind = OpKind::kMul;
    n->dtype = DType::kF32;
    n->shape = Shape({static_cast<std::int64_t>(bytes / 4)});
    n->buffer.handle = handle;
    n->buffer.size_bytes = bytes;
    return n;
  };
  NodePtr out = make(out_handle, out_bytes);
  for (std::uint64_t h : ins) out->inputs.push_back(make(h, out_bytes));
  g.nodes.push_back(out);
  g.outputs.push_back(out);
  g.inputs = out->inputs;
  return g;
}

backend::StreamCapabilities four_streams() {
  backend::StreamCapabilities c;
  c.stream_count = 4;
  c.concurrent_streams = 4;
  c.needs_explicit_events = true;
  return c;
}

backend::DeviceInfo wide_device() {
  backend::DeviceInfo d;
  d.compute_units = 64;
  d.max_threads_per_workgroup = 1024;
  return d;
}

}  // namespace

LSE_TEST(a_dependency_chain_stays_on_one_stream) {
  // b reads a's output, c reads b's: nothing may overlap, so nothing moves and
  // no event is spent.
  std::vector<FusionGroup> gs{placed(10, 256, {}), placed(11, 256, {10}),
                              placed(12, 256, {11})};
  const StreamPlan p = plan_streams(gs, four_streams(), wide_device());
  LSE_EXPECT_EQ(p.stream[0], 0u);
  LSE_EXPECT_EQ(p.stream[1], 0u);
  LSE_EXPECT_EQ(p.stream[2], 0u);
  LSE_EXPECT_EQ(p.waits_total, 0u);
  LSE_EXPECT_EQ(p.chain, 3u);
}

LSE_TEST(independent_consumers_of_one_value_fork_onto_their_own_streams) {
  // b and c both read a and write different buffers. b continues a's stream;
  // c has to move, and pays exactly one event to be ordered after a.
  std::vector<FusionGroup> gs{placed(10, 256, {}), placed(11, 256, {10}),
                              placed(12, 256, {10})};
  const StreamPlan p = plan_streams(gs, four_streams(), wide_device());
  LSE_EXPECT_EQ(p.stream[0], 0u);
  LSE_EXPECT_EQ(p.stream[1], 0u);
  LSE_EXPECT(p.stream[2] != p.stream[1]);
  LSE_EXPECT_EQ(p.waits_total, 1u);
  LSE_EXPECT_EQ(p.waits[2].size(), 1u);
  LSE_EXPECT_EQ(p.waits[2][0], 0u);
  LSE_EXPECT_EQ(p.record_after[0], 1);
  LSE_EXPECT_EQ(p.chain, 2u);
}

LSE_TEST(a_reused_buffer_is_a_dependency_even_with_no_value_between_them) {
  // c writes the buffer a wrote and b read. No node connects them; the slot
  // allocator put them on one allocation, and a write-after-read is real.
  std::vector<FusionGroup> gs{placed(10, 256, {}), placed(11, 256, {10}),
                              placed(10, 256, {})};
  const StreamPlan p = plan_streams(gs, four_streams(), wide_device());
  LSE_EXPECT(!p.waits[2].empty() || p.stream[2] == p.stream[1]);
  LSE_EXPECT(p.chain >= 2u);
}

LSE_TEST(a_group_that_fills_the_device_is_not_worth_an_event) {
  // Same fork, but each consumer alone saturates the machine: moving one buys
  // no overlap, so the cost model keeps it ordered and spends nothing.
  const std::size_t big = 64u * 1024u * 4u;  // 64K work items > 64 CU x 1024
  std::vector<FusionGroup> gs{placed(10, big, {}), placed(11, big, {10}),
                              placed(12, big, {10})};
  const StreamPlan p = plan_streams(gs, four_streams(), wide_device());
  LSE_EXPECT_EQ(p.streams_used, 1u);
  LSE_EXPECT_EQ(p.waits_total, 0u);
}

LSE_TEST(a_single_stream_device_gets_the_order_it_already_had) {
  std::vector<FusionGroup> gs{placed(10, 256, {}), placed(11, 256, {10}),
                              placed(12, 256, {10})};
  backend::StreamCapabilities one;  // stream_count 1, concurrent 1
  const StreamPlan p = plan_streams(gs, one, wide_device());
  for (std::uint32_t s : p.stream) LSE_EXPECT_EQ(s, 0u);
  LSE_EXPECT_EQ(p.waits_total, 0u);
  LSE_EXPECT_EQ(p.chain, 3u);
}

LSE_TEST(a_backend_whose_other_streams_cost_more_per_launch_keeps_the_order) {
  // Same fork as the spreading test, on a device with four concurrent streams
  // — but reaching any of them but the first costs extra per dispatch, so the
  // overlap can never pay for itself and nothing moves. The seam is intact and
  // the cost model, not a switch, is what declines.
  std::vector<FusionGroup> gs{placed(10, 256, {}), placed(11, 256, {10}),
                              placed(12, 256, {10})};
  backend::StreamCapabilities caps = four_streams();
  caps.uniform_launch_cost = false;
  const StreamPlan p = plan_streams(gs, caps, wide_device());
  for (std::uint32_t s : p.stream) LSE_EXPECT_EQ(s, 0u);
  LSE_EXPECT_EQ(p.waits_total, 0u);
  LSE_EXPECT_EQ(p.streams_used, 1u);
}

LSE_TEST(a_group_whose_buffers_are_unknown_is_ordered_after_everything) {
  // Nothing is known about the third group's bytes, so it may not be placed
  // beside anything: it waits for the tail of every stream in flight.
  std::vector<FusionGroup> gs{placed(10, 256, {}), placed(11, 256, {10}),
                              placed(12, 256, {10}), placed(0, 256, {})};
  const StreamPlan p = plan_streams(gs, four_streams(), wide_device());
  LSE_EXPECT(!p.waits[3].empty());
}

// ---- sharding: the vocabulary a partitioner reports its answer in ----------

namespace {

DeviceMesh two_way() { return DeviceMesh({{"tp", 2}}); }

Sharding replicated(std::uint8_t axes) {
  Sharding s;
  s.axis_count = axes;
  return s;
}

Sharding sharded(std::int8_t dim) {
  Sharding s;
  s.axis_count = 1;
  s.axes[0] = AxisPlacement::shard(dim);
  return s;
}

Sharding partial_sum() {
  Sharding s;
  s.axis_count = 1;
  s.axes[0] = AxisPlacement::partial();
  return s;
}

using Prop = graph::detail::ShardingPropagator;

}  // namespace

LSE_TEST(mesh_ranks_are_row_major_so_the_last_axis_is_contiguous) {
  const DeviceMesh mesh({{"pp", 2}, {"tp", 4}});
  LSE_EXPECT_EQ(mesh.rank_count(), 8u);
  LSE_EXPECT_EQ(mesh.axis_index("tp"), 1);
  LSE_EXPECT_EQ(mesh.axis_index("dp"), -1);
  LSE_EXPECT_EQ(mesh.coordinate(5, 0), 1);
  LSE_EXPECT_EQ(mesh.coordinate(5, 1), 1);
  const std::vector<dist::Rank> tp = mesh.group_along(5, 1);
  LSE_EXPECT(tp == std::vector<dist::Rank>({4, 5, 6, 7}));
  const std::vector<dist::Rank> pp = mesh.group_along(5, 0);
  LSE_EXPECT(pp == std::vector<dist::Rank>({1, 5}));
}

LSE_TEST(a_rank_outside_the_mesh_has_no_coordinate_and_no_group) {
  const DeviceMesh mesh({{"pp", 2}, {"tp", 4}});
  LSE_EXPECT_EQ(mesh.coordinate(8, 0), -1);
  LSE_EXPECT_EQ(mesh.coordinate(0, 2), -1);
  LSE_EXPECT(mesh.group_along(8, 0).empty());
}

LSE_TEST(uneven_shards_tile_the_global_extent_exactly) {
  const DeviceMesh mesh({{"tp", 3}});
  const Shape global{10};
  std::int64_t total = 0;
  for (dist::Rank r = 0; r < 3; ++r) {
    total += sharded(0).local_shape(global, mesh, r).dim(0);
  }
  LSE_EXPECT_EQ(total, 10);
  LSE_EXPECT_EQ(sharded(0).local_shape(global, mesh, 0).dim(0), 4);
  LSE_EXPECT_EQ(sharded(0).local_shape(global, mesh, 2).dim(0), 3);
}

LSE_TEST(two_mesh_axes_splitting_one_dimension_still_tile_it) {
  const DeviceMesh mesh({{"a", 2}, {"b", 3}});
  Sharding s;
  s.axis_count = 2;
  s.axes[0] = AxisPlacement::shard(0);
  s.axes[1] = AxisPlacement::shard(0);
  const Shape global{10};
  std::int64_t total = 0;
  for (dist::Rank r = 0; r < 6; ++r) total += s.local_shape(global, mesh, r).dim(0);
  LSE_EXPECT_EQ(total, 10);
}

LSE_TEST(a_placement_the_mesh_cannot_honour_degrades_to_replicated) {
  // Over-allocating is recoverable; returning an extent nobody holds is not.
  const DeviceMesh mesh = two_way();
  const Shape global{4, 8};
  LSE_EXPECT(sharded(5).local_shape(global, mesh, 0) == global);
  LSE_EXPECT(sharded(0).local_shape(global, mesh, 99) == global);
}

LSE_TEST(a_dimension_shorter_than_its_axis_leaves_the_tail_holding_nothing) {
  // Empty, not an error: the sum still tiles, and a rank with no rows is a
  // rank with no work rather than a plan that has to be rejected.
  const DeviceMesh mesh({{"tp", 4}});
  const Shape global{2, 8};
  LSE_EXPECT_EQ(sharded(0).local_shape(global, mesh, 0).dim(0), 1);
  LSE_EXPECT_EQ(sharded(0).local_shape(global, mesh, 1).dim(0), 1);
  LSE_EXPECT_EQ(sharded(0).local_shape(global, mesh, 3).dim(0), 0);
  LSE_EXPECT_EQ(sharded(0).local_shape(global, mesh, 3).dim(1), 8);
}

LSE_TEST(one_device_is_the_whole_tensor) {
  // The single-device case is the engine's normal case today, so a sharding
  // must be inert on it rather than a special case callers check for.
  const DeviceMesh mesh({{"tp", 1}});
  const Shape global{6, 8};
  LSE_EXPECT_EQ(mesh.rank_count(), 1u);
  LSE_EXPECT_EQ(mesh.coordinate(0, 0), 0);
  LSE_EXPECT(mesh.group_along(0, 0) == std::vector<dist::Rank>({0}));
  LSE_EXPECT(sharded(0).local_shape(global, mesh, 0) == global);
}

LSE_TEST(a_mesh_with_no_axes_is_one_rank_and_shards_nothing) {
  const DeviceMesh mesh;
  const Shape global{6, 8};
  LSE_EXPECT_EQ(mesh.rank_count(), 1u);
  LSE_EXPECT_EQ(mesh.axis_count(), 0u);
  LSE_EXPECT_EQ(mesh.axis_size(0), 1);
  LSE_EXPECT_EQ(mesh.coordinate(0, 0), -1);
  LSE_EXPECT(mesh.group_along(0, 0).empty());
  LSE_EXPECT(sharded(0).local_shape(global, mesh, 0) == global);
  // Nothing to propagate over and nothing to move.
  const Sharding in[1]{sharded(0)};
  const Shape shapes[1]{global};
  LSE_EXPECT(Prop::propagate(OpKind::kSum, in, shapes, mesh).input_reshards.empty());
  LSE_EXPECT(Prop::plan(sharded(0), replicated(0), mesh).empty());
}

LSE_TEST(a_scalar_has_no_dimension_to_split) {
  const DeviceMesh mesh = two_way();
  const Shape scalar;
  LSE_EXPECT_EQ(scalar.rank(), 0u);
  LSE_EXPECT(sharded(0).local_shape(scalar, mesh, 0) == scalar);
}

LSE_TEST(reshard_kinds_differ_by_more_than_a_constant) {
  const DeviceMesh mesh({{"tp", 4}});
  const Shape global{1024};
  auto bytes = [&](Reshard::Kind k) {
    Reshard r;
    r.kind = k;
    return r.traffic_bytes(global, DType::kF32, mesh);
  };
  LSE_EXPECT_EQ(bytes(Reshard::Kind::kAllGather), 3072u);
  LSE_EXPECT_EQ(bytes(Reshard::Kind::kReduceScatter), 3072u);
  // An all-reduce is a reduce-scatter followed by an all-gather.
  LSE_EXPECT_EQ(bytes(Reshard::Kind::kAllReduce), 6144u);
  // The factor of P that makes a free parallel axis beat splitting a
  // contraction whenever the graph has one.
  LSE_EXPECT_EQ(bytes(Reshard::Kind::kAllToAll), 768u);
  LSE_EXPECT_EQ(bytes(Reshard::Kind::kBroadcast), 4096u);
  LSE_EXPECT_EQ(bytes(Reshard::Kind::kNone), 0u);
}

LSE_TEST(an_axis_with_no_peers_carries_no_traffic) {
  const DeviceMesh mesh({{"tp", 1}});
  Reshard r;
  r.kind = Reshard::Kind::kAllReduce;
  LSE_EXPECT_EQ(r.traffic_bytes(Shape{1024}, DType::kF32, mesh), 0u);
}

LSE_TEST(a_tensor_smaller_than_the_mesh_still_costs_something_to_move) {
  // 8 bytes over 8 ranks: the all-to-all share is 8*7/64, which truncates to
  // zero. Zero reads as free, and free is the one price a collective that
  // actually runs must never be given.
  const DeviceMesh mesh({{"tp", 8}});
  Reshard r;
  r.kind = Reshard::Kind::kAllToAll;
  LSE_EXPECT(r.traffic_bytes(Shape{2}, DType::kF32, mesh) > 0u);
  r.kind = Reshard::Kind::kAllGather;
  LSE_EXPECT(r.traffic_bytes(Shape{1}, DType::kI8, mesh) > 0u);
  // Rounding up must not disturb a share that divides exactly.
  const DeviceMesh four({{"tp", 4}});
  Reshard e;
  e.kind = Reshard::Kind::kAllToAll;
  LSE_EXPECT_EQ(e.traffic_bytes(Shape{1024}, DType::kF32, four), 768u);
  // An empty tensor moves nothing; there is no collective to under-price.
  LSE_EXPECT_EQ(e.traffic_bytes(Shape{0}, DType::kF32, four), 0u);
}

LSE_TEST(a_sharding_renders_one_entry_per_mesh_axis) {
  Sharding s;
  s.axis_count = 2;
  s.axes[0] = AxisPlacement::shard(1);
  s.axes[1] = AxisPlacement::partial();
  LSE_EXPECT(s.to_string() == "[S1,P(sum)]");
  LSE_EXPECT(s.has_partial());
  LSE_EXPECT(!s.is_fully_replicated());
  LSE_EXPECT(replicated(2).is_fully_replicated());
}

LSE_TEST(a_column_parallel_matmul_costs_no_collective) {
  // x[4,64] @ w[128,64]^T with the weight split along its output dim. This is
  // the rule the whole column/row sandwich is built on.
  const DeviceMesh mesh = two_way();
  const Sharding in[2]{replicated(1), sharded(0)};
  const Shape shapes[2]{Shape{4, 64}, Shape{128, 64}};
  const Prop::Result r = Prop::propagate(OpKind::kLinear, in, shapes, mesh);
  LSE_EXPECT(r.input_reshards.empty());
  LSE_EXPECT(r.output.axes[0].placement == Placement::kSharded);
  LSE_EXPECT_EQ(static_cast<int>(r.output.axes[0].tensor_dim), 1);
}

LSE_TEST(a_row_parallel_matmul_yields_a_partial_and_defers_the_all_reduce) {
  const DeviceMesh mesh = two_way();
  const Sharding in[2]{sharded(1), sharded(1)};  // both split on K
  const Shape shapes[2]{Shape{4, 64}, Shape{128, 64}};
  const Prop::Result r = Prop::propagate(OpKind::kLinear, in, shapes, mesh);
  LSE_EXPECT(r.input_reshards.empty());
  LSE_EXPECT(r.output.axes[0].placement == Placement::kPartial);
  // propagate never decides to reduce; leaving the value partial is legal and
  // is what lets the reduction move past the next linear op.
  LSE_EXPECT(r.output_reshard.kind == Reshard::Kind::kNone);
}

LSE_TEST(a_replicated_operand_meets_a_split_contraction_for_free) {
  // The activation holds every K slice already and reads only its own.
  const DeviceMesh mesh = two_way();
  const Sharding in[2]{replicated(1), sharded(1)};
  const Shape shapes[2]{Shape{4, 64}, Shape{128, 64}};
  const Prop::Result r = Prop::propagate(OpKind::kLinear, in, shapes, mesh);
  LSE_EXPECT(r.input_reshards.empty());
  LSE_EXPECT(r.output.axes[0].placement == Placement::kPartial);
}

LSE_TEST(a_partial_activation_carries_through_a_matmul) {
  const DeviceMesh mesh = two_way();
  const Sharding in[2]{partial_sum(), replicated(1)};
  const Shape shapes[2]{Shape{4, 64}, Shape{128, 64}};
  const Prop::Result r = Prop::propagate(OpKind::kLinear, in, shapes, mesh);
  LSE_EXPECT(r.input_reshards.empty());
  LSE_EXPECT(r.output.axes[0].placement == Placement::kPartial);
}

LSE_TEST(a_bias_add_onto_a_sharded_activation_stays_sharded) {
  // The bias is replicated and rank 1, the activation rank 2: the shard dims
  // only agree once both are read from the trailing end.
  const DeviceMesh mesh = two_way();
  const Sharding in[2]{sharded(1), replicated(1)};
  const Shape shapes[2]{Shape{4, 128}, Shape{128}};
  const Prop::Result r = Prop::propagate(OpKind::kAdd, in, shapes, mesh);
  LSE_EXPECT(r.input_reshards.empty());
  LSE_EXPECT(r.output.axes[0].placement == Placement::kSharded);
  LSE_EXPECT_EQ(static_cast<int>(r.output.axes[0].tensor_dim), 1);
}

LSE_TEST(a_shard_of_a_broadcast_extent_is_gathered_not_assumed_replicated) {
  // [8,16] split on dim 1 times a [1] "split" on dim 0. The one element lives
  // on coordinate 0 alone — local_shape says so — so the other ranks have
  // nothing to broadcast and the operand has to be gathered. Treating an
  // extent-1 shard as replicated is silently wrong on every rank but the first.
  const DeviceMesh mesh({{"tp", 4}});
  const Sharding in[2]{sharded(1), sharded(0)};
  const Shape shapes[2]{Shape{8, 16}, Shape{1}};
  LSE_EXPECT_EQ(in[1].local_shape(Shape{1}, mesh, 0).dim(0), 1);
  LSE_EXPECT_EQ(in[1].local_shape(Shape{1}, mesh, 2).dim(0), 0);
  const Prop::Result r = Prop::propagate(OpKind::kMul, in, shapes, mesh);
  LSE_EXPECT_EQ(r.input_reshards.size(), 1u);
  LSE_EXPECT_EQ(r.input_reshards[0].first, 1u);
  LSE_EXPECT(r.input_reshards[0].second.kind == Reshard::Kind::kAllGather);
  // The real shard survives: only the degenerate one moved.
  LSE_EXPECT(r.output.axes[0].placement == Placement::kSharded);
  LSE_EXPECT_EQ(static_cast<int>(r.output.axes[0].tensor_dim), 1);
}

LSE_TEST(a_gathered_broadcast_extent_is_not_gathered_again_by_a_conflict) {
  // The degenerate operand moves once even when its neighbours disagree and
  // send the whole op down the conflict path.
  const DeviceMesh mesh({{"tp", 4}});
  const Sharding in[3]{sharded(0), sharded(1), sharded(0)};
  const Shape shapes[3]{Shape{8, 16}, Shape{8, 16}, Shape{1}};
  const Prop::Result r = Prop::propagate(OpKind::kWhere, in, shapes, mesh);
  std::size_t on_two = 0;
  for (const auto& [index, reshard] : r.input_reshards) {
    if (index == 2) ++on_two;
  }
  LSE_EXPECT_EQ(on_two, 1u);
  LSE_EXPECT(r.output.is_fully_replicated());
}

LSE_TEST(a_partial_operand_of_a_multiply_is_reduced_first) {
  const DeviceMesh mesh = two_way();
  const Sharding in[2]{partial_sum(), replicated(1)};
  const Shape shapes[2]{Shape{4, 128}, Shape{4, 128}};
  const Prop::Result r = Prop::propagate(OpKind::kMul, in, shapes, mesh);
  LSE_EXPECT_EQ(r.input_reshards.size(), 1u);
  LSE_EXPECT_EQ(r.input_reshards[0].first, 0u);
  LSE_EXPECT(r.input_reshards[0].second.kind == Reshard::Kind::kAllReduce);
  LSE_EXPECT(r.output.is_fully_replicated());
}

LSE_TEST(an_operand_is_never_resharded_twice_for_one_op) {
  // A partial selector and two operands split on dimensions that disagree: the
  // partial is reduced once, not once for being partial and again for the
  // conflict its neighbours caused.
  const DeviceMesh mesh = two_way();
  const Sharding in[3]{partial_sum(), sharded(0), sharded(1)};
  const Shape shapes[3]{Shape{4, 128}, Shape{4, 128}, Shape{4, 128}};
  const Prop::Result r = Prop::propagate(OpKind::kWhere, in, shapes, mesh);
  LSE_EXPECT_EQ(r.input_reshards.size(), 3u);
  std::size_t on_zero = 0;
  for (const auto& [index, reshard] : r.input_reshards) {
    if (index == 0) ++on_zero;
  }
  LSE_EXPECT_EQ(on_zero, 1u);
  LSE_EXPECT(r.output.is_fully_replicated());
}

LSE_TEST(a_sum_over_the_split_dimension_stays_partial) {
  const DeviceMesh mesh = two_way();
  const Sharding in[1]{sharded(1)};
  const Shape shapes[1]{Shape{4, 128}};
  const Prop::Result r = Prop::propagate(OpKind::kSum, in, shapes, mesh);
  LSE_EXPECT(r.input_reshards.empty());
  LSE_EXPECT(r.output.axes[0].placement == Placement::kPartial);
}

LSE_TEST(a_softmax_over_the_split_dimension_gathers_rather_than_guesses) {
  // The distributed form is an online pass with its own max and denominator
  // exchange — a different program, not this one plus a collective.
  const DeviceMesh mesh = two_way();
  const Sharding in[1]{sharded(1)};
  const Shape shapes[1]{Shape{4, 128}};
  const Prop::Result r = Prop::propagate(OpKind::kSoftmax, in, shapes, mesh);
  LSE_EXPECT_EQ(r.input_reshards.size(), 1u);
  LSE_EXPECT(r.input_reshards[0].second.kind == Reshard::Kind::kAllGather);
  LSE_EXPECT(r.output.is_fully_replicated());
}

LSE_TEST(a_shard_on_a_kept_dimension_shifts_when_the_reduced_one_drops) {
  const DeviceMesh mesh = two_way();
  const Sharding in[1]{sharded(1)};
  const Shape shapes[1]{Shape{4, 128}};
  const Prop::Result dropped =
      Prop::propagate(OpKind::kSum, in, shapes, mesh, /*reduce_dim=*/0);
  LSE_EXPECT(dropped.output.axes[0].placement == Placement::kSharded);
  LSE_EXPECT_EQ(static_cast<int>(dropped.output.axes[0].tensor_dim), 0);
  const Prop::Result kept =
      Prop::propagate(OpKind::kSum, in, shapes, mesh, 0, /*keepdims=*/true);
  LSE_EXPECT_EQ(static_cast<int>(kept.output.axes[0].tensor_dim), 1);
}

LSE_TEST(an_unmodelled_op_gathers_rather_than_inventing_a_sharding) {
  const DeviceMesh mesh = two_way();
  const Sharding in[1]{sharded(0)};
  const Shape shapes[1]{Shape{4, 128}};
  const Prop::Result r = Prop::propagate(OpKind::kTranspose, in, shapes, mesh);
  LSE_EXPECT_EQ(r.input_reshards.size(), 1u);
  LSE_EXPECT(r.output.is_fully_replicated());
}

LSE_TEST(replicated_to_sharded_moves_nothing) {
  // Every rank already holds the bytes and keeps its own slice, so this is
  // absent from the plan rather than present at zero cost.
  const DeviceMesh mesh = two_way();
  LSE_EXPECT(Prop::plan(replicated(1), sharded(0), mesh).empty());
}

LSE_TEST(a_partial_becomes_one_reduce_scatter_not_an_all_reduce_and_a_slice) {
  const DeviceMesh mesh = two_way();
  const std::vector<Reshard> p = Prop::plan(partial_sum(), sharded(1), mesh);
  LSE_EXPECT_EQ(p.size(), 1u);
  LSE_EXPECT(p[0].kind == Reshard::Kind::kReduceScatter);
  LSE_EXPECT_EQ(static_cast<int>(p[0].to_dim), 1);
}

LSE_TEST(changing_which_dimension_is_split_is_one_all_to_all) {
  const DeviceMesh mesh = two_way();
  const std::vector<Reshard> p = Prop::plan(sharded(0), sharded(1), mesh);
  LSE_EXPECT_EQ(p.size(), 1u);
  LSE_EXPECT(p[0].kind == Reshard::Kind::kAllToAll);
  LSE_EXPECT_EQ(static_cast<int>(p[0].from_dim), 0);
  LSE_EXPECT_EQ(static_cast<int>(p[0].to_dim), 1);
}

LSE_TEST(a_plan_reduces_while_the_tensor_is_still_split) {
  // Axis 0 needs a gather and axis 1 a reduction. Taken in axis order the
  // gather would run first and the reduction would then cost twice as much,
  // so the plan is ordered by what it does, not by which axis it is on.
  const DeviceMesh mesh({{"a", 2}, {"b", 2}});
  Sharding from;
  from.axis_count = 2;
  from.axes[0] = AxisPlacement::shard(0);
  from.axes[1] = AxisPlacement::partial();
  const std::vector<Reshard> p = Prop::plan(from, replicated(2), mesh);
  LSE_EXPECT_EQ(p.size(), 2u);
  LSE_EXPECT(p[0].kind == Reshard::Kind::kAllReduce);
  LSE_EXPECT_EQ(static_cast<int>(p[0].mesh_axis), 1);
  LSE_EXPECT(p[1].kind == Reshard::Kind::kAllGather);
  LSE_EXPECT_EQ(static_cast<int>(p[1].mesh_axis), 0);
}

namespace {

Array typed_host_array(const std::vector<float>& values, Shape shape,
                       DType dt) {
  Array a = Array::zeros(shape, dt);
  (void)a.eval();
  for (std::size_t i = 0; i < values.size(); ++i) {
    interpreter::store_element(*a.node(), i, values[i]);
  }
  return a;
}

// A packed plane cannot go through store_element: a lane is a bit pattern, and
// float would round every value past 2^24.
Array packed_host_array(const std::vector<std::uint32_t>& lanes, Shape shape) {
  Array a = Array::zeros(shape, DType::kU32);
  (void)a.eval();
  Node& n = *a.node();
  std::memcpy(interpreter::host_bytes(n), lanes.data(), lanes.size() * 4);
  n.host_dirty = true;
  n.device_dirty = false;
  return a;
}

// Reproducible, addressable by index, and — unlike a sampled sinusoid, whose
// products telescope — not self-cancelling over a long contraction. At K in
// the tens of thousands a sinusoidal fill sums to ~1e-3 while the terms it is
// built from sum to ~20, which leaves a reference dot product with no margin
// above the float32 error floor to discriminate against.
float spread(std::uint64_t i) {
  std::uint64_t z = i * 0x9e3779b97f4a7c15ull + 0x853c49e6748fea9bull;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
  z ^= z >> 31;
  return static_cast<float>(static_cast<std::int32_t>(z >> 32)) *
         (1.0f / 2147483648.0f);
}

struct QuantPlane {
  std::vector<std::uint32_t> packed;
  std::vector<float> scales;
  std::vector<float> biases;
  // What the codes actually encode. A reference contraction against this
  // cancels the quantization error instead of spending it on the tolerance.
  std::vector<float> dequantized;
  // The weights before quantization. Their distance from `dequantized` is the
  // error the storage format itself carries, which is the yardstick a kernel's
  // own error is measured against.
  std::vector<float> source;
  std::size_t lanes_per_row = 0;
  std::size_t groups_per_row = 0;

  // Root-mean-square of what quantizing the weights cost.
  [[nodiscard]] double round_trip_rmse() const {
    double acc = 0.0;
    for (std::size_t i = 0; i < source.size(); ++i) {
      const double d = static_cast<double>(source[i]) -
                       static_cast<double>(dequantized[i]);
      acc += d * d;
    }
    return source.empty() ? 0.0
                          : std::sqrt(acc / static_cast<double>(source.size()));
  }
};

QuantPlane quantize_plane(const quant::GroupAffine& q, std::int64_t n,
                          std::int64_t k) {
  QuantPlane p;
  p.lanes_per_row = q.packed_words(static_cast<std::size_t>(k));
  p.groups_per_row = q.group_count(static_cast<std::size_t>(k));
  p.packed.resize(static_cast<std::size_t>(n) * p.lanes_per_row);
  p.scales.resize(static_cast<std::size_t>(n) * p.groups_per_row);
  p.biases.resize(p.scales.size());
  p.dequantized.resize(static_cast<std::size_t>(n * k));
  p.source.resize(static_cast<std::size_t>(n * k));

  std::vector<float> row(static_cast<std::size_t>(k));
  std::vector<bfloat16_t> s(p.groups_per_row), b(p.groups_per_row);
  for (std::int64_t r = 0; r < n; ++r) {
    for (std::int64_t c = 0; c < k; ++c) {
      row[static_cast<std::size_t>(c)] =
          0.05f * spread(static_cast<std::uint64_t>(r * k + c));
    }
    const auto ro = static_cast<std::size_t>(r);
    std::copy(row.begin(), row.end(),
              p.source.begin() +
                  static_cast<std::ptrdiff_t>(ro * static_cast<std::size_t>(k)));
    q.quantize_row<bfloat16_t>(row.data(), static_cast<std::size_t>(k),
                               p.packed.data() + ro * p.lanes_per_row, s.data(),
                               b.data());
    q.dequantize_row<bfloat16_t>(p.packed.data() + ro * p.lanes_per_row,
                                 s.data(), b.data(), static_cast<std::size_t>(k),
                                 p.dequantized.data() +
                                     ro * static_cast<std::size_t>(k));
    for (std::size_t g = 0; g < p.groups_per_row; ++g) {
      p.scales[ro * p.groups_per_row + g] = s[g].to_float();
      p.biases[ro * p.groups_per_row + g] = b[g].to_float();
    }
  }
  return p;
}

// Whether a 4-bit contraction takes the integer path on the device this suite
// is running against. Not a preference — the kernel reads the same flag, and a
// device without the instruction decodes with the float codec instead.
bool integer_quant_path(int bits) {
  if (bits != 4) return false;
  Scheduler* sc = default_scheduler();
  if (sc == nullptr) return false;
  const auto* amd = backend::device_extension<backend::AmdDeviceInfo>(
      sc->backend().device_info());
  return amd != nullptr && amd->has_dot4_iu8;
}

// What one output of a quantized contraction is allowed to be off by, against
// a reference taken over the weights the codes encode.
//
// Two independent terms. The float one is the backward error of a float32 dot
// product: it tracks the sum of the term magnitudes, because the result
// cancels down to a small fraction of that and would set an unmeetable bar.
// The other is what the integer path's activation quantization can cost — it
// rounds each activation to int8 against the largest magnitude in its own
// group, so each carries at most half a step of that group and each product at
// most |w| times it.
//
// Worst case summed, not an RMS: this is the bound the kernel cannot exceed on
// any row, which is exactly what a per-tensor activation scale fails on a row
// with an outlier, since its step is the whole row's rather than the group's.
double admissible_error(const float* x, const float* w, std::int64_t k,
                        std::int64_t group_size, bool integer) {
  double mag = 0.0;
  for (std::int64_t c = 0; c < k; ++c) {
    mag += std::fabs(static_cast<double>(x[c]) * static_cast<double>(w[c]));
  }
  double quant = 0.0;
  if (integer) {
    for (std::int64_t g0 = 0; g0 < k; g0 += group_size) {
      double amax = 0.0;
      double wsum = 0.0;
      for (std::int64_t c = g0; c < g0 + group_size && c < k; ++c) {
        amax = std::max(amax, std::fabs(static_cast<double>(x[c])));
        wsum += std::fabs(static_cast<double>(w[c]));
      }
      quant += amax / 254.0 * wsum;
    }
  }
  return 1e-6 + 1e-6 * mag + quant;
}

// MLX's SwitchGLU layout: one plane of E * N rows, read as [E, N, lanes]. It is
// quantize_plane over E * N rows and nothing more, which is the point — the
// stack is not a new storage format, only a longer plane the kernel indexes
// into. Rows must differ across experts as well as within one, or a kernel that
// ignores the expert term passes.
QuantPlane quantize_stack(const quant::GroupAffine& q, std::int64_t experts,
                          std::int64_t n, std::int64_t k) {
  return quantize_plane(q, experts * n, k);
}

// Reference contraction against what the codes encode: row `r` of `x` times the
// matrix of expert `idx[r * keep + slot]`. Deliberately the same address
// arithmetic the kernel uses, in double.
std::vector<double> indexed_reference(const QuantPlane& p,
                                      const std::vector<float>& x,
                                      const std::vector<float>& idx,
                                      std::int64_t rows, std::int64_t keep,
                                      std::int64_t slot, std::int64_t n,
                                      std::int64_t k,
                                      std::vector<double>* magnitude) {
  std::vector<double> want(static_cast<std::size_t>(rows * n), 0.0);
  if (magnitude != nullptr) magnitude->assign(want.size(), 0.0);
  for (std::int64_t r = 0; r < rows; ++r) {
    const auto e =
        static_cast<std::int64_t>(idx[static_cast<std::size_t>(r * keep + slot)]);
    for (std::int64_t o = 0; o < n; ++o) {
      const std::int64_t wrow = e * n + o;
      double acc = 0.0, mag = 0.0;
      for (std::int64_t c = 0; c < k; ++c) {
        const double t =
            static_cast<double>(x[static_cast<std::size_t>(r * k + c)]) *
            static_cast<double>(
                p.dequantized[static_cast<std::size_t>(wrow * k + c)]);
        acc += t;
        mag += std::fabs(t);
      }
      want[static_cast<std::size_t>(r * n + o)] = acc;
      if (magnitude != nullptr) {
        (*magnitude)[static_cast<std::size_t>(r * n + o)] = mag;
      }
    }
  }
  return want;
}

// The three planes of a stack as graph inputs, at the checkpoint's own widths.
struct StackedArrays {
  Array packed, scales, biases;
};

StackedArrays stacked_arrays(const QuantPlane& p, std::int64_t experts,
                             std::int64_t n) {
  StackedArrays a;
  a.packed = packed_host_array(
      p.packed,
      Shape{experts, n, static_cast<std::int64_t>(p.lanes_per_row)});
  a.scales = typed_host_array(
      p.scales, Shape{experts, n, static_cast<std::int64_t>(p.groups_per_row)},
      DType::kBF16);
  a.biases = typed_host_array(
      p.biases, Shape{experts, n, static_cast<std::int64_t>(p.groups_per_row)},
      DType::kBF16);
  return a;
}

}  // namespace

// The whole point of the op: a weight that stays packed all the way to the
// register produces the same numbers as the contraction against the weights it
// encodes. The reference is computed here in double rather than with
// graph::linear, whose kernel narrows to f16 operands above a tile and would
// then be the less precise side of the comparison.
//
// Two bounds, and neither is a fixed epsilon.
//
// Per output, `admissible_error` — the worst case a correct implementation of
// this path can produce for this row and this column. That one is what catches
// a scaling bug: a per-tensor activation scale exceeds it by 30x on the
// outlier row below, because its step is the whole row's largest magnitude
// rather than the group's.
//
// Then an RMS bound against the weight round-trip RMSE of the bit width under
// test. A 4-bit contraction on a device with v_dot4_i32_iu8 quantizes the
// activation to int8 per group of 64 and contracts in integers, so it is not
// bit-comparable to an fma chain and never was going to be; what has to hold
// is that its own error stays inside the error the weight format it decodes
// already carries. Measured on gfx1151, as a fraction of e_w * ||x||:
//
//   4-bit, smooth row        0.087 - 0.109
//   4-bit, one 30x outlier   0.60  - 1.08
//   every other width        < 1e-5   (the float codec, f32 rounding only)
//
// so the bars are 1.5 and 0.02. The outlier row is the honest cost of the
// specified granularity: one activation 30x its neighbours spends most of the
// int8 range of its own group of 64, and the other 63 quantize against it.
//
// Every bit width MLX emits, at one row (decode) and at more rows than a tile
// (prefill): 3, 5 and 6 bits are the ones whose codes straddle two lanes. Two
// K's, because K = 128 leaves the 4-bit lane run entirely in the tail loop and
// K = 1024 is the first that fills the aligned one. Two activation fills,
// because a uniformly benign row cannot tell a per-group activation scale from
// a per-tensor one — the outlier row can.
LSE_TEST(quant_linear_matches_the_weights_it_encodes) {
  constexpr double kIntegerShare = 1.5;
  constexpr double kFloatShare = 0.02;
  for (int bits : {2, 3, 4, 5, 6, 8}) {
    for (std::int64_t kk : {std::int64_t{128}, std::int64_t{1024}}) {
      for (bool outlier : {false, true}) {
        for (std::int64_t rows : {std::int64_t{1}, std::int64_t{20}}) {
          auto spec = quant::GroupAffine::make(bits, 64);
          LSE_EXPECT(spec.ok());
          if (!spec.ok()) continue;
          const quant::GroupAffine q = *spec;

          constexpr std::int64_t kN = 32;
          const QuantPlane p = quantize_plane(q, kN, kk);

          std::vector<float> x(static_cast<std::size_t>(rows * kk));
          for (std::size_t i = 0; i < x.size(); ++i) {
            x[i] = 0.1f * std::sin(0.7f * static_cast<float>(i));
          }
          if (outlier) {
            // One element per row, 30x the rest of it. Its own group of 64
            // keeps its own scale, so every other group is unaffected; a
            // single scale for the row would quantize all of them against
            // this one number instead.
            for (std::int64_t r = 0; r < rows; ++r) {
              const auto at = static_cast<std::size_t>(
                  r * kk + (r * 37 + 11) % kk);
              x[at] = 3.0f;
            }
          }

          Array xa = host_array(x, Shape{rows, kk});
          Array pa = packed_host_array(
              p.packed, Shape{kN, static_cast<std::int64_t>(p.lanes_per_row)});
          Array sa = typed_host_array(
              p.scales, Shape{kN, static_cast<std::int64_t>(p.groups_per_row)},
              DType::kBF16);
          Array ba = typed_host_array(
              p.biases, Shape{kN, static_cast<std::int64_t>(p.groups_per_row)},
              DType::kBF16);
          // The plane reached the graph at its own width; nothing widened it.
          LSE_EXPECT(pa.dtype() == DType::kU32);

          Array got = quant_linear(xa, pa, sa, ba, bits, 64);
          LSE_EXPECT(got.node()->kind == OpKind::kQuantMatMul);
          LSE_EXPECT(got.node()->inputs.size() == 4);

          const std::vector<float> g = drain(got);
          LSE_EXPECT(g.size() == static_cast<std::size_t>(rows * kN));
          if (g.size() != static_cast<std::size_t>(rows * kN)) continue;

          // What an error the size of the weight format's own would do to
          // this contraction: e_w per weight, against this row of x, over a
          // dot product whose terms carry independent signs.
          const double e_w = p.round_trip_rmse();
          const bool integer = integer_quant_path(bits);
          double err = 0.0;
          double yardstick = 0.0;
          double worst = 0.0;
          for (std::int64_t r = 0; r < rows; ++r) {
            double xnorm = 0.0;
            for (std::int64_t c = 0; c < kk; ++c) {
              const double v = x[static_cast<std::size_t>(r * kk + c)];
              xnorm += v * v;
            }
            const double bar = e_w * std::sqrt(xnorm);
            for (std::int64_t o = 0; o < kN; ++o) {
              double want = 0.0;
              for (std::int64_t c = 0; c < kk; ++c) {
                want += static_cast<double>(x[static_cast<std::size_t>(r * kk + c)]) *
                        static_cast<double>(
                            p.dequantized[static_cast<std::size_t>(o * kk + c)]);
              }
              const double d =
                  static_cast<double>(g[static_cast<std::size_t>(r * kN + o)]) -
                  want;
              // The hard bound first: what any correct implementation of this
              // path can be off by, row by row and column by column.
              LSE_EXPECT_NEAR(
                  g[static_cast<std::size_t>(r * kN + o)], want,
                  admissible_error(&x[static_cast<std::size_t>(r * kk)],
                                   &p.dequantized[static_cast<std::size_t>(o * kk)],
                                   kk, 64, integer));
              err += d * d;
              yardstick += bar * bar;
              const double rel = std::fabs(d) / (bar + 1e-12);
              if (rel > worst) worst = rel;
            }
          }
          const auto count = static_cast<double>(rows * kN);
          const double err_rms = std::sqrt(err / count);
          const double bar_rms = std::sqrt(yardstick / count);
          const double share = integer ? kIntegerShare : kFloatShare;
          // 1e-6 is the float32 accumulation floor at these magnitudes; it is
          // what carries the widths whose weight error is smaller than it.
          const bool within = err_rms <= 1e-6 + share * bar_rms;
          LSE_EXPECT(within);
          if (!within) {
            std::printf(
                "       bits=%d K=%lld rows=%lld outlier=%d: rms %.3e "
                "e_w*|x| %.3e ratio %.3f worst %.2f\n",
                bits, static_cast<long long>(kk),
                static_cast<long long>(rows), outlier ? 1 : 0, err_rms,
                bar_rms, err_rms / (bar_rms + 1e-30), worst);
          }
        }
      }
    }
  }
}

// Output row r must be the contraction of activation row r. The kernel stages
// that row in LDS when it fits and reads it straight from global memory when
// it does not, and those are two different index expressions, so a K on each
// side of the LDS budget is the only thing that covers both. Qwen3.8-27B's
// mlp.down_proj (K = 17408) was the first shape in a shipped checkpoint to
// cross the line, onto a path that had dropped the row term: every token of a
// prefill pass came out as token 0. One row — decode — cannot see that, and
// neither can a K that always stages, which is why both were green.
LSE_TEST(quant_linear_row_survives_the_unstaged_path) {
  Scheduler* sc = default_scheduler();
  const std::uint32_t lds =
      sc != nullptr ? sc->backend().device_info().lds_bytes_per_workgroup : 0;

  constexpr int kBits = 8;
  constexpr std::int64_t kGroup = 64;
  constexpr std::int64_t kN = 16;
  constexpr std::int64_t kRows = 5;

  // Staging is `K * sizeof(float) <= lds_bytes`, so the smallest multiple of
  // the group size past `lds_bytes / 4` is the first K that cannot stage —
  // the narrowest crossing of the boundary, and cheap to reference in double.
  // A device that reports no budget stages unconditionally; 20480 keeps the
  // case honest against whatever device answers next.
  const std::int64_t unstaged =
      lds != 0 ? (static_cast<std::int64_t>(lds / 4) / kGroup + 1) * kGroup
               : std::int64_t{20480};

  auto spec = quant::GroupAffine::make(kBits, kGroup);
  LSE_EXPECT(spec.ok());
  if (!spec.ok()) return;
  const quant::GroupAffine q = *spec;

  for (const std::int64_t k : {std::int64_t{640}, unstaged}) {
    const QuantPlane p = quantize_plane(q, kN, k);

    // Rows must differ from one another, or a kernel that broadcasts one of
    // them passes.
    std::vector<float> x(static_cast<std::size_t>(kRows * k));
    for (std::int64_t r = 0; r < kRows; ++r) {
      for (std::int64_t c = 0; c < k; ++c) {
        x[static_cast<std::size_t>(r * k + c)] =
            0.1f * spread(static_cast<std::uint64_t>(r * k + c) + (1ull << 40));
      }
    }

    Array xa = host_array(x, Shape{kRows, k});
    Array pa = packed_host_array(
        p.packed, Shape{kN, static_cast<std::int64_t>(p.lanes_per_row)});
    Array sa = typed_host_array(
        p.scales, Shape{kN, static_cast<std::int64_t>(p.groups_per_row)},
        DType::kBF16);
    Array ba = typed_host_array(
        p.biases, Shape{kN, static_cast<std::int64_t>(p.groups_per_row)},
        DType::kBF16);

    Array got = quant_linear(xa, pa, sa, ba, kBits, kGroup);
    const std::vector<float> g = drain(got);
    LSE_EXPECT_EQ(g.size(), static_cast<std::size_t>(kRows * kN));
    if (g.size() != static_cast<std::size_t>(kRows * kN)) return;

    // A silent host fallback would make this a no-op on the one path it exists
    // to guard.
    if (sc != nullptr && sc->backend().emitter() != nullptr) {
      LSE_EXPECT_EQ(sc->last_trace().host_fallbacks, 0u);
      LSE_EXPECT_EQ(sc->last_trace().host_groups, 0u);
    }

    for (std::int64_t r = 0; r < kRows; ++r) {
      for (std::int64_t o = 0; o < kN; ++o) {
        double want = 0.0;
        double mag = 0.0;
        for (std::int64_t c = 0; c < k; ++c) {
          const double t =
              static_cast<double>(x[static_cast<std::size_t>(r * k + c)]) *
              static_cast<double>(
                  p.dequantized[static_cast<std::size_t>(o * k + c)]);
          want += t;
          mag += std::fabs(t);
        }
        // Backward-error bound for a float32 dot product: the tolerance tracks
        // the sum of magnitudes, not the result, which cancels down to a small
        // fraction of it and would set an unmeetable bar at this K.
        LSE_EXPECT_NEAR(g[static_cast<std::size_t>(r * kN + o)], want,
                        1e-6 + 1e-6 * mag);
      }
    }

    // The shape of the original failure: every output row byte-identical to
    // row 0. Named separately so a regression says which invariant broke.
    for (std::int64_t r = 1; r < kRows; ++r) {
      bool duplicates_row0 = true;
      for (std::int64_t o = 0; o < kN && duplicates_row0; ++o) {
        duplicates_row0 = g[static_cast<std::size_t>(r * kN + o)] ==
                          g[static_cast<std::size_t>(o)];
      }
      LSE_EXPECT(!duplicates_row0);
    }
  }
}

// The indexed form against every width MLX emits. Two things have to hold at
// once and each hides the other's failure: the contraction must equal the
// weights the codes encode (the quant half) and it must be the *selected*
// expert's weights (the index half). A kernel that always read expert 0 would
// pass a test whose rows all route to expert 0, and a kernel that decoded at
// the wrong width would pass a test that only compared expert ids.
LSE_TEST(quant_linear_indexed_matches_the_expert_it_selects) {
  constexpr std::int64_t kExperts = 6;
  constexpr std::int64_t kN = 24;
  constexpr std::int64_t kK = 128;
  constexpr std::int64_t kRows = 5;
  constexpr std::int64_t kKeep = 3;

  // Row r routes to a different expert in every slot, and no two rows share a
  // slot's expert. Expert 0 appears only in row 4's last slot, so a body that
  // dropped the index term reads a matrix nothing else asks for.
  const std::vector<float> idx{
      5, 3, 1,  //
      2, 4, 3,  //
      1, 5, 4,  //
      4, 2, 5,  //
      3, 1, 0,  //
  };

  for (int bits : {2, 3, 4, 5, 6, 8}) {
    auto spec = quant::GroupAffine::make(bits, 64);
    LSE_EXPECT(spec.ok());
    if (!spec.ok()) continue;
    const QuantPlane p = quantize_stack(*spec, kExperts, kN, kK);

    std::vector<float> x(static_cast<std::size_t>(kRows * kK));
    for (std::size_t i = 0; i < x.size(); ++i) {
      x[i] = 0.1f * spread(static_cast<std::uint64_t>(i) + (1ull << 33));
    }

    Array xa = host_array(x, Shape{kRows, kK});
    const StackedArrays w = stacked_arrays(p, kExperts, kN);
    Array ia = host_array(idx, Shape{kRows, kKeep});
    // The stack reached the graph as a rank-3 u32 plane; nothing widened it and
    // nothing unstacked it.
    LSE_EXPECT(w.packed.dtype() == DType::kU32);
    LSE_EXPECT(w.packed.shape().rank() == 3u);

    for (std::int64_t slot = 0; slot < kKeep; ++slot) {
      Array got = quant_linear_indexed(xa, w.packed, w.scales, w.biases, ia,
                                       static_cast<int>(slot), bits, 64);
      LSE_EXPECT(got.node()->kind == OpKind::kMoEDispatch);
      LSE_EXPECT(got.node()->inputs.size() == 5u);
      LSE_EXPECT(got.shape() == (Shape{kRows, kN}));

      const std::vector<float> g = drain(got);
      LSE_EXPECT_EQ(g.size(), static_cast<std::size_t>(kRows * kN));
      if (g.size() != static_cast<std::size_t>(kRows * kN)) continue;

      std::vector<double> mag;
      const std::vector<double> want = indexed_reference(
          p, x, idx, kRows, kKeep, slot, kN, kK, &mag);
      // Same admissible bound as the unstacked op, against the expert this
      // row actually routed to.
      const bool integer = integer_quant_path(bits);
      for (std::size_t i = 0; i < want.size(); ++i) {
        const auto r = static_cast<std::int64_t>(i) / kN;
        const auto e = static_cast<std::int64_t>(
            idx[static_cast<std::size_t>(r * kKeep + slot)]);
        const auto col = e * kN + static_cast<std::int64_t>(i) % kN;
        LSE_EXPECT_NEAR(
            g[i], want[i],
            admissible_error(&x[static_cast<std::size_t>(r * kK)],
                             &p.dequantized[static_cast<std::size_t>(col * kK)],
                             kK, 64, integer));
      }

      // And it is not some other expert's answer that happens to be close: the
      // same row against the wrong expert must differ everywhere.
      for (std::int64_t r = 0; r < kRows; ++r) {
        const auto e =
            static_cast<std::int64_t>(idx[static_cast<std::size_t>(
                r * kKeep + slot)]);
        const std::int64_t other = (e + 1) % kExperts;
        std::vector<float> wrong_idx(idx.size(), static_cast<float>(other));
        const std::vector<double> wrong = indexed_reference(
            p, x, wrong_idx, kRows, kKeep, slot, kN, kK, nullptr);
        bool all_close = true;
        for (std::int64_t o = 0; o < kN && all_close; ++o) {
          const auto i = static_cast<std::size_t>(r * kN + o);
          all_close = std::fabs(static_cast<double>(g[i]) - wrong[i]) <
                      admissible_error(
                          &x[static_cast<std::size_t>(r * kK)],
                          &p.dequantized[static_cast<std::size_t>(
                              (e * kN + o) * kK)],
                          kK, 64, integer);
        }
        LSE_EXPECT(!all_close);
      }
    }
  }
}

// The row-offset hazard, one op over. quant_linear shipped with the row term
// missing on its unstaged arm and every prefill token came out as token 0 —
// fluent, and uniformly wrong. The indexed body is the same two index
// expressions (zero against a staged panel, `row * K` against global memory)
// so it carries the identical hazard, and only a K on each side of the LDS
// budget covers both. Routing makes it worse here than there: each row also
// reads a different expert, so a dropped row term silently contracts row 0
// against row r's expert.
LSE_TEST(quant_linear_indexed_row_survives_the_unstaged_path) {
  Scheduler* sc = default_scheduler();
  const std::uint32_t lds =
      sc != nullptr ? sc->backend().device_info().lds_bytes_per_workgroup : 0;

  constexpr int kBits = 8;
  constexpr std::int64_t kGroup = 64;
  constexpr std::int64_t kExperts = 4;
  constexpr std::int64_t kN = 16;
  constexpr std::int64_t kRows = 5;
  constexpr std::int64_t kKeep = 2;
  constexpr std::int64_t kSlot = 1;

  // Smallest multiple of the group size past `lds_bytes / 4`: the first K that
  // cannot stage, which is the narrowest crossing of the boundary.
  const std::int64_t unstaged =
      lds != 0 ? (static_cast<std::int64_t>(lds / 4) / kGroup + 1) * kGroup
               : std::int64_t{20480};

  auto spec = quant::GroupAffine::make(kBits, kGroup);
  LSE_EXPECT(spec.ok());
  if (!spec.ok()) return;

  // Every row a different expert, so row confusion and expert confusion cannot
  // cancel into a right answer.
  const std::vector<float> idx{0, 3, 1, 2, 2, 1, 3, 0, 0, 2};

  for (const std::int64_t k : {std::int64_t{640}, unstaged}) {
    const QuantPlane p = quantize_stack(*spec, kExperts, kN, k);

    std::vector<float> x(static_cast<std::size_t>(kRows * k));
    for (std::int64_t r = 0; r < kRows; ++r) {
      for (std::int64_t c = 0; c < k; ++c) {
        x[static_cast<std::size_t>(r * k + c)] =
            0.1f * spread(static_cast<std::uint64_t>(r * k + c) + (1ull << 41));
      }
    }

    Array xa = host_array(x, Shape{kRows, k});
    const StackedArrays w = stacked_arrays(p, kExperts, kN);
    Array ia = host_array(idx, Shape{kRows, kKeep});

    Array got = quant_linear_indexed(xa, w.packed, w.scales, w.biases, ia,
                                     static_cast<int>(kSlot), kBits, kGroup);
    const std::vector<float> g = drain(got);
    LSE_EXPECT_EQ(g.size(), static_cast<std::size_t>(kRows * kN));
    if (g.size() != static_cast<std::size_t>(kRows * kN)) return;

    // A silent host fallback would make this a no-op on the one path it exists
    // to guard.
    if (sc != nullptr && sc->backend().emitter() != nullptr) {
      LSE_EXPECT_EQ(sc->last_trace().host_fallbacks, 0u);
      LSE_EXPECT_EQ(sc->last_trace().host_groups, 0u);
    }

    std::vector<double> mag;
    const std::vector<double> want =
        indexed_reference(p, x, idx, kRows, kKeep, kSlot, kN, k, &mag);
    for (std::size_t i = 0; i < want.size(); ++i) {
      LSE_EXPECT_NEAR(g[i], want[i], 1e-6 + 1e-6 * mag[i]);
    }

    // The exact shape of the original failure, named separately so a regression
    // says which invariant broke. Rows 0 and 4 route to the same expert, so a
    // dropped row term makes them identical while every other pair still
    // differs by its expert.
    bool duplicates_row0 = true;
    for (std::int64_t o = 0; o < kN && duplicates_row0; ++o) {
      duplicates_row0 = g[static_cast<std::size_t>(4 * kN + o)] ==
                        g[static_cast<std::size_t>(o)];
    }
    LSE_EXPECT(!duplicates_row0);
  }
}

// The device kernel against the host reference, on the same graph and the same
// bytes. This is the only oracle a mis-decoded router has: the interpreter's
// arm addresses the stack in place exactly as the emitted body does, so a
// disagreement is a decode bug rather than a layout one. Reported as a
// magnitude-relative worst case, because a plain absolute difference at K in
// the thousands says more about float32 than about the kernel.
LSE_TEST(quant_linear_indexed_agrees_with_the_host_reference) {
  Scheduler* sc = default_scheduler();
  if (sc == nullptr || sc->backend().emitter() == nullptr) return;

  constexpr std::int64_t kExperts = 8;
  constexpr std::int64_t kN = 32;
  constexpr std::int64_t kK = 512;  // the 35B's hidden_size
  constexpr std::int64_t kRows = 4;
  constexpr std::int64_t kKeep = 2;
  const std::vector<float> idx{7, 2, 0, 5, 3, 6, 6, 1};

  // 6 and 8 bits are the two widths the MoE checkpoints actually store, and in
  // the 6-bit one they appear together: experts at 6, routers overridden to 8.
  for (int bits : {6, 8}) {
    auto spec = quant::GroupAffine::make(bits, 64);
    LSE_EXPECT(spec.ok());
    if (!spec.ok()) continue;
    const QuantPlane p = quantize_stack(*spec, kExperts, kN, kK);

    std::vector<float> x(static_cast<std::size_t>(kRows * kK));
    for (std::size_t i = 0; i < x.size(); ++i) {
      x[i] = 0.1f * spread(static_cast<std::uint64_t>(i) + (1ull << 37));
    }

    for (std::int64_t slot = 0; slot < kKeep; ++slot) {
      const auto run = [&](Scheduler::Mode mode) {
        const auto prev = sc->mode();
        sc->set_mode(mode);
        Array xa = host_array(x, Shape{kRows, kK});
        const StackedArrays w = stacked_arrays(p, kExperts, kN);
        Array ia = host_array(idx, Shape{kRows, kKeep});
        Array got = quant_linear_indexed(xa, w.packed, w.scales, w.biases, ia,
                                         static_cast<int>(slot), bits, 64);
        std::vector<float> out = drain(got);
        const std::uint64_t fallbacks = sc->last_trace().host_fallbacks;
        sc->set_mode(prev);
        // The device run must really have gone to the device, or the two sides
        // are the same code and the comparison is vacuous.
        if (mode != Scheduler::Mode::kHostOnly) LSE_EXPECT_EQ(fallbacks, 0u);
        return out;
      };
      const std::vector<float> dev = run(Scheduler::Mode::kDeviceFirst);
      const std::vector<float> host = run(Scheduler::Mode::kHostOnly);
      LSE_EXPECT_EQ(dev.size(), host.size());
      if (dev.size() != host.size()) continue;

      std::vector<double> mag;
      const std::vector<double> want =
          indexed_reference(p, x, idx, kRows, kKeep, slot, kN, kK, &mag);

      double worst_rel = 0.0, worst_dev = 0.0, worst_host = 0.0;
      for (std::size_t i = 0; i < dev.size(); ++i) {
        const double scale = mag[i] > 0.0 ? mag[i] : 1.0;
        const double d = dev[i];
        const double h = host[i];
        worst_rel = std::max(worst_rel, std::fabs(d - h) / scale);
        worst_dev = std::max(worst_dev, std::fabs(d - want[i]) / scale);
        worst_host = std::max(worst_host, std::fabs(h - want[i]) / scale);
      }
      std::printf("       %d-bit slot %d: device-vs-host %.3e, "
                  "device-vs-double %.3e, host-vs-double %.3e (K=%d)\n",
                  bits, static_cast<int>(slot), worst_rel, worst_dev,
                  worst_host, static_cast<int>(kK));
      // Both sides sum the same terms in a different order in f32; the bound is
      // the backward error of the dot product, relative to the sum of term
      // magnitudes rather than to the cancelled result.
      LSE_EXPECT(worst_rel < 1e-6);
      LSE_EXPECT(worst_dev < 1e-6);
      LSE_EXPECT(worst_host < 1e-6);
    }
  }
}

// A stacked weight carrying its own planes picks the routed quantized
// contraction, the way a rank-2 one picks quant_linear. That is the seam:
// ops::routed_experts says linear_indexed once and the storage format decides,
// so no model file learns a second spelling.
LSE_TEST(a_stacked_group_affine_weight_picks_its_own_contraction) {
  constexpr std::int64_t kExperts = 4, kN = 8, kK = 64;
  constexpr int kBits = 6, kGroup = 64;
  const std::int64_t lanes = kK * kBits / 32;

  Array x = host_array(std::vector<float>(kK, 0.5f), Shape{1, kK});
  Array packed = packed_host_array(
      std::vector<std::uint32_t>(
          static_cast<std::size_t>(kExperts * kN * lanes), 0u),
      Shape{kExperts, kN, lanes});
  Array sa = typed_host_array(
      std::vector<float>(static_cast<std::size_t>(kExperts * kN), 1.0f),
      Shape{kExperts, kN, 1}, DType::kBF16);
  auto planes = std::make_shared<QuantPlanes>();
  planes->scales = sa.node();
  planes->biases = sa.node();
  planes->bits = kBits;
  planes->group_size = kGroup;
  planes->in_features = kK;
  packed.node()->quant = planes;

  // The expert axis survives: only the last axis of the plane counts lanes, so
  // a config check against {experts, out, in} can still be made.
  LSE_EXPECT(weight_shape(packed) == (Shape{kExperts, kN, kK}));

  Array idx = host_array({2.0f, 0.0f}, Shape{1, 2});
  Array y = linear_indexed(x, packed, idx, 0);
  LSE_EXPECT(y.node()->kind == OpKind::kMoEDispatch);
  LSE_EXPECT(y.node()->inputs.size() == 5u);
  // iattrs[0] stays the slot, as it is for the dense form; the geometry follows
  // it. All four are mixed into the emitted device function's name, which is
  // what gives a 6-bit expert and an 8-bit router distinct bodies.
  LSE_EXPECT_EQ(y.node()->iattrs[0], 0);
  LSE_EXPECT_EQ(y.node()->iattrs[1], kBits);
  LSE_EXPECT_EQ(y.node()->iattrs[2], kGroup);

  // A dense stack still takes the dense op: the fork is on the weight, not on
  // a flag the caller sets.
  Array dense = host_array(
      std::vector<float>(static_cast<std::size_t>(kExperts * kN * kK), 0.25f),
      Shape{kExperts, kN, kK});
  Array d = linear_indexed(x, dense, idx, 0);
  LSE_EXPECT(d.node()->kind == OpKind::kMoEDispatch);
  LSE_EXPECT(d.node()->inputs.size() == 3u);
}

// The 6-bit checkpoint's landmine end to end: 80 modules are overridden to 8
// bits and the rest are 6, so two contractions in the same layer must carry
// different geometry. Mis-decoding a router permutes which experts win and the
// model still reads fluently, so the only place this can be caught is here.
LSE_TEST(an_overridden_width_reaches_the_routed_contraction) {
  constexpr std::int64_t kExperts = 4, kN = 8, kK = 64;
  Array x = host_array(std::vector<float>(kK, 0.5f), Shape{1, kK});
  Array idx = host_array({1.0f}, Shape{1, 1});

  const auto build = [&](int bits) {
    const std::int64_t lanes = kK * bits / 32;
    Array packed = packed_host_array(
        std::vector<std::uint32_t>(
            static_cast<std::size_t>(kExperts * kN * lanes), 0u),
        Shape{kExperts, kN, lanes});
    Array sa = typed_host_array(
        std::vector<float>(static_cast<std::size_t>(kExperts * kN), 1.0f),
        Shape{kExperts, kN, 1}, DType::kBF16);
    auto planes = std::make_shared<QuantPlanes>();
    planes->scales = sa.node();
    planes->biases = sa.node();
    planes->bits = bits;
    planes->group_size = 64;
    planes->in_features = kK;
    packed.node()->quant = planes;
    return linear_indexed(x, packed, idx, 0);
  };

  Array at6 = build(6);
  Array at8 = build(8);
  LSE_EXPECT_EQ(at6.node()->iattrs[1], 6);
  LSE_EXPECT_EQ(at8.node()->iattrs[1], 8);
  // Same logical shape from two different plane widths, which is what the
  // shapes-vs-config cross-check in the loader depends on.
  LSE_EXPECT(at6.shape() == at8.shape());
  LSE_EXPECT(at6.node()->inputs[1]->shape.dim(2) == 12);
  LSE_EXPECT(at8.node()->inputs[1]->shape.dim(2) == 16);

  // A plane whose lane count solves to a different width than the attrs claim
  // is refused rather than run at the claimed one — the case an ignored
  // override produces.
  Array wrong = packed_host_array(
      std::vector<std::uint32_t>(
          static_cast<std::size_t>(kExperts * kN * 12), 0u),
      Shape{kExperts, kN, 12});
  Array sa = typed_host_array(
      std::vector<float>(static_cast<std::size_t>(kExperts * kN), 1.0f),
      Shape{kExperts, kN, 1}, DType::kBF16);
  Array bad = quant_linear_indexed(x, wrong, sa, sa, idx, 0, 8, 64);
  LSE_EXPECT(!bad.eval().ok());
}

// The gather half of the same op. A tied head reads this table as a matrix and
// as rows, so both readers have to agree with the codes on disk.
LSE_TEST(quant_embedding_gathers_the_rows_it_encodes) {
  for (int bits : {2, 3, 4, 5, 6, 8}) {
    auto spec = quant::GroupAffine::make(bits, 64);
    LSE_EXPECT(spec.ok());
    if (!spec.ok()) continue;
    const quant::GroupAffine q = *spec;

    constexpr std::int64_t kVocab = 9, kDim = 128;
    const auto lanes_per_row = q.packed_words(kDim);
    const auto groups_per_row = q.group_count(kDim);

    std::vector<std::uint32_t> packed(
        static_cast<std::size_t>(kVocab) * lanes_per_row);
    std::vector<float> scales(static_cast<std::size_t>(kVocab) * groups_per_row);
    std::vector<float> biases(scales.size());
    std::vector<float> table(static_cast<std::size_t>(kVocab * kDim));

    for (std::int64_t r = 0; r < kVocab; ++r) {
      std::vector<float> row(static_cast<std::size_t>(kDim));
      for (std::int64_t c = 0; c < kDim; ++c) {
        row[static_cast<std::size_t>(c)] =
            0.05f * std::cos(0.31f * static_cast<float>(r * kDim + c));
      }
      const auto ro = static_cast<std::size_t>(r);
      std::vector<bfloat16_t> s(groups_per_row), b(groups_per_row);
      q.quantize_row<bfloat16_t>(row.data(), static_cast<std::size_t>(kDim),
                                 packed.data() + ro * lanes_per_row, s.data(),
                                 b.data());
      q.dequantize_row<bfloat16_t>(packed.data() + ro * lanes_per_row, s.data(),
                                   b.data(), static_cast<std::size_t>(kDim),
                                   table.data() + ro * static_cast<std::size_t>(kDim));
      for (std::size_t g = 0; g < groups_per_row; ++g) {
        scales[ro * groups_per_row + g] = s[g].to_float();
        biases[ro * groups_per_row + g] = b[g].to_float();
      }
    }

    Array pa = packed_host_array(
        packed, Shape{kVocab, static_cast<std::int64_t>(lanes_per_row)});
    Array sa = typed_host_array(
        scales, Shape{kVocab, static_cast<std::int64_t>(groups_per_row)},
        DType::kBF16);
    Array ba = typed_host_array(
        biases, Shape{kVocab, static_cast<std::int64_t>(groups_per_row)},
        DType::kBF16);

    const std::vector<float> want_ids{8.0f, 0.0f, 5.0f, 3.0f};
    Array ids = host_array(want_ids, Shape{1, 4});

    // The table carries its own planes, so the model kernel asks for an
    // embedding and the storage format picks the op.
    auto planes = std::make_shared<QuantPlanes>();
    planes->scales = sa.node();
    planes->biases = ba.node();
    planes->bits = bits;
    planes->group_size = 64;
    planes->in_features = kDim;
    pa.node()->quant = planes;

    LSE_EXPECT(weight_shape(pa) == (Shape{kVocab, kDim}));
    Array got = embedding(pa, ids);
    LSE_EXPECT(got.node()->kind == OpKind::kQuantEmbedding);
    LSE_EXPECT(got.node()->inputs.size() == 4);
    LSE_EXPECT(got.shape() == (Shape{1, 4, kDim}));

    const std::vector<float> g = drain(got);
    LSE_EXPECT(g.size() == want_ids.size() * static_cast<std::size_t>(kDim));
    for (std::size_t t = 0; t < want_ids.size(); ++t) {
      const auto row = static_cast<std::size_t>(want_ids[t]);
      for (std::int64_t c = 0; c < kDim; ++c) {
        LSE_EXPECT_NEAR(g[t * static_cast<std::size_t>(kDim) +
                          static_cast<std::size_t>(c)],
                        table[row * static_cast<std::size_t>(kDim) +
                              static_cast<std::size_t>(c)],
                        1e-6);
      }
    }
  }
}

// A group-affine row whose scale plane does not cover it exactly is the one
// shape error that produces plausible numbers rather than a crash: `i /
// group_size` walks off the end of the row's own scales and into the next
// row's, and on the last row past the plane entirely. The device kernels
// already decline these; so must the host arm, or a shape the loader would
// reject reaches it as a fallback and answers with someone else's scales.
LSE_TEST(a_scale_plane_that_does_not_cover_the_row_is_refused) {
  constexpr int kBits = 4, kGroup = 64;

  // 12 lanes of 4-bit codes is 96 weights: one whole group of 64 and half of
  // another. Truncating division makes 96/64 == 1, so a one-group plane looks
  // right to any check that does not multiply back.
  {
    std::vector<std::uint32_t> packed(4 * 12, 0x11111111u);
    std::vector<float> sc(4, 1.0f), bi(4, 0.0f);
    Array pa = packed_host_array(packed, Shape{4, 12});
    Array sa = typed_host_array(sc, Shape{4, 1}, DType::kBF16);
    Array ba = typed_host_array(bi, Shape{4, 1}, DType::kBF16);
    Array x = host_array(std::vector<float>(96, 1.0f), Shape{1, 96});
    Array y = quant_linear(x, pa, sa, ba, kBits, kGroup);
    LSE_EXPECT(!y.eval().ok());
  }

  // The gather half: 32 lanes is 256 weights needing four groups, and two are
  // offered. Groups 2 and 3 would resolve into the next vocabulary row.
  {
    std::vector<std::uint32_t> packed(4 * 32, 0x11111111u);
    std::vector<float> sc{1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<float> bi(8, 0.0f);
    Array pa = packed_host_array(packed, Shape{4, 32});
    Array sa = typed_host_array(sc, Shape{4, 2}, DType::kBF16);
    Array ba = typed_host_array(bi, Shape{4, 2}, DType::kBF16);
    Array ids = host_array({0.0f}, Shape{1});
    Array e = quant_embedding(pa, sa, ba, ids, kBits, kGroup);
    LSE_EXPECT(!e.eval().ok());
  }

  // An activation row wider than the weight is the same error one operand
  // over, and the worst of the three: the row count is derived from the
  // plane's width while the output was sized from x's leading axes, so a
  // 512-wide row against a 256-wide weight writes two rows into a one-row
  // output.
  {
    std::vector<std::uint32_t> packed(4 * 32, 0x11111111u);
    std::vector<float> sc(4 * 4, 1.0f), bi(4 * 4, 0.0f);
    Array pa = packed_host_array(packed, Shape{4, 32});
    Array sa = typed_host_array(sc, Shape{4, 4}, DType::kBF16);
    Array ba = typed_host_array(bi, Shape{4, 4}, DType::kBF16);
    for (std::int64_t width : {std::int64_t{512}, std::int64_t{128}}) {
      Array x = host_array(
          std::vector<float>(static_cast<std::size_t>(width), 1.0f),
          Shape{1, width});
      Array y = quant_linear(x, pa, sa, ba, kBits, kGroup);
      LSE_EXPECT(!y.eval().ok());
    }
  }

  // A biases plane that disagrees with the scales plane is the same class of
  // error one axis over.
  {
    std::vector<std::uint32_t> packed(4 * 16, 0x11111111u);
    std::vector<float> sc(4 * 2, 1.0f), bi(4, 0.0f);
    Array pa = packed_host_array(packed, Shape{4, 16});
    Array sa = typed_host_array(sc, Shape{4, 2}, DType::kBF16);
    Array ba = typed_host_array(bi, Shape{4, 1}, DType::kBF16);
    Array x = host_array(std::vector<float>(128, 1.0f), Shape{1, 128});
    Array y = quant_linear(x, pa, sa, ba, kBits, kGroup);
    LSE_EXPECT(!y.eval().ok());
  }
}

// linear() and embedding() route on the weight, not on a flag the caller sets,
// which is what lets every lse::ops weight struct stay a plain Array.
LSE_TEST(a_group_affine_weight_picks_its_own_contraction) {
  Array x = host_array(std::vector<float>(64, 0.5f), Shape{1, 64});
  Array plain = host_array(std::vector<float>(64 * 8, 0.25f), Shape{8, 64});
  LSE_EXPECT(linear(x, plain).node()->kind == OpKind::kLinear);
  LSE_EXPECT(weight_shape(plain) == (Shape{8, 64}));

  Array packed = packed_host_array(std::vector<std::uint32_t>(8 * 8, 0u),
                                   Shape{8, 8});
  Array sa = typed_host_array(std::vector<float>(8, 1.0f), Shape{8, 1},
                              DType::kBF16);
  auto planes = std::make_shared<QuantPlanes>();
  planes->scales = sa.node();
  planes->biases = sa.node();
  planes->bits = 4;
  planes->group_size = 64;
  planes->in_features = 64;
  packed.node()->quant = planes;

  LSE_EXPECT(weight_shape(packed) == (Shape{8, 64}));
  LSE_EXPECT(linear(x, packed).node()->kind == OpKind::kQuantMatMul);
}

LSE_TEST(a_scheduler_holds_a_device_set_and_stamps_what_it_evaluates) {
  Scheduler* sched = default_scheduler();
  LSE_EXPECT(sched != nullptr);
  if (sched == nullptr) return;

  // The set is the scheduler's device vocabulary now. One member is the whole
  // set on this box; what matters is that it IS a set and that the primary is
  // the backend everything used to reach directly.
  backend::IDeviceSet& set = sched->devices();
  LSE_EXPECT(set.size() >= 1);
  LSE_EXPECT(set.primary() < set.size());
  LSE_EXPECT(&set.device(set.primary()) == &sched->backend());

  Array a = host_array({1.0f, 2.0f, 3.0f, 4.0f}, Shape{4});
  Array b = host_array({10.0f, 20.0f, 30.0f, 40.0f}, Shape{4});
  Array c = add(a, b);
  LSE_EXPECT_OK(c.materialize());

  // Every buffer the evaluation allocated names the device that allocated it,
  // which is what makes "may this kernel read these bytes" answerable at all.
  const backend::DeviceIndex primary = set.residency(set.primary());
  LSE_EXPECT(c.node()->buffer.valid());
  if (primary.bound()) {
    LSE_EXPECT(c.node()->buffer.residency == primary);
    LSE_EXPECT_EQ(set.member_of(c.node()->buffer.residency), set.primary());
  }
  LSE_EXPECT_OK(set.may_read(c.node()->buffer.residency, set.primary()));

  const std::vector<float> out = drain(c);
  LSE_EXPECT_EQ(out.size(), std::size_t{4});
  LSE_EXPECT_NEAR(out[0], 11.0f, 1e-6);
  LSE_EXPECT_NEAR(out[3], 44.0f, 1e-6);
}

LSE_TEST(a_scheduler_over_a_bare_backend_is_the_set_that_backend_is) {
  auto be = backend::create_backend("cpu");
  LSE_EXPECT(be.ok());
  if (!be.ok()) return;
  LSE_EXPECT_OK((*be)->init(0));

  Scheduler sched(**be);
  LSE_EXPECT_EQ(sched.devices().size(), std::size_t{1});
  LSE_EXPECT(&sched.backend() == be->get());
  LSE_EXPECT(sched.devices().residency(0) == (*be)->device_index());
}


// ---------------------------------------------------------------------------
// The timing report. Two defects motivated these: partition_ns bracketed the
// whole dispatch loop and so contained emit, compile and submit, and emit_ns
// bracketed the JIT and so contained compile. Both made the report sum to more
// than the step it described, and the second made a cold run's "emit" 98% a
// compile that was also printed on its own line.
// ---------------------------------------------------------------------------

namespace {

std::uint64_t attributed_span_ns(const Scheduler::Trace::Spans& s) {
  return s.partition.ns + s.schedule.ns + s.emit.ns + s.jit_lookup.ns +
         s.jit_compile.ns + s.bind.ns + s.submit.ns + s.host_wait.ns +
         s.readback.ns + s.host_exec.ns;
}

// Every host span in the engine is steady_clock, and each number says so on its
// own rather than by convention.
void expect_all_host_steady(const Scheduler::Trace::Spans& s) {
  const backend::ClockDomain host = backend::ClockDomain::kHostSteady;
  LSE_EXPECT(s.partition.clock == host);
  LSE_EXPECT(s.schedule.clock == host);
  LSE_EXPECT(s.emit.clock == host);
  LSE_EXPECT(s.jit_lookup.clock == host);
  LSE_EXPECT(s.jit_compile.clock == host);
  LSE_EXPECT(s.bind.clock == host);
  LSE_EXPECT(s.submit.clock == host);
  LSE_EXPECT(s.host_wait.clock == host);
  LSE_EXPECT(s.readback.clock == host);
  LSE_EXPECT(s.host_exec.clock == host);
  LSE_EXPECT(s.step.clock == host);
  LSE_EXPECT(s.unattributed.clock == host);
}

}  // namespace

LSE_TEST(the_spans_of_a_step_do_not_overlap_and_the_remainder_is_reported) {
  Scheduler* sched = default_scheduler();
  if (sched == nullptr) {
    std::printf("       no scheduler: skipped\n");
    return;
  }
  Array x = Array::full(Shape{1, 64}, DType::kF32, 1.0f);
  Array w = Array::full(Shape{64, 64}, DType::kF32, 0.01f);
  Array y = silu(linear(x, w));
  LSE_EXPECT_OK(y.eval());

  const Scheduler::Trace::Spans& s = sched->last_trace().spans;
  expect_all_host_steady(s);

  // The step happened, so it took time.
  LSE_EXPECT(s.step.ns > 0);
  // DISJOINT: the parts plus the reported remainder are the step exactly. Under
  // the old counters this identity was off by the whole of emit + launch, which
  // partition_ns also contained.
  const std::uint64_t parts = attributed_span_ns(s);
  LSE_EXPECT_EQ(parts + s.unattributed.ns, s.step.ns);
  // Which also means no single span can exceed the step it sits in.
  LSE_EXPECT(s.partition.ns <= s.step.ns);
  LSE_EXPECT(s.emit.ns <= s.step.ns);
  LSE_EXPECT(s.jit_compile.ns <= s.step.ns);
  LSE_EXPECT(s.host_wait.ns <= s.step.ns);

  std::printf("       step=%.3f ms attributed=%.3f unattributed=%.3f (%.1f%%)\n",
              s.step.seconds() * 1e3,
              static_cast<double>(parts) * 1e-6,
              s.unattributed.seconds() * 1e3,
              100.0 * static_cast<double>(s.unattributed.ns) /
                  static_cast<double>(s.step.ns));
}

// The other exit: a host-only step never reaches the phase dispatch loop, so it
// closes its pre-dispatch region at Partitioner::partition instead and spends
// its time in the interpreter. Both paths have to balance, and the interpreter's
// time has to be a span rather than an unexplained remainder.
LSE_TEST(a_host_only_step_balances_too_and_names_its_interpreter_time) {
  auto be = backend::create_backend("cpu");
  LSE_EXPECT(be.ok());
  if (!be.ok()) return;
  LSE_EXPECT_OK((*be)->init(0));

  Scheduler sched(**be);
  sched.set_mode(Scheduler::Mode::kHostOnly);
  Array x = Array::full(Shape{256}, DType::kF32, 1.5f);
  Array y = silu(x + x) * x;
  const NodePtr roots[] = {y.node()};
  LSE_EXPECT_OK(sched.eval(roots, true));

  const Scheduler::Trace::Spans& s = sched.last_trace().spans;
  expect_all_host_steady(s);
  LSE_EXPECT_EQ(attributed_span_ns(s) + s.unattributed.ns, s.step.ns);
  // Nothing was emitted, compiled or submitted on this path.
  LSE_EXPECT_EQ(s.emit.ns, std::uint64_t{0});
  LSE_EXPECT_EQ(s.jit_compile.ns, std::uint64_t{0});
  LSE_EXPECT_EQ(s.submit.ns, std::uint64_t{0});
  LSE_EXPECT_EQ(s.host_wait.ns, std::uint64_t{0});
  // The interpreter ran, and it is named rather than left in the remainder.
  LSE_EXPECT(s.host_exec.ns > 0);
  LSE_EXPECT(s.partition.ns > 0);
  std::printf("       host-only step=%.3f ms partition=%.3f host_exec=%.3f "
              "unattributed=%.3f\n",
              s.step.seconds() * 1e3, s.partition.seconds() * 1e3,
              s.host_exec.seconds() * 1e3, s.unattributed.seconds() * 1e3);
}

LSE_TEST(a_cold_compile_is_charged_to_compile_and_never_to_emit) {
  // The JIT reads LSE_CACHE_DIR when a JitCache is built, which is the first
  // time a scheduler dispatches. So this needs a scheduler of its own over a
  // fresh directory; deleting the cache the rest of the suite shares would make
  // every later case pay for it.
  auto be = backend::create_default_backend();
  if (!be.ok()) {
    std::printf("       no backend: skipped\n");
    return;
  }
  auto owned = be.release();
  if (const Status up = owned->init(0); !up.ok()) {
    std::printf("       backend declined: %s\n", up.message().c_str());
    return;
  }
  if (owned->emitter() == nullptr || owned->compiler() == nullptr ||
      !owned->compiler()->available()) {
    std::printf("       no device compiler: skipped\n");
    return;
  }

  const std::string dir = "/tmp/lse-cold-spans-" + std::to_string(::getpid());
  std::string saved;
  const bool had = std::getenv("LSE_CACHE_DIR") != nullptr;
  if (had) saved = std::getenv("LSE_CACHE_DIR");
  ::setenv("LSE_CACHE_DIR", dir.c_str(), 1);

  Scheduler cold(*owned);
  // A shape the shared cache is unlikely to hold, so the compile is real even
  // if the fresh directory somehow is not.
  Array x = Array::full(Shape{1, 37}, DType::kF32, 1.0f);
  Array y = silu(x + x) * x;
  const NodePtr roots[] = {y.node()};
  const Status ran = cold.eval(roots, true);

  if (had) ::setenv("LSE_CACHE_DIR", saved.c_str(), 1);
  else ::unsetenv("LSE_CACHE_DIR");
  std::error_code ec;
  std::filesystem::remove_all(dir, ec);

  LSE_EXPECT_OK(ran);
  if (!ran.ok()) return;

  const Scheduler::Trace::Spans& s = cold.last_trace().spans;
  const Scheduler::JitStats jit = cold.jit_stats();
  expect_all_host_steady(s);
  LSE_EXPECT_EQ(attributed_span_ns(s) + s.unattributed.ns, s.step.ns);

  // jit_compile is the JitCache's own measurement, not a second reading of it.
  LSE_EXPECT_EQ(s.jit_compile.ns, jit.compile_ns);
  std::printf("       compiles=%llu compile=%.3f ms emit=%.3f ms "
              "jit_lookup=%.3f ms step=%.3f ms\n",
              static_cast<unsigned long long>(jit.compiles),
              s.jit_compile.seconds() * 1e3, s.emit.seconds() * 1e3,
              s.jit_lookup.seconds() * 1e3, s.step.seconds() * 1e3);

  if (jit.compiles == 0) {
    std::printf("       nothing compiled (warm disk): containment still holds\n");
    return;
  }
  LSE_EXPECT(s.jit_compile.ns > 0);
  // The point of the change. Emitting HIP text is microseconds; comgr is tens
  // of milliseconds. While compile was nested inside emit, emit could never be
  // the smaller of the two, and on a cold prefill it read as 98.5% compile.
  LSE_EXPECT(s.emit.ns < s.jit_compile.ns);
  // And it is not hiding in the lookup span either.
  LSE_EXPECT(s.emit.ns + s.jit_lookup.ns + s.jit_compile.ns <= s.step.ns);
}

LSE_TEST(the_wait_span_is_host_scoped_and_invents_no_device_duration) {
  Scheduler* sched = default_scheduler();
  if (sched == nullptr) {
    std::printf("       no scheduler: skipped\n");
    return;
  }
  Array x = Array::full(Shape{1, 64}, DType::kF32, 2.0f);
  Array y = silu(x + x);
  LSE_EXPECT_OK(y.eval());

  const Scheduler::Trace& t = sched->last_trace();
  // The blocking wait is host wall and says so.
  LSE_EXPECT(t.spans.host_wait.clock == backend::ClockDomain::kHostSteady);
  // The device interval that wait contains is UNKNOWN, and asking for it is a
  // refusal rather than the host number wearing a device label.
  LSE_EXPECT(!t.device_exec.known());
  auto device = t.device_exec.duration_ns();
  LSE_EXPECT(!device.ok());
  std::printf("       device_exec: %s\n", device.status().message().c_str());

  // A backend may publish its clock and still be unable to read a tick. That
  // is two questions, and the published clock must not become a duration.
  auto clock = sched->backend().device_clock();
  if (clock.ok()) {
    std::printf("       backend clock: %s %llu Hz %u bits ordinal %u\n",
                std::string(backend::clock_domain_name(clock->domain)).c_str(),
                static_cast<unsigned long long>(clock->ticks_per_second),
                static_cast<unsigned>(clock->valid_bits),
                static_cast<unsigned>(clock->ordinal));
    LSE_EXPECT(clock->known());
    LSE_EXPECT(!t.device_exec.duration_ns().ok());
  }
}

LSE_TEST(the_span_split_costs_one_clock_read_per_boundary) {
  // The split reads steady_clock five times per fusion group where the four old
  // counters read four: emit open, emit close / jit open, jit close / bind open,
  // bind close / submit open, submit close. Adjacent spans share the tick
  // between them, so the added cost is exactly one read per group.
  constexpr int kReps = 200'000;
  std::uint64_t sink = 0;
  double best = 1e30;
  for (int trial = 0; trial < 5; ++trial) {
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < kReps; ++i) {
      sink += static_cast<std::uint64_t>(
          std::chrono::steady_clock::now().time_since_epoch().count());
    }
    const double ns =
        static_cast<double>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - t0)
                .count()) /
        kReps;
    best = std::min(best, ns);
  }
  LSE_EXPECT(sink != 0);
  // 315 fusion groups is one lemonseed decode token.
  const double per_token_us = best * 315.0 * 1e-3;
  std::printf("       steady_clock::now() = %.1f ns (best of 5 x %d); "
              "+1 read/group = %.1f us per 315-group token\n",
              best, kReps, per_token_us);
  // A decode token is ~9 ms. 100 us of added instrumentation would be ~1%,
  // which is the line this must stay well under.
  LSE_EXPECT(per_token_us < 100.0);
}

LSE_TEST_MAIN()
