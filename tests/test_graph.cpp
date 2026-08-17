#include "lse/graph/graph.hpp"

#include <cmath>
#include <cstring>
#include <sstream>
#include <unordered_map>
#include <vector>

#include "harness.hpp"
#include "lse/graph/interpreter.hpp"
#include "lse/graph/ops.hpp"
#include "lse/graph/program.hpp"
#include "lse/graph/sharding.hpp"
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

}  // namespace

// The whole point of the op: a weight that stays packed all the way to the
// register produces the same numbers as the contraction against the weights it
// encodes. The reference is computed here in double rather than with
// graph::linear, whose kernel narrows to f16 operands above a tile and would
// then be the less precise side of the comparison.
//
// Every bit width MLX emits, at one row (decode) and at more rows than a tile
// (prefill): 3, 5 and 6 bits are the ones whose codes straddle two lanes.
LSE_TEST(quant_linear_matches_the_weights_it_encodes) {
  for (int bits : {2, 3, 4, 5, 6, 8}) {
    for (std::int64_t rows : {std::int64_t{1}, std::int64_t{20}}) {
      auto spec = quant::GroupAffine::make(bits, 64);
      LSE_EXPECT(spec.ok());
      if (!spec.ok()) continue;
      const quant::GroupAffine q = *spec;

      constexpr std::int64_t kN = 32, kK = 128;
      const auto lanes_per_row = q.packed_words(kK);
      const auto groups_per_row = q.group_count(kK);

      std::vector<float> x(static_cast<std::size_t>(rows * kK));
      for (std::size_t i = 0; i < x.size(); ++i) {
        x[i] = 0.1f * std::sin(0.7f * static_cast<float>(i));
      }

      std::vector<std::uint32_t> packed(
          static_cast<std::size_t>(kN) * lanes_per_row);
      std::vector<float> scales(static_cast<std::size_t>(kN) * groups_per_row);
      std::vector<float> biases(scales.size());
      std::vector<float> dequantized(static_cast<std::size_t>(kN * kK));

      for (std::int64_t r = 0; r < kN; ++r) {
        std::vector<float> row(static_cast<std::size_t>(kK));
        for (std::int64_t c = 0; c < kK; ++c) {
          row[static_cast<std::size_t>(c)] =
              0.05f * std::cos(0.13f * static_cast<float>(r * kK + c));
        }
        const auto ro = static_cast<std::size_t>(r);
        std::vector<bfloat16_t> s(groups_per_row), b(groups_per_row);
        q.quantize_row<bfloat16_t>(row.data(), static_cast<std::size_t>(kK),
                                   packed.data() + ro * lanes_per_row, s.data(),
                                   b.data());
        q.dequantize_row<bfloat16_t>(
            packed.data() + ro * lanes_per_row, s.data(), b.data(),
            static_cast<std::size_t>(kK),
            dequantized.data() + ro * static_cast<std::size_t>(kK));
        for (std::size_t g = 0; g < groups_per_row; ++g) {
          scales[ro * groups_per_row + g] = s[g].to_float();
          biases[ro * groups_per_row + g] = b[g].to_float();
        }
      }

      Array xa = host_array(x, Shape{rows, kK});
      Array pa = packed_host_array(
          packed, Shape{kN, static_cast<std::int64_t>(lanes_per_row)});
      Array sa = typed_host_array(
          scales, Shape{kN, static_cast<std::int64_t>(groups_per_row)},
          DType::kBF16);
      Array ba = typed_host_array(
          biases, Shape{kN, static_cast<std::int64_t>(groups_per_row)},
          DType::kBF16);
      // The plane reached the graph at its own width; nothing widened it.
      LSE_EXPECT(pa.dtype() == DType::kU32);

      Array got = quant_linear(xa, pa, sa, ba, bits, 64);
      LSE_EXPECT(got.node()->kind == OpKind::kQuantMatMul);
      LSE_EXPECT(got.node()->inputs.size() == 4);

      const std::vector<float> g = drain(got);
      LSE_EXPECT(g.size() == static_cast<std::size_t>(rows * kN));
      for (std::int64_t r = 0; r < rows; ++r) {
        for (std::int64_t o = 0; o < kN; ++o) {
          double want = 0.0;
          for (std::int64_t c = 0; c < kK; ++c) {
            want += static_cast<double>(x[static_cast<std::size_t>(r * kK + c)]) *
                    static_cast<double>(
                        dequantized[static_cast<std::size_t>(o * kK + c)]);
          }
          LSE_EXPECT_NEAR(g[static_cast<std::size_t>(r * kN + o)], want,
                          1e-5 + 1e-5 * std::fabs(want));
        }
      }
    }
  }
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

LSE_TEST_MAIN()
