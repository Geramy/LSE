#include "harness.hpp"
#include "lse/graph/interpreter.hpp"
#include "lse/graph/ops.hpp"
#include "lse/graph/program.hpp"
#include "lse/ops/linear_attention.hpp"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <string_view>
#include <vector>

using namespace lse;
using namespace lse::graph;
namespace {
bool native = false;
Scheduler& scheduler() {
  auto* s = default_scheduler();
  if (!s) std::abort();
  if (!native) s->set_mode(Scheduler::Mode::kHostOnly);
  return *s;
}
Array upload(Shape shape, DType dtype, float phase) {
  auto& s = scheduler();
  Array a = Array::zeros(shape, dtype);
  LSE_EXPECT_OK(interpreter::ensure_output_buffer(*a.node(), s.backend()));
  for (std::size_t i = 0; i < a.shape().elem_count(); ++i) {
    const auto n = static_cast<int>(i % 19);
    interpreter::store_element(*a.node(), i, phase + static_cast<float>(n - 9) * .03125f);
  }
  a.node()->materialized = true;
  a.node()->host_dirty = true;
  a.node()->device_dirty = false;
  LSE_EXPECT_OK(interpreter::sync_to_device(*a.node(), s.backend()));
  Array leaf = Array::from_buffer(a.node()->buffer, shape, dtype);
  leaf.node()->member = a.node()->member;
  leaf.node()->host_mirror = a.node()->host_mirror;
  leaf.node()->host_dirty = a.node()->host_dirty;
  leaf.node()->device_dirty = a.node()->device_dirty;
  return leaf;
}
Array raw_rate(const Array& log, ops::DecayRate decay) {
  Array a = exp(cast(log, DType::kF32));
  return decay == ops::DecayRate::kSoftplusExpALog ? softplus(a) : a;
}
void exact(Array a, Array b) {
  std::vector<float> av(a.shape().elem_count()), bv(b.shape().elem_count());
  const auto as = a.to_host(av.data(), av.size() * sizeof(float));
  const auto bs = b.to_host(bv.data(), bv.size() * sizeof(float));
  LSE_EXPECT(as.ok() && bs.ok());
  if (!as.ok() || !bs.ok()) return;
  LSE_EXPECT_EQ(av.size(), bv.size());
  for (std::size_t i = 0; i < std::min(av.size(), bv.size()); ++i) {
    LSE_EXPECT(std::isfinite(av[i]) && std::isfinite(bv[i]));
    LSE_EXPECT_EQ(std::bit_cast<std::uint32_t>(av[i]),
                  std::bit_cast<std::uint32_t>(bv[i]));
  }
}
void strict_trace() {
  if (!native) return;
  const auto& t = scheduler().last_trace();
  LSE_EXPECT(t.device_groups > 0);
  LSE_EXPECT_EQ(t.host_groups, 0u);
  LSE_EXPECT_EQ(t.host_fallbacks, 0u);
}
bool reaches(const Program& p, const NodePtr& node) {
  return std::find(p.nodes().begin(), p.nodes().end(), node) != p.nodes().end();
}
ops::GatedDeltaNetSpec spec(ops::DecayRate decay, int heads = 2, int dim = 4) {
  ops::GatedDeltaNetSpec s;
  s.key_heads = s.value_heads = heads;
  s.key_head_dim = s.value_head_dim = dim;
  s.decay = decay;
  return s;
}
ops::GatedDeltaNetWeights weights() {
  ops::GatedDeltaNetWeights w;
  w.in_proj_q = upload({8,8}, DType::kF32, .11f);
  w.in_proj_k = upload({8,8}, DType::kF32, -.08f);
  w.in_proj_v = upload({8,8}, DType::kF32, .19f);
  w.in_proj_a = upload({2,8}, DType::kF32, .05f);
  w.in_proj_b = upload({2,8}, DType::kF32, -.01f);
  w.conv_q_w = w.conv_k_w = w.conv_v_w = upload({8,3}, DType::kF32, .09f);
  w.conv_q_b = w.conv_k_b = w.conv_v_b = Array::zeros({8}, DType::kF32);
  w.a_log = upload({2}, DType::kBF16, -.23f);
  w.dt_bias = upload({2}, DType::kBF16, .12f);
  w.norm = Array::full({4}, DType::kF32, 1.f);
  w.gate_proj = upload({8,8}, DType::kF32, .03f);
  w.out_proj = upload({8,8}, DType::kF32, .07f);
  return w;
}
}  // namespace

LSE_TEST(preparation_keeps_exact_fp32_rate_and_owned_leaf) {
  for (const auto decay : {ops::DecayRate::kExpALog, ops::DecayRate::kSoftplusExpALog}) {
    auto s = spec(decay, native ? 48 : 2);
    ops::GatedDeltaNetWeights w;
    w.a_log = upload({s.key_heads}, DType::kBF16, -.35f);
    const auto source = w.a_log.node();
    LSE_EXPECT_OK(ops::prepare_gated_delta_rate(w, s));
    strict_trace();
    const auto n = w.prepared_rate.node();
    LSE_EXPECT(n && n->kind == OpKind::kBuffer && n->inputs.empty());
    LSE_EXPECT(n->materialized && n->buffer.valid() && n->buffer.storage);
    LSE_EXPECT(n->dtype == DType::kF32 && n->member == source->member);
    auto expected = raw_rate(w.a_log, decay);
    exact(expected, w.prepared_rate);
    std::weak_ptr<void> lifetime = n->buffer.storage;
    auto retained = w.prepared_rate;
    w = {};
    LSE_EXPECT(!lifetime.expired());
    exact(expected, retained);
  }
}

LSE_TEST(replay_omits_rate_graph_and_preserves_recurrent_output_bits) {
  const int heads = native ? 48 : 2, dim = native ? 128 : 4;
  for (const auto decay : {ops::DecayRate::kExpALog, ops::DecayRate::kSoftplusExpALog}) {
    const auto s = spec(decay, heads, dim);
    ops::GatedDeltaNetWeights w;
    w.a_log = upload({heads}, DType::kBF16, -.42f);
    LSE_EXPECT_OK(ops::prepare_gated_delta_rate(w, s));
    const auto leaf = w.prepared_rate.node();
    const auto storage = leaf->buffer.storage;
    for (int rows : {1,4}) {
      const Shape vector{1,rows,heads,dim}, scalar{1,rows,heads};
      auto q = upload(vector, DType::kF32, .12f);
      auto k = upload(vector, DType::kF32, -.09f);
      auto v = upload(vector, DType::kF32, .2f);
      auto a = upload(scalar, DType::kF32, .15f);
      auto bias = upload({heads}, DType::kBF16, .07f);
      auto beta = upload(scalar, DType::kF32, .4f);
      auto initial = upload({1,heads,dim,dim}, DType::kF32, .04f);
      auto old_alpha = exp(neg(raw_rate(w.a_log, decay) * softplus(a + bias)));
      auto new_alpha = exp(neg(w.prepared_rate * softplus(a + bias)));
      Array old_state, new_state;
      auto old_output = gated_delta_step(q,k,v,old_alpha,beta,initial,&old_state);
      auto new_output = gated_delta_step(q,k,v,new_alpha,beta,initial,&new_state);
      const NodePtr old_roots[]{old_output.node(),old_state.node()};
      const NodePtr new_roots[]{new_output.node(),new_state.node()};
      Program old_program, new_program;
      for (int replay = 0; replay < 3; ++replay) {
        if (replay) {
          old_program.reset_compute(); new_program.reset_compute();
          // Rebind a dynamic input while preserving the prepared parameter.
          auto replacement = upload(scalar, DType::kF32, .15f + static_cast<float>(replay) * .125f);
          a.node()->buffer = replacement.node()->buffer;
          a.node()->host_mirror = replacement.node()->host_mirror;
          a.node()->host_dirty = replacement.node()->host_dirty;
          a.node()->device_dirty = replacement.node()->device_dirty;
        }
        LSE_EXPECT(leaf->materialized && leaf->buffer.storage == storage);
        LSE_EXPECT_OK(scheduler().eval(old_roots, false, &old_program)); strict_trace();
        LSE_EXPECT_OK(scheduler().eval(new_roots, false, &new_program)); strict_trace();
        LSE_EXPECT(reaches(old_program, w.a_log.node()));
        LSE_EXPECT(!reaches(new_program, w.a_log.node()));
        LSE_EXPECT(reaches(new_program, leaf));
        LSE_EXPECT_EQ(old_program.compute_count() - new_program.compute_count(),
                       decay == ops::DecayRate::kExpALog ? 2u : 3u);
        exact(old_output, new_output); exact(old_state, new_state);
      }
    }
  }
}

LSE_TEST(public_raw_weights_and_prepared_weights_match_full_gdn_and_conv_state) {
  if (native) LSE_SKIP("full-op contract uses the CPU reference");
  for (const auto decay : {ops::DecayRate::kExpALog, ops::DecayRate::kSoftplusExpALog}) {
    const auto s = spec(decay);
    const auto raw = weights(); auto prepared = raw;
    LSE_EXPECT(!raw.prepared_rate.valid());
    LSE_EXPECT_OK(ops::prepare_gated_delta_rate(prepared, s));
    for (int rows : {1,3}) {
      auto x = upload({1,rows,8}, DType::kF32, .08f);
      ops::GatedDeltaNetState a, b;
      a.conv_q = a.conv_k = a.conv_v = upload({1,2,8}, DType::kF32, .02f);
      a.recurrent = upload({1,2,4,4}, DType::kF32, .04f);
      b = a;
      auto old = ops::gated_delta_net(x, raw, s, &a);
      auto next = ops::gated_delta_net(x, prepared, s, &b);
      LSE_EXPECT(old.ok() && next.ok());
      if (!old.ok() || !next.ok()) continue;
      const NodePtr old_roots[]{old->node(),a.recurrent.node()};
      const NodePtr new_roots[]{next->node(),b.recurrent.node()};
      Program old_program, new_program;
      for (int replay = 0; replay < 2; ++replay) {
        if (replay) { old_program.reset_compute(); new_program.reset_compute(); }
        LSE_EXPECT_OK(scheduler().eval(old_roots, false, &old_program));
        LSE_EXPECT_OK(scheduler().eval(new_roots, false, &new_program));
        exact(*old,*next); exact(a.recurrent,b.recurrent);
        exact(a.conv_q,b.conv_q); exact(a.conv_k,b.conv_k); exact(a.conv_v,b.conv_v);
        LSE_EXPECT(!reaches(new_program, raw.a_log.node()));
      }
    }
  }
}

LSE_TEST(prepared_rate_rejects_wrong_rule_shape_dtype_or_missing_buffer) {
  if (native) LSE_SKIP("invalid-input contract uses the CPU reference");
  auto s = spec(ops::DecayRate::kExpALog);
  auto w = weights(); LSE_EXPECT_OK(ops::prepare_gated_delta_rate(w, s));
  auto x = Array::zeros({1,1,8}, DType::kF32);
  auto wrong_rule = s; wrong_rule.decay = ops::DecayRate::kSoftplusExpALog;
  LSE_EXPECT(!ops::gated_delta_net(x,w,wrong_rule,nullptr).ok());
  for (int reason = 0; reason < 4; ++reason) {
    auto bad = w;
    auto n = std::make_shared<Node>(*w.prepared_rate.node());
    bad.prepared_rate = Array(n);
    if (reason == 0) n->shape = Shape{3};
    if (reason == 1) n->dtype = DType::kBF16;
    if (reason == 2) n->buffer = {};
    if (reason == 3) n->materialized = false;
    LSE_EXPECT(!ops::gated_delta_net(x,bad,s,nullptr).ok());
  }
  const auto saved = w.prepared_rate.node();
  w.a_log = {};
  LSE_EXPECT(!ops::prepare_gated_delta_rate(w,s).ok());
  LSE_EXPECT(w.prepared_rate.node() == saved);
}

int main(int argc, char** argv) {
  native = argc == 2 && std::string_view(argv[1]) == "--gpu-rate";
  if (native) {
    setenv("LSE_REQUIRE_DEVICE_KERNELS", "1", 1);
  } else {
    setenv("LSE_BACKEND", "cpu", 1); setenv("LSE_POOL", "cpu", 1);
    unsetenv("LSE_REQUIRE_DEVICE_KERNELS");
  }
  return test::run_all();
}
