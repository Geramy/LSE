// A user-registered primitive must be indistinguishable from a built-in: same
// fusion behaviour, same evaluation path, same codegen contract.
#include "lse/graph/primitive.hpp"

#include <cmath>
#include <vector>

#include <array>
#include <cstdlib>

#include "harness.hpp"
#include "lse/graph/interpreter.hpp"
#include "lse/graph/ops.hpp"
#include "lse/graph/fallback.hpp"
#include "lse/graph/primitive_library.hpp"

using namespace lse;
using namespace lse::graph;

namespace {

// The whole definition of a new fusable op.
struct Swish final : ElementwisePrimitive<Swish, 1> {
  static constexpr std::string_view kName = "test.swish";
  static constexpr std::array<DialectExpr, 1> kSources{{{Dialect::kHip, "$0 / (1.0f + __expf(-$0))"}}};
  static float apply(float x) { return x / (1.0f + std::exp(-x)); }
};

struct WeightedSum final : ElementwisePrimitive<WeightedSum, 2> {
  static constexpr std::string_view kName = "test.wsum";
  static constexpr std::array<DialectExpr, 1> kSources{{{Dialect::kHip, "0.25f * $0 + 0.75f * $1"}}};
  static float apply(float a, float b) { return 0.25f * a + 0.75f * b; }
};

// These tests are about the host path specifically. Without pinning the mode
// they pass only on a machine with no GPU, and quietly assert nothing on one
// that has a working JIT.
class HostOnly_ {
 public:
  HostOnly_() {
    sched_ = default_scheduler();
    if (sched_ != nullptr) {
      prev_ = sched_->mode();
      sched_->set_mode(Scheduler::Mode::kHostOnly);
    }
  }
  ~HostOnly_() {
    if (sched_ != nullptr) sched_->set_mode(prev_);
  }

 private:
  Scheduler* sched_ = nullptr;
  Scheduler::Mode prev_ = Scheduler::Mode::kDeviceFirst;
};

std::vector<float> drain(Array& a) {
  if (!a.eval().ok()) return {};
  std::vector<float> out(a.shape().elem_count());
  for (std::size_t i = 0; i < out.size(); ++i) {
    out[i] = interpreter::load_element(*a.node(), i);
  }
  return out;
}

// Host-only: a GPU backend must fall back to the CPU for this rather than fail.
struct HostOnly final : ElementwisePrimitive<HostOnly, 1> {
  static constexpr std::string_view kName = "test.hostonly";
  static constexpr std::array<DialectExpr, 1> kSources{{{Dialect::kHip, "$0"}}};
  static float apply(float x) { return x * 3.0f; }
  bool has_device_impl() const noexcept override { return false; }
};

}  // namespace

LSE_REGISTER_PRIMITIVE(HostOnly);
LSE_REGISTER_PRIMITIVE(Swish);
LSE_REGISTER_PRIMITIVE(WeightedSum);

LSE_TEST(builtins_are_registered_as_primitives) {
  // Built-ins go through the same registry, so there is no privileged path.
  for (const char* name : {"add", "mul", "silu", "gelu", "rsqrt", "clamp"}) {
    LSE_EXPECT(find_primitive(name) != nullptr);
  }
  LSE_EXPECT(registered_primitives().size() >= 14u);
}

LSE_TEST(user_primitive_is_registered) {
  const Primitive* p = find_primitive("test.swish");
  LSE_EXPECT(p != nullptr);
  if (!p) return;
  LSE_EXPECT_EQ(p->arity(), 1u);
  LSE_EXPECT(p->fusion_class() == FusionClass::kElementwise);
}

LSE_TEST(duplicate_registration_is_rejected) {
  static const Swish other{};
  auto s = register_primitive(&other);
  LSE_EXPECT(!s.ok());
  LSE_EXPECT(s.code() == StatusCode::kAlreadyExists);
}

LSE_TEST(unknown_primitive_reports_what_is_available) {
  auto r = custom("nope.not.here", {Array::full(Shape{4}, DType::kF32, 1.0f)});
  LSE_EXPECT(!r.ok());
  LSE_EXPECT(r.status().code() == StatusCode::kNotFound);
  LSE_EXPECT(r.status().message().find("silu") != std::string::npos);
}

LSE_TEST(wrong_arity_is_rejected) {
  Array x = Array::full(Shape{4}, DType::kF32, 1.0f);
  auto r = custom("test.wsum", {x});
  LSE_EXPECT(!r.ok());
  LSE_EXPECT(r.status().code() == StatusCode::kInvalidArgument);
}

LSE_TEST(user_primitive_computes_correctly) {
  Array x = Array::full(Shape{8}, DType::kF32, 1.5f);
  auto y = custom("test.swish", {x});
  LSE_EXPECT(y.ok());
  if (!y.ok()) return;
  const auto out = drain(*y);
  const double want = 1.5 / (1.0 + std::exp(-1.5));
  LSE_EXPECT_EQ(out.size(), 8u);
  for (float v : out) LSE_EXPECT_NEAR(v, want, 1e-6);
}

LSE_TEST(user_primitive_matches_the_equivalent_builtin) {
  // test.swish and the built-in silu are the same function, so they must agree
  // bit-for-bit through the shared evaluation path.
  Array x = Array::full(Shape{16}, DType::kF32, -0.75f);
  auto mine = custom("test.swish", {x});
  LSE_EXPECT(mine.ok());
  if (!mine.ok()) return;
  Array theirs = silu(x);
  const auto a = drain(*mine);
  const auto b = drain(theirs);
  if (std::getenv("LSE_TRACE_GROUPS") != nullptr) {
    Scheduler* sc = default_scheduler();
    if (sc != nullptr) {
      const auto& t = sc->last_trace();
      std::printf("       silu: device_groups=%u host_groups=%u fallbacks=%u\n",
                  t.device_groups, t.host_groups, t.host_fallbacks);
      for (const auto& r : t.host_group_reasons) {
        std::printf("       reason: %s\n", r.c_str());
      }
      for (const auto& d : t.group_descriptions) {
        std::printf("       group: %s\n", d.c_str());
      }
    }
  }
  LSE_EXPECT_EQ(a.size(), b.size());
  for (std::size_t i = 0; i < a.size(); ++i) LSE_EXPECT_NEAR(a[i], b[i], 0.0);
}

LSE_TEST(binary_user_primitive_broadcasts) {
  Array a = Array::full(Shape{2, 4}, DType::kF32, 4.0f);
  Array b = Array::full(Shape{4}, DType::kF32, 8.0f);
  auto y = custom("test.wsum", {a, b});
  LSE_EXPECT(y.ok());
  if (!y.ok()) return;
  LSE_EXPECT_EQ(y->shape()[0], 2);
  LSE_EXPECT_EQ(y->shape()[1], 4);
  const auto out = drain(*y);
  for (float v : out) LSE_EXPECT_NEAR(v, 0.25 * 4.0 + 0.75 * 8.0, 1e-6);
}

LSE_TEST(user_primitive_fuses_with_builtins_into_one_kernel) {
  // The point of the whole design: a custom op is not a launch boundary.
  Array x = Array::full(Shape{32}, DType::kF32, 1.0f);
  auto s = custom("test.swish", {x * x});
  LSE_EXPECT(s.ok());
  if (!s.ok()) return;
  Array y = *s + x;

  const NodePtr roots[] = {y.node()};
  const auto groups = Partitioner::partition(roots);
  LSE_EXPECT_EQ(groups.size(), 1u);
  // constant, mul, swish, add
  LSE_EXPECT_EQ(groups[0].nodes.size(), 4u);
}

LSE_TEST(user_primitive_fuses_into_a_reduction_prologue) {
  Array x = Array::full(Shape{4, 8}, DType::kF32, 2.0f);
  auto s = custom("test.swish", {x});
  LSE_EXPECT(s.ok());
  if (!s.ok()) return;
  Array y = sum(*s, -1);
  const NodePtr roots[] = {y.node()};
  const auto groups = Partitioner::partition(roots);
  LSE_EXPECT_EQ(groups.size(), 1u);
  LSE_EXPECT(groups[0].anchor == OpKind::kSum);
}

LSE_TEST(emitted_hip_substitutes_arguments) {
  const Primitive* p = find_primitive("test.swish");
  LSE_EXPECT(p != nullptr);
  if (!p) return;
  const std::string args[] = {"v_in"};
  EmitContext ctx;
  ctx.inputs = args;
  ctx.out = "v_out";
  const std::string src = p->emit_device(ctx);
  LSE_EXPECT(src.find("v_out =") != std::string::npos);
  LSE_EXPECT(src.find("v_in") != std::string::npos);
  LSE_EXPECT(src.find("$0") == std::string::npos);
}

LSE_TEST(substitution_parenthesizes_compound_arguments) {
  // "$0 * $0" with argument "a + b" must not become "a + b * a + b".
  const std::string args[] = {"a + b"};
  const std::string got = substitute("$0 * $0", args);
  LSE_EXPECT(got == "(a + b) * (a + b)");
}

LSE_TEST(substitution_leaves_unrelated_dollars_alone) {
  const std::string args[] = {"x"};
  LSE_EXPECT(substitute("$0 + $", args) == "(x) + $");
}


LSE_TEST(one_file_defines_several_primitives_sharing_helpers) {
  auto lib = PrimitiveLibrary::load_file("tests/fixtures/example_primitives.hip");
  LSE_EXPECT(lib.ok());
  if (!lib.ok()) {
    std::printf("       %s\n", lib.status().to_string().c_str());
    return;
  }
  LSE_EXPECT_EQ((*lib)->entries().size(), 3u);

  for (const char* n : {"example.mish", "example.swiglu", "example.clip3"}) {
    LSE_EXPECT(find_primitive(n) != nullptr);
  }

  const Primitive* mish = find_primitive("example.mish");
  const Primitive* swiglu = find_primitive("example.swiglu");
  LSE_EXPECT(mish != nullptr && swiglu != nullptr);
  if (!mish || !swiglu) return;

  LSE_EXPECT_EQ(mish->arity(), 1u);
  LSE_EXPECT_EQ(swiglu->arity(), 2u);
  LSE_EXPECT_EQ(find_primitive("example.clip3")->arity(), 3u);

  // Both come from the same file, so the shared __device__ helpers are emitted
  // once per kernel rather than duplicated.
  LSE_EXPECT(mish->preamble_id() == swiglu->preamble_id());
  LSE_EXPECT(mish->device_preamble().find("lse_softplus") != std::string_view::npos);
  LSE_EXPECT(mish->device_preamble().find("lse_sigmoid_") != std::string_view::npos);
}

LSE_TEST(library_primitive_emits_a_call_to_its_function) {
  if (find_primitive("example.swiglu") == nullptr) return;
  const std::string args[] = {"g", "u"};
  EmitContext ctx;
  ctx.inputs = args;
  ctx.out = "dst";
  const std::string src = find_primitive("example.swiglu")->emit_device(ctx);
  LSE_EXPECT(src == "dst = lse_swiglu(g, u);");
}

LSE_TEST(library_primitives_fuse_like_builtins) {
  if (find_primitive("example.mish") == nullptr) return;
  Array x = Array::full(Shape{32}, DType::kF32, 1.0f);
  auto m = custom("example.mish", {x * x});
  LSE_EXPECT(m.ok());
  if (!m.ok()) return;
  const NodePtr roots[] = {m->node()};
  const auto groups = Partitioner::partition(roots);
  LSE_EXPECT_EQ(groups.size(), 1u);
}

LSE_TEST(device_only_primitive_refuses_host_evaluation) {
  // No silent zeros: a CPU run of device-only source must fail loudly.
  const HostOnly_ pinned;
  if (find_primitive("example.mish") == nullptr) return;
  Array x = Array::full(Shape{4}, DType::kF32, 1.0f);
  auto m = custom("example.mish", {x});
  LSE_EXPECT(m.ok());
  if (!m.ok()) return;
  auto s = m->eval();
  LSE_EXPECT(!s.ok());
  LSE_EXPECT(s.code() == StatusCode::kUnimplemented);
  LSE_EXPECT(s.message().find("JIT") != std::string::npos);
}

LSE_TEST(annotation_parser_reports_bad_input) {
  auto missing_fn = parse_primitive_annotations("// LSE_PRIMITIVE: foo arity=1\n");
  LSE_EXPECT(!missing_fn.ok());

  auto bad_arity = parse_primitive_annotations(
      "// LSE_PRIMITIVE: foo arity=9\n__device__ float f(float x){return x;}");
  LSE_EXPECT(!bad_arity.ok());

  auto no_name = parse_primitive_annotations(
      "// LSE_PRIMITIVE: \n__device__ float f(float x){return x;}");
  LSE_EXPECT(!no_name.ok());
}

LSE_TEST(file_without_annotations_is_rejected) {
  auto lib = PrimitiveLibrary::load_source("__device__ float f(float x){return x;}",
                                           "<inline>");
  LSE_EXPECT(!lib.ok());
  LSE_EXPECT(lib.status().code() == StatusCode::kNotFound);
}

LSE_TEST(missing_library_file_reports_io_error) {
  auto lib = PrimitiveLibrary::load_file("/nope/not/here.hip");
  LSE_EXPECT(!lib.ok());
  LSE_EXPECT(lib.status().code() == StatusCode::kIoError);
}

LSE_TEST(primitives_declare_which_targets_they_support) {
  const Primitive* host_only = find_primitive("test.hostonly");
  const Primitive* device_only = find_primitive("example.mish");
  const Primitive* both = find_primitive("silu");

  LSE_EXPECT(host_only != nullptr && both != nullptr);
  if (!host_only || !both) return;

  LSE_EXPECT(host_only->has_host_impl());
  LSE_EXPECT(!host_only->has_device_impl());
  LSE_EXPECT(both->has_host_impl());
  LSE_EXPECT(both->has_device_impl());
  if (device_only) {
    LSE_EXPECT(!device_only->has_host_impl());
    LSE_EXPECT(device_only->has_device_impl());
  }
}

LSE_TEST(host_only_primitive_still_computes) {
  Array x = Array::full(Shape{4}, DType::kF32, 2.0f);
  auto y = custom("test.hostonly", {x});
  LSE_EXPECT(y.ok());
  if (!y.ok()) return;
  const auto out = drain(*y);
  for (float v : out) LSE_EXPECT_NEAR(v, 6.0, 1e-6);
}

LSE_TEST(device_gap_is_reported_for_host_only_primitives) {
  Node n;
  n.set_kind(OpKind::kCustom);
  n.prim = find_primitive("test.hostonly");
  LSE_EXPECT(n.prim != nullptr);
  if (!n.prim) return;

  auto be = backend::create_backend("cpu");
  LSE_EXPECT(be.ok());
  if (!be.ok()) return;

  const std::string gap = Scheduler::device_gap(n, **be);
  LSE_EXPECT(!gap.empty());
  LSE_EXPECT(gap.find("no device implementation") != std::string::npos);
}

LSE_TEST(a_node_runnable_on_neither_target_is_an_error) {
  // Device-only source on a host-only executor must fail loudly, not silently
  // produce zeros.
  const HostOnly_ pinned;
  if (find_primitive("example.mish") == nullptr) return;
  Array x = Array::full(Shape{4}, DType::kF32, 1.0f);
  auto m = custom("example.mish", {x});
  LSE_EXPECT(m.ok());
  if (!m.ok()) return;
  auto s = m->eval();
  LSE_EXPECT(!s.ok());
  LSE_EXPECT(s.code() == StatusCode::kUnimplemented);
}

namespace {

// A handler can do anything; this one writes a sentinel so the test can prove
// it ran instead of the default host path.
class SentinelFallback final : public FallbackHandler {
 public:
  explicit SentinelFallback(float value) : value_(value) {}
  std::string_view name() const noexcept override { return "test.sentinel"; }
  bool can_handle(const Node& n, const backend::IBackend&) const override {
    return n.prim != nullptr && n.prim->name() == "test.hostonly";
  }
  Status execute(Node& n, backend::IBackend&) const override {
    for (std::size_t i = 0; i < n.element_count(); ++i) {
      interpreter::store_element(n, i, value_);
    }
    n.materialized = true;
    return OkStatus();
  }

 private:
  float value_;
};

// Claims a chosen op kind even though the backend could run it. Toggling
// `enabled_` at runtime is what per-op device placement looks like.
class OpRouter final : public FallbackHandler {
 public:
  explicit OpRouter(OpKind claim) : claim_(claim) {}
  std::string_view name() const noexcept override { return "test.router"; }
  bool intercepts() const noexcept override { return true; }
  bool can_handle(const Node& n, const backend::IBackend&) const override {
    return enabled_ && n.kind == claim_;
  }
  Status execute(Node& n, backend::IBackend&) const override {
    ++claimed_;
    for (std::size_t i = 0; i < n.element_count(); ++i) {
      interpreter::store_element(n, i, -1.0f);
    }
    n.materialized = true;
    return OkStatus();
  }
  void set_enabled(bool on) { enabled_ = on; }
  std::size_t claimed() const { return claimed_; }

 private:
  OpKind claim_;
  bool enabled_ = true;
  mutable std::size_t claimed_ = 0;
};

class NeverAccepts final : public FallbackHandler {
 public:
  std::string_view name() const noexcept override { return "test.never"; }
  bool can_handle(const Node&, const backend::IBackend&) const override {
    return false;
  }
  Status execute(Node&, backend::IBackend&) const override {
    return LSE_ERROR(kInternal, "should not be called");
  }
};

}  // namespace

LSE_TEST(default_chain_has_the_host_interpreter) {
  const auto names = default_fallback_chain().handler_names();
  bool found = false;
  for (const auto& n : names) found = found || n == "host-interpreter";
  LSE_EXPECT(found);
}

LSE_TEST(chain_orders_by_priority_then_insertion) {
  FallbackChain chain;
  static const NeverAccepts a;
  static const SentinelFallback b{1.0f};
  LSE_EXPECT_OK(chain.add(&a, 0));
  LSE_EXPECT_OK(chain.add(&b, 100));
  const auto names = chain.handler_names();
  LSE_EXPECT_EQ(names.size(), 2u);
  LSE_EXPECT(names[0] == "test.sentinel");
  LSE_EXPECT(names[1] == "test.never");
}

LSE_TEST(chain_rejects_duplicates_and_nulls) {
  FallbackChain chain;
  static const NeverAccepts a;
  LSE_EXPECT_OK(chain.add(&a));
  LSE_EXPECT(!chain.add(&a).ok());
  LSE_EXPECT(!chain.add(nullptr).ok());
  LSE_EXPECT(!chain.remove("nope").ok());
  LSE_EXPECT_OK(chain.remove("test.never"));
  LSE_EXPECT_EQ(chain.size(), 0u);
}

LSE_TEST(chain_takes_ownership_when_asked) {
  FallbackChain chain;
  LSE_EXPECT_OK(chain.add_owned(std::make_unique<SentinelFallback>(2.0f), 5));
  LSE_EXPECT_EQ(chain.size(), 1u);
  // Still resolvable after the caller's unique_ptr is long gone.
  LSE_EXPECT(chain.handler_names()[0] == "test.sentinel");
}

LSE_TEST(a_custom_handler_can_take_over_execution) {
  auto be = backend::create_backend("cpu");
  LSE_EXPECT(be.ok());
  if (!be.ok()) return;
  LSE_EXPECT_OK((*be)->init(0));

  FallbackChain chain;
  LSE_EXPECT_OK(chain.add_owned(std::make_unique<SentinelFallback>(42.0f), 100));
  LSE_EXPECT_OK(chain.add(host_interpreter_fallback(), 0));

  Scheduler sched(**be);
  sched.set_fallback_chain(&chain);

  Array x = Array::full(Shape{4}, DType::kF32, 2.0f);
  auto y = custom("test.hostonly", {x});
  LSE_EXPECT(y.ok());
  if (!y.ok()) return;

  const NodePtr roots[] = {y->node()};
  LSE_EXPECT_OK(sched.eval(roots));

  // 42 from the sentinel, not 6 from the host implementation.
  for (std::size_t i = 0; i < 4; ++i) {
    LSE_EXPECT_NEAR(interpreter::load_element(*y->node(), i), 42.0, 1e-6);
  }

  const auto& trace = sched.last_trace();
  LSE_EXPECT_EQ(trace.host_fallbacks, 1u);
  LSE_EXPECT_EQ(trace.fallback_handlers.size(), 1u);
  LSE_EXPECT(trace.fallback_handlers[0] == "test.sentinel");
  (*be)->shutdown();
}

LSE_TEST(an_empty_chain_fails_loudly) {
  auto be = backend::create_backend("cpu");
  if (!be.ok()) return;
  LSE_EXPECT_OK((*be)->init(0));

  FallbackChain empty;
  Scheduler sched(**be);
  sched.set_fallback_chain(&empty);

  Array x = Array::full(Shape{4}, DType::kF32, 1.0f);
  auto y = custom("test.hostonly", {x});
  if (!y.ok()) return;
  const NodePtr roots[] = {y->node()};
  auto s = sched.eval(roots);
  LSE_EXPECT(!s.ok());
  LSE_EXPECT(s.message().find("no fallback handler") != std::string::npos);
  (*be)->shutdown();
}

LSE_TEST(an_intercepting_handler_claims_ops_the_backend_could_run) {
  auto be = backend::create_backend("cpu");
  if (!be.ok()) return;
  LSE_EXPECT_OK((*be)->init(0));

  OpRouter router{OpKind::kMul};
  FallbackChain chain;
  LSE_EXPECT_OK(chain.add(&router, 100));
  LSE_EXPECT_OK(chain.add(host_interpreter_fallback(), 0));

  Scheduler sched(**be);
  sched.set_fallback_chain(&chain);

  Array x = Array::full(Shape{4}, DType::kF32, 3.0f);
  Array y = x * x;
  const NodePtr roots[] = {y.node()};
  LSE_EXPECT_OK(sched.eval(roots));

  // The multiply was routed away, so the value is the router's sentinel, not 9.
  LSE_EXPECT_NEAR(interpreter::load_element(*y.node(), 0), -1.0, 1e-6);
  LSE_EXPECT_EQ(sched.last_trace().intercepted, 1u);
  LSE_EXPECT_EQ(router.claimed(), 1u);
  (*be)->shutdown();
}

LSE_TEST(routing_can_be_switched_off_at_runtime) {
  auto be = backend::create_backend("cpu");
  if (!be.ok()) return;
  LSE_EXPECT_OK((*be)->init(0));

  OpRouter router{OpKind::kMul};
  router.set_enabled(false);

  FallbackChain chain;
  LSE_EXPECT_OK(chain.add(&router, 100));
  LSE_EXPECT_OK(chain.add(host_interpreter_fallback(), 0));

  Scheduler sched(**be);
  sched.set_fallback_chain(&chain);

  Array x = Array::full(Shape{4}, DType::kF32, 3.0f);
  Array y = x * x;
  const NodePtr roots[] = {y.node()};
  LSE_EXPECT_OK(sched.eval(roots));

  // Router declined, so the normal path computed 3*3.
  LSE_EXPECT_NEAR(interpreter::load_element(*y.node(), 0), 9.0, 1e-6);
  LSE_EXPECT_EQ(sched.last_trace().intercepted, 0u);
  LSE_EXPECT_EQ(router.claimed(), 0u);
  (*be)->shutdown();
}

LSE_TEST(non_intercepting_handlers_are_not_consulted_for_runnable_nodes) {
  auto be = backend::create_backend("cpu");
  if (!be.ok()) return;
  LSE_EXPECT_OK((*be)->init(0));

  // SentinelFallback is not an interceptor, so it must not claim ordinary work.
  FallbackChain chain;
  LSE_EXPECT_OK(chain.add_owned(std::make_unique<SentinelFallback>(42.0f), 100));
  LSE_EXPECT_OK(chain.add(host_interpreter_fallback(), 0));

  Scheduler sched(**be);
  sched.set_fallback_chain(&chain);

  Array x = Array::full(Shape{4}, DType::kF32, 3.0f);
  Array y = x * x;
  const NodePtr roots[] = {y.node()};
  LSE_EXPECT_OK(sched.eval(roots));
  LSE_EXPECT_NEAR(interpreter::load_element(*y.node(), 0), 9.0, 1e-6);
  LSE_EXPECT_EQ(sched.last_trace().intercepted, 0u);
  (*be)->shutdown();
}

LSE_TEST_MAIN()
