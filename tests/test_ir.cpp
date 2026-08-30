// The kernel IR: structured regions, the verifier, and one test per pass.
//
// Every pass test lowers the SAME body twice — once untouched, once through
// the pass under test — and asserts on both. A test that only checked the
// optimized side would still pass if the pass did nothing and the recorder had
// happened to produce the answer already.
#include <string>
#include <vector>

#include "harness.hpp"
#include "lse/ir/args.hpp"
#include "lse/ir/dialect.hpp"
#include "lse/ir/env.hpp"
#include "lse/ir/index.hpp"
#include "lse/ir/lower.hpp"
#include "lse/ir/pass/cse.hpp"
#include "lse/ir/pass/dce.hpp"
#include "lse/ir/alias.hpp"
#include "lse/ir/pass/factor_hoist.hpp"
#include "lse/ir/pass/lds_fold.hpp"
#include "lse/ir/pass/pass.hpp"
#include "lse/ir/recorder.hpp"
#include "lse/ir/space.hpp"
#include "lse/ir/verify.hpp"
#include "lse/math.hpp"

namespace ir = lse::ir;
namespace env = lse::ir::env;

namespace {

std::string_view test_scalar(ir::Scalar s) noexcept {
  switch (s) {
    case ir::Scalar::kU32: return "unsigned int";
    case ir::Scalar::kF32: return "float";
    case ir::Scalar::kBool: return "bool";
    default: return "float";
  }
}
std::string test_vec(ir::Scalar s, int n, std::string_view name) {
  return "typedef " + std::string(test_scalar(s)) + " " + std::string(name) +
         " __attribute__((ext_vector_type(" + std::to_string(n) + ")));";
}
const ir::TypeTable kTypes{test_scalar, test_vec};

constexpr ir::PrimitiveSource kSources[] = {
    {"fma", "fmaf($0, $1, $2)"},
    {"barrier", "__syncthreads()"},
    {"shared", "__shared__"},
    {"thread.local_id", "threadIdx.x"},
    {"thread.workgroup_id.x", "blockIdx.x"},
    {"thread.workgroup_id.y", "blockIdx.y"},
};
const ir::DialectSourceTable kTable{
    std::span<const ir::PrimitiveSource>(kSources)};

std::size_t count_of(const std::string& hay, std::string_view needle) {
  std::size_t n = 0;
  for (std::size_t p = hay.find(needle); p != std::string::npos;
       p = hay.find(needle, p + 1)) {
    ++n;
  }
  return n;
}

std::size_t run_one(ir::Body& body, std::unique_ptr<ir::Pass> pass) {
  return pass->run(body);
}

// Two GEMV-shaped stages over one activation buffer, the way the emitter
// splices fused siblings: each stages the same row into its own scratch under
// its own tile guard, barriers, and reads it back.
struct StagedRun {
  std::uint32_t k = 64;
  std::uint32_t tiles_a = 8;
  std::uint32_t tiles_b = 8;
  bool same_source = true;
  // The buffers the stages stage from, named so a test can hand the store
  // epilogue a longer name that contains one of them.
  std::string_view source = "b0";
  std::string_view other = "b1";
};

void build_two_stages(ir::KernelBody& kb, const StagedRun& cfg) {
  env::Emit e{&kb};
  const ir::Buffer<ir::f32> x(&kb, &kb.types(), std::string(cfg.source));
  const ir::Buffer<ir::f32> y(&kb, &kb.types(), std::string(cfg.other));
  const std::uint32_t tiles[2] = {cfg.tiles_a, cfg.tiles_b};
  for (int stage = 0; stage < 2; ++stage) {
    const auto lid = e.let(lse::math::local_id());
    const auto tile = e.let(lse::math::workgroup_id_x());
    const auto row = e.let(lse::math::workgroup_id_y());
    auto scratch = e.lds<ir::f32>(cfg.k);
    if (auto g = e.when(tile < tiles[stage] && row < 1u)) {
      for (auto t : e.range(lid, e.u32(cfg.k), 256)) {
        const ir::Buffer<ir::f32>& src =
            (stage == 1 && !cfg.same_source) ? y : x;
        scratch[t] = src[row * cfg.k + t].read();
      }
      e.barrier();
      auto acc = e.var(0.0f);
      for (auto t : e.range(cfg.k)) {
        acc = lse::math::fma(scratch[t].read(), e.f32(2.0f), acc.read());
      }
    }
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// Structure
// ---------------------------------------------------------------------------

LSE_TEST(control_flow_is_regions_not_a_flat_statement_list) {
  ir::KernelBody kb(kTypes, kTable);
  env::Emit e{&kb};
  auto acc = e.var(0.0f);
  if (auto g = e.when(e.thread_id() < 4u)) {
    for (auto t : e.range(8u)) {
      acc = acc.read() + e.f32(1.0f);
      (void)t;
    }
  }

  const ir::Body& b = kb.ir();
  // Exactly one `if` at the top level, and its body is a nested region rather
  // than a begin/end marker in the same list.
  std::size_t top_ifs = 0;
  ir::OpId the_if = ir::kNoOp;
  for (ir::OpId id : b.region(b.entry()).ops) {
    if (b.op(id).kind == ir::OpKind::kIf) {
      ++top_ifs;
      the_if = id;
    }
    LSE_EXPECT(b.op(id).kind != ir::OpKind::kFor);  // the loop is INSIDE the if
  }
  LSE_EXPECT_EQ(top_ifs, 1u);
  LSE_EXPECT_EQ(b.op(the_if).regions.size(), 1u);

  const ir::Region& body = b.region(b.op(the_if).regions[0]);
  LSE_EXPECT(b.region(b.op(the_if).regions[0]).parent == the_if);
  std::size_t fors = 0;
  ir::OpId the_for = ir::kNoOp;
  for (ir::OpId id : body.ops) {
    if (b.op(id).kind == ir::OpKind::kFor) {
      ++fors;
      the_for = id;
    }
  }
  LSE_EXPECT_EQ(fors, 1u);
  // A loop DEFINES its induction variable — it is a value with a type, not a
  // name the printer chose.
  LSE_EXPECT(b.op(the_for).result != ir::kNoValue);
  LSE_EXPECT(b.value(b.op(the_for).result).type.elem == ir::Scalar::kU32);
  LSE_EXPECT(!b.value(b.op(the_for).result).name.empty());
  LSE_EXPECT_EQ(b.op(the_for).regions.size(), 1u);
  LSE_EXPECT(ir::verify(b).ok());
}

LSE_TEST(values_carry_types_and_operands) {
  ir::KernelBody kb(kTypes, kTable);
  env::Emit e{&kb};
  const auto sum = e.let(e.thread_id() + 3u);

  const ir::Body& b = kb.ir();
  const ir::Operation& bind = b.op(b.value(sum.id()).def);
  LSE_EXPECT(bind.kind == ir::OpKind::kBind);
  LSE_EXPECT(bind.type.elem == ir::Scalar::kU32);
  LSE_EXPECT_EQ(bind.operands.size(), 1u);
  const ir::Operation& add = b.op(b.value(bind.operands[0]).def);
  LSE_EXPECT(add.kind == ir::OpKind::kBinary);
  LSE_EXPECT(add.key == "+");
  LSE_EXPECT_EQ(add.operands.size(), 2u);
  LSE_EXPECT(b.op(b.value(add.operands[1]).def).kind == ir::OpKind::kConst);
  LSE_EXPECT_EQ(b.op(b.value(add.operands[1]).def).imm, 3);
}

LSE_TEST(a_memory_reference_says_where_it_lives) {
  ir::KernelBody kb(kTypes, kTable);
  env::Emit e{&kb};
  const ir::Buffer<ir::f32> buf(&kb, &kb.types(), "b0");
  auto tile = e.lds<ir::f32>(32);
  auto frag = e.local<ir::f32, 4>();

  const ir::Body& b = kb.ir();
  LSE_EXPECT(b.value(buf.id()).type.space == ir::Space::kGlobal);
  LSE_EXPECT(b.value(tile.id()).type.space == ir::Space::kWorkgroup);
  LSE_EXPECT_EQ(b.value(tile.id()).type.lanes, 32u);
  LSE_EXPECT(b.value(frag.id()).type.space == ir::Space::kPrivate);
}

// ---------------------------------------------------------------------------
// The verifier
// ---------------------------------------------------------------------------

LSE_TEST(verifier_rejects_a_use_its_definition_does_not_dominate) {
  ir::KernelBody kb(kTypes, kTable);
  env::Emit e{&kb};
  ir::ValueId inner = ir::kNoValue;
  if (auto g = e.when(e.thread_id() < 4u)) {
    inner = e.let(e.thread_id() * 2u).id();
  }
  LSE_EXPECT(ir::verify(kb.ir()).ok());

  // Now use it after the region it was declared in, which the recorder's C++
  // scoping makes impossible but a pass could do by mistake.
  ir::Operation use;
  use.kind = ir::OpKind::kBind;
  use.type = ir::scalar_type(ir::Scalar::kU32);
  use.operands = {inner};
  kb.ir().add(std::move(use), "leaked");
  LSE_EXPECT(!ir::verify(kb.ir()).ok());
}

// ---------------------------------------------------------------------------
// Literal vs runtime extents
// ---------------------------------------------------------------------------

LSE_TEST(a_literal_extent_is_baked_and_a_runtime_extent_is_a_dispatch_constant) {
  ir::KernelBody kb(kTypes, kTable);
  env::Emit e{&kb};
  const auto n = e.extent("N", 1024);
  const auto m = e.runtime_extent("M", "k.rows");
  LSE_EXPECT(n.text() == "1024u");
  LSE_EXPECT(m.text() == "k.rows");
  LSE_EXPECT_EQ(kb.ir().runtime_extents().size(), 1u);
  LSE_EXPECT(kb.ir().runtime_extents()[0] == "M");
  // A baked extent linearizes to a number; a runtime one stays a symbol, which
  // is precisely what keeps it out of an address stride.
  LSE_EXPECT(ir::AffineExpr::of(kb.ir(), n.id()).as_constant().has_value());
  LSE_EXPECT(!ir::AffineExpr::of(kb.ir(), m.id()).as_constant().has_value());
}

LSE_TEST(a_runtime_extent_is_legal_as_an_outer_bound_and_in_a_guard) {
  ir::KernelBody kb(kTypes, kTable);
  env::Emit e{&kb};
  const ir::Buffer<ir::f32> buf(&kb, &kb.types(), "b0");
  const auto rows = e.runtime_extent("M", "k.rows");
  if (auto g = e.when(e.thread_id() < rows)) {
    for (auto r : e.range(e.u32(0), rows)) {
      // The address is affine over the induction variable and a LITERAL
      // extent; the runtime extent never enters it.
      buf[r * 1024u + e.thread_id()] = e.f32(1.0f);
    }
  }
  LSE_EXPECT(ir::verify(kb.ir()).ok());
}

LSE_TEST(a_runtime_extent_is_refused_in_address_arithmetic) {
  ir::KernelBody kb(kTypes, kTable);
  env::Emit e{&kb};
  const ir::Buffer<ir::f32> buf(&kb, &kb.types(), "b0");
  const auto rows = e.runtime_extent("M", "k.rows");
  // A non-constant stride: the whole point of baking extents in is that this
  // cannot happen.
  buf[e.thread_id() * rows] = e.f32(1.0f);
  const lse::Status s = ir::verify(kb.ir());
  LSE_EXPECT(!s.ok());
  LSE_EXPECT(s.message().find("address arithmetic") != std::string::npos);
}

LSE_TEST(a_runtime_extent_is_refused_as_an_inner_trip_count) {
  ir::KernelBody kb(kTypes, kTable);
  env::Emit e{&kb};
  auto acc = e.var(0.0f);
  const auto cols = e.runtime_extent("K", "k.cols");
  for (auto r : e.range(4u)) {
    (void)r;
    for (auto c : e.range(e.u32(0), cols)) {
      (void)c;
      acc = acc.read() + e.f32(1.0f);
    }
  }
  const lse::Status s = ir::verify(kb.ir());
  LSE_EXPECT(!s.ok());
  LSE_EXPECT(s.message().find("inner trip count") != std::string::npos);
}

// ---------------------------------------------------------------------------
// Iteration space
// ---------------------------------------------------------------------------

LSE_TEST(a_dimension_carries_the_cost_of_splitting_it) {
  ir::KernelBody kb(kTypes, kTable);
  ir::IterationSpace& sp = kb.ir().space();
  sp.add(ir::Dim{"tokens", 128, ir::DimKind::kParallel, 1, "k.token_base"});
  sp.add(ir::Dim{"k", 1024, ir::DimKind::kReduction, 256, {}});
  sp.add(ir::Dim{"layer", 20, ir::DimKind::kSequential, 1, {}});

  LSE_EXPECT_EQ(sp.items(), 128 * 1024 * 20);
  const ir::Dim* tokens = sp.find("tokens");
  const ir::Dim* kdim = sp.find("k");
  const ir::Dim* layer = sp.find("layer");
  LSE_EXPECT(tokens != nullptr && kdim != nullptr && layer != nullptr);
  if (tokens == nullptr || kdim == nullptr || layer == nullptr) return;
  LSE_EXPECT(tokens->kind == ir::DimKind::kParallel);
  LSE_EXPECT(tokens->splittable());
  LSE_EXPECT_EQ(tokens->max_windows(), 128);
  // Splitting K costs a collective; the kind is what says so.
  LSE_EXPECT(kdim->kind == ir::DimKind::kReduction);
  LSE_EXPECT_EQ(kdim->max_windows(), 4);
  LSE_EXPECT(layer->kind == ir::DimKind::kSequential);
  LSE_EXPECT(ir::to_string(ir::DimKind::kReduction) == "reduction");
  LSE_EXPECT(ir::verify(kb.ir()).ok());
}

LSE_TEST(a_dimension_that_takes_a_window_must_say_it_is_splittable) {
  ir::KernelBody kb(kTypes, kTable);
  kb.ir().space().add(
      ir::Dim{"tokens", 128, ir::DimKind::kParallel, 0, "k.token_base"});
  const lse::Status s = ir::verify(kb.ir());
  LSE_EXPECT(!s.ok());
  LSE_EXPECT(s.message().find("no granularity") != std::string::npos);
}

LSE_TEST(a_window_base_may_offset_an_address_but_not_scale_one) {
  // Legal: the base moves the origin, the stride stays literal.
  {
    ir::KernelBody kb(kTypes, kTable);
    env::Emit e{&kb};
    const ir::Buffer<ir::f32> buf(&kb, &kb.types(), "b0");
    const auto base = e.window_base("tokens", "k.token_base");
    for (auto r : e.range(16u)) {
      buf[(base + r) * 1024u + e.thread_id()] = e.f32(1.0f);
    }
    LSE_EXPECT(ir::verify(kb.ir()).ok());
  }
  // Illegal: a base used as a multiplier is a varying stride, which is the
  // whole thing baking extents in exists to prevent.
  {
    ir::KernelBody kb(kTypes, kTable);
    env::Emit e{&kb};
    const ir::Buffer<ir::f32> buf(&kb, &kb.types(), "b0");
    const auto base = e.window_base("tokens", "k.token_base");
    buf[e.thread_id() * base] = e.f32(1.0f);
    const lse::Status s = ir::verify(kb.ir());
    LSE_EXPECT(!s.ok());
    LSE_EXPECT(s.message().find("more than an offset") != std::string::npos);
  }
  // Illegal: a base is where a window starts, never how long it is.
  {
    ir::KernelBody kb(kTypes, kTable);
    env::Emit e{&kb};
    auto acc = e.var(0.0f);
    const auto base = e.window_base("tokens", "k.token_base");
    for (auto t : e.range(e.u32(0), base)) {
      (void)t;
      acc = acc.read() + e.f32(1.0f);
    }
    const lse::Status s = ir::verify(kb.ir());
    LSE_EXPECT(!s.ok());
    LSE_EXPECT(s.message().find("not a trip count") != std::string::npos);
  }
}

// ---------------------------------------------------------------------------
// CSE
// ---------------------------------------------------------------------------

LSE_TEST(cse_folds_a_recomputed_binding_and_does_nothing_without_it) {
  ir::KernelBody kb(kTypes, kTable);
  env::Emit e{&kb};
  const auto a = e.let(lse::math::local_id() / 32u);
  const auto b = e.let(lse::math::local_id() / 32u);
  auto acc = e.var(0.0f);
  acc = acc.read() + lse::math::cast<ir::f32>(a + b);

  // Without the pass both bindings are there — otherwise the assertion below
  // would prove nothing.
  const std::string before = ir::lower(kb.ir());
  LSE_EXPECT_EQ(count_of(before, "(threadIdx.x / 32u)"), 2u);

  const std::size_t fired = run_one(kb.ir(), ir::make_cse());
  LSE_EXPECT(fired >= 1);
  const std::string after = ir::lower(kb.ir());
  LSE_EXPECT_EQ(count_of(after, "(threadIdx.x / 32u)"), 1u);
  LSE_EXPECT(ir::verify(kb.ir()).ok());
}

LSE_TEST(cse_never_merges_two_different_buffers) {
  ir::KernelBody kb(kTypes, kTable);
  env::Emit e{&kb};
  const ir::Buffer<ir::f32> in(&kb, &kb.types(), "in0");
  const ir::Buffer<ir::f32> out(&kb, &kb.types(), "out");
  out[e.thread_id()] = in[e.thread_id()].read();

  (void)run_one(kb.ir(), ir::make_cse());
  const std::string src = ir::lower(kb.ir());
  // Two symbols of the same type are not one value just because their ops
  // look alike. This exact confusion turned `in0` into `out` and produced
  // fluent nonsense.
  LSE_EXPECT(src.find("out[i] = in0[i];") != std::string::npos);
}

LSE_TEST(cse_leaves_a_value_a_raw_statement_names_alone) {
  ir::KernelBody kb(kTypes, kTable);
  kb.set_store([](std::string_view index, std::string_view value) {
    return "b9[" + std::string(index) + "] = " + std::string(value) + ";\n";
  });
  env::Emit e{&kb};
  const auto row = e.let(lse::math::workgroup_id_y());
  e.store(row, e.f32(1.0f));
  const auto again = e.let(lse::math::workgroup_id_y());
  auto acc = e.var(0.0f);
  acc = acc.read() + lse::math::cast<ir::f32>(again);

  (void)run_one(kb.ir(), ir::make_cse());
  const std::string src = ir::lower(kb.ir());
  // The store's text names the first binding, so its declaration has to stay
  // even though the second is an identical expression.
  const std::string first_name = kb.ir().value(row.id()).name;
  LSE_EXPECT(src.find("const unsigned int " + first_name + " = blockIdx.y;") !=
             std::string::npos);
  LSE_EXPECT(src.find("b9[" + first_name + "]") != std::string::npos);
  LSE_EXPECT(ir::verify(kb.ir()).ok());
}

// ---------------------------------------------------------------------------
// lds_fold
// ---------------------------------------------------------------------------

LSE_TEST(lds_fold_merges_two_identical_stagings) {
  ir::KernelBody kb(kTypes, kTable);
  build_two_stages(kb, StagedRun{});

  const std::string before = ir::lower(kb.ir());
  LSE_EXPECT_EQ(count_of(before, "__shared__ float"), 2u);

  (void)run_one(kb.ir(), ir::make_cse());
  const std::size_t fired = run_one(kb.ir(), ir::make_lds_fold());
  LSE_EXPECT(fired >= 1);
  (void)run_one(kb.ir(), ir::make_dce());
  const std::string after = ir::lower(kb.ir());
  LSE_EXPECT_EQ(count_of(after, "__shared__ float"), 1u);
  // Both stages read the surviving array.
  LSE_EXPECT(ir::verify(kb.ir()).ok());
}

LSE_TEST(lds_fold_drops_the_second_fill_only_when_the_first_guard_covers_it) {
  // Wide stage first: the narrow stage's guard implies it, so its fill is pure
  // repetition behind a barrier and goes.
  {
    ir::KernelBody kb(kTypes, kTable);
    build_two_stages(kb, StagedRun{64, 16, 4, true});
    (void)run_one(kb.ir(), ir::make_cse());
    (void)run_one(kb.ir(), ir::make_lds_fold());
    (void)run_one(kb.ir(), ir::make_dce());
    const std::string after = ir::lower(kb.ir());
    LSE_EXPECT_EQ(count_of(after, "__shared__ float"), 1u);
    LSE_EXPECT_EQ(count_of(after, "] = b0["), 1u);
  }
  // Narrow stage first: the wide stage runs in workgroups the narrow one never
  // filled, so its fill must survive.
  {
    ir::KernelBody kb(kTypes, kTable);
    build_two_stages(kb, StagedRun{64, 4, 16, true});
    (void)run_one(kb.ir(), ir::make_cse());
    (void)run_one(kb.ir(), ir::make_lds_fold());
    (void)run_one(kb.ir(), ir::make_dce());
    const std::string after = ir::lower(kb.ir());
    LSE_EXPECT_EQ(count_of(after, "__shared__ float"), 1u);
    LSE_EXPECT_EQ(count_of(after, "] = b0["), 2u);
  }
}

LSE_TEST(lds_fold_refuses_two_stagings_from_different_sources) {
  ir::KernelBody kb(kTypes, kTable);
  build_two_stages(kb, StagedRun{64, 8, 8, false});
  (void)run_one(kb.ir(), ir::make_cse());
  const std::size_t fired = run_one(kb.ir(), ir::make_lds_fold());
  LSE_EXPECT_EQ(fired, 0u);
  const std::string after = ir::lower(kb.ir());
  LSE_EXPECT_EQ(count_of(after, "__shared__ float"), 2u);
}

// "What this body writes" is read off raw statement text, and that read has to
// match whole identifiers. A store epilogue into `b10` does not write `b1`, but
// a substring search says it does — and a staging whose source is falsely marked
// written never folds. The case is real, not hypothetical: a 13-binding
// quantized sibling run stores through b10, b11 and b12.
LSE_TEST(lds_fold_reads_a_raw_store_as_identifiers_not_substrings) {
  ir::KernelBody kb(kTypes, kTable);
  kb.set_store([](std::string_view index, std::string_view value) {
    return "b10[" + std::string(index) + "] = " + std::string(value) + ";\n";
  });
  build_two_stages(kb, StagedRun{64, 8, 8, true, "b1", "b2"});
  {
    env::Emit e{&kb};
    e.store(e.let(lse::math::workgroup_id_y()), e.f32(1.0f));
  }
  const std::string before = ir::lower(kb.ir());
  LSE_EXPECT_EQ(count_of(before, "__shared__ float"), 2u);
  LSE_EXPECT(before.find("b10[") != std::string::npos);

  (void)run_one(kb.ir(), ir::make_cse());
  LSE_EXPECT(run_one(kb.ir(), ir::make_lds_fold()) >= 1);
  (void)run_one(kb.ir(), ir::make_dce());
  LSE_EXPECT_EQ(count_of(ir::lower(kb.ir()), "__shared__ float"), 1u);
  LSE_EXPECT(ir::verify(kb.ir()).ok());
}

// The other direction of the same predicate: a store that really does name the
// staged buffer must still block the fold, or the pass approves a staging whose
// source this kernel has already overwritten.
LSE_TEST(lds_fold_still_refuses_when_the_store_names_the_staged_buffer) {
  ir::KernelBody kb(kTypes, kTable);
  kb.set_store([](std::string_view index, std::string_view value) {
    return "b1[" + std::string(index) + "] = " + std::string(value) + ";\n";
  });
  build_two_stages(kb, StagedRun{64, 8, 8, true, "b1", "b2"});
  {
    env::Emit e{&kb};
    e.store(e.let(lse::math::workgroup_id_y()), e.f32(1.0f));
  }
  (void)run_one(kb.ir(), ir::make_cse());
  LSE_EXPECT_EQ(run_one(kb.ir(), ir::make_lds_fold()), 0u);
  LSE_EXPECT_EQ(count_of(ir::lower(kb.ir()), "__shared__ float"), 2u);
}

// ---------------------------------------------------------------------------
// Workgroup scratch accounting
// ---------------------------------------------------------------------------

// The number a fused group is admitted on has to be the number its text asks
// for, and separate declarations SUM: the hardware gives each one its own LDS
// offset even in disjoint block scopes (two `__shared__ float[2176]` in sibling
// braces report sharedSizeBytes 17408 on gfx1151, against 8704 for one).
LSE_TEST(a_body_reports_the_workgroup_bytes_its_text_declares) {
  ir::KernelBody kb(kTypes, kTable);
  env::Emit e{&kb};
  LSE_EXPECT_EQ(kb.ir().workgroup_bytes(), 0u);

  auto a = e.lds<ir::f32>(64);
  LSE_EXPECT_EQ(kb.ir().workgroup_bytes(), 256u);
  auto b = e.lds<ir::f32>(64);
  // Summed, not maximized. A max would still say 256 here.
  LSE_EXPECT_EQ(kb.ir().workgroup_bytes(), 512u);
  // 16-byte aligned per array, the way kir::Lds prices a reservation: three
  // floats cost a whole 16-byte line, so this is 512 + 16 and not 512 + 12.
  auto c = e.lds<ir::f32>(3);
  LSE_EXPECT_EQ(kb.ir().workgroup_bytes(), 528u);
  LSE_EXPECT_EQ(kb.ir().workgroup_bytes(), kb.lds().used());

  auto acc = e.var(0.0f);
  acc = acc.read() + a[0].read() + b[0].read() + c[0].read();
  LSE_EXPECT_EQ(count_of(ir::lower(kb.ir()), "__shared__ float"), 3u);
}

// An array a fold left unread but DCE has not deleted is still an array the
// compiler charges for, so it is still counted. Correctness of the budget can
// therefore not depend on either pass having fired.
LSE_TEST(workgroup_bytes_counts_scratch_until_it_is_actually_deleted) {
  ir::KernelBody kb(kTypes, kTable);
  env::Emit e{&kb};
  auto used = e.lds<ir::f32>(64);
  auto unread = e.lds<ir::f32>(64);
  (void)unread;
  auto acc = e.var(0.0f);
  acc = acc.read() + used[0].read();

  LSE_EXPECT_EQ(kb.ir().workgroup_bytes(), 512u);
  LSE_EXPECT(run_one(kb.ir(), ir::make_dce()) >= 1);
  LSE_EXPECT_EQ(kb.ir().workgroup_bytes(), 256u);
  LSE_EXPECT_EQ(count_of(ir::lower(kb.ir()), "__shared__ float"), 1u);
}

// ---------------------------------------------------------------------------
// DCE
// ---------------------------------------------------------------------------

LSE_TEST(dce_deletes_an_allocation_nothing_reads_and_nothing_else) {
  ir::KernelBody kb(kTypes, kTable);
  env::Emit e{&kb};
  auto used = e.lds<ir::f32>(16);
  auto unused = e.lds<ir::f32>(16);
  (void)unused;
  auto acc = e.var(0.0f);
  acc = acc.read() + used[0].read();

  const std::string before = ir::lower(kb.ir());
  LSE_EXPECT_EQ(count_of(before, "__shared__ float"), 2u);

  const std::size_t fired = run_one(kb.ir(), ir::make_dce());
  LSE_EXPECT(fired >= 1);
  const std::string after = ir::lower(kb.ir());
  LSE_EXPECT_EQ(count_of(after, "__shared__ float"), 1u);
  LSE_EXPECT(after.find(kb.ir().value(used.id()).name) != std::string::npos);
  LSE_EXPECT(ir::verify(kb.ir()).ok());
}

LSE_TEST(dce_keeps_what_a_raw_statement_names) {
  ir::KernelBody kb(kTypes, kTable);
  kb.set_store([](std::string_view index, std::string_view value) {
    return "b9[" + std::string(index) + "] = " + std::string(value) + ";\n";
  });
  env::Emit e{&kb};
  const auto row = e.let(lse::math::workgroup_id_y());
  e.store(row, e.f32(2.0f));

  (void)run_one(kb.ir(), ir::make_dce());
  const std::string after = ir::lower(kb.ir());
  const std::string name = kb.ir().value(row.id()).name;
  LSE_EXPECT(after.find("const unsigned int " + name + " = blockIdx.y;") !=
             std::string::npos);
}

// ---------------------------------------------------------------------------
// The pipeline
// ---------------------------------------------------------------------------

LSE_TEST(the_default_pipeline_runs_every_pass_and_verifies) {
  ir::KernelBody kb(kTypes, kTable);
  build_two_stages(kb, StagedRun{});
  std::vector<ir::PassStat> stats;
  LSE_EXPECT(ir::default_pipeline().run(kb.ir(), &stats).ok());
  LSE_EXPECT_EQ(stats.size(), 4u);
  LSE_EXPECT(stats[0].name == "cse");
  LSE_EXPECT(stats[1].name == "factor_hoist");
  LSE_EXPECT(stats[2].name == "lds_fold");
  LSE_EXPECT(stats[3].name == "dce");
  LSE_EXPECT(stats[2].fired >= 1);
  LSE_EXPECT_EQ(count_of(ir::lower(kb.ir()), "__shared__ float"), 1u);
}

LSE_TEST_MAIN()

LSE_TEST(factor_hoist_lifts_an_invariant_scale_out_of_a_sum) {
  // The rms_norm-into-linear shape reduced to its arithmetic: a contraction
  // whose every term carries the same scalar. The scale belongs outside the
  // sum, and moving it there is what lets it be computed AFTER the loop
  // instead of before -- which is what would let a norm and the linear that
  // consumes it share one k-loop instead of taking one launch each.
  ir::KernelBody kb(kTypes, kTable);
  env::Emit e{&kb};
  const ir::Buffer<ir::f32> x(&kb, &kb.types(), "in0");
  const ir::Buffer<ir::f32> sb(&kb, &kb.types(), "in1");
  const ir::Buffer<ir::f32> out(&kb, &kb.types(), "out");

  auto s = sb[e.thread_id()].read();   // invariant: defined before the loop
  auto acc = e.var(0.0f);
  for (auto k : e.range(64u)) {
    acc = acc.read() + (x[k].read() * s);
  }
  out[e.thread_id()] = acc.read();

  const std::size_t fired = run_one(kb.ir(), ir::make_factor_hoist());
  LSE_EXPECT_EQ(fired, 1u);
  LSE_EXPECT(ir::verify(kb.ir()).ok());

  const std::string after = ir::lower(kb.ir());
  const std::size_t open = after.find("for (");
  LSE_EXPECT(open != std::string::npos);
  const std::size_t close = after.find('}', open);
  LSE_EXPECT(close != std::string::npos);
  const std::string in_loop = after.substr(open, close - open);
  const std::string after_loop = after.substr(close);
  // The multiply left the loop body...
  LSE_EXPECT_EQ(count_of(in_loop, "*"), 0u);
  // ...and is paid once, after it.
  LSE_EXPECT(count_of(after_loop, "*") >= 1u);
}

LSE_TEST(factor_hoist_refuses_an_accumulator_that_does_not_start_at_zero) {
  // s * (init + SUM) is not init + s * SUM. The rewrite is only valid from a
  // zero start, and a pass that fired here would be silently wrong.
  ir::KernelBody kb(kTypes, kTable);
  env::Emit e{&kb};
  const ir::Buffer<ir::f32> x(&kb, &kb.types(), "in0");
  const ir::Buffer<ir::f32> sb(&kb, &kb.types(), "in1");
  const ir::Buffer<ir::f32> out(&kb, &kb.types(), "out");

  auto s = sb[e.thread_id()].read();
  auto acc = e.var(1.0f);              // <- not zero
  for (auto k : e.range(64u)) {
    acc = acc.read() + (x[k].read() * s);
  }
  out[e.thread_id()] = acc.read();

  LSE_EXPECT_EQ(run_one(kb.ir(), ir::make_factor_hoist()), 0u);
  LSE_EXPECT(ir::verify(kb.ir()).ok());
}

LSE_TEST(factor_hoist_refuses_a_factor_that_moves_with_the_loop) {
  // Both halves of the product vary with k, so nothing distributes out.
  ir::KernelBody kb(kTypes, kTable);
  env::Emit e{&kb};
  const ir::Buffer<ir::f32> x(&kb, &kb.types(), "in0");
  const ir::Buffer<ir::f32> y(&kb, &kb.types(), "in1");
  const ir::Buffer<ir::f32> out(&kb, &kb.types(), "out");

  auto acc = e.var(0.0f);
  for (auto k : e.range(64u)) {
    acc = acc.read() + (x[k].read() * y[k].read());
  }
  out[e.thread_id()] = acc.read();

  LSE_EXPECT_EQ(run_one(kb.ir(), ir::make_factor_hoist()), 0u);
  LSE_EXPECT(ir::verify(kb.ir()).ok());
}

LSE_TEST(factor_hoist_lifts_a_scale_out_of_an_fma_contraction) {
  // The form a real contraction is actually written in. The emitter spells
  // its inner loop with the dialect's fma row, so a matcher that only knew
  // `acc + (a * b)` would never fire on anything the model emits.
  ir::KernelBody kb(kTypes, kTable);
  env::Emit e{&kb};
  const ir::Buffer<ir::f32> x(&kb, &kb.types(), "in0");
  const ir::Buffer<ir::f32> sb(&kb, &kb.types(), "in1");
  const ir::Buffer<ir::f32> out(&kb, &kb.types(), "out");

  auto s = sb[e.thread_id()].read();   // invariant: the norm scale
  auto acc = e.var(0.0f);
  for (auto k : e.range(64u)) {
    acc = lse::math::fma(x[k].read(), s, acc.read());
  }
  out[e.thread_id()] = acc.read();

  LSE_EXPECT_EQ(run_one(kb.ir(), ir::make_factor_hoist()), 1u);
  LSE_EXPECT(ir::verify(kb.ir()).ok());

  const std::string after = ir::lower(kb.ir());
  const std::size_t open = after.find("for (");
  const std::size_t close = after.find('}', open);
  LSE_EXPECT(open != std::string::npos && close != std::string::npos);
  // The fma is gone from the loop; a plain add took its place.
  LSE_EXPECT_EQ(count_of(after.substr(open, close - open), "fma"), 0u);
  LSE_EXPECT(count_of(after.substr(close), "*") >= 1u);
}

LSE_TEST(factor_hoist_finds_a_scale_the_loop_itself_computed) {
  // Invariance is a property of the DEPENDENCY CHAIN, not of position. This
  // scale is written inside the loop body, so a test that asked only "was it
  // defined outside" would refuse the hoist -- but it is built from two
  // constants and moves with nothing, so it is invariant and the factor comes
  // out just the same.
  ir::KernelBody kb(kTypes, kTable);
  env::Emit e{&kb};
  const ir::Buffer<ir::f32> x(&kb, &kb.types(), "in0");
  const ir::Buffer<ir::f32> out(&kb, &kb.types(), "out");

  auto acc = e.var(0.0f);
  for (auto k : e.range(64u)) {
    const auto s = e.f32(2.0f) * e.f32(4.0f);   // computed IN the loop
    acc = acc.read() + (x[k].read() * s);
  }
  out[e.thread_id()] = acc.read();

  LSE_EXPECT_EQ(run_one(kb.ir(), ir::make_factor_hoist()), 1u);
  LSE_EXPECT(ir::verify(kb.ir()).ok());
}

LSE_TEST(factor_hoist_pulls_a_factor_out_of_a_nested_product) {
  // The factor is buried in a product tree rather than sitting at the top of
  // the term. Flattening to sum-of-products finds it wherever it is, which a
  // shape-matcher keyed on one spelling would not.
  ir::KernelBody kb(kTypes, kTable);
  env::Emit e{&kb};
  const ir::Buffer<ir::f32> x(&kb, &kb.types(), "in0");
  const ir::Buffer<ir::f32> y(&kb, &kb.types(), "in1");
  const ir::Buffer<ir::f32> sb(&kb, &kb.types(), "in2");
  const ir::Buffer<ir::f32> out(&kb, &kb.types(), "out");

  auto s = sb[e.thread_id()].read();
  auto acc = e.var(0.0f);
  for (auto k : e.range(64u)) {
    acc = acc.read() + ((x[k].read() * s) * y[k].read());
  }
  out[e.thread_id()] = acc.read();

  LSE_EXPECT_EQ(run_one(kb.ir(), ir::make_factor_hoist()), 1u);
  LSE_EXPECT(ir::verify(kb.ir()).ok());
  const std::string after = ir::lower(kb.ir());
  const std::size_t open = after.find("for (");
  const std::size_t close = after.find('}', open);
  LSE_EXPECT_EQ(count_of(after.substr(open, close - open), "in2"), 0u);
}

namespace {

// The two memory ops in a body, in order.
std::vector<ir::OpId> memory_ops(const ir::Body& b, ir::RegionId r) {
  std::vector<ir::OpId> out;
  for (ir::OpId id : b.region(r).ops) {
    const ir::Operation& o = b.op(id);
    if (o.erased) continue;
    ir::Access probe;
    if (ir::access_of(b, id, &probe)) out.push_back(id);
    for (ir::RegionId sub : o.regions) {
      for (ir::OpId n : memory_ops(b, sub)) out.push_back(n);
    }
  }
  return out;
}

}  // namespace

LSE_TEST(alias_proves_two_offsets_of_one_buffer_are_different_elements) {
  ir::KernelBody kb(kTypes, kTable);
  env::Emit e{&kb};
  const ir::Buffer<ir::f32> x(&kb, &kb.types(), "in0");
  const ir::Buffer<ir::f32> out(&kb, &kb.types(), "out");
  const auto i = e.let(e.thread_id());
  out[e.thread_id()] = x[i].read() + x[i + e.u32(1u)].read();

  const auto ops = memory_ops(kb.ir(), kb.ir().entry());
  LSE_EXPECT(ops.size() >= 2);
  ir::Access a, b;
  LSE_EXPECT(ir::access_of(kb.ir(), ops[0], &a));
  LSE_EXPECT(ir::access_of(kb.ir(), ops[1], &b));
  // in0[i] and in0[i+1] are one apart: provably different elements.
  LSE_EXPECT(ir::alias_of(a, b) == ir::Alias::kNo);
  LSE_EXPECT(ir::alias_of(a, a) == ir::Alias::kMust);
}

LSE_TEST(alias_refuses_to_separate_two_different_buffer_symbols) {
  // THE UNSOUND MOVE THIS EXISTS TO PREVENT. Bindings are __restrict__, but
  // plan_slots recycles slots, so two symbols can name one allocation. Only a
  // caller holding the slot map may claim otherwise.
  ir::KernelBody kb(kTypes, kTable);
  env::Emit e{&kb};
  const ir::Buffer<ir::f32> x(&kb, &kb.types(), "in0");
  const ir::Buffer<ir::f32> y(&kb, &kb.types(), "in1");
  const ir::Buffer<ir::f32> out(&kb, &kb.types(), "out");
  const auto i = e.let(e.thread_id());
  out[e.thread_id()] = x[i].read() + y[i].read();

  const auto ops = memory_ops(kb.ir(), kb.ir().entry());
  LSE_EXPECT(ops.size() >= 2);
  ir::Access a, b;
  LSE_EXPECT(ir::access_of(kb.ir(), ops[0], &a));
  LSE_EXPECT(ir::access_of(kb.ir(), ops[1], &b));
  LSE_EXPECT(ir::alias_of(a, b) == ir::Alias::kMaybe);

  // With an oracle that actually knows they are separate allocations, it can.
  const ir::DistinctAllocations distinct =
      [](ir::ValueId p, ir::ValueId q) { return p != q; };
  LSE_EXPECT(ir::alias_of(a, b, &distinct) == ir::Alias::kNo);
}

LSE_TEST(alias_keeps_a_variable_index_difference_unknown) {
  // in0[i] against in0[j]: the difference is not a constant, so nothing is
  // known and the answer has to be kMaybe rather than a guess.
  ir::KernelBody kb(kTypes, kTable);
  env::Emit e{&kb};
  const ir::Buffer<ir::f32> x(&kb, &kb.types(), "in0");
  const ir::Buffer<ir::f32> out(&kb, &kb.types(), "out");
  const auto i = e.let(e.thread_id());
  const auto j = e.let(lse::math::workgroup_id_x());
  out[e.thread_id()] = x[i].read() + x[j].read();

  const auto ops = memory_ops(kb.ir(), kb.ir().entry());
  LSE_EXPECT(ops.size() >= 2);
  ir::Access a, b;
  LSE_EXPECT(ir::access_of(kb.ir(), ops[0], &a));
  LSE_EXPECT(ir::access_of(kb.ir(), ops[1], &b));
  LSE_EXPECT(ir::alias_of(a, b) == ir::Alias::kMaybe);
}

LSE_TEST(may_reorder_lets_two_reads_past_each_other_but_not_a_write) {
  ir::Access r1, r2, w;
  r1.buffer = 7; r1.index = ir::AffineExpr::constant(0); r1.is_write = false;
  r2.buffer = 7; r2.index = ir::AffineExpr::constant(0); r2.is_write = false;
  w = r1; w.is_write = true;
  // Same element, both reads: order does not matter.
  LSE_EXPECT(ir::may_reorder(r1, r2));
  // Same element, one writes: it does.
  LSE_EXPECT(!ir::may_reorder(r1, w));
  // A write to a provably different element may move.
  ir::Access w2 = w;
  w2.index = ir::AffineExpr::constant(4);
  LSE_EXPECT(ir::may_reorder(r1, w2));
}

LSE_TEST(alias_separates_lds_from_a_global_buffer_without_an_oracle) {
  // The one case where two distinct symbols ARE provably distinct memory:
  // they live in different address spaces. Slot recycling can put two global
  // bindings on one allocation, but it cannot put a global binding and a
  // __shared__ array on one -- so this needs no slot map to decide, and
  // without it every staging fold would refuse itself against its own fill.
  ir::KernelBody kb(kTypes, kTable);
  env::Emit e{&kb};
  const ir::Buffer<ir::f32> g(&kb, &kb.types(), "in0");
  auto lds = kb.shared<ir::f32, 64>("s0");
  const auto i = e.let(e.thread_id());
  lds[i] = g[i].read();

  const auto ops = memory_ops(kb.ir(), kb.ir().entry());
  LSE_EXPECT(ops.size() >= 2);
  ir::Access w, r;
  bool got_w = false, got_r = false;
  for (ir::OpId id : ops) {
    ir::Access acc;
    if (!ir::access_of(kb.ir(), id, &acc)) continue;
    if (acc.is_write && !got_w) { w = acc; got_w = true; }
    if (!acc.is_write && !got_r) { r = acc; got_r = true; }
  }
  LSE_EXPECT(got_w && got_r);
  LSE_EXPECT(w.space != r.space);
  LSE_EXPECT(ir::alias_of(w, r) == ir::Alias::kNo);
  LSE_EXPECT(ir::may_reorder(w, r));
}
