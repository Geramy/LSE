#include "harness.hpp"
#include "lse/backends/hrx/arch_database.hpp"
#include "lse/backends/hrx/hipc/hip_sources.hpp"
#include "lse/backends/hrx/loomc/loom_sources.hpp"
#include "lse/dispatch/arch/tuning.hpp"
#include "lse/dispatch/quant.hpp"

#include <array>

using namespace lse;
namespace {
struct Fixture {
  backend::DeviceInfo device;
  backend::AmdDeviceInfo amd;
  std::array<Shape, 4> inputs;
  std::array<DType, 4> dtypes{DType::kF32, DType::kU32, DType::kBF16, DType::kBF16};
  graph::DialectSourceTable intrinsics;
  graph::KernelShapes shapes;

  Fixture(int m, int n, int k, int bits, bool loom)
      : intrinsics(loom ? backend::loom_sources() : backend::hip_sources()) {
    device.arch = "gfx1201";
    device.compute_units = 64;
    device.wavefront_size = 32;
    device.max_threads_per_workgroup = 1024;
    device.lds_bytes_per_workgroup = 65536;
    backend::apply_arch_defaults(device, amd);
    device.extension_id = backend::AmdDeviceInfo::kExtensionId;
    device.extension = &amd;
    inputs = {Shape{m, k}, Shape{n, k * bits / 32}, Shape{n, k / 64}, Shape{n, k / 64}};
    shapes.inputs = inputs;
    shapes.input_dtypes = dtypes;
    shapes.output = Shape{m, n};
    shapes.output_dtype = DType::kF32;
    shapes.iattrs = {bits, 64, 0, 0};
    shapes.device = &device;
    shapes.intrinsics = &intrinsics;
  }
  dispatch::QuantPlan plan(bool indexed = false) const {
    return dispatch::quant_plan(shapes, indexed);
  }
};

void scalar(const dispatch::QuantPlan& p) {
  LSE_EXPECT(p.matrix == nullptr);
  LSE_EXPECT(p.implementation == dispatch::QuantMatrix::kNone);
}
void baseline(const dispatch::QuantPlan& p) {
  scalar(p);
  LSE_EXPECT(!p.int8_activations);
  LSE_EXPECT(!p.rotate_decode_panel);
  LSE_EXPECT_EQ(p.decode_columns, 1u);
  LSE_EXPECT_EQ(p.prefill_rows, 1u);
  LSE_EXPECT_EQ(p.row_ladder_ceiling, 0u);
}
}

LSE_TEST(quant_M8_rate_panel_is_shape_and_device_qualified) {
  for (bool loom : {false, true}) {
    Fixture f(8, 48, 5120, 4, loom);
    f.inputs[0] = Shape{1, 8, 5120};
    f.shapes.output = Shape{1, 8, 48};
    LSE_EXPECT(f.plan().shared_activation_panel && f.plan().int8_activations);
    LSE_EXPECT_EQ(dispatch::q4_shared_panel_rows(f.shapes), 8u);
    LSE_EXPECT_EQ(dispatch::q4_shared_panel_load_chunks(f.shapes), 1u);
    LSE_EXPECT(!f.plan(true).shared_activation_panel);
    f.shapes.staged.name = "owned_row";
    LSE_EXPECT(!f.plan().shared_activation_panel);
    f.shapes.staged.name = {};
    f.device.arch = "gfx1100";
    LSE_EXPECT(!f.plan().shared_activation_panel);
    f.device.arch = "gfx1201";
    f.device.wavefront_size = 64;
    LSE_EXPECT(!f.plan().shared_activation_panel);
    f.device.wavefront_size = 32;
    f.device.max_threads_per_workgroup = 128;
    LSE_EXPECT(!f.plan().shared_activation_panel);
    f.device.max_threads_per_workgroup = 1024;
    f.amd.has_dot4_iu8 = false;
    LSE_EXPECT(!f.plan().shared_activation_panel);
    f.amd.has_dot4_iu8 = true;
    f.dtypes[2] = f.dtypes[3] = DType::kF32;
    LSE_EXPECT(!f.plan().shared_activation_panel);
    for (const auto m : {1, 6, 7, 9})
      LSE_EXPECT(!Fixture(m, 48, 5120, 4, loom).plan().shared_activation_panel);
    // The 4-row (MTP=3) verify pass shares the panel too.
    LSE_EXPECT(Fixture(4, 48, 5120, 4, loom).plan().shared_activation_panel);
    LSE_EXPECT(!Fixture(8, 49, 5120, 4, loom).plan().shared_activation_panel);
    LSE_EXPECT(!Fixture(8, 48, 6144, 4, loom).plan().shared_activation_panel);
  }
}
LSE_TEST(quant_ffn_m512_uses_architecture_defaults) {
  for (bool loom : {false, true}) {
    for (int bits : {4, 6}) {
      for (auto [n, k] : {std::pair{17408, 5120}, std::pair{5120, 17408}}) {
        Fixture f(512, n, k, bits, loom);
        const auto p = f.plan();
        LSE_EXPECT(p.matrix != nullptr);
        if (!p.matrix) continue;
        LSE_EXPECT(p.implementation == (bits == 4 ? dispatch::QuantMatrix::kInt8Lds
                                                : dispatch::QuantMatrix::kBF16));
        LSE_EXPECT_EQ(p.int8_activations, bits == 4);
        LSE_EXPECT(p.matrix->acc == (bits == 4 ? math::MatrixElem::kI32
                                            : math::MatrixElem::kF32));
        LSE_EXPECT(p.matrix->operand == (bits == 4 ? math::MatrixElem::kSU8
                                                : math::MatrixElem::kBF16));
      }
    }
  }
}

LSE_TEST(quant_defaults_are_shape_specific) {
  for (bool loom : {false, true}) {
    for (int m : {1, 2, 3, 7, 8, 9, 32, 512}) {
      for (int bits : {4, 6, 8}) {
        Fixture f(m, 17, 64, bits, loom);
        const auto p = f.plan();
        LSE_EXPECT_EQ(p.int8_activations, bits == 4 && (m <= 8 || m >= 16));
        LSE_EXPECT_EQ(p.matrix != nullptr, bits == 4 && m >= 16);
        if (bits != 4 || m < 16) scalar(p);
      }
    }
    Fixture decode(1, 17408, 5120, 6, loom);
    scalar(decode.plan());
    LSE_EXPECT_EQ(decode.plan().decode_columns, 4u);
    LSE_EXPECT(decode.plan().rotate_decode_panel);
    for (auto [m, rows] : {std::pair{2, 2u}, std::pair{4, 4u}, std::pair{32, 8u},
                           std::pair{256, 8u}}) {
      Fixture f(m, 17408, 5120, 6, loom);
      scalar(f.plan());
      LSE_EXPECT_EQ(f.plan().prefill_rows, rows);
    }
    Fixture tail(1, 19, 5120, 6, loom);
    LSE_EXPECT_EQ(tail.plan().decode_columns, 1u);
    Fixture other(512, 10240, 5120, 6, loom);
    scalar(other.plan());
  }
}

LSE_TEST(quant_mtp_int8_requires_supported_architecture_and_intrinsics) {
  for (bool loom : {false, true}) {
    for (int m : {2, 3, 7, 8}) {
      for (int fault = 0; fault < 7; ++fault) {
        Fixture f(m, 17, 1024, 4, loom);
        if (fault == 0) f.device.arch = "gfx1200";
        if (fault == 1) f.device.wavefront_size = 64;
        if (fault == 2) f.amd.has_dot4_iu8 = false;
        if (fault == 3) f.intrinsics = {};
        if (fault == 4) {
          f.shapes.iattrs[1] = 128;
          f.inputs[2] = f.inputs[3] = Shape{17, 8};
        }
        if (fault == 5) f.dtypes[0] = DType::kBF16;
        baseline(f.plan(fault == 6));
      }
      Fixture allowed(m, 17, 1024, 4, loom);
      LSE_EXPECT(allowed.plan().int8_activations);
      scalar(allowed.plan());
    }
    for (int m : {1, 512}) {
      Fixture previous(m, 17, 1024, 4, loom);
      previous.device.arch = "gfx1151";
      LSE_EXPECT(previous.plan().int8_activations);
    }
  }
}

LSE_TEST(quant_mtp_ffn_row_ladder_uses_measured_shapes) {
  const auto* primitive = dynamic_cast<const graph::KernelPrimitiveBase*>(
      graph::find_primitive("quant_linear"));
  LSE_EXPECT(primitive != nullptr);
  if (!primitive) return;
  for (bool loom : {false, true}) {
    for (int m : {3, 7}) {
      for (auto [n, k] : {std::pair{17408, 5120}, std::pair{5120, 17408}}) {
        Fixture f(m, n, k, 4, loom);
        const auto plan = f.plan();
        LSE_EXPECT_EQ(plan.row_ladder_ceiling, m == 3 ? 4u : 8u);
        LSE_EXPECT(plan.int8_activations);
        scalar(plan);
        const auto geometry = primitive->plan(f.shapes);
        LSE_EXPECT_EQ(geometry.workgroup_count[0], static_cast<std::uint32_t>(n / 8));
        LSE_EXPECT_EQ(geometry.workgroup_count[1], 1u);
        LSE_EXPECT_EQ(geometry.workgroup_size[0], 256u);
        LSE_EXPECT(geometry.lds_bytes <= f.device.lds_bytes_per_workgroup);
        const std::uint32_t minimum = k == 5120 ? (m == 3 ? 33088u : 33152u)
                                               : (m == 3 ? 56128u : 56192u);
        f.device.lds_bytes_per_workgroup = minimum;
        LSE_EXPECT_EQ(f.plan().row_ladder_ceiling, m == 3 ? 4u : 8u);
        --f.device.lds_bytes_per_workgroup;
        LSE_EXPECT_EQ(f.plan().row_ladder_ceiling, 0u);
      }
    }
    for (int m : {1, 2, 4, 5, 6, 8, 9, 32, 512}) {
      Fixture f(m, 17408, 5120, 4, loom);
      LSE_EXPECT_EQ(f.plan().row_ladder_ceiling, 0u);
    }
    for (int fault = 0; fault < 11; ++fault) {
      Fixture f(3, 17408, 5120, 4, loom);
      if (fault == 0) f.device.arch = "gfx1200";
      if (fault == 1) f.device.wavefront_size = 64;
      if (fault == 2) f.amd.has_dot4_iu8 = false;
      if (fault == 3) f.intrinsics = {};
      if (fault == 4) f.dtypes[2] = f.dtypes[3] = DType::kF32;
      if (fault == 5) f.device.max_threads_per_workgroup = 255;
      if (fault == 6) {
        f.inputs[1] = Shape{17409, 640};
        f.inputs[2] = f.inputs[3] = Shape{17409, 80};
        f.shapes.output = Shape{3, 17409};
      }
      if (fault == 7) f.shapes.iattrs[0] = 8;
      if (fault == 9) f.shapes.staged = {"caller", 5120};
      if (fault == 10) f.shapes.staged_quant.codes = "caller";
      LSE_EXPECT_EQ(f.plan(fault == 8).row_ladder_ceiling, 0u);
    }
  }
}

LSE_TEST(quant_matrix_defaults_require_lds_and_threads) {
  for (bool loom : {false, true}) {
    Fixture q4(512, 17408, 5120, 4, loom);
    q4.device.lds_bytes_per_workgroup = dispatch::kQ4MatrixLdsBytes;
    LSE_EXPECT(q4.plan().matrix != nullptr);
    --q4.device.lds_bytes_per_workgroup;
    scalar(q4.plan());
    q4.device.lds_bytes_per_workgroup = 65536;
    q4.device.max_threads_per_workgroup = 255;
    scalar(q4.plan());
    Fixture q6(512, 17408, 5120, 6, loom);
    q6.device.max_threads_per_workgroup = 128;
    q6.device.lds_bytes_per_workgroup = 16384;
    LSE_EXPECT(q6.plan().matrix != nullptr);
    --q6.device.lds_bytes_per_workgroup;
    scalar(q6.plan());
    q6.device.lds_bytes_per_workgroup = 65536;
    q6.device.max_threads_per_workgroup = 127;
    scalar(q6.plan());
    Fixture batch(32, 17408, 5120, 6, loom);
    batch.device.lds_bytes_per_workgroup = 8191;
    LSE_EXPECT_EQ(batch.plan().prefill_rows, 1u);
    batch.device.lds_bytes_per_workgroup = 8192;
    LSE_EXPECT_EQ(batch.plan().prefill_rows, 2u);
    batch.device.lds_bytes_per_workgroup = 16383;
    LSE_EXPECT_EQ(batch.plan().prefill_rows, 2u);
    batch.device.lds_bytes_per_workgroup = 16384;
    LSE_EXPECT_EQ(batch.plan().prefill_rows, 8u);
  }
}

LSE_TEST(quant_dispatch_declines_structurally_unsupported_requests) {
  for (bool loom : {false, true}) {
    for (int fault = 0; fault < 12; ++fault) {
      Fixture f(512, 17408, 5120, 6, loom);
      if (fault == 0) f.shapes.device = nullptr;
      if (fault == 1) f.shapes.intrinsics = nullptr;
      if (fault == 2) f.intrinsics = {};
      if (fault == 3) f.dtypes[0] = DType::kBF16;
      if (fault == 4) f.dtypes[1] = DType::kI32;
      if (fault == 5) f.dtypes[3] = DType::kF32;
      if (fault == 6) f.shapes.output_dtype = DType::kBF16;
      if (fault == 7) f.inputs[1] = Shape{17408, 1};
      if (fault == 8) f.inputs[2] = Shape{17408, 1};
      if (fault == 9) f.inputs[1] = Shape{1, 17408, 960};
      if (fault == 10) f.shapes.iattrs[0] = 0;
      if (fault == 11) f.shapes.output = Shape{};
      if (fault == 2) scalar(f.plan());
      else baseline(f.plan());
    }
    Fixture indexed(512, 17408, 5120, 6, loom);
    baseline(indexed.plan(true));
    Fixture host(512, 17408, 5120, 6, loom);
    host.device.arch = "host";
    host.device.extension = nullptr;
    baseline(host.plan());
    Fixture wave64(512, 17408, 5120, 6, loom);
    wave64.device.wavefront_size = 64;
    baseline(wave64.plan());
    Fixture staged(512, 17408, 5120, 6, loom);
    staged.shapes.staged = {"caller", 5120};
    baseline(staged.plan());
    staged.shapes.staged = {};
    staged.shapes.staged_quant.codes = "caller";
    baseline(staged.plan());
  }
}

LSE_TEST(quant_dispatch_checks_index_capacity_and_scale_formats) {
  for (bool loom : {false, true}) {
    Fixture activation_overflow(131072, 128, 65536, 4, loom);
    baseline(activation_overflow.plan());
    Fixture packed_overflow(1, 1 << 30, 64, 4, loom);
    baseline(packed_overflow.plan());
    for (DType scale : {DType::kF32, DType::kF16, DType::kBF16}) {
      Fixture q4(512, 17408, 5120, 4, loom);
      q4.dtypes[2] = q4.dtypes[3] = scale;
      LSE_EXPECT(q4.plan().matrix != nullptr);
      Fixture q6(512, 17408, 5120, 6, loom);
      q6.dtypes[2] = q6.dtypes[3] = scale;
      LSE_EXPECT_EQ(q6.plan().matrix != nullptr, scale == DType::kBF16);
    }
  }
}

LSE_TEST(q4_prefill_range_uses_existing_matrix_tile_continuously) {
  for (bool loom : {false, true}) {
    for (int m : {16, 17, 32, 64, 128, 256, 257, 511, 512, 513, 1024, 4096, 4097}) {
      for (auto [n, k] : {std::pair{17408, 5120}, std::pair{5120, 17408}}) {
        Fixture f(m, n, k, 4, loom);
        const auto p = f.plan();
        LSE_EXPECT(p.int8_activations && p.matrix != nullptr);
        LSE_EXPECT(p.implementation == (m <= 4096 ? dispatch::QuantMatrix::kInt8Lds
                                                : dispatch::QuantMatrix::kInt8));
        LSE_EXPECT(p.matrix && p.matrix->acc == math::MatrixElem::kI32);
      }
    }
    for (int m : {9, 15}) {
      Fixture f(m, 17408, 5120, 4, loom);
      baseline(f.plan());
    }
    for (int fault = 0; fault < 9; ++fault) {
      Fixture f(64, 17408, 5120, 4, loom);
      if (fault == 0) f.device.arch = "gfx1200";
      if (fault == 1) f.device.wavefront_size = 64;
      if (fault == 2) f.shapes.device = nullptr;
      if (fault == 3) f.dtypes[0] = DType::kBF16;
      if (fault == 4) f.dtypes[3] = DType::kF32;
      if (fault == 5) f.intrinsics = {};
      if (fault == 6) f.device.lds_bytes_per_workgroup = dispatch::kQ4MatrixLdsBytes - 1;
      if (fault == 7) f.device.max_threads_per_workgroup = 255;
      LSE_EXPECT(f.plan(fault == 8).matrix == nullptr);
    }
    Fixture q6(64, 17408, 5120, 6, loom);
    scalar(q6.plan());
  }
}

LSE_TEST(q4_generic_prefill_projections_use_the_same_continuous_range) {
  for (bool loom : {false, true}) {
    for (int m : {16, 32, 64, 128, 256, 511}) {
      for (auto [n, k] : {std::pair{10240, 5120}, std::pair{6144, 5120},
                          std::pair{12288, 5120}, std::pair{1024, 5120},
                          std::pair{5120, 6144}}) {
        Fixture f(m, n, k, 4, loom);
        const auto plan = f.plan();
        LSE_EXPECT(plan.int8_activations);
        LSE_EXPECT(plan.matrix != nullptr);
        LSE_EXPECT(plan.implementation == dispatch::QuantMatrix::kInt8);
        if (plan.matrix) LSE_EXPECT(plan.matrix->acc == math::MatrixElem::kI32);
      }
    }
  }
}


LSE_TEST(quant_panel_adjacent_loads_use_measured_shapes_and_device_admission) {
  for (bool loom : {false, true}) {
    for (int n : {17408, 10240, 6144}) {
      Fixture f(4, n, 5120, 4, loom);
      LSE_EXPECT(f.plan().shared_activation_panel);
      LSE_EXPECT_EQ(dispatch::q4_shared_panel_load_chunks(f.shapes), 2u);
      for (int fault = 0; fault < 9; ++fault) {
        Fixture declined(4, n, 5120, 4, loom);
        if (fault == 0) declined.device.arch = "gfx1200";
        if (fault == 1) declined.device.wavefront_size = 64;
        if (fault == 2) declined.amd.has_dot4_iu8 = false;
        if (fault == 3) declined.device.max_threads_per_workgroup = 255;
        if (fault == 4) declined.shapes.staged = {"caller", 5120};
        if (fault == 5) declined.shapes.staged_quant.codes = "caller";
        if (fault == 6) declined.dtypes[2] = declined.dtypes[3] = DType::kF32;
        if (fault == 7) declined.intrinsics = {};
        if (fault == 8) declined.shapes.device = nullptr;
        LSE_EXPECT(!declined.plan().shared_activation_panel);
        LSE_EXPECT_EQ(dispatch::q4_shared_panel_load_chunks(declined.shapes), 1u);
      }
    }
    for (auto [n, k] : {std::pair{17408, 5120}, {5120, 17408},
                        {10240, 5120}, {6144, 5120}, {5120, 6144},
                        {12288, 5120}, {248320, 5120}}) {
      Fixture measured(8, n, k, 4, loom);
      LSE_EXPECT(measured.plan().shared_activation_panel);
      LSE_EXPECT_EQ(dispatch::q4_shared_panel_load_chunks(measured.shapes),
                    k == 5120 && (n == 17408 || n == 10240 || n == 12288 || n == 248320)
                        ? 4u : 2u);
    }
    for (auto [m, n, k] : {std::array{4, 5120, 17408}, {4, 12288, 5120},
                           {4, 5120, 6144}, {4, 248320, 5120},
                           {6, 17408, 5120},
                           {3, 17408, 5120}, {4, 17, 5120}}) {
      Fixture unchanged(m, n, k, 4, loom);
      LSE_EXPECT_EQ(dispatch::q4_shared_panel_load_chunks(unchanged.shapes), 1u);
    }
  }
}

// gfx1151's verify passes take int8 activations, the shared panel (4 and 8
// rows) and the row ladder (3 and 7) from its own header, as gfx1201's do from its; a part with no
// such rows (gfx1100) keeps the one-row-at-a-time contraction.
LSE_TEST(verify_pass_panel_is_qualified_per_part) {
  for (const int m : {3, 4, 7, 8}) {
    Fixture f(m, 17408, 5120, 4, true);
    f.inputs[0] = Shape{1, m, 5120};
    f.shapes.output = Shape{1, m, 17408};
    f.device = {};
    f.amd = {};
    f.device.arch = "gfx1151";
    f.device.compute_units = 40;
    f.device.wavefront_size = 32;
    f.device.max_threads_per_workgroup = 1024;
    f.device.lds_bytes_per_workgroup = 65536;
    backend::apply_arch_defaults(f.device, f.amd);
    f.device.extension_id = backend::AmdDeviceInfo::kExtensionId;
    f.device.extension = &f.amd;
    const auto halo = f.plan();
    LSE_EXPECT(halo.int8_activations);
    LSE_EXPECT_EQ(halo.shared_activation_panel, m == 4 || m == 8);
    LSE_EXPECT_EQ(halo.row_ladder_ceiling, m == 3 ? 4u : m == 7 ? 8u : 0u);
    f.device.arch = "gfx1100";
    const auto other = f.plan();
    LSE_EXPECT(!other.int8_activations && !other.shared_activation_panel);
  }
}

// Each part's dispatch rows come from its own header and only from there:
// a device gets its part's rows followed by the generic ones, an unknown part
// the generic rows alone, and no part reads another's.
LSE_TEST(arch_tuning_hands_each_part_only_its_own_rows) {
  namespace arch = dispatch::arch;
  const auto& r9700 = arch::tuning("gfx1201");
  const auto& halo = arch::tuning("gfx1151");
  const auto& other = arch::tuning("gfx1100");
  LSE_EXPECT(r9700.arch == "gfx1201");
  LSE_EXPECT(halo.arch == "gfx1151");
  LSE_EXPECT(other.arch.empty());
  const auto own = [](const arch::Tuning& t, const auto& rows) {
    for (const auto& row : rows)
      if (!row.arch.empty() && row.arch != t.arch) return false;
    return true;
  };
  for (const arch::Tuning* t : {&r9700, &halo, &other}) {
    LSE_EXPECT(own(*t, t->quant_matrix_shapes) && own(*t, t->quant_int8_rows) &&
               own(*t, t->quant_matrix_ranges) && own(*t, t->quant_scalar_shapes) &&
               own(*t, t->quant_row_ladders) && own(*t, t->quant_panel_devices) &&
               own(*t, t->q4_swiglu_shapes) && own(*t, t->q4_matrix_panel_shapes) &&
               own(*t, t->q8_matrix_rules) && own(*t, t->flash_wmma) &&
               own(*t, t->flash_cache) && own(*t, t->decode) &&
               own(*t, t->split_short) && own(*t, t->wave_l2));
    // The generic admissions follow every part's own rows.
    LSE_EXPECT(!t->quant_int8_rows.empty() && t->quant_int8_rows.back().arch.empty());
    LSE_EXPECT(!t->quant_matrix_ranges.empty() &&
               t->quant_matrix_ranges.back().arch.empty());
  }
  // gfx1201's measured rows, unchanged by the move into its header.
  LSE_EXPECT_EQ(r9700.quant_matrix_shapes.size(), 4u);
  LSE_EXPECT_EQ(r9700.q4_matrix_panel_shapes.size(), 6u);
  LSE_EXPECT_EQ(r9700.q8_matrix_rules.size(), 2u);
  LSE_EXPECT_EQ(r9700.decode.size(), 1u);
  LSE_EXPECT(r9700.flash_prefill && !halo.flash_prefill && !other.flash_prefill);
  LSE_EXPECT(other.quant_matrix_shapes.empty() && other.flash_wmma.empty());
}

LSE_TEST_MAIN()
