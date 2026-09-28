#include "harness.hpp"
#include "lse/backends/hrx/arch_database.hpp"
#include "lse/backends/hrx/hipc/hip_sources.hpp"
#include "lse/backends/hrx/loomc/loom_sources.hpp"
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

LSE_TEST(quant_scalar_defaults_are_shape_specific) {
  for (bool loom : {false, true}) {
    for (int m : {1, 2, 3, 7, 8, 9, 32, 512}) {
      for (int bits : {4, 6, 8}) {
        Fixture f(m, 17, 64, bits, loom);
        const auto p = f.plan();
        LSE_EXPECT_EQ(p.int8_activations, bits == 4 && (m <= 8 || m == 512));
        LSE_EXPECT_EQ(p.matrix != nullptr, bits == 4 && m == 512);
        if (bits != 4 || m != 512) scalar(p);
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
LSE_TEST_MAIN()
