#include "lse/graph/graph.hpp"

#include "harness.hpp"
#include "lse/backends/hrx/probe_emit.hpp"
#include "lse/backends/hrx/loomc/loomc_compiler.hpp"
#include "lse/backends/hrx/arch_database.hpp"
#include "lse/backends/hrx/probe_matrix.hpp"

using namespace lse;
using namespace lse::backend;
using namespace lse::backend::hrx_kernels;

DeviceInfo probe_fixture() {
  DeviceInfo d;
  d.arch = "gfx1201";
  d.compute_units = 64;
  d.wavefront_size = 32;
  d.max_threads_per_workgroup = 1024;
  d.lds_bytes_per_workgroup = 65536;
  return d;
}
LSE_TEST(loom_stream_and_touch_compile_through_native_compiler) {
  const auto d = probe_fixture();
  const LoomcCompiler compiler;
  LSE_EXPECT(compiler.available());
  if (!compiler.available()) return;
  for (std::uint32_t load_bytes : {4u, 8u, 16u}) {
    const std::uint32_t threads = d.compute_units * 8u * 256u;
    const std::uint32_t span = threads * (load_bytes / 4u);
    const std::uint32_t elems = ((512u << 20) / 4u / span) * span;
    auto emitted = emit_loom_stream_probe(d, elems, threads, load_bytes);
    LSE_EXPECT(emitted.ok());
    if (!emitted.ok()) { std::fprintf(stderr, "%s\n", emitted.status().to_string().c_str()); return; }
    LSE_EXPECT(emitted->dialect == graph::Dialect::kLoom);
    LSE_EXPECT_EQ(emitted->binding_order.size(), 2u);
    LSE_EXPECT_EQ(emitted->constants.total_bytes, 4u);
    LSE_EXPECT_EQ(emitted->dims.workgroup_size[0], 256u);
    LSE_EXPECT_EQ(emitted->dims.workgroup_count[0], threads / 256u);
    auto code = compiler.compile(emitted->source, "gfx1201");
    LSE_EXPECT(code.ok());
    if (!code.ok()) { std::fprintf(stderr, "%s\n", code.status().to_string().c_str()); return; }
    LSE_EXPECT(!code->code.empty());
  }
  auto touch = emit_loom_touch_probe(d);
  LSE_EXPECT(touch.ok());
  if (!touch.ok()) return;
  LSE_EXPECT_EQ(touch->binding_order.size(), 1u);
  LSE_EXPECT_EQ(touch->constants.total_bytes, 4u);
  LSE_EXPECT_EQ(touch->dims.workgroup_size[0], 32u);
  auto code = compiler.compile(touch->source, "gfx1201");
  LSE_EXPECT(code.ok());
  if (!code.ok()) std::fprintf(stderr, "%s\n", code.status().to_string().c_str());
}
LSE_TEST(loom_probes_decline_unknown_geometry_and_partial_strides) {
  auto d = probe_fixture();
  LSE_EXPECT(!emit_loom_stream_probe(d, 1025, 256, 16).ok());
  LSE_EXPECT(!emit_loom_stream_probe(d, 1024, 256, 3).ok());
  d.wavefront_size = 0;
  LSE_EXPECT(!emit_loom_touch_probe(d).ok());
  LSE_EXPECT(!emit_loom_stream_probe(d, 1024, 256, 16).ok());
}
LSE_TEST(loom_matrix_probe_compiles_shared_hip_tile_and_geometry) {
  auto d=probe_fixture();AmdDeviceInfo amd;apply_arch_defaults(d,amd);d.extension_id=AmdDeviceInfo::kExtensionId;d.extension=&amd;
  const LoomcCompiler compiler;unsigned compiled=0;
  for(auto storage:{DType::kF16,DType::kBF16,DType::kI32})for(const auto& row:math::matrix_core_table()) {
    const auto want=storage==DType::kF16?math::MatrixElem::kF16:storage==DType::kBF16?math::MatrixElem::kBF16:math::MatrixElem::kI8;
    if(row.target!=math::MatrixTarget::kRdna4||row.operand!=want||row.acc!=(storage==DType::kI32?math::MatrixElem::kI32:math::MatrixElem::kF32)||row.m!=16||row.n!=16||row.k_step!=16)continue;
    auto emitted=emit_loom_matrix_rate_probe(d,row,storage);LSE_EXPECT(emitted.ok());
    if(!emitted.ok()){std::fprintf(stderr,"%s\n",emitted.status().to_string().c_str());continue;}
    const auto dims=probe_matrix::rate_dims(d,row);
    LSE_EXPECT_EQ(emitted->binding_order.size(),3u);LSE_EXPECT_EQ(emitted->constants.total_bytes,4u);
    LSE_EXPECT_EQ(emitted->dims.workgroup_size[0],dims.workgroup_size[0]);LSE_EXPECT_EQ(emitted->dims.workgroup_count[0],dims.workgroup_count[0]);
    LSE_EXPECT_EQ(emitted->binding_order[0]->element_count(),256u*1024u/static_cast<unsigned>(row.pack));
    LSE_EXPECT_EQ(emitted->binding_order.back()->element_count(),256u*256u);
    auto object=compiler.compile(emitted->source,d.arch);LSE_EXPECT(object.ok());if(!object.ok())std::fprintf(stderr,"%s\n",object.status().to_string().c_str());else ++compiled;
    auto wrong=d;wrong.wavefront_size=0;LSE_EXPECT(!emit_loom_matrix_rate_probe(wrong,row,storage).ok());
    LSE_EXPECT(!emit_loom_matrix_rate_probe(d,row,DType::kU8).ok());
  }
  LSE_EXPECT_EQ(compiled,3u);
}
LSE_TEST_MAIN()
