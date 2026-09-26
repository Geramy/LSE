// Measured kernel-occupancy census over the REAL decode Q6 GEMV.
//
// Host-only: loomc compiles for gfx1201 without a device, the census is
// occupancy arithmetic over the code object's MEASURED resources (VGPR, LDS,
// required workgroup, spill), and the per-shape bandwidth to correlate against
// is the frozen cand-decode-gemv-micro capture (gpu-run/run.log). No GPU is
// opened, so this runs while another workload holds the device.
//
// The four shapes are the ones that dominate the decode capture (RUN.md):
//   upgate   N=17408  K=5120   grid 544  -> 48.0 GB/s
//   down     N=5120   K=17408  grid 160  -> 47.8 GB/s
//   gdn      N=10240  K=5120   grid 320  -> 27.9 GB/s
//   lm_head  N=248320 K=5120   grid 7760 -> 341.8 GB/s
// Each is driven through the production `quant_linear` node (M=1, 6-bit,
// group64) so the emitter produces the actual `emit_q6_decode_quad` source,
// compiled for gfx1201 and read back for its resources.

#include "harness.hpp"

#include <cstdio>
#include <string>
#include <vector>

#include "lse/opt/kernel_census.hpp"
#include "lse/opt/occupancy.hpp"
#include "lse/graph/ops.hpp"
#include "lse/graph/graph.hpp"
#include "lse/backends/hrx/loomc/loom_emitter.hpp"
#include "lse/backends/hrx/loomc/loomc_compiler.hpp"
#include "lse/backends/hrx/arch_database.hpp"
#include "lse/core/shape.hpp"

using namespace lse;
using namespace lse::graph;
using namespace lse::opt;
namespace backend = lse::backend;

namespace {

// The device the R9700 reports. gfx1201, 64 CUs, 32-lane waves, 64 KB LDS per
// workgroup — the same facts the production backend feeds the emitter, so the
// kernel emitted here is byte-for-byte the one that runs (the quad path gates
// on arch=="gfx1201" and wave==32).
backend::DeviceInfo gfx1201() {
  backend::DeviceInfo info;
  info.name = "R9700";
  info.arch = "gfx1201";
  info.compute_units = 64;
  info.wavefront_size = 32;
  info.max_threads_per_workgroup = 1024;
  info.lds_bytes_per_workgroup = 65536;
  backend::AmdDeviceInfo amd;
  backend::apply_arch_defaults(info, amd);
  info.extension_id = backend::AmdDeviceInfo::kExtensionId;
  info.extension = &amd;
  return info;
}

struct ShapeSpec {
  const char* name;
  int n;
  int k;
  // Workgroup count the frozen capture ran this shape at (from run.log).
  unsigned grid;
  // Effective GB/s the frozen capture measured (from run.log). The correlation
  // target; pasted, not re-measured, so no GPU is needed.
  double gbps;
};

// The four decode GEMV shapes that dominate the capture (RUN.md / run.log).
constexpr ShapeSpec kShapes[] = {
    {"upgate", 17408, 5120, 544, 48.0},
    {"down", 5120, 17408, 160, 47.8},
    {"gdn_in_proj", 10240, 5120, 320, 27.9},
    {"lm_head", 248320, 5120, 7760, 341.8},
};

// A shape-only buffer leaf: the emitter reads only the node's shape and dtype,
// so no device buffer is needed to emit the source.
NodePtr leaf(Shape shape, DType type) {
  auto n = std::make_shared<Node>();
  n->set_kind(OpKind::kBuffer);
  n->shape = std::move(shape);
  n->dtype = type;
  return n;
}

// Emit the real decode GEMV for one shape: a production `quant_linear` node
// (M=1, 6-bit, group64) partitioned and emitted by the Loom emitter.
Result<EmittedKernel> emit_gdgv(const backend::DeviceInfo& info, int n, int k) {
  // Packed 6-bit plane [N, K*6/32] u32, scales/biases [N, K/64] bf16, x row
  // [1, K] f32. The quad path requires input dtype order f32, u32, bf16, bf16.
  const int words = k * 6 / 32;
  const int groups = k / 64;
  auto x = leaf(Shape{1, k}, DType::kF32);
  auto packed = leaf(Shape{static_cast<int64_t>(n), static_cast<int64_t>(words)},
                     DType::kU32);
  auto scales = leaf(Shape{static_cast<int64_t>(n), static_cast<int64_t>(groups)},
                     DType::kBF16);
  auto biases = leaf(Shape{static_cast<int64_t>(n), static_cast<int64_t>(groups)},
                     DType::kBF16);
  Array y = quant_linear(Array(x), Array(packed), Array(scales),
                         Array(biases), 6, 64);
  const NodePtr roots[] = {y.node()};
  auto groups_out = Partitioner::partition(roots);
  if (groups_out.empty()) {
    return LSE_ERROR(kInternal, "no fusion groups for the GEMV node");
  }
  // The group carrying the quant_linear node is the one to emit.
  for (const FusionGroup& g : groups_out) {
    if (std::find(g.nodes.begin(), g.nodes.end(), y.node()) == g.nodes.end())
      continue;
    backend::LoomEmitter emitter;
    return emitter.emit(g, info);
  }
  return LSE_ERROR(kInternal, "the GEMV node is not in any fusion group");
}

struct CensusRow {
  const ShapeSpec& spec;
  opt::KernelCensus census;
  std::string entry;
  unsigned workgroup_threads = 0;
};

}  // namespace

LSE_TEST(the_decode_gemv_census_reads_measured_resources) {
  const backend::LoomcCompiler compiler;
  if (!compiler.available()) {
    std::printf("       (skipped: loomc not in this build)\n");
    return;
  }
  const backend::DeviceInfo info = gfx1201();
  const DeviceCapacity cap = DeviceCapacity::of(info);

  std::vector<CensusRow> rows;
  for (const ShapeSpec& spec : kShapes) {
    auto emitted = emit_gdgv(info, spec.n, spec.k);
    if (!emitted.ok()) {
      std::printf("       emit %s FAILED: %s\n", spec.name,
                  emitted.status().to_string().c_str());
      continue;
    }
    auto obj = compiler.compile(emitted->source, "gfx1201");
    if (!obj.ok()) {
      std::printf("       compile %s FAILED: %s\n", spec.name,
                  obj.status().to_string().c_str());
      continue;
    }
    LSE_EXPECT(!obj->resources.empty());
    if (obj->resources.empty()) continue;
    const backend::KernelResources& r = obj->resources.front();
    // The object states its own launch geometry; use it as the census thread
    // count so the residency the model seats is the one the kernel was built
    // for. Fall back to the capture's grid only if it does not.
    const std::uint32_t threads =
        r.required_workgroup_size.known() ? r.required_workgroup_size.value[0]
                                          : 256u;
    LaunchGeometry launch;
    launch.threads = threads;
    launch.workgroups = spec.grid;
    rows.push_back({spec, census(cap, r, launch), r.entry, threads});
  }

  LSE_EXPECT_EQ(rows.size(), 4u);
  if (rows.size() != 4) return;

  std::printf("       == decode Q6 GEMV occupancy census (gfx1201, M=1) ==\n");
  std::printf(
      "       %-12s %5s %5s %7s %5s %4s %6s %9s %11s %15s %6s\n", "shape",
      "vgpr", "sgpr", "lds", "priv", "spill", "wg_th", "waves/simd", "wgs/pool",
      "binding", "GB/s");
  for (const CensusRow& row : rows) {
    const auto& r = row.census.resources;
    auto u = [](const backend::DeviceFact<std::uint32_t>& f) -> std::uint32_t {
      return f.known() ? f.value : 0;
    };
    std::printf(
        "       %-12s %5u %5u %7u %5u %4d %6u %9u %9u %-11s %6.1f\n", row.spec.name,
        u(r.vector_registers), u(r.scalar_registers),
        u(r.workgroup_segment_bytes), u(r.private_segment_bytes),
        r.spilled() == backend::SpillState::kSpilled ? 1 : 0, row.workgroup_threads,
        row.census.occupancy.waves_per_simd,
        row.census.occupancy.workgroups_per_pool,
        to_string(row.census.occupancy.binding).data(), row.spec.gbps);
  }
  std::printf("       capacity: %s", cap.describe().c_str());
}

// The census itself, on a synthetic resource set: the register arm must bind
// when the VGPR count is high, the answer must degrade (not guess) when a fact
// is unknown, and a kernel that states a required workgroup larger than the
// caller ran it with must be flagged as geometry-mismatched.
LSE_TEST(the_census_pairs_measured_resources_with_the_occupancy_model) {
  backend::DeviceInfo info = gfx1201();
  const DeviceCapacity cap = DeviceCapacity::of(info);

  backend::KernelResources low;
  low.entry = "low";
  low.vector_registers = backend::DeviceFact<std::uint32_t>::queried(32);
  low.workgroup_segment_bytes =
      backend::DeviceFact<std::uint32_t>::queried(0);
  low.required_workgroup_size =
      backend::DeviceFact<std::array<std::uint32_t, 3>>::queried({256, 1, 1});
  LaunchGeometry l;
  l.threads = 256;
  l.workgroups = 544;
  KernelCensus c = census(cap, low, l);
  LSE_EXPECT(c.occupancy.seated());
  LSE_EXPECT(!c.geometry_mismatched());

  // A register-hungry kernel: the register arm binds only when the capacity
  // knows the VGPR file. The real gfx1201 capacity does not (the tree declines
  // to declare the compiler-model register count), so build a capacity that
  // does, and confirm the arm then takes the binding.
  DeviceCapacity regcap = cap;
  regcap.vector_registers_per_simd =
      backend::DeviceFact<std::uint32_t>::queried(512);
  regcap.vector_register_alloc_granule =
      backend::DeviceFact<std::uint32_t>::queried(8);
  backend::KernelResources heavy;
  heavy.entry = "heavy";
  heavy.vector_registers = backend::DeviceFact<std::uint32_t>::queried(200);
  heavy.workgroup_segment_bytes =
      backend::DeviceFact<std::uint32_t>::queried(0);
  heavy.required_workgroup_size =
      backend::DeviceFact<std::array<std::uint32_t, 3>>::queried({256, 1, 1});
  KernelCensus ch = census(regcap, heavy, l);
  LSE_EXPECT(ch.occupancy.seated());
  LSE_EXPECT(ch.occupancy.binding == OccupancyLimit::kVectorRegisters);
  // And the same kernel on the real (register-unknown) capacity must NOT claim
  // a register binding it cannot count: the arm drops out and slots bind.
  KernelCensus ch_real = census(cap, heavy, l);
  LSE_EXPECT(ch_real.occupancy.binding == OccupancyLimit::kWaveSlots);

  // The caller ran it with a smaller workgroup than it requested: mismatched.
  LaunchGeometry small;
  small.threads = 128;
  small.workgroups = 1088;
  KernelCensus cm = census(cap, low, small);
  LSE_EXPECT(cm.geometry_mismatched());
}

LSE_TEST_MAIN()
