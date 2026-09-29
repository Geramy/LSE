#include "harness.hpp"
#include "lse/backends/hrx/arch_database.hpp"
#include "lse/backends/hrx/loomc/loom_emitter.hpp"
#include "lse/graph/graph.hpp"
#include "lse/graph/ops.hpp"

#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <optional>
#include <unistd.h>

#include "lse/graph/jit.hpp"
#include "lse/opt/measurements.hpp"

namespace {
using namespace lse;
using namespace lse::graph;

Array leaf(Shape shape, DType type = DType::kF32) {
  auto node = std::make_shared<Node>();
  node->shape = shape;
  node->dtype = type;
  node->materialized = true;
  return Array(node);
}

FusionGroup subtraction(bool reverse = false, bool alias = false) {
  auto a = leaf(Shape{128}), b = alias ? a : leaf(Shape{128});
  auto out = reverse ? b - a : a - b;
  FusionGroup group;
  group.nodes = {out.node()};
  group.inputs = alias ? std::vector<NodePtr>{a.node()}
                       : std::vector<NodePtr>{a.node(), b.node()};
  group.outputs = {out.node()};
  group.anchor = out.node()->kind;
  group.anchor_class = out.node()->fclass;
  return group;
}

backend::DeviceInfo device() {
  backend::DeviceInfo info;
  info.arch = "gfx1201";
  info.wavefront_size = 32;
  info.max_threads_per_workgroup = 1024;
  info.lds_bytes_per_workgroup = 65536;
  return info;
}
}  // namespace

LSE_TEST(loom_cache_rebinds_new_graphs_without_retaining_old_nodes) {
  backend::LoomEmitter emitter;
  auto info = device();
  std::weak_ptr<Node> old_input;
  std::string source;
  {
    auto group = subtraction();
    old_input = group.inputs[0];
    auto emitted = emitter.emit(group, info);
    LSE_EXPECT(emitted.ok());
    if (!emitted.ok()) return;
    source = emitted->source;
  }
  LSE_EXPECT(old_input.expired());
  auto rebuilt = subtraction();
  auto cached = emitter.emit(rebuilt, info);
  auto fresh = backend::LoomEmitter{}.emit(rebuilt, info);
  LSE_EXPECT(cached.ok() && fresh.ok());
  if (!cached.ok() || !fresh.ok()) return;
  LSE_EXPECT(cached->source == source && cached->source == fresh->source);
  LSE_EXPECT(cached->binding_order == fresh->binding_order);
  LSE_EXPECT(cached->binding_order.front() == rebuilt.inputs.front());
  LSE_EXPECT_EQ(emitter.cache_stats().hits, 1u);
  LSE_EXPECT_EQ(emitter.cache_stats().entries, 1u);
}

LSE_TEST(loom_cache_identity_distinguishes_edges_aliases_constants_and_devices) {
  backend::LoomEmitter emitter;
  auto info = device();
  auto normal = subtraction(), reversed = subtraction(true);
  // The old display signature omits edges and collides for a-b and b-a.
  LSE_EXPECT_EQ(normal.signature(), reversed.signature());
  LSE_EXPECT(emitter.cache_key(normal, info) != emitter.cache_key(reversed, info));
  auto aliased = subtraction(false, true);
  LSE_EXPECT(emitter.cache_key(normal, info) != emitter.cache_key(aliased, info));
  auto wider = info;
  wider.wavefront_size = 64;
  LSE_EXPECT(emitter.cache_key(normal, info) != emitter.cache_key(normal, wider));
  wider = info;
  wider.lds_bytes_per_workgroup /= 2;
  LSE_EXPECT(emitter.cache_key(normal, info) != emitter.cache_key(normal, wider));
  for (const auto* group : {&normal, &reversed, &aliased}) {
    auto result = emitter.emit(*group, info);
    LSE_EXPECT(result.ok());
  }
  LSE_EXPECT_EQ(emitter.cache_stats().misses, 3u);
  LSE_EXPECT_EQ(emitter.cache_stats().hits, 0u);
  auto constant = subtraction();
  constant.inputs[0]->set_kind(OpKind::kConstant);
  constant.inputs[0]->attrs[0] = 1.0f;
  const auto one = emitter.cache_key(constant, info);
  constant.inputs[0]->attrs[0] = 2.0f;
  LSE_EXPECT(one != emitter.cache_key(constant, info));
}

LSE_TEST(loom_cache_reuses_q6_kernel_source_and_launch_metadata) {
  auto info = device();
  backend::AmdDeviceInfo amd;
  backend::apply_arch_defaults(info, amd);
  info.extension_id = backend::AmdDeviceInfo::kExtensionId;
  info.extension = &amd;
  auto out = quant_linear(leaf(Shape{1, 4096}),
      leaf(Shape{17, 768}, DType::kU32),
      leaf(Shape{17, 64}, DType::kBF16),
      leaf(Shape{17, 64}, DType::kBF16), 6, 64);
  const NodePtr roots[] = {out.node()};
  const auto groups = Partitioner::partition(roots);
  LSE_EXPECT_EQ(groups.size(), 1u);
  if (groups.size() != 1) return;
  backend::LoomEmitter emitter;
  auto cold_start = std::chrono::steady_clock::now();
  auto cold = emitter.emit(groups[0], info);
  auto cold_end = std::chrono::steady_clock::now();
  LSE_EXPECT(cold.ok());
  if (!cold.ok()) return;
  constexpr unsigned repeats = 100;
  for (unsigned i = 0; i < repeats; ++i) {
    auto hot = emitter.emit(groups[0], info);
    LSE_EXPECT(hot.ok());
    if (!hot.ok()) return;
    LSE_EXPECT(hot->source == cold->source);
    LSE_EXPECT(hot->binding_order == cold->binding_order);
    LSE_EXPECT_EQ(hot->lds_bytes, cold->lds_bytes);
    LSE_EXPECT_EQ(hot->dims.workgroup_count[0], cold->dims.workgroup_count[0]);
    LSE_EXPECT_EQ(hot->constants.total_bytes, cold->constants.total_bytes);
  }
  const auto hot_end = std::chrono::steady_clock::now();
  LSE_EXPECT_EQ(emitter.cache_stats().hits, repeats);
  LSE_EXPECT_EQ(emitter.cache_stats().misses, 1u);
  std::printf("    Q6 source emission: cold %.1f us, warm mean %.1f us (%u calls; host only)\n",
      std::chrono::duration<double, std::micro>(cold_end - cold_start).count(),
      std::chrono::duration<double, std::micro>(hot_end - cold_end).count() / repeats,
      repeats);
}

namespace {
struct ArtifactCompiler final : IKernelCompiler {
  mutable int compiles = 0;
  std::string id = "exact-source-test.v1";
  static std::string entry(std::string_view text) {
    const auto at = text.find("export(\"");
    if (at == std::string_view::npos) return {};
    const auto begin = at + 8;
    return std::string(text.substr(begin, text.find('"', begin) - begin));
  }
  Result<CompiledKernel> compile(std::string_view text,
                                 std::string_view) const override {
    ++compiles;
    CompiledKernel out;
    out.code.resize(text.size());
    std::memcpy(out.code.data(), text.data(), text.size());
    backend::KernelResources r;
    r.entry = entry(text);
    r.vector_registers = backend::DeviceFact<std::uint32_t>::queried(17);
    out.resources.push_back(r);
    out.census = census(out.code);
    return out;
  }
  std::vector<backend::KernelCensus> census(
      std::span<const std::byte> bytes) const override {
    backend::KernelCensus c;
    c.entry = entry({reinterpret_cast<const char*>(bytes.data()), bytes.size()});
    c.instructions = backend::DeviceFact<std::uint32_t>::queried(19);
    return {c};
  }
  bool available() const override { return true; }
  std::string identity() const override { return id; }
};
struct ArtifactBackend final : backend::IBackend {
  backend::DeviceInfo info = device();
  int loads = 0;
  std::uint64_t owner = 100;
  Status init(int) override { return OkStatus(); }
  void shutdown() noexcept override {}
  const backend::DeviceInfo& device_info() const noexcept override { return info; }
  Result<backend::DeviceBuffer> allocate(std::size_t, backend::MemoryClass,
                                         backend::Stream) override {
    return LSE_ERROR(kUnimplemented, "stub");
  }
  void deallocate(backend::DeviceBuffer&) noexcept override {}
  Status copy_h2d(const void*, backend::DeviceBuffer&, std::size_t,
                  std::size_t) override { return LSE_ERROR(kUnimplemented, "stub"); }
  Status copy_d2h(const backend::DeviceBuffer&, void*, std::size_t,
                  std::size_t) override { return LSE_ERROR(kUnimplemented, "stub"); }
  Result<backend::KernelHandle> load_executable(
      std::string_view name, std::span<const std::byte> bytes) override {
    const std::string_view text(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    if (ArtifactCompiler::entry(text) != name)
      return LSE_ERROR(kNotFound, "executable does not export requested entry");
    backend::KernelHandle handle;
    handle.executable = owner + static_cast<std::uint64_t>(++loads);
    handle.name = name;
    return handle;
  }
  Status launch(const backend::KernelHandle&, const backend::LaunchDims&,
                const backend::DispatchArgs&, const backend::DispatchTarget&) override {
    return LSE_ERROR(kUnimplemented, "stub");
  }
  Status synchronize() override { return OkStatus(); }
  std::string_view name() const noexcept override { return "artifact-test"; }
  std::span<const KernelToolchain> toolchains() const noexcept override { return {}; }
};
struct ArtifactDirectory {
  std::filesystem::path path;
  std::optional<std::string> dump;
  ArtifactDirectory() {
    static unsigned n = 0;
    path = std::filesystem::temp_directory_path() /
        ("lse-exact-source-" + std::to_string(::getpid()) + "-" + std::to_string(++n));
    std::filesystem::remove_all(path);
    std::filesystem::create_directories(path);
    if (const auto* before = std::getenv("LSE_HIP_DUMP")) dump = before;
    ::setenv("LSE_HIP_DUMP", (path / "dump").c_str(), 1);
  }
  ~ArtifactDirectory() {
    if (dump) ::setenv("LSE_HIP_DUMP", dump->c_str(), 1);
    else ::unsetenv("LSE_HIP_DUMP");
    std::filesystem::remove_all(path);
  }
};
EmittedKernel artifact_source(std::string_view entry = "probe_a",
                              std::string_view width = "128",
                              std::string_view dtype = "f32") {
  EmittedKernel out;
  out.dialect = Dialect::kLoom;
  out.entry_name = entry;
  out.source = "kernel.def export(\"" + out.entry_name + "\") @" + out.entry_name +
      "() {\n  %cfg = index.constant 256 : index\n} launch(%a: buffer) {\n" +
      "  %view = buffer.view %a[%zero] : buffer -> view<" + std::string(width) +
      "x" + std::string(dtype) + ", #dense>\n  kernel.return\n}\n";
  return out;
}
}  // namespace

LSE_TEST(source_identity_changes_only_the_self_export_and_is_idempotent) {
  auto a = artifact_source("probe_a"), b = artifact_source("probe_b");
  a.dims.workgroup_count[0] = 7;
  a.lds_bytes = 1024;
  LSE_EXPECT(finalize_source_identity(a, "lse_loom_"));
  LSE_EXPECT(finalize_source_identity(b, "lse_loom_"));
  LSE_EXPECT(a.source == b.source && a.entry_name == b.entry_name);
  LSE_EXPECT(a.content_addressed && b.content_addressed);
  LSE_EXPECT_EQ(a.dims.workgroup_count[0], 7u);
  LSE_EXPECT_EQ(a.lds_bytes, 1024u);
  const auto source = a.source, name = a.entry_name;
  LSE_EXPECT(finalize_source_identity(a, "lse_loom_"));
  LSE_EXPECT(a.source == source && a.entry_name == name);
  auto embedded = artifact_source("probe");
  embedded.source += "// probe_suffix\n";
  LSE_EXPECT(finalize_source_identity(embedded));
  LSE_EXPECT(embedded.source.find("probe_suffix") != std::string::npos);
  auto absent = artifact_source();
  absent.entry_name = "absent";
  LSE_EXPECT(!finalize_source_identity(absent));
  LSE_EXPECT(!absent.content_addressed);
}

LSE_TEST(loom_compiled_body_ignores_bound_producer_metadata_but_keeps_edges) {
  auto one = subtraction(), two = subtraction();
  one.inputs[0]->set_kind(OpKind::kConstant);
  two.inputs[0]->set_kind(OpKind::kConstant);
  one.inputs[0]->attrs[0] = 1;
  two.inputs[0]->attrs[0] = 2;
  backend::LoomEmitter emitter;
  const auto info = device();
  LSE_EXPECT(emitter.cache_key(one, info) != emitter.cache_key(two, info));
  auto a = emitter.emit(one, info), b = emitter.emit(two, info);
  LSE_EXPECT(a.ok() && b.ok());
  if (!a.ok() || !b.ok()) return;
  LSE_EXPECT(a->content_addressed && b->content_addressed);
  LSE_EXPECT(a->entry_name == b->entry_name && a->source == b->source);
  LSE_EXPECT(a->binding_order.front() == one.inputs.front());
  LSE_EXPECT(b->binding_order.front() == two.inputs.front());
  auto reversed = emitter.emit(subtraction(true), info);
  auto aliased = emitter.emit(subtraction(false, true), info);
  LSE_EXPECT(reversed.ok() && aliased.ok());
  if (reversed.ok()) LSE_EXPECT(a->source != reversed->source);
  if (aliased.ok()) LSE_EXPECT(a->source != aliased->source);
}

LSE_TEST(compiled_source_aliases_load_once_and_preserve_resource_names) {
  ArtifactDirectory dir;
  ArtifactCompiler compiler;
  ArtifactBackend backend;
  auto a = artifact_source("one"), b = artifact_source("two");
  a.dims.workgroup_count[0] = 7;
  b.dims.workgroup_count[0] = 9;
  a.traffic.stated = b.traffic.stated = true;
  a.traffic.workgroups = 7;
  b.traffic.workgroups = 9;
  a.binding_order = {leaf({128}).node()};
  b.binding_order = {leaf({128}).node()};
  LSE_EXPECT(finalize_source_identity(a));
  LSE_EXPECT(finalize_source_identity(b));
  JitCache cache(backend, compiler, dir.path.string());
  auto first = cache.get_or_compile(0, 11, a);
  auto second = cache.get_or_compile(0, 22, b);
  LSE_EXPECT(first.ok() && second.ok());
  if (!first.ok() || !second.ok()) return;
  LSE_EXPECT_EQ(compiler.compiles, 1);
  LSE_EXPECT_EQ(backend.loads, 1);
  LSE_EXPECT(first->name == a.entry_name && second->name == a.entry_name);
  LSE_EXPECT_EQ(first->executable, second->executable);
  LSE_EXPECT(cache.try_get(0, 11, a.dialect) != nullptr);
  LSE_EXPECT(cache.try_get(0, 22, b.dialect) != nullptr);
  const auto* r = cache.resources(0, 22, b.dialect, b.entry_name);
  const auto* c = cache.census(0, 22, b.dialect, b.entry_name);
  LSE_EXPECT(r && c);
  if (r) LSE_EXPECT_EQ(r->vector_registers.value, 17u);
  if (c) LSE_EXPECT_EQ(c->instructions.value, 19u);
  LSE_EXPECT(r == cache.resources(0, 11, a.dialect, a.entry_name));
  LSE_EXPECT(cache.resources(0, 22, b.dialect, b.structural_entry_name) == r);
  auto& measured = opt::KernelMeasurements::instance();
  LSE_EXPECT(measured.known(a.structural_entry_name));
  LSE_EXPECT(measured.census_known(b.structural_entry_name));
  LSE_EXPECT_EQ(measured.traffic(a.structural_entry_name).workgroups, 7u);
  LSE_EXPECT_EQ(measured.traffic(b.structural_entry_name).workgroups, 9u);
  LSE_EXPECT_EQ(a.dims.workgroup_count[0], 7u);
  LSE_EXPECT_EQ(b.dims.workgroup_count[0], 9u);
  LSE_EXPECT(a.binding_order.front() != b.binding_order.front());
  std::size_t objects = 0;
  for (const auto& entry : std::filesystem::directory_iterator(dir.path))
    objects += entry.path().extension() == ".co";
  LSE_EXPECT_EQ(objects, 1u);
}

LSE_TEST(compiled_source_warm_alias_verifies_the_entire_disk_source) {
  ArtifactDirectory dir;
  ArtifactCompiler compiler;
  ArtifactBackend backend;
  auto source = artifact_source();
  LSE_EXPECT(finalize_source_identity(source));
  {
    JitCache cache(backend, compiler, dir.path.string());
    LSE_EXPECT_OK(cache.get_or_compile(0, 31, source).status());
  }
  {
    opt::KernelMeasurements::instance().clear();
    JitCache cache(backend, compiler, dir.path.string());
    LSE_EXPECT(opt::KernelMeasurements::instance().known(source.structural_entry_name));
    LSE_EXPECT(opt::KernelMeasurements::instance().census_known(source.structural_entry_name));
    LSE_EXPECT_OK(cache.get_or_compile(0, 32, source).status());
    LSE_EXPECT_EQ(compiler.compiles, 1);
    LSE_EXPECT_EQ(cache.stats().disk_hits, 1u);
    LSE_EXPECT(cache.resources(0, 32, source.dialect, source.entry_name));
    LSE_EXPECT(cache.census(0, 32, source.dialect, source.entry_name));
  }
  for (const auto& entry : std::filesystem::directory_iterator(dir.path)) {
    if (entry.path().extension() != ".source") continue;
    auto altered = source.source;
    altered[altered.size() - 2] = 'x';
    std::ofstream out(entry.path());
    out << altered;
  }
  {
    JitCache cache(backend, compiler, dir.path.string());
    LSE_EXPECT_OK(cache.get_or_compile(0, 33, source).status());
    LSE_EXPECT_EQ(compiler.compiles, 2);
    LSE_EXPECT_EQ(cache.stats().disk_hits, 0u);
  }
}

LSE_TEST(compiled_source_retains_type_bound_launch_and_dialect_variants) {
  ArtifactDirectory dir;
  ArtifactCompiler compiler;
  ArtifactBackend backend;
  auto ordinary = artifact_source(), wider = artifact_source("wider", "256");
  auto integer = artifact_source("integer", "128", "u32");
  auto launch = artifact_source("launch");
  const auto at = launch.source.find("256 : index");
  launch.source.replace(at, 3, "128");
  for (auto* source : {&ordinary, &wider, &integer, &launch})
    LSE_EXPECT(finalize_source_identity(*source));
  JitCache cache(backend, compiler, dir.path.string());
  std::uint64_t signature = 40;
  for (const auto* source : {&ordinary, &wider, &integer, &launch})
    LSE_EXPECT_OK(cache.get_or_compile(0, ++signature, *source).status());
  LSE_EXPECT_EQ(compiler.compiles, 4);
  auto other_dialect = ordinary;
  other_dialect.dialect = Dialect::kHip;
  LSE_EXPECT_OK(cache.get_or_compile(0, 41, other_dialect).status());
  LSE_EXPECT_EQ(compiler.compiles, 5);
}

LSE_TEST(compiled_source_preserves_toolchain_geometry_and_device_ownership) {
  ArtifactDirectory dir;
  ArtifactCompiler compiler;
  ArtifactBackend first, second;
  first.owner = 100;
  second.owner = 200;
  auto source = artifact_source();
  LSE_EXPECT(finalize_source_identity(source));
  {
    JitCache cache(first, compiler, dir.path.string());
    auto result = cache.get_or_compile(0, 51, source);
    LSE_EXPECT(result.ok() && result->executable == 101u);
  }
  {
    JitCache cache(second, compiler, dir.path.string());
    auto result = cache.get_or_compile(0, 52, source);
    LSE_EXPECT(result.ok() && result->executable == 201u);
    LSE_EXPECT_EQ(compiler.compiles, 1);
  }
  second.info.compute_units = 17;
  {
    JitCache cache(second, compiler, dir.path.string());
    LSE_EXPECT_OK(cache.get_or_compile(0, 53, source).status());
    LSE_EXPECT_EQ(compiler.compiles, 2);
  }
  compiler.id = "exact-source-test.v2";
  {
    JitCache cache(second, compiler, dir.path.string());
    LSE_EXPECT_OK(cache.get_or_compile(0, 54, source).status());
    LSE_EXPECT_EQ(compiler.compiles, 3);
  }
  second.info.arch = "gfx1100";
  {
    JitCache cache(second, compiler, dir.path.string());
    LSE_EXPECT_OK(cache.get_or_compile(0, 55, source).status());
    LSE_EXPECT_EQ(compiler.compiles, 4);
  }
}

LSE_TEST_MAIN()
