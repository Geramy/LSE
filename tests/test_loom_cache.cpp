#include "harness.hpp"
#include "lse/backends/hrx/arch_database.hpp"
#include "lse/backends/hrx/loomc/loom_emitter.hpp"
#include "lse/graph/graph.hpp"
#include "lse/graph/ops.hpp"
#include "lse/graph/program.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <unordered_set>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <optional>
#include <barrier>
#include <thread>
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

LSE_TEST(loom_cache_separates_contiguous_views_from_owned_slice_bindings) {
  backend::LoomEmitter emitter;
  const auto info = device();
  auto source = leaf({1, 4, 8});
  auto left = slice(source, 1, 0, 2);
  auto right = slice(source, 1, 2, 4);
  auto sum = left + right;
  FusionGroup group;
  group.nodes = {sum.node()};
  group.inputs = {left.node(), right.node()};
  group.outputs = {sum.node()};
  group.anchor = sum.node()->kind;
  group.anchor_class = sum.node()->fclass;
  const auto view_key = emitter.cache_key(group, info);
  auto view = emitter.emit(group, info);
  LSE_EXPECT(view.ok());
  if (!view.ok()) return;
  LSE_EXPECT(view->source.find("buffer.assume.noalias") == std::string::npos);
  left.node()->requires_owned_storage = true;
  right.node()->requires_owned_storage = true;
  LSE_EXPECT(view_key != emitter.cache_key(group, info));
  auto owned = emitter.emit(group, info);
  LSE_EXPECT(owned.ok());
  if (!owned.ok()) return;
  LSE_EXPECT(owned->source.find("buffer.assume.noalias") != std::string::npos);
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
  mutable int resource_reads = 0;
  std::uint32_t metadata_version = 0;
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
  std::uint32_t resource_metadata_version() const noexcept override {
    return metadata_version;
  }
  std::vector<backend::KernelResources> resources(
      std::span<const std::byte> bytes) const override {
    ++resource_reads;
    backend::KernelResources r;
    r.entry = entry({reinterpret_cast<const char*>(bytes.data()), bytes.size()});
    r.vector_registers = backend::DeviceFact<std::uint32_t>::queried(17);
    r.private_segment_bytes = backend::DeviceFact<std::uint32_t>::queried(192);
    return {r};
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

namespace {
std::string cache_text(const std::filesystem::path& path) {
  std::ifstream in(path);
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}
void cache_write(const std::filesystem::path& path, std::string_view text) {
  std::ofstream out(path);
  out << text;
}
struct CacheFamily {
  std::string stem;
  std::filesystem::path meta, code, source, alias;
};
CacheFamily seed_cache_family(const std::filesystem::path& dir,
                              std::string_view version, unsigned key,
                              std::string_view entry = "lse_loom_123") {
  CacheFamily out;
  out.stem = version.empty() ? std::to_string(key)
      : "lse-" + std::string(version) + "-" + std::to_string(key);
  out.meta = dir / (out.stem + ".meta");
  out.code = dir / (out.stem + ".11.co");
  out.source = dir / (out.stem + ".source");
  const auto alias_stem = version.empty() ? std::to_string(key + 1000)
      : "lse-" + std::string(version) + "-" + std::to_string(key + 1000);
  out.alias = dir / (alias_stem + ".alias");
  const auto ownership = version.empty() ? std::string{}
      : "lse-kernel-cache-v1\n" + std::string(version) + "\n";
  cache_write(out.meta, ownership + "gfx1201\n000000000000000b\n" +
      std::string(entry) + "\nresource_version 0\n");
  cache_write(out.code, "complete old code object");
  cache_write(out.source, "complete old emitted source");
  const auto alias_owner = version.empty() ? std::string{}
      : "lse-kernel-alias-v1\n" + std::string(version) + "\n";
  cache_write(out.alias, alias_owner + "lse_loom_456\n" + out.stem + "\n11\n");
  return out;
}
void expect_cache_family(const CacheFamily& family, bool present) {
  for (const auto& path : {family.meta, family.code, family.source, family.alias})
    LSE_EXPECT(std::filesystem::exists(path) == present);
}
}  // namespace

LSE_TEST(cache_dump_overlap_preserves_all_cache_ownership_versions_and_user_files) {
  ArtifactDirectory dir;
  const auto older = seed_cache_family(dir.path, "0.0.0", 61);
  const auto same = seed_cache_family(dir.path, kernel_cache_version(), 62);
  const auto newer = seed_cache_family(dir.path, "999.0.0", 63);
  const auto user = dir.path / "user-data.loom";
  const auto generated = dir.path / "lse_loom_123.loom";
  cache_write(user, "user source");
  cache_write(generated, "generated source");
  ::setenv("LSE_HIP_DUMP", dir.path.c_str(), 1);
  // This is the first purge in this test process, including debug cleanup.
  purge_kernel_artifacts(dir.path.string());
  expect_cache_family(older, false);
  expect_cache_family(same, true);
  expect_cache_family(newer, true);
  LSE_EXPECT(cache_text(user) == "user source");
  LSE_EXPECT(!std::filesystem::exists(generated));
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

LSE_TEST(cached_resource_reader_refreshes_facts_without_recompiling_code) {
  ArtifactDirectory dir;
  ArtifactCompiler compiler;
  ArtifactBackend backend;
  auto source = artifact_source();
  LSE_EXPECT(finalize_source_identity(source));
  {
    JitCache cache(backend, compiler, dir.path.string());
    LSE_EXPECT_OK(cache.get_or_compile(0, 61, source).status());
    LSE_EXPECT_EQ(compiler.compiles, 1);
  }
  compiler.metadata_version = 1;
  auto& measured = opt::KernelMeasurements::instance();
  measured.clear();
  {
    JitCache cache(backend, compiler, dir.path.string());
    LSE_EXPECT(!measured.known(source.entry_name));
    LSE_EXPECT(!measured.known(source.structural_entry_name));
    LSE_EXPECT_OK(cache.get_or_compile(0, 61, source).status());
    LSE_EXPECT_EQ(compiler.compiles, 1);
    LSE_EXPECT_EQ(compiler.resource_reads, 1);
    LSE_EXPECT_EQ(cache.stats().disk_hits, 1u);
    const auto r = measured.lookup(source.structural_entry_name);
    LSE_EXPECT(r.private_segment_bytes.known());
    LSE_EXPECT_EQ(r.private_segment_bytes.value, 192u);
    LSE_EXPECT(r.spilled() == backend::SpillState::kUnknown);
  }
  measured.clear();
  {
    JitCache cache(backend, compiler, dir.path.string());
    const auto r = measured.lookup(source.structural_entry_name);
    LSE_EXPECT(r.private_segment_bytes.known());
    LSE_EXPECT_EQ(r.private_segment_bytes.value, 192u);
    LSE_EXPECT_OK(cache.get_or_compile(0, 61, source).status());
    LSE_EXPECT_EQ(compiler.compiles, 1);
    LSE_EXPECT_EQ(compiler.resource_reads, 1);
    LSE_EXPECT_EQ(cache.stats().disk_hits, 1u);
  }
}


LSE_TEST(cache_release_ownership_removes_only_complete_older_families) {
  ArtifactDirectory dir;
  ArtifactCompiler compiler;
  ArtifactBackend backend;
  const auto legacy = seed_cache_family(dir.path, "", 71);
  const auto older = seed_cache_family(dir.path, "0.0.0", 72);
  const auto same = seed_cache_family(dir.path, kernel_cache_version(), 73);
  const auto newer = seed_cache_family(dir.path, "999.0.0", 74);
  const auto unrelated = dir.path / "notes.txt";
  const auto orphan = dir.path / "789.11.co";
  const auto foreign = seed_cache_family(dir.path, "", 75, "user_kernel");
  const auto temp = dir.path / (older.stem + ".meta.tmp123");
  const auto empty_code = dir.path / (older.stem + ".12.co");
  const auto notes_code = dir.path / (older.stem + ".notes.co");
  cache_write(unrelated, "retain user file");
  cache_write(orphan, "unattributed code object");
  cache_write(temp, "incomplete publication");
  cache_write(empty_code, "");
  cache_write(notes_code, "unrelated family-adjacent file");
  JitCache cache(backend, compiler, dir.path.string());
  expect_cache_family(legacy, false);
  expect_cache_family(older, false);
  expect_cache_family(same, true);
  expect_cache_family(newer, true);
  expect_cache_family(foreign, true);
  LSE_EXPECT(cache_text(unrelated) == "retain user file");
  LSE_EXPECT(std::filesystem::exists(orphan));
  LSE_EXPECT(std::filesystem::exists(temp));
  LSE_EXPECT(std::filesystem::exists(empty_code));
  LSE_EXPECT(cache_text(notes_code) == "unrelated family-adjacent file");
  LSE_EXPECT_EQ(compiler.compiles, 0);
}

LSE_TEST(cache_release_ownership_preserves_partial_malformed_and_symlink_entries) {
  ArtifactDirectory dir;
  ArtifactCompiler compiler;
  ArtifactBackend backend;
  const auto partial = seed_cache_family(dir.path, "0.0.0", 81);
  cache_write(partial.meta, "lse-kernel-cache-v1\n0.0.0\ngfx1201\n");
  const auto missing = seed_cache_family(dir.path, "0.0.0", 82);
  std::filesystem::remove(missing.code);
  const auto malformed = seed_cache_family(dir.path, "", 83);
  cache_write(malformed.meta, "gfx1201\nb trailing\nlse_loom_123\n");
  const auto mismatched = seed_cache_family(dir.path, "999.0.0", 84);
  cache_write(mismatched.meta,
      "lse-kernel-cache-v1\n0.0.0\ngfx1201\nb\nlse_loom_123\n");
  const auto unknown = seed_cache_family(dir.path, "0.0.0", 85);
  cache_write(unknown.meta,
      "lse-kernel-cache-v2\n0.0.0\ngfx1201\nb\nlse_loom_123\n");
  const auto symlink = seed_cache_family(dir.path, "0.0.0", 86);
  const auto outside = dir.path / "user-owned-metadata";
  std::filesystem::rename(symlink.meta, outside);
  std::filesystem::create_symlink(outside, symlink.meta);
  const auto directory = seed_cache_family(dir.path, "0.0.0", 87);
  std::filesystem::remove(directory.meta);
  std::filesystem::create_directory(directory.meta);
  JitCache cache(backend, compiler, dir.path.string());
  for (const auto& family : {partial, malformed, mismatched, unknown, symlink, directory})
    expect_cache_family(family, true);
  LSE_EXPECT(std::filesystem::exists(missing.meta));
  LSE_EXPECT(std::filesystem::exists(missing.source));
  LSE_EXPECT(std::filesystem::exists(missing.alias));
  LSE_EXPECT(std::filesystem::exists(outside));
  LSE_EXPECT(std::filesystem::is_symlink(symlink.meta));
}

LSE_TEST(cache_release_alias_cleanup_requires_matching_owned_artifact) {
  ArtifactDirectory dir;
  const auto older = seed_cache_family(dir.path, "0.0.0", 91);
  const auto wrong_hash = dir.path / "lse-0.0.0-2001.alias";
  const auto wrong_owner = dir.path / "lse-999.0.0-2002.alias";
  const auto traversal = dir.path / "lse-0.0.0-2003.alias";
  cache_write(wrong_hash, "lse-kernel-alias-v1\n0.0.0\nlse_loom_456\n" +
      older.stem + "\n12\n");
  cache_write(wrong_owner, "lse-kernel-alias-v1\n999.0.0\nlse_loom_456\n" +
      older.stem + "\n11\n");
  cache_write(traversal, "lse-kernel-alias-v1\n0.0.0\nlse_loom_456\n../" +
      older.stem + "\n11\n");
  purge_kernel_artifacts(dir.path.string());
  expect_cache_family(older, false);
  for (const auto& path : {wrong_hash, wrong_owner, traversal})
    LSE_EXPECT(std::filesystem::exists(path));
}

LSE_TEST(cache_release_cleanup_uses_selected_directory_at_each_startup) {
  ArtifactDirectory dir;
  const auto selected = dir.path / "explicit";
  const auto other = dir.path / "environment";
  std::filesystem::create_directories(selected);
  std::filesystem::create_directories(other);
  const auto old_selected = seed_cache_family(selected, "0.0.0", 101);
  const auto old_other = seed_cache_family(other, "0.0.0", 102);
  const auto* before = std::getenv("LSE_CACHE_DIR");
  const std::optional<std::string> saved = before ? std::optional<std::string>(before)
                                                 : std::nullopt;
  ::setenv("LSE_CACHE_DIR", other.c_str(), 1);
  ArtifactCompiler compiler;
  ArtifactBackend backend;
  { JitCache cache(backend, compiler, selected.string()); }
  expect_cache_family(old_selected, false);
  expect_cache_family(old_other, true);
  // A second selected cache is not hidden behind a process-global once_flag.
  { JitCache cache(backend, compiler); }
  expect_cache_family(old_other, false);
  if (saved) ::setenv("LSE_CACHE_DIR", saved->c_str(), 1);
  else ::unsetenv("LSE_CACHE_DIR");
}

LSE_TEST(cache_release_publication_keeps_same_release_warm_and_newer_facts_separate) {
  ArtifactDirectory dir;
  ArtifactCompiler compiler;
  ArtifactBackend backend;
  auto source = artifact_source("release_probe");
  LSE_EXPECT(finalize_source_identity(source));
  {
    JitCache cache(backend, compiler, dir.path.string());
    LSE_EXPECT_OK(cache.get_or_compile(0, 111, source).status());
  }
  std::filesystem::path meta;
  for (const auto& item : std::filesystem::directory_iterator(dir.path)) {
    if (item.path().extension() != ".meta") continue;
    meta = item.path();
    LSE_EXPECT(item.path().filename().string().starts_with(
        "lse-" + std::string(kernel_cache_version()) + "-"));
  }
  LSE_EXPECT(!meta.empty());
  LSE_EXPECT(cache_text(meta).starts_with(
      "lse-kernel-cache-v1\n" + std::string(kernel_cache_version()) + "\n"));
  const auto future = seed_cache_family(dir.path, "999.0.0", 112, "future_resources");
  cache_write(future.meta, cache_text(future.meta) +
      "res future_resources 99 - - - - - - - - - - - - - - - - - - - - - -\n");
  auto& measured = opt::KernelMeasurements::instance();
  measured.clear();
  {
    JitCache cache(backend, compiler, dir.path.string());
    LSE_EXPECT(!measured.known("future_resources"));
    LSE_EXPECT(measured.known(source.structural_entry_name));
    LSE_EXPECT_OK(cache.get_or_compile(0, 111, source).status());
    LSE_EXPECT_EQ(cache.stats().disk_hits, 1u);
  }
  LSE_EXPECT_EQ(compiler.compiles, 1);
  expect_cache_family(future, true);
}

LSE_TEST(cache_release_concurrent_publications_remain_complete_and_reusable) {
  ArtifactDirectory dir;
  std::barrier start(2);
  bool ok[2]{};
  auto worker = [&](unsigned index) {
    ArtifactCompiler compiler;
    ArtifactBackend backend;
    auto source = artifact_source("concurrent_release");
    const bool finalized = finalize_source_identity(source);
    JitCache cache(backend, compiler, dir.path.string());
    start.arrive_and_wait();
    ok[index] = finalized && cache.get_or_compile(0, 121 + index, source).ok();
  };
  std::thread one(worker, 0), two(worker, 1);
  one.join();
  two.join();
  LSE_EXPECT(ok[0] && ok[1]);
  ArtifactCompiler compiler;
  ArtifactBackend backend;
  auto source = artifact_source("concurrent_release");
  LSE_EXPECT(finalize_source_identity(source));
  JitCache cache(backend, compiler, dir.path.string());
  LSE_EXPECT_OK(cache.get_or_compile(0, 123, source).status());
  LSE_EXPECT_EQ(compiler.compiles, 0);
  LSE_EXPECT_EQ(cache.stats().disk_hits, 1u);
  for (const auto& item : std::filesystem::directory_iterator(dir.path))
    LSE_EXPECT(item.path().filename().string().find(".tmp") == std::string::npos);
}

LSE_TEST(cache_publication_reclaim_preserves_unrelated_partial_and_symlink_siblings) {
  ArtifactDirectory dir;
  ArtifactCompiler compiler;
  ArtifactBackend backend;
  auto source = artifact_source("changing_source");
  JitCache cache(backend, compiler, dir.path.string());
  LSE_EXPECT_OK(cache.get_or_compile(0, 125, source).status());
  std::filesystem::path stem, original;
  for (const auto& item : std::filesystem::directory_iterator(dir.path)) {
    if (item.path().extension() == ".meta") {
      stem = item.path();
      stem.replace_extension();
    }
    if (item.path().extension() == ".co") original = item.path();
  }
  LSE_EXPECT(!stem.empty() && !original.empty());
  if (stem.empty() || original.empty()) return;
  const auto notes = stem.string() + ".notes.co";
  const auto empty = stem.string() + ".12.co";
  const auto linked = stem.string() + ".13.co";
  const auto target = dir.path / "user-code";
  cache_write(notes, "user notes");
  cache_write(empty, "");
  cache_write(target, "user binary");
  std::filesystem::create_symlink(target, linked);
  source = artifact_source("changing_source", "256");
  LSE_EXPECT_OK(cache.get_or_compile(0, 125, source).status());
  LSE_EXPECT_EQ(compiler.compiles, 2);
  LSE_EXPECT(!std::filesystem::exists(original));
  LSE_EXPECT(cache_text(notes) == "user notes");
  LSE_EXPECT(std::filesystem::exists(empty));
  LSE_EXPECT(std::filesystem::is_symlink(linked));
  LSE_EXPECT(cache_text(target) == "user binary");
}

LSE_TEST(cache_release_codec_storage_variants_keep_distinct_compiled_sources) {
  ArtifactDirectory dir;
  ArtifactCompiler compiler;
  ArtifactBackend backend;
  JitCache cache(backend, compiler, dir.path.string());
  std::uint64_t signature = 130;
  for (const auto dtype : {"f32", "f16", "bf16", "fp8", "bf8"}) {
    auto source = artifact_source("same_shape", "128", dtype);
    LSE_EXPECT(finalize_source_identity(source));
    LSE_EXPECT_OK(cache.get_or_compile(0, ++signature, source).status());
  }
  LSE_EXPECT_EQ(compiler.compiles, 5);
}


LSE_TEST(retained_emission_refuses_changed_buffer_alias_class) {
  backend::LoomEmitter emitter;
  const auto info = device();
  auto group = subtraction();
  for (std::size_t i = 0; i < group.inputs.size(); ++i) {
    group.inputs[i]->buffer.handle = 11 + i;
    group.inputs[i]->buffer.size_bytes = 128 * sizeof(float);
  }
  group.outputs[0]->buffer.handle = 13;
  group.outputs[0]->buffer.size_bytes = 128 * sizeof(float);
  Program program;
  program.retain(group.outputs, {}, {group}, group.nodes);
  auto& retained = program.groups()[0];
  const auto key = emitter.cache_key(retained, info);
  auto kernel = emitter.emit(retained, info);
  LSE_EXPECT(kernel.ok());
  if (!kernel.ok()) return;
  LSE_EXPECT(program.cache_emission(0, retained, nullptr, &emitter, key,
                                   info.arch, kernel.release()) != nullptr);
  std::uint64_t cached_key = 0;
  LSE_EXPECT(program.cached_emission(0, retained, nullptr, &emitter,
                                    info.arch, &cached_key) != nullptr);
  retained.inputs[1]->buffer = retained.inputs[0]->buffer;
  program.reset_compute();
  LSE_EXPECT(program.cached_emission(0, retained, nullptr, &emitter,
                                    info.arch, &cached_key) == nullptr);
  LSE_EXPECT(key != emitter.cache_key(retained, info));
  auto aliased = emitter.emit(retained, info);
  LSE_EXPECT(aliased.ok());
  if (aliased.ok())
    LSE_EXPECT(aliased->source.find("buffer.assume.noalias") == std::string::npos);
  // Disjoint windows in one slab still permit the original noalias source.
  retained.inputs[1]->buffer.offset = retained.inputs[0]->buffer.size_bytes;
  LSE_EXPECT_EQ(key, emitter.cache_key(retained, info));
}

// ---- launch index ----------------------------------------------------------

namespace {
// Accepts a bundle: every requested export must be in the object.
struct BundleBackend final : backend::IBackend {
  backend::DeviceInfo info = device();
  int loads = 0;
  int bundle_loads = 0;
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
    if (text.find("export(\"" + std::string(name) + "\")") == std::string_view::npos)
      return LSE_ERROR(kNotFound, "executable does not export ", std::string(name));
    backend::KernelHandle handle;
    handle.executable = 1000 + static_cast<std::uint64_t>(++loads);
    handle.name = name;
    return handle;
  }
  Result<std::vector<backend::KernelHandle>> load_executables(
      std::span<const std::string> names, std::span<const std::byte> bytes) override {
    ++bundle_loads;
    return IBackend::load_executables(names, bytes);
  }
  Status launch(const backend::KernelHandle&, const backend::LaunchDims&,
                const backend::DispatchArgs&, const backend::DispatchTarget&) override {
    return LSE_ERROR(kUnimplemented, "stub");
  }
  Status synchronize() override { return OkStatus(); }
  std::string_view name() const noexcept override { return "bundle-test"; }
  std::span<const KernelToolchain> toolchains() const noexcept override { return {}; }
};

FusionGroup scaled(std::int64_t width) {
  auto a = leaf(Shape{width}), b = leaf(Shape{width});
  auto out = a * b;
  FusionGroup group;
  group.nodes = {out.node()};
  group.inputs = {a.node(), b.node()};
  group.outputs = {out.node()};
  group.anchor = out.node()->kind;
  group.anchor_class = out.node()->fclass;
  return group;
}

// A prepared set as Scheduler::prepare hands it over: the groups' cache keys
// and their full emissions.
struct PreparedSet {
  std::vector<FusionGroup> groups;
  std::vector<std::uint64_t> signatures;
  std::vector<EmittedKernel> kernels;
  std::vector<JitCache::Preload> preload;
};
PreparedSet prepared(const backend::LoomEmitter& emitter, const backend::DeviceInfo& info,
                     std::vector<FusionGroup> groups) {
  PreparedSet out;
  out.groups = std::move(groups);
  for (const FusionGroup& g : out.groups) {
    out.signatures.push_back(emitter.cache_key(g, info));
    auto emitted = emitter.emit(g, info);
    LSE_EXPECT(emitted.ok());
    out.kernels.push_back(emitted.ok() ? emitted.release() : EmittedKernel{});
  }
  for (std::size_t i = 0; i < out.kernels.size(); ++i)
    out.preload.push_back({out.signatures[i], &out.kernels[i]});
  return out;
}
std::vector<FusionGroup> three_groups() { return {subtraction(), subtraction(true), scaled(256)}; }

std::vector<std::filesystem::path> files_with(const std::filesystem::path& dir,
                                              std::string_view suffix) {
  std::vector<std::filesystem::path> out;
  std::error_code ec;
  for (std::filesystem::recursive_directory_iterator it(dir, ec), end; !ec && it != end; ++it)
    if (it->path().string().ends_with(suffix)) out.push_back(it->path());
  return out;
}
void flip_byte(const std::filesystem::path& path, std::size_t from_end) {
  std::fstream f(path, std::ios::in | std::ios::out | std::ios::binary);
  f.seekg(0, std::ios::end);
  const auto size = static_cast<std::size_t>(f.tellg());
  const auto at = static_cast<std::streamoff>(size - 1 - std::min(from_end, size - 1));
  f.seekg(at);
  char c = 0;
  f.read(&c, 1);
  c = static_cast<char>(c ^ 0x5a);
  f.seekp(at);
  f.write(&c, 1);
}
}  // namespace

LSE_TEST(launch_index_makes_a_prepared_set_resident_without_writing_source) {
  ArtifactDirectory dir;
  const auto info = device();
  {
    ArtifactCompiler compiler;
    BundleBackend backend;
    backend::LoomEmitter emitter;
    auto set = prepared(emitter, info, three_groups());
    JitCache cache(backend, compiler, dir.path.string());
    LSE_EXPECT_OK(cache.preload(0, set.preload, true));
    LSE_EXPECT_EQ(compiler.compiles, 1);
    LSE_EXPECT_EQ(files_with(cache.launch_index_dir(), ".launch").size(), 1u);
  }
  // A second process: new emitter, new cache, the same groups rebuilt.
  ArtifactCompiler compiler;
  BundleBackend backend;
  backend::LoomEmitter emitter;
  JitCache cache(backend, compiler, dir.path.string());
  const auto groups = three_groups();
  std::vector<std::uint64_t> signatures;
  for (const auto& g : groups) signatures.push_back(emitter.cache_key(g, info));
  auto restored = cache.restore(0, Dialect::kLoom, signatures, emitter);
  LSE_EXPECT(restored.ok() && *restored);
  LSE_EXPECT_EQ(compiler.compiles, 0);
  LSE_EXPECT_EQ(backend.bundle_loads, 1);
  LSE_EXPECT_EQ(cache.stats().index_hits, 1u);
  LSE_EXPECT_EQ(cache.stats().index_misses, 0u);
  // Nothing was emitted to get here...
  LSE_EXPECT_EQ(emitter.cache_stats().misses, 0u);
  LSE_EXPECT_EQ(emitter.cache_stats().entries, groups.size());
  for (std::size_t i = 0; i < groups.size(); ++i) {
    LSE_EXPECT(cache.try_get(0, signatures[i], Dialect::kLoom) != nullptr);
    // ...and launching writes no text either: the description is the index's.
    auto launch = emitter.emit_launch(groups[i], info);
    auto fresh = backend::LoomEmitter{}.emit(groups[i], info);
    LSE_EXPECT(launch.ok() && fresh.ok());
    if (!launch.ok() || !fresh.ok()) continue;
    LSE_EXPECT(launch->source.empty());
    LSE_EXPECT(launch->entry_name == fresh->entry_name);
    LSE_EXPECT(launch->structural_entry_name == fresh->structural_entry_name);
    LSE_EXPECT(launch->content_addressed == fresh->content_addressed);
    LSE_EXPECT(launch->binding_order == fresh->binding_order);
    LSE_EXPECT_EQ(launch->constants.total_bytes, fresh->constants.total_bytes);
    LSE_EXPECT_EQ(launch->constants.fields.size(), fresh->constants.fields.size());
    for (int d = 0; d < 3; ++d) {
      LSE_EXPECT_EQ(launch->dims.workgroup_count[d], fresh->dims.workgroup_count[d]);
      LSE_EXPECT_EQ(launch->dims.workgroup_size[d], fresh->dims.workgroup_size[d]);
    }
    LSE_EXPECT_EQ(launch->lds_bytes, fresh->lds_bytes);
    const auto* handle = cache.try_get(0, signatures[i], Dialect::kLoom);
    LSE_EXPECT(handle != nullptr && handle->name == fresh->entry_name);
  }
  LSE_EXPECT_EQ(emitter.cache_stats().misses, 0u);
  // The object a restored kernel stands for is the one its source names: a
  // later compile request with the real text is a memory hit, not a rebuild.
  auto again = backend::LoomEmitter{}.emit(groups[0], info);
  LSE_EXPECT(again.ok());
  if (again.ok()) LSE_EXPECT_OK(cache.get_or_compile(0, signatures[0], *again).status());
  LSE_EXPECT_EQ(compiler.compiles, 0);
}

LSE_TEST(launch_index_key_changes_with_every_input_that_changes_source) {
  ArtifactDirectory dir;
  const auto info = device();
  ArtifactCompiler compiler;
  BundleBackend backend;
  backend::LoomEmitter emitter;
  auto set = prepared(emitter, info, three_groups());
  {
    JitCache cache(backend, compiler, dir.path.string());
    LSE_EXPECT_OK(cache.preload(0, set.preload, true));
  }
  auto misses = [&](const std::vector<FusionGroup>& groups, backend::DeviceInfo at,
                    std::string compiler_id = "exact-source-test.v1") {
    ArtifactCompiler other;
    other.id = std::move(compiler_id);
    BundleBackend device_backend;
    device_backend.info = at;
    backend::LoomEmitter fresh;
    JitCache cache(device_backend, other, dir.path.string());
    std::vector<std::uint64_t> signatures;
    for (const auto& g : groups) signatures.push_back(fresh.cache_key(g, at));
    auto restored = cache.restore(0, Dialect::kLoom, signatures, fresh);
    return restored.ok() && !*restored && cache.stats().index_rejects == 0 &&
           fresh.cache_stats().entries == 0;
  };
  // The control: the same groups on the same device are a hit.
  LSE_EXPECT(!misses(three_groups(), info));
  // A shape.
  LSE_EXPECT(misses({subtraction(), subtraction(true), scaled(512)}, info));
  // The edges (a-b against b-a) in the set's place.
  LSE_EXPECT(misses({subtraction(true), subtraction(), scaled(256)}, info));
  // A device fact the emitter writes against.
  auto halved = info;
  halved.lds_bytes_per_workgroup /= 2;
  LSE_EXPECT(misses(three_groups(), halved));
  auto cus = info;
  cus.compute_units = 40;
  LSE_EXPECT(misses(three_groups(), cus));
  auto arch = info;
  arch.arch = "gfx1151";
  LSE_EXPECT(misses(three_groups(), arch));
  // The compiler.
  LSE_EXPECT(misses(three_groups(), info, "exact-source-test.v2"));
  // A subset of the set is another set.
  LSE_EXPECT(misses({subtraction(), subtraction(true)}, info));
}

LSE_TEST(launch_index_is_scoped_to_the_engine_build_identity) {
  // The digest names this build's whole source tree, which is where every
  // kernel template is: an edit to one within a release is a new identity,
  // a new index directory and a new key, so no entry it wrote is reachable.
  const std::string_view build = engine_build_identity();
  LSE_EXPECT_EQ(build.size(), 64u);
  LSE_EXPECT(std::all_of(build.begin(), build.end(), [](char c) {
    return std::isxdigit(static_cast<unsigned char>(c)) != 0;
  }));
  ArtifactDirectory dir;
  ArtifactCompiler compiler;
  BundleBackend backend;
  JitCache cache(backend, compiler, dir.path.string());
  LSE_EXPECT(std::filesystem::path(cache.launch_index_dir()).filename().string() ==
             "launch-" + std::string(kernel_cache_version()) + "-" +
                 std::string(build.substr(0, 16)));
  // Every engine source and header is hashed, not a list of kernel files that
  // a new template could be missing from.
  std::ifstream listed_file(LSE_IDENTITY_SOURCES);
  std::unordered_set<std::string> listed;
  for (std::string line; std::getline(listed_file, line);) listed.insert(line);
  std::size_t expected = 0;
  for (const char* tree : {"src", "include"}) {
    for (const auto& item : std::filesystem::recursive_directory_iterator(
             std::filesystem::path(LSE_SOURCE_ROOT) / tree)) {
      if (!item.is_regular_file()) continue;
      const auto rel = std::filesystem::relative(item.path(), LSE_SOURCE_ROOT).generic_string();
      if (rel.find("/.") != std::string::npos) continue;
      ++expected;
      if (!listed.contains(rel)) std::printf("    not in the build identity: %s\n", rel.c_str());
      LSE_EXPECT(listed.contains(rel));
    }
  }
  LSE_EXPECT(expected > 300);
  for (const char* must : {"src/backends/hrx/loomc/loom_emitter.cpp",
                           "src/backends/hrx/loomc/loom_sources.cpp",
                           "src/kernels/quant_linear.cpp", "include/lse/dispatch/cache.hpp"})
    LSE_EXPECT(listed.contains(must));

  // An entry that names another build is never served, even if found.
  auto info = device();
  backend::LoomEmitter emitter;
  auto set = prepared(emitter, info, three_groups());
  LSE_EXPECT_OK(cache.preload(0, set.preload, true));
  const auto entries = files_with(cache.launch_index_dir(), ".launch");
  LSE_EXPECT_EQ(entries.size(), 1u);
  if (entries.size() != 1) return;
  std::string bytes = cache_text(entries[0]);
  const auto at = bytes.find(std::string(build));
  LSE_EXPECT(at != std::string::npos);
  if (at == std::string::npos) return;
  bytes[at] = bytes[at] == '0' ? '1' : '0';
  cache_write(entries[0], bytes);
  backend::LoomEmitter fresh;
  JitCache other(backend, compiler, dir.path.string());
  auto restored = other.restore(0, Dialect::kLoom, set.signatures, fresh);
  LSE_EXPECT(restored.ok() && !*restored);
  LSE_EXPECT_EQ(other.stats().index_rejects, 1u);
}

LSE_TEST(launch_index_rejects_a_corrupt_entry_and_rebuilds_it) {
  ArtifactDirectory dir;
  const auto info = device();
  ArtifactCompiler compiler;
  BundleBackend backend;
  backend::LoomEmitter emitter;
  auto set = prepared(emitter, info, three_groups());
  std::filesystem::path entry;
  {
    JitCache cache(backend, compiler, dir.path.string());
    LSE_EXPECT_OK(cache.preload(0, set.preload, true));
    const auto entries = files_with(cache.launch_index_dir(), ".launch");
    LSE_EXPECT_EQ(entries.size(), 1u);
    if (entries.size() != 1) return;
    entry = entries[0];
  }
  for (const std::size_t from_end : {std::size_t{0}, std::size_t{40}, std::size_t{100000}}) {
    flip_byte(entry, from_end);
    backend::LoomEmitter fresh;
    JitCache cache(backend, compiler, dir.path.string());
    auto restored = cache.restore(0, Dialect::kLoom, set.signatures, fresh);
    LSE_EXPECT(restored.ok() && !*restored);
    LSE_EXPECT_EQ(cache.stats().index_rejects, 1u);
    LSE_EXPECT(!std::filesystem::exists(entry));
    LSE_EXPECT_EQ(fresh.cache_stats().entries, 0u);
    LSE_EXPECT(cache.try_get(0, set.signatures[0], Dialect::kLoom) == nullptr);
    // The miss path rebuilds the entry; the object on disk was sound.
    LSE_EXPECT_OK(cache.preload(0, set.preload, true));
    LSE_EXPECT(std::filesystem::exists(entry));
  }
  LSE_EXPECT_EQ(compiler.compiles, 1);
  // Truncated.
  {
    const std::string bytes = cache_text(entry);
    cache_write(entry, bytes.substr(0, bytes.size() / 2));
    backend::LoomEmitter fresh;
    JitCache cache(backend, compiler, dir.path.string());
    auto restored = cache.restore(0, Dialect::kLoom, set.signatures, fresh);
    LSE_EXPECT(restored.ok() && !*restored);
    LSE_EXPECT_EQ(cache.stats().index_rejects, 1u);
    LSE_EXPECT_OK(cache.preload(0, set.preload, true));
  }
  // The bundle object damaged in place, same size: rejected by its checksum,
  // removed, and compiled again rather than loaded.
  const auto objects = files_with(dir.path, ".co");
  LSE_EXPECT_EQ(objects.size(), 1u);
  if (objects.size() != 1) return;
  flip_byte(objects[0], 10);
  {
    backend::LoomEmitter fresh;
    JitCache cache(backend, compiler, dir.path.string());
    const int loads_before = backend.bundle_loads;
    auto restored = cache.restore(0, Dialect::kLoom, set.signatures, fresh);
    LSE_EXPECT(restored.ok() && !*restored);
    LSE_EXPECT_EQ(cache.stats().index_rejects, 1u);
    LSE_EXPECT(!std::filesystem::exists(objects[0]));
    LSE_EXPECT_EQ(backend.bundle_loads, loads_before);  // the damaged bytes never load
    LSE_EXPECT_OK(cache.preload(0, set.preload, true));
    LSE_EXPECT_EQ(compiler.compiles, 2);
  }
  backend::LoomEmitter fresh;
  JitCache cache(backend, compiler, dir.path.string());
  auto restored = cache.restore(0, Dialect::kLoom, set.signatures, fresh);
  LSE_EXPECT(restored.ok() && *restored);
  LSE_EXPECT_EQ(cache.stats().index_rejects, 0u);
}

LSE_TEST(launch_index_misses_when_the_object_or_the_resident_set_differs) {
  ArtifactDirectory dir;
  const auto info = device();
  ArtifactCompiler compiler;
  BundleBackend backend;
  backend::LoomEmitter emitter;
  auto set = prepared(emitter, info, three_groups());
  {
    JitCache cache(backend, compiler, dir.path.string());
    LSE_EXPECT_OK(cache.preload(0, set.preload, true));
  }
  // One kernel of the set already resident from elsewhere: the bundle would
  // load a set this process does not need, so it is a miss, not a partial hit.
  {
    backend::LoomEmitter fresh;
    JitCache cache(backend, compiler, dir.path.string());
    LSE_EXPECT_OK(cache.get_or_compile(0, set.signatures[2], set.kernels[2]).status());
    auto restored = cache.restore(0, Dialect::kLoom, set.signatures, fresh);
    LSE_EXPECT(restored.ok() && !*restored);
    LSE_EXPECT_EQ(cache.stats().index_rejects, 0u);
  }
  // The bundle object gone (a cache cleaned by hand): a miss, not damage.
  for (const auto& path : files_with(dir.path, ".co")) std::filesystem::remove(path);
  backend::LoomEmitter fresh;
  JitCache cache(backend, compiler, dir.path.string());
  auto restored = cache.restore(0, Dialect::kLoom, set.signatures, fresh);
  LSE_EXPECT(restored.ok() && !*restored);
  LSE_EXPECT_EQ(cache.stats().index_rejects, 0u);
  LSE_EXPECT_EQ(cache.stats().index_misses, 1u);
  LSE_EXPECT(!files_with(cache.launch_index_dir(), ".launch").empty());
}

LSE_TEST(launch_index_concurrent_writers_and_readers_see_only_complete_entries) {
  ArtifactDirectory dir;
  const auto info = device();
  constexpr unsigned kWriters = 4;
  constexpr unsigned kRounds = 25;
  std::barrier start(kWriters + 1);
  std::atomic<unsigned> failures{0};
  std::atomic<unsigned> rejects{0};
  std::atomic<bool> writing{true};
  auto writer = [&](unsigned index) {
    ArtifactCompiler compiler;
    BundleBackend backend;
    backend::LoomEmitter emitter;
    // Two writers share a set, two write their own.
    auto groups = index % 2 == 0 ? three_groups()
        : std::vector<FusionGroup>{subtraction(), scaled(64 * (index + 1))};
    auto set = prepared(emitter, info, std::move(groups));
    start.arrive_and_wait();
    for (unsigned round = 0; round < kRounds; ++round) {
      JitCache cache(backend, compiler, dir.path.string());
      if (!cache.preload(0, set.preload, true).ok()) ++failures;
    }
  };
  auto reader = [&] {
    ArtifactCompiler compiler;
    BundleBackend backend;
    const auto groups = three_groups();
    start.arrive_and_wait();
    while (writing.load()) {
      backend::LoomEmitter emitter;
      std::vector<std::uint64_t> signatures;
      for (const auto& g : groups) signatures.push_back(emitter.cache_key(g, info));
      JitCache cache(backend, compiler, dir.path.string());
      auto restored = cache.restore(0, Dialect::kLoom, signatures, emitter);
      if (!restored.ok()) ++failures;
      rejects += static_cast<unsigned>(cache.stats().index_rejects);
    }
  };
  std::vector<std::thread> threads;
  for (unsigned i = 0; i < kWriters; ++i) threads.emplace_back(writer, i);
  std::thread read(reader);
  for (auto& t : threads) t.join();
  writing = false;
  read.join();
  LSE_EXPECT_EQ(failures.load(), 0u);
  LSE_EXPECT_EQ(rejects.load(), 0u);
  ArtifactCompiler compiler;
  BundleBackend backend;
  backend::LoomEmitter emitter;
  const auto groups = three_groups();
  std::vector<std::uint64_t> signatures;
  for (const auto& g : groups) signatures.push_back(emitter.cache_key(g, info));
  JitCache cache(backend, compiler, dir.path.string());
  auto restored = cache.restore(0, Dialect::kLoom, signatures, emitter);
  LSE_EXPECT(restored.ok() && *restored);
  LSE_EXPECT_EQ(compiler.compiles, 0);
  // Writers leave entries and objects, never their temporaries.
  for (const auto& item : std::filesystem::recursive_directory_iterator(dir.path))
    LSE_EXPECT(item.path().filename().string().find(".tmp") == std::string::npos);
  LSE_EXPECT_EQ(files_with(cache.launch_index_dir(), ".launch").size(), 3u);
}

LSE_TEST(launch_index_prunes_other_builds_but_keeps_recent_ones) {
  ArtifactDirectory dir;
  const std::string version(kernel_cache_version());
  auto seed = [&](const std::string& name, int age_minutes, bool notes = false) {
    const auto path = dir.path / name;
    std::filesystem::create_directories(path);
    cache_write(path / "1.launch", "entry");
    if (notes) cache_write(path / "user-notes.txt", "mine");
    std::filesystem::last_write_time(
        path, std::filesystem::file_time_type::clock::now() - std::chrono::minutes(age_minutes));
    return path;
  };
  const auto older_release = seed("launch-0.0.1-00000000000000aa", 1);
  const auto newer_release = seed("launch-999.0.0-00000000000000bb", 1);
  std::vector<std::filesystem::path> builds;
  for (int i = 0; i < 5; ++i) {
    char hex[17];
    std::snprintf(hex, sizeof hex, "%016x", 0xc0 + i);
    builds.push_back(seed("launch-" + version + "-" + hex, 10 * (i + 1)));
  }
  const auto foreign = dir.path / ("launch-" + version + "-notabuild");
  std::filesystem::create_directories(foreign);
  const auto keep_file = seed("launch-" + version + "-00000000000000ff", 100, true);
  ArtifactCompiler compiler;
  BundleBackend backend;
  JitCache cache(backend, compiler, dir.path.string());
  LSE_EXPECT(!std::filesystem::exists(older_release));
  LSE_EXPECT(std::filesystem::exists(newer_release));
  // The three most recently used other builds of this release stay.
  for (int i = 0; i < 5; ++i) LSE_EXPECT(std::filesystem::exists(builds[i]) == (i < 3));
  LSE_EXPECT(std::filesystem::exists(foreign));
  // A pruned directory holding anything but entries keeps that and itself.
  LSE_EXPECT(std::filesystem::exists(keep_file / "user-notes.txt"));
  LSE_EXPECT(!std::filesystem::exists(keep_file / "1.launch"));
  LSE_EXPECT(std::filesystem::exists(cache.launch_index_dir()));
}

LSE_TEST_MAIN()
