#include "lse/backends/hrx/loomc/loomc_compiler.hpp"

#include "lse/backends/hrx/code_object.hpp"

#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <utility>

#if LSE_HAVE_LOOMC
#include "lse/backends/hrx/compiler_image_identity.hpp"

#include "loomc/loomc.h"
#include "loomc/target/amdgpu.h"
#endif

#if LSE_HAVE_LOOMC
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <sys/mman.h>
#if defined(__APPLE__)
#include <TargetConditionals.h>
#include <malloc/malloc.h>
#if TARGET_OS_IOS
#include <os/proc.h>
#endif
#elif defined(__linux__)
#include <malloc.h>
#include <fstream>
#include <string>
#endif

namespace {
// Every loomc allocation goes through this: the system allocator, with what
// a compile asked for counted, so a compile that runs out of host memory says
// how much it wanted and how much the process had, and a traced run
// (LSE_TRACE_COMPILE_MEMORY=1) reports each compile's peak.
struct CompileMemory {
  std::atomic<std::uint64_t> live{0}, peak{0}, largest{0};
  void add(std::uint64_t n) {
    const auto now = live.fetch_add(n, std::memory_order_relaxed) + n;
    auto p = peak.load(std::memory_order_relaxed);
    while (now > p && !peak.compare_exchange_weak(p, now, std::memory_order_relaxed)) {}
    auto l = largest.load(std::memory_order_relaxed);
    while (n > l && !largest.compare_exchange_weak(l, n, std::memory_order_relaxed)) {}
  }
  void sub(std::uint64_t n) { live.fetch_sub(std::min(n, live.load(std::memory_order_relaxed)), std::memory_order_relaxed); }
};
CompileMemory g_compile_memory;

// What a block really holds, so a free takes back what its allocation added.
// Without it a free subtracts nothing and the count is every byte the compile
// ever asked for: on Linux that read as 8-15 GiB "peak" for a 2.4 MB source,
// and as hundreds of GiB "held" across a cold start.
std::uint64_t allocation_size(void* p) {
#if defined(__APPLE__)
  return p != nullptr ? malloc_size(p) : 0;
#elif defined(__linux__)
  return p != nullptr ? malloc_usable_size(p) : 0;
#else
  (void)p;
  return 0;
#endif
}

std::uint64_t available_memory() {
#if defined(__APPLE__) && TARGET_OS_IOS
  return os_proc_available_memory();
#elif defined(__linux__)
  // MemAvailable: what the kernel could hand out without swapping.
  std::ifstream meminfo("/proc/meminfo");
  std::string line;
  while (std::getline(meminfo, line))
    if (line.rfind("MemAvailable:", 0) == 0)
      return std::strtoull(line.c_str() + 13, nullptr, 10) * 1024;
  return 0;
#else
  return 0;
#endif
}

// The largest address range the process can still reserve, to tell an
// exhausted address space from exhausted memory.
std::uint64_t largest_reservable() {
  std::uint64_t lo = 0, hi = std::uint64_t{1} << 40;
  while (hi - lo > (std::uint64_t{16} << 20)) {
    const std::uint64_t mid = lo + (hi - lo) / 2;
    void* p = ::mmap(nullptr, mid, PROT_NONE, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (p != MAP_FAILED) {
      ::munmap(p, mid);
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return lo;
}

loomc_status_t accounted_ctl(void* self, loomc_allocator_command_t command, const void* params,
                             void** inout_ptr) {
  (void)self;
  const loomc_allocator_t system = loomc_allocator_system();
  std::uint64_t before = 0;
  if ((command == LOOMC_ALLOCATOR_COMMAND_REALLOC || command == LOOMC_ALLOCATOR_COMMAND_FREE) &&
      inout_ptr != nullptr)
    before = allocation_size(*inout_ptr);
  const std::uint64_t asked =
      params != nullptr && command != LOOMC_ALLOCATOR_COMMAND_FREE
          ? static_cast<const loomc_allocator_alloc_params_t*>(params)->byte_length
          : 0;
  const loomc_status_t status = system.ctl(system.self, command, params, inout_ptr);
  if (command == LOOMC_ALLOCATOR_COMMAND_FREE) {
    g_compile_memory.sub(before);
    return status;
  }
  if (status != nullptr) {
    std::fprintf(stderr,
                 "lse: loomc allocation of %llu bytes failed (compile holds %.1f MiB, peak %.1f MiB; "
                 "available %.1f MiB; largest reservable address range %.1f MiB)\n",
                 static_cast<unsigned long long>(asked), g_compile_memory.live.load() / 1048576.0,
                 g_compile_memory.peak.load() / 1048576.0, available_memory() / 1048576.0,
                 largest_reservable() / 1048576.0);
    return status;
  }
  const std::uint64_t after = inout_ptr != nullptr ? allocation_size(*inout_ptr) : 0;
  g_compile_memory.sub(before);
  g_compile_memory.add(after != 0 ? after : asked);
  return status;
}

loomc_allocator_t accounted_allocator() { return loomc_allocator_t{nullptr, accounted_ctl}; }

bool trace_compile_memory() {
  static const bool on = [] {
    const char* v = std::getenv("LSE_TRACE_COMPILE_MEMORY");
    return v != nullptr && v[0] == '1';
  }();
  return on;
}
}  // namespace
#endif

namespace lse::backend {

namespace {

#if LSE_HAVE_LOOMC

#ifndef LSE_LOOMC_VERSION
#define LSE_LOOMC_VERSION "unknown"
#endif

// Every knob an invocation runs with, in one place, so identity() reads the
// same values the calls do and a flag edit reaches the JIT cache key without
// anyone remembering to do anything.
struct Options {
  // Lower all the way to target-low prepared for AMDGPU emission; SOURCE_LOW
  // stops before ABI and resource materialization and cannot be emitted.
  loomc_target_pipeline_kind_t pipeline =
      LOOMC_TARGET_PIPELINE_KIND_PREPARED_LOW;
  loomc_target_control_flow_lowering_t control_flow =
      LOOMC_TARGET_CONTROL_FLOW_LOWERING_CFG;
  // A malformed body reports as far as it can in one pass rather than one
  // diagnostic per compile attempt.
  std::uint32_t source_to_low_max_errors = 20;
  // Zero: the hot path wants the in-place module transform and the
  // diagnostics, not a serialized compile artifact beside them.
  loomc_compile_artifact_flags_t compile_artifacts = 0;
};

constexpr Options kOptions{};

// The manifest carries counts only — no binding ordinals and no constant
// offsets — so nothing here parses it and no manifest descriptor is attached.
// The kernarg layout comes from the AMDHSA metadata note the HAL already
// parses; `launch(...)` declaration order is the whole mapping.
constexpr const char* kArtifactFormat = LOOMC_ARTIFACT_FORMAT_AMDGPU_HSACO;

const char* pipeline_name(loomc_target_pipeline_kind_t kind) {
  return kind == LOOMC_TARGET_PIPELINE_KIND_PREPARED_LOW ? "prepared_low"
                                                         : "source_low";
}

const char* control_flow_name(loomc_target_control_flow_lowering_t cf) {
  return cf == LOOMC_TARGET_CONTROL_FLOW_LOWERING_CFG ? "cfg"
                                                      : "structured_low";
}

// Resolve the function this compiler actually calls. DYLD_LIBRARY_PATH and
// loader interposition can select a different image than CMake discovered.
// Content hashing is deliberately paid once per process, outside compilation.
const std::string& library_stamp() {
  static const std::string stamp = detail::compiler_image_identity(
      reinterpret_cast<const void*>(&loomc_compile_module));
  return stamp;
}

StatusCode map_code(loomc_status_code_t code) {
  switch (code) {
    case LOOMC_STATUS_INVALID_ARGUMENT:
    case LOOMC_STATUS_FAILED_PRECONDITION:
    case LOOMC_STATUS_OUT_OF_RANGE:
      return StatusCode::kInvalidArgument;
    case LOOMC_STATUS_RESOURCE_EXHAUSTED:
      return StatusCode::kOutOfMemory;
    case LOOMC_STATUS_UNIMPLEMENTED:
      return StatusCode::kUnimplemented;
    case LOOMC_STATUS_CANCELLED:
      return StatusCode::kCancelled;
    case LOOMC_STATUS_NOT_FOUND:
      return StatusCode::kNotFound;
    case LOOMC_STATUS_ALREADY_EXISTS:
      return StatusCode::kAlreadyExists;
    default:
      return StatusCode::kInternal;
  }
}

// A dropped non-OK loomc_status_t leaks its rich payload, and there are enough
// call sites here that "remember to free it" is not a plan.
class OwnedStatus {
 public:
  explicit OwnedStatus(loomc_status_t status) noexcept : status_(status) {}
  ~OwnedStatus() { loomc_status_free(status_); }

  OwnedStatus(const OwnedStatus&) = delete;
  OwnedStatus& operator=(const OwnedStatus&) = delete;

  [[nodiscard]] bool ok() const noexcept { return loomc_status_is_ok(status_); }

  // The status's own rendering, which at the release status mode carries the
  // source location and annotations loomc attached to it.
  [[nodiscard]] Status to_status(std::string_view what) const {
    loomc_host_size_t length = 0;
    loomc_status_format(status_, 0, nullptr, &length);
    std::string text(length, '\0');
    if (length > 0) {
      loomc_status_format(status_, length + 1, text.data(), &length);
      text.resize(length);
    }
    if (text.empty()) {
      text = loomc_status_code_string(loomc_status_code(status_));
    }
    return Status(map_code(loomc_status_code(status_)),
                  std::string(what) + ": " + text);
  }

 private:
  loomc_status_t status_;
};

// Releases a loomc handle on scope exit. One template rather than five guard
// structs, because every handle here has the same shape.
template <typename T, void (*Release)(T*)>
class Owned {
 public:
  Owned() = default;
  ~Owned() { Release(handle_); }

  Owned(const Owned&) = delete;
  Owned& operator=(const Owned&) = delete;

  T** out() noexcept { return &handle_; }
  [[nodiscard]] T* get() const noexcept { return handle_; }

 private:
  T* handle_ = nullptr;
};

using OwnedResult = Owned<loomc_result_t, loomc_result_release>;
using OwnedSource = Owned<loomc_source_t, loomc_source_release>;
using OwnedModule = Owned<loomc_module_t, loomc_module_release>;

// Renders the ERROR diagnostics of a failed result. Warnings and notes can
// appear on a successful result too, so severity gates what lands in a failure
// message and `loomc_result_succeeded` gates whether there is a failure at all.
std::string render_errors(const loomc_result_t* result) {
  std::string out;
  const loomc_host_size_t count = loomc_result_diagnostic_count(result);
  for (loomc_host_size_t i = 0; i < count; ++i) {
    const loomc_diagnostic_t* d = loomc_result_diagnostic_at(result, i);
    if (d == nullptr || d->severity != LOOMC_DIAGNOSTIC_SEVERITY_ERROR) continue;
    out += "\n  ";
    out.append(d->code.data, d->code.size);
    // The post-parse verifier reports with no location at all, so line 0 is a
    // normal answer and not a missing field.
    if (d->range.start_line != 0) {
      out += ':' + std::to_string(d->range.start_line) + ':' +
             std::to_string(d->range.start_column);
    }
    out += ": ";
    out.append(d->message.data, d->message.size);
  }
  return out;
}

Status result_status(const loomc_result_t* result, std::string_view stage,
                     std::string_view arch) {
  if (result == nullptr) {
    return LSE_ERROR(kInternal, "loom ", stage, " produced no result");
  }
  if (loomc_result_succeeded(result)) return OkStatus();
  if (loomc_result_state(result) == LOOMC_RESULT_STATE_CANCELLED) {
    return LSE_ERROR(kCancelled, "loom ", stage, " cancelled");
  }
  return LSE_ERROR(kCompileError, "loom ", stage, " failed (", arch, "):",
                   render_errors(result));
}

// `gfx942:sramecc+:xnack-` is a legal architecture string for HRX to report and
// the feature suffixes are not decoration: they select a different code object.
// The selector before the first colon is what Loom calls the target, and the
// suffixes become structured feature states rather than being dropped.
loomc_amdgpu_target_identity_t parse_identity(const std::string& arch,
                                              std::string* out_selector) {
  loomc_amdgpu_target_identity_t identity{};
  const std::size_t first = arch.find(':');
  *out_selector = arch.substr(0, first);
  identity.target =
      loomc_make_string_view(out_selector->data(), out_selector->size());
  std::size_t pos = first;
  while (pos != std::string::npos) {
    const std::size_t begin = pos + 1;
    const std::size_t next = arch.find(':', begin);
    const std::string feature = arch.substr(
        begin, next == std::string::npos ? std::string::npos : next - begin);
    pos = next;
    if (feature.size() < 2) continue;
    const char sign = feature.back();
    if (sign != '+' && sign != '-') continue;
    const loomc_amdgpu_target_feature_state_t state =
        sign == '+' ? LOOMC_AMDGPU_TARGET_FEATURE_ON
                    : LOOMC_AMDGPU_TARGET_FEATURE_OFF;
    const std::string name = feature.substr(0, feature.size() - 1);
    if (name == "sramecc") identity.amdhsa_features.sramecc = state;
    if (name == "xnack") identity.amdhsa_features.xnack = state;
  }
  return identity;
}

#endif  // LSE_HAVE_LOOMC

}  // namespace

#if LSE_HAVE_LOOMC

struct LoomcCompiler::State {
  // A workspace is mutable and not internally synchronized, and
  // loomc_compile_module rewrites its module in place, so one invocation at a
  // time holds the whole toolchain. compile() is const and may be called from
  // any thread the JIT runs on, which is what makes this a lock rather than a
  // comment.
  std::mutex mutex;
  bool started = false;
  loomc_target_environment_t* env = nullptr;
  loomc_context_t* context = nullptr;
  // Never trimmed between compiles. Releasing the module already returns its
  // arena blocks to the workspace, so the high-water mark is one kernel's
  // lowering; trimming would hand those blocks back to the allocator for the
  // next compile to fault in again.
  loomc_workspace_t* workspace = nullptr;
  loomc_compiler_t* compiler = nullptr;
  loomc_pass_program_t* pass_program = nullptr;
  // Profiles are per-target and reusable; rebuilding one per kernel is pure
  // cost on a path whose whole compile is ~1 ms.
  std::unordered_map<std::string, loomc_target_profile_t*> profiles;

  ~State() {
    for (auto& [arch, profile] : profiles) loomc_target_profile_release(profile);
    loomc_pass_program_release(pass_program);
    loomc_compiler_release(compiler);
    loomc_workspace_release(workspace);
    loomc_context_release(context);
    loomc_target_environment_release(env);
  }

  // Called under `mutex`.
  Status ensure() {
    if (started) {
      return pass_program != nullptr
                 ? OkStatus()
                 : LSE_ERROR(kCompileError, "loom toolchain failed to start");
    }
    started = true;

    {
      OwnedStatus st(
          loomc_target_environment_create_amdgpu(accounted_allocator(), &env));
      if (!st.ok()) return st.to_status("loomc_target_environment_create_amdgpu");
    }

    loomc_context_target_options_t target_options{};
    target_options.type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS;
    target_options.structure_size = sizeof(target_options);
    target_options.target_environment = env;
    loomc_context_options_t context_options{};
    context_options.type = LOOMC_STRUCTURE_TYPE_CONTEXT_OPTIONS;
    context_options.structure_size = sizeof(context_options);
    context_options.next = &target_options;
    {
      OwnedStatus st(loomc_context_create(&context_options,
                                          accounted_allocator(), &context));
      if (!st.ok()) return st.to_status("loomc_context_create");
    }
    {
      OwnedStatus st(
          loomc_workspace_create(nullptr, accounted_allocator(), &workspace));
      if (!st.ok()) return st.to_status("loomc_workspace_create");
    }
    {
      OwnedStatus st(loomc_compiler_create(context, nullptr,
                                           accounted_allocator(), &compiler));
      if (!st.ok()) return st.to_status("loomc_compiler_create");
    }

    loomc_target_pipeline_options_t pipeline_options{};
    pipeline_options.type = LOOMC_STRUCTURE_TYPE_TARGET_PIPELINE_OPTIONS;
    pipeline_options.structure_size = sizeof(pipeline_options);
    pipeline_options.identifier = loomc_make_cstring_view("lse-prepared-low");
    pipeline_options.kind = kOptions.pipeline;
    pipeline_options.control_flow_lowering = kOptions.control_flow;
    pipeline_options.source_to_low_max_errors = kOptions.source_to_low_max_errors;
    OwnedResult result;
    OwnedStatus st(loomc_pass_program_create_from_target_pipeline(
        context, &pipeline_options, accounted_allocator(), &pass_program,
        result.out()));
    if (!st.ok()) {
      return st.to_status("loomc_pass_program_create_from_target_pipeline");
    }
    return result_status(result.get(), "pipeline setup", "");
  }

  // Called under `mutex`. The returned profile stays owned by the cache.
  Result<loomc_target_profile_t*> profile_for(const std::string& arch) {
    const auto it = profiles.find(arch);
    if (it != profiles.end()) return it->second;

    std::string selector;
    loomc_amdgpu_profile_options_t profile_options{};
    profile_options.type = LOOMC_STRUCTURE_TYPE_AMDGPU_PROFILE_OPTIONS;
    profile_options.structure_size = sizeof(profile_options);
    profile_options.identifier = loomc_make_cstring_view("lse");
    profile_options.identity = parse_identity(arch, &selector);
    loomc_target_profile_t* profile = nullptr;
    OwnedStatus st(loomc_target_profile_create_amdgpu(
        env, &profile_options, accounted_allocator(), &profile));
    if (!st.ok()) {
      return st.to_status("no loom target profile for '" + arch + "'");
    }
    profiles.emplace(arch, profile);
    return profile;
  }
};

#else

struct LoomcCompiler::State {};

#endif  // LSE_HAVE_LOOMC

LoomcCompiler::LoomcCompiler() : state_(std::make_unique<State>()) {}
LoomcCompiler::~LoomcCompiler() = default;

std::vector<KernelCensus> LoomcCompiler::census(
    std::span<const std::byte> object) const {
  return read_code_object_census(object);
}

std::uint32_t LoomcCompiler::resource_metadata_version() const noexcept {
  return kCodeObjectResourceVersion;
}

std::vector<KernelResources> LoomcCompiler::resources(
    std::span<const std::byte> object) const {
  return read_code_object_resources(object);
}

bool LoomcCompiler::available() const {
#if LSE_HAVE_LOOMC
  return true;
#else
  return false;
#endif
}

std::string LoomcCompiler::identity() const {
#if LSE_HAVE_LOOMC
  // Package metadata is supplementary; the resident image and its contents
  // identify the compiler actually selected by the dynamic loader.
  std::string id = "loomc." LSE_LOOMC_VERSION " ";
  id += library_stamp();
  id += " pipeline=";
  id += pipeline_name(kOptions.pipeline);
  id += " control_flow=";
  id += control_flow_name(kOptions.control_flow);
  id += " max_errors=" + std::to_string(kOptions.source_to_low_max_errors);
  id += " compile_artifacts=" + std::to_string(kOptions.compile_artifacts);
  id += " runtime_globals=none";
  id += " format=";
  id += kArtifactFormat;
  id += " manifest=none";
  return id;
#else
  return "no-compiler";
#endif
}

Result<graph::CompiledKernel> LoomcCompiler::compile(
    std::string_view source, std::string_view arch) const {
  if (source.empty()) return LSE_ERROR(kInvalidArgument, "empty source");
  if (arch.empty()) return LSE_ERROR(kInvalidArgument, "no target arch");

#if LSE_HAVE_LOOMC
  State& s = *state_;
  const std::lock_guard<std::mutex> lock(s.mutex);
  LSE_RETURN_IF_ERROR(s.ensure());
  // What this compile costs the host, reported when traced, and always for a
  // compile that needed more than 64 MiB at once.
  struct CompileReport {
    std::size_t source_bytes;
    std::uint64_t held_before = g_compile_memory.live.load();
    std::uint64_t available_before = available_memory();
    CompileReport(std::size_t bytes) : source_bytes(bytes) {
      g_compile_memory.peak.store(held_before);
      g_compile_memory.largest.store(0);
    }
    ~CompileReport() {
      const auto peak = g_compile_memory.peak.load() - std::min(g_compile_memory.peak.load(), held_before);
      if (!trace_compile_memory() && peak < (std::uint64_t{64} << 20)) return;
      std::fprintf(stderr,
                   "lse: loomc compile of %zu bytes: peak %.1f MiB over %.1f MiB held, largest allocation "
                   "%.1f MiB; available %.1f -> %.1f MiB\n",
                   source_bytes, peak / 1048576.0, held_before / 1048576.0,
                   g_compile_memory.largest.load() / 1048576.0, available_before / 1048576.0,
                   available_memory() / 1048576.0);
    }
  } report(source.size());

  const std::string arch_key(arch);
  auto profile = s.profile_for(arch_key);
  LSE_RETURN_IF_ERROR(profile.status());

  loomc_source_options_t source_options{};
  source_options.type = LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS;
  source_options.structure_size = sizeof(source_options);
  source_options.format = LOOMC_SOURCE_FORMAT_TEXT;
  source_options.identifier = loomc_make_cstring_view("lse_kernel.loom");
  source_options.contents = loomc_make_byte_span(source.data(), source.size());
  // COPY, not BORROWED: the source handle is reachable from every diagnostic
  // range this call renders, and the compile contract does not promise the
  // caller's bytes outlive the handle.
  source_options.storage = LOOMC_SOURCE_STORAGE_COPY;
  OwnedSource loom_source;
  {
    OwnedStatus st(loomc_source_create(&source_options, accounted_allocator(),
                                       loom_source.out()));
    if (!st.ok()) return st.to_status("loomc_source_create");
  }

  OwnedModule module;
  {
    OwnedResult result;
    OwnedStatus st(loomc_module_deserialize_from_source(
        s.context, s.workspace, loom_source.get(), nullptr,
        accounted_allocator(), module.out(), result.out()));
    if (!st.ok()) return st.to_status("loomc_module_deserialize_from_source");
    LSE_RETURN_IF_ERROR(result_status(result.get(), "parse", arch));
  }

  // Every `kernel.def` in the module is specialized to this arch. Asking the
  // module rather than the caller is what keeps the entry name stated once, in
  // the source that defines it.
  loomc_module_function_query_options_t query{};
  query.type = LOOMC_STRUCTURE_TYPE_MODULE_FUNCTION_QUERY_OPTIONS;
  query.structure_size = sizeof(query);
  query.kind = LOOMC_MODULE_FUNCTION_KIND_KERNEL;
  std::vector<loomc_module_function_t> functions;
  {
    loomc_host_size_t count = 0;
    OwnedResult result;
    OwnedStatus st(loomc_module_query_functions(module.get(), &query,
                                                accounted_allocator(), 0,
                                                nullptr, &count, result.out()));
    if (!st.ok()) return st.to_status("loomc_module_query_functions");
    LSE_RETURN_IF_ERROR(result_status(result.get(), "kernel query", arch));
    if (count == 0) {
      return LSE_ERROR(kCompileError,
                       "loom source declares no kernel.def entry point");
    }
    functions.resize(count);
    OwnedResult filled;
    OwnedStatus st2(loomc_module_query_functions(
        module.get(), &query, accounted_allocator(), count, functions.data(),
        &count, filled.out()));
    if (!st2.ok()) return st2.to_status("loomc_module_query_functions");
    LSE_RETURN_IF_ERROR(result_status(filled.get(), "kernel query", arch));
    functions.resize(count);
  }

  // Symbol views borrow from the module and loomc_compile_module mutates it, so
  // anything needed after that call is copied out first.
  std::vector<std::string> symbols;
  symbols.reserve(functions.size());
  for (const loomc_module_function_t& f : functions) {
    symbols.emplace_back(f.symbol_name.data, f.symbol_name.size);
  }

  std::vector<loomc_target_specialization_t> specializations(symbols.size());
  for (std::size_t i = 0; i < symbols.size(); ++i) {
    specializations[i].function_symbol =
        loomc_make_string_view(symbols[i].data(), symbols[i].size());
    specializations[i].target_profile = *profile;
  }

  loomc_target_specialization_options_t spec_options{};
  spec_options.type = LOOMC_STRUCTURE_TYPE_TARGET_SPECIALIZATION_OPTIONS;
  spec_options.structure_size = sizeof(spec_options);
  spec_options.specializations = specializations.data();
  spec_options.specialization_count = specializations.size();

  loomc_compile_options_t compile_options{};
  compile_options.type = LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS;
  compile_options.structure_size = sizeof(compile_options);
  compile_options.next = &spec_options;
  compile_options.module_name =
      loomc_make_string_view(symbols.front().data(), symbols.front().size());
  compile_options.artifact_flags = kOptions.compile_artifacts;
  {
    OwnedResult result;
    OwnedStatus st(loomc_compile_module(s.compiler, s.workspace, s.pass_program,
                                        module.get(), &compile_options,
                                        accounted_allocator(), result.out()));
    if (!st.ok()) return st.to_status("loomc_compile_module");
    LSE_RETURN_IF_ERROR(result_status(result.get(), "lowering", arch));
  }

  // Emission consumes the function-version facts the compile above retained, so
  // nothing may compile this module again between the two calls. No runtime
  // globals are requested: LSE dispatches through the native HRX ABI and
  // reads neither the feedback channel nor a sanitizer config global.
  const std::string artifact_id = symbols.front() + ".hsaco";
  loomc_emit_options_t emit_options{};
  emit_options.type = LOOMC_STRUCTURE_TYPE_EMIT_OPTIONS;
  emit_options.structure_size = sizeof(emit_options);
  emit_options.artifact_format = loomc_make_cstring_view(kArtifactFormat);
  emit_options.identifier =
      loomc_make_string_view(artifact_id.data(), artifact_id.size());
  emit_options.artifact_flags = LOOMC_EMIT_ARTIFACT_FLAG_PRIMARY;

  std::vector<std::byte> code;
  {
    OwnedResult result;
    OwnedStatus st(loomc_emit_module(s.env, s.workspace, module.get(),
                                    &emit_options, accounted_allocator(),
                                    result.out()));
    if (!st.ok()) return st.to_status("loomc_emit_module");
    LSE_RETURN_IF_ERROR(result_status(result.get(), "emit", arch));

    const loomc_host_size_t count = loomc_result_artifact_count(result.get());
    for (loomc_host_size_t i = 0; i < count; ++i) {
      const loomc_artifact_t* a = loomc_result_artifact_at(result.get(), i);
      if (a == nullptr || a->kind != LOOMC_ARTIFACT_KIND_EXECUTABLE) continue;
      // Artifact bytes are borrowed from the result, so they are copied out
      // before it is released. An HSACO is one contiguous segment; a
      // segmented sequence is gathered in order.
      loomc_byte_span_t span{};
      if (loomc_byte_sequence_try_get_contiguous_span(a->contents, &span)) {
        const auto* p = reinterpret_cast<const std::byte*>(span.data);
        code.assign(p, p + span.data_length);
      } else {
        const loomc_byte_sequence_callback_t gather = {
            +[](void* user_data, loomc_byte_span_t segment) -> loomc_status_t {
              auto* out = static_cast<std::vector<std::byte>*>(user_data);
              const auto* p = reinterpret_cast<const std::byte*>(segment.data);
              out->insert(out->end(), p, p + segment.data_length);
              return loomc_ok_status();
            },
            &code,
        };
        OwnedStatus gathered(loomc_byte_sequence_enumerate(a->contents, gather));
        if (!gathered.ok()) {
          return gathered.to_status("loomc_byte_sequence_enumerate");
        }
      }
      break;
    }
  }
  if (code.empty()) {
    return LSE_ERROR(kCompileError, "loom emit produced no code object for ",
                     arch);
  }
  graph::CompiledKernel out;
  out.code = std::move(code);
  // loomc emits an AMDGPU HSACO, so the same note reader serves it. What comes
  // back is genuinely thinner than the HIP path's: measured over every loom
  // object this engine has cached, the register counts, segment sizes and the
  // kernel's own required workgroup size are all present and the two spill
  // counts are absent from every one. They therefore read as unknown, which is
  // what the interface exists to be able to say.
  out.resources = read_code_object_resources(out.code);
  out.census = read_code_object_census(out.code);
  return out;
#else
  return LSE_ERROR(kUnimplemented,
                   "no loom toolchain in this build; configure with an HRX "
                   "install so libloomc is found");
#endif
}

}  // namespace lse::backend
