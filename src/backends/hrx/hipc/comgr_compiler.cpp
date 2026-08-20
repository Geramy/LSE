#include "lse/backends/hrx/hipc/comgr_compiler.hpp"

#include "lse/backends/hrx/code_object.hpp"

#include <cstdlib>
#include <string>
#include <vector>

#if LSE_HAVE_COMGR
#include <amd_comgr/amd_comgr.h>
#endif

namespace lse::backend {

namespace {

#if LSE_HAVE_COMGR

struct DataGuard {
  amd_comgr_data_t handle{};
  bool live = false;
  ~DataGuard() {
    if (live) amd_comgr_release_data(handle);
  }
};

struct DataSetGuard {
  amd_comgr_data_set_t handle{};
  bool live = false;
  ~DataSetGuard() {
    if (live) amd_comgr_destroy_data_set(handle);
  }
};

struct ActionGuard {
  amd_comgr_action_info_t handle{};
  bool live = false;
  ~ActionGuard() {
    if (live) amd_comgr_destroy_action_info(handle);
  }
};

Status check(amd_comgr_status_t s, const char* what) {
  if (s == AMD_COMGR_STATUS_SUCCESS) return OkStatus();
  const char* text = nullptr;
  amd_comgr_status_string(s, &text);
  return LSE_ERROR(kCompileError, what, ": ",
                   text ? text : "unknown comgr error");
}

// Concatenates every log entry in a data set, so a compile failure reports the
// compiler's own diagnostics rather than just a status code.
std::string collect_logs(amd_comgr_data_set_t set) {
  std::string out;
  std::size_t count = 0;
  if (amd_comgr_action_data_count(set, AMD_COMGR_DATA_KIND_LOG, &count) !=
      AMD_COMGR_STATUS_SUCCESS) {
    return out;
  }
  for (std::size_t i = 0; i < count; ++i) {
    amd_comgr_data_t entry{};
    if (amd_comgr_action_data_get_data(set, AMD_COMGR_DATA_KIND_LOG, i, &entry) !=
        AMD_COMGR_STATUS_SUCCESS) {
      continue;
    }
    std::size_t size = 0;
    if (amd_comgr_get_data(entry, &size, nullptr) == AMD_COMGR_STATUS_SUCCESS &&
        size > 0) {
      std::string text(size, '\0');
      if (amd_comgr_get_data(entry, &size, text.data()) ==
          AMD_COMGR_STATUS_SUCCESS) {
        out += text;
      }
    }
    amd_comgr_release_data(entry);
  }
  return out;
}

#ifndef LSE_ROCM_INCLUDE_DIR
#define LSE_ROCM_INCLUDE_DIR ""
#endif
#ifndef LSE_CLANG_RESOURCE_DIR
#define LSE_CLANG_RESOURCE_DIR ""
#endif

std::string rocm_include_dir() {
  if (const char* env = std::getenv("LSE_ROCM_INCLUDE")) return env;
  return LSE_ROCM_INCLUDE_DIR;
}

std::string clang_resource_dir() {
  if (const char* env = std::getenv("LSE_CLANG_RESOURCE")) return env;
  return LSE_CLANG_RESOURCE_DIR;
}

// <root>/include -> <root>, which is what --rocm-path expects.
std::string rocm_root(const std::string& include_dir) {
  const std::size_t pos = include_dir.rfind("/include");
  return pos == std::string::npos ? include_dir : include_dir.substr(0, pos);
}

// Options for the source -> bitcode action. Shared with identity() so the JIT
// cache key cannot drift from what was actually passed to the frontend.
std::vector<std::string> frontend_options() {
  std::vector<std::string> opts = {"-O3", "-x", "hip"};
  const std::string include_dir = rocm_include_dir();
  if (!include_dir.empty()) {
    opts.push_back("-isystem");
    opts.push_back(include_dir);
    opts.push_back("--rocm-path=" + rocm_root(include_dir));
  }
  const std::string resource_dir = clang_resource_dir();
  if (!resource_dir.empty()) {
    opts.push_back("-isystem");
    opts.push_back(resource_dir);
  }
  return opts;
}

// The codegen action does not inherit the compile action's options; without an
// explicit -O3 the backend runs at its default and every kernel ships
// unscheduled ISA (measured 13 GB/s vs 227 GB/s on the same GEMV source).
// Not constexpr: amd_comgr_action_info_set_option_list takes `const char**`,
// so const elements would not convert.
const char* kBackendOptions[] = {"-O3"};

Result<graph::CompiledKernel> compile_comgr(std::string_view source,
                                            std::string_view arch) {
  const std::string target = "amdgcn-amd-amdhsa--" + std::string(arch);

  DataSetGuard input;
  LSE_RETURN_IF_ERROR(check(amd_comgr_create_data_set(&input.handle),
                            "amd_comgr_create_data_set"));
  input.live = true;

  DataGuard src;
  LSE_RETURN_IF_ERROR(check(
      amd_comgr_create_data(AMD_COMGR_DATA_KIND_SOURCE, &src.handle),
      "amd_comgr_create_data"));
  src.live = true;
  LSE_RETURN_IF_ERROR(check(
      amd_comgr_set_data(src.handle, source.size(), source.data()),
      "amd_comgr_set_data"));
  LSE_RETURN_IF_ERROR(check(amd_comgr_set_data_name(src.handle, "lse_kernel.hip"),
                            "amd_comgr_set_data_name"));
  LSE_RETURN_IF_ERROR(check(amd_comgr_data_set_add(input.handle, src.handle),
                            "amd_comgr_data_set_add"));

  ActionGuard action;
  LSE_RETURN_IF_ERROR(check(amd_comgr_create_action_info(&action.handle),
                            "amd_comgr_create_action_info"));
  action.live = true;
  LSE_RETURN_IF_ERROR(check(
      amd_comgr_action_info_set_language(action.handle, AMD_COMGR_LANGUAGE_HIP),
      "amd_comgr_action_info_set_language"));
  LSE_RETURN_IF_ERROR(check(
      amd_comgr_action_info_set_isa_name(action.handle, target.c_str()),
      "amd_comgr_action_info_set_isa_name"));
  // Without this the log data set comes back empty and a failure reports
  // nothing useful.
  LSE_RETURN_IF_ERROR(check(
      amd_comgr_action_info_set_logging(action.handle, true),
      "amd_comgr_action_info_set_logging"));

  // Kept alive for the duration of the option list.
  const std::vector<std::string> owned = frontend_options();
  std::vector<const char*> options;
  options.reserve(owned.size());
  for (const std::string& o : owned) options.push_back(o.c_str());
  LSE_RETURN_IF_ERROR(check(
      amd_comgr_action_info_set_option_list(action.handle, options.data(),
                                            options.size()),
      "amd_comgr_action_info_set_option_list"));

  DataSetGuard output;
  LSE_RETURN_IF_ERROR(check(amd_comgr_create_data_set(&output.handle),
                            "amd_comgr_create_data_set(out)"));
  output.live = true;

  const amd_comgr_status_t compiled = amd_comgr_do_action(
      AMD_COMGR_ACTION_COMPILE_SOURCE_WITH_DEVICE_LIBS_TO_BC, action.handle,
      input.handle, output.handle);
  if (compiled != AMD_COMGR_STATUS_SUCCESS) {
    return LSE_ERROR(kCompileError, "HIP source failed to compile:\n",
                     collect_logs(output.handle));
  }

  // Codegen and link must not run in HIP language mode: the source-language
  // action info makes clang treat amdgcn as a host target.
  ActionGuard backend_action;
  LSE_RETURN_IF_ERROR(check(amd_comgr_create_action_info(&backend_action.handle),
                            "amd_comgr_create_action_info(backend)"));
  backend_action.live = true;
  LSE_RETURN_IF_ERROR(check(
      amd_comgr_action_info_set_isa_name(backend_action.handle, target.c_str()),
      "amd_comgr_action_info_set_isa_name(backend)"));
  LSE_RETURN_IF_ERROR(check(
      amd_comgr_action_info_set_logging(backend_action.handle, true),
      "amd_comgr_action_info_set_logging(backend)"));
  LSE_RETURN_IF_ERROR(check(
      amd_comgr_action_info_set_option_list(
          backend_action.handle, kBackendOptions, std::size(kBackendOptions)),
      "amd_comgr_action_info_set_option_list(backend)"));

  DataSetGuard relocatable;
  LSE_RETURN_IF_ERROR(check(amd_comgr_create_data_set(&relocatable.handle),
                            "amd_comgr_create_data_set(reloc)"));
  relocatable.live = true;
  const amd_comgr_status_t assembled = amd_comgr_do_action(
      AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE, backend_action.handle,
      output.handle, relocatable.handle);
  if (assembled != AMD_COMGR_STATUS_SUCCESS) {
    return LSE_ERROR(kCompileError, "codegen to relocatable failed:\n",
                     collect_logs(relocatable.handle));
  }

  DataSetGuard executable;
  LSE_RETURN_IF_ERROR(check(amd_comgr_create_data_set(&executable.handle),
                            "amd_comgr_create_data_set(exe)"));
  executable.live = true;
  const amd_comgr_status_t linked = amd_comgr_do_action(
      AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE, backend_action.handle,
      relocatable.handle, executable.handle);
  if (linked != AMD_COMGR_STATUS_SUCCESS) {
    return LSE_ERROR(kCompileError, "link to executable failed:\n",
                     collect_logs(executable.handle));
  }

  std::size_t count = 0;
  LSE_RETURN_IF_ERROR(check(
      amd_comgr_action_data_count(executable.handle,
                                  AMD_COMGR_DATA_KIND_EXECUTABLE, &count),
      "amd_comgr_action_data_count"));
  if (count == 0) {
    return LSE_ERROR(kCompileError, "compilation produced no code object");
  }

  amd_comgr_data_t code{};
  LSE_RETURN_IF_ERROR(check(
      amd_comgr_action_data_get_data(executable.handle,
                                     AMD_COMGR_DATA_KIND_EXECUTABLE, 0, &code),
      "amd_comgr_action_data_get_data"));

  std::size_t size = 0;
  Status s = check(amd_comgr_get_data(code, &size, nullptr),
                   "amd_comgr_get_data(size)");
  if (!s.ok()) {
    amd_comgr_release_data(code);
    return s;
  }
  graph::CompiledKernel out;
  out.code.resize(size);
  s = check(
      amd_comgr_get_data(code, &size, reinterpret_cast<char*>(out.code.data())),
      "amd_comgr_get_data(bytes)");
  amd_comgr_release_data(code);
  if (!s.ok()) return s;

  // The note is read from the bytes rather than from the still-live `code`
  // handle so both compilers in this backend go through one reader; the walk
  // is microseconds against a ~350 ms compile.
  out.resources = read_code_object_resources(out.code);
  out.census = read_code_object_census(out.code);
  return out;
}

#endif  // LSE_HAVE_COMGR

}  // namespace

std::vector<KernelCensus> ComgrCompiler::census(
    std::span<const std::byte> object) const {
  return read_code_object_census(object);
}

bool ComgrCompiler::available() const {
#if LSE_HAVE_COMGR
  return true;
#else
  return false;
#endif
}

std::string ComgrCompiler::identity() const {
#if LSE_HAVE_COMGR
  std::size_t major = 0;
  std::size_t minor = 0;
  amd_comgr_get_version(&major, &minor);
  // The option lists come from the same place the actions read them, so a
  // flag edit reaches the cache key without anyone remembering to do anything.
  // What this does NOT see is an in-place ROCm patch that leaves the comgr
  // version and the install paths untouched; purge the cache for those.
  std::string id = "comgr." + std::to_string(major) + "." + std::to_string(minor);
  for (const std::string& opt : frontend_options()) id += ' ' + opt;
  id += " |";
  for (const char* opt : kBackendOptions) {
    id += ' ';
    id += opt;
  }
  return id;
#else
  return "no-compiler";
#endif
}

Result<graph::CompiledKernel> ComgrCompiler::compile(
    std::string_view source, std::string_view arch) const {
  if (source.empty()) return LSE_ERROR(kInvalidArgument, "empty source");
  if (arch.empty()) return LSE_ERROR(kInvalidArgument, "no target arch");

#if LSE_HAVE_COMGR
  return compile_comgr(source, arch);
#else
  return LSE_ERROR(kUnimplemented,
                   "no compiler toolchain in this build; configure with ROCm "
                   "so amd_comgr is found");
#endif
}

}  // namespace lse::backend
