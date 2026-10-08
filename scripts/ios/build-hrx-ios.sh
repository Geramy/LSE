#!/usr/bin/env bash
# Builds HRX (libhrx) and loomc as static archives for iOS arm64 devices, so the
# engine can be linked in-process into an iPad app. This only compiles and
# links on the Mac: nothing is signed, installed or launched.
#
# The iPad reaches the GPU through the same driver and transport as macOS, so
# the macOS coarse host adapter stays on and the AMDGPU HAL is the only GPU
# driver. HSA is linked statically (no dlopen of libhsa), and every CPU code
# path that would load or map executable code is left out: no executable
# loaders, no executable plugins, no local-sync driver. libhrx still links the
# local-task driver for hrx_cpu_initialize(); with no loaders it can move
# buffers but cannot load executables.
#
# Environment:
#   HRX_SOURCE           unpatched hrx-system tree at the pinned revision.
#                        Copied, never modified. Default: fetched into
#                        $LSE_IOS_WORK/deps/hrx.
#   HSA_HEADERS_SOURCE   hsa-runtime-headers checkout (hsa/ and aqlprofile-sdk/).
#                        Default: fetched into $LSE_IOS_WORK/deps/hsa-headers.
#   HSA_IOS_LIBRARY      static libhsa-runtime64.a built for iOS arm64.
#   FLATCC_SOURCE        flatcc source tree; fetched by CMake when missing.
#   LOOM_AMDGPU_ISA_XML  directory holding amdgpu_isa_*.xml; fetched when
#                        missing.
#   IOS_DEPLOYMENT_TARGET  minimum iOS version (default 26.0).
#   LSE_IOS_WORK         output root (default build/ios in this checkout).
#   LSE_BUILD_JOBS       parallel jobs (default: all cores).
#
# Outputs, under $LSE_IOS_WORK/hrx-build:
#   hrx-ios-libs.txt      absolute paths of every archive the final link
#                         needs, in link order (HSA last)
#   hrx-ios-includes.txt  include directories for hrx_runtime.h and loomc
#   ios-libs/             the merged libloomc_ios.a and libhrx_ios.a
#   ios-probe/            the link test and its exact link line
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
dev="$HOME/Documents/Development"
work="${LSE_IOS_WORK:-$root/build/ios}"
# The same hrx-system and hsa-runtime-headers pins as .github/scripts/build-macos.sh.
hrx_rev=631c0b7854b51a593b89761cc6dcae88f2ec3940
hsa_headers_rev=4285513114a70f7cf4830c89279c8cfa57b901bb
hrx_source="${HRX_SOURCE:-$work/deps/hrx}"
hsa_headers="${HSA_HEADERS_SOURCE:-$work/deps/hsa-headers}"
hsa_library="${HSA_IOS_LIBRARY:-$dev/mac_linuxgpu-worktrees/hsa-ios/build/hsa-ios/libhsa-runtime64.a}"
flatcc_source="${FLATCC_SOURCE:-}"
isa_xml="${LOOM_AMDGPU_ISA_XML:-}"
deployment="${IOS_DEPLOYMENT_TARGET:-26.0}"
jobs="${LSE_BUILD_JOBS:-$(sysctl -n hw.ncpu)}"
# LSE's HRX patch series, in the order build-macos.sh applies it, without the
# Linux multi-GPU device, the RDNA4 VRAM kernarg path and its macOS BAR-write
# bracket, and with the iOS static runtime last. The
# macOS coarse host adapter stays on: the iPad reaches the GPU through the
# same transport.
hrx_patches=(loom-concat-destination-reserved-once stream-queue-affinity
  macos-coarse-host-adapter kv-fragment-addressing gfx12-vopd-identical-source
  ios-static-runtime)

die() { echo "$*" >&2; exit 1; }

[[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]] ||
  die 'This build requires an Apple Silicon macOS host.'
for tool in cmake ninja python3 rsync git; do
  command -v "$tool" >/dev/null || die "missing required tool: $tool"
done
fetch() {
  local url="$1" rev="$2" path="$3"
  if [[ ! -d "$path/.git" ]]; then
    git init -q "$path"
    git -C "$path" remote add origin "$url"
    git -C "$path" fetch --depth 1 origin "$rev"
    git -C "$path" checkout -q --detach FETCH_HEAD
  fi
  [[ "$(git -C "$path" rev-parse HEAD)" == "$rev" &&
     -z "$(git -C "$path" status --porcelain)" ]] ||
    die "dependency revision or contents differ: $path"
}
[[ -n "${HRX_SOURCE:-}" ]] ||
  fetch https://github.com/ROCm/hrx-system.git "$hrx_rev" "$hrx_source"
[[ -n "${HSA_HEADERS_SOURCE:-}" ]] ||
  fetch https://github.com/iree-org/hsa-runtime-headers.git "$hsa_headers_rev" "$hsa_headers"
[[ -f "$hrx_source/libhrx/include/hrx_runtime.h" ]] ||
  die "HRX_SOURCE is not an hrx-system tree: $hrx_source"
[[ -f "$hsa_headers/include/hsa/hsa.h" ]] ||
  die "HSA_HEADERS_SOURCE has no include/hsa/hsa.h: $hsa_headers"
[[ -f "$hsa_library" ]] || die "HSA_IOS_LIBRARY not found: $hsa_library"
lipo -info "$hsa_library" 2>/dev/null | grep -q arm64 ||
  die "HSA_IOS_LIBRARY is not an arm64 archive: $hsa_library"
for patch in "${hrx_patches[@]}"; do
  [[ -f "$root/patches/hrx/$patch.patch" ]] || die "missing patches/hrx/$patch.patch"
done

# The host compiler is Xcode's Apple clang so the archives link into an Xcode
# app against the SDK libc++. The AMDGPU HAL configure step still insists on
# finding llvm-ar, llvm-link, ld.lld and llvm-objcopy, which Xcode does not
# ship; they only build device code, and the prebuilt device binaries mean
# nothing here invokes them. Appending (not prepending) keeps every host tool
# resolving to Xcode.
llvm="${AMDGPU_LLVM_BIN:-$(brew --prefix llvm@21)/bin}"
[[ "$("$llvm/llvm-config" --version)" == 21.1.8 ]] ||
  die 'The qualified AMDGPU toolchain is LLVM 21.1.8 (brew install llvm@21).'
lld="${AMDGPU_LLD:-$(brew --prefix lld@21)/bin/ld.lld}"
"$lld" --version | grep -q 'LLD 21.1.8' ||
  die 'The qualified AMDGPU linker is LLD 21.1.8 (brew install lld@21).'
export PATH="$PATH:$llvm:$(dirname "$lld")"

cc="$(xcrun --find clang)"
cxx="$(xcrun --find clang++)"
ar="$(xcrun --find ar)"
ranlib="$(xcrun --find ranlib)"
libtool="$(xcrun --find libtool)"
sdk="$(xcrun --sdk iphoneos --show-sdk-path)"
archive_args=("-DCMAKE_AR=$ar" "-DCMAKE_RANLIB=$ranlib"
  "-DCMAKE_C_COMPILER_AR=$ar" "-DCMAKE_C_COMPILER_RANLIB=$ranlib"
  "-DCMAKE_CXX_COMPILER_AR=$ar" "-DCMAKE_CXX_COMPILER_RANLIB=$ranlib")
# The HSA and aqlprofile headers reach the build through the hsa-runtime64
# package written below, so no hsa-runtime-headers FetchContent is involved.
fetch_args=()
if [[ -f "$flatcc_source/include/flatcc/flatcc.h" ]]; then
  fetch_args+=("-DFETCHCONTENT_SOURCE_DIR_FLATCC=$flatcc_source")
fi

src="$work/hrx-source"
host="$work/hrx-host-tools"
build="$work/hrx-build"
probe="$build/ios-probe"
libs="$build/ios-libs"

# 1. A fresh copy of the source with LSE's patch series on top. The copy gets its
# own git repository: inside this checkout, git apply would otherwise resolve
# paths against the enclosing repository and silently skip them.
echo "== copying $hrx_source"
[[ -L "$src" ]] && die "refusing generated source symlink: $src"
rm -rf "$src"
mkdir -p "$src"
rsync -a --exclude=/.git --exclude=/build/ --exclude=__pycache__ \
  "$hrx_source/" "$src/"
git -C "$src" init -q
for patch in "${hrx_patches[@]}"; do
  git -C "$src" apply --check "$root/patches/hrx/$patch.patch"
  git -C "$src" apply "$root/patches/hrx/$patch.patch"
done

# 2. Generators that run during the build (embed_data, flatcc) must run on the
# Mac, not on the device. Build them for macOS and point the iOS build at them.
echo "== building macOS host tools"
cmake -S "$src" -B "$host" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$cc" -DCMAKE_CXX_COMPILER="$cxx" "${archive_args[@]}" \
  -DIREE_BUILD_TESTS=OFF -DIREE_BUILD_BENCHMARKS=OFF \
  -DIREE_HAL_DRIVER_DEFAULTS=OFF -DIREE_HAL_DRIVER_LOCAL_SYNC=OFF \
  -DIREE_HAL_DRIVER_LOCAL_TASK=OFF -DIREE_HAL_DRIVER_NULL=OFF \
  -DIREE_HAL_EXECUTABLE_LOADER_DEFAULTS=OFF \
  -DIREE_HAL_EXECUTABLE_PLUGIN_DEFAULTS=OFF \
  -DLOOM_BUILD=OFF -DLIBHRX_BUILD=OFF -DIREE_ENABLE_LIBBACKTRACE=OFF \
  ${fetch_args[@]+"${fetch_args[@]}"} > "$work/hrx-host-tools-configure.log"
cmake --build "$host" --target iree-c-embed-data iree-flatcc-cli \
  --parallel "$jobs" > "$work/hrx-host-tools-build.log"
mkdir -p "$host/bin"
cp "$host/build_tools/iree-c-embed-data" "$host/tools/iree-flatcc-cli" \
  "$host/bin/"

# 3. hrx-system links static HSA through an hsa-runtime64 CMake package. The
# iOS runtime is a bare archive, so describe it as one.
hsa_package="$work/hsa-package/lib/cmake/hsa-runtime64"
mkdir -p "$hsa_package"
cat > "$hsa_package/hsa-runtime64-config.cmake" <<EOF
if(NOT TARGET hsa-runtime64::hsa-runtime64)
  add_library(hsa-runtime64::hsa-runtime64 STATIC IMPORTED GLOBAL)
  set_target_properties(hsa-runtime64::hsa-runtime64 PROPERTIES
    IMPORTED_LOCATION "$hsa_library"
    INTERFACE_INCLUDE_DIRECTORIES "$hsa_headers/include")
endif()
EOF

# Loom reads the AMDGPU ISA XML from inside its own binary directory. Seed it
# from an existing download when there is one rather than fetching ~90 MB.
mkdir -p "$build"
isa_dest="$build/_deps/amdgpu_isa_xml-src"
if [[ -f "$isa_xml/amdgpu_isa_rdna4.xml" ]]; then
  mkdir -p "$isa_dest"
  rsync -a "$isa_xml/" "$isa_dest/"
  fetch_args+=("-DFETCHCONTENT_SOURCE_DIR_LOOM_AMDGPU_ISA_XML=$isa_dest")
fi

# 4. The link probe. hrx's exports come from its macOS export list; loomc's
# from the public headers, minus the SPIR-V and LLVM IR targets that are not
# built here. The probe references every one of them, so linking it proves the
# archive set is complete. It is also how the archive closure is taken from
# CMake below instead of being maintained by hand.
mkdir -p "$probe"
sed -e 's/^_//' -e '/^[[:space:]]*$/d' \
  "$src/libhrx/cmake/hrx_exports_macos.txt" > "$probe/hrx-symbols.txt"
python3 - "$src/loom/binding/c/include/loomc" > "$probe/loomc-symbols.txt" <<'PY'
import pathlib, re, sys
root = pathlib.Path(sys.argv[1])
names = set()
for header in sorted(root.rglob('*.h')):
    if re.match(r'target/(spirv|llvmir)', header.relative_to(root).as_posix()):
        continue
    text = re.sub(r'//[^\n]*|/\*.*?\*/', '', header.read_text(), flags=re.S)
    for m in re.finditer(r'LOOMC_API_EXPORT\b[^;{]*?\b(loomc_\w+)\s*\(', text):
        names.add(m.group(1))
print('\n'.join(sorted(names)))
PY
{
  printf '%s\n' '// Generated by scripts/ios/build-hrx-ios.sh.' \
    '#include <stddef.h>'
  sed 's/.*/extern void &(void);/' "$probe/hrx-symbols.txt" "$probe/loomc-symbols.txt"
  printf '%s\n' 'static void (*const k_symbols[])(void) = {'
  sed 's/.*/    \&&,/' "$probe/hrx-symbols.txt" "$probe/loomc-symbols.txt"
  printf '%s\n' '};' \
    'void (*const *hrx_ios_api_symbols(size_t* count))(void) {' \
    '  *count = sizeof(k_symbols) / sizeof(k_symbols[0]);' \
    '  return k_symbols;' '}'
} > "$probe/api_symbols.c"
cat > "$probe/link_test.c" <<'EOF'
// Generated by scripts/ios/build-hrx-ios.sh. Links, never runs.
#include <stdint.h>
#include <stdio.h>

#include "hrx_runtime.h"
#include "loomc/loomc.h"
#include "loomc/target/amdgpu.h"

void (*const* hrx_ios_api_symbols(size_t* count))(void);

int main(void) {
  hrx_status_t status = hrx_gpu_initialize(0);
  int device_count = 0;
  if (hrx_status_is_ok(status)) {
    hrx_status_ignore(hrx_gpu_device_count(&device_count));
    hrx_status_ignore(hrx_gpu_shutdown());
  } else {
    hrx_status_ignore(status);
  }

  loomc_context_t* context = NULL;
  loomc_status_t loomc_status =
      loomc_context_create(NULL, loomc_allocator_system(), &context);
  if (loomc_status_is_ok(loomc_status)) {
    loomc_context_release(context);
  } else {
    loomc_status_free(loomc_status);
  }

  size_t count = 0;
  void (*const* symbols)(void) = hrx_ios_api_symbols(&count);
  uintptr_t digest = (uintptr_t)&loomc_target_profile_create_amdgpu;
  for (size_t i = 0; i < count; ++i) digest ^= (uintptr_t)symbols[i];
  printf("%d %zu %lx\n", device_count, count, (unsigned long)digest);
  return 0;
}
EOF
cat > "$probe/link-probe.cmake" <<'EOF'
# Included through IREE_CMAKE_TRY_FILE once every repository target exists.
add_executable(hrx_ios_link_probe
  "${CMAKE_BINARY_DIR}/ios-probe/link_test.c"
  "${CMAKE_BINARY_DIR}/ios-probe/api_symbols.c")
target_link_libraries(hrx_ios_link_probe PRIVATE
  libhrx_src_libhrx_hrx_static
  loom_binding_c_loomc
  loom_binding_c_target_amdgpu_amdgpu
  loom_binding_c_target_amdgpu_iree_hal_iree_hal
  loom_binding_c_target_iree_hal_iree_hal
  "-framework IOKit" "-framework CoreFoundation" c++)
EOF

# 5. Configure and build for iOS. Configure from a clean cache so options from
# an earlier run cannot linger; objects are kept, so a rerun is incremental.
echo "== configuring iOS arm64 (deployment target $deployment)"
rm -f "$build/CMakeCache.txt"
cmake -S "$src" -B "$build" -G Ninja \
  -DCMAKE_SYSTEM_NAME=iOS -DCMAKE_OSX_SYSROOT=iphoneos \
  -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_OSX_DEPLOYMENT_TARGET="$deployment" \
  -DCMAKE_C_COMPILER="$cc" -DCMAKE_CXX_COMPILER="$cxx" "${archive_args[@]}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_FLAGS=-DIREE_HAL_AMDGPU_MACOS_COARSE_HOST_ADAPTER=1 \
  -DBUILD_SHARED_LIBS=OFF -DIREE_HOST_BIN_DIR="$host/bin" \
  -DIREE_BUILD_TESTS=OFF -DIREE_BUILD_BENCHMARKS=OFF -DIREE_BUILD_SAMPLES=OFF \
  -DIREE_ENABLE_LIBBACKTRACE=OFF \
  -DIREE_HAL_DRIVER_DEFAULTS=OFF -DIREE_HAL_DRIVER_AMDGPU=ON \
  -DIREE_HAL_AMDGPU_TARGETS=gfx1201 \
  -DIREE_HAL_AMDGPU_DEVICE_BINARY_BUILD_MODE=prebuilt \
  -DIREE_HAL_AMDGPU_LIBHSA_STATIC=ON -DIREE_ROCM_DEPENDENCY_MODE=auto \
  -Dhsa-runtime64_DIR="$hsa_package" \
  -DIREE_HAL_DRIVER_LOCAL_SYNC=OFF -DIREE_HAL_DRIVER_LOCAL_TASK=ON \
  -DIREE_HAL_DRIVER_NULL=OFF \
  -DIREE_HAL_EXECUTABLE_LOADER_DEFAULTS=OFF \
  -DIREE_HAL_EXECUTABLE_LOADER_EMBEDDED_ELF=OFF \
  -DIREE_HAL_EXECUTABLE_LOADER_SYSTEM_LIBRARY=OFF \
  -DIREE_HAL_EXECUTABLE_LOADER_VMVX_MODULE=OFF \
  -DIREE_HAL_EXECUTABLE_PLUGIN_DEFAULTS=OFF \
  -DIREE_HAL_EXECUTABLE_PLUGIN_EMBEDDED_ELF=OFF \
  -DIREE_HAL_EXECUTABLE_PLUGIN_SYSTEM_LIBRARY=OFF \
  -DLOOM_BUILD=ON -DLOOM_TARGET_DEFAULTS=OFF \
  -DLOOM_TARGET_AMDGPU=ON -DLOOM_TARGET_AMDGPU_TARGETS=gfx1201 \
  -DLOOM_TARGET_LLVMIR=OFF -DLOOM_TARGET_IREE_VM=OFF \
  -DLOOM_TARGET_SPIRV=OFF -DLOOM_TARGET_X86=OFF -DLOOM_TARGET_WASM=OFF \
  -DLOOM_TARGET_ARCH_LLVMIR=ON -DLOOM_EXECUTE_IREE_VM=OFF \
  -DLIBHRX_BUILD_HIP_BINDING=OFF -DLIBHRX_BUILD_CUDA_BINDING=OFF \
  -DLIBHRX_BUILD_CTS=OFF -DLIBHRX_BUILD_PASSTHROUGH=OFF -DHRX_ENABLE_ZSTD=OFF \
  -DIREE_CMAKE_TRY_FILE="$probe/link-probe.cmake" \
  ${fetch_args[@]+"${fetch_args[@]}"} > "$work/hrx-ios-configure.log"
# LOOM_TARGET_ARCH_LLVMIR: loomc's op registry links the LLVM IR target facts
# even with that target off; only the slice is built, not its emitter.
# LOOM_EXECUTE_IREE_VM: only loom's command-line tools use it.

echo "== building iOS archives"
cmake --build "$build" --target hrx_ios_link_probe --parallel "$jobs" \
  > "$work/hrx-ios-build.log"

# 6. The archive closure, in CMake's own link order, read off the probe link.
link_command="$(cd "$build" && ninja -t commands hrx_ios_link_probe | tail -n 1)"
inputs="$probe/cmake-link-inputs.txt"
python3 - "$build" "$hsa_library" "$link_command" > "$inputs" <<'PY'
import os, shlex, sys
build, hsa, command = sys.argv[1], os.path.realpath(sys.argv[2]), sys.argv[3]
seen = set()
for arg in shlex.split(command):
    if arg in ('&&', ':') or not arg.endswith(('.a', '.o')):
        continue
    if arg.endswith(('link_test.c.o', 'api_symbols.c.o')):
        continue
    path = os.path.realpath(os.path.join(build, arg))
    if path == hsa or path in seen:
        continue
    seen.add(path)
    print(path)
PY
loom_inputs=()
hrx_inputs=()
while IFS= read -r input; do
  case "$input" in
    "$build"/loom/*) loom_inputs+=("$input") ;;
    *) hrx_inputs+=("$input") ;;
  esac
done < "$inputs"
(( ${#loom_inputs[@]} && ${#hrx_inputs[@]} )) ||
  die "unexpected link closure; see $inputs"

# One archive per component: loomc (Loom and its C binding, including the
# ALWAYSLINK binding objects, which carry no static initialisers and are all
# reached through the public API) and hrx (libhrx with the IREE runtime, the
# AMDGPU HAL and flatcc). loomc depends on the IREE base libraries inside
# libhrx_ios.a, so it comes first.
echo "== merging archives"
rm -rf "$libs"
mkdir -p "$libs"
"$libtool" -static -no_warning_for_no_symbols -o "$libs/libloomc_ios.a" \
  "${loom_inputs[@]}" 2> "$libs/libtool-loomc.log"
"$libtool" -static -no_warning_for_no_symbols -o "$libs/libhrx_ios.a" \
  "${hrx_inputs[@]}" 2> "$libs/libtool-hrx.log"

printf '%s\n' "$libs/libloomc_ios.a" "$libs/libhrx_ios.a" \
  "$(cd "$(dirname "$hsa_library")" && pwd)/$(basename "$hsa_library")" \
  > "$build/hrx-ios-libs.txt"
printf '%s\n' "$src/libhrx/include" "$src/loom/binding/c/include" \
  > "$build/hrx-ios-includes.txt"

# 7. Link the test against exactly what the consumer will use.
echo "== link test"
target="arm64-apple-ios$deployment"
include_args=()
while IFS= read -r dir; do include_args+=("-I$dir"); done < "$build/hrx-ios-includes.txt"
archives=()
while IFS= read -r lib; do archives+=("$lib"); done < "$build/hrx-ios-libs.txt"
for unit in link_test api_symbols; do
  "$cc" -target "$target" -isysroot "$sdk" -O2 "${include_args[@]}" \
    -c "$probe/$unit.c" -o "$probe/$unit.o"
done
link=("$cc" -target "$target" -isysroot "$sdk" -Wl,-dead_strip
  "$probe/link_test.o" "$probe/api_symbols.o" "${archives[@]}"
  -framework IOKit -framework CoreFoundation -lc++
  -o "$probe/hrx_ios_link_test")
printf '%q ' "${link[@]}" > "$probe/link-line.txt"
echo >> "$probe/link-line.txt"
"${link[@]}"

# Nothing in the result may load code at run time.
imports="$(nm -u "$probe/hrx_ios_link_test")"
if grep -Eq '^_(dlopen|dlsym|NSAddImage|NSCreateObjectFileImageFromMemory)$' <<< "$imports"; then
  die "the linked result imports a dynamic loader entry point; see nm -u $probe/hrx_ios_link_test"
fi
if grep -Eq '^_(pthread_jit_write_protect_np|sys_icache_invalidate)$' <<< "$imports"; then
  die "the linked result imports a JIT entry point; see nm -u $probe/hrx_ios_link_test"
fi
if otool -L "$probe/hrx_ios_link_test" | tail -n +2 |
    grep -Ev '^[[:space:]]+/(usr/lib|System/Library)/' | grep -q .; then
  die "the linked result needs a dylib outside the system; see otool -L $probe/hrx_ios_link_test"
fi

echo
echo "Archives (link order):"; sed 's/^/  /' "$build/hrx-ios-libs.txt"
du -ch "${archives[@]}" | tail -n 1 | awk '{print "  total size " $1}'
echo "Include directories:"; sed 's/^/  /' "$build/hrx-ios-includes.txt"
echo "Link line: $probe/link-line.txt"
