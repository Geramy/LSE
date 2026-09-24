#!/usr/bin/env bash
# Hosted arm64 build. No GPU access or system-extension installation.
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
work="${LSE_MACOS_WORK:-$root/build/macos}"
[[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]] || {
  echo 'This build requires an Apple Silicon macOS host.' >&2; exit 1;
}
llvm="${AMDGPU_LLVM_BIN:-$(brew --prefix llvm@21)/bin}"
[[ "$("$llvm/llvm-config" --version)" == 21.1.8 ]] || {
  echo 'The qualified host compiler is LLVM 21.1.8 (brew install llvm@21).' >&2; exit 1;
}
lld="${AMDGPU_LLD:-$(brew --prefix lld@21)/bin/ld.lld}"
"$lld" --version | grep -q 'LLD 21.1.8' || {
  echo 'The qualified AMDGPU linker is LLD 21.1.8 (brew install lld@21).' >&2; exit 1;
}
PATH="$llvm:$(dirname "$lld"):$PATH"
export PATH
export MACOSX_DEPLOYMENT_TARGET=15.0
# LLVM's PATH must not select host archive tools implicitly: Darwin ld expects
# Darwin archive indices, including for translation units with no symbols.
# Resolve the tools from the selected Xcode, independently of Homebrew PATH.
darwin_ar="$(xcrun --find ar)"
darwin_ranlib="$(xcrun --find ranlib)"
darwin_archive_args=("-DCMAKE_AR=$darwin_ar" "-DCMAKE_RANLIB=$darwin_ranlib"
  "-DCMAKE_C_COMPILER_AR=$darwin_ar" "-DCMAKE_C_COMPILER_RANLIB=$darwin_ranlib"
  "-DCMAKE_CXX_COMPILER_AR=$darwin_ar" "-DCMAKE_CXX_COMPILER_RANLIB=$darwin_ranlib")
command -v cargo >/dev/null
command -v ninja >/dev/null
mkdir -p "$work/deps"
fetch() {
  local url="$1" rev="$2" path="$3"
  if [[ ! -d "$path/.git" ]]; then
    git init -q "$path"
    git -C "$path" remote add origin "$url"
    git -C "$path" fetch --depth 1 origin "$rev"
    git -C "$path" checkout --detach FETCH_HEAD
  fi
  [[ "$(git -C "$path" rev-parse HEAD)" == "$rev" ]] || {
    echo "Dependency revision differs: $path" >&2; exit 1;
  }
  [[ -z "$(git -C "$path" status --porcelain)" ]] || {
    echo "Dependency is modified: $path" >&2; exit 1;
  }
}
mac_rev=7cbbdb59c80be82d5f6e6624a1b15743200176b1
hrx_rev=5927b0e0fafdefb5c8b41aa71bca8fd28791ad7c
hsa_headers_rev=4285513114a70f7cf4830c89279c8cfa57b901bb
fetch https://github.com/lemonade-sdk/mac-amdgpu.git "$mac_rev" "$work/deps/mac-amdgpu"
fetch https://github.com/ROCm/hrx-system.git "$hrx_rev" "$work/deps/hrx"
fetch https://github.com/iree-org/hsa-runtime-headers.git "$hsa_headers_rev" "$work/deps/hsa-headers"
fetch https://github.com/crusoecloud/fastokens.git 7973014e4f3a6028ac48f305704eacd64d0b4ef6 "$work/deps/fastokens"
# Never build an old engine snapshot from the driver repository. Adapt this
# workflow's exact LSE commit; reject stale hunks instead of silently skipping.
# Recreate only the generated source trees so a local rerun cannot retain
# yesterday's added files after re-extracting the current commit.
python3 - "$work" <<'PY_CLEAN'
from pathlib import Path
import shutil, sys
for name in ('source', 'hrx-source'):
    path = Path(sys.argv[1]) / name
    if path.is_symlink():
        raise SystemExit(f'Refusing generated source symlink: {path}')
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
PY_CLEAN
git -C "$root" archive HEAD | tar -x -C "$work/source"
mkdir -p "$work/source/reference/fastokens"
git -C "$work/deps/fastokens" archive HEAD | tar -x -C "$work/source/reference/fastokens"
git -C "$work/deps/hrx" archive HEAD | tar -x -C "$work/hrx-source"
# Isolate the extracted trees from the enclosing repository. Otherwise git
# apply treats paths as outside the current prefix and silently skips them.
git -C "$work/source" init -q
git -C "$work/hrx-source" init -q
git -C "$work/source" apply --check "$root/.github/patches/macos-portability.patch"
git -C "$work/source" apply "$root/.github/patches/macos-portability.patch"
git -C "$work/hrx-source" apply --check "$work/deps/mac-amdgpu/patches/hrx/macos-coarse-host-adapter.patch"
git -C "$work/hrx-source" apply "$work/deps/mac-amdgpu/patches/hrx/macos-coarse-host-adapter.patch"
# The committed Cargo.lock determines transitive tokenizer dependencies.
python3 - "$work/source/cmake/LSERust.cmake" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1]); s = p.read_text()
s = s.replace('COMMAND ${LSE_CARGO} rustc --release', 'COMMAND ${LSE_CARGO} rustc --locked --release')
s = s.replace('COMMAND ${LSE_CARGO} build --release', 'COMMAND ${LSE_CARGO} build --locked --release')
p.write_text(s)
PY
jobs="${LSE_BUILD_JOBS:-3}"
cmake -S "$work/hrx-source" -B "$work/hrx-build" -G Ninja \
  "${darwin_archive_args[@]}" -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_DEPLOYMENT_TARGET=15.0 -DCMAKE_C_COMPILER="$llvm/clang" -DCMAKE_CXX_COMPILER="$llvm/clang++" \
  -DCMAKE_C_FLAGS=-DIREE_HAL_AMDGPU_MACOS_COARSE_HOST_ADAPTER=1 \
  -DIREE_BUILD_TESTS=OFF -DIREE_BUILD_BENCHMARKS=OFF \
  -DIREE_CLANG_BINARY="$llvm/clang" -DIREE_LLVM_LINK_BINARY="$llvm/llvm-link" -DIREE_LLD_BINARY="$lld" \
  -DIREE_HAL_DRIVER_DEFAULTS=OFF -DIREE_HAL_DRIVER_AMDGPU=ON \
  -DIREE_HAL_AMDGPU_TARGETS=gfx1201 -DLOOM_BUILD=ON \
  -DLOOM_TARGET_DEFAULTS=OFF -DLOOM_TARGET_AMDGPU=ON -DLOOM_TARGET_AMDGPU_TARGETS=gfx1201 \
  -DLOOM_TARGET_LLVMIR=ON -DLOOM_TARGET_IREE_VM=ON -DLOOM_TARGET_SPIRV=ON -DLOOM_TARGET_X86=ON \
  -DLIBHRX_BUILD_HIP_BINDING=OFF -DLIBHRX_BUILD_CTS=OFF -DIREE_ENABLE_LIBBACKTRACE=OFF \
  -DFETCHCONTENT_SOURCE_DIR_HSA_RUNTIME_HEADERS="$work/deps/hsa-headers"
cmake --build "$work/hrx-build" --target hrx loomc_shared --parallel "$jobs"
# HSA uses C++20 jthread; the hosted Xcode SDK library lacks it. Compile
# and link against the same qualified LLVM libc++ that the package bundles.
cmake -S "$work/deps/mac-amdgpu/hsa" -B "$work/hsa-build" -G Ninja \
  "${darwin_archive_args[@]}" -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_DEPLOYMENT_TARGET=15.0 -DBUILD_TESTING=OFF \
  -DCMAKE_C_COMPILER="$llvm/clang" -DCMAKE_CXX_COMPILER="$llvm/clang++" \
  "-DCMAKE_SHARED_LINKER_FLAGS=-L$llvm/../lib/c++ -Wl,-rpath,$llvm/../lib/c++" \
  "-DCMAKE_EXE_LINKER_FLAGS=-L$llvm/../lib/c++ -Wl,-rpath,$llvm/../lib/c++"
cmake --build "$work/hsa-build" --target hsa-runtime64 --parallel "$jobs"
cmake -S "$work/source" -B "$work/lse-build" -G Ninja \
  "${darwin_archive_args[@]}" -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_DEPLOYMENT_TARGET=15.0 -DCMAKE_CXX_COMPILER="$llvm/clang++" \
  "-DCMAKE_EXE_LINKER_FLAGS=-L$llvm/../lib/c++ -Wl,-rpath,$llvm/../lib/c++" \
  -DLSE_ENABLE_CPU=ON -DLSE_ENABLE_HRX=ON -DLSE_BUILD_TESTS=ON -DLSE_GPU_TARGETS=gfx1201 \
  -DLSE_HRX_INCLUDE_DIR="$work/hrx-source/libhrx/include" \
  -DLSE_HRX_LIBRARY="$work/hrx-build/libhrx/src/libhrx/libhrx.dylib" \
  -DLSE_LOOMC_INCLUDE_DIR="$work/hrx-source/loom/binding/c/include" \
  -DLSE_LOOMC_LIBRARY="$work/hrx-build/loom/binding/c/libloomc.dylib"
# All these tests are host-only; do not run the entire suite on a runner with
# no external AMD GPU, and never report CPU checks as GPU qualification.
tests=(test_kernel_env test_ir test_dtype test_shape test_quant test_graph test_backend_cpu
  test_primitive test_trace test_loom_print test_loom_repeat test_loom_gdn
  test_loom_extent test_loom_conv test_loom_words test_loom_flash
  test_generation_stats test_http_timings test_server_shutdown test_dispatch_profile test_loom_cache
  test_pointwise_fusion test_probe_measurement test_probe_policy test_quant_prefill
  test_token_ids test_submission_tuner test_submission_constants test_submission_decode
  test_decode_sample test_loaded_runtime test_hrx_copy_route test_gdn_pair
  test_gdn_scheduler test_scheduler_epilogue test_loom_matrix test_loom_dot
  test_fp8_conversion test_quant_operand_policy test_cooperative_rms)
cmake --build "$work/lse-build" --target lse lse-server compile_loom_matrix --parallel "$jobs"
# The host suite must not discover a real GPU on a developer's machine.
# Some tests enumerate the default backend, so give them a CPU-only build.
cmake -S "$work/source" -B "$work/host-tests" -G Ninja \
  "${darwin_archive_args[@]}" -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_DEPLOYMENT_TARGET=15.0 \
  -DCMAKE_CXX_COMPILER="$llvm/clang++" \
  "-DCMAKE_EXE_LINKER_FLAGS=-L$llvm/../lib/c++ -Wl,-rpath,$llvm/../lib/c++" \
  -DLSE_ENABLE_CPU=ON -DLSE_ENABLE_HRX=OFF -DLSE_BUILD_TESTS=ON
cmake --build "$work/host-tests" --target "${tests[@]}" --parallel "$jobs"
regex="$(IFS='|'; echo "${tests[*]}")"
ctest --test-dir "$work/host-tests" --output-on-failure -R "^($regex)$"
# This HRX-linked test uses CpuBackend allocations and simulated kernel results;
# it never opens a GPU. It is not available in the CPU-only CMake configuration.
cmake --build "$work/lse-build" --target test_matrix_probe_lifecycle --parallel "$jobs"
LSE_BACKEND=cpu ctest --test-dir "$work/lse-build" --output-on-failure \
  -R '^test_matrix_probe_lifecycle$'
python3 "$work/source/tests/test_server_cli.py" "$work/lse-build/lse-server"
mkdir -p "$work/native-fixtures"
"$work/lse-build/tests/compile_loom_matrix" "$work/native-fixtures"
"$work/lse-build/lse" --help > "$work/lse-build/help.txt"
python3 "$root/.github/scripts/package-macos.py" \
  --root "$root" --work "$work" --llvm "$llvm/.." --tag "${LSE_RELEASE_TAG:-snapshot}"
printf 'macOS arm64 host tests and native gfx1201 compilation passed; no GPU execution was attempted.\n'
