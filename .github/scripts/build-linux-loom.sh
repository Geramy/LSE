#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
work="$root/build/loom"
deps="$root/build/deps/linux-loom"
[[ "$(uname -s)" == Linux ]] || {
  echo 'This compiler build requires Linux.' >&2; exit 1;
}
hrx_rev=5927b0e0fafdefb5c8b41aa71bca8fd28791ad7c
fetch() {
  local url="$1" rev="$2" path="$3"
  if [[ ! -d "$path/.git" ]]; then
    git init -q "$path"
    git -C "$path" remote add origin "$url"
    git -C "$path" fetch --depth 1 origin "$rev"
    git -C "$path" checkout --detach FETCH_HEAD
  fi
  [[ "$(git -C "$path" rev-parse HEAD)" == "$rev" &&
     -z "$(git -C "$path" status --porcelain)" ]] || {
    echo "Dependency revision or contents differ: $path" >&2; exit 1;
  }
}
mkdir -p "$work" "$deps"
fetch https://github.com/ROCm/hrx-system.git "$hrx_rev" "$deps/hrx-$hrx_rev"
python3 - "$work/source" <<'PY'
from pathlib import Path
import shutil, sys
source = Path(sys.argv[1])
if source.is_symlink():
    raise SystemExit(f'Refusing generated source symlink: {source}')
if source.exists():
    shutil.rmtree(source)
source.mkdir()
PY
git -C "$deps/hrx-$hrx_rev" archive HEAD | tar -x -C "$work/source"
git -C "$work/source" init -q
# LSE's Loom patches (patches/hrx), then its HRX runtime patches (patches/).
# loom-grow-arrays-only-when-full keeps loomc from asking for gigabytes on a
# kernel with many matrix operands, such as the prefill GEMM's 64x64 tiles.
# gfx120x-hdp-kernarg-publication lets RDNA4 keep kernel arguments in VRAM.
# loom-vmem-load-latency schedules global loads with their real latency.
# loom-allocation-entry-preamble (upstream #1377) keeps allocation's split and
# replica copies after the entry live-in/resource preamble.
patches=(symbolic-memo-touched-reset.patch gfx12-vopd-identical-source.patch cooperative-matrix-operands.patch loop-invariant-motion.patch rdna4-prefetch-address-span.patch loom-grow-arrays-only-when-full.patch gfx120x-hdp-kernarg-publication.patch loom-vmem-load-latency.patch loom-allocation-entry-preamble.patch)
for patch in "${patches[@]}"; do
  path="$root/patches/hrx/$patch"
  git -C "$work/source" apply --check "$path"
  git -C "$work/source" apply "$path"
done
for patch in "$root"/patches/*.patch; do
  git -C "$work/source" apply --check "$patch"
  git -C "$work/source" apply "$patch"
done
rocm="${ROCM_PATH:-/opt/rocm}"
if [[ ! -f "$rocm/include/hsa/hsa.h" ]]; then
  for candidate in "$rocm"/core-*; do
    if [[ -f "$candidate/include/hsa/hsa.h" ]]; then rocm="$candidate"; break; fi
  done
fi
if [[ -x "$rocm/llvm/bin/clang" ]]; then
  default_cc="$rocm/llvm/bin/clang"
else
  default_cc="$(command -v clang)"
fi
cc="${CC:-$default_cc}"
export PATH="$(dirname "$cc"):$PATH"
cxx="${CXX:-${cc}++}"
# Build the runtime and compiler together for the native K/V address API.
# The upstream loom-compile configuration requires the VM execution target when
# its VM emitter is enabled.
cmake -S "$work/source" -B "$work/build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DLIBHRX_BUILD=ON \
  -DCMAKE_C_COMPILER="$cc" -DCMAKE_CXX_COMPILER="$cxx" \
  -DIREE_ROCM_PATH="$rocm" -DLIBHRX_BUILD_HIP_BINDING=OFF -DLIBHRX_BUILD_CTS=OFF \
  -DIREE_BUILD_TESTS=OFF -DIREE_BUILD_BENCHMARKS=OFF \
  -DIREE_ENABLE_LIBBACKTRACE=OFF -DIREE_HAL_DRIVER_DEFAULTS=OFF \
  -DIREE_HAL_DRIVER_AMDGPU=ON -DIREE_HAL_DRIVER_HIP=OFF \
  -DIREE_HAL_DRIVER_HIP_RCCL=OFF -DIREE_HAL_DRIVER_VULKAN=OFF \
  -DLOOM_BUILD=ON -DLOOM_TARGET_DEFAULTS=OFF -DLOOM_EXECUTE_DEFAULTS=OFF \
  -DLOOM_TARGET_AMDGPU=ON -DLOOM_TARGET_AMDGPU_TARGETS=loom_defaults \
  -DLOOM_TARGET_IREE_VM=ON -DLOOM_EXECUTE_IREE_VM=ON -DLOOM_TARGET_LLVMIR=ON \
  -DLOOM_TARGET_SPIRV=ON -DLOOM_TARGET_X86=ON
cmake --build "$work/build" --target hrx loomc_shared --parallel "${LSE_BUILD_JOBS:-3}"
python3 - "$work" "$deps" "$hrx_rev" <<'PY'
from pathlib import Path
import hashlib, json, os, re, subprocess, sys
work, deps = map(Path, sys.argv[1:3])
hrx_rev = sys.argv[3]
library = (work / 'build/loom/binding/c/libloomc.so').resolve(strict=True)
include = work / 'source/loom/binding/c/include'
version = work / 'build/loom/binding/c/cmake/loomc/loomc-config-version.cmake'
if not version.is_file() or not (include / 'loomc/loomc.h').is_file():
    raise SystemExit('Missing built Loom C API headers or version metadata')
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
patches = []
for name in ('symbolic-memo-touched-reset.patch', 'gfx12-vopd-identical-source.patch', 'cooperative-matrix-operands.patch', 'loop-invariant-motion.patch', 'rdna4-prefetch-address-span.patch', 'loom-grow-arrays-only-when-full.patch'):
    path = work.parents[1] / 'patches/hrx' / name
    patches.append({'path': f'patches/hrx/{name}', 'sha256': digest(path)})
for path in sorted((work.parents[1] / 'patches').glob('*.patch')):
    patches.append({'path': 'patches/' + path.name, 'sha256': digest(path)})
cache = {}
for line in (work / 'build/CMakeCache.txt').read_text().splitlines():
    if match := re.match(r'([^/#:][^:]*):[^=]+=(.*)', line):
        cache[match[1]] = match[2]
compilers = {key: subprocess.check_output([cache[key], '--version'], text=True).splitlines()[0]
             for key in ('CMAKE_C_COMPILER', 'CMAKE_CXX_COMPILER')}
configuration = {key: value for key, value in cache.items()
                 if key.startswith(('LOOM_TARGET_', 'LOOM_EXECUTE_', 'IREE_HAL_DRIVER_'))
                 or key in ('CMAKE_BUILD_TYPE', 'LIBHRX_BUILD', 'IREE_BUILD_TESTS', 'IREE_BUILD_BENCHMARKS')}
manifest = {
    'hrx_revision': hrx_rev,
    'patches': patches,
    'library': str(library), 'library_sha256': digest(library),
    'include_directory': str(include), 'version_file': str(version),
    'amdgpu_targets': 'loom_defaults',
    'compiler_only': False, 'host_compilers': compilers,
    'configuration': configuration,
    'build_script_sha256': digest(work.parents[1] / '.github/scripts/build-linux-loom.sh'),
    'cmake': subprocess.check_output(['cmake', '--version'], text=True).splitlines()[0],
}
(work / 'BUILD.json').write_text(json.dumps(manifest, indent=2) + '\n')
variables = {
    'LSE_HRX_INCLUDE_DIR': work / 'source/libhrx/include',
    'LSE_HRX_LIBRARY': (work / 'build/libhrx/src/libhrx/libhrx.so').resolve(strict=True),
    'LSE_LOOMC_INCLUDE_DIR': include,
    'LSE_LOOMC_LIBRARY': library,
    'LSE_LOOMC_VERSION_FILE': version,
    'LSE_LOOMC_MANIFEST': work / 'BUILD.json',
}
for key, value in variables.items():
    print(f'{key}={value}')
if github_env := os.environ.get('GITHUB_ENV'):
    with open(github_env, 'a') as out:
        for key, value in variables.items():
            out.write(f'{key}={value}\n')
PY
