#!/usr/bin/env bash
# Clones and builds the HRX backend into reference/hrx-system.
#
# Not part of the CMake configure: this is an IREE-derived tree with its own
# build driver and it takes tens of minutes, which is not something a configure
# step should do behind someone's back. CMake fetches fastokens because that is
# seconds and a path dependency; this is neither.
set -euo pipefail

repo="${LSE_HRX_REPO:-https://github.com/ROCm/hrx-system.git}"
ref="${LSE_HRX_REF:-}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# `python` is not a command on a stock Ubuntu 26.04; dev.py wants an interpreter
# by name, so pick one that exists rather than the one its docs happen to spell.
py="${PYTHON:-}"
if [[ -z "$py" ]]; then
  for c in python3 python; do command -v "$c" >/dev/null && { py="$c"; break; }; done
fi
if [[ -z "$py" ]]; then echo "no python interpreter found; set PYTHON" >&2; exit 1; fi
dir="$root/reference/hrx-system"
rocm="${ROCM_PATH:-/opt/rocm}"

if [[ ! -d "$rocm" ]]; then
  echo "no ROCm at $rocm; set ROCM_PATH" >&2
  exit 1
fi

# TheRock splits an install into components, so /opt/rocm holds core-*/ and
# extras-*/ and no include/ or lib/ of its own. hrx-system wants a flat root and
# stops with "HSA runtime headers were not found" against the outer directory.
# Descend to the component that actually carries them, newest first.
if [[ ! -e "$rocm/include/hsa/hsa.h" ]]; then
  for cand in $(ls -d "$rocm"/core-* 2>/dev/null | sort -r); do
    if [[ -e "$cand/include/hsa/hsa.h" ]]; then
      echo "componentised ROCm: using $cand"
      rocm="$cand"
      break
    fi
  done
fi
if [[ ! -e "$rocm/include/hsa/hsa.h" ]]; then
  echo "no hsa/hsa.h under $rocm -- install the ROCm runtime dev package" >&2
  echo "  (TheRock: amdrocm-runtime-dev<ver>, classic: hsa-rocr-dev)" >&2
  exit 1
fi

if [[ ! -d "$dir/.git" ]]; then
  echo "cloning $repo"
  git clone --recursive "$repo" "$dir"
fi
if [[ -n "$ref" ]]; then
  git -C "$dir" checkout "$ref"
  git -C "$dir" submodule update --init --recursive
fi

# The AMDGPU driver is built with clang targeting amdgcn, and it will not take
# gcc: IREE_ROCM_PATH supplies device libraries, not a host compiler. ROCm ships
# a clang beside the runtime, which is the one that matches it.
cc="${CC:-}"; cxx="${CXX:-}"
if [[ -z "$cc" ]]; then
  for cand in "$rocm/llvm/bin/clang" "$(command -v clang || true)"; do
    [[ -x "$cand" ]] && { cc="$cand"; cxx="${cand}++"; break; }
  done
fi
if [[ -z "$cc" ]]; then
  echo "no clang found for the AMDGPU driver; install clang or set CC/CXX" >&2
  exit 1
fi
echo "host compiler: $cc"

cd "$dir"
"$py" dev.py cmake configure -DIREE_HAL_DRIVER_AMDGPU=ON -DIREE_ROCM_PATH="$rocm" \
  -DCMAKE_C_COMPILER="$cc" -DCMAKE_CXX_COMPILER="$cxx"
"$py" dev.py cmake build

install="$dir/build/hrx-install"
echo
echo "HRX built. Configure this tree with:"
echo "  cmake -S . -B build -GNinja -DCMAKE_CXX_COMPILER=g++-16 \\"
echo "        -DLSE_HRX_ROOT=$install"
