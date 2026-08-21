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
dir="$root/reference/hrx-system"
rocm="${ROCM_PATH:-/opt/rocm}"

if [[ ! -d "$rocm" ]]; then
  echo "no ROCm at $rocm; set ROCM_PATH" >&2
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

cd "$dir"
python dev.py cmake configure -DIREE_HAL_DRIVER_AMDGPU=ON -DIREE_ROCM_PATH="$rocm"
python dev.py cmake build

install="$dir/build/hrx-install"
echo
echo "HRX built. Configure this tree with:"
echo "  cmake -S . -B build -GNinja -DCMAKE_CXX_COMPILER=g++-16 \\"
echo "        -DLSE_HRX_ROOT=$install"
