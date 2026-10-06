#!/bin/bash
# Builds the RDNA4 matrix-core microbenchmarks: gfx12 code objects with Homebrew LLVM/LLD 21
# (the toolchain LSE's macOS build already requires) and host launchers on the HSA runtime.
# usage: scripts/rdna4-matrix/build.sh [out-dir]   (HSA_PREFIX defaults to /usr/local)
set -euo pipefail
SRC=$(cd "$(dirname "$0")" && pwd)
OUT=${1:-$SRC/build}
HSA=${HSA_PREFIX:-/usr/local}
LLVM=$(brew --prefix llvm@21)/bin
LLD=$(brew --prefix lld@21)/bin
MCPU=${MCPU:-gfx1201}
mkdir -p "$OUT"
cl() {  # cl <source> <object> [defines...]
  local src=$1 obj=$2; shift 2
  "$LLVM/clang" -x cl -cl-std=CL2.0 -nogpulib --target=amdgcn-amd-amdhsa -mcpu=$MCPU -O3 "$@" -c "$SRC/$src" -o "$OUT/$obj.o"
  "$LLD/ld.lld" -shared "$OUT/$obj.o" -o "$OUT/$obj.hsaco"
  rm -f "$OUT/$obj.o"
}
cl peak.cl peak
cl probe.cl probe
for cfg in 0_0 1_0 2_0 3_0 4_0 0_1 1_1 3_1 4_1 0_2 1_2 3_2 4_2; do
  cl gemm.cl gemm_$cfg -DMODE=${cfg%_*} -DRESCALE=${cfg#*_} -DWN=4 -DWM=2 -DTN=2 -DTM=4 -DBKB=64
done
for q in 0 1; do for bt in 0 1; do cl q4gemm.cl q4_${q}_bt$bt -DQMODE=$q -DBIASTAIL=$bt; done; done
for host in peak probe gemm q4bench; do
  "$LLVM/clang++" -std=c++20 -O2 -I"$HSA/include" "$SRC/$host.cpp" -L"$HSA/lib" -lhsa-runtime64 -Wl,-rpath,"$HSA/lib" -o "$OUT/$host"
done
echo "built into $OUT"
