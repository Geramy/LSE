#!/bin/sh
set -eu
if [ "$#" -ne 3 ]; then
  echo "usage: $0 /path/to/probe-build edge|perf /path/to/output-prefix" >&2
  exit 2
fi
probe_build=$1
probe_mode=$2
probe_prefix=$3
probe_binary="$probe_build/experiments/q4-int8-splitk4/q4_int8_splitk4_probe"
case "$probe_mode" in
  edge) probe_native_mode=edge; probe_count=1 ;;
  perf) probe_native_mode=perf-m4; probe_count=20 ;;
  *) echo "mode must be edge or perf" >&2; exit 2 ;;
esac
mkdir -p "$(dirname -- "$probe_prefix")"
LSE_REQUIRE_DEVICE_KERNELS=1 HRX_PROFILE_MODE=dispatch \
  HRX_PROFILE_FILE="$probe_prefix.ireeprof" \
  "$probe_binary" "$probe_native_mode" "$probe_prefix" --native "$probe_count" \
  > "$probe_prefix.log" 2>&1
cat "$probe_prefix.log"
