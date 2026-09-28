#!/bin/sh
set -eu
if [ "$#" -lt 1 ]; then
  echo "usage: $0 /path/to/probe-build [extra CMake options...]" >&2
  exit 2
fi
probe_build=$1
shift
probe_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
probe_repo=$(CDPATH= cd -- "$probe_dir/../.." && pwd)
cmake -S "$probe_repo" -B "$probe_build" -G Ninja \
  -DLSE_BUILD_SPLITK4_PROBE=ON -DLSE_ENABLE_HRX=ON \
  -DLSE_BUILD_TESTS=OFF -DLSE_BUILD_CLI=OFF -DLSE_BUILD_SERVER=OFF "$@"
cmake --build "$probe_build" --target q4_int8_splitk4_probe -j "${PROBE_JOBS:-4}"
