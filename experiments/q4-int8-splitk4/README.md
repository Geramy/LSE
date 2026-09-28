# Archived Q4 INT8 WMMA split-K4 experiment

**Rejected:** actual M4 FFN up/down were 34.19%/49.10% slower in GPU CP time than the accepted global-panel DOT4 path, including preparation/merge. Correctness passed. See [RESULTS.md](RESULTS.md) and the compact JSON summaries.

This experiment is isolated on `experiments/q4-int8-splitk4-20260928`, based on `0817710340308f1e129fc3d9c19c864bee9429a0`. The original inactive Q4 WMMA source is retained in the branch base history. No active main-branch dispatch rule is added. A default-off CMake target builds only this probe.

## Reproduction

Requirements: a C++26 compiler, CMake 3.24+, Ninja, HRX headers/library, Loomc headers/library, and a gfx1201 wave32 device. The source, oracles and scripts are portable; the recorded performance conclusions apply to this measured device/runtime. Use an isolated build directory.

```sh
export CXX=clang++
experiments/q4-int8-splitk4/build.sh /tmp/lse-splitk4-build \
  -DLSE_HRX_INCLUDE_DIR=/path/to/libhrx/include \
  -DLSE_HRX_LIBRARY=/path/to/libhrx.dylib \
  -DLSE_LOOMC_INCLUDE_DIR=/path/to/loom/binding/c/include \
  -DLSE_LOOMC_LIBRARY=/path/to/libloomc.dylib

experiments/q4-int8-splitk4/run.sh /tmp/lse-splitk4-build edge /tmp/splitk4/edge
experiments/q4-int8-splitk4/run.sh /tmp/lse-splitk4-build perf /tmp/splitk4/perf
```

On Linux, provide the corresponding `.so` libraries. Normal loader installation/RPATH applies; the experiment does not require a library-path override. The original run used Clang 21.1.8, `-O2 -fno-fast-math`, installed HSA 32 us default and HRX CP dispatch capture.

The four edge cases check empty/ragged K partitions, N17 tails and realistic activations. The two performance cases are M4/N17408/K5120 and M4/N5120/K17408; each receives 32 warm calls per arm and ABBA20 (40 measured calls per arm). Both timing arms include two kernels: accepted activation prep+DOT4, or split-K4 matrix partial+merge. No model files are required.

To inspect CP timestamps, use the matching `iree-profile` binary:

```sh
iree-profile export --format=ireeperf-jsonl \
  --output=/tmp/splitk4/perf.jsonl /tmp/splitk4/perf.ireeprof
python3 experiments/q4-int8-splitk4/analyze.py /tmp/splitk4/perf
```

Offline compile gates are available without initializing a device:

```sh
/tmp/lse-splitk4-build/experiments/q4-int8-splitk4/q4_int8_splitk4_probe \
  edge /tmp/splitk4/offline-edge --offline 1
```

## Constraints

The candidate keeps the N16-major packed-weight copy alongside raw weights; one-time packing/upload is excluded from warmed timings and its storage cost is unresolved. Its group64 INT8 activation codec and cross-partition FP32 association differ from the accepted chunk8 DOT4 codec. Each is checked against its own independent component oracle, with FP32 partials and FP32 final accumulation. No bitwise equality between different codecs is claimed.

All four edges and both full shapes passed finite/completeness checks, every partial-plane oracle, exact active raw-panel words, readonly inputs and allocation guards:656 device dispatches, zero host/fallback. No perplexity sweep was run after the performance rejection.

## Archive compatibility

The committed planning shim uses a private namespace rather than replacing a project archive object. The frozen panel producer now explicitly declines epilogues so it satisfies the generic typed terminal-output contract, and its private symbol identity avoids collisions. These changes do not alter its arithmetic. `manifest.json` records the original measured binary/source hashes and exact compiler/runtime hashes; binaries and large raw capture files are excluded from this branch. Reproduction scripts are supplied; no new GPU measurement is claimed for the archive wrapper.
