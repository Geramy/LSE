### Combined M8 Q4 gate/up execution

The optimizer selects one shared-panel SwiGLU consumer for the measured gfx1201
M8/N17408/K5120 shape. It removes one launch and the gate temporary. Independent
FP32 projection accumulation order and the existing SiLU intrinsic are preserved.
The architecture and shape rule is in the central dispatch table.

Full-size native component checks match all 139,264 output bits. Paired component
GPU time falls 27.69%, and wall time falls 20.50%. The combined kernel uses 120 VGPRs
versus 88 for each separate kernel, with zero private scratch and zero LDS. The
measured gains include the higher register pressure; occupied waves were not measured.

### Measured workload and results

Measurements use source `257cc97f3bbc4fa1ae1877f068103415ea944857` and local server
SHA256 `30aa8dd5651d7fd541a48f3a7fa0c35c782a13e6a1dcfcbe8cb8296c734e6582`.
The release adds a version bump, documentation and packaging changes. These rates
were measured locally on macOS/R9700 gfx1201. They are not Linux throughput results.

Qwen3.8-27B Q4 target, Q8 auxiliary/draft weights, BF16 KV, FP32 floating
accumulation, temperature 0.6, top-k 20, top-p 0.95, seed 1234, batch/ubatch 1024 and
KV capacity 262100. Two 1024-token coding requests generate 384 tokens each, with
383 timed decode tokens. MTP uses depth 3; DFlash2 uses all seven proposals in block 8.
Each mode starts a new process with an empty private kernel cache. The resident
request reuses compiled code and zero prompt KV tokens. Compilation is included.
No profiler or competing GPU workload runs during throughput timing.

| Mode | Cold PP/s | Cold TPS | Resident PP/s | Resident TPS |
|---|---:|---:|---:|---:|
| Baseline | 446.28 | 24.23 | 634.34 | 24.86 |
| MTP=3 | 403.89 | 37.44 | 607.88 | 48.42 |
| DFlash2, seven proposals | 418.73 | 33.42 | 629.80 | 45.64 |

Matched resident DFlash2 improves 43.44 → 45.64 TPS (+5.07%). Both requests preserve
exact generated choices, acceptance counts and speculative pass counts. Verifier
time falls 70.7413 → 66.5850 ms per pass; draft time remains about 14.8 ms. All modes
record zero host groups and zero host fallbacks. MTP=3 remains faster on this prompt.
One cold/resident pair per mode does not establish long-context Pi throughput.
The 29/49/103 TPS targets remain unmet.

[Full method and resource results](https://github.com/Geramy/LSE/blob/v0.4.16/docs/benchmarks/m8-gate-up-pair-2026-09-29.md).

### Retained optimizations and compatibility

All accepted M8 down WMMA, M1024 prefill panels, GDN panel reuse, typed attention,
conditional DFlash2 sampling, memory retirement, buffer views, model generation
configuration and thinking/tool-call HTTP behavior remain active. No additional
perplexity run is needed for the bit-identical gate/up change.

Kernel cache ownership advances to 0.4.16. Startup removes only complete,
identifiable older LSE artifact families from the selected cache directory.
The default is `~/.lse/cache/`; `--cache-dir` overrides it. Foreign, partial,
symlink and newer-version entries are preserved.

Use `bin/lse` or `bin/lse-server` in the archives. The macOS arm64 package bundles
HSA/HRX/Loom and the dependency closure; MacAMDGPU must be installed and activated
separately. The Linux x86_64 package bundles HRX/Loom and uses compatible installed
ROCm/HSA. Each archive contains source/runtime manifests and a SHA256 checksum.
The macOS build runner checks host behavior, native gfx1201 compilation and
relocated launchers. It has no external AMD GPU; actual GPU execution was checked
locally in the qualified component and HTTP runs above.
