# Combined M8 Q4 gate/up execution

Date: 2026-09-29. The graph optimizer selects the combined kernel for the measured M8 gate/up shape. No launch option is required.

## Change

The optimizer replaces `SiLU(quant_linear(x, gate)) * quant_linear(x, up)` with `quant_swiglu.q4_shared_panel.v1`. Both projections read one existing group8 activation panel. The combined consumer removes one launch and the gate temporary. Each projection retains its own integer dot products, FP32 accumulation order, affine restoration and wave reduction. SiLU uses the existing backend intrinsic.

The rule is in `include/lse/dispatch/quant_shapes.hpp`: gfx1201, wave32, Q4/group64, M8/N17408/K5120, WG256. Input is F32 `[1,8,5120]`; packed weights are U32 `[17408,640]`; scales and biases are BF16 `[17408,80]`; the shared panel is U32 `[8,2000]`. Output is F32 `[1,8,17408]`. The optimizer applies the rewrite before retaining a device execution plan. It preserves the output node and checks intermediate consumers, requested roots, storage contracts, placement and the qualified DOT4 schedule. The accepted kernels for other shapes retain their existing selection.

## Component measurement

The control runs the shared producer, gate+SiLU, then up+multiply. The candidate runs the identical producer and one combined consumer. Both control shaders match the accepted artifacts byte for byte. Full-sized deterministic synthetic buffers use distinct gate/up weights and affine patterns.

Twenty ABBA rounds provide 40 samples per arm, after 16 warm chains per arm. One complete prepared chain is measured per sample. Passive GPU timestamps measure execution; ordinary completion synchronization measures wall time. Compilation, uploads, output checks and allocation are outside timing.

| Mean time | Accepted chain | Combined chain | Reduction |
|---|---:|---:|---:|
| GPU execution sum | 0.255890 ms | 0.185033 ms | 27.69% |
| Host wall time | 0.371035 ms | 0.294984 ms | 20.50% |
| GPU consumers | 0.250978 ms | 0.180111 ms | 28.24% |

Every paired round improves. All 139,264 output bits match the accepted device chain. Independent arithmetic, raw panel codes/steps/sums, input guards, output padding, zero replay and restored-input replay pass. No perplexity or L2 test is added for this bit-identical change.

## Resources and ISA

| Property | Gate+SiLU | Up+multiply | Combined |
|---|---:|---:|---:|
| VGPR | 88 | 88 | 120 |
| SGPR | 20 | 24 | 38 |
| LDS bytes | 0 | 0 | 0 |
| Private scratch bytes | 0 | 0 | 0 |
| Instructions | 1417 | 1250 | 1996 |
| Global load instructions | 106 | 114 | 116 |
| Load waits | 86 | 94 | 91 |
| Dual-issue pairs | 3 | 3 | 21 |
| DOT4 instructions | 160 | 160 | 320 |

The combined kernel uses more VGPRs. It reduces activation-panel loads and preserves both weight streams. It removes a 557,056-byte gate temporary and its write/read. Static instruction counts do not measure cache hits, bandwidth or occupied waves. Register allocation can lower the occupancy ceiling; the measured component and HTTP improvements include that cost. No hardware occupancy claim is made.

The registered production kernel emits the exact prototype source and code object. Qualified code object SHA256: `94272c39fbd8cf1664a1cea8fd3a1a96a0e17fc604b99e98d9af87787ef1eba5`. That exact object appears in the candidate HTTP cache, confirming active selection.

## Matched DFlash2 HTTP comparison

Qwen3.8-27B Q4 target; Q8 DFlash2 block8, seven proposals. Two fixed 1024-token coding prompts, 384 generated tokens and 383 timed decode tokens. Temperature 0.6, top-k 20, top-p 0.95, seed 1234, BF16 KV, FP32 floating accumulation, batch/ubatch 1024 and KV capacity 262100. Each process starts with an empty private disk cache. The second request reuses compiled code and reuses zero prompt KV tokens. Compilation is included. Throughput runs use no profiler and no competing GPU workload.

| Request | Control PP/s | Combined PP/s | Control TPS | Combined TPS |
|---|---:|---:|---:|---:|
| Cold | 417.81 | 418.73 | 32.30 | 33.42 |
| Resident | 627.55 | 629.80 | 43.44 | 45.64 |

Resident decode improves **5.07%** in this matched pair. Verifier time falls from 70.7413 to 66.5850 ms per pass; draft time remains 14.8116 to 14.8003 ms. Both requests have identical generated choices, acceptance counts and speculative pass counts. Resident acceptance is 281/369, with 102 passes. The JIT kernel count falls from 283 to 282. All requests execute zero host groups and zero host fallbacks.

Control server SHA256: `b78f36a45355f5cade9bf5b960f5080e5e6607fc3e86e5eccd14a79c8795329f`.
Candidate server SHA256: `30aa8dd5651d7fd541a48f3a7fa0c35c782a13e6a1dcfcbe8cb8296c734e6582`.
Bundled HSA/HRX/Loom identities and actual mapped paths are recorded with each run. This is a single comparison pair; it does not establish long-context Pi throughput.

## Same-executable mode comparison

Each mode uses the same new executable, request pair, storage format and launch settings. Each mode starts a new process with an empty cache.

| Mode | Cold PP/s | Cold TPS | Resident PP/s | Resident TPS |
|---|---:|---:|---:|---:|
| Baseline | 446.28 | 24.23 | 634.34 | 24.86 |
| MTP=3 | 403.89 | 37.44 | 607.88 | 48.42 |
| DFlash2 | 418.73 | 33.42 | 629.80 | 45.64 |

Resident MTP acceptance is 80.70%; resident DFlash2 acceptance is 76.15%. Baseline and DFlash2 remain below the requested 29/103 TPS targets; MTP remains below 49 TPS. These results apply to the measured 1024-token coding prompts.

## Verification and evidence

The graph/typed-reference suite passes 5/5. Existing activation-panel 9/9 and dispatch 10/10 tests pass. Tests cover output identity, one shared producer, escaping intermediates, incompatible device/geometry/storage/placement, invalid raw buffers and signed-code arithmetic with distinct weights. The registered emitted source/object match the qualified native prototype. Independent component review found no arithmetic or timing blocker.

Local evidence is in `build/release/pi-performance/m8-gate-up-pair-component/` and `m8-gate-up-pair-http/`. It includes raw requests/responses, commands, binary/runtime hashes, compile resources, passive timestamps, ISA counts, correctness logs and the exact cached-object selection check.
