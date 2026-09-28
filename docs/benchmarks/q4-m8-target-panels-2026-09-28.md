# Q4 eight-row target panels — 2026-09-28

## Accepted scope

The Q4 shared activation panel now covers eight-row target projections at N/K: 17408/5120, 5120/17408, 10240/5120, 6144/5120, 12288/5120, 5120/6144 and 248320/5120. These seven entries extend `kQ4PanelShapes`; the existing producer and eight-row consumer are unchanged. Original per-shape K splits, activation codec and FP32 restoration/reduction order are retained. DFlash2 continues to use three proposals by default.

## Native correctness and component timing

All seven shapes passed registered native execution against the original path: complete raw and fused residual output bits, every U32 panel word, input immutability, replay and guarded buffers. Each final reported trace had three device groups, zero host groups and zero fallbacks. An independent quantization component oracle also passed. No logits-L2 or perplexity run was needed for bitwise-equivalent arithmetic.

Bounded ABBA20 measurements on gfx1201 used three warmups and 40 measured calls per arm. GPU timestamps use HRX's physical-device frequency metadata (100 MHz). Candidate time includes preparation and consumption; no host-clock correlation was available.

| Eight-row contraction | Original GPU ms | Panel prep + consumer ms | GPU change | Wall change |
|---|---:|---:|---:|---:|
| FFN up/gate, N17408/K5120 | 0.324098 | 0.005048 + 0.178333 = 0.183381 | −43.42% | −40.28% |
| FFN down, N5120/K17408 | 0.422311 | 0.005216 + 0.215269 = 0.220485 | −47.79% | −44.63% |

Both candidate arms beat both original arms. The full six-case fixture, including four small projections kept inactive, completed 774 device dispatches with zero host groups, fallbacks or measured compiles. The M8 consumers used 92 VGPRs, 20/24 SGPRs, zero LDS and zero scratch; originals used 86/87 VGPRs, 24/28 SGPRs and 33024/56096 LDS bytes, with zero scratch. These resources are measured metadata, not an occupancy claim.

## HTTP validation and limits

The initial private depth-seven executable incorrectly retained canonical graph/kernel archives through absolute `-force_load` arguments. Its intended M8 policy overlay was inactive. Those results are preserved and marked invalid for panel selection. The corrected link replaces absolute and relative archive arguments; its linker map proves private ownership of `ops.cpp`, `quant.cpp` and `quant_linear_panel.cpp`. Offline emission proved all seven prepared U32 panel routes and the representative FFN consumer binding before HTTP validation.

On the same two-turn Pi request, legacy and corrected depth-seven runs produced identical assistant JSON, generation counts and acceptance counters. The follow-up generated 159 tokens in 67 passes, accepting 92/158 proposals, with zero host groups/fallbacks:

| Depth-seven follow-up | Legacy M8 | Corrected M8 panels |
|---|---:|---:|
| Verification ms | 10035.184 | 6508.480 |
| Verification ms/pass | 149.779 | 97.142 |
| Timed decode tokens/s | 13.961 | 20.140 |

Verification fell 35.14%. Depth seven remains slower than the current depth-three control (31.337 timed tokens/s, 160 generated tokens in 65 passes). The sampled depth-three response differs, so this is a workload comparison, not an identical-token depth comparison. The seven panel entries are accepted; the default depth stays three.

The corrected candidate started after clearing the kernel cache. By the end of the first response, cumulative process JIT reached 79.614 seconds. The follow-up added 31 variants and 15.532 seconds of JIT. Those prompt rates include compilation and do not describe warm prefill. A separate cold depth-three control likewise added 15 variants and 15.528 seconds of follow-up JIT; its assistant JSON matched the prior warm depth-three control on both turns. This table change does not resolve cold compilation latency.

## Evidence

Native evidence: `mac_amdgpu/build/release/pi-performance` and the frozen qualification directory `/private/tmp/lse-q4-panel-m8-qualification` (`gpu-m8.log`, `cp-timings.json`, `resources.json`, `manifest.json`). Native executable SHA256: `6db6051a3273bf6f31f8b280c10a91a8ff2e6b66eeb46e4527c5788fe79bf4f3`.

HTTP source base: `17aafcba2042ca69b6eeeaad3dc7d884206de715`; corrected executable SHA256: `42a917bf96b7b4b3bdd6f42426169789d9262fb4852ac01e775cc851250ff67c`. HTTP results: `depth7-m8-mapped` (invalid overlay), `depth7-m8-corrected-cold`, and `depth3-current-cold` under `mac_amdgpu/build/release/pi-performance`. The private manifest and linker map are in `/private/tmp/lse-dflash-depth`. Raw request payloads are not tracked.
