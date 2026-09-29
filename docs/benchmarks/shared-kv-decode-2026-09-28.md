# Shared-KV single-token decode attention

## Change

The gfx1201 split-attention shape table selects six query heads per workgroup when six query heads share one KV head. The existing tiled QK/PV body reads each K/V value once for those six heads. FP32 accumulation, masks, paging, partial-record ABI, and merge remain unchanged. Empty global partitions publish all 258 neutral fields before a workgroup-uniform return. The unchanged-policy submission-tuner end wait is removed; policy changes still retire all backend streams.

## Matched GPU component measurements

R9700 gfx1201 wave32; query heads 24, KV heads 4, dimension 256, page size 16. Partial plus merge, eight warmups per arm, ABBA blocks of 20; 40 timed invocations per arm. Passive GPU timestamps. Both candidate blocks beat both original blocks in each winning six-head case.

| KV format | Live keys / table | Previous GPU ms | Six-head GPU ms | Change |
| --- | ---: | ---: | ---: | ---: |
| F16 | 5207 / 8192 | 0.217655 | 0.105209 | -51.7% |
| F16 | 14000 / 16384 | 0.472087 | 0.210866 | -55.3% |
| BF16 | 5207 / 8192 | 0.207091 | 0.105114 | -49.2% |
| BF16 | 14000 / 16384 | 0.461380 | 0.216098 | -53.2% |

Two- and three-head tiles were measured and rejected in favor of six. No alternative tile bodies or switches were added to production. Empty exit alone improved BF16 5.2K attention by 13.2%, but did not establish a 14K gain (+1.5% in that comparison). Head reuse provides the measured long-context gain.

All timed code objects reported known zero private scratch allocation. Spill counts are unknown. Allocation guards, input bytes, double-reference outputs, poisoned partial records, empty replay, and restored replay passed. The canonical native runtime passed 80/80 cases; the existing T1 oracle now covers both grouped and ungrouped heads. A separate production fixture passed FP32, FP16, BF16, FP8, and BF8 with maximum absolute error below 9.62e-8, 30 device dispatches, and zero host/fallback. Six focused host suites passed. No perplexity run was made.

## Real HTTP measurements

Same two Pi turns, Q4 target, BF16 KV, model defaults temperature 1 / top-k 20 / top-p .95. Each process starts with a fresh private cache. First prompt has 5207 tokens. Follow-up reuses 5262 tokens and adds 23. Compilation is included in first-request timing.

| Mode | First PP/s | First decode TPS | Follow-up TPS | Follow-up acceptance |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 344.21 | 22.58 | 23.82 | — |
| MTP=3 | 313.32 | 24.74 | 40.25 | 75.6% |
| DFlash2 | 328.65 | 26.40 | 37.53 | 72.5% |

Baseline follow-up improved from 22.82 to 23.82 TPS (+4.4%). The isolated tuner change measured 22.82 → 22.81 follow-up TPS and did not establish a gain. MTP/DFlash bypass that tuner and mainly verify multiple query rows, so they do not inherit the full T1 component gain. Their recorded follow-up rates remain near their prior results.

All recorded responses are byte-identical to their respective preceding controls. There were zero host groups and zero fallback groups. These are single matched workload observations, not a 29/49/103 TPS or 600 PP/s guarantee.

## Sampling and remaining waits

`LSE_TIME_STEPS=1` now reports `[sampling-spans]` and `[readback-spans]`. Readback reports completion waiting separately from the subsequent transfer; sampling includes CPU selection, softmax, and RNG. These diagnostics do not change token selection.

A separate passive GPU/CPU diagnostic on the same baseline and BF16 KV recorded 162 full-vocabulary readbacks/samples:

| Operation | Mean ms | Median ms |
| --- | ---: | ---: |
| 1 MB logits transfer | 0.302287 | 0.282770 |
| CPU sampling | 0.110074 | 0.097292 |
| Readback extra completion wait | 0.000813 | 0.000791 |

The main mandatory GPU completion wait occurs earlier in scheduler evaluation. The later readback wait averages 0.813 microseconds, not one millisecond. These results do not support an extra avoidable 1 ms logits drain. The prior slower GPU top-k implementation remains removed. CPU follow-up inference-thread samples spent 84.0% in HSA waiting; this includes outstanding GPU execution and is not 84% wasted overhead.

The real baseline profile confirms the selected T1 partial grid is 256 workgroups for 24 heads and an 8192-token table, rather than 1536. T1 partial averages 96.32 microseconds and merge 24.99 microseconds in that diagnostic. Q4 projection and FFN execution remain the largest decode GPU families.

## Evidence

Local artifacts are in `mac_amdgpu/build/release/pi-performance/`: `decode-attention-ab/`, `baseline-tuner-{before,after}/`, `{baseline,mtp3,dflash}-head6-after/`, and `baseline-head6-profile/`. These retain commands, executable hashes, sampling settings, metrics, outputs, component resource facts, CPU samples, and passive timestamp captures. Raw requests/responses are not committed.

HTTP measured server SHA256: `30e2a4b4bec44829f8f0da4e47762a39f785ac9673987d73bee7b6c49e815cc2`. The diagnostic executable adds only profiling spans and has SHA256 `0cac93052074bc5edc5bdba521f58391a5c5848b1741ebd23211aabb7c7039ba`.
