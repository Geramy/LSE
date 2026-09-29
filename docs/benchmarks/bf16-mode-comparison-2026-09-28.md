# BF16 mode comparison — 2026-09-28

This is an **unreleased source candidate**, based on `9cb4cd8` plus the recorded
working changes. It includes generalized typed WMMA attention, automatic buffer
views and the replay-cover correction. It does not describe a published archive.

## Two-turn Pi observations

All three modes use the same server executable and mapped HSA runtime, Q4 target,
BF16 target KV, HRX/Loom on gfx1201 and the captured Pi chat requests. MTP uses its
Q8 draft module with depth 3; DFlash2 uses a Q8 drafter with three returned proposals.
The request's model label contains Q6; the served weights are `qwen38-27b-q4`.
Sampling comes from model defaults: temperature 1, top-k 20, top-p 0.95, low reasoning.

Each mode starts a fresh process with an empty private disk kernel cache. The first
prompt has 5,207 tokens. The byte-identical follow-up request has 5,285 tokens:
5,262 cached and 23 new. The second turn keeps the same process and resident state.
Model loading is outside the reported prompt rate; first prefill includes JIT.

| Mode | First prefill, tokens/s | First decode, tokens/s | Follow-up decode, tokens/s |
| --- | ---: | ---: | ---: |
| Baseline | 348.69 | 22.55 | 22.93 |
| MTP=3 | 318.86 | 26.19 | 40.19 |
| DFlash2 | 333.95 | 26.15 | 38.12 |

The first response is byte identical across all modes. MTP and DFlash2 responses
are byte identical on both turns. Baseline's follow-up differs and is longer.
The decode timer excludes the first generated token:

| Mode | First timed / generated tokens | Follow-up timed / generated tokens | Follow-up prefill, s |
| --- | ---: | ---: | ---: |
| Baseline | 54 / 55 | 104 / 105 | 0.670 |
| MTP=3 | 54 / 55 | 91 / 92 | 0.646 |
| DFlash2 | 54 / 55 | 91 / 92 | 0.670 |

All six requests report zero host groups and zero host fallbacks. Speculative
acceptance is 32/47 then 59/78 for MTP, and 39/43 then 58/80 for DFlash2. These are
single observations. Output length and acceptance differ; no repeated-run mean
or general throughput throughput claim follows from this comparison.

## Compilation conditions

The compiler counters are process-cumulative. First-request totals cover both
prefill and decode; follow-up values below are deltas of those cumulative totals.
There are zero disk-cache hits in all modes.

| Mode | First cumulative compiles / JIT s | Follow-up additional compiles / JIT s |
| --- | ---: | ---: |
| Baseline | 254 / 2.003 | 8 / 0.209 |
| MTP=3 | 345 / 2.656 | 2 / 0.143 |
| DFlash2 | 348 / 2.632 | 9 / 0.173 |

The first decode and follow-up have different compilation and resident-state
conditions. These data do not isolate JIT or any one kernel as the cause of the
rate difference. No CPU-sampled or passive-profile wall times are substituted
for these normal HTTP measurements.

## BF16 model-quality check

One later-build measurement scored exactly 1,024 next-token targets from the
pinned public WikiText2 corpus used by the earlier FP32/FP16 pair. Token IDs,
checkpoint config/index and tokenizer identities match those references.

| Configuration | Scored targets | CE | PPL |
| --- | ---: | ---: | ---: |
| Earlier FP32 KV / scalar Flash12 |1,024|1.590237269669|4.904912577255|
| Earlier FP16 KV / WMMA v1 |1,024|1.589521984382|4.901405419912|
| Current BF16 KV / generalized WMMA v2 |1,024|1.578957440359|4.849896867834|

The BF16 run selected all 32 `attention.flash.wmma16.v2` routes across two carried
M512 passes, at actual table capacities 512 then 1,024. All 254,279,680 logits were
finite; 3,142 device groups and zero host/fallback groups were reported. The owner
exited 0 using the mapped b7 HSA runtime and a fresh private cache.

This checks teacher-forced prefill in 512-token chunks. It does not test sampled
conversation quality or single-token decode. The earlier pair used its own
frozen executable; this BF16 point is a later build, not a contemporaneous
three-arm rerun or evidence of general quality improvement. FP8/BF8 retain
component coverage without a model-quality result.

## Provenance

- HTTP executable SHA256: `635ccc66422d2fb769f8f98730c6d52c37e5d6414136775f68f10870d62b50b1`.
- Mapped HSA SHA256: `b7f8216e32fa6ab0e5b2fce87c518ce67c3fffa4c628cceece983c53e3cea90d`.
- BF16 quality harness SHA256: `b0bf254d84c56020f2c74ecb5a3c962308b4a32f827d2bef9e01333d90ab5972`.
- Pinned token IDs SHA256: `0c4fa61cd7f49f2533b241c6624b079907468eb7a9e06901fbec8effb09c34df`.

Local evidence under `mac_amdgpu/build/release/pi-performance/`:
`bf16-mode-comparison.json`, `baseline-bf16-wave-views-cold/`,
`mtp3-bf16-wave-views-cold/`, `dflash-bf16-wave-views-cold/`, and
`kv-bf16-quality.{json,log}` plus its build/run manifests. The mode folders retain
requests, responses, metrics, cache state, commands and runtime/binary identities.
See [native WMMA checks](flash-wmma-general-2026-09-28.md) and the
[earlier corpus/quality method](kv-storage-attention-2026-09-28.md).
