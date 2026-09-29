# KV storage and FP16 matrix attention — 2026-09-28

## Later BF16 quality point

The generalized BF16 KV/WMMA v2 family subsequently scored the same 1,024 pinned
targets at **PPL 4.849896867834** (CE 1.578957440359). All 32 M512 attention routes
selected v2 at capacities 512 and 1,024; all 254,279,680 logits were finite,
with 3,142 device groups and zero host/fallback groups. This is one later-build
prefill measurement against the historical references below. It does not
establish sampled-conversation quality or a statistical improvement. See the
[BF16 comparison](bf16-mode-comparison-2026-09-28.md) for provenance and scope.

## Earlier FP16 matched pair

One ordered pair compared the accepted FP32 KV/Flash12 path with **FP16 KV plus FP16-operand WMMA attention**, preserving FP32 accumulation/output. Both used the same frozen executable, checkpoint, public corpus and settings. This is a combined numerical change; the result does not isolate KV rounding from matrix-operand rounding.

| Arm | Scored targets | CE | PPL | Device groups | Host / fallback |
| --- | ---: | ---: | ---: | ---: | ---: |
| FP32 KV / Flash12 | 1,024 | 1.590237269669 | 4.904912577255 | 3,142 | 0 / 0 |
| FP16 KV / matrix attention | 1,024 | 1.589521984382 | 4.901405419912 | 3,142 | 0 / 0 |

PPL changed by -0.071503% on this corpus. BF16, FP8 and BF8 were outside this earlier two-arm quality comparison. FP32 was the default at that time; current model dtype policy is described in [KV cache formats](../KV_CACHE.md).

### Method and route proof

Exactly 1,025 natural-text token IDs yielded 1,024 next-token targets through two carried 512-input public `HybridLM::hidden` and native `lm_head` calls. All 254,279,680 logits were finite; CE/PPL and mixer cursors were checked. No CPU LM-head fallback, generator hooks or model-distance tests were used.

The FP16 arm selected `attention.flash.wmma16.f16.v1` for all 16 paged layers in each pass: 32 selections total, M512 at actual capacities 512 then 1024. The FP32 arm selected 32 `attention.flash.qtile12` routes. Materialized KV storage dtype, logical width, physical pitch and byte extent were asserted.

Both fresh owners exited 0. The entire relevant persistent kernel cache, including sidecars, was cleared before each process; both had zero disk hits. Cold compile totals were 51 objects/30,482.222ms (FP32) and 51/634.626ms (FP16). These quality-harness timings are not warmed HTTP throughput measurements.

### Corpus and binary provenance

Public dataset: [Salesforce WikiText](https://huggingface.co/datasets/Salesforce/wikitext), `wikitext-2-raw-v1` test shard pinned to revision `b08601e04326c79dfdd32d625aee71d232d685c3` (CC-BY-SA3.0/GFDL). Raw test rows [0,128) were newline-joined and tokenized by the public checkpoint tokenizer without special/chat tokens; the first 1,025 IDs were retained. The native tokenizer was required to reproduce them.

| Identity | SHA256 |
| --- | --- |
| Token IDs JSON | `0c4fa61cd7f49f2533b241c6624b079907468eb7a9e06901fbec8effb09c34df` |
| Public source text | `ea4b10a941594324412c8ab1ea4c496c4ea9257616b7bb373c8c6a2c6f19dcab` |
| Checkpoint tokenizer | `06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523` |
| Checkpoint config | `14b65a0ee06517060a6bbd979bb1a8ff54e7b304b1a1f01d54344b88b8285e85` |
| Sharded checkpoint index | `13b840162b4cb35c66fef7df072f7dbb4717908204364f5e5d9f9655a2758fa8` |
| Matched harness executable | `b9d6316d3bfd84d8c11bb1dcf639e0a0a8b027aebc2800f17e56ed6a3d489c6c` |
| Loaded mapped HSA runtime | `b7f8216e32fa6ab0e5b2fce87c518ce67c3fffa4c628cceece983c53e3cea90d` |

Both arms used `build/models/qwen38-27b-q4` and actual dyld-loaded `/private/tmp/lse-hsa-upload/libhsa-runtime64.0.1.0.dylib`. Matching canonical headers/archive hashes and exact compile/link commands are recorded in the build manifest.

## Storage footprint

For 16 full-attention layers with 4 KV heads and D256, including both K and V:

| Format | Bytes per allocated token | At capacity 16,384 |
| --- | ---: | ---: |
| FP32 | 131,072 | 2 GiB |
| FP16 / BF16 | 65,536 | 1 GiB |
| FP8 / BF8, including scales | 33,280 | 520 MiB |

The quality run's final capacity 1,024 used 128 MiB FP32 versus 64 MiB FP16. These totals exclude weights, recurrent state, MTP, DFlash's FP32 private ring and temporary buffers; capacity can exceed live length.

## Cold HTTP observation

One Pi two-turn comparison used a cleared persistent cache before each fresh process. This is an ordered startup/continuation observation, not a matched decode-throughput qualification.

| Observation | FP32 baseline | FP16 KV + WMMA |
| --- | ---: | ---: |
| First prefill | 91.529 s | 29.006 s |
| Follow-up prefill | 16.138 s | 0.742 s |
| Process-total JIT compile time | 92.911 s | 15.915 s |
| Reported kernel variants | 427 | 427 |
| Follow-up verifier time per pass | 58.639 ms | 59.595 ms |
| First decode tokens/s | 26.641 | 23.894 |
| Follow-up decode tokens/s | 31.337 | 27.100 |

The lower prefill wall time includes substantially lower cold compiler cost. It does not isolate a warmed prefill kernel gain. Generated text/token counts and speculative acceptance differed (follow-up acceptance 59.35% versus 67.86%), so decode TPS is not used to claim a performance improvement. Follow-up verification time per pass was roughly unchanged. FP16 remains an optional launch setting; FP32 remains the default. HTTP artifacts are under the canonical release Pi-performance `fp16-wmma-cold` comparison.

## Evidence

Private evidence root: `/private/tmp/lse-kv-storage`. `ppl-fp32.json` and `ppl-fp16.json` contain quality, finite/counter and per-pass route/storage checks; `ppl-pair-run.json` records owner/cache/runtime identity; `ppl-corpus-manifest.json` and `ppl-build-manifest.json` pin corpus/binary provenance. HTTP performance and the detailed native component table are recorded separately.

## Native attention component timing

Both arms below use FP16 KV storage. The matrix path uses FP16 operands and FP32
accumulators. GPU command timestamps use the device frequency of 100 MHz. Ten
measured calls per strategy follow warmups in ABBA order. These measurements
isolate attention execution from compilation and storage format.

| Queries | Live keys | Table capacity | Existing Flash ms | Matrix Flash ms |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 512 | 512 | 2.183832 | 1.129232 |
| 512 | 1,024 | 1,024 | 4.738216 | 2.388848 |
| 512 | 2,048 | 2,048 | 9.321944 | 4.806216 |
| 512 | 4,096 | 4,096 | 16.635180 | 9.757996 |
| 512 | 5,610 | 8,192 | 24.496884 | 13.304296 |
| 512 | 14,000 | 16,384 | 71.616336 | 39.754288 |
| 64 | 3,991 | 4,096 | 5.303560 | 2.623952 |
| 128 | 5,610 | 8,192 | 7.076668 | 4.508920 |
| 256 | 14,000 | 16,384 | 39.177532 | 23.003120 |
| 16 | 5,610 | 8,192 | 2.093932 | 1.809160 |

The 16-query case at capacity 2,048 was slower (0.676160 to 1.059816 ms), as was
the batch-three 17-query case (0.419480 to 0.572380 ms). The central shape table
excludes those scopes. All 12 native correctness cases passed, with 606 device
dispatches and zero host execution or fallback. The checks covered the independent
chosen-format reference, finite outputs, repeated output bits, empty replay,
read-only inputs and guarded allocations. Resource counts were 112–116 VGPRs,
24,768 LDS bytes and zero scratch. They do not establish measured occupancy.

For 16 queries at capacity 8,192, direct compiler time fell from 16,624.956 to
157.201 ms. Source size fell from 1,456,073 bytes / 19,879 lines to 63,588 bytes /
1,235 lines. This timing excludes server startup. Component evidence is under
`/private/tmp/lse-flash-wmma-integrate`.
