# Earlier release and development measurements

These records describe their original source, requests and sampling settings.
Use the current README and final mode report for the latest controlled comparison.

## v0.5.3: refusable CPU fallback and the current mac_linuxgpu runtime

1. **`--no-cpu-fallback`** on `lse` and `lse-server`: startup fails when no device backend comes up or `--pool` names a CPU device, and a request fails when an operation has no device kernel. Each error names the cause. Without it, CPU fallback is logged, counted and reported in `lse_warnings`.
2. **No CPU-only run with the option:** without `--pool`, a GPU backend that fails to start no longer leaves the model on the CPU interpreter. `LSE_REQUIRE_DEVICE_KERNELS=1` has the same effect as the option.
3. **iOS build** against mac_linuxgpu build 247 and later (owner and selector calls; tested on 246 and 247). The HSA runtime is built under `build/ios/hsa`, so the mac_linuxgpu checkout is only read. `SKIP_HRX=1` links the newly built runtime.

## v0.5.2: model-defined limits, thinking levels and sampling defaults

1. **No default output limit:** a reply runs to an end-of-turn token, a stop sequence, the request's `max_tokens`, a `generation_config` limit or a full context. The 256 and 4096 defaults are gone.
2. **Full context:** `finish_reason: "length"`, `stop_reason: "context_full"` and `lse_context` on every response. A prompt with no room left gets HTTP 400 with `code: "context_full"`.
3. **Thinking levels** are read from the model's chat template and listed in `/v1/models`; an undefined level is refused, not aliased.
4. **Sampling defaults** come from `generation_config.json`, are reported with their sources, and use neutral defaults otherwise. `top_k`, `min_p` and `presence_penalty` are accepted on every request surface.

## v0.5.1: iPad prefill fixes

1. **loomc:** four passes doubled arrays on every append; a prefill kernel asked for 16 GiB and iPadOS refused it. Fixed; compiles peak around 150 MiB.
2. **Host memory:** weight host mirrors are released after upload, 3.0 GiB down to about 150 MiB.
3. **Fallbacks:** the device-path fallback is removed; CPU fallback stays, reported and optional (`--no-cpu-fallback`). On an iPad Pro (M4) with an R9700, a full agent turn decoded at **68.9 tok/s** at 94% acceptance, 20.5 GiB peak.

## v0.5.0: libLSE, iPadOS, DFlash2 BF16 loading and memory estimates

1. **libLSE:** the engine as an in-process library behind a plain C API ([`include/lse/lse.h`](../include/lse/lse.h)): `lse_open`, `lse_request` with the server's JSON, `lse_cancel`, `lse_status`, a log callback and an optional HTTP adapter. `lse-server` is a thin command line over it, and both transports answer identically.
2. **iOS and iPadOS:** `scripts/ios/build-ios.sh` builds `LSE.xcframework`. On an iPad Pro (M4) with an R9700 over Thunderbolt, through mac_linuxgpu's embedded driver, Qwen3.8-27B Q4 with the Q8 DFlash2 draft decoded at **40.5 tok/s** warm at 73% acceptance (the M5 Max Mac measured 43.3 tok/s at 74% with v0.4.24) and **57–70 tok/s** at 85–93%, with about 34 s model load and **0** CPU fallbacks.
3. **DFlash2 BF16 checkpoints:** `--dflash2-model` converts a BF16 draft to Q8/group64 once, bit-identical to `scripts/convert_dflash2_q8.py`, and caches the result.
4. **Model info and memory estimates:** `lse_model_info`/`--model-info` and `lse_estimate`/`--estimate`, plus `/v1/lse/model_info` and `/v1/lse/estimate` for a running server. Weights, KV, recurrent state, RoPE and the DFlash2 ring follow the allocators' own rules; a host-backend load of Qwen3.8-27B Q4 with its Q8 drafts matches within 0.001%. `/v1/models` reports `context_length`, `kv_len` and `kv_cache_dtype`.

## v0.4.24: HumanEval+ through 32K and driver HSA runtime

1. **HumanEval+ through 32K:** FlashPrefill V2 (alpha 0.1) prefill is **51–55% faster at 32K** across Baseline, MTP3 and DFlash2 on Qwen3.8-27B Q4, R9700/HRX/LOOM, over **1,368 completed generations**. Correctness matches dense prefill at Standard and 32K, with one additional failure per mode at 16K. [Results and data](benchmarks/flashprefill-humaneval-32k.json).
2. **Decode refinements:** single-token RMS normalization on gfx1201 keeps each row in registers across the reduction with **bit-identical output**; the single-token Q4 FFN down projection (5120×17408) now uses activation panels; and the submission tuner accepts repeatable wins of **2%** (previously 5%) when the sample ranges separate.
3. **macOS uses the driver's HSA runtime:** the macOS package **no longer bundles libhsa-runtime64**, which could shadow the driver's copy and fail to find the GPU. LSE now loads the HSA runtime installed by the GPU driver ([mac_linuxgpu](https://github.com/lemonade-sdk/mac_linuxgpu)) from `/usr/local/lib` or `/Library/MacAMDGPU/runtime`. Install or update the driver before upgrading.

## v0.4.23: FlashPrefill V2

1. **Faster long-context prefill:** FlashPrefill V2 reached **632.1 pp/s at 16K** and **604.9 pp/s at 32K** on Qwen3.8-27B Q4, R9700/HRX/LOOM. Default on for supported configurations; use `--FlashPrefillV2=off` for dense prefill.
2. **MTP and DFlash2 support:** at 16K, prompt throughput reached **582.6 pp/s with MTP3** and **615.2 pp/s with DFlash2**. Both on/off pairs matched the 64-token output and aggregate acceptance. Drafting and verification stay dense. Includes approximately **5× faster block selection** and Q4 SwiGLU bias-tail unrolling.
3. **Previously observed DFlash2 peaks:** **67.6 tok/s decode** and **96% acceptance**. These are separate workload peaks; the new sparse tests establish prefill gains, not a decode speedup. Perplexity is not yet measured.

## v0.4.22: faster Q4 prefill

1. Faster Q4 prefill with optimized nibble expansion and paired activation staging: **597.9 pp/s peak** on a warm full prompt with BF16 KV and no prompt cache reuse.
2. Fused eligible single-token Q4 SwiGLU gate/up projections and activation; updated dispatch regression coverage.
3. Fixed RDNA4 prefetch address/span lowering in bundled Loom. Previously observed live DFlash2 peaks: **67.6 tok/s decode** and **96% acceptance** on Qwen3.8-27B Q4, R9700/macOS. These are separate workload peaks.

[Prefill measurements](benchmarks/q4-prefill-2026-09-30.md).

## v0.4.21: real-world interactive performance

Real-world interactive use on the local macOS R9700 (`gfx1201`), running
Qwen3.8-27B Q4 with the Q8 DFlash2 block-8 drafter through HRX/Loom, measured:

- **Decode: 35.8 tokens/s overall**, with a **36.1 tokens/s median** per request
  and an observed range of **20.5–67.6 tokens/s**.
- **Prefill: 243.6 tokens/s overall**, ranging from **32.5–390.2 tokens/s**
  across short follow-ups and larger input chunks.
- **DFlash2 acceptance: 86% median** per request, ranging from **59–96%**.

These summarize user-supplied server logs. Overall rates are total logged tokens
divided by total phase time (approximately, because printed times are rounded).
Prompt counts represent newly processed input, not the full retained context;
short follow-ups include request overhead. Decode timing excludes the first token.
The results describe this live workload, not a fixed-context benchmark or a
before/after speedup measurement.

[Release details and downloads](https://github.com/Geramy/LSE/releases/tag/v0.4.21).

## v0.4.19: long-context KV growth and split attention

Consumed prefill chunk graphs and the scheduler's previous program are released
before a paged KV pool moves. Pools above 32K tokens grow in 4K-token steps to
limit allocation pressure. The local R9700 completed a 65,354-token prompt
with BF16 KV and Q8 DFlash2, followed by a 32-token decode request. The test
peaked at 29.72 GB of reserved device memory. A shorter deterministic response
matched v0.4.18 exactly. See the
[long-context report](benchmarks/kv-growth-2026-09-29.md).

One split partial and one merge kernel family now serve both single-token
decode and short verification queries. A matched cold 65,126-token Q4+DFlash2
request improved from 8.51 to 14.57 decode tokens/s; prefill measured 147.7
versus 147.0 prompt tokens/s. Both runs produced the same 64-token response.
See the [attention comparison](benchmarks/long-context-attention-2026-09-29.md).

## v0.4.18: smaller weight slabs

The weight allocator now uses 512 MiB slabs on macOS HRX, avoiding large
unused tails in the Q4 target and Q8 DFlash2 draft allocations. Matched driver
counters fell by about 1.71 GB after load and after a request. Repeated warm
requests measured 545.5 versus 548.3 prompt tokens/s and 49.9 decode
tokens/s for both builds. Output and acceptance matched. This does not resolve
the remaining long-context workspace and allocation high-water. See
[the slab report](benchmarks/weight-slab-memory-2026-09-29.md).

## v0.4.17: completed prefill workspace retirement

This release fixes overlapping target activation workspaces when ragged prefill
changes chunk width, and retires completed verifier graphs before a new request.
DFlash2 retains one wide context program during prefill and releases completed
wide programs afterward. Narrow decode replay, live state, KV and compiled
kernels remain available.

The matched 6143-token request reduces reserved VRAM from 29.77 to 28.26 GB.
Output and acceptance results match exactly. Prefill and decode rates differ
by less than 0.5% in that comparison. Eight cached follow-ups also preserve
output, acceptance and prefix reuse, with only 12.14 MB reserved-memory growth.
The reported allocation failure has not yet been replayed. See
[the memory report](benchmarks/prefill-workspace-memory-2026-09-29.md).

## Development: full-width DFlash2

The current development build verifies all seven proposals from the block-8
checkpoint. Conditional drafting uses the request temperature and preserves
target sampling through probability-ratio rejection and residual sampling.
Both prefill batch limits default to 1024. Use `--batch-size 2048 --ubatch-size 2048`
to select larger passes. The HTTP server accepts `--temperature 0.6`; an explicit
request temperature overrides it.

On exact 1024- and 2048-token coding requests at temperature 0.6, the second
requests measured **501.38 / 508.09 prompt tokens/s** and **35.04 / 34.53 decode
tokens/s**, respectively. Both generated 384 tokens, reused no prompt KV and had
zero host fallbacks. The process retained compiled code, but further compilation
remained inside these timings. Different temperatures produce different answers.
These are workload measurements, not general rates or achieved 103 TPS/600 PP/s
goals. Published v0.4.13 includes the full-width changes and the M8 optimization below.
See [full-width DFlash2 results](benchmarks/dflash-fullwidth-2026-09-29.md).

The next M8 Q4 down-projection optimization measured **40.21 TPS** on the
second 1024-token request, versus **35.42 TPS** for its matched control.
Prefill remained about **500 prompt tokens/s**. Both use seven proposals,
temperature 0.6 and BF16 KV. A 1024-target perplexity comparison was
4.8660 versus 4.8675. Completed prefill workspace is released before decode
to fix the reproduced second-request allocation failure at batch 4096.
See [M8 WMMA and memory results](benchmarks/wmma-m8-down-2026-09-29.md).

The next M1024 down-projection schedule measured **522.58 prompt tokens/s**,
versus **507.24** for the matched control. Decode was effectively unchanged:
**40.78 versus 40.55 TPS**. Both responses and acceptance statistics match exactly.
This prefill change is newer than v0.4.13. See
[M1024 prefill results](benchmarks/prefill-m1024-down-2026-09-29.md).

WMMA attention now skips entire windows that are outside the row or fully
causally masked. The component time falls 30.74% at 1024 live keys and 7.45%
at 5207 live keys. The matched resident HTTP request reaches **526.26 prompt
tokens/s**; decode remains about **40.7 TPS**. Responses and sampling statistics
match the control exactly. See
[attention window results](benchmarks/attention-causal-windows-2026-09-29.md).

Cooperative activation staging for M1024 gate/up now reaches **568.03 prompt
tokens/s**, versus **526.26** for its matched control. Decode remains **40.73 TPS**.
Both responses and acceptance statistics match exactly. The change preserves
the accepted M8 and M1024 down emitted kernels and launch plans. See
[cooperative prefill results](benchmarks/prefill-m1024-up-2026-09-29.md).

A separate comparison uses one binary and the same 1024-token requests,
temperature 0.6, BF16 KV and batch/ubatch 1024. Each mode starts with an empty
private disk cache. The resident request reuses compiled code but no prompt KV.

| Mode | Resident prompt tokens/s | Resident decode tokens/s | Acceptance |
| --- | ---: | ---: | ---: |
| Baseline | 572.30 | 24.79 | — |
| MTP=3 | 560.78 | 48.93 | 80.70% |
| DFlash2, seven proposals | 568.03 | 40.73 | 76.15% |

These source changes follow published v0.4.13. The performance targets remain
unmet. This coding request does not establish rates for long-context Pi chat.
See [mode comparison and timing limits](benchmarks/forward-modes-2026-09-29.md).

The same cooperative schedule now covers the exact M1024 QKV, GDN gate and
attention projection shapes. The matched resident request reaches **619.08
prompt tokens/s**, versus **568.03** before these entries. Decode remains
**40.68 TPS**. Responses and acceptance statistics match exactly. This exceeds
600 PP/s on this controlled 1024-token workload, not at every context length.
See [projection results and validation](benchmarks/prefill-projections-2026-09-29.md).

The M8 gate/up and QKV DOT4 schedules now consume two activation rows at a time.
The matched resident DFlash2 request improves from **40.68 to 42.41 decode
tokens/s**. Prefill remains above 600 at **616.78 prompt tokens/s**. Responses,
usage and sampling statistics match exactly. All previously accepted schedules
are retained. See [M8 scheduling results](benchmarks/m8-dot4-rowpairs-2026-09-29.md).


## v0.4.12: typed attention and automatic buffer views

This version adds typed WMMA attention, cooperative single-token attention, and
automatic contiguous buffer views. Eligible slices no longer launch copy kernels.
K/V storage defaults to BF16 for models that declare BF16, and FP16 otherwise.

Single-token split attention now shares K/V loads across six query heads.
The matched component check reduced BF16 attention time by 49% at 5.2K live
keys and 53% at 14K. Empty partitions publish neutral records and return early.
The submission tuner skips its end wait when the policy did not change.

The measurements below use the same Q4 target, BF16 KV, binary/runtime, and
requests. Each mode starts with an empty private kernel cache. First prefill
includes compilation. Sampling uses model defaults 1/20/0.95. The first prompt
has 5,207 tokens; the follow-up has 5,262 cached tokens and 23 new tokens.

| Mode | First prefill, tokens/s | First decode, tokens/s | Follow-up decode, tokens/s |
| --- | ---: | ---: | ---: |
| Baseline | 344.21 | 22.58 | 23.82 |
| MTP=3 | 313.32 | 24.74 | 40.25 |
| DFlash2 | 328.65 | 26.40 | 37.53 |

Baseline follow-up increased from 22.82 to 23.82 tokens/s in the matched check.
The speculative modes remain near their previous rates. Responses match their
preceding controls, with zero host groups or fallbacks. These are individual
workload measurements, not a general throughput guarantee. See the
[shared-KV decode report](benchmarks/shared-kv-decode-2026-09-28.md) for
component results, first/follow-up timing limits, and CPU sampling costs.

## v0.4.11: split decode attention and MTP prompt reuse

The monolithic single-token attention kernel is removed. Split attention now
covers long contexts without the old 4K capacity cutoff. On the recorded Pi
requests, plain decode measured **19.71 / 19.72 tokens/s**, compared with
12.91 / 12.70 before the change. The sampled responses changed.

MTP now retains verified state between requests. Follow-up prefill fell from
**14.25 s to 0.71 s**, with 5,310 cached tokens and 23 new tokens. Recorded decode
measured **32.68 / 31.90 tokens/s with MTP=3** and **27.97 / 35.26 tokens/s with
DFlash2**. The two modes produced identical responses in these candidate runs.
First prompt rates were **293.95 and 291.75 tokens/s**, respectively.
These are individual Pi workload measurements, not a 46–48 tokens/s guarantee.
See the [methods and limits](benchmarks/pi-execution-profile-2026-09-28.md#remove-monolithic-single-token-attention-and-retain-mtp-sessions).

## v0.4.10: shorter compilation stalls and optional KV formats

Identical Loom kernel bodies now share compiled code. Attention page traversal
stays in generated control flow instead of expanding each page into source.
In the same two-turn Pi conversation with an empty starting cache, total JIT
compilation fell from **92.91 s to 3.35 s**. First prefill fell from **91.53 s to
16.88 s**; follow-up prefill fell from **16.14 s to 0.76 s**. Both responses and
proposal acceptance counts matched. The final paired-load build measured **28.59 / 34.03 decode tokens/s** and
**310.72 prompt tokens/s** on those turns. Scratch allocation is now preserved
in optimizer facts and reported by dispatch profiling.
See the [execution profile](benchmarks/pi-execution-profile-2026-09-28.md).


## v0.4.9: chat sampling and prompt reuse

The CLI and server load each model's supported settings from `generation_config.json`.
If fields are missing, LSE uses embedded model settings and the fallback table under
`src/models`. Explicit CLI flags and HTTP request parameters override these defaults.
See [sampling defaults](SAMPLING.md) for precedence and supported fields.

Pi can return streamed reasoning without invalidating the previous prompt cache.
The HTTP response reports reused tokens in `usage.prompt_tokens_details.cached_tokens`.
In a two-turn Qwen3.8 Q4 + Q8 DFlash2 chat, the second prefill fell from 14.11 s
for 5296 tokens to 0.93 s for 23 new tokens after the serialization fix.
With the model's temperature 1, top-k 20 and top-p 0.95 settings, decode measured
23.98 and 28.82 tokens/s on the two turns. These measurements describe those
requests; output length and proposal acceptance affect the result.
See the [Pi chat report](benchmarks/pi-chat-2026-09-28.md).

## v0.4.8: lower VRAM use

KV growth previously kept obsolete recurrent-state graphs and their buffers alive.
The fix releases those graphs while keeping the current state and compiled kernels.
On one 14K Q4 + Q8 DFlash2 request, live allocations fell from **30.72 to 22.33 GiB**.
Sampled peak driver reservations fell from **30.93 to 24.86 GiB**.

The performance check measured 323.41 prompt tokens/s and 42.41 decode tokens/s
on the second identical request, with temperature 0 and 100% proposal acceptance.
**These are warmed synthetic results, not expected Pi or coding-session rates.**
The first request measured 15.51 decode tokens/s even without new kernel compilation;
request setup and reuse remain under investigation.
See the [memory report](benchmarks/kv-growth-memory-2026-09-28.md).

## v0.4.19

This version releases consumed prefill graphs before paged KV growth and uses
smaller growth steps above 32K tokens. The local R9700 with a Q4 target, Q8
DFlash2 draft and BF16 KV completed a 65,354-token prompt and a subsequent
32-token decode request. Peak reserved GPU memory in the extension was 29.72 GB.
One split-attention kernel family now covers single-token decode and short
verification queries. In a matched cold 65,126-token HTTP request, DFlash2
decode increased from 8.51 to 14.57 tokens/s; prefill stayed near 147 tokens/s.
The output text matched. See the
[attention comparison](docs/benchmarks/long-context-attention-2026-09-29.md).
The configured KV limit is not a measured usable capacity; see the
[long-context test](docs/benchmarks/kv-growth-2026-09-29.md).
The v0.4.18 weight-allocation reduction and v0.4.17 workspace-retirement fix
remain active; see [weight slab measurements](docs/benchmarks/weight-slab-memory-2026-09-29.md) and
[prefill memory ownership](#prefill-memory-ownership).
