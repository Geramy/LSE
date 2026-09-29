# Earlier release and development measurements

These records describe their original source, requests and sampling settings.
Use the current README and final mode report for the latest controlled comparison.

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
