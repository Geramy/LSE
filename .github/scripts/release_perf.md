## Long-context fixes

Q4 WMMA now handles supported prefill batches from 16 rows upward, including
remainder batches previously routed through scalar kernels. Architecture and
shape policies are stored in typed dispatch header tables. Short-query split
attention covers up to 16,384 keys, and HTTP requests reuse matching consumed
prefixes with DFlash2.

On a synthetic 5,610-token request, first-process prefill fell from 112.813 s
to 28.558 s. At 14,000 live tokens, the second decode measured **34.95 tok/s**
with 100% draft acceptance on repetitive text. These samples do not predict
Pi-agent throughput. Retained KV capacities triggered additional cold kernel
compilation, so the second prefill was not a warmed throughput measurement.

The macOS Loom dependency also clears only touched symbolic memo entries.
Six exact attention sources compiled in **71.542 s instead of 117.065 s**
(38.89% less time), with byte-identical GPU code objects. Those measurements
used one matched pair per source. The HTTP measurements above predate this
compiler fix.

See [the long-context report](https://github.com/Geramy/LSE/blob/master/docs/benchmarks/long-context-http-2026-09-28.md)
for workload boundaries, isolated GPU timings, CPU diagnosis and validation.
No new model perplexity or logits-L2 sweep was run for these changes.

## Flash12 key reuse

The accepted Flash12 implementation reuses each key load across twelve query
rows. At query batch 512, isolated attention GPU time fell from
**99.457 to 25.202 ms** at 5,610 live keys and from **232.793 to 93.772 ms**
at 14,000 live keys. Complete output arrays match bit for bit; allocated
registers and LDS are unchanged in those cases, with zero scratch. These are
component timings, not end-to-end engine speedups.

See [the Flash12 report](https://github.com/Geramy/LSE/blob/master/docs/benchmarks/flash12-key-reuse-2026-09-28.md)
for matched GPU timings, independent reference checks and emitted-code details.

## Earlier short-context qualification

Recorded on **gfx1201 / AMD R9700**, Apple Silicon with Thunderbolt 5,
Qwen3.8-27B-Q4, native HRX/Loom, MTP off. The warmed matched request used a
1,024-token prompt and 64 generated tokens, two warmups and three measured
requests per configuration, with flush 64 and 64 us completion polling.

| Stage | Qualified Q4 profile |
|---|---:|
| Prefill | **444.532 PP/s** |
| Decode | **24.2635 TPS** |

These are recorded measurements of the accepted qualified INT8 profile,
including split-128 WG128 decode. They establish **400+ PP/s and 23+ TPS**
on this system. The measurement predates the final automatic profile selection
and wave32 normalization promotion; it is not a benchmark of the release archive.

The checkpoint-qualified Q4 INT8 M1/M512 paths, Q4 M512 FFN LDS v2,
FP32 Flash12 prefill, shared-exponential FP32 decode attention, and split-128
WG128 decode are selected by default where qualified. Other kernels and operand
types remain available with their existing diagnostic controls. Unknown
checkpoints retain the exact FP32 path. Floating-point accumulation stays FP32;
integer dot products accumulate in INT32 before FP32 affine restoration.

**Quality uses perplexity.** New qualification windows contain 1,024–2,048
actual target tokens. The 2,046-target automatic Q4 prefill result was 8.127312,
matching explicit INT8 selection, versus 8.100851 for FP32 activations (+0.3266%).
The 1,024-target automatic teacher-forced result was 7.20144530149 versus
7.17831687877 with explicit INT8 (+0.3222%). These are separate fixtures and are
not compared with each other. The accepted Q6 staged-BF16 M512 paths retain
their documented historical 1,022-target qualification; FP8/BF8 alternatives
remain inactive.
