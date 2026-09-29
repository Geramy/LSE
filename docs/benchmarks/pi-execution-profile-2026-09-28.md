# Pi HTTP execution profile — 2026-09-28

## Workload and measurement

This profile uses two consecutive Pi chat turns with the local Qwen Q4 target,
Q8 DFlash2, depth 3, FP32 KV storage, and model sampling defaults. The first
prompt has 5,207 tokens. The second turn reuses 5,310 tokens and adds 23 tokens.
The GPU is an AMD Radeon AI PRO R9700 (`gfx1201`, wave32).

The engine source is `29279e1`; the server SHA-256 is
`16370d88719a10a0357dcb490a9e0f418bf6d88fd83647d6920da31b94842102`.
The mapped-transfer HSA library was confirmed in the process image. Its SHA-256
is `b7f8216e32fa6ab0e5b2fce87c518ce67c3fffa4c628cceece983c53e3cea90d`.

The disk kernel cache was empty before server start. Both turns use the same
process. LSE submit profiling and host spans were enabled. Passive GPU dispatch
timestamps and four CPU stack samples were collected. Per-dispatch forced waits
were disabled. There were zero host kernel groups and zero CPU fallbacks.

The capture contains 255,950 GPU dispatches, 427 distinct entries, and 18.724 s
of summed GPU execution, including startup and both requests. Every dispatch
entry maps to an LSE submit record. Device timestamps use a 100 MHz clock.

## Measured GPU costs

| Operation | Dispatches | GPU execution, ms |
| --- | ---: | ---: |
| Prefill Q4 FFN up/gate | 1,664 | 3,737.7 |
| Prefill Q4 projections | 2,704 | 2,634.5 |
| Attention, across phases | 272 | 2,173.6 |
| Prefill Q4 FFN down | 832 | 1,680.8 |
| Decode Q4 FFN up/gate | 13,696 | 1,588.9 |
| Decode Q4 projections | 22,256 | 1,224.5 |
| GDN chunk scan, across phases | 8,880 | 924.6 |
| Decode Q4 FFN down | 6,848 | 764.2 |
| Attention partial/merge, across phases | 3,296 | 628.6 |
| Decode vocabulary head | 204 | 421.6 |

Q4 projections and FFN operations account for about 62% of total recorded GPU
execution. The largest individual kernel is the M512/N17408/K5120 Q4 FFN
projection: 1,280 dispatches and 3,592.0 ms total.

Prefill and decode labels for quantized operations come from their actual
matrix row count. Other operations can serve both phases. These durations are
execution sums, not GPU occupancy or the percentage of wall time spent busy.

## CPU work and completion waits

| Inference-thread capture | Stack samples | Inside compiler | Inside HSA wait |
| --- | ---: | ---: | ---: |
| First prefill | 9,490 | 83.8% | 1.5% |
| Follow-up prefill | 2,484 | 87.7% | 4.7% |
| First decode | 2,545 | 17.1% | 55.1% |
| Follow-up decode | 2,590 | 0.0% | 84.1% |

These are wall-stack samples of the inference thread, not process CPU-use
percentages. A completion wait includes time while queued GPU work executes.
The 84.1% figure does not mean that 84.1% is an avoidable sleep.

LSE records 97,880.4 ms in compilation, 2,875.4 ms in JIT lookup, 962.6 ms in
emission, 571.6 ms in binding, and 528.7 ms in submission. Completion waits
account for 5,039.1 ms and readback for 1,577.0 ms. The last two can include GPU
execution and must not be added to GPU execution times as independent overhead.

Instrumentation changes request timing. This capture is used to locate costs,
not to advertise a new prompt or decode rate.

## Optimizations selected from the evidence

### Keep attention page traversal in generated control flow

The existing source expanded the page traversal at C++ generation time. Keeping
it as a device loop reduces the representative M16 attention source from
1,441,864 to 246,432 bytes (19,622 to 3,806 lines). Direct compilation falls
from 14.55 s to 0.191 s.

Seven native cases match every baseline output bit and pass masks, guards and
empty-partition checks. The cases issue 390 device dispatches with no CPU
fallback. Alternating measurements show no material steady execution change:

| Case | Previous GPU mean, ms | Page-loop GPU mean, ms |
| --- | ---: | ---: |
| M16, 8K capacity | 1.9113 | 1.9045 |
| M512, 5K context | 24.8243 | 24.7566 |
| M512, 14K context | 95.3847 | 95.5180 |

This change targets compilation stalls. The 14K measurements show clock drift
and overlapping distributions; they do not establish a steady GPU speedup.

### Share compiled code for identical generated bodies

A separate frozen 427-source FP16 corpus contains 372 unique bodies after only
the self-export name is normalized. Thus 55 graph identities generate duplicate
code. Compiled artifacts can be shared while invocation bindings, launch
metadata and structural optimizer measurements stay distinct. Exact source is
checked on memory and disk cache hits.

The structured emission cache still avoids repeated code generation. Sharing
compiled code does not merge graph bindings or remove shape distinctions that
change the actual generated computation.

### Reject activation preparation that slows the complete operation

A separate Q4 M512 activation-panel experiment removes consumer LDS and
barriers. It passes exact panel and output checks, but the preparation plus
consumer chain is slower:

| Operation | Existing mean, ms | Experiment mean, ms | Change |
| --- | ---: | ---: | ---: |
| FFN up | 2.5102 | 4.5592 | +81.6% |
| FFN down | 2.9689 | 3.1455 | +5.9% |

The rejected implementation is removed. Lower LDS use alone is insufficient
reason to retain an optimization.

## Integrated HTTP verification

The rebuilt server uses both compiler changes. The same two-turn workload runs
with FP32 KV, the same mapped HSA library, and an empty disk cache. Instrumented
CPU/GPU tracing is disabled for these request measurements; LSE host spans remain
enabled in both baseline and candidate.

| Measurement | Earlier baseline | Rebuilt server |
| --- | ---: | ---: |
| First prefill, s | 91.529 | 16.876 |
| First prefill, tokens/s | 56.89 | 308.54 |
| Follow-up prefill, s | 16.138 | 0.760 |
| Total JIT compilation, s | 92.911 | 3.351 |
| Unique compilations | 427 | 372 |
| First decode, tokens/s | 26.64 | 27.11 |
| Follow-up decode, tokens/s | 31.34 | 31.28 |

Both assistant responses, generated token counts, proposal counts and acceptance
counts match exactly. There are zero host groups and zero CPU fallbacks. The
candidate server SHA-256 is
`2f7d63bff665a1955e2c374f370f34962700ea3692d905596800543fed80e62f`.
These are individual paired workload measurements. Model loading is outside
prompt timing. They establish a compilation-stall improvement, not a steady
decode speedup. No additional perplexity test was run for these exact changes.

The coherent build passes all nine compiled-cache host cases. Frozen-corpus
replay compiles 372 bodies, reuses 55 aliases, then performs zero new compiles or
loads on a second pass. A fresh owner loads 372 objects from disk with zero
compilations. Of the final 15 emissions, seven reuse existing bodies and eight
require new bodies.

Thirteen of fourteen focused host CTest suites pass. The remaining `test_jit`
suite passes 108 of 111 cases; three fixtures require RDNA3.5 or CDNA3 target
descriptors that are absent from this local Loom build. No target policy or test
expectation was changed to conceal those failures. The actual gfx1201 HTTP run
above verifies native execution on the installed GPU.

## Remove repeated DFlash feature-graph traversal

The steady decode sample contains 33 of 2,590 inference-thread samples in
`Program::retain` while appending verified features to DFlash context. The
entire append-context branch contains 75 samples. It constructs a slice of an
already-materialized contiguous feature buffer, then traverses the target graph
to materialize that slice again. The 65 corresponding slice dispatches consume
only 0.329 ms total GPU time; the avoidable cost is mainly host planning.

A bounded buffer view now retains the allocation and its offset, residency and
owner while detaching the old computation graph. The helper validates the full
source extent and storage state. It does not change recurrence, context cursors,
readback ordering or required GPU completion waits.

All 25 DFlash host cases pass, including offsets, ownership lifetime, opaque
buffer mirrors, invalid bounds, rejection and continuation. A native two-turn
comparison reproduces both responses and all proposal counts exactly. It removes
19 device groups in the first turn and 46 in the second, with no host fallback.
Unique compilations fall from 372 to 369.

| Measurement | Before buffer view | After buffer view |
| --- | ---: | ---: |
| First prefill, s | 16.876 | 16.846 |
| Follow-up prefill, s | 0.760 | 0.756 |
| First decode, tokens/s | 27.11 | 27.20 |
| Follow-up decode, tokens/s | 31.28 | 32.13 |

This single pair shows a small improvement; it is not a statistical speedup
claim. The required completion waits remain. Of the 2,178 steady samples inside
HSA waits, 1,572 occur while completing target verification and 446 occur on
draft-result readback paths. These waits depend on pending GPU results.

## Pair adjacent M4 activation-panel loads

The M4 profile identifies two additional hot projections, N10240/K5120 and
N6144/K5120, totaling 6.423 ms per target verification body. The existing Q4
consumer now reads two neighboring activation chunks together: 16 bytes of
packed codes and 8 bytes of FP32 step bits per row. Weight loads, integer DOT4
order, FP32 accumulation and affine restoration, K partitioning, wave reduction
and fused epilogues retain their arithmetic order.

The central shape table selects paired loads only for the following three
M4/K5120 shapes under existing gfx1201/wave32 Q4/group64 panel admission.
Each component comparison uses one warmed A/B/B/A sequence of 20 projections
per arm. Both arms include the same activation-preparation kernel. These CP
means sum preparation and consumer execution, excluding compilation and host
validation.

| M4 projection | Current CP mean, ms | Paired-load CP mean, ms | Reduction |
| --- | ---: | ---: | ---: |
| N17408/K5120, SiLU | 0.122309 | 0.098667 | 19.33% |
| N10240/K5120 | 0.085419 | 0.059385 | 30.48% |
| N6144/K5120 | 0.054874 | 0.038126 | 30.52% |

Both candidate arms beat both baseline arms for each selected shape. Baseline
drift is visible; these component results do not establish an HTTP throughput
gain. The N5120/K17408 down experiment increases CP execution from 0.111918 to
0.114716 ms (+2.50%), so down retains its current implementation. Other shapes,
row counts, formats and devices also retain their current selections.

All 768 native dispatches pass complete bit comparisons against the current
consumer, an independent activation codec and contraction reference, finite
outputs, unchanged inputs and allocation guards, with zero host/fallback
launches. Tests include non-dyadic distinct-row activations, zero chunks,
BF16 affine scales/biases, N17 tails, SiLU, SiLU*external input and two residual
adds. Each owner starts fresh after clearing the disk cache; cold compilation
is outside the warmed timing.

The selected consumers reduce VGPR use from 66 to 53, with unchanged launch
geometry and zero declared LDS or scratch bytes. The producer is unchanged.
These resource counts do not measure occupancy. Implementation cache keys
include the selected load width; no new registered alternative or buffer
layout is introduced. No additional model-quality test or HTTP gain is claimed
for this exact arithmetic-order change.

## Detect private scratch allocation in optimizer facts

The native code-object reader previously missed MessagePack uint8 and uint32
values. It could also mistake an integer payload byte for a string boundary.
The corrected reader preserves fixed private bytes per work-item, LDS bytes
per workgroup, register counts and required launch dimensions. Missing or
truncated metadata remains unknown. Positive private allocation is not proof
of register spills or a count of executed memory traffic.

A resource-reader revision is stored separately from compiler identity. An old
cached object stays reusable. Its stale resource facts are withheld from the
optimizer until the object loads, then the metadata is refreshed from the
existing bytes and saved without compilation. Later starts can preload the
corrected facts.

Occupancy and arrangement comparisons now retain private allocation. Among
legal alternatives, a known zero-private option wins over one with known
private allocation before modeled occupancy or traffic is compared. Two
positive allocations only break a modeled tie. Unknown values do not supply
zero. A settled fusion admission can be demoted once after measurements prove
a private-memory or register-spill regression against usable alternatives;
already-compiled Programs are unchanged.

The current Loom emitter does not use that fusion-admission path. Its active
JIT does record resource facts, and existing `LSE_PROFILE_DISPATCH=submit` or
`serial` profiling now reports positive private allocation and confirmed spills
when the canonical object loads. The record includes the entry, architecture,
artifact and source hash, typed allocation units, and unknown spill counts.
This detects the problem in active Loom code; it does not invent an alternative
kernel or establish a throughput gain from scratch avoidance.

## Final native HTTP check

The canonical Release build includes the paired M4 loads and corrected resource
metadata. The same two-turn run starts with an empty kernel cache and uses the
same mapped HSA library, sampling defaults, FP32 KV and Q8 DFlash2 depth 3.
Both complete responses, token counts, proposal counts and acceptance counts
match the feature-view baseline exactly. Device group counts are unchanged;
there are zero host groups and zero CPU fallbacks.

| Measurement | Feature-view baseline | Paired M4 loads |
| --- | ---: | ---: |
| First prefill, s | 16.846 | 16.758 |
| First prefill, tokens/s | 309.10 | 310.72 |
| Follow-up prefill, s | 0.756 | 0.747 |
| First decode, tokens/s | 27.20 | 28.59 |
| Follow-up decode, tokens/s | 32.13 | 34.03 |
| Unique compilations | 369 | 369 |
| Total JIT compilation, s | 3.379 | 3.269 |

Decode rate increases 5.12% and 5.90% in this single comparison. Component GPU
measurements above support the kernel selection; this is not a repeated-run
statistical claim. The server SHA-256 is
`f37492b16ae84fe74aef24e8c40e321f8bca2f5854d4978972bdd035b45b1b50`.

All 369 active code objects report fixed private allocation of zero with the
corrected reader. All omit allocator spill counts, which remain unknown.
Six objects report 164–186 VGPRs. The result finds register pressure to inspect;
it does not find fixed private scratch as the cause of the remaining slowdown.

The coherent build passes eight of nine focused host suites. `test_jit` passes
120 of 123 cases, including all six new resource-reader cases and all six new
optimizer cases. Its three failures are the same unavailable RDNA3.5/CDNA3
compiler-target fixtures described above. The cache migration test verifies
one metadata refresh, no compilation, and reuse of refreshed facts on a later
startup. No additional perplexity run was used.

## Remaining limit

The actual Pi workload has not established 451 prompt tokens/s or 46–48 decode
tokens/s. Cold compilation, steady GPU execution and speculative acceptance
are separate measured costs. Short warmed synthetic tests must not be presented
as the performance of this conversation workload.
