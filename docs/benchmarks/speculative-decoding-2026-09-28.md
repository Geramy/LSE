# Speculative decoding measurements — September 28, 2026

## End-to-end results

Native gfx1201 on macOS, 64 CUs, wave32, 32,367 MiB reported device memory. Local Q4 27B target, Q8 MTP module, and Q8 DFlash2 drafter. HTTP completions use a fixed 1,024-token prose prompt, temperature zero, and 64 generated tokens. The first generated token is excluded from decode timing, leaving 63 timed tokens. Each mode receives one warmup followed by two measured requests. HSA blocked polling is 32 microseconds. No profiler is enabled in this table.

| Mode | Earlier release TPS | Current warm TPS | Mean TPS | Final prompt tokens/s |
|---|---:|---:|---:|---:|
| Plain Q4 | 24.22 / 24.18 | Not rerun | 24.20 baseline | 466.26 baseline |
| MTP, three proposals | 23.12 / 23.18 | 36.50 / 36.56 | 36.53 | 467.20 |
| DFlash2, four-row target verification | 15.15 / 15.16 | 38.07 / 38.38 | 38.23 | 471.47 |

Both current measurements include shared Q4 activation panels. The panel change preserved the complete generated continuation and acceptance counters in both modes. Relative to the immediately preceding snapshots, MTP improved from 26.74 to 36.37 TPS (+36.0%) and DFlash from 28.04 to 36.84 TPS (+31.4%). The initial FFN-only panel checkpoint reached 33.14 and 33.94 TPS; extending the same kernel to five measured projection shapes added 9.7% and 8.5%, respectively. Earlier release and current continuations differ, so comparison to the earlier-release column is a fixed-input measurement rather than an identical-output claim. The current MTP and DFlash continuations match for these 64 generated tokens.

The subsequent two-barrier RMS change and seven-row DFlash vocabulary panel reached the current table results. The complete continuations and acceptance counters remained unchanged. Final DFlash draft time fell from 413.12 to 357.23 ms; verification was 1,245.60 ms. MTP draft and verification were 339.32 and 1,383.66 ms. Relative to the preceding seven-shape panel checkpoint, mean DFlash throughput improved 3.77%; MTP increased 0.44%.

The measured requests compile no new kernels. The prompt column reports the final warm request; cold startup and initial warmup are excluded. This run loaded the installed `/Library/MacAMDGPU/runtime/libhsa-runtime64.dylib` without a `DYLD_LIBRARY_PATH` override.

For DFlash, reducing target verification from eight rows to four lowered verification time from 3,251 ms to 1,807 ms. The trained drafter still evaluates all eight positions; only the returned proposal prefix is shorter. Both measured schedules used 22 target passes and accepted 41 proposals. The final split-attention change preserved the continuation and acceptance counters, reducing draft time from 440 ms to 400 ms. The shared-panel change then reduced verification from 1,809 to 1,406 ms, preserving 41 of 53 accepted proposals and 22 target passes. MTP verification fell from 2,030 to 1,558 ms, preserving 38 of 54 accepted proposals and 25 target passes. The five-shape extension further reduced verification to 1,391 ms for MTP and 1,257 ms for DFlash, with the same counters and continuations.

The requested 43 TPS MTP and 88–105 TPS DFlash targets have not been achieved.

## Implemented changes

- Submit pending device work before recording ordered-copy dependencies.
- Keep MTP hidden states on the device between target and draft passes.
- Commit accepted recurrent-state prefixes without rerunning their full FFN layers.
- Refresh prebound reshape aliases through normal scheduler completion and replay.
- Coalesce short target-attention key loads across a wave.
- Use a circular DFlash context cache and split draft attention across key windows.
- Reuse Q8 matrix fragments across draft rows and stage four quantization groups per preparation round where measured faster.
- Transpose top-k scratch storage to distribute neighboring lanes across shared-memory banks.
- Default MTP to three proposals and DFlash to three verified proposals plus the anchor.
- Prepare exact Q4 activation codes once per shared input and reuse them across seven measured M4 FFN, attention/recurrent projection and vocabulary-head shapes. Paired global code-word loads remove repeated local staging while preserving DOT4 and FP32 restoration order.
- Preserve packed U32 storage through typed terminal output and CPU references, with capability-dependent dispatch cache keys.

Floating-point accumulation remains FP32. Integer matrix and DOT4 products use INT32 with FP32 scale/bias restoration. Q8 matrix activation quantization can change draft proposals; the target verifier remains authoritative.

## Focused evidence

| Component | Baseline | Candidate | Scope |
|---|---:|---:|---|
| Top-k, 7 × 248,320, k=16 | 2.936 ms | 0.600 ms | Full retained replay and drain |
| Draft attention, 1,024 live keys | 0.502 ms | 0.210 ms | Partial plus merge, retained replay |
| Draft attention, 2,055 live keys | 0.812 ms | 0.443 ms | Partial plus merge, retained replay |
| Q8 query projection, M8/N4096/K5120 | 0.764 ms | 0.092 ms | Native dispatch wall time |
| Q8 output projection, M8/N5120/K4096 | 0.772 ms | 0.093 ms | Native dispatch wall time |
| Q4 up, M4/N17408/K5120 | 0.1930 ms | 0.1444 ms | Original vs shared-panel preparation plus consumer |
| Q4 down, M4/N5120/K17408 | 0.2326 ms | 0.1388 ms | Original vs shared-panel preparation plus consumer |
| Shared-panel down, paired code-word loads | 0.1397 ms | 0.1132 ms | Scalar vs paired loads, including preparation |

Kernel measurements exclude compilation and validation from warm timing. Native fixtures checked independent component references, complete outputs, allocation guards and input preservation, with zero host fallback. Top-k selected values and indices matched exactly, including ties, NaNs, infinities and ragged chunks. Split draft attention covered wrapped caches, live counts 17/1,024/2,055 and rewinds, with maximum absolute component error below 9e-8.

Focused host gates passed: prefix recovery 6/6, scheduler 5/5, Q8 dispatch 4/4, target attention 4/4, top-k 4/4, DFlash 12/12, MTP options 3/3, server 7/7. Runtime tests recorded 56 passed and 5 skipped. No new perplexity sweep or L2 model-quality test was run for this checkpoint.

The production panel gates compared all seven enabled shapes and fused residual epilogues against the registered original kernel bit for bit. Independent activation-code references, complete outputs, input preservation and allocation guards passed with zero host fallback. Typed-storage and initial panel host fixtures each passed 5/5 in GPU-enabled and CPU-only builds. The extended panel fixture passed 6/6, including cross-projection reuse and all seven emission/cache cases. The CPU-only macOS DFlash fixture passed 9/9 after its linker dependency repair. No new perplexity sweep was run for this exact-arithmetic change.

Additional exact component evidence includes preparation and consumer GPU time:

| M4 projection N/K | Original ms | Shared panel ms | Reduction |
|---|---:|---:|---:|
| 10,240 / 5,120 | 0.10269 | 0.07129 | 30.6% |
| 6,144 / 5,120 | 0.06362 | 0.04661 | 26.7% |
| 12,288 / 5,120 | 0.12235 | 0.08251 | 32.6% |
| 5,120 / 6,144 | 0.06248 | 0.04283 | 31.5% |
| 248,320 / 5,120 | 2.67618 | 1.59546 | 40.4% |

## Remaining cost

LSE submission/step timing and GPU dispatch timestamps show FFN projections consume about 44–45% of target GPU execution time after the M4 panel changes. GPU timestamps measure dispatch duration, not occupancy. The requested 32-microsecond policy is a blocked completion-observation interval; it is not a fixed delay inserted after every kernel. Active spinning did not improve the earlier end-to-end comparison.

The shared-panel path is active for the seven measured M4 shapes. Alternate matrix, weight-layout and workgroup experiments remain inactive; additional projection shapes are not enabled without measurement. The macOS release dependency pin includes the committed 32-microsecond HSA policy; existing release archives retain their previously bundled libraries.

## Seven-row vocabulary panel and RMS reduction

The DFlash vocabulary projection M7/N248320/K5120 now shares one activation panel and computes eight rows per workgroup, masking the unused row. Its original K partition and FP32 restoration/reduction order remain unchanged. Central dispatch selects only the measured shape, with a separate implementation identity. The original four-row paths retain their identities.

ABBA20 native measurements, including preparation, reduced GPU execution from 5.1550 to 2.9710 ms (42.37%) and wall time from 5.1677 to 2.9895 ms. Both candidate samples beat both originals. The consumer uses 92 VGPRs, no LDS and no private scratch. The production raw and fused-residual cases preserved all 1,738,240 output bits, panel codes, input contents and allocation guards, with zero host fallback. Focused panel tests passed 7/7.

RMS normalization preserves its 256-lane FP32 tree while using two workgroup barriers instead of nine. Six native shapes passed exact outputs and guards across 516 device dispatches; focused production tests passed 7/7. GPU median time changed from 6.52 to 6.26 microseconds for M4/K5120 and from 35.20 to 33.44 microseconds for M512/K5120. The small-shape wall measurements were noisy; these component results are distinct from the HTTP measurements above.

These changes are newer than the v0.4.5 release archive. No additional perplexity sweep was run.

## Six-row panel coverage

The same rows-eight consumer now covers all seven measured six-row projections in the central shape table. The original K partitions, activation codec and FP32 restoration order are preserved. Across 1,251 private native device dispatches, full outputs, independent component references, panel codes, input preservation and guards passed with zero host fallback. The integrated graph/emission/cache fixture passed 8/8.

| Six-row projection N/K | Original GPU ms | Panel + consumer GPU ms | Reduction |
|---|---:|---:|---:|
| 17,408 / 5,120 | 0.279111 | 0.147690 | 47.09% |
| 5,120 / 17,408 | 0.434995 | 0.162687 | 62.60% |
| 10,240 / 5,120 | 0.173718 | 0.099233 | 42.88% |
| 6,144 / 5,120 | 0.102697 | 0.059113 | 42.44% |
| 12,288 / 5,120 | 0.195377 | 0.105805 | 45.85% |
| 5,120 / 6,144 | 0.104534 | 0.061012 | 41.63% |
| 248,320 / 5,120 | 4.233792 | 2.438142 | 42.41% |

These component wins do not establish a wider-verification throughput win. A preliminary five-proposal HTTP candidate reached 32.81 / 32.67 TPS versus 38.16 / 38.42 at three proposals. It required 18 rather than 22 verifier passes, but six-row target attention fell through to Flash8 because the short-attention default table omitted that width. The profiler attributed 303.7 ms (22.9% of verifier GPU execution) to those six-row attention calls. This schedule remains under investigation; the six-row kernel improvement is independent of selecting it as the default. No new perplexity sweep was run.

## Complete short-attention width coverage

The short-verifier routing table no longer enumerates selected query counts. It admits the continuous supported range of two through eight rows for the existing target geometry and 1,024/2,048-key capacities. The omitted widths therefore cannot fall through to Flash8 in this scope. One-row decoding retains its existing optimized decode kernel.

| Queries / capacity | Flash8 GPU ms | Split + merge GPU ms | Reduction |
|---|---:|---:|---:|
| 2 / 1,024 | 0.964462 | 0.079893 | 91.72% |
| 2 / 2,048 | 2.015485 | 0.158861 | 92.12% |
| 5 / 1,024 | 0.988058 | 0.158902 | 83.92% |
| 5 / 2,048 | 1.974116 | 0.297555 | 84.93% |
| 6 / 1,024 | 0.988858 | 0.281579 | 71.52% |
| 6 / 2,048 | 1.970753 | 0.377763 | 80.83% |

Native ABBA20 measurements exclude three initial warmups per arm and the final empty replay. All six cases passed full independent double-precision component references, empty replay, input preservation and allocation guards: 792 device dispatches, zero host fallback. The focused host routing/emission suite passed. This change reuses the existing short-attention implementation; no new quantization arithmetic or perplexity sweep was introduced. Integrated throughput is reported separately from these component measurements.
