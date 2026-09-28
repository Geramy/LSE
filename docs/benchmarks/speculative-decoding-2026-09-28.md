# Speculative decoding measurements — September 28, 2026

## End-to-end results

Native gfx1201 on macOS, 64 CUs, wave32, 32,367 MiB reported device memory. Local Q4 27B target, Q8 MTP module, and Q8 DFlash2 drafter. HTTP completions use a fixed 1,024-token prose prompt, temperature zero, and 64 generated tokens. The first generated token is excluded from decode timing, leaving 63 timed tokens. Each mode receives one warmup followed by two measured requests. HSA blocked polling is 32 microseconds. No profiler is enabled in this table.

| Mode | Earlier release TPS | Current warm TPS | Mean TPS | Final prompt tokens/s |
|---|---:|---:|---:|---:|
| Plain Q4 | 24.22 / 24.18 | Not rerun | 24.20 baseline | 466.26 baseline |
| MTP, three proposals | 23.12 / 23.18 | 26.82 / 26.65 | 26.74 | 452.62 |
| DFlash2, four-row target verification | 15.15 / 15.16 | 28.07 / 28.02 | 28.04 | 464.14 |

The MTP measurement predates the later draft-only attention/top-k changes and the general scheduler fix replacing equivalent model-local view refresh. The DFlash measurement includes those changes. Earlier release and current continuations differ, so these are fixed-input measurements rather than claims of identical-output speedup. The four-row DFlash continuation matches current MTP for these 64 generated tokens.

For DFlash, reducing target verification from eight rows to four lowered verification time from 3,251 ms to 1,807 ms. The trained drafter still evaluates all eight positions; only the returned proposal prefix is shorter. Both measured schedules used 22 target passes and accepted 41 proposals. The final split-attention change preserved the continuation and acceptance counters, reducing draft time from 440 ms to 400 ms. Final verification time was 1,809 ms; 41 of 53 compared proposals were accepted.

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

Floating-point accumulation remains FP32. Integer matrix and DOT4 products use INT32 with FP32 scale/bias restoration. Q8 matrix activation quantization can change draft proposals; the target verifier remains authoritative.

## Focused evidence

| Component | Baseline | Candidate | Scope |
|---|---:|---:|---|
| Top-k, 7 × 248,320, k=16 | 2.936 ms | 0.600 ms | Full retained replay and drain |
| Draft attention, 1,024 live keys | 0.502 ms | 0.210 ms | Partial plus merge, retained replay |
| Draft attention, 2,055 live keys | 0.812 ms | 0.443 ms | Partial plus merge, retained replay |
| Q8 query projection, M8/N4096/K5120 | 0.764 ms | 0.092 ms | Native dispatch wall time |
| Q8 output projection, M8/N5120/K4096 | 0.772 ms | 0.093 ms | Native dispatch wall time |

Kernel measurements exclude compilation and validation from warm timing. Native fixtures checked independent component references, complete outputs, allocation guards and input preservation, with zero host fallback. Top-k selected values and indices matched exactly, including ties, NaNs, infinities and ragged chunks. Split draft attention covered wrapped caches, live counts 17/1,024/2,055 and rewinds, with maximum absolute component error below 9e-8.

Focused host gates passed: prefix recovery 6/6, scheduler 5/5, Q8 dispatch 4/4, target attention 4/4, top-k 4/4, DFlash 12/12, MTP options 3/3, server 7/7. Runtime tests recorded 56 passed and 5 skipped. No new perplexity sweep or L2 model-quality test was run for this checkpoint.

## Remaining cost

LSE submission/step timing and GPU dispatch timestamps show FFN projections still consume about 55% of target GPU execution time. GPU timestamps measure dispatch duration, not occupancy. The requested 32-microsecond policy is a blocked completion-observation interval; it is not a fixed delay inserted after every kernel. Active spinning did not improve the earlier end-to-end comparison.

Private activation-panel and alternate matrix experiments are not active defaults in this checkpoint. The macOS release dependency pin includes the committed 32-microsecond HSA policy; existing release archives retain their previously bundled libraries.
