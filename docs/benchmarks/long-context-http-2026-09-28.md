# Long-context HTTP investigation — 2026-09-28

## Workload and baseline

Synthetic 5,610-token prose prompt, Qwen3.8-27B Q4 target and Q8 DFlash2, three proposals, greedy sampling, 16 generated tokens, declared KV capacity 262,100. This is not the supplied Pi conversation; its complete tool outputs and sampling settings were unavailable. The original executable is preserved in `baseline-bin` with SHA-256 `6d04d87f8cc9c18f1b9d56a3f1aecbdffa7c8a47080f3d9276f9bc5dad5f9298`.

The first diagnostic request used the built-in LSE spans and dispatch profiler. It reported 38,742 device groups and zero host groups. It compiled 291 kernels for 143.165 seconds total; those cold compilation costs are included in its 239.034-second prompt and 17.516-second decode wall timings. These are not warmed throughput measurements.

| Remainder batch | Summed scalar Q4 GPU execution |
|---|---:|
| 256 rows | 46.697 s |
| 128 rows | 23.483 s |
| 64 rows | 11.848 s |
| 32 rows | 6.030 s |
| Total | 88.058 s |

These are sums of dispatch durations, not GPU occupancy or nonoverlapping wall time. The same workload spends about 8.720 summed seconds in attention. Declared KV capacity does not itself scan or allocate all 262,100 entries: live demand grows the pool through capacity rungs.

## Changes published in `43e8d7d`

- Admit group-64 Q4 WMMA for prefill remainder rows on gfx1201, preserving structural and resource validation. The arithmetic is the same implementation used for the existing 512-row path.
- Admit short-query split attention through its safe 128-part/16,384-key merge limit.
- Retain consumed DFlash2 target and draft state at request completion, and let HTTP reuse exact consumed token prefixes. A changed prefix cold-starts. Fully cached repeated prompts are rescored because a request-local Generator has no retained final logits.
- Centralize architecture, quantization and shape tuning rows in dispatch headers.

## Focused native evidence

| Operation | Previous | Candidate |
|---|---:|---:|
| Q4 FFN up, M64 | 50.001 ms | 0.468 ms |
| Q4 FFN down, M64 | 12.138 ms | 0.531 ms |
| Q4 FFN up, M256 | 201.725 ms | 1.539 ms |
| Q4 FFN down, M256 | 47.647 ms | 1.217 ms |
| Attention M4, live 5,610 / table 8,192 | 5.460 ms | 0.818 ms |
| Attention M4, live 14,000 / table 16,384 | 12.573 ms | 2.171 ms |

Q4 entries are ABBA host elapsed means. Attention entries are CP device means for the full partial-and-merge operation. They are isolated operations, not end-to-end token rates. Native checks cover independent arithmetic references, bounds, preserved inputs, poisoned padding and empty attention rows. Q4 candidate rows are bit-identical to the existing approved 512-row implementation. No model L2 or perplexity sweep was run.

## CPU interpretation

macOS 100% process CPU corresponds to one logical core's execution capacity. Worker threads already exist. Sleeping GPU waits appear in wall-time stack samples but do not represent equivalent CPU execution. The source and samples include graph preparation, first-use kernel compilation/loading, HSA queue service, and sampling; no CPU inference fallback was found. Request CPU deltas and warmed HTTP results are recorded separately in JSON.

## Baseline with disk-cached kernels

Same 5,610-token request: prompt 112.813 s (49.73 tok/s), decode 2.795 s (5.37 tok/s). Zero new compilations, 506 disk object loads. This is the first request in a new process, so it includes executable loading and graph setup; it is not an in-process warm decode. CPU 25.32 s / elapsed 115.61 s = 21.90% average.

## Integrated HTTP results

LSE source commit `43e8d7d`. All five requests reported zero host groups and zero host fallbacks. Nine focused CPU suites passed, including 22 DFlash2 cases; the final attention header extraction passed both attention suites.

| Request | Fresh prompt tokens | Prefill seconds | Decode tok/s | New compile seconds | Mean process CPU |
|---|---:|---:|---:|---:|---:|
| full-prefill | 5610 | 28.558 | 5.38 | 1.174 | 26.6% |
| warm-full-prefill | 5610 | 148.507 | 26.48 | 128.242 | 97.4% |
| continuation | 9 | 1.028 | 12.37 | 0.065 | 29.0% |
| context-14000-first | 14000 | 131.618 | 31.73 | 49.323 | 52.5% |
| context-14000-warm | 14000 | 155.616 | 34.95 | 73.778 | 61.5% |

Despite the harness filenames containing `warm`, the second full-prefill requests encountered new retained-table shapes and were not compilation-free. Prefill timing includes those compilation costs. The two 14K decode samples generated 64 tokens, counted 63 decode tokens and had 100% draft acceptance on repetitive synthetic text; these rates are not a prediction for Pi sampling or arbitrary prompts. The continuation retokenized the previous response plus an appended request and reused the consumed prefix.

The second 5,610-token request compiled six new attention variants and ten small KV-write objects. The earlier request had encountered remainder widths at small KV pool capacities; its successor reused the 8,192-token pool. The six new Flash sources are approximately 1.01–1.45 MB. A CPU sample put 2,319 of 2,351 generation-thread stacks inside Loom compilation, dominated by symbolic proof memo clearing. This is distinct from CPU tensor fallback and is under investigation in the compiler dependency; the published dispatch changes do not resolve this compilation cost.

## Compiler memo reset

A dependency backport tracks memo entries touched by each symbolic proof and resets only those entries. Fact-only cache writes are tracked too. It preserves all proof passes and optimization decisions.

| Attention source rows | Original compile | Touched-entry reset |
|---|---:|---:|
| 2 | 13.433 s | 7.715 s |
| 8 | 13.573 s | 7.830 s |
| 32 | 22.332 s | 13.905 s |
| 64 | 22.459 s | 14.038 s |
| 128 | 22.608 s | 13.971 s |
| 256 | 22.661 s | 14.083 s |
| Total | 117.065 s | 71.542 s |

These are isolated cold compiles of the six exact retained-KV attention sources, using the same machine and build configuration, with one matched pair per source. Compile time fell 38.89%; all six emitted code objects are byte-identical to the original compiler output. The candidate passed 33 symbolic-expression tests, including sparse reset, opposite-branch fact isolation and allocation failure. This is a compiler latency improvement, not a measured increase in GPU execution speed. The HTTP timings above predate this compiler change.

## Follow-up attention implementation

[Flash12 key reuse](flash12-key-reuse-2026-09-28.md) subsequently reduced isolated
M512 attention GPU time by 74.66% at 5,610 live keys and 59.72% at 14,000 keys,
with bit-identical complete outputs. The HTTP measurements in this report
predate that kernel change as well as the compiler backport.
