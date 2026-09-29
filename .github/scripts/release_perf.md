### Measured source and workload

These measurements use source `c11f103c86627e66c36d1b109fb02281c40ad686` and server SHA256 `4c635d610f6bfb7545b81ededa5a356f1b2a1b5389bdd28d736a5d24837a88dd`. They describe that measured build. A later build needs its own comparison before it can claim the same rates.

macOS/R9700 gfx1201, Qwen Q4 target, Q8 auxiliary/draft weights, BF16 KV, temperature 0.6, top-k 20, top-p 0.95, seed 1234, batch/ubatch 1,024 and configured KV capacity 262,100. Each 1,024-token coding request generates 384 tokens, with 383 timed for decode. MTP uses depth 3. DFlash2 uses all seven proposals in block 8.

Each mode starts with an empty private kernel cache. The resident request retains compiled code and reuses zero prompt KV. Compilation is included. No host groups or fallbacks were recorded. One pair per mode is not a statistical estimate or a long-context Pi replay.

| Mode | Cold PP/s | Cold TPS | Resident PP/s | Resident TPS |
|---|---:|---:|---:|---:|
| Baseline | 442.18 | 24.14 | 624.10 | 24.73 |
| MTP=3 | 403.95 | 37.96 | 608.19 | 48.62 |
| DFlash2, seven proposals | 417.14 | 31.39 | 616.78 | 42.41 |

All three controlled resident requests exceed 600 PP/s. The 29 TPS baseline, 49 TPS MTP and 103 TPS DFlash2 goals remain unmet. Rates vary with prompt, sampling and live context.

The matched projection table change improved resident DFlash2 prefill from 568.03 to 619.08 PP/s (+8.99%). The subsequent M8 row-pair schedule improved resident decode from 40.68 to 42.41 TPS (+4.27%). Each scoped comparison preserved exact responses and speculative statistics. Different generation modes can produce different answers.

[Final methods and results](https://github.com/Geramy/LSE/blob/c11f103c86627e66c36d1b109fb02281c40ad686/docs/benchmarks/forward-modes-final-2026-09-29.md), [M8 schedule](https://github.com/Geramy/LSE/blob/c11f103c86627e66c36d1b109fb02281c40ad686/docs/benchmarks/m8-dot4-rowpairs-2026-09-29.md), [prefill projections](https://github.com/Geramy/LSE/blob/c11f103c86627e66c36d1b109fb02281c40ad686/docs/benchmarks/prefill-projections-2026-09-29.md).

### Retained optimizations and compatibility

Full-width conditional DFlash2 sampling, completed prefill workspace retirement, the qualified M8 down WMMA path and the faster paired DOT4 projection schedules remain active. Floating-point accumulation remains FP32. Compact Q4 weights remain unchanged. The cooperative M1024 panels and uniform masked-attention window skip passed complete output checks; measured component kernels have zero private scratch. No additional perplexity run was added for these bit-identical schedule changes.

The macOS archive bundles HSA, HRX, Loom and its runtime dependency closure. Use `bin/lse` or `bin/lse-server`; install and activate MacAMDGPU separately. Linux uses the bundled HRX/Loom runtimes and compatible installed ROCm/HSA. Each archive includes a source/runtime manifest and checksum.

K/V defaults to model-declared BF16 for BF16 checkpoints and FP16 otherwise, with explicit overrides. Batch/ubatch 1,024 and seven-proposal DFlash2 remain defaults. `--temperature` sets the server default; request settings take precedence. Launch examples use 0.6.

### Later v0.4.15 DFlash2 snapshot

Source `cb285b136fbb7d45b23ce4d0ffc7f0dfb0a4d665`, local server SHA256
`b78f36a45355f5cade9bf5b960f5080e5e6607fc3e86e5eccd14a79c8795329f`.
One matching DFlash2 cold/resident pair measures 413.74 PP/s / 32.01 TPS cold
and 616.38 PP/s / 43.05 TPS resident. Complete responses and acceptance statistics
match v0.4.14; zero host groups/fallbacks and zero prompt KV reuse. The real GPU
cache uses the release namespace. Resident decode differs by about +1.5% in one
pair, not a statistical estimate or long-context Pi guarantee. Baseline/MTP were
not rerun; their table above remains explicitly the older v0.4.14 snapshot.
See `docs/benchmarks/m8-gdn-http-2026-09-29.md`.
