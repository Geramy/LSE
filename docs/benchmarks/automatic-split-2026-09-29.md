# Automatic split attention at long context

## Change

Short-query attention derives its partition count from the actual K/V table
capacity. The selector checks partial-buffer indexing, partial LDS, and merge
LDS requirements instead of imposing a 65,536-key limit. Four-query tiles and
empty-partition skipping apply to compatible widths of four through eight rows
at capacities of at least 8,192 keys. Existing shorter-context choices remain.

## Isolated GPU check

Radeon AI PRO R9700 (gfx1201), BF16 K/V, eight queries, 65,656 live keys,
69,632-key table, permuted physical pages, 64 warm repetitions. These timings
include host submission and synchronization in the same executable.

| Execution | Mean time |
| --- | ---: |
| Contiguous split attention | 7.510 ms |
| Fragmented split attention | 8.122 ms |
| Forced unsplit attention | 71.790 ms |

Contiguous and fragmented split outputs matched bit for bit. A double-precision
CPU reference checked the first head's first and last query, across all 256
output components: maximum absolute error was 7.65e-10 for split attention and
1.02e-6 for unsplit attention. This reference check did not cover every head.

## HTTP check

One cold 65,536-token coding request, 128 generated tokens, Q4 target, Q8
DFlash2 with seven proposals, BF16 K/V, temperature 0.6, top-k 20, top-p 0.95,
seed 1234, batch and microbatch 1024. No host groups or host fallbacks occurred.

| Metric | Preserved earlier executable | Automatic split executable |
| --- | ---: | ---: |
| Prefill | 146.10 PP/s | 145.92 PP/s |
| Decode | 5.723 TPS | 13.870 TPS |
| Verifier per speculative step | 660.684 ms | 258.365 ms |
| Draft acceptance | 78.05% | 79.67% |

Both responses contained the same 579 characters, byte for byte. Decode timing
covers 127 tokens after the initial token. The observed TPS ratio is 2.424.
The preserved executable's complete source lineage is unavailable, and the
executables contain other revision differences. Do not attribute the entire
HTTP improvement to this change. The isolated check above tests the attention
choice directly. Neither result predicts throughput for every prompt or context.

Host dispatch tests cover capacities through 262,144 keys and LDS boundaries.
Those tests do not establish that the complete model fits or performs well at
that context length. No perplexity test was run for this dispatch change.
