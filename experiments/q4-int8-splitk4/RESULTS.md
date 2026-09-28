# Q4 INT8 WMMA real split-K4 probe

## Result

Rejected for active use. Both candidate GPU timing arms are slower than both accepted global-panel arms for both actual M4 FFN shapes. No production source changed and no perplexity run was needed after this performance rejection.

| Shape | Active prep+DOT4 GPU ms | Split-K4 partial+merge GPU ms | GPU change | Active wall ms | Split-K4 wall ms |
|---|---:|---:|---:|---:|---:|
| M4 FFN up (17408×5120) | 0.111922 | 0.150186 | +34.19% | 0.128914 | 0.167139 |
| M4 FFN down (5120×17408) | 0.104908 | 0.156416 | +49.10% | 0.122084 | 0.173583 |

Timings are device CP sums of both kernels, with 32 warm calls per arm followed by ABBA20 (40 measured calls per arm). Wall measurements include submission and completion. HRX dispatch profiling was enabled; installed system HSA 32 us default, gfx1201 wave32, no library-path override. Raw weights are retained; one-time N16-major packing/upload is excluded from warmed timings.

## Correctness and launch structure

Four edge cases covered empty partitions at K64/K128, uneven partitions at K320, realistic K5120, N17 tails and M4 rows. Both actual full FFN shapes passed independent group64 activation-codec/component oracles for every partial and final output, finite/completeness checks, exact active chunk8 DOT4 panel oracle, readonly inputs and allocation guards. Edges issued 80 native dispatches; actual shapes issued 576 native dispatches. All were device launches with zero host/fallback.

The matrix path uses gridY4 for four real K partitions; the partial tensor is FP32[4,M,N], followed by a left-associated FP32 partition merge. Up uses 136×4 workgroups and down 40×4 instead of the original unsplit136/40. Partials use 72 VGPR,6656 bytes LDS and no scratch; the accepted panel uses 66 VGPR and zero LDS/scratch. Matrix activation quantization remains per group64, while the accepted DOT4 contraction uses its existing chunk8 codec. This experiment therefore does not claim bitwise equality between those two codecs.

## Provenance

The experiment copied the frozen `q4-int8-weight-layout-20260928` candidate and the accepted paired global-panel probe. This branch retains the source and compact CP/resource summaries. Large raw capture files and binaries remain in the local measurement directory. Original measured source and binary hashes are recorded in `manifest.json`. See `README.md` for an isolated reproduction.
