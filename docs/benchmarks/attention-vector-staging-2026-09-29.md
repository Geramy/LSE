# Vector staging for paged attention

Measured on Radeon AI PRO R9700 (gfx1201), macOS, 2026-09-29.

## Change

The Flash WMMA attention kernel loads V with contiguous 16-byte loads and stages a 16-by-128 tile in padded LDS. WMMA reads its operands from that tile. The FP32 accumulation order is unchanged. The LDS requirement includes the staging buffer, so dispatch admission still checks the complete allocation.

A typed Loom load policy requests low retention in the near cache and high retention in the far cache. The attention shape table enables this policy for gfx1201 with at least 1,024 query rows and 65,536-token capacity. It applies to K/V data loads only. Pointer-table reads, stores, and short split-attention kernels retain their existing policy. The hint does not reserve or pin cache space.

The gfx1201 L2 fallback metadata is corrected from 4 MiB to 8 MiB. This metadata correction is not assigned a separate performance gain.

## HTTP comparison

Each measurement starts a new server with an empty kernel cache. Both binaries use the same saved request and seed, Q4 target, Q8 DFlash2 with seven proposals, BF16 K/V, temperature 0.6, top-k 20, top-p 0.95, and batch/ubatch 1,024. Each request generates 128 tokens; server decode timing counts 127 tokens after the first.

| Context | Metric | Baseline | Optimized |
|---:|---|---:|---:|
| 1,024 | Prefill, tokens/s | 413.16 | 408.43 |
| 1,024 | Decode, tokens/s | 27.75 | 27.55 |
| 65,536 | Prefill, tokens/s | 146.06 | 224.02 |
| 65,536 | Prompt time, seconds | 448.70 | 292.54 |
| 65,536 | Decode, tokens/s | 13.85 | 13.79 |

At 64K, prefill throughput increases 53.4% and prompt time decreases 34.8%. Decode is essentially unchanged. At 1K, prefill is 1.15% lower and decode is 0.70% lower in this single comparison. These are individual measurements, not statistical confidence intervals or guarantees for other prompts.

Both comparisons have byte-identical generated text, identical acceptance counts by proposal position, identical dispatch counts, and zero CPU fallback. Acceptance is 87.18% at 1K and 79.67% at 64K. No perplexity test was run because this change preserves the kernel's arithmetic.

## Isolated attention measurement

The synthetic fragmented BF16 fixture uses 1,024 query rows, 65,520 live keys, and capacity 65,536. Each candidate is timed over eight launches. These are host submission plus completion times.

| Kernel | Mean time per launch |
|---|---:|
| Baseline, first measurement | 911.09 ms |
| Vector staging | 396.42 ms |
| Vector staging and cache hint | 365.69 ms |
| Baseline, repeated measurement | 916.33 ms |

Every output element matches the baseline bit for bit. A double-precision reference checks all 256 components in the first head's first and last query; maximum absolute error is 7.27e-7.

The final kernel uses 136 VGPRs, 44 SGPRs, and 29,120 bytes of LDS, with zero scratch memory and zero spills. Its ISA contains four 128-bit global load sites and no scalar half-load sites. The additional LDS is 4,352 bytes. Register count alone is not an occupancy measurement.

## Validation and provenance

- 22 emitter and attention tests pass, including LDS limits, cache-policy selection, and separation of data loads from pointer-table loads and stores.
- Native FP32, FP16, BF16, FP8, and BF8 cases pass with causal/sliding masks, padded rows, and ragged dimensions. Narrow and packed formats include widths 20 and 256. These tests require GPU dispatch and reject CPU fallback.
- The production-generated source matches the measured hinted source after normalization of the export name.
- Baseline server SHA-256: `47ff1130028137154b11f2eae365be4216fdb49a8467a145685656603e972af4`.
- Optimized server SHA-256: `3043362f4a52880c2f1e780f312619766e7c7fd9419877e60bf4f140020f2b87`.
- 64K request SHA-256: `903fb7607c827013f0c5adf5473cc0485756cc03a1e106c32baa7a8eb69bfe29`.
- The optimized binary uses the same local build libraries as the baseline plus this attention and cache-policy change. Independent Linux HIP repairs are not included in these measured binaries.

An initial integration measurement was discarded after its generated source showed that a stale build object omitted the staging change. The object was rebuilt, the source and ISA were inspected, and all integration checks and HTTP measurements reported here used the rebuilt binaries.
