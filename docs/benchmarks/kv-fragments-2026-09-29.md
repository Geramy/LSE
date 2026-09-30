# K/V fragment storage — local validation

## Scope

This change affects paged attention K/V only. A scheduler-owned K/V manager
packs 256 KiB fragments into 256 MiB arenas. Layers share available slots on
the same backend and stream. Growth preserves existing K/V bytes and addresses;
attention and K/V-write kernels access the fragments through a device address
table. Weight, activation, recurrent-state, and general allocation policy are
unchanged. There is no CPU offload.

The final shared arena can be partially filled. Each K/V tensor can have an
unused tail smaller than 256 KiB. Existing cache capacity is reused across
session restarts, and arenas are freed after their final fragment lease ends.

## Full-engine memory check

Local R9700, Qwen3.8-27B Q4 target, Q8 DFlash2, BF16 K/V, FP32 attention
accumulation, batch and microbatch 1024, temperature 0.6, top-k 20, top-p 0.95,
seed 1234, configured context limit 262100. Two requests run in one server.
Both generated 128 tokens. The second prompt replaces the generated continuation,
so its reported cached-token count is zero.

| Request | Prompt tokens | PP/s | Timed decode tokens | TPS | Draft acceptance |
| --- | ---: | ---: | ---: | ---: | ---: |
| First | 61,452 | 165.88 | 127 | 26.21 | 97.3% |
| Follow-up | 68,173 | 139.31 | 127 | 10.10 | 100% |

Both completed with zero host groups and zero host fallbacks. The final request
reached 68,301 total tokens. Peak device VRAM use was **27,828,781,056 bytes
(27.83 GB)**. Minimum free device VRAM was **6,110,478,336 bytes (6.11 GB)**.
These repeated prompts have high draft acceptance and do not represent a coding
conversation. The configured 262100 limit is not a tested usable capacity.

The earlier contiguous growth check peaked at 29,027,713,024 bytes, but generated
only one token per request and reused a prefix on the second request. It is
not a matched TPS comparison. An intermediate fragment build with separate
per-layer managers peaked at 31,853,756,416 bytes; that ownership was corrected
to use the scheduler's shared K/V manager.

## Isolated attention comparison

BF16 inputs, permuted page IDs, nonadjacent fragments, 64 warm repetitions.
Timings include host submission and synchronization. Every comparison matched
contiguous attention output bit for bit; K/V writes also matched expected bytes.

| Live keys | Table capacity | Queries | Contiguous ms | Fragmented ms | Difference |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16,367 | 16,384 | 1 | 0.223689 | 0.243486 | +8.9% |
| 16,367 | 16,384 | 8 | 1.828763 | 1.914127 | +4.7% |
| 16,367 | 16,384 | 128 | 25.326921 | 25.494913 | +0.7% |
| 68,173 | 69,632 | 8 | 72.955789 | 74.805003 | +2.5% |

The existing optimized short-query split-attention rules stop at 65,536 keys.
The 69,632-key bucket exceeds that limit in both storage layouts. This is a
separate dispatch limitation; no attention shape rules were changed here.
Fragment indirection has measurable cost. These results do not establish a
zero-cost storage change or a whole-engine speedup.

## Verification and build status

- `test_kv_memory` and `test_kv_cache` passed.
- Arena packing, freed-slot reuse, retained storage lifetime, stable address
  tables, cross-fragment growth, and block-allocation order are covered.
- The compiler test passed borrowed-address lowering, bounded-index narrowing,
  and rejection of invalid storage bounds/alignment through its verifier.
- The HRX/compiler patch applies to pinned source `5927b0e0`.
- The 68K GPU probe passed after native address export was isolated into
  `hrx_buffer_get_device_address`. Ordinary host-mapping and transfer helpers
  retain their previous behavior. The HTTP measurement above preceded that
  API isolation; it was not repeated afterward.

