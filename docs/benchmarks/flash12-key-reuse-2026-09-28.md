# Flash12 key reuse — 2026-09-28

Flash12 now loads each key value once and reuses it across its twelve query rows. Each row retains its ascending-depth FP32 fused multiply-add order. Query staging, causal/sliding masks, online softmax, reduction trees, value accumulation, paging and launch geometry are unchanged. The header Flash rule selects key reuse; Flash8 retains its prior emitted body. No new registered path or launch override is introduced.

## Native GPU results

Measured on gfx1201, wave32, driver 204. Shapes are batch 1, query 512, 24 query heads, 4 KV heads, head dimension 256 and paged block 16, with causal masking. These are component timings, not end-to-end prefill or decode throughput.

| Live keys / table capacity | Original GPU mean | Key reuse GPU mean | Speedup | Original wall mean | Key reuse wall mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5,610 / 8,192 | 99.456938 ms | 25.202206 ms | 3.946× | 99.483856 ms | 25.222526 ms |
| 14,000 / 16,384 | 232.792939 ms | 93.771574 ms | 2.483× | 232.815488 ms | 93.790458 ms |

Each arm has 16 warm calls, then ABBA20: 40 measured calls per arm. Both candidate GPU and wall samples beat both original samples in each case. All 228 dispatch events are valid and use the device's 100 MHz timestamp scale. No host-clock correlation is claimed. Compiles and readbacks are outside the measured intervals; each call launches one native kernel.

## Correctness

Every one of the 3,145,728 output floats in each query 512 case matches the original shader bit for bit. A chronological double reference independently checks 55,296 outputs per case, across first rows, tile boundaries, midpoint and the final tile. Maximum absolute reference errors are 2.23517418e-8 and 2.18860805e-8; all remaining output floats are required finite.

A separate batch 3/query 17 case checks every 313,344 outputs. Its live lengths are 16,342, 143 and 0, with permuted pages, poisoned unused KV, invalid table IDs and poisoned queries in the padded batch row. All outputs match the original shader bits and full double reference; maximum absolute reference error is 2.63098627e-8. This adds 12 native dispatches.

All three cases preserve input bytes and 64-byte allocation guards. Replaying the same kernels with empty metadata overwrites every output with exact zero. Combined gates execute 240 native dispatches and zero host/fallback dispatches. No perplexity run is needed for this bitwise-equivalent component change.

## Generated code

Both actual query 512 cases retain 48 VGPRs, 64 SGPRs, 37,056 bytes LDS and zero scratch. The ragged case changes 46→47 VGPRs and retains 68 SGPRs, 37,056 bytes LDS and zero scratch. There is no resource-based evidence of an occupancy regression; no resident-workgroup count is inferred from this metadata.

The QK depth loop replaces twelve key loads with one key load and twelve independent FMA chains. Static `global_load_b32` sites decrease 281→270; static FMA sites remain 3,133 and barrier sites remain 44. Native disassembly contains no dual FMA packets. The measured gain comes from key reuse and reduced repeated depth-loop work, rather than a claimed dual-issue gain.

## Evidence and integration

Baseline source revision: `4a9335d085f7e93d6d5ed218e47abf618b9c0ea1`. Native fixture SHA256: `5fcbac7b8d09f9648917a94dfd3cd6d7f7ad1d499a4251b3ae2979c355f591b6`.

Local evidence is archived under `mac_amdgpu/build/pp-optimization/flash12-kreuse-20260928/`: source snapshots, fixture and compile commands, code objects, Loom source, disassembly, resource metadata, raw profile captures, JSONL exports, ABBA statistics and SHA256 manifest. The measured private QK body and integrated Flash12 body are equal after whitespace normalization; the retained Flash8 QK body matches the baseline after whitespace normalization.

The existing primitive identity is retained. The JIT cache validates emitted-source hashes before using an existing disk artifact. The integrated HTTP server rebuilt successfully, and `test_loom_flash` plus `test_attention_short` passed. No new end-to-end HTTP timing was run for this change; this report makes no end-to-end speedup claim.
