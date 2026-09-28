# Skip globally empty short-attention partitions

The M4/table8192 shape rule now skips partitions starting beyond the global live KV length. Each workgroup first writes its complete partial record: maximum `-Inf`, denominator `+0`, and 256 FP32 numerators `+0`. It then returns before query/key/value loads and LDS reductions. The launch geometry, merge, partial layout and nonempty FP32 arithmetic remain unchanged. Other query counts and table capacities retain their existing rules.

## Component measurements

Paged FP32 attention on gfx1201: batch 1, 24 query heads, 4 KV heads, head dimension 256, block size 16, capacity 8,192, causal mask. Both arms use the accepted four-query kernel. ABBA 20 provides 40 measured calls per arm after 16 warmups. GPU timestamps use the device metadata frequency of 100 MHz, with no host clock correlation.

| Live KV tokens | Current query4 GPU time | Empty shortcut GPU time | Change |
| ---: | ---: | ---: | ---: |
| 5,207 | 0.381818 ms | 0.322881 ms | -15.44% |
| 5,270 | 0.381834 ms | 0.333373 ms | -12.69% |

Times include partial and merge. Both candidate arms beat both baseline arms. Respectively 23 and 22 of 64 partitions are globally empty. Wall means were 0.402386→0.343382 ms and 0.401304→0.353816 ms. These component measurements do not establish an HTTP TPS improvement.

## Correctness and resources

Two timed cases and a batch-3 sliding-window case produced 480 native dispatches with zero host execution or fallback. Complete partial records and final outputs are bitwise baseline-equal. The independent chronological double oracle passed; maximum absolute error on the masked edge was 2.60770321e-8. Poisoned unused pages/table entries, shorter rows, padded batches, readonly inputs, allocation guards and metadata-empty replay passed. The shortcut uses the global live extent; per-row validity predicates remain unchanged.

Partial resources: VGPR 58→64, SGPR 54→64, LDS unchanged at 2,048 bytes, scratch zero. Workgroup 128, wave 32, launch geometry and allocation sizes are unchanged. Empty workgroups return before the three barrier pairs; no occupancy count is inferred from static register counts.

Base source: `69348e5`. Native fixture SHA256: `7c69c0f9f08640651db1e8c43b01238ab6e6e2b9cc579b9165112a98b4860d30`.
