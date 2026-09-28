# Four-query key/value reuse for short split attention

Measured September 28, 2026 on gfx1201/wave32, driver 204. Both actual cases use batch 1, 24 query heads, 4 KV heads, head dimension 256, block 16, causal masking, 14,000 live keys and a 16,384-key table. Baseline is the active short split partial128 plus merge, not the superseded Flash8 kernel.

| Query width | Original GPU partial + merge | Four-query GPU partial + merge | Speedup | Original wall | Four-query wall |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 2.073208 ms | 0.924168 ms | 2.243× | 2.092273 ms | 0.942055 ms |
| 8 | 3.883307 ms | 1.611319 ms | 2.410× | 3.903062 ms | 1.631165 ms |

ABBA20 follows 16 warm calls per arm, yielding 40 measured calls per arm. Both candidate CP arms and both wall arms beat both baseline arms in each case. Timings include partial and merge dispatches. Compiles, oracle work and readbacks are outside measured intervals. All 456 actual-case dispatch events are valid and scaled by the device's 100 MHz clock; no host-clock correlation is claimed.

Partial-only CP means are 1.997205→0.848697 ms for M4 and 3.803547→1.532285 ms for M8. Merge remains approximately 0.076/0.080 ms. These are component timings. The separate HTTP comparison below measures server throughput.

## Exactness and memory safety

Every final output float is bitwise baseline-equal: 24,576 at M4 and 49,152 at M8. Full partial buffers are also bitwise equal: 3,170,304 and 6,340,608 FP32 values. Complete chronological double-reference outputs pass with maximum absolute error 2.32830644e-9 for both actual cases.

The additional batch3/M6 sliding-window257 case has live lengths 16,342, 143 and 0, permuted page mappings, poisoned unused KV, invalid table IDs and poisoned Q in its padded batch row. Its final query tile has two live and two padded query rows. All 110,592 outputs and 14,266,368 partial values match the original bits; full double reference max error is 2.23517418e-8. Its 24 dispatch events are valid.

Across all three cases, metadata-empty replay produces exact zero outputs and full partial records with -infinity maximum, zero denominator and zero numerator. Input bytes and every 64-byte allocation guard are preserved. Total: 480 native dispatches, zero host/fallback dispatches. No perplexity sweep is needed for bitwise-identical component arithmetic.

## Implementation and resources

A workgroup handles four query rows for one head and one 128-key partition. It loads each shared K value once, preserves each row's eight ascending strided FP32 FMAs, and retains the XOR16/8/4/2/1 reduction tree. V pairs are shared across rows while each row keeps its original ascending-key numerator/denominator accumulation and FMA operand order. Row-specific causal/sliding predicates are retained. The partial layout remains [B,H,M,P,258], and the existing merge is unchanged.

| M / stage | Original workgroups | Four-query workgroups | Original VGPR / SGPR / LDS | Four-query VGPR / SGPR / LDS | Scratch |
| --- | ---: | ---: | --- | --- | ---: |
| 4 / partial | 12,288 | 3,072 | 36 / 44 / 512 B | 58 / 54 / 2,048 B | 0 B |
| 8 / partial | 24,576 | 6,144 | 36 / 44 / 512 B | 58 / 60 / 2,048 B | 0 B |
| 4 / merge | 96 | 96 | unchanged | unchanged | 0 B |
| 8 / merge | 192 | 192 | unchanged | unchanged | 0 B |

Registers and LDS increase, so this report does not infer resident workgroup count. Native execution proves a timing gain for the two measured shapes without scratch spills. Partial disassembly contains two dual-FMA packets for the first QK products of the four rows (four `v_dual_fmaak_f32` component mnemonics); the baseline contains none. No claim is made about sustained dual-issue occupancy or physical DRAM traffic.

The production header rule selects four-query tiling for M4–8 at actual table capacity 16,384. Shorter tables and other widths retain the prior single-query body. Performance is directly measured at M4/M8; the M6 ragged case validates padding and mask arithmetic. LDS eligibility and launch geometry consume the same tile policy. There is no new registered production path or launch override; emitted-source hash validation handles existing cache keys.

## Provenance

Release baseline: 3acaad16b66d11765b5eef98523266fd067330ee. Shared release source was untouched during private measurements. Raw captures, JSONL exports, ABBA statistics, fixtures, compile commands, code objects, disassembly, source snapshots and SHA256 manifest are saved beside this report. The measured private tiled body matches the prepared production body after tile-symbol substitution and whitespace normalization; the retained single-query body matches the release baseline after whitespace normalization. Shared host/emission and HTTP integration gates remain separate.

## HTTP comparison at 14,000 tokens

The shared server build passed `test_attention_short` and `test_attention_decode`.
Each server process then received one initialization request and one measured request.
Both used Qwen3.8-27B Q4, the same Q8 DFlash2 draft, depth 3, FP32 KV, and a 32 µs blocked-poll interval.
The context limit was 262,100 tokens. The actual prompt contained 14,000 tokens.
Each request generated 64 tokens; the timing counter covers 63 decode tokens after the first token.

| Second request | Original schedule | Four-query schedule | Change |
| --- | ---: | ---: | ---: |
| Decode throughput | 33.43 tok/s | 42.56 tok/s | +27.32% |
| Decode time | 1,884.540 ms | 1,480.215 ms | −21.45% |
| Target verification | 1,573.101 ms | 1,168.232 ms | −25.74% |
| Draft time | 311.039 ms | 311.572 ms | +0.17% |
| Accepted / tested proposals | 47 / 47 | 47 / 47 | Same |
| Verification passes | 16 | 16 | Same |
| Device / host / fallback groups | 87,319 / 0 / 0 | 87,319 / 0 / 0 | Same |

Generated text, finish reason, and usage counts matched exactly.
This is one ordered baseline/candidate pair on a repetitive synthetic prompt.
The 100% acceptance rate does not represent general coding workloads.
No perplexity test was run for this change.

The second baseline prefill took 88.282 seconds, including 43.688 seconds of new compilation.
The candidate prefill took 44.775 seconds with no new compilation.
These prefill times do not establish a prefill speedup: the shared disk cache favored the candidate.
The measured change targets the four-row verification pass.
The initialization requests are excluded from the decode comparison.

### HTTP artifacts

The local artifact directory is `mac_amdgpu/build/pp-optimization/query4-http-20260928`.
It contains both server binaries, launch plans, prompt hash, complete responses, server logs, summary, and `SHA256SUMS`.

- Baseline binary SHA256: `f6187afdade5619c61bb6fbbaab1d27b5c03b7c1748708226388ef3fed338ce1`.
- Candidate binary SHA256: `a30e732efd570800c0779a76c09b5b4dba660ab8c9ef83be79fdba7e3ac4e814`.
- Prompt SHA256: `db18c5f26dda21cd17ed5b4207f1f767cd5128623d40c3d291f62c93e7e7fac9`.
