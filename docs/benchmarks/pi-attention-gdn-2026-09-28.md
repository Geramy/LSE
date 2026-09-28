# Pi attention and GDN preparation

Measured on September 28, 2026 with an R9700 (gfx1201), driver 204, Qwen3.8-27B Q4, Q8 DFlash2 with three proposals, and FP32 KV. The declared context limit is 262,100 tokens. Sampling comes from the checkpoint: temperature 1.0, top-k 20, top-p 0.95.

## Changes

- Prepare the immutable GDN decay rate once when weights load. Use the same device FP32 cast, exponential, and optional softplus operations. Keep the owned result as a leaf so replay does not recompute it. This removes 96 small dispatches per Qwen target pass.
- Select the existing four-query short-attention layout for four through eight query rows at an allocated capacity of 8,192 tokens. The shape table already selects it at 16,384. The kernel body, partial record layout, FP32 accumulation, merge operation, and generation depth are unchanged.

## Native attention timing

Batch 1, 24 query heads, four KV heads, head dimension 256, 16-token pages, causal mask. Each arm has 16 warmup calls and 40 timed calls in ABBA20 order. GPU timestamps include partial and merge.

| Query rows | Live KV tokens | Previous GPU ms | Four-query GPU ms | Reduction |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 5,207 | 0.772475 | 0.395907 | 48.75% |
| 4 | 5,270 | 0.780802 | 0.394516 | 49.47% |
| 6 | 5,270 | 1.148430 | 0.711695 | 38.03% |
| 8 | 5,270 | 1.520242 | 0.713241 | 53.08% |

Both candidate timing blocks beat both original blocks in every case. All partial records and final outputs match the prior implementation bit for bit. A separate ragged batch with sliding masks, poisoned padding, empty rows, and permuted pages also passes. Across 936 native dispatches, there are no host groups or fallbacks. Inputs and allocation guards are unchanged.

Partial VGPR use increases from 36 to 58; LDS increases from 512 to 2,048 bytes. Scratch use remains zero. The four-query partial contains two static dual-FMA packets. These facts do not establish sustained dual-issue occupancy.

## Normal streaming HTTP result

The same two Pi user prompts, tool schemas, and low thinking setting are used before and after the changes. Assistant text, reasoning, token counts, and acceptance counts match exactly.

| Measurement | Previous | Updated |
| --- | ---: | ---: |
| First-turn decode | 23.98 tok/s | 24.87 tok/s |
| Follow-up decode | 28.82 tok/s | 31.28 tok/s |
| Follow-up verification | 4,283.635 ms | 3,836.616 ms |
| Follow-up draft | 1,208.512 ms | 1,220.896 ms |
| Follow-up prefill, 23 new tokens | 0.891 s | 0.890 s |
| Follow-up cached tokens | 5,310 | 5,310 |
| Follow-up timed decode tokens | 159 | 159 |
| Follow-up verifier passes | 65 | 65 |
| Follow-up accepted/tested proposals | 95/140 | 95/140 |
| Follow-up device groups | 158,376 | 151,752 |
| Host groups/fallbacks | 0/0 | 0/0 |

The follow-up improves by 8.54% in this ordered comparison. The updated process compiles seven changed kernels in its first request (246.2 ms total); the follow-up adds no compilations. First-turn prefill is 18.71 seconds for 5,207 tokens, versus 18.29 seconds before the changes. These results do not establish a prefill gain or a 46–48 TPS conversation rate. Kernel upload cost and prefill attention remain under investigation.

## Verification

- GDN CPU contract: 4/4 tests.
- GDN GPU: both decay conventions, H48/D128, one and four rows, three replays with changed dynamic inputs; exact FP32 rate, output, and recurrent-state bits. Every evaluated case requires device dispatches and rejects host fallback.
- Model CPU suite: 51/51 tests.
- Short-attention routing suite: 7/7 tests.
- Native attention gates described above.
- Two-turn streaming HTTP comparison described above.

No perplexity run is needed for these bitwise-equivalent operations. The local evidence is under `mac_amdgpu/build/release/pi-performance/attention-gdn` and the focused native fixtures.
