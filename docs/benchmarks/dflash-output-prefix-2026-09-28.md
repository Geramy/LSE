# DFlash output-prefix qualification

A post-v0.4.11 candidate keeps all eight noncausal draft-body rows and trims only final output processing for requests of one to three proposals. It uses the existing measured M4 Q4 vocabulary head, then computes three TopK and selector positions. Wider requests retain seven output positions. No kernel body, precision, or default verification depth changes.

## Native whole-output operation

Actual geometry: Q4 head N248320/K5120, Q8 selector projection N256/K5120, TopK16, codebook embedding, selector scores and typed path walk. Deterministic finite synthetic inputs and weights; the independent predecessor/successor graph inputs share codebook storage to bound fixture memory. A small one-layer module fixture checks eight-row prefix equivalence through context wrap, rewind, overwrite, and replay. The full five-layer model body is unchanged and is not included in this component timing.

After eight paired warmups, ABBA20 yields 40 timed evaluations of each arm:

| Metric | Head7, selector7 | Head4, selector3 | Reduction |
| --- | ---: | ---: | ---: |
| Mean passive GPU time | 3.624536 ms | 1.870375 ms | 48.40% |
| Mean wall time | 3.850205 ms | 2.219861 ms | 42.34% |
| Dispatches per evaluation | 18 | 20 | Two extra prefix copies |
| Head VGPR allocation | 96 | 62 | 34 registers |
| Head LDS / private allocation | 0 / 0 B | 0 / 0 B | No added scratch |

Both candidate GPU and wall arms were faster than both baseline arms. Wall time includes graph replay and completion wait; JIT and correctness readback precede the timed samples. Spill counts are unreported, distinct from known zero private bytes.

| Component GPU mean | Full output | Prefix output |
| --- | ---: | ---: |
| Q4 vocabulary head | 2.974835 ms | 1.491141 ms |
| Complete TopK16 chain | 0.513597 ms | 0.277879 ms |
| Q8 selector projection | 0.074446 ms | 0.025824 ms |
| Codebook / anchor / concatenate | 0.014604 ms | 0.017992 ms |
| Selector scores and walk | 0.031219 ms | 0.026298 ms |
| Additional prefix slice copies | — | 0.014362 ms |

Complete first-three normalized states, vocabulary logits, TopK values and IDs, selector gates, scores and tokens matched bitwise. Finite values, typed validation status zero, allocation guards and strict zero host/fallback execution passed. The capture contains 1,862 native dispatches across 98 complete evaluations and 34 canonical entries; timed subsets contain 1,520 dispatches. GPU duration uses the device metadata clock, 100,000,000 Hz.

The kernel cache was validated as the real `~/.lse/cache` directory with no LSE server running; 1,060 entries were cleared before the fresh process. `DYLD_PRINT_LIBRARIES` confirmed the requested HSA path `mac_amdgpu/build/hsa/libhsa-runtime64.0.1.0.dylib`, SHA256 `b7f8216e32fa6ab0e5b2fce87c518ce67c3fffa4c628cceece983c53e3cea90d`.

Fixture binary SHA256: `78c335e2ccfa0b26c8bcfd55e8d24051f53e25e9d4b44482c62f9ad11220a30d`. Sources, canonical archive hashes, passive capture identity, typed resource facts and raw ABBA means are in `manifest.json`, `resources.json` and `results.json`.

## Limits

The table above measures output components, without a perplexity run.

## Integrated HTTP comparison

The same two-turn Pi conversation, FP32 KV and model sampling defaults produced
byte-identical responses and identical accepted/tested proposal counts:

| Metric | Full output | Prefix output |
| --- | ---: | ---: |
| First prompt, 5,207 tokens | 291.75 tokens/s | 308.26 tokens/s |
| First decode, 102 timed tokens | 27.97 tokens/s | 28.96 tokens/s |
| Follow-up decode, 198 timed tokens | 35.26 tokens/s | 36.66 tokens/s |
| Follow-up prefill, 23 new tokens | 0.738 s | 0.729 s |

Decode improved approximately 3.5% and 4.0% in this single pair. Both runs had
zero host groups and zero fallback groups. Process and disk caches started empty;
the second conversation turn reused each running process. The candidate used
a fresh isolated cache directory. These measurements do not reach 46–48 tokens/s.
Artifacts: `dflash-split-cold` and `dflash-prefix-cold` under
`mac_amdgpu/build/release/pi-performance`. Candidate server SHA256:
`01dcf624adc20b0c49adbc38317e0071f24df2957695ebe9686eabd8f995627f`.

## Remaining work

 The unchanged eight-row body still contains the largest Q8 FFN and ring-attention work. Uncomputed output positions four to seven are no longer validated; consumed bucket positions preserve finite-score validation. Requests of one or two proposals reuse the same three-position output bucket.
