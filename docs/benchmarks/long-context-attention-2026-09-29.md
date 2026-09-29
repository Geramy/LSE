# Long-context split attention, 2026-09-29

The Qwen3.8-27B Q4 target and Q8 DFlash2 draft use BF16 paged KV on the local
R9700. This comparison uses the same model files, HRX/Loom runtime, 1024-token
batch and microbatch, temperature 0.6, top-k 20, top-p 0.95, seed 1234, and
262100 configured KV limit. Each server starts fresh. The prompt repeats a
short sequence; its high draft acceptance does not represent a coding chat.

The candidate extends short-query split attention from 16K through 65K keys.
Four verifier query rows share each K/V load. The merge stage writes partition
weights in 128-lane strides so it can combine 512 partitions. It retains the
existing FP32 dot products, softmax state, and output layout. The separate
single-query and short-query primitive registrations are replaced with one
split partial primitive and one merge primitive. The generated body still
selects the appropriate head or row tile from the query shape.

| Cold HTTP request | Current baseline | Consolidated split | Observed change |
| --- | ---: | ---: | ---: |
| 7,521 prompt tokens, prefill | 426.32 PP/s | 427.29 PP/s | +0.23% |
| 7,521 prompt tokens, 63 timed decode | 19.47 TPS | 20.18 TPS | +3.64% |
| 33,126 prompt tokens, prefill | 293.78 PP/s | 286.11 PP/s | -2.61% |
| 33,126 prompt tokens, 63 timed decode | 11.55 TPS | 13.00 TPS | +12.54% |
| 33K target verification, total | 4.410 s | 3.413 s | -22.61% |
| 65,126 prompt tokens, prefill | 147.70 PP/s | 147.02 PP/s | -0.46% |
| 65,126 prompt tokens, 63 timed decode | 8.51 TPS | 14.57 TPS | +71.26% |
| 65K target verification, total | 6.444 s | 3.192 s | -50.47% |

All three pairs produced the same 64-token text. The 33K pair had 55/55 accepted
proposals in eight verification passes. At 65K, the baseline accepted 55/55 in
eight passes; the candidate accepted 54/55 in nine passes. The text remained
identical. All work stayed on the device. The 65K candidate completed without
an allocation error; a near-end counter sample showed 28.76 GB reserved and
5.18 GB free. Samples do not establish the exact peak.

The baseline re-used most compiled kernels from disk. The first candidate 33K
run compiled 426 kernels and the 65K run compiled 118; baseline counts were 2
and 3. These are single cold runs in fixed order, so the prefill differences
are not a controlled steady-state speed estimate. The changed split path is
used during verification; prefill continues to use flash WMMA. No separate
per-kernel GPU timestamp capture was taken for this comparison.

A separate 512-key flash WMMA window candidate measured 219.81 PP/s on the same
33K prompt, versus 286.11 PP/s with the 256-key window. Its 18.03 TPS decode
result cannot be attributed to that window because verification used split
attention. The larger window remains out of the active source.

The split operation has two GPU stages: independent key partitions write
partial softmax records, then the merge reads them. Combining those stages
without a global synchronization protocol would remove parallel key work.
The code has one registered split partial kernel family and one registered
merge family, instead of separate decode and short-query versions of each.
The path is qualified through 65K keys. Larger contexts still need separate
performance and memory tests.

Raw HTTP responses and server logs are retained locally under
`build/release/pi-performance/long-context/` in the mac_amdgpu workspace.
Temporary comparison binaries were removed after testing. The active source
contains the consolidated split path and retains the 256-key flash WMMA window.
