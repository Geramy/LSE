# Completed prefill workspace retirement

## Change

The server releases completed target forward graphs before the next request's
prefill. Within prefill, it releases the completed target workspace when the
chunk width changes. Consecutive chunks with the same width retain replay.
Recurrent state is detached into leaves that share its existing device buffers.
Paged KV pools, weights, compiled kernels and verifier rollback remain intact.

DFlash2 retains at most one wide context-projection program during prefill.
Completed wide context programs are released after prefill. Narrow context
programs and draft programs remain available for decode.

## Method

Control: published-source v0.4.16, revision `2e705a0`.
Candidate: the same revision with the memory ownership changes. Native server
SHA-256: `5c9ded4c3b672440e179522f1cb9d3107aaff1c151d422f1eb7af44eda154a04`.
Both use the same bundled HSA/HRX/Loom libraries and local Q4 target plus Q8
DFlash2 checkpoint. Launch settings: BF16 KV, FP32 accumulation, KV capacity
262100, batch/ubatch 1024, temperature 0.6. Request settings: top-k 20, top-p
0.95, seed 1234, at most 128 generated tokens.

Only one server runs at a time. Requests are saved and replayed unchanged.
VRAM is sampled through the driver's allocator counters every 0.5 seconds.
Numbers below use decimal GB. Driver counters measure reserved allocations;
they include allocator slabs that can remain reserved after logical buffers are
freed. Sampled peaks are lower bounds, not precise instantaneous peaks.

## Ragged prefill comparison

The process first handles a 5120-token request. The second request has 6143
tokens, including a 1023-token remainder that exercises multiple prefill widths.
Neither request reuses prior KV. Each generates 128 tokens (127 timed decode
tokens). Responses and acceptance counts match exactly.

| Metric | Control | Candidate | Change |
| --- | ---: | ---: | ---: |
| Reserved VRAM after second request | 29.7696 GB | 28.2555 GB | -1.5141 GB |
| Sampled peak reserved VRAM | 30.0870 GB | 28.8731 GB | -1.2139 GB |
| Second request prefill | 457.82 tok/s | 458.49 tok/s | +0.14% |
| Second request decode | 68.21 tok/s | 67.88 tok/s | -0.48% |

There is no measured throughput improvement. The small timing differences do
not establish a performance regression. The high decode rate belongs to this
synthetic prompt's 100% measured proposal acceptance, and is not a general rate.

## Validation and limits

`test_dflash2` and `test_runtime` pass. Added cases check release of completed
graph owners, unchanged recurrent storage and values, and unchanged DFlash2
ring contents, rewind and repeated drafting.

Earlier eight-turn cached-follow-up and four-turn 8192-token boundary checks
matched text, cached-prefix counts and acceptance statistics. Those candidate
checks preceded the final change to retire between prefill widths. The final
eight-turn check under `matched-final` also matches all text, cached-prefix
lengths and acceptance counters. Reserved VRAM increases only 12.14 MB across
those eight turns (28.2432 to 28.2554 GB). Aggregate decode rate over the matched
472 timed tokens is 42.89 tok/s for control and 42.94 tok/s for the final build.

The reported allocation failure has not been replayed. Replaying that private
session requires user approval. The synthetic test establishes excess workspace
reservation and its reduction; it does not prove that every cause of the
reported 31.11 GB allocation failure has been removed.

No perplexity test was run. This change modifies ownership and lifetime, not
kernel arithmetic, weight quantization or sampling.

Raw requests, responses, server logs, allocator samples and build/test logs are
retained locally under `mac_amdgpu/build/release/pi-performance/memory-followups/`. Temporary backend allocation logging was removed
from the final binary.
