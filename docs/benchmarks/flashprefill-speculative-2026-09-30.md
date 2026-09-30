# FlashPrefill V2 with speculative decoding — 2026-09-30

## Execution contract

FlashPrefill V2 applies only to target prompt prefill. MTP draft, DFlash2 draft,
and target verification remain dense, including verification with multiple
queries. The existing `--FlashPrefillV2=off` switch disables prompt sparsity;
default alpha is 0.1 on the supported Qwen3.5-family, gfx1201 Wave32, HRX/LOOM,
BF16/FP32-KV configuration. BLASST remains unavailable with speculation.

A phase is carried from the generator through each model layer. Forward
retention is keyed by phase and query width. A replacement cannot change phase
before any cache/cursor mutation. Paged writes, rewind, recurrent-state carry,
accepted-prefix commit, MTP history and DFlash2 context storage are unchanged.
The batch scheduler also explicitly marks its non-speculative prompt chunks.

## Native paired checks

R9700, Qwen3.8-27B Q4, BF16 KV, capacity 65,536, batch/ubatch 1,024, exact
16,384-token code-review prompt, greedy generation capped at 64 tokens. MTP
uses the Q8 module at depth 3; DFlash2 uses the Q8 draft at block width 8.
Each run starts a fresh server and empty private kernel cache. Prompt and decode
times include cold compilation. One on/off pair per speculative method.

| Mode | Dense prompt tok/s | FlashPrefill prompt tok/s | Throughput gain | Accepted, off/on | Greedy text |
|---|---:|---:|---:|---|---|
| MTP3 | 474.8 | 582.6 | 22.7% | 47/47, 47/47 | Identical 64 tokens |
| DFlash2 | 485.0 | 615.2 | 26.8% | 49/60, 49/60 | Identical 64 tokens |

All four runs recorded zero host groups and zero fallback. DFlash2 exercised
rejected drafts and rollback. DFlash2 acceptance by position differed even
though the aggregate matched: approximate prompt state changes the drafts.

Cold decode throughput was 19.89/20.18 tok/s for MTP and 15.14/13.16 tok/s for
DFlash2 (off/on). The DFlash2 on run compiled more kernels and spent more time
in verification; these measurements establish prompt improvement, not a decode
speedup. Repeated warm decode profiling remains separate work.

A followup completion reused 16,447 cached prompt tokens and processed six new
tokens in every run, then immediately returned EOS in both configurations.
This checks prefix handoff and short prompt tails, not sustained followup
text generation. The fixture suite separately covers accepted-prefix state,
rejections, cancellation, stop boundaries, retained requests and feature capture.

Matching short outputs and aggregate acceptance do not establish perplexity,
retrieval accuracy or long-form equivalence. Approximate prompt KV, recurrent
state and captured draft features can affect subsequent generation.

Artifacts: `build/release/pi-performance/flashprefill-speculative/` in the
MacAMDGPU workspace: `results.json`, each run's command/request/result/response,
and server logs. Binary and request hashes are equal within each pair.

## Regression checks

- Explicit phase routing at widths 1, 2, 4, 8 and 17: prompt uses FlashPrefill
  only for multirow input; speculative calls stay dense at every width.
- Equal-width prompt and verifier graphs have different retained roots;
  repeated verifier calls replay their own root.
- A replacement changing phase fails without modifying the cursor.
- CLI coverage admits both speculative methods with FlashPrefill and continues
  rejecting BLASST with speculation.

Host checks passed: sparse phase/graph 3/3, hybrid retention 1/1, MTP options
3/3, DFlash2 35/35, DFlash2 walk 5/5, prefix commit 6/6, feature capture 5/5,
and 133 CLI cases. The final batch-scheduler tagging does not alter the HTTP
generator path used by the native measurements.
