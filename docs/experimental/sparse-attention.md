# Experimental sparse attention through HRX / LOOM

The HTTP server enables FlashPrefill V2 by default for Qwen3.5-family
models with 256-wide heads on gfx1201 Wave32 through LOOM, using BF16 or FP32
KV. The default alpha is 0.1; decode remains dense. Disable it with
`--FlashPrefillV2=off`. Batch/ubatch remain 1024.

MTP and DFlash2 use FlashPrefill V2 for target prompt prefill only. Their draft
and verification passes stay dense. Other devices, architectures and KV formats
retain dense attention automatically. Explicit sparse modes still require a supported
configuration. Linear-attention layers and KV retention are unchanged.
BLASST remains opt-in.

## Launch

For the measured R9700 baseline path, no sparse-mode or calibration arguments
are needed. To force dense prefill, add:

```sh
--FlashPrefillV2=off
```

`--FlashPrefillV2=on` explicitly requests FlashPrefill V2; unsupported explicit
requests fail. `--attention-prefill dense` also overrides the automatic default.
Conflicting explicit prefill switches are rejected regardless of order.

For BLASST, add these arguments to an existing baseline server command:

```sh
--no-mtp --dialect loom \
--attention-prefill blasst --attention-decode blasst \
--attention-calibration /path/to/attention-calibration.json
```

Each mode can independently be `dense` or `blasst`; prefill also accepts
`flashprefill-v2`. BLASST remains unavailable with MTP/DFlash2. Execution phase
is explicit: prompt chunks use prefill, ordinary decode uses decode, and every
speculative draft/verification call is dense regardless of query width.
Singleton prompt tails are dense. Retained forward programs are keyed by phase
and width so a short prompt tail cannot be replayed as a verifier. Replacement
passes cannot change phase; KV writes, rewind and accepted-prefix commit retain
the existing semantics.

Approximate prefill can change prompt KV, recurrent state, captured DFlash2
features and subsequent tokens. Dense verification preserves the speculative
protocol for that resulting target state; it does not imply equivalence to an
all-dense prompt or unchanged acceptance.

BLASST requires a calibration file; FlashPrefill V2 optionally accepts one to
override alpha 0.1. The file must name the exact `--model` argument and provide a
coefficient for each enabled phase:

```json
{
  "version": 1,
  "model": "/absolute/path/to/model",
  "prefill": {"scale": 0.0},
  "decode": {"scale": 0.0}
}
```

**The zero coefficients above disable pruning and are for validation. They
are not tuned sparsity settings.** For each live sequence length L, BLASST
uses `lambda = min(scale / max(L, 1), 1)`. BLASST omits a tile only when
its maximum relative to the running maximum is strictly below lambda.
FlashPrefill V2 uses `scale` directly as the relative aggregate probe-mass
threshold alpha in [0, 1]. Nonzero
coefficients must be calibrated against retrieval and generation-quality
fixtures for that model and this tile schedule. The measured alpha 0.1 setting has matched short greedy outputs at 16K/32K;
perplexity and broader model-quality calibration remain unmeasured.

## BLASST and shared runtime behavior

- Prefill retains the existing 16-query × 256-key WMMA schedule. All live query
  rows must agree that a tile is negligible before the workgroup skips softmax,
  V loads and PV. A vote barrier precedes scratch reuse, and every barrier
  under the skip branch is reached uniformly by the workgroup.
- Decode splits history into 1024-key partitions, each with eight 128-key
  tiles and a local FP32 running maximum, denominator and numerator. A
  second kernel merges partition records with stable softmax rescaling.
  Local maxima make pruning more conservative than a global running maximum;
  coefficients must be calibrated for this partition schedule. The first
  valid tile in each partition is always retained.
- Causal and sliding masks are applied before pruning. Empty rows produce
  zeros. All KV pages remain resident according to the ordinary cache policy;
  pruning never evicts tokens.
- Coefficients and the sparse mode enter graph attributes and compilation
  cache identity. Unsupported sparse native kernels fail instead of silently
  executing dense attention on the CPU.

## Validation commands

```sh
./build/tests/test_sparse_attention
./build/tests/test_sparse_attention --gpu
python3 tests/test_server_cli.py ./build/lse-server
```

The GPU test compares the tiled implementation against an independent CPU
reference with reversed physical pages, causal offsets, sliding masks, empty
sequences, padded batch rows, and partial key/query tiles. It tests F32 and BF16
storage with pruning disabled and enabled, and asserts zero host fallbacks.
It checks partitioned decode records and stable merging, nonconstant pooled
K/V means and counts, actual omitted blocks, mean correction, query tiles
crossing key boundaries, and the one-block server warmup shape.
The gfx1201 fixture run passed 59 cases with maximum absolute error below
1e-7 against the tiled CPU reference. Host routing (2), WMMA (8), split
decode (10), and CLI validation (111 cases) also passed. These numerical
fixtures do not establish model quality at nonzero thresholds. Other KV
encodings require additional GPU qualification.

Before performance promotion, compare dense and sparse at 8K/32K/64K under
identical token sequences, report total prefill/decode time, and test quality
with nonzero model-specific coefficients. Kernel-only savings cannot establish
an application speedup.

Full-server Q4/BF16-KV smoke tests also completed with BLASST in both phases
and with FlashPrefill V2 prefill plus dense decode, using zero coefficients.
Both had zero CPU fallbacks. The first generated response matched dense;
the second differed for both modes. Zero pruning does not guarantee bitwise
equivalence across accumulation schedules. These smoke tests establish launch
and execution coverage, not calibrated model quality or a performance gain.

## FlashPrefill V2 prototype

The prefill option launches three native LOOM kernels: pooled mean K/V and
actual counts per 256-key block; stable aggregate exponential probe scores
across each 16-query tile; and selected-block WMMA with mean correction.
Selection compares each block's aggregate mass with alpha times the peak
block mass. Sink, recent, diagonal and partial key blocks remain exact.
An omitted block must be fully visible to every live query row; its mean
key logit plus log(token count) and mean value update both denominator and
numerator. The entire KV cache is retained.

This first implementation uses a dense block mask, synchronous pooling and
probe loops. PackGQA, compact block lists, selector reuse and overlapping
pipelines are future performance work. Long-context measurements are listed below. Causal paged attention with 256-wide heads is
supported; sliding-window FlashPrefill V2 is rejected.

Algorithm references: [BLASST](https://arxiv.org/html/2512.12087v3) and
[FlashPrefill V2](https://arxiv.org/html/2608.19758v1). This implementation adapts
the pruning rule and synchronous schedules to LOOM; it does not reproduce the
papers' NVIDIA kernels or their reported speedups.

## Selector optimization and tuning headers

The v2 selector assigns a Wave32 to each key-block probe, distributes the
256-component dot product over its lanes, and retains eight pooled-key
components per lane across the query tile. An XOR reduction combines the
partial dot products. The attention algorithm and alpha remain unchanged;
the FP32 reduction order differs. A 129-block reference fixture covers
strided wave assignment, padded rows, full-width probes and query tails.

Performance tuning tables use the `*_tuneconfig.h` suffix:

- `include/lse/dispatch/attention_tuneconfig.h`
- `include/lse/dispatch/quant_tuneconfig.h`
- `include/lse/dispatch/q8_tuneconfig.h`
- `include/lse/backends/hrx/submission_tuneconfig.h`
- `include/lse/ops/sparse_attention_tuneconfig.h`

`kFlashPrefillSelectorThreads` in `attention_tuneconfig.h` controls the selector
workgroup size; 128 threads (four Wave32 groups) is the measured setting.

On the 5,207-token profiling workload, aggregate selector GPU time changed
from 75.76 ms to 15.20 ms; attention time stayed approximately 673 ms. This is a
selector-kernel improvement, not an established total-model speedup. The
response matched the original selector. All 59 GPU reference cases passed.


## Long-context execution check

One cold-process pair per length, Q4/BF16 KV, capacity 65536, batch/ubatch
1024, greedy 64-token output, exploratory alpha 0.1 and dense decode:

| Prompt tokens | Dense prompt tok/s | FlashPrefill V2 prompt tok/s |
| ---: | ---: | ---: |
| 16384 | 494.10 | 627.87 |
| 32768 | 378.10 | 604.92 |

The prompt and binary were identical within each pair; prompt times include
compilation from an empty private cache. Both output pairs matched exactly,
with zero CPU fallback. These are code-review prompt execution checks;
retrieval and broader model-quality calibration remain open.

The merged master default-on/opt-out smoke pair at 16K measured **632.09**
versus **492.91** prompt tok/s, respectively. Both used the same binary and
request, cold private compilation caches, alpha 0.1 for FlashPrefill, and
batch/ubatch 1024. The 64-token greedy output matched, with zero CPU fallback.
The new flag passed 130 CLI checks. This is the prefill peak quoted for v0.4.23.

## Speculative prompt prefill

MTP3 and DFlash2 native 16K checks passed with the same 64-token greedy output
and aggregate acceptance as dense controls, with zero CPU fallback. See the
[speculative phase and measurement report](../benchmarks/flashprefill-speculative-2026-09-30.md).
