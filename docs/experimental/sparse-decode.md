# Sparse decode attention

At long context a decode step is bound by memory bandwidth, and the KV cache
becomes a large share of the bytes it reads: at 128K tokens Qwen3.8-27B's 16
full-attention layers hold 8.6 GB of bf16 K/V against 15 GB of 4-bit weights.
Sparse decode reads only part of the cache in decode and verify passes. It
spends a little compute on choosing what to read, and the engine decides per
device where that pays.

It is approximate: it changes output. `--sparse-decode=off` keeps every pass
exact.

## What a sparse pass does

1. **Summaries.** Each attention layer keeps, per 16-token page and key head,
   four bf16 rows:
   - the keys' center and radius ((max + min) / 2, (max - min) / 2 per channel);
   - the key mean;
   - the value mean.

   They live in a pool beside the K/V pools, indexed by physical block, sized
   to the block ceiling once (1/8 of the K/V pool's bytes). Every pass that
   writes keys (prefill, decode, verify, tree) refreshes the pages from 64
   tokens before its first row to its last. A page's final summary is
   therefore taken after its last write, whatever a rejected draft or a tree
   commit rewrote below the pass.
2. **Scores.** For every far page, every query head of a key head and every
   row of the pass, the score is Quest's bound q·center + |q|·radius,
   relative to that row's bound over the eight newest pages. A page's score is
   its largest over the rows: one union across a verify chain's or tree's
   rows, so the target verifies against one consistent context. Decode uses a
   wave per page with the channels over the lanes. Passes with 16 or more rows
   per key head use the matrix core: two bf16 products, [q | |q|] against
   [center; radius].
3. **Selection, per key head.** The four sink pages and the 32 newest pages
   (the recent window, which also holds the pass's own rows) are always read.
   The far pages with the highest scores fill the rest of the budget. They are
   found by a bitwise bisection over unique keys (relative mass bits, page
   index in the low bits), then a scan writes their block ids between the
   sink and the recent window.
4. **Attention.** The step descriptor's positions move down by the pages left
   out, so the causal and tree masks hold unchanged. The existing exact kernels
   then run over the selected pages, reading a block table per key head: the
   1-row split partial and merge, or the WMMA flash split for chains and trees.
5. **Mean correction.** Each far page left out still contributes
   16·exp(scale·q·key mean) with the value mean, as FlashPrefill V2's omitted
   blocks do. These contributions go into split records that the merge
   combines with the exact ones.

## When it is used

`--sparse-decode=auto` (the default) predicts each pass's attention time both
ways, for each KV rung and pass width, at the smallest context the rung holds:

- exact: live K/V bytes over the measured exact-attention bandwidth;
- sparse: score, selection and correction time, plus the budget's K/V.

Sparse is taken only where it saves at least 20% of the exact attention time.
The inputs are all measured on the device at load and cached per device and
engine build in the kernel cache:

- the roofline (streaming bandwidth, dispatch cost, vector and matrix rates);
- the sparse kernels' own times on synthetic paged K/V (exact split attention
  at 64K and 128K, scores at 1 and 8 rows, selection and mean correction).

The load log prints both and the crossover for 1, 4, 8 and 16-row passes. A
device that was never measured stays exact.

| Flag | Default | Meaning |
|---|---|---|
| `--sparse-decode=auto\|on\|off` | `auto` | `auto`: where the measured costs predict a gain. `on`: every decode and verify pass whose context exceeds the budget. `off`: exact. |
| `--sparse-decode-budget N\|P%` | `25%` (at least 4096) | Tokens a pass reads, sink and recent window included, or a share of the context. |

Supported on gfx1201 and gfx1151 (Wave32, Loom), for Qwen3.5-family models
with 256-wide heads and bf16 or fp16 KV, on one device, up to 256K tokens.
`--sparse-decode=on` fails at load anywhere else.

## Quality

Perplexity and KL divergence come from `--perplexity-tail N --perplexity-step S`,
which prefills each window's context and then runs its last N tokens through
decode (S = 1) or verify (S = 8) passes. Decode-time attention is therefore
what gets scored, against an exact base recorded with `--sparse-decode=off`.
See [PERPLEXITY.md](../PERPLEXITY.md).
