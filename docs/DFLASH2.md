# DFlash2 drafting

Enable the separate DFlash2 drafter with `--dflash2=on` and select its checkpoint
with `--dflash2-model`. Without `--dflash2-model`, LSE uses the draft it pairs
with the target (for Qwen3.8-27B, `incoai/Qwen3.8-27B-DFlash2` at the revision
below) from the Hugging Face cache, downloading it first when it is not cached;
`--offline` turns a missing draft into an error that names the `pull` command.
See [Flags](../README.md#flags). This replaces native MTP for that process. The target
model verifies proposals and determines every emitted token.

## HTTP server

```sh
./build/lse-server \
  --model /path/to/qwen38-27b-q4 \
  --dflash2=on \
  --dflash2-model /path/to/qwen38-27b-dflash2-q8 \
  --served-name qwen38-q4 --kv-len 2048 \
  --host 127.0.0.1 --port 8080
```

Conditional draft selection uses the request's sampling settings. The target
verifies proposals with probability-ratio rejection and residual sampling; a
sampled request verifies, each step, the prefix of the draft block expected to
decode fastest (see [adaptive verify width](#adaptive-verify-width)). Greedy
requests retain deterministic draft selection and verify all seven proposals,
or with [draft trees](#draft-trees) the tree the policy picks.
DFlash2 remains opt-in. Current matched results and limits are in the
[final mode comparison](benchmarks/forward-modes-final-2026-09-29.md).

```sh
curl http://127.0.0.1:8080/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen38-q4","prompt":"Explain how GPU caches improve inference.","temperature":0,"max_tokens":16}'
```

## Initial implementation measurement (historical)

The complete Q4-target/Q8-drafter HTTP path succeeded with device kernels
required. Two identical greedy requests used 1024 prompt tokens, 16 output
tokens, and KV capacity 2048. Cold and warm requests returned identical text.
The second request reported:

| Measurement | Warm result |
| --- | ---: |
| Prompt evaluation | 421.2897 tokens/s |
| Decode | 14.6172 tokens/s |
| Accepted proposals | 10 / 13 (76.9%) |
| Speculative steps | 5 |
| Draft time | 200.095 ms |
| Verification time | 788.161 ms |
| New kernel compilations | 0 |

The cold request spent 32.361 seconds in JIT compilation; the cumulative kernel
count was 127. This short measurement establishes working native HTTP execution.
The 43 tokens/s decode target remains unmet. The final native-MTP=2 measurement was
25.86 tokens/s with 32 output tokens; its different output length prevents
a controlled speed comparison with this DFlash2 request.

## Compatible checkpoint

The compatible drafter is
[`incoai/Qwen3.8-27B-DFlash2`](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2),
revision `dedf8df68adfb1afeaf7b7480c0a0243108177b4`. Its BF16 weights contain
1,924,404,480 parameters in five layers and occupy 3,848,817,896 bytes. It shares
the target's embedding and output head; it is a separate drafter from the native
one-layer MTP head.

Source weight SHA256:

```text
67fc76d68dc5a9415511a4f394ef744d67510cd20e93b37cc2cc7d28e4bab65c
```

Target feature taps are post-block outputs at zero-based layers
`[5, 19, 33, 47, 61]`, before final normalization. The drafter has width 5120,
32 query heads, 8 KV heads, head dimension 128, FFN width 17408, and RoPE theta
10,000,000. The loader validates the target feature geometry.

Every draft evaluates the complete trained block of eight positions: one anchor
and seven mask tokens. Returning a shorter prefix preserves that computation.
The target verifier consumes the anchor plus up to seven proposals (all seven
for greedy requests and with `--adaptive-dflash2=off`). Health and timing
fields report depth seven; the trained draft block remains eight positions.
Attention is noncausal within the draft block, including future mask positions;
context keys obey the 2048-position sliding window. Learned dynamic convolutions
have two taps and groups of 16 channels. The selector conditions each choice on
its predecessor using top-16 candidates and rank-256 codebooks.

## Adaptive verify width

A verify pass costs more the more rows it carries, and a proposal behind a
likely rejection rarely pays for its row. With `--adaptive-dflash2=on` (the
default; `lse_config.adaptive_dflash2`), each step of a sampled request
verifies the prefix of the draft block that maximizes expected tokens per
second (`src/runtime/draft_width.cpp`):

- **Acceptance estimate.** Proposal j's chance of acceptance, given the ones
  before it were accepted, comes from the draft's own conditional
  distribution at j: its largest probability, mapped through a calibration
  table (24 bins) that learns from every proposal the target checks how often
  proposals of that confidence are accepted. Old observations fade (half
  weight after 512 more checks, about one 640-token request), so the table
  follows the text.
- **Costs.** The wall time of each verify width (1 to 8 rows) and of a draft
  is measured on the device for this model; the first steps of a process
  visit every width twice, and a width unmeasured for 512 steps is measured
  again on a step where that costs little (after 2,048 steps, whatever it
  costs). A shared level follows the context length, and a per-width offset
  holds what the extra rows cost. Costs are kept per doubling of the context
  (below 2K, 2K, 4K, ... 256K): a wide pass's extra rows cost more the
  longer the context. A context first reached starts from the nearest one
  measured; the first width measured there moves the level, every width
  measured there takes its own cost at once, and the widths not yet measured
  there follow the growth the measured ones show, linear in rows. A context
  left before it had measured every width (a process's first request is
  mostly warm-up) starts over that way on its return, so it explores nothing
  again. A step
  that recorded a program (a width's first pass in a request), compiled or
  loaded a kernel, or tried a kernel's variants on the device is left out,
  and so is each width's first pass in a newly reached context. While a
  cost rests on fewer than two samples a lower sample replaces it; a slow outlier
  moves an estimate by at most a quarter, a fast one moves it whole. A cold
  first request (kernels compiling and variants being tried) therefore
  cannot price the wide widths or tree rungs out for good. Nothing is
  hard-coded, so each GPU tunes itself: on the R9700 the step costs of 1 to 8
  rows measured 33.7, 35.3, 36.2, 35.9, 36.1, 36.5, 38.1 and 38.7 ms (verify
  pass, acceptance walk and context append; 640-token essay, warm), and
  before the verify widths moved onto the four- and eight-row kernels the 3,
  5, 6 and 7-row passes took 48 to 89 ms and the policy avoided them.
- **Stopping rule.** Proposal j is verified when some prefix ending at or
  after j adds more expected tokens per added nanosecond than the long-run
  rate; positions after j enter at their mean acceptance, not their own
  confidence. Whether proposal j is verified therefore depends only on the
  draft's distributions up to j, never on token j itself, so every verified
  proposal is still a sample of the draft's conditional and rejection
  sampling keeps the target's distribution exactly.
  `a_draft_side_stopping_rule_keeps_the_target_distribution` in
  `tests/test_runtime.cpp` checks this on a two-position example, and
  `dflash2_adaptive_verify_width_keeps_the_target_distribution` in
  `tests/test_dflash2.cpp` runs the generator itself, with policies that
  take every width and skip drafts, and tests the emitted transitions of a
  target with known probabilities against them (chi-square), beside a plain
  decode and the fixed width.
- **Skipping the draft.** When drafting is expected to lose to a plain
  one-row step, the next steps skip the draft, with a probing draft after
  every four plain steps. The draft keeps no state between drafts beyond its
  context ring, and every verify pass, plain or not, appends its verified
  rows to that ring, so a skipped draft leaves nothing to repair.

Without draft trees, greedy requests always verify all seven proposals (with
them, see [draft trees](#draft-trees)). The target's logits differ
in their last bits between pass widths (the GEMM shape and the Gated DeltaNet
chunking follow the rows of a pass), so a different width can flip a near-tie
between two tokens; master's greedy DFlash2 output already differs from plain
decoding for the same reason. Verifying the whole block keeps greedy output
byte-identical whatever the policy has learned.

Verify passes of different widths alternate without rebuilding: a retained
pass whose chain predecessor did not just run takes the carried state of the
pass that did by exchanging buffers (`graph::Program::adopt_carries`). Before
this, every width change rebuilt the pass (about 25 ms).

Timings report `spec_proposed` (proposals the passes carried), `spec_tested`
(proposals the target checked before the first rejection), `spec_accepted`,
`spec_mean_width` (rows per pass), `spec_plain_steps` (passes with no
proposals) and `spec_adaptive`; `/health` reports `dflash2_adaptive` and
`mtp_adaptive`. Acceptance rate is accepted over tested;
with a shorter prefix fewer of the doubtful proposals are tested at all, so
compare accepted over proposed as well.

DSpark ([arXiv 2607.05147](https://huggingface.co/papers/2607.05147))
schedules verify prefixes the same way, from a learned confidence head and a
profiled width curve.

Measured on the R9700 (macOS, driver build 266, master 1f1c3cd plus this
change), the pinned DFlash2 configuration at temperature 0.6: three
interleaved pairs of servers, `--adaptive-dflash2=off` then on, each pair
running 12 seeds of every prompt (36 requests per prompt and arm). Fixed
output is the same text on every run of a seed; adaptive output is a
different sample of the same distribution, so the comparison is over 36
texts each.

| Prompt | Fixed (median) | Adaptive (median) | Change | Mean change (95% CI) | Acceptance fixed / adaptive | Rows per pass fixed / adaptive |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 640-token essay | 62.0 tok/s | 67.4 tok/s | +8.8% | +8.4% (+6.8 to +10.0) | 67.2% / 69.5% | 7.95 / 5.79 |
| Code (HumanEval-style, 640 tokens) | 152.8 tok/s | 152.2 tok/s | -0.3% | -1.0% (-3.7 to +1.7) | 97.2% / 97.1% | 7.96 / 7.79 |
| 2K prompt, 256 tokens | 70.6 tok/s | 70.9 tok/s | +0.4% | +2.3% (-0.9 to +5.8) | 75.0% / 76.0% | 7.91 / 5.94 |
| 4K prompt, 256 tokens | 64.9 tok/s | 66.7 tok/s | +2.7% | +2.5% (-1.0 to +6.3) | 74.4% / 75.8% | 7.91 / 5.87 |

On a Radeon 8060S (gfx1151), where a 4-row verify pass costs 92 ms and an
8-row one 116 ms, three interleaved pairs measured essay +5.8%, 2K +6.9%,
code +0.6% and 4K -2.9% (means; essay and code 9 runs per arm, 2K and 4K
3).

The same policy drives MTP with `--adaptive-mtp=on` (the default;
`--adaptive-mtp=off` keeps `--mtp-depth` every step). On the
R9700 against `--mtp-depth 3` (same method, 36 requests per prompt and
arm) it chains about 6.8 deep on code, +25.4% (+21.3 to +29.2), but loses
on prose: essay -5.5% (-6.9 to -4.2), 2K -1.9%, 4K -3.0%. Later policy
changes (warm-up before pricing widths, cheap refreshes) were not measured
on MTP at that scale; a two-seed run of the essay was still about 3 to 4%
below depth 3. An MTP chain's draft cost climbs in steps (about 3.6, 6.4,
11.7, 11.9 and 15.7 ms for 1 to 5 proposals), so the depth choice is
close; on prose `--adaptive-mtp=off` is the faster setting today.
The chain depth is chosen from each position's mean acceptance. A position
fades only when it is checked (half weight after 20 checks), one never
checked starts at the mean of the position before it, and positions
holding little evidence for how often chains reach them are probed at most
every 16 steps: priced at the last well-measured position's mean, the
deeper chain worth most is drafted when it beats the chosen depth on this
device's costs, and not at all when it cannot. Without this, a process
whose first chains were 3 deep priced every deeper position at the prior
and stayed at depth 3.

Greedy requests produced byte-identical text to master with the policy on
and off (essay, code, 2K and 4K prompts).

## Draft trees

A DFlash2 draft scores, for each of its seven proposal positions, its top 16
candidates given each candidate of the position before: a first-order lattice
of conditional distributions. A chain verifies one path through it. With
draft trees (`--dflash2-tree=on|off`, `lse_config.dflash2_tree`; by default
on whenever the adaptive policy runs, which then takes a tree or a chain
each step from the device's measured costs) a step can verify many paths in
one target pass (`src/runtime/draft_tree.cpp`):

- **Construction** follows DDTree ([arXiv 2604.12989](https://arxiv.org/abs/2604.12989)):
  a node's value is the product of the conditional acceptance estimates along
  its path, and the best nodes are popped from a heap in value order. A child
  is never worth more than its parent, so the first B pops form the tree that
  maximizes expected accepted nodes over all B-node trees, and the expected
  value of every smaller budget is a prefix sum of the same order. Each
  candidate's estimate is its draft probability mapped through a calibration
  table that learns from every candidate of every row a walk reaches.
- **Size.** The adaptive policy prices a tree step whole (draft, verify and
  commit), per row rung (4, 8, 15 and 31 rows), measured on the device, and
  each step takes the tree rung or chain prefix with the highest expected
  tokens per second.
- **Verification.** Rows are laid out in depth-first preorder, best child
  first. Full-attention layers take the tree's ancestor mask (`MaskKind::kTree`,
  positions per row); Gated DeltaNet layers propagate state along the tree,
  each row from its parent's state, as SpecLA's state-resident serial form
  ([arXiv 2607.16673](https://arxiv.org/abs/2607.16673)) does for a chain and
  TreeWY's per-node recurrence ([arXiv 2608.20961](https://arxiv.org/abs/2608.20961))
  writes for a tree (`gdn.tree_scan.v1`); the causal convolution reads each
  row's ancestors (`causal_conv1d.tree.v1`).
- **Acceptance.** The walk starts at the root: the target answers at a node
  (its argmax, or a sample of its distribution at that node), and the walk
  descends into the child holding that token or stops and emits the answer.
  Every emitted token is the target's own answer at its context, so sampled
  output keeps the target distribution exactly; for a candidate set fixed
  before the target is consulted this is also the most a lossless rule can
  accept (DDTree, SpecInfer [arXiv 2305.09781](https://arxiv.org/abs/2305.09781)).
- **Commit.** Only the accepted path is kept: its K/V rows move into place
  (`kv_page_write.rows.v1`), the Gated DeltaNet state is replayed along the
  path (`gdn.path_state.v1`), the convolution tail takes the path's inputs,
  and the path's features go to the DFlash2 context ring.

Fast wide passes make trees pay. Every projection of a 9 to 31-row tree pass
runs on the 16-row int8 matrix panel, one workgroup applying each weight load
to every 16-row tile (9 to 14 rows run as 15, 17 to 30 as 31); attention takes
`attention.flash_split.wmma16.v1`, the flash tile over a share of the key
windows per workgroup, merged by the split merge. The matrix panel widths are
tree-only (`graph::TreePassScope`). The flash split also serves chain passes
of 2 to 8 rows over FP16 or BF16 K/V (MTP and DFlash2 verify chains, an MTP
draft's first pass): every query head and row of a key head shares its tiles,
so each window of keys and values is read once per tile instead of once per
two heads and four rows. A chain's logits differ from the scalar split's in
their last bits (BF16 matrix operands); one-row decode keeps the scalar
kernel.

Verify pass time on the Linux R9700 (gfx1201), ms, median of 7 per width
after a warm request, master's chain passes against the same widths as tree
passes:

| Rows | Chain, short context | Tree, short context | Chain, 4K context | Tree, 4K context |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 34.1 | 34.1 | 35.4 | 35.3 |
| 2 | 34.8 | 35.1 | 39.8 | 38.7 |
| 3 | 35.3 | 35.3 | 42.5 | 38.6 |
| 4 | 36.0 | 35.9 | 40.7 | 39.3 |
| 5 | 38.7 | 39.3 | 46.3 | 42.3 |
| 6 | 39.8 | 39.4 | 47.2 | 42.9 |
| 7 | 41.4 | 41.0 | 49.1 | 44.4 |
| 8 | 44.0 | 43.6 | 51.6 | 46.9 |
| 9 | 55.3 | 41.9 | 60.4 | 45.2 |
| 12 | 56.2 | 42.9 | 60.9 | 46.6 |
| 15 | 57.1 | 43.7 | 62.3 | 47.2 |
| 16 | 58.9 | 56.3 | 62.8 | 59.5 |
| 20 | 59.7 | 52.8 | 67.3 | 58.6 |
| 24 | 60.5 | 53.1 | 68.1 | 59.4 |
| 31 | 62.9 | 54.9 | 70.3 | 61.0 |
| 32 | 65.1 | 64.2 | 72.1 | 70.0 |

One row is a plain step either way. A tree of 2 to 8 rows costs what a chain does at a short
context and less at 4K (the flash split); trees use 15 and 31 rows, never
16 or 32, which are prompt widths and keep the prompt kernels.

Measured on the Linux R9700 (ROCm), 6 interleaved pairs of servers
(`--dflash2-tree=off` then on), 4 seeds each, so 24 requests per prompt and
arm at temperature 0.6. Changes are the geometric mean of per-seed ratios
with a 95% t interval:

| Prompt | Chain | Tree | Tokens/step chain / tree | Tree rows | Change (95% CI) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 640-token essay | 56.0 tok/s | 61.5 tok/s | 2.92 / 3.50 | 15.1 | +9.8% (+7.2 to +12.4) |
| Code (640 tokens) | 127.0 tok/s | 131.1 tok/s | 7.04 / 7.47 | 10.8 | +3.4% (-1.3 to +8.4) |
| 2K prompt, 256 tokens | 57.6 tok/s | 65.2 tok/s | 3.32 / 4.08 | 16.6 | +13.1% (+7.8 to +18.7) |
| 4K prompt, 256 tokens | 58.0 tok/s | 65.6 tok/s | 3.44 / 4.15 | 15.7 | +12.9% (+8.0 to +18.0) |

On the Mac R9700 (macOS, 2 pairs, 8 requests per prompt and arm, the first
2K and 4K request of the first chain server left out for its kernel
compiles): essay +12.9% (+8.1 to +17.8), code +7.5% (-1.6 to +17.6), 2K
+15.6% (+1.2 to +32.2), 4K +11.6% (+0.0 to +24.6); 66.4 against 75.0, 151.6
against 162.3, 71.0 against 82.2 and 70.5 against 78.7 tok/s.

On a Radeon 8060S (gfx1151), whose 15- and 31-row passes take the GEMM
rather than the matrix panel, the policy keeps mostly to chains and trees
of 4 and 8 rows; against `--dflash2-tree=off` (3 pairs, 12 requests per
prompt and arm): essay -0.3% (-3.0 to +2.5), code -0.3% (-4.4 to +3.9), 2K
-3.1% (-9.6 to +3.9), 4K +5.0% (-0.8 to +11.2).

Greedy output is the target's greedy continuation up to near-ties. Each
emitted token is the target's argmax at its row, but a tree row's logits come
from a different pass shape than a chain's or a plain step's (other kernels,
other sums), so where the top two tokens are within rounding of each other
the choice can flip, and the text diverges from there, as chain DFlash2
output already differs from plain decoding. Scored afterwards by one prompt
pass of the target (`--perplexity`, 384 generated tokens each), the tokens
that pass does not rank first, and by how much (nats):

| Prompt | Plain decode | Chain | Tree |
| --- | --- | --- | --- |
| Essay | 3, at most 0.08 | 3, at most 0.08 | 10, at most 0.29 |
| Code | 1, 0.09 | 4, at most 0.18 | 5, at most 0.66 |
| 2K prompt | 8, at most 1.15 | 8, at most 0.39 | 9, at most 0.55 |

Every such token is a near-tie at the scale on which the scoring pass itself
disagrees with plain decoding.

`test_draft_tree` checks the pop order, optimality against brute force, the
layout and the walk, and runs 300,000 sampled walks per budget against the
target distribution (chi-square). `test_tree_verify --gpu` checks the tree
scan, convolution and path state bit for bit against the chain kernels along
every path, the tree attention against a double reference (to 1.3e-4 with
the flash split), and the K/V row moves; `--gemm M` checks every projection
at M rows of a tree pass against a double reference (5e-3 relative, as the
8-row panel).

## Automatic Q8 conversion

`--dflash2-model` accepts the original BF16 checkpoint as well as a converted
Q8 directory. LSE reads the checkpoint itself to tell them apart. If the config
has a `quantization` block, the directory is loaded as is. If the config has
none and the single `model.safetensors` holds BF16 matrices, it is a source
checkpoint. On first use LSE converts a source checkpoint to affine Q8/group64
(`src/model/dflash2_convert.cpp`) and loads the result.

The conversion is byte-identical to the Python script below. That covers
`model.safetensors`, `config.json` and `source-repository.json`. It streams the
source with bounded memory (one 128-row block at a time) and builds the output
in a temporary directory, then renames it into place. On an M-series Mac it takes
about 6 s. The result is cached:

- By default it goes in `lse-q8g64/` inside the source directory. That works in
  an HF cache snapshot too.
- With `LSE_DFLASH2_CACHE_DIR` set, it goes in
  `$LSE_DFLASH2_CACHE_DIR/<org>-<name>@<revision>-q8g64/`.

On later runs LSE checks the cached manifest against the source's sha256 and
reuses the cached conversion. The source hash comes from a stat stamp, or from
the blob name in an HF cache, so a warm start does not re-read the source.

- **Recorded origin:** for an HF cache snapshot, the repository and revision
  come from the snapshot path. Otherwise they come from an `hf-origin.json`
  (`{"repository": ..., "revision": ...}`) beside the weights.
- **Progress:** goes to stderr as
  `[dflash2-convert] phase=<hash-source|quantize|finish> done=N total=N percent=P`.
- **Turning it off:** set `LSE_DFLASH2_AUTOCONVERT=0` to load a BF16 checkpoint
  unconverted.

`tests/test_dflash2_convert.cpp` checks the conversion against Python-produced
hashes on a synthetic checkpoint. To run it on the real checkpoint, set
`LSE_DFLASH2_CONVERT_SOURCE` to the snapshot below. You can also set
`LSE_DFLASH2_CONVERT_REFERENCE` to a directory the script wrote, and the test
compares against it byte for byte.

## Reproduce Q8 conversion

The converter requires Python and NumPy; the initial conversion used NumPy
2.5.3. It reads the source checkpoint without changing it and requires a new
output directory.

```sh
python3 -m venv .venv-dflash2
.venv-dflash2/bin/python -m pip install numpy==2.5.3
.venv-dflash2/bin/python scripts/convert_dflash2_q8.py \
  "$HOME/.cache/huggingface/hub/models--incoai--Qwen3.8-27B-DFlash2/snapshots/dedf8df68adfb1afeaf7b7480c0a0243108177b4" \
  /path/to/qwen38-27b-dflash2-q8 \
  --source-repository incoai/Qwen3.8-27B-DFlash2 \
  --source-revision dedf8df68adfb1afeaf7b7480c0a0243108177b4
```

All 49 matrix weights and selector codebooks use affine Q8 with group size 64.
Normalization weights and convolution bases keep BF16 storage. Floating
activations and accumulation remain FP32. Scale and bias use BF16; codes are
selected against the stored scale and bias, matching LSE's quantizer convention.
The output includes `config.json` and a `source-repository.json` manifest with
source revision, input/output SHA256, byte count, and tensor count.

The prepared Q8 weights contain 179 tensors and occupy 2,044,950,184 bytes.
Converted weight SHA256:

```text
cc3b5742f8edf02c4edcc734ef66e7c4c5f660b43438641ee36c5a72f8414188
```

## Focused verification

`test_dflash2` covers config compatibility, selector paths, noncausal and sliding
attention masks, grouped convolution, cache rewind/replacement/reset, and
isolation from overwritten target feature buffers. A shorter proposal request
also retains the full trained block's prefix.

`test_dflash2 --gpu-attention` compares the native attention primitive with its
CPU reference for eight queries, head dimension 128, and live context counts 17
and 2055. The native run passed with maximum absolute errors `2.24e-8` and
`3.84e-9`, respectively. It required actual device groups and kernel launches,
with zero host groups or fallbacks.

`test_dflash2 --gpu-selector` exercises all 1792 scores in the seven-position,
top-16, rank-256 selector. Native execution used one device group and zero host
groups or fallbacks. Its maximum absolute difference from independent double
reference math was `2.14577e-6`. The fixture checks finite outputs, exact shape,
and a mixed rounding bound: `gamma_(rank+2) * sum(abs(pred * gate * succ)) +
2u/(1-u) * abs(reference)`, where `u` is FP32 unit roundoff and
`gamma_n = nu/(1-nu)`. The absolute term covers dot-product rounding and
cancellation; the relative term covers unary addition and reference narrowing.

## Vocabulary selection

Large FP32 top-k rows use parallel 512-candidate chunks and repeated merges,
carrying original vocabulary indices throughout. Small rows retain the existing
bitonic route. Equal scores select the smaller original index; positive and
negative infinity retain their normal ordering. The chunked route puts NaNs
first, ordered by original index, so invalid logits stay visible and the
DFlash2 selector rejects nonfinite scores. This explicit behavior replaces the
legacy large-row loop's inconsistent handling of NaNs and all-negative-infinity
rows. `test_topk_parallel` covers that policy and native emission; its separate
`--gpu-topk` mode compares the legacy and parallel vocabulary paths with actual
device execution required. On the measured GPU, the five-run warm wall-time
median for shape `7 x 248320`, `k=16`, fell from `663.126584 ms` to
`2.950625 ms`, about 225 times faster. Both values and original indices matched
exactly, with zero host groups or fallbacks. This measures vocabulary selection,
not complete-engine decode speed.

The separate `--gpu-topk-edges` run also passed four 4097-wide rows selecting
16 candidates: NaNs, both infinities, ties across chunk boundaries and an
all-negative-infinity row. Every value and original index matched the independent
CPU oracle, with four device groups, zero host groups and zero fallbacks.

## Primary references

- [Authors' DFlash2 architecture and acceptance report](https://inco.ai/blog/dflash2/)
- [Pinned reference implementation](https://github.com/z-lab/dflash/blob/07ebd93db9f472af339b644bb70221ad8428328a/dflash/model_mlx.py)
- [Checkpoint model card](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2)
- [Original DFlash paper](https://arxiv.org/abs/2602.06036)
