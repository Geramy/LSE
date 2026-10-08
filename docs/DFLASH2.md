# DFlash2 drafting

Enable the separate DFlash2 drafter with `--dflash2=on` and select its checkpoint
with `--dflash2-model`. Without `--dflash2-model`, LSE uses the draft it pairs
with the target (for Qwen3.8-27B, `incoai/Qwen3.8-27B-DFlash2` at the revision
below) from the Hugging Face cache, downloading it first when it is not cached;
`--offline` turns a missing draft into an error that names the `pull` command.
See [Get a model](../README.md#get-a-model). This replaces native MTP for that process. The target
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
requests retain deterministic draft selection and verify all seven proposals.
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
  holds what the extra rows cost. A step that recorded a program (a width's
  first pass in a request) or compiled a kernel is left out. Nothing is
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

Greedy requests always verify all seven proposals. The target's logits differ
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

Greedy requests produced byte-identical text to master with the policy on
and off (essay, code, 2K and 4K prompts).

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
