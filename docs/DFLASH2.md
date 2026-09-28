# DFlash2 drafting

Enable the separate DFlash2 drafter with `--dflash2=on` and select its checkpoint
with `--dflash2-model`. This replaces native MTP for that process. The target
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

This first implementation uses deterministic draft selection. Target sampling
still governs output, but stochastic draft sampling and rejection-residual
selection are not implemented. Acceptance at nonzero temperature may therefore
differ from the authors' reported rates. DFlash2 remains opt-in.

```sh
curl http://127.0.0.1:8080/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen38-q4","prompt":"Explain how GPU caches improve inference.","temperature":0,"max_tokens":16}'
```

## Bounded HTTP measurement

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
Attention is noncausal within the draft block, including future mask positions;
context keys obey the 2048-position sliding window. Learned dynamic convolutions
have two taps and groups of 16 channels. The selector conditions each choice on
its predecessor using top-16 candidates and rank-256 codebooks.

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
