# KV cache formats

Use `--kv-cache-dtype` when you start `lse` or `lse-server`:

```bash
lse-server --model /path/to/model --pool hrx:0 \
  --kv-cache-dtype fp16 --kv-len 32768
```

The setting applies to the target model and MTP paged KV cache. DFlash2 keeps its
private ring in FP32. Model weights are unchanged. A request cannot change the
format of a running server.

| Value | Storage | Scale |
| --- | --- | --- |
| `fp32` | FP32 | None |
| `fp16` | IEEE FP16 | None |
| `bf16` | BF16 | None |
| `fp8` | OCP E4M3FN | One FP32 scale per token and KV head, separately for K and V |
| `bf8` | OCP E5M2 | One FP32 scale per token and KV head, separately for K and V |

The CLI option overrides `kv_cache_dtype` in the model configuration. A top-level
`kv_cache_dtype` overrides the same setting in `text_config`. If neither is set,
LSE uses BF16 when the model declares BF16, and FP16 otherwise. Model dtype
metadata is read in this order: `text_config.dtype`, `text_config.torch_dtype`,
`dtype`, then `torch_dtype`; null values are skipped. `--kv-cache-dtype fp32`
selects an explicit FP32 control. The startup log reports the selected format.

The 8-bit formats use round-to-nearest-even and finite saturation. Each packed
vector contains its codes and scale. Attention reads that storage directly; LSE
does not retain an expanded FP32 copy. WMMA uses FP16 Q/K/P/V matrix operands for
FP16 storage and BF16 operands for the other four formats, after decoding or
conversion. Matrix accumulators, online softmax state and attention output remain
FP32. The single-token and short-query split kernels retain FP32 floating-point
calculation while reading typed KV storage.

## Memory

For Qwen3.8-27B's 16 full-attention layers, four KV heads and head dimension 256:

| Format | KV bytes per allocated token | KV at 16,384 allocated tokens |
| --- | ---: | ---: |
| FP32 | 131,072 | 2 GiB |
| FP16 or BF16 | 65,536 | 1 GiB |
| FP8 or BF8, including scales | 33,280 | 520 MiB |

These figures include both K and V. They exclude model weights, recurrent state,
MTP, the DFlash2 ring and temporary buffers. Allocation follows page capacity;
live token count can be smaller than allocated capacity.

An MTP module adds one attention layer in the same format: 4,096 bytes per token
in BF16 for Qwen3.8-27B. The DFlash2 ring is fixed at its sliding window
(84,172,800 bytes for the Qwen3.8-27B draft) and does not grow with context.

### Allocation granularity

Tokens are paged in blocks of 16. How blocks become device memory depends on
the kernel dialect:

- **Loom (the default on every platform):** each layer's K and V are backed by 256 KiB
  fragments covering the blocks in use, drawn from 256 MiB arenas that all
  layers share. A layer's address table is reserved for the whole `--kv-len`.
- **HIP (`--dialect hip`, Linux only):** each layer's K and V are contiguous pools. A pool starts at 8 blocks
  and doubles up to 2,048 blocks (32K tokens), then grows 256 blocks (4K tokens)
  at a time, never past `--kv-len`. While a pool moves to its next size, that
  layer's old and new pools are briefly both allocated.

### Estimating before loading

`lse-server --model-info` lists the bytes per token and per block for every
format, computed by the same rules the allocator uses. `lse-server --estimate`
(or `lse_estimate` in libLSE, or `/v1/lse/estimate` on a running server) adds
the paging granularity, the MTP or DFlash2 cache, weights, recurrent state and
workspace for a whole configuration, and with `device_memory_bytes` reports the
longest `--kv-len` that fits:

```bash
lse-server --model /path/to/qwen38-27b-q4 --kv-len 65536 --kv-cache-dtype fp8 \
  --estimate='{"device_memory_bytes": 34359738368}'
```

## GPU selection

The central shape table selects `attention.flash.wmma16.v2` for eligible paged
batches on gfx1201/wave32 across all five storage formats. Query width starts at
2; logical K and V head widths can differ and reach 512. The rule requires the
matching F16/BF16 matrix capability, 256-thread workgroups, sufficient LDS, valid
metadata/table geometry and a power-of-two page block that divides the 256-key
window. It has no 64-query minimum or fixed 16,384-token table ceiling. Traversal
uses clamped live metadata; inactive pages and padded rows are guarded.

Single-token and selected short-query shapes keep their split-attention routes.
Other configurations use their established attention implementation.

Native qualification covers all five formats, ragged/padded rows, causal,
sliding and unmasked attention, nondivisible head tails, blocks 2/8/16, empty
replay, finite scaled values around 1e30, and the maximum Dh=Dv512 boundary.
FP16/BF16 additionally cover M512 at live lengths 5,610 and 14,000. All 39 cases
passed with 137 device dispatches and zero host/fallback groups. See the
[WMMA component report](benchmarks/flash-wmma-general-2026-09-28.md) for resources,
precision contracts and passive device durations.

## Model quality

An earlier matched 1,024-token pair measured perplexity 4.9049 for FP32 KV with
scalar Flash12 and 4.9014 for FP16 KV with the narrower FP16-operand WMMA v1 path.
A later BF16 KV/generalized WMMA v2 check scored the same 1,024 pinned targets at
PPL 4.849896867834, with 3,142 device groups, zero host/fallback and all logits
finite. All 32 paged layers across two M512 passes selected v2 at capacities 512
and 1,024. This is one later-build prefill measurement against historical references;
it does not establish sampled-conversation quality or a statistical improvement.
FP8/BF8 retain component checks without a model-quality result. See the
[current BF16 report](benchmarks/bf16-mode-comparison-2026-09-28.md) and
[original quality method](benchmarks/kv-storage-attention-2026-09-28.md).
