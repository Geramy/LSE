# KV cache formats

Use `--kv-cache-dtype` when you start `lse` or `lse-server`:

```bash
lse-server --model /path/to/model --pool hrx:0 --dialect loom \
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

The CLI option overrides `kv_cache_dtype` in the model configuration. When neither
is present, LSE uses FP32. The startup log reports the selected format.

The 8-bit formats use round-to-nearest-even and finite saturation. Each packed
vector contains its codes and scale. Attention reads that storage directly; LSE
does not retain an expanded FP32 copy. Floating-point accumulation, softmax state,
and attention output remain FP32. The admitted FP16 matrix attention path also
uses FP16 query and probability operands with FP32 matrix accumulators.

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

## GPU selection

The central attention shape table selects the measured matrix path for FP16 KV
on gfx1201. It covers the admitted 64–512-query batches through a 16,384-token
table capacity, plus the 16-query batch at capacity 8,192. Other configurations
use their established attention kernels. BF16 and the 8-bit KV formats do not
select the FP16 matrix path.

Storage codecs and all supported attention routes passed native component checks.
See the accompanying benchmark report for model perplexity and HTTP measurements;
component correctness alone does not establish equal quality or throughput for
all formats.
