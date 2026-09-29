### Pi chat: shorter compilation stalls

Measured on macOS with an AMD R9700 (`gfx1201`), native HRX/Loom, the mapped HSA
runtime, Qwen Q4 weights, Q8 DFlash2, depth 3, and FP32 KV storage. Sampling uses
temperature 1, top-k 20 and top-p 0.95. Each process starts with an empty disk
kernel cache. Both turns run in the same process.

| Measurement | Earlier baseline | Compiler fixes |
| --- | ---: | ---: |
| First prefill, 5,207 tokens | 91.53 s | **16.88 s** |
| First prefill rate | 56.89 tokens/s | **308.54 tokens/s** |
| Follow-up prefill, 23 new tokens | 16.14 s | **0.76 s** |
| Total kernel compilation | 92.91 s | **3.35 s** |
| Unique compilations | 427 | **372** |
| First decode | 26.64 tokens/s | **27.11 tokens/s** |
| Follow-up decode | 31.34 tokens/s | **31.28 tokens/s** |

Both responses and acceptance counts match exactly. These are individual paired
workload measurements; prompt timing excludes model loading. The changes reduce
compilation stalls. They do not establish a steady decode speedup or 46–48
tokens/s on Pi chat. See the [CPU/GPU profile and verification report](https://github.com/Geramy/LSE/blob/master/docs/benchmarks/pi-execution-profile-2026-09-28.md).

The compiler shares code only when complete generated bodies match. Attention
page traversal remains a device loop. Invocation bindings, device ownership,
launch metadata and optimizer measurements remain distinct.

A subsequent DFlash buffer-view fix removes repeated traversal of materialized
feature graphs, 65 copy dispatches and three compiled kernels. The same two
responses remain exact. The final pair measures 309.10 prompt tokens/s and
27.20 / 32.13 decode tokens/s. This small decode change is not a statistical
speedup claim.

### Optional KV storage

`--kv-cache-dtype` accepts `fp32`, `fp16`, `bf16`, `fp8` and `bf8`. FP32 remains
the default. Target and MTP paged caches use the selected format; the DFlash2
private ring remains FP32. FP16 halves paged KV storage and selects matrix
attention on supported gfx1201 shapes. Floating-point accumulation stays FP32.

One paired check with 1,024 scored target tokens measured perplexity **4.9049**
for FP32 and **4.9014** for FP16 plus matrix attention. BF16 and the 8-bit formats
have component validation; this model comparison covers FP16. See [KV formats](https://github.com/Geramy/LSE/blob/master/docs/KV_CACHE.md).

### macOS runtime

The macOS archive bundles the mapped-transfer HSA update. Its launcher selects
the bundled runtime; no `DYLD_LIBRARY_PATH` setting is required. The activated
MacAMDGPU DriverKit extension is still required and is installed separately.
The Linux archive uses the installed ROCm/HRX runtime.
