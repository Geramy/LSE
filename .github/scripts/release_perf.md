### Removed monolithic decode attention

The old single-token attention implementation and its dispatch entries are
removed. Split attention now covers long tables subject to actual index and LDS
limits. Native boundary checks include a 262,144-key capacity with 8 KiB merge
LDS; this is not a full 262K-context throughput result.

### MTP prompt reuse

MTP preserves verified target state, aligns its draft cache, and keeps an owned
last target hidden row between requests. On the two-turn Pi workload, follow-up
prefill falls from **14.247 s to 0.709 s**: 5,310 tokens reused, 23 new tokens.

Measured on macOS/R9700 gfx1201 with Qwen Q4, Q8 draft weights, FP32 KV,
temperature 1, top-k 20 and top-p 0.95. Each process starts with an empty kernel
cache. The first prompt has 5,207 tokens.

| Mode | First prompt tokens/s | First decode tokens/s | Follow-up decode tokens/s |
| --- | ---: | ---: | ---: |
| Plain Q4 | 318.42 | 19.71 | 19.72 |
| MTP=3 | 293.95 | 32.68 | 31.90 |
| DFlash2, depth 3 | 291.75 | 27.97 | 35.26 |

MTP and DFlash produce identical responses in both candidate turns. MTP first
response and acceptance counts match the previous build. Plain responses and
speculative follow-up responses changed from the previous build, so those rate
comparisons are not exact-output comparisons. DFlash first-turn rate was 28.59
before this change; this pair does not establish an improvement there.
The higher 46–48 tokens/s target remains unmet on these Pi requests.

All 13 new MTP state tests and 18 native attention boundary cases pass. The
three HTTP runs use zero CPU fallback groups. No new perplexity run was needed.
See the [profile, measurements and limits](https://github.com/Geramy/LSE/blob/master/docs/benchmarks/pi-execution-profile-2026-09-28.md).

### Packages

Linux x86_64 uses the installed ROCm/HRX runtime. The macOS arm64 archive bundles
HSA, HRX and Loom; its launcher selects those libraries without a
`DYLD_LIBRARY_PATH` override. The activated MacAMDGPU driver is installed
separately. FP32 remains the default KV format; optional FP16/BF16/FP8/BF8 storage
and the existing kernel cache remain available.
