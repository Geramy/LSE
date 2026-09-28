Recorded on **gfx1201 / AMD R9700**, Apple Silicon with Thunderbolt 5,
Qwen3.8-27B-Q4, native HRX/Loom, MTP off. The warmed matched request used a
1,024-token prompt and 64 generated tokens, two warmups and three measured
requests per configuration, with flush 64 and 64 us completion polling.

| Stage | Qualified Q4 profile |
|---|---:|
| Prefill | **444.532 PP/s** |
| Decode | **24.2635 TPS** |

These are recorded measurements of the accepted qualified INT8 profile,
including split-128 WG128 decode. They establish **400+ PP/s and 23+ TPS**
on this system. The measurement predates the final automatic profile selection
and wave32 normalization promotion; it is not a benchmark of the release archive.

The checkpoint-qualified Q4 INT8 M1/M512 paths, Q4 M512 FFN LDS v2,
FP32 Flash12 prefill, shared-exponential FP32 decode attention, and split-128
WG128 decode are selected by default where qualified. Other kernels and operand
types remain available with their existing diagnostic controls. Unknown
checkpoints retain the exact FP32 path. Floating-point accumulation stays FP32;
integer dot products accumulate in INT32 before FP32 affine restoration.

**Quality uses perplexity.** New qualification windows contain 1,024–2,048
actual target tokens. The 2,046-target automatic Q4 prefill result was 8.127312,
matching explicit INT8 selection, versus 8.100851 for FP32 activations (+0.3266%).
The 1,024-target automatic teacher-forced result was 7.20144530149 versus
7.17831687877 with explicit INT8 (+0.3222%). These are separate fixtures and are
not compared with each other. The accepted Q6 staged-BF16 M512 paths retain
their documented historical 1,022-target qualification; FP8/BF8 alternatives
remain inactive.
