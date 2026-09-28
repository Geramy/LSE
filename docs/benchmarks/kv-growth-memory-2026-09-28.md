# KV growth memory retention fix

## Cause and correction

When a target KV pool grew, retained forward programs still owned earlier
recurrent-state computation graphs. Those graphs kept old activation buffers and
KV allocations alive. Their KV bindings were obsolete, so the programs could
not replay.

The fix keeps the current recurrent-state buffers, detaches their old computation
history, and releases retained programs when the KV allocation moves. Compiled
kernels and the current KV contents remain available. Kernel arithmetic, FP32 KV
storage, model weights, and the prefill plan are unchanged.

## Measured memory

Radeon AI PRO R9700, gfx1201, macOS ARM64, Qwen3.8-27B Q4 with the Q8 DFlash2
module, three verified proposals, 14,000 prompt tokens and 64 generated tokens.
Configured KV limit: 262,100. The target KV pool grows on demand; its capacity at
this context is 16,384 slots, about 2 GiB in FP32.

| Measurement | Before | After | Reduction |
|---|---:|---:|---:|
| Live backend allocations after model load | 18.23 GiB | 18.23 GiB | — |
| Live backend allocations at the last target pass | 30.72 GiB | 22.33 GiB | **8.39 GiB** |
| Peak driver VRAM reservation sampled at 1 Hz | 30.93 GiB | 24.86 GiB | **6.06 GiB** |

Live allocation accounting records each backend allocation and its final release.
It does not add tensor views or shared aliases more than once. Driver accounting
sums two disjoint VRAM allocator regions. It includes reusable runtime pool
reservations, so it differs from live allocation bytes. The peak is a sampled
value, not a hardware high-water counter.

These memory figures are from one request per instrumented build. Generated text,
finish reason, usage and proposal acceptance matched. Both runs reported zero CPU
fallbacks and 87,319 device groups.

## Performance check

Diagnostic allocation logging was removed from the final executable. The second
request, after one initialization request, used the same prompt and parameters as
the prior accepted query4 comparison.

| Measurement | Prior accepted build | Memory fix |
|---|---:|---:|
| Prompt processing | 312.67 tokens/s | 323.41 tokens/s |
| Decode | 42.56 tokens/s | 42.41 tokens/s |
| Decode duration, 63 timed tokens | 1,480.215 ms | 1,485.627 ms |
| Target verification | 1,168.232 ms | 1,165.472 ms |
| CPU fallback groups | 0 | 0 |

These are individual samples, not a throughput distribution. The 0.36% decode
change does not establish a speed regression or improvement. No material speed
loss was observed. The prompt is repetitive and accepted all 47 proposals, so
these rates do not predict ordinary coding conversations. The measured requests
compiled no new kernels. Their generated output and usage matched exactly.

## Regression coverage

The new test crosses two KV growth boundaries with a small hybrid recurrent and
attention model. Weak references verify that earlier recurrent computation nodes
are released. It also exercises subsequent decode and session restart. The test
fails on the original implementation at both growth boundaries and passes with
the fix. The runtime suite reports 57 passed and 5 skipped. No perplexity run was
needed for this ownership-only change.

Local evidence directory: `build/pp-optimization/vram-audit-20260928` in the
mac_amdgpu checkout. It contains allocation logs, driver samples, HTTP responses,
source patch, executable hash and summary. The pre-fix source is LSE `78730ae`.
