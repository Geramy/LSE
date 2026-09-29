### Completed prefill memory fix

Release completed target prefill programs before decode and clear completed DFlash2 context programs on cold reset. Keep materialized recurrent state, paged KV, weights and compiled kernels. This fixes a reproduced second-request 64 MiB allocation failure when using batch/ubatch4096. Both exact4096-token requests now complete. The result is an allocation-lifetime fix; no peak-VRAM reduction is claimed without a measurement.

### Full-width DFlash2

All seven block8 proposal positions are active. The conditional drafter follows the request temperature and uses probability-ratio acceptance with residual sampling. Both prefill batch limits default to1024; supported launch values are128–4096. `--temperature` sets a server default; explicit request values override it. The launch examples use0.6. The checkpoint generation file remains unchanged.

### M8 Q4 down WMMA

The measured gfx1201 M8/N5120/K17408 down projection uses a shared padded fragment panel and INT32 WMMA with FP32 affine restoration. Compact Q4 weights are preserved. Its panel producer and consumer have zero private scratch and zero LDS. Faster paired DOT4 gate/up and other accepted schedules remain active.

A matched1024-target M8 quality pair measured perplexity4.8660→4.8675 (+0.031%). Both arms used BF16 KV, finite FP32 logits and zero host/fallback execution. Native codec, padding, residual, replay and buffer-guard checks pass. Opaque token replay also now reads authoritative device inputs instead of an absent host mirror.

### Measured HTTP throughput

macOS/R9700 gfx1201, Qwen Q4 target, Q8 DFlash2 block8, BF16 target KV, temperature0.6, top-k20, top-p0.95, seed1234, batch/ubatch1024. Two exact1024-token prompts each generated384 tokens. Each process starts with an empty cache. The second request reuses compiled kernels but zero prompt KV; previously unseen kernels can still compile. Timings include compilation.

| Request | Control prompt tokens/s | Release prompt tokens/s | Control decode tokens/s | Release decode tokens/s |
|---|---:|---:|---:|---:|
| First, cold process |357.49|360.08|29.50|30.19|
| Second, resident process |501.79|500.20|35.42|40.21|

Second-request verification falls90.01→77.43ms/pass. Both runs execute102 passes and accept281 proposals. Text differs; this is a matched-settings measurement, not exact-output timing. The result is a13.5% second-request TPS gain, with unchanged prefill throughput. Baseline and MTP were not remeasured. The29TPS baseline,49TPS MTP,103TPS DFlash2 and600PP/s goals remain unmet.

See [measurements, quality and limits](https://github.com/Geramy/LSE/blob/master/docs/benchmarks/wmma-m8-down-2026-09-29.md).

### Packages and compatibility

Build both Linux x86_64 and macOS arm64 from the same tagged source. The macOS archive bundles HSA/HRX/Loom; its launcher selects them without a system library override. An activated MacAMDGPU driver remains required. Linux uses compatible installed ROCm/HSA and the packaged HRX/Loom runtimes. Each archive includes a source/runtime manifest and checksum. CI verifies host behavior, compilation and relocation; actual gfx1201 GPU execution was checked on the local macOS eGPU.

Model-declared BF16 KV is the default for the current Qwen checkpoint; FP16 is the standard default for other models unless overridden. FP8/BF8 and FP32 KV remain explicit options. Floating-point accumulation remains FP32. The temperature parser is compatible with the macOS15 deployment target.
