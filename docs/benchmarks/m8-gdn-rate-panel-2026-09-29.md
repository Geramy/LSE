# M8 GDN rate projection reuse

## Change

One central shape-table entry admits M8/N48/K5120 Q4/group64 to the existing shared group8 activation-panel consumer on its qualified device configuration. The GDN alpha and beta projections use the same input as QKV and the gate. All four contractions share one producer, with four consumer references. No new kernel body, runtime option, weight layout or persistent weight copy is added.

The existing shapes retain their selected schedules. Floating-point accumulation stays FP32. The accepted M8 down WMMA and two-row gate/up schedules remain active.

## Component timing

Preparation-inclusive alpha/beta chains, 20 interleaved ABBA rounds after warmup. Each arm has 40 samples. The shared producer is included once in both arms because the larger projections already require it. Code compilation is outside timing. Passive device timestamps measure GPU execution; wall time includes submission and completion.

| Metric | Control ms | Shared-panel ms | Reduction |
|---|---:|---:|---:|
| Mean GPU chain | 0.050042 | 0.024935 | 50.17% |
| Median GPU chain | 0.048920 | 0.025380 | — |
| Mean wall chain | 0.237459 | 0.187337 | 21.11% |
| Median wall chain | 0.236438 | 0.185459 | — |

The candidate is faster in all 20 paired rounds for GPU and wall time. Mean alpha time is 22.418 → 10.049 microseconds; beta is 22.910 → 10.045 microseconds. One producer takes about 4.7–4.8 microseconds. Native timing covers 360 direct device launches, with zero host groups or fallbacks.

Both consumers retain 96 VGPRs and zero private scratch. LDS falls from 33,024 bytes to zero; three barrier pairs are removed. Static global-load counts increase: alpha 58 → 202 and beta 42 → 186. The measured improvement includes that tradeoff. No achieved occupancy or DRAM-bandwidth claim is made.

## Correctness and integration

Captured alpha and beta control source and code objects match the active profile shaders byte-for-byte. Complete eight-row raw, alpha and beta outputs (384 elements each) match exactly. The fixture includes the actual alpha exp/softplus and beta sigmoid/clamp epilogues. Independent maximum absolute arithmetic errors are 1.93e-7, 4.09e-8 and 8.09e-8 respectively. Codec, readonly input, allocation/padded guards and zero/rebound replay checks pass.

The real graph admission check proves that QKV, alpha, beta and gate share one panel node. A cloned input gets its own producer; retired weak-cache entries expire. Shape, device, wave size, thread capacity, DOT4 support, staged-source and affine-format exclusions are covered.

The coherent production rebuild completes 23 rebuild/link steps. Four focused suites pass: quant defaults, quant dispatch, Q4 activation panel and Q4 matrix panel. Registered producer and raw/fused consumer shaders and resource metadata match all eight private measured stages. The registered native check passes 24 device dispatches, zero host groups/fallbacks and complete output/codec/replay/guard comparisons.

The first registered check was run inside the workspace sandbox and failed before dispatch at HSA user-client discovery. The identical binary, code objects and runtime libraries pass outside that sandbox. This was an access failure, not a shader correctness failure. No runtime installation or driver change was made.

## Scope and evidence

This is a component gain. Baseline, MTP and DFlash2 token throughput have not been remeasured for this table entry. The previous v0.4.14 comparison remains explicitly scoped to source c11f103 and its measured binary. No additional perplexity run is added for this bit-identical schedule selection.

Local evidence is stored under build/release/pi-performance/m8-rate-panel-component: source/code-object hashes, resource census, raw CP samples, timing distributions, graph proof, coherent build and CPU logs, registered emission identity and native loader/output logs. Bundled HSA, HRX and Loom hashes match the earlier successful native probe. The accepted v0.4.14 executable is preserved in its release artifact directory.
