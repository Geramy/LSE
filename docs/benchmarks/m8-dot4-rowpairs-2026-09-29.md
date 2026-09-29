# M8 Q4 two-row activation scheduling

## Change

Only the shape-table entries for M8/N17408/K5120 and M8/N10240/K5120 select the new four-chunk load schedule. It decodes four weight words once, consumes two independent activation rows at a time, and combines four activation steps into one 16-byte load per row. Each row keeps the same chunk order, FP32 FMAs, scale/bias restoration and split reduction. The group8 producer codec, compact weights, grid, wave32/WG256 and output dtype remain unchanged. Other rows and shapes retain their accepted schedules.

## Component result

One ABBA20 pair per shape after 16 warm chains per arm. Preparation is included; no compilation during timing.

| N/K | Wall control ms | Wall candidate ms | GPU control ms | GPU candidate ms | GPU reduction |
|---|---:|---:|---:|---:|---:|
|17408/5120|0.189767|0.160598|0.170748|0.141404|17.19%|
|10240/5120|0.121180|0.104653|0.101932|0.084868|16.74%|

Complete output bits, group8 raw panel codec, independent arithmetic oracle, zero replay, readonly inputs and padded/allocation guards pass. The integrated native test also passes complete fused-epilogue output comparisons. No additional perplexity run is added for this bit-identical scheduling change.

Actual captured gate/up/QKV baseline shaders and code objects match the offline control byte-for-byte. Candidate VGPRs fall 100 → 88; LDS and private scratch stay zero. Static global loads decrease by 16 and load waits by 22 per shader. DOT4 instruction count stays 160. VOPD pairs decrease 5 → 3; this tradeoff is retained in the evidence. The measured result is faster despite fewer dual instruction pairs. Register counts do not establish measured occupancy.

## Matched HTTP result

Exact 1024-token coding requests, 384 generated / 383 timed decode tokens, Qwen3.8-27B Q4 target and Q8 DFlash2 block8/seven proposals. BF16 KV, FP32 floating accumulation, temperature 0.6, top-k 20, top-p 0.95, seed 1234, batch/ubatch 1024 and KV capacity 262100. Empty private disk cache per process; second request compiled code resident, zero prompt KV reuse. Compilation is included. No profiler or competing CPU/GPU workload is present.

| Request | Control PP/s | Candidate PP/s | Control TPS | Candidate TPS |
|---|---:|---:|---:|---:|
|First, cold process|416.77|417.14|30.47|31.39|
|Second, resident process|619.08|616.78|40.68|42.41|

Resident decode improves 4.27%; verifier elapsed 7823.174 → 7440.270 ms over 102 passes. Warm prefill is effectively unchanged (−0.37% in this single pair). Both requests, generated choices, usage, acceptance counts and all probability statistics match exactly. Resident 281/369 proposals accepted (76.15%); zero host groups/fallbacks. JIT count and device group count remain unchanged.

Focused activation-panel 9/9, matrix-panel 10/10 and dispatch 10/10 suites pass. Independent review found no concrete issue with scope, bounds or alignment. The accepted M8/M1024 down emitted bodies and complete plans remain byte-identical. No packed weights are duplicated. No peak-VRAM or occupancy claim is made.

## Identity and limits

Candidate starts at source 8150263 with this scoped change. Server SHA256: `4c635d610f6bfb7545b81ededa5a356f1b2a1b5389bdd28d736a5d24837a88dd`. Native identity and actual bundled HSA/HRX/Loom mappings are logged.

This is a controlled coding workload. It does not establish these rates for long-context Pi chat. The baseline and speculative decode targets remain unmet. The final same-binary resident comparison measured baseline 24.73 TPS, MTP=3 48.62 TPS and DFlash2 42.41 TPS; all three prefill rates exceed 600 PP/s on these 1024-token requests. See [final mode comparison](forward-modes-final-2026-09-29.md).

Local evidence: m8-dot4-rowpair-component/ CP samples, source/object hashes, captured-baseline equality and ISA census; m8-dot4-rowpair-integrated-native* logs/identity; m8-dot4-rowpair-http-{candidate,comparison.json}; prefill-projection-cooperative-http-candidate/ control. Raw requests stay local.
