# Cooperative M1024 Q4 projection shapes

## Change

The exact M1024/N10240,6144,12288/K5120 Q4 group64 shapes now use the accepted cooperative gate/up schedule. Each uses BF16 affine planes, F32 input/output, gfx1201/wave32/WG256 and 4,608 bytes LDS, with required barrier support. The shape table selects this existing body; no new numeric body or producer is added. Projections with the same input share one 5.625 MiB typed activation panel. Compact weights are unchanged. FP32 floating accumulation is retained.

## Component measurement

One ABBA20 pair per shape, preparation included; 16 warm chains per arm. Passive physical GPU timestamps:

| Shape N/K | Control ms | Producer ms | Consumer ms | Candidate chain ms | Reduction |
|---|---:|---:|---:|---:|---:|
| 10240/5120 | 3.790614 | 0.108032 | 2.262790 | 2.370822 | 37.46% |
| 6144/5120 | 1.793577 | 0.084470 | 1.075163 | 1.159633 | 35.35% |
| 12288/5120 | 3.869093 | 0.107384 | 3.055268 | 3.162652 | 18.26% |

Complete output bits match the prior WMMA body for all rows and columns. Independent group64 arithmetic oracle, raw codec, zero rows/replay, readonly inputs and allocation/padded-row guards pass. 513 native dispatches, zero host/fallback/compiles during timing. Consumer: 110 → 108 VGPRs, 16 → 18 SGPRs, LDS 6,656 → 4,608 bytes. Producer: 27 VGPRs, 22 SGPRs, LDS 0. Private scratch is zero for both. These are static resources, not measured occupancy/cache hits. Outliers remain included.

## Integrated HTTP comparison

Exact 1024-token requests,384 generated / 383 timed decode tokens, temperature 0.6, top-k 20, top-p 0.95, seed 1234, BF16 KV, batch/ubatch 1024, capacity 262100, Qwen3.8-27B Q4 and Q8 DFlash2 block8 with seven proposals. Each process starts with an empty private disk cache. The second retains compiled code but reuses zero prompt KV. Compilation remains included. CPU/GPU profiling and competing workloads are absent.

| Request | Control PP/s | Candidate PP/s | Control TPS | Candidate TPS |
|---|---:|---:|---:|---:|
| First, cold process |383.41|416.77|30.49|30.47|
| Second, resident process |568.03|619.08|40.73|40.68|

Resident prefill improves 8.99%; elapsed 1802.733 → 1654.074 ms. Decode is effectively unchanged (−0.12% in this pair). Requests, generated choices, usage and every acceptance/probability statistic match exactly. Resident: 102 passes, 281/369 proposals accepted (76.15%). All requests have zero host groups/fallbacks. Cumulative JIT compiles: 284 → 283; device groups increase by 16 for additional preparation dispatches. No packed weights are duplicated. No peak-VRAM number was measured.

This exceeds 600 PP/s on this controlled resident 1024-token workload. It does not establish 600 PP/s at every context length. The 29 baseline, 49 MTP and 103 DFlash2 TPS goals remain unmet. The earlier same-binary comparison measured 24.79 / 48.93 / 40.73 TPS before these prefill-only table entries; do not label that comparison as the new binary.

## Qualification and identity

CPU panel 10/10 and dispatch 10/10 pass; registered native qkv/gdn-z/attn-q checks pass complete baseline output/oracle/codec/replay/guards. Independent review found no concrete issue. The accepted M8 and M1024 down emitted bodies and complete launch plans remain byte-identical. No perplexity run is added for these bit-identical schedule changes.

Control is source67a0b4e plus accepted cooperative-up changes (de3afcf), measured server SHA256 8e5521f0700efefd2ad0d832c8c1492c4d0f32758536f59fa181cf38141906e6. Final capability-only barrier admission fix leaves normal Loom bodies unchanged.
Candidate starts at c2bd566 with these table/test changes, server SHA256 `18a4d9cd0aef525fda402ee8fcadd61b2acb44f6d9a4f6fe675fc07f238abf6f`. Bundled HSA/HRX/Loom identities are recorded.

Local evidence: prefill-projection-cooperative-component/ contains CP samples, resources, frozen shaders/objects, hashes and retained-emission equality; prefill-projection-native-identity.json and native logs; prefill-projection-cooperative-http-candidate/ request/response/command/log files; prefill-projection-cooperative-http-comparison.json. Raw requests stay local.
