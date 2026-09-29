# Final same-binary mode comparison

One pair per mode on the same R9700/gfx1201 GPU and Qwen3.8-27B Q4 target. Identical 1024-token coding requests, 384 generated / 383 timed decode tokens. BF16 KV, FP32 floating accumulation, launch temperature 0.6, top-k 20, top-p 0.95, seed 1234, batch/ubatch 1024 and KV capacity 262100. MTP uses the Q8 auxiliary model at depth3; DFlash2 uses Q8 block8/seven proposals.

Each mode starts with an empty private disk cache. The second request retains compiled code and reuses zero prompt KV. Compilation is included. No profiling or competing workload is present. Zero host groups and fallbacks in all requests. This is not a long-context Pi replay or a statistical estimate.

| Mode | Cold PP/s | Cold TPS | Resident PP/s | Resident TPS | Resident acceptance |
|---|---:|---:|---:|---:|---:|
|Baseline|442.18|24.14|624.10|24.73|—|
|MTP=3|403.95|37.96|608.19|48.62|80.70%|
|DFlash2, seven proposals|417.14|31.39|616.78|42.41|76.15%|

All three resident requests exceed 600 PP/s on this workload. Baseline 29 TPS, MTP3 49 TPS and DFlash2 103 TPS goals remain unmet. Do not apply these rates to every prompt, sampling setting or total context length.

The M8 scheduling change improves matched DFlash2 resident decode from 40.68 to 42.41 TPS (+4.27%) with exact responses and sampling statistics. It does not alter baseline single-token or MTP four-row decode kernels. Baseline/MTP rates are effectively unchanged from the preceding pair. See the scoped scheduling report.

Server SHA256: `4c635d610f6bfb7545b81ededa5a356f1b2a1b5389bdd28d736a5d24837a88dd`. Source8150263 plus the qualified M8 row-pair patch and its final documentation. Normal requests and sampling settings are byte-identical across modes, but mode-specific RNG execution can produce different answers.

Raw evidence remains local under final-forward-baseline-http/, final-forward-mtp3-http/, m8-dot4-rowpair-http-candidate/ and FINAL-FORWARD-MODE-COMPARISON.json. Runtime library identities are in the server logs.
