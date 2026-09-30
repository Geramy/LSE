1. Faster Q4 prefill with optimized nibble expansion and paired activation staging: **597.9 pp/s peak** on a warm full prompt with BF16 KV and no prompt cache reuse.
2. Fused eligible single-token Q4 SwiGLU gate/up projections and activation; updated dispatch regression coverage.
3. Fixed RDNA4 prefetch address/span lowering in bundled Loom. Previously observed live DFlash2 peaks: **67.6 tok/s decode** and **96% acceptance** on Qwen3.8-27B Q4, R9700/macOS. These are separate workload peaks.
