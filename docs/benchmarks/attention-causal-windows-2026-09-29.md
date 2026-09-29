# WMMA attention: skip fully masked windows

A workgroup-uniform predicate skips the entire QK / online softmax / PV body when a 256-key window starts beyond the row length or beyond the last query's causal position. All lanes in the workgroup use the same query tile, row length, offset and window index. All barriers inside the skipped body share that predicate. Windows that may contribute retain the accepted arithmetic and ordering.

## One component ABBA comparison

M1024, 24 query heads / 4 KV heads, QK and value width 256, BF16 storage and FP32 accumulation, WG256, 1536 workgroups. Each arm has 16 warm invocations, 40 timed invocations in ABBA20 order, and one empty replay. Compilation/allocation/readback are outside timing. Passive physical GPU timestamps use the reported 100 MHz device frequency.

| Live keys | Control GPU ms | Guard GPU ms | Reduction |
|---|---:|---:|---:|
| 1024, empty preceding context |4.486440|3.107093|30.74%|
| 5207, 4183 preceding keys |25.565322|23.661592|7.45%|

Wall means are 4.5100→3.1326 ms and 25.5898→23.6838 ms. These are component results, not engine token rates. The longer context still requires attention over the preceding keys.

Full output bits match in both large cases and all ragged edge cases. Independent double reference uses actual BF16/FP16 rounded operands; maximum large-case error 1.78e-4 and 1.21e-5. Causal, noncausal and sliding FP16/BF16 edge cases, padded poisoned rows, readonly inputs, zero replay and 64-byte allocation guards pass. The initial private performance fixture compared unrounded FP32 values and stopped on the accepted baseline; its oracle was corrected to the actual stored values before this pair. No production arithmetic or tolerance was changed.

Large-case resource counts: control 120 VGPR / 40 SGPR, candidate 136 / 46; both LDS 24768 bytes and private scratch 0. Register use increases; no achieved-occupancy or cache-counter claim is made. 228 native device dispatches in the performance pair; zero host or fallback dispatches. The fixture's historical arm label 'split' identifies this private window-guard candidate, not the separate split decode kernel.

## Integration

The existing host attention emission suite passes 7/7. The registered native runtime integration suite passes 80/80 with an empty private cache and the same bundled runtime. Matched HTTP qualification completes with byte-identical requests, identical generated responses and all speculative acceptance/probability statistics. First process: 370.02→370.35 PP/s and 30.42→30.42TPS. Second resident request: 522.58→526.26 PP/s (+0.71%) and 40.78→40.66TPS (−0.30%, effectively unchanged). Temperature 0.6/top-k 20/top-p 0.95/seed 1234, seven proposals, BF16 KV, batch/ubatch 1024; no prompt KV reuse, host groups or fallbacks. Compilation remains included. One pair does not establish a universal rate. No perplexity run was added for this bit-identical change. Control source c1ee42d, server SHA256 3a5d1ed0121c36465f3e713d9bfc0aaf00c93343a4c35452b98b3a41c6504884; candidate server SHA256 fae118c70699bf6d5087fa7490ac5c2071df3f5e39e966149013a89282fe2b81. Requests/response/command/mapped-library logs are in ../attention-causal-window-http-candidate and ../prefill-m1024-http-candidate; exact comparisons in ../attention-causal-window-http-comparison.json. The 600 PP/s and 103 DFlash2 TPS targets remain unmet.

GPU-SUMMARY.json contains distributions. Raw passive .ireeprof/.jsonl, offline and native logs, exact private source and hashes are retained here. The existing M8/M1024 down and gate/up paths remain intact.
