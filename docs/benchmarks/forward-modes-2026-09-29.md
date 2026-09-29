# Same-binary mode comparison

Sequential same-binary processes, fresh private disk cache each process; second request compiled code resident, no prompt KV reuse; temperature0.6 top_k20 top_p0.95 seed1234 BF16 KV batch/ubatch1024, capacity262100;1024 input384 output383 timed decode. Compilation included. Zero host/fallback. One pair each, not statistical estimates.

| Mode | Cold PP/s | Cold TPS | Resident PP/s | Resident TPS | Resident acceptance |
|---|---:|---:|---:|---:|---:|
| Baseline | 409.45 | 24.20 | 572.30 | 24.79 | — |
| MTP=3 | 376.22 | 38.16 | 560.78 | 48.93 | 80.70% |
| DFlash2, seven proposals | 383.41 | 30.49 | 568.03 | 40.73 | 76.15% |

MTP=3 resident: 128 passes, 255/316 proposals accepted, 2.99 timed tokens/pass; draft 11.94ms/pass, verify 48.90ms/pass. DFlash2 resident: 102 passes, 281/369 accepted, 3.75 timed tokens/pass; draft 14.71ms/pass, verify 76.61ms/pass. The wider DFlash2 pass produces more tokens but its verification cost is higher. These spans include execution and completion costs; they are not pure GPU kernel time.

Binary SHA256: `8e5521f0700efefd2ad0d832c8c1492c4d0f32758536f59fa181cf38141906e6`. Files: `forward-optimized-baseline-http, forward-optimized-mtp3-http, prefill-m1024-up-http-candidate`. Source starts at 67a0b4e plus the qualified cooperative-up patch. Final capability-only barrier admission fix was added after these timings; it leaves normal Loom GPU bodies unchanged.

The 29 baseline, 49 MTP, 103 DFlash2 TPS and 600 PP/s targets remain unmet. This coding request is not a long-context Pi replay. Do not apply these rates to every context, prompt or sampling setting.
