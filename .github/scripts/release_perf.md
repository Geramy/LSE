Measured on **gfx1201 (AMD R9700, driver 197)**, Qwen3.8-27B-Q6, 1024-token
prompt, 3 reps each. All tensors device-resident, compile-free, with
`LSE_REQUIRE_DEVICE_KERNELS=1`.

| Stage | This release | v0.4.1 | Delta |
|---|---|---|---|
| Prefill | **229 PP/s** (median; reps 229.25 / 227.20 / 225.84) | ~151 PP/s | **+51.7%** |
| Decode | **16.3 TPS** (median; reps 16.30 / 16.22 / 16.18) | 16.3 TPS | no regression |

**What changed.** The GDN projection GEMMs (`in_proj_qkv`, and the full
attention `qkv`/`v`/`o` projections) now select the staged-BF16 WMMA path at
M=1024 instead of falling to the scalar path. All four large shapes went
scalar→WMMA at ~3.0x each, taking the device total from 14053 ms to 9656 ms
for the 1024-token prefill. Decode sits at the measured aggregate-GEMV
bandwidth ceiling (~415 GB/s versus the ~549 GB/s that 25 TPS would need),
so it is unchanged; the M=1024 decode selection fix is included as well.

**Quality.** PPL-neutral: these are pure kernel-selection changes (records
table only). Greedy text is identical to the f32 reference on the standard
prompts, and the 0.005 logit rel-L2 gate holds where it applies.
