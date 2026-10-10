# Earlier release and development measurements

These records describe their original source, requests and sampling settings.
Use the current README and final mode report for the latest controlled comparison.

## v0.6.0: context-aware draft policy, long-context attention on one WMMA flash split

Requires a mac_linuxgpu HSA runtime with `hsa_amd_queue_create` (build 267 or later) on macOS. The Linux archive bundles its HSA runtime.

1. **Context-aware draft policy:** the adaptive dMTP and DFlash2 width keeps verify and draft costs per doubling of the context, so a long request no longer chooses wide trees priced at a short context. A cold first request (kernel compiles or variant trials in its passes) no longer holds the width down for the rest of the session.
2. **Chain passes on the flash split:** dMTP and DFlash2 verify chains of 2 to 8 rows over FP16 or BF16 K/V take the WMMA flash split that tree passes take, so each window of keys and values is read once per key head. Chain-pass logits can differ from 0.5.9 in their last bits; perplexity is unchanged (7.1467). One-row decode keeps the scalar split, whose value loop no longer branches per row.
3. **Fragment-ready V cache:** FP16 and BF16 value pools are stored dimension-major per 16-token block, so the flash tile reads each value fragment in one 16-byte read and the split decode kernels read four keys of a dimension at once.
4. **Faster tree tile:** on RDNA4 the narrowed probabilities are kept in scratch and each P fragment is one read; a tree pass reads its ancestor mask only for its own keys. Output is byte-identical.

## v0.5.9: cached device memory released before an allocation that would not fit

Requires a mac_linuxgpu HSA runtime with `hsa_amd_queue_create` (build 267 or later) on macOS. The Linux archive bundles its HSA runtime.

1. **Memory pressure handled before the allocation:** a fresh device allocation the device cannot cover first trims every cache (freed buffers kept for reuse, scheduler constants, idle sessions, HRX's idle pool slabs on macOS). Before, the caches were trimmed only after an allocation had failed, and on HRX a failed allocation also failed the stream, so the server stopped serving until restarted.
2. **Buffer reuse cache:** keeps the most recently freed sizes within an eighth of the device.
3. **Clear out-of-memory errors:** an allocation that still cannot fit is refused before the runtime sees it where the runtime allocates from the driver (Linux), and the error states the bytes requested, free and cached.
4. **K/V arenas** are reported under kv_cache.

## v0.5.8: HRX System main, DFlash2 draft trees, gfx1151 prefill and decode, startup fault fix

Requires a mac_linuxgpu HSA runtime with `hsa_amd_queue_create` (build 267 or later) on macOS. The Linux archive bundles its HSA runtime.

1. **Built on HRX System main** (ROCm/hrx-system 631c0b785). Two local HRX patches remain until upstream has them: gfx120x kernel arguments in VRAM (ROCm/hrx-system#1331) and a loomc register-allocation fix (#1443).
2. **DFlash2 draft trees, on by default:** one target pass verifies a tree of draft candidates when the measured costs make that faster than a chain; `--dflash2-tree=on|off` forces either.
3. **gfx1151 prefill and decode:** two query tiles per workgroup in prefill attention, vector loads for its probability fragments, the gate and up projections as one GEMM where registers allow.
4. **Kernel variants:** the prefill GEMM's grouped epilogue is offered as a measured variant, and variant trials cover kernels whose inputs were written in place.
5. **Startup GPU fault fix:** host DMA takes only whole-page ranges; other copies go through the staging buffer.
6. **Binding checker:** `LSE_CHECK_BINDINGS=1` checks every launch's buffers for liveness, bounds and the kernel's declared signature.
7. **Linux archive bundles the HSA runtime** (TheRock ROCr 1.21 with `librocprofiler-register` and its `rocm_sysdeps` libraries) and the GCC 16 C++ runtime, and links no `amd_comgr`; a system needs only the amdgpu kernel driver with KFD and access to `/dev/kfd`.

## v0.5.7: adaptive verify width, fast verify passes at every width, faster warm starts, perplexity scoring, gfx1151 prefill and decode

Requires mac_linuxgpu v0.1.151 (build 255) or later on macOS; kernel arguments in VRAM need v0.1.161 (build 265).

1. **Adaptive DFlash2 and MTP verify width, on by default.** Each sampled step verifies as many draft tokens as the draft's confidence and the measured cost of each width on the device make worthwhile. `--adaptive-dflash2=off` and `--adaptive-mtp=off` (`lse_config.adaptive_dflash2`, `adaptive_mtp`) keep the fixed widths; a request that sets `mtp_depth` keeps that depth, and greedy (temperature 0) requests always verify the fixed width, so greedy output is unchanged. On the R9700 at temperature 0.6, adaptive against fixed DFlash2 (median of 36 runs each): essay 62.0 -> **67.4** tok/s (+8.8%), code 152.8 -> 152.2 (-0.3%), 2K prompt +0.4%, 4K prompt +2.7%. Adaptive MTP against `--mtp-depth 3`: code 88.8 -> **113.4** tok/s (+27.7%), essay -5.3%, 2K -1.8%, 4K -2.4%.
2. **Every verify width on the four- or eight-row kernels.** A part names the row count each verify width runs as; on gfx1201 2-3 rows run on the four-row kernels and 5-7 rows on the eight-row ones, with the extra rows masked. DFlash2 verify pass on the R9700, ms per pass at rows 1..8: 31.4 35.2 48.8 33.0 74.9 51.5 89.2 37.6 -> 31.4 32.3 32.7 33.0 33.7 35.2 36.0 37.7.
3. **Warm starts prepare kernels without writing their source.** The kernel cache keeps a launch index beside the compiled objects, keyed by what is known before any source exists, so a warm start makes every prepared kernel resident without generating Loom text. Damaged entries are reported, removed and rebuilt.
4. **Perplexity and KL divergence.** `lse-server --perplexity FILE` scores a text through the same prefill kernels, attention and KV dtype as serving, with llama.cpp's chunk method by default (`--perplexity-method sliding` for sliding windows), and `--perplexity-kld-base-out` / `--perplexity-kld` record and compare a top-k KL-divergence base. Qwen3.8-27B Q4 (MLX), wikitext-2, c512, 580 chunks: **7.1467 ± 0.0464** on the R9700 on macOS and Linux and on gfx1151. See `docs/PERPLEXITY.md`.
5. **Kernel variants measured on the device.** A kernel can offer several layouts of one computation; the first time it runs, every variant runs on the step's operands, any whose output differs from variant 0 in a byte is refused, and the fastest is kept only when it beats variant 0 by more than 2%. Decisions are cached in `variants-<release>.txt` in the kernel cache; `LSE_AUTOTUNE=off` runs variant 0 everywhere.
6. **RDNA3.5 (gfx1151).** Whole-fragment WMMA operand loads in the prefill GEMM (Q4 GEMM 18 -> 31 TFLOPS; prefill at 1024 tokens 331 -> 505 tok/s), FlashPrefill V2 on by default (32768 tokens 533 -> 696 tok/s), four-row MTP verify passes in one row group with the SwiGLU pair fused, the odd verify widths on the four- and eight-row kernels, two columns per wave in the eight-row attention projections, the eight-row SwiGLU panel staged in scratch and per-shape K splits for the 8-bit draft's matrix panel. The target's eight-row down projection keeps one K slice, so its summation order and greedy output are unchanged. gfx1201's emitted kernels are byte-identical across these changes.
7. **Linux loomc memory accounting** subtracts frees (`malloc_usable_size`), so a compile's reported peak is no longer the sum of everything it allocated.

## v0.5.6: Loom by default on every platform, RDNA3.5 (gfx1151) matrix cores, faster verify passes

Requires mac_linuxgpu v0.1.151 (build 255) or later on macOS; kernel arguments in VRAM need v0.1.161 (build 265).

1. **Loom is the default kernel dialect on every platform.** `lse`, `lse-server` and libLSE (a NULL `lse_config.dialect`) generate Loom kernels unless `--dialect hip` is named. In 0.5.5 Linux builds defaulted to HIP, and `lse-server` without `--dialect` failed at startup on Linux while preparing kernels. The legacy HIP dialect stays selectable. `lse_estimate` with no dialect reports fragmented K/V storage, and FlashPrefill V2 and BLASST no longer need `--dialect loom` named. A user primitive that spells only HIP source runs on the host interpreter unless `--dialect hip` is named, as on macOS.
2. **RDNA3.5 (gfx1151) matrix cores on the Loom path.** Each architecture's tuning rules and matrix-instruction rows live in their own headers (`include/lse/dispatch/arch/{gfx1201,gfx1151,generic}.hpp`, `include/lse/math/matrix_{rdna3,rdna4,cdna3}.hpp`); gfx1201's rows moved there unchanged. The Loom dialect spells the RDNA3/3.5 WMMA rows, so gfx1151 runs the 4-bit prefill GEMM, the flash prefill attention and the 4- and 8-row int8 decode panels on its own WMMA instructions, splits decode attention across the part, and runs MTP and DFlash2 verify passes on the shared int8 panel. The 4-bit prefill GEMM's wide tile is chosen per part (gfx1151: 256x128 from 64x32 wave tiles). On an MS-S1 MAX (Radeon 8060S, ROCm 7.13, Ubuntu 24.04) with the pinned Qwen3.8-27B Q4 configuration: plain decode **13.3** tok/s (640 tokens), MTP=3 **15.1** tok/s (acceptance 65%), DFlash2 **18.4** tok/s (acceptance 67%), prefill 250 to 320 tok/s; 0.5.5 failed at startup on gfx1151. The Q6 staged-bf16 kernels, the 64-row prefill panel tiles and FP8 WMMA are RDNA4-only; gfx1151 uses the generic contraction there. The gfx1151 work leaves gfx1201's kernel sources and greedy output unchanged.
3. **Faster verify passes.** An 8-row verify pass's k and v projections and a 4-row pass's k, v and gated-delta-net a and b projections read the shared activation panel; each 4-row down-projection wave computes two adjacent columns. 640-token decode on mac_linuxgpu build 265: DFlash2 55.6 -> **56.4** tok/s (+1.4%), MTP=3 55.1 -> **55.7** (+1.1%) from the shared panel, and MTP=3 55.8 -> **56.7** (+1.6%) from the two-column down projection; acceptance and greedy output unchanged.
4. **Offline gfx1151 capture:** `LSE_CAPTURE_TARGET=gfx1151-loom` loads a whole model and emits, and with `LSE_TEST_REAL_COMPILE` compiles, every request shape's kernels for gfx1151 with no GPU.

## v0.5.5: faster decode and short prompts, HRX kernel arguments in VRAM

Requires mac_linuxgpu v0.1.151 (build 255) or later; kernel arguments in VRAM need v0.1.161 (build 265). Measured on builds 264 and 265.

1. **Decode:** the single-row 4-bit decode contraction issues all of a column's weight, scale and bias reads before staging the activations; partial RoPE rotates in one launch; one-token passes reshape heads instead of transposing them; a decode row's GDN q/k preparation is one launch. Plain decode runs 352 fewer launches per token. 640-token decode on build 264: plain **30.9** tok/s (before: 28.6), DFlash2 **51.8** (48.8), MTP=3 **53.1** (46.4).
2. **Short prefill:** 4-bit GEMMs of 65 to 767 rows are tiled from 48x32 wave tiles, so one row tile holds a whole pass of up to 192 rows. Warm TTFT on build 264: 137 tokens **0.138 s** (before: 0.182 s), 271 tokens 0.236 s (0.276 s), 532 tokens 0.393 s (0.425 s), 646 tokens 0.468 s (0.490 s).
3. **Speculative verify and draft:** the 8-row down projection widens each weight word with one shift and one mask per operand register; a vocabulary-projection wave owns four columns at 6 to 8 rows, so a verify pass or DFlash2 draft reads each activation run once for all four; a 4-row MTP=3 verify pass runs its gate and up projections as one fused SwiGLU launch; a DFlash2 draft pass's 8-row 8-bit projections read a shared activation panel made once per input. DFlash2 640-token decode on build 264: **53.3** tok/s (before: 52.4) from the down projection; on build 265 the vocabulary projection takes DFlash2 from 52.4 to **53.6** tok/s, the fused 4-row SwiGLU takes MTP=3 from 53.6 to **54.4**, and the shared 8-bit panel cuts a draft step from 10.4 to 8.0 ms, taking DFlash2 from 53.1 to **55.1** tok/s.
4. **HRX kernel arguments in VRAM:** `gfx120x-hdp-kernarg-publication` (the change in ROCm/hrx-system#1331) on the Linux and macOS builds, and on macOS `hsa-bar-write-bracket`, which brackets each submission's BAR stores under the driver's gate. 640-token decode on build 265: plain **31.8** tok/s (before: 30.7), DFlash2 **54.2** (53.0), MTP=3 **54.0–55.4** (53.6). Linux with ROCm 7.13: DFlash2 +4.9 to +7.5%, plain +2.2 to +3.7%.
5. **Loom load scheduling:** `loom-vmem-load-latency` sets loomc's AMDGPU global-load latency to 320 cycles instead of 16 (macOS and Linux builds), so the 4-bit GEMM issues its loads ahead of the math that waits on them. On build 265: prefill GEMM M 656 x 5120 x 17408 1.728 -> **1.378** ms, 0.350 ms saved (+25%), warm TTFT at 648 tokens 475 -> 453 ms, 640-token decode plain 31.63 -> 31.73 tok/s, DFlash2 54.16 -> 54.30.
6. **Kernels prepared at load:** the server makes every request shape's kernels resident before it reports ready, loaded together as one code object per shape set (about 260 executables instead of about 4000). On build 265, first request at each size after a cold start: 137 tokens **250 ms** (before: 344 ms), 271 tokens 273 ms (407 ms), 532 tokens 402 ms (514 ms), 1060 tokens 727 ms (1115 ms); first 640-token decode after load 52.4 tok/s (49.4). Load takes 10.2 s instead of 3.8 s, host footprint is 654 MiB instead of 390 MiB, and the KV pool peaks at 2 GiB while preparing.
7. **RDNA4 matrix-core tools:** standalone microbenchmarks and an FP8 Q4 prefill GEMM prototype in `scripts/rdna4-matrix`.

## v0.5.4: twice the prefill, faster speculative decode and HRX patches in-tree

Requires mac_linuxgpu v0.1.151 (build 255) or later; measured on v0.1.156 (build 260).

1. **Prefill:** a tiled f16 matrix-core GEMM for 4- and 8-bit projections of 16 rows or more (64x64 wave tiles on wide passes, K slices when the grid underfills), a GDN prefill scan with a thread pair per state row, flash prefill attention staged through workgroup memory, a wave per row for `l2_normalize` and one padded pass for a prompt's remainder. Warm 4K prompt (4230 tokens) **1,455** tok/s, up from 678 in 0.5.3 on the same driver; warm 1060 tokens **1,307** tok/s.
2. **Time to first token:** issue-ordered clears of fresh state and KV, reuse of freed device buffers, kernels loaded together at a step's first cache miss and the last DFlash2 context pass after the first token. Warm 137-token TTFT **0.18 s** (0.5.3: 0.68 s); the first request after start 0.37 s.
3. **Decode:** occupancy-tiled short-query attention, four-chunk lm_head and query loads, the decode down projection over every compute unit, preloaded GDN scans that read shared key heads in place, fused verify q/k preparation, batched verify launches, a device top-k for sampled verification and MTP drafts chained on the device. 640-token decode: DFlash2 **49.9** tok/s, MTP=3 **50.8** tok/s, plain **28.5** tok/s (0.5.3: 40.9, 44.9, 27.5).
4. **HRX patches in LSE:** the macOS, iOS and Linux builds apply `patches/hrx` instead of fetching the archived mac-amdgpu repository; loomc grows its arrays only when full, so the 64x64-tile kernels compile.

## v0.5.3: faster load and decode, refusable CPU fallback and the current mac_linuxgpu runtime

Requires mac_linuxgpu v0.1.151 (build 255) or later.

1. **`--no-cpu-fallback`** on `lse` and `lse-server`: startup fails when no device backend comes up or `--pool` names a CPU device, and a request fails when an operation has no device kernel. Each error names the cause. Without it, CPU fallback is logged, counted and reported in `lse_warnings`.
2. **No CPU-only run with the option:** without `--pool`, a GPU backend that fails to start no longer leaves the model on the CPU interpreter. `LSE_REQUIRE_DEVICE_KERNELS=1` has the same effect as the option.
3. **iOS build** against mac_linuxgpu build 247 and later (owner and selector calls; tested on 246 and 247). The HSA runtime is built under `build/ios/hsa`, so the mac_linuxgpu checkout is only read. `SKIP_HRX=1` links the newly built runtime.
4. **Faster model load:** the pinned Qwen3.8-27B Q4 + DFlash2 configuration loads in **3.9 s** on driver build 255 (16 GT/s link), down from 16–28 s: streamed weight uploads through a mapped staging ring with `pread`, RoPE tables on all cores, the tokenizer loaded concurrently and mapped load buffers.
5. **Faster decode:** about +4% on a 640-token decode (row-at-a-time verify logits, small host writes queued behind the stream). Measured on driver 255: 2K prompt **592** tok/s prefill, **28.5** tok/s decode; 4K prompt **628** tok/s prefill, **38.4** tok/s decode.

## v0.5.2: model-defined limits, thinking levels and sampling defaults

1. **No default output limit:** a reply runs to an end-of-turn token, a stop sequence, the request's `max_tokens`, a `generation_config` limit or a full context. The 256 and 4096 defaults are gone.
2. **Full context:** `finish_reason: "length"`, `stop_reason: "context_full"` and `lse_context` on every response. A prompt with no room left gets HTTP 400 with `code: "context_full"`.
3. **Thinking levels** are read from the model's chat template and listed in `/v1/models`; an undefined level is refused, not aliased.
4. **Sampling defaults** come from `generation_config.json`, are reported with their sources, and use neutral defaults otherwise. `top_k`, `min_p` and `presence_penalty` are accepted on every request surface.

## v0.5.1: iPad prefill fixes

1. **loomc:** four passes doubled arrays on every append; a prefill kernel asked for 16 GiB and iPadOS refused it. Fixed; compiles peak around 150 MiB.
2. **Host memory:** weight host mirrors are released after upload, 3.0 GiB down to about 150 MiB.
3. **Fallbacks:** the device-path fallback is removed; CPU fallback stays, reported and optional (`--no-cpu-fallback`). On an iPad Pro (M4) with an R9700, a full agent turn decoded at **68.9 tok/s** at 94% acceptance, 20.5 GiB peak.

## v0.5.0: libLSE, iPadOS, DFlash2 BF16 loading and memory estimates

1. **libLSE:** the engine as an in-process library behind a plain C API ([`include/lse/lse.h`](../include/lse/lse.h)): `lse_open`, `lse_request` with the server's JSON, `lse_cancel`, `lse_status`, a log callback and an optional HTTP adapter. `lse-server` is a thin command line over it, and both transports answer identically.
2. **iOS and iPadOS:** `scripts/ios/build-ios.sh` builds `LSE.xcframework`. On an iPad Pro (M4) with an R9700 over Thunderbolt, through mac_linuxgpu's embedded driver, Qwen3.8-27B Q4 with the Q8 DFlash2 draft decoded at **40.5 tok/s** warm at 73% acceptance (the M5 Max Mac measured 43.3 tok/s at 74% with v0.4.24) and **57–70 tok/s** at 85–93%, with about 34 s model load and **0** CPU fallbacks.
3. **DFlash2 BF16 checkpoints:** `--dflash2-model` converts a BF16 draft to Q8/group64 once, bit-identical to `scripts/convert_dflash2_q8.py`, and caches the result.
4. **Model info and memory estimates:** `lse_model_info`/`--model-info` and `lse_estimate`/`--estimate`, plus `/v1/lse/model_info` and `/v1/lse/estimate` for a running server. Weights, KV, recurrent state, RoPE and the DFlash2 ring follow the allocators' own rules; a host-backend load of Qwen3.8-27B Q4 with its Q8 drafts matches within 0.001%. `/v1/models` reports `context_length`, `kv_len` and `kv_cache_dtype`.

## v0.4.24: HumanEval+ through 32K and driver HSA runtime

1. **HumanEval+ through 32K:** FlashPrefill V2 (alpha 0.1) prefill is **51–55% faster at 32K** across Baseline, MTP3 and DFlash2 on Qwen3.8-27B Q4, R9700/HRX/LOOM, over **1,368 completed generations**. Correctness matches dense prefill at Standard and 32K, with one additional failure per mode at 16K. [Results and data](benchmarks/flashprefill-humaneval-32k.json).
2. **Decode refinements:** single-token RMS normalization on gfx1201 keeps each row in registers across the reduction with **bit-identical output**; the single-token Q4 FFN down projection (5120×17408) now uses activation panels; and the submission tuner accepts repeatable wins of **2%** (previously 5%) when the sample ranges separate.
3. **macOS uses the driver's HSA runtime:** the macOS package **no longer bundles libhsa-runtime64**, which could shadow the driver's copy and fail to find the GPU. LSE now loads the HSA runtime installed by the GPU driver ([mac_linuxgpu](https://github.com/lemonade-sdk/mac_linuxgpu)) from `/usr/local/lib` or `/Library/MacAMDGPU/runtime`. Install or update the driver before upgrading.

## v0.4.23: FlashPrefill V2

1. **Faster long-context prefill:** FlashPrefill V2 reached **632.1 pp/s at 16K** and **604.9 pp/s at 32K** on Qwen3.8-27B Q4, R9700/HRX/LOOM. Default on for supported configurations; use `--FlashPrefillV2=off` for dense prefill.
2. **MTP and DFlash2 support:** at 16K, prompt throughput reached **582.6 pp/s with MTP3** and **615.2 pp/s with DFlash2**. Both on/off pairs matched the 64-token output and aggregate acceptance. Drafting and verification stay dense. Includes approximately **5× faster block selection** and Q4 SwiGLU bias-tail unrolling.
3. **Previously observed DFlash2 peaks:** **67.6 tok/s decode** and **96% acceptance**. These are separate workload peaks; the new sparse tests establish prefill gains, not a decode speedup. Perplexity is not yet measured.

## v0.4.22: faster Q4 prefill

1. Faster Q4 prefill with optimized nibble expansion and paired activation staging: **597.9 pp/s peak** on a warm full prompt with BF16 KV and no prompt cache reuse.
2. Fused eligible single-token Q4 SwiGLU gate/up projections and activation; updated dispatch regression coverage.
3. Fixed RDNA4 prefetch address/span lowering in bundled Loom. Previously observed live DFlash2 peaks: **67.6 tok/s decode** and **96% acceptance** on Qwen3.8-27B Q4, R9700/macOS. These are separate workload peaks.

[Prefill measurements](benchmarks/q4-prefill-2026-09-30.md).

## v0.4.21: real-world interactive performance

Real-world interactive use on the local macOS R9700 (`gfx1201`), running
Qwen3.8-27B Q4 with the Q8 DFlash2 block-8 drafter through HRX/Loom, measured:

- **Decode: 35.8 tokens/s overall**, with a **36.1 tokens/s median** per request
  and an observed range of **20.5–67.6 tokens/s**.
- **Prefill: 243.6 tokens/s overall**, ranging from **32.5–390.2 tokens/s**
  across short follow-ups and larger input chunks.
- **DFlash2 acceptance: 86% median** per request, ranging from **59–96%**.

These summarize user-supplied server logs. Overall rates are total logged tokens
divided by total phase time (approximately, because printed times are rounded).
Prompt counts represent newly processed input, not the full retained context;
short follow-ups include request overhead. Decode timing excludes the first token.
The results describe this live workload, not a fixed-context benchmark or a
before/after speedup measurement.

[Release details and downloads](https://github.com/Geramy/LSE/releases/tag/v0.4.21).

## v0.4.19: long-context KV growth and split attention

Consumed prefill chunk graphs and the scheduler's previous program are released
before a paged KV pool moves. Pools above 32K tokens grow in 4K-token steps to
limit allocation pressure. The local R9700 completed a 65,354-token prompt
with BF16 KV and Q8 DFlash2, followed by a 32-token decode request. The test
peaked at 29.72 GB of reserved device memory. A shorter deterministic response
matched v0.4.18 exactly. See the
[long-context report](benchmarks/kv-growth-2026-09-29.md).

One split partial and one merge kernel family now serve both single-token
decode and short verification queries. A matched cold 65,126-token Q4+DFlash2
request improved from 8.51 to 14.57 decode tokens/s; prefill measured 147.7
versus 147.0 prompt tokens/s. Both runs produced the same 64-token response.
See the [attention comparison](benchmarks/long-context-attention-2026-09-29.md).

## v0.4.18: smaller weight slabs

The weight allocator now uses 512 MiB slabs on macOS HRX, avoiding large
unused tails in the Q4 target and Q8 DFlash2 draft allocations. Matched driver
counters fell by about 1.71 GB after load and after a request. Repeated warm
requests measured 545.5 versus 548.3 prompt tokens/s and 49.9 decode
tokens/s for both builds. Output and acceptance matched. This does not resolve
the remaining long-context workspace and allocation high-water. See
[the slab report](benchmarks/weight-slab-memory-2026-09-29.md).

## v0.4.17: completed prefill workspace retirement

This release fixes overlapping target activation workspaces when ragged prefill
changes chunk width, and retires completed verifier graphs before a new request.
DFlash2 retains one wide context program during prefill and releases completed
wide programs afterward. Narrow decode replay, live state, KV and compiled
kernels remain available.

The matched 6143-token request reduces reserved VRAM from 29.77 to 28.26 GB.
Output and acceptance results match exactly. Prefill and decode rates differ
by less than 0.5% in that comparison. Eight cached follow-ups also preserve
output, acceptance and prefix reuse, with only 12.14 MB reserved-memory growth.
The reported allocation failure has not yet been replayed. See
[the memory report](benchmarks/prefill-workspace-memory-2026-09-29.md).

## Development: full-width DFlash2

The current development build verifies all seven proposals from the block-8
checkpoint. Conditional drafting uses the request temperature and preserves
target sampling through probability-ratio rejection and residual sampling.
Both prefill batch limits default to 1024. Use `--batch-size 2048 --ubatch-size 2048`
to select larger passes. The HTTP server accepts `--temperature 0.6`; an explicit
request temperature overrides it.

On exact 1024- and 2048-token coding requests at temperature 0.6, the second
requests measured **501.38 / 508.09 prompt tokens/s** and **35.04 / 34.53 decode
tokens/s**, respectively. Both generated 384 tokens, reused no prompt KV and had
zero host fallbacks. The process retained compiled code, but further compilation
remained inside these timings. Different temperatures produce different answers.
These are workload measurements, not general rates or achieved 103 TPS/600 PP/s
goals. Published v0.4.13 includes the full-width changes and the M8 optimization below.
See [full-width DFlash2 results](benchmarks/dflash-fullwidth-2026-09-29.md).

The next M8 Q4 down-projection optimization measured **40.21 TPS** on the
second 1024-token request, versus **35.42 TPS** for its matched control.
Prefill remained about **500 prompt tokens/s**. Both use seven proposals,
temperature 0.6 and BF16 KV. A 1024-target perplexity comparison was
4.8660 versus 4.8675. Completed prefill workspace is released before decode
to fix the reproduced second-request allocation failure at batch 4096.
See [M8 WMMA and memory results](benchmarks/wmma-m8-down-2026-09-29.md).

The next M1024 down-projection schedule measured **522.58 prompt tokens/s**,
versus **507.24** for the matched control. Decode was effectively unchanged:
**40.78 versus 40.55 TPS**. Both responses and acceptance statistics match exactly.
This prefill change is newer than v0.4.13. See
[M1024 prefill results](benchmarks/prefill-m1024-down-2026-09-29.md).

WMMA attention now skips entire windows that are outside the row or fully
causally masked. The component time falls 30.74% at 1024 live keys and 7.45%
at 5207 live keys. The matched resident HTTP request reaches **526.26 prompt
tokens/s**; decode remains about **40.7 TPS**. Responses and sampling statistics
match the control exactly. See
[attention window results](benchmarks/attention-causal-windows-2026-09-29.md).

Cooperative activation staging for M1024 gate/up now reaches **568.03 prompt
tokens/s**, versus **526.26** for its matched control. Decode remains **40.73 TPS**.
Both responses and acceptance statistics match exactly. The change preserves
the accepted M8 and M1024 down emitted kernels and launch plans. See
[cooperative prefill results](benchmarks/prefill-m1024-up-2026-09-29.md).

A separate comparison uses one binary and the same 1024-token requests,
temperature 0.6, BF16 KV and batch/ubatch 1024. Each mode starts with an empty
private disk cache. The resident request reuses compiled code but no prompt KV.

| Mode | Resident prompt tokens/s | Resident decode tokens/s | Acceptance |
| --- | ---: | ---: | ---: |
| Baseline | 572.30 | 24.79 | — |
| MTP=3 | 560.78 | 48.93 | 80.70% |
| DFlash2, seven proposals | 568.03 | 40.73 | 76.15% |

These source changes follow published v0.4.13. The performance targets remain
unmet. This coding request does not establish rates for long-context Pi chat.
See [mode comparison and timing limits](benchmarks/forward-modes-2026-09-29.md).

The same cooperative schedule now covers the exact M1024 QKV, GDN gate and
attention projection shapes. The matched resident request reaches **619.08
prompt tokens/s**, versus **568.03** before these entries. Decode remains
**40.68 TPS**. Responses and acceptance statistics match exactly. This exceeds
600 PP/s on this controlled 1024-token workload, not at every context length.
See [projection results and validation](benchmarks/prefill-projections-2026-09-29.md).

The M8 gate/up and QKV DOT4 schedules now consume two activation rows at a time.
The matched resident DFlash2 request improves from **40.68 to 42.41 decode
tokens/s**. Prefill remains above 600 at **616.78 prompt tokens/s**. Responses,
usage and sampling statistics match exactly. All previously accepted schedules
are retained. See [M8 scheduling results](benchmarks/m8-dot4-rowpairs-2026-09-29.md).


## v0.4.12: typed attention and automatic buffer views

This version adds typed WMMA attention, cooperative single-token attention, and
automatic contiguous buffer views. Eligible slices no longer launch copy kernels.
K/V storage defaults to BF16 for models that declare BF16, and FP16 otherwise.

Single-token split attention now shares K/V loads across six query heads.
The matched component check reduced BF16 attention time by 49% at 5.2K live
keys and 53% at 14K. Empty partitions publish neutral records and return early.
The submission tuner skips its end wait when the policy did not change.

The measurements below use the same Q4 target, BF16 KV, binary/runtime, and
requests. Each mode starts with an empty private kernel cache. First prefill
includes compilation. Sampling uses model defaults 1/20/0.95. The first prompt
has 5,207 tokens; the follow-up has 5,262 cached tokens and 23 new tokens.

| Mode | First prefill, tokens/s | First decode, tokens/s | Follow-up decode, tokens/s |
| --- | ---: | ---: | ---: |
| Baseline | 344.21 | 22.58 | 23.82 |
| MTP=3 | 313.32 | 24.74 | 40.25 |
| DFlash2 | 328.65 | 26.40 | 37.53 |

Baseline follow-up increased from 22.82 to 23.82 tokens/s in the matched check.
The speculative modes remain near their previous rates. Responses match their
preceding controls, with zero host groups or fallbacks. These are individual
workload measurements, not a general throughput guarantee. See the
[shared-KV decode report](benchmarks/shared-kv-decode-2026-09-28.md) for
component results, first/follow-up timing limits, and CPU sampling costs.

## v0.4.11: split decode attention and MTP prompt reuse

The monolithic single-token attention kernel is removed. Split attention now
covers long contexts without the old 4K capacity cutoff. On the recorded Pi
requests, plain decode measured **19.71 / 19.72 tokens/s**, compared with
12.91 / 12.70 before the change. The sampled responses changed.

MTP now retains verified state between requests. Follow-up prefill fell from
**14.25 s to 0.71 s**, with 5,310 cached tokens and 23 new tokens. Recorded decode
measured **32.68 / 31.90 tokens/s with MTP=3** and **27.97 / 35.26 tokens/s with
DFlash2**. The two modes produced identical responses in these candidate runs.
First prompt rates were **293.95 and 291.75 tokens/s**, respectively.
These are individual Pi workload measurements, not a 46–48 tokens/s guarantee.
See the [methods and limits](benchmarks/pi-execution-profile-2026-09-28.md#remove-monolithic-single-token-attention-and-retain-mtp-sessions).

## v0.4.10: shorter compilation stalls and optional KV formats

Identical Loom kernel bodies now share compiled code. Attention page traversal
stays in generated control flow instead of expanding each page into source.
In the same two-turn Pi conversation with an empty starting cache, total JIT
compilation fell from **92.91 s to 3.35 s**. First prefill fell from **91.53 s to
16.88 s**; follow-up prefill fell from **16.14 s to 0.76 s**. Both responses and
proposal acceptance counts matched. The final paired-load build measured **28.59 / 34.03 decode tokens/s** and
**310.72 prompt tokens/s** on those turns. Scratch allocation is now preserved
in optimizer facts and reported by dispatch profiling.
See the [execution profile](benchmarks/pi-execution-profile-2026-09-28.md).


## v0.4.9: chat sampling and prompt reuse

The CLI and server load each model's supported settings from `generation_config.json`.
If fields are missing, LSE uses embedded model settings and the fallback table under
`src/models`. Explicit CLI flags and HTTP request parameters override these defaults.
See [sampling defaults](SAMPLING.md) for precedence and supported fields.

Pi can return streamed reasoning without invalidating the previous prompt cache.
The HTTP response reports reused tokens in `usage.prompt_tokens_details.cached_tokens`.
In a two-turn Qwen3.8 Q4 + Q8 DFlash2 chat, the second prefill fell from 14.11 s
for 5296 tokens to 0.93 s for 23 new tokens after the serialization fix.
With the model's temperature 1, top-k 20 and top-p 0.95 settings, decode measured
23.98 and 28.82 tokens/s on the two turns. These measurements describe those
requests; output length and proposal acceptance affect the result.
See the [Pi chat report](benchmarks/pi-chat-2026-09-28.md).

## v0.4.8: lower VRAM use

KV growth previously kept obsolete recurrent-state graphs and their buffers alive.
The fix releases those graphs while keeping the current state and compiled kernels.
On one 14K Q4 + Q8 DFlash2 request, live allocations fell from **30.72 to 22.33 GiB**.
Sampled peak driver reservations fell from **30.93 to 24.86 GiB**.

The performance check measured 323.41 prompt tokens/s and 42.41 decode tokens/s
on the second identical request, with temperature 0 and 100% proposal acceptance.
**These are warmed synthetic results, not expected Pi or coding-session rates.**
The first request measured 15.51 decode tokens/s even without new kernel compilation;
request setup and reuse remain under investigation.
See the [memory report](benchmarks/kv-growth-memory-2026-09-28.md).

## v0.4.19

This version releases consumed prefill graphs before paged KV growth and uses
smaller growth steps above 32K tokens. The local R9700 with a Q4 target, Q8
DFlash2 draft and BF16 KV completed a 65,354-token prompt and a subsequent
32-token decode request. Peak reserved GPU memory in the extension was 29.72 GB.
One split-attention kernel family now covers single-token decode and short
verification queries. In a matched cold 65,126-token HTTP request, DFlash2
decode increased from 8.51 to 14.57 tokens/s; prefill stayed near 147 tokens/s.
The output text matched. See the
[attention comparison](docs/benchmarks/long-context-attention-2026-09-29.md).
The configured KV limit is not a measured usable capacity; see the
[long-context test](docs/benchmarks/kv-growth-2026-09-29.md).
The v0.4.18 weight-allocation reduction and v0.4.17 workspace-retirement fix
remain active; see [weight slab measurements](docs/benchmarks/weight-slab-memory-2026-09-29.md) and
[prefill memory ownership](#prefill-memory-ownership).
