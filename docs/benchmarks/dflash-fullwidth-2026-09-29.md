# Full-width DFlash2 execution and sampling

Measured on macOS with R9700/gfx1201, the local Qwen3.8-27B group-affine Q4 target and Q8 DFlash2 block-8 checkpoint. Target/MTP KV is BF16 and attention accumulation is FP32. The draft uses its FP32 ring cache.

## Changes

- Verify all seven trained proposal positions. Terminal requests return only the required prefix while retaining one fixed seven-position head shape.
- Upload tokens, anchor, offset and ring metadata in one retained input buffer.
- Use one native two-tap convolution instead of separate slice and arithmetic operations.
- Use paired activation-panel loads for five measured eight-row Q4 projection shapes.
- Skip empty short-attention partitions at the qualified 8192/16384 table capacities.
- Sample conditional selector distributions at the request temperature. Verify proposals with probability-ratio acceptance and positive residual sampling. Greedy requests retain deterministic selection.
- Share target distribution construction with ordinary sampling. Preserve existing penalties, filter order and baseline seeded sampling.
- Report tested and accepted proposals by position, candidate coverage and probability overlap.
- Default both prefill batch limits to 1024. Add `--batch-size` and `--ubatch-size` for powers of two from 128 to 4096; microbatch must not exceed batch.
- Add `--temperature` to the HTTP server. Explicit request temperature overrides this launch default.

These development changes are separate from published v0.4.12, which retains three proposals.

## Request measurements

Each process starts with an empty private disk cache. Each prompt has exactly the listed token count. Each request generates 384 tokens. The second request uses a different leading user token, so it reuses no prompt KV. It retains compiled code, but can still compile new shapes. All timings include these costs. Top-k 20, top-p 0.95, seed1234. Each run has zero host groups and host fallbacks.

| Prompt tokens | Target temperature | Proposal policy | First PP/s | First TPS | Second PP/s | Second TPS | Second acceptance |
|---:|---:|---|---:|---:|---:|---:|---:|
|1024|1.0|Probabilistic|356.07|26.15|504.98|27.69|66.7%|
|1024|0.6|Probabilistic|355.56|29.06|501.38|35.04|76.4%|
|1024|0|Greedy|360.14|30.86|504.52|31.65|73.6%|
|2048|0.6|Probabilistic, final binary|409.47|32.79|508.09|34.53|76.2%|

The final2048 requests omit their temperature field and use launch `--temperature 0.6`. Both batch limits are1024. This checks server-default precedence. The final sampled graph does not execute the unused deterministic selector-walk kernel.

The preceding deterministic0.6 control measured33.11 TPS on the second1024 request, compared with 35.04 TPS for probabilistic drafting. That control used the prior2048-row prefill candidate; the new build uses 1024. The continuations and acceptance paths differ. This single comparison does not isolate a universal implementation speedup.

At temperature 1, probabilistic drafting did not beat the earlier deterministic1024 result29.39 TPS. The user selected launch temperature 0.6. Model files retain their own generation configuration; launch and explicit request overrides remain separate.

## Acceptance and verifier cost

The 1024 temperature 0.6 second request uses 102 verifier passes and 383 timed output tokens:3.75 outputs/pass. Verification costs 91.18 ms/pass and drafting15.12 ms/pass. The 2048 request produces 3.79 outputs/pass, with 93.41 ms verification and 15.54 ms drafting.

Greedy acceptance is 73.6%, not95%. This prompt still has draft/target argmax disagreement; removing target sampling alone does not make the seven-proposal block fully accepted. Aggregate acceptance is not a per-position survival curve. The response now provides the actual position counts.

For sampled1024 temperature 1, the measured candidate support covers about91% of target mass, but target/draft overlap is about69% over tested positions. Correct rejection sampling accepts according to overlap. Randomizing drafts while retaining exact-match verification would instead reduce acceptance to the dot product of distributions.

The small-M WMMA verifier is still undergoing native qualification. No WMMA throughput gain is claimed here. The 103 TPS DFlash2 and 600 PP/s goals remain unmet.

## Matched execution-only Pi result

Before probabilistic drafting, the paired Q4 loads and empty-partition shortcut improved the seven-proposal Pi follow-up from 21.03 to 24.64 TPS. Verification fell106.31 to 87.96 ms/pass. Both requests have identical text, usage, acceptance and step counts. The follow-up has 5262 cached and 23 new prompt tokens; its tiny-prefix prompt rate is not full-prefill throughput.

This full-width run still trails the preceding three-proposal Pi result37.53 TPS. That width comparison is not an identical-output experiment.

## Native component evidence

Five Q4 M8 shapes have complete bitwise output matches, independent quantization-codec checks, unchanged allocation guards, ABBA device timing and zero private scratch. Paired loads reduce GPU consumer time14.4–32.1% depending on shape. The arithmetic is unchanged.

Four skip-empty attention comparisons preserve complete outputs and intermediate records bit for bit. Partial-plus-merge GPU time falls55.5% at 1024 live keys in an8192 table,44.2% at 2048/8192,15.9% at 5207/8192, and 4.7% at 14000/16384. These capacities are part of the measurement, not a substitute for smaller allocated tables.

The typed native fixture passes 10 cases across FP32, FP16, BF16, FP8 and BF8 KV;60 device dispatches and zero host/fallback dispatches. It checks encoded storage, causal tails, shuffled pages, empty replay, input immutability and guards.

## Correctness

- DFlash2 CPU:32/32, including conditional predecessor choice, probability normalization, repeatable sampling, rejected suffix cancellation and retained next-turn state.
- Shared sampler CPU:9/9, including exact enumeration recovering the target distribution, identical/disjoint support, filtering, invalid inputs, independent RNG streams and ordinary32-token seeded golden sequence.
- Server CLI:89 cases; CLI:57 cases. Invalid batch sizes, microbatch ordering and nonfinite temperatures fail before backend startup.
- Q4 panel and short-attention focused suites:9/9 and 8/8.

No L2 or new perplexity sweep was used. A WMMA candidate that changes activation quantization will require one1024-token perplexity check before promotion.

## Reproduce

```sh
/path/to/lse-server --model /path/to/qwen38-27b-q4 \
  --dflash2=on --dflash2-model /path/to/qwen38-27b-dflash2-q8 \
  --pool hrx:0 --dialect loom --kv-cache-dtype bf16 --kv-len 262100 \
  --batch-size 1024 --ubatch-size 1024 --temperature 0.6 \
  --served-name qwen-q4 --host 127.0.0.1 --port 8080
```

