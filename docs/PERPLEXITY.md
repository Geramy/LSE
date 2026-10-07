# Perplexity and KL divergence

LSE can score a fixed text and report its perplexity, using the same method
as llama.cpp's `llama-perplexity`. It can also compare two runs by KL
divergence, as `llama-perplexity --kl-divergence` does. The text runs through
the same prompt prefill a request gets, so the score reflects exactly what is
served:

- the same kernels and pass plan (`--batch-size`, `--ubatch-size`),
- the same prefill attention (FlashPrefill V2 when the engine turns it on,
  dense with `--FlashPrefillV2=off`),
- the same KV cache dtype (`--kv-cache-dtype`), and
- the same checks: with `LSE_REQUIRE_DEVICE_KERNELS=1` or `--no-cpu-fallback`,
  any operation without a device kernel fails the run instead of running on
  the CPU.

Use it to check output quality whenever a speed change lands. Run the same
text before and after. Compare the two perplexities or, more sensitively,
compare the second run against a KL-divergence base recorded by the first.

Only the target model is scored. A speculative decoder (MTP or DFlash2) does
not change the target's logits, so it is not used even when it is loaded.
Check speculative decoding separately: greedy output must match the
non-speculative output, and acceptance is reported in the timings.

## Quick start

```bash
export LSE_REQUIRE_DEVICE_KERNELS=1
M="--model <model-dir> --pool hrx:0 --dialect loom --batch-size 1024 --ubatch-size 1024 \
   --kv-cache-dtype bf16 --kv-len 262100 --no-mtp"

# Perplexity, llama.cpp style: wikitext-2 test, 512-token chunks.
lse-server $M --perplexity wiki.test.raw --perplexity-output ppl.json

# A quick check on the first 40 chunks only.
lse-server $M --perplexity wiki.test.raw --perplexity-chunks 40

# KL divergence: record a base once, then compare another build or setting to it.
lse-server $M --FlashPrefillV2=off --perplexity wiki.test.raw --perplexity-kld-base-out dense.kld
lse-server $M --perplexity wiki.test.raw --perplexity-kld dense.kld --perplexity-output fp2.json
```

`lse-server` loads the model, scores the file, writes the JSON and exits. It
does not open an HTTP port. `wiki.test.raw` is the file llama.cpp uses: run
`scripts/get-wikitext-2.sh` from llama.cpp, or download
`https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip`.

| Option | Meaning |
|---|---|
| `--perplexity FILE` | UTF-8 text, tokenized the way a prompt is, with no chat template |
| `--perplexity-tokens FILE` | Score a JSON array of token ids instead of text |
| `--perplexity-method M` | `chunks` (default, llama.cpp's method) or `sliding` |
| `--perplexity-ctx N` | Tokens per chunk or window (default 512, llama.cpp's `-c`) |
| `--perplexity-stride N` | `sliding` only: tokens between window starts (default: the ctx) |
| `--perplexity-chunks N` | Score only the first N chunks or windows (llama.cpp's `--chunks`) |
| `--perplexity-kld-base-out FILE` | Record a KL-divergence base (llama.cpp's `--kl-divergence-base` when it writes) |
| `--perplexity-kld-top-k K` | Token ids recorded per scored token in that base (1 to 32, default 32) |
| `--perplexity-kld FILE` | Compare against a recorded base (llama.cpp's `--kl-divergence`) |
| `--perplexity-output PATH` | Where to write the JSON (default: standard output) |
| `--perplexity-token-ids` | Include the token ids in the JSON |

Every other `lse-server` option means what it means when serving.
`--FlashPrefillV2=off` selects dense prefill. Each chunk must fit `--kv-len`.

## Methods

**chunks** (default) is llama.cpp's method. The whole text is tokenized once.
It is split into `floor(tokens / ctx)` non-overlapping chunks of `ctx` tokens,
and the remainder is dropped. In each chunk, the first `ctx/2` tokens are
context only. The model's predictions from position `ctx/2` onward are scored
against the tokens that follow, so each chunk scores `ctx - ctx/2 - 1` tokens
(255 at ctx 512). The text must hold at least two chunks.

llama.cpp puts a BOS token at the start of each chunk only when the model's
vocabulary asks for one (`add_bos_token`). Qwen3.5 and Qwen3.6 GGUF vocabularies
do not, so llama.cpp inserts no BOS for this family, and neither does LSE. The
default reproduces llama.cpp's token windows and scored positions exactly.
Use the same tokenizer and the same `wiki.test.raw`, and LSE's numbers are
directly comparable to llama.cpp's for the same weights.

**sliding** cuts windows of `ctx` tokens that start every `stride` tokens.
Each window scores only the tokens that no earlier window scored.

- With `stride == ctx`, windows don't overlap. Each window scores every token
  except its first, which has no context in that window.
- With `stride < ctx`, every token after the first token of the text is scored
  exactly once. After the first window, each scored token has at least
  `ctx - stride` tokens of context.

Each chunk or window runs in a fresh session, as a request without a session
id does.

## Output

```json
{
  "object": "lse.perplexity",
  "tokens": 297193, "token_ids_sha256": "e0847dfd…",
  "method": "chunks", "ctx": 512, "stride": 512,
  "windows_scored": 580, "scored_tokens": 147900,
  "mean_nll": 0.0, "perplexity": 0.0, "perplexity_uncertainty": 0.0,
  "elapsed_seconds": 0.0, "prefill_tokens": 296960, "prefill_tokens_per_second": 0.0,
  "config": {"device_arch": "gfx1201", "prefill_attention": "flashprefill-v2",
             "prefill_attention_scale": 0.1, "kv_cache_dtype": "bf16", "kv_capacity": 262100,
             "batch_size": 1024, "ubatch_size": 1024, "head_rows": 128,
             "cpu_fallback_allowed": false},
  "kld": {"base": "dense.kld", "top_k": 32, "tokens": 147900,
          "base_perplexity": 0.0, "perplexity": 0.0, "delta_perplexity": 0.0,
          "perplexity_ratio": 0.0, "mean_log_ratio": 0.0, "mean_log_ratio_uncertainty": 0.0,
          "mean_kld": 0.0, "mean_kld_uncertainty": 0.0, "kld_p99": 0.0, "kld_max": 0.0,
          "same_top": 0.0, "same_top_uncertainty": 0.0,
          "mean_delta_p": 0.0, "mean_delta_p_uncertainty": 0.0, "rms_delta_p": 0.0,
          "mean_base_top_mass": 0.0},
  "windows": [{"index": 0, "begin": 0, "end": 512, "first_target": 257, "scored": 255,
               "nll_sum": 0.0, "mean_nll": 0.0, "perplexity": 0.0, "seconds": 0.0}]
}
```

- `mean_nll` is the mean negative log-likelihood in nats per scored token, and
  `perplexity` is `exp(mean_nll)`.
- `perplexity_uncertainty` is llama.cpp's "±": the standard error of the
  mean NLL, propagated to perplexity.
- `token_ids_sha256` is the SHA-256 of the token ids as little-endian 32-bit
  integers. If another implementation computes the same hash, it scored the
  same tokens.
- `prefill_tokens_per_second` counts every token sent through prefill,
  context included, over the whole run.
- The `kld` fields follow llama.cpp's KL-divergence report:
  - `same_top` is the fraction of tokens where both runs rank the same token
    first.
  - `delta_p` is the change in the probability of the correct token.
  - `mean_log_ratio` is the mean of `nll - base nll`, which is
    `log(PPL / base PPL)`.

## KL divergence with top-k storage

llama.cpp stores every vocabulary entry of every scored token in its base
file, as 16-bit values. For a 248k-entry vocabulary that is about 0.5 MB per
token, or about 70 GB for the wikitext-2 test set. Reading full logit rows
back from the GPU would cost the same again in transfers. LSE stores instead,
per scored token:

- the base run's negative log-likelihood, and
- its 32 most likely token ids with their log-probabilities.

That is 264 bytes per token, about 39 MB for wikitext-2 at ctx 512.

The comparison run reads its own log-probability of those same 32 ids from
the device. The probability outside them is treated as one more outcome.
`mean_kld` is therefore the KL divergence between the two distributions
coarsened to 33 outcomes: the base's top 32 tokens, and everything else. It
never exceeds the full-vocabulary KL divergence, and it is zero only when the
two runs agree on those 32 tokens and on the remaining mass.
`mean_base_top_mass` reports how much of the base distribution the 32 ids
hold. Top-1 agreement uses the comparison run's own device argmax, and
`delta_p`, PPL and the log ratio use exact log-likelihoods.

A base and a comparison must score the same token ids with the same method,
ctx, stride and chunk limit. The run checks this and refuses a mismatch.

## How a chunk is scored

`Generator::score` runs the chunk through `prefill_passes`, the plan a prompt
of that length gets. Each pass calls `HybridLM::hidden` in the prompt
attention phase. After each pass, the LM head is applied to that pass's
scored rows, 128 rows at a time, and two device kernels reduce each row.

- `logits.lse_pick.v1` (LOOM, one workgroup per row) returns:
  - the row's maximum,
  - the sum of `exp(logit - maximum)`, and
  - the logits of the requested ids: the next token, plus the base's top-k
    ids when comparing.
- `topk_pairs` returns the row's top-k ids when recording a base or comparing.

The host then forms log-probabilities in double precision. Nothing
vocabulary-sized leaves the GPU. The CPU backend has reference
implementations of both kernels, which the unit tests use.

Serving applies the LM head to one row, so it uses the matrix-vector form of
the head kernels. Scoring applies it to 128 rows at once, so it uses the
matrix form. The hidden states come from the serving prefill in both cases.

## C API

`lse_perplexity(engine, request_json, &json_out, &err)` in
[`include/lse/lse.h`](../include/lse/lse.h) does the same thing in process.
The request object takes the following fields:

- `text` or `tokens`
- `method`, `ctx`, `stride` and `chunks`
- `kld_base_out`, `kld_top_k` and `kld_base`
- `include_windows` and `include_token_ids`

The call queues behind generation in flight, as a request does.

## Reference comparison

[`R9700_QUALIFICATION.md`](R9700_QUALIFICATION.md) compares LSE against
mlx-lm, using the same 4-bit checkpoint and identical token windows.
