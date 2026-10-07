# Perplexity

LSE can score a fixed text and report its perplexity. The text runs through
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
text before and after, and compare the two perplexities.

Only the target model is scored. A speculative decoder (MTP or DFlash2) does
not change the target's logits, so it is not used even when it is loaded.
Check speculative decoding separately: greedy output must match the
non-speculative output, and acceptance is reported in the timings.

## Command line

```bash
export LSE_REQUIRE_DEVICE_KERNELS=1
lse-server --model <model-dir> --pool hrx:0 --dialect loom \
  --batch-size 1024 --ubatch-size 1024 --kv-cache-dtype bf16 --kv-len 262100 --no-mtp \
  --perplexity wikitext-2-raw-test.txt --perplexity-window 1024 \
  --perplexity-output result.json
```

`lse-server` loads the model, scores the file, writes the JSON and exits. It
does not open an HTTP port.

| Option | Meaning |
|---|---|
| `--perplexity FILE` | UTF-8 text, tokenized the way a prompt is, with no chat template |
| `--perplexity-tokens FILE` | Score a JSON array of token ids instead of text |
| `--perplexity-window N` | Tokens per window (default 1024) |
| `--perplexity-stride N` | Tokens between window starts, 1 to N (default: the window) |
| `--perplexity-max-windows N` | Score only the first N windows (default: all) |
| `--perplexity-output PATH` | Where to write the JSON (default: standard output) |
| `--perplexity-token-ids` | Include the token ids in the JSON |

Every other `lse-server` option means what it means when serving.
`--FlashPrefillV2=off` selects dense prefill. Each window must fit `--kv-len`.

## Windows

The text is split with the standard sliding-window method. Windows of
`window` tokens start every `stride` tokens. Each window scores only the
tokens that no earlier window scored. Each token is predicted from the tokens
before it in its own window.

- With `stride == window`, windows don't overlap. Each window scores every
  token except its first, which has no context in that window.
- With `stride < window`, every token after the first token of the text is
  scored exactly once. After the first window, each scored token has at least
  `window - stride` tokens of context.

Each window runs in a fresh session, as a request without a session id does.

## Output

```json
{
  "object": "lse.perplexity",
  "tokens": 297193,
  "token_ids_sha256": "e65198...",
  "window": 1024, "stride": 1024,
  "windows_scored": 291, "scored_tokens": 296902,
  "nll_sum": 0.0, "mean_nll": 0.0, "perplexity": 0.0,
  "config": {"device_arch": "gfx1201", "prefill_attention": "flashprefill-v2",
             "prefill_attention_scale": 0.1, "kv_cache_dtype": "bf16",
             "kv_capacity": 262100, "batch_size": 1024, "ubatch_size": 1024,
             "head_rows": 128, "cpu_fallback_allowed": false},
  "windows": [{"index": 0, "begin": 0, "end": 1024, "first_target": 1, "scored": 1023,
               "nll_sum": 0.0, "mean_nll": 0.0, "perplexity": 0.0, "seconds": 0.7}]
}
```

`mean_nll` is the mean negative log-likelihood in nats per scored token.
`perplexity` is `exp(mean_nll)`. `token_ids_sha256` is the SHA-256 of the
token ids as little-endian 32-bit integers. If another implementation
computes the same hash, it scored the same tokens.

## How a window is scored

`Generator::score` runs the window through `prefill_passes`, the plan a
prompt of that length gets. Each pass calls `HybridLM::hidden` in the prompt
attention phase. After each pass, the LM head is applied to that pass's
scored rows, 128 rows at a time. One device kernel,
`logits.lse_pick.v1` (LOOM, one workgroup per row), reduces each logit row to
three values:

- its maximum,
- the sum of `exp(logit - maximum)`, and
- the logit of the next token.

The host then forms `max + log(sum) - target` in double precision.
That is 12 bytes per scored token instead of a whole vocabulary row. There is
no host path on a GPU backend. The CPU backend has a reference implementation
of the same kernel, which the unit tests use.

Serving applies the LM head to one row, so it uses the matrix-vector form of
the head kernels. Scoring applies it to 128 rows at once, so it uses the
matrix form. The hidden states come from the serving prefill in both cases.

## C API

`lse_perplexity(engine, request_json, &json_out, &err)` in
[`include/lse/lse.h`](../include/lse/lse.h) does the same thing in process.
The request object takes `text` or `tokens`, `window`, `stride`,
`max_windows`, `include_windows` and `include_token_ids`. The call queues
behind generation in flight, as a request does.

## Reference comparison

[`R9700_QUALIFICATION.md`](R9700_QUALIFICATION.md) compares LSE against
mlx-lm, using the same 4-bit checkpoint and identical token windows.
