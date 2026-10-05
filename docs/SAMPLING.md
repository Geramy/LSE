# Sampling defaults

HTTP, the in-process API and the `lse` command line use the checkpoint's own
sampling defaults. For each field, the first of these that sets it wins:

1. An explicit request parameter or CLI flag.
2. `generation_config.json` beside `config.json`.
3. The model config's embedded `generation_config` object.
4. A top-level model-config sampling field.
5. LSE's neutral defaults: temperature 1, top-k off (0), top-p off (1), min-p off
   (0), repetition penalty off (1), presence penalty off (0).

The loader reads `temperature`, `top_k`, `top_p`, `min_p`, `repetition_penalty`,
`presence_penalty` and `do_sample`. `do_sample: false` makes the model default
greedy. It also reads `max_new_tokens` and `max_length`, which are the only
output limits LSE applies without being asked; see [API](API.md#output-limits).
There's no per-family table. A model whose files name nothing gets the neutral
defaults, and `/v1/models` reports them as `lse_default`.

Missing and null fields inherit. An explicit temperature of 0 is greedy, and a
top-k of 0 (or -1 in a request) turns top-k off. Malformed metadata in the
model's files, or malformed request fields, are an error. HTTP also handles
seed and frequency penalty; an explicit repetition penalty takes priority over
the frequency-penalty mapping. At startup, the server logs each default and
where it came from. `/v1/models` and model info report them under
`generation_defaults`, with a `sources` map. `Config::to_json` writes every
default into its `generation_config`, so a converted or repacked model keeps
them.
