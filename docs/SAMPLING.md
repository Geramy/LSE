# Sampling defaults

HTTP and CLI use the checkpoint's supported sampling defaults. Each field uses
this priority, highest first:

1. Explicit request parameter or CLI flag.
2. `generation_config.json` beside `config.json`.
3. The model config's embedded `generation_config` object.
4. A top-level model-config sampling field.
5. The central model-family fallback table.
6. Generic defaults: temperature 1, top-k disabled, top-p 1, repetition penalty 1.

The loader handles temperature, top-k, top-p and repetition penalty.
`do_sample:false` makes the model default greedy. Other generation fields are
not implemented by this loader. The supported Qwen3.5 family fallback is
1/20/0.95; files and explicit fields take priority for every model type.

Missing and null fields inherit. Explicit temperature 0 is greedy and top-k 0
disables top-k. Malformed supplied metadata or request fields return an error.
HTTP also retains seed and frequency-penalty handling; an explicit repetition
penalty takes priority over the frequency-penalty mapping. The server logs its
loaded temperature, top-k and top-p once at startup.
