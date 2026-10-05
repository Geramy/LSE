# Output limits, context, thinking levels and sampling defaults

This page specifies the request fields and response fields that control how long
a reply runs, what happens when the context fills, which thinking levels a model
offers and which sampling defaults it uses. The HTTP server and the in-process C
API (`lse_request`, `lse_model_info`) share the same JSON, so every shape below
applies to both. Fields that are not part of the OpenAI schema use LSE's own
names, and OpenAI clients ignore them.

## Output limits

LSE has no output limit of its own. A reply ends at the first of these:

| End | `finish_reason` | `stop_reason` |
|---|---|---|
| The model emits an end-of-turn token | `stop` (`tool_calls` when it made a call) | `stop_token` |
| One of the request's `stop` strings | `stop` | `stop_sequence` |
| A token limit (see the order below) | `length` | `max_tokens` |
| The prompt plus the reply fill the context | `length` | `context_full` |
| The client disconnected or cancelled | — | `cancelled` |

A request's token limit comes from the first of these that is set:

1. The request's `max_tokens`, or `max_completion_tokens` (if both are sent, `max_tokens` is used). Each must be a positive integer; `null` means not sent.
2. The operator's cap, `lse-server --max-tokens N` (`lse_config.max_tokens`). The default is 0, which means no cap. With a cap, a request asking for more than the cap gets HTTP 400 (`param: "max_tokens"`).
3. `max_new_tokens` from the model's `generation_config.json`, when that file sets it.
4. No limit.

If the model's `generation_config.json` sets `max_length`, it bounds the prompt
plus the reply for items 2 to 4. Whatever the limit, the prompt plus the reply
never exceed the context (`context_length`, the engine's `--kv-len`).

LSE looked for model-defined output limits in `generation_config.json`
(`max_new_tokens`, `max_length`), in `config.json` (top-level fields and an
embedded `generation_config` object) and in the chat template. The Qwen3.6 and
Qwen3.8 checkpoints set none of them. `tokenizer_config.json`'s
`model_max_length` is the trained context length, not an output limit, so it is
not used as one.

## Context full

### A reply that fills the context

When a reply fills the context, it ends normally with HTTP 200. The text
generated so far is kept. The choice reports `finish_reason: "length"` and
`stop_reason: "context_full"`.

Every completion response carries `lse_context`, so a client can offer to compact
the conversation before or after the context fills. For a streamed reply, it is
in the final chunk, the one with `finish_reason`.

```json
{
  "object": "chat.completion",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "..."},
    "finish_reason": "length",
    "stop_reason": "context_full",
    "logprobs": null
  }],
  "usage": {"prompt_tokens": 40, "completion_tokens": 472, "total_tokens": 512,
            "prompt_tokens_details": {"cached_tokens": 0}},
  "lse_context": {"tokens_used": 512, "context_length": 512, "tokens_remaining": 0},
  "timings": {"...": "..."}
}
```

| Field | Type | Meaning |
|---|---|---|
| `choices[0].stop_reason` | string | `stop_token`, `stop_sequence`, `max_tokens`, `context_full` or `cancelled` |
| `lse_context.tokens_used` | integer | Prompt tokens (whole conversation, including cached ones) plus reply tokens |
| `lse_context.context_length` | integer | The most tokens the engine holds; the same as `/v1/models` `context_length` |
| `lse_context.tokens_remaining` | integer | `context_length - tokens_used`, never below 0 |

The final chunk of a stream has the same fields:

```json
{"object": "chat.completion.chunk",
 "choices": [{"index": 0, "delta": {}, "finish_reason": "length", "stop_reason": "context_full"}],
 "lse_context": {"tokens_used": 512, "context_length": 512, "tokens_remaining": 0},
 "timings": {"...": "..."}}
```

`/v1/completions` uses the same `stop_reason` and `lse_context` fields. Its
choices carry `text` instead of `message`.

### A prompt that fills the context

A request whose prompt leaves no room for a reply (`prompt_tokens >= context_length`)
gets HTTP 400 before any work starts. The error code is the same, `context_full`:

```json
{
  "error": {
    "message": "the prompt is 600 tokens and the context holds 512; compact or shorten the conversation",
    "type": "invalid_request_error",
    "code": "context_full",
    "param": "messages",
    "lse_context": {"tokens_used": 600, "context_length": 512, "tokens_remaining": 0}
  }
}
```

`param` is `messages` on `/v1/chat/completions` and `prompt` on `/v1/completions`.
Here, `tokens_used` is the size of the prompt. Over `lse_request`, this arrives as
`LSE_EVENT_ERROR` with status 400 and the same body.

A client can read `error.code == "context_full"`, or `stop_reason == "context_full"`,
and does not need to match message strings. LSE does no compaction itself.

## Thinking levels

The thinking levels come from the model's own chat template. LSE reads the
template from `chat_template.jinja` beside `config.json`. If that file is absent,
it uses `tokenizer_config.json` `chat_template` (a string, or the entry named
`default` in a list), then `chat_template.json`. LSE renders the template with
probe conversations, using its own Jinja interpreter, and compares the results:

- **Toggle**: the template variable `enable_thinking`, when setting it changes the rendering.
- **Effort levels**: the values of the template variable `reasoning_effort` that the template compares against and that render without `raise_exception`.
- **Instruction**: the text the template inserts before the system prompt for that level.
- **Generation prompt**: the assistant opening the template appends for that level. LSE uses it to open the reply.
- **Default**: the level whose rendering matches the template's rendering with neither variable set.

No model has hard-coded strings. A model starts without loading if its template
can't be read: `lse_open` fails and names the file and the construct.

### `/v1/models`

Each model entry has a `thinking` object. The native API has the same object in
`lse_model_info` and `GET /v1/lse/model_info` (and under `served` there).

```json
"thinking": {
  "supported": true,
  "source": "/models/qwen38-27b-q4/chat_template.jinja",
  "toggle": "enable_thinking",
  "effort": "reasoning_effort",
  "default_level": "xhigh",
  "levels": [
    {"id": "none",   "enable_thinking": false, "reasoning_effort": null,
     "instruction": null, "opens_reasoning": false, "default": false},
    {"id": "xhigh",  "enable_thinking": true,  "reasoning_effort": "xhigh",
     "instruction": "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.",
     "opens_reasoning": true, "default": true},
    {"id": "medium", "enable_thinking": true,  "reasoning_effort": "medium",
     "instruction": null, "opens_reasoning": true, "default": false},
    {"id": "low",    "enable_thinking": true,  "reasoning_effort": "low",
     "instruction": "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.",
     "opens_reasoning": true, "default": false}
  ]
}
```

| Field | Type | Meaning |
|---|---|---|
| `supported` | boolean | The template defines at least one level |
| `source` | string or null | The file the template came from; null when the model has none |
| `toggle` | string or null | `"enable_thinking"` when the template has the switch |
| `effort` | string or null | `"reasoning_effort"` when the template has effort levels |
| `default_level` | string or null | The level a request that names none gets |
| `levels[].id` | string | The value to send as `reasoning_effort` |
| `levels[].enable_thinking` | boolean or null | What the level passes to the template's toggle; null without one |
| `levels[].reasoning_effort` | string or null | What it passes to the effort variable; null when none |
| `levels[].instruction` | string or null | Template text placed first in the system prompt; null when the template adds none |
| `levels[].opens_reasoning` | boolean | The reply starts inside a `<think>` block, so its text up to `</think>` is returned as `reasoning_content` |
| `levels[].default` | boolean | This is `default_level` |

Level ids, in the order they're listed:

- `none` means thinking is off (`enable_thinking: false`). It is present only when the template has the toggle.
- Each effort value the template accepts, in the order the template names them, with thinking on.
- `on` means thinking is on, for a template that has the toggle but no effort levels.

| Model template | `levels` | `default_level` |
|---|---|---|
| Qwen3.8 (`chat_template.jinja`) | `none`, `xhigh`, `medium`, `low` | `xhigh` |
| Qwen3.6 (`tokenizer_config.json`) | `none`, `on` | `on` |
| SmolLM2, or a model with no template | (empty), `supported: false` | null |

Qwen3.8's template has no `high` level. Its `medium` level adds no instruction,
and its `xhigh` and `low` levels each add their own.

### Request fields

| Field | Values |
|---|---|
| `reasoning_effort` | A `levels[].id`. Also accepted as `thinking_level`, `chat_template_kwargs.reasoning_effort` or `reasoning.effort` |
| `enable_thinking` | boolean. Also accepted as `thinking` (a boolean, or `{"type": "enabled" \| "disabled"}`) or `chat_template_kwargs.enable_thinking` |

Resolution:

1. If `reasoning_effort` is sent, it must be a level id. The switch, if also sent, must agree with that level's `enable_thinking`.
2. If only `enable_thinking: false` is sent, the level is `none`. If only `enable_thinking: true` is sent, the level is the default when the default thinks, otherwise the first level that thinks.
3. If neither is sent, the level is `default_level`.
4. On a model without thinking controls, `reasoning_effort: "none"` and `enable_thinking: false` are accepted, since the model already doesn't think. Asking such a model to think is an error.

Errors are HTTP 400, `type: "invalid_request_error"`, and include `levels`, the
model's level ids:

| `error.code` | When |
|---|---|
| `unsupported_reasoning_effort` | The level is not one the template defines, e.g. `high` or `minimal` on Qwen3.8. No level is aliased to another |
| `thinking_unsupported` | Thinking was asked of a model whose template has no controls, or `enable_thinking: false` was sent to a template that can't switch thinking off |
| `conflicting_thinking` | Two spellings of the switch or the level disagree, or the level contradicts the switch |
| `invalid_thinking` | A field has the wrong type |

```json
{"error": {"message": "reasoning_effort 'high' is not defined by this model's chat template; levels: none, xhigh, medium, low",
           "type": "invalid_request_error", "code": "unsupported_reasoning_effort",
           "param": "reasoning_effort", "levels": ["none", "xhigh", "medium", "low"]}}
```

On `/v1/completions` the prompt is passed through unchanged. There, the thinking
fields only decide whether `<think>` markers in the output are split into
`reasoning`. They are checked for type and agreement, but not against the
template.

## Sampling defaults

The model's own files supply the defaults for any field a request leaves out
(or sends as `null`). For each field, the first of these that sets it wins:

1. The request.
2. `generation_config.json` beside `config.json`.
3. `config.json`'s embedded `generation_config` object, then its top-level fields.
4. LSE's neutral defaults.

`lse-server --temperature` overrides the temperature default and reports
`server_option` as its source. `do_sample: false` in the model's files makes the
default temperature 0 (greedy). There's no per-model or per-family table.

### `/v1/models`

Each model entry has a `generation_defaults` object. The native API has the same
object in `lse_model_info` and `GET /v1/lse/model_info`.

```json
"generation_defaults": {
  "temperature": 1.0, "top_k": 20, "top_p": 0.95, "min_p": 0.0,
  "repetition_penalty": 1.0, "presence_penalty": 0.0,
  "max_new_tokens": null, "max_length": null,
  "sources": {
    "temperature": "generation_config.json", "top_k": "generation_config.json",
    "top_p": "generation_config.json", "min_p": "lse_default",
    "repetition_penalty": "lse_default", "presence_penalty": "lse_default",
    "max_new_tokens": "lse_default", "max_length": "lse_default"
  }
}
```

`sources` values are `generation_config.json`, `config.json`, `server_option` or
`lse_default`. The example shows Qwen3.8's `generation_config.json`
(`temperature` 1.0, `top_k` 20, `top_p` 0.95).

### LSE's neutral defaults

These defaults apply to fields the model's files don't set, and LSE reports them
as `lse_default`. They mean plain sampling from the model's distribution:

| Field | Default | Meaning of the default |
|---|---|---|
| `temperature` | 1.0 | The model's own distribution (0 or less is greedy) |
| `top_k` | 0 | Off |
| `top_p` | 1.0 | Off |
| `min_p` | 0.0 | Off |
| `repetition_penalty` | 1.0 | Off |
| `presence_penalty` | 0.0 | Off |
| `max_new_tokens`, `max_length` | null | No limit |

### Request fields

Each of these fields is accepted on `/v1/chat/completions` and `/v1/completions`,
and so through `lse_request` too:

| Field | Type and range |
|---|---|
| `temperature` | number; 0 or less is greedy |
| `top_k` | integer; 0 or -1 turns it off |
| `top_p` | number in [0, 1]; 1 turns it off |
| `min_p` | number in [0, 1]; drops tokens below `min_p` times the top token's probability |
| `presence_penalty` | number; subtracted once from the logit of each distinct token seen in the last 64 tokens |
| `repetition_penalty` | number > 0; divides a seen token's logit (or multiplies a negative one) |
| `frequency_penalty` | number; mapped to `repetition_penalty = 1 + frequency_penalty` when positive (an explicit `repetition_penalty` wins) |
| `seed` | nonnegative integer |

Sampling applies the penalties first, then temperature, then top-k, top-p and
min-p, in that order. The `lse` command line takes `--top-k`, `--top-p`,
`--min-p`, `--presence-penalty`, `--repeat-penalty` and `--temperature`, each
defaulting to the model's value.
