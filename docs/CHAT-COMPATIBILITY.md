# Thinking and function tools

LSE serves text chat through `/v1/chat/completions`, including ordinary JSON
responses and server-sent events. It translates Qwen3.8's trained tool syntax
into structured function calls and renders tool results back into the next
prompt. Tool execution belongs to the client.

## Run with pi

Start an HTTP server with the Q4 target and Q8 MTP module:

```sh
./build/lse-server --model /path/to/qwen38-27b-q4 \
  --mtp /path/to/qwen38-27b-mtp-q8 --mtp-depth 3 \
  --pool hrx:0 --kv-len 32768 \
  --served-name qwen38-q4 --host 127.0.0.1 --port 8080
```

Merge the `lse` provider from [pi-models.example.json](pi-models.example.json)
into `~/.pi/agent/models.json`, keeping existing providers. Set the configured
context window to the server's KV capacity. pi 0.85.1 reserves 4096 context
tokens before allocating output; a 4096-token advertised window leaves only one
output token. The example uses 32768 on both sides. Then select it with:

```sh
pi --provider lse --model qwen38-q4 --thinking low
```

`--thinking off` works too. `thinkingFormat: "qwen"` makes pi send the
`enable_thinking` switch. The server also accepts `reasoning_effort`, `thinking`
and `chat_template_kwargs.enable_thinking`. The accepted `reasoning_effort`
values are the levels the model's own chat template defines, as listed in
`/v1/models` `thinking.levels`. Qwen3.8's template defines `none`, `xhigh`,
`medium` and `low`; a level it doesn't define (`high`, `minimal`) gets HTTP 400
with `code: "unsupported_reasoning_effort"`, not an alias. Effort controls the
prompt instruction; it does not impose a separate reasoning-token budget. See
[API](API.md#thinking-levels).

Use `api: "openai-completions"`. The Responses API (`/v1/responses`) is not
implemented. `supportsStrictMode: false` is required because LSE does not perform
grammar-constrained JSON-schema decoding; requests with `strict: true` receive
HTTP 400 instead of silently claiming that guarantee. Schema argument types
still guide conversion of XML parameter values into JSON. Strings such as
`"007"` remain strings.

## Wire behavior

- Leading `developer` and `system` messages form the Qwen system prompt.
- Text and text-part arrays are supported; image parts receive HTTP 400.
- Assistant function-call history and `role: "tool"` results retain their IDs.
  Parallel results are ordered by their matching call IDs before rendering.
- `tool_choice` accepts `auto`, `none`, `required`, or a named function.
  `parallel_tool_calls: false` rejects a generated second call. Choice requirements
  steer the prompt and validate completed output; failure is a model-output error.
- Complete calls return `tool_calls` containing `id`, `type: "function"`,
  `function.name`, and JSON text in `function.arguments`. Completed tool turns
  finish with `tool_calls`; token limits finish with `length`.
- Streaming begins with the assistant role. Reasoning arrives incrementally in
  `delta.reasoning_content`; final answer text arrives in `delta.content`.
  Function calls emit indexed deltas after a complete call has been parsed.
  Partial calls at a token limit never become executable tool-call objects.
- There is no default output limit. A reply ends at an end-of-turn token, a stop
  sequence, the request's `max_tokens`, or a full context. Each choice says which
  in `stop_reason`, and every response reports `lse_context` (`tokens_used`,
  `context_length`, `tokens_remaining`). A prompt that fills the context gets
  HTTP 400 with `code: "context_full"`. See [API](API.md#context-full).
- `stream_options.include_usage: true` adds a final usage chunk with empty
  `choices`, followed by `[DONE]`. Usage counts reasoning and output tokens.
- The assistant turn opens with the generation prompt the model's chat template
  defines for the level. For Qwen3.x thinking levels it already opens `<think>`,
  so an unfinished reasoning passage remains reasoning even when generation never
  emits an opening or closing tag.

## Standard OpenAI client

Clients must select Chat Completions and use the server's `/v1` base URL.
For example, with the standard Python SDK:

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="local")
response = client.chat.completions.create(
    model="qwen38-q4",
    messages=[{"role": "user", "content": "What is 2 plus 3?"}],
    reasoning_effort="low",
    max_completion_tokens=256,
)
print(response.choices[0].message.content)
```

Use function tools with `strict: false` (or omit `strict`). Return the assistant
message and a tool result carrying its `tool_call_id` in the next request.
Frameworks that default to Responses must select their Chat Completions adapter.

## Verification

`test_chat_protocol` covers every two-piece split and bytewise splitting of a
reasoning/tool response, incremental and truncated thinking, XML and JSON calls,
multiple IDs and indices, typed arguments, tool choices, truncated calls,
nullable reasoning history, out-of-order results and invalid requests.

The bounded live check can be repeated against an already running server:

```sh
python3 scripts/test-chat-compat.py --base-url http://127.0.0.1:8080 \
  --model qwen38-q4 --output /tmp/lse-chat-check
```

It saves requests and responses and asserts thinking on/off, thinking truncation,
nonstream and stream calls, a call/result/answer round trip, named/required/none
choices, and streaming usage. These are API integration checks; no perplexity
sweep is involved.

The installed pi client can be checked separately with an isolated provider
configuration and only its `read` tool enabled:

```sh
python3 scripts/test-pi-compat.py --base-url http://127.0.0.1:8080/v1 \
  --model qwen38-q4 --context-window 32768 --output /tmp/lse-pi-check
```

## Native results — 2026-09-28

Qwen3.8-27B Q4 with the Q8 MTP module, depth 2, gfx1201/Loom, device kernels
required, and a 64 us HSA wait. pi 0.85.1 used a matching 32768-token context.

| Check | Result |
|---|---|
| Thinking disabled through direct and template flags | Clean `READY` answer |
| Thinking enabled, low effort | Reasoning separated from final answer `5` |
| Thinking truncated at eight tokens | Reasoning deltas only; `length` finish |
| Named nonstream function call | `lookup`, JSON arguments, call ID, `tool_calls` finish |
| Tool-result follow-up | Answer included the supplied `sapphire-517` value |
| Required streamed function call | Indexed call delta, usage-only final chunk, `[DONE]` |
| Tool choice `none` | Plain answer without tool calls |
| Actual pi, thinking off | Successful `read`, then exact `marigold-826` answer |
| Actual pi, thinking low | Thinking events, successful `read`, then exact answer |
| OpenAI JavaScript SDK 6.40.0 | Nonstream and stream completions plus usage parsed successfully |

The initial pi run with a 4096-token advertised context failed because pi
clamped its output budget to one token. Both runs passed with matching 32768
settings. Their complete two-turn elapsed times were 193.20 s (off) and
248.29 s (low). Server prefill durations ranged from 59.87 to 143.37 s;
decode durations were 0.33–3.13 s. These checks include compilation and varying
prompt lengths. They establish protocol behavior, not warm throughput, and
expose a remaining prefill latency issue.

Local requests, responses, pi events and SDK output are saved under
`mac_amdgpu/build/pp-optimization/chat-compat-20260928/`. Eleven parser regressions,
the existing server and CLI tests passed. The graph-only link targets reported
by the preceding platform CI failure were also rebuilt and passed after moving
attention policy into the graph library.

Protocol references: [OpenAI Chat API](https://developers.openai.com/api/reference/resources/chat),
[pi model configuration](https://github.com/earendil-works/pi/blob/main/packages/coding-agent/docs/models.md).
The Qwen framing follows the local Qwen3.8 checkpoint's `chat_template.jinja`.
