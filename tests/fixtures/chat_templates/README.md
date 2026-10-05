Chat templates copied verbatim from published checkpoints, for the thinking
control tests (test_chat_template):

- qwen3.8/chat_template.jinja, generation_config.json: mlx-community/Qwen3.8-27B-8bit
  (thinking toggle plus reasoning_effort xhigh/medium/low).
- qwen3.6/tokenizer_config.json: Qwen/Qwen3.6-27B, chat_template only
  (thinking toggle, no effort levels).
- smollm2/chat_template.jinja: HuggingFaceTB/SmolLM2-135M-Instruct, from the
  GGUF metadata (no thinking controls).
