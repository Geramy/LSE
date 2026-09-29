# Lemon Seed Engine (LSE)

LSE runs text models on AMD GPUs. It provides an HTTP server and a command-line program.
It generates GPU kernels for the model and device, then stores compiled kernels in a local cache.

- **HTTP server:** Chat Completions, text completions, reasoning output, and function tool calls.
- **Model formats:** MLX group-affine Q4, Q6, and Q8 weights; BF16, FP16, and FP32 weights.
- **Speculative decoding:** Native multi-token prediction (MTP) or an optional DFlash2 draft model.
- **GPU execution:** HRX with HIP or Loom kernel source, subject to platform support.
- **CPU backend:** Reference execution and a fallback when available GPU backends cannot start.

[Install](#install-a-release) · [Start the server](#start-the-http-server) ·
[MTP and DFlash2](#select-a-decoding-mode) · [Client setup](#connect-a-client) ·
[Performance](#measured-performance) · [Build](#build-from-source) ·
[Troubleshooting](#troubleshooting)

## Kernel cache

The CLI and HTTP server create `~/.lse/cache/` automatically and reuse compiled
kernels across launches. Use `--cache-dir /path/to/cache` to select another
location. The startup log prints the selected directory. The flag takes precedence
over the legacy `LSE_CACHE_DIR` environment override. No environment setting is
required. Cache entries check compiler identity, device properties and kernel source.

## Current source candidate: BF16 mode comparison

These unreleased measurements use source `9cb4cd8` plus working changes, the same
server binary/runtime, a Q4 target and BF16 KV. Each mode starts with an empty
private kernel cache; first prefill includes compilation. Sampling uses model
defaults 1/20/0.95. The first prompt has 5,207 tokens; the identical follow-up has
5,262 cached tokens and 23 new tokens.

| Mode | First prefill, tokens/s | First decode, tokens/s | Follow-up decode, tokens/s |
| --- | ---: | ---: | ---: |
| Baseline | 348.69 | 22.55 | 22.93 |
| MTP=3 | 318.86 | 26.19 | 40.19 |
| DFlash2 | 333.95 | 26.15 | 38.12 |

All first responses match. MTP=3 and DFlash2 responses match on both turns;
baseline's follow-up differs. There are zero host groups/fallbacks. These are
single observations, with different first/follow-up JIT conditions; they do not
establish a general throughput guarantee. See the
[mode comparison and limits](docs/benchmarks/bf16-mode-comparison-2026-09-28.md).

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
See the [methods and limits](docs/benchmarks/pi-execution-profile-2026-09-28.md#remove-monolithic-single-token-attention-and-retain-mtp-sessions).

## v0.4.10: shorter compilation stalls and optional KV formats

Identical Loom kernel bodies now share compiled code. Attention page traversal
stays in generated control flow instead of expanding each page into source.
In the same two-turn Pi conversation with an empty starting cache, total JIT
compilation fell from **92.91 s to 3.35 s**. First prefill fell from **91.53 s to
16.88 s**; follow-up prefill fell from **16.14 s to 0.76 s**. Both responses and
proposal acceptance counts matched. The final paired-load build measured **28.59 / 34.03 decode tokens/s** and
**310.72 prompt tokens/s** on those turns. Scratch allocation is now preserved
in optimizer facts and reported by dispatch profiling.
See the [execution profile](docs/benchmarks/pi-execution-profile-2026-09-28.md).

## KV storage

K/V storage defaults to BF16 when the model declares BF16, including the local
Qwen3.8 checkpoints, and FP16 otherwise. LSE reads `dtype` or `torch_dtype` from
`text_config` before checking the top level. An explicit model `kv_cache_dtype`
setting overrides this default; `--kv-cache-dtype` overrides both. Supported
values are `fp32`, `fp16`, `bf16`, `fp8` and `bf8`. The setting applies to target
and MTP paged caches; DFlash2 retains its private FP32 ring. Attention accumulation
remains FP32.

FP16 and BF16 each halve paged KV storage relative to FP32. Eligible gfx1201
paged batches use WMMA directly from all five formats: FP16 storage uses FP16
Q/K/P/V matrix operands; FP32, BF16, FP8 and BF8 storage use BF16 operands after
decoding or conversion. Matrix accumulators, softmax state and output remain
FP32. Single-token and selected short-query split attention keep FP32 calculation.

An earlier matched 1,024-token check measured perplexity **4.9049 with FP32 KV**
and **4.9014 with FP16 KV plus WMMA v1**. The generalized family has native
component coverage across all five formats. A later BF16 v2 check scored 1,024
pinned targets at PPL **4.8499**, with zero host fallback. This prefill result
does not establish sampled-conversation quality or a statistical improvement.
See [KV cache formats](docs/KV_CACHE.md) for selection and evidence.

## v0.4.9: chat sampling and prompt reuse

The CLI and server load each model's supported settings from `generation_config.json`.
If fields are missing, LSE uses embedded model settings and the fallback table under
`src/models`. Explicit CLI flags and HTTP request parameters override these defaults.
See [sampling defaults](docs/SAMPLING.md) for precedence and supported fields.

Pi can return streamed reasoning without invalidating the previous prompt cache.
The HTTP response reports reused tokens in `usage.prompt_tokens_details.cached_tokens`.
In a two-turn Qwen3.8 Q4 + Q8 DFlash2 chat, the second prefill fell from 14.11 s
for 5296 tokens to 0.93 s for 23 new tokens after the serialization fix.
With the model's temperature 1, top-k 20 and top-p 0.95 settings, decode measured
23.98 and 28.82 tokens/s on the two turns. These measurements describe those
requests; output length and proposal acceptance affect the result.
See the [Pi chat report](docs/benchmarks/pi-chat-2026-09-28.md).

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
See the [memory report](docs/benchmarks/kv-growth-memory-2026-09-28.md).

## Supported platforms

| Platform | GPU requirements | Kernel source |
|---|---|---|
| Linux x86_64 | ROCm 7.x and [HRX](https://github.com/ROCm/hrx-system) | HIP or Loom |
| macOS on Apple Silicon | An external AMD GPU and the activated [MacAMDGPU driver](https://github.com/lemonade-sdk/mac-amdgpu) | Loom |
| CPU | A build with the CPU backend | CPU reference execution |

The tested macOS GPU is the R9700 (`gfx1201`). The macOS package includes HSA, HRX, Loom, and their runtime libraries.
It does not install the DriverKit extension.

The macOS binaries target macOS 15 or later. GPU use also requires a macOS version supported by MacAMDGPU.
See the driver instructions for that requirement.

Linux release targets include `gfx942`, `gfx1150`, `gfx1151`, `gfx1200`, and `gfx1201`.
Check each release for its build targets and runtime requirements.

## Install a release

Use the archive for your operating system from [Releases](https://github.com/Geramy/LSE/releases).
The examples below use `v0.4.11`.

Each install procedure sets `LSE_BIN` for the later commands. Use the same terminal for those commands.

### Linux x86_64

1. Download the archive and checksum.

   ```bash
   lse_tag=v0.4.11
   lse_asset="lse-${lse_tag}-linux-x86_64"
   curl -fLO "https://github.com/Geramy/LSE/releases/download/${lse_tag}/${lse_asset}.tar.gz"
   curl -fLO "https://github.com/Geramy/LSE/releases/download/${lse_tag}/${lse_asset}.tar.gz.sha256"
   ```

2. Check the checksum. Continue only if the check reports `OK`.

   ```bash
   sha256sum -c "${lse_asset}.tar.gz.sha256"
   ```

3. Extract the archive.

   ```bash
   tar -xzf "${lse_asset}.tar.gz"
   cd "$lse_asset"
   LSE_BIN="$PWD"
   ```

4. Make the installed HRX and ROCm libraries available to the system loader.
   Replace the HRX path below with your installation path.

   ```bash
   export LD_LIBRARY_PATH="/path/to/hrx-install/lib:/opt/rocm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
   ```

### macOS on Apple Silicon

1. Install and activate the [MacAMDGPU driver](https://github.com/lemonade-sdk/mac-amdgpu).
2. Download the archive and checksum.

   ```bash
   lse_tag=v0.4.11
   lse_asset="lse-${lse_tag}-macos-arm64"
   curl -fLO "https://github.com/Geramy/LSE/releases/download/${lse_tag}/${lse_asset}.tar.gz"
   curl -fLO "https://github.com/Geramy/LSE/releases/download/${lse_tag}/${lse_asset}.tar.gz.sha256"
   ```

3. Check the checksum. Continue only if the check reports `OK`.

   ```bash
   shasum -a 256 -c "${lse_asset}.tar.gz.sha256"
   ```

4. Extract the archive.

   ```bash
   tar -xzf "${lse_asset}.tar.gz"
   cd "$lse_asset"
   LSE_BIN="$PWD/bin"
   ```

Use the programs in `bin/`. These launchers select the runtime libraries supplied with the package.
You do not need to set `DYLD_LIBRARY_PATH` yourself.

### Check the GPU

```bash
"$LSE_BIN/lse" --devices
```

Confirm that the output lists an HRX device before you load a model.
The examples below select the first HRX device with `--pool hrx:0`.

## Start the HTTP server

Set the model location. Replace the example path with your Q4 checkpoint directory.

```bash
LSE_MODEL="/absolute/path/to/qwen38-27b-q4"
```

`--model` also accepts a Hugging Face repository ID or a supported `.safetensors` file.
A repository ID can cause a model download.

Start ordinary decoding first:

```bash
"$LSE_BIN/lse-server" \
  --model "$LSE_MODEL" \
  --no-mtp \
  --pool hrx:0 --dialect loom \
  --kv-len 32768 \
  --served-name qwen38-q4 \
  --host 127.0.0.1 --port 8080
```

Confirm that startup reports `device hrx` and `generates loom`.
On Linux, you can select `--dialect hip` instead.
A dialect request is a preference. If unavailable, LSE reports the change and selects an available toolchain.

In another terminal, check the server:

```bash
curl -fsS http://127.0.0.1:8080/health
```

Send a chat request:

```bash
curl -fsS http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen38-q4",
    "messages": [{"role": "user", "content": "Say hello."}],
    "enable_thinking": false,
    "max_tokens": 64
  }'
```

Press **Ctrl+C** in the server terminal to stop it.
The default shutdown grace period is 30 seconds.

## Select a decoding mode

Stop the current server before you start another server on port 8080.
Use a draft module that matches the target model.

| Mode | Selection | Operation |
|---|---|---|
| Ordinary decoding | `--no-mtp` | The target model generates each next token. |
| MTP | `--mtp PATH --mtp-depth 3` | The MTP module proposes three tokens. The target verifies them. |
| DFlash2 | `--dflash2=on --dflash2-model PATH` | A separate draft model proposes tokens. The target verifies them. |

Speculative decoding speed depends on draft cost, verification cost, context length, and accepted proposals.
A larger proposal count does not always increase speed.

### MTP with three proposals

Use a matching Q8 MTP module. For Qwen3.8-27B, see
[`mlx-community/Qwen3.8-27B-MTP-8bit`](https://huggingface.co/mlx-community/Qwen3.8-27B-MTP-8bit).

```bash
"$LSE_BIN/lse-server" \
  --model "$LSE_MODEL" \
  --mtp /absolute/path/to/qwen38-27b-mtp-q8 \
  --mtp-depth 3 \
  --pool hrx:0 --dialect loom --kv-len 32768 \
  --served-name qwen38-q4 --host 127.0.0.1 --port 8080
```

MTP depth accepts values from 1 to 7. Its default is 3.
A request can override this value with `"mtp_depth": 3`.
Without `--no-mtp`, LSE can use an MTP module found beside the target model.

### DFlash2 with a Q8 draft model

Prepare the matching Q8 draft model with the [DFlash2 conversion instructions](docs/DFLASH2.md#reproduce-q8-conversion).

```bash
"$LSE_BIN/lse-server" \
  --model "$LSE_MODEL" \
  --dflash2=on \
  --dflash2-model /absolute/path/to/qwen38-27b-dflash2-q8 \
  --pool hrx:0 --dialect loom --kv-len 32768 \
  --served-name qwen38-q4 --host 127.0.0.1 --port 8080
```

DFlash2 replaces MTP for this server process. It evaluates a draft block of eight positions.
The default verifier checks the anchor token and up to three proposals.
See [DFlash2](docs/DFLASH2.md) for model compatibility and sampling limits.

## Connect a client

Use these settings:

| Setting | Value for the examples above |
|---|---|
| API | OpenAI-compatible Chat Completions |
| Base URL | `http://127.0.0.1:8080/v1` |
| Model ID | `qwen38-q4` |
| API key | The key set with `--api-key`, if used |
| Context limit | `32768`, to match `--kv-len` |

Use the [pi setup guide](docs/CHAT-COMPATIBILITY.md#run-with-pi) for thinking controls and function tools.
The [example pi configuration](docs/pi-models.example.json) contains the required provider fields.

LSE returns reasoning in `reasoning_content` and function calls in `tool_calls`.
The client executes tools and sends their results in the next request.
Streaming responses use server-sent events. Set `"stream": true` to request them.

| Endpoint | Support |
|---|---|
| `GET /health` | Server and speculation status |
| `GET /v1/models`, `GET /v1/models/{id}` | Loaded model information |
| `POST /v1/chat/completions` | Text chat, reasoning, tools, and streaming |
| `POST /v1/completions` | Text completions and streaming |

Current API limits:

- The server runs one generation request at a time. Other requests wait.
- `n` must be 1.
- Image and video input are unavailable.
- Strict JSON-schema generation is unavailable. Omit `strict` or set it to `false` for function tools.
- The Responses, embeddings, audio, images, and moderation endpoints return HTTP 501.
- `frequency_penalty` maps to a multiplicative repetition penalty. It does not use the exact OpenAI additive formula.

For remote access, set `--host 0.0.0.0` and an API key.
The server has no request rate limit or per-client accounting.
See the [client compatibility guide](docs/CHAT-COMPATIBILITY.md) for complete behavior and test results.

## Models and context

| Registered architecture | Model families |
|---|---|
| `qwen3.5` | Dense Qwen3.5, Qwen3.6, and Qwen3.8 |
| `qwen3.5-moe` | MoE models from these families |
| `lemonseed` | LemonSeed models with Mixture-of-Depths |

List the model architectures and local model cache:

```bash
"$LSE_BIN/lse" --list-models
"$LSE_BIN/lse" --list-cache
```

This build runs the text model. It does not run a checkpoint's vision tower.
Q4, Q6, and Q8 refer to the stored weight format.
Floating-point paths use FP32 accumulation. Integer dot products use INT32 accumulation before FP32 scale and bias operations.

`--kv-len` sets the context limit. KV storage grows with use.
A larger limit does not evaluate unused tokens. A longer active context increases attention work.
Set the client context limit to the same value as the server limit.

The HTTP server can reuse an exact consumed prompt prefix for ordinary decoding and DFlash2.
A changed prefix requires new prefill. MTP also retains verified state for an exact continuation.

## Measured performance

**Prefill** processes input tokens. **Decode** generates output tokens.
Both rates below use tokens per second.

### Current BF16 source candidate

The current baseline, MTP=3 and DFlash2 comparison is shown [above](#current-source-candidate-bf16-mode-comparison).
It uses one binary and identical requests; its method and quality limits are in the
[BF16 report](docs/benchmarks/bf16-mode-comparison-2026-09-28.md).

### Earlier FP32 Pi chat measurements

The same two-turn Pi workload used Qwen3.8-27B Q4, Q8 DFlash2 with three proposals,
temperature 1, top-k 20 and top-p 0.95. Each process started with an empty disk
kernel cache. The first prompt contained 5,207 tokens. The second reused the
conversation prefix and added 23 tokens.

| KV and attention | First prefill tokens/s, including compilation | First decode tokens/s | Follow-up decode tokens/s |
| --- | ---: | ---: | ---: |
| Earlier FP32 baseline | 56.89 | 26.64 | 31.34 |
| FP32 with bounded attention source and compiled-code reuse | 308.54 | 27.11 | 31.28 |
| FP32 with materialized DFlash feature views | 309.10 | 27.20 | 32.13 |
| Final FP32 run with paired M4 activation loads | **310.72** | **28.59** | **34.03** |
| Earlier optional FP16 with matrix attention | 179.51 | 23.89 | 27.10 |

After the compiler changes, FP32 compilation took 3.35 seconds across 372 unique
kernels, compared
with 92.91 seconds across 427 kernels in the baseline. Follow-up prefill took
0.760 seconds. Both FP32 runs generated identical responses and acceptance
counts. This establishes a reduction in cold compilation stalls; steady decode
is effectively unchanged. The subsequent buffer-view change removes 65 copy
dispatches and three compiled kernels while preserving both responses; its
small decode improvement is from one pair, not a statistical result.
Prompt rates exclude model loading. See the
[execution profile](docs/benchmarks/pi-execution-profile-2026-09-28.md).

The earlier FP16 run predates the compiler fixes. It generated different text,
with 59% follow-up acceptance versus 68% for FP32, and does not establish a
decode speedup. These measurements used an explicit FP32 control; the current
model policy defaults to BF16 for BF16 checkpoints and FP16 otherwise. See the
[KV and attention report](docs/benchmarks/kv-storage-attention-2026-09-28.md).

### Earlier 14K synthetic result

The v0.4.8 source reached these rates on an R9700 (`gfx1201`) with Qwen3.8-27B Q4 and a Q8 DFlash2 draft.

| Mode | Prompt tokens | Prefill | Decode |
|---|---:|---:|---:|
| DFlash2, three verified proposals | 14,000 | **312.67** | **42.56** |

The request used FP32 KV, temperature zero, and a 32 µs blocked-poll interval.
It generated 64 tokens; the decode timer covers 63 tokens after the first token.
The sample followed one initialization request and required no new compilation.
The synthetic prompt was repetitive and gave 100% draft acceptance. General coding workloads can give different rates.

In one baseline/candidate comparison, decode increased from 33.43 to 42.56 tokens per second (**27.32%**).
Generated text matched exactly. Both runs used GPU kernels with zero host fallback.
The baseline prefill compiled new kernels, so its prefill rate is not a comparable performance baseline.
These are local source-build results, not measurements of the downloadable archive.
See the [14K HTTP comparison](docs/benchmarks/short-query4-key-reuse-2026-09-28.md#http-comparison-at-14000-tokens).

### Earlier short-context results

These measurements used an R9700 (`gfx1201`), a Qwen3.8-27B Q4 target, and Q8 draft modules.
The prompt contained 1,024 tokens. Each request generated 64 tokens at temperature zero.
The measurements exclude initial compilation and warmup.

| Mode | Prefill | Decode |
|---|---:|---:|
| Ordinary Q4 | 466.26 | 24.20 |
| MTP, three proposals | 467.20 | 36.53 |
| DFlash2, three verified proposals | 471.47 | 38.23 |

These measurements predate v0.4.6. They do not measure the release archive.
See the [speculative decoding report](docs/benchmarks/speculative-decoding-2026-09-28.md) for the full method and results.

Recent component results:

- Flash12 attention GPU time decreased by 74.66% at 5,610 live keys and 59.72% at 14,000 keys.
- Six attention kernels compiled in 71.54 seconds instead of 117.07 seconds with the macOS compiler change.
- Both changes preserved the tested outputs. These component results do not establish the same improvement in total engine speed.

See the [Flash12 report](docs/benchmarks/flash12-key-reuse-2026-09-28.md)
and [long-context report](docs/benchmarks/long-context-http-2026-09-28.md).
Model quality uses perplexity on 1,024–2,048 target tokens.

## Troubleshooting

| Symptom | Check |
|---|---|
| The executable is missing | Linux programs are at the archive root. macOS launchers are in `bin/`. |
| No HRX device appears | Check runtime libraries and driver status. On macOS, confirm that the DriverKit extension is active. |
| The server reports CPU execution | Check `--devices`. Use `--pool hrx:0` after the GPU runtime can start. |
| First requests are slow | Check kernel compilation counts. New model shapes and KV capacities can require new kernels. |
| A later request has slow prefill | Check fresh prompt tokens and new compilation time. A request is not warm merely because it is second. |
| macOS CPU use is near 100% | This can indicate one busy compiler thread. It does not, by itself, show CPU model execution. |
| Long-context decode is slower | Compare the active token count, draft acceptance, and verification time with the benchmark workload. |
| pi permits almost no output | Match the client context limit to `--kv-len`. See the pi guide for its output reservation. |
| A client requests `/v1/responses` | Select its Chat Completions adapter. |

HTTP response `timings` includes prefill, decode, compilation, and speculation statistics.
The built-in dispatch profiler supports `LSE_PROFILE_DISPATCH=submit` and `LSE_PROFILE_DISPATCH=serial`.
Serial mode waits after each dispatch and changes execution timing. Use ordinary execution for throughput measurements.

Print the complete command options:

```bash
"$LSE_BIN/lse-server" --help
"$LSE_BIN/lse" --help
```

## Build from source

Use [BUILD_INSTRUCTIONS.md](BUILD_INSTRUCTIONS.md) for platform setup and CMake options.
The main build requirements are:

| Platform | Host compiler | Other tools |
|---|---|---|
| Linux | GCC 16 or later, with C++26 reflection | CMake 3.24+, Ninja, Rust/Cargo, ROCm, HRX |
| macOS | LLVM 21.1.8 and LLD 21.1.8 | Xcode command-line tools, CMake, Ninja, Rust/Cargo |

On Linux, `scripts/bootstrap-hrx.sh` builds the pinned HRX dependency.
CMake obtains the pinned `fastokens` source when no checkout exists.
On macOS, use the complete build script:

```bash
bash .github/scripts/build-macos.sh
```

That script builds the committed source and its pinned dependencies.
Use `Release` or `RelWithDebInfo` for performance measurements.

## Engine structure

LSE records tensor operations in a graph. The optimizer combines operations and selects supported kernel implementations.
The compiler generates device code. HRX submits that code to the GPU.
The disk cache checks device, compiler, and emitted-source identity before reuse.

Architecture and shape policies are in these headers:

- [Q4 and Q6 policies](include/lse/dispatch/quant_shapes.hpp)
- [Q8 policies](include/lse/dispatch/q8_shapes.hpp)
- [Attention policies](include/lse/dispatch/attention_shapes.hpp)
- [Matrix instruction table](include/lse/math.hpp)

The engine includes device probes, a cost model, tracing, and dispatch profiling.
A device pool can discover multiple devices. Model execution currently uses one selected device.
Multi-device model partitioning and continuous HTTP batching remain development work.

## More documentation

| Topic | Document |
|---|---|
| Build and runtime setup | [Build instructions](BUILD_INSTRUCTIONS.md) |
| Thinking, tools, pi, and API limits | [Client compatibility](docs/CHAT-COMPATIBILITY.md) |
| DFlash2 model and Q8 conversion | [DFlash2](docs/DFLASH2.md) |
| Q4 compute selection | [INT8 policy](docs/INT8_POLICY.md) |
| Quantized weights and compute formats | [Quantized operands](docs/QUANT_OPERANDS.md) |
| FP8 and BF8 conversion | [FP8 conversion](docs/FP8_CONVERSION.md) |
| Measured optimization results | [Benchmark reports](docs/benchmarks/) |

## License

LSE uses the [MIT license](LICENSE.md).
