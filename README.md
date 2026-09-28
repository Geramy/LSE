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
The examples below use `v0.4.6`.

Each install procedure sets `LSE_BIN` for the later commands. Use the same terminal for those commands.

### Linux x86_64

1. Download the archive and checksum.

   ```bash
   lse_tag=v0.4.6
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
   lse_tag=v0.4.6
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
A changed prefix requires new prefill. MTP requests currently start with fresh model state.

## Measured performance

**Prefill** processes input tokens. **Decode** generates output tokens.
Both rates below use tokens per second.

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
