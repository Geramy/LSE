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
required. Cache entries include the engine release version and check compiler
identity, device properties and kernel source. Startup removes complete older
LSE-owned artifact families from the selected directory. It preserves current
and newer releases, unrelated files, incomplete records and symlinks. An update
can compile kernels again; later launches reuse the current release cache.

## Current release: v0.4.18

This version reduces unused GPU weight-allocation space for the Q4 target and
Q8 DFlash2 draft by about 1.71 GB in a matched R9700 test. Warm decode stayed
at 49.9 tokens/s, and output and acceptance matched. See the
[weight slab measurements](docs/benchmarks/weight-slab-memory-2026-09-29.md).
The v0.4.17 workspace-retirement fix remains active; see
[prefill memory ownership](#prefill-memory-ownership).

The server verifies all seven proposals from the DFlash2 block-8 checkpoint.
Conditional drafting uses the request temperature. Probability-ratio rejection
and residual sampling preserve the target sampling distribution.

Prefill uses shared Q4 activation panels for the measured M1024 projection
shapes. M8 gate/up and QKV projections now consume two activation rows at a time.
The accepted M8 down WMMA, typed attention, buffer views and memory retirement
remain active. Floating-point accumulation remains FP32.

M8 GDN alpha and beta projections now reuse the activation panel shared by QKV
and the GDN gate. Their preparation-inclusive paired GPU time falls from
0.0500 to 0.0249 ms, with complete fused outputs matching exactly. This is a
component result. The later matched DFlash2 HTTP check measured 616.38 PP/s
and 43.05 TPS on its resident request, with exact responses and unchanged acceptance.
See the [v0.4.15 check](docs/benchmarks/m8-gdn-http-2026-09-29.md).

The current source combines the measured M8 Q4 gate/up sequence in one kernel.
Its matched DFlash2 resident decode rate increases from 43.44 to 45.64 TPS, with
identical output. The latest same-executable resident comparison measures
**634.34 PP/s / 24.86 TPS** for baseline, **607.88 PP/s / 48.42 TPS** for MTP=3,
and **629.80 PP/s / 45.64 TPS** for seven-proposal DFlash2. These are 1024-token
coding requests at temperature 0.6. See the
[combined gate/up report](docs/benchmarks/m8-gate-up-pair-2026-09-29.md) for cold
results, resource costs and method. These rates do not apply to every context
or Pi conversation.

| Optimization | Evidence |
| --- | --- |
| Full-width DFlash2 and conditional sampling | [DFlash2 results](docs/benchmarks/dflash-fullwidth-2026-09-29.md) |
| M8 down WMMA and prefill memory retirement | [M8 and memory results](docs/benchmarks/wmma-m8-down-2026-09-29.md) |
| M1024 down activation panel | [Down prefill results](docs/benchmarks/prefill-m1024-down-2026-09-29.md) |
| Skip fully masked attention windows | [Attention results](docs/benchmarks/attention-causal-windows-2026-09-29.md) |
| Cooperative M1024 gate/up | [Gate/up results](docs/benchmarks/prefill-m1024-up-2026-09-29.md) |
| Cooperative QKV, GDN gate and attention projections | [Projection results](docs/benchmarks/prefill-projections-2026-09-29.md) |
| M8 activation register lifetime | [Decode scheduling results](docs/benchmarks/m8-dot4-rowpairs-2026-09-29.md) |
| Combined M8 gate/up | [Gate/up pairing results](docs/benchmarks/m8-gate-up-pair-2026-09-29.md) |
| M8 GDN alpha/beta panel reuse | [Rate projection results](docs/benchmarks/m8-gdn-rate-panel-2026-09-29.md) |

The launch examples use temperature 0.6 and batch/ubatch 1024. Explicit request
parameters override launch defaults. Earlier versions and measurements remain
in [release history](docs/RELEASE_HISTORY.md).

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

## Prefill memory ownership

This release frees completed forward workspaces before a new request and
when prefill changes chunk width. Consecutive chunks of the same width retain
replay. Model weights, compiled kernels and live KV storage remain resident.
DFlash2 also releases completed large context-projection programs.

A matched 6143-token ragged request reduced reserved GPU memory from 29.77 GB
to 28.26 GB, with identical output and acceptance counts. Prefill and decode
rates remained within 0.5% in that comparison. See the
[memory report](docs/benchmarks/prefill-workspace-memory-2026-09-29.md) for the
method and the unresolved reproduction of the reported allocation failure.

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
The Linux archive bundles its selected HRX runtime and patched Loom compiler;
root and `bin/` launchers load those libraries and forward the existing CLI arguments.
A compatible Linux C/C++ runtime, ROCm 7.x, HSA and GPU driver remain required.
`BUILD.json` records the compiler source pins, patch hashes and bundled library hashes.
Check each release for its build targets and runtime requirements.

## Install a release

Use the archive for your operating system from [Releases](https://github.com/Geramy/LSE/releases).
The examples below use `v0.4.18`.

Each install procedure sets `LSE_BIN` for the later commands. Use the same terminal for those commands.

### Linux x86_64

1. Download the archive and checksum.

   ```bash
   lse_tag=v0.4.18
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

4. If ROCm is outside the system loader paths, add the runtime directory from
   your matching ROCm installation. The launchers select the bundled HRX and Loom.

   ```bash
   export LD_LIBRARY_PATH="/opt/rocm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
   ```

### macOS on Apple Silicon

1. Install and activate the [MacAMDGPU driver](https://github.com/lemonade-sdk/mac-amdgpu).
2. Download the archive and checksum.

   ```bash
   lse_tag=v0.4.18
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
  --temperature 0.6 --batch-size 1024 --ubatch-size 1024 \
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
  --temperature 0.6 --batch-size 1024 --ubatch-size 1024 \
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
  --temperature 0.6 --batch-size 1024 --ubatch-size 1024 \
  --served-name qwen38-q4 --host 127.0.0.1 --port 8080
```

DFlash2 replaces MTP for this server process. It evaluates a draft block of eight positions.
The verifier checks the anchor token and all seven proposals.
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

### Combined M8 gate/up (v0.4.16 measurements)

Same executable, 1024-token coding prompts, 384 generated tokens, temperature 0.6,
top-k 20, top-p 0.95, BF16 KV and batch/ubatch 1024. Each mode starts with an empty
kernel cache. The resident request reuses compiled code and no prompt KV.

| Mode | Cold PP/s | Cold TPS | Resident PP/s | Resident TPS |
|---|---:|---:|---:|---:|
| Baseline | 446.28 | 24.23 | 634.34 | 24.86 |
| MTP=3 | 403.89 | 37.44 | 607.88 | 48.42 |
| DFlash2 | 418.73 | 33.42 | 629.80 | 45.64 |

DFlash2 improves 5.07% against a matched 43.44 TPS control; output and acceptance
are identical. Higher register use is included in these measurements. See the
[gate/up report](docs/benchmarks/m8-gate-up-pair-2026-09-29.md). The published
v0.4.15 binaries retain the results below.

### v0.4.15 DFlash2 check

Measured source `cb285b1`. One matching cold/resident pair uses the same
1,024-token requests and sampling settings as the earlier comparison below.

| DFlash2, seven proposals | Prefill tokens/s | Decode tokens/s |
| --- | ---: | ---: |
| Cold | 413.74 | 32.01 |
| Resident | 616.38 | 43.05 |

Both complete responses and acceptance statistics match v0.4.14. There are zero
host groups, fallbacks or reused prompt KV tokens. The new release cache namespace
is verified. The resident decode difference is about +1.5% in this single pair.
It is not a statistical result or a long-context Pi guarantee. Baseline and MTP=3
were not rerun for this addition. See the [HTTP check](docs/benchmarks/m8-gdn-http-2026-09-29.md).

### Last same-binary BF16 measurements

Measured source `c11f103` (v0.4.14). One binary, identical 1024-token coding requests and 384 generated tokens per
request. The decode rate times 383 tokens after the first token. Each mode starts
with an empty private disk cache. The resident request retains compiled code
but reuses zero prompt KV. Compilation is included. All requests have zero host
fallbacks. The GPU is R9700/gfx1201; target is Qwen3.8-27B Q4 with Q8 draft modules.

Sampling: temperature 0.6, top-k 20, top-p 0.95 and seed 1234. BF16 KV,
FP32 floating accumulation, batch/ubatch 1024 and configured KV capacity 262100.

| Mode | Cold prefill tokens/s | Cold decode tokens/s | Resident prefill tokens/s | Resident decode tokens/s |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 442.18 | 24.14 | 624.10 | 24.73 |
| MTP=3 | 403.95 | 37.96 | 608.19 | 48.62 |
| DFlash2, seven proposals | 417.14 | 31.39 | 616.78 | 42.41 |

This is one pair per mode, not a statistical estimate or long-context Pi replay.
The 600 PP/s threshold is met on these resident requests. The 29 baseline,
49 MTP and 103 DFlash2 TPS targets remain unmet. See the
[final mode comparison](docs/benchmarks/forward-modes-final-2026-09-29.md) for binary
identity, request hashes, acceptance and timing limits.

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
| Earlier versions and measurements | [Release history](docs/RELEASE_HISTORY.md) |

## License

LSE uses the [MIT license](LICENSE.md).
