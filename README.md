# Lemon Seed Engine (LSE)

A modular C++ training and inference engine for the
[lemonseed](https://github.com/Geramy/lemonseed) hybrid LLM architecture
(Gated DeltaNet + gated GQA + sparse MoE with Mixture-of-Depths), running on the
[HRX](https://github.com/ROCm/hrx-system) native runtime. Weights:
[lemonseed-1.5b-base](https://huggingface.co/lemonade-sdk/lemonseed-1.5b-base).

## Design in one paragraph

Ops are lazy: they record into a DAG and execute only when a host-visible read
demands a value. On demand the graph is partitioned into fusion groups, each
group is emitted as HIP source, compiled with `amd_comgr` into an AMDGPU code
object, cached on disk, and dispatched through the native HRX ABI
(`hrx_stream_dispatch`) — not through HIP. Every extension seam (backend,
transport, quantization scheme, layer, sampler) is a CRTP base that owns the
shared algorithms and calls into the derived type for the primitives.

## Install a release

Releases are built by the **Build & Release** workflow in the Actions tab and
carry ahead-of-time kernels for the architectures selected for that build,
which the release notes list.

```bash
# Pick the asset from https://github.com/Geramy/LSE/releases
curl -LO https://github.com/Geramy/LSE/releases/download/<tag>/lse-<tag>-linux-x86_64.tar.gz
curl -LO https://github.com/Geramy/LSE/releases/download/<tag>/lse-<tag>-linux-x86_64.tar.gz.sha256
sha256sum -c lse-<tag>-linux-x86_64.tar.gz.sha256

tar -xzf lse-<tag>-linux-x86_64.tar.gz
cd lse-<tag>-linux-x86_64
```

The binary loads the ROCm and HRX runtimes at start-up, so both have to be on
the library path:

```bash
export LD_LIBRARY_PATH=/opt/rocm/lib:/path/to/hrx-install/lib:$LD_LIBRARY_PATH
./lse --devices        # what this build can see, and what it will not answer
```

`--devices` is the first thing to run: if it reports no HRX device, the library
path is wrong and the engine will fall back to the CPU backend, which runs the
same models far slower rather than failing.

## Run

```bash
# a repo id, a checkpoint directory, or a .safetensors file
./lse -m mlx-community/Qwen3.8-27B-4bit -n 256 "The capital of France is"

./lse --list-cache                 # models in the HF cache, and whether this build loads each
./lse -m <model> --stats           # timings, launch counts and the device cost model
./lse -m <model> -b 4 -p "..." -p "..."   # decode several sequences as one batch
```

Speculative decoding is on whenever the checkpoint ships a multi-token
prediction module, or point at one:

```bash
./lse -m mlx-community/Qwen3.8-27B-4bit --mtp <path-or-repo-id> -n 256 "..."
```

### `lse` options

| Option | Default | |
|---|---|---|
| `-m, --model NAME` | `$LSE_MODEL` | Checkpoint directory, `.safetensors`, or an HF repo id. A bare name resolves when it is unique |
| `-n, --max-tokens N` | 256 | Tokens to generate |
| `-t, --temperature F` | 0.8 | 0 or less is greedy |
| `--top-k N` | off | Keep the N most likely tokens |
| `--top-p F` | 1.0 | Nucleus threshold |
| `--repeat-penalty F` | 1.0 | Above 1 discourages repeats |
| `-s, --seed N` | 0 | Sampler seed |
| `--mtp PATH` | beside the model | Multi-token-prediction module for speculative decoding |
| `--arch NAME` | detected | Force a model kernel instead of detecting one |
| `--tokenizer REPO` | `Qwen/Qwen3.6-27B` | HF repo for `tokenizer.json`, used only when the model directory has none |
| `--kv-len N` | `max(2*train_seq, 2048)` | Allocate the KV cache for N tokens and keep that shape |
| `-b, --batch N` | 1 | Decode N copies of the prompt as one batch |
| `-p, --prompt TEXT` | | One more sequence for the batch; repeatable, and differing lengths put the rows at different positions |
| `--kv-blocks N` | no limit | Blocks one attention layer's pool may hold; below what the batch needs, sequences are preempted and resume |
| `--pool LIST` | `$LSE_POOL` | Devices this run may use, backend-qualified and best first: `hrx:0,cpu:0` |
| `--dialect NAME` | the device's choice | Source dialect to generate kernels in: `hip` or `loom` |
| `--list-models` | | Print the registered model kernels and exit |
| `--list-cache` | | List the models in the HF cache and whether this build can load each, and exit |
| `--devices` | | Report every device this build can see, and what it will not answer |
| `--stats` | | Print timings, launch counts and the cost model when done |
| `--debug` | | Print the HIP dump path and file count |

## Serve an OpenAI API

`lse-server` answers the OpenAI wire format, so anything that already speaks it
works by changing the base URL.

```bash
./lse-server -m mlx-community/Qwen3.8-27B-4bit --port 8080
```

### `lse-server` options

| Option | Default | |
|---|---|---|
| `-m, --model NAME` | `$LSE_MODEL` | Checkpoint directory, `.safetensors`, or an HF repo id |
| `--host ADDR` | `127.0.0.1` | Address to bind. `0.0.0.0` serves every interface, not just this machine |
| `--port N` | 8080 | Port to bind |
| `--api-key KEY` | none | Require `Authorization: Bearer KEY`. Without it every request is served unauthenticated |
| `--served-name ID` | the model argument | Model id reported by `/v1/models` |
| `--max-tokens N` | 4096 | Refuse requests asking for more |
| `--mtp PATH` | beside the model | Multi-token-prediction module for speculative decoding |
| `--tokenizer REPO` | `Qwen/Qwen3.6-27B` | HF repo for `tokenizer.json` when the model directory has none |
| `--kv-len N` | from the config | Allocate the KV cache for N tokens |

Binding beyond localhost gives anyone who can reach the machine use of the
GPU, so pair `--host 0.0.0.0` with `--api-key`:

```bash
./lse-server -m <model> --host 0.0.0.0 --api-key "$(openssl rand -hex 24)"
```

There is no rate limit and no per-client accounting, and generation is
serialized, so `--max-tokens` is what stops one caller holding the device.

| Endpoint | |
|---|---|
| `GET /health` | liveness |
| `GET /v1/models`, `GET /v1/models/{id}` | the loaded model |
| `POST /v1/chat/completions` | streaming and non-streaming |
| `POST /v1/completions` | streaming and non-streaming |

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Say hello"}],"max_tokens":64}'
```

Set `"stream": true` for server-sent events: an opening chunk carrying the
assistant role, a chunk per delta, a final chunk with `finish_reason` and
`usage`, then `data: [DONE]`.

Honoured: `messages` (string content or the array-of-parts form),
`prompt`, `stream`, `max_tokens`, `max_completion_tokens`, `temperature`,
`top_p`, `top_k`, `seed`, `frequency_penalty`, and `stop` as a string or an
array. `n` must be 1. Prompts are framed as ChatML, which is what the models
this engine targets are trained on, rather than evaluated from the
checkpoint's own Jinja template.

Two places the wire format and this engine disagree, both resolved toward the
wire so a client gets what the API promises:

- `temperature` defaults to **1.0**, as the API specifies, not to the CLI's 0.8.
- `frequency_penalty` is additive in the API and multiplicative here, so it is
  mapped rather than passed through. 0 is off on both sides.

Endpoints outside that set — `/v1/embeddings`, `/v1/responses`,
`/v1/audio/*`, `/v1/images/*`, `/v1/moderations` — answer `501` naming
themselves rather than `404`.

One model on one device, so generation is serialized and concurrent requests
queue. The engine decodes several sequences in one step; putting that behind
the server is future work and does not change the wire format.

## Requirements

| | Version | Why |
|---|---|---|
| **g++** | **16 or newer** | The host sources are C++26 and use P2996 static reflection. Kernel argument structs are reflected over to derive their ABI layout, so the tree does not compile without it. |
| CMake | 3.24 or newer | |
| Ninja | any | Generator used by the commands below |
| cargo | any | Builds the `fastokens` FFI shim the tokenizer links |
| ROCm | 7.x, with `amd_comgr` | HRX backend only. Supplies the compiler the JIT calls at runtime |
| hrx-system | built and installed | HRX backend only |

**clang cannot build this tree.** P2996 reflection is enabled by `-freflection`,
which the build applies only for GNU 16 and newer; on any other compiler the
reflection headers stop with `no member named 'meta' in namespace 'std'`.
CMake picks up `g++-16` from `PATH` on a fresh configure, but it does not
override a compiler already cached in an existing build directory — pass
`-DCMAKE_CXX_COMPILER=g++-16` if you are reconfiguring one, or check
`build/CMakeCache.txt` if the reflection headers fail.

Host code is C++26. Device code stays at C++20 so generated kernels do not
depend on host-only language features.

## Build

```bash
cmake -S . -B build -GNinja -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DCMAKE_CXX_COMPILER=g++-16
cmake --build build
ctest --test-dir build --output-on-failure
```

The core library and CPU backend build with no GPU and no external packages.

To enable the HRX backend, build `hrx-system` and point at its install:

```bash
cmake -S . -B build -GNinja -DCMAKE_CXX_COMPILER=g++-16 \
      -DLSE_HRX_ROOT=/path/to/hrx-install
```

The HRX runtime is loaded at run time, so its libraries have to be findable:

```bash
export LD_LIBRARY_PATH=/opt/rocm/lib:/path/to/hrx-install/lib:$LD_LIBRARY_PATH
```

Without it the HRX backend fails to initialize and the engine falls back to the
CPU backend, which runs the same models roughly two hundred times slower.

### Options

| Option | Default | Purpose |
|---|---|---|
| `LSE_ENABLE_HRX` | ON | Build the HRX backend |
| `LSE_ENABLE_CPU` | ON | Build the CPU reference backend |
| `LSE_BUILD_TESTS` | ON | Build the test suite |
| `LSE_GPU_TARGETS` | `gfx1151;gfx1201;gfx942` | AOT kernel targets |
| `LSE_ROCM_PATH` | `/opt/rocm` | ROCm root (flat or TheRock layout) |
| `LSE_WERROR` | OFF | Warnings as errors |
| `LSE_ASAN` | OFF | AddressSanitizer + UBSan |

## Current

**Compiler**

- Tracing JIT: lazy tensor DAG, fusion-group partitioning, HIP emission,
  `amd_comgr` compilation, disk cache, native-ABI dispatch with no HIP runtime.
- Kernel IR with regions and typed SSA values, a verifier run after every pass,
  and an iteration space whose dimensions carry their kind (parallel, reduction,
  sequential).
- Optimization passes: common subexpression elimination, dead code elimination,
  and LDS folding — which collapses the identical shared-memory stagings of
  fused siblings into one.
- Sibling fusion, retained and replayed programs, and batched command buffers.
- JIT cache keyed on group signature, target architecture, and the compiler's
  own reported identity, so a toolchain change invalidates stale objects.

**Kernels**

- Authored as ordinary C++ against a reflection-based surface — one body is
  either executed on the host or recorded into device source, with argument
  binding derived from struct layout.
- Matrix-core descriptor table covering WMMA and MFMA across RDNA3/3.5, RDNA4
  and CDNA3, keyed by target, accumulator, operand and shape. Adding an operand
  family is a table row.
- Weights are held and moved in the checkpoint's own format; conversion happens
  inside the kernel at the register boundary. bf16 native, with Q8/Q6/Q4 block
  codecs.

**Measurement and distribution**

- Device qualification probe: measured DRAM bandwidth, dispatch cost, and
  matrix-core throughput per operand family, plus per-ordered-pair link latency
  and bandwidth fitted separately. Every number carries its provenance.
- Cost model answering throughput at a given queue depth, with split proportions
  for uneven pools.
- QuickReduce two-shot compressed all-reduce; execution-stream seam; loopback
  transport exercising 2/4/8 ranks on one box.
- CPU reference backend for numerics checking against every device kernel.

**Model**

- Gated DeltaNet, gated GQA with KV cache, sparse MoE (8 experts, top-2),
  Mixture-of-Depths, chunked prefill, and device-side argmax.

**Measured** — [lemonseed-1.5b-base](https://huggingface.co/lemonade-sdk/lemonseed-1.5b-base)
(bf16) at 102.3 tok/s median decode on one gfx1151 APU (Strix Halo,
227–233 GB/s), 17 test suites green, zero warnings under the full warning set.

## Upcoming

- **Multi-device** — device enumeration, buffer residency, and per-group
  placement chosen from the measured cost model rather than configuration.
- **Paged KV cache** with runtime batch extents, replacing contiguous
  per-session state.
- **Continuous batching** across sessions, which hides link latency behind queue
  depth.
- **Device-resident collectives** over a real peer path.
- **Tracing seam** exporting Perfetto traces, with a rocprofiler adapter, so
  engine spans and kernel traces share one timeline and one clock.
- **Iteration-space windows** — fusion and sharding as one mechanism, where the
  level of parallelism (data, expert, pipeline, tensor) is which dimension gets
  split, derived from dimension kind, queue depth and measured link cost.
- **Heterogeneous pools** — a device without fp8 matrix cores runs the fp16
  route for the same operation and stays a full pool member, taking a smaller
  share proportional to its measured throughput.
- **Communication layer** — one asynchronous client/server API over TCP and
  RDMA, with the transport hidden from callers.
- **Batched serving** — the server's requests decoded together in one step
  rather than queued, and `/v1/embeddings` once the engine exposes pooling.
- **Multi-machine** — a control plane that ships IR rather than code objects, so
  each peer compiles for its own architecture.
- **Runtime-adaptive optimization** — variant tournaments judged on measured
  time, with what was learned persisted so the next process starts tuned.

## License

Non-commercial use is free for everyone; commercial use requires a separate
license, with exceptions for the organizations listed in Exhibit A. See
**[LICENSE.md](LICENSE.md)**.
