# Lemon Seed Engine (LSE)

A modular C++ inference engine for hybrid LLMs -- Gated DeltaNet
interleaved with gated GQA, dense or sparse-MoE feed-forward -- running on the
[HRX](https://github.com/ROCm/hrx-system) native runtime on AMD GPUs. Kernels
are generated at run time from the model's own shapes rather than selected from
a library.

## Platforms

LSE builds and runs on **Linux + ROCm** (the primary path) and **macOS +
Apple Silicon** (via the [MacAMDGPU](https://github.com/lemonade-sdk/mac-amdgpu)
DriverKit driver and HSA runtime). The same `lse` / `lse-server` binaries are
produced on both; the difference is the GPU runtime and the AOT kernel target.
Build steps for each are in **[BUILD_INSTRUCTIONS.md](BUILD_INSTRUCTIONS.md)**.

On macOS there is no native AMD GPU driver, so the GPU path runs through
MacAMDGPU (a DriverKit system extension exposing an HSA runtime for the
HRX/Loom path). The [reproduction guide](https://github.com/lemonade-sdk/mac-amdgpu/blob/main/docs/LSE_QUICKSTART.md)
has the driver install + pinned LSE build; GPU execution needs the driver
activated, the build and host tests do not. Use the `macos-arm64` release asset
for Apple Silicon — Linux release binaries are not macOS builds and vice versa.

HIP (`--dialect hip`) and Loom (`--dialect loom`) share packed-Q6
interpretation, tiling, operand selection, and FP8/BF8 conversion. The
optimizer selects accepted kernels from shape, capabilities, accuracy, and
matched timing evidence — no manual kernel selection. See
[operand selection](docs/QUANT_OPERANDS.md) and
[policy](docs/INT8_POLICY.md).

## Reported performance

Medians, warm JIT cache, GPU otherwise idle, best of several runs. `—` = not
measured on that platform. These are LSE results, not llama.cpp-parity claims.

| Platform / GPU | Model | Prefill (tok/s) | Decode (tok/s) |
|---|---|---|---|
| Linux / gfx1151 (Strix Halo, 220 GB/s) | Qwen3.5-0.8B-4bit, 1601 tok | 1703 | — |
| Linux / gfx1151 (Strix Halo, 220 GB/s) | Qwen3.8-27B-4bit, 401 tok | 28.7 | 11.9 (15.0 with MTP) |
| Linux / gfx1151 (Strix Halo, 220 GB/s) | lemonseed-1.5b-base (bf16) | — | 102.3 |
| macOS / gfx1201 (R9700, 1090 GB/s) | Qwen3.8-27B-4bit, 401 tok | 112 | 20.9 (27.9 with MTP) |
| macOS / gfx1201 (R9700, 1090 GB/s) | **Qwen3.8-27B-Q6, 1024 tok** | **229** | **16.3** |
| macOS / gfx1201 (R9700, 1090 GB/s) | Qwen3.8-27B-MLX-6bit, 512/129 | 88.6 | 17.45 |

The **Qwen3.8-27B-Q6 / 1024-token** row is the current gfx1201 result
(v0.4.2): prefill 229 PP/s (median of 229.25 / 227.20 / 225.84) and decode
16.3 TPS (median of 16.30 / 16.22 / 16.18), all device-resident and
compile-free. Prefill is **+51.7%** over the ~151 PP/s at v0.4.1 — the GDN
projection GEMMs (in_proj_qkv, full-attention qkv/v/o) now select the staged
BF16 WMMA path at M=1024 instead of scalar (all four large shapes went
scalar→WMMA at ~3.0×, device total 14053→9656 ms); PPL-neutral (pure selection
change, greedy text identical). Decode is at the measured aggregate-GEMV
bandwidth ceiling (~415 GB/s vs ~549 GB/s needed for 25 tok/s).

Decode on the APU is bandwidth-bound at 82% of the measured DRAM rate; on the
R9700 it is not, which is where the headroom is. 19 test suites green, zero
warnings under the full warning set.

## Models

The architecture is what a checkpoint is loaded as, not its name, so a family
shares one kernel. `--list-models` prints what a build registers.

| Kernel | Checkpoints | Notes |
|---|---|---|
| `qwen3.5` | Qwen3.5, Qwen3.6, Qwen3.8 dense | Hybrid: three Gated DeltaNet layers to each full-attention one. Verified on Qwen3.5-0.8B and Qwen3.8-27B |
| `qwen3.5-moe` | The A3B-style MoE variants of the same families | MLX's SwitchGLU layout, experts stacked as one plane per projection |
| `lemonseed` | [lemonseed-1.5b-base](https://huggingface.co/lemonade-sdk/lemonseed-1.5b-base) | Adds Mixture-of-Depths |

Weights are read in MLX group-affine form at 4, 6 or 8 bits, or bf16/f16/f32. A
multi-token-prediction module is used when the checkpoint has one, which is
what `--mtp` names and `--no-mtp` declines; the text tower loads on its own
where a checkpoint also ships a vision tower, which this build does not run.

## Design in one paragraph

Ops are lazy: they record into a DAG and execute only when a host-visible read
demands a value. On demand the graph is partitioned into fusion groups, each
group is emitted as HIP or Loom source, compiled with the selected toolchain into
an AMDGPU code object, cached on disk, and dispatched through the native HRX ABI
(`hrx_stream_dispatch`) — not through HIP. Every extension seam (backend,
transport, quantization scheme, layer, sampler) is a CRTP base that owns the
shared algorithms and calls into the derived type for the primitives.

## Install a release

Releases are built by the **Build & Release** workflow in the Actions tab and
carry ahead-of-time kernels for the architectures selected for that build,
which the release notes list.

The **macOS ARM64** workflow builds on GitHub-hosted Apple Silicon runners.
It packages `lse`, `lse-server`, HRX, Loom and the MacAMDGPU HSA runtime, checks
host behavior and native kernel compilation, and tests the package after moving
it to a different directory. These hosted checks do not execute AMD GPU kernels.

For **Apple Silicon** (binaries target macOS 15 or later; the current
MacAMDGPU driver requires macOS Tahoe 26.2 or later):

```bash
# Pick the macos-arm64 asset from https://github.com/Geramy/LSE/releases
curl -LO https://github.com/Geramy/LSE/releases/download/<tag>/lse-<tag>-macos-arm64.tar.gz
curl -LO https://github.com/Geramy/LSE/releases/download/<tag>/lse-<tag>-macos-arm64.tar.gz.sha256
shasum -a 256 -c lse-<tag>-macos-arm64.tar.gz.sha256
tar -xzf lse-<tag>-macos-arm64.tar.gz
cd lse-<tag>-macos-arm64
./bin/lse --devices
./bin/lse --pool hrx:0 --dialect loom -m /path/to/model --no-mtp -n 128 "Hello"
./bin/lse-server --pool hrx:0 --dialect loom -m /path/to/model --no-mtp --port 8080
```

Install, approve and initialize the
[MacAMDGPU driver](https://github.com/lemonade-sdk/mac-amdgpu#hardware-requirements)
separately before GPU use. Hardware qualification currently covers the R9700
(`gfx1201`) on Apple Silicon; this is not general support for every AMD GPU.
The archive does not install a system extension. Use its `bin/` launchers to
load the bundled runtime libraries. Binaries are ad-hoc signed, not Developer ID
notarized. The package uses Loom; HIP/comgr is not included on macOS.

For **Linux x86_64**:

```bash
# Pick the asset from https://github.com/Geramy/LSE/releases
curl -LO https://github.com/Geramy/LSE/releases/download/<tag>/lse-<tag>-linux-x86_64.tar.gz
curl -LO https://github.com/Geramy/LSE/releases/download/<tag>/lse-<tag>-linux-x86_64.tar.gz.sha256
sha256sum -c lse-<tag>-linux-x86_64.tar.gz.sha256

tar -xzf lse-<tag>-linux-x86_64.tar.gz
cd lse-<tag>-linux-x86_64
```

The binary finds the ROCm runtime itself, searching `$ROCM_PATH`, `/opt/rocm`
and the versioned installs beside it and taking the first that carries the
symbols the backend needs. A machine with a distro ROCm sitting beside a newer
one needs nothing exported:

```bash
./lse --devices        # what this build can see, and what it will not answer
```

Set `$ROCM_PATH` if the install is somewhere else. `LD_LIBRARY_PATH` still wins
where it is set, which is what you want when pinning a particular runtime.

`--devices` is the first thing to run. If it reports no HRX device the engine
falls back to the CPU backend, which runs the same models far slower rather
than failing, and it will name the backend that declined and why -- a missing
runtime symbol reads very differently from a machine with no GPU in it.

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

### Choose HIP or Loom

Both the CLI and server accept **`--dialect hip`** or **`--dialect loom`**.
The HIP code generator is named `hipc` in the source; its command-line value is
`hip`, not `hipc`. Both paths dispatch through HRX.

| Platform / path | Flags | Compiler and runtime |
|---|---|---|
| Linux, HIP source | `--pool hrx:0 --dialect hip` | HIP code generation and ROCm `amd_comgr`, with HRX |
| Linux, Loom source | `--pool hrx:0 --dialect loom` | Loom compiler and HRX; requires a build that includes Loom |
| Apple Silicon + MacAMDGPU | `--pool hrx:0 --dialect loom` | Native macOS Loom, HRX and MacAMDGPU HSA runtime |

```bash
# macOS AMDGPU inference (also valid for a Loom-enabled Linux build)
./lse --pool hrx:0 --dialect loom -m /path/to/model --no-mtp -n 128 "Hello"
./lse-server --pool hrx:0 --dialect loom -m /path/to/model --no-mtp --port 8080

# Linux HIP-source path
./lse --pool hrx:0 --dialect hip -m /path/to/model --no-mtp -n 128 "Hello"
./lse-server --pool hrx:0 --dialect hip -m /path/to/model --no-mtp --port 8080
```

The macOS package supports Loom; it does not include a macOS HIP compiler.
`--dialect` is a preference among the device's available toolchains. If a device
does not declare the requested dialect, LSE reports that fact and uses its own
choice. Check the startup `generates hip` / `generates loom` line to confirm the
selected path. `--pool hrx:0` selects the first HRX device; omit it for automatic
device selection. Check `--devices` before loading a model.

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
| `--no-mtp` | off | Decode one token per pass, ignoring any MTP module |
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
| `--no-mtp` | off | Decode one token per pass, ignoring any MTP module |
| `--tokenizer REPO` | `Qwen/Qwen3.6-27B` | HF repo for `tokenizer.json` when the model directory has none |
| `--kv-len N` | from the config | Allocate the KV cache for N tokens |
| `--pool LIST` | `$LSE_POOL` | Device selection, for example `hrx:0` |
| `--dialect NAME` | the device's choice | Kernel source dialect: `hip` or `loom` (not `hipc`) |

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
| fastokens | checked out beside the tree | Tokenizer only. A path dependency, see below |

Developed on Ubuntu 25.10 with g++ 16.2, CMake 3.31, Ninja 1.12, cargo 1.93 and
ROCm 7.2.1. Older ROCm 7.x should work; ROCm 6 does not, because the backend
needs `hsa_amd_vmem_address_reserve_align`.

### Installing the toolchain

Ubuntu 25.10 carries g++-16 directly. On 24.04 it comes from the toolchain PPA:

```bash
# Ubuntu 25.10 and newer
sudo apt install g++-16 cmake ninja-build git curl pkg-config

# Ubuntu 24.04
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt update && sudo apt install g++-16 cmake ninja-build git curl pkg-config
```

`cargo` from the distro is usually old enough to matter; rustup is the reliable
route:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
. "$HOME/.cargo/env"
```

ROCm comes from AMD's repository, not the distro's:

```bash
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://repo.radeon.com/rocm/rocm.gpg.key \
  | sudo gpg --dearmor -o /etc/apt/keyrings/amdrocm.gpg
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/amdrocm.gpg] \
https://repo.amd.com/rocm/packages-multi-arch/ubuntu2404 noble main" \
  | sudo tee /etc/apt/sources.list.d/rocm.list
sudo apt update && sudo apt install rocm
sudo usermod -aG render,video "$USER"   # log out and back in
```

Check it took with `rocminfo | grep gfx` -- the name it prints is the target
this engine will compile kernels for.

### The two checkouts that are not in this repo

Both live under `reference/`, which is deliberately not tracked.

**fastokens** is a path dependency of `third_party/fastokens-ffi`, so the
tokenizer does not build without it. **Configure fetches it for you** at the
revision CI pins, into `reference/fastokens`; an existing checkout is left
alone, whatever it is sitting on. Override with `-DLSE_FASTOKENS_REF` or
`-DLSE_FASTOKENS_REPO`, or clone it yourself:

```bash
git clone https://github.com/crusoecloud/fastokens.git reference/fastokens
git -C reference/fastokens checkout 7973014e4f3a6028ac48f305704eacd64d0b4ef6
```

**hrx-system** is the GPU backend, and there is a script for it:

```bash
./scripts/bootstrap-hrx.sh          # ROCM_PATH, LSE_HRX_REPO, LSE_HRX_REF
```

It checks out the revision this tree is tested against rather than whatever
`main` is -- upstream moved 819 commits inside a fortnight -- and applies the
patches in `patches/`, which are fixes we need and have sent upstream. Applying
is idempotent, so a checkout that already carries them, or an upstream that has
taken them, is left alone.

It clones and builds, then prints the `-DLSE_HRX_ROOT` to configure with. This
one is not folded into `configure` on purpose: it is an IREE-derived tree with
its own build driver and it takes tens of minutes, which is not something a
configure step should start on its own. To drive it by hand instead, follow its
`BUILDING.md`:

```bash
git clone https://github.com/ROCm/hrx-system.git reference/hrx-system
cd reference/hrx-system
python dev.py cmake configure -DIREE_HAL_DRIVER_AMDGPU=ON -DIREE_ROCM_PATH=/opt/rocm
python dev.py cmake build
``` Without it the tree still
builds and the tests still pass -- on the CPU backend, which is two orders of
magnitude slower and is not what you want to measure anything on.

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

Full build steps for both platforms are in **[BUILD_INSTRUCTIONS.md](BUILD_INSTRUCTIONS.md)**
(Linux + ROCm and macOS + Apple Silicon). The core library and CPU backend build
on either with no GPU and no external packages; the HRX (GPU) backend needs the
platform runtime. Quick Linux build:

```bash
cmake -S . -B build -GNinja -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DCMAKE_CXX_COMPILER=g++-16
cmake --build build
ctest --test-dir build --output-on-failure
```

`RelWithDebInfo` is what the numbers in this README were taken on; a `Debug`
build is perhaps twenty times slower and will mislead you about everything.
Enable the HRX backend with `-DLSE_HRX_ROOT=/path/to/hrx-install`; without a
working HRX backend the engine falls back to the CPU backend (the same models,
roughly two hundred times slower) and says so on the way past. See
[BUILD_INSTRUCTIONS.md](BUILD_INSTRUCTIONS.md) for the macOS build, the AOT
`LSE_GPU_TARGETS`, and the full option table.

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

Measured throughput per platform and GPU is in the
[**Reported performance**](#reported-performance) table above.

## Upcoming

- **Multi-device execution** — a pool opens, probes and reports every member
  today, and a model still loads onto one of them. Splitting the work across
  them, and choosing which split from the measured cost model rather than from
  configuration, is the open half.
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

MIT — see **[LICENSE.md](LICENSE.md)**.

### macOS decode submission policy

Single-device gfx1201 Loom decode measures submission intervals during ordinary
warm decode steps, excludes JIT/fallback/repartition samples, and retains only
stable improvements over the 16-dispatch baseline. Explicit
`LSE_FLUSH_INTERVAL` overrides selection; `LSE_AUTO_BATCH=0` disables it.
The measured macOS short-context rates use explicit flush64 and 64 µs polling
overrides, KV128 and no MTP. They are HTTP end-to-end rates and are not
equivalent to other engines' benchmark workloads.
See [the macOS measurements](https://github.com/lemonade-sdk/mac-amdgpu/blob/main/docs/LSE_PERFORMANCE.md).
