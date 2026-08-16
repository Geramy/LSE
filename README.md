# Lemon Seed Engine (LSE)

A modular C++ training and inference engine for the
[lemonseed](https://github.com/Geramy/lemonseed) hybrid LLM architecture
(Gated DeltaNet + gated GQA + sparse MoE with Mixture-of-Depths), running on the
[HRX](https://github.com/ROCm/hrx-system) native runtime.

## Design in one paragraph

Ops are lazy: they record into a DAG and execute only when a host-visible read
demands a value. On demand the graph is partitioned into fusion groups, each
group is emitted as HIP source, compiled with `amd_comgr` into an AMDGPU code
object, cached on disk, and dispatched through the native HRX ABI
(`hrx_stream_dispatch`) — not through HIP. Every extension seam (backend,
transport, quantization scheme, layer, sampler) is a CRTP base that owns the
shared algorithms and calls into the derived type for the primitives.

## Build

```bash
cmake -S . -B build -GNinja -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build
ctest --test-dir build --output-on-failure
```

The core library and CPU backend build with no GPU and no external packages.

To enable the HRX backend, build `hrx-system` and point at its install:

```bash
cmake -S . -B build -GNinja -DLSE_HRX_ROOT=/path/to/hrx-install
```

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

**Measured** — 102.4 tok/s median decode on one gfx1151 APU (Strix Halo,
227–233 GB/s), 16 test suites green, zero warnings under the full warning set.

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
- **Multi-machine** — a control plane that ships IR rather than code objects, so
  each peer compiles for its own architecture.
- **Runtime-adaptive optimization** — variant tournaments judged on measured
  time, with what was learned persisted so the next process starts tuned.

## License

Non-commercial use is free for everyone; commercial use requires a separate
license, with exceptions for the organizations listed in Exhibit A. See
**[LICENSE.md](LICENSE.md)**.
