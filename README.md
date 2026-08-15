# Lemon Seed Engine (LSE)

A modular C++ training and inference engine for the
[lemonseed](https://github.com/Geramy/lemonseed) hybrid LLM architecture
(Gated DeltaNet + gated GQA + sparse MoE with Mixture-of-Depths), running on the
[HRX](https://github.com/ROCm/hrx-system) native runtime.

See **[PLAN.md](PLAN.md)** for the full design and milestones.

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

## Layout

```
include/lse/     public headers — core, quant, backend, graph, model,
                 tokenizer, runtime, train, dist, server
src/backends/    one directory per backend (hrx, cpu), self-registering
third_party/     fastokens FFI crate + vendored single-header deps
reference/       upstream repos, read-only (gitignored)
tests/           dependency-free harness
```

## Status

Phase 0: foundation builds, 4 test suites green. See PLAN.md §11 for milestones.
