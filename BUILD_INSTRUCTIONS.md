# Building Lemon Seed Engine

Two first-class targets (plus an in-process iOS/iPadOS library, below): **Linux + ROCm** (the original, full path) and
**macOS + Apple Silicon** (via the [mac_linuxgpu](https://github.com/lemonade-sdk/mac_linuxgpu)
driver and HSA runtime). Both produce the same `lse` / `lse-server`
binaries and the same in-process library, libLSE; the difference is the GPU
runtime and the kernel AOT target.

The core library and CPU backend build on either platform with **no GPU and no
external packages**. The HRX (GPU) backend is what needs the platform-specific
runtime below.

---

## Linux + ROCm

This is the primary build. It needs a C++26 compiler with P2996 reflection
(`g++-16`), CMake 3.24+, Ninja, and (for the GPU backend) a ROCm install.

### Toolchain

```bash
# Ubuntu 25.10 and newer ship g++-16; older needs the GCC 16 PPA or a manual build.
g++-16 --version        # must report 16.x
cmake --version         # 3.24+
ninja --version
```

ROCm lives at `/opt/rocm` by default; point `LSE_ROCM_PATH` elsewhere if yours
does not.

### Build

```bash
cmake -S . -B build -GNinja \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DCMAKE_CXX_COMPILER=g++-16 \
      -DLSE_ROCM_PATH=/opt/rocm \
      -DLSE_GPU_TARGETS="gfx1201;gfx1150;gfx1151;gfx942"
cmake --build build
ctest --test-dir build --output-on-failure
```

`RelWithDebInfo` is the configuration the measured numbers in this repo were
taken on. A `Debug` build is roughly twenty times slower and misleads you about
everything.

### Enabling the HRX (GPU) backend

`hrx-system` is not vendored. Build it, then point at its install:

```bash
# build hrx-system once (per your ROCm) and install it somewhere, e.g. /opt/hrx
cmake -S . -B build -GNinja -DCMAKE_CXX_COMPILER=g++-16 \
      -DLSE_HRX_ROOT=/opt/hrx
```

The build records the `hrx-install` path so its libraries are found without help;
the ROCm runtime is located at start-up. To pin a particular ROCm:

```bash
export ROCM_PATH=/opt/rocm-6.x
# or
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
```

If the HRX backend cannot initialize, LSE falls back to the CPU backend (the
same models, roughly two hundred times slower) and says so on the way past.
`--no-cpu-fallback` makes that an error instead.

### AOT targets

| Target | Family |
|---|---|
| `gfx942` | CDNA3 (MI300X) |
| `gfx1150`, `gfx1151` | RDNA3.5 (Strix Halo / Ryzen AI) |
| `gfx1200`, `gfx1201` | RDNA4 (Radeon AI PRO R9700) |

Pass the ones you have as `LSE_GPU_TARGETS` (semicolon-separated). The default
is `gfx1151;gfx1201;gfx942`.

---

## macOS + Apple Silicon

macOS has no native AMD GPU driver, so the GPU path runs through
[mac_linuxgpu](https://github.com/lemonade-sdk/mac_linuxgpu): the unmodified
upstream Linux amdgpu and amdkfd driver running as a DriverKit system
extension, which installs the HSA runtime for the HRX/Loom path. It replaces
the earlier MacAMDGPU driver. The build itself
is a normal Apple-Silicon (arm64) host build; it needs LLVM 21.1.8 + LLD 21.1.8
(from Homebrew), CMake, Ninja, and the Rust toolchain.

> **GPU execution requires the mac_linuxgpu driver installed and activated on
> the host.** The build and the host test-suite run with no GPU; only the live
> inference needs the driver. See
> [Building and installing](https://github.com/lemonade-sdk/mac_linuxgpu#building-and-installing)
> in the mac_linuxgpu README for the driver install/activation steps.

### Toolchain

```bash
brew install llvm@21 lld@21 cmake ninja rust
# must be exactly 21.1.8
$(brew --prefix llvm@21)/bin/llvm-config --version    # 21.1.8
$(brew --prefix lld@21)/bin/ld.lld --version | grep 'LLD 21.1.8'
```

### Build

The CI script [`build-macos.sh`](.github/scripts/build-macos.sh) is the
reference — it fetches the pinned HRX dependencies, applies LSE's HRX patch
series from `patches/hrx/`, configures with the Darwin archive tools, and builds both
HRX/Loom and LSE. To run it by hand:

```bash
bash .github/scripts/build-macos.sh
```

Or the equivalent manual configure, with the pinned revisions and the Darwin
`ar`/`ranlib` from the active Xcode (Homebrew's clang would otherwise pull in
its own archive tools, which Darwin `ld` cannot read):

```bash
llvm="$(brew --prefix llvm@21)/bin"
lld="$(brew --prefix lld@21)/bin/ld.lld"
export PATH="$llvm:$(dirname "$lld"):$PATH"
export MACOSX_DEPLOYMENT_TARGET=15.0
darwin_ar="$(xcrun --find ar)"
darwin_ranlib="$(xcrun --find ranlib)"

cmake -S . -B build/macos -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DCMAKE_CXX_COMPILER="$llvm/clang++" \
      -DCMAKE_C_COMPILER_AR="$darwin_ar" -DCMAKE_RANLIB="$darwin_ranlib" \
      -DCMAKE_C_COMPILER_AR="$darwin_ar" -DCMAKE_CXX_COMPILER_AR="$darwin_ar" \
      -DCMAKE_CXX_COMPILER_RANLIB="$darwin_ranlib" \
      -DLSE_HRX_ROOT=<built hrx-install> \
      -DLSE_GPU_TARGETS="gfx1201"
cmake --build build/macos
ctest --test-dir build/macos --output-on-failure
```

The HRX dependency is built the same way (LLVM 21.1.8, the macOS coarse host
adapter flag, `gfx1201` target) — `build-macos.sh` does this and pins the
revisions so a build is reproducible rather than tracking a moving `main`.

### Running on the GPU

With the driver activated and the model present:

```bash
export DYLD_LIBRARY_PATH=<hrx-install>/lib:$DYLD_LIBRARY_PATH   # so libhsa-runtime64 is found
./build/macos/lse-server -m <model-dir> --pool hrx:0
```

Use the `macos-arm64` release asset for Apple Silicon; it bundles HRX and Loom
and loads the HSA runtime the mac_linuxgpu driver installs, so the build above
also needs the driver (or `DYLD_LIBRARY_PATH` pointing at an HSA runtime) to
reach the GPU. Linux release binaries are not macOS builds and vice versa.

---

## libLSE (in-process C API)

Every build with the tokenizer (that is, with Rust/Cargo available) also builds
libLSE, the engine as a library behind the plain C header
[`include/lse/lse.h`](include/lse/lse.h). The CMake target is `lse_api`; the
archive is `libLSE.a`. `lse-server` is a thin `main` over it, so an app that
links it gets exactly the server's behavior without a socket.

```bash
cmake --build build --target lse_api        # -> build/libLSE.a
```

From CMake, link the target, which brings the engine's own libraries with it:

```cmake
set(LSE_BUILD_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory(path/to/LemonSeed-Engine lse EXCLUDE_FROM_ALL)
target_link_libraries(my_app PRIVATE lse::api)   # C or C++; include <lse/lse.h>
```

The API, in order of use:

| Function | Purpose |
|---|---|
| `lse_model_info(model, &json, &err)` | Describe a checkpoint from its config and tensor headers; no device |
| `lse_estimate(&cfg, options, &json, &err)` | Device memory `lse_open(&cfg)` would allocate, by component; no device |
| `lse_config_init(&cfg)` | Fill an `lse_config` with the `lse-server` defaults, one field per flag |
| `lse_open(&cfg, &err)` | Open the devices and load the model; `lse_status(NULL, ...)` reports progress meanwhile |
| `lse_request(...)`, `lse_cancel(...)` | OpenAI-shaped JSON requests answered through a callback, streamed or whole |
| `lse_status(engine, &json)` | Load phase, model, request counters, last timings, device bytes held |
| `lse_http_start`, `lse_http_stop`, `lse_http_wait` | Optionally serve the same engine over HTTP |
| `lse_set_log_callback(cb, user)` | Receive the log lines `lse-server` prints |
| `lse_close(engine)`, `lse_free(p)` | Release the engine; release every string the library returned |

Example:

```c
#include <stdio.h>
#include <string.h>
#include <lse/lse.h>

static void on_response(void *user, lse_request_id id, lse_event event,
                        int status, const char *data, size_t len) {
  if (event == LSE_EVENT_CHUNK) printf("%.*s\n", (int)len, data);   /* one SSE chunk */
  if (event == LSE_EVENT_RESPONSE || event == LSE_EVENT_ERROR)
    printf("%d %.*s\n", status, (int)len, data);
}

int main(void) {
  char *info = NULL, *err = NULL;
  if (lse_model_info("/models/qwen38-27b-q4", &info, &err) == LSE_OK) {
    puts(info);                      /* architecture, KV layers, KV bytes per token, ... */
    lse_free(info);
  }

  lse_config cfg;
  lse_config_init(&cfg);             /* the lse-server defaults */
  cfg.model = "/models/qwen38-27b-q4";
  cfg.dflash2 = 1;
  cfg.dflash2_model = "/models/qwen38-27b-dflash2-q8";
  cfg.kv_len = 32768;
  cfg.kv_cache_dtype = "bf16";
  cfg.pool = "hrx:0";                /* cfg.dialect NULL: Loom, the default */

  char *plan = NULL;                 /* what lse_open(&cfg) would allocate */
  if (lse_estimate(&cfg, "{\"device_memory_bytes\": 34359738368}", &plan, &err) == LSE_OK) {
    puts(plan);
    lse_free(plan);
  }

  lse_engine *engine = lse_open(&cfg, &err);   /* blocks until the model is ready */
  if (engine == NULL) { fprintf(stderr, "%s\n", err); lse_free(err); return 1; }

  const char *body =
      "{\"messages\":[{\"role\":\"user\",\"content\":\"Say hello.\"}],"
      "\"max_tokens\":64,\"stream\":true}";
  lse_request_id id;
  lse_request(engine, "POST", "/v1/chat/completions", body, strlen(body),
              on_response, NULL, &id);
  /* ... wait for LSE_EVENT_DONE or LSE_EVENT_ERROR; lse_cancel(engine, id) stops it ... */

  /* Optional: serve the same engine over HTTP as well, until lse_http_stop. */
  if (lse_http_start(engine, "127.0.0.1", 8080, &err) == LSE_OK) lse_http_wait(engine, &err);
  lse_close(engine);
  return 0;
}
```

`lse-server --model-info` and `--estimate` print the two planning
answers from the command line.

---

## iOS / iPadOS (in-process library)

Requirements: an Apple Silicon Mac with Xcode and the iOS SDK, CMake, Ninja,
Rust with the `aarch64-apple-ios` target, and a
[mac_linuxgpu](https://github.com/lemonade-sdk/mac_linuxgpu) checkout (it
provides the HSA runtime the framework links statically).

On an iPad with the mac_linuxgpu driver embedded in the app, the engine runs
inside the app: there is no executable and no subprocess. The app links
`LSE.xcframework` (libLSE plus HRX, loomc, the tokenizer and the static HSA
runtime) and talks to the engine through `include/lse/lse.h`: `lse_open`
loads the model, `lse_request` takes the same JSON requests the HTTP server
does, and `lse_http_start` optionally serves the OpenAI-compatible HTTP API
from the same engine.

```bash
rustup target add aarch64-apple-ios
MAC_LINUXGPU_DIR=<mac_linuxgpu checkout> bash scripts/ios/build-ios.sh
# -> build/ios/LSE.xcframework

# Later builds can reuse the HRX/loomc archives from the first one:
SKIP_HRX=1 MAC_LINUXGPU_DIR=<mac_linuxgpu checkout> bash scripts/ios/build-ios.sh

# The release asset is the framework zipped with its top-level directory:
ditto -c -k --keepParent build/ios/LSE.xcframework lse-v0.5.0-ios-arm64.xcframework.zip
```

`IOS_DEPLOYMENT_TARGET` (default 26.0), `LSE_IOS_WORK` (default `build/ios`)
and `LSE_BUILD_JOBS` adjust the build. In Xcode, add `LSE.xcframework` to the
app target (Do Not Embed: it is a static library) and `import LSE` from Swift.

The script builds the HSA runtime (`make hsa-ios` in mac_linuxgpu, with its
output in `build/ios/hsa`, so the checkout is only read), HRX and
loomc as static archives (`scripts/ios/build-hrx-ios.sh`, which fetches
hrx-system at the same pin as the macOS build and applies the same
`patches/hrx/` series plus `ios-static-runtime.patch` to a copy of it), the
engine with `CMAKE_SYSTEM_NAME=iOS`, then prelinks them into one object that
exports only the `lse_*` API. Kernels are generated on the device by loomc as
GPU code objects (data); nothing is compiled for the CPU at run time. The
kernel cache defaults to the app's `Library/Caches/lse/kernels`.

---

## Options (both platforms)

| Option | Default | Purpose |
|---|---|---|
| `LSE_ENABLE_HRX` | ON | Build the HRX (GPU) backend |
| `LSE_ENABLE_CPU` | ON | Build the CPU reference backend |
| `LSE_BUILD_TESTS` | ON | Build the test suite |
| `LSE_GPU_TARGETS` | `gfx1151;gfx1201;gfx942` | AOT kernel targets |
| `LSE_ROCM_PATH` | `/opt/rocm` | ROCm root (Linux) |
| `LSE_HRX_ROOT` | — | Path to a built `hrx-install` (enables GPU) |
| `LSE_WERROR` | OFF | Warnings as errors |
| `LSE_ASAN` | OFF | AddressSanitizer + UBSan |

## Checking memory estimates against real checkpoints

`test_model_info` checks model info and memory estimates against fixture
checkpoints and against what a load on the host backend allocates. A gated
companion repeats the comparison for real checkpoints; it copies the whole
model into RAM, so it runs only when pointed at one:

```bash
LSE_BACKEND=cpu \
LSE_TEST_MODEL_TARGET=/models/qwen38-27b-q4 \
LSE_TEST_MODEL_DFLASH2=/models/qwen38-27b-dflash2-q8 \
  build/tests/test_model_info_real
```

Set `LSE_TEST_MODEL_MTP` instead of `LSE_TEST_MODEL_DFLASH2` to check an MTP
module. Activation and workspace need a forward pass on a GPU and are not part
of this check.

## Release builds

Releases are built and published by GitHub Actions — see the workflows in
[`.github/workflows/`](.github/workflows/): `build-release.yml` (Linux/ROCm, all
selected GPU families) and `macos.yml` (macOS arm64). Dispatch either with
`release=true` and a tag to attach the artifacts to a GitHub release.
