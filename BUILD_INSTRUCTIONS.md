# Building Lemon Seed Engine

Two first-class targets: **Linux + ROCm** (the original, full path) and
**macOS + Apple Silicon** (via the [MacAMDGPU](https://github.com/lemonade-sdk/mac-amdgpu)
DriverKit driver and HSA runtime). Both produce the same `lse` / `lse-server`
binaries; the difference is the GPU runtime and the kernel AOT target.

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
[MacAMDGPU](https://github.com/lemonade-sdk/mac-amdgpu): a DriverKit system
extension that exposes an HSA runtime for the HRX/Loom path. The build itself
is a normal Apple-Silicon (arm64) host build; it needs LLVM 21.1.8 + LLD 21.1.8
(from Homebrew), CMake, Ninja, and the Rust toolchain.

> **GPU execution requires the MacAMDGPU driver installed and activated on the
> host.** The build and the host test-suite run with no GPU; only the live
> inference needs the driver. See the
> [MacAMDGPU quickstart](https://github.com/lemonade-sdk/mac-amdgpu/blob/main/docs/LSE_QUICKSTART.md)
> for the driver install/activation steps.

### Toolchain

```bash
brew install llvm@21 lld@21 cmake ninja rust
# must be exactly 21.1.8
$(brew --prefix llvm@21)/bin/llvm-config --version    # 21.1.8
$(brew --prefix lld@21)/bin/ld.lld --version | grep 'LLD 21.1.8'
```

### Build

The CI script [`build-macos.sh`](.github/scripts/build-macos.sh) is the
reference — it fetches the pinned HRX/mac-amdgpu deps, applies the macOS
portability patches, configures with the Darwin archive tools, and builds both
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
./build/macos/lse-server -m <model-dir> --pool hrx:0 --dialect loom
```

Use the `macos-arm64` release asset for Apple Silicon; it bundles the HSA/HRX/Loom
runtime. Linux release binaries are not macOS builds and vice versa.

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

## Release builds

Releases are built and published by GitHub Actions — see the workflows in
[`.github/workflows/`](.github/workflows/): `build-release.yml` (Linux/ROCm, all
selected GPU families) and `macos.yml` (macOS arm64). Dispatch either with
`release=true` and a tag to attach the artifacts to a GitHub release.
