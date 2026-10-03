#!/usr/bin/env bash
# Builds libLSE for iOS/iPadOS arm64 devices and packages it as
# build/ios/LSE.xcframework: one static library carrying the engine, HRX,
# loomc, the tokenizer and the mac_linuxgpu HSA runtime, plus include/lse/lse.h
# and a module map, so an app imports it from Swift as `import LSE`.
#
# This compiles and links on the Mac only; nothing is signed or installed.
#
# Steps:
#   1. the HSA runtime as a static archive (make hsa-ios in mac_linuxgpu)
#   2. HRX and loomc as static archives (scripts/ios/build-hrx-ios.sh)
#   3. the engine for iOS (CMake; library targets only, no executables)
#   4. one relocatable object prelinked from all of it, exporting only lse_*,
#      so the engine's self-registering modules are kept without -force_load
#      and none of its internals can collide with the app's symbols
#   5. LSE.xcframework
#
# Environment:
#   MAC_LINUXGPU_DIR       mac_linuxgpu checkout providing the HSA runtime
#                          (default ../mac_linuxgpu next to this checkout)
#   HRX_SOURCE, HSA_HEADERS_SOURCE, ...  passed through to build-hrx-ios.sh
#   SKIP_HRX=1             reuse an existing build/ios/hrx-build
#   IOS_DEPLOYMENT_TARGET  minimum iOS version (default 26.0)
#   LSE_IOS_WORK           output root (default build/ios in this checkout)
#   LSE_BUILD_JOBS         parallel jobs (default: all cores)
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
work="${LSE_IOS_WORK:-$root/build/ios}"
mac_linuxgpu="${MAC_LINUXGPU_DIR:-$(cd "$root/.." && pwd)/mac_linuxgpu}"
deployment="${IOS_DEPLOYMENT_TARGET:-26.0}"
jobs="${LSE_BUILD_JOBS:-$(sysctl -n hw.ncpu)}"
die() { echo "$*" >&2; exit 1; }

[[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]] ||
  die 'This build requires an Apple Silicon macOS host.'
for tool in cmake ninja cargo rustup xcrun; do
  command -v "$tool" >/dev/null || die "missing required tool: $tool"
done
rustup target list --installed | grep -qx aarch64-apple-ios ||
  die 'The Rust iOS target is missing: rustup target add aarch64-apple-ios'
[[ -f "$mac_linuxgpu/hsa/CMakeLists.txt" ]] ||
  die "MAC_LINUXGPU_DIR has no hsa/: $mac_linuxgpu"
sdk_version="$(xcrun --sdk iphoneos --show-sdk-version)"
mkdir -p "$work"

echo "== 1/5 HSA runtime (iOS static) from $mac_linuxgpu"
make -C "$mac_linuxgpu" hsa-ios HSA_IOS_DEPLOYMENT_TARGET="$deployment"
hsa_library="$mac_linuxgpu/build/hsa-ios/libhsa-runtime64.a"
[[ -f "$hsa_library" ]] || die "make hsa-ios produced no $hsa_library"

echo "== 2/5 HRX and loomc (iOS static)"
hrx_build="$work/hrx-build"
if [[ "${SKIP_HRX:-0}" != 1 || ! -f "$hrx_build/hrx-ios-libs.txt" ]]; then
  HSA_IOS_LIBRARY="$hsa_library" IOS_DEPLOYMENT_TARGET="$deployment" \
    LSE_IOS_WORK="$work" LSE_BUILD_JOBS="$jobs" \
    bash "$root/scripts/ios/build-hrx-ios.sh"
fi
hrx_libs=()
while IFS= read -r line; do [[ -n "$line" ]] && hrx_libs+=("$line"); done < "$hrx_build/hrx-ios-libs.txt"
includes=()
while IFS= read -r line; do [[ -n "$line" ]] && includes+=("$line"); done < "$hrx_build/hrx-ios-includes.txt"
hrx_include="${includes[0]}"
loomc_include="${includes[1]}"
hrx_archive="$hrx_build/ios-libs/libhrx_ios.a"
loomc_archive="$hrx_build/ios-libs/libloomc_ios.a"

echo "== 3/5 engine (iOS, library only)"
lse_build="$work/lse-build"
# Device power (low power on background, device loss after sleep) when the
# HSA runtime has it.
hsa_power=OFF
if nm -g "$hsa_library" 2>/dev/null | grep -q ' T _mac_hsa_agent_prepare_low_power$'; then
  hsa_power=ON
fi
echo "   HSA device power: $hsa_power"
cmake -S "$root" -B "$lse_build" -G Ninja \
  -DCMAKE_SYSTEM_NAME=iOS -DCMAKE_OSX_SYSROOT=iphoneos \
  -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_OSX_DEPLOYMENT_TARGET="$deployment" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLSE_BUILD_TESTS=OFF -DLSE_ENABLE_CPU=ON -DLSE_ENABLE_HRX=ON \
  -DLSE_GPU_TARGETS=gfx1201 \
  -DLSE_HRX_INCLUDE_DIR="$hrx_include" -DLSE_HRX_LIBRARY="$hrx_archive" \
  -DLSE_LOOMC_INCLUDE_DIR="$loomc_include" -DLSE_LOOMC_LIBRARY="$loomc_archive" \
  -DLSE_HSA_POWER="$hsa_power" \
  > "$work/lse-configure.log"
cmake --build "$lse_build" --target lse_api --parallel "$jobs" > "$work/lse-build.log" ||
  { tail -40 "$work/lse-build.log"; die "engine build failed (see $work/lse-build.log)"; }

# The engine's archives: every static library the lse_api target links,
# which CMake knows; read it from the generated link closure rather than
# keeping a second list here.
engine_libs=()
while IFS= read -r lib; do engine_libs+=("$lib"); done < <(
  find "$lse_build" -name 'liblse_*.a' -o -name 'libLSE.a' | sort)
fastokens="$root/third_party/fastokens-ffi/target/aarch64-apple-ios/release/libfastokens_ffi.a"
[[ -f "$fastokens" ]] || die "missing $fastokens"

echo "== 4/5 prelink"
out="$work/xcframework-input"
rm -rf "$out"; mkdir -p "$out/headers"
exports="$out/exports.txt"
# Exactly the C API: every function the header declares.
sed -n 's/^LSE_API [^(]*[ *]\(lse_[a-z_]*\)(.*/_\1/p' "$root/include/lse/lse.h" | sort -u > "$exports"
[[ "$(wc -l < "$exports")" -eq "$(grep -c '^LSE_API' "$root/include/lse/lse.h")" ]] ||
  die "could not read every declaration of the API from lse.h"
force=()
for lib in "${engine_libs[@]}"; do force+=(-force_load "$lib"); done
# -force_load keeps every engine module (backends, kernels and model
# architectures register from translation units nothing references); HRX,
# loomc, HSA and the tokenizer contribute only what the engine reaches.
xcrun -sdk iphoneos ld -r -arch arm64 \
  -platform_version ios "$deployment" "$sdk_version" \
  "${force[@]}" "$fastokens" "${hrx_libs[@]}" \
  -exported_symbols_list "$exports" \
  -o "$out/LSE.o"
xcrun -sdk iphoneos libtool -static -o "$out/libLSE.a" "$out/LSE.o"

cp "$root/include/lse/lse.h" "$out/headers/lse.h"
# The libraries the prelinked object still needs from the system: the C++
# runtime, IOKit/CoreFoundation for the driver transport, iconv for the
# tokenizer. Autolinked by any target that imports the module.
cat > "$out/headers/module.modulemap" <<'EOF'
module LSE {
  header "lse.h"
  link "c++"
  link "iconv"
  link framework "IOKit"
  link framework "CoreFoundation"
  export *
}
EOF

echo "== 5/5 LSE.xcframework"
rm -rf "$work/LSE.xcframework"
xcodebuild -create-xcframework -library "$out/libLSE.a" -headers "$out/headers" \
  -output "$work/LSE.xcframework" > "$work/xcframework.log"

# Link check: an iOS executable that calls the API must link with nothing but
# the system libraries the module map names, and must not import dlopen.
probe="$out/probe"
mkdir -p "$probe"
cat > "$probe/main.c" <<'EOF'
#include "lse.h"
int main(void) {
  lse_config config;
  lse_config_init(&config);
  char *err = 0;
  lse_engine *engine = lse_open(&config, &err);
  if (engine) lse_close(engine);
  lse_free(err);
  return (int)lse_abi_version() - 1;
}
EOF
xcrun -sdk iphoneos clang -target "arm64-apple-ios$deployment" -I"$out/headers" \
  -Wl,-dead_strip "$probe/main.c" "$out/libLSE.a" \
  -lc++ -liconv -framework IOKit -framework CoreFoundation -o "$probe/lse_link_check"
# CPU JIT is not allowed on iOS: the engine must never ask for it. (The
# prelinked object cannot be dead-stripped inside, so IREE's ELF module code
# and its sys_icache_invalidate import ride along; no CPU executable loader is
# built, so nothing can reach them.)
if nm -u "$probe/lse_link_check" | grep -Eq '^_pthread_jit_write_protect_np$'; then
  die "the linked engine imports a JIT entry point"
fi
# dlopen remains only for optional probes that fail cleanly on iOS (the
# RDMA verbs library, the opt-in SQ profiler); the HSA runtime and HRX are
# linked in and never loaded.

echo
echo "LSE.xcframework: $work/LSE.xcframework"
ls -la "$out/libLSE.a" "$probe/lse_link_check"
otool -L "$probe/lse_link_check" | tail -n +2
