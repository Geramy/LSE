# PART 0 — OP FEASIBILITY GATE: fp8 WMMA on gfx1201

**Verdict: (a) OP EXISTS AND IS EMISSABLE — PROCEED to PART 1.**

The gfx1201 ISA has a packed-8-bit WMMA instruction with f32 accumulate,
loomc's AMDGPU model models it (contract + fragment layout + scalar fp8
conversion + descriptor), and the LSE tree already carries the dialect row,
the HIP builtin spelling, and the OCP e4m3 pack/decode helpers (upstream 0.4.1
commit 42374ae, now on master 5e6ad94 and in the candidate tree). No shared
`build/hrx-macos-source/loom` edit is required — LSE's Loom-source splicing
path reaches the op through the existing `vector.mma` spelling, which the
0.4.1 tree already generates for this exact row.

---

## 1. The instruction

**gfx1201 (RDNA4) `v_wmma_f32_16x16x16_fp8_fp8`** — 16x16x16 tile, wave32,
**f32 accumulator** (the accumulate type is f32; cite below), 8-bit packed
operands (two fp8 per i32 register per lane). The mixed bf8/fp8 forms
(`v_wmma_f32_16x16x16_bf8_fp8`, `fp8_bf8`, `bf8_bf8`) also exist.

There is NO `v_mma_f32_16x16x32_f8f6f4`-style selector instruction on gfx1201:
the selector-driven `v_wmma_f32_16x16x128_f8f6f4` (f8/f6/f4 operand-width
selectors, K=128) is **gfx1250-only** (see §2.3). For the R9700 (gfx1201)
the relevant instruction is the packed8 16x16x16 WMMA.

## 2. Citations — loomc AMDGPU model (`build/hrx-macos-source/loom`)

### 2.1 Feature profile: gfx1201 gets the fp8 WMMA contract
- `py/loom/target/arch/amdgpu/target_info.py:1126` — `rdna4_processor_info()`
  sets `matrix_feature_profile=AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX12` for
  every rdna4 processor, and `target_info.py:1425` registers
  `rdna4_processor_info("gfx1201", 0x04E)`.
- `target_info.py:116-118` — the `wmma_gfx12` profile features are
  `("wmma_gfx12", "swmmac_gfx12")` — a single-generation profile shared by
  gfx1200/gfx1201 (the gfx1250-specific profiles at lines 120-126 carry the
  `wmma_gfx1250` / `wmma_gfx1250_scale_f8f6f4` features instead).
- The descriptor set for gfx1201 is `amdgpu.rdna4.core`
  (`target_info.py:1110` default key; `target_info.py:1255-1263` flags it with
  `AMDGPU_DESCRIPTOR_SET_INFO_FLAGS_RDNA4_VOPD_NONCANONICAL_FP8` — i.e. the
  rdna4 set is where the VOPD fp8 encoding lives, in contrast to the
  gfx1250 sets which carry `VOPD_PACKED_BF16`).

### 2.2 Contracts: the fp8 WMMA rows are `wmma_gfx12`-gated
`py/loom/target/arch/amdgpu/matrix_contracts.py` (all four mixed rows
`wmma.f32.16x16x16.{fp8,bf8}.{fp8,bf8}` sit at ~lines 1969-2024):

- `wmma.f32.16x16x16.fp8.fp8` — `features=("wmma_gfx12",)`,
  `tile_shape=(16,16,16)`, lhs/rhs `payload("fp8", 2, 8)` (2 i32 registers,
  8 elements per lane), **accumulator `payload("f32", 8, 8)`**,
  `result payload("f32", 8, 8)` — **f32 accumulate, cited here.**
  `fragment_layout="rdna4_wmma_f32_16x16x16_packed8"`.
- Same row shape for `.bf8.bf8`, `.bf8.fp8`, `.fp8.bf8`.

The CDNA4 f8f6f4 selector MFMAs in the same file
(`matrix.mfma.f32.16x16x128.f8f6f4`, lines ~930-953) are `mfma_gfx950`-gated —
not a gfx12 path.

### 2.3 Fragment layout
- `src/loom/target/arch/amdgpu/matrix/types.h:241-242` —
  `LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_RDNA4_WMMA_F32_16X16X16_PACKED8 = 22`,
  "RDNA4 WMMA 16x16x16 packed fp8/bf8 input, f32 accumulator/result layout."
- `py/loom/target/arch/amdgpu/matrix_fragment_layouts.py:945` —
  `("rdna4_wmma_f32_16x16x16_packed8", 16, 8, 8)`: K=16 reduction, 8 source
  elements / 8-bit per lane. The 64/128-K packed8 layouts at lines 946-947 are
  the `gfx125x_wmma_f32_16x16x{64,128}` rows, i.e. the gfx1250 generation.

### 2.4 Descriptors (python overlay layer)
- `py/loom/target/arch/amdgpu/descriptors/matrix.py:261-277` —
  `_v_wmma_f32_16x16x16_packed8_overlay` builds
  `amdgpu.v_wmma_f32_16x16x16_fp8_fp8` / `_bf8_bf8` / mixed keys with
  `input_units=2, accumulator_units=8` (i.e. `V_WMMA_F32_16X16X16_FP8_FP8`
  VOPD, 8 f32 VGPR result).
- The gfx1250 f8f6f4 WMMA is a different instruction:
  `py/loom/target/arch/amdgpu/descriptors/rdna4.py:531-583`
  (`v_wmma_f32_16x16x128_f8f6f4`, matrix_a_fmt/matrix_b_fmt selectors,
  K=128), registered only in the `gfx125x` descriptor sets
  (`rdna4.py:1109-1112`). `descriptors/rdna4_test.py:177` tests it against
  the gfx125x set, not gfx1201.

### 2.5 Scalar fp8 conversion exists in loomc
- `src/loom/target/arch/amdgpu/lower/narrow_float/fp8.c` + `fp8_packed.c`,
  `fp8_lane.c`, `fp8_plan.c`, `fp8_descriptors.c`,
  `fp8_vector_conversion.c` — the f8e4m3/f8e5m2 scalar pack/convert
  lowering (e4m3 saturation/RNE semantics in `vector_conversion.c:479ff`).

## 3. Citations — LSE tree (candidate `cand-fp8-mma/lse-source`)

### 3.1 The matrix-core table already has the row
`include/lse/math.hpp`:
- `557-577` — the RDNA4 packed8 rows:
  `wmma12.f32.16x16x16.fp8_fp8` (kRdna4, kF32 acc, kFp8 operand,
  `Scalar::kI32, 2` registers × pack 4, `c = f32×8`, cap `kWmma12Fp8`,
  throughput 400, `OperandLayout::kLaneRowSplitK`,
  `AccLayout::kRowBlockHalfWave`, **`emittable()==true`**), the chained
  2-instruction `k_step=32` form (kUnmeasured — we do not use it), and the
  `bf8_bf8` row.
- `378` — `MatrixCap::kWmma12Fp8 = 1u << 8`.
- `531` — `device_matrix_caps` for kRdna4 includes
  `cap_bits(MatrixCap::kWmma12Fp8)` unconditionally (no device flag needed).
- `466-470` — `emittable()`: the fp8_fp8 row is NOT marked kUnmeasured, so it
  emits. (Honest caveat: the table comment at line 520 says the RDNA4 fp8
  layout was hypothesized from the su8 ISA-calculator derivation; the
  iu8/bf16 split-K + row-block-half-wave layouts are the ones device-verified
  on gfx1201 by cand-ffn-wmma2. The fp8 row reuses exactly those two layouts
  — same 8-bit-packed operand geometry, same accumulator. The GPU probe arm
  in RUN.md is the device confirmation, but the *emission* gate passes on
  table state today.)

### 3.2 The Loom spelling exists for the row
`src/backends/hrx/loomc/loom_sources.cpp:340-380` — `matrix_spellings()`
emits, for every emittable kRdna4 16x16x16 row, the `vector.mma` Loom source
with `encoding.define #matrix_operand<element_format=f8e4m3, payload_elements=8,
payload_registers=2>` (the `kFp8` case at line ~349: `a_format = b_format =
"f8e4m3"`) and the fragment/mma splice at line 373. **So the 0.4.1 tree
already generates a gfx1201-emittable Loom source for
`wmma12.f32.16x16x16.fp8_fp8`.**

### 3.3 The HIP builtin spelling exists
`src/backends/hrx/hipc/hip_sources.cpp:139-142`:
`{"wmma12.f32.16x16x16.fp8_fp8",
 "__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12($0, $1, $2)"}`
and the `bf8_bf8` row. (OCP helpers at lines 162-172: `pack4.fp8.ocp`,
`value.fp8.0..3`, `pack4.bf8.ocp`, `value.bf8.0..3` — from upstream 42374ae.)

### 3.4 The OCP FP8 shared helpers (upstream 42374ae, on master 5e6ad94)
`include/lse/math/fp8.hpp` — `Fp8Format<kFp8>` (e4m3: 3-bit mantissa, bias 7,
max finite 448), `fp8_value`, `fp8_bits` (host RNE-even oracle, exact-dyadic
tiebreak), `pack_fp8`, `unpack_fp8` + the device
`pack4.fp8.ocp`/`value.fp8.N` intrinsics. `docs/FP8_CONVERSION.md`,
`tests/test_fp8_conversion.cpp` (+ native arm) cover the conversion contract:
SATFINITE clamp to ±448, RNE, e4m3 overflow → NaN on device.

### 3.5 Does the LSE tree emit the fp8 WMMA anywhere today?
No kernel emits it yet — the OCP FP8 commit adds the *conversion* surface
(pack/decode), not a GEMM. This candidate is the first emitter of
`wmma12.f32.16x16x16.fp8_fp8` in LSE.

## 4. What is missing (work for PART 1, all candidate-tree local)

1. `src/kernels/ffn_fp8_q6.cpp` — the `quant_linear.q6_fp8_mma_prefill`
   kernel (new file; modeled on `ffn_wmma2_q6.cpp`).
2. `include/lse/kernels/ffn_fp8_q6.hpp` — selector declaration.
3. `src/kernels/quant_linear.cpp` — one routing hook
   (`ffn_fp8_q6_for(s)` first in the chain) + include.
4. `src/backends/hrx/loomc/loom_emitter.cpp` — latch
   `LSE_FFN_FP8_MMA` into the emission identity (one array element).
5. `ffn_check.cpp` — CPU check suite (candidate dir, not in LSE).
6. **No edits to `build/hrx-macos-source/loom` (shared) and no edits to
   `include/lse/math.hpp`'s table are needed**: the row, the spelling, the
   conversion helpers, and the capability all exist. The only shared-tree
   surface in play is the *pinned* `build/lse-macos-adapter` archive, which
   this candidate links against unmodified.

## 5. Serialization-rule check (int-mma worker)

Checked before writing anything to shared loomc:
- `find build/hrx-macos-source \( -name '*.c' -o -name '*.h' -o -name '*.cpp'
  -o -name '*.py' \) -newermt '10 minutes ago'` → **0 files** (15:33 PDT).
- `cand-int-mma/` contains only its own `lse-source` + `part0-ops.md`
  (15:25-15:30); its emitter work is in its own candidate tree
  (`cand-int-mma/lse-source`), and its PART 0 verdict is already written.
- The i8 rows in `math.hpp`/`hip_sources.cpp`/`loom_sources.cpp` predate this
  session (visible in the cand-ffn-f32 base copy).
**No shared-loomc edit is mid-flight, and none is needed for fp8.**

## 6. Accumulate-type citation (for the report)

The f8f6f4/packed8 WMMA on gfx1201 accumulates in **f32**: loomc contract
`wmma.f32.16x16x16.fp8.fp8` accumulator payload `("f32", 8, 8)`
(`matrix_contracts.py` ~line 1976); LSE row `wmma12.f32.16x16x16.fp8_fp8`
has `c_elem = Scalar::kF32, c_len = 8` (`math.hpp:564-568`). There is no
narrow-accumulate fp8 form on the gfx12 WMMA (the f16/bf16 narrow forms on
RDNA3/wave64 are a different generation). Cross-K accumulation therefore
stays in f32 registers exactly as in the bf16 record kernel.
