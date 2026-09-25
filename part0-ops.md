# PART 0 — OP FEASIBILITY GATE: int8 WMMA for the Loom emitter / loomc (gfx1201)

**Verdict: PASS — branch (a). The op exists and is emittable.**
`wmma12.i32.16x16x16.i8` (Loom dialect row key: `wmma12.i32.16x16x16.iu8`,
i.e. i8-operand / i32-accumulator, 16×16×16, wave 32) is:

1. in the Loom IR matrix-core table with **measured (on-device-verified,
   gfx1201) layouts** → `emittable() == true`;
2. spelled by the pinned Loom emitter into `vector.mma` on `vector<2xi32>`
   fragments (6-row spelling table);
3. lowered by the pinned loomc for the `amdgpu.gfx12` descriptor set (which
   covers gfx1200/gfx1201) to the real ISA instruction
   `v_wmma_i32_16x16x16_iu8`.

No Loom IR op addition was needed (option (b) does not apply). This file
records the full inventory with file:line citations, the fp16-exactness
analysis for the record, and the one open item that moves to PART 2
(loomc compile of the int8 `vector.mma` on gfx1201 — the pinned
`source_low_mma.loom-test` corpus pins that lowerings for gfx12 live on
`gfx12-generic`/`gfx1250`, not on the `gfx1200`/`gfx1201` rows; the CPU
suite's loomc gate resolves it on the exact pinned build).

All loomc citations are against `build/hrx-macos-source/loom` (the hrx
macOS source this machine's loomc is built from); all Loom IR / emitter
citations are against `cand-int-mma/lse-source` (cloned from
`cand-ffn-f32/lse-source`).

---

## 1. Loom IR side (the table the kernel asks)

`include/lse/math.hpp` — `detail::kMatrixCore` (the single table every
kernel row lookup goes through):

- line 542: `{"wmma12.i32.16x16x16.iu8", MatrixTarget::kRdna4,
  MatrixElem::kI32, MatrixElem::kI8, Scalar::kI32 /*a*/ a_len=2,
  Scalar::kI32 /*b*/ b_len=2, Scalar::kI32 /*c*/ c_len=8, pack=4,
  m=n=k=16, wave=32, cap=kWmma12Int8, chained=1, k_step=16, throughput=400,
  OperandLayout::kLaneRowSplitK, AccLayout::kRowBlockHalfWave}` — **this is
  the single-instruction row; both layout fields are measured, so
  `emittable()` (defined just above the table, `math.hpp:~436`) is true.**
- line 548: same key, the **chained double-K variant** (a_len=b_len=4,
  chained=2, k_step=32) — `OperandLayout::kUnmeasured`, deliberately not
  emittable.
- line 556: `wmma12.i32.16x16x16.su8` (signed i8 activation × u8 code, the
  llama.cpp mixed-sign form), also `kLaneRowSplitK` +
  `kRowBlockHalfWave`, i.e. emittable. Our kernel uses the plain iu8 row
  with both operands biased unsigned (both sides u8), so no su8 is needed.
- `MatrixCap::kWmma12Int8` (line ~377) is granted for every RDNA4 device by
  `device_matrix_caps` in `include/lse/kernels/wmma.hpp` (line ~95: the
  kRdna4 arm ORs `kWmma12F16 | kWmma12Int8 | kWmma12Int4 | kWmma12Fp8`), so
  the cap gate a selector must pass is device-uniform.

The i8 row's measured layouts are the same `kLaneRowSplitK` operand fill
(lane L holds row L%16, half-wave `hi = L/16` takes k-halves 8..15) and
`kRowBlockHalfWave` accumulator mapping (D[m][n] = v{m%8}{n + 16*(m/8)},
from AMD's matrix instruction calculator for `v_wmma_i32_16x16x16_iu8`,
comment at `math.hpp:417`) that the production 13.7-TFLOPS bf16 row
(`wmma12.f32.16x16x16.bf16`, line 530) uses. The only differences are the
register widths: operands `vector<2xi32>` (two 4-byte-packed int8s per lane
per fragment, vs 8×bf16) and accumulator `vector<8xi32>` (vs 8×f32).

Precedent for on-device verification: the table comment at `math.hpp:522`
states the RDNA4 split-K / block-acc layouts were "VERIFIED ON DEVICE
(gfx1201)" by the generalized tile + device-vs-oracle suite, and
`tests/test_jit.cpp:4816`
(`matrix_core_int8_linear_matches_a_host_integer_reference`) is the int8
A/B against an integer host reference.

## 2. Loom emitter side (Loom source spelling)

`src/backends/hrx/loomc/loom_sources.cpp` — `matrix_spellings()` (line
~330): it walks `math::matrix_core_table()` and emits a spelling for every
row with `target==kRdna4 && emittable() && chained==1 && 16x16x16 &&
wave==32 && kLaneRowSplitK && kRowBlockHalfWave`. Its format switch (line
~352) handles `case math::MatrixElem::kI8: a_format = b_format = "i8"`.
Therefore `wmma12.i32.16x16x16.iu8` is in the table with the spelling

```
$t3 = encoding.define #matrix_operand<element_format=i8, payload_elements=8, payload_registers=2> : encoding<schema>
$t4 = encoding.define #matrix_operand<element_format=i8, payload_elements=8, payload_registers=2> : encoding<schema>
$t5 = vector.fragment<lhs> $0 shape [16,16] using {schema = $t3} : vector<2xi32>
$t6 = vector.fragment<rhs> $1 shape [16,16] using {schema = $t4} : vector<2xi32>
$t7 = vector.fragment<init> $2 shape [16,16] : vector<8xi32>
$r = vector.mma $t5, $t6, $t7 : vector<2xi32>, vector<2xi32>, vector<8xi32>
```

Unit-tested at `tests/test_loom_matrix.cpp:36`
(`loom_matrix_uses_shared_measured_rows_and_declines_other_layouts`): the
test expects **exactly 6** RDNA4 single-instruction emittable rows — f16,
bf16, iu8, su8, fp8_fp8, bf8_bf8 — and iterates the whole table asserting
each present row's spelling exists. `tests/test_loom_matrix.cpp:64`
(`loom_matrix_shared_ir_preserves_vector_results`) prints the i8 row through
the real dialect table and asserts `vector.mma` appears. The int8 row is
therefore **already unit-tested on the emitter side** (no new test needed;
the suite is re-run in PART 2).

Kernel-facing surface: `include/lse/kernels/wmma.hpp` —
`MatrixTile<... kRdna4, kI32, kI8, 16,16,16>` resolves `kRow` to the
measured row, `geometry_of(Tile::kRow)` yields `wave=32, slots=8, frag=2,
halves=2, split_k=true, lane_k=8, slot_step=1, half_rows=8` (the same
consumption indices the bf16 production kernel uses), and `math::mma<Op>`
calls the dialect op by key. This is exactly how
`src/kernels/ffn_wmma2_q6.cpp` (the 13.7-TFLOPS record's emitter path)
consumes `wmma12.f32.16x16x16.bf16`; the int8 kernel is the same skeleton
with `kI32/kI8` and u8 staging.

## 3. loomc side (target ISA tables, pinned build)

- **ISA instruction exists in the gfx12 descriptor set.**
  `loom/py/loom/target/arch/amdgpu/descriptors/sets.py:1861`:
  `_v_wmma_i32_16x16x16_iu8_overlay(op_sel_hi_field="OPSEL_HI")` inside
  `_rdna4_core_overlays()` (line 1498) → `_gfx12_core_overlays()` (line
  1905) → `_gfx12_core_overlay_descriptors()` (line 1909), the descriptor
  set materialized for the gfx12 core ISA (gfx1200/gfx1201). The overlay
  itself (`matrix.py:363-377`) carries `mnemonic="v_wmma_i32_16x16x16_iu8"`.
  The same overlay is in the gfx11 (RDNA3) set at `sets.py:1413` — the
  instruction has existed since RDNA3; RDNA4 carries it unchanged (wave32,
  16×16×16, int32 accumulate).
- **The compiler knows the contract for gfx12.**
  `loom/py/loom/target/arch/amdgpu/matrix_contracts.py:1849-1864`:
  `AmdgpuMatrixContract(name="wmma.i32.16x16x16.iu8.gfx12", family="wmma",
  features=("wmma_gfx12",), flags=("sign_select","clamp"),
  tile_shape=(16,16,16), lhs=payload("iu8",2,8), rhs=payload("iu8",2,8),
  accumulator=payload("i32",8,8), result=payload("i32",8,8),
  intrinsic_name="llvm.amdgcn.wmma.i32.16x16x16.iu8",
  fragment_layout="rdna4_wmma_i32_16x16x16_iu8")`. The fragment layout
  (16, 8, 8 per `matrix_fragment_layouts.py:948`) matches the Loom IR row's
  measured widths (a/b 2 i32, c 8 i32 per lane).
- **The Loom `vector.mma` lowering pattern for i8 is pinned in the corpus.**
  `loom/src/loom/target/arch/amdgpu/test/source_low/source_low_mma.loom-test`
  (the `i8_i8_wmma_gfx1100` case, line ~3288) lowers
  `vector.mma ... : vector<4xi32>, vector<4xi32>, vector<8xi32>` to
  `v_wmma_i32_16x16x16_iu8` with `neg_lo/neg_hi` sign-select immediates;
  `source_low_mma_swmmac_gfx12_generic.loom-test:87-121`
  (`swmmac_i32_iu8`, target `amdgpu.gfx12.generic.core`) shows the same
  `element_format=i8` fragment schema on the gfx12 core set. The i8
  `payload_elements`/`payload_registers` values differ by lane width only
  (gfx11: 16/4 per lane; gfx12: 8/2 per lane), which is what our emitter
  spelling carries.
- **Open item for PART 2 (recorded, not papered over):** in the pinned
  corpus the gfx12 *row-specific* MMA lowerings are pinned on
  `amdgpu.gfx12.generic.core` / `gfx1250` targets, while the concrete
  `amdgpu.target<gfx1200>` rows in `source_low_mma.loom-test` only pin the
  fp8 swmmac path. Whether `loomc`'s contract resolution accepts the i8
  `vector.mma` on the concrete `gfx1201` target (vs declining to generic)
  is exactly what the PART 2 CPU suite's `LoomcCompiler::compile(src,
  "gfx1201")` gate measures. If it declines, the gate fails loudly (it is a
  hard loomc error, not a silent wrong instruction), and this project stops
  at that evidence — which is the correct outcome, not a paper-over.

## 4. fp16-exactness analysis (required for the record; why fp16 WMMA
    cannot replace the int path)

Even if the int8 row were unavailable, an fp16-operand WMMA
(`wmma12.f32.16x16x16.f16`) would NOT give exact Q6 products:

- A Q6 weight dequantizes to `w = c·d + t` with 6-bit code `c ∈ [0,64)`,
  group scale `d`, bias `t`. Staging `w` to fp16 and multiplying by an
  fp16-staged activation introduces **two independent roundings per
  product**; the product of two rounded 8-sig-bit values is not exact, and
  the bf16-staging floor we measured (code context ε = 0.0229,
  `cand-ffn-wmma2/gate-r2-report.md`) is precisely this error class.
- The concrete exactness bound the task asks for: an fp16 "product path"
  limited to integer operands can represent products of codes `c_a·c_b`
  exactly only while the product fits fp16's 11-bit mantissa (2^11 = 2048
  exactly representable integers). 6-bit codes give products up to
  **63·63 = 3969 > 2048** — i.e. even with per-group scale factored out
  and both operands restricted to the code range, fp16 cannot hold every
  integer product exactly. (For the biased-u8 form, 63·63 is the same
  bound on the bias-cross term.) The int32 core product, by contrast, is
  exact for the full u8×u8 range (max 255·255·16 = 1,040,400 « 2^31-1
  across the K-tile of 16). This is why the cited llama.cpp RDNA4 scheme
  (`research-precision-tradeoffs.md` §1.2: `mma.cuh:1388`
  `__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12`, u8 pre-bias,
  int32 in-core, fp32 group scales, fp32 cross-K) uses the integer core,
  and why the activation side stays on an int8 grid with per-K-group fp32
  scale + zero.
- f16 staging is strictly worse than bf16 staging (10 vs 8 mantissa bits,
  but a 65504 max and subnormal cliff at 2^-14 that the 6-bit-scale
  magnitudes routinely hit); it is not a candidate.

## 5. What this verdict enables (PART 1 design inputs)

- Kernel row: `wmma12.i32.16x16x16.iu8` (single-instruction, k_step=16).
  K-tile = 16; a 64-K block = 4 consecutive instructions, the same block
  cadence the bf16 record used (16-element swizzled LDS, two barriers per
  block).
- Fragment shapes: A/B `vector<2xi32>`, C `vector<8xi32>`; accumulator is
  **int32** per slot; per-group fp32 scale/zero are folded after the
  integer core exactly as `mmq-vec-dot.cuh:1058` does
  (`sum += C.x[l] * sc * x_df * dB`).
- Bias (pre-biased u8) cross term: with both operands biased by a constant
  (activation by +128, weight code by +32 — the Q4_0/Q6_K trick,
  `mmq-load-tiles.cuh` cited in the research doc), the integer sum carries
  three exact correction terms (x-mean, code-mean, constant) that are
  applied per group in fp32 from the same scale/zero the activation
  quantizer already computes. No weight ever becomes a float.

**Status: gate PASS → proceed to PART 1 (kernel).**
