# Quantized storage and compute operands

HIP and Loom use the same quantized kernel implementation. A checkpoint's
storage format does not directly specify its matrix operands: MLX affine Q6
stores integer codes plus scale/bias metadata, not six-bit floating-point
values. Kernels retain packed Q6 in VRAM and convert workgroup tiles on-chip.

The shared gfx1201 path implements staged BF16 and scaled OCP E4M3/E5M2
high/residual operands with three FP32-accumulating matrix products. Conversion
matches HIP finite saturation and nearest-even rounding, including explicit
NaN/infinity handling. Hardware tests cover all 256 decode byte values and
rounding edges; model kernels also check exceptional blocks before narrowing.

Operand precision and accumulation precision are separate choices. AMD's
[rocWMMA type table](https://rocm.docs.amd.com/projects/rocWMMA/en/docs-7.2.0/api-reference/api-reference-guide.html)
lists gfx12 FP8, BF8, FP16, and BF16 matrix inputs with FP32 accumulation;
INT8 matrix products accumulate in INT32 before scaling to floating point.
The [OCP FP8 specification](https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-06-20-pdf)
defines E4M3 and E5M2 as different exponent/mantissa tradeoffs. Narrowing can
change model predictions even when FP32 accumulation is preserved, so candidate
operands must pass matched 1,024–2,048-scored-token perplexity qualification.
No BF32 accumulator type is defined in this backend; existing floating-point
accumulation remains FP32.

Selection lives in `src/dispatch/quant.cpp`, alongside the hardware and
shape tables. The two gfx1201/wave32 M512 FFN shapes with group64 Q6, F32
activations/output, and BF16 scale/bias use staged BF16. Q6 decode uses the
four-column scalar body; scalar prefill reuses an eight-row activation panel
at M32 or larger. Storage, intrinsics, and resource requirements are checked
in dispatch. Model identity and quality measurements are not runtime inputs.

The accepted historical Qwen3.8-27B comparison measured PPL 9.062509 versus
9.062366 on 1,022 actual next-token targets; its original 1,024-target label
counted input tokens. This accepted result is retained without another run.
The standard for new model-quality comparisons is 1,024–2,048 actual targets.

| M / N / K | FP8 E4M3 | BF8 E5M2 | Staged BF16 |
|---|---:|---:|---:|
| 64 / 17408 / 5120 | 4.171 ms | 1.988 ms | **1.878 ms** |
| 512 / 17408 / 5120 | 18.451 ms | 9.671 ms | **6.711 ms** |
| 64 / 5120 / 17408 | 4.375 ms | 3.662 ms | **3.005 ms** |
| 512 / 5120 / 17408 | 13.622 ms | 11.046 ms | **8.313 ms** |

These are means of eight warm host evaluation-plus-retirement intervals, with
complete outputs and guards checked. They include submission overhead and are
not raw GPU timestamps or whole-model TPS. Original scalar up-projection times
were 4.946 and 18.448 ms at M64 and M512 in a preceding controlled comparison.

The E4M3 Qwen Q6 candidate produced finite outputs on a 64-token prompt, but
that prompt is below the required 1,024–2,048 scored tokens for perplexity.
E4M3 was slower than staged BF16; BF8 also failed arithmetic safety checks on
two fixtures. Neither is enabled as the automatic Q6 winner. Their
implementations remain available for further qualification.

Inputs around 1e-38 exposed a separate discrepancy from the CPU reference in
both the original scalar GPU path and the exceptional-block fallback. Its
cause remains unverified despite the code object's requested NO_FLUSH mode.
It is not counted as a passing arithmetic safety test.

The selected implementation and dispatch table revision participate in
HIP/Loom emission and JIT identity. FP8/BF8 and the BF16 residual body remain
registered, without a default route or temporary precision switch.

Q4 dispatch is documented in [INT8_POLICY.md](INT8_POLICY.md). Full model
measurements are in the [Mac performance report](https://github.com/lemonade-sdk/mac-amdgpu/blob/main/docs/LSE_PERFORMANCE.md).
