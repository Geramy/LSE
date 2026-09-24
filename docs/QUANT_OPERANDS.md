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

Selection combines the affine storage contract, device capabilities, available
intrinsics, workgroup resources, accepted accuracy evidence and matched costs.
The initial accepted profile is gfx1201/64 CU with group64 Q6, F32 activations
and output, and BF16 scale/bias metadata. Four exact shapes are qualified:
M64 or M512 with (N,K)=(17408,5120) or (5120,17408). Staged BF16 wins all four.
Unknown shapes and single-token decode retain the existing floating-point
implementation. This is an initial measured profile, not a claim of optimal
selection for every shape or GPU. No manual Q6 precision switch is needed.

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

The actual E4M3 Qwen Q6 candidate's 64-token prompt produced all 248,320 finite
logits, 0.12055% relative L2 difference, KL divergence 1.26e-6, and identical
argmax/top-20 membership versus the FP32 baseline. This single prompt does not
establish broad model quality. E4M3 was slower than staged BF16; BF8 additionally
failed cancellation-safe absolute-error checks on two fixtures. Neither is
enabled as the automatic Q6 winner. Their implementations remain available for
further qualified profiles.

Inputs around 1e-38 exposed a separate discrepancy from the CPU reference in
both the original scalar GPU path and the exceptional-block fallback. Its
cause remains unverified despite the code object's requested NO_FLUSH mode.
It is not counted as a passing relative-precision test.

The selected implementation name participates in HIP/Loom emission and JIT
identity. A profile cannot reuse a different operand implementation's cached
code. Timing updates that preserve the selected implementation do not change
that identity. The old manual `LSE_Q6_WMMA` qualification switch is retired.

`LSE_HRX_INT8=1` is a separate explicit accuracy policy for activation-quantized
Q4 kernels; see [INT8 policy](INT8_POLICY.md). Full model timings and reproduction
details are in the [Mac performance report](https://github.com/lemonade-sdk/mac-amdgpu/blob/main/docs/LSE_PERFORMANCE.md).
