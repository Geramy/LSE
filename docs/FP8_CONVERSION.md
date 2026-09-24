# Shared OCP FP8 conversion

`lse/math/fp8.hpp` supplies the same host and typed kernel API to HIP and Loom:

```cpp
using E = lse::math::MatrixElem;
auto bits = lse::math::pack_fp8<E::kFp8>(a, b, c, d);
auto a_rounded = lse::math::unpack_fp8<E::kFp8, 0>(bits);
```

`kFp8` means OCP E4M3 (448 maximum finite); `kBf8` means OCP E5M2
(57344 maximum finite). The first operand occupies the low byte. Packing uses
round-to-nearest, ties-to-even and clamps **finite** overflow. Signed zero is
preserved. As in HIP, E4M3 infinity becomes NaN and E5M2 infinity remains
infinity. NaN class is preserved; device NaN payload and sign are unspecified.
The host implementation returns a signed canonical NaN. These are not the
incompatible FNUZ formats used by gfx94x.

HIP uses AMD's packed conversion and byte-select decode builtins. Loom uses
its typed vector narrowing and scalar extension, which lower to the native
gfx12 conversion instructions. The source explicitly preserves HIP's
exceptional-value behavior: Loom's E4M3 narrowing alone also saturates infinity,
so an infinity is converted to NaN before that narrowing. Both dialects clamp
finite overflow before conversion. A model feeder still needs to check
nonfinite operands and apply its own accuracy/exceptional-block policy.

References:
- [HIP FP8 device support and conversion API](https://rocm.docs.amd.com/projects/HIP/en/docs-6.3.0/reference/fp8_numbers.html)
- [AMD's HIP conversion implementation](https://github.com/ROCm/clr/blob/develop/hipamd/include/hip/amd_detail/amd_hip_fp8.h)

`test_fp8_conversion` covers every finite representation and adjacent rounding
midpoint, both signs, saturation, subnormal boundaries, and exceptional values.
`test_fp8_conversion_native` compiles both formats and every byte-select decode
through the shared typed API to gfx1201 HSACOs. Existing unrelated model kernel
selection is unchanged by adding these rows; callers must qualify the target
and required format rows.

R9700 hardware qualification additionally passed two replays of each format,
1024 encodes and 1024 decodes per replay, all 256 decode bytes, midpoint
neighbors, finite overflow, F32 subnormals, signed zero, and NaN/Inf classes.
Every finite/zero/infinity result was compared bitwise; NaNs were compared by
class. Four independent 64-byte-offset buffers retained their eight guards and
unchanged inputs through confirmed retirement. This qualifies conversion, not
an approximate model kernel's accuracy or automatic selection policy.
