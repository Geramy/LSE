# Activation INT8 policy

Set `LSE_HRX_INT8=1` before launching LSE to allow its existing Q4 mixed-sign dot
and integer WMMA model kernels. These kernels quantize floating-point
activations to signed INT8 and therefore introduce extra rounding. They remain
subject to the existing architecture, shape and accuracy constraints.

Unset, `0`, and every value other than exactly `1` keep activation conversion
disabled. The setting is latched at first kernel selection for the process;
changing its environment after compiling or retaining a program has no effect.
Both HIP and Loom use the same policy. Emission and persistent JIT keys include
its effective value, so an enabled process cannot reuse disabled-policy code.

This flag does not disable genuine integer operations, raw dot/matrix
instructions, or the validated integer calibration probe. It does not enable
INT8 activation conversion for Q6 or Q8: their existing accuracy gate continues
to use floating-point arithmetic. FP8/BF8 instruction support is independent of
this policy and does not requantize a model's stored weights or activations.
