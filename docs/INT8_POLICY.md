# Q4 compute dispatch

Hardware, dtype, shape, and resource selection lives in
`src/dispatch/quant.cpp`. On supported devices, group64 Q4 uses INT8
activation conversion for single-row decode and prefill at M512 or larger.
The two gfx1201 M512 FFN shapes use the winning activation LDS matrix layout;
other matrix shapes use the standard INT8 layout. Intermediate row counts
retain floating-point arithmetic. Indexed operations retain the scalar path.

There are no model hashes, quality flags, perplexity values, or environment
switches in execution. The selected implementation and dispatch revision
identify cached kernels. Other operand types and kernel bodies remain in the
source without becoming default selections.

Integer dot and matrix products accumulate in INT32 before FP32 affine
restore. Floating-point paths retain FP32 accumulation. Q6 and Q8 activations
are not converted to INT8 by this table.

## Accepted measurements

Matched current-engine prefill perplexity scores 2,046 actual next-token
targets, after an excluded warmup: FP32 activations yield PPL 8.100851 and INT8
activations 8.127312 (+0.3266%). Automatic and explicit INT8 selection both yield
8.127312. The corpus SHA-256 is
`694b24612e47cbbdc03a333e0b7dec407cd13945d63e63c64ccd0ed4adebad07`.

Matched teacher-forced decode scores 1,024 actual targets after a 511-token
prefix: automatic selection yields PPL 7.20144530149 and explicit INT8
7.17831687877 (+0.3222%). Automatic selection retains exact arithmetic for
M2 through M256 prefix chunks. Both runs have finite outputs, zero host groups
or fallbacks, and successful exits. This corpus SHA-256 is
`4d207d8fc8c7298a0489133e104c3e12ecee9c58f2c9828dec178725cd9f17b0`.
These historical results were accepted for this checkpoint and device. Perplexity on
1,024–2,048 actual targets is the standard model-quality test.

Further kernel measurements are in [QUANT_OPERANDS.md](QUANT_OPERANDS.md) and
[the Mac performance report](https://github.com/lemonade-sdk/mac-amdgpu/blob/main/docs/LSE_PERFORMANCE.md).
