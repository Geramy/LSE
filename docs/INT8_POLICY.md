# Activation INT8 policy

The default policy automatically selects the qualified Q4 INT8 scalar path at
M1 and integer WMMA path at M512 for the pinned Qwen3.8-27B checkpoint on
gfx1201/wave32. Qualification requires the complete configuration and all three
mapped weight-shard SHA-256 hashes. The immutable weight profile is propagated
into graph and compiler identities. Unknown checkpoints, feature slices,
indexed operations, rank-three batches and draft-model weights retain FP32
activation arithmetic. Row-only permutations preserve the profile because
codes, scales and biases move together without changing each output's math.

`LSE_HRX_INT8=0` disables activation conversion. Exactly `1` retains the existing
explicit diagnostic policy, subject to device and shape restrictions. Unset
selects the automatic policy; other values disable conversion. The setting is
latched at first kernel selection. HIP and Loom share this policy, and emission
and persistent JIT identities distinguish its effective selection. The
automatic WMMA crossover remains M512 regardless of a diagnostic minimum-M
override. `LSE_Q4_DECODE_EXACT=1` retains exact scalar decode.

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
These results were accepted for this checkpoint and device. Perplexity on
1,024–2,048 actual targets is the standard model-quality test.

Integer dot and matrix products accumulate in INT32 before FP32 affine restore.
Floating-point paths retain FP32 accumulation. This policy does not convert
Q6 or Q8 activations to INT8. FP8/BF8 implementations remain available without
being selected as defaults. Details and frozen evidence are in the
[Mac performance report](https://github.com/lemonade-sdk/mac-amdgpu/blob/main/docs/LSE_PERFORMANCE.md).
