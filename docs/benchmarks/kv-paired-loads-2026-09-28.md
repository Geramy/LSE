# Paired 16-bit KV loads — 2026-09-28

On gfx1201/wave32, split attention reads adjacent FP16/BF16 values with one
4-byte load and widens both values to FP32. Attention products, reductions and
stable partial merging remain FP32. This comparison changes only the V-load
width; the query/key algorithm, inputs, masks and launch geometry match within
each pair.

## Matched component timings

Each case uses B1, 24 query heads, 4 KV heads, head dimension 256, block size 16
and a causal mask. Each arm warms 16 chains, followed by A/B/B/A blocks of 20
chains: 40 measured samples per arm. GPU CP times below include partial and
merge dispatches. Both paired-load blocks are faster than both scalar blocks
in all six cases.

| KV format | Queries | Capacity / live tokens | Two scalar loads, mean ms | Paired load, mean ms | Change |
|---|---:|---:|---:|---:|---:|
| FP16 | 4 | 8192 / 5207 | 0.346840 | 0.321032 | −7.44% |
| BF16 | 4 | 8192 / 5207 | 0.357076 | 0.334474 | −6.33% |
| FP16 | 4 | 16384 / 14000 | 0.896949 | 0.844768 | −5.82% |
| BF16 | 4 | 16384 / 14000 | 0.914686 | 0.864200 | −5.52% |
| FP16 | 1 | 16384 / 14000 | 0.524440 | 0.502972 | −4.09% |
| BF16 | 1 | 16384 / 14000 | 0.516974 | 0.481807 | −6.80% |

An earlier T1 comparison used the superseded serial-QK body; its timing was
invalidated for current policy and was not used to select a capacity exception.
The T1 rows above use the current cooperative wave-QK body in both arms.

## Correctness and resources

Every partial record and final output is bitwise identical between load widths
and repeated invocations. An independent double-precision reference reads the
exact stored-format values; current T1 maximum absolute component errors are
5.96e-8 for FP16 and 6.71e-8 for BF16. Ragged/sliding/padded-row gates and poisoned
empty-metadata replay also pass: every partial/final field is overwritten,
readonly input bytes remain unchanged, and 64-byte allocation guards survive.

Each actual timing process completes 228 native device dispatches with zero
host dispatches, fallback or measured compiles. Two additional format edge
processes complete 76 dispatches each. All generated objects report zero
private allocation; LDS and launch geometry are unchanged.

| Partial body | Scalar VGPR / SGPR | Paired VGPR / SGPR | LDS bytes |
|---|---:|---:|---:|
| T4, capacity 8192, both formats | 64 / 64 | 64 / 64 | 2048 |
| T4, capacity 16384, both formats | 58 / 54 | 58 / 54 | 2048 |
| Current T1, FP16 | 36 / 44 | 34 / 44 | 512 |
| Current T1, BF16 | 36 / 44 | 36 / 44 | 512 |

The merge source and launch plan are unchanged. These component results do not
establish model quality or an HTTP throughput gain.

## Reproduction and identity

Each case starts a fresh process and loads frozen code objects directly;
persistent JIT is bypassed. An owned isolated cache remains empty and the user
cache is untouched. Actual HSA identity is checked from dyld output.

Evidence is retained in `/private/tmp/lse-narrow-kv-load`: `cp-summary.json`
contains the T4 distributions, and `cp-summary-current-t1.json` contains the
current T1 distributions. Source/archive hashes, code-object resources and
process/runtime identities are in the corresponding `source-manifest`,
`build-manifest`, `resources` and `native-runs` JSON files.

- T4 fixture SHA256: `c8c5774577cc0a456bd3d14f4b34487218eda807d7cdf19b061f9dbebafd5b06`.
- Current T1 fixture SHA256: `91922eec38b19cf3b402283471165f489db4f663f141fb7e3591f14c7b331dd6`.
- Current `sdpa.cpp` SHA256: `c6d9727d540ea1a51224203dc4c7e409be494c3a68b90ddb7ab0a17afc196473`.
- `kv_storage.hpp` SHA256: `2564f8991b3128660382e87b698795c48090b935ffc019fda62dfca8d8a7727c`.
- Loaded `libhsa-runtime64.0.1.0.dylib` SHA256: `b7f8216e32fa6ab0e5b2fce87c518ce67c3fffa4c628cceece983c53e3cea90d`.
