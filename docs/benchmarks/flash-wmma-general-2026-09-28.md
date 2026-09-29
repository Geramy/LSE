# Generalized WMMA attention — 2026-09-28

`attention.flash.wmma16.v2` handles paged F32, F16, BF16, scaled FP8 and
scaled BF8 KV. Matrix operands are F16 for F16 storage and BF16 otherwise.
QK/PV accumulators, online softmax state and output remain FP32. This changes
operand rounding; it does not retain scalar FP32 attention arithmetic.

The central rule selects gfx1201/wave32, 256 threads, query widths at least 2,
and positive head/value widths up to 512. Other devices retain the existing
attention implementation. Masks, live metadata clamps, ragged/padded batches,
page resolution and zero output for empty attention are preserved. Sliding
window 0 follows the CPU causal convention. Aligned F16/BF16 K loads use 16 bytes;
PV loads remain scalar because the matrix fragment layout is strided.

## Native component checks

All 35 small cases passed on the registered production family, with 105 device
dispatches and zero host/fallback groups:

- All five storage formats: causal/sliding/none, logical widths20/28,
  query widths2/3/17, blocks2/8/16, Dh512/Dv320 and Dh=Dv512.
- Ragged rows and a padded batch use poisoned inactive table/KV cells.
- F16 covers capacity262144 with63 live keys.
- F32/BF16/FP8/BF8 cover finite V values around 1e30 with bounded Q/K.
- Complete output matches an independent oracle using the selected operand
  conversions. Every output must be finite, repeated output is bit identical,
  empty metadata replay overwrites with zero, and inputs plus 64-byte allocation
  guards remain unchanged. The 1e30 case reports amplitude-normalized component
  error; no model/logit L2 check is used.

FP16 and BF16 also passed four actual Qwen geometry cases: M512,
QH24/KVH4, Dh=Dv256, block16, capacities8192/16384 and live5610/14000.
Each checks all 3,145,728 outputs for finite completion and exact repeat/empty
replay; an independent oracle checks eight query rows across every head/channel.
Maximum absolute error is 6.25e-8 to1.99e-7. These cases add 32 device dispatches,
zero host/fallback. Total native coverage is 39 cases and 137 dispatches.

| Compiled geometry | VGPR | SGPR | LDS bytes | Reported private bytes |
| --- | ---: | ---: | ---: | ---: |
| Qwen D256, F16/BF16 | 120 | 40 | 24,768 | 0 |
| Dh512/Dv320, F16/BF16 | 132 | 46 | 32,960 | 0 |
| Dh512/Dv320, F32/FP8/BF8 | 128 | 48 | 32,960 | 0 |
| Dh=Dv512, all five formats | 148 | 40 | 32,960 | 0 |

Private 0 is queried metadata, not an inferred occupancy or invented spill count.
The maximum admitted Dh=Dv512 boundary passed all five formats: complete
2,048-output oracle, finite completion, exact repeat, empty replay, input integrity
and guards. Maximum absolute error is 2.97e-9 to 2.24e-8. This gate adds 15 device
dispatches and zero host/fallback; no performance measurement was made.

## Passive device durations

Two warmups precede three measured replays; initial JIT is excluded. CP ticks
were checked against the device's 100 MHz timestamp frequency.

| Storage | M512/live5610 median | M512/live14000 median |
| --- | ---: | ---: |
| F16 | 13.878 ms | 41.308 ms |
| BF16 | 14.214 ms | 42.115 ms |

These are sequential component measurements, not a paired speedup or HTTP
throughput claim. Empty replay takes 12.4–13.64µs. Host wall measurements contain
submission and completion wait and are reported separately in the local logs.

A later BF16 v2 model check scored the same 1,024 pinned targets as the historical
FP32/FP16 references at PPL 4.849896867834. All 32 M512 attention routes selected
v2; all 254,279,680 logits were finite, with 3,142 device groups and zero
host/fallback. This single later-build prefill result does not establish sampled
conversation quality or a statistical improvement. See the
[BF16 method and limits](bf16-mode-comparison-2026-09-28.md).
The earlier FP16 storage/WMMA quality pair belongs to the narrower v1 path.

## Addressing and provenance

The reusable `kv_load_vector` keeps known vector/column indices separate, avoiding
unsigned division by non-power-of-two logical widths. WMMA page IDs enter the
ordinary address domain directly; the packed scalar-word conversion caused a
native compiler address-range rejection. No page clamp or alternate kernel
was added.

Final kernel SHA256 is `60cf88165258c6ac9dbef9fab7ed6efd75bcc87a203a5b1bf023efd0fe06c02c`;
KV helper SHA256 is `2564f8991b3128660382e87b698795c48090b935ffc019fda62dfca8d8a7727c`.
The explicitly mapped HSA dylib SHA256 is
`b7f8216e32fa6ab0e5b2fce87c518ce67c3fffa4c628cceece983c53e3cea90d`.
Local fixture, native logs, CP summary and provenance are under
`/private/tmp/lse-flash-wmma-general/`. Each Scheduler process uses a fresh
private cache directory; the shared persistent cache was untouched. The final
packed gate overlays only the new Flash object on a copied canonical archive.
