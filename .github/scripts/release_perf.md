### Fragmented K/V memory management

Paged K/V on the Loom backend now uses a dedicated shared memory manager.
It packs 256 KiB fragments into 256 MiB arenas and fills available slots before
allocating another arena. Growing a cache preserves its existing K/V addresses
and data. This removes whole-pool K/V copies during growth. Weights and general
tensor allocations keep their existing policy.

The local Q4 target with Q8 DFlash2 and BF16 K/V completed 68,301 total tokens.
Peak VRAM was 27.83 GB, with at least 6.11 GB free. Isolated fragment-addressing
cost was 0.7–8.9% at 16K context and 2.5% for eight queries at 68K context.
Automatic short-query split attention now covers compatible capacities above
65,536 keys, subject to device resource limits. In a same-executable attention
check at 65,656 live keys, split attention measured 8.12 ms versus 71.79 ms for
forced unsplit attention. The configured maximum context is not a tested usable
capacity.

### Long-context prefill

Vector staging and a shape-gated cache hint improve the Flash WMMA attention
kernel on gfx1201. A matched cold 64K HTTP comparison measured 146.06 to 224.02
prefill tokens/s (+53.4%), with prompt time falling from 448.70 to 292.54 seconds.
Decode was 13.85 versus 13.79 tokens/s. At 1K, prefill was 413.16 versus 408.43
and decode was 27.75 versus 27.55 tokens/s. These are single measurements.
Both request pairs produced identical text and acceptance counts with zero CPU
fallback. See the [measurement report](https://github.com/Geramy/LSE/blob/v0.4.20/docs/benchmarks/attention-vector-staging-2026-09-29.md).

See [K/V storage](https://github.com/Geramy/LSE/blob/v0.4.20/docs/KV-STORAGE.md) and the
[measurement report](https://github.com/Geramy/LSE/blob/v0.4.20/docs/benchmarks/kv-fragments-2026-09-29.md).
The archives bundle the matching HRX runtime and Loom compiler.
