### Fragmented K/V memory management

Paged K/V on the Loom backend now uses a dedicated shared memory manager.
It packs 256 KiB fragments into 256 MiB arenas and fills available slots before
allocating another arena. Growing a cache preserves its existing K/V addresses
and data. This removes whole-pool K/V copies during growth. Weights and general
tensor allocations keep their existing policy.

The local Q4 target with Q8 DFlash2 and BF16 K/V completed 68,301 total tokens.
Peak VRAM was 27.83 GB, with at least 6.11 GB free. Isolated fragment-addressing
cost was 0.7–8.9% at 16K context and 2.5% for eight queries at 68K context.
A matched short-context server throughput comparison has not been run.
The existing optimized attention shape limit above 65,536 keys remains.
This release is a K/V growth and memory-management fix; it does not claim a
throughput improvement or validate the full configured context capacity.
See [K/V storage](https://github.com/Geramy/LSE/blob/v0.4.20/docs/KV-STORAGE.md) and the
[measurement report](https://github.com/Geramy/LSE/blob/v0.4.20/docs/benchmarks/kv-fragments-2026-09-29.md).
The archives bundle the matching HRX runtime and Loom compiler.
