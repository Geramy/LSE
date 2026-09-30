# Paged K/V storage

The Loom path stores paged attention K/V in a dedicated memory manager. The
scheduler shares this manager across model layers and HTTP sessions. Weight,
activation, recurrent-state, and general tensor allocation do not use it.

## Allocation

- Each arena is 256 MiB (268,435,456 bytes).
- Each fragment is 256 KiB. An arena has exactly 1,024 fragment slots.
- K and V from different layers can occupy the same arena.
- The manager uses available slots on the same backend and stream before it
  allocates another arena. Released slots can be reused.
- A cache grows by adding fragments. Its existing K/V bytes do not move.

The last arena can be partially filled. Each K/V tensor can also have an
unused tail smaller than one fragment. These are bounded capacity reserves;
there is no separate 256 MiB reserve for every layer.

Released sequence blocks are reused within their cache. The cache keeps its
high-water capacity across a session restart to avoid allocation during the
next request. Destroying the cache releases its fragments. An arena is freed
when its final fragment lease is released.

## Addressing and lifetime

Each K/V tensor has a small device table of 64-bit fragment addresses. The
manager reserves this table for the configured context capacity; it does not
reserve all of the corresponding K/V data. Adding resident fragments updates
only the new table entries. Existing entries and the table allocation remain
stable within that capacity.

Attention and K/V-write kernels resolve logical indices through the table.
Vector accesses stay vector accesses. A kernel binding retains the table and
its fragment storage, including when an older graph still holds the binding.

Logical pool shapes still use the existing compilation buckets. Crossing a
shape bucket can rebuild a graph, but does not copy the old K/V pool. Growing
within a bucket does not create a kernel variant for each fragment.

## Backend requirements

This path needs the Loom `buffer.from_address` operation and an HRX native
device-address export. Release builds supply these through the pinned HRX patches. The Linux
bootstrap also applies `patches/0004-hrx-kv-fragment-addressing.patch`. Other toolchains retain their
existing contiguous storage implementation. Fragment tables execute on their
owning device; copying a table to another device is not a K/V migration.

K/V precision and FP32 attention accumulation are unchanged. This feature does
not offload active K/V to system RAM.

## Checks

`test_kv_memory` checks arena packing, slot reuse, retained storage lifetime,
block-allocation order, stable table addresses, and growth across a fragment
boundary. `test_kv_cache` checks the existing cache formats and writes.

`kv_fragment_probe` compares contiguous and fragmented BF16 K/V on a GPU. It
uses nonadjacent fragments and permuted page IDs, checks K/V writes, and compares
attention outputs bit for bit for 1, 8, and 128 queries. Its timings include host
submission and synchronization; they are not GPU-only profiler measurements.
