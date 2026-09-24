# Q6 INT8 prefetch experiment

This branch is experimental and is not selected by the working default.
Two consecutive K iterations fetch operands before consuming them in the
original arithmetic order. The explicit INT8 policy remains required.

On R9700, all 71 numerical cases passed nine repeats; output hashes match the
previous four-column INT8 implementation. Native resources are 69 VGPR,
84 SGPR and zero scratch, with unchanged LDS and launch geometry.

Performance is not accepted. The earlier single-iteration candidate regressed
on both large FFN projections despite passing numerical tests. A direct
identical-buffer control passed its first shape but failed the FP32 oracle
after changing shapes; investigation must resolve that failure before its
timings can be used. Do not relax the oracle or infer a cache defect yet.

Local mac_amdgpu artifacts: build/perf-q6-int8-quad-prefetch2 and
build/perf-q6-int8-shared-buffers. The corrected diagnostic supports isolated
shape runs; the failed logs and binaries remain preserved. Full-model quality
and speed qualification remain pending.
