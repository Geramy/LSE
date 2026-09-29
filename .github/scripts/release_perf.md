### Weight slab VRAM reduction

The macOS HRX weight allocator now packs the Q4 target and Q8 DFlash2 draft
into 512 MiB slabs. In matched local R9700 runs, reserved GPU memory after
loading fell from 20.583 to 18.870 GB and after a request from 24.694 to
22.978 GB. Warm repeated requests measured 545.5 versus 548.3 prompt
tokens/s and 49.9 decode tokens/s in both builds. Generated output hashes and
80% DFlash2 acceptance matched. These numbers use decimal GB and driver
reserved-memory counters; timing does not show a throughput regression, but
does not establish identical performance for every workload.

[Method and limits](https://github.com/Geramy/LSE/blob/v0.4.18/docs/benchmarks/weight-slab-memory-2026-09-29.md).

### Earlier prefill workspace memory fixes

Completed target graphs are released before the next request's prefill and
when the prefill chunk width changes. Consecutive full-size chunks keep replay.
DFlash2 releases completed large context-projection programs while preserving
narrow decode programs. Live recurrent state, KV pools, weights and compiled
kernels remain reusable.

The matched local macOS/R9700 test uses Qwen3.8-27B Q4, Q8 DFlash2, BF16 KV,
FP32 accumulation, batch/ubatch 1024, temperature 0.6, top-k 20, top-p 0.95,
seed 1234 and KV capacity 262100. A 5120-token request precedes a 6143-token
request with a 1023-token remainder. Each generates 128 tokens.

| Metric | v0.4.16 control | Memory fix |
| --- | ---: | ---: |
| Reserved VRAM after ragged request | 29.77 GB | 28.26 GB |
| Sampled peak reserved VRAM | 30.09 GB | 28.87 GB |
| Ragged request prefill | 457.82 tok/s | 458.49 tok/s |
| Ragged request decode | 68.21 tok/s | 67.88 tok/s |

Responses and acceptance results match exactly. The synthetic decode request
has 100% proposal acceptance; its rate is not a general DFlash2 rate. The small
timing differences do not establish a throughput improvement or regression.
An eight-turn cached-follow-up comparison also matches all outputs and cache
lengths. Reserved VRAM grows only 12.14 MB over those turns. Aggregate decode
is 42.89 tok/s for control and 42.94 tok/s for the memory fix.

Measurements use memory-fix source `ec71e14`; v0.4.17 adds version, documentation
and packaging changes. Numbers use decimal GB and driver reserved-memory
counters sampled every 0.5 seconds. Sampling can miss instantaneous peaks.
The reported allocation failure has not been replayed. The result
establishes reduced workspace retention, not elimination of every possible OOM.

[Full memory method and limits](https://github.com/Geramy/LSE/blob/v0.4.17/docs/benchmarks/prefill-workspace-memory-2026-09-29.md).
The earlier combined M8 gate/up kernels and all accepted typed attention,
activation panels, buffer views and speculative decoding paths remain active.
[Earlier all-mode throughput](https://github.com/Geramy/LSE/blob/v0.4.17/docs/benchmarks/m8-gate-up-pair-2026-09-29.md)
uses a different 1024-token workload and is not a new v0.4.17 measurement.
