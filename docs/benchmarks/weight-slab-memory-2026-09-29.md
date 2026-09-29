# Weight slab VRAM reduction

## Change

LSE previously reserved up to one sixteenth of device capacity per weight
slab on macOS HRX, about 2022.9 MiB on the tested R9700. It now uses 512 MiB
slabs, as the other backends already do. This bounds unused tail space while
keeping driver allocation calls far below one call per tensor. Tensor bytes,
placement class and GPU arithmetic are unchanged.

## Method

The control is published v0.4.17. The candidate changes only the slab size.
Both use the local Qwen3.8-27B Q4 target and Q8 DFlash2 draft, BF16 KV,
FP32 accumulation, batch and ubatch 1024, KV capacity 262100 and temperature
0.6. Each request uses top-k 20, top-p 0.95 and seed 1234. One 1321-token
request generates 32 tokens, followed by four 1321-token requests requesting
128 tokens. The last three requests measure warm throughput.

Only one server runs at a time. Driver counters report reserved GPU
allocations, including unused tails; they are not live tensor-byte counts.
Memory and throughput comparisons below are from matched runs. Build and
release packaging can shift baseline driver reservations slightly, so the
slab-byte difference is the direct accounting of this change.

| Measurement | v0.4.17 | 512 MiB slabs |
| --- | ---: | ---: |
| Weight slabs after target and draft load | 9 | 32 |
| Weight slab reservation | 17.780 GiB | 16.184 GiB |
| Driver reserved VRAM after load | 20.583 GB | 18.870 GB |
| Driver reserved VRAM after request | 24.694 GB | 22.978 GB |
| Final repeated warm prefill | 545.5 tok/s | 548.3 tok/s |
| Final repeated warm decode | 49.9 tok/s | 49.9 tok/s |
| Reported weight-load time | 4.244 s | 4.260 s |

The deterministic responses and 80% DFlash2 acceptance matched. The first
requests included compilation and submission tuning, so they are excluded
from warm throughput. The timing comparison does not prove identical speed
for every workload; it shows no regression in this matched run.

## Scope

This change frees approximately 1.71 GB of reserved VRAM. It does not
offload active weights or KV to host RAM. The earlier prefill workspace
retirement remains active. At long context, graph workspaces and backend
allocation high-water still need a separate ownership audit. This release
does not claim to resolve the reported GPU-service failure near 18K tokens.
