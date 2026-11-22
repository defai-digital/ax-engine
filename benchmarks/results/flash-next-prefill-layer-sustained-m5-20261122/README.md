# Flash Next layer checkpoints under sustained mixed serving

MacBook Pro M5 Max 128 GB, internal SSD, MLX and MLX Metal 0.32.4 and
Rust 1.99.1. One release-server executable from experimental
`646827ee3` ran physical-only, 16-layer, 16-layer and physical-only.
The two-slot resident MXFP4 profile kept greedy direct decoding,
physical prefill settings and the pinned publisher pack fixed.
Drafting, n-gram acceleration, coalescing and prefix reuse were off.

Each arm replayed the same 220 offered arrivals, with 132 512-token,
66 2,048-token and 22 8,192-token prompts. Every measured reply
matched all 96 output IDs from the
[single-slot oracle](../flash-next-mxfp4-mixed-serving-m5-20261121/README.md).
All five queue and stream counters drained in all four arms.
Sampled pressure stayed normal, peak server footprint was 75.83 GiB
and all arms met the 90% CPU idle floor before and after the run.
The two checkpoint arms published 447 and 429 suspended steps.
Each arm exercised 46 medium prompts and three long prompts, binding
every layer advance to its original prompt cursor and the peer's
decode step.

| Run order | Short p95 gap tail | Short largest gap | 8K p95 arrival to first token | 8K p95 from client dispatch | Total output rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Physical 1 | 344.10 ms | 933.70 ms | 49.488 s | 12.180 s | 22.896 tok/s |
| 16 layers 1 | 235.65 ms | 747.03 ms | 92.038 s | 13.416 s | 21.765 tok/s |
| 16 layers 2 | 229.55 ms | 650.27 ms | 61.539 s | 12.672 s | 22.564 tok/s |
| Physical 2 | 645.87 ms | 1,022.25 ms | 76.267 s | 13.140 s | 22.325 tok/s |

The short gap tail is the p95 of each short request's p95 token gap.
It falls 31.5% and 64.5% in the two orders. Matching short requests by
ID finds 22 of 132 and 61 of 132 with a smaller individual p95.
Median matched changes are increases of 3.27 ms and 0.41 ms.
The cohort benefit is concentrated in its tail. From offered arrival,
8K p95 first-token latency rises 86.0% and falls 19.3%. This metric
includes waiting for a client lane. Total output throughput falls 4.9%
and rises 1.1%. The opposing long-latency and throughput results keep
16-layer scheduling experimental.

The [de-identified evidence](evidence.tar.xz) contains raw HTTP stream
observations, request-local steps, sampled pressure and the completed
receipt. The [driver archive](driver.tar.xz) and [input receipt](inputs.sha256)
bind the run. Replay checks 880 measured responses and 84,480 output
IDs and recomputes the [comparison](comparison.json), including paired
request changes and client-lane wait.

```sh
python3 benchmarks/results/flash-next-prefill-layer-sustained-m5-20261122/replay.py
```
