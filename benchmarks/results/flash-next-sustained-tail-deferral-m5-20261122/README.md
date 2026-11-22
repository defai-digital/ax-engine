# Flash Next repeated early and late arrivals on M5 Max

A MacBook Pro M5 Max 128 GB with internal NVMe SSD and macOS 27.2.1
ran the same codesign-verified `14235e84b` release-server executable
used in the [single-pair](../flash-next-server-prefill-tail-fixed-m5-20261122/README.md)
and [cancellation](../flash-next-primary-tail-holdouts-m5-20261122/README.md)
experiments. The run used Rust 1.99.1, MLX and MLX Metal 0.32.4, and the
pinned resident MXFP4 pack. The final inventory check verified all
49 publisher files.

Each of four processes exercises the same 12 pairs sampled from the
[frozen mixed-serving corpus](../flash-next-mxfp4-mixed-serving-m5-20261121/README.md).
Each pair starts a 512-token request and submits an 8,192-token
request at a specified short output token. Four pairs use early
arrivals at token 48 or 65. Eight pairs use late arrivals at token
84, 86, 88, or 92. Each request generates 96 greedy output tokens.
The four processes use physical, held, held, and physical order.

All 9,216 output IDs match the archived solo outputs. Every burst
drains all five work counters and all 1,247 pressure samples are
normal. Both held processes defer all eight late arrivals and leave
all four early arrivals on the existing physical path. Each physical
process starts long prefill in a shared step for all eight late
arrivals. Every process clears the 90% CPU-idle floor at both samples.

| Process order | Mode | Short maximum gap, median | Long TTFT, median | Output tokens/s | CPU idle before / after |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | Physical split | 743.70 ms | 10.966 s | 12.50 | 91.78% / 93.11% |
| 2 | Tail hold | 21.56 ms | 10.499 s | 12.90 | 91.80% / 95.23% |
| 3 | Tail hold | 21.57 ms | 10.572 s | 12.66 | 91.94% / 97.00% |
| 4 | Physical split | 769.51 ms | 11.294 s | 11.97 | 98.86% / 92.80% |

The table covers each process's eight late arrivals. The short gap
is the median of the per-burst maximum gaps after the trigger.
The long TTFT clock starts at that trigger, including any waiting.
Output throughput divides both streams' 192 tokens per burst by the
sum of the burst windows. Time between bursts is excluded.

The result establishes repeatability for deliberate early and late
arrivals in the experimental resident two-slot profile. The long
requests enter one at a time alongside short peers. Natural offered
load and the single-slot product qualification retain their own gates.

The archive preserves the tested runner, input digests, pressure
samples, and all 48 raw bursts. Replay binds the executable and
corpus to the preceding evidence, recomputes both matched pairs,
and checks the arrivals, route markers, token IDs, and drainage.

```sh
shasum -a 256 -c SHA256SUMS
python3 -B replay.py | cmp comparison.json -
```
