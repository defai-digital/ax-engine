# Flash Next bounded short-first crossover on M5 Max

MacBook Pro M5 Max 128 GiB, internal NVMe SSD, macOS 27.2.1,
MLX and MLX Metal 0.32.4, and Rust and Cargo 1.99.1. The same signed
release-server binary runs both positions of the experimental switch
on the pinned MXFP4 pack `ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3`,
with mapped resident weights, two slots, and MTP off.

Each arm replays four distinct 8,192/512-token prompt pairs. The short
prompt is offered half a second after the long prompt and both
requests generate 96 greedy tokens. Under FIFO each long request
completes its 8,192-token prefill before the short request runs.
With short-first enabled, the short request runs whole while the
long peer is at the 2,048-token cut. The replay checks the marker,
the physical prefill cuts, and each request's solo token oracle.

| Order | Arm | CPU idle before / after | Timing eligible | Short turns | Matched output IDs |
| ---: | --- | ---: | --- | ---: | ---: |
| 1 | FIFO | 91.43% / 89.49% | No | 0 | 768 |
| 2 | Short-first | 90.71% / 87.88% | No | 4 | 768 |
| 3 | Short-first | 99.90% / 67.83% | No | 4 | 768 |
| 4 | FIFO | 93.80% / 99.18% | Yes | 0 | 768 |

All 3,072 output token IDs match. Every arm drains all five work
counters. All 881 pressure samples are normal, and peak footprint
is 75.75 GiB. Both paired orders miss the fixed CPU-idle requirement.
The archive retains descriptive raw latencies and emits no paired
timing comparison for either order. The single-request product
profile remains standard.

The first attempt wrote the FIFO streams and traces, then the analyzer
received the trace file path in place of parsed rows. The corrected
analyzer replayed the initial 768 IDs and the fresh crossover persisted
its CPU and drain observations before analysis. The retry kept the
signed server, workload, runtime patch, and threshold identical.

```sh
shasum -a 256 -c SHA256SUMS
python3.12 -B replay.py | cmp comparison.json -
```
