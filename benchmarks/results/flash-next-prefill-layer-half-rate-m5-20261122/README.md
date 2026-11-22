# Flash Next layer checkpoints at half offered load

MacBook Pro M5 Max 128 GB, internal SSD, MLX and MLX Metal 0.32.4 and
Rust 1.99.1. The experiment uses the same two-slot resident MXFP4
profile and release-server executable as the
[full-rate layer comparison](../flash-next-prefill-layer-sustained-m5-20261122/README.md).
It doubles the original arrival offsets from an offered rate of
0.25 to 0.125 requests/s. The frozen corpus, greedy settings and
96 generated tokens per request stay fixed. Each treatment is repeated
with reversed process order.

All 880 measured responses preserve their prompt IDs and all 84,480
generated IDs from the [single-slot oracle](../flash-next-mxfp4-mixed-serving-m5-20261121/README.md).
Both layer arms exercise genuine suspended work with a decoding peer.
They suspend 22 and 21 medium requests and 11 long requests each.
Every arm drains all five queue and stream counters and exceeds the
90% CPU-idle floor before and after timing. All 7,390 pressure
samples are normal. Peak server footprint is 75.82 GiB.

| Run order | Short p95 of per-request maximum gap | 8K dispatch TTFT p95 | Overall arrival-to-first-token p95 | Succeeded output rate |
| --- | ---: | ---: | ---: | ---: |
| Physical 1 | 683.58 ms | 10.711 s | 9.563 s | 12.059 tok/s |
| 16 layers 1 | 471.99 ms | 11.921 s | 10.043 s | 12.059 tok/s |
| 16 layers 2 | 475.01 ms | 12.228 s | 11.345 s | 12.058 tok/s |
| Physical 2 | 714.71 ms | 11.689 s | 9.520 s | 12.058 tok/s |

In paired order, the short gap tail falls 31.0% and 33.5%.
The long dispatch TTFT p95 rises 11.3% and 4.6%. Overall
arrival-to-first-token p95 rises by 0.480 and 1.824 seconds.
Client-lane p95 wait stays close to 10 ms in all four arms.
Completed output rate is virtually identical at this scheduled
cadence. The matched median change in each short request's p95
gap is within 0.03 ms of zero in both orders. This supports a
short-stream tail improvement at this arrival rate and keeps
16-layer scheduling experimental.

The [de-identified evidence](evidence.tar.xz) retains client streams,
request-local steps, sampled pressure and the terminal receipt.
The [driver](driver.tar.xz) and [input hashes](inputs.sha256) bind
the experiment to the frozen pack and measured executable. Replay
checks that only the arrival offsets change relative to the
full-rate workload, confirms the treated steps, checks output
identity and recomputes the [comparison](comparison.json).

```sh
python3 -B benchmarks/results/flash-next-prefill-layer-half-rate-m5-20261122/replay.py
```
