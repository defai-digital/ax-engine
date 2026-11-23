# Flash Next natural-arrival replay with process CPU accounting

The [previous two-slot crossover](../flash-next-natural-short-drain-m5-20261123/README.md)
is repeated on MacBook Pro M5 Max 128 GB, internal SSD, macOS 27.2.1
and MLX 0.32.4. The same signed release-server executable, pinned MXFP4
pack and 220 offered arrivals are used in each arm. Resident placement,
2,049 scheduled tokens and disabled MTP stay within the experimental
two-slot serving profile. All four arms match **84,480** prior output
IDs and drain all five lifecycle gauges with **4,067** normal pressure
samples.

| Comparison, mixed → priority drain | First order | Reverse order |
| --- | ---: | ---: |
| Matched short turns with smaller maximum inter-token gap | 17/17 | 16/16 |
| Median per-turn maximum gap | 840.40 → 24.60 ms | 900.07 → 23.78 ms |
| Delivered output rate | 23.245 → 23.122 tok/s | 22.847 → 23.346 tok/s |
| 512-token offered TTFT p95 | 29.30 → 36.48 s | 44.83 → 25.34 s |
| 2,048-token dispatch TTFT p95 | 2.92 → 13.81 s | 3.09 → 13.55 s |

All 33 matched 2,048-token successors start prefill one engine step
after the peer's last long decode in both policies. Drain lowers
the short output gap in both orders and extends the successor's
dispatch-to-first-token latency. Queue wait before dispatch moves
in opposite directions between the orders.

## Host CPU across each client's full window

The [process sampler](../../../scripts/observe_serving_cpu.py) separates
CPU consumed by the server and load generator. The residual uses host
busy CPU time minus those measured processes. It includes other host
activity and sampling-boundary effects.

| Arm | Host idle | Server average cores | Client average cores | Residual average cores |
| --- | ---: | ---: | ---: | ---: |
| Mixed first | 85.49% | 0.773 | 0.0039 | 1.821 |
| Drain first | 85.98% | 0.777 | 0.0039 | 1.730 |
| Drain second | 89.14% | 0.743 | 0.0037 | 1.198 |
| Mixed second | 86.91% | 0.765 | 0.0038 | 1.576 |

The two drain arms finish at 88.89% and 88.84% idle. All four
full observation windows also fall below 90% idle. The comparisons
establish output and lifecycle identity and a bounded short-gap
effect. The measured rates retain their original host load and
do not support a general throughput or latency promotion. The
single-slot resident profile remains the product standard.

The [comparison](comparison.json) includes source CPU intervals,
per-minute assignment and request-level admission joins. The
[sealed evidence](evidence.tar.xz) retains every portable workload,
trace, pressure and CPU record. The replayer checks the prior
campaign digests and rebuilds both comparisons.

```sh
shasum -a 256 -c SHA256SUMS
python3.12 -B replay.py | cmp comparison.json -
```
