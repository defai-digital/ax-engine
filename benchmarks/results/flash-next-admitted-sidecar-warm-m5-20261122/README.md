# Flash Next exact sidecar candidate warmed serving on M5 Max

MacBook Pro M5 Max 128 GB, internal NVMe SSD, MLX 0.32.4. The four arms
use the exact `838b331a3` release-server binary that passed the
[delivered-snapshot readiness repeat](../flash-next-admitted-sidecar-delivered-m5-20261122/README.md).
The paired reader experiment uses the original warm-serving runner,
which keeps the trunk on guarded mmap and changes only sidecar
payload reads. All four arms match 448 output IDs per arm. Health,
32K overflow rejection, the direct route and normal pressure hold.

| Pair | Mapped sidecar | Positional sidecar | Mapped whole startup | Positional whole startup |
| --- | ---: | ---: | ---: | ---: |
| 512-token first request | 11.990 s | 0.558 s | 158.339 s | 96.450 s |
| 8,192-token first request | 6.450 s | 0.523 s | 82.354 s | 92.442 s |

The mapped short-first process ends at 85.49% CPU idle, below the
90% floor. The original harness fails on that arm. The long-first
pair meets the host conditions. Its 512-, 2,048- and 8,192-token warm
TTFT medians increase by 2.57%, 1.19% and 4.07%. Its decode medians
remain inside the 3% band. The 8,192-token TTFT also misses the band.
The all-arm gate fails and the candidate remains opt-in.

The [raw terminal receipt](receipt.json.gz) retains all observations.
The [offline audit](replay.py) checks the two distinct misses and binds
the candidate to the previous startup receipt. The numeric short-first
comparison is retained with its failed timing eligibility.

```bash
python3 benchmarks/results/flash-next-admitted-sidecar-warm-m5-20261122/replay.py
```
