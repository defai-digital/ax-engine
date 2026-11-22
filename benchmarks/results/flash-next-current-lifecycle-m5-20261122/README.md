# Flash Next current native lifecycle on M5 Max

MacBook Pro M5 Max 128 GB, internal NVMe SSD, macOS 27.2.1. These
controls bind the same pinned pack, inventory and production binary
as the [fresh delivery](../flash-next-mxfp4-current-delivery-m5-20261122/README.md).

All 21 actions pass across disabled, default and required modes. Each
group exercises normal requests, SSE, token budgets one and two, stop,
disconnect and recovery. Every checked counter drains to zero and
each owned server exits cleanly. CPU idle is 92.40% before the phase
and 97.33% after it.

| Mode | Lifecycle actions | HTTP ready on the staged pack |
| --- | ---: | ---: |
| Disabled | 7/7 | 59.322 s |
| Default | 7/7 | 52.263 s |
| Required | 7/7 | 51.245 s |

The lifecycle gate passes. The staged-pack timings describe these
launches. The fresh delivered-copy readiness receipt remains in force.

Replay the [raw lifecycle result](lifecycle.json) and compare the
[pinned gate result](comparison.json).

```bash
python3 benchmarks/results/flash-next-current-lifecycle-m5-20261122/replay.py
```
