# Flash Next automatic reader lifecycle on M5 Max

The signed `88c480c03` release-server binary ran against the pinned
MXFP4 checkpoint on MacBook Pro M5 Max 128 GB, internal SSD, and
macOS 27.2.1. It shares the executable and publisher inventory with the
[fresh delivery](../flash-next-auto-reader-fresh-m5-20261122/README.md)
and [frozen QA](../flash-next-auto-reader-qa-m5-20261122/README.md).

Both modes match the seven prior HTTP actions in stable response fields.
Baseline requests, SSE, one- and two-token budgets, stop, unfinished
stream disconnect, and recovery all pass. The five generation counters
drain after every action, and both server processes exit cleanly.

| MTP mode | HTTP ready | Memory plan to prefetch completion | Runner admission to HTTP listening |
| --- | ---: | ---: | ---: |
| Disabled | 129.887 s | 120.782 s | 5.241 s |
| Default | 108.667 s | 98.951 s | 5.505 s |

The observed disabled startup exceeds the 120-second staged readiness
bound. CPU idle fell from 92.51% before the lifecycle to 87.29% after it.
The observations are ineligible for a timing comparison. Both actual
sidecar pre-reads take less than half a second. The earlier interval
includes resident weight processing, so it cannot identify the allocation
or I/O operation that accounts for the observed spread.

The replay verifies the action order, stable responses, readiness values,
cleanup, and the startup intervals from both raw logs.

```sh
shasum -a 256 -c SHA256SUMS
python3.12 -B replay.py | cmp comparison.json -
```
