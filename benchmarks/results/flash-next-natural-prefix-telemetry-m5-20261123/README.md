# Flash Next natural prefix learning with L1 telemetry on M5 Max

Four fresh processes served the pinned Flash Next MXFP4 pack on a
MacBook Pro M5 Max 128 GB with macOS 27.2.1, an internal SSD, Rust
1.99.1, and MLX and MLX Metal 0.32.4. The campaign repeats the
[previous M5 workload](../flash-next-natural-prefix-crossover-m5-20261123/README.md)
in off, on, on, off order with the same signed executable and seven
unseeded cases. This repeat also retains the existing shared-cache
gauges from the request's SSE store reports.

All four arms match the 1,921 token IDs from the same-host cold oracle.
The enabled arms record three expected 2,048-token hits each. The
publisher inventory verifies all 49 pack members before and after.

The first pair meets both endpoint CPU idle floors. The last disabled
arm ends at 69.35% idle against the 90% floor, leaving the full timing
crossover ineligible. The previous campaign's eligible reversed pair
and this campaign's eligible forward pair remain independent results.

| First pair, target first-token latency | Off | On |
| --- | ---: | ---: |
| Second question with cold prefill | 2.083 s | 2.111 s |
| Third question on the same document | 3.068 s | 1.520 s |
| Related 2,048-token suffix | 4.351 s | 2.630 s |
| Second and third questions, sum | 5.151 s | 3.630 s |

The second-question reports list eight charged entries and
1,170,506 KiB of shared L1 in all four arms. The store count rises
from one to two in the enabled arms. These are aggregate shared-cache
values associated with the request's reported step. The replay also
retains separate whole-process RSS and physical-footprint readings.

```sh
shasum -a 256 -c SHA256SUMS
python3.12 -B replay.py | cmp comparison.json -
python3.12 -B l1_usage.py evidence.tar.xz | cmp l1-usage.json -
```
