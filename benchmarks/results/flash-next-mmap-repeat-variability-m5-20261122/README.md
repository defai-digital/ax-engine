# Flash Next warmed request variability with the same M5 mmap reader

This replays the 512-token-first mmap controls from the two
[read-ahead campaigns](../flash-next-mmap-read-ahead-m5-20261122/README.md).
Both controls used the same server executable, publisher files, initial
selected-page observations and one-request resident 32K profile on the
MacBook Pro M5 Max 128 GB. One ran first in its campaign and the other
ran second. Each matches all 448 output IDs, stops and route fields.
Both have normal sampled pressure and pass the CPU-idle guard.

| Warm prompt | Original TTFT | Reversed-campaign TTFT | TTFT change | Decode change |
| --- | ---: | ---: | ---: | ---: |
| 512 tokens | 0.522 s | 0.513 s | −1.78% | −0.15% |
| 2,048 tokens | 1.643 s | 1.764 s | +7.34% | +1.81% |
| 8,192 tokens | 8.987 s | 9.750 s | +8.49% | −0.12% |

The unchanged mmap reader varies beyond the 3% TTFT limit across these
two campaigns. The data shows the size of one observed between-run
difference. It cannot attribute the difference to process order,
temperature, page residency outside the sample, or any single cause.
The original campaign's long-first mmap run is timing-ineligible and
contributes no measurements here.

Use the observed spread to plan same-reader repeats with fixed workload
and optional resource observations. The aligned reader's measured
regressions and the 3% promotion guard still apply.

```sh
shasum -a 256 -c SHA256SUMS
python3 -B replay.py | cmp comparison.json -
```
