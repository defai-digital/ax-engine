# Flash Next automatic reader startup phases on M5 Max

This retained-pack launch uses the same signed current-source candidate and
publisher inventory as the
[automatic-reader crossover](../flash-next-current-auto-reader-crossover-m5-20261123/README.md).
The MacBook Pro M5 Max 128 GB uses the internal SSD, macOS 27.2.1, Rust
1.99.1, and MLX and MLX Metal 0.32.4. The instrumented server admits one
resident sequence with MTP off. All 49 publisher members and the derived
manifest are checked before and after serving. All seven stable API
actions match the current-main oracle with zero final verifier steps.

| Measurement | Time |
| --- | ---: |
| Process launch to HTTP readiness | 131.700 s |
| Resident tensor phase | 120.336 s |
| Shard reads and MLX allocation within that phase | 120.321 s |
| Synchronous shard evaluation within that phase | 0.348 ms |
| MTP sidecar phase with automatic pre-read | 1.445 s |

The trace binds 1,942 selected tensors over the same 15 publisher shards,
for 68,516,576,280 selected bytes. The resident phase accounts for 91.37%
of readiness and exceeds the 120-second staged bound on its own. CPU idle
is 97.45% before and 93.43% after. The system-wide VM interval from before
launch through the lifecycle actions records 57.41 GiB of page-ins, 3.36
MiB of decompressions, and zero page-outs or swap-outs. These counters
include other activity during the interval. The launch identifies resident
reads and allocation as the measured startup cost worth testing next.

The replay independently parses the raw shard and phase markers, checks
publisher selection against the frozen geometry, and verifies the stable
API actions against the current-main receipt.

```sh
shasum -a 256 -c SHA256SUMS
python3.12 -B replay.py | cmp comparison.json -
```
