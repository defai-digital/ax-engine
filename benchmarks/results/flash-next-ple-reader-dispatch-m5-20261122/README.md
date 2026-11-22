# Flash Next native PLE reader dispatch on M5 Max

MacBook Pro M5 Max 128 GB, internal NVMe SSD, macOS 27.2.1, Rust
1.99.1, MLX and MLX Metal 0.32.4. The exact [M2 probe binary and
workload](../flash-next-ple-reader-dispatch-m2-20261122/README.md) ran in
1/32/32/1 thread order on one verified local view of the MXFP4 pack.
All four arms match the M2 payload digests, requested rows and read
volume across 128 PLE table shards. CPU idle is 90.69% before and
96.20% after, with normal memory pressure at all sampled boundaries.

| Reader threads | Initial decode p50 | Repeated decode p50 | Initial prefill p50 | Repeated prefill p50 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 4.459 ms | 0.0275 ms | 1711.053 ms | 13.534 ms |
| 32 | 0.199 ms | 0.1585 ms | 24.815 ms | 25.070 ms |
| 32 | 0.174 ms | 0.1609 ms | 25.372 ms | 25.225 ms |
| 1 | 0.0274 ms | 0.0185 ms | 13.358 ms | 12.387 ms |

The first process has a distinct initial cost on the reused view.
These timings leave startup page residency and process order
confounded. Both repeated decode and repeated prefill favor one thread
in each order. Keep the existing default and use the separate
persistent worker experiment to evaluate dispatch setup overhead.
This diagnostic measures native table reads and payload assembly.

The [comparison](comparison.json) replays every stage in the
[raw result](observed-m5.json.gz). Each decode stage contains 288
samples and each prefill stage contains four. P95 values are included
for reproducibility.

```sh
shasum -a 256 -c SHA256SUMS
python3 -B replay.py
```
