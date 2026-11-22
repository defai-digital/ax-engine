# Flash Next MXFP4 startup phases on M5 Max

MacBook Pro M5 Max 128 GB, internal SSD, macOS 27.2.1, Rust 1.99.1 and
MLX 0.32.4. The `426f64829` DEBUG executable ran the pinned delivered
MXFP4 snapshot at `ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3` with
resident experts and one active request. All 28 HTTP controls matched
the prior pinned receipt and all four runs recorded zero MTP verifier
steps. Both modes initialize the packaged MTP sidecar at model load.

| Sequence | MTP policy | HTTP ready | Shard read and allocation | Sync eval | MTP sidecar |
| --- | --- | ---: | ---: | ---: | ---: |
| Disabled first | Disabled | 79.349 s | 58.984 s | 0.461 ms | 10.896 s |
| Disabled first | Default | 99.447 s | 79.667 s | 0.179 ms | 10.625 s |
| Default first | Default | 84.358 s | 64.622 s | 0.352 ms | 10.645 s |
| Default first | Disabled | 108.491 s | 88.603 s | 0.163 ms | 11.403 s |

Each process materialized the same 1,942 tensors across 15 shards. The
read and allocation timer accounts for 74% to 82% of HTTP readiness.
It includes allocation and copying. The separate sync eval timer is
under one millisecond in every start. The second start took longer in
each order and system-wide page-in bytes were higher. Cache state is
uncontrolled. The run identifies where elapsed time accumulated.
It leaves the relative contributions of storage, allocation and other
system activity open.

The replay keeps the trunk-admission lap separate from the subsequent
wall interval, since the memory-plan line falls inside that lap. It
counts shard timers inside the resident-tensor phase once. The
[sealed evidence](evidence.tar.xz) retains four checked startup traces,
the original phase and HTTP receipts, VM samples, the earlier loader
selection and the prior HTTP control. The [audit](comparison.json)
includes the original phase receipt SHA-256.

The instrumented starts are diagnostic. The frozen ordinary delivery
evidence still records 157.292 seconds against a 120-second readiness
limit. A separate, controlled [loader comparison](../flash-next-native-loader-m5-20261121/README.md)
found faster startup with a lazy CPU reader and higher TTFT at 2K and
8K. The next comparison separately measures first request and warmed
requests before selecting a runtime change.

```bash
python3 benchmarks/results/flash-next-mxfp4-startup-phases-m5-20261121/replay.py
```
