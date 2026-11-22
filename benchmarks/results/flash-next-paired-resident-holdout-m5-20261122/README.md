# Flash Next resident-reader startup holdout on M5 Max

MacBook Pro M5 Max 128 GB, internal SSD, macOS 27.2.1, MLX and MLX
Metal 0.32.4. The holdout extends the
[original two-shard comparison](../flash-next-paired-resident-reads-m5-20261122/README.md)
to shards 17 and 18, which contribute to the consistently slower
resident-copy intervals in the
[startup attribution](../flash-next-resident-startup-attribution-m5-20261122/README.md).
The binary and opposing arm orders match the M2 holdout. The reader
fills two disjoint ranges of large tensors into the same type of
MLX allocation.

All 18 arms match the publisher checksums, selected 316 tensors,
allocation sizes and Metal outputs. The short-read control rejects
partial payloads. Every tested boundary has normal pressure and
host CPU idle passes at 91.49% before and 90.99% after. Before timed
population, each clone has zero selected pages resident and each
reused input has all selected pages resident.

| Input | Serial pread median | Paired pread median | Paired time change |
| --- | ---: | ---: | ---: |
| Shard 17, fresh clone | 456.34 ms | 455.99 ms | −0.08% |
| Shard 18, fresh clone | 451.43 ms | 450.00 ms | −0.32% |
| Shard 17, reused | 347.59 ms | 265.55 ms | −23.60% |
| Shard 18, reused | 342.84 ms | 258.49 ms | −24.60% |

On the selected startup-sensitive shards, the measured difference
on fresh clones is below 0.4%. The reused-input improvement supports
further cached I/O work and leaves the measured delivery readiness gap
open. Retain the existing resident loader policy.

The [comparison](comparison.json) and [raw result](result.json)
can be replayed with the committed paired-reader and M2 holdout
packages. The replay checks file residency, exact payload and Metal
results, arm order, pressure, and the native reader identity.

```sh
python3 -B benchmarks/results/flash-next-paired-resident-holdout-m5-20261122/replay.py
```
