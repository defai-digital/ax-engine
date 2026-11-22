# Flash Next paired resident-shard reads on M5 Max

MacBook Pro M5 Max 128 GB, internal SSD, MLX and MLX Metal 0.32.4.
The comparison uses the exact reader and pinned MXFP4 selection from the
[M2 functional control](../flash-next-paired-resident-reads-m2-20261122/README.md).
The pinned MLX reader's positional batches motivated this experiment.
It runs serial and paired positional reads into the same kind of
allocator-owned storage. Paired mode splits each tensor of at least
64 MiB into two disjoint ranges before creating the MLX array.

Both shards are tested with opposing serial/paired orders. Every
arm matches the original selected payload SHA-256, allocation,
array count and Metal output for all 266 selected tensors across
the two shards. The short-read negative control passes in both
modes. All 18 arms report normal pressure before and after the read.
CPU idle is 98.72% before and 98.36% afterward. The clone arms have
zero resident pages before each read, and the reused arms have every
selected page resident. The mmap sample is a reused-input control.

| Shard and input | Serial pread median | Paired pread median |
| --- | ---: | ---: |
| 21, fresh clone | 188.68 ms | 184.95 ms |
| 24, fresh clone | 186.16 ms | 177.31 ms |
| 21, reused | 138.92 ms | 115.41 ms |
| 24, reused | 134.81 ms | 107.59 ms |

Paired reads take about 2–5% less time on fresh clones and 17–20% less
time on reused inputs. The timed scope is isolated shard population
under the shared allocation and tensor order. The holdout targets the
shards with repeatably higher startup copy costs. Whole-model readiness,
warmed serving, and the reader default remain separate decisions.

The [raw result](result.json) and [comparison](comparison.json) replay
against the earlier M2 source, reader build and exact selection.

```sh
python3 -B benchmarks/results/flash-next-paired-resident-reads-m5-20261122/replay.py
```
