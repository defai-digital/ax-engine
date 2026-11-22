# Flash Next mmap visitation with fixed allocation order

The follow-up uses the same two publisher shards on Mac Studio M2 Ultra
192 GB with an external APFS SSD and MLX 0.32.4. Both arms allocate and
touch the destinations in tensor-name order before mapping the file.
Only the order of source range visits varies during the timed copy.
Array construction and the byte and Metal checks use tensor-name order.
All eight arms match the full payload and allocation signatures and
record normal memory pressure and zero initially resident source pages.

| Shard | First arm in pair | Name-order copy | Physical-order copy |
| --- | --- | ---: | ---: |
| 21 | Name | 85.555 s | 25.590 s |
| 21 | Physical | 27.419 s | 85.077 s |
| 24 | Physical | 75.851 s | 47.510 s |
| 24 | Name | 26.607 s | 92.795 s |

Each shard shows a speedup in one process order and a slowdown in
the reverse order. This reversal persists after fixing destination
allocation and first-touch order. The study retains the name-order
product loader. The [raw receipt](result.json) records allocation and
touch time, copy time and total wall time separately.

Apply the [probe changes](fixed-order.patch) to the original experiment
sources to reconstruct the native probe and driver. The
[build receipt](build.json) binds the resulting source and executable.
The replay binds both published shards to the previous startup trace
and reproduces [comparison.json](comparison.json).

```sh
python3 benchmarks/results/flash-next-resident-mmap-order-m2-20261122/fixed-allocation/replay.py
```
