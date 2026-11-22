# Flash Next resident mmap copy order on M2 Ultra

Mac Studio M2 Ultra 192 GB, external APFS SSD, macOS 27.2.1 and
MLX and MLX Metal 0.32.4. The two publisher MXFP4 shards each contain
158 tensors. The [instrumented M5 default startup](../flash-next-delivery-bound-startup-m5-20261122/README.md)
materialized all 158 tensors from each shard. Their observed copy
and allocation costs were 9.325 and 10.541 seconds.

One probe executable uses guarded mapped copies to owned MLX storage
in both arms. It visits the tensor ranges in either name or physical
byte-offset order and allocates each destination during that visit.
The probe verifies the delivered manifest and the complete publisher
hashes, uses each of the two process orders per shard, and checks
full payload bytes, allocated sizes and Metal output in canonical
name order. Every clone started with zero resident selected pages.
All eight arms recorded normal memory pressure before and after.

| Shard | First arm in pair | Name-order copy | Physical-order copy |
| --- | --- | ---: | ---: |
| 21 | Name | 24.379 s | 27.011 s |
| 21 | Physical | 90.397 s | 27.161 s |
| 24 | Physical | 79.269 s | 26.671 s |
| 24 | Name | 32.405 s | 33.203 s |

Physical order improves one pair per shard. In the other pairs it
increases time by 10.8% and 2.5%. The copy loop varies source page
visits and the order of MLX destination allocation together. Keep
name-order production loading and isolate these effects before an
M5 integration decision.

Replay checks the [outcome](result.json) against the original M5
resident startup trace and the pinned publisher inventory. It
reproduces the [comparison](comparison.json) and binds the native
[build inputs](build.json).

```sh
python3 benchmarks/results/flash-next-resident-mmap-order-m2-20261122/replay.py
```
