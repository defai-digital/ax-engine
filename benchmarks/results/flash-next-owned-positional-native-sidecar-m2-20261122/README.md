# Flash Next required-MTP positional sidecar smoke on M2 Ultra

Mac Studio M2 Ultra 192 GB, external SSD, macOS 27.2.1, Rust 1.99.1
and MLX 0.32.4. Both controls use `8204ff924` and the pinned installed
MXFP4 pack. The harness verifies all delivered members before launch.
The first process uses the mapped sidecar reader. The second enables
the descriptor-bound positional reader. The trunk uses the same mapped
path in both processes. Initial cache residency is uncontrolled.

| Sidecar route | HTTP ready | Resident trunk | Sidecar phase |
| --- | ---: | ---: | ---: |
| Mapped | 27.279 s | 10.350 s | 2.007 s |
| Positional | 241.429 s | 226.205 s | 0.868 s |

Both routes produce the same response choices and activation counters
for all seven required-MTP checks, including 107 verified steps each.
The kill switch refuses required activation with the positional reader
enabled. Pressure is normal before and after both runs. The logs retain
the same 15 resident shards and 1,942 trunk tensors.

The sidecar phase is shorter in this sample. The unchanged trunk phase
accounts for the large readiness difference between processes. These
timings support the functional route check and motivate an isolated
M5 comparison with recorded initial page residency.

Replay [comparison.json](comparison.json) against the
[sealed evidence](evidence.tar.xz).

```bash
python3 benchmarks/results/flash-next-owned-positional-native-sidecar-m2-20261122/replay.py
```
