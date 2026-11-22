# Flash Next MTP sidecar owned-storage reads on M2 Ultra

Mac Studio M2 Ultra 192 GB, external APFS SSD, macOS 27.2.1 and
MLX 0.32.4. The comparison uses the same executable as the
[two-shard owned-storage trial](../flash-next-owned-positional-shards-m2-20261122/README.md),
but selects all 31 BF16 tensors in the pinned publisher sidecar.
It independently verifies the publisher file and the selected payload.

| File state | Mapped copy | Positional read |
| --- | ---: | ---: |
| Fresh APFS clones, two runs | 27.329–27.940 s | 2.122–2.130 s |
| Reused file, two runs | 1.222–1.238 s | 0.598–0.599 s |

The fresh clones start with zero resident selected pages. The reused
file starts with all 318,256 selected pages resident. Each group
counterbalances its two read orders. All eight arms reproduce the
same 5,214,301,696 selected bytes, allocator size and Metal probe
digest, with normal pressure before and after each process.

The timings cover filling MLX-owned storage and constructing arrays.
Native head normalization, attachment, verifier execution and M5
required-MTP readiness are separate measurements. The descriptor-bound
native prototype keeps the guarded mmap route as its default.

The [raw receipt](result.json) replays against the pinned pack
inventory and [comparison.json](comparison.json).

```bash
python3 benchmarks/results/flash-next-owned-positional-sidecar-m2-20261122/replay.py
```
