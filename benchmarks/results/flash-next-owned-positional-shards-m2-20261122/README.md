# Flash Next reads into owned MLX storage on M2 Ultra

Mac Studio M2 Ultra 192 GB, external APFS SSD, macOS 27.2.1, MLX 0.32.4,
and Apple clang 21.0.0. Both diagnostic readers allocate MLX storage
and construct an available row-contiguous array over it. The mapped
arm byte-copies from its file mapping. The positional arm uses bounded
`pread` calls into the same kind of allocation. The prototype keeps
file cache enabled and publishes an array only after a complete read.

| Publisher shard | Fresh mapped | Fresh positional | Reused mapped | Reused positional |
| --- | ---: | ---: | ---: | ---: |
| 21 | 19.073 s | 3.902 s | 0.258 s | 0.198 s |
| 24 | 40.724 s | 4.088 s | 0.251 s | 0.199 s |

The second shard reverses each pair's order. Each fresh APFS clone has
zero initially resident selected pages. Both reused arms start with all
selected pages resident. All eight runs begin and end at normal sampled
pressure. Timers stop after filling and constructing the arrays, before
SHA-256 verification and the Metal probes.

Both readers match the independent source digest for all 266 selected
tensors and 3,220,919,680 bytes across the two shards. Both produce the
same allocator byte totals and the same Metal probe digests on each
shard. Shard 21 includes 41 offsets unaligned for the tensor dtype. The
prototype copies these as bytes. The partial-read fixture reaches EOF
after four of eight requested bytes and rejects the result before
array construction.

This single-shard experiment measures materialization on an external
SSD. Full-model startup, first request, warmed serving and M5 release
readiness remain separate measurements.

Replay the [raw result](result.json) against the pinned publisher
inventory and compare with [comparison.json](comparison.json).

```bash
python3 benchmarks/results/flash-next-owned-positional-shards-m2-20261122/replay.py
```
