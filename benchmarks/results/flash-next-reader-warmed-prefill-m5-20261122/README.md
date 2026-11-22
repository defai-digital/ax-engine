# Flash Next matched reader prefill on M5 Max

MacBook Pro M5 Max 128 GB and internal NVMe SSD. These four arms use
the [same Rust 1.99.1 and MLX 0.32.4 native probe](../flash-next-reader-warmed-prefill-m2-20261122/README.md)
at `3130d9d12`, pinned MXFP4 publisher inventory and deterministic
prompt IDs. Each process uses a fresh APFS clone and a common 512-token
warmup. The two pairs reverse the reader order and the order of the
2,048- and 8,192-token measurements. Between the two measured periods,
each arm runs two extra unmeasured 8,192-token prefills.

| Order | Prompt tokens | Mmap, first period | Positional, first period | Mmap, second period | Positional, second period |
| --- | ---: | ---: | ---: | ---: | ---: |
| Short first | 2,048 | 1.701 s | 1.833 s | 1.799 s | 1.829 s |
| Short first | 8,192 | 9.023 s | 9.588 s | 9.981 s | 9.562 s |
| Long first | 2,048 | 1.669 s | 1.701 s | 1.772 s | 1.763 s |
| Long first | 8,192 | 8.782 s | 9.221 s | 9.245 s | 9.378 s |

The original reader geometry, first greedy token and serialized state
match for both lengths in every arm. All 48 untraced measurements
agree with the corresponding first token from their eight traced
passes. System pressure stays normal and CPU idle exceeds 90% before
and after every process.

The 8,192-token positional median is 4.99–6.26% slower in the first
period. In the second period it ranges from 4.20% faster to 1.45% slower.
The mmap arms also slow down between periods. Elapsed time and added
warmups therefore vary together. The traced MoE compute difference
changes sign between the two 8K pairs, and PLE row time changes by
less than 13 ms in either 8K pair. The data does not isolate a stage
that explains the warm serving regression. The existing
[full-trunk serving result](../flash-next-owned-positional-native-m5-20261122/README.md)
continues to govern the full-trunk reader. The
[sidecar-only serving result](../flash-next-sidecar-reader-warm-m5-20261122/README.md)
has a separate reader and its own warm serving measurements.

Replay the [raw receipt](receipt.json) and compare against
[comparison.json](comparison.json).

```bash
python3 benchmarks/results/flash-next-reader-warmed-prefill-m5-20261122/replay.py
```
