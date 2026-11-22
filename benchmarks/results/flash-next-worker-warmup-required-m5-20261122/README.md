# Flash Next Required MTP worker warmup on M5 Max

MacBook Pro M5 Max 128 GB, internal NVMe SSD, macOS 27.2.1, Rust
1.99.1, MLX and MLX Metal 0.32.4. The same executable, resident
one-slot 32K profile and verified MXFP4 snapshot continue from the
[completed default phase](../flash-next-worker-warmup-default-m5-20261122/README.md).
The complete eight-process campaign exits successfully and rechecks
the publisher inventory.

All four Required processes match 17 prompt/output trajectories,
2,176 generated IDs, stop states and each request's canonical MTP
report. Each process executes 1,266 verifier steps and records zero
route errors. The three/one/one/three worker passes keep the three
HTTP warmup streams.

| Process | Worker passes | Worker warmup | HTTP readiness | CPU idle before and after |
| --- | ---: | ---: | ---: | ---: |
| Required 1 | 3 | 2.579 s | 117.102 s | 93.90%, 89.10% |
| Required 2 | 1 | 1.830 s | 125.003 s | 94.85%, 92.89% |
| Required 3 | 1 | 1.795 s | 137.905 s | 92.80%, 80.61% |
| Required 4 | 3 | 2.521 s | 118.007 s | 97.37%, 98.19% |

Each paired order contains an arm that ends below the 90% CPU idle
floor. The retained-copy observations support functional parity.
The comparison emits timing deltas for the eligible default pairs
and marks both Required pairs ineligible. Keep the current worker
startup policy and the unchanged fresh-delivery readiness criterion.

The [comparison](comparison.json) replays the [terminal
campaign and Required observations](evidence.tar.xz) together with
the original default-phase raw archive.

```sh
shasum -a 256 -c SHA256SUMS
python3 -B replay.py
```
