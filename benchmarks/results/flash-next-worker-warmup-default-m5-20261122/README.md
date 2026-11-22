# Flash Next default worker warmup on M5 Max

MacBook Pro M5 Max 128 GB, internal NVMe SSD, macOS 27.2.1, Rust
1.99.1, MLX and MLX Metal 0.32.4. The [same experimental binary
as the M2 run](../flash-next-worker-warmup-once-m2-20261122/README.md)
serves the retained publisher-verified MXFP4 snapshot with the
resident one-slot 32K profile and product-default MTP policy.

The completed default phase uses three/one/one/three worker passes.
It matches all 17 response trajectories and 2,176 output IDs per
process. Stop reasons and routes match. All four CPU idle samples
at each boundary pass the 90% floor, and all 887 sampled pressures
are normal. Each process performs the three HTTP warmup streams.

| Process | Worker passes | Plan to worker start | Worker phase | HTTP readiness |
| --- | ---: | ---: | ---: | ---: |
| Default 1 | 3 | 96.579 s | 2.541 s | 102.494 s |
| Default 2 | 1 | 113.711 s | 1.375 s | 117.911 s |
| Default 3 | 1 | 109.032 s | 1.401 s | 112.923 s |
| Default 4 | 3 | 110.984 s | 3.016 s | 117.095 s |

The single pass reduces the worker phase by 1.166 and 1.615 seconds.
Total readiness changes by +15.417 and -4.172 seconds in the two
paired orders. The interval before worker warmup accounts for most
of the opposing results. First-request TTFT increases by 2.47% and
2.31%, respectively.

Retain the existing startup policy. This [comparison](comparison.json)
covers the default phase on the retained snapshot. The Required phase
has separate observations, and the fresh-delivery board uses its
original [delivery receipt](../flash-next-mxfp4-current-delivery-m5-20261122/README.md).

The [phase archive](evidence.tar.xz) preserves the original interim
progress receipt and all four raw HTTP and startup records.

```sh
shasum -a 256 -c SHA256SUMS
python3 -B replay.py
```
