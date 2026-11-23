# Flash Next explicit two-slot admission on M5 Max and M2 Ultra

The codesign-verified `56e2b1a79` candidate runs the resident MXFP4
snapshot at `ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3` on MacBook Pro
M5 Max 128 GB and Mac Studio M2 Ultra 192 GB, both on macOS 27.2.1 and
MLX 0.32.4. The isolated build recompiles the changed engine core. Native
HTTP exercises the explicit 2,048-token logical and 512-token physical
prefill profile with MTP, coalescing and prefix restoration off.

The [previous signed reader](../flash-next-guarded-reader-fresh-m5-20261123/README.md)
reproduces early admission on M2. The second 8,192-token request becomes
Runnable at zero progress while the first holds the lane. With the new
server, the second request stays Waiting through the first request's
2,048- and 4,096-token checkpoints. A third submitted request receives
`429 concurrency_limit`. Cancelling the first admits the second at its
original 2,048-token boundary. An eligible 512- and 8,192-token pair
shares a step of one decode token and 512 prefill tokens.

All 288 measured token IDs on each host match that host's own
uncontended outputs. Both cases drain the five work counters. M5 records
143 normal pressure samples and 75.525 GiB peak physical footprint. M2
records 110 normal samples and 85.894 GiB peak footprint. Both retain
the mapped reader. This scope covers experimental admission safety.
Sustained throughput and product concurrency qualification have separate
evidence requirements. The ordinary product profile remains one request.

```sh
shasum -a 256 -c SHA256SUMS
python3.12 -B replay.py | cmp comparison.json -
```
