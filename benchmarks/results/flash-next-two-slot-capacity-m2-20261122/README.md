# Flash Next native two-slot HTTP capacity on M2 Ultra

This control ran on a Mac Studio M2 Ultra 192 GB with external APFS
storage, macOS 27.2.1, Rust 1.99.1, and MLX and MLX Metal 0.32.4.
The [release-server build](build.json) came from `38c90b27a` and loaded
the pinned MXFP4 pack `ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3`.
Publisher inventory verification passed before and after the run.

TensorFold at `f0cc0a134` checks for a free lane and available memory
on admission and returns the lane's reservation when its work ends.
This control checks the corresponding externally visible boundary in
the AX Engine native HTTP server. The test enables two resident
request slots and physical prefill splitting with 32,768 total KV
tokens. MTP, n-gram acceleration, direct coalescing, and prefix reuse
are off. The Flash Next product profile remains one resident request
at a time on M5 Max 128 GB.

The server first runs the 512- and 8,192-token prompts serially, then
repeats the same burst twice in the same process. In each burst the
long request arrives at short output token 48. Its first 512-token
prefill cut shares a scheduler step with the short decode. At occupancy
two, a third client receives HTTP `429` with code `concurrency_limit`.
The replacement's prefill shares scheduler step 419 or 612 with the
long request after the original short stream completes.

All six completed requests match their serial controls exactly, totaling
576 generated IDs. Both bursts drain all five generation counters,
and `/health` stays healthy. All 119 observed memory pressure values
are normal. The evidence covers two short functional bursts on M2.
Primary-tier throughput, sustained concurrency, and one-slot product
readiness have separate gates.

Replay the [raw HTTP observations](evidence.tar.xz) and compare them with
the [recorded verdict](comparison.json). The script reuses the verified
prompts and native runner from the [early-arrival campaign](../flash-next-server-prefill-early-m2-20261122/README.md).

```sh
shasum -a 256 -c SHA256SUMS
python3 -B replay.py | cmp comparison.json -
```
