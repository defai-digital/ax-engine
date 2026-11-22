# Flash Next PLE worker pool through native HTTP on M5 Max

MacBook Pro M5 Max 128 GB, internal NVMe SSD, macOS 27.2.1, Rust
1.99.1, MLX and MLX Metal 0.32.4. The [same executable and sealed
driver as the M2 functional run](../flash-next-ple-serving-m2-20261122/README.md)
serve the pinned resident MXFP4 pack in the default one-slot, 32K
context profile. Both treatments configure 32 PLE reader threads.
The ordinary implementation uses scoped reads and the candidate
reports 31 persistent workers, with the caller doing the first chunk.

The five processes run a conditioning arm, then off/on/on/off. All
five match 31 prompt and output arrays, 3,968 generated token IDs
per process, and every stop state. Direct generation records zero
verifier steps and route errors. All 1,389 pressure samples show
normal memory pressure.

The first scored pair meets the 90% CPU idle floor on both sides.
Changes below compare the pool with the ordinary reader.

| Prompt tokens | TTFT change | Decode rate change |
| ---: | ---: | ---: |
| 512 | +1.47% | -1.38% |
| 2,048 | -1.32% | -1.37% |
| 8,192 | +4.40% | -0.004% |

The 8,192-token TTFT regression exceeds the existing 3% warmed
latency comparison band. The ordinary arm in reverse order closes
at 81.15% CPU idle. Replay retains its functional observations and
marks the pair timing-ineligible. Keep the current default, consistent
with the separate [serial and pooled staging
measurements](../flash-next-ple-reader-pool-m5-20261122/README.md).

The [comparison](comparison.json) replays the [terminal
receipt](result.json) and all five archived HTTP streams. Its readiness
observations use the already delivered snapshot and support the
reader diagnosis.

```sh
shasum -a 256 -c SHA256SUMS
python3 -B replay.py
```
