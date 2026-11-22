# Flash Next startup trace with a different delivery manifest

MacBook Pro M5 Max 128 GB, internal NVMe SSD, macOS 27.2.1, Rust
1.99.1 and MLX 0.32.4. Instrumented server `c2358dd81` ran the pinned
MXFP4 payload on an earlier delivered snapshot with guarded mmap and
the startup trace enabled. The native harness verified the same
publisher inventory before and after all three modes.

| Mode | HTTP ready | Resident tensors | Shard copy and allocation | Sidecar |
| --- | ---: | ---: | ---: | ---: |
| Disabled | 81.338 s | 63.555 s | 63.541 s | 8.956 s |
| Required | 80.262 s | 61.293 s | 61.279 s | 8.786 s |
| Default | 64.323 s | 49.815 s | 49.799 s | 5.102 s |

All 21 response choices and activation counters matched the frozen
delivery controls. Required MTP verified 107 steps, default and disabled
verified zero, and the kill-switch control failed closed. The prelaunch
CPU idle was 90.50%. All 136 sampled pressure observations were normal.

The wrapper rejected the trace after functional completion. Its selected
snapshot had a different derived `model-manifest.json` from the frozen
delivery receipt. The manifests differ in one top-level field. The
selected snapshot includes a `dropped_tensors` provenance record with
count 972. The frozen manifest omits that field. The trace retains both
manifests and the failed phase status. The rejection occurred before the
post-run CPU idle check.

The trace locates most of the observed startup time in resident shard
copy and allocation on this snapshot. These timers include storage
access and cannot attribute their separate costs. The frozen default
delivery readiness remains 157.292 seconds against the 120-second gate.

The [sealed evidence](evidence.tar.xz) retains the raw functional
receipt, launch input digest manifest, both native manifests, phase logs,
and pressure samples. Replay against [comparison.json](comparison.json).

```bash
python3 benchmarks/results/flash-next-delivered-startup-phases-m5-20261122/replay.py
```
