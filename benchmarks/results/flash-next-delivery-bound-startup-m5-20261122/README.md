# Flash Next startup on the frozen delivered snapshot

MacBook Pro M5 Max 128 GB, internal NVMe SSD, macOS 27.2.1, Rust 1.99.1
and MLX 0.32.4. This diagnostic uses instrumented server `c2358dd81`
with guarded mmap and repeats the three original control modes in
their original order. The pinned publisher inventory and the derived
manifest match the frozen delivery receipt. The launcher checks the
manifest before startup. The native harness checks the inventory and
manifest throughout the controls.

| Mode | HTTP ready | Resident tensors | Shard copy and allocation | Sidecar |
| --- | ---: | ---: | ---: | ---: |
| Disabled | 102.478 s | 83.085 s | 83.071 s | 10.470 s |
| Required | 136.833 s | 115.935 s | 115.918 s | 10.966 s |
| Default | 109.064 s | 88.296 s | 88.282 s | 11.634 s |

All 21 response choices and activation counters match the frozen
controls. Required MTP verified 107 steps. Default and disabled each
verified zero. The kill-switch control failed closed. CPU idle was
91.70% before and 96.63% after the run. All 184 pressure samples were
normal.

Default readiness in this trace is 109.064 seconds, below the existing
120-second limit. About 81% of that observation is resident shard copy
and allocation, which includes storage access. The instrumented server
and repeat launch differ from the frozen fresh-delivery run. Its
157.292-second default readiness remains the release gate input.

The [sealed evidence](evidence.tar.xz) retains the matched manifest, raw
functional receipts, phase logs and VM samples. Reproduce
[comparison.json](comparison.json) from the repository root.

```bash
python3 benchmarks/results/flash-next-delivery-bound-startup-m5-20261122/replay.py
```
