# Flash Next positional MTP sidecar reads on M5 Max

Four fresh internal-SSD clones use one instrumented `8204ff924`
release-server binary with Rust 1.99.1 and MLX 0.32.4. The experiment
changes only the sidecar payload reader. It verifies the publisher
inventory before and after all four arms, checks the delivered model
manifest and keeps the guarded mmap trunk. All four clones start with
zero of the 318,256 selected sidecar file pages resident.

| Policy and reader | Sidecar stage | Resident trunk stage | HTTP readiness | CPU idle after |
| --- | ---: | ---: | ---: | ---: |
| Required with mmap | 12.624 s | 154.822 s | 177.889 s | 90.94% |
| Required with positional reads | 0.525 s | 82.892 s | 92.768 s | 92.25% |
| Default with positional reads | 0.593 s | 85.203 s | 94.741 s | 91.94% |
| Default with mmap | 11.832 s | 86.082 s | 106.981 s | 87.10% |

All 28 responses and activation counters match the fresh production
controls by policy. Each required arm records 107 verifier steps and
each default arm records zero. All route error counters stay at zero.
The previous [M2 test](../flash-next-owned-positional-native-sidecar-m2-20261122/README.md)
exercises the same binary and the required-MTP kill switch.

The isolated sidecar stage takes roughly 11–12 s less with positional
reads. The required-arm wall difference also includes a 72 s change
in trunk loading. The final default mmap arm falls below the original
90% post-run CPU-idle floor, so the full timing comparison remains a
diagnostic. The controls exercise the native response and activation
contracts. A fresh product readiness and sustained serving comparison
is still needed for a default-reader decision.

The [raw receipt](receipt.json.gz) and [replayed comparison](comparison.json)
retain the exact controls and original startup measurements.

```bash
python3 benchmarks/results/flash-next-owned-positional-native-sidecar-m5-20261122/replay.py
```
