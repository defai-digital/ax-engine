# Flash Next full-model QSA tile evaluation under a physical memory guard

Mac Studio M2 Ultra 192 GB, macOS 27.2.1, external USB SSD, MLX
and MLX Metal 0.32.4, Rust 1.99.1. Test-only code loads the pinned
49-member resident MXFP4 pack and the frozen 8,192-token input from the
[mixed-step attribution](../flash-next-shared-step-clock-m5-20261123/README.md).
It compares lazy evaluation with evaluation after each unchanged
selected-attention tile. Every isolated arm starts a fresh process.
The signed executable and pack inventory are verified before the run
and the pack is checked again after the arms finish.

The 512-row plans use four sets of `512, 512, 512, 511, 1`.
The 2,048-row plans retain four sets of `2047, 1`. Both runs keep a
96 GiB physical-footprint guard and stop at warning or critical pressure.

| Arm | Sampled peak physical footprint | MLX peak | Result |
| --- | ---: | ---: | --- |
| 512 lazy | 90.53 GiB | 80.52 GiB | Completed |
| 512 serial | 90.12 GiB | 80.52 GiB | Completed, 432 evaluated tiles |
| 2,048 serial | 94.91 GiB | 89.36 GiB | Completed, 324 evaluated tiles |
| 2,048 lazy | 99.95 GiB | Unavailable | Guard terminated process |

The 512 arms have identical full-state and final-logit SHA-256 digests.
Peak MLX allocation is the same. The stock 512 arm starts at 84.31% CPU
idle, below the 90% timing threshold. The 2,048-row serial arm completes
within the guard. The lazy arm crosses the guard and has no terminal
state or logits, so the comparison supplies a capacity signal and
leaves 2,048-row state identity and comparative timing open.

All 918 isolated samples report normal pressure. Swap is unchanged
within each observed arm. An initial combined test stopped at
110.73 GiB with no per-arm receipts. That run records the guard
failure and cannot identify the responsible shape or evaluation mode.
The fresh-process crossover resolves the arm for this repeat.

The [synthetic grouping study](../flash-next-qsa-tile-eval-groups-20261123/README.md)
motivates testing two tiles per evaluation at wider shapes. This M2
result covers per-tile evaluation and one prompt. Primary-tier
confirmation, a serving comparison, and admission changes retain
separate criteria.

The [portable capture](evidence.tar.xz) includes both builds' source
receipts, pinned input files, raw pressure samples, and per-arm
outcomes. The executable is bound by its recorded SHA-256.
The [comparison](comparison.json) verifies the guard and the completed
exactness pair without attributing a terminal output to the failed arm.

```sh
shasum -a 256 -c SHA256SUMS
python3.12 -B replay.py | cmp comparison.json -
```
