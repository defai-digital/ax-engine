# Flash Next paired QSA prefill repeat with eligible M5 timing

MacBook Pro M5 Max 128 GB, internal SSD, macOS 27.2.1, MLX and MLX
Metal 0.32.4, Rust 1.99.1. The signed executable, verified 49-member
MXFP4 pack, and 8,192-token input match the
[original paired-gather probe](../flash-next-paired-qsa-prefill-20261123/README.md).
The warmed arms retain four repetitions of physical widths
`512, 512, 512, 511, 1` and run paired, stock, stock, paired.
Both paired arms exercise 432 Metal gather dispatches. All four
arms retain bitwise final logits and serialized state.

| Order | Stock model forwards | Paired model forwards | Change |
| --- | ---: | ---: | ---: |
| First | 8.9826 s | 8.5832 s | -4.45% |
| Reverse | 9.3627 s | 9.3028 s | -0.64% |
| Median | 9.1727 s | 8.9430 s | -2.50% |

CPU idle was 92.30% before and 90.31% after, both above the 90%
endpoint threshold. The Mach CPU trace records 91.20% idle across
the diagnostic process window. That window includes loading and
checks as well as timed forwards. All 168 pressure samples are
normal. The paired route raises peak MLX-tracked allocation by
`116,377,616` bytes (111.0 MiB) in both orders.

This result covers one fixed prompt and native model-forward latency.
The HTTP prefill crossover and a default decision retain separate
criteria. Prompt length, tile geometry and arithmetic remain fixed
in this probe.

The [capture](evidence.tar.xz) stores the original results, CPU ticks,
pressure measurements, pinned input files, and signed build digests.
The [comparison](comparison.json) includes per-chunk timings and the
exact order of the measured arms.

```sh
shasum -a 256 -c SHA256SUMS
python3.12 -B replay.py | cmp comparison.json -
```
