# Flash Next buffered resident reader on M5 Max

One signed release-server executable tests the earlier aligned reader patch
on source `bde3b0319`. The MacBook Pro M5 Max 128 GB uses the internal SSD,
macOS 27.2.1, Rust 1.99.1, and MLX and MLX Metal 0.32.4. Each process serves
the pinned Flash Next MXFP4 pack in the ordinary resident 32K single-slot
profile with MTP off. All 49 publisher files are checked before and after
each arm. The 14-request schedule starts with a 512-token prompt in every
process and uses an independent APFS clone for each launch.

| Arm | HTTP ready | Resident load | CPU idle before / after |
| --- | ---: | ---: | ---: |
| Mapped first | 172.461 s | 149.243 s | 97.31% / 96.32% |
| Buffered first | 30.091 s | 8.222 s | 96.80% / 95.94% |
| Buffered repeat | 30.089 s | 8.034 s | 94.64% / 96.86% |

The eligible first pair reaches HTTP readiness 142.370 seconds sooner
with aligned buffered reads. Warmed median TTFT changes by +1.05%, −0.85%,
and −1.18% at 512, 2,048, and 8,192 input tokens. Warmed decode changes by
−0.04%, −0.54%, and −0.84% at the same lengths. All six measures pass the
existing 3% band. The second buffered start repeats readiness within
0.003 seconds of the first. Its warmed TTFT differs by at most 0.38% and
warmed decode by at most 1.66% across the three lengths.

Each arm reproduces all 448 [signed M5 output IDs](../flash-next-aligned-native-default-m5-20261122/README.md),
the response and route fields, both health checks, and the 32K overflow
rejection. Pressure remains normal in all 109 samples. The independent
reference became eligible before the final mapped arm. The original
owner exited before admitting that arm and the three completed receipts
remain sealed. The reverse pair and any default change remain open.

The same candidate source passes four focused checks on Mac mini M4
Pro 64 GiB with macOS 27.0 and the same MLX 0.32.4 library. These cover
unaligned tensor values and ownership, short read and range rejection,
reader eligibility, and mapped tensor selection.

```sh
shasum -a 256 -c SHA256SUMS
python3.12 -B replay.py | cmp comparison.json -
```
