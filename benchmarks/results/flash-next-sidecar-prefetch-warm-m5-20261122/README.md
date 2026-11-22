# Flash Next sidecar pre-read and warm serving on M5 Max

MacBook Pro M5 Max 128 GB, internal NVMe SSD, macOS 27.2.1,
Rust 1.99.1, and MLX and MLX Metal 0.32.4. Each arm uses a fresh
APFS clone of the same 49-file pinned MXFP4 pack and the signed
executable from the [delivered-copy diagnostic](../flash-next-sidecar-prefetch-delivery-m5-20261122/README.md).
The one-request resident profile advertises 32,768 total tokens.
Greedy outputs follow the [existing 448-token oracle](../flash-next-sidecar-reader-warm-m5-20261122/README.md).

| Run order | Sidecar reader | First prompt | HTTP ready | Resident trunk | Sidecar stage | CPU idle before / after |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | Mapped | 512 | 164.390 s | 142.879 s | 11.747 s | 90.91% / 91.93% |
| 2 | Pre-read, mapped copy | 512 | 88.343 s | 77.188 s | 1.001 s | 93.20% / 97.64% |
| 3 | Pre-read, mapped copy | 8,192 | 94.442 s | 85.318 s | 0.943 s | 98.84% / 95.79% |
| 4 | Mapped | 8,192 | 106.279 s | 84.823 s | 12.167 s | 98.84% / 98.35% |

Both pre-read arms read exactly 5,214,301,696 bytes. The pre-read takes
0.398 and 0.394 seconds and is included in the sidecar stage. The
65.691-second trunk difference accounts for most of the short-first
readiness difference. In the long-first pair, the trunk times differ
by 0.496 seconds and the sidecar stage is 11.224 seconds shorter.

All 56 responses match the prior oracle's 1,792 output IDs across
the four arms. Health checks and the 32K completion and overflow
controls pass. The 176 pressure samples are normal and each arm meets
the 90% CPU idle floor. The two pairs meet the existing 3% bands for
warmed TTFT and decode at all three prompt lengths.

| First prompt | Warm prompt | TTFT change with pre-read | Decode rate change with pre-read |
| --- | ---: | ---: | ---: |
| 512 | 512 | −1.77% | +3.64% |
| 512 | 2,048 | −1.75% | +1.90% |
| 512 | 8,192 | −5.17% | +1.55% |
| 8,192 | 512 | +0.55% | +0.58% |
| 8,192 | 2,048 | −0.71% | +0.28% |
| 8,192 | 8,192 | −3.18% | +5.41% |

The [software and M4 check receipt](software-and-m4.json) binds
experimental source commit `7265da7ca` to the exact tested patch.
It records 4,787 passing Rust tests, 345 passing Python tests,
the full script run, both qualification dry runs, and all 15
supported-tier loader and switch inventory checks.

The pre-read is opt-in. A default decision requires the original
empty-cache delivery and release gates. The archive retains all
native warm observations, startup phase logs, and sampled pressure.

```sh
shasum -a 256 -c SHA256SUMS
python3 -B replay.py | cmp comparison.json -
```
