# Flash Next retained-delivery lifecycle in reverse order

MacBook Pro M5 Max 128 GB, internal NVMe SSD, macOS 27.2.1, Rust
1.99.1, and MLX and MLX Metal 0.32.4. This run uses the signed
`eb57f768a` executable, frozen MXFP4 pack and
lifecycle controls from the
[initial retained-delivery pair](../flash-next-sidecar-retained-lifecycle-m5-20261122/README.md).
Independent APFS clones keep the pre-read arm first in this repeat.

| Reader | Disabled HTTP ready | Default HTTP ready | CPU idle before / after |
| --- | ---: | ---: | ---: |
| Prefetch | 112.552 s | 112.540 s | 97.86% / 92.92% |
| Mapped | 121.059 s | 120.980 s | 90.18% / 91.47% |

All 28 HTTP lifecycle actions match the frozen output and usage
controls, including the five drained counters. Both modes in both
arms record zero verifier steps.

Both arms meet the CPU idle and normal-pressure requirements.
Default readiness with pre-read is 8.440 seconds shorter in the reverse order.
The pre-read arm meets the 120-second staged limit in both modes.
The product decision still requires an eligible empty-cache delivery
and the release board retains the pinned ordinary MLX-VLM comparison.

Replay on macOS with Python 3.12 or newer.

```sh
shasum -a 256 -c SHA256SUMS
python3.12 -B replay.py | cmp comparison.json -
```
