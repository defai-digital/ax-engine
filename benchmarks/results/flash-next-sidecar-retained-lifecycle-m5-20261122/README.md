# Flash Next retained-delivery lifecycle with sidecar pre-read

MacBook Pro M5 Max 128 GB, internal NVMe SSD, macOS 27.2.1, Rust
1.99.1, and MLX and MLX Metal 0.32.4. Both arms use the signed
release-server from `eb57f768a` with the pinned MXFP4
pack at `ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3`. Independent APFS
clones share one verified retained delivery. The mapped arm runs
first. Each arm runs the disabled and default modes in that order.

| Reader | Disabled HTTP ready | Default HTTP ready | CPU idle before / after |
| --- | ---: | ---: | ---: |
| Mapped | 117.486 s | 123.539 s | 96.32% / 96.80% |
| Prefetch | 97.901 s | 119.778 s | 98.68% / 98.47% |

All 28 lifecycle actions match the
[frozen HTTP control](../flash-next-native-lifecycle-38c90b27a-m5-20261122/README.md).
All five work counters drain, both modes record zero verifier steps,
and the four bounding memory-pressure readings are normal. Each
pre-read process reads 5,214,301,696 bytes, taking
0.395 s and 0.400 s.

Default HTTP readiness is 3.761 seconds shorter and finishes
0.222 seconds within the 120-second staged limit. This
is one measured order on retained storage. The option remains opt-in
pending an eligible empty-cache delivery comparison. The complete
release board also requires the pinned ordinary MLX-VLM reference.

The archive retains the exact staged inputs, lifecycle receipts and
sidecar route markers. Run the replay on macOS with Python 3.12+.

```sh
shasum -a 256 -c SHA256SUMS
python3.12 -B replay.py | cmp comparison.json -
```
