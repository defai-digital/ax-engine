# Flash Next sidecar pre-read on the delivered M5 Max pack

A MacBook Pro M5 Max 128 GB with internal NVMe SSD and macOS 27.2.1
ran the pinned, verified resident MXFP4 snapshot with the exact
codesign-verified binary from the [M2 native pre-read run](../flash-next-native-sidecar-prefetch-m2-20261122/README.md).
The diagnostic used Rust 1.99.1 and MLX and MLX Metal 0.32.4.

All 21 controls match the [original delivery evidence](../flash-next-fresh-delivery-38c90b27a-m5-20261122/README.md).
Required MTP records 107 verifier steps. The Required plus
`AX_NO_SPEC` control fails with the expected activation error and no
pre-read. The three positive arms read exactly 5,214,301,696 bytes
each, followed by the existing mapped materializer.

| Policy | Historical HTTP ready | Pre-read diagnostic ready | Resident trunk | Sidecar stage, including pre-read | Pre-read | CPU idle before / after |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Disabled MTP | 108.096 s | 109.413 s | 98.452 s | 1.724 s | 0.400 s | 92.79% / 79.99% |
| Required MTP | 126.269 s | 92.364 s | 81.267 s | 1.320 s | 0.392 s | 92.70% / 83.37% |
| Default | 125.260 s | 101.426 s | 90.101 s | 1.755 s | 0.419 s | 93.78% / 81.65% |

The Required and default observations fall within the 120-second
band. All six pressure samples are normal. All three arms miss the
90% CPU idle floor at their after-run sample. The historical
measurement uses a different executable. The initial page cache
reflects ordinary system state in each run. These observations establish functional
parity on the M5 delivered pack. Performance attribution and a default
decision require the same-executable paired startup and warmed
serving evidence.

The archive preserves the native responses, phase logs, build and
host samples. Replay checks the original delivery results, M2
build and source patch, and the mode, route and timing observations.

```sh
shasum -a 256 -c SHA256SUMS
python3 -B replay.py | cmp comparison.json -
```
