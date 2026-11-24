# Flash Next cold delivery with the patched URL dependency

The signed `1e8ea4845` CLI and server used Rust 1.99.1, `url` 2.5.9 and
MLX 0.32.4 on MacBook Pro M5 Max 128 GB, macOS 27.2.1 and internal SSD.
The CLI downloaded the pinned MXFP4 pack from an empty cache. The
runner verified all 49 files and 132,261,877,478 bytes in the publisher
inventory. The transfer took 1,020.286 seconds and recorded 5,078
progress counters, with no resumed or failed attempts.

After download and inventory verification, the launcher released the
shared slot while waiting for the HTTP controls. It sampled 90.34% CPU
idle at control admission. Readiness measures process start to HTTP
model-list availability.

| Mode | Context limit | HTTP readiness | Matched controls | Verified MTP steps |
| --- | ---: | ---: | ---: | ---: |
| Disabled | 32,768 | 32.231 s | 7 | 0 |
| Required | 16,384 | 106.479 s | 7 | 107 |
| Default | 32,768 | 33.101 s | 7 | 0 |

All three modes selected resident placement with one active request.
The 21 stable responses and verifier counters match the
[consolidated native control](../flash-next-consolidated-native-m5-20261124/README.md).
The incompatible Required override was rejected. The delivery gate
passes, and the three delivery launch measurements meet their recorded
readiness limits. Full qualification still needs the current-build
lifecycle, QA and timing matrix, plus the independent ordinary reference.

The portable archive retains the locked dependency resolution,
transport stream, signed-build hashes, HTTP control results and startup
logs. The source digest map records scratch-path normalization.

```sh
shasum -a 256 -c SHA256SUMS
python3.12 -E -s -B replay.py | cmp comparison.json -
```
