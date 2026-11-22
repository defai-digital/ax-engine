# Flash Next fresh delivery on the `38c90b27a` M5 production build

The native product CLI started with an empty cache on a MacBook Pro
M5 Max 128 GB with internal NVMe storage and macOS 27.2.1. The CLI
and server use the `release-server` profile, Rust 1.99.1 and MLX and
MLX Metal 0.32.4. The download took 1,110.398 seconds without a failed
attempt. Verification covered all 49 publisher files and 132,261,877,478
bytes of the pinned MXFP4 revision `ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3`.

| Mode | HTTP ready | Frozen limit | HTTP controls | Plan to runner admission |
| --- | ---: | ---: | ---: | ---: |
| Disabled, initial load | 108.096 s | 300 s | 7/7 | 102.197 s |
| Required | 126.269 s | 120 s | 7/7 | 119.005 s |
| Product default | 125.260 s | 120 s | 7/7 | 118.735 s |

All 21 request bodies, response choices, stable usage fields and
activation counters match the [prior production controls](../flash-next-mxfp4-current-delivery-m5-20261122/README.md).
Required verifies 107 steps and fails closed under the kill switch.
Disabled and default each verify zero steps. Replay retains the original
response bytes and excludes the two runtime durations from usage identity.

The delivery gate passes. Both warm starts exceed the frozen readiness
limit. The plan-to-admission interval accounts for most of each start.
TensorFold `f0cc0a134` separates model construction from opening its
listener. The corresponding log intervals here locate the duration before
AX runner admission without attributing it to any single loader phase.
All 54 passive pressure readings are normal. CPU-idle samples starting
within the two warm startup intervals range from 93.55% to 98.70%.
Lifecycle, staged QA, the native matrix and the independent reference have
their own receipts and release gates.

The [evidence archive](evidence.tar.xz) preserves the complete download
event sequence, the raw native-control result, the three timestamped
startup events per mode and the passive samples. The downloaded snapshot
path is redacted consistently in the two delivery records. The archive
keeps their original and portable digests.

```sh
shasum -a 256 -c SHA256SUMS
python3 -B replay.py | cmp comparison.json -
```
