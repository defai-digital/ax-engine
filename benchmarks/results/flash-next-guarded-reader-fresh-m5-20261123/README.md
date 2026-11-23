# Flash Next guarded reader fresh delivery on M5 Max

The signed `d2220c26f` server and its matching product CLI delivered the
pinned Flash Next MXFP4 pack at `ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3`
from an empty cache on MacBook Pro M5 Max 128 GB with internal SSD,
macOS 27.2.1 and MLX 0.32.4. The transfer took 1,390.139 seconds and
recorded 7,504 progress counters. The original delivery runner verified
all 49 publisher files and 132,261,877,478 bytes of content, checked the
native model manifest, and rechecked the publisher inventory after the
functional controls. There were no resumed or failed transfer attempts.

| Native mode | Context tokens | HTTP readiness | HTTP controls | Verified MTP steps |
| --- | ---: | ---: | ---: | ---: |
| Disabled | 32,768 | 30.212 s | 7 | 0 |
| Required | 16,384 | 114.947 s | 7 | 107 |
| Default | 32,768 | 30.213 s | 7 | 0 |

The 21 stable responses and verifier counters match the prior signed
[delivery corpus](../flash-next-probe-followup-fresh-m5-20261123/README.md).
The incompatible Required override still fails before activation. All
three modes meet the frozen readiness limits. The runtime patch also
passes the software gates on the main integration tree, including 4,807
Rust and 345 Python tests. The signed
[automatic crossover](../flash-next-automatic-buffered-reader-m5-20261123/README.md)
and [retained Required run](../flash-next-stock-required-buffered-candidate-m5-20261123/README.md)
cover the same server. Full release qualification requires the independent
reference, retained QA, lifecycle and generation matrix.

```sh
shasum -a 256 -c SHA256SUMS
python3.12 -B replay.py | cmp comparison.json -
```
