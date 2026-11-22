# Flash Next resident native refresh on M5 Max

MacBook Pro M5 Max, 128 GB, internal NVMe SSD, macOS 27.2.1, Rust
1.99.1, MLX and MLX Metal 0.32.4. The release-server build is from
`fd85d821f`. This run uses an APFS view of the retained publisher
snapshot, verifies all 49 pinned files and binds the exact MXFP4
manifest. The ordinary profile advertises one resident request and
32,768 total context tokens.

The native runner passes 21 HTTP controls. The explicit bypass and
Required MTP conflict fails before any observed native payload phase,
in 0.506 seconds. Both direct QA modes complete 105 requests and
retain 101 passing checks each. The replay matches all 210 raw QA
records against the signed v7.9.0 native outputs, stop reasons and
usage totals. Both direct lifecycle processes pass seven actions
each, drain all five counters and exit cleanly.

Four generation processes reproduce all 124 signed matrix output
arrays. All measured direct rows report zero verifier steps and route
errors. QA CPU idle meets the timing floor at 95.90% before and 98.75%
afterward. Lifecycle CPU idle is 91.98% and 86.47%. Matrix CPU idle is
91.83% and 85.70%. Timing from the lifecycle and matrix observations
is ineligible under the 90% floor.

| Resident snapshot readiness | Seconds |
| --- | ---: |
| Disabled | 120.489 |
| Required | 140.952 |
| Product default | 118.823 |

The separate [fresh delivery](../flash-next-mxfp4-current-delivery-m5-20261122/README.md)
and [release board](../flash-next-current-native-board-m5-20261122/README.md)
retain their measured executable and gates. The reused-snapshot
default result and native identity checks leave the release verdict
at five of nine gates.

The [evidence archive](evidence.tar.xz) includes the measured driver,
pinned inventory, staged inputs, raw QA rows, all matrix repeats and the
online comparison. The [comparison](comparison.json) can be independently
replayed after checking SHA-256. Changing a raw matrix token triggers
failure even when the recorded summary remains intact.

```sh
shasum -a 256 -c SHA256SUMS
python3 -B replay.py
```
