# Flash Next storage-probe follow-up fresh delivery on M5 Max

The signed `26c0ccefe` release-server binary completed a new delivery on
MacBook Pro M5 Max 128 GB with internal NVMe SSD, macOS 27.2.1, and
MLX 0.32.4. The pinned Flash Next MXFP4 pack at
`ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3` started from an empty cache.
The download verified 49 publisher files totaling 132,261,877,478 bytes
in 1,001.417 seconds without a failed attempt. Replay verifies all
5,194 transfer progress events against the signed inventory.

| Native mode | HTTP readiness | CPU idle before / after | Verifier steps |
| --- | ---: | ---: | ---: |
| Disabled | 94.949 s | 97.13% / 98.92% | 0 |
| Required | 112.088 s | 98.27% / 95.72% | 107 |
| Default | 112.099 s | 98.43% / 99.10% | 0 |

All 21 native HTTP controls reproduce the signed outputs and activation
counters. The reader switch is absent. Each mode automatically pre-reads
5,214,301,696 bytes of sidecar data. Pressure remains normal and both CPU
idle samples exceed the 90% floor in every mode. The first post-delivery
start meets the readiness gate.

The [retained matrix and mapped opt-out](../flash-next-probe-followup-retained-m5-20261122/README.md)
use the same signed executable. The separate
[paired startup experiment](../flash-next-sidecar-fresh-delivery-pair-m5-20261122/README.md)
measures the pre-read speedup against the mapped path on its own exact build.

```sh
shasum -a 256 -c SHA256SUMS
python3.12 -B replay.py | cmp comparison.json -
```
