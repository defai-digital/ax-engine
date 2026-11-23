# Flash Next automatic reader after empty-cache delivery on M5

The signed `88c480c03` release-server build downloaded the pinned MXFP4
pack into an empty native cache on a MacBook Pro M5 Max 128 GB with an
internal NVMe SSD. The native delivery verified 49 publisher members
and 132,261,877,478 bytes. All 21 HTTP responses, stable usage fields,
and MTP activation counters match the previously recorded controls.
The reader environment override was absent in all three launches.

| MTP policy | HTTP ready | Prefetched bytes | CPU idle before / after | Verifier steps |
| --- | ---: | ---: | ---: | ---: |
| Disabled | 98.865 s | 5,214,301,696 | 97.11% / 92.23% | 0 |
| Required | 112.553 s | 5,214,301,696 | 98.35% / 93.10% | 107 |
| Default | 110.482 s | 5,214,301,696 | 98.43% / 89.95% | 0 |

All three launches meet the delivery readiness bound with normal memory
pressure. The default arm's final CPU sample falls below 90%, so the
campaign is ineligible for a timing comparison. The required activation
kill-switch control rejected before sidecar payload startup. The archive
covers the automatic delivery checkpoint. Explicit opt-out, continued
native testing, and the independent MLX-VLM reference are tracked
separately.

The portable evidence binds the native controls to the signed build and
replays all 4,805 recorded byte-progress events and the per-mode reader
markers. The downloader recorded only placeholder home paths.

```sh
shasum -a 256 -c SHA256SUMS
python3.12 -B replay.py | cmp comparison.json -
```
