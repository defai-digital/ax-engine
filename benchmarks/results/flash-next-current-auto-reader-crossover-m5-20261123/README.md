# Flash Next automatic sidecar reader crossover on M5 Max

Four fresh processes used one signed release-server binary built from
`5c43cf896` and the sealed auto-reader patch. The MacBook Pro M5 Max
128 GB used its internal SSD, macOS 27.2.1, Rust 1.99.1, and MLX and
MLX Metal 0.32.4. Each process served the retained Flash Next MXFP4
pack in the resident single-slot profile. Each arm verified all
49 publisher members and the derived manifest before and after serving.

| Order | Reader | HTTP ready | CPU idle before / after | Stable API actions |
| --- | --- | ---: | ---: | ---: |
| 1 | Explicitly mapped | 144.622 s | 91.37% / 89.46% | 7/7 |
| 2 | Automatic pre-read | 144.714 s | 94.48% / 91.54% | 7/7 |
| 3 | Automatic pre-read | 130.687 s | 90.77% / 96.72% | 7/7 |
| 4 | Explicitly mapped | 141.760 s | 97.44% / 95.92% | 7/7 |

Both automatic starts pre-read 5,214,301,696 bytes. All 28 actions match
the [current-main lifecycle](../flash-next-main-lifecycle-m5-20261123/README.md)
in stable response and drainage fields. Every server exits cleanly
with zero final MTP verifier steps.

The final CPU-idle sample in the first arm falls below 90%. The
reverse-order pair qualifies and automatic pre-read reaches HTTP
readiness 11.073 seconds earlier, a 7.81% reduction. Both automatic
starts exceed the 120-second retained readiness bound. The automatic
reader remains isolated and the readiness gate remains open for this
build.

The original on-host inputs and the archived receipts reproduce the
comparison with Python 3.12.

```sh
shasum -a 256 -c SHA256SUMS
python3.12 -B replay.py | cmp comparison.json -
```
