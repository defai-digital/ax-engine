# Flash Next main-branch lifecycle on M5 Max

The signed and unpatched `5c43cf896` release-server build used the pinned
Flash Next MXFP4 pack on a MacBook Pro M5 Max 128 GB with its internal SSD,
macOS 27.2.1, Rust 1.99.1, and MLX and MLX Metal 0.32.4. The executable is
byte-identical to the [prefix switch controls](../flash-next-natural-prefix-switch-controls-m5-20261123/README.md).
All 49 publisher members and the derived manifest were checked before
and after serving.

| MTP mode | HTTP ready | Stable actions matched | Final verifier steps |
| --- | ---: | ---: | ---: |
| Ordinary default | 142.775 s | 7/7 | 0 |
| Explicitly disabled | 126.782 s | 7/7 | 0 |

The two resident runs match each other and the
[previous signed lifecycle](../flash-next-probe-followup-lifecycle-m5-20261122/README.md)
in response text, finish reason, token counts, activation and drainage.
Both servers shut down cleanly. CPU idle was 97.20% before and 97.17%
after the cohort. Both starts exceed the staged 120-second readiness
bound, and the startup logs show the mapped sidecar route without
a pre-read. These launches used the retained pack and two fresh
single-slot processes. The earlier startup observations with
automatic pre-read are from a separate signed branch.

The canonical lifecycle runner completed and emitted the raw result.
A supplemental comparison also included request timing fields and
returned status 1. The replay compares stable fields using the prior
signed verifier and checks both modes against the raw lifecycle gate.

```sh
shasum -a 256 -c SHA256SUMS
python3.12 -B replay.py | cmp comparison.json -
```
