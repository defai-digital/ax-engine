# Flash Next scheduler-facing natural EOS controls

One signed Rust 1.99.1 release-server test executable runs the pinned
resident MXFP4 pack on Mac Studio M2 Ultra 192 GB with external SSD and
MacBook Pro M5 Max 128 GB with internal NVMe SSD. Both use macOS 27.2.1
and MLX 0.32.4. The harness verifies the 49 publisher files and derived
manifest before and after each run. All 35 M2 and 38 M5 pressure samples
are normal.

| Frozen QA workload | Input tokens | Generated tokens including EOS | Final MTP decision |
| --- | ---: | ---: | --- |
| Sort numbers | 35 | 11 | Bonus EOS |
| Bracketed list | 34 | 8 | Correction EOS |
| Long lookup | 29,774 | 5 | Bonus EOS |

Each direct and MTP path stops at its first configured terminal token,
`248046`, before the 64-token budget. The 24 output IDs match across
both routes and devices. The native test checks that each request's
state is released before continuing and checks all verifier decisions
against the same-state host argmax. Ten drafts are accepted per device.
Replay requires the correction and bonus termination paths.

The controls disable the cost controller to exercise MTP. They measure
natural greedy stopping and request cleanup. The product's MTP
performance and default-promotion gates remain open.

The pinned software suite passes 4,788 Rust and 345 Python tests,
strict Clippy, the script checks, and both qualification dry runs.
[Software receipt](software.json).

```sh
shasum -a 256 -c SHA256SUMS
python3.12 -B replay.py | cmp comparison.json -
```
