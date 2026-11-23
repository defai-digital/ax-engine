# Guarded Flash Next AX versus the bounded MLX-VLM reference

This offline replay compares the signed guarded-reader AX build
`c1a7c3f9dfb8` with the
[bounded reference](../flash-next-affine8-reference-m5-20261121/README.md).
Both measured runs use MacBook Pro M5 Max 128 GiB, internal NVMe SSD,
macOS 27.2.1, and the pinned AXQ MXFP4 pack at
`ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3`.
The same frozen QA and timing prompts are used. AX uses MLX 0.32.4.
The reference uses MLX 0.32.2 and MLX-VLM `6102cb4a` with
bounded positional reads for affine8 PLE. The measurements came
from separate campaigns with normal pressure and eligible CPU endpoints.

The replay regrades all 210 AX responses and the 105 reference answers.
Both AX direct modes pass 101 items and match in text. The reference
passes 100 items. Four failures are shared. The remaining failure,
`format_csv_pair`, is reference-only. The bounded reference reproduces
all 105 signed ordinary QA rows and grades and all 15 signed timing
trajectories.

| Input tokens | AX default TTFT | Bounded reference TTFT | AX default decode | Bounded reference decode |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 0.527 s | 2.197 s | 48.64 tok/s | 38.31 tok/s |
| 2,048 | 1.837 s | 5.646 s | 45.79 tok/s | 37.14 tok/s |
| 8,192 | 10.510 s | 15.149 s | 44.19 tok/s | 37.15 tok/s |

Across both native arms, all six numeric comparisons meet the
existing ratio limits. AX TTFT is 0.24–0.69 times the bounded reference,
and decode is 1.19–1.29 times its rate. AX uses loopback streaming,
while the reference runs in process. The original signatures,
measurements, and profile labels are retained. The approved ordinary
reference remains required for a release verdict.

The replay validates both native archives and the bounded-to-signed
parity chain before using the release evaluator's grading and numeric
formulas. It does not evaluate or emit a release verdict.

```sh
shasum -a 256 -c SHA256SUMS
python3.12 -B replay.py | cmp comparison.json -
```
