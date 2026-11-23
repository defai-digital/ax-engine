# Flash Next logit selection component measurements

MLX 0.32.4 routes topk through its Metal partition implementation.
That implementation merge-sorts the full row. This probe compares
the sort-based Flash Next reads with selections built from the same
MLX argmax. The bonus probe gathers the selected logit. The correction
probe masks that index, selects and gathers the remaining maximum,
and falls back to the original sort for unusual magnitudes.

Each capture measures 60 materialized F32 rows at width 248,320 after
eight warmups and checks 45 adversarial inputs. Bonus equivalence
requires the same token and finite-value decision. Correction
equivalence additionally requires identical margin bits. The selected
gather and scatter variants match their observables in all captures.
The raw mask alternative demonstrates margin differences on subnormals.

| Capture | Sort-based bonus | Gather bonus | Sort-based correction | Scatter correction |
| --- | ---: | ---: | ---: | ---: |
| M2 Ultra 192 GB | 521.56 µs | 285.94 µs | 556.54 µs | 383.73 µs |
| M5 Max 128 GB eligible repeat | 597.19 µs | 280.21 µs | 636.35 µs | 375.38 µs |

Both devices run macOS 27.2.1. The first M5 component run ends at
80.80% CPU idle. The repeat records 95.80% before and 98.18% after.
The table reports median component wall times. Native serving has
additional work outside these component reads.

~~~sh
shasum -a 256 -c SHA256SUMS
python3.12 -B replay.py | cmp comparison.json -
~~~
