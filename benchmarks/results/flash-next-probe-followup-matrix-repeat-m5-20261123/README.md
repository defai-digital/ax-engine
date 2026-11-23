# Flash Next storage-probe follow-up timing repeat on M5 Max

The signed `26c0ccefe` release-server binary repeats the retained matrix
on the MacBook Pro M5 Max 128 GB with internal NVMe SSD, macOS 27.2.1,
and MLX 0.32.4. It binds the same executable and publisher inventory
as the [first matrix](../flash-next-probe-followup-retained-m5-20261122/README.md)
and [fresh delivery](../flash-next-probe-followup-fresh-m5-20261123/README.md).
All 124 signed output arrays match. Each of the five server launches
uses the resident one-request path, automatically pre-reads 5,214,301,696
bytes, and reaches HTTP readiness. Both CPU idle samples qualify at
97.99% before and 91.73% after.

| Input tokens | Default median TTFT | Default median decode |
| ---: | ---: | ---: |
| 512 | 0.546 s | 47.89 tokens/s |
| 2,048 | 1.626 s | 45.92 tokens/s |
| 8,192 | 10.012 s | 44.36 tokens/s |

The passive observer starts during the fourth process. It observes
85.00% weighted idle over 310.35 seconds of startup and generation
intervals, and 86.93% over 45.07 seconds after the last measured cell.
These samples characterize part of the run. The frozen evaluator uses
the same before and after samples as the original matrix.

Replaying the release board with this eligible repeat keeps six gates
passing. Integrity now depends on the independent reference's CPU
samples, and the same reference must complete independent quality and
relative timing. The matrix keeps MTP inactive throughout.

```sh
shasum -a 256 -c SHA256SUMS
python3.12 -B replay.py | cmp comparison.json -
```
