# Flash Next two-client fixed-width repayment sweep

The sweep uses the [sealed M5 Max step traces](../flash-next-mxfp4-prefill-step-cost-m5-20261121/README.md)
of a 512-token request decoding while an 8,192-token peer prefills.
The replay holds each of the 20 measured prefill widths and runner
costs fixed. It tests repayment shares from 0% through 100% in one
percentage-point increments at p10, p50 and p90 observed decoder costs.
The final prefill slice is also the measured first output step of
the long request. Zero repayment reproduces that schedule and
11.016 seconds of runner time from the first short token to the first
long token.

| Decoder cost estimate | Largest share with a long first token before short completion | Added runner time for that first token | First share to reduce short p95 | Unstarted long slices at short completion |
| --- | ---: | ---: | ---: | ---: |
| p10 | 11% | 1.421 s | 25% | 8/20 |
| p50 | 12% | 1.529 s | 25% | 8/20 |
| p90 | 12% | 1.569 s | 26% | 8/20 |

Of 303 projections, none lowers the 735.99 ms runner-gap p95 while
also keeping the long request's first token before short completion.
The first p95 gain in each scenario is 7.03 ms. The maximum runner
gap stays at 747.38 ms in those cases. Repayment spaces expensive
slices, and each slice retains its measured cost.

These are fixed-cost projections for one pair of request arrivals.
The replay has no measured costs for the remaining solo prefill slices
after the short request completes, so it leaves that long first-token
time unestimated. The result supports keeping the current scheduler
policy. Before tuning the repayment knob in a live campaign, measure
another policy that reduces or avoids expensive shared intervals while
preserving the existing numerical controls and both clients' latency.

Recompute [comparison.json](comparison.json) from the original archive.

```bash
python3 benchmarks/results/flash-next-prefill-repayment-sweep-m5-20261122/replay.py
```
