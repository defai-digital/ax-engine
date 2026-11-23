# Flash Next cross-step PLE read-ahead feasibility on M2 Ultra

Mac Studio, Apple M2 Ultra, 192 GiB unified memory, internal SSD, macOS
27.2.1, and MLX and MLX Metal 0.32.4. The signed test executable uses
Rust 1.99.1 and builds from `6a5d97bbc` with the
[recorded test-only patch](source.patch.gz). All 49 publisher files and
the derived manifest match before and after the run.

MTPLX's one-span lookahead motivated this diagnostic. It prepares the
next 512-token physical prefill at the end of each 2,048-token
scheduler slice. It claims the result through AX's existing exact
row-ID and layout checks. The control keeps within-slice lookahead.
Four distinct 8,192-token prompts from the frozen mixed workload run
in paired AB, BA, AB and BA orders. Both arms use the same runner and
a 32-token output budget.

| Measure | Control | Cross-step candidate |
| --- | ---: | ---: |
| Ready matches at observed boundaries | 0 of 12 | 12 of 12 |
| Median host-side gather at those boundaries | 40.190 ms | 8.420 ms |

All four pairs match their output and stop reason, totaling 56
matched IDs. Total native prefill changes by about 0.02% to 0.80% in
the four pairs. Input and pack reuse can warm pages, so these
measurements show feasibility within this M2 run. A primary-tier
long-context comparison needs measured initial page residency and
a meaningful end-to-end first-token reduction before the scheduler
carries future spans. The test leaves the shipping route as it was.

```sh
shasum -a 256 -c SHA256SUMS
python3.12 -B replay.py | cmp comparison.json -
```
