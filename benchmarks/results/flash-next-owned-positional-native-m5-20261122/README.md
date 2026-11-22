# Flash Next resident positional reads on M5 Max

The four-arm M5 Max 128 GB experiment uses the same instrumented
`8729b80c2` release-server binary, the pinned MXFP4 pack and a fresh
internal-SSD APFS clone for each arm. The comparison reverses reader
order between the 512-token and 8,192-token first-request arms. It
verifies all 49 publisher files before and after observation. Each arm
serves the ordinary one-request 32K resident profile, passes both
health checks and rejects an overflowing request. CPU idle exceeds
90% before and after each arm, and sampled pressure stays normal.

| First prompt | Mapped readiness | Positional readiness | Mapped trunk load | Positional trunk load |
| --- | ---: | ---: | ---: | ---: |
| 512 tokens | 86.221 s | 22.060 s | 67.511 s | 7.288 s |
| 8,192 tokens | 42.170 s | 22.100 s | 29.357 s | 7.256 s |

All four arms match all 448 greedy output IDs per arm with zero MTP
verifier steps and zero route errors. Positional reads reduce the time
from process start to first token from 87.691 to 22.881 s in the
short-first pair and from 51.404 to 32.423 s in the long-first pair.

| First prompt | Warm 512 TTFT change | Warm 2,048 TTFT change | Warm 8,192 TTFT change |
| --- | ---: | ---: | ---: |
| 512 tokens | +1.28% | +7.54% | +7.66% |
| 8,192 tokens | -1.71% | +9.23% | +6.23% |

All decode medians meet the 3% band. Four of six warmed TTFT cells
exceed the +3% band. This keeps positional reads experimental despite
the large startup improvement. Ready-state MLX active allocations agree
within 256 bytes. The probe records file-page residency and process
memory but does not attribute the warmed TTFT changes to individual
prefill stages or PLE lookups. The delivered-copy release readiness
gate retains its separate production receipt.

The [raw receipt](receipt.json.gz) retains the original process
and request records and exact binary and runner digests. Replay the
[comparison](comparison.json) through the fixed workload and warm bands.

```bash
python3 benchmarks/results/flash-next-owned-positional-native-m5-20261122/replay.py
```
