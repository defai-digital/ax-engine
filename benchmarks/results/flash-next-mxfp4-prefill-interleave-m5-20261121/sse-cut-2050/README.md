# Flash Next 2,050-token native split audit on M5 Max

This holdout appends a two-token tail after the 2,048-token logical
boundary. It uses the server, model revision, resident profile
and M5 hardware from the [2,048-token audit](../sse-cut-2048/README.md).

The [strict audit](audit.json) matches all 28 arrays of 96 output
IDs to same-build serial controls. Four requests carried their
own logical boundary across the 512, 512, 512, 511 and 1 physical
prefill widths while the peer continued decoding. The two-token
tail ran as one logical item. All six disconnect and replacement
controls passed. Measured pressure stayed normal.

| Nonserial observations | Split disabled | Split enabled |
| --- | ---: | ---: |
| Largest observed inter-token gap | 1.825 s | 0.597 s |
| Median combined output throughput | 25.68 tokens/s | 23.46 tokens/s |
| Median per-trial first-token latency | 2.134 s | 2.527 s |
| Peak sampled process footprint | 72.268 GiB | 72.132 GiB |

The timing comparison covers four nonserial trials per policy.
The [raw receipt](result.json.gz) and [holdout IDs](holdout-prompts.jsonl)
bind this result. Replay the same verifier and executable SHA-256
from the 2,048-token audit.
