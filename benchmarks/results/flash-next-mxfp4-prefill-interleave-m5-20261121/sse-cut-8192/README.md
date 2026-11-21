# Flash Next 8,192-token native split audit on M5 Max

The long holdout uses the [same server, model revision, resident
profile and hardware](../sse-cut-2048/README.md). It completes the
four-length fixed-prompt test.

The [strict audit](audit.json) matched 28 arrays of 96 output IDs.
Four requests published their own exclusive boundaries while a
peer was already decoding. The observed logical ends were 2,048,
4,096, 6,144 and 8,192. All six disconnect and replacement controls
passed. Resource observations stayed at normal pressure.

| Nonserial observations | Split disabled | Split enabled |
| --- | ---: | ---: |
| Largest observed inter-token gap | 3.045 s | 0.828 s |
| Median of each trial's p95 inter-token gap | 0.049 s | 0.713 s |
| Median combined output throughput | 7.82 tokens/s | 7.72 tokens/s |
| Median per-trial first-token latency | 11.911 s | 13.075 s |
| Peak sampled process footprint | 76.106 GiB | 75.663 GiB |

Each arm has four nonserial trials. The lower largest gap came
with a substantial rise in the median trial p95 gap in this
workload. Sustained mixed arrivals remain a separate experiment.

The [raw receipt](result.json.gz) and [holdout IDs](holdout-prompts.jsonl)
bind this result. Replay the same verifier and executable SHA-256
from the 2,048-token audit.
