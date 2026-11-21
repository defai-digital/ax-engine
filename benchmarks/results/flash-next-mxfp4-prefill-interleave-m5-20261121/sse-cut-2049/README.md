# Flash Next 2,049-token native split audit on M5 Max

This holdout appends a one-token tail after the 2,048-token logical
boundary. It uses the exact server, model revision, resident
profile and M5 hardware from the [2,048-token audit](../sse-cut-2048/README.md).

The [strict audit](audit.json) matches all 28 arrays of 96 output
IDs to the same-build serial controls. Four requests carried
their own boundary across the 512, 512, 512, 511 and 1 physical
prefill widths while a peer was already decoding. The final prompt
token remained in its own logical item. All six disconnect and
replacement controls passed, and measured pressure stayed normal.

| Nonserial observations | Split disabled | Split enabled |
| --- | ---: | ---: |
| Largest observed inter-token gap | 1.755 s | 0.500 s |
| Median combined output throughput | 26.01 tokens/s | 24.80 tokens/s |
| Median per-trial first-token latency | 2.077 s | 2.327 s |
| Peak sampled process footprint | 72.274 GiB | 72.143 GiB |

The timing comparison covers four nonserial trials per policy.
The [raw receipt](result.json.gz) and [holdout IDs](holdout-prompts.jsonl)
bind this result independently of the 2,048-token case. Replay the
same verifier and executable SHA-256 from the 2,048-token audit.
