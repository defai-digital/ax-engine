# Qwen 3 + Gemma 4 vs mlxcel flip decision

Decision: **not_yet**

Candidate `ax-engine` vs baseline `mlxcel`; medians over 3 cache-isolated repetitions.

| Scenario | AX tok/s | mlxcel tok/s | Throughput ratio | TTFT ratio | Stream-gap ratio | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| S0 | 107.98 | 94.85 | 1.138x | 0.729x | 0.877x | FAIL |
| S1 | 6.79 | 19.46 | 0.349x | 2.914x | 0.273x | FAIL |
| S2 | 79.64 | 74.09 | 1.075x | 0.755x | 1.926x | FAIL |
| S3 | 74.82 | 110.61 | 0.676x | 16.867x | 1.967x | FAIL |

Locked gates:

- throughput ratio ≥ 1.15
- p95 TTFT ratio ≤ 0.90
- interactive p95 stream-gap ratio ≤ 0.90
- absolute interactive p95 stream gap ≤ 50.00 ms
- zero candidate request errors, HTTP 503s, and lifecycle errors

Failed gates:

- S0: median_throughput_ratio
- S1: median_throughput_ratio, median_ttft_p95_ratio
- S2: median_throughput_ratio, median_stream_gap_p95_ratio
- S3: median_throughput_ratio, median_ttft_p95_ratio, median_stream_gap_p95_ratio, absolute_stream_gap_p95
