# Qwen 3 + Gemma 4 vs mlxcel flip decision

Decision: **not_yet**

Candidate `ax-engine` vs baseline `mlxcel`; medians over 3 cache-isolated repetitions.

| Scenario | AX tok/s | mlxcel tok/s | Throughput ratio | TTFT ratio | Stream-gap ratio | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| S0 | 109.67 | 93.82 | 1.169x | 0.755x | 0.802x | PASS |
| S1 | 14.28 | 19.54 | 0.731x | 1.377x | 0.807x | FAIL |
| S2 | 21.49 | 74.00 | 0.290x | 0.921x | 2.309x | FAIL |
| S3 | 89.78 | 110.49 | 0.813x | 7.561x | 1.841x | FAIL |

Locked gates:

- throughput ratio ≥ 1.15
- p95 TTFT ratio ≤ 0.90
- interactive p95 stream-gap ratio ≤ 0.90
- absolute interactive p95 stream gap ≤ 50.00 ms
- zero candidate request errors, HTTP 503s, and lifecycle errors

Failed gates:

- S1: median_throughput_ratio, median_ttft_p95_ratio
- S2: median_throughput_ratio, median_ttft_p95_ratio, median_stream_gap_p95_ratio
- S3: median_throughput_ratio, median_ttft_p95_ratio, median_stream_gap_p95_ratio, absolute_stream_gap_p95
