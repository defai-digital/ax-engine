# Qwen 3 + Gemma 4 vs mlxcel flip decision

Decision: **not_yet**

Candidate `ax-engine` vs baseline `mlxcel`; medians over 3 cache-isolated repetitions.

| Scenario | AX tok/s | mlxcel tok/s | Throughput ratio | TTFT ratio | Stream-gap ratio | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| S0 | 109.46 | 94.32 | 1.161x | 0.847x | 0.852x | PASS |
| S1 | 14.73 | 19.45 | 0.758x | 1.328x | 0.801x | FAIL |
| S2 | 109.39 | 74.07 | 1.477x | 0.999x | 0.766x | FAIL |
| S3 | 90.21 | 110.35 | 0.818x | 7.391x | 1.838x | FAIL |

Locked gates:

- throughput ratio ≥ 1.15
- p95 TTFT ratio ≤ 0.90
- interactive p95 stream-gap ratio ≤ 0.90
- absolute interactive p95 stream gap ≤ 50.00 ms
- zero candidate request errors, HTTP 503s, and lifecycle errors

Failed gates:

- S1: median_throughput_ratio, median_ttft_p95_ratio
- S2: median_ttft_p95_ratio
- S3: median_throughput_ratio, median_ttft_p95_ratio, median_stream_gap_p95_ratio, absolute_stream_gap_p95
