# Qwen 3 + Gemma 4 vs mlxcel flip decision

Decision: **not_yet**

Candidate `ax-engine` vs baseline `mlxcel`; medians over 3 cache-isolated repetitions.

| Scenario | AX tok/s | mlxcel tok/s | Throughput ratio | TTFT ratio | Stream-gap ratio | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| S0 | 104.94 | 94.58 | 1.109x | 0.731x | 0.852x | FAIL |
| S1 | 5.66 | 19.70 | 0.288x | 3.541x | 23.718x | FAIL |
| S2 | 79.26 | 73.97 | 1.071x | 0.769x | 1.882x | FAIL |
| S3 | 91.86 | 109.98 | 0.835x | 4.039x | 1.408x | FAIL |

Locked gates:

- throughput ratio ≥ 1.15
- p95 TTFT ratio ≤ 0.90
- interactive p95 stream-gap ratio ≤ 0.90
- absolute interactive p95 stream gap ≤ 50.00 ms
- zero candidate request errors, HTTP 503s, and lifecycle errors

Failed gates:

- S0: median_throughput_ratio
- S1: median_throughput_ratio, median_ttft_p95_ratio, median_stream_gap_p95_ratio, absolute_stream_gap_p95
- S2: median_throughput_ratio, median_stream_gap_p95_ratio
- S3: median_throughput_ratio, median_ttft_p95_ratio, median_stream_gap_p95_ratio
