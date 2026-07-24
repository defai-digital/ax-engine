# Qwen 3 + Gemma 4 vs mlxcel flip decision

Decision: **not_yet**

Candidate `ax-engine` vs baseline `mlxcel`; medians over 3 cache-isolated repetitions.

| Scenario | AX tok/s | mlxcel tok/s | Throughput ratio | TTFT ratio | Stream-gap ratio | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| S0 | 108.11 | 94.76 | 1.141x | 0.754x | 0.866x | FAIL |
| S1 | 6.33 | 19.57 | 0.323x | 3.146x | 0.265x | FAIL |
| S2 | 79.92 | 73.97 | 1.080x | 0.772x | 2.008x | FAIL |
| S3 | 80.73 | 110.28 | 0.732x | 15.103x | 1.874x | FAIL |

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
