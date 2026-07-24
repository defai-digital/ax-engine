# Qwen 3 + Gemma 4 vs mlxcel flip decision

Decision: **not_yet**

Candidate `ax-engine` vs baseline `mlxcel`; medians over 3 cache-isolated repetitions.

| Scenario | AX tok/s | mlxcel tok/s | Throughput ratio | TTFT ratio | Stream-gap ratio | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| S0 | 108.93 | 93.86 | 1.161x | 0.922x | 0.822x | FAIL |
| S1 | 14.15 | 19.59 | 0.722x | 1.394x | 0.778x | FAIL |
| S2 | 21.37 | 74.12 | 0.288x | 0.927x | 2.336x | FAIL |
| S3 | 88.99 | 110.09 | 0.808x | 6.891x | 1.852x | FAIL |

Locked gates:

- throughput ratio ≥ 1.15
- p95 TTFT ratio ≤ 0.90
- interactive p95 stream-gap ratio ≤ 0.90
- absolute interactive p95 stream gap ≤ 50.00 ms
- zero candidate request errors, HTTP 503s, and lifecycle errors

Failed gates:

- S0: median_ttft_p95_ratio
- S1: median_throughput_ratio, median_ttft_p95_ratio
- S2: median_throughput_ratio, median_ttft_p95_ratio, median_stream_gap_p95_ratio
- S3: median_throughput_ratio, median_ttft_p95_ratio, median_stream_gap_p95_ratio, absolute_stream_gap_p95
