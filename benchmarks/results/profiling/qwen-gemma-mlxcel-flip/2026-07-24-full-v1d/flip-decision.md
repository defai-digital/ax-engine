# Qwen 3 + Gemma 4 vs mlxcel flip decision

Decision: **not_yet**

Candidate `ax-engine` vs baseline `mlxcel`; medians over 3 cache-isolated repetitions.

| Scenario | AX tok/s | mlxcel tok/s | Throughput ratio | TTFT ratio | Stream-gap ratio | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| S0 | 110.45 | 94.75 | 1.166x | 0.748x | 0.827x | PASS |
| S1 | 6.34 | 19.26 | 0.329x | 3.088x | 0.259x | FAIL |
| S2 | 80.82 | 73.97 | 1.093x | 0.901x | 2.262x | FAIL |
| S3 | 82.95 | 110.29 | 0.752x | 14.794x | 1.842x | FAIL |

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
