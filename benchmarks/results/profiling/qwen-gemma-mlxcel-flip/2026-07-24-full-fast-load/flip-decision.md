# Qwen 3 + Gemma 4 vs mlxcel flip decision

Decision: **not_yet**

Candidate `ax-engine` vs baseline `mlxcel`; medians over 3 cache-isolated repetitions.

| Scenario | AX tok/s | mlxcel tok/s | Throughput ratio | TTFT ratio | Stream-gap ratio | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| S0 | 108.05 | 93.80 | 1.152x | 1.179x | 0.797x | FAIL |
| S1 | 14.82 | 19.94 | 0.743x | 1.354x | 0.775x | FAIL |
| S2 | 72.77 | 74.09 | 0.982x | 0.827x | 2.272x | FAIL |
| S3 | 90.26 | 110.38 | 0.818x | 7.422x | 1.838x | FAIL |

Locked gates:

- throughput ratio ≥ 1.15
- p95 TTFT ratio ≤ 0.90
- interactive p95 stream-gap ratio ≤ 0.90
- absolute interactive p95 stream gap ≤ 50.00 ms
- zero candidate request errors, HTTP 503s, and lifecycle errors

Failed gates:

- S0: median_ttft_p95_ratio
- S1: median_throughput_ratio, median_ttft_p95_ratio
- S2: median_throughput_ratio, median_stream_gap_p95_ratio
- S3: median_throughput_ratio, median_ttft_p95_ratio, median_stream_gap_p95_ratio, absolute_stream_gap_p95
