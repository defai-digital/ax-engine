# Qwen 3 + Gemma 4 vs mlxcel flip decision

Decision: **not_yet**

Candidate `ax-engine` vs baseline `mlxcel`; medians over 3 cache-isolated repetitions.

| Scenario | AX tok/s | mlxcel tok/s | Throughput ratio | TTFT ratio | Stream-gap ratio | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| S0 | 72.37 | 94.71 | 0.764x | 4.016x | 1.147x | FAIL |
| S1 | 6.46 | 19.65 | 0.329x | 3.095x | 0.982x | FAIL |
| S2 | 68.50 | 74.19 | 0.923x | 4.007x | 1.352x | FAIL |
| S3 | 55.35 | 110.81 | 0.499x | 12.380x | 2.432x | FAIL |

Locked gates:

- throughput ratio ≥ 1.15
- p95 TTFT ratio ≤ 0.90
- interactive p95 stream-gap ratio ≤ 0.90
- absolute interactive p95 stream gap ≤ 50.00 ms
- zero candidate request errors, HTTP 503s, and lifecycle errors

Failed gates:

- S0: median_throughput_ratio, median_ttft_p95_ratio, median_stream_gap_p95_ratio
- S1: median_throughput_ratio, median_ttft_p95_ratio, median_stream_gap_p95_ratio
- S2: median_throughput_ratio, median_ttft_p95_ratio, median_stream_gap_p95_ratio
- S3: median_throughput_ratio, median_ttft_p95_ratio, median_stream_gap_p95_ratio, absolute_stream_gap_p95

