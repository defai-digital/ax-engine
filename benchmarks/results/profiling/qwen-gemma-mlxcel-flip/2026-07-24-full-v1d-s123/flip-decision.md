# Qwen 3 + Gemma 4 vs mlxcel flip decision

Decision: **not_yet**

Candidate `ax-engine` vs baseline `mlxcel`; medians over 3 cache-isolated repetitions.

| Scenario | AX tok/s | mlxcel tok/s | Throughput ratio | TTFT ratio | Stream-gap ratio | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| S0 | 110.47 | 94.64 | 1.167x | 0.752x | 0.808x | PASS |
| S1 | 4.80 | 19.60 | 0.245x | 4.160x | 0.249x | FAIL |
| S2 | 81.01 | 74.09 | 1.093x | 0.764x | 2.228x | FAIL |
| S3 | 87.63 | 110.30 | 0.794x | 11.712x | 1.404x | FAIL |

Locked gates:

- throughput ratio ≥ 1.15
- p95 TTFT ratio ≤ 0.90
- interactive p95 stream-gap ratio ≤ 0.90
- absolute interactive p95 stream gap ≤ 50.00 ms
- zero candidate request errors, HTTP 503s, and lifecycle errors

Failed gates:

- S1: median_throughput_ratio, median_ttft_p95_ratio
- S2: median_throughput_ratio, median_stream_gap_p95_ratio
- S3: median_throughput_ratio, median_ttft_p95_ratio, median_stream_gap_p95_ratio
