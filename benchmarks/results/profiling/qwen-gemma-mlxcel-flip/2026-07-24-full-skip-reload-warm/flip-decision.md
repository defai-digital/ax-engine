# Qwen 3 + Gemma 4 vs mlxcel flip decision

Decision: **not_yet**

Candidate `ax-engine` vs baseline `mlxcel`; medians over 3 cache-isolated repetitions.

| Scenario | AX tok/s | mlxcel tok/s | Throughput ratio | TTFT ratio | Stream-gap ratio | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| S0 | 108.91 | 93.99 | 1.159x | 0.959x | 0.825x | FAIL |
| S1 | 14.80 | 19.51 | 0.758x | 1.326x | 0.808x | FAIL |
| S2 | 89.88 | 74.00 | 1.214x | 0.835x | 2.002x | FAIL |
| S3 | 90.27 | 110.47 | 0.817x | 7.184x | 1.829x | FAIL |

Locked gates:

- throughput ratio ≥ 1.15
- p95 TTFT ratio ≤ 0.90
- interactive p95 stream-gap ratio ≤ 0.90
- absolute interactive p95 stream gap ≤ 50.00 ms
- zero candidate request errors, HTTP 503s, and lifecycle errors

Failed gates:

- S0: median_ttft_p95_ratio
- S1: median_throughput_ratio, median_ttft_p95_ratio
- S2: median_stream_gap_p95_ratio
- S3: median_throughput_ratio, median_ttft_p95_ratio, median_stream_gap_p95_ratio, absolute_stream_gap_p95
