# Qwen 3 + Gemma 4 vs mlxcel flip decision

Decision: **not_yet**

Candidate `ax-engine` vs baseline `mlxcel`; medians over 3 cache-isolated repetitions.

| Scenario | AX tok/s | mlxcel tok/s | Throughput ratio | TTFT ratio | Stream-gap ratio | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| S0 | 109.36 | 93.96 | 1.164x | 0.841x | 0.819x | PASS |
| S1 | 14.76 | 19.46 | 0.758x | 1.326x | 0.800x | FAIL |
| S2 | 109.37 | 73.99 | 1.478x | 1.011x | 0.774x | FAIL |
| S3 | 90.13 | 110.57 | 0.815x | 7.633x | 1.831x | FAIL |

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
