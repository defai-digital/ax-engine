# MLX 0.32.1 Qwen 3.8 MTP multirow A/B

This directory contains the raw native-serving artifacts for the experimental
`AX_MLX_MTP_MULTIROW_BATCH` route. The run used AX Engine 7.1.5 working-tree
code, the pinned PyPI MLX 0.32.1 runtime, an Apple M5 Max with 128 GB unified
memory, and the pinned Qwen 3.8 27B AXQ 6-bit MTP snapshot.

The workload is four deterministic prompts of exactly 155 input tokens and a
forced 256-token output (`temperature=0`, `ignore_eos=true`). EngineCore prefix
reuse and both MLX prefix-cache tiers were disabled. Each measured c2/c4 file
contains eight requests; an unrecorded four-request c4 batch warmed each fresh
server first.

| Mode | Concurrency | Trial | Output tok/s | Oracle-identical streams |
| --- | ---: | ---: | ---: | ---: |
| MTP baseline, feature off | 2 | 1 | 24.296 | 8/8 |
| MTP baseline, feature off | 4 | 1 | 24.314 | 8/8 |
| MTP-to-direct batch | 2 | 1 | 42.566 | 7/8 |
| MTP-to-direct batch | 2 | 2 | 44.047 | 8/8 |
| MTP-to-direct batch | 4 | 1 | 74.422 | 8/8 |
| MTP-to-direct batch | 4 | 2 | 77.405 | 8/8 |
| Singleton control, feature on | 1 | 1 | 23.976 | 4/4 |

The tensor route reached every eligible row: the c2 artifacts report 16
candidate, suspended, and batched-forward rows; c4 reports 26 of each. No
measured request failed and every request emitted exactly 256 tokens.

The performance result is positive, but the route is **not promoted**: one of
32 multirow streams differed from the feature-off greedy oracle. The feature
therefore remains default-off and still obeys the ordinary structural and
batched-decode certification gates. See `summary.json` for the compact verdict
and the other JSON files for raw request observations, route decisions, and
output-token hashes.
