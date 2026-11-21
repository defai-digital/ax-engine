# Flash Next native output and timing matrix on M5 Max

Both runs used MacBook Pro M5 Max 128 GB, macOS 27.2.1, internal SSD,
Rust 1.99.1 and MLX 0.32.4. The hash-bound `b36b7b287`
`release-server` binary exercised the same MXFP4 pack revision
`ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3`. Each mode ran in two
alternating processes after a discarded warm pass. Each process ran
16 greedy trajectories and five requests per input length, with two
warmups and three measured requests of 128 output tokens. The frozen
matrix uses a 4,096-block KV pool. The
[ordinary-launch QA](../flash-next-mxfp4-ordinary-qa-20261121/README.md)
separately records the normal 32K profile.

| Run | CPU idle before | CPU idle after | Same-mode repeat identity | Explicit direct and default |
| --- | ---: | ---: | --- | --- |
| First | 97.14% | 87.61% | 31/31 per mode | 16/16 trajectories and 9/9 measured cells |
| Retry | 95.32% | 74.90% | 31/31 per mode | 16/16 trajectories and 9/9 measured cells |

The selected repeats in both runs used identical prompt IDs and
generated identical token arrays for every recorded trajectory and
fixed cell. Route errors and MTP verifier steps were zero. The
following medians describe the selected repeats in the second run.

| Input tokens | Explicit direct TTFT | Default TTFT | Explicit direct decode | Default decode |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 0.544 s | 0.541 s | 48.99 tok/s | 49.52 tok/s |
| 2,048 | 1.623 s | 1.766 s | 45.56 tok/s | 45.76 tok/s |
| 8,192 | 10.505 s | 10.618 s | 44.36 tok/s | 44.52 tok/s |

The frozen timing gate requires at least 90% CPU idle at both ends.
Both runs provide functional and diagnostic evidence. The current
release timing comparison with the pinned MLX-VLM runtime remains open.

The [first result](first-result.json.gz) and
[retry result](retry-result.json.gz) include the selected repeats,
timing samples, and prompt hashes. The
[first raw runs](first-raw-runs.json.gz) and
[retry raw runs](retry-raw-runs.json.gz) preserve both per-mode repeats
for independent comparison.
