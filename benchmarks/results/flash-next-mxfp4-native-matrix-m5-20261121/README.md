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
fixed cell. Compared with the [signed v7.9.0 matrix](../qualification/2026-10-24-flash-next-mxfp4-m5max-release-v7.9.0/generate-matrix/result.json),
both runs match all input hashes and all 62 output arrays across the
two arms, including every warmup, measured cell and trajectory.
Route errors and MTP verifier steps were zero. The
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

## Quiet-host repeat

The third run completed on the same current binary, pack and matrix.
CPU idle was 98.98% before and 97.56% afterward. The two repeats in
each mode matched all 31 output arrays, and all 124 arrays across the
four measured processes match the signed v7.9.0 matrix. The comparison
checked all input hashes, complete output IDs, zero route errors and
zero MTP verifier steps.

| Input tokens | Explicit direct TTFT | Default TTFT | Explicit direct decode | Default decode |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 0.546 s | 0.550 s | 49.72 tok/s | 49.54 tok/s |
| 2,048 | 1.640 s | 1.626 s | 46.28 tok/s | 46.44 tok/s |
| 8,192 | 9.410 s | 9.533 s | 44.39 tok/s | 45.07 tok/s |

The signed native default-arm medians were 0.487, 1.572 and 12.012 s
TTFT, with decode at 50.72, 47.10 and 45.78 tok/s. The current native
observations show lower 8K TTFT, higher short-prompt TTFT and lower
decode rates across the measured lengths. The executable and macOS
version differ between those campaigns. The current independent
MLX-VLM comparison remains open.

The [quiet result](quiet-result.json.gz) and
[raw repeats](quiet-raw-runs.json.gz) retain the timing and token evidence.

## Independent reference stop

The pinned MLX-VLM process used the same verified inventory on the
M5. Before launch, CPU idle was 92.77%. It materialized the weights
in 19.971 seconds and completed one of 105 QA items in 406.908
seconds. The monitor stopped the owned process at critical system
memory pressure (level 4), before the performance matrix started.
System swap usage rose from 1,433.94 MiB to 23,338.81 MiB.
Peak compressor occupancy was 73.334 GiB.

The [partial receipt](reference-partial.json.gz),
[47 system samples](reference-monitor.jsonl.gz), and
[terminal status](reference-status.json) preserve the attempt.
The independent QA and timing gates remain open.
