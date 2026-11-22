# Flash Next bounded reference CPU phase trace

This MacBook Pro M5 Max 128 GB run uses the pinned MLX-VLM graph and
bounded affine8 PLE reader. It binds the same MXFP4 publisher inventory
and workload hashes to the [current production native matrix](../flash-next-current-matrix-m5-20261122/README.md).
All 105 QA outputs and grades and all 15 fixed-window trajectories
match the signed MLX-VLM record. All 25 pressure samples are normal.
Peak tracked MLX memory is 67.91 GiB.

The six native-to-reference numeric comparisons meet their thresholds.
The reference runner's CPU idle checks are 92.50% before and 82.19%
after, below the frozen 90% floor on exit. This bounded profile supplies
diagnostic evidence. The release comparison still needs the pinned
ordinary reference and valid timing evidence.

The original reference program runs unchanged under the passive Darwin
tick observer. Each row records host CPU counters and the two bounding
phases. The table weights idle by total ticks and includes only intervals
whose start and end report the same phase.

| Phase | Full observed intervals | Weighted host idle |
| --- | ---: | ---: |
| Initial hashes | 9 | 86.06% |
| QA | 18 | 88.58% |
| Timing matrix | 33 | 86.46% |
| Final hashes | 9 | 84.96% |

The low-idle periods include the timing matrix as well as final hashing.
The passive counters measure the whole host. They do not attribute CPU
time to individual processes or replace the runner's idle checks.

The [bounded result](bounded.json.gz), [CPU intervals](cpu-phases.jsonl.gz)
and [memory samples](memory-samples.jsonl.gz) preserve the raw observations.
Replay validates the exact sources, workload binding and CPU delta chain.

```bash
python3 benchmarks/results/flash-next-bounded-reference-cpu-m5-20261122/replay.py
```
