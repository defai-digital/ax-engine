# Flash Next current native matrix on M5 Max

The production server, pinned MXFP4 inventory and internal SSD are the
same as the [fresh delivery](../flash-next-mxfp4-current-delivery-m5-20261122/README.md).
The matrix retains the discarded warming process and two independent
processes for each of the disabled and product-default modes. Each
recorded process covers 16 deterministic prompts and 15 fixed-length
runs, with two warmups and three timed runs at each length.

Each same-mode repeat matches all 31 full token arrays. The selected
processes also match all 31 arrays between modes, including the six
warmup outputs. All arms record zero route errors and zero MTP verifier
steps. Host CPU idle is 93.88% before and 90.31% after, passing the
original 90% timing floor.

| Prompt tokens | Disabled median TTFT | Default median TTFT | Disabled decode | Default decode |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 0.538 s | 0.540 s | 49.27 tok/s | 50.25 tok/s |
| 2,048 | 1.658 s | 1.633 s | 45.85 tok/s | 46.54 tok/s |
| 8,192 | 9.801 s | 9.731 s | 44.76 tok/s | 44.82 tok/s |

The table uses the original selection rule, which keeps the process
with the lowest sum of median TTFT across the three prompt lengths in
each mode. These are native serving measurements. The release decision
also needs the independent reference and delivery readiness evidence.

The [result](result.json.gz) and [raw repeats](raw-runs.json.gz) retain
the byte-verifiable original results. Replay verifies the staged scripts
and delivery binding, reconstructs fixed workload hashes, checks raw
repeats, and reproduces the [comparison](comparison.json).

```bash
python3 benchmarks/results/flash-next-current-matrix-m5-20261122/replay.py
```
