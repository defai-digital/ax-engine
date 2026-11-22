# Flash Next long-workload MTP decision audit on M5 Max

MacBook Pro M5 Max 128 GB, internal NVMe SSD, macOS 27.2.1 and MLX
0.32.4. The `9df17f26a` release-server test build uses Rust 1.99.1.
It verifies all 49 publisher pack members and the delivered manifest
before and after the run. The native runner uses a 2,048-token
prefill item and the canonical singleton verifier. The test disables
the MTP cost controller so it exercises the verification path.

| Workload | Input tokens | Output budget | Verifier decisions | Accepted drafts | Rejected drafts |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen math prompt | 69 | 64 | 34 | 29 | 4 |
| Fixed timing-window prompt | 8,192 | 64 | 39 | 24 | 14 |
| Frozen long lookup prompt | 29,774 | 8 | 4 | 3 | 0 |

All 56 accepted drafts match the greedy verifier decision on the same
state. The 77 decisions account for 133 output positions after the
three first tokens. The runner test also checks counter agreement,
position alignment, request completion and state cleanup. Independent
direct runs match all 136 generated IDs on this device. All 46 sampled
memory-pressure readings are normal.

The same executable on [M2 Ultra](../flash-next-expanded-mtp-audit-m2-20261122/README.md)
matches the M5 on the short and lookup prompts. The 8,192-token
direct trajectory changes at zero-based output position 29 and in
35 positions in total across chips. The MTP output follows its
respective direct trajectory on each chip. The test captures token
decisions; it does not measure the cross-chip logit margin.

These controls cover bounded greedy execution on the primary device.
The long lookup generates eight tokens. MTP-S, MTP-P and MTP-D remain
open. The audit is not an acceleration or default-policy measurement.

The [raw decisions](report.json.gz), [run receipt](receipt.json) and
[workload provenance](../flash-next-expanded-mtp-audit-m2-20261122/cases-provenance.json)
bind the original observations to the frozen QA and timing-window inputs.

```bash
python3 benchmarks/results/flash-next-expanded-mtp-audit-m5-20261122/replay.py
```
