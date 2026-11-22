# Flash Next long-workload MTP decision audit on M2 Ultra

Mac Studio M2 Ultra 192 GB, external SSD, macOS 27.2.1. The `9df17f26a`
release-server test build uses Rust 1.99.1 and MLX 0.32.4. It validates
the 49-member MXFP4 pack before and after the test. The native runner
uses the canonical singleton verifier and a 2,048-token prefill item.
The test disables the MTP cost controller to exercise verification.

| Workload | Input tokens | Output budget | Verifier decisions | Accepted drafts | Rejected drafts |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen math prompt | 69 | 64 | 34 | 29 | 4 |
| Fixed timing-window prompt | 8,192 | 64 | 39 | 24 | 14 |
| Frozen long lookup prompt | 29,774 | 8 | 4 | 3 | 0 |

The host oracle recomputes greedy decisions from the verifier's actual
same-state logit rows. All 56 accepted drafts select the target token.
The captured decisions account for all 133 generated positions after
the three first tokens. The test verifies route counters, position
alignment, normal completion and cleared request state. The separate
direct runs match all 136 generated IDs. The bound lookup prompt uses
an eight-token audit budget.

This is functional evidence from the M2 best-effort tier. The M5
same-state run, MTP speed and default promotion have separate scope.
The [raw observations](report.json.gz), [execution receipt](receipt.json)
and [workload provenance](cases-provenance.json) bind the original
inputs and executable. The replay reconstructs the prompt IDs from
the frozen QA dataset and fixed-window generator.

```bash
python3 benchmarks/results/flash-next-expanded-mtp-audit-m2-20261122/replay.py
```
