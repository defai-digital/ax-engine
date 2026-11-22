# Flash Next current native QA on M5 Max

MacBook Pro M5 Max 128 GB, internal NVMe SSD, macOS 27.2.1. Both
arms use the production server, publisher inventory and pinned MXFP4
revision from the [fresh delivery](../flash-next-mxfp4-current-delivery-m5-20261122/README.md).
Both use the ordinary single-request 32K resident profile with zero
sampled verifier steps and route errors.

| Mode | Closed-answer checks | Peak sampled RSS | 29,774-token lookup |
| --- | ---: | ---: | ---: |
| Disabled | 101/105 | 78.88 GiB | 43.627 s |
| Product default | 101/105 | 78.85 GiB | 42.840 s |

All 105 response text pairs match. The retained checker failures are
`format_status_colon`, `instruction_alphabet_first`,
`knowledge_water_formula`, and `science_gravity_earth`. The frozen
memory and lookup gates pass. Host CPU idle is 96.40% before the
phase and 94.00% after it. Reference-relative quality and performance
remain subject to the independent runtime evidence.

The [sealed evidence](evidence.tar.xz) retains all 210 native rows and
their original summary. Regrade with the original frozen scripts and
verify [comparison.json](comparison.json).

```bash
python3 benchmarks/results/flash-next-current-qa-m5-20261122/replay.py
```
