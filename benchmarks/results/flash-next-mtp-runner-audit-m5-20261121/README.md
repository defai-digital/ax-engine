# Flash Next canonical MTP runner audit on M5 Max

MacBook Pro M5 Max 128 GB, internal SSD, macOS 27.2.1. The native Rust
1.99.1 release-server test at `3978cf05ad8f749caa31a61a0e1ff15ada2e9c73`
used MLX 0.32.4 and MXFP4 pack `ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3`.
All 49 pack files were verified before and after the successful run.
The [launch repair](launcher-repair.json) binds the unchanged test
binary and the pinned dynamic library through the `caffeinate` child.

| Input tokens | Output budget | Verified steps | Accepted drafts | Rejected drafts | Direct output matches |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16 | 64 | 36 | 27 | 9 | 64/64 |
| 512 | 64 | 35 | 28 | 6 | 64/64 |
| 16 | 2 | 1 | 0 | 0 | 2/2 |

The production-facing runner supplied each canonical-singleton
verification decision to a test-only host argmax over the verifier's
actual logits. The replay checks the accept-or-correct decision, output
slice and full output budget. Direct greedy controls matched all 130
output IDs. The 72 steps include two without a draft, and the two-token
case accepted none.

This is bounded same-state greedy evidence from synthetic token prompts,
a resident runner and the cost policy disabled to exercise the verifier.
The active MTP-S, MTP-P and MTP-D qualification records remain separate.
The test contributes correctness evidence and a source-bound receipt.

The [raw decisions](audit.json.gz), [build](build.json), [pack binding](pack.json)
and [provenance](provenance.json) bind the recorded workload. Replay from
the repository root and compare stdout against [comparison.json](comparison.json).

```bash
python3 benchmarks/results/flash-next-mtp-runner-audit-m5-20261121/replay.py
```
