# Flash Next 512-token private MTP transactions on M5 Max

MacBook Pro M5 Max 128 GB, macOS 27.2.1, internal SSD and MLX 0.32.4.
The Rust 1.99.1 test executable built at `c755bbebb` used the canonical
singleton schedule and the pinned MXFP4 pack at
`ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3`. All 49 publisher members
and the derived model manifest were verified before and after this run.
The [prompt](prompt.json) is the first 512 IDs of the second
[archived 2K prompt](../flash-next-mxfp4-prefill-partition-m5-20261121/prompts.jsonl.gz).

| Draft head | Independent reader | Cases |
| --- | --- | ---: |
| Trained | [Passed](trained-reader.json) | 15 |
| Permuted output | [Passed](permuted-reader.json) | 13 |

Both readers validated forced verifier controls and distinct terminal
cases. The trained head also covered failure after an accepted token.
The [trained](trained.json.gz) and [permuted](permuted.json.gz) producer
receipts bind private state and tensor files by SHA-256. The [run receipt](result.json)
binds both readers, the test binary, the prompt and the pack inventory.

This extends the [46-token transaction controls](../qualification/2026-11-18-flash-next-mxfp4-runner-state/README.md)
to a 512-token prompt. End-to-end MTP-S, MTP-P and MTP-D remain open.
