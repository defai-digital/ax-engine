# Flash Next MXFP4 two-user prefill step cost on M5 Max

MacBook Pro M5 Max 128 GB, internal SSD, macOS 27.2.1, Rust 1.99.1,
MLX 0.32.4 and pinned MXFP4 at `ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3`.
The diagnostic uses the [six-arm selective experiment](../flash-next-mxfp4-selective-prefill-m5-20261121/README.md)
executable with resident experts, two slots and 8K-only splitting. Each
burst pairs a 512-token prompt and an 8,192-token prompt. Both bursts
match all 384 output IDs to the archived solo trajectories, drain all
five queue and stream metrics and preserve normal pressure across 83 samples.

| Runner step | Count | Median native runner time |
| --- | ---: | ---: |
| Decode only | 210 | 22.01 ms |
| Shared decode with 511-token prefill | 4 | 730.58 ms |
| Shared decode with 512-token prefill | 12 | 727.70 ms |

Shared timing comes from the existing native SSE step fields. The
analysis matches all participants for each counted step and counts the
wall interval once. All 16 shared observations occurred in the early
arrival burst. The later arrival supplied decode-only timing and no
observed concurrent prefill slice. The shared medians apply to those
observed steps and do not establish timing across arrival positions.

Changing the spacing between existing physical slices can protect more
decode turns between slices. In this burst the existing width already
spent about 730 ms inside a shared step. A shorter bound within one
step would require a separate physical-width or execution experiment
with exact greedy output and state checks. The serving defaults retain
the current single-active-request resident profile.

The [sealed evidence](evidence.tar.xz) retains exact timing traces,
prompt data, pressure samples, an original receipt and client JSON
with the local corpus path and bind address redacted. Replaying checks
the portable client records against the prior solo tokens.

```bash
python3 benchmarks/results/flash-next-mxfp4-prefill-step-cost-m5-20261121/replay.py
```
