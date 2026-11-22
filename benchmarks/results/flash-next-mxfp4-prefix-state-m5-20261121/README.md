# Flash Next fixed-boundary prefix state on M5 Max

MacBook Pro M5 Max 128 GB, internal SSD, macOS 27.2.1. The isolated
probe used MLX 0.32.4, Rust 1.99.1, and the pinned MXFP4 pack at
`ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3`. It reused the [source patch
and prompts](../flash-next-mxfp4-prefix-state-m2-20261121/README.md) from
the earlier fixed-cut study. The M5 receipt binds the packaged binary,
source patch, model manifest, publisher inventory and HTTP controls.

On both 2,048-token prefixes, a preserved final singleton produced
identical standalone and cold-cut states. The alternate partition
changed both saved digests. Cold continuation and restored state
matched through serialization and rebind, the 64-token suffix, all 32
subsequent greedy IDs per prompt, and the state after decode. The Rust
probe passed and all 52 pressure samples were normal, with a peak
sampled footprint of 74.30 GiB.

The [original launcher receipt](original-state-receipt.json) marked the
aggregate failed because it required every continuation's first output
to differ from the standalone seed output. Case 0 yielded 5,513 in
both runs. Case 1 yielded 45,527 after the suffix and 5,513 from the
standalone seed, providing a direct witness against substituting the
saved first token. The replay checks all state and output axes for both
cases and requires a distinguishing case. The exact original status
and both complete output arrays remain available.

The paired [HTTP controls](http-control.json.gz) bind the same two
extension prompts and their exact seed IDs. With the seeds cached and
a peer decoding, both extensions advanced 2,048 and then 64 prompt
tokens, emitted their first token after the suffix, and matched the
corresponding solo cold output. The complete first branch matched 96
IDs. The cancellation branch also produced the distinguishing 45,527
before disconnect. Each branch shared both prefill steps with the
decoding peer. The evidence establishes those logical scheduler cuts
alongside exact-prompt reuse and strict-extension cold fallback.

## Physical follow-up

Two additional processes repeated the HTTP controls with the same
executable, exact prompts, model manifest and pack inventory. The
physical trace recorded three post-listen cuts per process, each with
`512, 512, 512, 511, 1` rows, alongside request-keyed SSE advances of
2,048 and 64. The L1 process matched 96 paired, one pre-cancellation
and 96 replacement output IDs. Both peers matched their full solo
output arrays, and the exact repeat restored 2,048 tokens. All 157
pressure samples were normal. Peak sampled footprint was 72.0 GiB with
the cache disabled and 72.9 GiB with exact-prompt L1 enabled.

The original launcher counted six `31, 1` startup pairs on the L1
process as request forwards and failed its final audit. The retained
[trace extraction](physical-traces.json) records line numbers, the
single structured listen boundary per process, and the original log
hashes. Each startup pair precedes its listener boundary. The physical
events lack request IDs, so the replay corroborates the aggregate
trace against the three request-keyed cold controls in each process.
The [HTTP receipt](physical-http-control.json.gz) and pressure samples
retain their original bytes under compression.

Strict extensions still prefill cold. Admission of a warm extension
still needs request-bound runtime snapshot and suffix-state proof,
uncached-row accounting, and cold recovery coverage.

From the repository root, replay the evidence and compare the output
with [comparison.json](comparison.json).

```bash
python3 benchmarks/results/flash-next-mxfp4-prefix-state-m5-20261121/replay.py
python3 benchmarks/results/flash-next-mxfp4-prefix-state-m5-20261121/replay_physical.py
```

The second replay emits [physical-comparison.json](physical-comparison.json).
