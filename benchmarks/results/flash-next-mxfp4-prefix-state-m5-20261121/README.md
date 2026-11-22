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

An extension admission experiment still needs the physical cut and
suffix state under the actual executor, a request-bound snapshot,
uncached-row accounting and a cold recovery control.

From the repository root, replay the evidence and compare the output
with [comparison.json](comparison.json).

```bash
python3 benchmarks/results/flash-next-mxfp4-prefix-state-m5-20261121/replay.py
```
