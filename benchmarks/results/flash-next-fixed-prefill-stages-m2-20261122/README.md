# Flash Next stage costs at the exact split prefill geometry

The M2 Ultra 192 GB measurement uses the pinned resident MXFP4 pack,
MLX 0.32.4 and Rust 1.99.1. The [run binding](run.json) records the
profile binary, prompt and source hashes. All 49 publisher members
were verified. The input is the second 8,192-token prompt from the
[M5 partition experiments](../flash-next-mxfp4-prefill-partition-m5-20261121/README.md).

Four logical 2,048-token items each use physical forwards of 512, 512,
512, 511 and 1 tokens. The first 19 forwards omit the output head.
The final singleton produces one logit row. The same resident model
executes this schedule first without stage barriers and again with
synchronized stage tracking. The final logits and serialized state
match bit for bit.

The synchronized model work takes 42.695 s. Expert computation
(`moe_compute`) accounts for 23.478 s, or 55.0%. Gated DeltaNet
accounts for 6.470 s across the two reported stages. The cache-only
stateless work after the final layer's attention state update,
including the final mixer, accounts for 0.529 s, or 1.24% of the
traced total. This places the MoE path ahead of the proposed
final-layer shortcut for the next investigation.

The untraced model run takes 22.647 s, with a median of 1.365 s across
the 511/512-token forwards. Stage barriers change the total, and the
probe excludes client traffic and scheduler waits. These fractions
characterize this M2 diagnostic. Apply the same measurement on M5
before making a performance decision for the primary tier.

Replay the [raw attribution](attribution.json) and compare the result
with [comparison.json](comparison.json).

```bash
cd benchmarks/results/flash-next-fixed-prefill-stages-m2-20261122
python3 summarize.py attribution.json | diff -u comparison.json -
```
