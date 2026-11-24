# Flash Next layer-two GDN replay after router and decay tiling

Two fresh resident processes on MacBook Pro M5 Max 128 GB, macOS
27.2.1 and MLX 0.32.4 repeat the narrow stock and wide combined arms
from the [projection crossover](../flash-next-router-decay-crossover-m5-20261124/README.md).
Both use the same signed Rust 1.99.1 test binary, pinned MXFP4 pack,
8,192-token prompt, and 64 greedy output tokens. Each arm's full-model
state and output receipt and both decision logit vectors repeat
bitwise. The wide arm engages 144 decay and 192 router projections.

The trace compares all 2,047 common rows in zero-based layer 2. The
incoming attention tensor, QKV projection, convolved and normalized
Q/K/V, and all seven decay stages agree bitwise. The beta input differs
only at zero-based row 1504 of the retained span. Initial recurrence
state agrees. The live recurrence differs at 373 output rows, from
row 1504 through the last difference at row 1919, and in the final state.

For either arm's fixed inputs, whole, lazy-split and fenced-split
replays reproduce that arm's live result and final state exactly.
Holding each partition fixed across the two input sets reproduces
the same 373 row differences. The trace localizes the first observed
layer-two difference to the beta path. Raw beta projection and its
following activation remain separate candidates for a targeted trace.

All 159 pressure samples are normal. Sampled process footprint peaks
at 76.29 GiB for the narrow control and 86.57 GiB for the wide trial.
This diagnostic retains the production 512-row decision from the
crossover.

```sh
shasum -a 256 -c SHA256SUMS
python3.12 -E -s -B replay.py | cmp comparison.json -
```
