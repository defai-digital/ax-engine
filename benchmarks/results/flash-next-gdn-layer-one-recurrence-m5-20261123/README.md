# Flash Next continuous layer-one GDN witnesses

Two fresh M5 Max 128 GB processes on macOS 27.2.1 and MLX 0.32.4
repeat the same pinned MXFP4 pack and physical spans as the
[layer-one comparison](../flash-next-gdn-layer-one-drift-m5-20261123/README.md).
The signed Rust 1.99.1 test captures all 2,047 common prompt rows at
the first differing layer. Each arm retains its prior 64 completion
IDs, states and decision logits. At the 23 retained positions inside
the common span, all seven previously sampled stages match their own
arm's archived operator data.

The incoming attention branch already differs at zero-based rows
458, 1334 and 1901. These are the only differing raw QKV projection,
decay and beta rows. The causal convolution spreads each difference
across four rows, yielding 12 differing normalized Q/K/V rows. The
live recurrence differs at every row from 458 through 2046 and in its
final state. The earlier trace first sampled the effect at 493.

With either arm's materialized input set and zero state fixed, whole,
lazy-split and fenced-split replays exactly reproduce that arm's live
output and final state. The remaining layer-one drift in this workload
enters before the recurrence. The three incoming differences provide
positions for inspecting the preceding layer and attention preprocessing.

Both arms stay inside the 96 GiB guard across 153 normal pressure
samples. The higher peak sampled footprint is 86.37 GiB.

`python3 -B replay.py` regenerates `comparison.json` and verifies the
signed inputs, prior operator rows, full-model controls, memory guard,
and state and output digests.
