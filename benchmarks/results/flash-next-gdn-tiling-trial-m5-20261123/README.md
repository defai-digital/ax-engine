# Flash Next full-model GDN decay tiling trial

MacBook Pro M5 Max 128 GB, internal SSD, macOS 27.2.1 and MLX 0.32.4.
The signed Rust and Cargo 1.99.1 `release-server` test binary uses
the 49-member MXFP4 pack at `ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3`.
Each arm runs in a fresh process through the native scheduler on the
sealed 8,192-token prompt and generates 64 greedy tokens. The model
is resident with MTP disabled. Prefill QSA uses the same stock policy
in every arm.

The test-only candidate sends only the decay projection through the
existing 512-row projection tiler at wider eligible BF16 MXFP4 spans.
The four first wide physical spans each have 2,047 tokens. The
candidate keeps the original cut geometry for the full trunk. Both
512-row arms are exact historical negative controls. The wide trial
runs in stock, candidate, stock order, followed by a candidate GDN
trace. Timings cover the scheduler and exclude model load and the
offline trace replay.

| Arm | Decay projections tiled | First token | Total | MLX peak |
| --- | ---: | ---: | ---: | ---: |
| 512-row stock | 0 | 8.846 s | 10.593 s | 74.12 GiB |
| 512-row candidate switch | 0 | 8.974 s | 10.757 s | 74.12 GiB |
| Wide stock, first | 0 | 9.670 s | 11.468 s | 84.02 GiB |
| Wide candidate | 144 | 9.573 s | 11.368 s | 84.02 GiB |
| Wide stock, last | 0 | 9.552 s | 11.317 s | 83.97 GiB |

Both narrow arms and both wide stock arms retain their own archived
bits in every retained state, ID and decision row. The candidate layer
zero GDN projection, decay, recurrence and final state match the
narrow arm over all 2,047 common prompt rows. The entire serialized
layer zero state agrees at all four retained positions.

At the first common committed position, 2,047, the other 47 layers
still differ. The candidate matches the narrow arm's first two tokens
and differs in 60 of 64 positions overall. The first-token time and
MLX peak both fall in the wide stock range at the table's precision.
These results support a local correctness repair and leave the 512-row
release default as the validated path. All 370 pressure samples are
normal, with an 86.36 GiB peak footprint below the 96 GiB guard.
The [post-tiling layer-zero trace](../flash-next-post-decay-residual-m5-20261123/README.md)
also matches the narrow activations at 12 sampled positions.

`python3 -B replay.py` verifies the signed inputs, native controls,
logit rows, timings, sampled memory, per-layer state, and the GDN
trace. It regenerates `comparison.json`.
