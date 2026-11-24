# Flash Next layer-one drift after decay tiling

Fresh 512-row stock and wide tiling runs use the same signed M5 Max
128 GB binary, pinned MXFP4 pack, prompt and stock QSA configuration
as the [full-model crossover](../flash-next-gdn-tiling-trial-m5-20261123/README.md).
Both repeat their own 64 completion IDs, retained state witnesses and
two decision-logit rows bitwise. The wide arm tiles 144 decay
projections.

At 24 sampled positions, layer one's attention input, raw post-conv
Q/K/V, logged decay before its final exponential and beta match
bitwise. The first observed difference is in the FP32 GDN core
output, starting at zero-based position 493. The core differs at 18
sampled positions, with a maximum absolute delta of 1.0925e-5. The
remaining cause could be an earlier unsampled row or one of the
unmeasured recurrence inputs. This trace establishes the next
observable boundary for attribution.

All 143 memory-pressure samples are normal. The 86.32 GiB peak
footprint remains inside the 96 GiB guard. `python3 -B replay.py`
verifies the signed inputs and regenerates `comparison.json`.
