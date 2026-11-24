# Flash Next layer-zero trace after decay tiling

This repeats the wide tiling arm on the same signed M5 Max 128 GB
binary, pinned MXFP4 pack, prompt and physical cuts as the
[full-model crossover](../flash-next-gdn-tiling-trial-m5-20261123/README.md).
It retains the candidate's 64 completion IDs, every available layer
state witness and both decision-logit rows bitwise.

All captured layer-zero activations match the earlier 512-row trace
at the 12 selected positions. This covers the attention HC input,
GDN core and gated output, attention HC output, and the MLP HC input
and output. Layer one remains the first differing serialized state
in the full-model crossover. The layer-zero activation check covers
the retained positions. The test collects 36 normal pressure samples
inside the 96 GiB guard.

`python3 -B replay.py` verifies the capture against both signed runs
and regenerates `comparison.json`.
