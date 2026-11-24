# Flash Next layer-zero router witnesses

Two fresh MacBook Pro M5 Max 128 GB runs on macOS 27.2.1 and MLX 0.32.4
reuse the signed binary, pinned MXFP4 pack and prompt from the
[continuous layer-one replay](../flash-next-gdn-layer-one-recurrence-m5-20261123/README.md).
They repeat their respective 64 completion IDs, state witnesses and
decision logits bitwise. The comparison inspects the three known
layer-one input differences, their immediate neighbors and two
additional controls, for 11 positions in all.

The sampled layer-zero GDN, attention HC write and MLP HC input all
match. The first sampled differences appear in the raw MoE router
logits at zero-based rows 458, 1334, 1335 and 1901. The maximum
absolute router difference is 0.03125. All sampled routed expert
indices, expert outputs before weighting and the shared-expert path
match. The routed contribution and final MLP HC write differ at 458,
1334 and 1901, the same three rows seen entering layer one.

An isolated M5 check loads the exact affine 8-bit group-32 router with
the recorded BF16 inputs. A 2,047-row projection and a padded 2,048-row
projection both reproduce the wide logits, including the four changes.
Projection-only tiles of `512,512,512,511` reproduce all eleven
canonical router rows. The isolated input uses zeros outside the
retained positions. Full-model and timing validation remain open for
the router layout candidate.

The run has 171 normal pressure observations and remains inside the
96 GiB footprint guard. `python3 -B replay.py` verifies the signed
inputs, the earlier full-model controls, and the isolated projection
row hashes. It regenerates `comparison.json`. With the pinned pack
and MLX installed, the component can be repeated with
`python3 -B router_projection_component.py SNAPSHOT OBSERVED INVENTORY`.
