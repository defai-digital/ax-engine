# Flash Next prefill geometry changes state before decode

MacBook Pro M5 Max 128 GB, internal SSD, macOS 27.2.1, Rust 1.99.1,
MLX 0.32.4. Both fresh signed diagnostic processes use the pinned
49-member resident MXFP4 pack. AX Engine's normal scheduler runs one
request through the default 2,048-token scheduler budget, MTP off.
The physical prefill limit is 512 or 2,048. Grouped QSA, paired gather,
indexed attention and prefix reuse are off.

Both arms reproduce all 64 output IDs from their corresponding
[signed HTTP controls](../flash-next-qsa-grouped-http-m5-20261123/README.md)
on the same 8,192-token input. The first emitted token is 5525 for both
widths. The first shared physical boundary is 2,047 input tokens.
All 48 layer-state payload hashes already differ at that boundary,
including layer zero. The snapshot hashes omit only the process-local
owner field. The covered 2,048 and 8,192 boundaries also differ.

| Consumed positions | Greedy IDs at 512 / 2,048 | Largest absolute difference across full logit rows |
| --- | --- | ---: |
| 8,192 | 5525 / 5525 | 1.32031250 |
| 8,193 | 248044 / 5513 | 2.03906250 |

At the second output each arm has consumed the same first output ID.
The 512-row winner leads the 2,048-row winner by 0.46875 in its own
logits. The 2,048-row winner leads the 512-row winner by 0.5 in its own
logits. The observed divergence precedes sampling and is already
present in the prefill state. This single-input probe leaves
cross-geometry quality and the first differing operator open.

All 143 pressure samples are normal and inside the 96 GiB
footprint guard. The direct runner probe without the scheduler
matched only the first two archived outputs in the 512-row arm.
The accepted probe exercises the archived scheduler budget and
requires complete output identity for each geometry. The archive
contains physical cut plans, per-layer hashes and four full F32
logit rows. The replay binds the prior HTTP archive and the exact input.

```sh
shasum -a 256 -c SHA256SUMS
python3.12 -B replay.py | cmp comparison.json -
```
