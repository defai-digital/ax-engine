# Flash Next prefill geometry drift reaches the first GDN core

MacBook Pro M5 Max 128 GB, internal SSD, macOS 27.2.1, Rust 1.99.1,
and MLX 0.32.4. The same signed test executable and 49-member MXFP4
pack repeat the [scheduler-backed geometry probe](../flash-next-prefill-geometry-divergence-m5-20261123/README.md).
Both arms reproduce their own 64 archived HTTP IDs, physical cuts,
layer-state hashes and two complete output logit rows with the
sampled operator capture enabled.

The first-layer trace compares 156 rows across 12 selected positions
and 13 stages. Embedding, attention input, the post-convolution
GDN Q/K/V tensors, decay log and beta match bitwise at every sampled
position. The first recorded difference is in the FP32 GDN core
output. Eight sampled core rows differ, beginning at zero-based
token 510 with maximum absolute difference 8.73e-10 at that row.
The largest sampled core difference is 3.9e-06. Downstream
layer-zero MLP inputs differ by up to 0.015625. The test records
153 normal pressure samples within the 96 GiB guard.

The sampled pre-normalization inputs do not isolate the
normalization and recurrence from each other. Position 510 is the
first *sampled* difference. The test gives no quality verdict
for changing physical prefill width. The current width and optional
grouped QSA policy remain in effect.

```sh
shasum -a 256 -c SHA256SUMS
python3.12 -B replay.py | cmp comparison.json -
```
