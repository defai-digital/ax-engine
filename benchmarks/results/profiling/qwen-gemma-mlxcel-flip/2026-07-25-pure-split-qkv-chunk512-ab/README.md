# Pure-wall A/B: Gemma4 split vs packed QKV at chunk-512 (mbp-m5)

Profile residual: `pre_sdpa_qkv_proj` ~1.09s. Hypothesis: long pure chunk
seq=512 fell outside old split range [127,511] onto packed; extend range.

## Result

| | cold mean |
|--|--|
| packed (kill-switch, old behavior for seq=512) | **8853 ms** |
| split (cap 8192) | **9122 ms** |

**ratio_split_over_packed = 1.030** → keep max=511 (packed for chunk-512).

Not a ≥7.5% pure cut; attention QKV policy residual closed for this polarity.
