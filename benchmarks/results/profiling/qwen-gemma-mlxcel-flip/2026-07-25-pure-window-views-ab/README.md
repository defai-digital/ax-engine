# Pure-wall A/B: multi_token_window_views (mbp-m5)

Profile residual: `sdpa` ~1.24s with Gemma4 SWA (window=1024). Default ON
presents sliding layers with a retained K/V window view.

## Result

| | cold mean |
|--|--|
| ON (default) | **8873 ms** |
| OFF | **9088 ms** |

**ratio_on_over_off = 0.976** (~2.4% win for ON). Keep default ON.
Not enough alone for ≥7.5% pure cut toward thr≥21.
