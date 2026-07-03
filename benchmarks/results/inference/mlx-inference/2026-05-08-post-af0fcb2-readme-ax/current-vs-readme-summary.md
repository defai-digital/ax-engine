# AX Engine README Model Refresh - Current vs README

- Current artifact dir: `benchmarks/results/mlx-inference/2026-05-08-post-af0fcb2-readme-ax`
- README baseline artifact dir: `benchmarks/results/mlx-inference/2026-05-07-v4.4.1-readme-refresh`
- Scope: AX-only refresh for all README performance-table models, prompt tokens 128 and 512, generation tokens 128, repetitions 3.
- Reference rows (`mlx_lm`, `mlx_swift_lm`) are reused from the README baseline artifacts; AX direct and AX default n-gram rows were rerun at the current HEAD.

## Aggregate Current-vs-README Delta

- Prefill/direct AX average delta: `-2.4%`
- Direct decode AX average delta: `-6.0%`
- Default n-gram decode AX average delta: `-5.4%`

## Largest N-Gram Decode Regressions

| Model | Prompt | README n-gram | Current n-gram | Delta | Current vs mlx_lm | Claim |
|---|---:|---:|---:|---:|---:|---|
| Qwen 3.5 9B 4-bit | 512 | 91.1 | 53.2 | -41.7% | -43.1% | `ngram_no_draft_direct_fallback` |
| Gemma 4 E2B 6-bit | 128 | 421.3 | 374.5 | -11.1% | +132.2% | `ngram_acceleration_effective_throughput` |
| Gemma 4 E2B 6-bit | 512 | 417.3 | 372.5 | -10.7% | +141.5% | `ngram_acceleration_effective_throughput` |
| Gemma 4 E2B 8-bit | 128 | 453.3 | 405.0 | -10.6% | +190.5% | `ngram_acceleration_effective_throughput` |
| Gemma 4 E2B 5-bit | 512 | 448.5 | 400.9 | -10.6% | +125.1% | `ngram_acceleration_effective_throughput` |

## Largest N-Gram Decode Improvements

| Model | Prompt | README n-gram | Current n-gram | Delta | Current vs mlx_lm | Claim |
|---|---:|---:|---:|---:|---:|---|
| Qwen Coder Next 4-bit | 128 | 137.2 | 221.0 | +61.0% | +139.8% | `ngram_acceleration_effective_throughput` |
| Gemma 4 E4B 4-bit | 128 | 318.4 | 325.3 | +2.2% | +168.2% | `ngram_acceleration_effective_throughput` |
| Qwen 3.5 9B 4-bit | 128 | 193.7 | 196.9 | +1.7% | +106.9% | `ngram_acceleration_effective_throughput` |
| Qwen 3.6 35B A3B UD-MLX 4-bit | 512 | 275.3 | 279.7 | +1.6% | +170.9% | `ngram_acceleration_effective_throughput` |
| Qwen 3.6 35B A3B UD-MLX 4-bit | 128 | 278.8 | 274.8 | -1.4% | +155.4% | `ngram_acceleration_effective_throughput` |

## Full Per-Row Summary

| Model | Prompt | Prefill current vs README | Direct decode current vs README | N-gram decode current vs README | N-gram current vs mlx_lm |
|---|---:|---:|---:|---:|---:|
| Gemma 4 E2B 4-bit | 128 | 3,228.7 vs 3,366.7 (-4.1%) | 173.8 vs 185.6 (-6.3%) | 513.4 vs 572.1 (-10.3%) | +160.0% |
| Gemma 4 E2B 4-bit | 512 | 7,233.1 vs 7,803.5 (-7.3%) | 167.8 vs 180.2 (-6.9%) | 508.5 vs 566.2 (-10.2%) | +165.0% |
| Gemma 4 E2B 5-bit | 128 | 3,089.7 vs 3,308.4 (-6.6%) | 156.6 vs 170.3 (-8.1%) | 405.2 vs 451.8 (-10.3%) | +121.5% |
| Gemma 4 E2B 5-bit | 512 | 7,046.2 vs 7,573.0 (-7.0%) | 149.6 vs 163.7 (-8.6%) | 400.9 vs 448.5 (-10.6%) | +125.1% |
| Gemma 4 E2B 6-bit | 128 | 3,056.9 vs 3,220.5 (-5.1%) | 141.6 vs 153.6 (-7.8%) | 374.5 vs 421.3 (-11.1%) | +132.2% |
| Gemma 4 E2B 6-bit | 512 | 6,959.5 vs 7,375.6 (-5.6%) | 137.8 vs 148.2 (-7.0%) | 372.5 vs 417.3 (-10.7%) | +141.5% |
| Gemma 4 E2B 8-bit | 128 | 3,077.4 vs 3,209.1 (-4.1%) | 132.7 vs 136.8 (-3.1%) | 405.0 vs 453.3 (-10.6%) | +190.5% |
| Gemma 4 E2B 8-bit | 512 | 6,862.5 vs 7,434.7 (-7.7%) | 122.6 vs 132.7 (-7.6%) | 403.3 vs 449.9 (-10.4%) | +199.7% |
| Gemma 4 E4B 4-bit | 128 | 2,396.8 vs 2,469.9 (-3.0%) | 110.4 vs 114.1 (-3.2%) | 325.3 vs 318.4 (+2.2%) | +168.2% |
| Gemma 4 E4B 4-bit | 512 | 4,064.1 vs 4,069.3 (-0.1%) | 107.6 vs 106.7 (+0.8%) | 314.5 vs 322.1 (-2.4%) | +162.1% |
| Gemma 4 26B A4B 4-bit | 128 | 1,157.1 vs 1,221.3 (-5.3%) | 111.5 vs 119.3 (-6.5%) | 244.4 vs 270.1 (-9.5%) | +106.7% |
| Gemma 4 26B A4B 4-bit | 512 | 2,819.8 vs 2,958.1 (-4.7%) | 108.9 vs 116.5 (-6.5%) | 213.3 vs 220.7 (-3.4%) | +88.5% |
| Gemma 4 31B 4-bit | 128 | 520.1 vs 556.5 (-6.5%) | 25.0 vs 27.2 (-8.0%) | 61.4 vs 64.5 (-4.8%) | +134.1% |
| Gemma 4 31B 4-bit | 512 | 674.0 vs 714.4 (-5.7%) | 24.4 vs 26.4 (-7.7%) | 57.0 vs 62.8 (-9.1%) | +128.8% |
| Qwen 3.5 9B 4-bit | 128 | 1,945.3 vs 1,917.1 (+1.5%) | 91.6 vs 93.5 (-2.0%) | 196.9 vs 193.7 (+1.7%) | +106.9% |
| Qwen 3.5 9B 4-bit | 512 | 2,726.0 vs 2,770.9 (-1.6%) | 91.4 vs 94.2 (-3.0%) | 53.2 vs 91.1 (-41.7%) | -43.1% |
| Qwen 3.6 35B A3B UD-MLX 4-bit | 128 | 1,081.2 vs 1,011.1 (+6.9%) | 120.3 vs 121.1 (-0.7%) | 274.8 vs 278.8 (-1.4%) | +155.4% |
| Qwen 3.6 35B A3B UD-MLX 4-bit | 512 | 2,675.3 vs 2,498.2 (+7.1%) | 120.5 vs 120.2 (+0.3%) | 279.7 vs 275.3 (+1.6%) | +170.9% |
| Qwen 3.6 35B A3B 5-bit | 128 | 992.4 vs 960.8 (+3.3%) | 135.9 vs 132.7 (+2.4%) | 263.3 vs 278.7 (-5.5%) | +125.4% |
| Qwen 3.6 35B A3B 5-bit | 512 | 2,616.2 vs 2,432.7 (+7.5%) | 134.5 vs 131.8 (+2.1%) | 261.4 vs 275.1 (-5.0%) | +129.9% |
| Qwen 3.6 35B A3B 6-bit | 128 | 916.1 vs 956.7 (-4.2%) | 109.1 vs 118.9 (-8.2%) | 241.6 vs 257.9 (-6.3%) | +134.8% |
| Qwen 3.6 35B A3B 6-bit | 512 | 2,349.7 vs 2,333.3 (+0.7%) | 108.0 vs 118.3 (-8.7%) | 238.1 vs 254.1 (-6.3%) | +135.6% |
| Qwen 3.6 35B A3B 8-bit | 128 | 912.0 vs 942.7 (-3.3%) | 96.8 vs 107.3 (-9.8%) | 241.6 vs 258.3 (-6.4%) | +158.1% |
| Qwen 3.6 35B A3B 8-bit | 512 | 2,336.0 vs 2,298.3 (+1.6%) | 95.9 vs 105.5 (-9.1%) | 237.7 vs 257.8 (-7.8%) | +159.9% |
| Qwen Coder Next 4-bit | 128 | 835.1 vs 855.4 (-2.4%) | 93.6 vs 103.1 (-9.2%) | 221.0 vs 137.2 (+61.0%) | +139.8% |
| Qwen Coder Next 4-bit | 512 | 2,619.1 vs 2,596.7 (+0.9%) | 90.4 vs 103.0 (-12.2%) | 245.3 vs 258.6 (-5.1%) | +171.5% |
| GLM 4.7 Flash 4-bit | 128 | 823.5 vs 870.0 (-5.3%) | 93.4 vs 104.9 (-11.0%) | 254.8 vs 280.3 (-9.1%) | +174.0% |
| GLM 4.7 Flash 4-bit | 512 | 2,225.0 vs 2,395.4 (-7.1%) | 91.5 vs 103.7 (-11.7%) | 249.1 vs 274.9 (-9.4%) | +175.5% |

## CSV

- `benchmarks/results/mlx-inference/2026-05-08-post-af0fcb2-readme-ax/current-vs-readme-summary.csv`
