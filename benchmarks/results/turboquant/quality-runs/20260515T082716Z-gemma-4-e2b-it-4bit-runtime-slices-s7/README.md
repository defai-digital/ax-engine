# TurboQuant Runtime Slice S7 Microbench

This run records the same-shape standalone fused cold-decode microbenchmark
after sharing the flat hot-tail merge helper between runtime and the
microbenchmark. It is diagnostic evidence only; it does not promote TurboQuant
runtime support.

Artifact:

- `microbench-gemma-e2b-shape-post-s7.json`

Shape:

- model shape: Gemma 4 E2B-style grouped-query decode
- cold tokens: 7936
- hot tokens: 256
- query heads: 8
- KV heads: 1
- head dim: 512
- repetitions: 3
- warmup: 1

Validation:

```bash
python3 scripts/check_turboquant_microbench_artifact.py --min-cold-tokens 7936 \
  benchmarks/results/turboquant/quality-runs/20260515T082716Z-gemma-4-e2b-it-4bit-runtime-slices-s7/microbench-gemma-e2b-shape-post-s7.json
```

Result summary:

| surface | median wall time |
| --- | ---: |
| `two_stage_scores` cold Metal kernel | 5218 us |
| `dim_parallel` cold Metal kernel | 279466 us |
| shared flat hot-tail host merge | 949 us |

Quality remained within the fused-kernel microbench gate:
`two_stage_scores.max_abs_diff=3.864988684654236e-7`,
`two_stage_scores.min_cosine_similarity=0.9999997019767761`, and hot-tail
merge `max_abs_diff=2.849847078323364e-7`.
