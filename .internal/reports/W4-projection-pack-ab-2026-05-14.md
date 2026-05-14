# W4 Linear-Attention Projection Pack A/B

Date: 2026-05-14

## Scope

This report records the first real-model split-vs-packed AX MLX projection-pack
A/B for Qwen3.6 35B A3B 8-bit.

The run used the checked-in Qwen reference rows for `mlx_lm` and
`mlx_swift_lm`, then refreshed AX direct rows twice with
`AX_MLX_LINEAR_ATTENTION_PROFILE=1`:

- default split `qkv/z/a/b` projections as `ax_engine_mlx`
- experimental loader-time packed `qkvz/ba` projections as
  `ax_engine_mlx_linear_pack`

Artifact:

- `benchmarks/results/mlx-inference/2026-05-14-qwen36-linear-pack-ab/qwen3_6-35b-a3b-8bit-linear-pack-ab.json`
- `benchmarks/results/mlx-inference/2026-05-14-qwen36-linear-pack-ab/qwen3_6-35b-a3b-8bit-linear-pack-ab.md`

## Result

| Prompt tokens | Split prefill tok/s | Packed prefill tok/s | Packed/split | Split projection ms | Packed projection ms | Projection ratio | Verdict |
|---:|---:|---:|---:|---:|---:|---:|---|
| 128 | 722.1 | 738.3 | 1.023x | 661.6 | 645.6 | 0.976x | candidate win |
| 512 | 1,897.4 | 1,865.8 | 0.983x | 980.0 | 994.5 | 1.015x | neutral/noisy |

Promotion gate:

```text
python3 scripts/check_mlx_forward_profile_artifact.py \
  benchmarks/results/mlx-inference/2026-05-14-qwen36-linear-pack-ab/qwen3_6-35b-a3b-8bit-linear-pack-ab.json \
  --require-pack-candidate-win \
  --min-pack-candidate-wins 2 \
  --min-pack-candidate-win-prompts 2
```

Outcome:

```text
MLX forward profile artifact check failed: benchmarks/results/mlx-inference/2026-05-14-qwen36-linear-pack-ab/qwen3_6-35b-a3b-8bit-linear-pack-ab.json pack comparison is not a candidate win: .internal/models/Qwen3.6-35B-A3B-8bit prompt=512: neutral/noisy
```

## Conclusion

The packed projection path is a valid experiment, but this A/B does not justify
turning it on by default or claiming a general TTFT/prefill improvement.

The 128-token shape improved enough to pass the candidate-win threshold, but
512-token did not. The next performance work should not polish the current
loader-time concatenation into a default. It should inspect why `qkvz` becomes
more expensive at 512 tokens even though the split substage total falls, then
decide whether the real fix is a lower-level fused projection/scan path rather
than loader-only packing.
