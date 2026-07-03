# Gemma 4 E2B 4-bit v4.8.0 Protocol Comparison

This rerun uses the same high-level benchmark protocol as the v4.8.0 README
artifact: 128/512 prompt tokens, 128 generation tokens, 5 repetitions, and
15 seconds cooldown.

## Artifacts

- v4.8.0 reference: `benchmarks/results/mlx-inference/2026-05-13-ax-direct-ngram-r2/gemma-4-e2b-it-4bit.json`
- Current rerun: `benchmarks/results/mlx-inference/2026-05-14-gemma-e2b-4bit-v48-protocol/gemma-4-e2b-it-4bit.json`

## Build And Protocol

| Artifact | Commit | Repetitions | Cooldown |
| --- | --- | ---: | ---: |
| v4.8.0 reference | `1d86ce9f13ca1b01241ce45032d208a32ab74335` | 5 | 15s |
| Current rerun | `67befd2b49d5848b897d05ffeff63448eb236e24` | 5 | 15s |

The current checkout had unrelated dirty files during the run, so this artifact
is useful as local evidence but should not replace the public README table
without a clean-tree rerun.

## Median Throughput Delta

| Prompt | Engine | v4.8.0 prefill tok/s | Current prefill tok/s | Delta | v4.8.0 decode tok/s | Current decode tok/s | Delta |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | `mlx_lm` | 2,443.1 | 2,563.9 | +4.9% | 213.5 | 192.2 | -10.0% |
| 128 | `ax_engine_mlx` | 3,909.7 | 3,419.8 | -12.5% | 191.3 | 154.5 | -19.3% |
| 128 | `ax_engine_mlx_ngram_accel` | 3,949.5 | 3,608.7 | -8.6% | 591.5 | 534.1 | -9.7% |
| 512 | `mlx_lm` | 7,768.5 | 6,996.8 | -9.9% | 210.1 | 167.2 | -20.4% |
| 512 | `ax_engine_mlx` | 8,499.2 | 7,714.3 | -9.2% | 184.3 | 164.1 | -11.0% |
| 512 | `ax_engine_mlx_ngram_accel` | 8,600.7 | 7,811.4 | -9.2% | 581.5 | 532.4 | -8.4% |

## Relative Decode Position

| Prompt | Engine | v4.8.0 vs `mlx_lm` | Current vs `mlx_lm` |
| ---: | --- | ---: | ---: |
| 128 | `ax_engine_mlx` | 0.896x | 0.803x |
| 128 | `ax_engine_mlx_ngram_accel` | 2.770x | 2.778x |
| 512 | `ax_engine_mlx` | 0.877x | 0.981x |
| 512 | `ax_engine_mlx_ngram_accel` | 2.767x | 3.184x |

## Interpretation

The slowdown is not a single n-gram regression. The current `mlx_lm` decode
baseline is also lower than the v4.8.0 artifact, especially at 512 prompt
tokens, which points to run-condition noise, thermal state, or local machine
load as part of the absolute throughput drop.

AX direct decode still deserves follow-up. At 128 prompt tokens, direct decode
fell from 0.896x to 0.803x relative to `mlx_lm`, so this path regressed beyond
the baseline shift in this run. At 512 prompt tokens, the direct relative ratio
improved from 0.877x to 0.981x because `mlx_lm` slowed more than AX direct.

The n-gram path remains healthy for this draftable prompt shape. Its relative
decode position is stable or stronger than the v4.8.0 artifact, and the run
kept full n-gram acceptance for the measured requests.

## Recommended Next Step

Before changing runtime code, rerun the same command from a clean worktree and
record whether the 128-token direct-decode ratio remains near 0.80x. If it does,
profile direct decode only with `AX_MLX_DECODE_PROFILE=1` and compare the
per-layer input, QK/V norm, reshape/transpose, and scheduler timing against the
v4.8.0 reference.
