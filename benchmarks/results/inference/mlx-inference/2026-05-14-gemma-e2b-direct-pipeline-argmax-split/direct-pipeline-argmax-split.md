# Gemma 4 E2B Direct Pipeline Argmax Split

This diagnostic artifact separates the previous `argmax + async_eval(next)`
bucket into two production direct-pipeline counters:

- `ax_mlx_direct_pipeline_argmax_wall_us`
- `ax_mlx_direct_pipeline_async_eval_wall_us`

The run is intentionally short and is meant to validate the timing split, not
to replace headline throughput rows.

## Artifact

- JSON: `benchmarks/results/mlx-inference/2026-05-14-gemma-e2b-direct-pipeline-argmax-split/gemma-4-e2b-it-4bit-direct.json`
- Prompt shape: 128 prompt tokens, 128 generation tokens
- AX policy: `direct_no_ngram_acceleration`
- Repetitions: 1 measured run after 1 AX warmup
- Recorded commit: `b24e67fe328bd9ca2a9f092f2048bc933ba66372`

The artifact was produced from a dirty worktree immediately before committing
the argmax split, so the recorded commit is the pre-commit HEAD plus local
telemetry patches.

## Direct Pipeline Timing

| Metric | Value |
|---|---:|
| Decode tok/s | 159.4 |
| Decode wall us | 793,905 |
| Direct pipeline wall us | 789,221 |
| Direct pipeline steps | 127 |
| Direct pipeline wall us / step | 6,214.3 |

| Direct pipeline bucket | Wall us | Share of direct pipeline |
|---|---:|---:|
| Next forward graph build | 56,122 | 7.1% |
| `argmax(next_logits)` graph node | 78 | 0.0% |
| `async_eval(next_token)` submit | 732,090 | 92.8% |
| Pending token `eval` barrier | 585 | 0.1% |
| Pending token read | 53 | 0.0% |

## Reading

The token-selection graph node is not the bottleneck. The cost remains inside
the pure `async_eval(next_token)` call, while pending-token materialization and
CPU readback are negligible.

That narrows the next investigation to MLX async-submit backpressure or the
size/dependency structure of the next-token graph being submitted. A useful next
experiment is to compare `async_eval(next_token)` against submitting a cheaper
target from the same graph, or to inspect whether direct mode is queueing more
work than mlx_lm before each submit.
