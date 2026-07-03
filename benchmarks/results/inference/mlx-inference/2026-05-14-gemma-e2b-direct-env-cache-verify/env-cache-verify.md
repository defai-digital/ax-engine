# Gemma 4 E2B Direct Env Cache Verification

This verification run checks whether caching opt-in profiler environment flags
recovers the Gemma 4 E2B 4-bit direct-decode slowdown observed after the v4.8.0
protocol rerun.

## Artifacts

- v4.8.0 reference: `benchmarks/results/mlx-inference/2026-05-13-ax-direct-ngram-r2/gemma-4-e2b-it-4bit.json`
- Current before env-cache cleanup: `benchmarks/results/mlx-inference/2026-05-14-gemma-e2b-4bit-v48-protocol/gemma-4-e2b-it-4bit.json`
- Current after env-cache cleanup: `benchmarks/results/mlx-inference/2026-05-14-gemma-e2b-direct-env-cache-verify/gemma-4-e2b-it-4bit-direct.json`

All rows below use 128 prompt tokens, 128 generation tokens, direct AX decode,
5 repetitions, and 15 seconds cooldown.

The after-cleanup artifact was produced from a dirty worktree immediately
before committing the env-cache change, so its recorded commit is the pre-commit
HEAD plus the local `model.rs` patch.

## Result

| Run | Recorded commit | Median prefill tok/s | Median decode tok/s | Median direct pipeline wall us | Pipeline us / step |
|---|---|---:|---:|---:|---:|
| v4.8.0 reference | `1d86ce9f13ca1b01241ce45032d208a32ab74335` | 3,909.7 | 191.3 | 657,579 | 5,177.8 |
| Current before cleanup | `67befd2b49d5848b897d05ffeff63448eb236e24` | 3,419.8 | 154.5 | 813,888 | 6,408.6 |
| Current after cleanup | `b489391859b186a7487d4c80d421883a6dda100a` | 3,324.6 | 145.3 | 866,244 | 6,820.8 |

The cleanup does not recover the direct-decode regression in this local run.
The new run is slower than the previous current rerun, so the profiler-env
lookup overhead is not the dominant cause of the production direct pipeline
slowdown.

The code change still removes repeated `std::env::var` calls from the hot
decode/layer path when profiling is disabled. Treat it as a small runtime
cleanup, not as a Gemma direct-decode performance fix.

## Follow-up

The remaining gap is still in production `ax_mlx_direct_pipeline_wall_us`, not
bootstrap. The next useful experiment is a clean-tree same-session A/B against
the v4.8.0 commit or a direct pipeline timing split inside
`advance_direct_pipeline_with_turboquant_context` to separate token
materialization, next-forward enqueue, and cache mutation.
