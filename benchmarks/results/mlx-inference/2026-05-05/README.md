# MLX Inference Stack AX-Only Refresh: 2026-05-05

Hardware: Apple M5 Max, 128 GB unified memory, macOS 26.4.1.

This run refreshes AX Engine MLX rows from the root README performance table
after recent runtime changes. The `mlx_lm` primary reference rows and most
`mlx_swift_lm` secondary reference rows are reused from the prior public
artifacts; Gemma 4 26B A4B additionally refreshes its `mlx_swift_lm` rows
through the Gemma4 MoE benchmark-adapter path. AX rows are remeasured with the
same random-token prompt artifacts (`mlx_lm` seed=0), batch=1,
`prefill_step_size=2048`, 128 generated tokens, 3 measured trials + 1 warmup,
and `server_sse_runner_time_us`.

`ax_engine_mlx` is the direct same-policy AX comparison.
`ax_engine_mlx_ngram_accel` is effective throughput from AX n-gram acceleration
and must not be read as raw model decode speed. Acceleration rows include
fixed-schema `ngram_acceleration_telemetry` counters from route metadata for
auditability, including zero-valued draft, accept/reject, complete-miss,
no-draft, and cooldown counters.

## Artifacts

| Model | Result JSON | Prompt artifacts |
|---|---|---|
| Gemma 4 E2B | `gemma-4-e2b-it-4bit.json` | `gemma-4-e2b-it-4bit-prompts/` |
| Gemma 4 E2B 5-bit | `gemma-4-e2b-it-5bit.json` | `gemma-4-e2b-it-5bit-prompts/` |
| Gemma 4 E2B 6-bit | `gemma-4-e2b-it-6bit.json` | `gemma-4-e2b-it-6bit-prompts/` |
| Gemma 4 E2B 8-bit | `gemma-4-e2b-it-8bit.json` | `gemma-4-e2b-it-8bit-prompts/` |
| Gemma 4 26B A4B | `gemma-4-26b-a4b-it-4bit.json` | `gemma-4-26b-a4b-it-4bit-prompts/` |
| Gemma 4 31B | `gemma-4-31b-it-4bit.json` | `gemma-4-31b-it-4bit-prompts/` |
| Qwen 3.5 9B | `qwen3_5-9b-mlx-4bit.json` | `qwen3_5-9b-mlx-4bit-prompts/` |
| Qwen 3.6 35B A3B UD-MLX 4-bit | `qwen3_6-35b-a3b-ud-mlx-4bit.json` | `qwen3_6-35b-a3b-ud-mlx-4bit-prompts/` |
| Qwen 3.6 35B A3B 5-bit | `qwen3_6-35b-a3b-5bit.json` | `qwen3_6-35b-a3b-5bit-prompts/` |
| Qwen 3.6 35B A3B 6-bit | `qwen3_6-35b-a3b-6bit.json` | `qwen3_6-35b-a3b-6bit-prompts/` |
| Qwen 3.6 35B A3B 8-bit | `qwen3_6-35b-a3b-8bit.json` | `qwen3_6-35b-a3b-8bit-prompts/` |
| Qwen Coder Next | `qwen3-coder-next-4bit.json` | `qwen3-coder-next-4bit-prompts/` |

Logs: `ax-only-refresh.log` records the full AX direct/n-gram acceleration refresh;
`ax-spec-telemetry-refresh.log` records the follow-up n-gram acceleration refresh after
fixing the benchmark parser to retain the richer step route over the terminal
response route. `gemma-4-26b-a4b-sorted-moe-refresh.log` records the targeted
Gemma 4 26B A4B refresh after aligning AX's MoE expert gather with MLX
SwitchGLU's sorted-index path for large prefill batches.
`moe-sorted-gather-refresh.log` records the formal AX-only rerun for all MoE
models after adding the reusable reference-row refresh path to the benchmark
harness.
Gemma 4 26B A4B `mlx_swift_lm` rows were refreshed directly with
`scripts/mlx-swift-bench/.build/release/mlx-swift-bench` after enabling the
Gemma4 MoE `MLXVLM` factory path in the adapter.

## Public Review Notes

- All result JSON files contain reference rows plus refreshed AX rows.
- All n-gram acceleration AX rows contain non-empty `ngram_acceleration_telemetry`.
- Percentages in the root README are computed against each row's `mlx_lm` median.
- Qwen 3.5 9B keeps the already-refreshed `mlx_lm` baseline from the local
  2026-05-05 opt1 artifact and the secondary Swift row from the 2026-05-04
  public artifact; only its AX rows were remeasured in this refresh.
- Gemma 4 26B A4B was refreshed again after sorted MoE gather landed; its
  512-token direct prefill improved from about 778 tok/s to about 2,750 tok/s.
- Gemma 4 26B A4B now includes admitted `mlx_swift_lm` secondary-reference
  rows: 109.4 tok/s decode at 128 prompt tokens and 104.7 tok/s decode at 512
  prompt tokens.
- Qwen 3.6 35B A3B and Qwen Coder Next were refreshed with the same sorted MoE
  gather path. Their 512-token direct prefill rows now run ahead of the reused
  `mlx_lm` baselines instead of showing the prior unsorted-gather regression.
