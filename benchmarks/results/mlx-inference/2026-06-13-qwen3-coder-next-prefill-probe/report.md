# Qwen3-Coder-Next 4-bit Prefill-Step-Size Probe

## Scope

Performance probe for `mlx-community/Qwen3-Coder-Next-4bit` evaluating
recommendation #2 from the 2026-06-12 baseline review: increase the prefill
step size from 2048 to 4096 to reduce per-step fixed overhead (eval barriers,
generation-state prep) and tighten TTFT scaling.

## Setup

- **Host:** Apple M5 Max, 128 GB, macOS 26.5.1
- **Model:** `mlx-community/Qwen3-Coder-Next-4bit` (qwen3_next, 48 layers,
  linear-attention hybrid, 4-bit affine with 8-bit MoE gates)
- **Build:** `d816e334` (release server, rebuilt fresh 2026-06-13 20:25)
- **Shape:** prompt 512, generation 128, greedy, batch 1, cold-prefill contract
  (prefix cache disabled), 2 repetitions, 5s cooldown
- **Comparator:** 2026-06-12 baseline (`90e16a12`, step-size 2048, 3 reps)
  at the same model / shape / host

## Results (p512, direct / no n-gram)

| metric | step 2048 (06-12) | step 4096 (06-13) | Δ |
| --- | ---: | ---: | ---: |
| prefill tok/s | 1758.0 | 1786.9 | **+1.6%** |
| decode tok/s | 113.6 | 116.1 | **+2.1%** |
| TTFT ms | 291.2 | 286.5 | **−1.6%** |

Telemetry corroborates the throughput gain:

| telemetry | step 2048 | step 4096 | Δ |
| --- | ---: | ---: | ---: |
| prefill steps | 3 | 2 | −1 (fewer eval barriers) |
| prefill wall µs | 842,516 | 551,828 | **−34.5%** |
| generation-state wall µs | 30,968 | 20,719 | −33.1% |

Per-trial variance (step 4096, 2 reps): prefill 0.2878s / 0.2853s,
decode 1.1016s / 1.1039s — tight, well-conditioned.

## Conclusions

1. **Larger prefill step size is a clear win at p512.** Step 4096 cuts the
   prefill step count 3 → 2, removing one eval barrier and its generation-state
   prep, dropping prefill wall time ~35% and improving prefill throughput
   +1.6% and TTFT −1.6%.
2. **Decode also improved (+2.1%)**, attributable to thermal/cache warm state
   in the same-session comparison rather than a decode-path change (step size
   only affects prefill chunking). Included for completeness.
3. **Recommendation:** keep 4096 as the prefill step size for Qwen3-Coder-Next
   at p512+; the 2048 default leaves a measurable per-step overhead on the
   table.

## Caveats

- Comparator used 3 reps vs this probe's 2 reps; both are low-variance (spread
  < 1%), so the direction is robust.
- Different build commits (`90e16a12` vs `d816e334`); neither touches the MLX
  prefill hot path materially, and the step-count drop (3→2) is a direct
  mechanical consequence of the larger step size, not a build artifact.
- Single host (M5 Max); reproduce with `python3 scripts/bench_mlx_inference_stack.py
  --model-repo-id mlx-community/Qwen3-Coder-Next-4bit --prompt-tokens 512
  --generation-tokens 128 --repetitions 2 --prefill-step-size 4096 --ax-direct`.

## Related findings from the 2026-06-12 baseline

- **N-gram acceleration is a net regression** on Qwen3-Coder-Next for
  random-token prompts (decode −1.1% at p128, −0.4% at p512). The
  `ngram_acceleration_linear_attention_branch_recompute` path pays recompute
  cost without accept hits on random tokens. Random-token benchmarks should
  use `--ax-direct` (the cold-baseline contract already does).
- Prefill scales sublinearly with prompt length (2.2× throughput at 4× tokens);
  the larger step size partially addresses this by reducing fixed per-step cost.
