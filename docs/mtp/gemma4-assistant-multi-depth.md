# Gemma 4 assistant MTP — multi-depth drafting (decode speedup)

**Change.** Lift the Gemma 4 assistant drafter from 1 token/step to **depth 2**,
drafting recurrently via the assistant's own `post_projection` backbone-hidden
estimate. **1.10–1.20x decode throughput** on the 26B model while holding
assistant accept **>97%** on every fair-MTP suite. Default ships as depth-2 with
a **0.85** first-token / **0.999** deep confidence gate; correctness-preserving.

Files: `crates/ax-engine-mlx/src/runner.rs`
(`load_gemma4_assistant_mtp_runtime`, `gemma4_assistant_draft_token`),
`crates/ax-engine-mlx/src/gemma4_assistant_mtp.rs` (gate + depth constants).
Probe: `crates/ax-engine-mlx/src/bin/gemma_depth_probe.rs`.

## Why depth > 1 works (and was blocked)

The assistant was hard-capped at depth 1 (`config.max_depth.min(1)`), so Gemma's
speculative speedup was structurally limited to ~2x — while the Qwen MTP head
drafts depth-3. The cap was conservative, not fundamental:

- The assistant is **stateless per decode step**: it has no key/value projection
  of its own and re-reads the *target's* frozen KV cache. The multi-depth path
  peeks shared full/sliding K/V (and any sliding ring layout) once via
  `Gemma4AssistantDraftSession::bind_target_cache`, then reuses those arrays for
  every depth step — no cache surgery and no per-layer re-peek. Q RoPE uses
  `rope_dynamic` with a scalar Int32 offset array so position is a graph input
  (compile-ready shape; `AX_MLX_GEMMA4_ASSISTANT_COMPILE` remains opt-in).
- `gemma4_assistant_forward_one` already computes a `post_projection`
  "backbone hidden" estimate of the drafted position and **returned it only to
  have the caller discard it** (`let Ok((logits, _projected_hidden)) = …`). That
  projection exists for exactly one purpose: to seed the next recurrent draft.

Depth-2 drafting feeds that estimate back as the next step's backbone hidden
(token = draft-1, RoPE position + 1, same frozen target cache) and advances one
position. The 2nd-token query cannot attend draft-1's position (no KV exists for
it), but draft-1's signal flows through the residual stream — and the benchmark
confirms that is enough to keep the 2nd token at ~97% accept.

Gemma is **dense** attention (sliding + full), so a partial-reject rollback is a
cheap recompute of the committed prefix — unlike the 27B linear-attention path
that made tree-draft non-viable (`docs/mtp/tree-draft-phase-a.md`).

## The gate is mandatory and must stay tight

Each draft position is proposed only when its T=1.0 argmax confidence clears a
gate; a miss stops drafting (correctness-preserving — a short draft just verifies
fewer speculative positions, never a changed committed token). Unlike the
throughput-tuned Qwen head (gate 0.90), the Gemma deep gate must stay **tight**:
a wrong deep draft costs a full target recompute, so ungated multi-depth is
net-negative (a recompute storm — the greedy probe measured python_modules_long
collapsing to ~0.65 accept / 1.6 tok/fwd ungated).

The canonical 26B benchmark (M5 Max, T=0.6, top-p 0.95, top-k 20, chat-templated
flappy/long_code/python_modules_long, multi-prompt, n-gram stacking off) found a
uniform **0.999** gate strictly dominates a looser 0.99 deep gate: it raises
accept on every suite at **identical** throughput — the looser gate bought no
speed, only lower accept.

| deep gate (long_code) | assistant accept | decode tok/s |
|-----------------------|------------------|--------------|
| 0.99                  | 0.9656           | 140.4        |
| 0.995                 | 0.9650           | 138.4        |
| **0.999**             | **0.9769**       | 139.3        |

## Result (canonical, depth-2 @ uniform 0.999 vs depth-1)

| suite               | d1 accept | d2 accept | d1 tok/s | d2 tok/s | speedup |
|---------------------|-----------|-----------|----------|----------|---------|
| flappy              | 0.9874    | 0.9831    | 136.9    | 151.3    | 1.105x  |
| long_code           | 0.9812    | 0.9769    | 125.8    | 139.3    | 1.107x  |
| python_modules_long | 0.9901    | 0.9817    | 128.4    | 154.5    | 1.203x  |

All suites hold accept **>97%** at **1.10–1.20x** decode. Depth-2 is accept-neutral
vs depth-1 (the 2nd token only fires in confident contexts).

## Why depth-2, not depth-3

Depth-3 cannot satisfy the >97% accept constraint: the greedy probe measured the
hardest suite (python_modules_long) at ~0.937 accept at depth-3 even with a tight
gate. So **depth-2 @ 0.999 is the constrained optimum** — no further optimization
chance under "improve decode rate while holding >97% accept": depth-3 breaks the
constraint, a looser gate lowers accept for no speed, and the gate cannot go
above 0.999.

## Tuning knobs (env)

- `AX_MLX_GEMMA4_ASSISTANT_MTP_MAX_DEPTH` — draft depth ceiling (default **2**;
  set 1 to restore single-token drafting). Overrides the prepared contract's
  conservative `max_depth=1` (a runtime capability of the same weights).
- `AX_MLX_GEMMA4_ASSISTANT_MTP_DRAFT_MIN_CONFIDENCE` — first-token gate (default
  0.85).
- `AX_MLX_GEMMA4_ASSISTANT_MTP_DEEP_DRAFT_MIN_CONFIDENCE` — deep-position
  (2nd token+) gate (default 0.999; loosen toward 0.99 to trade accept for
  speculation on easy content).
- `AX_MLX_GEMMA4_ASSISTANT_LAZY_MULTI_DEPTH` — fuse multi-depth drafting into a
  **single materialize** (lazy argmax token chain + GPU-exact confidences).
  **Default OFF** (opt-in `=1`). Host gates still apply after materialisation
  (same accepted-prefix contract). Depth-1 and exact-cpu confidence mode keep
  the per-depth path. 12B same-artifact A/B was accept-neutral but not a clear
  decode win when deep drafts rarely clear the 0.999 deep gate.

## Reproduce

```bash
# Probe (greedy, deterministic; faithful per-step because the assistant is
# stateless): sweeps depth/gate schedules and reports tok/fwd + accept.
cargo run --release --bin gemma_depth_probe -- <target_model_dir> 400
#   AX_GEMMA_PROMPT_FILE=<ids>   AX_GEMMA_SCHEDULES="0.999;0.999,0.999"

# Canonical production A/B (T=0.6, chat templates, n-gram off):
AX_MLX_GEMMA4_ASSISTANT_MTP_MAX_DEPTH={1,2} \
  ax-engine-bench generate --mlx --mlx-model-artifacts-dir <dir> \
  --tokens <chat_templated_ids> --max-output-tokens 400 \
  --temperature 0.6 --top-p 0.95 --top-k 20 --json
# assistant accept = ax_mlx_gemma4_assistant_mtp_{accepted,draft}_tokens
# decode tok/s     = ax_mtp_emitted_tokens / (ax_mlx_decode_wall_us / 1e6)
```

Note: the `bench_gemma4_assistant_mtp.py` harness passes `--ax-mtp-max-depth` and
does not forward `AX_MLX_GEMMA4_ASSISTANT_MTP_MAX_DEPTH` through its inner
subprocess, so use the direct `ax-engine-bench generate` form above to A/B the
assistant depth.
