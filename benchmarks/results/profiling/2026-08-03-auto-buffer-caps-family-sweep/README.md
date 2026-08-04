# Auto buffer caps family sweep — 2026-08-03 (M3 Max 128 GB, MLX 0.32.0)

Raw artifacts for the `qwen3_5` exclusion from `AX_MLX_AUTO_BUFFER_CAPS`
(`crates/ax-engine-mlx/src/weights.rs`). Probe contract mirrors the 6-bit MTP
matrix measurement: `POST /v1/generate/stream`, prefill = first-step
`runner_time_us`, prefix caches disabled, sampled T=0.6/top_p=0.95/top_k=20,
`ignore_eos`, 2 warmups + measured reps, one server process per run.

- `probe.py` — the probe driver.
- `35b-head-caps-on.log` / `35b-head-caps-off.log` — Qwen3.6-35B-A3B-6bit-MTP,
  273-token flappy_pipes prompt (token ids from the 6.12.1 matrix artifact),
  1000 generated tokens. caps ON degrades one-way (816 → 579 tok/s and still
  falling); caps OFF stays flat (937 → 895).
- `35b-decode-ab.log` — same model, decode-rate comparison: ~45.5 tok/s both
  configurations (the decode-trace gather-QMM win does not materialize on the
  server sampled path).
- `35b-interleaved-x4.log` — interleaved ON/OFF/ON/OFF, 8 reps, 400 generated
  tokens: ON 874.6/917.5 mean tok/s (wobble), OFF 977.5/972.2 (flat,
  rep spread < 2%).
- `fixed-binary-35b-27b.log` — server built with the family exclusion, no env
  overrides: 35B mean 927.7 (flat-ish), 27B flat 152→156 tok/s.
- `glm-interleaved-x4.log` — GLM-4.7-Flash-4bit (`glm4_moe_lite`): prefill and
  decode parity within thermal drift → family kept raised.
- `coder-next-interleaved-x4.log` — Qwen3-Coder-Next-4bit (`qwen3_next`):
  raise costs ~5–6% prefill (600.7/592.9 ms ON vs 564.0/564.6 ms OFF),
  sampled decode parity (~63 tok/s) → kept raised for its documented greedy
  server decode win (+28%, 2026-07-17 python-Session A/B).

Machine-noise note: this is an interactive desktop (background daemons,
thermal drift visible as uniform slowdowns inside the GLM series); only
interleaved adjacent-pair differences were treated as signal.
