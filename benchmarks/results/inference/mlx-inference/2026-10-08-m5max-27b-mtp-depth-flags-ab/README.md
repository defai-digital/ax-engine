# Qwen 3.8 27B MXFP4 MTP: device-accept and cost-model depth A/B (2026-10-08)

MacBook Pro, Apple M5 Max, 128 GB unified memory, internal NVMe SSD (routine
bench host — signal only, never SKU qualification; not the Mac mini M4 Pro
64 GB best-experience SKU). macOS 27.0 (26A428).

Pack: `AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-MXFP4-MTP` @
`b2c5354f779e430d0c1733143db848a72b71c16e` (primary pin).

Binary under test: workspace 7.6.1, commit `ad9dfb1f` plus the 2026-10-08 MTP
change set (uncommitted patch: dead draft log-prob removal on greedy gate-0
paths, opt-in `AX_MLX_MTP_DEVICE_GREEDY_ACCEPT`, opt-in
`AX_MLX_MTP_COST_MODEL_DEPTH` cost-model depth controller with park), release
profile, `ax-engine-server` sha256
`4d86dd9fe52f6c1349d15a3e4c62cb9d76e96523f478e553c608603dc4726410`.
Toolchain: Homebrew cargo 1.98.0 (the host has no rustup, so the repo 1.97.1
pin is not honored — same provenance as the 2026-09-21 campaign), pip-wheel
MLX 0.32.3 from the repo `.venv`.

Contract (mirrors the 2026-09-15/21 peer campaigns): repository prompt suites
`flappy`, `long_code`, `python_modules_long`, `coding`, plus a
`random512` negative control (512 uniform random prompt tokens); 256 generated
tokens, greedy (seed 0), 2 warmups + 5 measured repetitions, 3 s cooldown,
prefix cache off, n-gram stacking off. Harness:
`scripts/bench_mlx_inference_stack.py`; MTP arms run
`--ax-ngram-accel --ax-mtp-policy required --ax-mtp-disable-ngram-stacking`,
the direct arm runs `--ax-direct`; mlx-lm lane skipped (internal flag A/B, not
an inference-stack claim). Every row's observed route is recorded in the lane
JSON (`mtp_head_only_verify_loop` / `direct_pipeline_baseline`).

## Measured (decode tok/s, median of per-case medians)

| Suite | direct | MTP (new code) | +device-accept | +cost-model | +both |
| --- | ---: | ---: | ---: | ---: | ---: |
| flappy | 35.35 | 70.35 (1.990x) | 70.31 (1.989x) | 69.38 (1.963x) | 69.30 (1.960x) |
| long_code | 35.35 | 73.46 (2.078x) | 73.49 (2.079x) | 70.31 (1.989x) | 70.25 (1.987x) |
| python_modules_long | 35.30 | 63.36 (1.795x) | 63.36 (1.795x) | 57.36 (1.625x) | 57.34 (1.624x) |
| coding | 35.37 | 59.05 (1.670x) | 59.07 (1.670x) | 57.20 (1.617x) | 57.21 (1.618x) |
| random512 | 35.27 | 54.56 (1.547x) | — | 54.94 (1.557x) | — |

Historical anchors on this host (different builds): MXFP4 MTP 76.70–76.90
tok/s on flappy (2026-09-15/21); direct AR 30.14 tok/s on the 6-bit pack
(2026-09-21). The 2026-10-08 direct row is 35.3 tok/s on MXFP4, so MTP ratios
are against a faster same-day direct baseline.

## Findings

- The MTP product path is intact on the change set: 1.62–2.08x over the
  same-binary direct baseline on every suite, route identity unchanged.
- `AX_MLX_MTP_DEVICE_GREEDY_ACCEPT=1` is performance-neutral (±0.1%). It
  removes the host-side acceptance loop and the full-window `predicted`
  download but keeps the two-eval split; its value is groundwork for a later
  eval-merge step, not a standalone win.
- `AX_MLX_MTP_COST_MODEL_DEPTH=1` is slower on these 256-token generations
  (-1.4% flappy, -4.3% long_code, -9.5% python_modules_long, -3.1% coding).
  On suites where MTP already wins, the warmup sweep + direct probes +
  duty-bounded staleness probes tax every request and never get to show the
  park stop-loss. The controller stays opt-in (default OFF); probe/warmup
  budget tuning or restriction to long-generation sessions is required before
  any default consideration.
- Negative control: even uniform random prompts leave MTP ahead (1.547x); no
  losing workload was found on this host/pack, so park never engaged.
- Greedy identity: every flagged arm (`device-accept`, `cost-model`, `both`)
  emits token-id sequences bit-identical to the unflagged MTP arm on every
  case (verified from captured `output_token_ids`). MTP-vs-direct streams
  diverge on some natural-language cases — the disclosed relaxed-profile
  near-tie semantics (ADR-033), unchanged by this patch set.

## Caveats

Single host, single day, one binary build; 256-token generations; bench-host
numbers are signal only and never qualify a SKU. This is not MTP-S/P/D
evidence; no public multiplier is licensed by these numbers. The mlx-lm
primary baseline was intentionally skipped for this internal A/B.

Raw: one JSON per suite/arm lane (per-case medians, trials, captured token
ids, route decisions) in this directory.
