# TurboQuant Quality Artifact Run

Generated: 20260514T202156Z

- Model directory: `.internal/models/gemma-4-e2b-it-4bit`
- Model id: `gemma-4-e2b-it-4bit`
- Model family: `gemma4`
- Head dim: `512`
- Baseline benchmark: `baseline.json`
- Candidate benchmark: `candidate.json`
- Decode outputs: `baseline-decode-outputs.json`, `candidate-decode-outputs.json`
- Quality metrics: `quality-metrics.json`
- Quality gate artifact: `quality-gate.json`
- Promotion readiness: `promotion-readiness.json`
- Same-shape fused decode microbench:
  `microbench-gemma-e2b-shape.json`
- Command log: `command.log`
- Commands: `commands.txt`
- Environment: `environment.txt`

This artifact is promotion evidence only if `quality-gate.json` validates and
`promotion-readiness.json` reports no blockers. Public support docs remain
experimental until that readiness report is clean.

Result: this is a negative performance artifact. The candidate uses the real
`fused_compressed_decode` path and passes exact replay quality, but
`promotion-readiness.json` still reports the performance blocker
`metrics.decode_tok_s_ratio_to_baseline must be >= 0.85`.
