# S1 single-process vs mlxcel — official 3-rep with prefix reuse (2026-07-28)

Locked S1 contract (Qwen3.5-9B interactive stream + Gemma 4 12B
13.8k-token long prefill from ONE AX process vs mlxcel v0.4.2 one
process per model), Apple M5 Max 128 GB, fresh processes per trial,
alternating order, greedy sampling.

- Candidate: branch `perf/prefix-reuse` (runtime 1750cc4c; target
  manifest `benchmarks/manifests/targets/ax-qwen-gemma-m5max-thr-quanta.json`
  with `AX_MLX_PREFIX_CACHE_MAX_BYTES=8589934592`).
- Baseline: mlxcel v0.4.2 @ 1b9a0018.

| rep | thr (>=1.15) | ttft p95 (<=0.90) | gap p95 (<=0.90) | errors |
| --- | --- | --- | --- | --- |
| 1 | 5.031 | 0.040 | 0.259 | 0 |
| 2 | 4.989 | 0.041 | 0.261 | 0 |
| 3 | 5.015 | 0.040 | 0.258 | 0 |

Files:
- `campaign.json` — run ledger (pair contracts, artifact SHAs, commands).
- `artifacts/` — the six raw trial artifacts (3 AX, 3 mlxcel).
- `px-cmp-r{1..3}.json` — per-rep gate evaluations
  (`scripts/compare_qwen_gemma_flip.py`).
- `audits/` — token-equivalence evidence:
  `canonical-warm_repeat.json` (5/5 PASS, the CI merge gate),
  `canonical-warm_extend.json` (2/5 — the known short-prompt recompute
  non-determinism documented in PERFORMANCE-RESULTS),
  `fin-s1-*`/`gt-s1-*` — the exact 13.8k S1 text passing both
  warm_repeat and warm_extend token-exactly, with and without the
  cold-grid snap.
