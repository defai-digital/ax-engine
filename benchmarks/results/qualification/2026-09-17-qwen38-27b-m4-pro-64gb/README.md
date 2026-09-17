# Qwen 3.8 27B product gate on Mac mini M4 Pro 64 GB

Local operator qualification of the installed, fully bundled release wheel.
This is **Candidate** evidence, not MTP Tier 2, `release_ready`, or a published
release attestation.

- Clean source: `8c8217b2` (full commit and artifact hashes in `summary.json`).
- Host: Mac mini M4 Pro, 64 GB, macOS 26.6.2.
- Pack: `AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-6bit-MTP` at
  `3e290738e96972307c6aeb9934ab170ca0eae1c1`.
- Direct: 32/32 hard QA, zero soft failures, 7/7 product-surface probes, no skips.
- MTP: 32/32 hard QA, zero soft failures, 7/7 product-surface probes, no skips.
- The 32 cases per route are **16 stratified questions in two stream modes**;
  they are not 32 independent questions or a representative accuracy benchmark.
- Actual server route probes: direct 0 draft / 0 verify; MTP 63 draft / 86 verify.
- Inline video was checked against a text-only control: 22 vs 16 prompt tokens.
  This proves tokenized input admission, not advanced visual understanding.
- The same 64-token greedy probe differed between direct and MTP starting at
  output index 25 (zero-based). Do not infer greedy equivalence from the QA pass.

`route-direct.json` and `route-mtp.json` preserve the selected original response
fields, including full route counters and output tokens; machine-local runtime
paths are omitted. `summary.json` contains the source/pack/wheel bindings, test
identities and product-surface results. Internal raw logs retain full local
paths and the failed intermediate runs.

Reproduction procedure and build-manifest schema:
[Testing](../../../../docs/TESTING.md#primary-qualification-mac-mini-m4-pro-64-gb).
Run `scripts/qualify_qwen38_27b.py --run` from the tested clean checkout with
the specified wheel, source-bound build manifest, and pinned model inventory.
A dry run or directory preflight does not reproduce this hardware result.
