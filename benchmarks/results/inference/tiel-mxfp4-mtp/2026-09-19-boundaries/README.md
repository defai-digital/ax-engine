# Tiel native API boundary checks

This follow-up adds debug-only explanations at existing residency-policy skip points. It does not change admission predicates, wired-limit override semantics, model execution, MTP depth, sampling, or cache behavior. The success message explicitly describes the wired limit as process-wide. See [operator diagnostics](../../../../../docs/mtp/tiel-prefill-diagnostics.md) for logging setup and limitations.

## Scope

- Host: MacBook Pro M5 Max, 128 GiB. All model runs were sequential on this host; neither pack ran on M3 hardware.
- Packs: the exact Tiel and Cyber-Tiel MXFP4 MTP exports qualified in the [preceding campaign](../2026-09-19-path/README.md).
- Base: `22a82562755ba14197f14a840be118f3be457f2a`. Rust 1.97.1, optimized `release-pyext`, MLX 0.32.2. `results.json` binds both extension hashes and final source hashes.
- Native Python streaming API; greedy, seed zero, ignore EOS, cold request KV, prefix reuse disabled, throughput MTP requested. This is not an HTTP, pressure, co-residency or performance qualification.
- Coding-prompt output budgets: 1, 2, 3, 4, 5, 16. One-token-prompt output budgets: 1, 2, 4, 5. Each runs twice within the same session. Budgets 3/4/5 straddle the three-draft plus target emission boundary.

## Results

| Check | Base + candidate, both packs | Result |
| --- | ---: | --- |
| Short valid requests | 80 | Exact output budget, no streaming overrun, one terminal response |
| Invalid requests | 16 | Expected exception type and message |
| Same-session recovery after rejection | 16 | Next valid 16-token request completed |

The four rejected inputs per process are zero budget, empty token input, negative budget and `2**32` budget. The first two raise `ValueError`; integer conversion boundaries raise `OverflowError`. All valid outputs match between base and candidate and across repetitions. The 16-token cases show actual MTP submissions and depth-three draft activity; short requests need not use every draft depth.

No generation defect was reproduced, so no numerical or output-truncation patch was made. No speed gain is claimed. The existing core/SDK zero-budget admission remains unchanged. The probes do not extend MTP defaults or certification.

Rust: 3,747 tests passed. Python: 209 passed, 26 skipped, 142 subtests passed. Formatting, MLX library Clippy, maturin, script checks, both qualification dry runs and canonical claims passed. Full-workspace Clippy still fails with exactly the previous error multiset in unchanged core tests. Post-run pack file hashes and MTP norms match the preceding campaign.

## Reproduce

```bash
python3 benchmarks/results/inference/tiel-mxfp4-mtp/2026-09-19-boundaries/verify.py --check-source
```

For model execution on the audited host, write `[{"token_ids": ...}]` from a recorded coding prompt into a cases JSON file. Use an explicit Python executable and `PYTHONPATH` pointing to the selected extension, matching its MLX dynamic library. Clear inherited `AX_*` / `MTPLX_*` flags, then run:

```bash
python probe.py --model MODEL_PATH --cases CASES_JSON --output RUN_JSON
```

Run one process at a time. The probe sets its own native/MTP/prefix controls and records the loaded extension hash. It does not configure a tracing subscriber; this matrix validates unchanged runtime behavior, not rendered CLI debug output.
