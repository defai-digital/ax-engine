# Tiel restricted-PATH regression

A PATH-only `hw.memsize` probe could return unknown memory in service environments that omit `/usr/sbin`, silently disabling the audited Tiel no-wire optimization. The fix reuses the existing hardware helper with its absolute sysctl fallback. It also removes duplicate brand-query code and redundant fallback calls. No new policy, allocator lifecycle, kernel, MTP route, cache limit or operator-override semantics were introduced.

## Contract

- Hardware: MacBook Pro, Apple M5 Max, 128 GiB, AC power. One benchmark process at a time. Neither pack was run on M3 hardware.
- Base: `cb4dc8b3fb518acdf931d212f1b2505bdae9a0b9`. Candidate source hashes and both extension hashes are bound in `trials.json`. Rust 1.97.1, optimized `release-pyext`, MLX 0.32.2.
- Exact packs: `AutomatosX/AX-Tiel-Coder-35B-A3B-MLX-AXQ-MXFP4-MTP` at `5ab39b24bfd7f65203be9b7823b1840486f58b6d`; `AutomatosX/AX-Cyber-Tiel-Coder-35B-A3B-MLX-AXQ-MXFP4-MTP` at `fe05e871ec69ad9ae8eac01fd285555514ac7daf`. Pack manifest files and MTP norms checked before and after.
- Same `python-lru` token arrays from the previous campaign; greedy native throughput MTP, depth cap 3, cold KV, prefix reuse off, exactly 256 output tokens, two warmups and five measured requests per process, three-second inter-request idle.
- Normal PATH: `/usr/bin:/bin:/usr/sbin:/sbin`; restricted PATH: `/usr/bin:/bin`. All inherited AX and MTPLX flags removed before setting the harness controls.
- TTFT: native API entry to first committed callback. Completion: API entry to final committed callback. Decode excludes the first emitted batch. Load, tokenization and HTTP are excluded.
- `bench_path.py` extends the prior harness only by reading the old wired limit through `set_wired_limit(0)` **after all timed requests**. The process then exits.
- Initial candidate runs were excluded after a regression-test fixture correction and final rebuild. Every published candidate run uses the final binary. Base runs precede final candidate runs; sequential runs do not eliminate temporal noise.

## Results

Medians of five measured requests per cell:

| Model | PATH | Base TTFT ms | Fixed TTFT ms | Base completion tok/s | Fixed completion tok/s | Base decode tok/s | Fixed decode tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Tiel | normal | 139.99 | 141.84 | 195.21 | 195.12 | 217.71 | 217.99 |
| Tiel | restricted | 250.20 | 143.08 | 181.20 | 194.79 | 219.32 | 217.90 |
| Cyber-Tiel | normal | 144.30 | 144.25 | 219.09 | 219.12 | 248.79 | 248.88 |
| Cyber-Tiel | restricted | 256.22 | 143.53 | 200.85 | 219.28 | 250.38 | 248.90 |

All 56 requests (40 measured and 16 warmups) returned identical output token IDs for each model across both binaries and PATH environments. The base restricted-PATH processes retained a wired limit of 103,903,852,953 bytes; all other processes reported zero. This ties the launch-environment regression to the policy guard. Recorded telemetry confirms MTP submissions, accepted drafts and depth-three activity for every request.

The improvement restores the existing optimization in a previously broken launch environment. Normal-PATH results do not establish a further speed gain. This is one short coding workload, not a refreshed MTPLX comparison, long-context qualification, contention test, or mixed-model residency certification. The previous [matched MTPLX comparison](../2026-09-19-wired/README.md) remains separate.

## Validation and reproduction

The isolated macOS subprocess regression failed before the fix and passed after it. It covers a missing PATH command, a failing command, empty output and an invalid sysctl key. No process-global test environment mutation or additional dependency is needed.

Rust: 3,747 passed. Python: 209 passed, 26 skipped, 142 subtests passed. Formatting, MLX library Clippy, maturin, script checks, both qualification dry runs and canonical claims passed. Full-workspace Clippy still fails with the same pre-existing error multiset in unchanged core tests; it is not reported as green.

```bash
python3 benchmarks/results/inference/tiel-mxfp4-mtp/2026-09-19-path/verify.py --check-source
```

For runtime reproduction, extract either model's input case from `trials.json` into a one-element JSON array. Run `bench_path.py --engine ax --model MODEL_PATH --cases CASES_JSON --output RUN_JSON --warmups 2 --reps 5 --cooldown 3`, using an explicit Python executable and `PYTHONPATH` pointing to the selected extension. Alternate the two PATH values above; keep all other environment and input settings identical and run each process alone. MLX dynamic-library resolution must match the installed extension.
