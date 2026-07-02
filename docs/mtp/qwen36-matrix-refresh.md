# Qwen3.6 AX-only Multi-Suite MTP Results

This page keeps the 2026-06-29 AX Engine-only Qwen3.6 pure-MTP result table
that used to live in the README. It is an artifact archive for comparing AX
Engine across prompt suites, not a front-page session or cross-engine
leaderboard. The README keeps only the current peer decode view and links here
for regression review.

These rows keep AX Engine pure-MTP measurements separate from peer-engine rows.
Use this table for AX Engine's Qwen3.6 MTP throughput across prompt suites, and
use the peer comparison for the `flappy`-only cross-engine view with the same
generated-token and repetition contract.

## Result Table

| Target | Suite | Depth | AX MTP decode | AX MTP prefill | AX MTP TTFT | AX accept |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6 27B 4-bit | `flappy` | 3 | 60.3 tok/s | 649.2 tok/s | 495 ms | 99.5% |
| Qwen3.6 27B 4-bit | `long_code` | 3 | 60.3 tok/s | 782.5 tok/s | 917 ms | 99.4% |
| Qwen3.6 27B 4-bit | `python_modules_long` | 3 | 49.4 tok/s | 660.7 tok/s | 522 ms | 97.2% |
| Qwen3.6 27B 6-bit | `flappy` | 3 | 40.5 tok/s | 621.5 tok/s | 521 ms | 99.4% |
| Qwen3.6 27B 6-bit | `long_code` | 3 | 38.9 tok/s | 718.4 tok/s | 999 ms | 99.5% |
| Qwen3.6 27B 6-bit | `python_modules_long` | 3 | 33.3 tok/s | 623.8 tok/s | 560 ms | 96.7% |
| Qwen3.6 35B-A3B 4-bit | `flappy` | 1 | 168.2 tok/s | 1,731.4 tok/s | 185 ms | 99.8% |
| Qwen3.6 35B-A3B 4-bit | `long_code` | 1 | 167.5 tok/s | 2,558.8 tok/s | 280 ms | 99.8% |
| Qwen3.6 35B-A3B 4-bit | `python_modules_long` | 1 | 160.8 tok/s | 1,843.2 tok/s | 185 ms | 98.3% |
| Qwen3.6 35B-A3B 6-bit | `flappy` | 1 | 134.9 tok/s | 1,491.8 tok/s | 216 ms | 99.8% |
| Qwen3.6 35B-A3B 6-bit | `long_code` | 1 | 133.6 tok/s | 2,281.8 tok/s | 314 ms | 99.8% |
| Qwen3.6 35B-A3B 6-bit | `python_modules_long` | 1 | 135.5 tok/s | 1,631.2 tok/s | 210 ms | 98.4% |

## Methodology

- Generated tokens: `1000`
- Measured repetitions: `5` per prompt case after the AX warmup pass
- Cooldown: 15 s between repetitions, 10 s between prompt cases
- Sampling: `temperature=0.6`, `top_p=0.95`, `top_k=20`
- Mode: pure MTP, with no MTP+n-gram stacking

Pure-MTP verification is enforced by the summary builder: AX MTP artifacts with
non-zero n-gram accepted, proposed, submitted, or hit-step telemetry fail
summary generation.

## Artifacts

- Matrix summary:
  [`summary.md`](../../benchmarks/results/mtp-qwen36-matrix/2026-06-29-qwen36-mtp-matrix/summary.md),
  [`summary.json`](../../benchmarks/results/mtp-qwen36-matrix/2026-06-29-qwen36-mtp-matrix/summary.json)
- Follow-up MTPLX enablement smoke:
  [`summary.md`](../../benchmarks/results/mtp-qwen36-matrix/2026-06-29-mtplx-enabled-smoke/summary.md)
- Peer enablement smoke:
  [`summary.md`](../../benchmarks/results/mtp-qwen36-matrix/2026-06-29-peer-mtp-enable-smoke/summary.md)

## Smoke Notes

The MTPLX enablement smoke verifies that the local reference MTPLX loader can
attach and run the Qwen3.6 27B 4-bit, 35B-A3B 4-bit, and 35B-A3B 6-bit MTP
heads after loading the reference checkout and complete safetensor artifacts.
That smoke uses 16 generated tokens, one measured repetition, and no cooldown,
so it is a loader validation artifact rather than a promoted throughput row.

The peer enablement smoke verifies lightning-mlx on the same 27B 4-bit,
35B-A3B 4-bit, and 35B-A3B 6-bit MTP artifacts after normalizing the local MTP
sidecar layout and 6-bit MTP group-size inference in the benchmark adapter.
Rapid-MLX is still listed as unsupported for these Qwen3.6 MTP rows because its
scheduler starts with the shared artifacts but skips MTP installation for this
generation flow, so running it would measure non-MTP decode.

## Related

- [Qwen3.6 MTP peer benchmark](qwen36-peer-comparison.md)
- [MTP docs index](README.md)
