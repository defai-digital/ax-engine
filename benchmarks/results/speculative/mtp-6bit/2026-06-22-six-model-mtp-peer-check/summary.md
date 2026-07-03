# 2026-06-22 Six-Model MTP Peer Check

Scope: `flappy` prompt suite, six 6-bit `download-mtp` targets, MTP-only. No `mtp-ngram` rows were run or promoted.

Method: sampled decode (`temperature=0.6`, `top_p=0.95`, `top_k=20`), 1000 generated tokens, 5 measured repetitions per prompt case, 30s cooldown, 10s inter-case cooldown, cold prefill/prefix cache disabled. AX rows used the prepared paths under `/Volumes/Ext4T/models/hub/.../snapshots/v1`.

The generic AX harness performs one warmup run per prompt case. The artifacts record `repetitions=5`, cooldown, prompt hashes, route identity, MTP depth, accept counters, prefill, decode, and TTFT.

| Target | AX MTP mode | Depth | AX decode median tok/s | AX prefill median tok/s | AX TTFT median ms | AX accept | MTPLX | lightning-mlx |
|---|---:|---:|---:|---:|---:|---:|---|---|
| `qwen3.6-27b-6bit` | Qwen fused sidecar | 3 | 42.1 | 632.7 | 508 | 99.5% | N/A: MTPLX 0.3.7 rejects the 6-bit `qwen-dense` runtime contract as unsupported | N/A for promoted matrix: Lightning rows are excluded by repo policy; ad-hoc smoke worked but is not promoted |
| `qwen3.6-35b-a3b` | Qwen fused sidecar | 1 | 141.5 | 1561.8 | 212 | 99.8% | N/A: MTPLX 0.3.7 is not a promoted 6-bit matrix peer for this package | N/A for promoted matrix: Lightning rows are excluded by repo policy |
| `gemma-4-12b` | Gemma assistant-MTP | 2 | 62.2 | 1701.7 | 214 | 99.3% | N/A: no MTPLX Gemma assistant-MTP runner for this package | N/A: no lightning-mlx Gemma assistant-MTP runner for this package |
| `gemma-4-26b` | Gemma assistant-MTP | 1 | 112.9 | 2395.0 | 148 | 99.8% | N/A: no MTPLX Gemma assistant-MTP runner for this package | N/A: no lightning-mlx Gemma assistant-MTP runner for this package |
| `gemma-4-31b` | Gemma assistant-MTP | 1 | 28.1 | 701.9 | 516 | 99.6% | N/A: no MTPLX Gemma assistant-MTP runner for this package | N/A: no lightning-mlx Gemma assistant-MTP runner for this package |
| `glm-4.7-flash` | GLM built-in MTP sidecar | 1 | 91.5 | 1694.5 | 163 | 98.2% | N/A: no MTPLX GLM built-in MTP runner for this package | N/A: no lightning-mlx GLM built-in MTP runner for this package |

Pure-MTP verification: every completed AX artifact has `ax_mtp_ngram_accepted_tokens=0` and no n-gram submitted/proposed tokens across all recorded prompt cases.

Artifacts:

- `qwen3.6-27b-6bit/flappy/ax_engine.json`
- `qwen3.6-35b-a3b/flappy/ax_engine.json`
- `gemma-4-12b/flappy/ax_engine.json`
- `gemma-4-26b/flappy/ax_engine.json`
- `gemma-4-31b/flappy/ax_engine.json`
- `glm-4.7-flash/flappy/ax_engine.json`

Peer checks:

- MTPLX environment was restored by reinstalling Homebrew `python@3.13`, making `/opt/homebrew/var/mtplx/venv-0.3.7/bin/python` runnable again.
- MTPLX smoke against `qwen3.6-27b-6bit` failed closed: `compatibility.can_run=false; runtime_compatibility=unsupported; qwen-dense runtime contract detected; not supported in v0.2.0`. The diagnostic-only `--allow-unverified-model` path was not used.
- Lightning raw optimized-model smoke could not run because `Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed` was not cached.
- Lightning prompt-suite ad-hoc smoke against the exact Qwen3.6 27B 6-bit package completed, but repo docs exclude Lightning rows from the promoted MTP matrix, so it remains diagnostic only.
