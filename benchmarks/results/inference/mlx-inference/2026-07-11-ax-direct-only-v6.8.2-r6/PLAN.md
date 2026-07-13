# Plan: AX-only direct-mode refresh (r6)

## Goal

Re-measure **direct** (non-speculative, n-gram off) AX Engine throughput for
the README Gemma 4 + Qwen 3.6 matrix so prefill / TTFT / decode medians reflect
the current `v6.8.2` tree. Compare against the 2026-07-11 clean high-water cells
in the README tables (AX column only).

## Why now

README direct cells show large **prefill and TTFT** deficits vs `mlx_lm` on
dense/large Gemma 4 and Qwen 3.6 rows while **decode** leads. Fresh AX-only
numbers on the current commit establish whether that gap is still present after
subsequent decode-path work.

## Contract (matches publication high-water)

| Setting | Value |
| --- | --- |
| Mode | `--ax-direct-only` → stack gets `--skip-mlx-lm --ax-direct` |
| Policy | `direct_no_ngram_acceleration` |
| Prompt tokens | 128, 512, 2048 |
| Generation | 128 (greedy) |
| Warmups / measure | 2 / 5 |
| Cooldown | 15 s |
| Prefill chunk | 2048 (server default for this path) |
| Prefix cache | disabled for cold prefill/TTFT |
| Binary | `target/release/ax-engine-server` (`--no-build-ax-engine` in row runs) |
| Load gates | default publication gates (`max_load_average=2.0`, top process CPU ≤50%) |
| Host | Apple M5 Max / 128 GB / macOS 26+ |

## Rows

All inventory rows from
`benchmarks/manifests/llama_cpp_metal/inventory.json` (Gemma 4 E2B/E4B/26B/31B
4/6-bit + Qwen 3.6 27B/35B-A3B 4/6-bit; 8-bit inventory rows if present may
skip when artifacts missing).

## Steps

1. **Preflight** — resolve HF snapshots (`config.json` + `model-manifest.json` + `*.safetensors`).
2. **Build** — clean `cargo build -p ax-engine-server --release` on current HEAD.
3. **Run** — `scripts/bench_ax_only_sweep.py --ax-direct-only` into this directory.
4. **Summarize** — extract median prefill / decode / TTFT vs 2026-07-11 clean README cells.

## Out of scope

- No `mlx_lm` or llama.cpp re-runs (AX-only).
- No MTP / n-gram / README chart rewrite until numbers are reviewed.
- No prefill-kernel code changes in this pass (measurement first).

## Expected runtime

~60–90 minutes for a full 12-row success path (prior clean run ~72 min).
