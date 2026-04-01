# Apple-to-Apple Benchmarks

This directory contains the canonical benchmark harness for comparing
`ax-engine` against `llama.cpp` on the same machine with the same model file,
same prompt/decode shapes, and the same serial outer-sample methodology.

## Quick Start

```bash
# Build release binaries first
cargo build -p ax-engine-bench --release

export REPO_DIR=$(pwd)

# Full comparison: AX Engine vs llama.cpp
./benchmarks/run_apple_to_apple.py \
  --model ./models/Qwen3-8B-Q4_K_M.gguf \
  --label qwen3-8b

# AX Engine only (fast iteration, no llama.cpp needed)
./benchmarks/run_apple_to_apple.py \
  --repo-dir "$REPO_DIR" \
  --model ./models/Qwen3-8B-Q4_K_M.gguf \
  --label qwen3-8b-dev \
  --ax-only

# AX Engine only for Qwen 3.5 9B
./benchmarks/run_apple_to_apple.py \
  --repo-dir "$REPO_DIR" \
  --model ./models/Qwen3.5-9B-Q4_K_M.gguf \
  --label qwen3.5-9b-ax-only \
  --ax-only \
  --prompt-tokens 512 \
  --decode-tokens 128 \
  --samples 5 \
  --cooldown-seconds 20

# AX Engine only, compared against a previous llama.cpp baseline
./benchmarks/run_apple_to_apple.py \
  --model ./models/Qwen3-8B-Q4_K_M.gguf \
  --label qwen3-8b-retest \
  --llama-baseline ./benchmarks/results/20260329-190458-001/llama/summary.json
```

## Modes

### Full mode (default)

Runs both AX Engine and `llama.cpp`, writes a side-by-side comparison. Requires
`llama-bench` installed (default: `/opt/homebrew/bin/llama-bench`).

### AX-only mode (`--ax-only`)

Runs only AX Engine. Produces a standalone summary with prefill/decode tok/s,
prefill plan, and command buffer count. Use this for fast iteration during
development when you only need AX numbers.

### Baseline-reuse mode (`--llama-baseline <path>`)

Runs only AX Engine, then compares against a previous `llama/summary.json`
from an earlier full run. Produces full comparison output (comparison.md/json/tsv)
without re-running `llama.cpp`. Useful when the `llama.cpp` baseline hasn't
changed but you've rebuilt AX.

## Methodology

The harness intentionally uses:

- serial execution, never parallel benchmarks
- one model at a time
- outer samples with median aggregation
- cooldown between samples
- matching prompt/decode shapes for both engines
- full GPU offload and Flash Attention enabled for `llama.cpp`

Current default settings:

- prompt tokens: `512`
- decode tokens: `128`
- decode depth: `512`
- samples: `5`
- cooldown: `20s`
- AX warmup iterations: `0`
- AX measure iterations: `1`
- `llama.cpp`: `-ngl 99 -b 2048 -ub 512 -ctk f16 -ctv f16 -fa 1 -t 12`

## All Options

```
--model MODEL              GGUF model path (required)
--label LABEL              Short run label (default: model stem)
--prompt-tokens N          Prefill token count (default: 512)
--decode-tokens N          Decode token count (default: 128)
--decode-depth N           Decode start depth (default: prompt-tokens)
--samples N                Outer sample count (default: 5)
--cooldown-seconds N       Cooldown between samples (default: 20)
--repo-dir PATH            Repository directory (defaults to script location or $REPO_DIR)
--ax-only                  Skip llama.cpp, report AX results only
--llama-baseline PATH      Reuse a previous llama/summary.json as baseline
--ax-bench PATH            AX bench binary (default: target/release/ax-engine-bench)
--llama-bench PATH         llama-bench binary (default: /opt/homebrew/bin/llama-bench)
--out-dir DIR              Output directory (default: benchmarks/results)
--timestamp YYYYMMDD-HHMMSS  Override result folder timestamp
```

## Listing Past Results

```bash
./benchmarks/list_results.py
./benchmarks/list_results.py --model-contains Qwen3.5
./benchmarks/list_results.py --json --limit 10
```
AX-only runs are included via `ax.json` and show in `ax_*` columns even when
`manifest.json`/`comparison.json` are not present.

## Output

The runner writes a timestamped directory under `benchmarks/results/` using the
format `YYYYMMDD-HHMMSS-001`.

**Full mode** output:

- `ax.json` — AX Engine raw results
- `llama/summary.json` — llama.cpp aggregated results
- `comparison.json` — Side-by-side comparison (machine-readable)
- `comparison.md` — Side-by-side comparison (human-readable)
- `comparison.tsv` — Tab-separated for spreadsheets
- `manifest.json` — Engine binaries and commands

**AX-only mode** output:

- `ax.json` — AX Engine raw results
- `summary.md` — Standalone AX summary with prefill plan and CB count

**Baseline-reuse mode** output: same as full mode, with the baseline
`llama/summary.json` copied into the run directory.

## Output Contract

`comparison.json` contains:

- model metadata
- benchmark parameters
- engine identity for AX and `llama.cpp`
- binary paths and AX command metadata
- AX prefill/decode medians
- `llama.cpp` prefill/decode medians
- percentage ratios
- artifact paths

`manifest.json` is the most direct machine-readable entrypoint when you need to
know which engine and which binaries produced a result folder.
`comparison.json` is the main summary used for future automation or README
refresh scripts.
