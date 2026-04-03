# Benchmarking

This guide is the canonical way to benchmark AX Engine and run apples-to-apples comparisons with `llama.cpp`.

## Goals

Use this guide when you need to:

- measure AX throughput or latency on a local GGUF
- compare AX against `llama.cpp` with aligned settings
- produce reproducible artifacts for PRs and docs

## Canonical Workflow

Use the benchmark harness in `benchmarks/` as the default path.

Build once from repo root:

```bash
cargo build -p ax-engine-bench --release
```

### Full comparison (AX + llama.cpp)

```bash
./benchmarks/run_apple_to_apple.py \
  --model ./models/<model>.gguf \
  --label <label> \
  --samples 5 \
  --cooldown-seconds 20
```

### AX-only (fast iteration)

```bash
./benchmarks/run_apple_to_apple.py \
  --model ./models/<model>.gguf \
  --label <label> \
  --ax-only
```

### AX-only with previous llama baseline

```bash
./benchmarks/run_apple_to_apple.py \
  --model ./models/<model>.gguf \
  --label <label> \
  --llama-baseline ./benchmarks/results/<run-id>/llama/summary.json
```

### llama-only with previous AX baseline

```bash
./benchmarks/run_apple_to_apple.py \
  --model ./models/<model>.gguf \
  --label <label> \
  --llama-only \
  --ax-baseline ./benchmarks/results/<run-id>/ax.json
```

## What The Harness Aligns

`benchmarks/run_apple_to_apple.py` keeps comparison methodology consistent:

- same GGUF model path for both engines
- same prompt/decode token counts and decode depth
- serial outer samples with median aggregation
- cooldown between samples
- AX deterministic mode (`--deterministic`)
- AX warmup and measurement pinned to `--warmup-iters 0 --measure-iters 1`
- `llama.cpp` run with full offload and explicit cache/attention settings

Current `llama.cpp` defaults used by the harness:

- `-ngl 99`
- `-b 2048`
- `-ub 512`
- `-ctk f16 -ctv f16`
- `-fa 1`
- `--no-warmup`
- `-t 12`
- `-r 1` per outer sample

If you change any of these, record it in the result notes.

## Output Artifacts

Runs are written under `benchmarks/results/` in folders like `YYYYMMDD-HHMMSS-001`.

Full comparison output:

- `ax.json`
- `llama/summary.json`
- `comparison.json`
- `comparison.md`
- `comparison.tsv`
- `manifest.json`

AX-only output:

- `ax.json`
- `summary.md`

llama-only output:

- `llama/summary.json`
- `summary.md`

Useful fields to keep for reviews:

- AX medians: `prefill_tok_per_sec_median`, `decode_tok_per_sec_median`
- AX plans: `prefill_plan`, `decode_plan`
- AX routing/context: `prefill_route_family`, `prefill_route_detail`, `q5k_prefill_mode`, `support_note`
- AX scheduling counters: `prefill_command_buffers`, `decode_command_buffers`, `decode_buffer_barriers`
- Comparison medians and ratios in `comparison.json`
- Exact binary/command provenance in `manifest.json`

## Manual Commands (When Needed)

Prefer the harness for published numbers. Use manual commands only for targeted debugging.

AX throughput benchmark:

```bash
./target/release/ax-engine-bench bench \
  --model ./models/<model>.gguf \
  --deterministic \
  --samples 5 \
  --cooldown-ms 20000 \
  --warmup-iters 0 \
  --measure-iters 1 \
  --json-output /tmp/ax.json
```

AX latency benchmark:

```bash
./target/release/ax-engine-bench bench \
  --model ./models/<model>.gguf \
  --intent latency
```

AX speculative benchmark:

```bash
./target/release/ax-engine-bench speculative \
  --model ./models/<target>.gguf \
  --draft-model ./models/<draft>.gguf \
  --deterministic \
  --samples 5 \
  --cooldown-ms 20000 \
  --json-output /tmp/ax-spec.json
```

Manual `llama.cpp` prefill sample:

```bash
/opt/homebrew/bin/llama-bench \
  -m ./models/<model>.gguf \
  -p 512 -n 0 -d 0 -r 1 \
  --no-warmup \
  -ngl 99 -b 2048 -ub 512 \
  -ctk f16 -ctv f16 -fa 1 -t 12 \
  -o json
```

Manual `llama.cpp` decode sample:

```bash
/opt/homebrew/bin/llama-bench \
  -m ./models/<model>.gguf \
  -p 0 -n 128 -d 512 -r 1 \
  --no-warmup \
  -ngl 99 -b 2048 -ub 512 \
  -ctk f16 -ctv f16 -fa 1 -t 12 \
  -o json
```

## AX-Specific Caveats

- If AX reports `prefill_mode=serial` or `PrefillPlan: mode=serial ...`, treat prefill numbers as fallback-path measurements, not normal GPU-prefill parity.
- Always capture `prefill_plan` and `decode_plan` with published numbers.
- For Q5_K-heavy/mixed models, keep `q5k_prefill_mode` in the report.
- For Qwen3.5 route experiments, record `qwen35_shared_timeline_slots` when non-default.

## Publishing Checklist

Before sharing AX vs `llama.cpp` claims:

- use one model file for both engines
- keep prompt/decode/depth aligned
- use repeated outer samples and report medians
- run serially (no concurrent inference jobs)
- include machine details (for example, `Apple M3 Max`)
- include AX commit and `llama.cpp` build/commit where available
- include exact commands or `manifest.json`
- attach `comparison.json` (or `ax.json` for AX-only)
- include plan fields (`prefill_plan`, `decode_plan`)

## Listing Existing Runs

```bash
./benchmarks/list_results.py
./benchmarks/list_results.py --model-contains Qwen3.5
./benchmarks/list_results.py --json --limit 20
```

For option details, see:

- `./benchmarks/run_apple_to_apple.py --help`
- `./target/release/ax-engine-bench --help`
- `./target/release/ax-engine-bench bench --help`
- `./benchmarks/README.md`
