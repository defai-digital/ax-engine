# Benchmarking

This document describes how to benchmark AX Engine correctly and how to compare it against `llama.cpp` without mixing incompatible settings.

## Goals

Use this guide when you want to:

- measure AX Engine throughput or latency on a local model
- compare AX Engine against `llama.cpp`
- generate numbers that are reproducible enough to publish in repo docs or PRs

## Benchmarking Principles

The main failure mode is not usually a busy machine. It is methodology drift.

For example:

- comparing AX throughput mode to a different `llama.cpp` execution mode
- changing prompt or decode lengths between tools
- using different KV cache dtypes
- forgetting that `llama.cpp` Flash Attention can materially change results
- comparing one noisy single run against a median from repeated runs

Treat every comparison as invalid until the following are aligned:

- same GGUF file
- same prompt token count
- same decode token count
- same quantization
- same KV dtype where possible
- full GPU offload where possible
- repeated samples, not one-off runs

## Environment Hygiene

Before running a benchmark set:

1. Build release binaries.
2. Avoid running multiple inference jobs at the same time.
3. Check that the machine is mostly idle.
4. Record the machine state alongside the result.

Useful checks:

```bash
uptime
top -l 1 -n 20 -o cpu
ps -axo pid,%cpu,%mem,etime,command | sort -k2 -nr | head -n 30
```

What to look for:

- no other inference engine using CPU or GPU
- no parallel `ax-bench`, `llama-bench`, `ollama`, or model-serving job
- enough free memory to avoid paging pressure

Normal desktop processes such as `WindowServer`, editors, or browsers are usually acceptable if the machine is otherwise mostly idle, but fewer background apps is better.

## Build

From the repo root:

```bash
cargo build --workspace --release
```

AX benchmark binary:

```text
./target/release/ax-bench
```

## AX Engine Benchmarks

### Throughput

Basic run:

```bash
./target/release/ax-bench bench --model ./models/<model>.gguf
```

Recommended repeated run for publishable numbers:

```bash
./target/release/ax-bench bench \
  --model ./models/<model>.gguf \
  --deterministic \
  --samples 5 \
  --measure-iters 1 \
  --cooldown-ms 500
```

This reports median and mean throughput. For throughput comparisons, prefer the median.

### Latency

Latency mode disables the throughput-oriented decode path and keeps per-token latency meaningful:

```bash
./target/release/ax-bench bench \
  --model ./models/<model>.gguf \
  --intent latency
```

### Decode Profile

Use this to inspect the decode hot path and identify where time is actually spent:

```bash
./target/release/ax-bench profile --model ./models/<model>.gguf
```

### Important AX Notes

- Throughput mode usually selects `pipelined` decode.
- Latency mode usually selects `single_cb` decode.
- The selected prefill and decode plans are printed in the output. Keep them with the benchmark record.
- `--llama-parity-preset` is not a cross-engine truth mode. It only applies a small set of AX env defaults and can change performance significantly.
- If AX prints `PrefillPlan: mode=serial reason=unsupported_quant:...`, treat that prefill result as a fallback-path measurement, not a normal GPU-prefill comparison.
- Mixed-quant GGUFs can trigger that behavior when active layer tensors use `Q5_K`, because AX defaults `Q5_K` to decode-only baseline.
- If you benchmark the experimental route with `AX_METAL_EXPERIMENTAL_Q5K_PREFILL=1`, record that env var alongside the run and do not compare it against default AX numbers as if it were shipped behavior.

## `llama.cpp` Comparison Runs

Use `llama-bench`, not `llama-cli`, for engine-to-engine throughput comparisons.

### Required Comparison Settings

At minimum, align the following:

- `-m` same GGUF
- `-p` same prompt tokens
- `-n` same decode tokens
- `-ctk f16 -ctv f16`
- `-ngl 99` to fully offload on a single Apple GPU where supported
- `-r 5` or more

Recommended command:

```bash
/opt/homebrew/bin/llama-bench \
  -m ./models/<model>.gguf \
  -p 512 \
  -n 128 \
  -r 5 \
  -ngl 99 \
  -ctk f16 \
  -ctv f16 \
  -fa 1 \
  -o json
```

### Flash Attention Requirement

If the model and build support it, explicitly set:

```bash
-fa 1
```

Do not rely on the default. In local testing on this repo, leaving Flash Attention off understated `llama.cpp` performance enough to change the headline conclusion.

### Reading `llama-bench` Output

`llama-bench` emits two result rows:

- `n_prompt > 0, n_gen = 0`: prompt processing throughput
- `n_prompt = 0, n_gen > 0`: generation throughput

Use those as the `prefill` and `decode` comparison values.

## Apples-to-Apples Comparison Checklist

Before publishing AX vs `llama.cpp` numbers, confirm:

- same hardware
- same model file
- same token counts
- same quantization
- same KV dtype
- same GPU offload intent
- `llama.cpp` Flash Attention explicitly recorded
- repeated runs with medians
- no active competing inference job

If any of those are missing, label the result as exploratory, not authoritative.

One more invalid-comparison case:

- if AX prefill fell back to `mode=serial reason=unsupported_quant:...` while `llama.cpp` stayed on its normal prompt fast path, do not treat the prefill delta as an apples-to-apples engine conclusion
- in that case, the decode comparison is still useful, but the prefill comparison is mostly measuring AX's current `Q5_K` support boundary

## Suggested Reporting Format

Include:

- date
- machine, for example `Apple M3 Max`
- AX commit
- `llama.cpp` build or commit if available
- exact command lines
- prompt and decode token counts
- median prefill tok/s
- median decode tok/s
- notable routing details such as AX `Mode`, `PrefillPlan`, and `Plan`

Example table:

| Model | Engine | Prefill tok/s | Decode tok/s | Notes |
|---|---|---:|---:|---|
| Llama 3 8B Q4_K_M | AX | 642.0 | 58.1 | `pipelined`, `f16kv_hd128` |
| Llama 3 8B Q4_K_M | llama.cpp | 771.4 | 64.8 | median from `samples_ts`, `-fa 1`, `-ctk f16`, `-ctv f16` |

## Current Repo Snapshot

Retested on March 25, 2026 on Apple M3 Max with the corrected comparison method:

- AX command shape:

```bash
./target/release/ax-bench bench \
  --model ./models/<model>.gguf \
  --deterministic \
  --samples 5 \
  --measure-iters 1 \
  --cooldown-ms 500
```

- `llama.cpp` command shape:

```bash
/opt/homebrew/bin/llama-bench \
  -m ./models/<model>.gguf \
  -p 512 \
  -n 128 \
  -r 5 \
  -ngl 99 \
  -ctk f16 \
  -ctv f16 \
  -fa 1 \
  -o json
```

Recorded results:

| Model | AX prefill | llama.cpp prefill | AX vs llama.cpp | AX decode | llama.cpp decode | AX vs llama.cpp | AX notes |
|---|---:|---:|---:|---:|---:|---:|---|
| Gemma 3 12B Q4_K_M | 418.5 | 477.7 | 87.6% | 39.3 | 39.1 | 100.5% | `pipelined`, `attn=splitk_hd256/profile_preferred` |
| Gemma 3 27B Q4_K_M | 161.3 | 191.3 | 84.3% | 17.7 | 14.6 | 121.2% | `pipelined`, `attn=f16kv_hd128/stable` |
| Llama 3 8B Q4_K_M | 642.0 | 771.4 | 83.2% | 58.1 | 64.8 | 89.7% | `pipelined`, `attn=f16kv_hd128/stable` |
| Llama 3 70B Q4_K_M | 70.1 | 71.4 | 98.2% | 8.0 | 7.1 | 112.7% | `pipelined`, `PrefillPlan=mode=gpu_batch`, `q5k_prefill=experimental` |
| Qwen3 8B Q4_K_M | 631.4 | 664.8 | 95.0% | 55.1 | 59.8 | 92.1% | `pipelined`, `attn=f16kv_hd128/stable` |
| Qwen3 14B Q4_K_M | 269.6 | 334.0 | 80.7% | 33.4 | 20.8 | 160.6% | `pipelined`, `attn=f16kv_hd128/stable` |
| Qwen3 32B Q4_K_M | 126.3 | 129.4 | 97.6% | 13.1 | 12.0 | 109.2% | `pipelined`, `attn=f16kv_hd128/stable` |

`llama.cpp` values above are medians from `samples_ts` with `-fa 1`, `-ctk f16`, and `-ctv f16`. `AX vs llama.cpp` over `100%` means AX was faster.

70B note:

- `models/meta-llama-3-70b-instruct.Q4_K_M.gguf` contains an active `Q5_K` tensor, so shipped AX behavior still defaults that model to decode-only baseline for prefill.
- The published 70B row above records only the post-fix validation result with `AX_METAL_EXPERIMENTAL_Q5K_PREFILL=1`.
- The default AX path is intentionally excluded from the headline table because it is a fallback-path result, not a representative fast-path comparison.

Interpretation:

- Gemma 3 12B: `llama.cpp` led on prefill, decode was effectively tied.
- Gemma 3 27B: `llama.cpp` led on prefill, while AX led on decode in this retest.
- Llama 3 8B: `llama.cpp` led on both prefill and decode in this retest.
- Llama 3 70B: with `AX_METAL_EXPERIMENTAL_Q5K_PREFILL=1`, AX was close to `llama.cpp` on prefill and slightly ahead on decode in this local retest; the default fallback path is intentionally not treated as a headline benchmark row.
- Qwen3 8B: `llama.cpp` led on both prefill and decode, but AX remained within the same general range.
- Qwen3 14B: `llama.cpp` led clearly on prefill, while AX led materially on decode in this retest.
- Qwen3 32B: prefill was close, with `llama.cpp` slightly ahead, while AX led modestly on decode.
- Earlier local comparisons that omitted `-fa 1` for `llama.cpp` were not reliable enough to use as headline repo claims.

## Common Mistakes

- Comparing AX throughput mode against AX latency mode
- Comparing AX against `llama.cpp` with `flash_attn=false`
- Using different prompt lengths between runs
- Using a different GGUF for each engine
- Publishing a single run without reporting variance
- Changing background workload halfway through a benchmark set

## Minimal Benchmark Set

If you only need one clean comparison for a PR or README update, use:

```bash
./target/release/ax-bench bench \
  --model ./models/<model>.gguf \
  --deterministic \
  --samples 5 \
  --measure-iters 1 \
  --cooldown-ms 500

/opt/homebrew/bin/llama-bench \
  -m ./models/<model>.gguf \
  -p 512 \
  -n 128 \
  -r 5 \
  -ngl 99 \
  -ctk f16 \
  -ctv f16 \
  -fa 1 \
  -o json
```

Keep both outputs with the PR.
