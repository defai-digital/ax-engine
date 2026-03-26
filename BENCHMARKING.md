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

Machine-readable artifact:

```bash
./target/release/ax-bench bench --model ./models/<model>.gguf --json
./target/release/ax-bench bench --model ./models/<model>.gguf --json-output /tmp/ax-bench.json
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
- Mixed-quant GGUFs with active `Q5_K` layer tensors now use AX's conservative GPU prefill route by default.
- For `Q5_K` prefill runs, also record whether AX auto-used:
  - `q5k_prefill=base`
  - `q5k_prefill=small_n`
- AX currently auto-selects `small_n` only when the mapped model is
  predominantly `Q5_K` and the prompt batch is small (`<= 32` tokens).
- `ax-bench bench`, `ax-bench profile`, and `ax-bench soak` JSON now emit this as a first-class field:
  - `q5k_prefill_mode`
- `ax-bench soak` summary now also prints:
  - `PrefillPlan: ...`
  - `Q5KPrefill: ...` when present
- `ax-bench speculative` summary now also prints:
  - `PrefillPlan: ...`
  - `Q5KPrefill: ...` when present
- `ax-bench speculative` also supports machine-readable artifacts now:
  - `--json`
  - `--json-output <path>`
- If you are doing route A/B validation, record any forced override explicitly:
  - `AX_METAL_EXPERIMENTAL_Q5K_PREFILL_VARIANT=base`
  - `AX_METAL_EXPERIMENTAL_Q5K_PREFILL_VARIANT=small`

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
- in that case, the decode comparison is still useful, but the prefill comparison is mostly measuring an unsupported-quant fallback rather than AX's normal GPU path
- if AX auto-used `q5k_prefill=small_n`, record that explicitly because it is narrower than the base conservative route

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

Retested on March 26, 2026 on Apple M3 Max with one fresh AX-vs-`llama.cpp` rerun set:

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
  -r 3 \
  -ngl 99 \
  -ctk f16 \
  -ctv f16 \
  -fa 1 \
  -o json
```

Recorded results:

| Model | AX prefill | llama.cpp prefill | AX vs llama.cpp | AX decode | llama.cpp decode | AX vs llama.cpp | AX notes |
|---|---:|---:|---:|---:|---:|---:|---|
| Gemma 3 12B Q4_K_M | 420.5 | 463.3 | 90.8% | 39.6 | 35.8 | 110.5% | `pipelined`, `attn=cache/stable` |
| Gemma 3 27B Q4_K_M | 155.7 | 170.2 | 91.5% | 17.8 | 15.1 | 117.9% | `pipelined`, `attn=f16kv_hd128/stable` |
| Llama 3 8B Q4_K_M | 673.6 | 639.2 | 105.4% | 61.1 | 47.1 | 129.8% | `pipelined`, `attn=mistral_bc64/experimental` |
| Llama 3 70B Q4_K_M | 55.0 | 66.8 | 82.4% | 6.0 | 6.3 | 95.6% | `pipelined`, `PrefillPlan=mode=gpu_batch`, `q5k_prefill=base` |
| Qwen3 8B Q4_K_M | 659.5 | 736.7 | 89.5% | 58.3 | 60.3 | 96.7% | `pipelined`, `attn=mistral_bc64/experimental` |
| Qwen3 14B Q4_K_M | 277.0 | 408.2 | 67.9% | 34.9 | 35.6 | 98.1% | `pipelined`, `attn=mistral_hd128/profile_preferred` |
| Qwen3 32B Q4_K_M | 104.9 | 150.7 | 69.6% | 9.2 | 14.9 | 61.7% | `pipelined`, `attn=mistral_bc64/experimental` |

`llama.cpp` values above are medians from `samples_ts` on the current local Homebrew `llama-bench` build (`build_commit 342d6125b`, `build_number 8500`) with `-fa 1`, `-ctk f16`, and `-ctv f16`. `AX vs llama.cpp` over `100%` means AX was faster.

70B note:

- `models/meta-llama-3-70b-instruct.Q4_K_M.gguf` contains an active `Q5_K` tensor.
- Shipped AX now routes that mixed-quant case through the conservative `Q5_K` GPU prefill path by default.
- The published 70B row above is therefore representative of current shipped behavior, not an opt-in validation path.

`Q5_K` prefill auto-routing snapshot:

- `models/meta-llama-3.1-8b-instruct-q5_k_m.gguf`
  - command shape:
    - `target/release/ax-bench bench --model ... --prompt-tokens 16 --decode-tokens 32 --warmup-iters 1 --measure-iters 1`
  - result:
    - `PrefillPlan: ... q5k_prefill=small_n`
    - prefill `108.1 tok/s`
    - decode `35.0 tok/s`
- `models/meta-llama-3-70b-instruct.Q4_K_M.gguf`
  - same command shape
  - result:
    - `PrefillPlan: ... q5k_prefill=base`
    - prefill `27.0 tok/s`
    - decode `8.2 tok/s`

Interpretation:

- AX currently auto-uses the small-`N` route only on the
  predominant-`Q5_K` 8B file
- the mixed-quant 70B file stays on the base conservative route
- benchmark records for `Q5_K` prefill should include the exact
  `q5k_prefill=...` label

Interpretation:

- Gemma 3 12B: after the March 26 Gemma-specific Metal fusions, AX is close on prefill and now leads on decode.
- Gemma 3 27B: after the March 26 Gemma-specific Metal fusions, AX is close on prefill and now leads on decode.
- Llama 3 8B: in this March 26 rerun, AX leads on both prefill and decode.
- Llama 3 70B: AX remains below current local `llama.cpp` on both prefill and decode, and the published row now reflects the shipped default mixed-quant route.
- Qwen3 8B: current local `llama.cpp` leads modestly on both prefill and decode.
- Qwen3 14B: current local `llama.cpp` leads strongly on prefill, while decode is close.
- Qwen3 32B: current local `llama.cpp` leads on both prefill and decode in this rerun.
- Earlier mixed-date rows and earlier local comparisons that omitted `-fa 1` for `llama.cpp` are not reliable enough to keep as headline repo claims.

March 26 exact-shape prefill route sanity check:

- Synthetic GPU microbench on the dominant `pp512` dense-model shapes showed the same ordering across `Llama 3 8B`, `Qwen3 14B`, and `Gemma 3 12B`: the current `f32` batch route beat `f16in`, and `f16in_bn32` was slower still.
  - artifact: `automatosx/tmp/gpu-microbench-exact-prefill-routes-suite2-2026-03-26.json`
- The paired FFN `f16in` kernel also lost to two separate `f16in` projections on those same shapes.
  - artifact: `automatosx/tmp/gpu-microbench-exact-prefill-routes-suite2-2026-03-26.json`
- End-to-end warmed `prefill-profile` reruns confirmed the same direction for the still-slower published rows:
  - `Llama 3 8B`: `AX_METAL_BATCH_F16_IO=1 AX_METAL_BATCH_F16_PAIR=1` dropped to `323.4 tok/s`, which remains far below the current headline prefill row
    - artifact: `automatosx/tmp/llama3-8b-prefill-profile-hot-pp512-f16in-pair-2026-03-26.json`
  - `Gemma 3 12B`: the same route dropped to `199.4 tok/s`, also well below the current headline prefill row
    - artifact: `automatosx/tmp/gemma3-12b-prefill-profile-hot-pp512-f16in-pair-2026-03-26.json`
- Conclusion: the remaining dense-model prefill gap is not hiding behind `f16in + pair`; current README rows should keep the existing dense-model `f16_io=off` fast path until a different kernel family shows a clear win.

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
