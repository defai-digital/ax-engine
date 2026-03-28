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
- no parallel `ax-engine-bench`, `llama-bench`, `ollama`, or model-serving job
- enough free memory to avoid paging pressure

Normal desktop processes such as `WindowServer`, editors, or browsers are usually acceptable if the machine is otherwise mostly idle, but fewer background apps is better.

## Build

From the repo root:

```bash
cargo build --workspace --release
```

AX benchmark binary:

```text
./target/release/ax-engine-bench
```

## AX Engine Benchmarks

### Throughput

Basic run:

```bash
./target/release/ax-engine-bench bench --model ./models/<model>.gguf
```

Machine-readable artifact:

```bash
./target/release/ax-engine-bench bench --model ./models/<model>.gguf --json
./target/release/ax-engine-bench bench --model ./models/<model>.gguf --json-output /tmp/ax-engine-bench.json
```

Recommended repeated run for publishable numbers:

```bash
./target/release/ax-engine-bench bench \
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
./target/release/ax-engine-bench bench \
  --model ./models/<model>.gguf \
  --intent latency
```

### Decode Profile

Use this to inspect the decode hot path and identify where time is actually spent:

```bash
./target/release/ax-engine-bench profile --model ./models/<model>.gguf
```

### Important AX Notes

- Throughput mode usually selects `pipelined` decode.
- Latency mode usually selects `single_cb` decode.
- The selected prefill and decode plans are printed in the output. Keep them with the benchmark record.
- `--llama-parity-preset` is not a cross-engine truth mode. It only applies a small set of AX env defaults and can change performance significantly.
- If AX prints `PrefillPlan: mode=serial reason=unsupported_quant:...`, treat that prefill result as a fallback-path measurement, not a normal GPU-prefill comparison.
- Mixed-quant GGUFs with active `Q5_K` layer tensors now use AX's GPU prefill route by default.
- For `Q5_K` prefill runs, also record whether AX auto-used:
  - `q5k_prefill=base`
  - `q5k_prefill=small_n`
- AX currently auto-selects `small_n` only when the mapped model is
  predominantly `Q5_K` and the prompt batch is in the small-batch window (`4..8` tokens).
- `ax-engine-bench bench`, `ax-engine-bench profile`, and `ax-engine-bench soak` JSON now emit this as a first-class field:
  - `q5k_prefill_mode`
- `ax-engine-bench soak` summary now also prints:
  - `PrefillPlan: ...`
  - `Q5KPrefill: ...` when present
- `ax-engine-bench speculative` summary now also prints:
  - `PrefillPlan: ...`
  - `Q5KPrefill: ...` when present
- `ax-engine-bench speculative` also supports machine-readable artifacts now:
  - `--json`
  - `--json-output <path>`
- If you are doing route A/B validation, record any forced override explicitly:
  - `AX_METAL_Q5K_PREFILL_VARIANT=base`
  - `AX_METAL_Q5K_PREFILL_VARIANT=small`

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
./target/release/ax-engine-bench bench \
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
| Llama 3 8B Q4_K_M | 675.8 | 639.2 | 105.7% | 61.2 | 47.1 | 129.9% | `pipelined`, `attn=mistral_bc64/experimental` |
| Llama 3 70B Q4_K_M | 55.7 | 66.8 | 83.4% | 6.5 | 6.3 | 103.2% | `pipelined`, `PrefillPlan=mode=gpu_batch`, `q5k_prefill=base` |
| Qwen3 8B Q4_K_M | 667.3 | 736.7 | 90.6% | 57.9 | 60.3 | 96.0% | `pipelined`, `attn=mistral_bc64/experimental`, `decode=f16kv_hd128_n2/profile_preferred` |
| Qwen3 14B Q4_K_M | 357.2 | 408.2 | 87.5% | 35.3 | 35.6 | 99.1% | `pipelined`, `attn=mistral_hd128/profile_preferred` |
| Qwen3 32B Q4_K_M | 126.0 | 150.7 | 83.6% | 16.6 | 14.9 | 111.4% | `pipelined`, `attn=mistral_bc64/experimental` |
| Qwen3.5 9B Q4_K_M | 122.6 | 732.2 | 16.7% | 25.0 | 48.9 | 51.1% | `sequential`, hybrid attention+SSM, GPU-unified prefill |

`llama.cpp` values above are medians from `samples_ts` on the current local Homebrew `llama-bench` build (`build_commit 342d6125b`, `build_number 8500`) with `-fa 1`, `-ctk f16`, and `-ctv f16`. `AX vs llama.cpp` over `100%` means AX was faster.

70B note:

- `models/meta-llama-3-70b-instruct.Q4_K_M.gguf` contains an active `Q5_K` tensor.
- Shipped AX now routes that mixed-quant case through the conservative `Q5_K` GPU prefill path by default.
- The published 70B row above is therefore representative of current shipped behavior, not an opt-in validation path.

`Q5_K` prefill auto-routing snapshot:

- `models/meta-llama-3.1-8b-instruct-q5_k_m.gguf`
  - command shape:
    - `target/release/ax-engine-bench bench --model ... --prompt-tokens 16 --decode-tokens 32 --warmup-iters 1 --measure-iters 1`
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
- on the predominant-`Q5_K` 8B file, the shipped base route now uses
  `f16` input staging outside the `4..8` small-window because that improved
  real-model prompt throughput without hurting decode
- benchmark records for `Q5_K` prefill should include the exact
  `q5k_prefill=...` label

March 26 `Q5_K` large-batch production pass:

- Exact-shape GPU microbench on the real `Q5_K` hot shapes showed the missing
  large-batch `f16in` route was worthwhile, while the small-window still
  preferred `small_n`.
  - artifact: `automatosx/tmp/q5k-gpu-microbench-f16in-2026-03-26.json`
  - `llama3_8b_q5k attn_qkv tokens512`: `4.45ms -> 3.70ms` (`1.20x`)
  - `llama3_8b_q5k ffn_up tokens512`: `14.13ms -> 11.83ms` (`1.19x`)
  - `llama3_8b_q5k attn_qkv_small_window tokens8`: `small_n` remained best
- Shipped planner change: AX now keeps `q5k_prefill=small_n` with `f16_io=off`
  for the `4..8` window, and uses `q5k_prefill=base` with `f16_io=on`
  outside that window.
- End-to-end `Meta-Llama-3-8B-Instruct-Q5_K_M.gguf` throughput improved from
  the earlier cooled `317.2 tok/s` prefill / `32.3 tok/s` decode baseline to
  `341.6 tok/s` prefill / `32.5 tok/s` decode on the new shipped default.
  - artifact: `automatosx/tmp/llama3-8b-q5k-bench-default-f16in-promoted-2026-03-26.json`
- Shipped kernel change: AX now uses the blocked `Q5_K` batch kernel by
  default for the large-`N` base route. Same-build A/B on
  `Meta-Llama-3-8B-Instruct-Q5_K_M.gguf` improved from `321.9 tok/s` prefill /
  `30.4 tok/s` decode with `AX_METAL_BATCH_Q5K_BLOCKED=0` to
  `343.1 tok/s` prefill / `32.8 tok/s` decode with the blocked route enabled.
  - artifacts:
    - `automatosx/tmp/llama3-8b-q5k-bench-post-blocked-kernel-disabled-2026-03-26.json`
    - `automatosx/tmp/llama3-8b-q5k-bench-post-blocked-kernel-2026-03-26.json`
- A short prompt sanity run still shows the intended small-window policy:
  `PrefillPlan: ... q5k_prefill=small_n f16_io=off`.
  - artifact: `automatosx/tmp/llama3-8b-q5k-bench-small-window-sanity-2026-03-26.json`

Interpretation:

- Gemma 3 12B: after the March 26 Gemma-specific Metal fusions, AX is close on prefill and now leads on decode.
- Gemma 3 27B: after the March 26 Gemma-specific Metal fusions, AX is close on prefill and now leads on decode.
- Llama 3 8B: after the March 26 route pass, the current shipped default is still the best tested path and AX remains ahead on both prefill and decode.
- Llama 3 70B: after the March 26 route pass, the shipped mixed-quant default remains the best balanced route; forced `q5k_prefill=small_n` improved prefill but hurt decode.
- Qwen3 8B: after a March 26 route pass, AX improved modestly but current local `llama.cpp` still leads on both prefill and decode.
- Qwen3 14B: after a March 26 route pass, AX is much closer on prefill and decode is effectively at parity; current local `llama.cpp` still holds a prefill lead.
- Qwen3 32B: after the March 26 exact-shape route pass, AX is still below current local `llama.cpp` on prefill but now leads on decode.
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

March 26 Qwen3 32B exact-shape route pass:

- Fresh current-default `Qwen3 32B` throughput landed at `126.0 tok/s` prefill and `16.6 tok/s` decode.
  - artifact: `automatosx/tmp/qwen3-32b-bench-post-qwen3-f16-path-2026-03-26.json`
- Forcing `fa2_hd128` moved prefill attention to `fa2_simd_hd128/experimental` and regressed to `116.4 tok/s` prefill and `10.1 tok/s` decode.
  - artifact: `automatosx/tmp/qwen3-32b-bench-fa2hd128-2026-03-26.json`
- Forcing decode `hd128_n2` regressed to `108.9 tok/s` prefill and `9.7 tok/s` decode.
  - artifact: `automatosx/tmp/qwen3-32b-bench-hd128n2-2026-03-26.json`
- Exact-shape GPU microbench on the real 32B hot shapes kept `q4_k/q6_k -> NR2` for decode, but still showed the dense prefill `f32` batch route beating `f16in` on `attn_qkv_fused`, `attn_wo`, and `ffn_down`.
  - artifact: `automatosx/tmp/qwen3-32b-gpu-microbench-2026-03-26.json`
- The `f16_io` path remains non-shippable for Qwen3 32B in this pass.
  - `AX_METAL_BATCH_F16_IO_QWEN3=1 AX_METAL_BATCH_F16_PAIR_QWEN3=1` fell to `64.2 tok/s` prefill and decode collapsed to zero generated throughput.
    - artifact: `automatosx/tmp/qwen3-32b-bench-f16io-pair-2026-03-26.json`
  - `AX_METAL_BATCH_F16_IO_QWEN3=1 AX_METAL_BATCH_F16_PAIR_QWEN3=0` still collapsed decode to zero generated throughput.
    - artifact: `automatosx/tmp/qwen3-32b-bench-f16io-no-pair-2026-03-26.json`
- Conclusion: keep the shipped `perfs/qwen3-32b.json` route as-is for now. The remaining 32B gap is not a missing profile flip; it is still inside the current dense prefill GPU execution path.
- Conclusion: the remaining dense-model prefill gap is not hiding behind `f16in + pair`; current README rows should keep the existing dense-model `f16_io=off` fast path until a different kernel family shows a clear win.

March 26 Qwen3 8B / 14B route pass:

- Fresh current-default `Qwen3 8B` throughput before any profile change landed at `654.5 tok/s` prefill and `56.1 tok/s` decode.
  - artifact: `automatosx/tmp/qwen3-8b-bench-refresh-2026-03-26.json`
- Forcing `fa2_hd128` on `Qwen3 8B` moved prefill attention to `fa2_simd_hd128/experimental` and landed at `655.5 tok/s` prefill and `57.6 tok/s` decode.
  - artifact: `automatosx/tmp/qwen3-8b-bench-fa2hd128-2026-03-26.json`
- Forcing decode `hd128_n2` on `Qwen3 8B` landed at `665.4 tok/s` prefill and `57.5 tok/s` decode.
  - artifact: `automatosx/tmp/qwen3-8b-bench-hd128n2-2026-03-26.json`
- Combining `fa2_hd128 + hd128_n2` on `Qwen3 8B` regressed prefill to `646.9 tok/s`, so it was not kept.
  - artifact: `automatosx/tmp/qwen3-8b-bench-fa2hd128-hd128n2-2026-03-26.json`
- Shipped change: AX now loads `perfs/qwen3-8b.json`, which keeps the existing `mistral_bc64` prefill route and sets `hd128_n2_default=true` for decode. The clean no-env post-change bench landed at `667.3 tok/s` prefill and `57.9 tok/s` decode.
  - artifact: `automatosx/tmp/qwen3-8b-bench-post-profile-2026-03-26.json`
- Follow-up March 26 matmul-parity pass: vectorizing the blocked Q4_K/Q6_K B-load path moved `Qwen3 8B` further to `725.1 tok/s` prefill while decode stayed flat at `57.8 tok/s`.
  - artifact: `automatosx/tmp/qwen3-8b-bench-post-blocked-bload-vectorize-2026-03-26.json`

- Fresh current-default `Qwen3 14B` throughput landed at `357.2 tok/s` prefill and `35.3 tok/s` decode.
  - artifact: `automatosx/tmp/qwen3-14b-bench-refresh-2026-03-26.json`
- Forcing `fa2_hd128` on `Qwen3 14B` regressed prefill to `346.8 tok/s`, even though decode nudged up slightly.
  - artifact: `automatosx/tmp/qwen3-14b-bench-fa2hd128-2026-03-26.json`
- Forcing `mistral_bc64` on `Qwen3 14B` regressed heavily to `276.2 tok/s` prefill and `34.9 tok/s` decode.
  - artifact: `automatosx/tmp/qwen3-14b-bench-bc64-2026-03-26.json`
- Forcing decode `hd128_n2` on `Qwen3 14B` regressed to `336.8 tok/s` prefill and `34.0 tok/s` decode.
  - artifact: `automatosx/tmp/qwen3-14b-bench-hd128n2-2026-03-26.json`
- Conclusion: keep the shipped `perfs/qwen3-14b.json` route as-is. The old README row was stale, but the current 14B default is already the best of the tested routes in this pass.

March 26 Llama 3 8B / 70B route pass:

- Fresh current-default `Llama 3 8B` throughput landed at `675.8 tok/s` prefill and `61.2 tok/s` decode.
  - artifact: `automatosx/tmp/llama3-8b-bench-refresh-2026-03-26.json`
- Forcing decode `hd128_n2` on `Llama 3 8B` regressed badly to `583.1 tok/s` prefill and `43.7 tok/s` decode.
  - artifact: `automatosx/tmp/llama3-8b-bench-hd128n2-2026-03-26.json`
- Forcing `fa2_hd128` on `Llama 3 8B` regressed to `638.1 tok/s` prefill and `50.9 tok/s` decode.
  - artifact: `automatosx/tmp/llama3-8b-bench-fa2hd128-2026-03-26.json`
- Conclusion: keep `Llama 3 8B` on the current default route. Neither tested override beat the shipped path.

- Fresh current-default `Llama 3 70B` throughput landed at `55.7 tok/s` prefill and `6.5 tok/s` decode.
  - artifact: `automatosx/tmp/llama3-70b-bench-refresh-2026-03-26.json`
- Forcing `AX_METAL_Q5K_PREFILL_VARIANT=small` improved prefill to `58.8 tok/s` but reduced decode to `5.9 tok/s`.
  - artifact: `automatosx/tmp/llama3-70b-bench-q5k-small-2026-03-26.json`
- Forcing decode `hd128_n2` regressed to `54.2 tok/s` prefill and `5.8 tok/s` decode.
  - artifact: `automatosx/tmp/llama3-70b-bench-hd128n2-2026-03-26.json`
- Conclusion: keep `perfs/llama3-70b.json` unchanged for now. The forced `small_n` variant is a prompt-throughput tradeoff, not a clear shipped win, and `hd128_n2` is worse.

March 27 Qwen3.5-9B and Qwen3-8B session:

- Qwen3.5 9B is the first hybrid attention+SSM (Mamba-2) model benchmarked in this repo.
- AX and `llama.cpp` benchmarks were run concurrently in this session; absolute numbers may be slightly depressed on both sides, but the ratios are representative.

Qwen3.5 9B results (best complete run pair from multiple iterations):

| Model | AX prefill | llama.cpp prefill | AX vs llama.cpp | AX decode | llama.cpp decode | AX vs llama.cpp | AX notes |
|---|---:|---:|---:|---:|---:|---:|---|
| Qwen3.5 9B Q4_K_M | 234.4 | 722.2 | 32.5% | 12.9 | 49.0 | 26.3% | `sequential`, `PrefillPlan=mode=gpu_batch kv=qwen35_hybrid recurrent=backend_owned` |

- AX command shape: `./target/release/ax-engine-bench bench --model ./models/Qwen3.5-9B-Q4_K_M.gguf`
- `llama.cpp` command shape: `llama-bench -m ./models/Qwen3.5-9B-Q4_K_M.gguf -p 512 -n 128 -r 3`
- `llama.cpp` was run without `-fa 1 -ctk f16 -ctv f16` in this session; with flash attention enabled, `llama.cpp` numbers would likely be higher.

Key observations:

- AX decode reports `Mode: sequential` with `Fallback: qwen35 hybrid decode currently uses sequential host orchestration`.
- Decode used 20,608 Metal command buffer submissions and 1,024 barriers for 128 tokens (161 submits/token), compared to 128 submissions (1/token) for the pure-transformer Qwen3 8B pipelined path.
- GPU attention KV allocated as F16, but the recurrent (SSM) state remains F32.
- Multiple iterations during the session showed steady improvement as code was updated:
  - Run 1: AX 225.5 prefill, 7.9 decode
  - Run 2: AX 223.7 prefill, 12.2 decode
  - Run 3: AX 234.4 prefill, 12.9 decode (used for headline)

Root causes for the gap:

1. No pipelined decode for the hybrid attention+SSM architecture. The double-buffered pipeline only supports pure-transformer models currently.
2. Sequential host orchestration: each layer's recurrent state (conv1d + SSM) requires CPU round-trips between Metal command buffer submissions.
3. Prefill uses 225 command buffer submissions vs 1 for pure-transformer models, because the recurrent layers are not yet fused into the GPU batch prefill graph.

Qwen3 8B refresh (same session):

| Model | AX prefill | llama.cpp prefill | AX vs llama.cpp | AX decode | llama.cpp decode | AX vs llama.cpp | AX notes |
|---|---:|---:|---:|---:|---:|---:|---|
| Qwen3 8B Q4_K_M | 524.4 | 545.9 | 96.1% | 39.5 | 38.6 | 102.3% | `pipelined`, `attn=mistral_bc64/experimental`, `decode=f16kv_hd128_n2/profile_preferred` |

- AX command shape: `./target/release/ax-engine-bench bench --model ./models/Qwen3-8B-Q4_K_M.gguf`
- `llama.cpp` command shape: `llama-bench -m ./models/Qwen3-8B-Q4_K_M.gguf -p 512 -n 128 -r 3`
- `llama.cpp` was run without `-fa 1 -ctk f16 -ctv f16`; the March 26 controlled run with those flags showed 736.7/60.3, so the ratio here overstates AX vs a properly-configured `llama.cpp`.
- AX decode reports `Mode: pipelined` with 128 command buffer submissions (1/token) and 0 barriers. KV dtype F16 throughout.
- The parity ratio (96% prefill, 102% decode) is consistent with the March 26 controlled run (98.7% prefill, 103.5% decode), confirming that Qwen3 8B remains near parity on the pure-transformer path.

Interpretation:

- Pure-transformer models (Qwen3 8B): AX is at parity with `llama.cpp` on both prefill and decode, thanks to the pipelined decode loop, F16 KV, and fused GPU dispatch.
- Hybrid attention+SSM models (Qwen3.5 9B): AX has a significant performance gap due to the sequential host orchestration required for recurrent layers. Pipelined decode for the hybrid architecture is the highest-value optimization target.

March 28 Qwen3.5-9B controlled retest:

- The March 27 `llama.cpp` baseline was run without `-fa 1 -ctk f16 -ctv f16`, which inflated the gap. The March 28 retests use the full controlled settings on both sides.
- Three sequential runs were performed to account for thermal warm-up. The third (thermally stable) run is used as the headline.

Warmed run (headline):

| Model | AX prefill | llama.cpp prefill | AX vs llama.cpp | AX decode | llama.cpp decode | AX vs llama.cpp | AX notes |
|---|---:|---:|---:|---:|---:|---:|---|
| Qwen3.5 9B Q4_K_M | 240.7 | 735.2 | 32.7% | 25.7 | 48.7 | 52.8% | `sequential`, `PrefillPlan=mode=gpu_batch kv=qwen35_hybrid recurrent=backend_owned` |

All five runs (sequential, AX then llama.cpp each iteration):

| Run | AX prefill | AX decode | llama.cpp prefill | llama.cpp decode | Prefill % | Decode % |
|---|---:|---:|---:|---:|---:|---:|
| Run 1 (cold) | 221.3 | 20.5 | 489.9 | 34.3 | 45.2% | 59.8% |
| Run 2 | 233.9 | 23.4 | 678.2 | 45.3 | 34.5% | 51.7% |
| Run 3 | 239.6 | 26.1 | 732.4 | 48.9 | 32.7% | 53.4% |
| Run 4 | 236.9 | 22.4 | 734.0 | 48.7 | 32.3% | 46.0% |
| Run 5 (warm, headline) | 240.7 | 25.7 | 735.2 | 48.7 | 32.7% | 52.8% |

Note: flash attention makes almost no difference for Qwen3.5 on `llama.cpp` (735 vs 727 without FA). The recurrent (SSM) layers don't use attention, so FA only affects the 8 full-attention layers out of 32.

- AX command shape: `./target/release/ax-engine-bench bench --model ./models/Qwen3.5-9B-Q4_K_M.gguf --deterministic --samples 5 --measure-iters 1 --cooldown-ms 500`
- `llama.cpp` command shape: `/opt/homebrew/bin/llama-bench -m ./models/Qwen3.5-9B-Q4_K_M.gguf -p 512 -n 128 -r 5 -ngl 99 -ctk f16 -ctv f16 -fa 1`
- `llama.cpp` build: `342d6125b (8500)`
- Machine idle, no concurrent inference jobs. Runs sequential (AX first, then llama.cpp each iteration).

Key observations:

- AX decode improved from 12.9 (Mar 27) to 25.7 tok/s (+99%) thanks to GPU-unified decode work.
- Both engines show significant thermal warm-up: AX prefill 221→241, llama.cpp prefill 490→735. The cold run 1 overstated AX's relative position.
- The thermally stable ratios are ~33% prefill and ~53% decode. Confirmed across 5 sequential runs.
- AX GPU dispatch overhead remains the dominant bottleneck: 265 prefill cmd_buf, 12,544 decode cmd_buf (vs 1/128 for pure-transformer models).
- Decode barriers: 1,024 (vs 0 for pipelined pure-transformer models).

Progress summary:

| Metric | Mar 27 AX | Mar 28 AX (warm) | Mar 27 ratio | Mar 28 ratio (warm) | Notes |
|---|---:|---:|---:|---:|---|
| Prefill | 234.4 | 240.7 | 32.5% | 32.7% | AX prefill stable; Mar 27 llama.cpp was without FA |
| Decode | 12.9 | 25.7 | 26.3% | 52.8% | +99% AX improvement from GPU-unified decode |

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
./target/release/ax-engine-bench bench \
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
