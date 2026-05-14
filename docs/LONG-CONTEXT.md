# Long-Context Behavior

This page explains what AX Engine can and cannot claim about long-context
inference today. It separates four surfaces that are often collapsed into one
phrase:

- cold long-prompt prefill
- long-running chat or agent sessions with repeated prefixes
- server-path concurrent long-prefill behavior
- experimental compressed-KV work

AX Engine's strongest current long-context evidence is repeated-prefix reuse in
long-running sessions. Cold 8k prefill and multi-request serving are measured
separately and should not be presented as solved by the same evidence.

## What AX Owns

AX Engine does not replace MLX tensor kernels. The repo-owned MLX runtime uses
MLX for matrix multiply, quantized matmul, attention, RMSNorm, RoPE, and graph
execution. AX owns the runtime behavior around that graph:

- chunked prefill to keep long prompt execution inside stable Metal command
  windows
- chunked KV growth through `slice_update`, avoiding per-token full-array
  concatenation
- decode-step KV materialization so MLX lazy graphs do not grow without bound
  across long sequences
- logical prefix reuse in `ax-engine-core`
- physical MLX KV snapshot restore in `ax-engine-mlx`
- route telemetry that distinguishes physical snapshot hits from warmup
  recompute and unsupported paths

The detailed implementation contract lives in [KV-CACHE.md](KV-CACHE.md).

## Current Evidence Snapshot

| Surface | Evidence | Current result | What it supports |
|---|---|---|---|
| Cold prefill scaling | [Qwen3-4B P1 prefill scaling](../benchmarks/results/mlx-inference/2026-05-07-real-p1/qwen3-4b-4bit-prefill-scaling/prefill-scaling.md) | AX/MLX prefill ratio was 1.159x at 1k, 0.982x at 2k, 0.913x at 4k, and 0.840x at 8k | AX has measured long-context behavior, but cold 8k prefill is still an optimization surface |
| Server startup and concurrency | [Qwen3-4B P2 startup/concurrency](../benchmarks/results/mlx-inference/2026-05-07-real-p2/qwen3-4b-4bit-p2-latency/p2-latency.md) | 8k benchmark-warm TTFT was 2509.7 ms; 4-request concurrent prefill was classified as serialized | This sets serving expectations; it does not prove continuous batching |
| Hot-prefix correctness | [Qwen3.5 warm-repeat equivalence](../benchmarks/results/mlx-inference/2026-05-13-hot-prefix-w2/equivalence-gate/warm_repeat/qwen3-5-9b-2026-05-13.json) | 5/5 prompts matched token-exactly, with 5 physical snapshot hits, 176 reused tokens, and 0 warmup tokens on the claimed hit path | AX can restore physical MLX prefix snapshots on the validated Qwen warm-repeat path |
| Multi-turn long session | `benchmarks/results/kv-long-context/*fix-final-2026-05-14.json` | Qwen3.5, Qwen3.6, and Gemma4 E2B show repeated physical prefix hits and reduced post-first-turn TTFT; GLM-4.7 still shows no prefix hits in this artifact family | Long-running session reuse is promising on supported cache layouts, but not uniform across every architecture |
| Compressed KV | TurboQuant quality and microbench artifacts | Experimental and off by default | Not a production long-context support claim yet |

## Multi-Turn Evidence

The 2026-05-14 `ax.kv_multiturn_chat_evidence.v1` artifacts use a 2048-token
initial prompt, 10 turns, 64 generated tokens per turn, and 128-token user
deltas. They measure whether later turns can avoid paying the full accumulated
context cost.

| Model artifact | First-turn TTFT | Last-turn TTFT | Prefix snapshot hits | Reused tokens | Interpretation |
|---|---:|---:|---:|---:|---|
| `qwen3.5-9b-4bit-multiturn-fix-final-2026-05-14.json` | 0.690 s | 0.077 s | 10 | 18,496 | Physical prefix reuse captures the long-session win |
| `qwen3.6-35b-a3b-4bit-multiturn-fix-final-2026-05-14.json` | 0.633 s | 0.109 s | 10 | 18,496 | Physical prefix reuse captures the long-session win |
| `gemma4-e2b-4bit-multiturn-fix-final-2026-05-14.json` | 0.235 s | 0.037 s | 10 | 18,784 | Physical prefix reuse captures the long-session win |
| `glm47-flash-4bit-multiturn-fix-final-2026-05-14.json` | 0.791 s | 1.716 s | 0 | 0 | MLA multi-turn prefix snapshot reuse is still a follow-up surface |

These rows are intentionally not mixed into the README short/mid-prompt
throughput table. They answer a different question: whether a long-running
session can reuse already-computed context instead of repeatedly pre-filling the
entire conversation.

## Claim Boundaries

Strong current claim:

> AX Engine has a repo-owned KV and prefix-reuse layer that can restore physical
> MLX prefix snapshots for validated cache layouts, giving long-running
> chat/agent sessions a path to avoid repeated full-context prefill.

Measured but bounded claim:

> AX Engine has checked-in 8k cold-prefill and server-path concurrency evidence,
> and those artifacts currently show cold 8k prefill and multi-request
> continuous batching as active optimization surfaces.

Do not claim yet:

- AX is universally faster than `mlx_lm` at 8k or longer cold-prefill contexts.
- AX has production continuous batching for concurrent long prompts.
- TurboQuant compressed KV is a production long-context path.
- Prefix reuse works equally across Qwen, Gemma, GLM, and future architectures
  without architecture-specific validation.

## Benchmarking Best Practices

Long-context benchmarking should use separate artifacts for separate questions.
The best current split is:

| Question | Best artifact | Engines | Notes |
|---|---|---|---|
| Cold long-prompt prefill and derived TTFT | `ax.long_context_comparison.v1` | AX, `mlx_lm`, optional `llama.cpp Metal` | AX and `mlx_lm` must share prompt-token hashes. `llama.cpp` is shape-compatible external GGUF evidence only. |
| Decode cost at existing depth | `ax.long_context_decode_at_depth.v1` | AX, `mlx_lm`, optional `llama.cpp Metal` | Needed because decode can slow as each token reads more KV. `llama.cpp` rows are admitted only with explicit `llama-bench n_depth` evidence. |
| Long-running session reuse | `ax.kv_multiturn_chat_evidence.v1` | AX first, then server references | Measures whether repeated conversation turns avoid full accumulated-context prefill. |
| Online long-prompt serving | `ax.mlx_concurrent_prefill.v1` and `ax.serving_benchmark.v1` | AX server, `llama-server`, optional vLLM reference | Must report TTFT/TPOT/E2E percentiles, queue delay, cache hits, failures, and concurrency/request-rate policy. |

The first implemented cross-engine gate is the cold long-prefill comparison
artifact. Generate an inference-stack run with long prompt lengths and the
optional llama.cpp Metal row:

```text
python3 scripts/bench_mlx_inference_stack.py \
  --model-dir /path/to/mlx-model \
  --prompt-tokens 1024,2048,4096,8192,16384 \
  --generation-tokens 1 \
  --repetitions 3 \
  --llama-cpp-bench /path/to/llama-bench \
  --llama-cpp-gguf /path/to/model.gguf \
  --llama-cpp-decode-at-depth \
  --llama-cpp-extra-args "-fa 1" \
  --output benchmarks/results/mlx-inference/<date>/<model>-long-context-source.json
```

Then build, validate, and render the comparison artifact:

```text
python3 scripts/build_long_context_comparison_artifact.py \
  benchmarks/results/mlx-inference/<date>/<model>-long-context-source.json \
  --output benchmarks/results/mlx-inference/<date>/<model>-long-context-comparison.json \
  --require-llama-cpp

python3 scripts/check_long_context_comparison_artifact.py \
  --require-llama-cpp \
  benchmarks/results/mlx-inference/<date>/<model>-long-context-comparison.json

python3 scripts/render_long_context_comparison_report.py \
  --require-llama-cpp \
  benchmarks/results/mlx-inference/<date>/<model>-long-context-comparison.json \
  --output benchmarks/results/mlx-inference/<date>/<model>-long-context-comparison.md
```

This gate deliberately does not prove prefix-cache reuse, decode-at-depth, or
continuous batching. It answers only: for the same long prompt shape, how do
AX and `mlx_lm` compare on prompt-hash-parity cold prefill, and where does the
shape-compatible `llama.cpp Metal` GGUF baseline sit?

For decode-at-depth, build a separate artifact from the completed long-context
source or comparison artifact:

```text
python3 scripts/build_long_context_decode_at_depth_artifact.py \
  benchmarks/results/mlx-inference/<date>/<model>-long-context-source.json \
  --output benchmarks/results/mlx-inference/<date>/<model>-decode-at-depth.json

python3 scripts/check_long_context_decode_at_depth_artifact.py \
  benchmarks/results/mlx-inference/<date>/<model>-decode-at-depth.json

python3 scripts/render_long_context_decode_at_depth_report.py \
  benchmarks/results/mlx-inference/<date>/<model>-decode-at-depth.json \
  --output benchmarks/results/mlx-inference/<date>/<model>-decode-at-depth.md
```

Use `--require-llama-cpp` only when the source has `llama.cpp Metal` rows with
explicit `llama-bench n_depth` evidence. The current shape-compatible
`llama.cpp` `pp`/`tg` rows are intentionally not enough for that claim. Add
`--llama-cpp-decode-at-depth` to the inference-stack run to capture
`-p 0 -n <generation> -d <prompt>` depth rows.

## How To Read Telemetry

Route metadata makes the long-context path visible:

| Field | Meaning |
|---|---|
| `ax_mlx_prefix_cache_hits` | Physical MLX prefix snapshot restored |
| `ax_mlx_prefix_cache_misses` | Snapshot path was eligible but no snapshot was present |
| `ax_mlx_prefix_cache_reused_tokens` | Tokens restored from physical snapshot state |
| `ax_mlx_prefix_cache_warmup_tokens` | Prefix tokens re-prefilled because physical state was not restored |
| `ax_mlx_prefix_cache_blocked_*` | Snapshot restore or store was blocked by policy, layout, or trim constraints |
| `prefix_reused_tokens` | Logical scheduler-level prefix reuse, not necessarily a physical GPU-state shortcut |

The key distinction is logical versus physical reuse. A logical prefix hit
proves the scheduler recognized shared tokens. A physical hit proves the MLX
runner restored GPU KV state and skipped the corresponding prefill work.

## Next Evidence Needed

Before making broader public claims, the project should add:

- a multi-model cold-prefill scaling campaign at 1k/2k/4k/8k/16k where the host
  can support it, rendered through `ax.long_context_comparison.v1`
- real multi-model decode-at-depth artifacts with `--llama-cpp-decode-at-depth`
  enabled so `--require-llama-cpp` can pass for representative reports
- long-session latency artifacts for GLM MLA after its physical snapshot path
  is fixed or deliberately documented as recompute-only
- server-path 2/4/8-request long-prefill artifacts that show whether overlap is
  real, partial, or serialized
- a compressed-KV promotion artifact that measures end-to-end long-context
  quality and decode throughput, not only kernel correctness
