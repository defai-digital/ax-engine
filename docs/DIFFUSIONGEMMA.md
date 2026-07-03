# DiffusionGemma Experimental Support

DiffusionGemma support is experimental. AX Engine has a native MLX graph path
for `mlx-community/diffusiongemma-26B-A4B-it-4bit`, but this model family uses
block diffusion rather than ordinary autoregressive next-token decoding. Keep
its measurements and claims separate from the README direct-decode tables until
the path has stronger correctness, latency, and serving evidence.

DiffusionGemma is a block-diffusion Gemma4 26B checkpoint. The first visible
output comes from a committed 256-token diffusion block, not from a single
next-token step.

Because of that generation shape, the rows below intentionally do not use the
plain `decode tok/s` or `TTFT` labels used for autoregressive models. In Qwen,
Gemma 4 text, and other next-token decoders, `TTFT` means prompt prefill plus
the first single-token decode step, and `decode tok/s` means the steady
token-by-token autoregressive loop. DiffusionGemma instead runs a bidirectional
denoise pass over a 256-token canvas, then performs a causal commit for that
block. The comparable boundary inside this runtime is therefore time to first
block and first-block decode. Treating these as ordinary TTFT/decode rows would
make the result look directly comparable to autoregressive throughput even
though the work per visible output boundary is different.

The charts keep the same 128 / 512 / 2,048 prompt-token layout as the
autoregressive sections for readability, but the values are AX first-block
telemetry. Peer bars are intentionally omitted rather than shown as zero:
current llama.cpp Metal cannot load the GGUF
(`unknown model architecture: 'diffusion-gemma'`), and `mlx_lm` 0.31.3 cannot
load the MLX snapshot (`Model type diffusion_gemma not supported.`).

> **Prompt realism matters for diffusion.** First-block decode is
> convergence-gated: the denoiser iterates until the canvas stabilises, so its
> throughput depends on the prompt being real, in-distribution text. These rows
> use prefixes of a coherent technical document tokenized with the model's own
> tokenizer (`DIFFUSION_PROMPT_TEXT` in
> [`scripts/bench_diffusion_gemma_direct.py`](../scripts/bench_diffusion_gemma_direct.py)),
> which converge in 12-16 denoise steps. Earlier revisions of this benchmark
> fed synthetic random token ids; those never converge, hit the denoise-step
> cap, and measured the failure mode (~25-35 tok/s at 41-48 steps) rather than
> realistic throughput. Decode throughput is input-dependent, so it does not
> scale cleanly with prompt length the way prefill does.

<table>
<tr>
<td>
<img width="100%"
  src="assets/perf-diffusiongemma-direct-decode-tok-s.svg"
  alt="AX direct DiffusionGemma first-block decode throughput">
</td>
<td>
<img width="100%"
  src="assets/perf-diffusiongemma-direct-prefill-tok-s.svg"
  alt="AX direct DiffusionGemma prefill throughput">
</td>
<td>
<img width="100%"
  src="assets/perf-diffusiongemma-direct-ttft-ms.svg"
  alt="AX direct DiffusionGemma time to first committed block">
</td>
</tr>
</table>

| Prompt tokens | AX first-block decode | Denoise steps | Committed block |
| ---: | ---: | ---: | ---: |
| 128 | 115.4 tok/s | 13 | 256 tokens |
| 512 | 92.1 tok/s | 16 | 256 tokens |
| 2048 | 118.1 tok/s | 12 | 256 tokens |

## Prefill And First-Block Latency

| Prompt tokens | AX direct prefill | AX time to first block | llama.cpp Metal 9650 | `mlx_lm` 0.31.3 |
| ---: | ---: | ---: | --- | --- |
| 128 | 1,073.2 tok/s | 2,337 ms | load blocked | load blocked |
| 512 | 2,743.2 tok/s | 2,964 ms | load blocked | load blocked |
| 2048 | 3,959.0 tok/s | 2,690 ms | load blocked | load blocked |

`time to first block` is prefill wall time plus the first 256-token
denoise-and-commit block. `first-block decode` is computed as
`256 / ax_mlx_diffusion_block_wall_us`. Use these rows to track AX's
DiffusionGemma path; do not compare them directly with ordinary autoregressive
TTFT or fixed-token decode throughput.

| Runtime path | Model artifact | Benchmark status |
| --- | --- | --- |
| AX direct MLX | `mlx-community/diffusiongemma-26B-A4B-it-4bit` | Measured: 2 warmup + 5 measured repetitions, 15 s cooldown, medians reported |
| llama.cpp Metal 9650 | 4-bit GGUF | Blocked at load: `unknown model architecture: 'diffusion-gemma'` |
| `mlx_lm` 0.31.3 | 4-bit MLX snapshot | Blocked at load: `Model type diffusion_gemma not supported.` |

## Estimated Weight-Bandwidth Diagnostic

This is a secondary diagnostic, not a headline performance metric and not a
measured GPU utilization counter. It estimates first-block weight traffic at
block granularity from the measured denoise-step count plus one causal commit
over the 16.54 GB MLX safetensors artifact. This run used 13 / 16 / 12 denoise
steps at 128 / 512 / 2,048 prompt tokens on realistic prompts. The chart shows
estimated weight bandwidth versus the M5 Max theoretical ceiling; the table
keeps the effective GB/s values.

<img src="assets/perf-diffusiongemma-direct-memory-bandwidth-share.svg"
  alt="DiffusionGemma estimated weight bandwidth vs theoretical peak">

| Prompt tokens | Estimated effective bandwidth | % of 614.4 GB/s M5 Max theoretical bandwidth |
| ---: | ---: | ---: |
| 128 | 104.4 GB/s | 17.0% |
| 512 | 101.2 GB/s | 16.5% |
| 2,048 | 99.2 GB/s | 16.1% |

At these prompt lengths, the first-block path reaches roughly 16% of
theoretical M5 Max bandwidth under this estimate. That should be read as "not
raw-memory-bandwidth saturated," not as "the GPU is only 16% utilized."
Per-step cost is broadly distributed across attention, the MoE and dense FFN
blocks, the router, and the vocab-projection LM head, so this path is dispatch-,
occupancy-, and kernel-mix-bound rather than saturated on a single
weight-streaming bottleneck. Further per-step gains require kernel-level
MoE/attention work; the practical wins are realistic prompts that converge
quickly, stopping exactly at convergence, and the block-diffusion structure
itself.

## Denoise Loop

[`crates/ax-engine-mlx/src/diffusion.rs`](../crates/ax-engine-mlx/src/diffusion.rs)
keeps denoise state, entropy-bound acceptance, and self-conditioning on the
GPU. Convergence is checked every step (`convergence_check_interval`, default
1): the per-step scalar sync is negligible, but a coarser grid overshoots the
true convergence step to the next multiple, wasting a full denoise pass.
Checking every step stops exactly at convergence. The CPU no longer round-trips
256 token positions on every denoise step; sampling and acceptance stay in lazy
MLX graph nodes that fuse with the forward evaluation.

The denoise loop can stop early when any configured convergence signal fires:

1. **Strict stability:** argmax is unchanged for `convergence_steps`
   consecutive checks and mean entropy is below `entropy_threshold` (default
   0.005).
2. **Low update rate:** the accepted-position update rate drops below
   `acceptance_rate_threshold` (default 1%), so another denoise pass is unlikely
   to change the block materially.
3. **Entropy plateau:** mean entropy stops decreasing materially after the early
   denoise phase, indicating diminishing returns from additional passes.

The benchmark rows above report the measured adaptive-convergence run as
recorded in the artifact. On realistic prompts the denoiser converges in
13 / 16 / 12 denoise steps at 128 / 512 / 2,048 prompt tokens, far short of the
48-step cap.

## Current Optimizations

The default decode path is the full-pipeline compiled closure: it compiles the
entire denoise step (forward + softmax + entropy + argmax + sampling +
acceptance) into one `MlxClosure`, collapsing ~280 per-step MLX C-API calls
into a single dispatched graph. Self-conditioning feedback is generated only
when the checkpoint actually carries `diffusion_self_conditioning` weights;
otherwise AX skips the cached embedding table and feedback path. Because the
full-pipeline path is a single pure compiled graph, it bypasses the per-layer
embedding cache and KV concat buffer; those apply only to the non-compiled
imperative fallback.

| Optimization | Default | Status | Toggle |
| --- | --- | --- | --- |
| Full pipeline (fused denoise step) | ON | Active in the default path; self-conditioning is skipped when checkpoint weights are absent | opt-out `AX_DIFFUSION_NO_FULL_PIPELINE=1` |
| Commit skip on converge | ON | Active; skips the causal commit when the block converges with >= 99% acceptance | opt-out `AX_DIFFUSION_NO_SKIP_COMMIT=1` |
| Compiled forward (forward-only) | ON when full pipeline is off | Fallback only; superseded by the full pipeline | opt-out `AX_DIFFUSION_NO_COMPILED_FORWARD=1` |
| Embedding cache | OFF | Opt-in; output-neutral but reachable only on the imperative fallback | opt-in `AX_DIFFUSION_EMBEDDING_CACHE=1` |
| KV concat buffer | OFF | Opt-in; known-divergent | opt-in `AX_DIFFUSION_KV_CONCAT_BUFFER=1` |

> **KV concat buffer is gated off by default.** Its `slice_update` reuse path is
> not bit-equivalent to the canonical `concatenate` path: on a 512-token block
> it diverges in ~237/256 committed tokens, which perturbs convergence
> (15 vs 17 denoise steps) and can introduce decode artifacts. It yields no
> throughput gain in any bit-exact configuration, so the default path and the
> imperative fallback use the canonical concatenate path.

## Benchmark Contract

The published rows use first-block telemetry instead of the standard fixed-token
autoregressive benchmark contract. `max_output_tokens=1` is enough to force
prefill plus one diffusion block, and the block counters still report the full
256-token denoise/commit cycle even though the caller receives only the first
emitted token.

Telemetry: SSE-emitted `ax_mlx_diffusion_*` counters cover block count, denoise
steps, convergence count, per-criterion convergence signals, near-miss
entropy/update-rate diagnostics, denoise wall time, commit wall time, and block
wall time, plus `diffusion` decode-route classification in
`bench_mlx_inference_stack.py`.

Artifacts: AX direct rows are
[`2026-07-03-readme-first-block/summary.json`](../benchmarks/results/diffusion-gemma-direct/2026-07-03-readme-first-block/summary.json),
with the human summary in
[`summary.md`](../benchmarks/results/diffusion-gemma-direct/2026-07-03-readme-first-block/summary.md).
Peer runtime blockers are recorded as load failures, so there are no llama.cpp
or `mlx_lm` result artifacts for this model family.

Run the full direct benchmark and regenerate the charts:

```bash
cargo build -p ax-engine-bench --bin ax-engine-bench
python3 scripts/bench_diffusion_gemma_direct.py
```

Render charts from an existing artifact:

```bash
python3 scripts/bench_diffusion_gemma_direct.py --skip-benchmark
```

## Decode Acceleration Model

DiffusionGemma's acceleration model is the diffusion block itself. It does not
stack with MTP or n-gram acceleration because those techniques assume an
autoregressive next-token loop.

| | MTP (speculative decoding) | DiffusionGemma (block diffusion) |
| --- | --- | --- |
| Generation | Draft-then-verify, one token at a time | 256-token blocks via bidirectional denoising |
| Forward pass | Causal only | Bidirectional (denoise) + causal (commit) |
| Needs draft model / assistant head | Yes | No |
| AX Engine decode path | `ngram_acceleration` / `mtp_head_only` | `diffusion` (early return, mutually exclusive) |

In the runner's `decode_one`, the diffusion path returns before the MTP/n-gram
branches are reached. `DiffusionConfig` carries canvas size, denoise steps,
entropy thresholds, convergence settings, and temperature schedule only; it has
no MTP fields.

## Supported Experimental Features

- Block-autoregressive discrete diffusion decode (canvas=256, up to 48 denoise
  steps)
- Entropy-bound position acceptance with argmax-based rejection
- Self-conditioning via GPU matmul (`prob x cached embedding table`) when
  checkpoint weights are present
- Linear temperature schedule
- Adaptive convergence detection
- Standard causal prefill
- Causal commit pass
- SSE telemetry counters for diffusion block timing, denoise steps, convergence
  signals, and near-miss entropy/update-rate diagnostics (`ax_mlx_diffusion_*`)
- `diffusion` decode-route classification in the benchmark harness

Not applicable:

- MTP / assistant-head speculative decoding
- N-gram acceleration
- Direct pipeline double-buffering
