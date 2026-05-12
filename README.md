# AX Engine

[![Preview Surfaces](https://github.com/defai-digital/ax-engine/actions/workflows/python-preview.yml/badge.svg?branch=main)](https://github.com/defai-digital/ax-engine/actions/workflows/python-preview.yml)
[![Coverage Report](https://github.com/defai-digital/ax-engine/actions/workflows/coverage.yml/badge.svg?branch=main)](https://github.com/defai-digital/ax-engine/actions/workflows/coverage.yml)

AX Engine is a Mac-first LLM inference runtime, local server, SDK layer, and
benchmark toolkit for Apple Silicon.

It is not "AX MLX" as a product. MLX is the primary Apple Silicon execution
backend for supported model families, while AX Engine also exposes explicit
compatibility routes for upstream `mlx-lm` and `llama.cpp` so users can stay on
one AX surface while model coverage grows.

> Requires **macOS 14 (Sonoma) or later** on **Apple Silicon M2 Max or newer** with **32 GB RAM minimum**.
> Rust 1.85+ for source builds.

### Supported Hardware

AX Engine targets high-memory Apple Silicon Macs running **macOS 14 (Sonoma) or later**.

| Machine | Minimum spec | Suggested spec |
|---|---|---|
| Mac Mini | M4 Pro, 32 GB | M4 Pro, 64 GB |
| MacBook Pro 14″ / 16″ | M2 Pro / M2 Max, 32 GB | M3 Max, 96 GB |
| Mac Studio | M2 Max / M2 Ultra, 32 GB | M4 Max, 96 GB |

M3, M4, M5 chip variants are supported across all three lines. M1 is not supported. M2 base chip (max 24 GB) is below the 32 GB minimum.

## 30-Second Setup

Install the released command-line tools and open the local web manager:

```bash
brew install defai-digital/ax-engine/ax-engine
ax-engine-manager --check
ax-engine-manager
```

Then connect it to a model and server:

```bash
# Download an mlx-community model and generate its manifest in one step
MODEL_DIR="$(python3 scripts/download_model.py mlx-community/Qwen3-4B-4bit --json | python3 -c 'import json,sys; print(json.load(sys.stdin)["dest"])')"

# Start the server
ax-engine-server --mlx --mlx-model-artifacts-dir "$MODEL_DIR" --port 8080

# In another terminal, open the web manager with live server metadata
ax-engine-manager --model-dir "$MODEL_DIR" --server-url http://127.0.0.1:8080
```

The manager opens a localhost page with model type, family, and size dropdowns,
a `[Download]` action, server port controls, and full local endpoint URLs.

Or from Python (after `maturin develop` or `pip install ax-engine`):

```python
from ax_engine import download_model, Session
path = download_model("mlx-community/Qwen3-4B-4bit")
with Session(mlx=True, mlx_model_artifacts_dir=str(path)) as s:
    print(s.generate([1, 2, 3], max_output_tokens=8).output_tokens)
```

`download_model()` downloads weights and auto-runs `ax-engine-bench generate-manifest`.
See [Getting a Model](#getting-a-model) for all paths including raw HF checkpoints,
and see [AX Engine Manager](docs/MANAGER.md) for the full web workflow.

## Why AX Engine

AX Engine gives local inference work a stable runtime contract:

- `ax-engine-server` exposes a local HTTP adapter over the runtime.
- `ax-engine-bench` records workload contracts, route identity, correctness,
  determinism, and performance evidence.
- `ax-engine-sdk`, Python bindings, and the JavaScript preview client provide
  thin integration surfaces over the same backend-resolution rules.
- Repo-owned MLX execution is optimized for supported Qwen and Gemma families.
- Delegated `mlx_lm.server` and `llama.cpp` routes cover explicit
  compatibility cases without turning delegated results into AX-owned
  throughput claims.

[mlx_lm](https://github.com/ml-explore/mlx-lm) and
[mlx-swift-lm](https://github.com/ml-explore/mlx-swift) remain the canonical
MLX references. AX Engine compares against them, learns from them, and delegates
to `mlx-lm` for unsupported MLX text models when requested. The AX-owned value
is the runtime layer around supported workloads: request lifecycle, scheduling,
KV/cache policy, n-gram acceleration, and auditable benchmark artifacts.

For supported transformer families on Apple Silicon, the AX-owned runtime layer
can produce higher effective throughput than the reference MLX runtimes on
matching benchmark shapes:

- **N-gram acceleration** reaches up to 3.4x mlx_lm decode
  throughput on high-hit benchmark rows — with no second draft model and no
  model changes
- **Coding-shaped decode is a natural fit when local repetition exists**:
  completion, edit loops, structured diffs, JSON/tool output, imports,
  indentation, and repeated identifiers often contain patterns that n-gram
  acceleration can predict and the target model can verify. Novel, high-entropy,
  or very short coding requests may see little or no gain.
- **AX-owned request lifecycle** provides deterministic, auditable scheduling,
  KV block management, and prefix reuse that upstream Python runtimes do not
  expose as stable contracts
- **workload-contract tooling** (`ax-engine-bench`) validates correctness,
  determinism, route identity, and regression across checked-in manifests, not
  just throughput snapshots

The thesis is not "our MLX tensor ops are faster." MLX compiles and executes the
same compute graph either way. The thesis is that **AX's decode strategy above
MLX** — how tokens are speculated, how requests are scheduled, how KV state is
materialized — produces measurably higher effective throughput on supported
workloads.

## Runtime Paths

| Path | Use it for | Current scope |
|---|---|---|
| Repo-owned MLX runtime | Supported Qwen/Gemma MLX model artifacts and repo-owned performance claims | Local Apple Silicon inference, token-based server/SDK requests, benchmarked direct and n-gram acceleration modes |
| `mlx_lm_delegated` | MLX text models that upstream `mlx-lm` supports before AX has a repo-owned graph | Blocking and SSE text generation through a user-provided `mlx_lm.server`; `/v1/generate`, `/v1/generate/stream`, and OpenAI-compatible completion/chat text endpoints. Streaming is delegated text compatibility evidence, not repo-owned token/KV performance |
| `llama_cpp` | GGUF and non-MLX local inference | Delegated llama.cpp server/CLI compatibility; route-contract evidence, not repo-owned MLX throughput |

The runtime report exposes `selected_backend`, `support_tier`, and
`resolution_policy` so callers and benchmark artifacts can distinguish these
paths.

For the exact OpenAI-shaped endpoint contract, including what is and is not
compatible today, see `docs/API-COMPATIBILITY.md`.

## Design

### Execution Layer

The repo-owned MLX path uses MLX directly for tensor operations via the official
`mlx-c` C API. Matrix multiply, quantized matmul, attention, RMSNorm, and RoPE
go through MLX's Apple-maintained Metal kernels. AX owns the runtime behavior
above that graph.

What AX Engine adds around model execution:

- **N-gram acceleration**: a bigram/trigram table built at runtime predicts
  up to 4 draft tokens per step. The target model verifies them in one forward
  pass over `[last_token, D1, …, D_n]`. An EMA accept-rate gate (α=0.1,
  threshold 0.5) disables acceleration after a bad sequence and re-enables when
  the table recovers. No second draft model required.
- **Scheduler and KV manager**: request lifecycle, batching, memory-blocked
  recovery, and execution planning live in `ax-engine-core` — deterministic,
  async-free, no framework dependencies. See [`docs/SCHEDULER.md`](docs/SCHEDULER.md)
  and [`docs/KV-CACHE.md`](docs/KV-CACHE.md) for design details.
- **Chunked KV cache**: keys and values grow in pre-allocated backing buffers via
  `slice_update`. Draft rollback is O(1) — only the sequence-length
  pointer moves. After each decode step, all KV buffers are evaluated with the
  output token to flatten the lazy-eval graph and prevent O(N²) graph depth.
- **Graph compilation**: `mlx_enable_compile()` is called once at startup so
  Metal shader compilation and dispatch tables are reused across steps with the
  same shape — equivalent to `mx.compile()` in mlx_lm.
- **GatedDelta linear attention**: hybrid architectures (Qwen3.5, Qwen3-Next)
  use a custom SIMD-group Metal kernel for the recurrent GatedDelta state update.
  All other ops in the same models (dense attention, FFN, projections) delegate
  to MLX's hardware-optimized paths.

### Memory Layer

`mlx_set_wired_limit(recommendedMaxWorkingSetSize)` wires model weights into GPU
memory at startup, preventing Metal from paging them between requests. A
dedicated GPU stream avoids cross-stream synchronization on the shared default
stream.

See [`docs/KV-CACHE.md`](docs/KV-CACHE.md) for a detailed description of the
two-layer KV cache architecture, prefix caching coordination, model-specific
cache variants, and memory pressure handling.

## Supported Models

| Family | Model | Architecture notes |
|---|---|---|
| Gemma 4 | gemma-4-e2b-it, gemma-4-e4b-it, gemma-4-26b-a4b-it, gemma-4-31b-it | Dense, per-layer embedding, and MoE variants; MLX affine 4/5/6/8-bit weights, sliding-window + full attention, K=V full-attention layers, logit softcapping |
| Qwen 3.5 | Qwen3.5-9B | Linear attention + MoE FFN, attn_output_gate per-head interleaving |
| Qwen 3.6 / Coder Next | Qwen3.6-35B-A3B 4/5/6/8-bit MLX, Qwen3-Coder-Next-4bit | `qwen3_next` architecture: GatedDelta linear attention (3 of every 4 layers) + full attention with per-head sigmoid gate (every 4th layer) + sparse top-k MoE with shared expert |

All models use MLX safetensors format with the AX `model-manifest.json`
descriptor. Each supported architecture has a hand-written forward pass in
`ax-engine-mlx`. Adding a new architecture means implementing the model graph,
not wiring up a generic loader.

Recent community-model checks are tracked according to the evidence they have.
On 2026-05-06, `mlx-community/GLM-4.7-Flash-4bit` was promoted to a repo-owned
MLX runtime path after the GLM MLA attention, sigmoid router, and latent-KV
cache contracts landed and an AX server benchmark completed.
See
`benchmarks/results/mlx-inference/2026-05-06/README.md` for commands and
artifacts. Before promoting any additional architecture, run
`scripts/probe_mlx_model_support.py --model-dir <model-dir>`: GLM now reports
`repo_owned_runtime_ready` when the runtime-ready manifest and local reference
files are present.

## Limitations

- **GatedDelta prefill (Qwen3.5)**: The recurrent state update in GatedDelta
  linear-attention layers serializes over time steps and cannot be parallelized.
  On **Qwen3.5 9B** this puts AX prefill ~9% behind mlx-swift-lm at 512 tokens;
  decode throughput is unaffected. **Qwen3-Next (Coder Next) is not affected** —
  AX prefill exceeds mlx-swift-lm by 2× on that architecture because the sparse
  MoE forward path dominates the runtime, not the GatedDelta layers.
- **Raw HuggingFace weights**: ax-engine loads MLX community (pre-sanitized)
  weights only. For hybrid architectures (Qwen3.5, Qwen3-Next), loading an
  unsanitized checkpoint now raises a hard error — norm weight mean is sampled at
  load time and a clear remediation message is shown. Convert first with
  `mlx_lm.convert`, or download a pre-sanitized model from mlx-community. See
  [Getting a Model](#getting-a-model).
- **N-gram acceleration rows**: effective-throughput measurements, not raw
  model-kernel speedups. The n-gram hit rate is prompt- and output-pattern
  dependent. Coding-shaped workloads with repeated local structure are the
  intended high-value case; random, high-entropy, very short, or deliberately
  diverse outputs may see little benefit, and the runtime backs off toward the
  direct path when the accept rate drops below threshold.
- **TurboQuant KV compression**: experimental and off by default. The
  `turboquant-shadow` and `turboquant-fused-experimental` modes are evidence and
  route-telemetry surfaces, not production support claims. The correctness quality
  gate (K8/V4 fused path, zero fallbacks) now passes for Gemma 4 E2B; the
  remaining blocker is a long-context performance promotion artifact (≥8192-token
  context) required before public docs can drop the experimental label. Run
  `scripts/check_turboquant_promotion_readiness.py` to see the current gate
  status before changing any public support wording.

## Performance ([methodology](docs/PERFORMANCE.md))

Full README generation-model benchmark refresh ran on 2026-05-11. Artifact directory: `benchmarks/results/mlx-inference/2026-05-11-full-readme-refresh/`. Fresh rows completed for all 14 README generation models with `mlx_lm`, `mlx_swift_lm`, AX direct same-policy, and AX default n-gram at 128/512 prompt tokens and 128 generated tokens. Qwen Coder Next was retested on 2026-05-12 after the local model artifact was fixed; source: `benchmarks/results/mlx-inference/2026-05-12-qwen-coder-retest-r2/qwen3-coder-next-4bit.json`. Each successful fresh model used 5 repetitions and a 10-second trial delay/cooling interval; `ax-engine-server` and the `mlx-swift-lm` adapter were rebuilt before the run.

**Prefill** — Across the fresh models, AX engine prefill is -9% to +131% vs mlx_lm. The weakest fresh row is Gemma 4 E2B at 512 tokens, while the strongest fresh row is Qwen 3.6 35B A3B at 128 tokens.

**Decode** — Direct decode (n-gram disabled) spans -12% to +27% vs mlx_lm across fresh rows. With n-gram acceleration (the default), fresh rows span -0.4% to +207% vs mlx_lm; all fresh models except the Qwen Coder Next 512-token default row are above mlx_lm in default mode.

**TTFT** — AX TTFT is lower than mlx_lm on 24/28 fresh prompt rows. Source: `benchmarks/results/mlx-inference/2026-05-11-full-readme-refresh/`. mlx_lm TTFT is derived from reported prefill throughput; ax engine TTFT is measured directly from per-step runner timing.

Additional long-context validation artifacts are checked in separately from the
short/mid-prompt public tables. On 2026-05-07, `mlx-community/Qwen3-4B-4bit`
was run on Apple M5 Max through the P1 prefill-scaling gate and the P2
startup/concurrent-prefill gate:
[P1 prefill scaling](benchmarks/results/mlx-inference/2026-05-07-real-p1/qwen3-4b-4bit-prefill-scaling/prefill-scaling.md),
[P2 startup and concurrency](benchmarks/results/mlx-inference/2026-05-07-real-p2/qwen3-4b-4bit-p2-latency/p2-latency.md).
These artifacts measure direct AX MLX behavior, not n-gram decode acceleration:
the 8k P1 AX/MLX prefill ratio was 0.840x, and the 4-request P2 concurrent
prefill row was classified as serialized. Treat them as expectation-management
evidence for long-context serving claims, not as proof of continuous batching.

### Prefill throughput (tok/s) — percentages vs mlx_lm

| Model | MLX quantization | Prompt tok | mlx_lm | mlx_swift_lm | ax engine |
|---|---|---:|---:|---:|---:|
| Gemma 4 E2B | 4-bit · group=64 · affine | 128 | 2,615.9 | 2,610.1 (-0.2%) | 3,276.7 (+25.3%) |
|        |        | 512 | 8,378.7 | 6,768.2 (-19.2%) | **7,661.6 (-8.6%)** |
| Gemma 4 E2B | 5-bit · group=64 · affine | 128 | 2,711.9 | 2,429.3 (-10.4%) | 3,224.3 (+18.9%) |
|        |        | 512 | 7,742.3 | 7,032.0 (-9.2%) | **7,375.0 (-4.7%)** |
| Gemma 4 E2B | 6-bit · group=64 · affine | 128 | 2,306.9 | 2,404.1 (+4.2%) | 3,230.9 (+40.1%) |
|        |        | 512 | 7,882.5 | 6,643.1 (-15.7%) | **7,566.4 (-4.0%)** |
| Gemma 4 E2B | 8-bit · group=64 · affine | 128 | 2,190.2 | 2,413.8 (+10.2%) | 3,276.1 (+49.6%) |
|        |        | 512 | 7,181.8 | 6,039.5 (-15.9%) | 7,611.5 (+6.0%) |
| Gemma 4 E4B | 4-bit · group=64 · affine | 128 | 1,745.0 | 1,954.7 (+12.0%) | 2,525.2 (+44.7%) |
|        |        | 512 | 4,568.3 | 4,246.1 (-7.1%) | **4,266.4 (-6.6%)** |
| Gemma 4 26B A4B | 4-bit · group=64 · affine | 128 | 735.7 | 1,259.3 (+71.2%) | 1,212.8 (+64.8%) |
|        |        | 512 | 2,120.0 | 2,929.9 (+38.2%) | 2,985.5 (+40.8%) |
| Gemma 4 31B | 4-bit · group=64 · affine | 128 | 366.2 | 644.8 (+76.1%) | 559.1 (+52.7%) |
|        |        | 512 | 638.8 | 811.5 (+27.0%) | 724.5 (+13.4%) |
| Qwen 3.5 9B | 4-bit · group=64 · affine | 128 | 968.2 | 1,795.9 (+85.5%) | 2,076.2 (+114.4%) |
|        |        | 512 | 1,787.8 | 2,402.1 (+34.4%) | 2,921.0 (+63.4%) |
| Qwen 3.6 35B A3B | UD-MLX 4-bit · group=64 · affine | 128 | 572.1 | 1,036.5 (+81.2%) | 1,052.8 (+84.0%) |
|        |        | 512 | 1,681.3 | 2,584.7 (+53.7%) | 2,719.5 (+61.8%) |
| Qwen 3.6 35B A3B | MLX 5-bit · group=64 · affine | 128 | 499.6 | 913.5 (+82.9%) | 1,019.7 (+104.1%) |
|        |        | 512 | 1,568.4 | 2,457.1 (+56.7%) | 2,608.5 (+66.3%) |
| Qwen 3.6 35B A3B | MLX 6-bit · group=64 · affine | 128 | 452.0 | 796.1 (+76.1%) | 974.3 (+115.6%) |
|        |        | 512 | 1,454.5 | 2,430.2 (+67.1%) | 2,516.5 (+73.0%) |
| Qwen 3.6 35B A3B | MLX 8-bit · group=64 · affine | 128 | 415.6 | 652.8 (+57.1%) | 961.9 (+131.5%) |
|        |        | 512 | 1,302.3 | 2,387.6 (+83.3%) | 2,463.4 (+89.1%) |
| Qwen Coder Next | 4-bit · group=64 · affine | 128 | 301.7 | 471.1 (+56.1%) | 770.1 (+155.2%) |
|        |        | 512 | 921.0 | 1,659.1 (+80.1%) | 1,742.4 (+89.2%) |
| GLM 4.7 Flash | 4-bit · group=64 · affine | 128 | 516.5 | 971.3 (+88.0%) | 868.8 (+68.2%) |
|        |        | 512 | 1,637.7 | 2,526.8 (+54.3%) | 2,356.8 (+43.9%) |

### Decode throughput (tok/s) — generation=128 tokens, temp=0

Higher is better. The direct AX column is the same-policy baseline with n-gram acceleration
disabled. The n-gram column is the default AX decode policy and the row to use for
user-facing throughput expectations. Qwen Coder Next was retested after the local
model artifact fix; its direct row improved. The default n-gram row is now effectively
a direct fallback — the retest telemetry shows zero n-gram drafts produced
(`ax_ngram_fallback_linear_no_draft_steps: 460/635`), a regression introduced between
the recovery build (cf0228e) and the current build by the ADR 0018 linear-attention
prefix-cache changes. This is tracked as a separate open issue.

| Model | MLX quantization | Prompt tok | mlx_lm | mlx_swift_lm | ax direct baseline | ax default n-gram |
|---|---|---:|---:|---:|---:|---:|
| Gemma 4 E2B | 4-bit · group=64 · affine | 128 | 196.6 | 194.6 (-1.0%) | 183.9 (-6.5%) | **564.9 (+187.3%)** |
|        |        | 512 | 189.6 | 189.8 (+0.1%) | 177.9 (-6.2%) | **556.0 (+193.3%)** |
| Gemma 4 E2B | 5-bit · group=64 · affine | 128 | 186.5 | 174.7 (-6.3%) | 165.6 (-11.2%) | **450.0 (+141.3%)** |
|        |        | 512 | 178.9 | 165.8 (-7.4%) | 157.9 (-11.7%) | **456.9 (+155.3%)** |
| Gemma 4 E2B | 6-bit · group=64 · affine | 128 | 174.0 | 164.5 (-5.5%) | 157.0 (-9.8%) | **427.6 (+145.8%)** |
|        |        | 512 | 166.5 | 158.5 (-4.8%) | 152.8 (-8.2%) | **423.5 (+154.4%)** |
| Gemma 4 E2B | 8-bit · group=64 · affine | 128 | 153.8 | 145.4 (-5.5%) | 141.0 (-8.3%) | **462.3 (+200.6%)** |
|        |        | 512 | 149.7 | 141.6 (-5.4%) | 137.0 (-8.5%) | **459.3 (+206.8%)** |
| Gemma 4 E4B | 4-bit · group=64 · affine | 128 | 136.9 | 131.9 (-3.7%) | 121.5 (-11.3%) | **352.1 (+157.1%)** |
|        |        | 512 | 133.2 | 126.4 (-5.1%) | 117.6 (-11.7%) | **348.3 (+161.5%)** |
| Gemma 4 26B A4B | 4-bit · group=64 · affine | 128 | 125.8 | 118.9 (-5.4%) | 122.4 (-2.7%) | **273.6 (+117.6%)** |
|        |        | 512 | 123.6 | 115.8 (-6.3%) | 119.4 (-3.4%) | **230.1 (+86.1%)** |
| Gemma 4 31B | 4-bit · group=64 · affine | 128 | 28.3 | 27.7 (-2.1%) | 27.7 (-2.1%) | **64.8 (+128.9%)** |
|        |        | 512 | 27.7 | 27.3 (-1.6%) | 27.3 (-1.6%) | **62.7 (+126.3%)** |
| Qwen 3.5 9B | 4-bit · group=64 · affine | 128 | 82.3 | 80.9 (-1.7%) | **104.3 (+26.7%)** | **205.6 (+149.9%)** |
|        |        | 512 | 81.8 | 78.9 (-3.6%) | **103.8 (+26.8%)** | **101.0 (+23.4%)** |
| Qwen 3.6 35B A3B | UD-MLX 4-bit · group=64 · affine | 128 | 115.3 | 117.4 (+1.8%) | **124.7 (+8.1%)** | **283.5 (+145.8%)** |
|        |        | 512 | 117.8 | 113.6 (-3.5%) | **120.1 (+2.0%)** | **279.8 (+137.6%)** |
| Qwen 3.6 35B A3B | MLX 5-bit · group=64 · affine | 128 | 121.7 | 116.5 (-4.2%) | **136.0 (+11.7%)** | **279.5 (+129.6%)** |
|        |        | 512 | 121.7 | 116.0 (-4.7%) | **132.4 (+8.8%)** | **279.1 (+129.4%)** |
| Qwen 3.6 35B A3B | MLX 6-bit · group=64 · affine | 128 | 112.1 | 108.2 (-3.6%) | **116.8 (+4.1%)** | **258.4 (+130.4%)** |
|        |        | 512 | 109.3 | 107.0 (-2.1%) | **119.9 (+9.7%)** | **256.1 (+134.4%)** |
| Qwen 3.6 35B A3B | MLX 8-bit · group=64 · affine | 128 | 102.9 | 99.2 (-3.5%) | **108.9 (+5.9%)** | **260.4 (+153.2%)** |
|        |        | 512 | 101.5 | 98.4 (-3.0%) | **108.0 (+6.4%)** | **249.7 (+146.0%)** |
| Qwen Coder Next | 4-bit · group=64 · affine | 128 | 90.2 | 89.1 (-1.2%) | **92.7 (+2.8%)** | **90.6 (+0.5%)** |
|        |        | 512 | 91.6 | 86.6 (-5.4%) | 91.3 (-0.3%) | **92.4 (+0.9%)** |
| GLM 4.7 Flash | 4-bit · group=64 · affine | 128 | 101.9 | 96.6 (-5.2%) | 101.1 (-0.8%) | **273.8 (+168.7%)** |
|        |        | 512 | 98.3 | 91.3 (-7.2%) | **101.7 (+3.5%)** | **268.0 (+172.6%)** |

### Time to first token (ms) — generation=128 tokens, temp=0

Lower is better. mlx_lm and mlx_swift_lm values are derived from reported prefill
throughput (`prompt_tokens / prefill_tok_s × 1000 ms`); ax engine values are directly
measured from per-step runner timing in the SSE event stream. Source:
`benchmarks/results/mlx-inference/2026-05-11-full-readme-refresh/` for the full table,
with Qwen Coder Next sourced from the 2026-05-12 retest-r2 artifact.

| Model | MLX quantization | Prompt tok | mlx_lm | mlx_swift_lm | ax engine |
|---|---|---:|---:|---:|---:|
| Gemma 4 E2B | 4-bit · group=64 · affine | 128 | 48.9 | 49.0 (+0.2%) | **39.1 (-20.2%)** |
|        |        | 512 | 61.1 | 75.6 (+23.8%) | 66.8 (+9.4%) |
| Gemma 4 E2B | 5-bit · group=64 · affine | 128 | 47.2 | 52.7 (+11.6%) | **39.7 (-15.9%)** |
|        |        | 512 | 66.1 | 72.8 (+10.1%) | 69.4 (+5.0%) |
| Gemma 4 E2B | 6-bit · group=64 · affine | 128 | 55.5 | 53.2 (-4.0%) | **39.6 (-28.6%)** |
|        |        | 512 | 65.0 | 77.1 (+18.7%) | 67.7 (+4.2%) |
| Gemma 4 E2B | 8-bit · group=64 · affine | 128 | 58.4 | 53.0 (-9.3%) | **39.1 (-33.1%)** |
|        |        | 512 | 71.3 | 84.8 (+18.9%) | **67.3 (-5.6%)** |
| Gemma 4 E4B | 4-bit · group=64 · affine | 128 | 73.4 | 65.5 (-10.7%) | **50.7 (-30.9%)** |
|        |        | 512 | 112.1 | 120.6 (+7.6%) | 120.0 (+7.1%) |
| Gemma 4 26B A4B | 4-bit · group=64 · affine | 128 | 174.0 | 101.6 (-41.6%) | **105.5 (-39.3%)** |
|        |        | 512 | 241.5 | 174.7 (-27.6%) | **171.5 (-29.0%)** |
| Gemma 4 31B | 4-bit · group=64 · affine | 128 | 349.5 | 198.5 (-43.2%) | **228.9 (-34.5%)** |
|        |        | 512 | 801.5 | 630.9 (-21.3%) | **706.7 (-11.8%)** |
| Qwen 3.5 9B | 4-bit · group=64 · affine | 128 | 132.2 | 71.3 (-46.1%) | **61.7 (-53.4%)** |
|        |        | 512 | 286.4 | 213.1 (-25.6%) | **175.3 (-38.8%)** |
| Qwen 3.6 35B A3B | UD-MLX 4-bit · group=64 · affine | 128 | 223.7 | 123.5 (-44.8%) | **121.6 (-45.7%)** |
|        |        | 512 | 304.5 | 198.1 (-35.0%) | **188.3 (-38.2%)** |
| Qwen 3.6 35B A3B | MLX 5-bit · group=64 · affine | 128 | 256.2 | 140.1 (-45.3%) | **125.5 (-51.0%)** |
|        |        | 512 | 326.5 | 208.4 (-36.2%) | **196.3 (-39.9%)** |
| Qwen 3.6 35B A3B | MLX 6-bit · group=64 · affine | 128 | 283.2 | 160.8 (-43.2%) | **131.4 (-53.6%)** |
|        |        | 512 | 352.0 | 210.7 (-40.2%) | **203.5 (-42.2%)** |
| Qwen 3.6 35B A3B | MLX 8-bit · group=64 · affine | 128 | 308.0 | 196.1 (-36.3%) | **133.1 (-56.8%)** |
|        |        | 512 | 393.1 | 214.4 (-45.5%) | **207.8 (-47.1%)** |
| Qwen Coder Next | 4-bit · group=64 · affine | 128 | 424.2 | 271.7 (-36.0%) | **166.2 (-60.8%)** |
|        |        | 512 | 555.9 | 308.6 (-44.5%) | **293.9 (-47.1%)** |
| GLM 4.7 Flash | 4-bit · group=64 · affine | 128 | 247.8 | 131.8 (-46.8%) | **147.3 (-40.5%)** |
|        |        | 512 | 312.6 | 202.6 (-35.2%) | **217.2 (-30.5%)** |

### Embedding throughput (tok/s) — runtime apples-to-apples

Measured on the same tokenized inputs with matching pooling (`last`) and normalization (`true`) settings across backends. Source: `benchmarks/results/embedding/2026-05-12-readme-refresh/`.

Single-request median throughput (ax-engine-py vs mlx-lm, same session):

| Model | mlx-lm (baseline) | ax-engine-py |
|---|---:|---:|
| Qwen3-Embedding 0.6B 8-bit | 1,424.7 | 1,314.8 (≈-7.7%) |
| Qwen3-Embedding 4B 4-bit | 518.7 | 442.1 (-14.8%) |
| Qwen3-Embedding 8B 4-bit DWQ | 311.5 | 264.4 (-15.1%) |

† The 0.6B model completes in ~7ms/sentence in this run, making it sensitive to thermal variance. Run-to-run gap typically ranges from -5% to -15%.

## Installation

### Homebrew

For tagged macOS arm64 releases, install the preview command-line tools from
the AutomatosX tap:

```bash
brew install defai-digital/ax-engine/ax-engine
```

This installs:

- `ax-engine-server`: local HTTP adapter over the SDK runtime
- `ax-engine-bench`: workload-contract, readiness, direct-generate, and
  benchmark-support CLI
- `ax-engine-manager`: local web manager for readiness, model downloads, server
  metadata, endpoint URLs, guarded job plans, and redacted support bundles
- the Homebrew `mlx-c` runtime dependency required by the released binaries

Check the installed tools:

```bash
ax-engine-server --help
ax-engine-bench doctor
ax-engine-manager --check
```

Homebrew is the quickest path for the released server and benchmark binaries.
If `ax-engine-bench doctor` fails with `Library not loaded:
/opt/homebrew/opt/mlx-c/lib/libmlxc.dylib`, install or repair the runtime with
`brew install mlx-c` and `brew reinstall defai-digital/ax-engine/ax-engine`.
Use the source build when you need the full Rust workspace, Python extension,
local examples, or changes that have not been tagged yet.

The release archive attached to GitHub is the Homebrew formula payload. It is
not a standalone installer with bundled dynamic libraries. Use Homebrew unless
you are prepared to provide `mlx-c` and its dynamic library path yourself.

### Source

Development builds require Rust and the MLX C runtime on Apple Silicon:

```bash
brew install mlx-c
cargo build --workspace --release
```

Python bindings are built from source:

```bash
maturin develop
python -m unittest discover -s python/tests -v
```

## Quick Start

The fastest local workflow is:

1. install or build the command-line tools;
2. download a supported MLX model and generate its manifest;
3. start the local server;
4. open `ax-engine-manager` to manage downloads, server port controls,
   readiness, endpoint URLs, benchmark artifacts, and support bundles.

For a complete manager walkthrough, see [docs/MANAGER.md](docs/MANAGER.md).

The commands below use source-build paths. If you installed with Homebrew, use
`ax-engine-server`, `ax-engine-bench`, and `ax-engine-manager` directly instead
of `./target/release/...`.

```bash
# Download a model and generate its manifest
MODEL_DIR="$(python3 scripts/download_model.py mlx-community/Qwen3-4B-4bit --json | python3 -c 'import json,sys; print(json.load(sys.stdin)["dest"])')"
# MODEL_DIR uses the Hugging Face Hub snapshot cache by default, e.g.
# ~/.cache/huggingface/hub/models--mlx-community--Qwen3-4B-4bit/snapshots/<hash>

# Check readiness without opening the browser
./target/release/ax-engine-manager --check --model-dir "$MODEL_DIR"

# HTTP inference server (repo-owned MLX runtime)
./target/release/ax-engine-server \
  --mlx \
  --mlx-model-artifacts-dir "$MODEL_DIR" \
  --port 8080

# Local web manager
./target/release/ax-engine-manager \
  --model-dir "$MODEL_DIR" \
  --server-url http://127.0.0.1:8080
```

```python
# Python bindings (after maturin develop)
import ax_engine

path = ax_engine.download_model("mlx-community/Qwen3-4B-4bit")
with ax_engine.Session(mlx=True, mlx_model_artifacts_dir=str(path)) as s:
    result = s.generate([1, 2, 3], max_output_tokens=32)
    print(result.output_tokens)
```

For an unsupported MLX text model that upstream `mlx-lm` can serve, keep AX
Engine as the CLI/server surface and delegate the model execution explicitly:

```bash
mlx_lm.server --model /path/to/local/mlx-model --host 127.0.0.1 --port 8090

./target/release/ax-engine-bench generate \
  --prompt "Hello from mlx-lm" \
  --support-tier mlx_lm_delegated \
  --mlx-lm-server-url http://127.0.0.1:8090
```

`mlx_lm_delegated` is a compatibility route, not a repo-owned MLX throughput
claim. It forwards text generation to upstream `mlx_lm.server`, preserves AX
sampling fields such as `temperature`, `top_p`, `top_k`, `repetition_penalty`,
and `seed`, and exposes blocking plus SSE text surfaces through AX. Streamed
chunks are delegated text deltas; they are not AX-owned token IDs, KV state, or
model-kernel throughput evidence. Tool calls and visual/multimodal inputs are
not yet AX compatibility contracts.

```bash
# Primary benchmark: AX vs mlx_lm vs mlx-swift-lm
python3 scripts/bench_mlx_inference_stack.py \
  --model-dir /path/to/local/mlx-model \
  --prompt-tokens 128,512 --generation-tokens 128 \
  --ax-compare-policies --repetitions 3 \
  --mlx-swift-lm-command './scripts/mlx-swift-bench/.build/release/mlx-swift-bench \
    --model {model} --prompt-token-ids {prompt_token_ids_path} \
    --generation-tokens {generation_tokens} --trials {trials} \
    --delay {delay} --prefill-step-size {prefill_step_size}' \
  --output benchmarks/results/mlx-inference/2026-05-04/gemma-4-e2b-it-4bit.json

# Secondary workload-contract benchmark
./target/release/ax-engine-bench scenario \
  --manifest benchmarks/manifests/scenario/chat_gemma4_e2b_short.json \
  --output-root benchmarks/results

# Smoke checks
./target/release/ax-engine-manager --check --model-dir "$MODEL_DIR"
bash scripts/check-server-preview.sh
bash scripts/check-python-preview.sh
```

## Getting a Model

ax-engine requires pre-sanitized MLX weights. The recommended source is
[mlx-community](https://huggingface.co/mlx-community) — models there are already
converted and validated. Loading an unsanitized raw HF checkpoint into a hybrid
architecture (Qwen3.5, Qwen3-Next) raises a hard error at load time.

### mlx-community model (recommended)

`download_model()` and `scripts/download_model.py` download weights and auto-generate
the required `model-manifest.json` in one step:

```bash
# Script (works with Homebrew install or source build)
python scripts/download_model.py mlx-community/Qwen3-4B-4bit

# For automation and manager integration, emit a parseable summary
python scripts/download_model.py mlx-community/Qwen3-4B-4bit --json

# Python SDK
from ax_engine import download_model
path = download_model("mlx-community/Qwen3-4B-4bit")
```

By default these helpers use the same Hugging Face Hub snapshot cache as
`mlx_lm` and `huggingface_hub`. If you already have `mlx_lm` installed, its
download also lands in that cache and ax-engine can auto-discover it:

```bash
python -m mlx_lm.generate --model mlx-community/Qwen3-4B-4bit --prompt "x" --max-tokens 1
ax-engine-bench generate-manifest ~/.cache/huggingface/hub/models--mlx-community--Qwen3-4B-4bit/snapshots/<hash>
ax-engine-server --mlx --resolve-model-artifacts hf-cache --preset qwen3_dense --port 8080
```

### Raw HuggingFace checkpoint

Raw checkpoints need sanitization before ax-engine can load them. Use `mlx_lm.convert`:

```bash
pip install mlx-lm
mlx_lm.convert --hf-path <org/model> --mlx-path /path/to/dest -q --q-bits 4
ax-engine-bench generate-manifest /path/to/dest
ax-engine-server --mlx --mlx-model-artifacts-dir /path/to/dest --port 8080
```

### Manifest generation

Both paths above require `model-manifest.json`. The download helpers generate it
automatically. To run it directly:

```bash
ax-engine-bench generate-manifest /path/to/model      # Homebrew or built binary
cargo run -p ax-engine-core --bin generate-manifest -- /path/to/model  # source
```

## SDKs

ax-engine-server exposes OpenAI-compatible HTTP endpoints, and several SDKs
wrap those endpoints or the in-process Rust session directly.

| Language | Package / path | LangChain |
|----------|---------------|-----------|
| **Python** | `python/ax_engine` | `ax_engine.langchain` — `AXEngineChatModel`, `AXEngineLLM` |
| **TypeScript / JS** | `javascript/ax-engine` (`@ax-engine/sdk`) | `@ax-engine/sdk/langchain` — `ChatAXEngine`, `AXEngineLLM` |
| **Go** | `sdk/go/axengine` | Use [langchaingo](https://github.com/tmc/langchaingo) OpenAI provider — see `examples/go/langchain/` |
| **Ruby** | `sdk/ruby` (`ax-engine-sdk`) | `ax_engine/langchain` — `ChatModel`, `LLM` (requires langchain-rb) |
| **Mojo** | `sdk/mojo/ax_engine.mojo` | Via Python — use `ax_engine.langchain` from Mojo's Python interop |

### TypeScript / JavaScript

```bash
npm install @ax-engine/sdk
```

```typescript
import AxEngineClient from "@ax-engine/sdk";

const client = new AxEngineClient({ baseUrl: "http://127.0.0.1:8080" });
const resp = await client.chatCompletion({
  messages: [{ role: "user", content: "Hello!" }],
  max_tokens: 128,
});
console.log(resp.choices[0].message.content);

// Streaming
for await (const event of client.streamChatCompletion({ messages: [...], stream: true })) {
  process.stdout.write(event.data.choices[0]?.delta?.content ?? "");
}
```

LangChain integration (requires `@langchain/core`):

```typescript
import { ChatAXEngine } from "@ax-engine/sdk/langchain";
import { HumanMessage } from "@langchain/core/messages";

const chat = new ChatAXEngine({ maxTokens: 128 });
const response = await chat.invoke([new HumanMessage("Hello!")]);
```

### Go

The Go SDK lives at `sdk/go/axengine` (module `github.com/ax-engine/ax-engine-go`).

```go
client := axengine.NewClient(nil)

resp, err := client.ChatCompletion(ctx, axengine.OpenAiChatCompletionRequest{
    Messages:  []axengine.OpenAiChatMessage{{Role: "user", Content: "Hello!"}},
    MaxTokens: axengine.Ptr(128),
})

// Streaming
ch, errCh := client.StreamChatCompletion(ctx, req)
for chunk := range ch {
    fmt.Print(*chunk.Choices[0].Delta.Content)
}
```

See `examples/go/` for runnable examples. For LangChain, point
[langchaingo](https://github.com/tmc/langchaingo)'s OpenAI provider at
`http://127.0.0.1:8080/v1` — see `examples/go/langchain/` and `docs/GO.md`.

### Ruby

The Ruby SDK lives at `sdk/ruby/` (`ax-engine-sdk` gem). Zero dependencies —
stdlib `net/http` only. Streaming uses a block interface.

```ruby
require "ax_engine"

client = AxEngine::Client.new

# Blocking chat
resp = client.chat_completion(
  messages: [{ role: "user", content: "Hello!" }],
  max_tokens: 128
)
puts resp.dig("choices", 0, "message", "content")

# Streaming
client.stream_chat_completion(
  messages: [{ role: "user", content: "Count from 1 to 5." }],
  max_tokens: 64
) do |event|
  print event.dig("data", "choices", 0, "delta", "content").to_s
end
```

LangChain via [langchain-rb](https://github.com/patterns-ai-core/langchain):

```ruby
require "ax_engine/langchain"

chat = AxEngine::Langchain::ChatModel.new(max_tokens: 256)
puts chat.chat(messages: [{ role: "user", content: "Hello!" }]).chat_completion
```

See `examples/ruby/` and `docs/RUBY.md` for full details.

### Python — LangChain

```python
from ax_engine.langchain import AXEngineChatModel
from langchain_core.messages import HumanMessage

chat = AXEngineChatModel(base_url="http://127.0.0.1:8080", max_tokens=256)
response = chat.invoke([HumanMessage(content="Hello!")])
print(response.content)

# Streaming
for chunk in chat.stream([HumanMessage(content="Count from 1 to 5.")]):
    print(chunk.content, end="", flush=True)
```

Requires `pip install langchain-core`. See `docs/PYTHON.md` for full details.

### Mojo

The Mojo SDK (`sdk/mojo/ax_engine.mojo`) wraps the Python `ax_engine` package
via Mojo's `PythonObject` interop. Requires the Python extension to be built
first (`maturin develop`).

```mojo
from sdk.mojo.ax_engine import Session

var session = Session(
    "qwen3_dense",
    mlx=True,
    mlx_model_artifacts_dir="/path/to/artifacts",
)
var result = session.generate("Hello from Mojo!", max_output_tokens=64)
print(result.output_text)
session.close()
```

## Workspace

```
crates/ax-engine-core    Engine state machine, scheduler, KV manager, sampler
crates/ax-engine-mlx     MLX model graph, n-gram acceleration, KV cache, runner
crates/mlx-sys           bindgen FFI over mlx-c; safe MlxArray RAII wrappers
crates/ax-engine-sdk     Session API, backend resolution (MLX, mlx-lm delegated, or llama.cpp)
crates/ax-engine-server  Axum HTTP/SSE adapter (OpenAI-compatible routes)
crates/ax-engine-bench   Manifest-driven workload-contract CLI
crates/ax-engine-py      PyO3 extension (ABI3, Python 3.10+)
javascript/ax-engine     TypeScript/JS HTTP SDK + LangChain adapter
sdk/go/axengine          Go HTTP SDK
sdk/ruby/                Ruby HTTP SDK (ax-engine-sdk gem)
sdk/mojo/                Mojo SDK (Python-interop)
```

Unsupported MLX text models can use the explicit delegated `mlx_lm_delegated`
route through a user-provided `mlx_lm.server`. Non-MLX inference routes through
the delegated `llama.cpp` contract.

## Development

```bash
cargo build --workspace                                           # build all crates
cargo test --quiet                                                # full Rust test suite
cargo clippy --all-targets --all-features -- -D warnings         # lint (CI gate)
cargo fmt                                                         # format
maturin develop                                                   # rebuild Python extension
python -m unittest discover -s python/tests -v                   # Python tests
```

Coverage is collected by the report-only GitHub Actions workflow in
`.github/workflows/coverage.yml`. It publishes Rust `cargo llvm-cov` and Python
`coverage.py` artifacts without enforcing a percentage threshold yet; add a gate
only after the project has a stable baseline across macOS, MLX, and PyO3 paths.

Public documentation is in `docs/`. Canonical benchmark manifests are in
`benchmarks/manifests/`. Key design documents:
[SDK / API](docs/SDK.md) ·
[Manager](docs/MANAGER.md) ·
[Python](docs/PYTHON.md) ·
[JavaScript / TypeScript](docs/JAVASCRIPT.md) ·
[Go](docs/GO.md) ·
[Ruby](docs/RUBY.md) ·
[Mojo](docs/MOJO.md) ·
[Scheduler](docs/SCHEDULER.md) ·
[KV Cache](docs/KV-CACHE.md) ·
[Benchmarking](docs/BENCH-DESIGN.md) ·
[Serving Benchmarks](docs/SERVING-BENCHMARKS.md)

## Contributing

AX Engine welcomes public contributions. See [CONTRIBUTING.md](CONTRIBUTING.md)
for guidelines.

## Community

- Website: [automatosx.com](https://automatosx.com)
- Discord: [Join us](https://discord.com/invite/cTavsMgu)
- Email: enquiry@defai.digital

## License

MIT License. See [LICENSE](LICENSE) for details.

Copyright (c) 2026 [DEFAI Private Limited](https://defai.digital)
