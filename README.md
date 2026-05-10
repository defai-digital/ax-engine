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

Qwen Coder Next single-model benchmark refresh is in progress on 2026-05-10.
Artifact directory:
`benchmarks/results/mlx-inference/2026-05-10-qwen-coder-refresh/`. The run
scope is `qwen3-coder-next-4bit` with `mlx_lm`, `mlx_swift_lm`, AX direct
same-policy, and AX default n-gram at 128/512 prompt tokens and 128 generated
tokens. Reference rows are running first; AX direct and AX n-gram will be
attempted after a 10-second cooling pause. The table rows below remain from the
last validated README state until the Qwen Coder refresh stage has a complete
or explicitly blocked artifact.

**Prefill** — Across the 13 fresh rows, AX engine prefill is +39–142% vs
mlx_lm at 128 tokens. At 512 tokens the spread is wider: Gemma E4B 4-bit is
5% behind mlx_lm, while Qwen 3.6 35B 8-bit is +94%. Coder Next is still shown
from the 2026-05-09 seed artifact until a sanitized local checkpoint is
available.

**Decode** — Direct decode (n-gram disabled) spans −17% to +13% vs mlx_lm on
the fresh rows. Qwen 3.6 35B variants remain positive in direct mode, while
Gemma E2B and E4B quantizations are behind. With n-gram acceleration (the
default), 25/26 fresh prompt rows are above mlx_lm, reaching +198%; the Qwen
3.5 512-token row backs off to roughly parity when no draft candidates are
available.

**TTFT** — AX TTFT is lower than mlx_lm on 23/26 fresh prompt rows. The best
fresh rows are the Qwen 3.6 35B variants (up to −59% at 128 tokens and −48% at
512 tokens); the only fresh regression above +5% is Gemma E4B 4-bit at 512
tokens. Source: `benchmarks/results/mlx-inference/2026-05-10-full-readme-refresh/`.
mlx_lm TTFT is derived from reported prefill throughput; ax engine TTFT is
measured directly from per-step runner timing.

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
| Gemma 4 E2B | 4-bit · group=64 · affine | 128 | 2,328.7 | 2,161.0 (-7.2%) | 3,238.0 (+39.0%) |
|      |      | 512 | 7,376.7 | 5,825.5 (-21.0%) | 7,187.4 (-2.6%) |
| Gemma 4 E2B | 5-bit · group=64 · affine | 128 | 2,099.0 | 1,951.1 (-7.0%) | 3,252.9 (+55.0%) |
|      |      | 512 | 6,982.7 | 5,392.1 (-22.8%) | 7,001.4 (+0.3%) |
| Gemma 4 E2B | 6-bit · group=64 · affine | 128 | 1,968.4 | 2,240.1 (+13.8%) | 2,947.7 (+49.8%) |
|      |      | 512 | 6,919.9 | 6,315.8 (-8.7%) | 6,826.5 (-1.3%) |
| Gemma 4 E2B | 8-bit · group=64 · affine | 128 | 1,977.8 | 2,225.3 (+12.5%) | 3,132.3 (+58.4%) |
|      |      | 512 | 6,451.9 | 5,598.0 (-13.2%) | 6,839.6 (+6.0%) |
| Gemma 4 E4B | 4-bit · group=64 · affine | 128 | 1,628.1 | 2,242.7 (+37.7%) | 2,435.5 (+49.6%) |
|      |      | 512 | 4,339.4 | 4,307.5 (-0.7%) | 4,115.6 (-5.2%) |
| Gemma 4 26B A4B | 4-bit · group=64 · affine | 128 | 651.5 | 1,274.3 (+95.6%) | 1,140.9 (+75.1%) |
|      |      | 512 | 1,979.8 | 2,834.8 (+43.2%) | 2,716.7 (+37.2%) |
| Gemma 4 31B | 4-bit · group=64 · affine | 128 | 332.9 | 623.6 (+87.3%) | 522.5 (+56.9%) |
|      |      | 512 | 590.0 | 783.4 (+32.8%) | 685.6 (+16.2%) |
| Qwen 3.5 9B | 4-bit · group=64 · affine | 128 | 1,122.4 | 1,913.1 (+70.4%) | 1,969.4 (+75.5%) |
|      |      | 512 | 2,226.5 | 3,113.8 (+39.8%) | 2,740.9 (+23.1%) |
| Qwen 3.6 35B A3B | UD-MLX 4-bit · group=64 · affine | 128 | 525.0 | 984.9 (+87.6%) | 1,032.5 (+96.7%) |
|      |      | 512 | 1,606.4 | 2,532.3 (+57.6%) | 2,623.5 (+63.3%) |
| Qwen 3.6 35B A3B | MLX 5-bit · group=64 · affine | 128 | 452.2 | 826.9 (+82.8%) | 1,007.4 (+122.8%) |
|      |      | 512 | 1,497.7 | 2,453.7 (+63.8%) | 2,571.7 (+71.7%) |
| Qwen 3.6 35B A3B | MLX 6-bit · group=64 · affine | 128 | 416.7 | 758.0 (+81.9%) | 964.3 (+131.4%) |
|      |      | 512 | 1,378.7 | 2,393.3 (+73.6%) | 2,445.8 (+77.4%) |
| Qwen 3.6 35B A3B | MLX 8-bit · group=64 · affine | 128 | 400.7 | 644.4 (+60.8%) | 968.0 (+141.6%) |
|      |      | 512 | 1,261.9 | 2,321.8 (+84.0%) | 2,449.0 (+94.1%) |
| Qwen Coder Next | 4-bit · group=64 · affine | 128 | 267.1 | 384.9 (+44.1%) | 714.4 (+167.4%) |
|      |      | 512 | 815.4 | 1,417.0 (+73.8%) | 1,665.1 (+104.2%) |
| GLM 4.7 Flash | 4-bit · group=64 · affine | 128 | 495.2 | 997.9 (+101.5%) | 869.7 (+75.6%) |
|      |      | 512 | 1,600.7 | 2,518.8 (+57.4%) | 2,384.3 (+49.0%) |

### Decode throughput (tok/s) — generation=128 tokens, temp=0

Higher is better. The direct AX column is the same-policy baseline with n-gram acceleration
disabled. The n-gram column is the default AX decode policy and the row to use for
user-facing throughput expectations. For Qwen 3.5 at 512 prompt tokens, the default
n-gram row can fall back below direct mode when no draft candidates are available.

| Model | MLX quantization | Prompt tok | mlx_lm | mlx_swift_lm | ax direct baseline | ax default n-gram |
|---|---|---:|---:|---:|---:|---:|
| Gemma 4 E2B | 4-bit · group=64 · affine | 128 | 204.9 | 187.9 (-8.3%) | 175.7 (-14.2%) | **526.6 (+157.0%)** |
|      |      | 512 | 196.7 | 187.3 (-4.8%) | 162.9 (-17.2%) | **526.9 (+167.8%)** |
| Gemma 4 E2B | 5-bit · group=64 · affine | 128 | 183.6 | 170.5 (-7.1%) | 164.0 (-10.7%) | **423.9 (+131.0%)** |
|      |      | 512 | 174.3 | 170.2 (-2.4%) | 154.2 (-11.5%) | **404.7 (+132.2%)** |
| Gemma 4 E2B | 6-bit · group=64 · affine | 128 | 153.0 | 149.4 (-2.3%) | 138.4 (-9.6%) | **392.5 (+156.6%)** |
|      |      | 512 | 158.8 | 148.9 (-6.2%) | 135.8 (-14.5%) | **394.1 (+148.2%)** |
| Gemma 4 E2B | 8-bit · group=64 · affine | 128 | 141.2 | 135.8 (-3.8%) | 130.2 (-7.8%) | **420.2 (+197.6%)** |
|      |      | 512 | 140.5 | 131.1 (-6.7%) | 125.3 (-10.8%) | **406.7 (+189.5%)** |
| Gemma 4 E4B | 4-bit · group=64 · affine | 128 | 127.2 | 122.4 (-3.8%) | 112.8 (-11.3%) | **325.8 (+156.2%)** |
|      |      | 512 | 123.7 | 119.5 (-3.4%) | 109.5 (-11.5%) | **325.4 (+163.1%)** |
| Gemma 4 26B A4B | 4-bit · group=64 · affine | 128 | 119.7 | 107.9 (-9.8%) | 108.1 (-9.7%) | **250.7 (+109.4%)** |
|      |      | 512 | 112.7 | 104.3 (-7.4%) | 107.0 (-5.1%) | **209.2 (+85.6%)** |
| Gemma 4 31B | 4-bit · group=64 · affine | 128 | 24.6 | 26.5 (+7.8%) | 24.3 (-1.1%) | **46.3 (+88.2%)** |
|      |      | 512 | 25.6 | 25.5 (-0.5%) | 25.1 (-2.1%) | **57.8 (+126.1%)** |
| Qwen 3.5 9B | 4-bit · group=64 · affine | 128 | 91.0 | 90.9 (-0.1%) | 93.8 (+3.1%) | **196.0 (+115.3%)** |
|      |      | 512 | 91.7 | 89.9 (-2.0%) | 94.3 (+2.8%) | 91.5 (-0.3%) |
| Qwen 3.6 35B A3B | UD-MLX 4-bit · group=64 · affine | 128 | 109.8 | 113.5 (+3.3%) | 119.6 (+8.9%) | **271.9 (+147.6%)** |
|      |      | 512 | 112.8 | 110.9 (-1.7%) | 118.4 (+5.0%) | **271.1 (+140.4%)** |
| Qwen 3.6 35B A3B | MLX 5-bit · group=64 · affine | 128 | 116.9 | 116.1 (-0.6%) | 131.9 (+12.9%) | **277.2 (+137.2%)** |
|      |      | 512 | 121.0 | 115.7 (-4.4%) | 130.8 (+8.0%) | **265.6 (+119.4%)** |
| Qwen 3.6 35B A3B | MLX 6-bit · group=64 · affine | 128 | 106.5 | 104.2 (-2.2%) | 117.9 (+10.6%) | **246.1 (+131.0%)** |
|      |      | 512 | 106.4 | 101.9 (-4.2%) | 116.7 (+9.8%) | **240.9 (+126.5%)** |
| Qwen 3.6 35B A3B | MLX 8-bit · group=64 · affine | 128 | 99.6 | 99.4 (-0.2%) | 106.9 (+7.3%) | **253.5 (+154.5%)** |
|      |      | 512 | 101.1 | 97.7 (-3.4%) | 108.0 (+6.8%) | **255.6 (+152.8%)** |
| Qwen Coder Next | 4-bit · group=64 · affine | 128 | 92.2 | 89.4 (-3.0%) | 89.3 (-3.1%) | **223.2 (+142.2%)** |
|      |      | 512 | 90.4 | 89.2 (-1.3%) | 89.2 (-1.3%) | **220.8 (+144.3%)** |
| GLM 4.7 Flash | 4-bit · group=64 · affine | 128 | 105.3 | 98.8 (-6.2%) | 104.4 (-0.8%) | **279.8 (+165.8%)** |
|      |      | 512 | 100.3 | 96.0 (-4.4%) | 103.9 (+3.5%) | **273.3 (+172.3%)** |

### Time to first token (ms) — generation=128 tokens, temp=0

Lower is better. mlx_lm and mlx_swift_lm values are derived from reported prefill
throughput (`prompt_tokens / prefill_tok_s × 1000 ms`); ax engine values are directly
measured from per-step runner timing in the SSE event stream. Source:
`benchmarks/results/mlx-inference/2026-05-10-full-readme-refresh/`.

| Model | MLX quantization | Prompt tok | mlx_lm | mlx_swift_lm | ax engine |
|---|---|---:|---:|---:|---:|
| Gemma 4 E2B | 4-bit · group=64 · affine | 128 | 55.0 | 59.2 (+7.8%) | **39.5 (-28.1%)** |
|      |      | 512 | 69.4 | 87.9 (+26.6%) | 71.2 (+2.6%) |
| Gemma 4 E2B | 5-bit · group=64 · affine | 128 | 61.0 | 65.6 (+7.6%) | **39.3 (-35.5%)** |
|      |      | 512 | 73.3 | 95.0 (+29.5%) | **73.1 (-0.3%)** |
| Gemma 4 E2B | 6-bit · group=64 · affine | 128 | 65.0 | 57.1 (-12.1%) | **43.4 (-33.2%)** |
|      |      | 512 | 74.0 | 81.1 (+9.6%) | 75.0 (+1.4%) |
| Gemma 4 E2B | 8-bit · group=64 · affine | 128 | 64.7 | 57.5 (-11.1%) | **40.9 (-36.9%)** |
|      |      | 512 | 79.4 | 91.5 (+15.3%) | **74.9 (-5.7%)** |
| Gemma 4 E4B | 4-bit · group=64 · affine | 128 | 78.6 | 57.1 (-27.4%) | **52.6 (-33.2%)** |
|      |      | 512 | 118.0 | 118.9 (+0.7%) | 124.4 (+5.4%) |
| Gemma 4 26B A4B | 4-bit · group=64 · affine | 128 | 196.5 | 100.4 (-48.9%) | **112.2 (-42.9%)** |
|      |      | 512 | 258.6 | 180.6 (-30.2%) | **188.5 (-27.1%)** |
| Gemma 4 31B | 4-bit · group=64 · affine | 128 | 384.5 | 205.3 (-46.6%) | **245.0 (-36.3%)** |
|      |      | 512 | 867.7 | 653.6 (-24.7%) | **746.7 (-13.9%)** |
| Qwen 3.5 9B | 4-bit · group=64 · affine | 128 | 114.0 | 66.9 (-41.3%) | **65.0 (-43.0%)** |
|      |      | 512 | 230.0 | 164.4 (-28.5%) | **186.8 (-18.8%)** |
| Qwen 3.6 35B A3B | UD-MLX 4-bit · group=64 · affine | 128 | 243.8 | 130.0 (-46.7%) | **124.0 (-49.1%)** |
|      |      | 512 | 318.7 | 202.2 (-36.6%) | **195.2 (-38.8%)** |
| Qwen 3.6 35B A3B | MLX 5-bit · group=64 · affine | 128 | 283.0 | 154.8 (-45.3%) | **127.1 (-55.1%)** |
|      |      | 512 | 341.8 | 208.7 (-39.0%) | **199.1 (-41.8%)** |
| Qwen 3.6 35B A3B | MLX 6-bit · group=64 · affine | 128 | 307.1 | 168.9 (-45.0%) | **132.7 (-56.8%)** |
|      |      | 512 | 371.4 | 213.9 (-42.4%) | **209.3 (-43.6%)** |
| Qwen 3.6 35B A3B | MLX 8-bit · group=64 · affine | 128 | 319.5 | 198.6 (-37.8%) | **132.2 (-58.6%)** |
|      |      | 512 | 405.7 | 220.5 (-45.6%) | **209.1 (-48.5%)** |
| Qwen Coder Next | 4-bit · group=64 · affine | 128 | 479.2 | 332.6 (-30.6%) | **179.2 (-62.6%)** |
|      |      | 512 | 627.9 | 361.3 (-42.5%) | **307.5 (-51.0%)** |
| GLM 4.7 Flash | 4-bit · group=64 · affine | 128 | 258.5 | 128.3 (-50.4%) | **147.2 (-43.1%)** |
|      |      | 512 | 319.9 | 203.3 (-36.4%) | **214.7 (-32.9%)** |

### Embedding throughput (tok/s) — runtime apples-to-apples

Measured on the same tokenized inputs with matching pooling (`last`) and normalization (`true`) settings across backends. Source: `benchmarks/results/embedding/ab-postfix/`.

Single-request median throughput (ax-engine-py vs mlx-lm, same session):

| Model | mlx-lm (baseline) | ax-engine-py |
|---|---:|---:|
| Qwen3-Embedding 0.6B 8-bit | 1,410.3 | 1,398.8 (≈-6%) † |
| Qwen3-Embedding 4B 4-bit | 536.6 | 444.3 (-17.2%) |
| Qwen3-Embedding 8B 4-bit DWQ | 319.8 | 280.4 (-12.3%) |

† The 0.6B model completes in ~6ms/sentence, making it sensitive to thermal variance. Run-to-run gap typically ranges from -5% to -10%.

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
