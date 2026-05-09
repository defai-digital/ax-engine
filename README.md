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

Install the released command-line tools and open the local TUI cockpit:

```bash
brew install defai-digital/ax-engine/ax-engine
ax-engine-manager --check
ax-engine-manager
```

Then connect it to a model and server:

```bash
# Download an mlx-community model and generate its manifest in one step
python scripts/download_model.py mlx-community/Qwen3-4B-4bit
MODEL_DIR="$HOME/.cache/ax-engine/models/mlx-community--Qwen3-4B-4bit"

# Start the server
ax-engine-server --mlx --mlx-model-artifacts-dir "$MODEL_DIR" --port 8080

# In another terminal, open the TUI cockpit with live server metadata
ax-engine-manager --model-dir "$MODEL_DIR" --server-url http://127.0.0.1:8080
```

Or from Python (after `maturin develop` or `pip install ax-engine`):

```python
from ax_engine import download_model, Session
path = download_model("mlx-community/Qwen3-4B-4bit")
with Session(mlx=True, mlx_model_artifacts_dir=str(path)) as s:
    print(s.generate([1, 2, 3], max_output_tokens=8).output_tokens)
```

`download_model()` downloads weights and auto-runs `ax-engine-bench generate-manifest`.
See [Getting a Model](#getting-a-model) for all paths including raw HF checkpoints,
and see [AX Engine Manager](docs/MANAGER.md) for the full TUI workflow.

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

AX Engine columns were refreshed on 2026-05-09 from
`benchmarks/results/mlx-inference/2026-05-09-post-v4.5.0/`. This run covers 25
commits since the q-slice-fix run, the two most performance-relevant being native
top-p/top-k sampling (`d3a8615`) and a TurboQuant fused decode hot-path improvement
(`685ca98`). The `mlx_lm` and `mlx_swift_lm` columns are matched reference rows
reused from the previous run.

**Prefill** — AX engine prefill is faster than mlx_lm on most models at short
prompts (+40–170% at 128 tokens), driven by chunked KV allocation and a tuned
pipeline. At 512-token prompts the gap narrows; several Gemma quantizations
(5-bit, 6-bit, E4B 4-bit) are 5–10% behind mlx_lm.

**Decode** — Direct decode (n-gram disabled): Qwen 3.6 35B variants are +3–8%
above mlx_lm; Gemma 4-bit models and most others are within ±4%; Gemma 5–8-bit
models are 5–15% below mlx_lm, a regression attributable to per-step sampling
overhead introduced in `d3a8615`. With n-gram acceleration (the default),
effective throughput reaches up to 3.1× mlx_lm; the speculator backs off on
high-entropy outputs.

**TTFT** — Qwen 3.6 and Coder Next TTFT leads are maintained: −37–63% vs
mlx_lm across all prompt sizes. Gemma E2B at 128 tokens: −29–39%. Several
512-token rows (E2B 5-bit, E2B 6-bit, E4B) are 7–12% above mlx_lm due to
prefill parity or regression in this run. Source:
`benchmarks/results/mlx-inference/2026-05-09-post-v4.5.0/`. mlx_lm TTFT is
derived from reported prefill throughput; ax engine TTFT is measured directly
from per-step runner timing.

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
| Gemma 4 E2B | 4-bit · group=64 · affine | 128 | 2,265.8 | 2,450.4 (+8.1%) | 3,413.2 (+50.7%) |
|    |    | 512 | 7,634.1 | 6,664.3 (-12.7%) | 7,744.1 (+1.4%) |
| Gemma 4 E2B | 5-bit · group=64 · affine | 128 | 2,267.5 | 2,393.9 (+5.6%) | 3,306.6 (+45.8%) |
|    |    | 512 | 8,405.7 | 6,742.6 (-19.8%) | 7,532.0 (-10.4%) |
| Gemma 4 E2B | 6-bit · group=64 · affine | 128 | 2,156.3 | 3,436.8 (+59.4%) | 3,058.0 (+41.8%) |
|    |    | 512 | 7,320.7 | 7,962.3 (+8.8%) | 6,833.5 (-6.7%) |
| Gemma 4 E2B | 8-bit · group=64 · affine | 128 | 1,911.7 | 3,082.0 (+61.2%) | 3,113.2 (+62.9%) |
|    |    | 512 | 6,582.8 | 6,758.1 (+2.7%) | 7,201.9 (+9.4%) |
| Gemma 4 E4B | 4-bit · group=64 · affine | 128 | 1,586.0 | 2,006.2 (+26.5%) | 2,339.7 (+47.5%) |
|    |    | 512 | 4,432.6 | 4,362.5 (-1.6%) | 4,101.3 (-7.5%) |
| Gemma 4 26B A4B | 4-bit · group=64 · affine | 128 | 545.3 | 1,227.3 (+125.1%) | 1,127.2 (+106.7%) |
|    |    | 512 | 1,620.7 | 2,938.6 (+81.3%) | 2,887.7 (+78.2%) |
| Gemma 4 31B | 4-bit · group=64 · affine | 128 | 336.5 | 641.6 (+90.7%) | 510.4 (+51.7%) |
|    |    | 512 | 563.5 | 760.6 (+35.0%) | 662.9 (+17.7%) |
| Qwen 3.5 9B | 4-bit · group=64 · affine | 128 | 1,131.5 | 2,101.1 (+85.7%) | 1,924.5 (+70.1%) |
|    |    | 512 | 2,285.3 | 3,165.8 (+38.5%) | 2,711.2 (+18.6%) |
| Qwen 3.6 35B A3B | UD-MLX 4-bit · group=64 · affine | 128 | 531.7 | 963.2 (+81.1%) | 981.5 (+84.6%) |
|    |    | 512 | 1,594.2 | 2,546.5 (+59.7%) | 2,517.2 (+57.9%) |
| Qwen 3.6 35B A3B | MLX 5-bit · group=64 · affine | 128 | 474.4 | 861.8 (+81.7%) | 960.8 (+102.5%) |
|    |    | 512 | 1,484.5 | 2,416.7 (+62.8%) | 2,434.7 (+64.0%) |
| Qwen 3.6 35B A3B | MLX 6-bit · group=64 · affine | 128 | 420.0 | 762.4 (+81.5%) | 908.9 (+116.4%) |
|    |    | 512 | 1,377.9 | 2,350.6 (+70.6%) | 2,328.1 (+69.0%) |
| Qwen 3.6 35B A3B | MLX 8-bit · group=64 · affine | 128 | 393.1 | 617.7 (+57.1%) | 923.2 (+134.8%) |
|    |    | 512 | 1,202.2 | 2,305.2 (+91.7%) | 2,275.8 (+89.3%) |
| Qwen Coder Next | 4-bit · group=64 · affine | 128 | 267.1 | 384.9 (+44.1%) | 714.4 (+167.4%) |
|    |    | 512 | 815.4 | 1,417.0 (+73.8%) | 1,665.1 (+104.2%) |
| GLM 4.7 Flash | 4-bit · group=64 · affine | 128 | 502.9 | 1,045.0 (+107.8%) | 819.2 (+62.9%) |
|    |    | 512 | 1,584.7 | 2,588.8 (+63.4%) | 2,230.9 (+40.8%) |

### Decode throughput (tok/s) — generation=128 tokens, temp=0

The direct AX column is a same-policy diagnostic baseline with n-gram acceleration
disabled. The n-gram column is the default AX decode policy and the row to use for
user-facing throughput expectations. For Qwen 3.5 at 512 prompt tokens, the default
n-gram row falls back to the direct pipeline after a no-draft probe window.

| Model | MLX quantization | Prompt tok | mlx_lm | mlx_swift_lm | ax direct baseline | ax default n-gram |
|---|---|---:|---:|---:|---:|---|
| Gemma 4 E2B | 4-bit · group=64 · affine | 128 | 197.5 | 192.4 (-2.6%) | 192.1 (-2.7%) | **581.5 (+194.5%)** |
|    |    | 512 | 191.9 | 179.5 (-6.5%) | 184.9 (-3.6%) | **575.1 (+199.6%)** |
| Gemma 4 E2B | 5-bit · group=64 · affine | 128 | 182.9 | 174.1 (-4.8%) | 169.7 (-7.2%) | **457.4 (+150.0%)** |
|    |    | 512 | 178.1 | 167.0 (-6.2%) | 164.6 (-7.6%) | **454.2 (+155.0%)** |
| Gemma 4 E2B | 6-bit · group=64 · affine | 128 | 161.3 | 153.0 (-5.1%) | 137.2 (-14.9%) | **377.9 (+134.3%)** |
|    |    | 512 | 154.2 | 147.1 (-4.6%) | 137.6 (-10.8%) | **403.3 (+161.5%)** |
| Gemma 4 E2B | 8-bit · group=64 · affine | 128 | 139.4 | 134.9 (-3.2%) | 125.3 (-10.1%) | **412.5 (+195.9%)** |
|    |    | 512 | 134.5 | 130.8 (-2.8%) | 128.2 (-4.7%) | **416.6 (+209.6%)** |
| Gemma 4 E4B | 4-bit · group=64 · affine | 128 | 121.3 | 116.4 (-4.0%) | 109.9 (-9.4%) | **332.2 (+173.9%)** |
|    |    | 512 | 120.0 | 117.9 (-1.7%) | 109.5 (-8.7%) | **340.6 (+184.0%)** |
| Gemma 4 26B A4B | 4-bit · group=64 · affine | 128 | 118.3 | 109.4 (-7.5%) | 115.6 (-2.2%) | **259.2 (+119.2%)** |
|    |    | 512 | 113.1 | 104.7 (-7.5%) | 111.0 (-1.8%) | **220.0 (+94.5%)** |
| Gemma 4 31B | 4-bit · group=64 · affine | 128 | 26.2 | 24.8 (-5.5%) | 25.2 (-3.8%) | **57.3 (+118.4%)** |
|    |    | 512 | 24.9 | 24.7 (-0.9%) | 23.8 (-4.5%) | **55.5 (+122.7%)** |
| Qwen 3.5 9B | 4-bit · group=64 · affine | 128 | 95.2 | 93.7 (-1.6%) | 91.9 (-3.5%) | **186.4 (+95.8%)** |
|    |    | 512 | 93.4 | 91.4 (-2.2%) | 89.9 (-3.8%) | 86.3 (-7.6%) |
| Qwen 3.6 35B A3B | UD-MLX 4-bit · group=64 · affine | 128 | 107.6 | 103.6 (-3.7%) | 104.3 (-3.1%) | **250.1 (+132.4%)** |
|    |    | 512 | 103.3 | 101.4 (-1.9%) | 107.1 (+3.7%) | **254.6 (+146.5%)** |
| Qwen 3.6 35B A3B | MLX 5-bit · group=64 · affine | 128 | 116.8 | 110.2 (-5.6%) | 124.1 (+6.3%) | **261.6 (+123.9%)** |
|    |    | 512 | 113.7 | 108.7 (-4.4%) | 122.6 (+7.8%) | **256.1 (+125.2%)** |
| Qwen 3.6 35B A3B | MLX 6-bit · group=64 · affine | 128 | 102.9 | 99.1 (-3.6%) | 106.1 (+3.1%) | **259.6 (+152.4%)** |
|    |    | 512 | 101.1 | 98.0 (-3.1%) | 106.0 (+4.9%) | **256.5 (+153.8%)** |
| Qwen 3.6 35B A3B | MLX 8-bit · group=64 · affine | 128 | 93.6 | 89.3 (-4.6%) | 98.0 (+4.7%) | **227.5 (+143.1%)** |
|    |    | 512 | 91.4 | 89.1 (-2.6%) | 97.6 (+6.8%) | **225.4 (+146.5%)** |
| Qwen Coder Next | 4-bit · group=64 · affine | 128 | 92.2 | 89.4 (-3.0%) | 89.3 (-3.1%) | **223.2 (+142.2%)** |
|    |    | 512 | 90.4 | 89.2 (-1.3%) | 89.2 (-1.3%) | **220.8 (+144.3%)** |
| GLM 4.7 Flash | 4-bit · group=64 · affine | 128 | 93.0 | 88.0 (-5.4%) | 91.0 (-2.1%) | **250.5 (+169.3%)** |
|    |    | 512 | 90.4 | 84.5 (-6.6%) | 88.3 (-2.3%) | **243.0 (+168.8%)** |

### Time to first token (ms) — generation=128 tokens, temp=0

Lower is better. mlx_lm and mlx_swift_lm values are derived from reported prefill
throughput (`prompt_tokens / prefill_tok_s × 1000 ms`); ax engine values are directly
measured from per-step runner timing in the SSE event stream. Source:
`benchmarks/results/mlx-inference/2026-05-09-post-v4.5.0/`.

| Model | MLX quantization | Prompt tok | mlx_lm | mlx_swift_lm | ax engine |
|---|---|---:|---:|---:|---:|
| Gemma 4 E2B | 4-bit · group=64 · affine | 128 | 56.5 | 52.2 (-7.5%) | **37.5 (-33.6%)** |
|    |    | 512 | 67.1 | 76.8 (+14.6%) | **66.1 (-1.4%)** |
| Gemma 4 E2B | 5-bit · group=64 · affine | 128 | 56.4 | 53.5 (-5.3%) | **38.7 (-31.4%)** |
|    |    | 512 | 60.9 | 75.9 (+24.7%) | 68.0 (+11.6%) |
| Gemma 4 E2B | 6-bit · group=64 · affine | 128 | 59.4 | 37.2 (-37.3%) | **41.9 (-29.5%)** |
|    |    | 512 | 69.9 | 64.3 (-8.1%) | 74.9 (+7.1%) |
| Gemma 4 E2B | 8-bit · group=64 · affine | 128 | 67.0 | 41.5 (-38.0%) | **41.1 (-38.6%)** |
|    |    | 512 | 77.8 | 75.8 (-2.6%) | **71.1 (-8.6%)** |
| Gemma 4 E4B | 4-bit · group=64 · affine | 128 | 80.7 | 63.8 (-20.9%) | **54.7 (-32.2%)** |
|    |    | 512 | 115.5 | 117.4 (+1.6%) | 124.8 (+8.1%) |
| Gemma 4 26B A4B | 4-bit · group=64 · affine | 128 | 234.7 | 104.3 (-55.6%) | **113.6 (-51.6%)** |
|    |    | 512 | 315.9 | 174.2 (-44.8%) | **177.3 (-43.9%)** |
| Gemma 4 31B | 4-bit · group=64 · affine | 128 | 380.4 | 199.5 (-47.6%) | **250.8 (-34.1%)** |
|    |    | 512 | 908.7 | 673.1 (-25.9%) | **772.3 (-15.0%)** |
| Qwen 3.5 9B | 4-bit · group=64 · affine | 128 | 113.1 | 60.9 (-46.1%) | **66.5 (-41.2%)** |
|    |    | 512 | 224.0 | 161.7 (-27.8%) | **188.8 (-15.7%)** |
| Qwen 3.6 35B A3B | UD-MLX 4-bit · group=64 · affine | 128 | 240.7 | 132.9 (-44.8%) | **130.4 (-45.8%)** |
|    |    | 512 | 321.2 | 201.1 (-37.4%) | **203.4 (-36.7%)** |
| Qwen 3.6 35B A3B | MLX 5-bit · group=64 · affine | 128 | 269.8 | 148.5 (-45.0%) | **133.2 (-50.6%)** |
|    |    | 512 | 344.9 | 211.9 (-38.6%) | **210.3 (-39.0%)** |
| Qwen 3.6 35B A3B | MLX 6-bit · group=64 · affine | 128 | 304.8 | 167.9 (-44.9%) | **140.8 (-53.8%)** |
|    |    | 512 | 371.6 | 217.8 (-41.4%) | **219.9 (-40.8%)** |
| Qwen 3.6 35B A3B | MLX 8-bit · group=64 · affine | 128 | 325.6 | 207.2 (-36.4%) | **138.7 (-57.4%)** |
|    |    | 512 | 425.9 | 222.1 (-47.8%) | **225.0 (-47.2%)** |
| Qwen Coder Next | 4-bit · group=64 · affine | 128 | 479.2 | 332.6 (-30.6%) | **179.2 (-62.6%)** |
|    |    | 512 | 627.9 | 361.3 (-42.5%) | **307.5 (-51.0%)** |
| GLM 4.7 Flash | 4-bit · group=64 · affine | 128 | 254.5 | 122.5 (-51.9%) | **156.2 (-38.6%)** |
|    |    | 512 | 323.1 | 197.8 (-38.8%) | **229.5 (-29.0%)** |

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
- `ax-engine-manager`: Ratatui local manager for readiness, server metadata,
  benchmark artifacts, guarded job plans, and redacted support bundles
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
4. open `ax-engine-manager` to inspect readiness, server metadata, benchmark
   artifacts, guarded job plans, and redacted support bundles.

For a complete manager walkthrough, see [docs/MANAGER.md](docs/MANAGER.md).

The commands below use source-build paths. If you installed with Homebrew, use
`ax-engine-server`, `ax-engine-bench`, and `ax-engine-manager` directly instead
of `./target/release/...`.

```bash
# Download a model and generate its manifest
python scripts/download_model.py mlx-community/Qwen3-4B-4bit
# prints the local path when ready, e.g. ~/.cache/ax-engine/models/mlx-community--Qwen3-4B-4bit
MODEL_DIR="$HOME/.cache/ax-engine/models/mlx-community--Qwen3-4B-4bit"

# Check readiness without entering terminal raw mode
./target/release/ax-engine-manager --check --model-dir "$MODEL_DIR"

# HTTP inference server (repo-owned MLX runtime)
./target/release/ax-engine-server \
  --mlx \
  --mlx-model-artifacts-dir "$MODEL_DIR" \
  --port 8080

# Local Ratatui cockpit
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

# For automation and future TUI integration, emit a parseable summary
python scripts/download_model.py mlx-community/Qwen3-4B-4bit --json

# Python SDK
from ax_engine import download_model
path = download_model("mlx-community/Qwen3-4B-4bit")
```

If you already have `mlx_lm` installed, its download also lands in the standard HF
cache that ax-engine can auto-discover:

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
[Benchmarking](docs/BENCH-DESIGN.md)

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
