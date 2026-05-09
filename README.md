# AX Engine

[![Preview Surfaces](https://github.com/defai-digital/ax-engine/actions/workflows/python-preview.yml/badge.svg?branch=main)](https://github.com/defai-digital/ax-engine/actions/workflows/python-preview.yml)
[![Coverage Report](https://github.com/defai-digital/ax-engine/actions/workflows/coverage.yml/badge.svg?branch=main)](https://github.com/defai-digital/ax-engine/actions/workflows/coverage.yml)

AX Engine is a Mac-first LLM inference runtime, local server, SDK layer, and
benchmark toolkit for Apple Silicon.

It is not "AX MLX" as a product. MLX is the primary Apple Silicon execution
backend for supported model families, while AX Engine also exposes explicit
compatibility routes for upstream `mlx-lm` and `llama.cpp` so users can stay on
one AX surface while model coverage grows.

> Requires macOS on Apple Silicon M4 or newer and Rust 1.85+.

## 30-Second Setup

Install the released command-line tools and verify the local runtime contract:

```bash
brew install defai-digital/ax-engine/ax-engine
```

```bash
ax-engine-bench doctor
ax-engine-server --help
```

This verifies the released AX Engine tools. Running inference requires choosing
a runtime path below: repo-owned MLX, delegated `mlx-lm`, or delegated
`llama.cpp`.

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
  async-free, no framework dependencies.
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

- **GatedDelta prefill**: Qwen3.5 and Qwen3-Next linear-attention layers use a
  custom Metal kernel with a serial time loop. For long prompts (512+ tokens),
  this puts AX prefill behind mlx-swift-lm on those models; decode throughput is
  unaffected.
- **Raw HuggingFace weights**: ax-engine loads MLX community (pre-sanitized)
  weights. Raw HF checkpoints for hybrid models need norm-weight `+1.0` and
  conv1d `moveaxis(2,1)` transformations that the converter does not apply.
- **N-gram acceleration rows**: effective-throughput measurements, not raw model-kernel
  speedups. n-gram hit rate is prompt/output-pattern dependent. Coding-like
  workloads with repeated local structure are the intended high-value case;
  random, high-entropy, very short, or deliberately diverse outputs may see
  little benefit, and the runtime can back off toward the direct path.
- **TurboQuant KV compression**: experimental and off by default. The
  `turboquant-shadow` and `turboquant-fused-experimental` modes are evidence
  and route-telemetry surfaces, not production support claims. Public support
  requires a passing long-context, model-level quality artifact and decode
  throughput gate; current Gemma 4 E2B local evidence reaches the fused
  compressed route with zero fallback but does not pass the promotion
  performance gate.

## Performance ([methodology](docs/PERFORMANCE.md))

AX Engine columns were refreshed on 2026-05-08 from
`benchmarks/results/mlx-inference/2026-05-08-ngram-fix/`. This run applied two
fixes relative to v4.4.2: (1) the direct decode double-buffer pipeline no longer
submits extra GPU command buffers for KV backing arrays — they are already in the
output token's computation graph via SDPA, so adding them explicitly created
per-layer overhead (~85 µs each); (2) dense-model n-gram draft length now always
uses `MAX_DRAFT_LEN=6` as the cap and lets the confidence gate prune chains
naturally, rather than pre-capping at `DEFAULT_DRAFT_LEN=4` for mid-range
posterior values. Most `mlx_lm` and `mlx_swift_lm` columns remain matched
reference rows reused by the harness.

Versus the v4.4.1 README artifact baseline, this run shows uniform improvements
across all 14 models on both paths. Direct decode is up +1–11% on most rows
(Gemma E4B +8–11%, Qwen 3.5 +10%); GLM and Gemma 26B p128 are within ±3%
noise. N-gram decode is up +1–5% on most rows, with larger gains on Gemma E4B
(+7–10%), Qwen 3.5 (+9–11%), and Qwen Coder Next p128 (+73%, from 137 to 237
tok/s — the draft-length fix restores full chain speculation during the warm-up
window). No row regresses beyond noise.

The direct AX column is a same-policy diagnostic baseline with n-gram
acceleration disabled, while the n-gram column is the default AX decode policy
and the row to use for user-facing throughput expectations.
The prefill table uses the direct AX row because n-gram acceleration is a
decode policy, not a prefill optimization.
For Qwen 3.5 at 512 prompt tokens, the default n-gram row is slightly above the
`mlx_lm` reference but below the direct path because it records zero n-gram
draft attempts in this run and falls back to the direct pipeline after the
no-draft probe window.

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

### Decode throughput (tok/s) — generation=128 tokens, temp=0

| Model | MLX quantization | Prompt tok | mlx_lm | mlx_swift_lm | ax direct baseline | ax default n-gram |
|---|---|---:|---:|---:|---:|---|
| Gemma 4 E2B | 4-bit · group=64 · affine | 128 | 197.5 | 192.4 (-2.6%) | 187.6 (-5.0%) | **584.0 (+195.7%)** |
|    |    | 512 | 191.9 | 179.5 (-6.5%) | 183.0 (-4.6%) | **577.0 (+200.7%)** |
| Gemma 4 E2B | 5-bit · group=64 · affine | 128 | 182.9 | 174.1 (-4.8%) | 172.5 (-5.7%) | **455.3 (+148.9%)** |
|    |    | 512 | 178.1 | 167.0 (-6.2%) | 166.4 (-6.6%) | **454.4 (+155.1%)** |
| Gemma 4 E2B | 6-bit · group=64 · affine | 128 | 161.3 | 153.0 (-5.1%) | 155.6 (-3.5%) | **426.9 (+164.7%)** |
|    |    | 512 | 154.2 | 147.1 (-4.6%) | 150.3 (-2.5%) | **423.1 (+174.4%)** |
| Gemma 4 E2B | 8-bit · group=64 · affine | 128 | 139.4 | 134.9 (-3.2%) | 139.4 (0.0%) | **460.4 (+230.2%)** |
|    |    | 512 | 134.5 | 130.8 (-2.8%) | 135.3 (+0.6%) | **454.5 (+237.9%)** |
| Gemma 4 E4B | 4-bit · group=64 · affine | 128 | 121.3 | 116.4 (-4.0%) | 122.9 (+1.3%) | **348.5 (+187.3%)** |
|    |    | 512 | 120.0 | 117.9 (-1.7%) | 118.9 (-0.9%) | **345.4 (+187.8%)** |
| Gemma 4 26B A4B | 4-bit · group=64 · affine | 128 | 118.3 | 109.4 (-7.5%) | 115.9 (-2.0%) | **273.4 (+131.1%)** |
|    |    | 512 | 113.1 | 104.7 (-7.5%) | 117.6 (+4.0%) | **233.6 (+106.5%)** |
| Gemma 4 31B | 4-bit · group=64 · affine | 128 | 26.2 | 24.8 (-5.5%) | 28.0 (+6.9%) | **65.3 (+149.2%)** |
|    |    | 512 | 24.9 | 24.7 (-0.9%) | 27.4 (+10.0%) | **63.6 (+155.4%)** |
| Qwen 3.5 9B | 4-bit · group=64 · affine | 128 | 95.2 | 93.7 (-1.6%) | 103.4 (+8.6%) | **210.5 (+121.1%)** |
|    |    | 512 | 93.4 | 91.4 (-2.2%) | 103.2 (+10.5%) | 101.2 (+8.4%) |
| Qwen 3.6 35B A3B | UD-MLX 4-bit · group=64 · affine | 128 | 107.6 | 103.6 (-3.7%) | 123.4 (+14.7%) | **285.6 (+165.4%)** |
|    |    | 512 | 103.3 | 101.4 (-1.9%) | 122.4 (+18.5%) | **280.7 (+171.7%)** |
| Qwen 3.6 35B A3B | MLX 5-bit · group=64 · affine | 128 | 116.8 | 110.2 (-5.6%) | 137.2 (+17.5%) | **284.1 (+143.2%)** |
|    |    | 512 | 113.7 | 108.7 (-4.4%) | 135.6 (+19.3%) | **280.3 (+146.6%)** |
| Qwen 3.6 35B A3B | MLX 6-bit · group=64 · affine | 128 | 102.9 | 99.1 (-3.6%) | 121.2 (+17.8%) | **259.9 (+152.6%)** |
|    |    | 512 | 101.1 | 98.0 (-3.1%) | 120.8 (+19.5%) | **257.5 (+154.7%)** |
| Qwen 3.6 35B A3B | MLX 8-bit · group=64 · affine | 128 | 93.6 | 89.3 (-4.6%) | 109.1 (+16.6%) | **263.6 (+181.6%)** |
|    |    | 512 | 91.4 | 89.1 (-2.6%) | 108.3 (+18.5%) | **259.3 (+183.7%)** |
| Qwen Coder Next | 4-bit · group=64 · affine | 128 | 92.2 | 89.4 (-3.0%) | 103.7 (+12.5%) | **236.9 (+157.0%)** |
|    |    | 512 | 90.4 | 89.2 (-1.3%) | 103.0 (+13.9%) | **261.3 (+189.1%)** |
| GLM 4.7 Flash | 4-bit · group=64 · affine | 128 | 93.0 | 88.0 (-5.4%) | 102.6 (+10.3%) | **275.6 (+196.3%)** |
|    |    | 512 | 90.4 | 84.5 (-6.6%) | 102.7 (+13.6%) | **270.6 (+199.3%)** |

### Prefill throughput (tok/s) — percentages vs mlx_lm

| Model | MLX quantization | Prompt tok | mlx_lm | mlx_swift_lm | ax engine |
|---|---|---:|---:|---:|---:|
| Gemma 4 E2B | 4-bit · group=64 · affine | 128 | 2,265.8 | 2,450.4 (+8.1%) | 3,412.5 (+50.6%) |
|    |    | 512 | 7,634.1 | 6,664.3 (-12.7%) | 7,723.5 (+1.2%) |
| Gemma 4 E2B | 5-bit · group=64 · affine | 128 | 2,267.5 | 2,393.9 (+5.6%) | 3,315.9 (+46.2%) |
|    |    | 512 | 8,405.7 | 6,742.6 (-19.8%) | 7,483.1 (-11.0%) |
| Gemma 4 E2B | 6-bit · group=64 · affine | 128 | 2,156.3 | 3,436.8 (+59.4%) | 3,226.1 (+49.6%) |
|    |    | 512 | 7,320.7 | 7,962.3 (+8.8%) | 7,342.0 (+0.3%) |
| Gemma 4 E2B | 8-bit · group=64 · affine | 128 | 1,911.7 | 3,082.0 (+61.2%) | 3,143.6 (+64.4%) |
|    |    | 512 | 6,582.8 | 6,758.1 (+2.7%) | 7,302.2 (+10.9%) |
| Gemma 4 E4B | 4-bit · group=64 · affine | 128 | 1,586.0 | 2,006.2 (+26.5%) | 2,544.1 (+60.4%) |
|    |    | 512 | 4,432.6 | 4,362.5 (-1.6%) | 4,278.4 (-3.5%) |
| Gemma 4 26B A4B | 4-bit · group=64 · affine | 128 | 545.3 | 1,227.3 (+125.1%) | 1,186.2 (+117.5%) |
|    |    | 512 | 1,620.7 | 2,938.6 (+81.3%) | 2,955.9 (+82.4%) |
| Gemma 4 31B | 4-bit · group=64 · affine | 128 | 336.5 | 641.6 (+90.7%) | 565.2 (+68.0%) |
|    |    | 512 | 563.5 | 760.6 (+35.0%) | 738.6 (+31.1%) |
| Qwen 3.5 9B | 4-bit · group=64 · affine | 128 | 1,131.5 | 2,101.1 (+85.7%) | 2,045.9 (+80.8%) |
|    |    | 512 | 2,285.3 | 3,165.8 (+38.5%) | 2,884.4 (+26.2%) |
| Qwen 3.6 35B A3B | UD-MLX 4-bit · group=64 · affine | 128 | 531.7 | 963.2 (+81.1%) | 1,076.7 (+102.5%) |
|    |    | 512 | 1,594.2 | 2,546.5 (+59.7%) | 2,701.5 (+69.5%) |
| Qwen 3.6 35B A3B | MLX 5-bit · group=64 · affine | 128 | 474.4 | 861.8 (+81.7%) | 1,020.8 (+115.2%) |
|    |    | 512 | 1,484.5 | 2,416.7 (+62.8%) | 2,602.4 (+75.3%) |
| Qwen 3.6 35B A3B | MLX 6-bit · group=64 · affine | 128 | 420.0 | 762.4 (+81.5%) | 973.0 (+131.7%) |
|    |    | 512 | 1,377.9 | 2,350.6 (+70.6%) | 2,481.6 (+80.1%) |
| Qwen 3.6 35B A3B | MLX 8-bit · group=64 · affine | 128 | 393.1 | 617.7 (+57.1%) | 992.3 (+152.4%) |
|    |    | 512 | 1,202.2 | 2,305.2 (+91.7%) | 2,471.7 (+105.6%) |
| Qwen Coder Next | 4-bit · group=64 · affine | 128 | 267.1 | 384.9 (+44.1%) | 893.6 (+234.5%) |
|    |    | 512 | 815.4 | 1,417.0 (+73.8%) | 2,843.6 (+248.7%) |
| GLM 4.7 Flash | 4-bit · group=64 · affine | 128 | 502.9 | 1,045.0 (+107.8%) | 863.2 (+71.6%) |
|    |    | 512 | 1,584.7 | 2,588.8 (+63.4%) | 2,385.9 (+50.6%) |

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
- the Homebrew `mlx-c` runtime dependency required by the released binaries

Check the installed tools:

```bash
ax-engine-server --help
ax-engine-bench doctor
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

The commands below use source-build paths. If you installed with Homebrew, use
`ax-engine-server` and `ax-engine-bench` directly instead of
`./target/release/...`.

```bash
# HTTP inference server (repo-owned MLX runtime)
./target/release/ax-engine-server \
  --mlx \
  --mlx-model-artifacts-dir /path/to/local/mlx-model \
  --port 8080

# Python bindings (after maturin develop)
python3 - <<'EOF'
import ax_engine
with ax_engine.Session(model_id='gemma4', mlx=True,
        mlx_model_artifacts_dir='/path/to/local/mlx-model') as s:
    result = s.generate([1, 2, 3], max_output_tokens=32)
    print(result.output_tokens)
EOF
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
bash scripts/check-server-preview.sh
bash scripts/check-python-preview.sh
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
`benchmarks/manifests/`.

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
