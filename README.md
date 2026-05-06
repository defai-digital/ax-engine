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

- **N-gram acceleration** reaches up to 2.4x mlx_lm decode
  throughput on high-hit benchmark rows — with no second draft model and no
  model changes
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
| `mlx_lm_delegated` | MLX text models that upstream `mlx-lm` supports before AX has a repo-owned graph | Text generation through a user-provided `mlx_lm.server`; `/v1/generate`, fake SSE over `/v1/generate/stream`, and OpenAI-compatible completion/chat text endpoints |
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
  speedups. n-gram hit rate is prompt/output-pattern dependent.

## Performance

**Apple M5 Max · 128 GB · macOS 26.4.1.** Random-token prompts (mlx_lm seed=0),
batch=1, prefill_step_size=2048, 3 timed trials + 1 warmup. AX rows below were
refreshed on 2026-05-06 from
`benchmarks/results/mlx-inference/2026-05-06-ax-rework/`. Reference
`mlx_lm`/`mlx_swift_lm` rows were reused from the matching checked-in artifacts
so this is an AX-only rework refresh. `ax engine` is the direct same-policy
comparison against `mlx_lm`; `ax engine + n-gram accel` reports observed
effective throughput, not raw model speed.

The 2026-05-06 AX rework is not a universal direct-decode win. Direct AX decode
is strongest on Qwen 3.6 4/5-bit, Qwen Coder Next, and the 512-token GLM 4.7
shape; Gemma direct decode remains below the `mlx_lm` baseline in this refresh.
N-gram acceleration is broadly effective-throughput positive, with the Qwen 3.5
512-token shape as the clear exception.

To reproduce the benchmark procedure on an Apple Silicon host, use
`scripts/reproduce-mlx-inference-benchmark.sh` with a local MLX model artifact
directory. The script records doctor output, command logs, prompt artifacts, and
raw JSON results under `benchmarks/community-results/local/` by default. See
`docs/BENCHMARKS.md` and `benchmarks/community-results/README.md` before
submitting or comparing external rows.

Gemma 4 E4B benchmark rows are pending; the model manifest and scenario
manifest are present. See `benchmarks/results/mlx-inference/2026-05-04/README.md`
for the run command.

### Decode throughput (tok/s) — generation=128 tokens, temp=0

| Model | MLX quantization | Prompt tok | mlx_lm | mlx_swift_lm | ax engine | ax engine + n-gram accel |
|---|---|---:|---:|---:|---:|---|
| Gemma 4 E2B | 4-bit · group=64 · affine | 128 | 197.5 | 192.4 (−2.6%) | 180.7 (−8.5%) | **547.5 (+177.3%)** |
|    |    | 512 | 191.9 | 179.5 (−6.5%) | 175.2 (−8.7%) | **532.3 (+177.4%)** |
| Gemma 4 E2B | 5-bit · group=64 · affine‖ | 128 | 182.9 | 174.1 (−4.8%) | 161.1 (−11.9%) | **424.2 (+131.9%)** |
|    |    | 512 | 178.1 | 167.0 (−6.2%) | 152.5 (−14.4%) | **424.0 (+138.1%)** |
| Gemma 4 E2B | 6-bit · group=64 · affine‖ | 128 | 161.3 | 153.0 (−5.1%) | 145.2 (−10.0%) | **404.1 (+150.5%)** |
|    |    | 512 | 154.2 | 147.1 (−4.6%) | 138.1 (−10.5%) | **396.5 (+157.1%)** |
| Gemma 4 E2B | 8-bit · group=64 · affine‖ | 128 | 139.4 | 134.9 (−3.2%) | 127.1 (−8.9%) | **424.5 (+204.5%)** |
|    |    | 512 | 134.5 | 130.8 (−2.8%) | 125.0 (−7.1%) | **421.2 (+213.0%)** |
| Gemma 4 26B A4B | 4-bit · group=64 · affine¶ | 128 | 118.3 | 109.4 (−7.5%) | 112.5 (−4.8%) | **250.1 (+111.5%)** |
|    |    | 512 | 113.1 | 104.7 (−7.5%) | 108.8 (−3.8%) | **199.8 (+76.6%)** |
| Gemma 4 31B | 4-bit · group=64 · affine | 128 | 26.2 | 24.8 (−5.5%) | 24.3 (−7.5%) | **60.7 (+131.6%)** |
|    |    | 512 | 24.9 | 24.7 (−0.9%) | 22.6 (−9.2%) | **57.4 (+130.3%)** |
| Qwen 3.5 9B | 4-bit · group=64 · affine | 128 | 96.5 | 93.7 (−2.9%) | 92.6 (−4.0%) | **196.2 (+103.2%) †** |
|    |    | 512 | 101.3 | 91.4 (−9.8%) | 93.0 (−8.1%) | 92.0 (−9.2%) † |
| Qwen 3.6 35B A3B | UD-MLX 4-bit · group=64 · affine§ | 128 | 107.6 | 103.6 (−3.7%) | 110.2 (+2.5%) | **253.1 (+135.3%) †** |
|    |    | 512 | 103.3 | 101.4 (−1.9%) | 109.4 (+5.9%) | **252.5 (+144.5%) †** |
| Qwen 3.6 35B A3B | MLX 5-bit · group=64 · affine§ | 128 | 116.8 | 110.2 (−5.6%) | 119.7 (+2.4%) | **257.5 (+120.5%) †** |
|    |    | 512 | 113.7 | 108.7 (−4.4%) | 117.5 (+3.3%) | **247.7 (+117.8%) †** |
| Qwen 3.6 35B A3B | MLX 6-bit · group=64 · affine§ | 128 | 102.9 | 99.1 (−3.6%) | 100.9 (−1.9%) | **234.5 (+127.9%) †** |
|    |    | 512 | 101.1 | 98.0 (−3.1%) | 98.3 (−2.7%) | **231.0 (+128.6%) †** |
| Qwen 3.6 35B A3B | MLX 8-bit · group=64 · affine§ | 128 | 93.6 | 89.3 (−4.6%) | 92.4 (−1.3%) | **233.3 (+149.2%) †** |
|    |    | 512 | 91.4 | 89.1 (−2.6%) | 91.2 (−0.3%) | **234.5 (+156.5%) †** |
| Qwen Coder Next | 4-bit · group=64 · affine‡ | 128 | 92.2 | 89.4 (−3.0%) | 92.5 (+0.4%) | **246.6 (+167.6%) †** |
|    |    | 512 | 90.4 | 89.2 (−1.3%) | 91.6 (+1.4%) | **243.6 (+169.6%) †** |
| GLM 4.7 Flash | 4-bit · group=64 · affine | 128 | 93.0 | 88.0 (−5.4%) | 91.9 (−1.2%) | **256.7 (+176.0%) †** |
|    |    | 512 | 90.4 | 84.5 (−6.6%) | 91.8 (+1.6%) | **253.0 (+179.8%) †** |

† Qwen 3.5, Qwen 3.6, Qwen Coder Next, and GLM 4.7 n-gram acceleration rows
are effective-throughput measurements from AX's n-gram acceleration policy, not
raw model decode speed. Qwen-family linear-attention rows use a rollback-safe
branch/recompute path for SSM state. Acceleration is prompt/output-pattern
dependent: in the 2026-05-06 AX-only refresh it is broadly positive, but the
Qwen 3.5 512-token shape regressed to 92.0 tok/s, or -9.2% vs the matching
`mlx_lm` row. Benchmark JSON artifacts include fixed-schema n-gram telemetry
fields; the throughput table uses median AX runner timing plus output-token
count.

‡ Qwen Coder Next uses MLX affine 4-bit globally, with 8-bit overrides for
router and shared-expert gate tensors.

§ Qwen 3.6 35B A3B includes the Unsloth UD-MLX 4-bit checkpoint plus
MLX-community 5/6/8-bit checkpoints. The 5/6-bit checkpoints are affine 5/6-bit
globally with 8-bit router and shared-expert gate overrides; the 8-bit
checkpoint is affine 8-bit throughout.

¶ Gemma 4 26B A4B is the public Gemma 4 MoE MLX model. Its checkpoint uses
affine 4-bit globally, with 8-bit overrides for dense MLP and router
projections. The `mlx_swift_lm` reference row loads this model via the
`MLXVLM` factory (required for MoE architectures) rather than the standard
Swift path; prompt tokenization is otherwise identical to all other Swift rows.

‖ Gemma 4 E2B 5/6/8-bit checkpoints use affine quantization at their
respective bit depths globally. These rows verify AX Engine's higher-bit
quantization support; the 4-bit row is the primary decode performance
reference.

### Prefill throughput (tok/s) — percentages vs mlx_lm

| Model | MLX quantization | Prompt tok | mlx_lm | mlx_swift_lm | ax engine |
|---|---|---:|---:|---:|---:|
| Gemma 4 E2B | 4-bit · group=64 · affine | 128 | 2,265.8 | 2,450.4 (+8.1%) | 3,247.8 (+43.3%) |
|    |    | 512 | 7,634.1 | 6,664.3 (−12.7%) | 7,834.1 (+2.6%) |
| Gemma 4 E2B | 5-bit · group=64 · affine‖ | 128 | 2,267.5 | 2,393.9 (+5.6%) | 3,137.6 (+38.4%) |
|    |    | 512 | 8,405.7 | 6,742.6 (−19.8%) | 7,252.8 (−13.7%) |
| Gemma 4 E2B | 6-bit · group=64 · affine‖ | 128 | 2,156.3 | 3,436.8 (+59.4%) | 3,139.6 (+45.6%) |
|    |    | 512 | 7,320.7 | 7,962.3 (+8.8%) | 7,149.9 (−2.3%) |
| Gemma 4 E2B | 8-bit · group=64 · affine‖ | 128 | 1,911.7 | 3,082.0 (+61.2%) | 3,052.9 (+59.7%) |
|    |    | 512 | 6,582.8 | 6,758.1 (+2.7%) | 7,296.4 (+10.8%) |
| Gemma 4 26B A4B | 4-bit · group=64 · affine¶ | 128 | 545.3 | 1,227.3 (+125.1%) | 1,225.9 (+124.8%) |
|    |    | 512 | 1,620.7 | 2,938.6 (+81.3%) | 2,735.1 (+68.8%) |
| Gemma 4 31B | 4-bit · group=64 · affine | 128 | 336.5 | 641.6 (+90.7%) | 518.5 (+54.1%) |
|    |    | 512 | 563.5 | 760.6 (+35.0%) | 670.0 (+18.9%) |
| Qwen 3.5 9B | 4-bit · group=64 · affine | 128 | 1,133.3 | 2,101.1 (+85.4%) | 1,900.4 (+67.7%) |
|    |    | 512 | 2,245.7 | 3,165.8 (+41.0%) | 2,694.3 (+20.0%) |
| Qwen 3.6 35B A3B | UD-MLX 4-bit · group=64 · affine§ | 128 | 531.7 | 963.2 (+81.1%) | 1,021.7 (+92.1%) |
|    |    | 512 | 1,594.2 | 2,546.5 (+59.7%) | 2,557.5 (+60.4%) |
| Qwen 3.6 35B A3B | MLX 5-bit · group=64 · affine§ | 128 | 474.4 | 861.8 (+81.7%) | 973.6 (+105.2%) |
|    |    | 512 | 1,484.5 | 2,416.7 (+62.8%) | 2,448.7 (+65.0%) |
| Qwen 3.6 35B A3B | MLX 6-bit · group=64 · affine§ | 128 | 420.0 | 762.4 (+81.5%) | 879.5 (+109.4%) |
|    |    | 512 | 1,377.9 | 2,350.6 (+70.6%) | 2,333.8 (+69.4%) |
| Qwen 3.6 35B A3B | MLX 8-bit · group=64 · affine§ | 128 | 393.1 | 617.7 (+57.1%) | 896.5 (+128.0%) |
|    |    | 512 | 1,202.2 | 2,305.2 (+91.7%) | 2,295.1 (+90.9%) |
| Qwen Coder Next | 4-bit · group=64 · affine‡ | 128 | 267.1 | 384.9 (+44.1%) | 834.6 (+212.4%) |
|    |    | 512 | 815.4 | 1,417.0 (+73.8%) | 2,485.6 (+204.8%) |
| GLM 4.7 Flash | 4-bit · group=64 · affine | 128 | 502.9 | 1,045.0 (+107.8%) | 827.0 (+64.4%) |
|    |    | 512 | 1,584.7 | 2,588.8 (+63.4%) | 2,270.6 (+43.3%) |

### Workload Contracts

The throughput tables above are MLX model-inference comparisons. Workload
contracts are a separate `ax-engine-bench` surface for checked-in scenario,
replay, matrix, route, correctness, determinism, and regression gates.

Canonical manifests live under `benchmarks/manifests/{scenario,replay,matrix}`.
See `docs/BENCHMARKS.md` for the evidence split, methodology, and
prompt-provenance requirements.

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
and `seed`, and exposes blocking plus fake-SSE text surfaces through AX. Tool
calls and visual/multimodal inputs are not yet AX compatibility contracts.

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
