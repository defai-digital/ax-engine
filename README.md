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

Recent community-model checks that lack an AX `model-manifest.json` are tracked
as reference-only evidence, not repo-owned AX support. On 2026-05-06,
`mlx-community/GLM-4.7-Flash-4bit` benchmarked successfully through
`mlx_lm.benchmark`; `mlx-community/DeepSeek-V4-Flash-2bit-DQ` downloaded but
failed closed because upstream `mlx-lm` did not support `model_type=deepseek_v4`
in this environment. See
`benchmarks/results/mlx-inference/2026-05-06/README.md` for commands and
artifacts. Before promoting either architecture, run
`scripts/probe_mlx_model_support.py --model-dir <model-dir>`: GLM currently
classifies as an implementation candidate because Apple `mlx-lm` and Apple
`mlx-swift-lm` both expose GLM4MoELite references. AX can now map GLM tensors
into a draft `model-manifest.json`, but that manifest is intentionally marked
`runtime_status.ready=false` until the GLM MLA attention, router, and latent-KV
cache paths exist. DeepSeek V4 remains fail-closed because the available
SwiftLM port is partial and drops checkpoint features that affect the forward
contract.

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
batch=1, prefill_step_size=2048, 3 timed trials + 1 warmup. All rows
below were refreshed on 2026-05-05 from
`benchmarks/results/mlx-inference/2026-05-05-rerun/`. `ax engine` is the direct
same-policy comparison against `mlx_lm`; `ax engine + n-gram accel` reports
observed effective throughput, not raw model speed.

To reproduce the benchmark procedure on an Apple Silicon host, use
`scripts/reproduce-mlx-inference-benchmark.sh` with a local MLX model artifact
directory. The script records doctor output, command logs, prompt artifacts, and
raw JSON results under `benchmarks/community-results/local/` by default. See
`docs/BENCHMARKS.md` and `benchmarks/community-results/README.md` before
submitting or comparing external rows.

Gemma 4 E4B benchmark rows are pending; the model manifest and scenario
manifest are present. See `benchmarks/results/mlx-inference/2026-05-04/README.md`
for the run command.

### Reference-only MLX checks

These rows use upstream `mlx_lm.benchmark` only. They show whether a community
MLX model can run today through the reference stack; they are not AX Engine
repo-owned runtime rows and should not be compared with `ax engine` columns
until a model manifest, graph implementation, server smoke, and AX benchmark
artifact exist.

| Model | Repo revision | Status | Prompt tok | mlx_lm prefill tok/s | mlx_lm decode tok/s | Peak memory |
|---|---|---|---:|---:|---:|---:|
| mlx-community/GLM-4.7-Flash-4bit | `1454cffb1a21737e162f508e5bc70be9def89276` | Reference benchmark passed; draft manifest candidate | 128 | 487.5 | 89.7 | 17.063 GB |
|    |    |    | 512 | 1,517.5 | 85.5 | 17.495 GB |
| mlx-community/DeepSeek-V4-Flash-2bit-DQ | `722bf559b7de93575b2320973cf2002e05bfe6c9` | Downloaded; fail-closed partial reference; benchmark blocked by `mlx_lm`: `Model type deepseek_v4 not supported` | 128 / 512 | - | - | - |

### Decode throughput (tok/s) — generation=128 tokens, temp=0

| Model | MLX quantization | Prompt tok | mlx_lm | mlx_swift_lm | ax engine | ax engine + n-gram accel |
|---|---|---:|---:|---:|---:|---|
| Gemma 4 E2B | 4-bit · group=64 · affine | 128 | 198.4 | 192.7 (−2.9%) | 176.4 (−11.1%) | **548.9 (+176.7%)** |
|    |    | 512 | 194.3 | 183.7 (−5.5%) | 171.8 (−11.6%) | **542.5 (+179.2%)** |
| Gemma 4 E2B | 5-bit · group=64 · affine‖ | 128 | 176.5 | 172.9 (−2.0%) | 161.1 (−8.7%) | **434.2 (+146.0%)** |
|    |    | 512 | 174.4 | 163.9 (−6.0%) | 150.4 (−13.7%) | **412.1 (+136.3%)** |
| Gemma 4 E2B | 6-bit · group=64 · affine‖ | 128 | 153.1 | 147.2 (−3.8%) | 137.7 (−10.0%) | **377.3 (+146.5%)** |
|    |    | 512 | 148.6 | 141.1 (−5.0%) | 130.4 (−12.2%) | **374.0 (+151.8%)** |
| Gemma 4 E2B | 8-bit · group=64 · affine‖ | 128 | 131.7 | 133.4 (+1.2%) | 127.8 (−3.0%) | **431.7 (+227.7%)** |
|    |    | 512 | 134.0 | 131.5 (−1.9%) | 124.6 (−7.0%) | **427.7 (+219.2%)** |
| Gemma 4 26B A4B | 4-bit · group=64 · affine¶ | 128 | 118.3 | 110.8 (−6.3%) | 115.8 (−2.1%) | **252.4 (+113.4%)** |
|    |    | 512 | 115.3 | 106.5 (−7.7%) | 111.3 (−3.5%) | **206.9 (+79.4%)** |
| Gemma 4 31B | 4-bit · group=64 · affine | 128 | 26.4 | 25.7 (−2.7%) | 25.7 (−2.6%) | **61.7 (+133.5%)** |
|    |    | 512 | 25.4 | 24.9 (−2.0%) | 24.9 (−1.7%) | **57.6 (+126.9%)** |
| Qwen 3.5 9B | 4-bit · group=64 · affine | 128 | 94.2 | 93.0 (−1.3%) | 94.6 (+0.4%) | **196.3 (+108.4%) †** |
|    |    | 512 | 93.9 | 90.6 (−3.5%) | 92.8 (−1.2%) | 91.8 (−2.2%) † |
| Qwen 3.6 35B A3B | UD-MLX 4-bit · group=64 · affine§ | 128 | 107.8 | 105.8 (−1.8%) | 111.7 (+3.7%) | **259.2 (+140.5%) †** |
|    |    | 512 | 107.2 | 105.0 (−2.1%) | 110.4 (+3.0%) | **257.5 (+140.1%) †** |
| Qwen 3.6 35B A3B | MLX 5-bit · group=64 · affine§ | 128 | 112.5 | 110.4 (−1.9%) | 120.6 (+7.2%) | **260.8 (+131.8%) †** |
|    |    | 512 | 112.4 | 108.8 (−3.2%) | 120.0 (+6.7%) | **258.0 (+129.5%) †** |
| Qwen 3.6 35B A3B | MLX 6-bit · group=64 · affine§ | 128 | 102.3 | 98.4 (−3.9%) | 105.5 (+3.1%) | **239.0 (+133.5%) †** |
|    |    | 512 | 102.1 | 97.0 (−5.1%) | 104.5 (+2.3%) | **236.3 (+131.3%) †** |
| Qwen 3.6 35B A3B | MLX 8-bit · group=64 · affine§ | 128 | 93.5 | 88.3 (−5.5%) | 94.2 (+0.7%) | **236.2 (+152.6%) †** |
|    |    | 512 | 92.9 | 88.1 (−5.2%) | 94.5 (+1.6%) | **232.1 (+149.8%) †** |
| Qwen Coder Next | 4-bit · group=64 · affine‡ | 128 | 88.6 | 85.2 (−3.8%) | 92.3 (+4.2%) | **247.1 (+178.8%) †** |
|    |    | 512 | 88.9 | 88.3 (−0.7%) | 94.6 (+6.4%) | **242.3 (+172.7%) †** |

† Qwen 3.5, Qwen 3.6, and Qwen Coder Next n-gram acceleration rows use a rollback-safe
branch/recompute path for SSM state. These are effective-throughput
measurements from AX's n-gram acceleration policy, not raw model decode speed.
Linear-attention acceleration is prompt/output-pattern dependent. Benchmark JSON
artifacts include fixed-schema n-gram telemetry fields; the throughput table
uses the median `server_sse_runner_time_us` timing plus output-token count.

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
| Gemma 4 E2B | 4-bit · group=64 · affine | 128 | 2,451.0 | 3,032.3 (+23.7%) | 3,289.5 (+34.2%) |
|    |    | 512 | 7,909.9 | 6,679.4 (−15.6%) | 7,741.2 (−2.1%) |
| Gemma 4 E2B | 5-bit · group=64 · affine‖ | 128 | 2,338.0 | 2,936.3 (+25.6%) | 3,159.3 (+35.1%) |
|    |    | 512 | 7,772.5 | 6,863.5 (−11.7%) | 7,403.9 (−4.7%) |
| Gemma 4 E2B | 6-bit · group=64 · affine‖ | 128 | 2,435.5 | 3,028.7 (+24.4%) | 3,009.4 (+23.6%) |
|    |    | 512 | 7,772.0 | 6,863.8 (−11.7%) | 6,888.6 (−11.4%) |
| Gemma 4 E2B | 8-bit · group=64 · affine‖ | 128 | 1,842.4 | 2,596.9 (+41.0%) | 3,049.3 (+65.5%) |
|    |    | 512 | 6,914.1 | 5,691.1 (−17.7%) | 7,336.8 (+6.1%) |
| Gemma 4 26B A4B | 4-bit · group=64 · affine¶ | 128 | 694.8 | 1,235.2 (+77.8%) | 1,239.0 (+78.3%) |
|    |    | 512 | 2,004.5 | 2,805.0 (+39.9%) | 2,827.0 (+41.0%) |
| Gemma 4 31B | 4-bit · group=64 · affine | 128 | 344.1 | 617.3 (+79.4%) | 538.7 (+56.6%) |
|    |    | 512 | 581.4 | 765.6 (+31.7%) | 709.5 (+22.0%) |
| Qwen 3.5 9B | 4-bit · group=64 · affine | 128 | 989.9 | 1,743.5 (+76.1%) | 1,905.2 (+92.5%) |
|    |    | 512 | 2,105.3 | 2,922.4 (+38.8%) | 2,720.1 (+29.2%) |
| Qwen 3.6 35B A3B | UD-MLX 4-bit · group=64 · affine§ | 128 | 479.5 | 854.7 (+78.3%) | 1,019.7 (+112.7%) |
|    |    | 512 | 1,492.0 | 2,511.7 (+68.3%) | 2,565.7 (+72.0%) |
| Qwen 3.6 35B A3B | MLX 5-bit · group=64 · affine§ | 128 | 455.2 | 811.9 (+78.3%) | 947.0 (+108.0%) |
|    |    | 512 | 1,460.0 | 2,449.5 (+67.8%) | 2,469.1 (+69.1%) |
| Qwen 3.6 35B A3B | MLX 6-bit · group=64 · affine§ | 128 | 399.9 | 696.3 (+74.1%) | 918.1 (+129.6%) |
|    |    | 512 | 1,323.5 | 2,340.2 (+76.8%) | 2,354.5 (+77.9%) |
| Qwen 3.6 35B A3B | MLX 8-bit · group=64 · affine§ | 128 | 368.2 | 585.5 (+59.0%) | 903.1 (+145.3%) |
|    |    | 512 | 1,199.5 | 2,234.5 (+86.3%) | 2,319.0 (+93.3%) |
| Qwen Coder Next | 4-bit · group=64 · affine‡ | 128 | 276.4 | 437.8 (+58.4%) | 852.1 (+208.3%) |
|    |    | 512 | 888.6 | 1,652.8 (+86.0%) | 2,686.0 (+202.3%) |

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
