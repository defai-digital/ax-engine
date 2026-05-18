# AX Engine

### Faster Inference: Prefill, Decode, and TTFT

<table>
<tr>
<td align="center"><strong>Prefill rate</strong></td>
<td align="center"><strong>Decode rate</strong></td>
<td align="center"><strong>TTFT</strong></td>
</tr>
<tr>
<td><img src="docs/assets/perf-prefill-box-whisker.svg" alt="Box-and-Whisker Plot comparing llama.cpp Metal, mlx_lm, ax_engine, and ax_engine plus n-gram prefill rates"></td>
<td><img src="docs/assets/perf-decode-box-whisker.svg" alt="Box-and-Whisker Plot comparing llama.cpp Metal, mlx_lm, ax_engine, and ax_engine plus n-gram decode rates"></td>
<td><img src="docs/assets/perf-ttft-box-whisker.svg" alt="Box-and-Whisker Plot comparing llama.cpp Metal, mlx_lm, ax_engine, and ax_engine plus n-gram TTFT"></td>
</tr>
</table>

AX Engine is a Mac-first LLM inference runtime, local server, SDK layer, and
benchmark toolkit for Apple Silicon.

AX Engine runs direct-support MLX model families on Apple Silicon, and keeps
other MLX text models or non-MLX models reachable through explicit `mlx-lm` and
`llama.cpp` compatibility routes. Users get one AX server, SDK, and benchmark
surface while repo-owned model coverage grows.

> Requires **macOS 14 (Sonoma) or later** on **Apple Silicon M2 Max or newer** with **32 GB RAM minimum**.
> Rust 1.85+ for source builds.

## 30-Second Setup

Install the released command-line tools and verify the runtime:

```bash
brew install defai-digital/ax-engine/ax-engine
ax-engine-bench doctor
```

Then download a model and start `ax-engine-server` from the CLI:

```bash
# Download an mlx-community model and generate its manifest in one step
MODEL_DIR="$(python3 scripts/download_model.py mlx-community/Qwen3-4B-4bit --json | python3 -c 'import json,sys; print(json.load(sys.stdin)["dest"])')"

# Start the local HTTP server. Keep this process running.
ax-engine-server \
  --mlx \
  --mlx-model-artifacts-dir "$MODEL_DIR" \
  --port 8080

# In another terminal, inspect the running server.
curl http://127.0.0.1:8080/v1/runtime
```

Or from Python (after `maturin develop` or `pip install ax-engine`):

```python
from ax_engine import download_model, Session
path = download_model("mlx-community/Qwen3-4B-4bit")
with Session(mlx=True, mlx_model_artifacts_dir=str(path)) as s:
    print(s.generate([1, 2, 3], max_output_tokens=8).output_tokens)
```

`download_model()` downloads weights and auto-runs `ax-engine-bench generate-manifest`.
See [Getting a Model](#getting-a-model) for all paths including raw HF checkpoints.

## Typical Hardware Stack ([hardware FAQ](docs/FAQ.md#what-hardware-does-ax-engine-support))

For local agent and chatbot workloads, AX Engine is a better fit for a small
model portfolio than for one model serving every workflow. See the
[FAQ model-stack guidance](docs/FAQ.md#what-model-stack-should-i-run-on-high-memory-apple-silicon)
for the full recommendation.

| Hardware | Recommended memory | Best fit |
|---|---:|---|
| Mac mini M4 Pro | 64 GB RAM | Compact always-on local chatbot and agent server |
| MacBook Pro M5 Max | 128 GB RAM | Portable high-throughput chatbot, agent, and coding stack |
| Mac Studio M3 Ultra | 256 GB RAM | Larger local model portfolio, longer contexts, and heavier parallel workloads |

| Role | Recommended model | Setup | App | Why |
|---|---|---|---|---|
| Default chatbot | Gemma 4 26B-A4B / 31B | 4-bit or 6-bit, 16K-32K | [ax-studio](https://github.com/defai-digital/ax-studio) | General assistant path for reasoning, chat, JSON/function calling, and on-device agent workflows |
| General agentic model | Qwen3.6-35B-A3B | 4-bit or 6-bit, 16K-32K | AX server / SDK | Strong general agent and coding balance; sparse MoE keeps active compute low |
| Coding specialist | Qwen3-Coder-Next | 6-bit + 16K default; 4-bit/5-bit + 32K when needed | [ax-code](https://github.com/defai-digital/ax-code) | Dedicated local coding-agent path for repo editing, tool use, and long coding sessions |

## Why AX Engine ([FAQ](docs/FAQ.md#is-ax-faster-because-it-replaces-mlx-kernels))

AX Engine gives local inference work a stable runtime contract:

- `ax-engine-server` exposes a local HTTP adapter over the runtime.
- `ax-engine-bench` records workload contracts, route identity, correctness,
  determinism, and performance evidence.
- `ax-engine-sdk`, Python bindings, and the JavaScript client provide
  thin integration surfaces over the same backend-resolution rules.
- Repo-owned MLX execution is tracked in
  [Direct Support Models](#direct-support-models-support-llm-models); delegated
  routes stay separate from AX-owned throughput claims.
- Delegated `mlx_lm.server` and `llama.cpp` routes cover explicit
  compatibility cases without turning delegated results into AX-owned
  throughput claims.

[mlx_lm](https://github.com/ml-explore/mlx-lm) is the canonical MLX reference.
AX Engine compares against `mlx_lm.benchmark` and keeps `mlx_lm.server` as the
explicit delegated compatibility route when AX does not yet have a repo-owned
graph.

For measured direct-support transformer families on Apple Silicon, the AX-owned
runtime layer can produce higher effective throughput than the reference MLX
runtimes on matching benchmark shapes:

- **N-gram acceleration** reaches up to 3.1x mlx_lm decode
  throughput on high-hit benchmark rows with no second draft model
- **Coding-shaped decode is a natural fit when local repetition exists**,
  including completion, edit loops, structured diffs, JSON/tool output, imports,
  indentation, and repeated identifiers
- **AX-owned request lifecycle** provides deterministic, auditable scheduling,
  KV block management, and prefix reuse that upstream Python runtimes do not
  expose as stable contracts
- **Long-session prefix reuse** restores physical MLX KV snapshots on validated
  cache layouts, so long-running chat and agent loops can avoid repeatedly
  pre-filling the same accumulated context. See
  [`docs/LONG-CONTEXT.md`](docs/LONG-CONTEXT.md) for the evidence boundary.
- **workload-contract tooling** (`ax-engine-bench`) validates correctness,
  determinism, route identity, and regression across checked-in manifests, not
  just throughput snapshots

See the [FAQ](docs/FAQ.md#is-ax-faster-because-it-replaces-mlx-kernels) for
the boundary between MLX kernels and AX-owned runtime behavior.

## Runtime Paths

| Path | Use it for | Current scope |
|---|---|---|
| Repo-owned MLX runtime | Direct-support MLX model families and repo-owned performance claims when backed by benchmark artifacts | Local Apple Silicon inference, token-based server/SDK requests, benchmarked direct and n-gram acceleration modes |
| `mlx_lm_delegated` | MLX text models that upstream `mlx-lm` supports before AX has a repo-owned graph | Blocking and SSE text generation through a user-provided `mlx_lm.server`; `/v1/generate`, `/v1/generate/stream`, and OpenAI-compatible completion/chat text endpoints. Streaming is delegated text compatibility evidence, not repo-owned token/KV performance |
| `llama_cpp` | GGUF and non-MLX local inference | Delegated llama.cpp server/CLI compatibility; route-contract evidence, not repo-owned MLX throughput |

The runtime report exposes `selected_backend`, `support_tier`, and
`resolution_policy` so callers and benchmark artifacts can distinguish these
paths.

For the exact OpenAI-shaped endpoint contract, including what is and is not
compatible today, see `docs/API-COMPATIBILITY.md`.

## Design

The repo-owned MLX path uses MLX directly for tensor operations through the
official `mlx-c` C API. MLX owns the Apple-maintained Metal kernels; AX owns the
runtime behavior above the graph: request lifecycle, scheduling, KV/cache
policy, n-gram acceleration, manifests, and benchmark evidence.

Design details live in the focused docs:
[Scheduler](docs/SCHEDULER.md) ·
[KV Cache](docs/KV-CACHE.md) ·
[Long Context](docs/LONG-CONTEXT.md) ·
[Benchmark Design](docs/BENCH-DESIGN.md) ·
[FAQ](docs/FAQ.md#is-ax-faster-because-it-replaces-mlx-kernels).

## Direct Support Models ([support LLM models](docs/SUPPORTED-MODELS.md))

Direct support means AX has a repo-owned `ax-engine-mlx` graph for the model
family and loads MLX safetensors through the AX manifest path. Other MLX text
models can still use the explicit `mlx_lm_delegated` compatibility route, but
delegated rows are not AX-owned throughput claims.

| Family | Direct model IDs | Current scope | Architecture notes |
|---|---|---|---|
| Gemma 4 | `gemma-4-e2b-it`, `gemma-4-e4b-it`, `gemma-4-26b-a4b-it`, `gemma-4-31b-it` | Repo-owned MLX runtime; MLX affine 4/5/6/8-bit weights where available | Dense, per-layer embedding, and MoE variants; sliding-window + full attention, K=V full-attention layers, logit softcapping |
| Gemma 3 | `gemma-3-1b-it` through `gemma-3-27b-it` | Repo-owned MLX runtime | GeGLU dense FFN; per-head QK norm; sliding-window local + global attention interleaving; embedding scale |
| Qwen 3.5 | `Qwen3.5-9B` | Repo-owned MLX runtime | Linear attention + MoE FFN; `attn_output_gate` per-head interleaving |
| Qwen 3.6 / Coder Next | `Qwen3.6-35B-A3B` 4/5/6/8-bit MLX, `Qwen3-Coder-Next-4bit` | Repo-owned MLX runtime | `qwen3_next`: GatedDelta linear attention, full attention with per-head sigmoid gate, sparse top-k MoE with shared expert |
| Qwen 3 | `Qwen3-0.6B` through `Qwen3-32B` | Repo-owned MLX runtime | SwiGLU dense FFN; per-head QK norm; optional MoE variants |
| GLM 4.7 Flash | `mlx-community/GLM-4.7-Flash-4bit` | Repo-owned MLX runtime after community-model promotion | MLA attention, sigmoid router, latent-KV cache support |
| LLaMA 3 / 3.1 / 3.2 / 3.3 | `Llama-3.1-8B-Instruct` and related | Repo-owned MLX runtime | SwiGLU dense FFN; LLaMA-3 RoPE scaling |
| LLaMA 4 | `Llama-4-Scout`, `Llama-4-Maverick` | Repo-owned MLX runtime | iRoPE; interleaved MoE with frequency-based dispatch; attention temperature scaling |
| Mistral 3 / Ministral | `Mistral-Small-3.1`, `Ministral-3B`, `Ministral-8B` | Repo-owned MLX runtime | SwiGLU dense FFN; sliding-window attention on all layers |
| Mixtral | `Mixtral-8x7B-Instruct`, `Mixtral-8x22B-Instruct` | Repo-owned MLX runtime | SWA + sparse top-2 MoE; `block_sparse_moe` weights |
| DeepSeek V3 / V3.2 | `DeepSeek-V3`, `DeepSeek-V3-0324` | Repo-owned MLX runtime | MLA attention; sigmoid MoE routing with optional correction bias; shared experts |

Direct-support models use MLX safetensors format with the AX
`model-manifest.json` descriptor. Each supported architecture has a hand-written
forward pass in `ax-engine-mlx`. Adding a new architecture means implementing
the model graph, not wiring up a generic loader.

Community-model checks are tracked by evidence level. Before promoting another
architecture, run
`scripts/probe_mlx_model_support.py --model-dir <model-dir>`; a model should
report `repo_owned_runtime_ready` only when its manifest, local reference files,
and runtime path are all present.

## Performance ([full performance docs](docs/PERFORMANCE.md))

<!-- readme-performance-artifacts: base=benchmarks/results/mlx-inference/2026-05-14-ax-direct-ngram-r4/ -->
The README keeps the common Gemma 4 and Qwen 3.6 generation benchmark rows
visible. Full result tables and interpretation live in
[`docs/PERFORMANCE.md`](docs/PERFORMANCE.md); benchmark methodology, test setup,
and reproduction details live in [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

These rows are a provenance-tracked composite from
`benchmarks/results/mlx-inference/2026-05-14-ax-direct-ngram-r4/`. All rows use
the same prompt contract, generation=128 shape, prompt SHA checks, 5
repetitions, a 15-second cooldown between trials, and production-build binaries.
Percentages are versus `mlx_lm`. The `llama.cpp Metal*` column is a
shape-compatible external reference; it does not share prompt-token hashes with
the MLX rows.

### Prefill throughput (tok/s) — percentages vs mlx_lm

| Model | MLX quantization | Prompt tok | llama.cpp Metal* | mlx_lm | ax engine |
|---|---|---:| ---: |---:|---:|
| Gemma 4 E2B | 4-bit | 128 | 3,532.9 | 2,443.1 | **3,906.4 (+59.9%)** |
|         |         | 512 | 7,232.1 | 7,768.5 | **8,361.2 (+7.6%)** |
|         |         | 2048 | — | — | — |
| Gemma 4 E2B | 5-bit | 128 | 3,427.4 | 2,544.0 | **3,818.0 (+50.1%)** |
|         |         | 512 | 7,159.5 | 8,085.2 | **8,251.7 (+2.1%)** |
|         |         | 2048 | — | — | — |
| Gemma 4 E2B | 6-bit | 128 | 3,431.5 | 2,374.6 | **3,799.9 (+60.0%)** |
|         |         | 512 | 7,061.6 | 7,742.7 | **8,193.6 (+5.8%)** |
|         |         | 2048 | — | — | — |
| Gemma 4 E2B | 8-bit | 128 | 3,698.8 | 2,144.7 | **3,801.7 (+77.3%)** |
|         |         | 512 | 7,747.4 | 7,221.3 | **8,237.2 (+14.1%)** |
|         |         | 2048 | — | — | — |
| Gemma 4 E4B | 4-bit | 128 | 2,238.1 | 1,757.9 | **3,002.8 (+70.8%)** |
|         |         | 512 | 4,343.5 | 4,502.1 | **4,652.5 (+3.3%)** |
|         |         | 2048 | — | — | — |
| Gemma 4 26B A4B | 4-bit | 128 | 1,937.7 | 730.5 | **1,318.5 (+80.5%)** |
|         |         | 512 | 3,387.1 | 2,090.9 | **3,149.3 (+50.6%)** |
|         |         | 2048 | — | — | — |
| Gemma 4 31B | 4-bit | 128 | 511.7 | 374.7 | **657.3 (+75.4%)** |
|         |         | 512 | 651.7 | 653.7 | **783.6 (+19.9%)** |
|         |         | 2048 | — | — | — |
| Qwen 3.6 35B A3B | 4-bit | 128 | 1,711.4 | 575.0 | **1,134.4 (+97.3%)** |
|         |         | 512 | 3,137.6 | 1,702.2 | **2,842.1 (+67.0%)** |
|         |         | 2048 | — | — | — |
| Qwen 3.6 35B A3B | MLX 5-bit | 128 | 1,563.4 | 499.2 | **1,122.3 (+124.8%)** |
|         |         | 512 | 2,882.8 | 1,585.8 | **2,742.0 (+72.9%)** |
|         |         | 2048 | — | — | — |
| Qwen 3.6 35B A3B | MLX 6-bit | 128 | 1,598.4 | 438.6 | **1,022.2 (+133.0%)** |
|         |         | 512 | 2,988.3 | 1,440.3 | **2,622.9 (+82.1%)** |
|         |         | 2048 | — | — | — |
| Qwen 3.6 35B A3B | MLX 8-bit | 128 | 1,664.2 | 406.4 | **1,038.4 (+155.5%)** |
|         |         | 512 | 3,122.8 | 1,298.5 | **2,606.6 (+100.7%)** |
|         |         | 2048 | — | — | — |

### Decode throughput (tok/s) — generation=128 tokens, temp=0

Higher is better. `ax direct baseline` disables n-gram acceleration.
`ax default n-gram` is the default AX decode policy and reports observed
effective throughput, not raw model-kernel speed.

| Model | MLX quantization | Prompt tok | llama.cpp Metal* | mlx_lm | ax direct baseline | ax default n-gram |
|---|---|---:| ---: |---:|---:|---:|
| Gemma 4 E2B | 4-bit | 128 | 162.3 | 213.5 | 190.1 (-10.9%) | **564.8 (+164.5%)** |
|         |         | 512 | 157.8 | 210.1 | 179.9 (-14.4%) | **573.3 (+172.8%)** |
|         |         | 2048 | — | — | — | — |
| Gemma 4 E2B | 5-bit | 128 | 147.0 | 196.9 | 174.0 (-11.6%) | **452.5 (+129.9%)** |
|         |         | 512 | 151.3 | 190.1 | 164.9 (-13.2%) | **458.7 (+141.3%)** |
|         |         | 2048 | — | — | — | — |
| Gemma 4 E2B | 6-bit | 128 | 138.5 | 175.4 | 156.2 (-11.0%) | **428.8 (+144.4%)** |
|         |         | 512 | 137.6 | 170.7 | 151.1 (-11.5%) | **424.6 (+148.8%)** |
|         |         | 2048 | — | — | — | — |
| Gemma 4 E2B | 8-bit | 128 | 131.4 | 154.3 | 139.6 (-9.6%) | **465.8 (+201.8%)** |
|         |         | 512 | 137.3 | 149.8 | 135.6 (-9.5%) | **460.3 (+207.2%)** |
|         |         | 2048 | — | — | — | — |
| Gemma 4 E4B | 4-bit | 128 | 104.5 | 137.4 | 122.6 (-10.8%) | **355.9 (+159.0%)** |
|         |         | 512 | 104.1 | 133.8 | 119.5 (-10.7%) | **355.9 (+165.9%)** |
|         |         | 2048 | — | — | — | — |
| Gemma 4 26B A4B | 4-bit | 128 | 107.2 | 128.2 | 120.5 (-6.0%) | **274.2 (+113.9%)** |
|         |         | 512 | 107.7 | 124.9 | 117.6 (-5.9%) | **233.2 (+86.7%)** |
|         |         | 2048 | — | — | — | — |
| Gemma 4 31B | 4-bit | 128 | 24.5 | 29.0 | 27.8 (-4.1%) | **65.4 (+125.7%)** |
|         |         | 512 | 24.1 | 28.4 | 27.0 (-5.1%) | **63.9 (+125.0%)** |
|         |         | 2048 | — | — | — | — |
| Qwen 3.6 35B A3B | 4-bit | 128 | 89.7 | 120.8 | **123.3 (+2.0%)** | **284.2 (+135.2%)** |
|         |         | 512 | 90.9 | 120.6 | **122.4 (+1.5%)** | **281.2 (+133.2%)** |
|         |         | 2048 | — | — | — | — |
| Qwen 3.6 35B A3B | MLX 5-bit | 128 | 98.5 | 125.7 | **137.2 (+9.1%)** | **284.2 (+126.1%)** |
|         |         | 512 | 86.9 | 125.8 | **136.5 (+8.5%)** | **281.4 (+123.7%)** |
|         |         | 2048 | — | — | — | — |
| Qwen 3.6 35B A3B | MLX 6-bit | 128 | 95.7 | 114.8 | **120.2 (+4.7%)** | **261.1 (+127.4%)** |
|         |         | 512 | 95.5 | 114.4 | **119.6 (+4.6%)** | **258.4 (+125.9%)** |
|         |         | 2048 | — | — | — | — |
| Qwen 3.6 35B A3B | MLX 8-bit | 128 | 90.3 | 105.1 | **108.0 (+2.8%)** | **263.1 (+150.4%)** |
|         |         | 512 | 90.1 | 104.4 | **107.2 (+2.7%)** | **259.4 (+148.5%)** |
|         |         | 2048 | — | — | — | — |

### Time to first token (ms) — generation=128 tokens, temp=0

Lower is better. `mlx_lm` values are derived from reported prefill throughput.
AX values are measured directly from per-step runner timing in the SSE event
stream.

| Model | MLX quantization | Prompt tok | llama.cpp Metal* | mlx_lm | ax engine |
|---|---|---:| ---: |---:|---:|
| Gemma 4 E2B | 4-bit | 128 | 36.2 | 52.4 | **32.8 (-37.5%)** |
|         |         | 512 | 70.8 | 65.9 | **61.2 (-7.1%)** |
|         |         | 2048 | — | — | — |
| Gemma 4 E2B | 5-bit | 128 | 37.3 | 50.3 | **33.5 (-33.4%)** |
|         |         | 512 | 71.5 | 63.3 | **62.0 (-2.0%)** |
|         |         | 2048 | — | — | — |
| Gemma 4 E2B | 6-bit | 128 | 37.3 | 53.9 | **33.7 (-37.5%)** |
|         |         | 512 | 72.5 | 66.1 | **62.5 (-5.5%)** |
|         |         | 2048 | — | — | — |
| Gemma 4 E2B | 8-bit | 128 | 34.6 | 59.7 | **33.7 (-43.6%)** |
|         |         | 512 | 66.1 | 70.9 | **62.2 (-12.3%)** |
|         |         | 2048 | — | — | — |
| Gemma 4 E4B | 4-bit | 128 | 57.2 | 72.8 | **42.6 (-41.5%)** |
|         |         | 512 | 117.9 | 113.7 | **110.0 (-3.2%)** |
|         |         | 2048 | — | — | — |
| Gemma 4 26B A4B | 4-bit | 128 | 66.1 | 175.2 | **97.1 (-44.6%)** |
|         |         | 512 | 151.2 | 244.9 | **162.6 (-33.6%)** |
|         |         | 2048 | — | — | — |
| Gemma 4 31B | 4-bit | 128 | 250.2 | 341.6 | **194.7 (-43.0%)** |
|         |         | 512 | 785.6 | 783.2 | **653.4 (-16.6%)** |
|         |         | 2048 | — | — | — |
| Qwen 3.6 35B A3B | 4-bit | 128 | 74.8 | 222.6 | **112.8 (-49.3%)** |
|         |         | 512 | 163.2 | 300.8 | **180.1 (-40.1%)** |
|         |         | 2048 | — | — | — |
| Qwen 3.6 35B A3B | MLX 5-bit | 128 | 81.9 | 256.4 | **114.0 (-55.5%)** |
|         |         | 512 | 177.6 | 322.9 | **186.7 (-42.2%)** |
|         |         | 2048 | — | — | — |
| Qwen 3.6 35B A3B | MLX 6-bit | 128 | 80.1 | 291.8 | **125.2 (-57.1%)** |
|         |         | 512 | 171.3 | 355.5 | **195.2 (-45.1%)** |
|         |         | 2048 | — | — | — |
| Qwen 3.6 35B A3B | MLX 8-bit | 128 | 76.9 | 314.9 | **123.3 (-60.9%)** |
|         |         | 512 | 164.0 | 394.3 | **196.4 (-50.2%)** |
|         |         | 2048 | — | — | — |

Embedding benchmarks are kept out of this README summary; see
[`docs/EMBEDDINGS.md`](docs/EMBEDDINGS.md) for embedding throughput, serving,
and cold-start measurements.

## Installation

### Homebrew

For tagged macOS arm64 releases, install the command-line tools from
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

The fastest local workflow is:

1. install or build the command-line tools;
2. download a supported MLX model and generate its manifest;
3. check model readiness;
4. start the local server and call its HTTP endpoints.

The commands below use source-build paths. If you installed with Homebrew, use
`ax-engine-server` and `ax-engine-bench` directly instead of
`./target/release/...`.

### Start `ax-engine-server` from the CLI

```bash
# Download a model and generate its manifest
MODEL_DIR="$(python3 scripts/download_model.py mlx-community/Qwen3-4B-4bit --json | python3 -c 'import json,sys; print(json.load(sys.stdin)["dest"])')"
# MODEL_DIR uses the Hugging Face Hub snapshot cache by default, e.g.
# ~/.cache/huggingface/hub/models--mlx-community--Qwen3-4B-4bit/snapshots/<hash>

# Check readiness
./target/release/ax-engine-bench doctor --mlx-model-artifacts-dir "$MODEL_DIR"

# HTTP inference server (repo-owned MLX runtime)
./target/release/ax-engine-server \
  --mlx \
  --mlx-model-artifacts-dir "$MODEL_DIR" \
  --port 8080

# In another terminal, inspect the running server
curl http://127.0.0.1:8080/v1/runtime

# Optional smoke generation request
curl http://127.0.0.1:8080/v1/generate \
  -H 'content-type: application/json' \
  -d '{
    "model": "qwen3_dense",
    "input_tokens": [1, 2, 3, 4],
    "max_output_tokens": 4,
    "sampling": {
      "temperature": 0.0,
      "top_p": 1.0,
      "top_k": 0,
      "seed": 1234
    }
  }'
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

`mlx_lm_delegated` is a compatibility route, not an AX-owned MLX throughput
claim. AX forwards text generation to upstream `mlx_lm.server`, preserves
sampling fields such as `temperature`, `top_p`, `top_k`, `repetition_penalty`,
and `seed`, and exposes blocking plus SSE text through the AX API. Streamed
chunks are delegated text deltas; they are not AX-owned token IDs, KV state, or
model-kernel throughput evidence. Tool calls and visual/multimodal inputs are
not compatibility contracts yet.

```bash
# Primary benchmark: AX vs mlx_lm
python3 scripts/bench_mlx_inference_stack.py \
  --model-dir /path/to/local/mlx-model \
  --prompt-tokens 128,512,2048 --generation-tokens 128 \
  --ax-compare-policies --repetitions 3 \
  --output benchmarks/results/mlx-inference/2026-05-04/gemma-4-e2b-it-4bit.json

# Secondary workload-contract benchmark
./target/release/ax-engine-bench scenario \
  --manifest benchmarks/manifests/scenario/chat_gemma4_e2b_short.json \
  --output-root benchmarks/results

# Smoke checks
./target/release/ax-engine-bench doctor --mlx-model-artifacts-dir "$MODEL_DIR"
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

# For automation, emit a parseable summary
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
bash scripts/check-mlx-telemetry.sh                              # Gemma/AX MLX telemetry gate
```

For Gemma/AX MLX telemetry and decode-profile changes, prefer the targeted
`scripts/check-mlx-telemetry.sh` gate. Use
`scripts/check-mlx-telemetry.sh --full-workspace` when the change touches shared
Rust contracts; that protected path compiles the workspace with
`cargo test --workspace --no-run --jobs 1` before running crate-by-crate tests.

Coverage is collected by the report-only GitHub Actions workflow in
`.github/workflows/coverage.yml`. It publishes Rust `cargo llvm-cov` and Python
`coverage.py` artifacts without enforcing a percentage threshold yet; add a gate
only after the project has a stable baseline across macOS, MLX, and PyO3 paths.

Public documentation is in `docs/`. Canonical benchmark manifests are in
`benchmarks/manifests/`. Key design documents:
[SDK / API](docs/SDK.md) ·
[Python](docs/PYTHON.md) ·
[JavaScript / TypeScript](docs/JAVASCRIPT.md) ·
[Go](docs/GO.md) ·
[Ruby](docs/RUBY.md) ·
[Mojo](docs/MOJO.md) ·
[Scheduler](docs/SCHEDULER.md) ·
[KV Cache](docs/KV-CACHE.md) ·
[Benchmarking](docs/BENCH-DESIGN.md) ·
[Serving Benchmarks](docs/SERVING-BENCHMARKS.md)

## Limitations

- **GatedDelta prefill (Qwen3.5)**: Qwen3.5 prefill can trail upstream MLX
  references on longer prompts; decode and Qwen3-Next are not affected in the
  same way.
- **Raw HuggingFace weights**: use pre-sanitized MLX community weights or
  convert first with `mlx_lm.convert`.
- **N-gram acceleration rows**: effective-throughput measurements, not raw
  model-kernel speedups.
- **TurboQuant KV compression**: experimental and off by default.

See the [FAQ limitations entry](docs/FAQ.md#what-are-the-current-limitations)
for details.

## Contributing

AX Engine welcomes community input through issue tickets, wishlist requests,
reproducible benchmark results, and documentation feedback. We generally do not
accept unsolicited code PRs, especially for runtime, model, kernel, scheduler,
cache, n-gram, or performance-tuning changes.

Performance tuning is tightly coupled: a local speedup can regress correctness,
TTFT, memory pressure, direct-vs-n-gram behavior, long-context behavior, serving
stability, or another model family. Please open an issue first with the problem,
target workload, and evidence so maintainers can choose the right validation
path. See [CONTRIBUTING.md](CONTRIBUTING.md) for issue, wishlist, and benchmark
result guidelines.

## Community

- Website: [automatosx.com](https://automatosx.com)
- Discord: [Join us](https://discord.com/invite/cTavsMgu)
- Email: enquiry@defai.digital

## License

Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

Copyright (c) 2026 [DEFAI Private Limited](https://defai.digital)
