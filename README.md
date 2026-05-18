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
| General agentic model | Qwen3.6-35B-A3B / Qwen3.6-27B | 35B A3B 4-bit; 27B 4/5/6/8-bit, 16K-32K | AX server / SDK | Strong general agent and coding balance; sparse MoE keeps active compute low |
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
| Qwen 3.6 / Coder Next | `Qwen3.6-35B-A3B` 4-bit MLX, `Qwen3.6-27B` 4/5/6/8-bit MLX, `Qwen3-Coder-Next-4bit` | Repo-owned MLX runtime | `qwen3_next`: GatedDelta linear attention, full attention with per-head sigmoid gate, sparse top-k MoE with shared expert |
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

<!-- readme-performance-artifacts: base=benchmarks/results/mlx-inference/2026-05-18-gguf-full-stack/ -->
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

<!-- llama-cpp-column-disclaimer -->
**`llama.cpp Metal*` column** — Shape-compatible reference produced by Metal-enabled `llama-bench`. `llama-bench` generates its own internal synthetic prompt tokens and does not consume the harness prompt JSON, so these numbers are NOT prompt-hash parity with the other columns. The intent is rough side-by-side context against a well-known third-party Metal runtime, not head-to-head comparison. MLX bit-widths are mapped to the nearest standard bartowski GGUF K-quant (4→Q4_K_M, 5→Q5_K_M, 6→Q6_K, 8→Q8_0). No percentage delta is shown for this column because the prompt is not shared. Source: `benchmarks/manifests/llama_cpp_metal/inventory.json`, `scripts/bench_llama_cpp_metal_sweep.py`.

### Prefill throughput (tok/s) — percentages vs mlx_lm

| Model | MLX quantization | Prompt tok | llama.cpp Metal* | mlx_lm | ax engine |
|---|---|---:| ---: |---:|---:|
| Gemma 4 E2B | 4-bit | 128 | 3,431.8 | 2,536.3 | **3,780.4 (+49.0%)** |
|         |         | 512 | 7,125.8 | 8,243.4 | **8,272.6 (+0.4%)** |
|         |         | 2048 | 7,062.2 | 17,851.0 | 8,011.4 (-55.1%) |
| Gemma 4 E2B | 5-bit | 128 | 3,382.0 | 2,399.6 | **3,783.3 (+57.7%)** |
|         |         | 512 | 7,175.2 | 8,064.6 | **8,257.1 (+2.4%)** |
|         |         | 2048 | 7,165.1 | 17,725.6 | 7,516.3 (-57.6%) |
| Gemma 4 E2B | 6-bit | 128 | 3,572.3 | 2,444.6 | **3,758.1 (+53.7%)** |
|         |         | 512 | 7,371.0 | 7,820.8 | **8,187.5 (+4.7%)** |
|         |         | 2048 | 7,334.2 | 17,373.8 | 7,633.0 (-56.1%) |
| Gemma 4 E2B | 8-bit | 128 | 3,459.5 | 2,295.9 | **3,667.8 (+59.8%)** |
|         |         | 512 | 7,776.0 | 7,516.3 | **8,052.7 (+7.1%)** |
|         |         | 2048 | 7,670.1 | 17,167.9 | 7,748.2 (-54.9%) |
| Gemma 4 E4B | 4-bit | 128 | 2,235.2 | 1,579.5 | **2,960.2 (+87.4%)** |
|         |         | 512 | 4,186.8 | 4,297.9 | **4,625.0 (+7.6%)** |
|         |         | 2048 | 4,373.4 | 7,361.5 | 4,713.4 (-36.0%) |
| Gemma 4 26B A4B | 4-bit | 128 | 1,940.3 | 730.3 | **1,304.0 (+78.6%)** |
|         |         | 512 | 3,404.4 | 2,095.6 | **3,126.5 (+49.2%)** |
|         |         | 2048 | 3,332.0 | 3,893.2 | **4,416.3 (+13.4%)** |
| Gemma 4 31B | 4-bit | 128 | 506.3 | 365.0 | **651.3 (+78.4%)** |
|         |         | 512 | 652.9 | 639.2 | **780.2 (+22.1%)** |
|         |         | 2048 | 604.4 | 741.5 | **747.5 (+0.8%)** |
| Qwen 3.6 27B | 4-bit | 128 | 542.4 | 431.2 | **762.9 (+76.9%)** |
|  |  | 512 | 764.6 | 754.0 | **915.1 (+21.4%)** |
|  |  | 2048 | 593.0 | 924.1 | **926.8 (+0.3%)** |
| Qwen 3.6 27B | 5-bit | 128 | 527.6 | 393.6 | **722.3 (+83.5%)** |
|  |  | 512 | 742.5 | 697.3 | **859.5 (+23.3%)** |
|  |  | 2048 | 611.9 | 868.0 | 862.3 (-0.7%) |
| Qwen 3.6 35B A3B | 4-bit | 128 | n/a | 575.0 | **1,134.4 (+97.3%)** |
|         |         | 512 | n/a | 1,702.2 | **2,842.1 (+67.0%)** |
|         |         | 2048 | n/a | — | — |

### Decode throughput (tok/s) — generation=128 tokens, temp=0

Higher is better. `ax direct baseline` disables n-gram acceleration.
`ax default n-gram` is the default AX decode policy and reports observed
effective throughput, not raw model-kernel speed.

| Model | MLX quantization | Prompt tok | llama.cpp Metal* | mlx_lm | ax direct baseline | ax default n-gram |
|---|---|---:| ---: |---:|---:|---:|
| Gemma 4 E2B | 4-bit | 128 | 157.6 | 211.2 | 183.1 (-13.3%) | **580.1 (+174.6%)** |
|         |         | 512 | 158.5 | 206.0 | 178.3 (-13.5%) | **576.1 (+179.7%)** |
|         |         | 2048 | 154.3 | 192.2 | 176.9 (-7.9%) | **531.0 (+176.3%)** |
| Gemma 4 E2B | 5-bit | 128 | 151.4 | 195.1 | 174.3 (-10.6%) | **448.4 (+129.9%)** |
|         |         | 512 | 153.8 | 188.3 | 168.0 (-10.8%) | **444.2 (+136.0%)** |
|         |         | 2048 | 154.2 | 181.3 | 157.0 (-13.4%) | **425.7 (+134.9%)** |
| Gemma 4 E2B | 6-bit | 128 | 151.6 | 174.0 | 156.6 (-10.0%) | **425.1 (+144.2%)** |
|         |         | 512 | 151.5 | 169.0 | 151.4 (-10.5%) | **420.0 (+148.5%)** |
|         |         | 2048 | 151.2 | 163.2 | 146.8 (-10.0%) | **396.2 (+142.7%)** |
| Gemma 4 E2B | 8-bit | 128 | 138.2 | 153.5 | 139.3 (-9.3%) | **458.8 (+198.9%)** |
|         |         | 512 | 138.4 | 149.1 | 135.3 (-9.3%) | **455.2 (+205.4%)** |
|         |         | 2048 | 138.1 | 144.5 | 131.8 (-8.8%) | **426.9 (+195.4%)** |
| Gemma 4 E4B | 4-bit | 128 | 108.7 | 136.1 | 121.9 (-10.4%) | **353.0 (+159.4%)** |
|         |         | 512 | 108.8 | 132.6 | 118.9 (-10.3%) | **353.1 (+166.3%)** |
|         |         | 2048 | 108.7 | 129.8 | 116.5 (-10.3%) | **275.8 (+112.4%)** |
| Gemma 4 26B A4B | 4-bit | 128 | 108.8 | 127.6 | 120.7 (-5.4%) | **271.8 (+113.0%)** |
|         |         | 512 | 108.4 | 124.1 | 117.6 (-5.2%) | **232.7 (+87.5%)** |
|         |         | 2048 | 108.9 | 119.5 | 111.5 (-6.6%) | **246.3 (+106.2%)** |
| Gemma 4 31B | 4-bit | 128 | 24.1 | 28.7 | 26.6 (-7.4%) | **65.2 (+127.1%)** |
|         |         | 512 | 24.0 | 28.2 | 27.2 (-3.8%) | **63.8 (+125.9%)** |
|         |         | 2048 | 22.8 | 27.0 | 25.9 (-4.2%) | **51.0 (+88.6%)** |
| Qwen 3.6 27B | 4-bit | 128 | 26.3 | 33.8 | 32.3 (-4.4%) | **32.3 (-4.5%)** |
|  |  | 512 | 26.2 | 33.6 | 32.8 (-2.3%) | **32.5 (-3.3%)** |
|  |  | 2048 | 23.8 | 33.3 | — (no decode)† | — (no decode)† |
| Qwen 3.6 27B | 5-bit | 128 | 22.6 | 28.2 | 26.6 (-5.8%) | 12.7 (-55.0%)‡ |
|  |  | 512 | 22.5 | 28.0 | 27.4 (-2.2%) | 27.0 (-3.8%) |
|  |  | 2048 | 21.2 | 27.8 | 27.2 (-2.3%) | 26.8 (-3.9%) |
| Qwen 3.6 35B A3B | 4-bit | 128 | n/a | 120.8 | **123.3 (+2.0%)** | **284.2 (+135.2%)** |
|         |         | 512 | n/a | 120.6 | **122.4 (+1.5%)** | **281.2 (+133.2%)** |
|         |         | 2048 | n/a | — | — | — |

† Qwen 3.6 27B 4-bit at prompt=2048 completed prefill but produced zero
decode tokens across all 5 trials (early EOS at decode step 0). Cell is
left as "—" because reporting `0.0 tok/s` would conflate a stop-token
exit with a throughput regression. See
`benchmarks/results/mlx-inference/2026-05-18-gguf-full-stack/qwen3_6-27b-4bit.json`.

‡ Qwen 3.6 27B 5-bit at prompt=128 shows the n-gram drafter
under-performing direct decode (`ngram_no_accept_fallback` cooldown).
The 512 and 2048 prompts on the same model recover; the 128 dip is
recorded as-is for transparency, not bolded.

### Time to first token (ms) — generation=128 tokens, temp=0

Lower is better. `mlx_lm` values are derived from reported prefill throughput.
AX values are measured directly from per-step runner timing in the SSE event
stream.

| Model | MLX quantization | Prompt tok | llama.cpp Metal* | mlx_lm | ax engine |
|---|---|---:| ---: |---:|---:|
| Gemma 4 E2B | 4-bit | 128 | 37.3 | 50.5 | **33.9 (-32.9%)** |
|         |         | 512 | 71.9 | 62.1 | **61.9 (-0.4%)** |
|         |         | 2048 | 290.0 | 114.7 | 255.6 (+122.8%) |
| Gemma 4 E2B | 5-bit | 128 | 37.8 | 53.3 | **33.8 (-36.6%)** |
|         |         | 512 | 71.4 | 63.5 | **62.0 (-2.3%)** |
|         |         | 2048 | 285.8 | 115.5 | 272.5 (+135.8%) |
| Gemma 4 E2B | 6-bit | 128 | 35.8 | 52.4 | **34.1 (-35.0%)** |
|         |         | 512 | 69.5 | 65.5 | **62.5 (-4.5%)** |
|         |         | 2048 | 279.2 | 117.9 | 268.3 (+127.6%) |
| Gemma 4 E2B | 8-bit | 128 | 37.0 | 55.8 | **34.9 (-37.4%)** |
|         |         | 512 | 65.8 | 68.1 | **63.6 (-6.7%)** |
|         |         | 2048 | 267.0 | 119.3 | 264.3 (+121.6%) |
| Gemma 4 E4B | 4-bit | 128 | 57.3 | 81.0 | **43.2 (-46.6%)** |
|         |         | 512 | 122.3 | 119.1 | **110.7 (-7.1%)** |
|         |         | 2048 | 468.3 | 278.2 | 434.5 (+56.2%) |
| Gemma 4 26B A4B | 4-bit | 128 | 66.0 | 175.3 | **98.2 (-44.0%)** |
|         |         | 512 | 150.4 | 244.3 | **163.8 (-33.0%)** |
|         |         | 2048 | 614.6 | 526.0 | **463.7 (-11.8%)** |
| Gemma 4 31B | 4-bit | 128 | 252.8 | 350.6 | **196.5 (-44.0%)** |
|         |         | 512 | 784.2 | 801.0 | **656.3 (-18.1%)** |
|         |         | 2048 | 3,388.2 | 2,761.8 | **2,739.7 (-0.8%)** |
| Qwen 3.6 27B | 4-bit | 128 | 236.0 | 296.8 | **167.8 (-43.5%)** |
|  |  | 512 | 669.7 | 679.1 | **559.5 (-17.6%)** |
|  |  | 2048 | 3,453.5 | 2,216.3 | **2,209.7 (-0.3%)** |
| Qwen 3.6 27B | 5-bit | 128 | 242.6 | 325.2 | **177.2 (-45.5%)** |
|  |  | 512 | 689.5 | 734.3 | **595.7 (-18.9%)** |
|  |  | 2048 | 3,346.8 | 2,359.4 | 2,375.1 (+0.7%) |
| Qwen 3.6 35B A3B | 4-bit | 128 | n/a | 222.6 | **112.8 (-49.3%)** |
|         |         | 512 | n/a | 300.8 | **180.1 (-40.1%)** |
|         |         | 2048 | n/a | — | — |

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
