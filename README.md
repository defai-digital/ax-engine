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

<!-- readme-performance-artifacts: base=benchmarks/results/mlx-inference/2026-05-18-ax-only-sweep/; reference=benchmarks/results/mlx-inference/2026-05-18-mlx-lm-llamacpp-sweep/; ax-overlay=benchmarks/results/mlx-inference/2026-05-19-ax-only-readme-refresh-gemma-ffn-compile-off/; ax-overlay=benchmarks/results/mlx-inference/2026-05-19-ax-only-readme-refresh-except-qwen35-r3/; ax-overlay=benchmarks/results/mlx-inference/2026-05-19-ax-qwen35-a3b-4bit-cold-prefill-refresh/ -->
The README keeps the common Gemma 4 and Qwen 3.6 generation benchmark rows
visible. Full result tables and interpretation live in
[`docs/PERFORMANCE.md`](docs/PERFORMANCE.md); benchmark methodology, test setup,
and reproduction details live in [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

These rows are a provenance-tracked composite. The `mlx_lm` reference rows for
the non-Qwen-35B rows come from
`benchmarks/results/mlx-inference/2026-05-18-mlx-lm-llamacpp-sweep/`. The
refreshed AX rows for Gemma 4 E2B 4/5/6/8-bit, Gemma 4 E4B 4-bit,
and Gemma 4 26B A4B 4-bit come from
`benchmarks/results/mlx-inference/2026-05-19-ax-only-readme-refresh-gemma-ffn-compile-off/`
and were rerun with the benchmark kill-switch
`AX_MLX_PREFILL_FFN_COMPILE=0` after the first default-compile benchmark
disconnected the server; the harness surfaced that failed run as
`RemoteDisconnected`. The fallback artifacts completed successfully and are
the values shown below.
The refreshed AX rows for Gemma 4 31B and Qwen 3.6 27B come from
`benchmarks/results/mlx-inference/2026-05-19-ax-only-readme-refresh-except-qwen35-r3/`.
The `llama.cpp Metal*` column is injected from
`benchmarks/manifests/llama_cpp_metal/inventory.json` and the
`benchmarks/results/mlx-inference/2026-05-18-llama-cpp-metal-gemma-e2b-4bit-depth-fa/`
Gemma 4 E2B 4-bit recheck. Refreshed Qwen 3.6 35B A3B 4-bit `mlx_lm`,
AX direct, and n-gram rows come from
`benchmarks/results/mlx-inference/2026-05-19-ax-qwen35-a3b-4bit-cold-prefill-refresh/`
with AX prefix cache disabled for cold prefill and TTFT measurement. The
refreshed non-Qwen-35B AX rows use generation=128, 3 repetitions, a 15-second
cooldown between trials, and production-build binaries; the Qwen 3.6 35B A3B
refresh uses 5 repetitions. MLX and AX rows also use matching prompt SHA checks.
Percentages are versus `mlx_lm`. The `llama.cpp Metal*` column is a
shape-compatible external reference; it does not share prompt-token hashes with
the MLX rows.

<!-- llama-cpp-column-disclaimer -->
**`llama.cpp Metal*` column** — Shape-compatible reference produced by Metal-enabled `llama-bench`. `llama-bench` generates its own internal synthetic prompt tokens and does not consume the harness prompt JSON, so these numbers are NOT prompt-hash parity with the other columns. The intent is rough side-by-side context against a well-known third-party Metal runtime, not head-to-head comparison. MLX bit-widths are mapped to the nearest standard bartowski GGUF K-quant (4→Q4_K_M, 5→Q5_K_M, 6→Q6_K, 8→Q8_0). No percentage delta is shown for this column because the prompt is not shared. Source: `benchmarks/manifests/llama_cpp_metal/inventory.json`, `scripts/bench_llama_cpp_metal_sweep.py`.

Note: The 2K `llama.cpp Metal*` prefill rows are long-context,
GGUF-runtime-reference rows, not MLX parity claims. Across this dataset, the 2K
`llama.cpp Metal*` prefill column commonly trails the MLX/AX rows, with the
largest gap on Gemma 4 E2B. The Gemma 4 E2B 4-bit row was produced with
`llama.cpp` b9110 (`ef22b3e4a`) and rechecked on b9200 (`3e12fbdea`) with Metal
offload, `-b/-ub 2048`, and flash attention enabled. The b9200 recheck improved
2K prefill only slightly, and no missing benchmark flag was found. This is our
benchmark boundary, not an upstream `llama.cpp` official bug statement.

### Prefill throughput (tok/s) — percentages vs mlx_lm

| Model | MLX quantization | Prompt tok | llama.cpp Metal* | mlx_lm | ax engine |
|---|---|---:| ---: |---:|---:|
| Gemma 4 E2B | 4-bit | 128 | 3,481.7 | 2,338.1 | **4,098.8 (+75.3%)** |
|         |         | 512 | 6,846.0 | 7,870.0 | **8,862.3 (+12.6%)** |
|         |         | 2048 | 7,643.1 | 18,014.7 | 8,339.2 (-53.7%) |
| Gemma 4 E2B | 5-bit | 128 | 3,398.4 | 2,238.5 | **3,966.3 (+77.2%)** |
|         |         | 512 | 6,860.3 | 7,469.9 | **8,887.8 (+19.0%)** |
|         |         | 2048 | 7,288.1 | 16,664.1 | 8,041.9 (-51.7%) |
| Gemma 4 E2B | 6-bit | 128 | 3,539.7 | 1,823.5 | **3,907.3 (+114.3%)** |
|         |         | 512 | 7,274.0 | 6,046.6 | **8,705.3 (+44.0%)** |
|         |         | 2048 | 7,623.2 | 15,332.1 | 7,966.8 (-48.0%) |
| Gemma 4 E2B | 8-bit | 128 | 3,694.3 | 1,605.0 | **3,841.5 (+139.3%)** |
|         |         | 512 | 7,481.0 | 6,332.9 | **8,589.6 (+35.6%)** |
|         |         | 2048 | 7,990.4 | 15,536.8 | 8,073.7 (-48.0%) |
| Gemma 4 E4B | 4-bit | 128 | 2,194.0 | 1,513.2 | **3,156.6 (+108.6%)** |
|         |         | 512 | 4,454.2 | 4,195.5 | **4,942.8 (+17.8%)** |
|         |         | 2048 | 4,426.6 | 7,325.4 | 4,839.7 (-33.9%) |
| Gemma 4 26B A4B | 4-bit | 128 | 1,911.4 | 496.4 | **1,457.3 (+193.5%)** |
|         |         | 512 | 3,484.5 | 1,621.0 | **3,091.8 (+90.7%)** |
|         |         | 2048 | 3,604.8 | 3,300.1 | **4,505.9 (+36.5%)** |
| Gemma 4 31B | 4-bit | 128 | 522.6 | 283.1 | **672.8 (+137.7%)** |
|         |         | 512 | 665.3 | 619.9 | **789.2 (+27.3%)** |
|         |         | 2048 | 560.3 | 733.9 | 720.6 (-1.8%) |
| Qwen 3.6 27B | 4-bit | 128 | 539.6 | 378.8 | **788.3 (+108.1%)** |
|  |  | 512 | 759.7 | 705.7 | **900.6 (+27.6%)** |
|  |  | 2048 | 664.3 | 895.2 | **908.6 (+1.5%)** |
| Qwen 3.6 27B | 5-bit | 128 | 520.8 | 278.8 | **741.5 (+165.9%)** |
|  |  | 512 | 733.4 | 599.5 | **851.0 (+42.0%)** |
|  |  | 2048 | 667.0 | 827.5 | **863.0 (+4.3%)** |
| Qwen 3.6 27B | 6-bit | 128 | 537.7 | 270.5 | **730.4 (+170.0%)** |
|  |  | 512 | 756.1 | 577.6 | **863.6 (+49.5%)** |
|  |  | 2048 | 689.3 | 798.2 | **866.8 (+8.6%)** |
| Qwen 3.6 27B | 8-bit | 128 | 559.4 | 219.3 | **719.8 (+228.1%)** |
|  |  | 512 | 798.2 | 520.2 | **877.7 (+68.7%)** |
|  |  | 2048 | 741.9 | 787.4 | **879.8 (+11.7%)** |
| Qwen 3.6 35B A3B | 4-bit | 128 | 1,706.9 | 539.4 | **1,358.0 (+151.8%)** |
|  |  | 512 | 3,146.6 | 1,599.5 | **3,066.7 (+91.7%)** |
|  |  | 2048 | 3,542.3 | 3,513.1 | 3,108.4 (-11.5%) |

### Decode throughput (tok/s) — generation=128 tokens, temp=0

Higher is better. `ax direct baseline` disables n-gram acceleration.
`ax default n-gram` is the default AX decode policy and reports observed
effective throughput, not raw model-kernel speed.

The bench prompts are `mlx_lm.benchmark` seed-0 random tokens, which is
the only way to keep prompt-hash parity across all four columns. The
n-gram column is sensitive to workload shape — published benchmarks
(Saxena 2024, vLLM, SpecDecode-Bench 2025, EfficientEdit 2025) all
report n-gram speculative decoding is an input-output overlap
technique: code editing / refactoring / summarization see large
speedups; fresh code generation and open-ended chat see modest
speedups or none.
[`docs/NGRAM-ACCELERATION.md`](docs/NGRAM-ACCELERATION.md) covers how
the drafter works, when each workload regime is expected to accelerate,
the
[when-it-helps section](docs/NGRAM-ACCELERATION.md#when-n-gram-acceleration-helps)
with literature citations and our own random-vs-real measurements,
and the
[decode-degeneracy](docs/NGRAM-ACCELERATION.md#decode-degeneracy-on-synthetic-benchmark-prompts)
validator that flags rows whose throughput was measured on a collapsed
output loop.

| Model | MLX quantization | Prompt tok | llama.cpp Metal* | mlx_lm | ax direct baseline | ax default n-gram |
|---|---|---:| ---: |---:|---:|---:|
| Gemma 4 E2B | 4-bit | 128 | 174.6 | 214.0 | 190.2 (-11.1%) | **492.2 (+130.0%)** |
|  |  | 512 | 165.2 | 210.3 | 183.0 (-13.0%) | **485.7 (+131.0%)** |
|  |  | 2048 | 171.9 | 200.9 | 175.8 (-12.5%) | **449.9 (+123.9%)** |
| Gemma 4 E2B | 5-bit | 128 | 154.8 | 195.2 | 172.7 (-11.5%) | **398.9 (+104.4%)** |
|  |  | 512 | 154.3 | 182.0 | 167.1 (-8.2%) | **394.4 (+116.6%)** |
|  |  | 2048 | 154.3 | 181.9 | 161.3 (-11.3%) | **370.6 (+103.7%)** |
| Gemma 4 E2B | 6-bit | 128 | 152.1 | 172.2 | 156.3 (-9.2%) | **368.5 (+114.0%)** |
|  |  | 512 | 152.0 | 166.3 | 151.4 (-9.0%) | **364.3 (+119.0%)** |
|  |  | 2048 | 152.2 | 162.5 | 146.7 (-9.7%) | **344.4 (+112.0%)** |
| Gemma 4 E2B | 8-bit | 128 | 136.1 | 153.0 | 139.1 (-9.1%) | **388.3 (+153.7%)** |
|  |  | 512 | 138.3 | 148.8 | 135.1 (-9.2%) | **382.3 (+156.9%)** |
|  |  | 2048 | 138.7 | 144.2 | 131.3 (-8.9%) | **359.5 (+149.3%)** |
| Gemma 4 E4B | 4-bit | 128 | 110.7 | 137.1 | 122.1 (-10.9%) | **310.8 (+126.8%)** |
|  |  | 512 | 110.8 | 133.6 | 119.1 (-10.8%) | **302.1 (+126.2%)** |
|  |  | 2048 | 110.7 | 130.6 | 116.4 (-10.8%) | **289.8 (+122.0%)** |
| Gemma 4 26B A4B | 4-bit | 128 | 112.6 | 127.9 | 121.3 (-5.2%) | **141.1 (+10.3%)** |
|  |  | 512 | 112.9 | 125.0 | 118.1 (-5.5%) | **195.0 (+56.0%)** |
|  |  | 2048 | 112.9 | 119.3 | 113.1 (-5.2%) | **237.0 (+98.6%)** |
| Gemma 4 31B | 4-bit | 128 | 25.0 | 28.9 | 26.8 (-7.3%) | **58.3 (+102.0%)** |
|  |  | 512 | 25.5 | 28.3 | 26.5 (-6.2%) | **56.2 (+98.4%)** |
|  |  | 2048 | 25.3 | 27.0 | 25.2 (-7.0%) | **53.5 (+97.9%)** |
| Qwen 3.6 27B | 4-bit | 128 | 26.0 | 34.0 | 31.9 (-6.2%) | **31.6 (-7.0%)** |
|  |  | 512 | 26.0 | 33.9 | 31.5 (-7.1%) | **31.5 (-7.2%)** |
|  |  | 2048 | 18.8 | 33.4 | 31.2 (-6.8%)† | **31.1 (-7.0%)**† |
| Qwen 3.6 27B | 5-bit | 128 | 23.5 | 21.6 | **26.5 (+22.7%)** | **26.3 (+21.7%)**‡ |
|  |  | 512 | 23.3 | 28.1 | 26.4 (-6.2%) | **27.1 (-3.8%)** |
|  |  | 2048 | 17.8 | 27.8 | 26.4 (-5.0%) | **26.9 (-3.4%)** |
| Qwen 3.6 27B | 6-bit | 128 | 21.3 | 24.0 | **24.4 (+1.6%)** | **24.1 (+0.5%)** |
|  |  | 512 | 21.3 | 24.8 | 24.3 (-1.8%) | **24.1 (-2.9%)** |
|  |  | 2048 | 15.4 | 24.6 | 24.1 (-1.9%) | **23.9 (-2.9%)** |
| Qwen 3.6 27B | 8-bit | 128 | 18.3 | 18.7 | 18.1 (-2.9%) | **18.0 (-3.5%)** |
|  |  | 512 | 18.2 | 18.6 | 18.1 (-2.8%) | **17.9 (-3.8%)** |
|  |  | 2048 | 12.7 | 18.4 | 17.9 (-2.5%) | **17.8 (-3.3%)** |
| Qwen 3.6 35B A3B | 4-bit | 128 | 108.1 | 140.1 | **148.8 (+6.2%)** | **340.0 (+142.7%)** |
|  |  | 512 | 108.2 | 136.5 | **149.9 (+9.8%)** | **334.6 (+145.2%)** |
|  |  | 2048 | 105.7 | 134.5 | **143.3 (+6.6%)** | **312.7 (+132.5%)** |

† Qwen 3.6 27B 4-bit at prompt=2048 originally produced zero decode tokens
because 4-bit quantization noise pushed an EOS token to argmax at decode
step 0 on the `mlx_lm.benchmark` random-token contract. The cell is now
measured with the AX server's `AX_MLX_IGNORE_EOS=1` env override
(benchmark-only knob added at `crates/ax-engine-mlx/src/runner.rs`), which
empties the terminal-token list so EOS no longer stops decode — matching
how `mlx_lm.benchmark` measures fixed `gen=N` throughput regardless of
stop-token argmax. Production deployments without that env var will still
honor EOS at step 0 on this specific synthetic prompt. Source:
`benchmarks/results/mlx-inference/2026-05-19-ax-only-readme-refresh-except-qwen35-r3/qwen3_6-27b-4bit.json`.

‡ Qwen 3.6 27B 5-bit at prompt=128 shows the n-gram drafter
under-performing direct decode on this refreshed run.
The 512 and 2048 prompts on the same model recover; the 128 dip is
recorded as-is for transparency.

### Time to first token (ms) — generation=128 tokens, temp=0

Lower is better. `mlx_lm` values are derived from reported prefill throughput.
AX values are measured directly from per-step runner timing in the SSE event
stream.

| Model | MLX quantization | Prompt tok | llama.cpp Metal* | mlx_lm | ax engine |
|---|---|---:| ---: |---:|---:|
| Gemma 4 E2B | 4-bit | 128 | 36.8 | 54.7 | **31.2 (-43.0%)** |
|         |         | 512 | 74.8 | 65.1 | **57.8 (-11.2%)** |
|         |         | 2048 | 268.0 | 113.7 | 245.6 (+116.0%) |
| Gemma 4 E2B | 5-bit | 128 | 37.7 | 57.2 | **32.3 (-43.6%)** |
|         |         | 512 | 74.6 | 68.5 | **57.6 (-16.0%)** |
|         |         | 2048 | 281.0 | 122.9 | 254.7 (+107.2%) |
| Gemma 4 E2B | 6-bit | 128 | 36.2 | 70.2 | **32.8 (-53.3%)** |
|         |         | 512 | 70.4 | 84.7 | **58.8 (-30.5%)** |
|         |         | 2048 | 268.7 | 133.6 | 257.1 (+92.5%) |
| Gemma 4 E2B | 8-bit | 128 | 34.6 | 79.7 | **33.3 (-58.2%)** |
|         |         | 512 | 68.4 | 80.8 | **59.6 (-26.3%)** |
|         |         | 2048 | 256.3 | 131.8 | 253.7 (+92.4%) |
| Gemma 4 E4B | 4-bit | 128 | 58.3 | 84.6 | **40.6 (-52.1%)** |
|         |         | 512 | 114.9 | 122.0 | **103.6 (-15.1%)** |
|         |         | 2048 | 462.7 | 279.6 | 423.2 (+51.4%) |
| Gemma 4 26B A4B | 4-bit | 128 | 67.0 | 257.8 | **87.8 (-65.9%)** |
|         |         | 512 | 146.9 | 315.8 | **165.6 (-47.6%)** |
|         |         | 2048 | 568.1 | 620.6 | **454.5 (-26.8%)** |
| Gemma 4 31B | 4-bit | 128 | 244.9 | 452.2 | **190.2 (-57.9%)** |
|         |         | 512 | 769.5 | 826.0 | **648.7 (-21.5%)** |
|         |         | 2048 | 3,655.2 | 2,790.6 | 2,842.0 (+1.8%) |
| Qwen 3.6 27B | 4-bit | 128 | 237.2 | 337.9 | **162.4 (-51.9%)** |
|  |  | 512 | 673.9 | 725.6 | **568.5 (-21.6%)** |
|  |  | 2048 | 3,083.1 | 2,287.7 | **2,254.1 (-1.5%)** |
| Qwen 3.6 27B | 5-bit | 128 | 245.8 | 459.0 | **172.6 (-62.4%)** |
|  |  | 512 | 698.1 | 854.1 | **601.7 (-29.6%)** |
|  |  | 2048 | 3,070.5 | 2,474.9 | **2,373.0 (-4.1%)** |
| Qwen 3.6 27B | 6-bit | 128 | 238.1 | 473.2 | **175.3 (-63.0%)** |
|  |  | 512 | 677.2 | 886.5 | **592.9 (-33.1%)** |
|  |  | 2048 | 2,971.2 | 2,565.6 | **2,362.8 (-7.9%)** |
| Qwen 3.6 27B | 8-bit | 128 | 228.8 | 583.6 | **177.8 (-69.5%)** |
|  |  | 512 | 641.5 | 984.2 | **583.3 (-40.7%)** |
|  |  | 2048 | 2,760.6 | 2,601.1 | **2,327.9 (-10.5%)** |
| Qwen 3.6 35B A3B | 4-bit | 128 | 75.0 | 237.3 | **94.3 (-60.3%)** |
|  |  | 512 | 162.7 | 320.1 | **167.0 (-47.8%)** |
|  |  | 2048 | 578.2 | 583.0 | 658.9 (+13.0%) |

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
