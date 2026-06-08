# AX Engine

AX Engine is a Mac-first LLM inference runtime, local server, SDK layer, and benchmark toolkit for Apple Silicon. It runs direct-support MLX model families natively, and routes other MLX text models or non-MLX models through explicit `mlx-lm` and `llama.cpp` compatibility routes.

## Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [Getting a Model](#getting-a-model)
- [Typical Hardware](#typical-hardware)
- [What AX Engine Does](#what-ax-engine-does)
- [Supported Models](#supported-models)
- [Performance](#performance)
  - [Gemma 4 12B](#gemma-4-12b)
  - [Speculative Decoding (MTP)](#speculative-decoding-mtp)
    - [Gemma 4](#gemma-4)
    - [Qwen 3.6](#qwen-36)
  - [Direct Decode · Prefill · TTFT](#direct-decode--prefill--ttft)
- [SDKs](#sdks)
- [Server Usage](#server-usage)
- [Workspace](#workspace)
- [Development](#development)
- [Benchmark Reference Projects](#benchmark-reference-projects)
- [Limitations](#limitations)
- [Contributing](#contributing)
- [Community](#community)
- [License](#license)

## Quick Start

**Install:**

```bash
pip install ax-engine
```

**Download a model and start the server:**

```bash
ax-engine serve qwen36-35b --download --port 8080
```

**Call it from any OpenAI client:**

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="local")
resp = client.chat.completions.create(
    model="my-model",
    messages=[{"role": "user", "content": "What is AGI?"}],
    max_tokens=128,
)
print(resp.choices[0].message.content)
```

**Or use the Python SDK directly:**

```python
from ax_engine import download_model, Session

path = download_model("mlx-community/Qwen3-4B-4bit")
with Session(mlx=True, mlx_model_artifacts_dir=str(path)) as s:
    print(s.generate([1, 2, 3], max_output_tokens=8).output_tokens)
```

> Requires **macOS 14 (Sonoma) or later** on **Apple Silicon M2 Max or newer** with **32 GB RAM minimum**.

## Installation

### Python (recommended)

```bash
pip install ax-engine
```

Requires macOS 14+, Apple Silicon (M2 Max or newer), Python 3.10+. The pip wheel includes the `ax-engine` orchestration CLI and the `ax-engine-server` binary — both are on your `PATH` after install.

Optional extras:

```bash
pip install "ax-engine[download]"  # Hugging Face Hub downloads plus mlx-lm tools
```

### Homebrew

For `ax-engine-bench` (workload-contract CLI), or as an alternative way to install `ax-engine-server`:

```bash
brew tap defai-digital/ax-engine
brew install ax-engine
ax-engine-bench doctor
```

If `doctor` fails with `Library not loaded: libmlxc.dylib`, run:
`brew install mlx-c && brew reinstall ax-engine`.

### Source

```bash
brew install mlx-c
cargo build --workspace --release
maturin develop  # Python bindings
```

## Getting a Model

AX Engine requires pre-sanitized MLX weights. The recommended source is [mlx-community](https://huggingface.co/mlx-community) — models there are already converted and validated.

### mlx-community (recommended)

`ax-engine download`, `download_model()`, and `scripts/download_model.py` download weights and auto-generate the required `model-manifest.json` in one step:

```bash
# List supported download targets
ax-engine download --list

# Download by alias
ax-engine download qwen36-35b --json
ax-engine download qwen36-27b --json
ax-engine download gemma4-e2b --json
ax-engine download gemma4-12b --json
ax-engine download gemma4-31b --json

# Download and serve in one command
ax-engine serve qwen36-35b --download --port 8080

# Raw mlx-community repo IDs are also accepted
ax-engine download mlx-community/Qwen3.6-35B-A3B-4bit --json
ax-engine download mlx-community/gemma-4-e2b-it-4bit --json

# Optional: copy snapshot to an explicit directory
ax-engine download qwen36-35b --dest /Volumes/Models/qwen36-35b

# Python SDK
from ax_engine import download_model
path = download_model("mlx-community/Qwen3.6-35B-A3B-4bit")
```

Built-in download aliases:

| Alias | Repo |
|---|---|
| `qwen36-35b` | `mlx-community/Qwen3.6-35B-A3B-4bit` |
| `qwen36-27b`, `qwen36-27b-5bit`, `qwen36-27b-6bit`, `qwen36-27b-8bit` | `mlx-community/Qwen3.6-27B-{4,5,6,8}bit` |
| `gemma4-e2b`, `gemma4-e2b-5bit`, `gemma4-e2b-6bit`, `gemma4-e2b-8bit` | `mlx-community/gemma-4-e2b-it-{4,5,6,8}bit` |
| `gemma4-12b`, `gemma4-12b-6bit` | `mlx-community/gemma-4-12B-it-{4,6}bit` |
| `gemma4-31b` | `mlx-community/gemma-4-31b-it-4bit` |

Leave downloads in the Hugging Face Hub cache by default — it's shared with `mlx_lm` and other HF-aware tools, avoiding duplicate copies of large weights. Use `--dest` only when you want an explicit copy outside the shared cache.

If you already have `mlx_lm` installed, its downloads land in the same cache and AX Engine can auto-discover them:

```bash
python -m mlx_lm.generate --model mlx-community/Qwen3-4B-4bit --prompt "x" --max-tokens 1
ax-engine-bench generate-manifest ~/.cache/huggingface/hub/models--mlx-community--Qwen3-4B-4bit/snapshots/<hash>
ax-engine serve ~/.cache/huggingface/hub/models--mlx-community--Qwen3-4B-4bit/snapshots/<hash> --port 8080
```

### Raw HuggingFace checkpoint

Raw checkpoints need sanitization before AX Engine can load them:

```bash
pip install mlx-lm
mlx_lm.convert --hf-path <org/model> --mlx-path /path/to/dest -q --q-bits 4
ax-engine-bench generate-manifest /path/to/dest
ax-engine serve /path/to/dest --port 8080
```

### Manifest generation

Both paths above require `model-manifest.json`. Download helpers generate it automatically. To run it directly:

```bash
ax-engine-bench generate-manifest /path/to/model      # pip, Homebrew, or built binary
cargo run -p ax-engine-core --bin generate-manifest -- /path/to/model  # source
```

## Typical Hardware

For local agent and chatbot workloads, AX Engine is a better fit for a small model portfolio than for one model serving every workflow. See the [FAQ model-stack guidance](docs/FAQ.md#what-model-stack-should-i-run-on-high-memory-apple-silicon) for the full recommendation.

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

## What AX Engine Does

AX Engine gives local inference work a stable runtime contract:

- **Repo-owned MLX execution** tracks [direct-support model families](#supported-models) separately from delegated routes — delegated results are not AX-owned throughput claims.
- **Dual-family speculative decoding** supports both Qwen3.6's fused MTP sidecar and Gemma 4's separate assistant-drafter contract in the same repo-owned runtime and benchmark tooling.
- **N-gram acceleration** reaches up to 3.1× mlx_lm decode throughput on high-hit benchmark rows with no second draft model.
- **Long-session prefix reuse** restores physical MLX KV snapshots on validated cache layouts, so long-running chat and agent loops avoid repeatedly pre-filling accumulated context. See [`docs/LONG-CONTEXT.md`](docs/LONG-CONTEXT.md).
- **Workload-contract tooling** (`ax-engine-bench`) validates correctness, determinism, route identity, and regression across checked-in manifests.
- **Delegated routes** (`mlx_lm_delegated`, `llama_cpp`) cover explicit compatibility cases without polluting AX-owned performance claims.

[mlx_lm](https://github.com/ml-explore/mlx-lm) is the canonical MLX reference. AX Engine compares against `mlx_lm.benchmark` and keeps `mlx_lm.server` as the explicit delegated compatibility route when AX does not yet have a repo-owned graph. See the [FAQ](docs/FAQ.md#is-ax-faster-because-it-replaces-mlx-kernels) for the boundary between MLX kernels and AX-owned runtime behavior.

Design details: [Scheduler](docs/SCHEDULER.md) · [KV Cache](docs/KV-CACHE.md) · [Long Context](docs/LONG-CONTEXT.md) · [Benchmark Design](docs/BENCH-DESIGN.md).

### Runtime Paths

| Path | Use it for | Current scope |
|---|---|---|
| Repo-owned MLX runtime | Direct-support MLX model families and repo-owned performance claims backed by benchmark artifacts | Local Apple Silicon inference, token-based server/SDK requests, direct and n-gram acceleration modes |
| `mlx_lm_delegated` | MLX text models that upstream `mlx-lm` supports before AX has a repo-owned graph | Blocking and SSE text generation through a user-provided `mlx_lm.server`; not AX-owned token/KV performance |
| `llama_cpp` | GGUF and non-MLX local inference | Delegated llama.cpp server/CLI compatibility; route-contract evidence, not repo-owned MLX throughput |

The runtime report exposes `selected_backend`, `support_tier`, and `resolution_policy` so callers and benchmark artifacts can distinguish these paths. For the exact OpenAI-shaped endpoint contract see `docs/API-COMPATIBILITY.md`.

## Supported Models

Direct support means AX has a repo-owned `ax-engine-mlx` graph for the model family and loads MLX safetensors through the AX manifest path. Other MLX text models can still use the explicit `mlx_lm_delegated` compatibility route.

| Family | Direct model IDs | Current scope | Architecture notes |
|---|---|---|---|
| Gemma 4 | `gemma-4-e2b-it`, `gemma-4-e4b-it`, `gemma-4-12b-it`, `gemma-4-26b-a4b-it`, `gemma-4-31b-it` | Repo-owned MLX runtime; MLX affine 4/5/6/8-bit weights; assistant-MTP benchmark path | Dense unified 12B, per-layer embedding, and MoE variants; sliding-window + full attention, logit softcapping |
| Qwen 3 | `Qwen3-4B-4bit` and manifest-backed dense checkpoints | Repo-owned MLX runtime | SwiGLU dense FFN; per-head QK norm |
| Qwen 3.5 | `Qwen3.5-9B-MLX-4bit` | Repo-owned MLX runtime | Linear attention + MoE FFN; `attn_output_gate` per-head interleaving |
| Qwen 3.6 / Coder Next | `Qwen3.6-35B-A3B` 4-bit, `Qwen3.6-27B` 4/5/6/8-bit, `Qwen3-Coder-Next-4bit` | Repo-owned MLX runtime | `qwen3_next`: GatedDelta linear attention, full attention with per-head sigmoid gate, sparse top-k MoE |

> GLM 4.7 Flash (`glm4_moe_lite`) was demoted from direct support to the `mlx_lm_delegated` passby route: native decode only reaches `mlx_lm` parity and the 4-bit export has no MTP head. The `glm4.7-flash-4bit` preset now selects the delegated tier and requires `--mlx-lm-server-url`. See [`docs/SUPPORTED-MODELS.md`](docs/SUPPORTED-MODELS.md).

Adding a new architecture means implementing the model graph in `ax-engine-mlx`, not wiring up a generic loader. Architecture code alone is not a direct-support claim — a model requires a repo-owned graph, manifest, smoke coverage, and benchmark evidence before promotion here. LLaMA, Mistral, Mixtral, DeepSeek, and unlisted Gemma/Qwen variants should use the explicit delegated route.

Before promoting another architecture or checkpoint, run `scripts/probe_mlx_model_support.py --model-dir <model-dir>`; a model should report `repo_owned_runtime_ready` only when its manifest, local reference files, and runtime path are all present.

Full list: [`docs/SUPPORTED-MODELS.md`](docs/SUPPORTED-MODELS.md).

## Performance

Full result tables and interpretation live in [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md). Benchmark methodology, test setup, and reproduction details live in [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).
### Gemma 4 12B

Gemma 4 12B (`model_type: gemma4_unified`) is a different implementation from the per-layer-embedding E2B/E4B and the MoE 26B/31B. **Upstream `mlx_lm` 0.31.3 cannot load it** — it fails with `ValueError: Model type gemma4_unified not supported`. The external reference here is **llama.cpp Metal** on a shape-compatible GGUF.

> [!NOTE]
> **AX Engine currently supports text-only input for Gemma 4 12B.** Image and audio modalities are planned for **v6.2.0**.

**AX beats llama.cpp Metal on this model in both modes.** In **direct** decode, AX runs **68 tok/s** on a bit-comparable 4-bit-FFN artifact vs llama.cpp's **57–61** (depth-matched), and the margin grows with context (+12% at 128 tokens → +26% at 2,048). On top of that, **depth-2 assistant-MTP** — which `mlx_lm` can't run and llama.cpp doesn't have — holds **62–69 tok/s at ≥97.6% assistant accept**. The earlier story (llama.cpp ahead by ~30%) was an artifact handicap: the upstream snapshot keeps the FFN at 8-bit and so reads ~1.5× the weight bytes; decode is bandwidth-bound, so matching the quantization closes the gap (see the bandwidth table below).

**Direct decode — AX native MLX vs llama.cpp Metal (mlx_lm N/A):**

<table>
<tr>
<td><img width="100%" src="docs/assets/perf-gemma4-12b-direct-decode-tok-s.svg" alt="Grouped bar chart comparing Gemma 4 12B 4-bit median direct decode throughput for AX Engine native MLX and llama.cpp Metal at 128/512/2048 prompt tokens; mlx_lm is not available because it has no gemma4_unified graph"></td>
<td><img width="100%" src="docs/assets/perf-gemma4-12b-direct-prefill-tok-s.svg" alt="Grouped bar chart comparing Gemma 4 12B 4-bit median prefill throughput for AX Engine native MLX and llama.cpp Metal at 128/512/2048 prompt tokens"></td>
<td><img width="100%" src="docs/assets/perf-gemma4-12b-direct-ttft-ms.svg" alt="Grouped bar chart comparing Gemma 4 12B 4-bit median time to first token for AX Engine native MLX and llama.cpp Metal at 128/512/2048 prompt tokens"></td>
</tr>
</table>

| Prompt tokens | AX decode | llama.cpp decode (depth 0) | llama.cpp decode (matched depth) | AX prefill | llama.cpp prefill | AX TTFT (ms) | llama.cpp TTFT (ms) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 68.0 | 60.4 | 60.7 | 1,180 | 1,242 | 108 | 103 |
| 512 | 68.1 | 57.7 | 56.6 | 1,886 | 1,748 | 271 | 293 |
| 2048 | 63.8 | 58.3 | 50.6 | 2,035 | 1,640 | 1,007 | 1,249 |

AX wins decode at every prompt size, and the margin widens with context (+12% / +20% / +26% vs the matched-depth column). The two llama.cpp decode columns matter: plain `llama-bench tg` decodes from an **empty context** (depth 0 — its best case), while AX decodes *after* the prompt prefill; the **matched-depth** column (`-d {prompt} -n 128`) is the apples-to-apples figure, and llama.cpp slows more with depth (58 → 51 at 2,048). AX prefill also leads at 2,048 (2,035 vs 1,640). The `llama.cpp Metal` columns are a **shape-compatible external GGUF baseline** (ggml-org Q4_K_M); `mlx_lm` is **absent because it cannot load `gemma4_unified`**.

> This table uses the bit-comparable **4-bit-FFN** AX artifact (`scripts/requantize_gemma4_12b_ffn_4bit.py`), ~4.5 bpw vs the Q4_K_M GGUF's ~4.8 bpw. The upstream `mlx-community/gemma-4-12B-it-4bit` snapshot keeps the FFN at **8-bit** (~10.4 GiB, ~1.5× the FFN bytes) and trails llama.cpp at ~46 tok/s — that's a *bytes-read* handicap, not a runtime one; see the memory-bandwidth analysis next.

**Memory bandwidth utilization:**

Decode is memory-bandwidth-bound on Apple Silicon: each token reads the model weights once, so decode tok/s is set by bytes-read and how close the engine gets to the memory ceiling. Measured M5 Max GPU peak read bandwidth ≈ 577 GB/s (MLX reduction over a 6 GB array).

<img src="docs/assets/perf-gemma4-12b-bandwidth.svg" alt="Horizontal bar chart showing percentage of M5 Max GPU peak memory bandwidth consumed per decode token for AX 8-bit FFN (86%), AX 4-bit FFN (80%), llama.cpp depth-0 (77%), and llama.cpp depth-512 (72%)">

| Engine / quantization | Weights/token | Decode tok/s | Effective BW | % of 577 GB/s peak |
|---|---:|---:|---:|---:|
| AX — 8-bit FFN (upstream 4bit snapshot) | 10.98 GB | 45.0 | 494 GB/s | 86% |
| AX — 4-bit FFN (re-quantized) | 6.74 GB | 68.1 | 459 GB/s | 80% |
| llama.cpp Q4_K_M — decode @ depth 512 | 7.38 GB | 56.6 | 418 GB/s | 72% |
| llama.cpp Q4_K_M — decode @ depth 0 (`tg`) | 7.38 GB | 60.4 | 446 GB/s | 77% |

AX sustains **as much or more memory bandwidth than llama.cpp** (459 vs 418 GB/s at matched depth) — both near the hardware ceiling, so neither is bandwidth-starved and AX is not under-utilizing memory. The direct-decode gap is purely *bytes read*: the upstream snapshot keeps the FFN at 8-bit (~10.4 GiB, ~1.5× the FFN bytes of the Q4_K_M GGUF). Re-quantizing to uniform 4-bit group-64 (~6.3 GiB, ~4.5 bpw, bit-comparable to Q4_K_M's ~4.8 bpw) makes AX direct decode **68.1 vs 56.6 tok/s — beating llama.cpp** at a fair, depth-matched comparison, with output verified coherent. Build it with `scripts/requantize_gemma4_12b_ffn_4bit.py`. (8-bit weights saturate bandwidth slightly better — 86% vs 80% of peak — because 4-bit needs more dequant compute per byte; that ~6% headroom lives in MLX's `quantized_matmul` kernel, not AX's runtime.)

**Assistant-MTP speculative decode (depth 2):**

On top of the 4-bit-FFN direct win, the assistant-MTP path (depth-2 draft, default `0.999` confidence gate) runs on the assistant bundle and adds a second speculative lever `mlx_lm` and llama.cpp don't have:

<table>
<tr>
<td><img width="100%" src="docs/assets/perf-gemma4-assistant-mtp-12b-decode-tok-s.svg" alt="Grouped box-and-whisker plot comparing Gemma 4 12B 4-bit assistant-MTP and assistant MTP+n-gram decode throughput across flappy, long_code, and python_modules_long prompt suites"></td>
<td><img width="100%" src="docs/assets/perf-gemma4-assistant-mtp-12b-accept-rate.svg" alt="Grouped box-and-whisker plot comparing Gemma 4 12B 4-bit assistant-MTP and assistant MTP+n-gram accept rate across flappy, long_code, and python_modules_long prompt suites"></td>
</tr>
<tr>
<td><img width="100%" src="docs/assets/perf-gemma4-assistant-mtp-12b-prefill-tok-s.svg" alt="Grouped box-and-whisker plot comparing Gemma 4 12B 4-bit assistant-MTP and assistant MTP+n-gram prefill throughput across flappy, long_code, and python_modules_long prompt suites"></td>
<td><img width="100%" src="docs/assets/perf-gemma4-assistant-mtp-12b-ttft-ms.svg" alt="Grouped box-and-whisker plot comparing Gemma 4 12B 4-bit assistant-MTP and assistant MTP+n-gram time-to-first-token across flappy, long_code, and python_modules_long prompt suites"></td>
</tr>
</table>

| Suite | Depth | AX MTP tok/s | AX MTP accept | AX MTP+ngram tok/s | AX MTP+ngram accept | n-gram accept | n-gram hits |
|---|---:|---:|---:|---:|---:|---:|---:|
| flappy | 2 | 64.3 | 98.7% | 61.9 | 98.7% | 86.3% | 106 |
| long_code | 2 | 67.1 | 98.7% | 68.8 | 98.6% | 65.9% | 46 |
| python_modules_long | 2 | 62.7 | 98.0% | 62.5 | 97.6% | 82.0% | 67 |

**Prefill and TTFT — same run:**

| Suite | AX MTP prefill | AX MTP+ngram prefill | AX MTP ttft ms | AX MTP+ngram ttft ms |
|---|---:|---:|---:|---:|
| flappy | 1,827 | 1,792 | 197 | 202 |
| long_code | 1,971 | 1,992 | 405 | 400 |
| python_modules_long | 1,809 | 1,811 | 202 | 201 |

Direct rows: 4-bit-FFN artifact, greedy-equivalent sampler, 128 generated tokens, 5 repetitions, 15 s cooldown, random-token prompts (mlx_lm.benchmark contract); llama.cpp decode shown at depth 0 (`tg`) and at matched context depth (`-d {prompt}`). MTP rows: depth-2 draft, temperature=0.6, top_p=0.95, top_k=20; 1,000 generated tokens, 5 repetitions, 10 s / 5 s cooldowns. Apple M5 Max · AX Engine v6.0.1 · llama.cpp b9430 (Metal) · mlx_lm 0.31.3 (no `gemma4_unified` support).

Full artifacts: [`2026-06-08-gemma-4-12b-it-4bit-direct`](benchmarks/results/mlx-inference/2026-06-08-gemma-4-12b-it-4bit-direct/gemma-4-12b-it-4bit.json) (direct; llama.cpp GGUF provenance in [`llama_cpp_gguf_provenance.json`](benchmarks/results/mlx-inference/2026-06-08-gemma-4-12b-it-4bit-direct/llama_cpp_gguf_provenance.json)) · [`2026-06-08-gemma4-12b-assistant-mtp`](benchmarks/results/gemma4-assistant-mtp/2026-06-08-gemma4-12b-assistant-mtp/summary.json) (assistant-MTP).

<details>
<summary>Prepare Gemma 4 12B assistant-MTP artifacts</summary>

Gemma 4 12B MLX target and assistant repos are already converted to MLX safetensors — they do not go through `ax-engine convert-mtplx` or `scripts/prepare_mtp_sidecar.py`. Download the target and matching assistant, then package them with the Gemma-specific helper:

```bash
hf download mlx-community/gemma-4-12B-it-4bit
hf download mlx-community/gemma-4-12B-it-assistant-4bit
python3 scripts/prepare_gemma4_assistant_mtp.py \
  --target mlx-community/gemma-4-12B-it-4bit \
  --assistant mlx-community/gemma-4-12B-it-assistant-4bit

hf download mlx-community/gemma-4-12B-it-6bit
hf download mlx-community/gemma-4-12B-it-assistant-6bit
python3 scripts/prepare_gemma4_assistant_mtp.py \
  --target mlx-community/gemma-4-12B-it-6bit \
  --assistant mlx-community/gemma-4-12B-it-assistant-6bit
```

The default outputs are quant-specific synthetic HF cache snapshots: `models--ax-local--gemma-4-12b-it-4bit-assistant-mtp/snapshots/v1/` and `models--ax-local--gemma-4-12b-it-6bit-assistant-mtp/snapshots/v1/`. Each package contains the target files, an `assistant/` subtree, and `ax_gemma4_assistant_mtp.json`. Generate or validate the AX manifest before serving:

```bash
ax-engine-bench generate-manifest \
  ~/.cache/huggingface/hub/models--ax-local--gemma-4-12b-it-4bit-assistant-mtp/snapshots/v1 \
  --validate
ax-engine-bench generate-manifest \
  ~/.cache/huggingface/hub/models--ax-local--gemma-4-12b-it-6bit-assistant-mtp/snapshots/v1 \
  --validate
```
</details>



### Speculative Decoding (MTP)

AX Engine's key Mac advantage is **dual-family speculative decoding** — it supports both Gemma 4's separate assistant-drafter contract and Qwen3.6's fused sidecar contract in one repo-owned runtime and benchmark surface. A single benchmark surface records route identity, sampler, prompt suite, cooldown, accept behavior, and artifact provenance so the two MTP families are comparable without pretending they use the same architecture.

#### Gemma 4

Unlike Qwen's fused `mtp.*` sidecar, Gemma 4's multi-token prediction uses a small **assistant drafter** that shares the target's tokenizer and embedding table, drafts one token per step from the target's last-layer hidden state, and attends to the target's own KV cache. AX runs it assistant-MTP-only (`mtp`) and with n-gram stacked on top (`mtp-ngram`).

A **draft confidence gate** (`AX_MLX_GEMMA4_ASSISTANT_MTP_DRAFT_MIN_CONFIDENCE`, default `0.999`) only proposes a draft when the drafter's top-token probability clears the threshold, keeping accept high while remaining correctness-preserving. Lower the gate toward `0` for more speculation on less predictable content.

No peer engine (MTPLX, Rapid-MLX, lightning-mlx) exposes a runnable Gemma 4 assistant-MTP path, so this benchmark has no peer comparison rows.

**Gemma 4 speculative decoding holds draft accept ≥98% on every cell below** (98.4–99.5% across 26B / 31B × {MTP, MTP+n-gram} × {flappy, long_code, python_modules_long}).


<table>
<tr>
<td align="center"><strong>Gemma 4 26B A4B 4-bit</strong></td>
<td align="center"><strong>Gemma 4 31B 4-bit</strong></td>
</tr>
<tr>
<td><img width="100%" src="docs/assets/perf-gemma4-assistant-mtp-26b-decode-tok-s.svg" alt="Grouped box-and-whisker plot comparing Gemma 4 26B A4B 4-bit assistant-MTP and assistant MTP+n-gram decode throughput across flappy, long_code, and python_modules_long prompt suites"></td>
<td><img width="100%" src="docs/assets/perf-gemma4-assistant-mtp-31b-decode-tok-s.svg" alt="Grouped box-and-whisker plot comparing Gemma 4 31B 4-bit assistant-MTP and assistant MTP+n-gram decode throughput across flappy, long_code, and python_modules_long prompt suites"></td>
</tr>
<tr>
<td><img width="100%" src="docs/assets/perf-gemma4-assistant-mtp-26b-accept-rate.svg" alt="Grouped box-and-whisker plot comparing Gemma 4 26B A4B 4-bit assistant-MTP and assistant MTP+n-gram accept rate across flappy, long_code, and python_modules_long prompt suites"></td>
<td><img width="100%" src="docs/assets/perf-gemma4-assistant-mtp-31b-accept-rate.svg" alt="Grouped box-and-whisker plot comparing Gemma 4 31B 4-bit assistant-MTP and assistant MTP+n-gram accept rate across flappy, long_code, and python_modules_long prompt suites"></td>
</tr>
<tr>
<td><img width="100%" src="docs/assets/perf-gemma4-assistant-mtp-26b-prefill-tok-s.svg" alt="Grouped box-and-whisker plot comparing Gemma 4 26B A4B 4-bit assistant-MTP and assistant MTP+n-gram prefill throughput across flappy, long_code, and python_modules_long prompt suites"></td>
<td><img width="100%" src="docs/assets/perf-gemma4-assistant-mtp-31b-prefill-tok-s.svg" alt="Grouped box-and-whisker plot comparing Gemma 4 31B 4-bit assistant-MTP and assistant MTP+n-gram prefill throughput across flappy, long_code, and python_modules_long prompt suites"></td>
</tr>
<tr>
<td><img width="100%" src="docs/assets/perf-gemma4-assistant-mtp-26b-ttft-ms.svg" alt="Grouped box-and-whisker plot comparing Gemma 4 26B A4B 4-bit assistant-MTP and assistant MTP+n-gram time-to-first-token across flappy, long_code, and python_modules_long prompt suites"></td>
<td><img width="100%" src="docs/assets/perf-gemma4-assistant-mtp-31b-ttft-ms.svg" alt="Grouped box-and-whisker plot comparing Gemma 4 31B 4-bit assistant-MTP and assistant MTP+n-gram time-to-first-token across flappy, long_code, and python_modules_long prompt suites"></td>
</tr>
</table>

| Model | Suite | Depth | AX MTP tok/s | AX MTP accept | AX MTP+ngram tok/s | AX MTP+ngram accept |
|---|---|---:|---:|---:|---:|---:|
| Gemma 4 26B A4B 4-bit | flappy | 1 | 126.1 | 99.3% | 129.1 | 99.5% |
| Gemma 4 26B A4B 4-bit | long_code | 1 | 125.5 | 99.1% | 127.1 | 99.1% |
| Gemma 4 26B A4B 4-bit | python_modules_long | 1 | 124.0 | 98.5% | 124.1 | 98.6% |
| Gemma 4 31B 4-bit | flappy | 1 | 37.9 | 99.3% | 38.2 | 99.2% |
| Gemma 4 31B 4-bit | long_code | 1 | 37.8 | 99.2% | 38.9 | 99.1% |
| Gemma 4 31B 4-bit | python_modules_long | 1 | 37.4 | 98.4% | 37.3 | 98.6% |

**Prefill and TTFT — same run:**

| Model | Suite | AX MTP prefill | AX MTP+ngram prefill | AX MTP ttft ms | AX MTP+ngram ttft ms |
|---|---|---:|---:|---:|---:|
| Gemma 4 26B A4B 4-bit | flappy | 2,711 | 2,739 | 130 | 129 |
| Gemma 4 26B A4B 4-bit | long_code | 3,991 | 3,977 | 203 | 205 |
| Gemma 4 26B A4B 4-bit | python_modules_long | 2,909 | 2,912 | 131 | 131 |
| Gemma 4 31B 4-bit | flappy | 747 | 751 | 475 | 486 |
| Gemma 4 31B 4-bit | long_code | 769 | 798 | 1,035 | 995 |
| Gemma 4 31B 4-bit | python_modules_long | 739 | 728 | 477 | 481 |

The gated assistant already captures the speculation, so stacking n-gram on top adds little — the two modes track closely. Sampler: temperature=0.6, top_p=0.95, top_k=20; 1,000 generated tokens, 5 repetitions, 10 s / 5 s cooldowns. Apple M5 Max · AX Engine v6.0.1.

Full artifacts: [`2026-06-07-gemma4-assistant-mtp`](benchmarks/results/gemma4-assistant-mtp/2026-06-07-gemma4-assistant-mtp/summary.json).

<details>
<summary>Reproduce this benchmark</summary>

```bash
python3 scripts/bench_gemma4_assistant_mtp.py \
  --models 26b-a4b-4bit,31b-4bit \
  --modes mtp,mtp-ngram \
  --suites flappy,long_code,python_modules_long \
  --max-tokens 1000 --repetitions 5
python3 scripts/render_gemma4_assistant_mtp_charts.py \
  --results-dir benchmarks/results/gemma4-assistant-mtp/<run-dir>
```

Artifacts land under `benchmarks/results/gemma4-assistant-mtp/`; SVGs render into `docs/assets/`. Tune the accept/throughput trade-off with `AX_MLX_GEMMA4_ASSISTANT_MTP_DRAFT_MIN_CONFIDENCE` (default `0.999`; `0` disables the gate).
</details>

#### Qwen 3.6

Three-engine MTP comparison (MTPLX 0.3.7, AX Engine MTP, AX Engine MTP+n-gram) using standard `Qwen/Qwen3.6-*` sidecars plus matching `mlx-community/*-4bit` MLX bases. No `Youssofal/*MTPLX*` bundles are used. All three engines run on the same prompt suites, token caps, sampler, warmup, repetition count, and cooldown.

AX MTP runs the default draft confidence gate (`AX_MLX_MTP_DRAFT_MIN_CONFIDENCE`). The accept columns below use the accept-maximizing `0.98` setting, which holds pure-MTP accept ≥99% on every row except the hardest `python_modules_long` suite (27B 97.6%, 35B-A3B 99.3%). The shipped default is `0.90`, which trades ~1–2 points of accept for +5–13% decode throughput (see `docs/MTP-DRAFT-GATE-THROUGHPUT.md`). Set the variable to `0.98` to restore the accept-maximizing behavior, or `0` to disable.

<table>
<tr>
<td align="center"><strong>Qwen3.6 27B 4-bit</strong></td>
<td align="center"><strong>Qwen3.6 35B-A3B 4-bit</strong></td>
</tr>
<tr>
<td><img width="100%" src="docs/assets/perf-mtp-fair-27b-decode-tok-s.svg" alt="Grouped box-and-whisker plot comparing Qwen3.6 27B 4-bit fair MTP decode throughput for MTPLX, AX Engine MTP, and AX Engine MTP+n-gram with flappy, long_code, and python_modules_long combined"></td>
<td><img width="100%" src="docs/assets/perf-mtp-fair-35b-a3b-decode-tok-s.svg" alt="Grouped box-and-whisker plot comparing Qwen3.6 35B-A3B 4-bit fair MTP decode throughput for MTPLX, AX Engine MTP, and AX Engine MTP+n-gram with flappy, long_code, and python_modules_long combined"></td>
</tr>
<tr>
<td><img width="100%" src="docs/assets/perf-mtp-fair-27b-accept-rate.svg" alt="Grouped box-and-whisker plot comparing Qwen3.6 27B 4-bit fair MTP accept rate for MTPLX, AX Engine MTP, and AX Engine MTP+n-gram with flappy, long_code, and python_modules_long combined"></td>
<td><img width="100%" src="docs/assets/perf-mtp-fair-35b-a3b-accept-rate.svg" alt="Grouped box-and-whisker plot comparing Qwen3.6 35B-A3B 4-bit fair MTP accept rate for MTPLX, AX Engine MTP, and AX Engine MTP+n-gram with flappy, long_code, and python_modules_long combined"></td>
</tr>
<tr>
<td><img width="100%" src="docs/assets/perf-mtp-fair-27b-prefill-tok-s.svg" alt="Grouped box-and-whisker plot comparing Qwen3.6 27B 4-bit fair MTP prefill throughput for MTPLX, AX Engine MTP, and AX Engine MTP+n-gram with flappy, long_code, and python_modules_long combined"></td>
<td><img width="100%" src="docs/assets/perf-mtp-fair-35b-a3b-prefill-tok-s.svg" alt="Grouped box-and-whisker plot comparing Qwen3.6 35B-A3B 4-bit fair MTP prefill throughput for MTPLX, AX Engine MTP, and AX Engine MTP+n-gram with flappy, long_code, and python_modules_long combined"></td>
</tr>
<tr>
<td><img width="100%" src="docs/assets/perf-mtp-fair-27b-ttft-ms.svg" alt="Grouped box-and-whisker plot comparing Qwen3.6 27B 4-bit fair MTP time-to-first-token for MTPLX, AX Engine MTP, and AX Engine MTP+n-gram with flappy, long_code, and python_modules_long combined"></td>
<td><img width="100%" src="docs/assets/perf-mtp-fair-35b-a3b-ttft-ms.svg" alt="Grouped box-and-whisker plot comparing Qwen3.6 35B-A3B 4-bit fair MTP time-to-first-token for MTPLX, AX Engine MTP, and AX Engine MTP+n-gram with flappy, long_code, and python_modules_long combined"></td>
</tr>
</table>

| Model | Suite | Depth | MTPLX tok/s | MTPLX accept | AX tok/s | AX accept | AX+ngram tok/s | AX+ngram accept |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.6 27B 4-bit | flappy | 3 | 56.1 | 100.0% (96.0–100.0) | 60.6 | 99.9% (98.3–100.0) | 57.4 | 99.1% (96.5–99.5) |
| Qwen3.6 27B 4-bit | long_code | 3 | 57.9 | 99.7% (98.4–100.0) | 54.9 | 99.9% (99.2–100.0) | 59.3 | 99.1% (98.4–99.7) |
| Qwen3.6 27B 4-bit | python_modules_long | 3 | 52.7 | 87.6% (81.2–95.0) | 47.8 | 97.6% (96.7–99.8) | 50.2 | 97.2% (95.3–98.6) |
| Qwen3.6 35B-A3B 4-bit | flappy | 1 | 104.3 | 49.5% (42.3–60.6) | 180.6 | 100.0% (99.1–100.0) | 182.3 | 99.6% (97.9–99.7) |
| Qwen3.6 35B-A3B 4-bit | long_code | 1 | 105.6 | 51.4% (43.1–66.7) | 179.1 | 100.0% (99.8–100.0) | 224.2 | 99.8% (99.0–100.0) |
| Qwen3.6 35B-A3B 4-bit | python_modules_long | 1 | 98.2 | 42.6% (37.0–46.1) | 182.4 | 99.3% (98.0–99.7) | 169.4 | 97.6% (96.2–98.4) |

Accept cells show median with `(min–max)` range across the suite's cases × 5 reps, so the run-to-run spread on the borderline `python_modules_long` suite is visible rather than hidden behind a single point.

**Prefill throughput (tok/s) — same run:**

MTPLX prefill is derived from `prompt_tokens / prompt_eval_time_s` (runner-level). AX prefill is measured at runner level. Both are pure GPU compute measurements.

| Model | Suite | Depth | MTPLX tok/s | AX MTP tok/s | AX MTP+ngram tok/s |
|---|---|---:|---:|---:|---:|
| Qwen3.6 27B 4-bit | flappy | 3 | 657 | 681 | 639 |
| Qwen3.6 27B 4-bit | long_code | 3 | 793 | 769 | 765 |
| Qwen3.6 27B 4-bit | python_modules_long | 3 | 680 | 692 | 671 |
| Qwen3.6 35B-A3B 4-bit | flappy | 1 | 1,520 | 1,831 | 1,836 |
| Qwen3.6 35B-A3B 4-bit | long_code | 1 | 2,431 | 2,735 | 2,707 |
| Qwen3.6 35B-A3B 4-bit | python_modules_long | 1 | 1,654 | 1,966 | 1,967 |

**Time to first token (ms) — same run:**

MTPLX TTFT is derived from `prompt_eval_time_s`. AX TTFT is a runner-time measurement. Both are pure prefill measurements.

| Model | Suite | Depth | MTPLX ms | AX MTP ms | AX MTP+ngram ms |
|---|---|---:|---:|---:|---:|
| Qwen3.6 27B 4-bit | flappy | 3 | 489 | 477 | 504 |
| Qwen3.6 27B 4-bit | long_code | 3 | 905 | 934 | 938 |
| Qwen3.6 27B 4-bit | python_modules_long | 3 | 509 | 505 | 505 |
| Qwen3.6 35B-A3B 4-bit | flappy | 1 | 213 | 176 | 176 |
| Qwen3.6 35B-A3B 4-bit | long_code | 1 | 295 | 262 | 265 |
| Qwen3.6 35B-A3B 4-bit | python_modules_long | 1 | 206 | 172 | 177 |

Sampler: temperature=0.6, top_p=0.95, top_k=20; 1,000 gen tokens, 5 repetitions, 30 s cooldown, 10 s inter-case cooldown. MTPLX 0.3.7 · AX Engine v6.0.1.

Full artifacts: [`2026-06-07-qwen36-fair`](benchmarks/results/mtp-fair/2026-06-07-qwen36-fair/summary.json) (full same-day run; MTPLX and AX MTP+n-gram rows at their defaults, AX pure-MTP rows at the accept-maximizing `0.98` gate).

<details>
<summary>Reproduce this benchmark</summary>

```bash
ax-engine convert-mtplx mlx-community/Qwen3.6-27B-4bit \
  --mtp-source Qwen/Qwen3.6-27B \
  --fair-base-only
ax-engine convert-mtplx mlx-community/Qwen3.6-35B-A3B-4bit \
  --mtp-source Qwen/Qwen3.6-35B-A3B \
  --fair-base-only
python3 scripts/bench_qwen36_mtp_fair.py \
  --engines mtplx ax \
  --modes mtp mtp-ngram \
  --models 27b-4bit 35b-a3b-4bit \
  --suites flappy long_code python_modules_long \
  --max-tokens 1000 \
  --repetitions 5 \
  --cooldown 30
```

`convert-mtplx` wraps the generic sidecar packager, applies model-specific defaults when optional knobs are omitted (Qwen3.6 27B depth 3; 35B-A3B depth 1), and validates `ax_mtp_sidecar_manifest.json` before reporting success. The generated `summary.md`, `summary.json`, and `decode-tok-s.svg` live under `benchmarks/results/mtp-fair/`. Full methodology and caveats in [`docs/PERFORMANCE.md#mtp-mode`](docs/PERFORMANCE.md#mtp-mode).
</details>

### Direct Decode · Prefill · TTFT

<!-- readme-performance-artifacts: reference=benchmarks/results/mlx-inference/2026-05-26-direct-mode-clean-refresh/; ax-overlay=benchmarks/results/mlx-inference/2026-06-04-ax-direct-ngram-readme-rerun/ -->

The tables below compare **direct (non-speculative) decode** across llama.cpp Metal, mlx_lm, and ax engine, covering Gemma 4 and Qwen 3.6 at 128/512/2048 prompt tokens. `ax direct baseline` disables n-gram acceleration, MTP, and assistant drafting to measure the repo-owned direct decode path. Bench prompts are `mlx_lm.benchmark` seed-0 random tokens, which keeps prompt-hash parity across MLX rows.

<table>
<tr>
<td></td>
<td align="center"><strong>Gemma 4</strong></td>
<td align="center"><strong>Qwen 3.6</strong></td>
</tr>
<tr>
<td align="center"><strong>Decode rate</strong></td>
<td><img src="docs/assets/perf-gemma4-decode-box-whisker.svg" alt="Grouped box-and-whisker plot comparing llama.cpp Metal, mlx_lm, and ax_engine direct decode rates for Gemma 4 models at 128/512/2048 prompt tokens with a red highest-median reference line"></td>
<td><img src="docs/assets/perf-qwen-decode-box-whisker.svg" alt="Grouped box-and-whisker plot comparing llama.cpp Metal, mlx_lm, and ax_engine direct decode rates for Qwen 3.6 models at 128/512/2048 prompt tokens with a red highest-median reference line"></td>
</tr>
<tr>
<td align="center"><strong>Prefill rate</strong></td>
<td><img src="docs/assets/perf-gemma4-prefill-box-whisker.svg" alt="Grouped box-and-whisker plot comparing llama.cpp Metal, mlx_lm, and ax_engine prefill rates for Gemma 4 models at 128/512/2048 prompt tokens with a red highest-median reference line"></td>
<td><img src="docs/assets/perf-qwen-prefill-box-whisker.svg" alt="Grouped box-and-whisker plot comparing llama.cpp Metal, mlx_lm, and ax_engine prefill rates for Qwen 3.6 models at 128/512/2048 prompt tokens with a red highest-median reference line"></td>
</tr>
<tr>
<td align="center"><strong>TTFT</strong></td>
<td><img src="docs/assets/perf-gemma4-ttft-box-whisker.svg" alt="Grouped box-and-whisker plot comparing llama.cpp Metal, mlx_lm, and ax_engine TTFT for Gemma 4 models at 128/512/2048 prompt tokens with a red lowest-median reference line"></td>
<td><img src="docs/assets/perf-qwen-ttft-box-whisker.svg" alt="Grouped box-and-whisker plot comparing llama.cpp Metal, mlx_lm, and ax_engine TTFT for Qwen 3.6 models at 128/512/2048 prompt tokens with a red lowest-median reference line"></td>
</tr>
</table>

> **`llama.cpp Metal*` column** — Shape-compatible reference produced by Metal-enabled `llama-bench`. `llama-bench` generates its own internal synthetic prompt tokens and does not consume the harness prompt JSON, so these numbers are **not** prompt-hash parity with the other columns. No percentage delta is shown. MLX bit-widths are mapped to the nearest standard GGUF K-quant (4→Q4_K_M, 5→Q5_K_M, 6→Q6_K, 8→Q8_0). Source: `benchmarks/manifests/llama_cpp_metal/inventory.json`, `scripts/bench_llama_cpp_metal_sweep.py`.

<details>
<summary>Benchmark provenance and methodology</summary>

The `mlx_lm` reference rows for the 12 Gemma 4 and Qwen 3.6 rows shown below come from `benchmarks/results/mlx-inference/2026-05-26-direct-mode-clean-refresh/`. The AX direct-mode cells come from the full 12-model AX-only rerun in `benchmarks/results/mlx-inference/2026-06-04-ax-direct-ngram-readme-rerun/` (v5.1.8, `5402992b`). The `llama.cpp Metal*` column is injected from `benchmarks/manifests/llama_cpp_metal/inventory.json` and the `2026-05-18-llama-cpp-metal-gemma-e2b-4bit-depth-fa/` Gemma 4 E2B 4-bit recheck.

Setup: generation=128, 5 measured repetitions, 15-second cooldown, AX prefix cache disabled for cold prefill and TTFT measurement, production-build binaries, matching prompt SHA checks. Long-greedy AX prefill rows are runner-time measurements of the cache-state prefix plus final prompt-token boundary — not full-logits prompt scoring throughput. Percentages are versus `mlx_lm`.

The 2K `llama.cpp Metal*` prefill rows are long-context, GGUF-runtime-reference rows. The Gemma 4 E2B 4-bit row was produced with llama.cpp b9110 and rechecked on b9200 with Metal offload, `-b/-ub 2048`, and flash attention enabled. The b9200 recheck improved 2K prefill only slightly — this is our benchmark boundary, not an upstream llama.cpp official bug statement.
</details>

#### Prefill throughput (tok/s) — percentages vs mlx_lm

| Model | MLX quantization | Prompt tok | llama.cpp Metal* | mlx_lm | ax engine |
|---|---|---:| ---: |---:|---:|
| Gemma 4 E2B | 4-bit | 128 | 3,481.7 | 2,338.1 | **6,044.9 (+158.5%)** |
|         |         | 512 | 6,846.0 | 7,870.0 | **17,238.5 (+119.0%)** |
|         |         | 2048 | 7,643.1 | 18,014.7 | **24,778.3 (+37.5%)** |
| Gemma 4 E2B | 5-bit | 128 | 3,398.4 | 2,238.5 | **6,019.3 (+168.9%)** |
|         |         | 512 | 6,860.3 | 7,469.9 | **16,846.5 (+125.5%)** |
|         |         | 2048 | 7,288.1 | 16,664.1 | **24,188.3 (+45.2%)** |
| Gemma 4 E2B | 6-bit | 128 | 3,539.7 | 1,823.5 | **5,700.8 (+212.6%)** |
|         |         | 512 | 7,274.0 | 6,046.6 | **16,336.4 (+170.2%)** |
|         |         | 2048 | 7,623.2 | 15,332.1 | **23,502.4 (+53.3%)** |
| Gemma 4 E2B | 8-bit | 128 | 3,694.3 | 1,605.0 | **5,452.1 (+239.7%)** |
|         |         | 512 | 7,481.0 | 6,332.9 | **15,679.5 (+147.6%)** |
|         |         | 2048 | 7,990.4 | 15,536.8 | **23,392.6 (+50.6%)** |
| Gemma 4 E4B | 4-bit | 128 | 2,194.0 | 1,513.2 | **3,409.6 (+125.3%)** |
|         |         | 512 | 4,454.2 | 4,195.5 | **7,000.6 (+66.9%)** |
|         |         | 2048 | 4,426.6 | 7,325.4 | **8,863.4 (+21.0%)** |
| Gemma 4 26B A4B | 4-bit | 128 | 1,911.4 | 496.4 | **1,339.1 (+169.7%)** |
|         |         | 512 | 3,484.5 | 1,621.0 | **3,055.2 (+88.5%)** |
|         |         | 2048 | 3,604.8 | 3,300.1 | **4,668.6 (+41.5%)** |
| Gemma 4 31B | 4-bit | 128 | 522.6 | 283.1 | **513.3 (+81.3%)** |
|         |         | 512 | 665.3 | 619.9 | **742.5 (+19.8%)** |
|         |         | 2048 | 560.3 | 733.9 | **782.7 (+6.6%)** |
| Qwen 3.6 27B | 4-bit | 128 | 539.6 | 378.8 | **583.6 (+54.1%)** |
|  |  | 512 | 759.7 | 705.7 | **827.1 (+17.2%)** |
|  |  | 2048 | 664.3 | 895.2 | **923.7 (+3.2%)** |
| Qwen 3.6 27B | 5-bit | 128 | 520.8 | 278.8 | **536.6 (+92.4%)** |
|  |  | 512 | 733.4 | 599.5 | **784.4 (+30.9%)** |
|  |  | 2048 | 667.0 | 827.5 | **883.3 (+6.7%)** |
| Qwen 3.6 27B | 6-bit | 128 | 537.7 | 270.5 | **509.7 (+88.4%)** |
|  |  | 512 | 756.1 | 577.6 | **762.6 (+32.0%)** |
|  |  | 2048 | 689.3 | 798.2 | **869.6 (+8.9%)** |
| Qwen 3.6 27B | 8-bit | 128 | 559.4 | 219.3 | **453.2 (+106.6%)** |
|  |  | 512 | 798.2 | 520.2 | **731.9 (+40.7%)** |
|  |  | 2048 | 741.9 | 787.4 | **868.1 (+10.2%)** |
| Qwen 3.6 35B A3B | 4-bit | 128 | 1,706.9 | 539.4 | **1,115.0 (+106.7%)** |
|  |  | 512 | 3,146.6 | 1,599.5 | **2,618.6 (+63.7%)** |
|  |  | 2048 | 3,542.3 | 3,513.1 | **3,700.6 (+5.3%)** |

#### Decode throughput (tok/s) — generation=128 tokens, temp=0

| Model | MLX quantization | Prompt tok | llama.cpp Metal* | mlx_lm | ax direct baseline |
|---|---|---:| ---: |---:|---:|
| Gemma 4 E2B | 4-bit | 128 | 174.6 | 214.0 | **235.8 (+10.2%)** |
|  |  | 512 | 165.2 | 210.3 | **226.6 (+7.8%)** |
|  |  | 2048 | 171.9 | 200.9 | **216.8 (+7.9%)** |
| Gemma 4 E2B | 5-bit | 128 | 154.8 | 195.2 | **210.6 (+7.9%)** |
|  |  | 512 | 154.3 | 182.0 | **203.3 (+11.7%)** |
|  |  | 2048 | 154.3 | 181.9 | **194.8 (+7.1%)** |
| Gemma 4 E2B | 6-bit | 128 | 152.1 | 172.2 | **186.7 (+8.4%)** |
|  |  | 512 | 152.0 | 166.3 | **180.9 (+8.8%)** |
|  |  | 2048 | 152.2 | 162.5 | **174.5 (+7.4%)** |
| Gemma 4 E2B | 8-bit | 128 | 136.1 | 153.0 | **163.3 (+6.7%)** |
|  |  | 512 | 138.3 | 148.8 | **158.7 (+6.7%)** |
|  |  | 2048 | 138.7 | 144.2 | **153.9 (+6.7%)** |
| Gemma 4 E4B | 4-bit | 128 | 110.7 | 137.1 | **144.4 (+5.3%)** |
|  |  | 512 | 110.8 | 133.6 | **141.4 (+5.8%)** |
|  |  | 2048 | 110.7 | 130.6 | **138.3 (+6.0%)** |
| Gemma 4 26B A4B | 4-bit | 128 | 112.6 | 127.9 | **135.3 (+5.7%)** |
|  |  | 512 | 112.9 | 125.0 | **132.0 (+5.6%)** |
|  |  | 2048 | 112.9 | 119.3 | **127.4 (+6.8%)** |
| Gemma 4 31B | 4-bit | 128 | 25.0 | 28.9 | **29.3 (+1.6%)** |
|  |  | 512 | 25.5 | 28.3 | **28.7 (+1.5%)** |
|  |  | 2048 | 25.3 | 27.0 | **27.5 (+1.8%)** |
| Qwen 3.6 27B | 4-bit | 128 | 26.0 | 34.0 | **35.0 (+3.1%)** |
|  |  | 512 | 26.0 | 33.9 | **34.2 (+0.9%)** |
|  |  | 2048 | 18.8 | 33.4 | **33.8 (+1.2%)** |
| Qwen 3.6 27B | 5-bit | 128 | 23.5 | 21.6 | **28.9 (+33.9%)** |
|  |  | 512 | 23.3 | 28.1 | **28.8 (+2.5%)** |
|  |  | 2048 | 17.8 | 27.8 | **28.6 (+2.8%)** |
| Qwen 3.6 27B | 6-bit | 128 | 21.3 | 24.0 | **25.7 (+6.9%)** |
|  |  | 512 | 21.3 | 24.8 | **25.6 (+3.4%)** |
|  |  | 2048 | 15.4 | 24.6 | **25.4 (+3.2%)** |
| Qwen 3.6 27B | 8-bit | 128 | 18.3 | 18.7 | **19.3 (+3.5%)** |
|  |  | 512 | 18.2 | 18.6 | **19.2 (+3.4%)** |
|  |  | 2048 | 12.7 | 18.4 | **19.1 (+3.9%)** |
| Qwen 3.6 35B A3B | 4-bit | 128 | 108.1 | 140.1 | **155.2 (+10.8%)** |
|  |  | 512 | 108.2 | 136.5 | **152.8 (+12.0%)** |
|  |  | 2048 | 105.7 | 134.5 | **151.8 (+12.9%)** |

> Qwen 3.6 27B 4-bit at prompt=2,048 originally produced zero decode tokens because 4-bit quantization noise pushed an EOS token to argmax at decode step 0 on the `mlx_lm.benchmark` random-token contract. The benchmark harness now sends `sampling.ignore_eos=true` for AX throughput runs, matching how `mlx_lm.benchmark` measures fixed `gen=N` throughput. Production requests default to `ignore_eos=false`. Source: `benchmarks/results/mlx-inference/2026-05-20-qwen27-4to5-direct-ngram-directcpp-r2/qwen3_6-27b-4bit.json`.

#### Time to first token (ms) — generation=128 tokens, temp=0

**Lower is better.** `mlx_lm` values are derived from reported prefill throughput. AX values are measured directly from per-step runner timing in the SSE event stream.

| Model | MLX quantization | Prompt tok | llama.cpp Metal* | mlx_lm | ax engine |
|---|---|---:| ---: |---:|---:|
| Gemma 4 E2B | 4-bit | 128 | 36.8 | 54.7 | **21.2 (-61.3%)** |
|         |         | 512 | 74.8 | 65.1 | **29.7 (-54.3%)** |
|         |         | 2048 | 268.0 | 113.7 | **82.7 (-27.3%)** |
| Gemma 4 E2B | 5-bit | 128 | 37.7 | 57.2 | **21.3 (-62.8%)** |
|         |         | 512 | 74.6 | 68.5 | **30.4 (-55.7%)** |
|         |         | 2048 | 281.0 | 122.9 | **84.7 (-31.1%)** |
| Gemma 4 E2B | 6-bit | 128 | 36.2 | 70.2 | **22.5 (-68.0%)** |
|         |         | 512 | 70.4 | 84.7 | **31.3 (-63.0%)** |
|         |         | 2048 | 268.7 | 133.6 | **87.1 (-34.8%)** |
| Gemma 4 E2B | 8-bit | 128 | 34.6 | 79.7 | **23.5 (-70.6%)** |
|         |         | 512 | 68.4 | 80.8 | **32.7 (-59.6%)** |
|         |         | 2048 | 256.3 | 131.8 | **87.5 (-33.6%)** |
| Gemma 4 E4B | 4-bit | 128 | 58.3 | 84.6 | **37.5 (-55.6%)** |
|         |         | 512 | 114.9 | 122.0 | **73.1 (-40.1%)** |
|         |         | 2048 | 462.7 | 279.6 | **231.1 (-17.4%)** |
| Gemma 4 26B A4B | 4-bit | 128 | 67.0 | 257.8 | **95.6 (-62.9%)** |
|         |         | 512 | 146.9 | 315.8 | **167.6 (-46.9%)** |
|         |         | 2048 | 568.1 | 620.6 | **438.7 (-29.3%)** |
| Gemma 4 31B | 4-bit | 128 | 244.9 | 452.2 | **249.4 (-44.8%)** |
|         |         | 512 | 769.5 | 826.0 | **689.5 (-16.5%)** |
|         |         | 2048 | 3,655.2 | 2,790.6 | **2,616.7 (-6.2%)** |
| Qwen 3.6 27B | 4-bit | 128 | 237.2 | 337.9 | **219.3 (-35.1%)** |
|  |  | 512 | 673.9 | 725.6 | **619.0 (-14.7%)** |
|  |  | 2048 | 3,083.1 | 2,287.7 | **2,217.1 (-3.1%)** |
| Qwen 3.6 27B | 5-bit | 128 | 245.8 | 459.0 | **238.5 (-48.0%)** |
|  |  | 512 | 698.1 | 854.1 | **652.7 (-23.6%)** |
|  |  | 2048 | 3,070.5 | 2,474.9 | **2,318.7 (-6.3%)** |
| Qwen 3.6 27B | 6-bit | 128 | 238.1 | 473.2 | **251.1 (-46.9%)** |
|  |  | 512 | 677.2 | 886.5 | **671.4 (-24.3%)** |
|  |  | 2048 | 2,971.2 | 2,565.6 | **2,355.0 (-8.2%)** |
| Qwen 3.6 27B | 8-bit | 128 | 228.8 | 583.6 | **282.5 (-51.6%)** |
|  |  | 512 | 641.5 | 984.2 | **699.6 (-28.9%)** |
|  |  | 2048 | 2,760.6 | 2,601.1 | **2,359.3 (-9.3%)** |
| Qwen 3.6 35B A3B | 4-bit | 128 | 75.0 | 237.3 | **114.8 (-51.6%)** |
|  |  | 512 | 162.7 | 320.1 | **195.5 (-38.9%)** |
|  |  | 2048 | 578.2 | 583.0 | **553.4 (-5.1%)** |

Embedding benchmarks are kept out of this README summary; see [`docs/EMBEDDINGS.md`](docs/EMBEDDINGS.md).

## SDKs

ax-engine-server exposes OpenAI-compatible HTTP endpoints, and several SDKs wrap those endpoints or the in-process Rust session directly.

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

See `examples/go/` for runnable examples. For LangChain, point [langchaingo](https://github.com/tmc/langchaingo)'s OpenAI provider at `http://127.0.0.1:8080/v1` — see `examples/go/langchain/` and `docs/GO.md`.

### Ruby

The Ruby SDK lives at `sdk/ruby/` (`ax-engine-sdk` gem). Zero dependencies — stdlib `net/http` only. Streaming uses a block interface.

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

The Mojo SDK (`sdk/mojo/ax_engine.mojo`) wraps the Python `ax_engine` package via Mojo's `PythonObject` interop. Requires the Python extension to be built first (`maturin develop`).

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

## Server Usage

The installed PyPI workflow uses `ax-engine serve` for the common local-serving path. `ax-engine-server` remains available as the backward-compatible low-level entrypoint when you need explicit runtime flags.

```bash
# Download a model and generate its manifest
MODEL_DIR="$(ax-engine download qwen36-35b --json | python3 -c 'import json,sys; print(json.load(sys.stdin)["dest"])')"

# Recommended: resolve and launch ax-engine-server
ax-engine serve "$MODEL_DIR" --port 8080

# Backward-compatible low-level path
./target/release/ax-engine-server \
  --mlx \
  --mlx-model-artifacts-dir "$MODEL_DIR" \
  --port 8080

# Inspect the running server
curl http://127.0.0.1:8080/v1/runtime

# Smoke generation request
curl http://127.0.0.1:8080/v1/generate \
  -H 'content-type: application/json' \
  -d '{
    "model": "qwen3_dense",
    "input_tokens": [1, 2, 3, 4],
    "max_output_tokens": 4,
    "sampling": { "temperature": 0.0, "top_p": 1.0, "top_k": 0, "seed": 1234 }
  }'
```

**Python bindings (after `maturin develop`):**

```python
import ax_engine

path = ax_engine.download_model("mlx-community/Qwen3-4B-4bit")
with ax_engine.Session(mlx=True, mlx_model_artifacts_dir=str(path)) as s:
    result = s.generate([1, 2, 3], max_output_tokens=32)
    print(result.output_tokens)
```

**Delegated route** (for unsupported MLX text models that `mlx-lm` can serve):

```bash
mlx_lm.server --model /path/to/local/mlx-model --host 127.0.0.1 --port 8090

./target/release/ax-engine-bench generate \
  --prompt "Hello from mlx-lm" \
  --support-tier mlx_lm_delegated \
  --mlx-lm-server-url http://127.0.0.1:8090
```

`mlx_lm_delegated` is a compatibility route, not an AX-owned MLX throughput claim. AX forwards text generation to upstream `mlx_lm.server` and preserves `temperature`, `top_p`, `top_k`, `repetition_penalty`, and `seed`. Streamed chunks are delegated text deltas — not AX-owned token IDs, KV state, or model-kernel throughput evidence.

**Check readiness and run benchmarks:**

```bash
# Readiness check
./target/release/ax-engine-bench doctor --mlx-model-artifacts-dir "$MODEL_DIR"
bash scripts/check-server-preview.sh
bash scripts/check-python-preview.sh

# Primary benchmark: AX vs mlx_lm
python3 scripts/bench_mlx_inference_stack.py \
  --model-dir /path/to/local/mlx-model \
  --prompt-tokens 128,512,2048 --generation-tokens 128 \
  --ax-compare-policies --repetitions 5 \
  --output benchmarks/results/mlx-inference/$(date +%F)/gemma-4-e2b-it-4bit.json

# Secondary workload-contract benchmark
./target/release/ax-engine-bench scenario \
  --manifest benchmarks/manifests/scenario/chat_gemma4_e2b_short.json \
  --output-root benchmarks/results
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

For Gemma/AX MLX telemetry and decode-profile changes, prefer the targeted `scripts/check-mlx-telemetry.sh` gate. Use `scripts/check-mlx-telemetry.sh --full-workspace` when the change touches shared Rust contracts; that protected path compiles the workspace with `cargo test --workspace --no-run --jobs 1` before running crate-by-crate tests.

Coverage is collected by the report-only GitHub Actions workflow in `.github/workflows/coverage.yml`. It publishes Rust `cargo llvm-cov` and Python `coverage.py` artifacts without enforcing a percentage threshold yet.

Public documentation is in `docs/`. Canonical benchmark manifests are in `benchmarks/manifests/`. Key design docs:
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

## Benchmark Reference Projects

AX Engine's benchmark design and compatibility checks are informed by local reference checkouts of related open-source projects. A row is published only when it fits the benchmark contract for the specific workload: comparable model artifacts, prompt and sampling policy, prefill/decode/TTFT definitions, repeatability, host/runtime metadata, and provenance.

| Project | Repository |
|---|---|
| ds4 | [antirez/ds4](https://github.com/antirez/ds4) |
| lightning-mlx | [samuelfaj/lightning-mlx](https://github.com/samuelfaj/lightning-mlx) |
| llama.cpp | [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) |
| mistral.rs | [EricLBuehler/mistral.rs](https://github.com/EricLBuehler/mistral.rs) |
| MLX | [ml-explore/mlx](https://github.com/ml-explore/mlx) |
| mlx-c | [ml-explore/mlx-c](https://github.com/ml-explore/mlx-c) |
| mlx-engine | [lmstudio-ai/mlx-engine](https://github.com/lmstudio-ai/mlx-engine) |
| mlx-lm | [ml-explore/mlx-lm](https://github.com/ml-explore/mlx-lm) |
| mlx-turboquant | [rachittshah/mlx-turboquant](https://github.com/rachittshah/mlx-turboquant) |
| MTPLX | [youssofal/MTPLX](https://github.com/youssofal/MTPLX) |
| Rapid-MLX | [raullenchai/Rapid-MLX](https://github.com/raullenchai/Rapid-MLX) |
| turboquant-mlx | [arozanov/turboquant-mlx](https://github.com/arozanov/turboquant-mlx) |
| vLLM | [vllm-project/vllm](https://github.com/vllm-project/vllm) |

Some reference projects are experimental, version-unstable, focused on a different serving route, or not shaped for the same Apple MLX/Metal measurement strategy, so those results remain implementation guidance or diagnostic evidence rather than public comparison rows.

## Limitations

- **Qwen3.5 long-prompt prefill**: Qwen3.5 prefill can trail upstream MLX references on longer prompts; decode and Qwen3-Next are not affected in the same way.
- **Raw HuggingFace weights**: use pre-sanitized MLX community weights or convert first with `mlx_lm.convert`.
- **N-gram acceleration rows**: effective-throughput measurements, not raw model-kernel speedups.
- **TurboQuant KV compression**: experimental and off by default.

See the [FAQ limitations entry](docs/FAQ.md#what-are-the-current-limitations) for details.

## Contributing

AX Engine welcomes community input through issue tickets, wishlist requests, reproducible benchmark results, and documentation feedback. We generally do not accept unsolicited code PRs, especially for runtime, model, kernel, scheduler, cache, n-gram, or performance-tuning changes.

Performance tuning is tightly coupled: a local speedup can regress correctness, TTFT, memory pressure, direct-vs-n-gram behavior, long-context behavior, serving stability, or another model family. Please open an issue first with the problem, target workload, and evidence so maintainers can choose the right validation path. See [CONTRIBUTING.md](CONTRIBUTING.md) for issue, wishlist, and benchmark result guidelines.

## Community

- Website: [automatosx.com](https://automatosx.com)
- Discord: [Join us](https://discord.com/invite/cTavsMgu)
- Email: enquiry@defai.digital

## License

Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

Copyright (c) 2026 [DEFAI Private Limited](https://defai.digital)
