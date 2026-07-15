# Supported LLM Models

AX Engine supports LLMs through a direct-first runtime contract. Direct support
is the default deployment path; delegated adapters are explicit compatibility
paths for migration, validation, or external reference rows. The path matters
because it defines who runs the model graph, which API features are available,
and what benchmark claims are allowed.

| Path | Use it for | Who runs the model | What the result means |
| --- | --- | --- | --- |
| Direct support | Model families with a repo-owned `ax-engine-mlx` graph | AX Engine on MLX | AX-owned token/KV/runtime behavior; performance claims still require benchmark artifacts |
| `mlx_lm_delegated` | Explicit compatibility checks for MLX text models before AX direct support | A user-provided `mlx_lm.server` | AX server/SDK compatibility over delegated text generation; not AX-owned MLX throughput |
| `llama_cpp` | Explicit GGUF/non-MLX compatibility checks or external reference rows | llama.cpp server or CLI | Delegated route-contract evidence; not AX-owned MLX throughput |
| Unsupported | Requests with no direct repo-owned path and no explicitly selected adapter | None | Fail closed |

Runtime metadata exposes the selected path through fields such as
`selected_backend`, `support_tier`, and `resolution_policy`. Preserve those
labels in benchmark artifacts and user-facing claims.

Promotion, freeze, and end-of-life decisions follow the
[model support policy](MODEL-SUPPORT-POLICY.md). In short, do not add new direct
support work for a model family that has had no meaningful upstream release or
artifact refresh within the last six months unless an owner records a specific
exception.

## Getting Model Artifacts

AX Engine requires pre-sanitized MLX safetensors plus a `model-manifest.json`.
The recommended source is [mlx-community](https://huggingface.co/mlx-community)
because those snapshots are already converted and validated. `ax-engine
download`, `download_model()`, and `scripts/download_model.py` download weights
and auto-generate the manifest in one step.

List direct-download aliases:

```text
ax-engine download --list
```

Download by alias:

```text
ax-engine download qwen3.5-9b --json
ax-engine download qwen36-35b --json
ax-engine download qwen36-27b --json
ax-engine download gemma4-e2b --json
ax-engine download gemma4-12b --json
ax-engine download gemma4-31b --json
ax-engine download llama3.3-70b --json
ax-engine download mistral-small --json
ax-engine download gpt-oss-20b --json
```

Download and serve in one command:

```text
ax-engine serve qwen36-35b --download --port 8080
ax-engine serve llama3.3-70b --download --port 8080
```

Raw `mlx-community` repo IDs are also accepted:

```text
ax-engine download mlx-community/Qwen3.6-35B-A3B-4bit --json
ax-engine download mlx-community/Qwen3-Coder-Next-6bit --json
ax-engine download mlx-community/gemma-4-e2b-it-4bit --json
ax-engine download mlx-community/gpt-oss-20b-MXFP4-Q4 --json
```

Copy a snapshot to an explicit directory only when needed:

```text
ax-engine download qwen36-35b --dest /Volumes/Models/qwen36-35b
```

Python SDK:

```python
from ax_engine import download_model

path = download_model("mlx-community/Qwen3.6-35B-A3B-4bit")
```

Built-in direct-download aliases:

**Primary productivity (Gemma / Qwen / GLM)**

| Alias | Repo |
| --- | --- |
| `qwen36-35b` | `mlx-community/Qwen3.6-35B-A3B-4bit` |
| `qwen36-27b`, `qwen36-27b-6bit` | `mlx-community/Qwen3.6-27B-{4,6}bit` |
| `gemma4-e2b`, `gemma4-e2b-6bit` | `mlx-community/gemma-4-e2b-it-{4,6}bit` |
| `gemma4-12b`, `gemma4-12b-6bit` | `mlx-community/gemma-4-12B-it-{4,6}bit` |
| `gemma4-26b` | `mlx-community/gemma-4-26b-a4b-it-4bit` |
| `gemma4-31b` | `mlx-community/gemma-4-31b-it-4bit` |
| `qwen3.5-9b` | `mlx-community/Qwen3.5-9B-MLX-4bit` |
| `glm4.7-flash-4bit` | `mlx-community/GLM-4.7-Flash-4bit` |

**Secondary — research / enterprise Llama**

| Alias | Repo |
| --- | --- |
| `llama3.1-8b` | `mlx-community/Llama-3.1-8B-Instruct-4bit` |
| `llama3.3-70b` | `mlx-community/Llama-3.3-70B-Instruct-4bit` |
| `llama4-scout` | `mlx-community/Llama-4-Scout-17B-16E-Instruct-4bit` |

**Secondary — European market Mistral**

| Alias | Repo |
| --- | --- |
| `mistral-small` | `mlx-community/Mistral-Small-3.1-24B-Instruct-2503-4bit` |
| `ministral-8b` | `mlx-community/Ministral-8B-Instruct-2410-4bit` |
| `devstral-small` | `mlx-community/Devstral-Small-2505-4bit` |

**Secondary — open reasoner GPT-OSS (MXFP4)**

| Alias | Repo | Notes |
| --- | --- | --- |
| `gpt-oss-20b` | `mlx-community/gpt-oss-20b-MXFP4-Q4` | Comfortable on 64–128 GB |
| `gpt-oss-120b` | `mlx-community/gpt-oss-120b-MXFP4-Q4` | Prefer 128 GB+; experts stay MXFP4-packed at runtime |

Leave downloads in the Hugging Face Hub cache by default. The cache is shared
with `mlx_lm` and other HF-aware tools, avoiding duplicate copies of large
weights. Use `--dest` only when you want an explicit copy outside the shared
cache.

### MTP Downloads

`ax-engine download-mtp` is the one-command path for supported local-agent MTP
targets. It downloads the base model and prepares AX MTP artifacts when the
model family has a repo-owned packaging path. The CLI command accepts the
canonical target names below plus their aliases; see the
[CLI reference](CLI.md#ax-engine) for optional flags such as `--output`,
`--force`, and `--json`:

```text
ax-engine download-mtp gemma-4-12b-4bit
ax-engine download-mtp qwen3.6-27b-6bit
ax-engine download-mtp qwen3.6-35b-a3b
ax-engine download-mtp gemma-4-12b
ax-engine download-mtp gemma-4-26b
ax-engine download-mtp gemma-4-31b
```

By default, generated MTP packages are written as synthetic Hugging Face Hub
cache snapshots under the active HF cache root (`HF_HUB_CACHE`, `HF_HOME`, or
`XDG_CACHE_HOME`). For example, the Quick Start target defaults to:

```text
~/.cache/huggingface/hub/models--ax-local--gemma-4-12b-it-4bit-assistant-mtp/snapshots/v1
```

The command prints the prepared path and a matching `ax-engine serve ...`
command. Use `--output <dir>` only when you need an explicit copy outside the
shared Hugging Face cache.

| Target | Base repo | Result |
| --- | --- | --- |
| `gemma-4-12b-4bit` | `mlx-community/gemma-4-12B-it-4bit` | Quick-start Gemma assistant-MTP package with `mlx-community/gemma-4-12B-it-assistant-4bit` |
| `qwen3.6-27b-6bit` | `mlx-community/Qwen3.6-27B-6bit` | Qwen fused MTP sidecar from `Qwen/Qwen3.6-27B` |
| `qwen3.6-35b-a3b` | `mlx-community/Qwen3.6-35B-A3B-6bit` | Qwen fused MTP sidecar from `Qwen/Qwen3.6-35B-A3B` |
| `gemma-4-12b` | `mlx-community/gemma-4-12B-it-6bit` | Gemma assistant-MTP package with `mlx-community/gemma-4-12B-it-assistant-6bit` |
| `gemma-4-26b` | `mlx-community/gemma-4-26b-a4b-it-6bit` | Gemma assistant-MTP package with `google/gemma-4-26b-a4b-it-assistant` |
| `gemma-4-31b` | `mlx-community/gemma-4-31b-it-6bit` | Gemma assistant-MTP package with `google/gemma-4-31b-it-assistant` |

For Qwen3.6, `download-mtp` wraps the standard download plus `convert-mtplx`
provenance flow. For Gemma 4, it downloads the target and assistant and runs
the Gemma assistant-MTP packager. The 4-bit Gemma 4 12B target is the simple
Quick Start path and a peer-comparison benchmark lane; the recommended
practical AX Engine MTP benchmark lane remains the six 6-bit targets.
Qwen3-Coder-Next remains a direct-decode
target; it is not a `download-mtp` target because its MLX base artifact does
not ship its own MTP head.

### Existing mlx_lm Downloads

If you already have `mlx_lm` installed, its downloads land in the same cache and
AX Engine can auto-discover them after manifest generation:

```text
python -m mlx_lm.generate --model mlx-community/Qwen3-4B-4bit --prompt "x" --max-tokens 1
ax-engine-bench generate-manifest ~/.cache/huggingface/hub/models--mlx-community--Qwen3-4B-4bit/snapshots/<hash>
ax-engine serve ~/.cache/huggingface/hub/models--mlx-community--Qwen3-4B-4bit/snapshots/<hash> --port 8080
```

### Raw Hugging Face Checkpoints

Raw checkpoints need sanitization before AX Engine can load them:

```text
pip install mlx-lm
mlx_lm.convert --hf-path <org/model> --mlx-path /path/to/dest -q --q-bits 4
ax-engine-bench generate-manifest /path/to/dest
ax-engine serve /path/to/dest --port 8080
```

### Manifest Generation

Download helpers generate `model-manifest.json` automatically. To run manifest
generation directly:

```text
ax-engine-bench generate-manifest /path/to/model
```

Source-tree workflows are covered in the
[Getting Started installation guide](GETTING-STARTED.md#source).

## Direct Support

Direct support means AX has a hand-written `ax-engine-mlx` model graph and
loads MLX safetensors through the AX manifest path. AX owns the request
lifecycle, token/KV handling, direct decode path, n-gram acceleration policy,
route telemetry, and benchmark artifact attribution for these models.

Direct support requires:

- MLX safetensors weights
- an AX `model-manifest.json`
- a repo-owned model implementation in `ax-engine-mlx`
- server or SDK smoke coverage

Public performance claims additionally require MLX inference-stack benchmark
evidence with a matching `mlx_lm.benchmark` baseline.

Current direct-support LLM families:

**Primary (productivity — deepest optim + public performance tables)**

| Family | Direct model IDs | Current scope | Notes |
| --- | --- | --- | --- |
| Gemma 4 | `gemma-4-e2b-it`, `gemma-4-e4b-it`, `gemma-4-12b-it`, `gemma-4-26b-a4b-it`, `gemma-4-31b-it` | Repo-owned MLX runtime; MLX affine 4/5/6-bit weights where available; assistant-MTP packaging for matched `*-assistant` drafters; processed/token-aligned Gemma4 unified media tensors on native generate and OpenAI token-array routes when the manifest contains unified media roles | Dense, unified 12B, per-layer embedding, and MoE variants; sliding-window + full attention; K=V full-attention layers; logit softcapping |
| Qwen 3 | `Qwen3-4B-4bit` and manifest-backed Qwen 3 dense checkpoints | Repo-owned MLX runtime | SwiGLU dense FFN; per-head QK norm; optional MoE variants require manifest evidence |
| Qwen 3.5 | `Qwen3.5-9B-MLX-4bit` / `qwen3.5-9b` preset | Repo-owned MLX runtime; MLX affine 4-bit weights | GatedDeltaNet linear attention + dense SwiGLU FFN; `attn_output_gate` per-head interleaving |
| Qwen 3.6 / Coder Next | `Qwen3.6-35B-A3B` 4-bit MLX, `Qwen3.6-27B` 4/5/6-bit MLX, `Qwen3-Coder-Next-4bit` | Repo-owned MLX runtime | `qwen3_next`: GatedDelta linear attention, full attention with per-head sigmoid gate, sparse top-k MoE with shared expert |
| GLM 4.7 Flash | `glm4_moe_lite` / `glm4.7-flash-4bit` | Repo-owned MLX runtime; MLX affine 4-bit weights | Flash MLA attention, sigmoid-routed MoE with dense+MoE layer split, shared expert; post-attention RMS norm |

**Secondary (preview direct — download aliases + server presets; share standard / family graphs)**

| Family | Direct model IDs / aliases | Current scope | Notes |
| --- | --- | --- | --- |
| Llama 3.x | `llama3.1-8b`, `llama3.3-70b` | Preview direct; MLX affine 4-bit; `llama3` / standard dense path | Research and enterprise baseline; Llama 3 chat template |
| Llama 4 | `llama4-scout` | Preview direct; MLX affine 4-bit; `llama4` MoE path | Fits ~128 GB class; Maverick full 4-bit is out of scope for that class |
| Mistral | `mistral-small`, `ministral-8b`, `devstral-small` | Preview direct; MLX affine 4-bit; `mistral3` / standard path | European market chat + coding; Instruct `[INST]` chat fallback |
| GPT-OSS | `gpt-oss-20b`, `gpt-oss-120b` | Preview direct; MXFP4 MoE experts **kept packed** at load (`gather_qmm` mode=`mxfp4`) | MoE decoder with 128 experts (top-4), SwiGLU, alternating full/sliding-128 attention, per-head learned attention sinks, YaRN RoPE (128K), GQA (64q/8k heads). Prefer MXFP4-Q4 downloads. Expert residency stays ~4-bit so 120B is practical on 128 GB class hosts; attention/router tensors remain higher precision |

Experimental direct-support model families:

| Family | Model ID | Current scope | Notes |
| --- | --- | --- | --- |
| DiffusionGemma | `mlx-community/diffusiongemma-26B-A4B-it-4bit` | Experimental repo-owned MLX block-diffusion path | Experimental rows live under [Performance Results](PERFORMANCE-RESULTS.md#diffusiongemma); benchmark boundary is first committed 256-token diffusion block, not autoregressive TTFT/decode. See [DiffusionGemma experimental support](DIFFUSIONGEMMA.md). |

All direct-support models use MLX safetensors format with the AX
`model-manifest.json` descriptor. Adding a new direct-support architecture
means implementing the model graph, not wiring up a generic loader.

Architecture code alone is not a certified performance claim. Secondary
families ship as **preview direct** with download aliases and chat fallbacks;
public README tok/s tables remain focused on primary Gemma/Qwen evidence until
paired benchmark artifacts exist. Mixtral, DeepSeek full, and unlisted
Gemma/Qwen variants stay unsupported by default. Use `mlx_lm_delegated` or
`llama_cpp` only when the caller explicitly wants a compatibility adapter.

Before promoting another architecture, run:

```text
scripts/probe_mlx_model_support.py --model-dir <model-dir>
```

A model should report `repo_owned_runtime_ready` only when its manifest, local
reference files, and runtime path are all present.

## `mlx_lm_delegated`

Use `mlx_lm_delegated` only when the caller explicitly opts into upstream
`mlx-lm` serving for an MLX text model that AX does not yet own. This is a
compatibility adapter, not an AX deployment default.

This path requires a running `mlx_lm.server`:

```text
mlx_lm.server --model /path/to/local/mlx-model --host 127.0.0.1 --port 8090

ax-engine-server \
  --support-tier mlx-lm-delegated \
  --mlx-lm-server-url http://127.0.0.1:8090
```

GLM 4.7 Flash is directly supported through the repo-owned MLX graph and the
`glm4.7-flash-4bit` preset selects the native MLX tier. It can still be served
through `mlx_lm_delegated` only by selecting the delegated tier explicitly:

```text
mlx_lm.server --model mlx-community/GLM-4.7-Flash-4bit --host 127.0.0.1 --port 8090

ax-engine-server \
  --model-id glm4_moe_lite \
  --support-tier mlx-lm-delegated \
  --mlx-lm-server-url http://127.0.0.1:8090
```

When omitted, GLM 4.7 Flash uses the native MLX graph directly.

Supported delegated surfaces:

- blocking text generation
- SSE text generation
- OpenAI-compatible text completion and chat response shapes
- text sampling fields forwarded to upstream where supported

Boundaries:

- text-only
- token-array prompts fail closed
- multimodal inputs fail closed
- streamed chunks are delegated text deltas, not AX-owned token IDs
- KV state and model-kernel throughput belong to upstream `mlx-lm`, not AX
- benchmark rows must be labeled as delegated route-contract evidence

`mlx-swift-lm` remains a benchmark/reference adapter where admitted by the
benchmark harness. It is not the default delegated backend.

## `llama_cpp`

Use `llama_cpp` for GGUF models and non-MLX local inference. AX keeps the same
server, SDK, and benchmark surfaces, but model execution is delegated to
llama.cpp.

This path can target a running llama.cpp server:

```text
llama-server -m /path/to/model.gguf --host 127.0.0.1 --port 8081

ax-engine-server \
  --support-tier llama-cpp \
  --llama-server-url http://127.0.0.1:8081
```

or a configured llama.cpp CLI path where the SDK/server command supports it.

Supported delegated surfaces depend on the configured llama.cpp adapter, but
the intended route is local text generation through AX's server and SDK
contracts.

Boundaries:

- not AX-owned MLX runtime support
- not prompt-hash parity with MLX rows unless an artifact explicitly proves it
- benchmark rows are delegated route-contract or shape-compatible external
  reference evidence
- performance numbers must not be merged into AX-owned MLX throughput tables
  without clear labeling

Local `.gguf` paths require explicit `llama_cpp` selection. The default server
route remains the repo-owned MLX runtime and will not silently turn GGUF inputs
into AX-owned support claims.

## Choosing A Path

| Situation | Choose | Why |
| --- | --- | --- |
| You want AX-owned performance and token/KV behavior for a listed family | Direct support | AX owns the MLX graph and runtime policy |
| You have an MLX text model that `mlx-lm` already serves but AX does not own | `mlx_lm_delegated` | Keeps AX API surfaces while upstream runs the model |
| You have GGUF weights or a non-MLX local model | `llama_cpp` | llama.cpp is the delegated local inference route |
| You have Gemma4 unified image/audio/video inputs already preprocessed into AX's validated `multimodal_inputs.gemma4_unified` tensor contract | Direct support | Native MLX can consume processed media tensors without raw media decoding in the hot path; OpenAI-shaped routes require pre-tokenized prompt tokens for span alignment |
| You need client-side preprocessing for image URLs/data URIs, WAV audio URLs/data URIs, OpenAI-style `input_audio` WAV base64, or decoded video frames | Direct support through the Python helper | The helper prepares the processed tensor contract before the request reaches the optimized runtime |
| You need server-side raw OpenAI media content-part decoding on native Gemma4 unified chat | Direct support when documented for that format | Inline PNG/JPEG, WAV/MP3, GIF, and FFmpeg-backed MP4/WebM are decoded into AX's processed tensor contract before the optimized runtime |
| You need multimodal input on delegated routes, remote media URL fetching, or encoded video decoding without `ffmpeg` | Unsupported unless explicitly documented elsewhere | Delegated routes are text-first, remote fetching is intentionally disabled, and MP4/WebM codec decode is outside MLX tensor kernels |

## Evidence Rules

Do not merge the three paths into one unlabeled model-support or throughput
table.

| Evidence type | Supports | Does not support |
| --- | --- | --- |
| MLX inference-stack artifacts from `scripts/bench_mlx_inference_stack.py` | Direct-support AX-vs-reference performance claims with matching `mlx_lm.benchmark` rows | Broad serving, concurrency, or unsupported-model claims |
| `ax-engine-bench` scenario/replay/matrix artifacts | Route, correctness, determinism, replay, regression, and delegated contract evidence | Raw model-inference throughput unless explicitly designed for that metric |
| `mlx_lm_delegated` checks | AX API compatibility with upstream `mlx_lm.server` | AX-owned token IDs, KV state, or MLX throughput |
| llama.cpp delegated artifacts | Non-MLX route-contract and backend prompt-cache behavior | AX-owned MLX throughput |

For benchmark methodology and artifact contracts, see
[`BENCHMARKS.md`](BENCHMARKS.md) and [`PERFORMANCE.md`](PERFORMANCE.md).

## Future Models

AX should not force an all-or-nothing answer for new model generations.
Depending on readiness, a new model may be:

- direct support after repo-owned graph implementation and evidence
- `mlx_lm_delegated` if upstream `mlx-lm` can serve it as text
- `llama_cpp` if the user has a GGUF/non-MLX route
- unsupported until one of those paths is credible

Support claims must be earned by actual validation evidence. For MLX support
claims, that evidence must name the reference runtime, AX decode mode, model
identity, prompt shape, host readiness state, and whether the row came from the
MLX inference stack or an `ax-engine-bench` workload-contract artifact.
