# Supported Models

AX Engine supports LLMs through a direct-first runtime contract. Direct support
is the default deployment path; delegated adapters are explicit compatibility
paths for migration, validation, or external reference rows. The path matters
because it defines who runs the model graph, which API features are available,
and what benchmark claims are allowed.

**Related:** [Getting Started](GETTING-STARTED.md) · [CLI](CLI.md) ·
[Model Support Policy](MODEL-SUPPORT-POLICY.md) · [MTP Docs](mtp/README.md) ·
[FAQ](FAQ.md)

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

## Native multimodal and speech support

This table describes the repo-owned MLX implementations, not every architecture
mlxcel can load. AX selects high-use models whose text backbone, media tower,
preprocessor, token expansion, cache behavior, and server contract can all be
validated locally. A family name alone is not enough: `/v1/models` derives its
media capabilities from the loaded manifest and fails closed when required
tower tensors are absent.

| Priority | Model / manifest family | Accepted input | Native surface | Implementation and validation boundary |
| --- | --- | --- | --- | --- |
| P0 | Qwen3-VL dense / MoE (`qwen3_vl`, `qwen3_vl_moe`) | Image, multi-image, video | OpenAI chat and `/v1/generate` | Conv3D temporal patches, factorized position interpolation, complete vision blocks and merger, DeepStack injection, and multimodal RoPE; dense 4B image path smoke-tested on Apple M3 |
| P0 | Visual Qwen3.5 (`qwen3_5` with `vision_tower.*`) | Image, multi-image, video | OpenAI chat and `/v1/generate` | Shares the Qwen visual stack with the Qwen3.5 gated-delta text backbone; text-only Qwen3.5 manifests do not advertise media; 4B image path smoke-tested on Apple M3 |
| P0 | MiniCPM-V 4.6 (`minicpmv4_6`) | Image, multi-image | OpenAI chat and `/v1/generate` | Dynamic SigLIP grid, 27-layer tower, layer-6 `VitMerger`, final pixel-shuffle merger, and version-specific placeholder expansion; OCR smoke-tested on Apple M3 |
| P1 | NVIDIA Nemotron 3 Nano Omni 30B-A3B (`nemotron_h` with media tensors) | Image, audio, ordered image+audio | OpenAI chat and `/v1/generate` | RADIO vision, connector MLP, Parakeet/Conformer audio, exact STFT framing, and mixed-media spans; image, JFK audio, and combined prompts smoke-tested on Apple M3 |
| P1 | Unlimited-OCR (`unlimited_ocr`) | Image | Processed native runtime input; OpenAI inline-image compatibility through the explicit vLLM Unlimited-OCR profile | SAM+CLIP dual vision and DeepSeek MoE language graph. Protected-prefix R-SWA keeps the entire image/text prefill and rotates only generated-token KV |
| P2 | Gemma 4 unified 12B (`gemma4` manifest with unified media roles) | Image, audio, video | OpenAI chat and `/v1/generate` | Encoder-free image/audio connector and sampled video frames. Gemma 4 E2B/E4B/26B/31B text manifests remain text-only unless their manifest actually contains compatible media roles |
| P2 | Whisper large-v3-turbo (`whisper`) | Audio | `/v1/audio/transcriptions`, `/v1/audio/translations`, `EngineSession::transcribe_audio` | Dedicated encoder-decoder ASR runtime, canonical multilingual vocabulary, log-mel frontend, KV-cached greedy decoding, transcribe/translate, WAV/MP3 decode; JFK transcription smoke-tested through both native and HTTP paths on Apple M3 |

Whisper is intentionally not exposed through text generation endpoints.
Transcription accepts `json`, `text`, or `verbose_json`; the current native
contract is temperature-0 greedy decoding without prompts or word/segment
timestamps. GLM-OCR is not supported and was not added.

Inline server media is data-URI/base64 only. Remote URL fetching remains
disabled. Qwen and Gemma accept bounded sampled video; MiniCPM and Nemotron do
not. See [Server](SERVER.md) for request shapes and
capability-discovery fields.

## Multi-model serving

`ax-engine-server` can keep more than one **direct** MLX model loaded in a
single process (`POST /v1/model/load` with `load_mode=add`). This is a product
allowlist, not “any downloaded model”:

| Allowlisted multi-model targets | Notes |
| --- | --- |
| Qwen 3.5 9B, Qwen 3.6 27B / 35B | Manifest family + architecture signature must match the requested `model_id` |
| Qwen3-Coder-Next | Same manifest-authoritative check (`qwen3_next` 512-expert signature) |
| Gemma 4 12B / 26B / 31B | Same manifest-authoritative check |
| EmbeddingGemma 300M, Qwen3-Embedding 0.6B / 4B / 8B | Embedding co-residency: serve `/v1/chat/completions` and `/v1/embeddings` from one process, routed by `model` |

The [AutomatosX org packages](https://huggingface.co/AutomatosX) publish these
targets under `AX-`-branded names (for example
`AutomatosX/AX-Qwen3.6-27B-MLX-OptiQ-4bit-MTP`); the branded id and the bare
family label resolve to the same allowlist target, and the retained
`model-manifest.json` stays authoritative either way.

Other families remain single-model hot-swap (`load_mode=replace` with only one
resident model) or fail closed when `add` would leave a multi-model registry.
Routing, memory preflight, idle eviction, and arbiter behavior:
[Server: Multi-model serving](SERVER.md#multi-model-serving).

## Getting Model Artifacts

AX Engine requires pre-sanitized MLX safetensors plus a `model-manifest.json`.
Two recommended sources:

- [AutomatosX](https://huggingface.co/AutomatosX) — AX-branded packs built for
  this engine. Chat packs bundle the speculative-decode extras in one repo
  (Qwen: fused `mtp.safetensors` sidecar; Gemma: `assistant/` weights plus the
  `ax_gemma4_assistant_mtp.json` contract) and ship a pre-generated
  `model-manifest.json`, so one download produces a serve-ready MTP directory
  with no separate `download-mtp` step. OptiQ variants carry mixed 4/8-bit
  per-tensor quantization; embedding packs cover EmbeddingGemma 300M and
  Qwen3-Embedding 0.6B/4B/8B. The managed catalog contains all 25 public
  repositories in the organization; every currently published Qwen 3.5,
  Qwen 3.6, and Gemma 4 variant is supported.
- [mlx-community](https://huggingface.co/mlx-community) — community MLX
  snapshots, already converted and validated. **Not download-managed**: their
  aliases below stay serve aliases for artifacts you already have, and
  downloads take the explicit raw `org/repo` form; two-repo MTP packaging
  uses the `download-mtp` flow below.

`ax-engine download`, `download_model()`, and `scripts/download_model.py`
download weights and auto-generate the manifest when a repo does not ship one.

List managed download aliases:

```text
ax-engine download --list
```

Download by managed alias:

```text
ax-engine download ax-qwen3.5-9b --json
ax-engine download ax-qwen3.6-35b --json
ax-engine download ax-qwen3.6-27b --json
ax-engine download ax-gemma4-12b --json
ax-engine download ax-gemma4-31b --json
ax-engine download ax-qwen3-coder-next --json
ax-engine download ax-embeddinggemma-300m --json
ax-engine download ax-qwen3-embedding-4b --json
```

Download and serve in one command:

```text
ax-engine serve ax-qwen3.6-35b --download --port 31418
ax-engine serve ax-qwen3-coder-next --download --port 31418
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
ax-engine download ax-qwen3.6-35b --dest /Volumes/Models/ax-qwen3.6-35b
```

Python SDK:

```python
from ax_engine import download_model

path = download_model("mlx-community/Qwen3.6-35B-A3B-4bit")
```

Built-in aliases. The AutomatosX tables are the managed download catalog; the
mlx-community-backed tables are **serve aliases** — they resolve presets and
already-downloaded artifacts, while their downloads take the raw repo id form:

**Serve aliases — primary productivity (Gemma / Qwen / GLM)**

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

**Managed download aliases — AutomatosX packs (MTP / assistant bundled,
manifest included)**

The bare alias selects the flagship OptiQ (Qwen/Gemma 4-bit) or DWQ
(embeddings) build; `-4bit` / `-6bit` variants select the plain quants. These
aliases promise the exact AutomatosX repo, so serve them through the
idempotent download flow: `ax-engine serve ax-qwen3.6-27b --download`.

| Alias | Repo |
| --- | --- |
| `ax-qwen3.5-9b`[`-4bit`,`-6bit`] | `AutomatosX/AX-Qwen3.5-9B-MLX-{OptiQ-4bit,4bit,6bit}-MTP` |
| `ax-qwen3.6-27b`[`-4bit`,`-6bit`] | `AutomatosX/AX-Qwen3.6-27B-MLX-{OptiQ-4bit,4bit,6bit}-MTP` |
| `ax-qwen3.6-35b`[`-4bit`,`-6bit`] | `AutomatosX/AX-Qwen3.6-35B-A3B-MLX-{OptiQ-4bit,4bit,6bit}-MTP` |
| `ax-gemma4-12b`[`-4bit`,`-6bit`] | `AutomatosX/AX-Gemma-4-12B-IT-MLX-{QAT-OptiQ-4bit,QAT-4bit,6bit}-Assistant-MTP` |
| `ax-gemma4-26b`[`-4bit`,`-6bit`] | `AutomatosX/AX-Gemma-4-26B-A4B-IT-MLX-{OptiQ-4bit,QAT-4bit,6bit}-Assistant-MTP` |
| `ax-gemma4-31b`[`-4bit`,`-6bit`] | `AutomatosX/AX-Gemma-4-31B-IT-MLX-{OptiQ-4bit,QAT-4bit,6bit}-Assistant-MTP` |
| `ax-qwen3-coder-next`[`-6bit`] | `AutomatosX/AX-Qwen3-Coder-Next-MLX-{4,6}bit` |
| `ax-diffusiongemma-26b` | `AutomatosX/AX-DiffusionGemma-26B-A4B-IT-MLX-4bit` |
| `ax-embeddinggemma-300m` | `AutomatosX/AX-EmbeddingGemma-300M-MLX-8bit` |
| `ax-qwen3-embedding-0.6b` | `AutomatosX/AX-Qwen3-Embedding-0.6B-MLX-8bit` |
| `ax-qwen3-embedding-4b` | `AutomatosX/AX-Qwen3-Embedding-4B-MLX-4bit-DWQ` |
| `ax-qwen3-embedding-8b` | `AutomatosX/AX-Qwen3-Embedding-8B-MLX-4bit-DWQ` |

**Serve aliases — secondary research / enterprise Llama**

| Alias | Repo |
| --- | --- |
| `llama3.1-8b` | `mlx-community/Llama-3.1-8B-Instruct-4bit` |
| `llama3.3-70b` | `mlx-community/Llama-3.3-70B-Instruct-4bit` |
| `llama4-scout` | `mlx-community/Llama-4-Scout-17B-16E-Instruct-4bit` |

**Serve aliases — secondary European market Mistral**

| Alias | Repo |
| --- | --- |
| `mistral-small` | `mlx-community/Mistral-Small-3.1-24B-Instruct-2503-4bit` |
| `ministral-8b` | `mlx-community/Ministral-8B-Instruct-2410-4bit` |
| `devstral-small` | `mlx-community/Devstral-Small-2505-4bit` |

**Serve aliases — secondary open reasoner GPT-OSS (MXFP4)**

| Alias | Repo | Notes |
| --- | --- | --- |
| `gpt-oss-20b` | `mlx-community/gpt-oss-20b-MXFP4-Q4` | Comfortable on 64–128 GB |
| `gpt-oss-120b` | `mlx-community/gpt-oss-120b-MXFP4-Q4` | Prefer 128 GB+; experts stay MXFP4-packed at runtime |

Leave downloads in the Hugging Face Hub cache by default. The cache is shared
with `mlx_lm` and other HF-aware tools, avoiding duplicate copies of large
weights. Use `--dest` only when you want an explicit copy outside the shared
cache.

### MTP Downloads

The AutomatosX packs above bundle their MTP artifacts in the direct download —
`ax-engine download ax-qwen3.6-27b` already produces an MTP-ready directory,
so `download-mtp` is not needed for them.

For the mlx-community bases, `ax-engine download-mtp` is the one-command path
for supported local-agent MTP targets. It downloads the base model and
prepares AX MTP artifacts when the model family has a repo-owned packaging
path. The CLI command accepts the
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
ax-engine serve ~/.cache/huggingface/hub/models--mlx-community--Qwen3-4B-4bit/snapshots/<hash> --port 31418
```

### Raw Hugging Face Checkpoints

Raw checkpoints need sanitization before AX Engine can load them:

```text
pip install mlx-lm
mlx_lm.convert --hf-path <org/model> --mlx-path /path/to/dest -q --q-bits 4
ax-engine-bench generate-manifest /path/to/dest
ax-engine serve /path/to/dest --port 31418
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
| Gemma 4 unified | `gemma-4-12b-it` | Repo-owned MLX runtime; MLX affine 4/5/6-bit weights where available; assistant-MTP packaging; processed image/audio/video on server routes only when the manifest contains all required media roles | Unified 12B connector; sliding-window + full attention; K=V full-attention layers; logit softcapping |
| Gemma 4 text | `gemma-4-e2b-it`, `gemma-4-e4b-it`, `gemma-4-26b-a4b-it`, `gemma-4-31b-it` | Repo-owned text runtime; MLX affine 4/5/6-bit weights where available; assistant-MTP packaging for matched `*-assistant` drafters | Per-layer embedding, dense, and MoE variants; no media capability is inferred from the Gemma family name |
| Qwen 3 | `Qwen3-4B-4bit` and manifest-backed Qwen 3 dense checkpoints | Repo-owned MLX runtime | SwiGLU dense FFN; per-head QK norm; optional MoE variants require manifest evidence |
| Qwen 3.5 | `Qwen3.5-9B-MLX-4bit` / `qwen3.5-9b` preset; visual checkpoints with `vision_tower.*` | Repo-owned MLX runtime; MLX affine 4-bit and OptiQ mixed 4/8-bit weights; image/video when the tower is present | GatedDeltaNet linear attention + dense SwiGLU FFN; `attn_output_gate` per-head interleaving; visual variants reuse the Qwen3-VL tower |
| Qwen3-VL | Dense and MoE `qwen3_vl*` MLX checkpoints | Repo-owned image/video chat runtime | Conv3D patch embed, full vision stack and merger, DeepStack, multimodal RoPE |
| MiniCPM-V 4.6 | `minicpmv4_6` MLX checkpoints | Repo-owned image/multi-image chat runtime | SigLIP + mid-tower/final mergers; OCR/document use |
| Nemotron 3 Nano Omni | `nemotron_h` manifests containing `vision_model.*` / `sound_encoder.*` | Repo-owned image/audio/mixed chat runtime | RADIO vision + Parakeet audio on the Nemotron-H hybrid backbone |
| Unlimited-OCR | `unlimited_ocr` | Preview repo-owned processed-image runtime | Dual vision, DeepSeek MoE, protected-prefix decode ring |
| Whisper large-v3-turbo | `whisper` | Repo-owned speech transcription/translation runtime | Dedicated audio endpoints; not a chat/text-generation model |
| Qwen 3.6 / Coder Next | `Qwen3.6-35B-A3B` 4-bit MLX, `Qwen3.6-27B` 4/5/6-bit MLX, `Qwen3-Coder-Next-4bit` | Repo-owned MLX runtime | `qwen3_next`: GatedDelta linear attention, full attention with per-head sigmoid gate, sparse top-k MoE with shared expert |
| GLM 4.7 Flash | `glm4_moe_lite` / `glm4.7-flash-4bit` | Repo-owned MLX runtime; MLX affine 4-bit weights | Flash MLA attention, sigmoid-routed MoE with dense+MoE layer split, shared expert; post-attention RMS norm |

**Secondary (preview direct — serve aliases + server presets; share standard / family graphs)**

| Family | Direct model IDs / aliases | Current scope | Notes |
| --- | --- | --- | --- |
| Llama 3.x | `llama3.1-8b`, `llama3.3-70b` | Preview direct; MLX affine 4-bit; `llama3` / standard dense path | Research and enterprise baseline; Llama 3 chat template |
| Llama 4 | `llama4-scout` | Preview direct; MLX affine 4-bit; `llama4` MoE path | Fits ~128 GB class; Maverick full 4-bit is out of scope for that class |
| Mistral | `mistral-small`, `ministral-8b`, `devstral-small` | Preview direct; MLX affine 4-bit; `mistral3` / standard path | European market chat + coding; Instruct `[INST]` chat fallback |
| GPT-OSS | `gpt-oss-20b`, `gpt-oss-120b` | Preview direct; MXFP4 MoE experts **kept packed** at load (`gather_qmm` mode=`mxfp4`), including OptiQ mixed MXFP4/affine overrides | MoE decoder with 128 experts (top-4), SwiGLU, alternating full/sliding-128 attention, per-head learned attention sinks, YaRN RoPE (128K), GQA (64q/8k heads). Prefer MXFP4-Q4 downloads. Expert residency stays ~4-bit so 120B is practical on 128 GB class hosts; attention/router tensors remain higher precision |

Experimental direct-support model families:

| Family | Model ID | Current scope | Notes |
| --- | --- | --- | --- |
| DiffusionGemma | `mlx-community/diffusiongemma-26B-A4B-it-4bit` | Experimental repo-owned MLX block-diffusion path | Experimental rows live under [Performance Results](PERFORMANCE-RESULTS.md#diffusiongemma); benchmark boundary is first committed 256-token diffusion block, not autoregressive TTFT/decode. See [DiffusionGemma experimental support](DIFFUSIONGEMMA.md). |

All direct-support models use MLX safetensors format with the AX
`model-manifest.json` descriptor. Adding a new direct-support architecture
means implementing the model graph, not wiring up a generic loader.

Architecture code alone is not a certified performance claim. Secondary
families ship as **preview direct** with download aliases and chat fallbacks;
public performance tok/s tables remain focused on primary Gemma/Qwen evidence until
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

## `vllm` delegated CUDA candidate

The `vllm` tier preserves AX server, SDK, admission, error, and OpenAI wire
contracts while a separately supervised vLLM process owns CUDA execution. One
provider implementation serves both Linux CPU architectures; platform
differences are exact runtime profiles:

| Profile | CPU/GPU target | Current evidence |
| --- | --- | --- |
| `cuda-linux-aarch64-thor-sm110` | NVIDIA Thor, `aarch64`, SM110 | Candidate: exact preflight plus real Unlimited-OCR non-stream, stream, parity, and admission tests |
| `cuda-linux-x86_64-a6000-sm86` | RTX A6000, `x86_64`, SM86 | Candidate: exact preflight plus real text-generation provider smoke; Unlimited-OCR and soak gates remain required |

Supported candidate surfaces:

- blocking and SSE OpenAI chat/completions
- exact `/v1/models` readiness and public/upstream model identity separation
- ordered text plus inline PNG/JPEG data-URI content for the
  `unlimited-ocr` model profile
- usage-only final stream chunks and `[DONE]`
- explicit bounded admission with typed 429 responses

Boundaries:

- no generation POST retry
- no worker lifecycle ownership and no silent backend fallback
- remote media URLs, audio, video, tools, and unknown provider extensions fail
  closed on the Unlimited-OCR route
- performance and GA claims are restricted to a named model, profile, artifact,
  and hardware receipt
- TensorRT-LLM and TensorRT Edge-LLM remain separate optimized providers; they
  are not aliases or automatic fallbacks for `vllm`

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
| You need CUDA OCR/VLM compatibility on a certified Thor or CUDA PC | `vllm` | Keeps one AX API/provider contract while the independent runtime profile owns CUDA details |
| You have GGUF weights or a non-MLX local model | `llama_cpp` | llama.cpp is the delegated local inference route |
| You have Gemma4 unified image/audio/video inputs already preprocessed into AX's validated `multimodal_inputs.gemma4_unified` tensor contract | Direct support | Native MLX can consume processed media tensors without raw media decoding in the hot path; manual OpenAI-shaped extension payloads require pre-tokenized prompt tokens for span alignment |
| You need client-side preprocessing for image URLs/data URIs, WAV audio URLs/data URIs, or OpenAI-style `input_audio` WAV base64 | Direct support through the Python helper | The helper prepares the processed tensor contract before the request reaches the optimized runtime |
| You need server-side raw OpenAI media content-part decoding on native Gemma4 unified chat | Direct support for image/audio/video | Inline PNG/JPEG, WAV/MP3, and bounded sampled GIF/MP4/WebM are decoded into AX's processed tensor contract before the optimized runtime |
| You need Qwen3-VL or visual Qwen3.5 image/video chat | Direct support | The manifest must contain the Qwen vision tower; inline media is expanded into ordered image/video spans |
| You need MiniCPM-V 4.6 OCR or multi-image chat | Direct support | The version-specific SigLIP/merger and placeholder path is selected from the manifest |
| You need Nemotron Omni image, audio, or mixed-media chat | Direct support | Both media towers are discovered from the loaded manifest; video is unsupported |
| You need local Whisper large-v3-turbo transcription or translation | Direct support | Use the dedicated OpenAI audio endpoints or Rust SDK; text generation fails closed |
| You need inline PNG/JPEG OCR through the certified vLLM profile | `vllm` | The Unlimited-OCR profile preserves ordered text/image parts and validates data URIs before forwarding |
| You need remote media URL fetching or video on delegated routes | Unsupported | Remote fetching and video are intentionally disabled |

## Evidence Rules

Do not merge the three paths into one unlabeled model-support or throughput
table.

| Evidence type | Supports | Does not support |
| --- | --- | --- |
| MLX inference-stack artifacts from `scripts/bench_mlx_inference_stack.py` | Direct-support AX-vs-reference performance claims with matching `mlx_lm.benchmark` rows | Broad serving, concurrency, or unsupported-model claims |
| `ax-engine-bench` scenario/replay/matrix artifacts | Route, correctness, determinism, replay, regression, and delegated contract evidence | Raw model-inference throughput unless explicitly designed for that metric |
| `mlx_lm_delegated` checks | AX API compatibility with upstream `mlx_lm.server` | AX-owned token IDs, KV state, or MLX throughput |
| vLLM delegated artifacts | AX API/wire compatibility on the named CUDA profile and model | AX-owned kernels or claims for an untested CUDA SKU |
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
