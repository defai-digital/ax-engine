# Supported LLM Models

AX Engine supports LLMs through three explicit runtime paths. The path matters
because it defines who runs the model graph, which API features are available,
and what benchmark claims are allowed.

| Path | Use it for | Who runs the model | What the result means |
|---|---|---|---|
| Direct support | Model families with a repo-owned `ax-engine-mlx` graph | AX Engine on MLX | AX-owned token/KV/runtime behavior; performance claims still require benchmark artifacts |
| `mlx_lm_delegated` | MLX text models that upstream `mlx-lm` supports before AX has a repo-owned graph | A user-provided `mlx_lm.server` | AX server/SDK compatibility over delegated text generation; not AX-owned MLX throughput |
| `llama_cpp` | GGUF models and non-MLX local inference | llama.cpp server or CLI | Delegated route-contract evidence; not AX-owned MLX throughput |
| Unsupported | Requests with no credible direct or delegated path | None | Fail closed |

Runtime metadata exposes the selected path through fields such as
`selected_backend`, `support_tier`, and `resolution_policy`. Preserve those
labels in benchmark artifacts and user-facing claims.

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

| Family | Direct model IDs | Current scope | Notes |
|---|---|---|---|
| Gemma 4 | `gemma-4-e2b-it`, `gemma-4-e4b-it`, `gemma-4-12b-it`, `gemma-4-26b-a4b-it`, `gemma-4-31b-it` | Repo-owned MLX runtime; MLX affine 4/5/6/8-bit weights where available; assistant-MTP packaging for matched `*-assistant` drafters; processed Gemma4 unified media tensors on the native generate API | Dense, unified 12B, per-layer embedding, and MoE variants; sliding-window + full attention; K=V full-attention layers; logit softcapping |
| Qwen 3 | `Qwen3-4B-4bit` and manifest-backed Qwen 3 dense checkpoints | Repo-owned MLX runtime | SwiGLU dense FFN; per-head QK norm; optional MoE variants require manifest evidence |
| Qwen 3.5 | `Qwen3.5-9B-MLX-4bit` | Repo-owned MLX runtime | Linear attention + MoE FFN; `attn_output_gate` per-head interleaving |
| Qwen 3.6 / Coder Next | `Qwen3.6-35B-A3B` 4-bit MLX, `Qwen3.6-27B` 4/5/6/8-bit MLX, `Qwen3-Coder-Next-4bit` | Repo-owned MLX runtime | `qwen3_next`: GatedDelta linear attention, full attention with per-head sigmoid gate, sparse top-k MoE with shared expert |

> GLM 4.7 Flash (`glm4_moe_lite`) was demoted from direct support to
> [`mlx_lm_delegated`](#mlx_lm_delegated). The repo-owned MLA graph remains
> in-tree but is no longer a promoted route: native decode only reaches parity
> with `mlx_lm` (see [`PERFORMANCE-DECODE-GAP.md`](PERFORMANCE-DECODE-GAP.md)),
> and the 4-bit export ships no MTP head, so AX's speculative decode cannot
> accelerate it. Serve GLM through mlx-lm passby.

All direct-support models use MLX safetensors format with the AX
`model-manifest.json` descriptor. Adding a new direct-support architecture
means implementing the model graph, not wiring up a generic loader.

Architecture code, tensor-role metadata, or comments are not public direct
support claims by themselves. LLaMA, Mistral, Mixtral, DeepSeek, and unlisted
Gemma/Qwen variants should use `mlx_lm_delegated` or `llama_cpp` when those
backends can serve them, or stay unsupported until repo-owned manifest, smoke,
and benchmark evidence are promoted here.

Before promoting another architecture, run:

```text
scripts/probe_mlx_model_support.py --model-dir <model-dir>
```

A model should report `repo_owned_runtime_ready` only when its manifest, local
reference files, and runtime path are all present.

## `mlx_lm_delegated`

Use `mlx_lm_delegated` when upstream `mlx-lm` can serve an MLX text model but
AX does not yet have a repo-owned graph for that architecture.

This path requires a running `mlx_lm.server`:

```text
mlx_lm.server --model /path/to/local/mlx-model --host 127.0.0.1 --port 8090

ax-engine-server \
  --support-tier mlx-lm-delegated \
  --mlx-lm-server-url http://127.0.0.1:8090
```

GLM 4.7 Flash (`glm4_moe_lite`) is the canonical delegated model. Its
`glm4.7-flash-4bit` preset now selects this tier, so serve it via mlx-lm passby:

```text
mlx_lm.server --model mlx-community/GLM-4.7-Flash-4bit --host 127.0.0.1 --port 8090

ax-engine-server \
  --preset glm4.7-flash-4bit \
  --mlx-lm-server-url http://127.0.0.1:8090
```

The preset fails closed when `--mlx-lm-server-url` is omitted; it does not fall
back to the native MLX graph. GLM was demoted because AX's native decode only
matches `mlx_lm` throughput and the 4-bit export has no MTP head for AX
speculation to exploit (see [`PERFORMANCE-DECODE-GAP.md`](PERFORMANCE-DECODE-GAP.md)).

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

Local `.gguf` paths resolve to `llama_cpp` rather than the repo-owned MLX
runtime.

## Choosing A Path

| Situation | Choose | Why |
|---|---|---|
| You want AX-owned performance and token/KV behavior for a listed family | Direct support | AX owns the MLX graph and runtime policy |
| You have an MLX text model that `mlx-lm` already serves but AX does not own | `mlx_lm_delegated` | Keeps AX API surfaces while upstream runs the model |
| You have GGUF weights or a non-MLX local model | `llama_cpp` | llama.cpp is the delegated local inference route |
| You have Gemma4 unified image/audio/video inputs already preprocessed into AX's validated `multimodal_inputs.gemma4_unified` tensor contract | Direct support | Native MLX can consume processed media tensors without raw media decoding in the hot path; OpenAI-shaped routes require pre-tokenized prompt tokens for span alignment |
| You need client-side preprocessing for image URLs/data URIs, WAV audio URLs/data URIs, OpenAI-style `input_audio` WAV base64, or decoded video frames | Direct support through the Python helper | The helper prepares the processed tensor contract before the request reaches the optimized runtime |
| You need server-side raw image/audio/video OpenAI media content-part decoding, encoded video container decoding, or multimodal input on delegated routes | Unsupported unless explicitly documented elsewhere | Current delegated routes are text-first and raw media preprocessing is not part of AX's OpenAI text adapter |

## Evidence Rules

Do not merge the three paths into one unlabeled model-support or throughput
table.

| Evidence type | Supports | Does not support |
|---|---|---|
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
