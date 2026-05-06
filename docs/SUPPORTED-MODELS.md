# Supported Models

AX Engine routes inference through three labeled shipping paths:

- repo-owned MLX runtime for supported Apple Silicon model artifacts
- explicit `mlx-lm` delegated compatibility for unsupported MLX text models
- `llama.cpp` delegated compatibility for GGUF and non-MLX inference

## Support Strategy

AX does not treat model support as a simple yes-or-no list.

Instead, support should be understood through:

- support tier
- selected backend
- capability set

This matters because a future model may be:

- available through the repo-owned MLX runtime
- available through the explicit `mlx-lm` delegated path
- available through the delegated `llama.cpp` path
- not supported yet

The current default route is:

- explicit repo-owned MLX requests route to `mlx`
- explicit `mlx_lm_delegated` routes to a user-provided `mlx_lm.server`
- all non-MLX inference routes to `llama.cpp`
- local `.gguf` model paths route to `llama.cpp`
- retired AX native mode is no longer a supported user-facing inference mode

The current routing decision is recorded in the private ADR set.

## Runtime Baseline

The repo-owned MLX runtime targets Apple Silicon Macs where the MLX runtime and
AX Engine's model graph are both available.

This means:

- repo-owned MLX runtime is the local Mac path for supported model artifacts
- `mlx_lm_delegated` is compatibility, not the repo-owned runtime
- non-MLX local inference should use delegated `llama.cpp`
- support-tier language should not be read as implying broad model support
- retired AX native mode should not be exposed as a shipping runtime
- model-inference benchmark claims for repo-owned MLX runtime must come from
  `scripts/bench_mlx_inference_stack.py` with a matching required
  `mlx_lm.benchmark` primary baseline and, optionally, an explicit
  `mlx-swift-lm` `BenchmarkHelpers` / `MLXLMCommon` secondary baseline adapter
- `ax-engine-bench` scenario, replay, matrix, compare, and autotune artifacts
  describe workload-contract evidence; delegated llama.cpp and
  `mlx_lm_delegated` manifests describe route-contract evidence only

## Support Tiers

### `mlx_certified`

Meaning:

- reserved for a future certified repo-owned runtime
- not currently used for the repo-owned MLX runtime
- must not be assigned without benchmark and correctness evidence

### `mlx_preview`

Meaning:

- repo-owned MLX runtime is selected
- MLX runtime behavior is still preview-grade unless explicitly certified
- not a claim that every MLX model architecture is supported

### `llama_cpp`

Meaning:

- request is handled through `llama.cpp`
- this should not be read as equivalent to repo-owned runtime feature coverage
  or performance
- `vLLM`, `mistral.rs`, and unlabeled MLX adapters are not part of the current
  shipping inference route

### `mlx_lm_delegated`

Meaning:

- request is handled by an explicitly configured `mlx_lm.server`
- this is broad MLX text-model compatibility, not repo-owned MLX runtime support
- supports text-only blocking generation, fake SSE over AX streaming endpoints,
  and OpenAI-compatible text completion/chat response shapes
- token-array prompts and multimodal inputs fail closed
- benchmark evidence must be labeled as delegated route-contract evidence, not
  repo-owned MLX throughput
- `mlx-swift-lm` remains a secondary benchmark/reference adapter, not the
  default delegated backend

### `unsupported`

Meaning:

- AX does not currently have a credible path for that model request

## Current Runtime Direction

The current runtime direction is:

- use `mlx` for explicit repo-owned MLX runtime requests
- use `mlx_lm_delegated` only when explicitly requested for MLX text-model
  compatibility
- use `llama_cpp` for all non-MLX inference
- promote MLX preview models only after reference-runtime comparison,
  correctness smoke coverage, and public benchmark artifacts are available

## Current Repo-Owned MLX Preview Models

| Family | Model | Evidence |
|---|---|---|
| Gemma 4 | gemma-4-e2b-it, gemma-4-e4b-it, gemma-4-26b-a4b-it, gemma-4-31b-it | MLX stack benchmark + workload-contract scenario; E2B affine 4/5/6/8-bit, 26B A4B MoE, and 31B dense have MLX stack benchmark + server smoke; E4B model manifest and scenario manifest are present, MLX stack benchmark run pending |
| Qwen 3.5 | Qwen3.5-9B | MLX stack benchmark + workload-contract scenario |
| Qwen 3.6 | Qwen3.6-35B-A3B 4/5/6/8-bit MLX | MLX stack benchmark, server smoke, Qwen3.5-MoE manifest regression test |
| Qwen 3 Coder Next | Qwen3-Coder-Next-4bit | MLX stack benchmark, server smoke, Qwen3Next MoE/linear-attention regression tests |

## Reference-Only MLX Community Checks

The following checks answer a narrower question: can upstream `mlx-lm` load and
benchmark the downloaded community model today? They are not repo-owned AX
runtime support until the model has an AX `model-manifest.json`, a hand-written
graph in `ax-engine-mlx`, server smoke coverage, and MLX stack benchmark rows
that include AX runtime results.

Use `scripts/probe_mlx_model_support.py --model-dir <model-dir>` before
starting or promoting a repo-owned implementation for a new MLX architecture.
The probe reads the model config, safetensors index, manifest readiness, and
local reference implementations, then classifies the artifact as
repo-owned-runtime-ready, an implementation candidate, known family, partial
reference, or unknown architecture with explicit blockers.

| Model | Config model_type | Current AX status | Latest local evidence |
|---|---|---|---|
| `mlx-community/GLM-4.7-Flash-4bit` | `glm4_moe_lite` | Repo-owned MLX runtime ready | `mlx_lm.benchmark`, `mlx_swift_lm`, refreshed `ax_engine_mlx` direct, and `ax_engine_mlx_ngram_accel` benchmarks passed on 2026-05-06; support probe reports `repo_owned_runtime_ready`; latest median AX direct decode is 94.8 tok/s at 128 prompt tokens and 94.1 tok/s at 512 prompt tokens; n-gram effective decode is 260.7 tok/s at 128 prompt tokens and 253.5 tok/s at 512 prompt tokens |
| `mlx-community/DeepSeek-V4-Flash-2bit-DQ` | `deepseek_v4` | Fail closed: partial reference only, not repo-owned AX support | Downloaded on 2026-05-06; `mlx_lm.benchmark` failed with `Model type deepseek_v4 not supported`; support probe finds the available SwiftLM port drops compressor/indexer and `tid2eid` hash-routing weights that are present in the checkpoint |

## Current Limitations And Problems

Gemma 4 E4B has model and scenario manifests but no MLX stack benchmark run
yet; its public benchmark rows are pending. All other repo-owned MLX preview
models above have completed benchmark rows.
Gemma 4 26B A4B MoE and Gemma 4 E2B 5/6/8-bit rows include both `mlx_lm` and
admitted `mlx_swift_lm` reference rows.
N-gram acceleration rows remain effective-throughput measurements from AX's
n-gram policy and must not be described as raw model-kernel speedups.

## Future Model Generations

If a newer model appears in the future, for example a later Qwen or Gemma
generation, AX should not force an all-or-nothing answer.

Depending on readiness, that model may resolve to:

- `mlx_preview`
- `mlx_lm_delegated`
- `llama_cpp`
- `unsupported`

where `mlx_preview` means repo-owned MLX runtime, and `mlx_lm_delegated`
means upstream `mlx-lm` compatibility through AX surfaces.

## Not A Broad Model Matrix

AX Engine is not trying to support every architecture through the repo-owned
runtime early. Unsupported models should resolve to a delegated compatibility
path or to `unsupported`, with that route visible in runtime metadata.

Deferred from the main path:

- multimodal models
- hybrid-only architectures
- MoE-first support
- broad delegated exceptions that lack route-contract evidence

## Important Note

This document describes product direction and support strategy, not a universal
backend guarantee.
The implementation is still in progress, and support claims must be earned by
actual benchmark and validation evidence. For MLX support claims, that evidence
must name the MLX reference runtime, AX decode mode, model identity, prompt
shape, host readiness state, and whether the row came from the MLX inference
stack or from an `ax-engine-bench` workload-contract artifact.
