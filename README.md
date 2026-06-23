# AX Engine

AX Engine is a Mac-first LLM inference runtime for Apple Silicon developers who
want local models to be fast, inspectable, and easy to serve. It is not just a
wrapper around `mlx_lm`: for direct-support Gemma, Qwen, GLM, and
DiffusionGemma families, AX Engine owns the MLX graph path, KV/runtime behavior,
server route, model packaging, and benchmark contract.

## Why AX Engine

AX Engine is built to win the full interactive local-model path, not just report
one isolated kernel number. In the current public direct-mode matrix, AX Engine
is ahead of `mlx_lm` in every listed prefill and TTFT row; direct decode is
tracked separately, with peer rows and model-specific boundaries kept visible.

- **First-class MTP:** one-command MTP package preparation through
  `ax-engine download-mtp`, including the Gemma 4 12B 4-bit quick-start target
  plus recommended 6-bit MTP benchmarking and 4-bit comparison lanes.
- **Simple local serving:** install the wheel, download or prepare a model, then
  run the printed `ax-engine serve ...` command for OpenAI-compatible local
  endpoints.
- **Repo-owned direct runtime:** direct-support Gemma, Qwen, GLM, and
  DiffusionGemma paths run in AX Engine's MLX runtime; delegated `mlx-lm` and
  `llama.cpp` routes stay explicit instead of hidden behind one label.
- **Serious benchmarking:** public results are tied to checked-in artifacts that
  record route identity, model snapshot, prompt suite, sampler, cooldowns,
  repetitions, MTP accept rate, prefill, decode, TTFT, and dirty-state
  provenance.
- **Agent-oriented support:** Qwen3-Coder-Next is called out separately from the
  Qwen3.6 family because it carries a coding-first architecture and validation
  path.

## Table of Contents

- [Why AX Engine](#why-ax-engine)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Getting a Model](#getting-a-model)
- [Typical Hardware](#typical-hardware)
- [What AX Engine Does](#what-ax-engine-does)
- [Performance](#performance)
  - [Speculative Decoding (MTP)](#speculative-decoding-mtp)
    - [6-bit MTP acceleration refresh (2026-06-23)](#6-bit-mtp-acceleration-refresh-2026-06-23)
    - [4-bit MTP comparison lane (2026-06-20)](#4-bit-mtp-comparison-lane-2026-06-20)
    - [GLM-4.7 Flash MTP validation session](#glm-47-flash-mtp-validation-session)
  - [Direct Decode · Prefill · TTFT](#direct-decode--prefill--ttft)
    - [Gemma 4 12B](#gemma-4-12b)
      - [Gemma 4 12B Multimodal](#gemma-4-12b-multimodal)
    - [DiffusionGemma](#diffusiongemma)
    - [Gemma 4 and Qwen 3.6](#gemma-4-and-qwen-36)
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

**Install** (macOS 26 Tahoe or later, Apple Silicon only — see [Typical Hardware](#typical-hardware)):

Upgrade pip first so pip 23+ can find the macOS wheel, and keep the package
spec quoted for zsh.

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -U "ax-engine[download]<7"
```

**Download a Gemma 4 12B MTP package:**

```bash
ax-engine download-mtp gemma-4-12b-4bit
# Then run the printed "ax-engine serve ..." command.
```

**Send one request from another terminal:**

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"gemma-4-12b-it","messages":[{"role":"user","content":"Say hello in one sentence."}],"max_tokens":64}'
```

For model choices, SDK examples, Homebrew, and source builds, see the
[Getting Started guide](docs/GETTING-STARTED.md) and [SDK docs](docs/sdk/README.md).

> Quick Start requires **macOS 26 (Tahoe) or later** on **Apple Silicon M2 Max or newer**. The Gemma 4 12B MTP path is intended for high-memory machines; use the memory tiers listed in [Typical Hardware](#typical-hardware). Earlier macOS releases are not supported — there is no wheel or binary for them.

## Installation

For platform requirements, troubleshooting, optional extras, Homebrew, source
builds, and release-channel diagnostics, see the
[Getting Started installation guide](docs/GETTING-STARTED.md#installation).

## Getting a Model

AX Engine loads pre-sanitized MLX safetensors plus an AX
`model-manifest.json`. Use `ax-engine download --list` for direct-decode
aliases, `ax-engine serve <alias> --download` for one-command serving, and
`ax-engine download-mtp <target>` for supported MTP packages.
Detailed aliases, MTP targets, raw checkpoint conversion, cache behavior, and
manifest commands live in
[Supported Models](docs/SUPPORTED-MODELS.md#getting-model-artifacts) and the
[CLI reference](docs/CLI.md#ax-engine).

```bash
# Serve a direct model in one command.
ax-engine serve qwen36-35b --download --port 8080

# Or inspect and download direct-model artifacts separately.
ax-engine download --list
ax-engine download qwen36-35b --json

# Prepare a Gemma 4 12B MTP package.
ax-engine download-mtp gemma-4-12b-4bit
```

Common acquisition paths:

| Model/package | Command | Runtime path |
|---|---|---|
| Direct MLX models | `ax-engine download <alias-or-mlx-community-repo>` | Repo-owned MLX graph |
| Gemma 4 12B quick-start MTP | `ax-engine download-mtp gemma-4-12b-4bit` | Gemma assistant-MTP |
| Qwen3.6 27B / 35B-A3B MTP | `ax-engine download-mtp qwen3.6-27b-6bit` or `qwen3.6-35b-a3b` | Qwen fused MTP sidecar |
| Gemma 4 12B / 26B / 31B 6-bit MTP | `ax-engine download-mtp gemma-4-12b`, `gemma-4-26b`, or `gemma-4-31b` | Gemma assistant-MTP |
| GLM-4.7 Flash MTP | `ax-engine download-mtp glm-4.7-flash` | GLM built-in MTP sidecar |
| Raw Hugging Face checkpoints | Convert with `mlx_lm.convert`, then run `ax-engine-bench generate-manifest` | Direct only after sanitization |

Direct-support model families:

| Family | Direct model IDs | Notes |
|---|---|---|
| Gemma 4 | `gemma-4-e2b-it`, `gemma-4-e4b-it`, `gemma-4-12b-it`, `gemma-4-26b-a4b-it`, `gemma-4-31b-it` | MLX affine 4/5/6-bit weights; assistant-MTP paths |
| Qwen 3 | `Qwen3-4B-4bit` and manifest-backed dense checkpoints | Dense SwiGLU graph |
| Qwen 3.5 | `Qwen3.5-9B-MLX-4bit` | GatedDeltaNet linear attention |
| Qwen 3.6 | `Qwen3.6-35B-A3B` 4-bit, `Qwen3.6-27B` 4/5/6-bit | `qwen3_next`; fused sidecar-MTP paths |
| Qwen3-Coder-Next | `Qwen3-Coder-Next-4bit` | Direct coding-agent path |
| GLM 4.7 Flash | `glm4_moe_lite` / `glm4.7-flash-4bit` | Flash MLA + MoE graph |

Direct support means AX owns the `ax-engine-mlx` graph and loads MLX safetensors
through the AX manifest path. Other MLX text models can use
`mlx_lm_delegated`; GGUF and non-MLX local inference can use `llama_cpp`.

## Typical Hardware

AX Engine targets high-memory Apple Silicon Macs running macOS 26 (Tahoe) or
later. Start at 32 GB unified memory for small models; use 64 GB, 128 GB, or
larger machines when running the recommended local chatbot, agent, and coding
model stack.

Full sizing tables and model-stack recommendations live in the
[hardware FAQ](docs/FAQ.md#what-hardware-does-ax-engine-support) and
[model-stack FAQ](docs/FAQ.md#what-model-stack-should-i-run-on-high-memory-apple-silicon).

## What AX Engine Does

AX Engine is a local inference runtime for high-memory Apple Silicon Macs. It
keeps model acquisition, serving, acceleration, and benchmark evidence behind
one explicit runtime contract:

- **Serve local models through stable APIs.** The server exposes OpenAI-shaped
  chat/completions, native generate routes, Ollama-compatible chat, SDK
  sessions, and route metadata.
- **Run supported MLX models in a repo-owned runtime.** Direct-support families
  use AX-owned model graphs, tokenizer/KV handling, scheduling, and runtime
  telemetry.
- **Prepare acceleration-ready packages.** `download-mtp` packages Qwen fused
  sidecars, Gemma assistant drafters, and GLM built-in MTP sidecars; n-gram
  acceleration remains a separate direct-runtime policy.
- **Keep long sessions efficient.** Prefix reuse restores validated physical
  MLX KV snapshots so chat and agent loops avoid repeatedly pre-filling the
  same context.
- **Benchmark the contract, not just kernels.** `ax-engine-bench` preserves
  route identity, sampler settings, prompt shape, correctness checks, and
  artifact provenance for public claims.

[mlx_lm](https://github.com/ml-explore/mlx-lm) is the canonical MLX reference.
AX Engine compares against `mlx_lm.benchmark` and uses `mlx_lm.server` only as
an explicit delegated compatibility route when AX does not yet own the model
graph. See the [FAQ](docs/FAQ.md#is-ax-faster-because-it-replaces-mlx-kernels)
for the boundary between MLX kernels and AX-owned runtime behavior.

Design details: [Architecture](docs/ARCHITECTURE.md) ·
[Scheduler](docs/SCHEDULER.md) · [KV Cache](docs/KV-CACHE.md) ·
[Long Context](docs/LONG-CONTEXT.md) · [Benchmark Design](docs/BENCH-DESIGN.md).

### Runtime Paths

| Path | Use it for | What AX owns |
|---|---|---|
| Repo-owned MLX runtime | Direct-support model families and AX-owned performance claims | Model graph, token/KV runtime, scheduling, acceleration policy, server/SDK route behavior |
| `mlx_lm_delegated` | MLX text models supported upstream before AX has a graph | AX route compatibility over a user-provided `mlx_lm.server`; not AX token/KV throughput |
| `llama_cpp` | GGUF and non-MLX local inference | AX route compatibility over llama.cpp server/CLI behavior; not AX MLX throughput |

Runtime reports expose `selected_backend`, `support_tier`, and
`resolution_policy` so callers and benchmark artifacts can distinguish direct
execution from delegated compatibility. For endpoint details, see
[`docs/API-COMPATIBILITY.md`](docs/API-COMPATIBILITY.md).

## Performance

Full result tables and interpretation live in
[`docs/PERFORMANCE.md`](docs/PERFORMANCE.md). Public claim boundaries live in
[`docs/performance/README.md`](docs/performance/README.md). Benchmark
methodology, test setup, and reproduction details live in
[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

### Speculative Decoding (MTP)

AX Engine supports three MTP packaging contracts in the repo-owned runtime: Qwen
fused sidecars, Gemma assistant drafters, and GLM built-in MTP sidecars. The
current benchmark design has two clearly labeled lanes: the recommended 6-bit
`download-mtp` matrix for practical AX Engine usage, and 4-bit comparison rows
for alignment with peer MTP-engine results. Same-package direct baselines are
used only to report AX MTP acceleration.

| Target | Preparation command | Benchmark mode |
|---|---|---|
| `qwen3.6-27b-6bit` | `ax-engine download-mtp qwen3.6-27b-6bit` | Qwen fused sidecar MTP |
| `qwen3.6-35b-a3b` | `ax-engine download-mtp qwen3.6-35b-a3b` | Qwen fused sidecar MTP |
| `gemma-4-12b` | `ax-engine download-mtp gemma-4-12b` | Gemma assistant-MTP |
| `gemma-4-26b` | `ax-engine download-mtp gemma-4-26b` | Gemma assistant-MTP |
| `gemma-4-31b` | `ax-engine download-mtp gemma-4-31b` | Gemma assistant-MTP |
| `glm-4.7-flash` | `ax-engine download-mtp glm-4.7-flash` | GLM built-in MTP sidecar |

Rules for current MTP benchmark artifacts:

- Use 6-bit `download-mtp` model packages for the recommended practical AX
  Engine lane.
- Use 4-bit MTP rows only as a clearly labeled comparison lane for alignment
  with other MTP-engine benchmark results.
- Use the prepared path returned by `ax-engine download-mtp`.
- Report `mtp` rows plus same-package direct baselines for AX acceleration.
  Do not run or promote `mtp-ngram` rows.
- Do not include Qwen3-Coder-Next, 5-bit, 8-bit, FFN-only, or GGUF variants in
  the recommended 6-bit MTP matrix.
- Direct rows are same-artifact denominators for `AX MTP / AX direct` decode
  acceleration, not a cross-model speed leaderboard.

The benchmark prompt suites remain `flappy`, `long_code`, and
`python_modules_long`, with sampled decode (`temperature=0.6`, `top_p=0.95`,
`top_k=20`), `1000` generated tokens, `5` measured repetitions, and recorded
cooldown. Recommended 6-bit artifacts should live under
`benchmarks/results/mtp-6bit/`; 4-bit comparison artifacts must stay clearly
labeled as comparison results. Every artifact records the exact model snapshot,
MTP package provenance, route identity, accept rate, prefill throughput, decode
throughput, TTFT, sampler, prompt suite, repetitions, and cooldown.

For production-like AX Engine guidance, use the 6-bit lane. The 4-bit lane is
published to make comparison with other MTP engines easier because many peer
benchmarks use 4-bit models. Historical MTP+n-gram artifacts remain useful for
debugging regressions, but they are not current README/PERFORMANCE MTP evidence.

#### 6-bit MTP acceleration refresh (2026-06-23)

This refresh covers all six 6-bit `download-mtp` targets across the three real
prompt suites. The chart compares each model and prompt suite with **MTP off**
(`AX direct`) and **MTP on** (`AX MTP`) side by side, using decode median tok/s
from the same prepared package. The speedup labels are `AX MTP decode median /
AX direct decode median`; they are same-package acceleration ratios, not a
cross-model speed leaderboard. For Gemma 4 12B's shape-compatible direct peer
comparison, see [Gemma 4 12B](#gemma-4-12b).

![AX MTP decode throughput with MTP off and MTP on](docs/assets/perf-mtp-6bit-ax-acceleration.svg)

| Target | Suite | AX direct decode | AX MTP decode | AX speedup | AX MTP prefill | AX MTP TTFT | AX accept | MTPLX | lightning-mlx |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| `qwen3.6-27b-6bit` | `flappy` | 18.6 tok/s | 42.1 tok/s | 2.26x | 632.7 tok/s | 508 ms | 99.5% | N/A | N/A |
| `qwen3.6-27b-6bit` | `long_code` | 18.5 tok/s | 34.1 tok/s | 1.85x | 693.7 tok/s | 1031 ms | 97.7% | N/A | N/A |
| `qwen3.6-27b-6bit` | `python_modules_long` | 17.5 tok/s | 33.7 tok/s | 1.92x | 614.6 tok/s | 566 ms | 96.7% | N/A | N/A |
| `qwen3.6-35b-a3b` | `flappy` | 46.2 tok/s | 141.5 tok/s | 3.06x | 1561.8 tok/s | 212 ms | 99.8% | N/A | N/A |
| `qwen3.6-35b-a3b` | `long_code` | 46.3 tok/s | 140.1 tok/s | 3.02x | 2381.1 tok/s | 301 ms | 98.5% | N/A | N/A |
| `qwen3.6-35b-a3b` | `python_modules_long` | 46.0 tok/s | 142.3 tok/s | 3.09x | 1690.4 tok/s | 205 ms | 98.9% | N/A | N/A |
| `gemma-4-12b` | `flappy` | 26.7 tok/s | 62.2 tok/s | 2.33x | 1701.7 tok/s | 214 ms | 99.3% | N/A | N/A |
| `gemma-4-12b` | `long_code` | 26.6 tok/s | 70.5 tok/s | 2.65x | 1951.6 tok/s | 409 ms | 99.1% | N/A | N/A |
| `gemma-4-12b` | `python_modules_long` | 27.1 tok/s | 63.2 tok/s | 2.33x | 1753.3 tok/s | 205 ms | 98.0% | N/A | N/A |
| `gemma-4-26b` | `flappy` | 45.7 tok/s | 112.9 tok/s | 2.47x | 2395.0 tok/s | 148 ms | 99.8% | N/A | N/A |
| `gemma-4-26b` | `long_code` | 46.8 tok/s | 113.6 tok/s | 2.43x | 3754.7 tok/s | 219 ms | 99.3% | N/A | N/A |
| `gemma-4-26b` | `python_modules_long` | 45.9 tok/s | 107.2 tok/s | 2.34x | 2597.7 tok/s | 147 ms | 98.9% | N/A | N/A |
| `gemma-4-31b` | `flappy` | 15.4 tok/s | 28.1 tok/s | 1.82x | 701.9 tok/s | 516 ms | 99.6% | N/A | N/A |
| `gemma-4-31b` | `long_code` | 15.3 tok/s | 27.3 tok/s | 1.78x | 747.8 tok/s | 1067 ms | 99.5% | N/A | N/A |
| `gemma-4-31b` | `python_modules_long` | 15.0 tok/s | 25.5 tok/s | 1.70x | 678.5 tok/s | 512 ms | 98.9% | N/A | N/A |
| `glm-4.7-flash` | `flappy` | 52.6 tok/s | 91.5 tok/s | 1.74x | 1694.5 tok/s | 163 ms | 98.2% | N/A | N/A |
| `glm-4.7-flash` | `long_code` | 51.9 tok/s | 77.9 tok/s | 1.50x | 2727.6 tok/s | 250 ms | 98.2% | N/A | N/A |
| `glm-4.7-flash` | `python_modules_long` | 52.4 tok/s | 72.9 tok/s | 1.39x | 1948.2 tok/s | 174 ms | 97.7% | N/A | N/A |

Methodology: `1000` generated tokens, `5` measured repetitions per prompt case
after the AX warmup pass, 30 s cooldown, 10 s inter-case cooldown, sampled
decode (`temperature=0.6`, `top_p=0.95`, `top_k=20`), pure MTP, and no
MTP+n-gram stacking. Peer rows are `N/A` when the peer runner cannot run the
same prepared 6-bit `download-mtp` package under a comparable prompt-suite
contract. MTPLX 0.3.7 rejects the Qwen dense runtime contract and has no Gemma
assistant-MTP or GLM built-in sidecar runner. Lightning-MLX remains
diagnostic-only under current policy after the silent-thinking pathology and
does not provide a comparable promoted row for these prepared packages.

Pure-MTP verification: all listed AX MTP artifacts record zero n-gram accepted,
proposed, submitted, and hit-step telemetry. Summary artifacts:
[`summary.md`](benchmarks/results/mtp-6bit/2026-06-22-six-model-mtp-full-three-suite-ax-gain/summary.md)
and
[`summary.json`](benchmarks/results/mtp-6bit/2026-06-22-six-model-mtp-full-three-suite-ax-gain/summary.json).

#### 4-bit MTP comparison lane (2026-06-20)

The 4-bit lane is not the recommended AX Engine deployment setting. It is kept
in the MTP section because peer engines commonly publish 4-bit MTP results, so
these rows make comparison easier. Use the 6-bit `download-mtp` packages above
for practical AX Engine usage.

Qwen3.6 rows compare against MTPLX on the same 4-bit base family, prompt suites,
sampler, 1,000 generated tokens, 5 measured repetitions, and cooldown contract:

| Model | Suite | Depth | AX MTP decode | MTPLX decode | AX / MTPLX | AX MTP prefill | AX MTP TTFT | AX accept |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.6 27B 4-bit | `flappy` | 3 | 61.4 tok/s | 56.1 tok/s | 1.09x | 677.7 tok/s | 474 ms | 99.7% |
| Qwen3.6 27B 4-bit | `long_code` | 3 | 60.5 tok/s | 57.9 tok/s | 1.04x | 789.4 tok/s | 909 ms | 99.6% |
| Qwen3.6 27B 4-bit | `python_modules_long` | 3 | 52.0 tok/s | 52.7 tok/s | 0.99x | 692.1 tok/s | 506 ms | 97.8% |
| Qwen3.6 35B-A3B 4-bit | `flappy` | 1 | 169.0 tok/s | 104.3 tok/s | 1.62x | 1,795.1 tok/s | 179 ms | 100.0% |
| Qwen3.6 35B-A3B 4-bit | `long_code` | 1 | 164.7 tok/s | 105.6 tok/s | 1.56x | 2,672.7 tok/s | 269 ms | 99.9% |
| Qwen3.6 35B-A3B 4-bit | `python_modules_long` | 1 | 166.7 tok/s | 98.2 tok/s | 1.70x | 1,973.5 tok/s | 174 ms | 97.9% |

Gemma rows are AX assistant-MTP comparison artifacts. No runnable peer benchmark
covers the same Gemma assistant-MTP contract: `mlx_lm` cannot load
`gemma4_unified`, llama.cpp does not expose a Gemma assistant-MTP path, and
available MTP peer tools target different sidecar contracts.

| Model | Suite | Depth | AX MTP decode | AX MTP prefill | AX MTP TTFT | AX accept | Peer MTP |
|---|---|---:|---:|---:|---:|---:|---|
| Gemma 4 12B 4-bit-FFN | `flappy` | 2 | 96.8 tok/s | 1,928.3 tok/s | 187 ms | 98.8% | N/A |
| Gemma 4 12B 4-bit-FFN | `long_code` | 2 | 92.3 tok/s | 2,040.5 tok/s | 390 ms | 99.4% | N/A |
| Gemma 4 12B 4-bit-FFN | `python_modules_long` | 2 | 82.9 tok/s | 1,830.5 tok/s | 195 ms | 97.9% | N/A |
| Gemma 4 26B A4B 4-bit | `flappy` | 1 | 128.8 tok/s | 2,690.0 tok/s | 131 ms | 99.4% | N/A |
| Gemma 4 26B A4B 4-bit | `long_code` | 1 | 136.7 tok/s | 4,026.1 tok/s | 202 ms | 99.2% | N/A |
| Gemma 4 26B A4B 4-bit | `python_modules_long` | 1 | 130.1 tok/s | 2,923.0 tok/s | 130 ms | 98.8% | N/A |
| Gemma 4 31B 4-bit | `flappy` | 1 | 39.4 tok/s | 723.5 tok/s | 487 ms | 99.4% | N/A |
| Gemma 4 31B 4-bit | `long_code` | 1 | 40.0 tok/s | 806.8 tok/s | 987 ms | 99.4% | N/A |
| Gemma 4 31B 4-bit | `python_modules_long` | 1 | 37.4 tok/s | 741.4 tok/s | 472 ms | 97.5% | N/A |

Artifacts:
[`Qwen3.6 4-bit fair summary`](benchmarks/results/mtp-fair/2026-06-20-qwen36-merged-ax-refresh/summary.md),
[`Qwen3.6 prefill/TTFT report`](benchmarks/results/mtp-fair/2026-06-20-qwen36-merged-ax-refresh/prefill-ttft-report.json),
and
[`Gemma 4 assistant-MTP summary`](benchmarks/results/gemma4-assistant-mtp/2026-06-20-gemma4-assistant-mtp-ax-mtp-only/summary.md).

#### GLM-4.7 Flash MTP validation session

GLM-4.7 Flash uses the built-in MTP tensors from `zai-org/GLM-4.7-Flash`.
`ax-engine download-mtp glm-4.7-flash` downloads the 6-bit MLX base
(`mlx-community/GLM-4.7-Flash-6bit`), extracts the built-in MTP layer into
`glm_mtp.safetensors`, and writes a self-contained AX package.

The first local validation session used the prepared package returned by
`download-mtp` and the `flappy` real-prompt suite. This is a smoke session, not
the promoted 5-repetition MTP matrix row: it used 32 generated tokens, 1 measured
repetition, no cooldown, sampled decode (`temperature=0.6`, `top_p=0.95`,
`top_k=20`), MTP depth 1, and no MTP+n-gram stacking. The direct baseline uses
the same package and prompt suite with MTP disabled.

| Mode | Route | Decode median | Prefill median | TTFT median | MTP evidence |
|---|---|---:|---:|---:|---|
| Direct baseline | `direct_single_decode_baseline` | 58.7 tok/s | 1,670 tok/s | 166 ms | no drafts |
| GLM built-in MTP | `mtp_head_only_verify_loop` | 90.3 tok/s | 1,690 tok/s | 163 ms | 54 drafted, 46 accepted, 85.2% accept |

In this smoke session, GLM built-in MTP was **1.54x** faster than direct decode
on median decode throughput. Treat this as path validation and a same-artifact
diagnostic comparison until the full 6-bit MTP matrix is rerun with 1,000
generated tokens, 5 measured repetitions, and recorded cooldown.

Artifacts: [`flappy-after-activation-fix.json`](benchmarks/results/mtp-6bit/2026-06-22-glm47-flash-mtp-smoke/flappy-after-activation-fix.json)
(MTP) and [`flappy-direct-baseline.json`](benchmarks/results/mtp-6bit/2026-06-22-glm47-flash-mtp-smoke/flappy-direct-baseline.json)
(direct baseline).

### Direct Decode · Prefill · TTFT

#### Gemma 4 12B

Gemma 4 12B (`model_type: gemma4_unified`) is reported separately from the per-layer-embedding E2B/E4B and MoE 26B/31B checkpoints because it has a distinct graph, multimodal tensor contract, and benchmark boundary. **Upstream `mlx_lm` 0.31.3 cannot load it** (`ValueError: Model type gemma4_unified not supported`), so the direct peer here is **llama.cpp Metal** on a shape-compatible GGUF.

> [!NOTE]
> **AX Engine's repo-owned native MLX route supports Gemma 4 12B text plus inline base64 image/audio/video chat.** Delegated compatibility routes remain text-first; `/v1/generate` accepts the processed `multimodal_inputs.gemma4_unified` tensor contract.

**At a glance:**

- **Direct decode:** AX native MLX reaches **61.7-66.0 tok/s** on the bit-comparable 4-bit-FFN artifact versus llama.cpp Metal's **56.9-59.2 tok/s** depth-matched range.
- **Context depth:** AX's direct margin is **+11% / +11% / +8%** versus llama.cpp matched-depth decode at 128 / 512 / 2,048 prompt tokens.
- **Assistant-MTP:** current `gemma-4-12b` MTP benchmarking lives in the [6-bit MTP acceleration refresh](#6-bit-mtp-acceleration-refresh-2026-06-23), where the 6-bit `download-mtp` package reaches **62.2-70.5 tok/s** and **2.33-2.65x** same-package speedup over MTP-off decode.
- **Why the earlier result flipped:** the upstream MLX snapshot keeps FFN weights at 8-bit, so it reads about **1.65x** the bytes of the re-quantized 4-bit-FFN artifact. Decode is bandwidth-bound; matching quantization closes the gap.

**Direct decode peer comparison:**

AX direct rows use the 4-bit-FFN MLX artifact and random-token prompts. `mlx_lm` is absent because it has no `gemma4_unified` graph. The llama.cpp rows are shape-compatible external GGUF references, not prompt-hash-parity MLX rows.

<table>
<tr>
<td><img width="100%" src="docs/assets/perf-gemma4-12b-direct-decode-tok-s.svg" alt="Grouped bar chart comparing Gemma 4 12B 4-bit median direct decode throughput for AX Engine native MLX and llama.cpp Metal at 128/512/2048 prompt tokens; mlx_lm is not available because it has no gemma4_unified graph"></td>
<td><img width="100%" src="docs/assets/perf-gemma4-12b-direct-prefill-tok-s.svg" alt="Grouped bar chart comparing Gemma 4 12B 4-bit median prefill throughput for AX Engine native MLX and llama.cpp Metal at 128/512/2048 prompt tokens"></td>
<td><img width="100%" src="docs/assets/perf-gemma4-12b-direct-ttft-ms.svg" alt="Grouped bar chart comparing Gemma 4 12B 4-bit median time to first token for AX Engine native MLX and llama.cpp Metal at 128/512/2048 prompt tokens"></td>
</tr>
</table>

| Prompt tokens | AX decode | llama.cpp decode (depth 0) | llama.cpp decode (matched depth) | AX prefill | llama.cpp prefill | AX TTFT (ms) | llama.cpp TTFT (ms) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 66.0 | 59.8 | 59.2 | 1,171 | 1,252 | 109 | 102 |
| 512 | 65.6 | 59.6 | 58.9 | 1,839 | 1,745 | 278 | 293 |
| 2048 | 61.7 | 59.7 | 56.9 | 2,004 | 1,690 | 1,022 | 1,212 |

Read the two llama.cpp decode columns carefully:

- `depth 0` is plain `llama-bench tg`, decoding from an empty context and representing llama.cpp's best case.
- `matched depth` uses `-d {prompt} -n 128`, so decode happens after the same prompt depth AX has already prefetched.
- AX wins the matched-depth comparison at every prompt size, and prefill also leads at 512 and 2,048 tokens.

The table uses the bit-comparable **4-bit-FFN** AX artifact (`scripts/requantize_gemma4_12b_ffn_4bit.py`), about 4.5 bpw versus the Q4_K_M GGUF's about 4.8 bpw. The upstream `mlx-community/gemma-4-12B-it-4bit` snapshot keeps the FFN at **8-bit** (~10.98 GB) and trails llama.cpp at about 46 tok/s. That is a bytes-read handicap, not an AX runtime result.

**Memory bandwidth share:**

Decode is memory-bandwidth-bound on Apple Silicon: each token reads the model weights once, so decode tok/s is set by bytes-read and how close the engine gets to the memory ceiling. Measured M5 Max GPU peak read bandwidth ≈ 577 GB/s (MLX reduction over a 6 GB array).

<img src="docs/assets/perf-gemma4-12b-bandwidth.svg" alt="100% stacked bar chart showing Gemma 4 12B effective decode bandwidth used versus theoretical headroom for AX 8-bit FFN (86%), AX 4-bit FFN (75%), llama.cpp depth-0 (76%), and llama.cpp depth-512 (75%)">

| Engine / quantization | Weights/token | Decode tok/s | Effective BW | % of 577 GB/s peak |
|---|---:|---:|---:|---:|
| AX — 8-bit FFN (upstream 4bit snapshot) | 10.98 GB | 45.0 | 494 GB/s | 86% |
| AX — 4-bit FFN (re-quantized) | 6.74 GB | 64.4 | 434 GB/s | 75% |
| llama.cpp Q4_K_M — decode @ depth 512 | 7.38 GB | 58.9 | 435 GB/s | 75% |
| llama.cpp Q4_K_M — decode @ depth 0 (`tg`) | 7.38 GB | 59.8 | 441 GB/s | 76% |

The bandwidth view is the key explanation: AX is not under-utilizing memory. The re-quantized AX row sustains **434 GB/s**, in the same band as llama.cpp's **435 GB/s** at matched depth. The remaining direct-decode difference is bytes read per token: uniform 4-bit group-64 reduces AX to **6.74 GB/token**, while Q4_K_M reads **7.38 GB/token**. The 8-bit-FFN upstream snapshot has higher bus utilization (86%) but worse speed because it reads far more data.

**Methodology and artifacts:**

Direct rows use the 4-bit-FFN artifact, greedy-equivalent sampler, 128 generated tokens, 5 repetitions, 15 s cooldown, and random-token prompts following the `mlx_lm.benchmark` contract. llama.cpp decode is shown both at depth 0 (`tg`) and at matched context depth (`-d {prompt}`). Host/runtime for the latest direct llama.cpp peer rerun: Apple M5 Max · llama.cpp b9700 / ggml 0.15.2 (Metal, flash-attn) · `mlx_lm` 0.31.3 has no `gemma4_unified` support. MTP methodology and artifacts live with [Speculative Decoding (MTP)](#speculative-decoding-mtp).

Full artifacts: [`2026-06-20-gemma-4-12b-it-4bit-direct`](benchmarks/results/mlx-inference/2026-06-20-gemma-4-12b-it-4bit-direct/gemma-4-12b-it-4bit.json) (AX direct rerun; chart artifact with retained llama.cpp reference rows in [`gemma-4-12b-it-4bit-with-llama-reference.json`](benchmarks/results/mlx-inference/2026-06-20-gemma-4-12b-it-4bit-direct/gemma-4-12b-it-4bit-with-llama-reference.json); llama.cpp GGUF provenance in [`llama_cpp_gguf_provenance.json`](benchmarks/results/mlx-inference/2026-06-09-gemma-4-12b-it-4bit-direct/llama_cpp_gguf_provenance.json)).

##### Gemma 4 12B Multimodal

Gemma 4 12B multimodal timing is reported separately from the text benchmark above because media inputs expand into validated Gemma4 unified soft-token spans before the MLX graph runs. The publication-grade timing artifact covers all **17 AX Engine image/audio/video cases** through both the native `/v1/generate/stream` prefill path and the OpenAI-compatible `/v1/chat/completions` path. The llama.cpp Metal peer rows are cold OpenAI chat endpoint rows for the supported image/audio cases, with prompt cache, slot prompt reuse, and context checkpoints disabled and raw llama.cpp timing/cache metadata recorded.

<table>
<tr>
<td><img width="100%" src="docs/assets/perf-gemma4-12b-multimodal-ttft-ms.svg" alt="Bar chart showing Gemma 4 12B multimodal prefill time to first token for AX Engine native MLX"></td>
<td><img width="100%" src="docs/assets/perf-gemma4-12b-multimodal-prefill-tok-s.svg" alt="Bar chart showing Gemma 4 12B multimodal prefill throughput for AX Engine native MLX"></td>
<td><img width="100%" src="docs/assets/perf-gemma4-12b-multimodal-peer-chat-ms.svg" alt="Grouped bar chart comparing Gemma 4 12B multimodal cold chat endpoint latency for llama.cpp Metal on the left and AX Engine on the right"></td>
</tr>
</table>

| Coverage | AX cases measured | Expanded input | Median runner prefill TTFT | Median prefill | Median AX chat E2E | llama.cpp peer endpoint |
|---|---:|---:|---:|---:|---:|---|
| Image | 5 | 275-535 tokens | 189.4-316.2 ms | 1,447.8-1,692.1 tok/s | 1,440.8-1,704.8 ms | 5 measured, 401.6-518.7 ms cold chat endpoint |
| Audio | 4 | 32-771 tokens | 75.8-419.4 ms | 422.1-1,838.4 tok/s | 1,466.5-1,819.2 ms | 3 measured, 338.0-464.5 ms cold chat endpoint; 1 skipped: llama.cpp audio cap unstable |
| Video | 4 | 92-2,355 tokens | 106.1-2,973.5 ms | 792.0-1,681.0 tok/s | 1,500.2-4,441.7 ms | 4 skipped: llama.cpp video path unsupported |
| Combined | 4 | 181-442 tokens | 133.2-256.7 ms | 1,359.1-1,721.6 tok/s | 1,532.4-1,771.6 ms | 1 measured, 507.9 ms cold chat endpoint; 3 skipped: video unsupported |

Rows use `/v1/generate/stream` with processed `multimodal_inputs.gemma4_unified` for runner-time prefill and `/v1/chat/completions` with inline media for client-wall E2E latency. This run used `max_output_tokens=8`, 1 warmup, 3 measured repetitions, `--max-batch-tokens 4096`, a release server binary, 128 GB unified memory, and a clean tracked worktree at `67ce2675a469cf5eecba687f348c649e663011b8`.

The llama.cpp peer rows use reference llama.cpp `19bba67c1` with Metal, `gemma-4-12B-it-Q4_K_M.gguf`, and `mmproj-gemma-4-12B-it-Q8_0.gguf`. They are OpenAI chat endpoint-latency rows for supported image/audio inputs, not native prefill rows and not a throughput comparison. The fair-peer launch contract is `--cache-ram 0 --no-cache-idle-slots --slot-prompt-similarity 0 --ctx-checkpoints 0` plus `--llama-cache-policy prompt_cache_disabled`; the artifact records raw llama.cpp `timings`, `prompt_tokens_details.cached_tokens`, server prompt token counts, and cache counts. Published peer rows require zero reported cached prompt tokens and server prompt-eval token counts at least as large as the cold request's reported prompt tokens. Video-containing peer rows are explicit skips because the local llama.cpp Gemma 4 path does not expose a like-for-like video contract, and `audio_cap` is skipped because this llama.cpp Gemma 4 audio path fails the warmup-plus-three-repetition contract on the largest audio fixture. The peer chart excludes one measured image case whose AX and llama.cpp output token counts differ, so chart bars compare matched-output rows only. For this Gemma 4 llama.cpp build, most peer text appears in `reasoning_content` rather than `message.content`, so the benchmark validates positive `response_chars`.

Full artifact: [`2026-06-09-gemma4-12b-multimodal-cold-peer-matrix`](benchmarks/results/gemma4-multimodal/2026-06-09-gemma4-12b-multimodal-cold-peer-matrix.json). Render charts with:

```bash
python3 scripts/render_gemma4_multimodal_charts.py \
  --artifact benchmarks/results/gemma4-multimodal/2026-06-09-gemma4-12b-multimodal-cold-peer-matrix.json \
  --assets-dir docs/assets
```

To reproduce the supported-case image/audio/video timing matrix from a Gemma 4 12B AX Engine server, use the matrix runner and validate the resulting artifact before publishing charts:

```bash
python3 scripts/bench_gemma4_multimodal.py \
  --url http://127.0.0.1:18080 \
  --model gemma-4-12B-it \
  --model-dir /path/to/gemma-4-12B-it-4bit \
  --cases all \
  --layers native_runtime_prefill,openai_chat_e2e \
  --warmup 1 \
  --repetitions 3 \
  --cooldown 1 \
  --max-output-tokens 8 \
  --server-command "target/release/ax-engine-server --model-id gemma-4-12B-it --mlx --mlx-model-artifacts-dir /path/to/gemma-4-12B-it-4bit --max-batch-tokens 4096 --port 18080" \
  --llama-url http://127.0.0.1:<peer-port> \
  --llama-binary /path/to/llama-server \
  --llama-gguf <path-to-gemma-4-12B-it-Q4_K_M.gguf> \
  --llama-mmproj <path-to-mmproj-gemma-4-12B-it-Q8_0.gguf> \
  --llama-cache-policy prompt_cache_disabled \
  --output benchmarks/results/gemma4-multimodal/gemma4-12b-multimodal-cold-peer-matrix.json

python3 scripts/check_gemma4_multimodal_benchmark_artifact.py \
  benchmarks/results/gemma4-multimodal/gemma4-12b-multimodal-cold-peer-matrix.json \
  --min-repetitions 3 \
  --require-modalities image,audio,video \
  --require-build-provenance \
  --readme-ready
```

For a fair llama.cpp peer rerun, launch `llama-server` with prompt cache, slot prompt reuse, and context checkpoints disabled for the peer server, for example `--cache-ram 0 --no-cache-idle-slots --slot-prompt-similarity 0 --ctx-checkpoints 0`, then validate with `--readme-ready`. Peer rows with unknown cache policy, reported cached prompt tokens, or server prompt-eval token counts that are too low for a cold prompt are rejected by the artifact checker. Without a matching Gemma 4 12B GGUF and multimodal projector, peer rows are explicit skips. Video rows remain explicit skips until the peer server exposes a like-for-like video path for Gemma 4 12B.

Gemma assistant-MTP package layout and cache-location details live in
[Supported Models](docs/SUPPORTED-MODELS.md#mtp-downloads).

#### DiffusionGemma

DiffusionGemma is a block-diffusion Gemma4 26B checkpoint, not an ordinary autoregressive decoder. AX runs it with a native MLX graph, but the measurement boundary is different from the direct-decode families below: the first visible output comes from a **committed 256-token diffusion block**, not from a single next-token step.

Because of that generation shape, the rows below intentionally do **not** use the
plain `decode tok/s` or `TTFT` labels used for autoregressive models. In Qwen,
Gemma 4 text, and other next-token decoders, `TTFT` means prompt prefill plus the
first single-token decode step, and `decode tok/s` means the steady
token-by-token autoregressive loop. DiffusionGemma instead runs a bidirectional
denoise pass over a 256-token canvas, then performs a causal commit for that
block. The comparable boundary inside this runtime is therefore **time to first
block** and **first-block decode**. Treating these as ordinary TTFT/decode rows
would make the result look directly comparable to autoregressive throughput even
though the work per visible output boundary is different.

The charts keep the same 128 / 512 / 2,048 prompt-token layout as the autoregressive sections for readability, but the values are AX first-block telemetry. Peer bars are intentionally omitted rather than shown as zero: current llama.cpp Metal cannot load the GGUF (`unknown model architecture: 'diffusion-gemma'`), and `mlx_lm` 0.31.3 cannot load the MLX snapshot (`Model type diffusion_gemma not supported.`).

<table>
<tr>
<td><img width="100%" src="docs/assets/perf-diffusiongemma-direct-decode-tok-s.svg" alt="Bar chart showing measured AX direct DiffusionGemma first-block decode throughput at 128, 512, and 2048 prompt tokens"></td>
<td><img width="100%" src="docs/assets/perf-diffusiongemma-direct-prefill-tok-s.svg" alt="Bar chart showing measured AX direct DiffusionGemma prefill throughput at 128, 512, and 2048 prompt tokens"></td>
<td><img width="100%" src="docs/assets/perf-diffusiongemma-direct-ttft-ms.svg" alt="Bar chart showing measured AX direct DiffusionGemma time to first committed block at 128, 512, and 2048 prompt tokens"></td>
</tr>
</table>

| Prompt tokens | AX first-block decode | Denoise steps | Committed block |
|---:|---:|---:|---:|
| 128 | 30.7 tok/s | 48 | 256 tokens |
| 512 | 58.9 tok/s | 25 | 256 tokens |
| 2048 | 32.1 tok/s | 48 | 256 tokens |

**Prefill and first-block latency:**

| Prompt tokens | AX direct prefill | AX time to first block | llama.cpp Metal 9650 | `mlx_lm` 0.31.3 |
|---:|---:|---:|---|---|
| 128 | 1,351.8 tok/s | 8,428 ms | load blocked | load blocked |
| 512 | 3,002.1 tok/s | 4,518 ms | load blocked | load blocked |
| 2048 | 4,031.4 tok/s | 8,475 ms | load blocked | load blocked |

`time to first block` is prefill wall time plus the first 256-token denoise-and-commit block. `first-block decode` is computed as `256 / ax_mlx_diffusion_block_wall_us`. Use these rows to track AX's DiffusionGemma path; do not compare them directly with ordinary autoregressive TTFT or fixed-token decode throughput.

| Runtime path | Model artifact | Benchmark status |
|---|---|---|
| AX direct MLX | `mlx-community/diffusiongemma-26B-A4B-it-4bit` | Measured: 1 warmup + 5 measured repetitions, 15 s cooldown, medians reported |
| llama.cpp Metal 9650 | 4-bit GGUF | Blocked at load: `unknown model architecture: 'diffusion-gemma'` |
| `mlx_lm` 0.31.3 | 4-bit MLX snapshot | Blocked at load: `Model type diffusion_gemma not supported.` |

**Memory bandwidth share:**

The bandwidth chart is an implementation-efficiency view, not a peer comparison. It estimates first-block traffic at block granularity from the measured denoise-step count plus one causal commit over the 16.54 GB MLX safetensors artifact. This rerun used **48 / 25 / 48** denoise steps at 128 / 512 / 2,048 prompt tokens, so the estimated traffic is much larger than a one-step early-exit block. The chart shows estimated bandwidth used versus the M5 Max theoretical ceiling; the table keeps the effective GB/s values.

<img src="docs/assets/perf-diffusiongemma-direct-memory-bandwidth-share.svg" alt="100% stacked bar chart showing estimated AX direct DiffusionGemma memory bandwidth share used versus theoretical headroom at 128, 512, and 2048 prompt tokens">

| Prompt tokens | Estimated effective bandwidth | % of 614.4 GB/s M5 Max theoretical bandwidth |
|---:|---:|---:|
| 128 | 97.3 GB/s | 15.8% |
| 512 | 98.9 GB/s | 16.1% |
| 2,048 | 101.8 GB/s | 16.6% |

At these prompt lengths, the first-block path uses roughly 16% of theoretical M5 Max bandwidth. The current bottleneck is therefore not raw memory bandwidth alone; the next optimization target is denoise graph reuse, dispatch overhead, and convergence behavior under stricter quality gates.

**Denoise loop optimization — GPU-native sampling:**

`crates/ax-engine-mlx/src/diffusion.rs` keeps denoise state, entropy-bound acceptance, and self-conditioning on the GPU. Convergence checks materialize only scalar counters and run every `convergence_check_interval` steps (default 4), reducing per-block GPU/CPU syncs from 48 to about 12. The CPU no longer round-trips 256 token positions on every denoise step; sampling and acceptance stay in lazy MLX graph nodes that can fuse with the forward evaluation.

**Adaptive convergence detection:**

The denoise loop can stop early when any configured convergence signal fires:

1. **Strict stability:** argmax is unchanged for `convergence_steps` consecutive checks and mean entropy is below `entropy_threshold` (default 0.005).

2. **Low update rate:** the accepted-position update rate drops below `acceptance_rate_threshold` (default 1%), so another denoise pass is unlikely to change the block materially.

3. **Entropy plateau:** mean entropy stops decreasing materially after the early denoise phase, indicating diminishing returns from additional passes.

The benchmark rows above report the measured adaptive-convergence run as recorded in the artifact. This rerun did **not** converge after one denoise step: it used 48 / 25 / 48 denoise steps at 128 / 512 / 2,048 prompt tokens. Time to first block therefore tracks the full measured denoise work for the 128- and 2,048-token rows and a mid-run early exit for the 512-token row.

**Denoise performance optimizations (enabled by default):**

The following optimizations are enabled by default for DiffusionGemma to maximize memory bandwidth utilization and reduce per-step overhead. Each can be individually disabled via opt-out environment variables.

| Optimization | What it does | Opt-out |
|---|---|---|
| KV concat buffer | Pre-allocates per-layer KV concatenation buffers on the first denoise step and updates only the canvas slice on subsequent steps via `slice_update`, avoiding re-copying the cached prompt prefix. Also caches the bidirectional attention mask per layer. | `AX_DIFFUSION_NO_KV_CONCAT_BUFFER=1` |
| Embedding cache | Caches per-layer embedding inputs across denoise steps when token IDs are unchanged, using a GPU-side sum fingerprint to detect changes. | `AX_DIFFUSION_NO_EMBEDDING_CACHE=1` |
| Compiled forward | Compiles the bidirectional denoise forward pass into an `MlxClosure` per block (when self-conditioning is off), collapsing ~250 per-step MLX C-API calls into one dispatched graph. | `AX_DIFFUSION_NO_COMPILED_FORWARD=1` |
| Commit skip on converge | Skips the causal commit forward pass (~40 ms) when the denoise loop converges with near-perfect acceptance (≥ 99%). | `AX_DIFFUSION_NO_SKIP_COMMIT=1` |

**Experimental opt-in optimizations:**

| Environment variable | What it does | Status |
|---|---|---|
| `AX_DIFFUSION_FULL_PIPELINE=1` | Compiles the entire denoise step (forward + softmax + entropy + argmax + sampling + acceptance) into a single `MlxClosure`. Supersedes the forward-only compiled closure when both are set. | Experimental / benchmarking |

Example usage for a single benchmark run with all optimizations:

```bash
python3 scripts/bench_diffusion_gemma_direct.py --bench-bin target/release/ax-engine-bench
```

These flags are read once per process. The default-on optimizations have been validated for token equivalence against the imperative path.

Artifacts: AX direct rows are [`2026-06-20-direct-first-block-rerun/summary.json`](benchmarks/results/diffusion-gemma-direct/2026-06-20-direct-first-block-rerun/summary.json), with the human summary in [`summary.md`](benchmarks/results/diffusion-gemma-direct/2026-06-20-direct-first-block-rerun/summary.md). Peer runtime blockers are recorded as load failures, so there are no llama.cpp or `mlx_lm` result artifacts for this model family.

Render charts with:

```bash
python3 scripts/bench_diffusion_gemma_direct.py --skip-benchmark
```

**Decode acceleration model — no MTP:**

DiffusionGemma's acceleration model is the diffusion block itself. It does not stack with MTP or n-gram acceleration because those techniques assume an autoregressive next-token loop:

| | MTP (speculative decoding) | DiffusionGemma (block diffusion) |
|---|---|---|
| Generation | Draft-then-verify, one token at a time | 256-token blocks via bidirectional denoising |
| Forward pass | Causal only | Bidirectional (denoise) + causal (commit) |
| Needs draft model / assistant head | Yes | No |
| AX Engine decode path | `ngram_acceleration` / `mtp_head_only` | `diffusion` (early return, mutually exclusive) |

In the runner's `decode_one`, the diffusion path returns before the MTP/n-gram branches are reached. `DiffusionConfig` carries canvas size, denoise steps, entropy thresholds, convergence settings, and temperature schedule only; it has no MTP fields.

**Supported features:**

- Block-autoregressive discrete diffusion decode (canvas=256, up to 48 denoise steps)
- Entropy-bound position acceptance with argmax-based rejection
- Self-conditioning via GPU matmul (prob × cached embedding table)
- Linear temperature schedule (configurable start/end)
- Adaptive convergence detection (stable argmax, mean entropy, low update rate, and entropy plateau)
- Standard causal prefill (same Gemma4 encoder, 4,073.3 tok/s median at the 2,048-token row)
- Causal commit pass (writes KV cache for subsequent blocks)
- SSE telemetry counters for diffusion block timing, denoise steps, convergence signals, and near-miss entropy/update-rate diagnostics (`ax_mlx_diffusion_*`)
- `diffusion` decode-route classification in benchmark harness

**Not applicable:**

- MTP / assistant-head speculative decoding (architecturally incompatible)
- N-gram acceleration (diffusion replaces the autoregressive decode loop)
- Direct pipeline double-buffering (not autoregressive)

**Benchmark contract:**

The published rows use first-block telemetry instead of the standard fixed-token autoregressive benchmark contract. `max_output_tokens=1` is enough to force prefill plus one diffusion block, and the block counters still report the full 256-token denoise/commit cycle even though the caller receives only the first emitted token.

Telemetry: SSE-emitted `ax_mlx_diffusion_*` counters cover block count, denoise steps, convergence count, per-criterion convergence signals, near-miss entropy/update-rate diagnostics, denoise wall time, commit wall time, and block wall time, plus `diffusion` decode-route classification in `bench_mlx_inference_stack.py`.

Run the full direct benchmark and regenerate the charts:

```bash
cargo build -p ax-engine-bench --bin ax-engine-bench
python3 scripts/bench_diffusion_gemma_direct.py
```

<!-- readme-performance-artifacts: reference=benchmarks/results/mlx-inference/2026-05-26-direct-mode-clean-refresh/; ax-overlay=benchmarks/results/mlx-inference/2026-06-22-ax-direct-readme-direct-only/ -->

#### Gemma 4 and Qwen 3.6

The family tables below compare **direct (non-speculative) decode** across llama.cpp Metal, mlx_lm, and ax engine, covering Gemma 4 and Qwen 3.6 at 128/512/2048 prompt tokens. `ax direct baseline` disables n-gram acceleration, MTP, and assistant drafting to measure the repo-owned direct decode path. Bench prompts are `mlx_lm.benchmark` seed-0 random tokens, which keeps prompt-hash parity across MLX rows.

The prefill and TTFT advantage is the practical direct-mode story. AX is ahead of `mlx_lm` in every listed prefill and TTFT cell below, while decode gains are smaller and model-dependent. That means the repo-owned MLX route is especially valuable for interactive requests where prompt ingestion dominates perceived latency: AX keeps prompt prefill, first-token timing, model-specific graph paths, and route metadata in one measured runtime path. These are cold-prefix rows, not prompt-cache, continuous-batching, or speculative-decoding claims.

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

> **`llama.cpp Metal*` column** — Shape-compatible reference produced by Metal-enabled `llama-bench`. `llama-bench` generates its own internal synthetic prompt tokens and does not consume the harness prompt JSON, so these numbers are **not** prompt-hash parity with the other columns. No percentage delta is shown. MLX bit-widths are mapped to the nearest Unsloth GGUF quant (4→Q4_K_M, 6→Q6_K), with explicit UD-* Unsloth Dynamic rows only when no standard root-level K-quant is published. Source: `benchmarks/manifests/llama_cpp_metal/inventory.json`, `scripts/bench_llama_cpp_metal_sweep.py`.

<details>
<summary>Benchmark provenance and methodology</summary>

The `mlx_lm` reference rows for the Gemma 4 and Qwen 3.6 rows shown below come from `benchmarks/results/mlx-inference/2026-05-26-direct-mode-clean-refresh/`. The AX direct-mode cells come from the direct-only AX rerun in `benchmarks/results/mlx-inference/2026-06-22-ax-direct-readme-direct-only/` (v6.5.2). The `llama.cpp Metal*` column is injected from `benchmarks/manifests/llama_cpp_metal/inventory.json` and the `2026-05-18-llama-cpp-metal-gemma-e2b-4bit-depth-fa/` Gemma 4 E2B 4-bit recheck.

Setup: generation=128, 5 measured repetitions, 15-second cooldown, AX prefix cache disabled for cold prefill and TTFT measurement, production-build binaries, matching prompt SHA checks. Long-greedy AX prefill rows are runner-time measurements of the cache-state prefix plus final prompt-token boundary — not full-logits prompt scoring throughput. Percentages are versus `mlx_lm`.

The 2K `llama.cpp Metal*` prefill rows are long-context, GGUF-runtime-reference rows. The Gemma 4 E2B 4-bit row was produced with llama.cpp b9110 and rechecked on b9200 with Metal offload, `-b/-ub 2048`, and flash attention enabled. The b9200 recheck improved 2K prefill only slightly — this is our benchmark boundary, not an upstream llama.cpp official bug statement.
</details>

#### Prefill throughput (tok/s) — percentages vs mlx_lm

| Model | MLX quantization | Prompt tok | llama.cpp Metal* | mlx_lm | ax engine |
|---|---|---:| ---: |---:|---:|
| Gemma 4 E2B | 4-bit | 128 | 3,481.7 | 2,338.1 | **6,047.1 (+158.6%)** |
|         |         | 512 | 6,846.0 | 7,870.0 | **16,880.4 (+114.5%)** |
|         |         | 2048 | 7,643.1 | 18,014.7 | **25,168.4 (+39.7%)** |
| Gemma 4 E2B | 6-bit | 128 | 3,539.7 | 1,823.5 | **5,595.1 (+206.8%)** |
|         |         | 512 | 7,274.0 | 6,046.6 | **15,912.0 (+163.2%)** |
|         |         | 2048 | 7,623.2 | 15,332.1 | **22,676.9 (+47.9%)** |
| Gemma 4 E4B | 4-bit | 128 | 2,194.0 | 1,513.2 | **3,444.8 (+127.7%)** |
|         |         | 512 | 4,454.2 | 4,195.5 | **7,027.8 (+67.5%)** |
|         |         | 2048 | 4,426.6 | 7,325.4 | **8,800.8 (+20.1%)** |
| Gemma 4 26B A4B | 4-bit | 128 | 1,911.4 | 496.4 | **1,345.5 (+171.0%)** |
|         |         | 512 | 3,484.5 | 1,621.0 | **3,047.3 (+88.0%)** |
|         |         | 2048 | 3,604.8 | 3,300.1 | **4,642.8 (+40.7%)** |
| Gemma 4 31B | 4-bit | 128 | 522.6 | 283.1 | **513.2 (+81.3%)** |
|         |         | 512 | 665.3 | 619.9 | **738.6 (+19.2%)** |
|         |         | 2048 | 560.3 | 733.9 | **774.5 (+5.5%)** |
| Qwen 3.6 27B | 4-bit | 128 | 539.6 | 378.8 | **580.3 (+53.2%)** |
|  |  | 512 | 759.7 | 705.7 | **833.0 (+18.0%)** |
|  |  | 2048 | 664.3 | 895.2 | **931.6 (+4.1%)** |
| Qwen 3.6 27B | 6-bit | 128 | 537.7 | 270.5 | **506.5 (+87.3%)** |
|  |  | 512 | 756.1 | 577.6 | **756.9 (+31.0%)** |
|  |  | 2048 | 689.3 | 798.2 | **862.9 (+8.1%)** |
| Qwen 3.6 35B A3B | 4-bit | 128 | 1,706.9 | 539.4 | **1,123.0 (+108.2%)** |
|  |  | 512 | 3,146.6 | 1,599.5 | **2,606.6 (+63.0%)** |
|  |  | 2048 | 3,542.3 | 3,513.1 | **3,754.7 (+6.9%)** |

#### Decode throughput (tok/s) — generation=128 tokens, temp=0

| Model | MLX quantization | Prompt tok | llama.cpp Metal* | mlx_lm | ax direct baseline |
|---|---|---:| ---: |---:|---:|
| Gemma 4 E2B | 4-bit | 128 | 174.6 | 214.0 | **236.0 (+10.3%)** |
|  |  | 512 | 165.2 | 210.3 | **226.7 (+7.8%)** |
|  |  | 2048 | 171.9 | 200.9 | **216.7 (+7.9%)** |
| Gemma 4 E2B | 6-bit | 128 | 152.1 | 172.2 | **186.0 (+8.0%)** |
|  |  | 512 | 152.0 | 166.3 | **180.2 (+8.4%)** |
|  |  | 2048 | 152.2 | 162.5 | **173.9 (+7.0%)** |
| Gemma 4 E4B | 4-bit | 128 | 110.7 | 137.1 | **143.4 (+4.6%)** |
|  |  | 512 | 110.8 | 133.6 | **140.4 (+5.1%)** |
|  |  | 2048 | 110.7 | 130.6 | **137.6 (+5.4%)** |
| Gemma 4 26B A4B | 4-bit | 128 | 112.6 | 127.9 | **134.9 (+5.5%)** |
|  |  | 512 | 112.9 | 125.0 | **131.7 (+5.3%)** |
|  |  | 2048 | 112.9 | 119.3 | **127.2 (+6.6%)** |
| Gemma 4 31B | 4-bit | 128 | 25.0 | 28.9 | **29.1 (+0.7%)** |
|  |  | 512 | 25.5 | 28.3 | **28.5 (+0.7%)** |
|  |  | 2048 | 25.3 | 27.0 | **27.3 (+1.0%)** |
| Qwen 3.6 27B | 4-bit | 128 | 26.0 | 34.0 | **34.4 (+1.2%)** |
|  |  | 512 | 26.0 | 33.9 | **34.5 (+1.8%)** |
|  |  | 2048 | 18.8 | 33.4 | **34.2 (+2.2%)** |
| Qwen 3.6 27B | 6-bit | 128 | 21.3 | 24.0 | **25.4 (+6.0%)** |
|  |  | 512 | 21.3 | 24.8 | **25.4 (+2.4%)** |
|  |  | 2048 | 15.4 | 24.6 | **25.2 (+2.3%)** |
| Qwen 3.6 35B A3B | 4-bit | 128 | 108.1 | 140.1 | **153.2 (+9.4%)** |
|  |  | 512 | 108.2 | 136.5 | **152.7 (+11.9%)** |
|  |  | 2048 | 105.7 | 134.5 | **150.8 (+12.2%)** |
> Qwen 3.6 27B 4-bit at prompt=2,048 originally produced zero decode tokens because 4-bit quantization noise pushed an EOS token to argmax at decode step 0 on the `mlx_lm.benchmark` random-token contract. The benchmark harness now sends `sampling.ignore_eos=true` for AX throughput runs, matching how `mlx_lm.benchmark` measures fixed `gen=N` throughput. Production requests default to `ignore_eos=false`. Source: `benchmarks/results/mlx-inference/2026-05-20-qwen27-4to5-direct-ngram-directcpp-r2/qwen3_6-27b-4bit.json`.

#### Time to first token (ms) — generation=128 tokens, temp=0

**Lower is better.** `mlx_lm` values are derived from reported prefill throughput. AX values are measured directly from per-step runner timing in the SSE event stream.

| Model | MLX quantization | Prompt tok | llama.cpp Metal* | mlx_lm | ax engine |
|---|---|---:| ---: |---:|---:|
| Gemma 4 E2B | 4-bit | 128 | 36.8 | 54.7 | **21.2 (-61.3%)** |
|         |         | 512 | 74.8 | 65.1 | **30.3 (-53.4%)** |
|         |         | 2048 | 268.0 | 113.7 | **81.4 (-28.4%)** |
| Gemma 4 E2B | 6-bit | 128 | 36.2 | 70.2 | **22.9 (-67.4%)** |
|         |         | 512 | 70.4 | 84.7 | **32.2 (-62.0%)** |
|         |         | 2048 | 268.7 | 133.6 | **90.3 (-32.4%)** |
| Gemma 4 E4B | 4-bit | 128 | 58.3 | 84.6 | **37.2 (-56.1%)** |
|         |         | 512 | 114.9 | 122.0 | **72.9 (-40.3%)** |
|         |         | 2048 | 462.7 | 279.6 | **232.7 (-16.8%)** |
| Gemma 4 26B A4B | 4-bit | 128 | 67.0 | 257.8 | **95.1 (-63.1%)** |
|         |         | 512 | 146.9 | 315.8 | **168.0 (-46.8%)** |
|         |         | 2048 | 568.1 | 620.6 | **441.1 (-28.9%)** |
| Gemma 4 31B | 4-bit | 128 | 244.9 | 452.2 | **249.4 (-44.8%)** |
|         |         | 512 | 769.5 | 826.0 | **693.2 (-16.1%)** |
|         |         | 2048 | 3,655.2 | 2,790.6 | **2,644.3 (-5.2%)** |
| Qwen 3.6 27B | 4-bit | 128 | 237.2 | 337.9 | **220.6 (-34.7%)** |
|  |  | 512 | 673.9 | 725.6 | **614.6 (-15.3%)** |
|  |  | 2048 | 3,083.1 | 2,287.7 | **2,198.3 (-3.9%)** |
| Qwen 3.6 27B | 6-bit | 128 | 238.1 | 473.2 | **252.7 (-46.6%)** |
|  |  | 512 | 677.2 | 886.5 | **676.5 (-23.7%)** |
|  |  | 2048 | 2,971.2 | 2,565.6 | **2,373.5 (-7.5%)** |
| Qwen 3.6 35B A3B | 4-bit | 128 | 75.0 | 237.3 | **114.0 (-52.0%)** |
|  |  | 512 | 162.7 | 320.1 | **196.4 (-38.6%)** |
|  |  | 2048 | 578.2 | 583.0 | **545.4 (-6.4%)** |
Embedding benchmarks are kept out of this README summary; see [`docs/EMBEDDINGS.md`](docs/EMBEDDINGS.md).

## SDKs

AX Engine SDK docs are organized under [`docs/sdk/`](docs/sdk/README.md).
Most SDKs target the OpenAI-compatible HTTP server; Python can also use the
in-process session API.

| SDK | Docs | Package / path |
|----------|---------------|-----------|
| **Rust** | [docs/sdk/rust.md](docs/sdk/rust.md) | `crates/ax-engine-sdk` |
| **Python** | [docs/sdk/python.md](docs/sdk/python.md) | `python/ax_engine` |
| **JavaScript / TypeScript** | [docs/sdk/javascript.md](docs/sdk/javascript.md) | `javascript/ax-engine` / `@ax-engine/sdk` |
| **Go** | [docs/sdk/go.md](docs/sdk/go.md) | `sdk/go/axengine` |
| **Ruby** | [docs/sdk/ruby.md](docs/sdk/ruby.md) | `sdk/ruby` / `ax-engine-sdk` |
| **Mojo** | [docs/sdk/mojo.md](docs/sdk/mojo.md) | `sdk/mojo/ax_engine.mojo` |

## Server Usage

Use `ax-engine serve` for normal local serving. It accepts a downloaded model
directory, a supported alias with `--download`, or the MTP package directory
printed by `ax-engine download-mtp`.

Start a direct-support model from an alias:

```bash
ax-engine serve qwen36-35b --download --port 8080
```

Or serve a prepared local package:

```bash
ax-engine serve "$MODEL_DIR" --port 8080
```

Inspect the runtime route before testing clients:

```bash
curl http://127.0.0.1:8080/v1/runtime
```

Send an OpenAI-compatible chat request:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"qwen3_dense","messages":[{"role":"user","content":"Say hello from AX."}],"max_tokens":64}'
```

For local-only development, HTTP authentication is disabled by default. To
require a bearer token on API routes, start with `--api-key` or set
`AX_ENGINE_API_KEY`; health probes remain unauthenticated. See
[`docs/SERVER.md`](docs/SERVER.md#authentication) for the full auth contract.

Use the low-level `ax-engine-server` entrypoint when you need explicit runtime
flags:

```bash
ax-engine-server \
  --mlx \
  --mlx-model-artifacts-dir "$MODEL_DIR" \
  --port 8080
```

Detailed endpoint examples, streaming, embeddings, Ollama-shaped adapters,
delegated `mlx_lm` routes, and server preview checks live in
[`docs/SERVER.md`](docs/SERVER.md) and
[`docs/GETTING-STARTED.md`](docs/GETTING-STARTED.md#first-commands).
Benchmark commands live with the performance docs instead of this usage
section.

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

Common development gates:

```bash
cargo build --workspace
cargo test --quiet
cargo clippy --all-targets --all-features -- -D warnings
cargo fmt
maturin develop
python -m unittest discover -s python/tests -v
bash scripts/check-mlx-telemetry.sh
```

For Gemma/AX MLX telemetry and decode-profile changes, prefer the targeted `scripts/check-mlx-telemetry.sh` gate. Use `scripts/check-mlx-telemetry.sh --full-workspace` when the change touches shared Rust contracts; that protected path compiles the workspace with `cargo test --workspace --no-run --jobs 1` before running crate-by-crate tests.

Coverage is collected by the report-only GitHub Actions workflow in `.github/workflows/coverage.yml`. It publishes Rust `cargo llvm-cov` and Python `coverage.py` artifacts without enforcing a percentage threshold yet.

Public documentation is in `docs/`. Canonical benchmark manifests are in `benchmarks/manifests/`. Key design docs:
[SDKs](docs/sdk/README.md) ·
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
- Discord: [Join us](https://discord.gg/aDhhburqJg)
- Email: enquiry@defai.digital

## License

Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

Copyright (c) 2026 [DEFAI Private Limited](https://defai.digital)
