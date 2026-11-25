# AX Engine

AX Engine is a **Mac-first** Apple Silicon inference runtime **optimized first
for Qwen 3.8 27B AXQ**. The Qwen 3.8 27B and Flash Next product scope is
**MLX + AXQ MXFP4 / MXFP8 with MTP sidecars**, plus affine 6-bit/group64 MTP.
The generic pinned selectors load MXFP4 MTP packs; the Qwen 3.8 27B MXFP8
pack is a catalog Candidate and Flash Next MXFP8 remains an explicit preview.
Standalone affine 4-bit and 8-bit and non-MTP variants for these two models
are unsupported. MTP execution remains explicit opt-in until its gates pass.
Install the macOS app with one curl command to download models, chat and serve
OpenAI-compatible endpoints locally.

Primary optimization target. Checkpoint Tier 1. MTP Tier 2 pending. AX certification record: Candidate (gates open).

The default pack is `qwen3.8-27b:axq`
([`AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-MXFP4-MTP`](https://huggingface.co/AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-MXFP4-MTP)
@ `b2c5354f779e430d0c1733143db848a72b71c16e`). Additional Qwen, Gemma, GLM, and
other certified families stay supported; they are not the first-run or
qualification center. Super-class Qwen 3.8 (2.4T) is experimental only.

NVIDIA/CUDA fleet serving lives in
[AX Serving](https://github.com/defai-digital/ax-serving). AX Engine remains the
local Apple Silicon runtime and no longer ships the former vLLM or TensorRT
provider bridges, runtime package, container, or CUDA qualification scripts.

Browse AutomatosX serve-ready chat / coding / embedding snapshots in the
[AutomatosX model collection on Hugging Face](https://huggingface.co/AutomatosX/models).
Additional native families (GLM 4.7 Flash, Nemotron Omni, Unlimited-OCR, Whisper,
MiniCPM-V, and others) are documented under
[Supported Models](docs/SUPPORTED-MODELS.md).

[Install for macOS](#install-ax-engine-for-macos) · [Performance](#performance) · [Documentation](docs/README.md)

## Performance

**MacBook Pro M5 Max 128 GB · Qwen 3.8 AXQ MXFP4 · One active request.**
Each model uses a pinned pack and the same requests across engines.
The 27B DFlash2 rows use identical drafter weights.

**Decode** measures visible output after the first SSE text chunk.
**Effective prefill** uses the first-text latency difference between
short and long inputs. A zero means the tested configuration did not
complete the workload.

<!-- qwen38-m5-modes:begin -->

### Qwen 3.8 27B MXFP4

![Qwen 3.8 27B decode and effective prefill by engine version and mode on M5 Max.](docs/assets/perf-qwen38-27b-m5-modes.svg)

| Engine and mode | Decode (output tok/s) | Effective prefill (input tok/s) |
| --- | ---: | ---: |
| AX Engine 8.1.0 (Direct greedy) | 35.35 | 965.41 |
| AX Engine 8.1.0 (DFlash2) | 82.83 | 895.73 |
| Splash 1.7.0 (DFlash2) | 108.95 | 1292.85 |
| oMLX 0.8.2 (Direct greedy) | 34.47 | 996.35 |
| oMLX 0.8.2 (DFlash2) | 40.02 | 1027.14 |
| mlx-lm 0.32.0 (Direct greedy) | 31.92 | 857.92 |
| MTPLX 2.12.3 (Direct greedy) | 31.18 | 800.05 |
| TensorFold 1.3.4 (Direct greedy) | 0 | 0 |
| TensorFold 1.3.4 (DFlash2) | 0 | 0 |
| ds4 19b3f501 (Direct greedy) | 0 | 0 |

### Qwen 3.8 Flash Next MXFP4

![Qwen 3.8 Flash Next decode and effective prefill by engine version and mode on M5 Max.](docs/assets/perf-qwen38-flash-next-m5-modes.svg)

| Engine and mode | Decode (output tok/s) | Effective prefill (input tok/s) |
| --- | ---: | ---: |
| AX Engine 8.1.0 (Direct greedy) | 50.94 | 3201.85 |
| oMLX 0.8.2 (Direct greedy) | 59.01 | 2284.24 |
| mlx-lm 0.32.0 (Direct greedy) | 0 | 0 |
| MTPLX 2.12.3 (Direct greedy) | 0 | 0 |
| TensorFold 1.3.4 (Direct greedy) | 0 | 0 |
| ds4 19b3f501 (Direct greedy) | 0 | 0 |
| Splash 1.7.0 (DFlash2) | 0 | 0 |

<!-- qwen38-m5-modes:end -->

[Raw results, route checks and reproduction](benchmarks/results/qwen38-m5-modes-20261125/README.md) ·
[AX concurrency, verifier and startup controls](#ax-runtime-controls)

## Hardware and model fit

**Prebuilt packages require macOS 26.2 or later on Apple Silicon.** Hardware tiers:

| Tier | Chips | What to expect |
| --- | --- | --- |
| Primary | **M5 and later** | Defaults, tuning and published performance are measured here. Best experience. |
| Supported | M3, M4 | Tested for correctness every release; may be slower than M5. |
| Best-effort | M2 | Runs, but is not tuned or qualified. |
| Not supported | M1 | Startup fails closed. |

Product SKUs:

- **Qwen 3.8 27B AXQ (`qwen3.8-27b:axq`)** — optimized on M5-class Macs (32 GB+ recommended). The 2026-11-18 product-health gate passed on MacBook Pro M5 Max 128 GB. The earlier published qualification record is the retired 6-bit pack on a Mac mini M4 Pro 64 GB, a supported M4-tier configuration. MXFP4 SKU qualification and MTP Tier 2 remain pending
- **Qwen 3.8 Flash Next MXFP4 MTP** (125B-A6B). Second SKU. Qualification and certification standard: MacBook Pro M5 Max 128 GB, single-stream resident (one request at a time, MTP off, contexts up to 32k). MXFP4 MTP target. The signed v7.9.0 release passed the direct-default gates on this standard. AX certification record: Candidate. Model MTP remains an experimental opt-in; MTP Tier 2 is pending.
  `qwen3.8-flash-next:axq` and `qwen3.8-flash-next:mxfp4` select the pinned
  MXFP4 MTP pack. Launching this pack on the standard host defaults to one
  engine job and a 32,768-token context budget, including output tokens.
  A signed v8.1.0 candidate reached HTTP readiness in **28–30 seconds**
  with automatic buffered reads on the eligible internal SSD.
  [Flash Next operating profile](docs/model-certifications/qwen3.8-flash-next.md#current-mxfp4-evidence).
- **Mac Studio, Ultra-class Apple Silicon with 192 GB+ RAM** (M3 Ultra or newer; M2 Ultra is best-effort) — larger-memory host for the same Flash Next pack.
  The 2026-10-05 direct-default release gates were measured here and stay bound to this host.

Compact single models (Qwen 3.5 9B 4-bit preferred) still fit **16 GB**. Prefer
4-bit for headroom on that class.

Wired-memory controls are available on every Mac AX Engine runs on (M2 and
newer); they are not M5-only. The automatic **no-wire optimization** currently targets only
the two audited Tiel MXFP4 MTP exports on **M5 Max with at least 128 GiB**.
Other supported hosts retain existing wiring. Model fit and measured speed
still depend on memory capacity and workload; see the
[Tiel residency policy](docs/mtp/tiel-prefill-diagnostics.md#residency-policy-for-the-audited-m5-max-exports).

## Why AX Engine

- **MLX-backed AXQ MX packs** — `qwen3.8-27b:axq` selects the pinned
  MXFP4 MTP pack. `qwen3.8-27b:mxfp8` selects the pinned MXFP8 MTP Candidate.
  Flash Next has its own `qwen4_exp` route and MXFP4 MTP selector. See the
  [format and migration policy](docs/QWEN38-FORMAT-POLICY.md).
- **DFlash2 on eligible greedy Qwen text** — Qwen 3.8 27B and Qwen 3.6
  35B-A3B draft a block with DFlash2 when the matching Apache-2.0 checkpoint
  is already on disk (`z-lab/Qwen3.8-27B-DFlash2`,
  `incoai/Qwen3.6-35B-A3B-DFlash2`). The server does not download it, and it
  is not inside the AXQ pack. Sampled requests, grammars, images, and a
  missing checkpoint stay on the normal routes. MTP stays explicit opt-in.
  See [DFlash2](docs/mtp/README.md#dflash2).
- **NAX attention on M5** — on M5 or later with macOS 26.2 or newer, MLX
  selects Neural Accelerator GEMM and SDPA. AX keeps native causal attention
  for allowlisted full-attention prefill of at least 1024 tokens, including
  Qwen 3.8 27B. Gemma stays off that policy. M3 and M4 do not arm it.
  `AX_MLX_NAX_ATTENTION=0` turns the attention policy off. See
  [NAX attention](docs/MLX-BACKEND.md#nax-attention).
- **Multi-model in one process** — keep allowlisted Qwen 3.5/3.6,
  Qwen 3.8 27B MXFP4, Qwen3-Coder-Next, Gemma 4 and embedding models
  resident (`load_mode=add`). Route chat and embeddings by request `model`.
  [Multi-model serving](docs/SERVER.md#multi-model-serving).
- **You own the stack you serve** — AX runs the MLX graph, KV/runtime, and
  OpenAI-compatible server for supported Gemma / Qwen / GLM (and other direct
  families); `mlx-lm` and `llama.cpp` stay optional compatibility adapters
- **Engine-owned scheduling and KV** — continuous batched decode
  (certification-gated), chunked prefill, preempt-and-recompute, and a paged
  KV ledger with cross-request prefix sharing run inside the engine, with a
  published `ax_runtime_*` saturation contract for fleet routers — see
  [Scheduling and KV runtime](#scheduling-and-kv-runtime)
- **Native media and speech** — image/video chat, mixed image+audio reasoning,
  OCR, and Whisper transcription/translation run through repo-owned MLX graphs
  with capability-gated OpenAI endpoints (checkpoint-authoritative; see media
  table below)
- **Clear fleet boundary** — AX Engine owns local Apple Silicon inference;
  [AX Serving](docs/AX-SERVING.md) owns fleet orchestration and NVIDIA/CUDA
  workers
- **Claims you can audit** — public rows ship with checked-in artifacts (route,
  model snapshot, sampler, accept rate, provenance)

## Components and workflow

![AX Engine components and workflow](docs/images/ax-engine-components.svg)

Clients enter through the HTTP server, the Python binding, or a language SDK.
`ax-engine-sdk` keeps one session per loaded model on that model's worker
thread. `ax-engine-core` admits the request, plans a decode-first batch, and
owns the paged KV ledger. `ax-engine-mlx` runs the graph, including DFlash2
block drafting for eligible greedy Qwen text, MTP, n-gram acceleration,
expert paging, and NAX attention on M5. Optional `mlx_lm.server` and llama.cpp
adapters stay on the compatibility route.
Normative detail is in [Architecture](docs/ARCHITECTURE.md).

## Install AX Engine for macOS

Use the curl installer on **Apple Silicon (M5 or later recommended; M3–M4 supported; M2 best-effort), macOS 26.2 or newer**.
It installs **AX Engine.app with its native inference runtime bundled**. You do
not need Homebrew, Python, Xcode or a Hugging Face account for public models.
Quit AX Engine before installing or upgrading, then run in Terminal:

```bash
curl -fsSL https://download.automatosx.com/ax-engine-desktop/install | bash
```

The installer verifies signed release metadata, the download and Apple's app
identity before installing `/Applications/AX Engine.app`. macOS may request your
password if that directory needs administrator access. It preserves models,
settings and conversations, and restores the previous app if replacement fails.
It does not install a global `ax-engine` command on your shell's PATH.

Open **AX Engine** from Applications. In **Models → Catalog**, choose a model
that fits your Mac and download it; use the app's server controls and streaming
chat after the model is ready. Model downloads are separate from installation.
The default Qwen 3.8 27B pack is about 17.4 GB; allow about 20 GB of free disk
space and prefer 32 GB or more of memory for it. Catalog inclusion does not
establish model qualification or memory fit.

### Update or verify the app

Quit AX Engine and run the same curl command again to update to the latest
stable GUI release. Automatic app updates are not available yet.

To download and verify the release without installing it:

```bash
curl -fsSL https://download.automatosx.com/ax-engine-desktop/install | bash -s -- --verify-only
```

For an existing user-writable Applications directory, pass
`--applications-dir "$HOME/Applications"` after `bash -s --`. Run with `--help`
for installer options. If the download fails, stop and retry later; do not
remove quarantine or disable Gatekeeper. See the [macOS app guide](docs/MACOS-APP.md)
for usage and [distribution guide](docs/MACOS-DISTRIBUTION.md) for release verification.

## Advanced: CLI and Python installation

The curl command above is the end-user app installation path. The channels below
are for shell automation, direct API serving or Python development; they do not
install the GUI app. Native CLI packages require Apple Silicon (M5 or later recommended; M3–M4 supported; M2 best-effort)
and macOS 26.2 or later.

| Goal | Channel |
| --- | --- |
| Python SDK and CLI in one environment | [pip](#install-with-pip-no-homebrew-needed) |
| CLI/server without Homebrew or the Python SDK | [Standalone archive](#standalone-download-no-homebrew-no-python-sdk) |
| CLI/server managed by Homebrew | [Homebrew](#homebrew) |

### Install with pip (no Homebrew needed)

1. Install Python from the official macOS installer: open
   [python.org/downloads/macos](https://www.python.org/downloads/macos/),
   download the **Python 3.13** installer (any 3.13.x), run it, then **quit and
   reopen Terminal**. Python 3.12 through 3.14 work; 3.15 does not yet work with
   the `download` extra because MLX publishes no 3.15 wheels.
2. Check the version. Use `python3.13` explicitly: the `python3` that ships with
   macOS is 3.9, cannot install AX Engine, and may open an Xcode Command Line
   Tools prompt.

   ```bash
   python3.13 --version   # must print: Python 3.13.x
   ```

3. Create an environment and install:

   ```bash
   python3.13 -m venv "$HOME/.venvs/ax-engine"
   source "$HOME/.venvs/ax-engine/bin/activate"
   python -m pip install --upgrade pip
   python -m pip install --upgrade "ax-engine[download]>=8.0.6,<9"
   ax-engine doctor
   ```

4. Make `ax-engine` available in every new Terminal (otherwise it is only found
   after running the `source` line above):

   ```bash
   echo 'export PATH="$HOME/.venvs/ax-engine/bin:$PATH"' >> ~/.zprofile
   ```

The wheel provides the Python package (`import ax_engine`), the `ax-engine` and
`ax-engine-server` commands, and the bench binary used by diagnostics. It
bundles the MLX runtime; Xcode and the Metal Toolchain are not needed.
`pip install` printing `No matching distribution found` almost always means an
old Python (3.9 is the usual culprit) or macOS older than 26 — see
[Python installation troubleshooting](docs/GETTING-STARTED.md#python-installation-troubleshooting).

### Standalone download (no Homebrew, no Python SDK)

A signed and notarized `arm64` tarball with the `ax-engine`, `ax-engine-server`
and `ax-engine-bench` binaries and the MLX runtime. Downloading with `curl`
avoids the browser quarantine prompt:

```bash
TAG=$(curl -fsSLI -o /dev/null -w '%{url_effective}' \
  https://github.com/defai-digital/ax-engine/releases/latest | sed 's|.*/||')
ASSET="ax-engine-$TAG-macos-arm64.tar.gz"
BASE="https://github.com/defai-digital/ax-engine/releases/download/$TAG"
cd "$(mktemp -d)"
curl -fsSLO "$BASE/$ASSET" && curl -fsSLO "$BASE/$ASSET.sha256"
shasum -a 256 -c "$ASSET.sha256"          # must print: OK
mkdir -p "$HOME/.local/ax-engine" && tar -xzf "$ASSET" -C "$HOME/.local/ax-engine"
echo 'export PATH="$HOME/.local/ax-engine:$PATH"' >> ~/.zprofile
export PATH="$HOME/.local/ax-engine:$PATH"
ax-engine doctor
```

Serving a model directory you already have needs nothing else. Downloading
models by alias (`ax-engine serve qwen3.8-27b:axq`) also needs Python 3.12+ and
`huggingface-hub`: follow
[Model download helpers](docs/GETTING-STARTED.md#model-download-helpers).

### Homebrew

```bash
brew install defai-digital/tap/ax-engine
ax-engine doctor
```

The self-contained release formula installs the release's pinned
`libmlx.dylib`, `libjaccl.dylib`, and precompiled `mlx.metallib`; it does not
build MLX from source. Native inference from a ready local model directory
needs no Python, Xcode, or Metal Toolchain. Model aliases and preparation
helpers use Python 3.12+; online downloads also require `huggingface-hub`.
Follow the
[model download helpers](docs/GETTING-STARTED.md#model-download-helpers) setup
before downloading or serving an alias.

If more than one install channel is present, `which -a ax-engine` lists every
copy; keep one channel per shell. See [Getting Started](docs/GETTING-STARTED.md)
for the full channel comparison and troubleshooting.

### Run the CLI server

Use [AX Engine for macOS](docs/MACOS-APP.md) for graphical model downloads,
resource charts, server controls and streaming chat. The terminal UI has been
retired; the CLI remains available for automation and local model paths.

**Serve Qwen 3.8 27B AXQ MXFP4 MTP**, then request from another
terminal. `qwen3.8-27b:axq` is the pinned AutomatosX MXFP4 MTP pack (same
checkpoint as `ax-qwen3.8-27b`). The command reuses the cached snapshot when
present and downloads it otherwise (about 17.4 GB; the first start waits for
the download). Listen defaults to `127.0.0.1:31418`:

```bash
ax-engine serve qwen3.8-27b:axq

curl http://127.0.0.1:31418/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"Say hello in one sentence."}],"max_tokens":64}'
```

**Option C — coding model** (resolve + serve; Tiel Coder 35B-A3B is a
development MXFP4 MTP pack, about 22 GB):

```bash
ax-engine serve tiel-coder-35b:axq --port 31418
```

If a download stops with `Repository Not Found` or `401`, that snapshot is not
publicly downloadable. Pick one from the live
[AutomatosX collection](https://huggingface.co/AutomatosX/models?sort=alphabetical);
`qwen3.8-27b:axq` above is public.

Python wheel, source builds, and troubleshooting:
[Getting Started](docs/GETTING-STARTED.md).

## Models

### Public AutomatosX model downloads

`ax-engine download --list` prints the curated public
[AutomatosX model collection](https://huggingface.co/AutomatosX/models?sort=alphabetical).
The GUI shows the runtime catalog in **Models → Catalog**, completed models in
**Your Models**, and current downloads in **Downloads**. Downloads are public,
unrestricted AutomatosX packs only; no Hugging Face account or token is needed.
Download other models independently and use the CLI with a local model path.
Local models remain accessible when the Hub is unavailable. Qwen 3.8 27B AXQ (MXFP4 MTP pack),
Qwen 3.6, Qwen 3.5, and Gemma 4 variants published there (plain 4-bit/6-bit,
QAT, OptiQ, AXQ where available) are first-class serve targets. Other native families (for example **GLM 4.7 Flash**, Nemotron
Omni, Unlimited-OCR, Whisper, MiniCPM-V) use the repo-owned runtime via serve
aliases, presets, or manual model directories; they are not all AutomatosX-managed
packages. Full matrix:
[Supported Models](docs/SUPPORTED-MODELS.md).

**Qwen 3.8 Super-class (2.4T) is experimental only, not a production target.**
Those packs can technically load through the SSD expert-stream path
(`--stream-experts`, default `auto`), but local inference is too slow even at
2-bit to recommend or certify. **Start local serving on Qwen 3.8 27B AXQ
MXFP4 MTP** (`qwen3.8-27b:axq`).

**Recommended starting packages** (serve-ready, match published benches):

| Goal | Alias / family | Why |
| --- | --- | --- |
| Default dense chat + MTP | `qwen3.8-27b:axq` (pinned AXQ MXFP4 MTP) | Qwen 3.8 27B; MXFP4 with MTP sidecar, qualification pending |
| MoE chat + MTP | `ax-qwen3.6-35b-a3b` (4-bit or 6-bit MTP) | Qwen 3.6 MoE pack |
| Dense chat + MTP (3.6) | `ax-qwen3.6-27b` (6-bit MTP preferred) | Secondary dense Qwen 3.6 pack |
| AXQ evaluation candidate | `qwen3.6-27b:axq` (pinned 6-bit) | Qwen 3.6 27B AXQ candidate; explicit until its checkpoint certification gates pass |
| Vision MoE Instruct AXQ | `ax-qwen3-vl-30b` / `ax-qwen3-vl-30b-4bit` | Qwen3-VL 30B-A3B Instruct AXQ packs; candidate, no MTP |
| Holo3 GUI-agent AXQ | `holo3-35b` / `holo3-35b:axq` | Qwen3.5-class 35B-A3B MoE; Tier 1 certified text path; no MTP |
| Ornith coding AXQ | `ornith-35b` / `ornith-35b:axq` | Qwen3.5-class 35B-A3B MoE coding agent; Tier 1 AXQ; no MTP |
| Ornith 1.5 MTP | `ornith-1.5-35b:axq` | Qwen3.5-class 35B-A3B MoE with HF per-expert MTP sidecar; development pack |
| Tiel coding MTP | `tiel-coder-35b:axq` / `cyber-tiel-coder-35b:axq` | Qwen3.5-class 35B-A3B MoE MXFP4 packs; native HF per-expert MTP mode; development, unqualified |
| GPT-OSS AXQ | `gpt-oss-20b:axq` / `gpt-oss-120b:axq` | AutomatosX AXQ; bare aliases stay mlx-community MXFP4-Q4 |
| Nemotron 3 Nano AXQ | `nemotron-3-nano` / `nemotron-3-nano:axq` | `nemotron_h` 30B-A3B; development AXQ |
| Muse-Glimmer image-text AXQ | `muse-glimmer-30b` / `muse-glimmer-30b:axq` | Meta dense 30B image-text agent; ATEM chat; development AXQ; no MTP |
| Multimodal chat + MTP | `ax-gemma4-12b` / 26B / 31B Assistant-MTP | Image/audio/video + assistant draft package |
| Coding agent | `ax-qwen3-coder-next` | Coding-focused MoE; multi-model friendly |
| Embeddings | `ax-embeddinggemma-300m` or Qwen3-Embedding aliases | Batched ingest scale in [full results](docs/PERFORMANCE-RESULTS.md#session-mode-embeddings) |

Repositories ending in `-MTP` or `-Assistant-MTP` already contain the prepared
sidecar or assistant artifacts and `model-manifest.json`. Download them with
the standard flow; do **not** run `download-mtp` afterward.

| Family | Role | Supported AutomatosX snapshots |
| --- | --- | --- |
| Qwen 3.5 9B | Chat / agent | [`AX-Qwen3.5-9B-MLX-4bit-MTP`](https://huggingface.co/AutomatosX/AX-Qwen3.5-9B-MLX-4bit-MTP)<br>[`AX-Qwen3.5-9B-MLX-6bit-MTP`](https://huggingface.co/AutomatosX/AX-Qwen3.5-9B-MLX-6bit-MTP)<br>[`AX-Qwen3.5-9B-MLX-OptiQ-4bit-MTP`](https://huggingface.co/AutomatosX/AX-Qwen3.5-9B-MLX-OptiQ-4bit-MTP) |
| Qwen 3.6 27B | Chat / agent / multimodal | [`AX-Qwen3.6-27B-MLX-4bit-MTP`](https://huggingface.co/AutomatosX/AX-Qwen3.6-27B-MLX-4bit-MTP)<br>[`AX-Qwen3.6-27B-MLX-6bit-MTP`](https://huggingface.co/AutomatosX/AX-Qwen3.6-27B-MLX-6bit-MTP)<br>[`AX-Qwen3.6-27B-MLX-OptiQ-4bit-MTP`](https://huggingface.co/AutomatosX/AX-Qwen3.6-27B-MLX-OptiQ-4bit-MTP)<br>AXQ candidates: [`AXQ-6bit`](https://huggingface.co/AutomatosX/AX-Qwen3.6-27B-MLX-AXQ-6bit-MTP) / [`AXQ-4bit`](https://huggingface.co/AutomatosX/AX-Qwen3.6-27B-MLX-AXQ-4bit-MTP) |
| Qwen 3.8 27B | Chat / agent / multimodal | Default selector `qwen3.8-27b:axq` for [`AXQ-MXFP4-MTP`](https://huggingface.co/AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-MXFP4-MTP). Explicit MXFP8 MTP Candidate via `qwen3.8-27b:mxfp8`. |
| Qwen 3.6 35B-A3B | Chat / agent / multimodal | [`AX-Qwen3.6-35B-A3B-MLX-4bit-MTP`](https://huggingface.co/AutomatosX/AX-Qwen3.6-35B-A3B-MLX-4bit-MTP)<br>[`AX-Qwen3.6-35B-A3B-MLX-6bit-MTP`](https://huggingface.co/AutomatosX/AX-Qwen3.6-35B-A3B-MLX-6bit-MTP)<br>[`AX-Qwen3.6-35B-A3B-MLX-OptiQ-4bit-MTP`](https://huggingface.co/AutomatosX/AX-Qwen3.6-35B-A3B-MLX-OptiQ-4bit-MTP) |
| Qwen3-VL 30B-A3B Instruct | Vision chat (image/video) | AXQ candidates: [`AXQ-6bit`](https://huggingface.co/AutomatosX/AX-Qwen3-VL-30B-A3B-Instruct-MLX-AXQ-6bit) / [`AXQ-4bit`](https://huggingface.co/AutomatosX/AX-Qwen3-VL-30B-A3B-Instruct-MLX-AXQ-4bit) (no MTP) |
| Holo3 35B-A3B | GUI agent (text path) | Certified AXQ: [`AXQ-6bit`](https://huggingface.co/AutomatosX/AX-Holo3-35B-A3B-MLX-AXQ-6bit) / [`AXQ-4bit`](https://huggingface.co/AutomatosX/AX-Holo3-35B-A3B-MLX-AXQ-4bit) (no MTP) |
| Ornith 1.0 35B | Coding agent | Certified AXQ: [`AXQ-6bit`](https://huggingface.co/AutomatosX/AX-Ornith-1.0-35B-MLX-AXQ-6bit) / [`AXQ-4bit`](https://huggingface.co/AutomatosX/AX-Ornith-1.0-35B-MLX-AXQ-4bit) (no MTP) |
| Tiel 35B-A3B | Coding agent | Development MXFP4 MTP: [`Tiel Coder`](https://huggingface.co/AutomatosX/AX-Tiel-Coder-35B-A3B-MLX-AXQ-MXFP4-MTP) / [`Cyber Tiel Coder`](https://huggingface.co/AutomatosX/AX-Cyber-Tiel-Coder-35B-A3B-MLX-AXQ-MXFP4-MTP); native MTP mode, no checkpoint or speed certification |
| GPT-OSS 20B / 120B | Open reasoner | Bare aliases: mlx-community MXFP4-Q4. AXQ: [`20B-6bit`](https://huggingface.co/AutomatosX/AX-gpt-oss-20b-MLX-AXQ-6bit) / [`120B-6bit`](https://huggingface.co/AutomatosX/AX-gpt-oss-120b-MLX-AXQ-6bit) |
| Nemotron 3 Nano 30B-A3B | Hybrid chat | AXQ candidates: [`AXQ-6bit`](https://huggingface.co/AutomatosX/AX-Nemotron-3-Nano-30B-A3B-MLX-AXQ-6bit) / [`AXQ-4bit`](https://huggingface.co/AutomatosX/AX-Nemotron-3-Nano-30B-A3B-MLX-AXQ-4bit) |
| Muse-Glimmer 30B | Image-text agent | AXQ candidates: [`AXQ-6bit`](https://huggingface.co/AutomatosX/AX-Muse-Glimmer-30B-MLX-AXQ-6bit) / [`AXQ-4bit`](https://huggingface.co/AutomatosX/AX-Muse-Glimmer-30B-MLX-AXQ-4bit) (no MTP) |
| Gemma 4 12B | Chat / multimodal | [`AX-Gemma-4-12B-IT-MLX-6bit-Assistant-MTP`](https://huggingface.co/AutomatosX/AX-Gemma-4-12B-IT-MLX-6bit-Assistant-MTP)<br>[`AX-Gemma-4-12B-IT-MLX-QAT-4bit-Assistant-MTP`](https://huggingface.co/AutomatosX/AX-Gemma-4-12B-IT-MLX-QAT-4bit-Assistant-MTP)<br>[`AX-Gemma-4-12B-IT-MLX-QAT-OptiQ-4bit-Assistant-MTP`](https://huggingface.co/AutomatosX/AX-Gemma-4-12B-IT-MLX-QAT-OptiQ-4bit-Assistant-MTP) |
| Gemma 4 26B-A4B | Chat / agent / multimodal | [`AX-Gemma-4-26B-A4B-IT-MLX-6bit-Assistant-MTP`](https://huggingface.co/AutomatosX/AX-Gemma-4-26B-A4B-IT-MLX-6bit-Assistant-MTP)<br>[`AX-Gemma-4-26B-A4B-IT-MLX-OptiQ-4bit-Assistant-MTP`](https://huggingface.co/AutomatosX/AX-Gemma-4-26B-A4B-IT-MLX-OptiQ-4bit-Assistant-MTP)<br>[`AX-Gemma-4-26B-A4B-IT-MLX-QAT-4bit-Assistant-MTP`](https://huggingface.co/AutomatosX/AX-Gemma-4-26B-A4B-IT-MLX-QAT-4bit-Assistant-MTP) |
| Gemma 4 31B | Chat / agent / multimodal | [`AX-Gemma-4-31B-IT-MLX-6bit-Assistant-MTP`](https://huggingface.co/AutomatosX/AX-Gemma-4-31B-IT-MLX-6bit-Assistant-MTP)<br>[`AX-Gemma-4-31B-IT-MLX-OptiQ-4bit-Assistant-MTP`](https://huggingface.co/AutomatosX/AX-Gemma-4-31B-IT-MLX-OptiQ-4bit-Assistant-MTP)<br>[`AX-Gemma-4-31B-IT-MLX-QAT-4bit-Assistant-MTP`](https://huggingface.co/AutomatosX/AX-Gemma-4-31B-IT-MLX-QAT-4bit-Assistant-MTP) |
| Qwen3-Coder-Next | Coding agent | [`AX-Qwen3-Coder-Next-MLX-4bit`](https://huggingface.co/AutomatosX/AX-Qwen3-Coder-Next-MLX-4bit)<br>[`AX-Qwen3-Coder-Next-MLX-6bit`](https://huggingface.co/AutomatosX/AX-Qwen3-Coder-Next-MLX-6bit) |
| DiffusionGemma 26B-A4B | Diffusion language model | [`AX-DiffusionGemma-26B-A4B-IT-MLX-4bit`](https://huggingface.co/AutomatosX/AX-DiffusionGemma-26B-A4B-IT-MLX-4bit) |
| EmbeddingGemma 300M | Embeddings | [`AX-EmbeddingGemma-300M-MLX-8bit`](https://huggingface.co/AutomatosX/AX-EmbeddingGemma-300M-MLX-8bit) |
| Qwen3-Embedding 0.6B | Embeddings | [`AX-Qwen3-Embedding-0.6B-MLX-8bit`](https://huggingface.co/AutomatosX/AX-Qwen3-Embedding-0.6B-MLX-8bit) |
| Qwen3-Embedding 4B / 8B | Embeddings | [`AX-Qwen3-Embedding-4B-MLX-4bit-DWQ`](https://huggingface.co/AutomatosX/AX-Qwen3-Embedding-4B-MLX-4bit-DWQ)<br>[`AX-Qwen3-Embedding-8B-MLX-4bit-DWQ`](https://huggingface.co/AutomatosX/AX-Qwen3-Embedding-8B-MLX-4bit-DWQ) |

Native multimodal and speech support is checkpoint-authoritative: AX advertises
only modalities whose required tower tensors are present in
`model-manifest.json`.

| Model family | Inputs | Native API surface | Current scope |
| --- | --- | --- | --- |
| Qwen3-VL; visual Qwen3.5; Qwen 3.6 | Image, video | Chat/generate | Conv3D visual patches, full ViT/merger, MRoPE, multi-image/video ordering; Qwen 3.6 27B image/video smoke-tested on M3 Max |
| Standard Gemma 4 E4B/26B/31B | Image, video | Chat/generate | Full bidirectional ViT, 2-D RoPE, spatial pooling, checkpoint standardization, and vision projection; catalogued sizes are E4B/26B/31B. E2B still loads from an explicit directory. Conformer audio is not yet native |
| Gemma 4 unified 12B | Image, audio, video | Chat/generate | Encoder-free image/audio connector and sampled per-frame video path; requires the unified connector roles |
| MiniCPM-V 4.6 | One or more images | Chat/generate | Dynamic SigLIP grid, mid-tower merger, OCR/document prompts |
| Nemotron 3 Nano Omni | Image, audio, or both | Chat/generate | RADIO vision plus Parakeet audio with ordered mixed-media spans |
| Unlimited-OCR | Image | Native processed-input runtime; delegated OpenAI OCR profile | Full prefill KV is protected; only generated-token KV uses the decode ring |
| [Unlimited-OCR-2](docs/UNLIMITED-OCR-2.md) | Image, ordered pages | Experimental native MXFP4 and MXFP8 SR-SWA via the Python image-request helper | Pinned-pack M5 Max functional controls with full retained KV |
| Whisper large-v3-turbo | Audio | `/v1/audio/transcriptions`, `/v1/audio/translations`, Rust SDK | WAV/MP3 to 16 kHz; multilingual transcribe/translate; text-generation routes fail closed |

GLM-OCR is not supported.

Download helpers inspect the source tensor index and automatically rebuild
older packaged manifests that omitted a declared Qwen or Gemma vision tower.
For an existing manual model directory, run
`ax-engine-bench generate-manifest --force /path/to/model`.

The default Hugging Face cache layout is
`models--AutomatosX--<repository>/snapshots/<revision>`. Use the shorter
`ax-*` aliases shown by `ax-engine download --list`; for example:

```bash
ax-engine serve qwen3.8-27b:axq
ax-engine serve qwen3.8-27b:axq --offline  # require the pinned MXFP4 MTP cache
```

Aliases, hardware sizing and qualification records are in the
[Supported Models](docs/SUPPORTED-MODELS.md) ·
[Qwen 3.8 27B AXQ certification](docs/model-certifications/qwen3.8-27b-axq.md) ·
[Flash Next certification](docs/model-certifications/qwen3.8-flash-next.md) ·
[Hardware FAQ](docs/FAQ.md#what-hardware-does-ax-engine-support) ·
[CLI](docs/CLI.md).

### Multi-model serving

One process can keep several **allowlisted** models loaded and route each
request by `model` (OpenAI, gRPC, Ollama, Anthropic). Add mode is limited to
Qwen 3.8 27B MXFP4, Qwen 3.5 9B, Qwen 3.6 27B/35B, Qwen3-Coder-Next,
Gemma 4 12B/26B/31B, EmbeddingGemma 300M/2 and Qwen3-Embedding 0.6B–8B.
AutomatosX `AX-` package names resolve to the same targets. Each model owns its own session and scheduler while a process
arbiter fair-rotates Metal turns (no fused cross-model batch).

```bash
# After a first model is already serving on :31418
curl -s http://127.0.0.1:31418/v1/model/load -H 'content-type: application/json' -d '{
  "model_id": "gemma-4-12b-it",
  "model_path": "/path/to/gemma-4-12b-artifacts",
  "load_mode": "add",
  "make_default": false
}'

curl -s http://127.0.0.1:31418/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "gemma-4-12b-it",
  "messages": [{"role": "user", "content": "Hi"}],
  "max_tokens": 32
}'
```

Full contract (load/unload, memory preflight, idle eviction, metrics labels):
[Server: Multi-model serving](docs/SERVER.md#multi-model-serving) ·
[Supported Models](docs/SUPPORTED-MODELS.md#multi-model-serving).

## Scheduling and KV runtime

Token-level scheduling is owned by the engine, not a gateway: each step the
scheduler builds a decode-first, token-budgeted batch with chunked prefill and
mixed prefill+decode routes, backed by a paged logical KV ledger. Full design:
[Scheduler](docs/SCHEDULER.md) · [KV Cache](docs/KV-CACHE.md) ·
[Serving Invariants](docs/SERVING-INVARIANTS.md).

- **Continuous batched decode** — structurally eligible decode requests share
  one batched forward (default on; `AX_MLX_BATCHED_DECODE=0` is the kill
  switch) behind a fail-closed bit-exact certification gate; host-sampled
  batching is a separate opt-in
- **Paged KV ledger with prefix sharing** — per-request block tables,
  ref-counted cross-request prefix reuse, tiered eviction, and an optional
  disk-durable prefix cache that survives restarts
- **Pressure handling** — KV memory-pressure throttling, preempt-and-recompute
  (newest in-flight prefill only, never decode), and server admission control
  (global and per-model concurrency caps → `429`)
- **Fleet telemetry contract** — `/metrics` publishes versioned `ax_runtime_*`
  saturation series (KV utilization, queue depth, batch headroom, TTFT p95,
  decode tok/s, error rate) that [AX Serving](docs/AX-SERVING.md) and other
  routers consume for node selection; token scheduling itself stays on-box

## AX runtime controls

| Model and workload | M5 result |
| --- | --- |
| **Qwen 3.8 27B MXFP4**, one request, explicit direct decode | **32.64 tok/s** across eight fixed-budget prompts. [Direct control](benchmarks/results/qwen-cross-engine-mlx-20261125/README.md#single-request-direct-controls) |
| **Qwen 3.8 27B MXFP4**, four concurrent DFlash2 requests | The existing verifier measured **85.07–88.05 tok/s**. The optional four-row cap measured **82.15–85.00 tok/s** with shorter stream gaps. [Cohort comparison](benchmarks/results/qwen-cross-engine-mlx-20261125/README.md#qwen-38-27b-dflash2) |
| **Flash Next MXFP4**, one resident request, explicit direct decode | **36.24 tok/s** across three 64-token prompts. [Direct control](benchmarks/results/qwen-cross-engine-mlx-20261125/README.md#single-request-direct-controls) |
| **Flash Next MXFP4**, one resident request, opt-in MTP | Canonical one-draft verification measured **33.35 tok/s**. The two-draft block cap measured **29.43–29.67 tok/s**, with shorter stream gaps than the three-draft block. [Verifier controls](benchmarks/results/qwen-cross-engine-mlx-20261125/README.md#flash-next-experimental-required-mtp-on-m5-max-128-gb) |
| **Flash Next MXFP4**, HTTP startup | **28–30 seconds** with automatic buffered reads versus **162–168 seconds** with mapped reads, with the pack already on the internal SSD. [Signed startup crossover](benchmarks/results/flash-next-automatic-buffered-reader-m5-20261123/README.md) |

The [27B](docs/model-certifications/qwen3.8-27b-axq.md) and
[Flash Next](docs/model-certifications/qwen3.8-flash-next.md) certification
records track the current candidate profiles and remaining gates.
[Performance Results](docs/PERFORMANCE-RESULTS.md) has the full M5 controls
and [historical benchmarks](docs/PERFORMANCE-RESULTS.md#historical-results).
[Benchmarks](docs/BENCHMARKS.md) explains reproduction.

## SDKs

Most clients use the OpenAI-compatible HTTP server. Python also has an
in-process session API.

| SDK | Docs |
| --- | --- |
| Rust | [docs/sdk/rust.md](docs/sdk/rust.md) |
| Python | [docs/sdk/python.md](docs/sdk/python.md) |
| JavaScript / TypeScript | [docs/sdk/javascript.md](docs/sdk/javascript.md) |
| Go | [docs/sdk/go.md](docs/sdk/go.md) |
| Ruby | [docs/sdk/ruby.md](docs/sdk/ruby.md) |
| Swift | [docs/sdk/swift.md](docs/sdk/swift.md) |
| Mojo *(experimental)* | [docs/sdk/mojo.md](docs/sdk/mojo.md) |

## Server

`ax-engine serve` is the normal entrypoint (see Quick Start). Default listen is
**`127.0.0.1:31418`** (not AX Serving’s `18080`). Port map, LAN bind, and
Serving vs Engine:

**[Network ports and settings](docs/PORTS.md)**

```bash
curl http://127.0.0.1:31418/v1/runtime
```

Auth, streaming, embeddings, Ollama-shaped routes:
[Server](docs/SERVER.md) · [API Compatibility](docs/API-COMPATIBILITY.md) ·
[OpenClaw](docs/OPENCLAW.md). Fleet / NVIDIA serving:
[AX Serving](docs/AX-SERVING.md).

## Documentation

| Need | Read |
| --- | --- |
| Docs hub | [docs/README.md](docs/README.md) |
| Install and first request | [Getting Started](docs/GETTING-STARTED.md) |
| **Ports, bind host, Engine vs Serving** | **[Ports](docs/PORTS.md)** |
| Models and MTP packages | [Supported Models](docs/SUPPORTED-MODELS.md) · [MTP Docs](docs/mtp/README.md) |
| Hardware / FAQ | [FAQ](docs/FAQ.md) |
| Full performance tables | [Performance Results](docs/PERFORMANCE-RESULTS.md) |
| Serving peer (newest) | [Serving peer detail](docs/performance/ax-vs-peer-mlx-serving-qwen36-2026-08-06.md) |
| Reproduce benchmarks | [Benchmarks](docs/BENCHMARKS.md) |
| Server / API / SDKs | [Server](docs/SERVER.md) · [API](docs/API-COMPATIBILITY.md) · [OpenClaw](docs/OPENCLAW.md) · [SDKs](docs/sdk/README.md) |
| Fleet / NVIDIA (AX Serving) | [AX Serving](docs/AX-SERVING.md) |
| Architecture | [Architecture](docs/ARCHITECTURE.md) |
| Scheduler / KV internals | [Scheduler](docs/SCHEDULER.md) · [KV Cache](docs/KV-CACHE.md) · [Serving Invariants](docs/SERVING-INVARIANTS.md) |

## Development

```bash
cargo build --workspace
cargo test --quiet
cargo clippy --all-targets --all-features -- -D warnings
cargo fmt --check
maturin develop
python -m unittest discover -s python/tests -v
```

Crate layout and conventions: [AGENTS.md](AGENTS.md) ·
[Architecture](docs/ARCHITECTURE.md).

## Limitations

- Qwen3.5 long-prompt prefill can trail upstream MLX references on longer prompts
- Raw HuggingFace / mlx-community snapshots load directly: `model-manifest.json` is auto-generated from `config.json` + safetensors headers on first load and weight sanitization is auto-detected, so `mlx_lm.convert` is not required
- N-gram acceleration is workload-dependent, not a raw kernel speedup
- NVIDIA/CUDA serving requires the separate AX Serving product

Details: [FAQ limitations](docs/FAQ.md#what-are-the-current-limitations).

## Contributing

Issues, wishlist items, reproducible benchmarks, and docs feedback are welcome.
Unsolicited code PRs for runtime, kernel, scheduler, or performance paths are
generally not accepted — open an issue first. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Community

- Website: [automatosx.com](https://automatosx.com)
- Discord: [Join us](https://discord.gg/MECsKdA6zF)
- Email: [enquiry@defai.digital](mailto:enquiry@defai.digital)

## Acknowledgments

AX Engine is grateful to the open-source foundations, benchmarking peers, and
community contributors listed in [Acknowledgments](docs/ACKNOWLEDGMENTS.md).

## License

Apache License, Version 2.0. See [LICENSE](LICENSE).

Copyright (c) 2026 [DEFAI Private Limited](https://defai.digital)
