# MLX-LM Model Acquisition PRD

## Background

ax-engine loads pre-sanitized MLX weights only (mlx-community format). Raw HuggingFace
checkpoints require two transformations ax-engine does not apply: norm weights are stored as
zero-centered deltas that need `+1.0` added, and conv1d weights need `moveaxis(2,1)`. The
current developer experience has three compounding problems:

1. **Silent failure**: a user who downloads a raw checkpoint and loads it gets wrong inference
   output with no error. This is the most dangerous issue.
2. **No download path**: there is no SDK function, CLI helper, or script for obtaining a model
   artifact that ax-engine can use. Users must manually discover mlx-community, run
   `mlx-lm`, and then invoke a Rust binary to generate the manifest.
3. **Hostile error messages**: when something does go wrong the errors either reference a Rust
   build command unfamiliar to Python users, or omit the next-step hint entirely.

## Goals

- **G1** Detect and hard-fail on unsanitized checkpoints before they silently corrupt output.
- **G2** Give every user a single Python call or script invocation to get from "nothing" to a
  loadable LLM model artifact.
- **G3** Every error on the MLX LLM model-loading path ends with a concrete remediation step.

## Non-goals

- Automatic conversion of raw HF checkpoints inside ax-engine (we continue to require
  pre-sanitized weights; we only improve detection and guidance).
- Tokenizer loading (remains user responsibility per existing design).
- Accepting arbitrary `repo_id` strings in `Session(model_id=...)` for auto-download
  (the `model_id` field stays a session label; `download_model()` is the explicit download API).

## User Stories

**US-1** As a new Python user, I run `pip install ax-engine` and want to get a model running in
under 5 minutes with no Rust toolchain knowledge.

**US-2** As a developer who downloaded a raw HuggingFace checkpoint by mistake, I expect a
clear error explaining exactly what's wrong and how to fix it — not silent garbage output.

**US-3** As a CLI user who just ran `python -m mlx_lm generate --model mlx-community/Qwen3-4B-4bit`, I
expect `ax-engine-server --resolve-model-artifacts hf-cache --preset qwen3_dense` to either
work or tell me the exact next command to run.

---

## Phase 1 — Safety & Error Quality (P0/P1)

**Scope**: No new user-visible features. Only error detection and message improvements.

### 1a. Unsanitized-weight detection (`crates/ax-engine-mlx/src/weights.rs`)

Add a heuristic check in `load_weights()`: after the layer loop, if the first layer has
linear-attention weights, sample the `norm` tensor mean absolute value. Sanitized
mlx-community norm weights are centred around `1.0`; raw HF delta weights are centred near
`0.0`. Threshold: `mean_abs < 0.15` → hard error.

New error variant: `WeightLoadError::UnsanitizedWeights(String)`.

Error message must include:
- what was observed (mean abs value)
- what was expected (~1.0 for sanitized weights)
- the remediation command: `pip install mlx-lm && mlx_lm.convert ...`

### 1b. Python SDK pre-flight (`python/ax_engine/__init__.py`)

In `Session.__init__`, before delegating to the Rust binding, check: if `mlx=True` and neither
`mlx_model_artifacts_dir` nor `AX_ENGINE_MLX_MODEL_ARTIFACTS_DIR` is set, raise `ValueError`
with a Python-friendly message pointing to `ax_engine.download_model()`.

### 1c. HF cache error hints (`crates/ax-engine-server/src/args.rs`)

Three targeted improvements:
- "no valid AX model artifacts" error: append a `generate-manifest` hint.
- "missing model-manifest.json" error (in `validate_preset_model_artifacts`): already has a
  Rust build hint; add a Python alternative: `python scripts/download_model.py <repo-id>`.
- The multiple-match error: no change needed (it already names the candidates).

---

## Phase 2 — Download Tooling (P1/P2)

**Scope**: New user-facing download surface.

### 2a. `ax-engine-bench generate-manifest` subcommand (`crates/ax-engine-bench`)

Add `generate-manifest <model-dir>` as a new subcommand to `ax-engine-bench`. This makes
manifest generation available to Homebrew users who have no `cargo` toolchain.

Implementation: thin dispatch to `ax_engine_core::convert::convert_hf_model_dir` and
`ax_engine_core::convert::write_manifest` — the same Rust functions used by the existing
standalone `generate-manifest` binary.

### 2b. `scripts/download_model.py`

MLX LLM model download script. Accepts a positional `repo_id` argument plus
`--dest` and `--force`. Performs:

1. `python -m mlx_lm generate --model <repo-id> --prompt x --max-tokens 1`
   to let `mlx-lm` acquire and validate the MLX LLM model
2. Post-download validation (safetensors present, config.json present)
3. Auto-runs manifest generation: tries `ax-engine-bench generate-manifest` first,
   falls back to `cargo run`, falls back to hint
4. Rejects embedding repo IDs; embedding artifacts are downloaded manually.

### 2c. `ax_engine.download_model()` (`python/ax_engine/__init__.py`)

```python
def download_model(
    repo_id: str,
    dest: str | Path | None = None,
    *,
    force: bool = False,
) -> Path
```

Runs `mlx-lm` to download the model. After download, calls `_try_generate_manifest()`
which tries `ax-engine-bench generate-manifest` (installed) then `cargo run` (dev repo).
Returns the local directory `Path` with manifest generated when possible.

`mlx-lm` is the model acquisition dependency; raise a Python-friendly error with
`pip install mlx-lm` if missing.

---

## Phase 3 — SDK Polish & Docs (P2/P3)

**Scope**: Documentation and example improvements.

### 3a. "Getting a Model" section in `docs/GETTING-STARTED.md`

Insert before "First Commands". Cover the two paths:
- **mlx-community (recommended)**: `download_model()` → `generate-manifest` → server
- **raw HF checkpoint**: `mlx_lm.convert` → `generate-manifest` → server

### 3b. `examples/python/quick_start.py`

End-to-end new-user example:
1. `download_model('mlx-community/Qwen3-4B-4bit')` (skips if already present)
2. Remind user to run `generate-manifest` if manifest absent
3. `Session(mlx=True, mlx_model_artifacts_dir=path)`
4. `session.generate(...)`

---

## Success Metrics

| Metric | Target |
|--------|--------|
| User loads unsanitized checkpoint | Hard error within `load_weights()` with remediation hint |
| `Session(mlx=True)` with no path | `ValueError` in Python before touching Rust, with download hint |
| HF cache miss error | Error includes `generate-manifest` command |
| New user time-to-first-token | Under 10 minutes with only `pip install` and Python knowledge |
