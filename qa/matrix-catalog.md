# QA matrix catalog (direct + ngram + MTP + embed + multimodal)

This documents the model matrix used for local verification. Paths are resolved
from the active Hugging Face hub cache at runtime by `scripts/run_qa_matrix.py`
(via inventory written to `$QA_SCRATCH/qa-matrix.txt`).

## Direct catalog aliases attempted

See `MODEL_PROFILES` / server presets: Gemma 4, Qwen 3.5/3.6, GLM 4.7 Flash,
Llama 3.1/3.3/4 Scout, Mistral Small/Ministral/Devstral, GPT-OSS 20B/120B.

## MTP packages attempted

ax-local assistant-MTP and Qwen fused MTP sidecars under
`models--ax-local--*` when present in the HF hub cache.

Packages must ship MTP weights (`mtp.safetensors`, `glm_mtp.safetensors`, or
Gemma assistant MTP metadata/weights). Catalog rows that only ship a base
package (for example `gemma-4-12B-it-4bit-ffn4` without an assistant MTP
sidecar) are **skipped** as `not_mtp_package`.

## Unified runner

```bash
cargo build -p ax-engine-server -p ax-engine-bench
# inventory: OK|mode|model_id|artifacts lines into $QA_SCRATCH/qa-matrix.txt
export QA_SCRATCH=/path/to/scratch
export QA_SAMPLE=8 QA_SEED=20260716
python3 scripts/run_qa_matrix.py --surface
# or filter:
python3 scripts/run_qa_matrix.py --modes direct ngram --surface
python3 scripts/run_qa_matrix.py --modes mtp --surface
# embedding packages (no chat bank; runs qa/embedding_probes.py):
python3 scripts/run_qa_matrix.py --modes embed --embed-tier standard
# multimodal packages (no chat bank; runs qa/multimodal_probes.py):
python3 scripts/run_qa_matrix.py --modes multimodal --multimodal-tier smoke
python3 scripts/run_qa_matrix.py --modes multimodal --multimodal-tier standard
```

### Multimodal cells

```text
OK|multimodal|gemma-4-12B-it|/path/to/gemma4-unified-snapshot
```

Requires `config.json` with `vision_config` + `image_token_id` (Gemma 4 unified).
Rows that fail package detection are **skipped** as `not_multimodal_package`.

**Tiers:** `smoke` (remote/video reject + image path) vs `standard` (+ solid-color content).

Classification:

- hard fail on `remote_media_rejected` / `video_rejected` / `multimodal_image` /
  `image_describe_smoke` → `engine_fail`
- hard fail on `image_color_content` → `model_quality` (quantization-sensitive)

Public product policy: **video is rejected** on OpenAI chat routes. Surface and
multimodal matrix probes assert reject; they do not require video generation.

### Embedding cells

```text
OK|embed|qwen3-emb|/path/to/Qwen3-Embedding-…/snapshots/<sha>
OK|embed|embeddinggemma|/path/to/embeddinggemma-…/snapshots/<sha>
```

Requires `tokenizer.json` + `config.json` under the artifacts dir.

**Tiers:** `smoke` (engine health) vs `standard` (default: + STS bank, pair-classification AP, retrieval Hit@1/MRR — MTEB-shaped original fixtures).

Classification:

- hard fail on `api_shape` / `l2_normalized` / `batch_vs_single` / `determinism` / `empty_rejected` → `engine_fail`
- hard fail on `sts_triple_ranking` / `pair_classification` / `retrieval` → `model_quality`

Full reference-oracle cosine checks remain in `scripts/verify_embedding_models.py`.
Full public MTEB is still external / not vendored.

`qa/run_full_qa.sh` is a **thin wrapper**: it writes a direct+ngram inventory from
a default HF-cache model list and calls `run_qa_matrix.py`.

Each cell invokes `qa/run_qa.py` with `--allow-partial` and reads the JSON
report (`totals.hard_passed` / `totals.items`) when present. Soft keyword
misses never force `model_quality`; only hard check failures do.

With `--surface` (or `QA_SURFACE=1`), product-surface probes run before the bank
sample; hard surface failures classify as `engine_fail` / `surface_hard_fail`.

### Mode flags (critical)

| mode | server flags | intent |
| --- | --- | --- |
| **direct** | `--disable-ngram-acceleration` | pure direct decode |
| **ngram** | (default; n-gram eligible) | n-gram acceleration path |
| **MTP** | `--mlx-mtp-disable-ngram-stacking` (and **not** `--disable-ngram-acceleration`) | pure StrictMtp path |
| **embed** | `--disable-ngram-acceleration` | embedding API QA only |
| **multimodal** | `--disable-ngram-acceleration` | vision policy + image path QA |

`--disable-ngram-acceleration` sets `mtp_requested=false` in the runner and
forces `MtpRequestRoute::DirectFallback`, which **bypasses** MTP decode. The
matrix runner rejects MTP server commands that include that flag.

### MTP path proof

Before serving QA, each MTP cell runs a short `ax-engine-bench generate` probe
and requires positive `route.crossover_decisions` telemetry, for example:

- fused MTP: `ax_mtp_source_mtp_proposer_wall_us` / `ax_mtp_verify_tokens` / `ax_mtp_draft_tokens` > 0
- Gemma assistant MTP: `ax_mlx_gemma4_assistant_mtp_enabled=1` with draft/verify activity, or `ax_mtp_verify_tokens` > 0

Cells that fail this gate are classified `engine_fail` with
`mtp_path_not_exercised`.

## CI wiring

| Script | Role |
| --- | --- |
| `scripts/check-qa.sh` | Offline (always) |
| `scripts/check-qa-model.sh` | Live model when CI mounts artifacts |

## Evidence (2026-07-16)

- **Direct**: 19 cells completed; 0 engine failures. Residual partials were
  model-quality on hard prompts (for example `unit_test_clamp`).
- **MTP re-run with correct flags + telemetry gate**: 8 true MTP packages
  exercised (4 Qwen fused + 4 Gemma assistant); 0 engine failures; 1 catalog
  row skipped as `not_mtp_package` (ffn4 base without MTP weights). Residual
  partials again model-quality only.
