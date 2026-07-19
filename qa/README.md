# AX Engine QA

Inference quality harness against a running `ax-engine-server`. It is **not** a
full public-benchmark replacement (MMLU / GSM8K / HumanEval / IFEval / MTEB /
MMMU / …). It is a practical gate for:

- server load + chat path health
- catastrophic garbage / repetition / encoding corruption
- basic capability across skill dimensions
- product-surface paths (concurrency, stream parity, cancel, tools, media policy)
- **multimodal serving health** (fail-closed remote/video, capability-honest image)
- **embedding model health** (API shape, L2, batch consistency, semantic order)

## Best practices (rethought)

| Principle | Practice in this harness |
| --- | --- |
| **Separate engine vs model quality** | `engine_fail` blocks CI; `model_quality` is partial by default |
| **Capability-honest soft skips** | Soft-skip vision only when `/v1/models` does **not** advertise image |
| **Fail closed for unsupported media** | Remote image URLs and public `video_url` must 4xx (not 200/5xx) |
| **Prove the path you claim** | MTP telemetry gate; multimodal package detection before probes |
| **Small stratified bank, large inventory** | Seeded samples beat a fixed dozen “golden” prompts |
| **Offline tests for the harness** | `check-qa.sh` always runs; GPU matrix is optional |
| **Do not pretend to be leaderboards** | No vendored MMLU/MTEB/VLMEval; optional external adapters only |

## Process / CI integration

| Gate | When | Command |
| --- | --- | --- |
| **Offline QA (required)** | Every CI `scripts` job + `check-scripts.sh` | `bash scripts/check-qa.sh` |
| **Model QA (required when artifacts mounted)** | CI `model-smoke` with MLX artifacts | `bash scripts/check-qa-model.sh` |
| **Full local matrix** | Operator machine with models | `python3 scripts/run_qa_matrix.py …` or `bash qa/run_full_qa.sh` |

Offline gate covers bank integrity, harness unit tests, and `py_compile`.
It never needs GPU weights.

Model gate starts a server against `$AX_ENGINE_MLX_MODEL_ARTIFACTS_DIR`, runs
**surface probes** (hard) and a small stratified bank sample (`--allow-partial`).

## Layout

| Piece | Role |
| --- | --- |
| `prompt_def.py` | `QaPrompt` dataclass (no bank import) |
| `question_bank.py` | Prompt bank (dozens of items, many categories) |
| `prompts.py` | Sampling API + bank validation |
| `checkers.py` | Hard vs soft quality checks |
| `client.py` | OpenAI chat + `/v1/generate` clients |
| `run_qa.py` | CLI runner (HTML + JSON reports) |
| `surface_probes.py` | Concurrent / stream parity / cancel / tools / media policy / multimodal |
| `multimodal_probes.py` | Multimodal tiers (smoke / standard) for matrix + CI |
| `embedding_probes.py` | Embedding QA tiers (smoke / standard) |
| `embedding_bank.py` | Original STS / pair / retrieval fixtures (MTEB-shaped) |
| `reporter.py` | Report generators |
| `run_full_qa.sh` | Thin wrapper → unified matrix (direct + ngram inventory) |
| `matrix-catalog.md` | Matrix inventory notes |
| `../scripts/run_qa_matrix.py` | **Unified** direct / ngram / MTP / embed / multimodal orchestrator |
| `../scripts/check-qa.sh` | Offline CI gate |
| `../scripts/check-qa-model.sh` | Live-model CI gate |

## Unified orchestration

There is **one** multi-model runner: `scripts/run_qa_matrix.py`.

Inventory lines:

```text
OK|direct|model_id|/path/to/artifacts
OK|ngram|model_id|/path/to/artifacts
OK|mtp|model_id|/path/to/artifacts
OK|embed|embedding_model_id|/path/to/artifacts
OK|multimodal|vision_model_id|/path/to/artifacts
```

```bash
# Operator matrix (MTP + direct from inventory)
export QA_SCRATCH=/path/to/scratch
# write inventory → $QA_SCRATCH/qa-matrix.txt
python3 scripts/run_qa_matrix.py --surface --modes direct mtp

# Embedding packages only
python3 scripts/run_qa_matrix.py --modes embed

# Multimodal packages only (policy + image path; standard = + color content)
python3 scripts/run_qa_matrix.py --modes multimodal --multimodal-tier smoke
python3 scripts/run_qa_matrix.py --modes multimodal --multimodal-tier standard

# Convenience default HF-cache list (direct + ngram)
bash qa/run_full_qa.sh
# equivalent wrapper that materializes inventory and calls run_qa_matrix.py
```

Mode flags:

| mode | Server flags | Suite |
| --- | --- | --- |
| **direct** | `--disable-ngram-acceleration` | chat bank + optional surface |
| **ngram** | default (n-gram eligible) | chat bank + optional surface |
| **MTP** | `--mlx-mtp-disable-ngram-stacking` (never `--disable-ngram-acceleration`) | chat bank + MTP path proof |
| **embed** | `--disable-ngram-acceleration` | `embedding_probes` only (no chat bank) |
| **multimodal** | `--disable-ngram-acceleration` | `multimodal_probes` only (no chat bank) |

`--mode` on `run_qa.py` remains a **report label only**.

## Embedding QA (market standard vs this harness)

### Industry layers

| Layer | Market standard | Typical metrics |
| --- | --- | --- |
| **1. Engine smoke** | OpenAI `/v1/embeddings` works | shape, finite, L2 |
| **2. Consistency** | batch ≈ single; determinism | cosine ≈ 1 |
| **3. Task micro-suite** | MTEB *task shapes* at tiny scale | STS ranking, pair accuracy, Hit@1 / MRR |
| **4. Reference oracle** | match trusted family impl | cosine vs mlx-lm / mlx-embeddings |
| **5. Full MTEB** | public leaderboard | dozens of datasets, nDCG@10, etc. |

### AX tiers

| Tier | CLI / matrix | Covers |
| --- | --- | --- |
| **smoke** | `--tier smoke` / `--embed-tier smoke` | layers 1–2 + 3 STS triples |
| **standard** (default) | `--tier standard` | smoke + pair classification + retrieval Hit@1/MRR + full STS bank |
| **oracle** | `scripts/verify_embedding_models.py` | layer 4 (optional publish gate) |
| **MTEB** | external | layer 5 — **not** vendored here |

JSON reports include `metrics` (`sts_triple_accuracy`, `pair_classification_accuracy`,
`retrieval_hit_at_1`, `retrieval_mrr`) and `market_alignment` metadata
(`schema_version: 2`).

### Probes

| Probe | Tier | Checks |
| --- | --- | --- |
| `api_shape` | smoke+ | HTTP 200, batch length, dim, finite |
| `l2_normalized` | smoke+ | ‖v‖ ≈ 1 |
| `batch_vs_single` | smoke+ | min cosine ≥ 0.999 (0.996 for Qwen 8B 4bit/DWQ) |
| `determinism` | smoke+ | re-run cosine ≈ 1 |
| `empty_rejected` | smoke+ | empty → 400 |
| `sts_triple_ranking` | smoke+ | similar > unrelated |
| `pair_classification` | standard | paraphrase accuracy @ cosine threshold |
| `retrieval` | standard | Hit@1 + MRR on tiny corpus |

Fixtures are **original** (`embedding_bank.py`) — MTEB-shaped, not MTEB data.

```bash
# Standard tier (default)
python3 qa/embedding_probes.py \
  --base-url http://127.0.0.1:31418 \
  --model qwen3-embedding \
  --tokenizer /path/to/tokenizer.json \
  --tier standard \
  --json-output /tmp/embed-qa.json

# Matrix (README embedding models)
# OK|embed|id|/path/to/snapshot
python3 scripts/run_qa_matrix.py --modes embed --embed-tier standard

# Family oracle (publish)
python3 scripts/verify_embedding_models.py --model-dir /path/to/snapshot
```

## Product-surface probes

```bash
# Against an already-running server
python3 qa/surface_probes.py \
  --base-url http://127.0.0.1:31418 \
  --model my-model \
  --json-output /tmp/surface.json
```

| Probe | Hard? | Notes |
| --- | --- | --- |
| `concurrent_chat` | yes | Parallel `/v1/chat/completions` |
| `stream_nonstream` | yes | Both paths return content; **parity** of normalized text (opt-out `--no-stream-parity`) |
| `cancel_request` | yes* | `/v1/requests` + cancel; soft-skip if API 404 |
| `tools_schema` | yes* | Tools payload must not 5xx; 400/422 soft-skip |
| `remote_media_rejected` | yes | Remote `https://…` image URL must 4xx |
| `video_rejected` | yes | Public `video_url` must 4xx (product policy) |
| `multimodal_image` | yes* | Tiny PNG; soft-skip **only** when model does not advertise image |

## Multimodal QA

```bash
# Against an already-running vision model
python3 qa/multimodal_probes.py \
  --base-url http://127.0.0.1:31418 \
  --model gemma-4-12B-it \
  --tier smoke \
  --json-output /tmp/mm.json

# Stronger content gate (solid blue color + describe)
python3 qa/multimodal_probes.py --model gemma-4-12B-it --tier standard --require-image
```

| Tier | Covers |
| --- | --- |
| **smoke** | remote reject, video reject, capability-honest image path |
| **standard** | smoke + solid-color content + describe/thinking-channel smoke |

Package detection: `config.json` with `vision_config` + `image_token_id` (Gemma 4 unified).

| Layer | Market analogue | AX |
| --- | --- | --- |
| Fail-closed media | OpenAI/vLLM security posture | hard probes |
| Vision path smoke | peer serving smoke | `multimodal_image` / describe |
| Tiny content | color / OCR digit micro-suite | `image_color_content` (standard) |
| Full VLM eval | VLMEvalKit / MMMU | **not** vendored |

Deeper Gemma operator suite (audio/speech, Pillow fixtures) remains
`scripts/qa_gemma4_multimodal.py --strict` (not offline-CI). Public product
routes **reject video**; that script’s historical GIF probe is for research
bring-up only — matrix/surface expect **reject**.

CI model-smoke runs multimodal **smoke** when the mounted package or live
card advertises image; text-only mounts skip cleanly.

Matrix: `python3 scripts/run_qa_matrix.py --modes multimodal --multimodal-tier smoke`
Surface on chat cells: `python3 scripts/run_qa_matrix.py --surface` (or `QA_SURFACE=1`).

## Hard vs soft checks (question bank)

| Kind | Examples | Effect |
| --- | --- | --- |
| **Hard** | length, repetition, garbage, unicode, exact_answer, regex (all), invoice_total | Drive `auto_pass` and exit code |
| **Soft** | keywords | Reported only |

`exact_match` modes: `auto` | `token` | `full` | `last_line` | `case` | `substring`.

## Question bank + sampling

Default: stratified sample of **12** from a larger bank with seed replay.

```bash
python3 qa/run_qa.py --list-categories
python3 qa/run_qa.py --validate-bank
python3 qa/run_qa.py --base-url http://127.0.0.1:31418 --model m --mode direct --sample 12 --seed 42
```

### Exit codes (`run_qa.py`)

| Exit | Meaning |
| --- | --- |
| `0` | All hard checks passed (or `--allow-partial`) |
| `1` | Hard failure / empty run |
| `2` | Bad CLI / bank validation failure |

JSON report (`schema_version: 1`) is written beside HTML unless `--no-json`.

## Reports

Generated HTML/JSON/logs under `qa/reports/` are **local artifacts** (gitignored).

## Adding prompts

1. Append a unique `QaPrompt` in `question_bank.py`.
2. Prefer `exact_answer` / regex over soft keywords.
3. Short answers → `exact_match='token'` or `'full'`.
4. Run:

```bash
bash scripts/check-qa.sh
```
