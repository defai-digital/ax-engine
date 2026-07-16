# AX Engine QA

Inference quality harness against a running `ax-engine-server`. It is **not** a
full public-benchmark replacement (MMLU / GSM8K / HumanEval / IFEval / …). It is
a practical gate for:

- server load + chat path health
- catastrophic garbage / repetition / encoding corruption
- basic capability across skill dimensions
- product-surface paths (concurrency, stream, cancel, tools, multimodal)

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
| `surface_probes.py` | Concurrent / stream / cancel / tools / multimodal probes |
| `reporter.py` | Report generators |
| `run_full_qa.sh` | Thin wrapper → unified matrix (direct + ngram inventory) |
| `matrix-catalog.md` | Matrix inventory notes |
| `../scripts/run_qa_matrix.py` | **Unified** direct / ngram / MTP orchestrator |
| `../scripts/check-qa.sh` | Offline CI gate |
| `../scripts/check-qa-model.sh` | Live-model CI gate |

## Unified orchestration

There is **one** multi-model runner: `scripts/run_qa_matrix.py`.

Inventory lines:

```text
OK|direct|model_id|/path/to/artifacts
OK|ngram|model_id|/path/to/artifacts
OK|mtp|model_id|/path/to/artifacts
```

```bash
# Operator matrix (MTP + direct from inventory)
export QA_SCRATCH=/path/to/scratch
# write inventory → $QA_SCRATCH/qa-matrix.txt
python3 scripts/run_qa_matrix.py --surface --modes direct mtp

# Convenience default HF-cache list (direct + ngram)
bash qa/run_full_qa.sh
# equivalent wrapper that materializes inventory and calls run_qa_matrix.py
```

Mode flags:

| mode | Server flags |
| --- | --- |
| **direct** | `--disable-ngram-acceleration` |
| **ngram** | default (n-gram eligible) |
| **MTP** | `--mlx-mtp-disable-ngram-stacking` (never `--disable-ngram-acceleration`) |

`--mode` on `run_qa.py` remains a **report label only**.

## Product-surface probes

```bash
# Against an already-running server
python3 qa/surface_probes.py \
  --base-url http://127.0.0.1:8080 \
  --model my-model \
  --json-output /tmp/surface.json
```

| Probe | Hard? | Notes |
| --- | --- | --- |
| `concurrent_chat` | yes | Parallel `/v1/chat/completions` |
| `stream_nonstream` | yes | Both paths return content / SSE |
| `cancel_request` | yes* | `/v1/requests` + cancel; soft-skip if API 404 |
| `tools_schema` | yes* | Tools payload must not 5xx; 400/422 soft-skip |
| `multimodal_image` | yes* | Tiny PNG; soft-skip if model rejects vision |

Deeper Gemma multimodal suite remains `scripts/qa_gemma4_multimodal.py`
(vision/audio/video; not offline-CI).

Matrix integration: `python3 scripts/run_qa_matrix.py --surface`
(or `QA_SURFACE=1`).

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
python3 qa/run_qa.py --base-url http://127.0.0.1:8080 --model m --mode direct --sample 12 --seed 42
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
