# MTP Compare Benchmark Results

This directory holds the legacy AX-vs-MTPLX comparison artifacts. New fair
Qwen3.6 MTP comparisons should use `benchmarks/results/mtp-fair/` and
`scripts/bench_qwen36_mtp_fair.py`, which exclude `Youssofal/*MTPLX*` bundles
and use provenance-recorded sidecars from standard `Qwen/Qwen3.6-*` shards plus
`mlx-community/*-4bit` base models.

The files here are retained for historical analysis only. Do not promote new
public Qwen3.6 MTP claims from this directory unless the result explicitly
states its model provenance and why the legacy optimized bundle comparison is
the intended target.

## Directory structure

```
mtp-compare/
  README.md                          (this file)
  <date>-ax-mtp-all/
    summary.md                       (human-readable results table)
    flappy/
      flappy.json                    (AX artifact: direct + n-gram rows for flappy suite)
      real-flappy-*-prompts/         (tokenized prompt artifacts)
    long_code/
      long_code.json                 (AX artifact: direct + n-gram rows for long_code suite)
      real-long_code-*-prompts/      (tokenized prompt artifacts)
```

## Running the benchmark

```bash
python3 scripts/prepare_qwen36_mtp_sidecar.py --model 27b
python3 scripts/prepare_qwen36_mtp_sidecar.py --model 35b
python3 scripts/bench_qwen36_mtp_fair.py \
  --models 27b-4bit 35b-a3b-4bit \
  --engines mtplx rapid_mlx ax_engine \
  --suites flappy long_code \
  --depth-policy fair-shared
```

## Sampling configuration

These benchmarks use temperature=0.6, top_p=0.95, top_k=20 to match MTPLX's
default sampling. Rows are labeled `sampling_not_distribution_exact` and cannot
be promoted as greedy-exact baselines. See `docs/PERFORMANCE.md §MTP Mode` for
claim boundaries.

## MTPLX reference JSON format

Create a file with the following structure after running MTPLX separately on
the same prompt suite files:

```json
{
  "mtplx_version": "0.3.7",
  "hardware": "Apple M5 Max 128GB",
  "run_date": "2026-05-27",
  "runner": "scripts/bench_mtplx_prompt_suites.py",
  "prompt_source": "benchmarks/prompts/mtp-suites/*.jsonl",
  "results": [
    {"model_bundle": "Speed",   "suite": "flappy",    "decode_tok_s": 59.2, "accept_rate": 0.995, "depth": 3},
    {"model_bundle": "Speed",   "suite": "long_code", "decode_tok_s": 59.8, "accept_rate": 0.996, "depth": 3},
    {"model_bundle": "Quality", "suite": "flappy",    "decode_tok_s": 43.0, "accept_rate": 0.994, "depth": 3},
    {"model_bundle": "Quality", "suite": "long_code", "decode_tok_s": 43.2, "accept_rate": 0.997, "depth": 3}
  ]
}
```

Then pass it with `--mtplx-results path/to/mtplx.json`.

The prompt-parity MTPLX runner used for the published d=3 comparison is:

```bash
/opt/homebrew/var/mtplx/venv-0.3.7/bin/python scripts/bench_mtplx_prompt_suites.py \
  --model ~/.cache/huggingface/hub/models--ax-local--Qwen3.6-27B-MTP/snapshots/v1 \
  --suite flappy \
  --prompts benchmarks/prompts/mtp-suites/flappy.jsonl \
  --output benchmarks/results/mtp-compare/$(date +%F)-mtplx-d3/speed-flappy/mtplx.json \
  --profile sustained \
  --depth 3 \
  --max-tokens 1000 \
  --repetitions 5 \
  --warmup-repetitions 1 \
  --cooldown 15 \
  --seed 42 \
  --disable-thinking
```

## Updating PERFORMANCE.md

After a new run, copy the `summary.md` numbers into the
`## MTP Mode` section of `docs/PERFORMANCE.md`. Put rows with matching MTPLX
reference artifacts in the comparison table. Put AX rows without a matching
MTPLX artifact in the AX-only smoke table. The section header comment marks the
update location.

## Prompt suites

Suite files live at `benchmarks/prompts/mtp-suites/`. Each JSONL file follows
the `ax.real_prompt.v1` schema (id, category, prompt, max_tokens). The schema
mirrors MTPLX's `PromptCase` so files can be reused across both tools.

| Suite | Cases | Category | max_tokens | N-gram profile |
|---|---:|---|---:|---|
| `flappy` | 4 | game_code | 1000 | High repetition (method stubs, game loops) |
| `long_code` | 4 | long_code | 1000 | Very high repetition (audit logs, test stubs, SQL) |
