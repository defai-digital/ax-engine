# MTP Compare Benchmark Results

This directory holds AX Engine n-gram acceleration benchmarks run against the
MTP-capable `Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed` and
`Youssofal/Qwen3.6-27B-MTPLX-Optimized-Quality` models using coding-shaped
real-prompt suites.

These benchmarks compare AX n-gram acceleration against MTPLX 0.3.7 as an
external reference on coding-shaped workloads. MTPLX results are injected from
a separate MTPLX run rather than produced by this harness.

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
# Both suites, AX rows only
python3 scripts/bench_mtp_compare.py \
  --model-dir /path/to/Qwen3.6-27B-MTPLX-Optimized-Speed \
  --output-dir benchmarks/results/mtp-compare/$(date +%F)-ax-mtp-all

# Both suites with MTPLX reference injection
python3 scripts/bench_mtp_compare.py \
  --model-dir /path/to/Qwen3.6-27B-MTPLX-Optimized-Speed \
  --mtplx-results benchmarks/results/mtp-compare/<date>-mtplx-ref/mtplx.json \
  --output-dir benchmarks/results/mtp-compare/$(date +%F)-ax-mtp-all

# Focused Speed smoke
python3 scripts/bench_mtp_compare.py \
  --model-dir /path/to/Qwen3.6-27B-MTPLX-Optimized-Speed \
  --suites flappy \
  --mtp-only \
  --ax-mtp-max-depth 2 \
  --repetitions 1 \
  --cooldown 5 \
  --output-dir benchmarks/results/mtp-compare/$(date +%F)-speed-depth2-smoke

# Focused Quality smoke
python3 scripts/bench_mtp_compare.py \
  --model-dir /path/to/Qwen3.6-27B-MTPLX-Optimized-Quality \
  --suites flappy \
  --mtp-only \
  --ax-mtp-max-depth 3 \
  --repetitions 1 \
  --cooldown 5 \
  --output-dir benchmarks/results/mtp-compare/$(date +%F)-quality-depth3-smoke
```

## Sampling configuration

These benchmarks use temperature=0.6, top_p=0.95, top_k=20 to match MTPLX's
default sampling. Rows are labeled `sampling_not_distribution_exact` and cannot
be promoted as greedy-exact baselines. See `docs/PERFORMANCE.md §MTP Mode` for
claim boundaries.

## MTPLX reference JSON format

Create a file with the following structure after running MTPLX separately:

```json
{
  "mtplx_version": "0.3.7",
  "hardware": "Apple M5 Max 128GB",
  "run_date": "2026-05-23",
  "results": [
    {"suite": "flappy",    "decode_tok_s": 47.7, "accept_rate": 0.694, "depth": 3},
    {"suite": "long_code", "decode_tok_s": 56.4, "accept_rate": 0.848, "depth": 3}
  ]
}
```

Then pass it with `--mtplx-results path/to/mtplx.json`.

## Updating PERFORMANCE.md

After a new run, copy the `summary.md` numbers into the
`## MTP Mode` section of `docs/PERFORMANCE.md`. The section header comment
marks the update location.

## Prompt suites

Suite files live at `benchmarks/prompts/mtp-suites/`. Each JSONL file follows
the `ax.real_prompt.v1` schema (id, category, prompt, max_tokens). The schema
mirrors MTPLX's `PromptCase` so files can be reused across both tools.

| Suite | Cases | Category | max_tokens | N-gram profile |
|---|---:|---|---:|---|
| `flappy` | 4 | game_code | 1000 | High repetition (method stubs, game loops) |
| `long_code` | 4 | long_code | 1000 | Very high repetition (audit logs, test stubs, SQL) |
