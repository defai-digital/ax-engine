# AX Engine QA

Inference quality harness against a running `ax-engine-server`. It is **not** a
full public-benchmark replacement (MMLU / GSM8K / HumanEval / IFEval / …). It is
a practical gate for:

- server load + chat path health
- catastrophic garbage / repetition / encoding corruption
- basic capability across skill dimensions

## Question bank + sampling

| Piece | Role |
| --- | --- |
| `question_bank.py` | Large prompt bank (dozens of items, many categories) |
| `prompts.py` | `QaPrompt` type + **stratified sampling** |
| `run_qa.py` | CLI runner |
| `checkers.py` | Heuristic + exact-answer checks |

**Default behavior:** each run samples **12** prompts from the bank with
stratified category coverage and a random seed. That reduces the chance that a
pipeline “memorizes” a fixed dozen items.

```bash
# Inspect bank
python3 qa/run_qa.py --list-categories
python3 qa/run_qa.py --list-prompts

# Typical run (server already up)
python3 qa/run_qa.py \
  --base-url http://127.0.0.1:8080 \
  --model llama3.1-8b \
  --streams false \
  --sample 12

# Reproducible sample
python3 qa/run_qa.py --base-url http://127.0.0.1:8080 --model m --seed 42 --sample 12

# Full bank (slower)
python3 qa/run_qa.py --base-url http://127.0.0.1:8080 --model m --all

# Only math + code
python3 qa/run_qa.py --base-url http://127.0.0.1:8080 --model m --categories math code --sample 8
```

`run_full_qa.sh` honors:

| Env | Meaning |
| --- | --- |
| `QA_SAMPLE` | Sample size (default 12) |
| `QA_SEED` | Fixed seed for replay |
| `QA_ALL=1` | Run entire bank |

## Categories (dimensions)

Reasoning, math, code, instruction following, creative, knowledge, common sense,
science, reading comprehension, translation, JSON/format discipline, testing,
summarization — aligned with common public-eval *skill axes*, with original items
suitable for automated checks.

## Adding prompts

1. Append a `QaPrompt` in `question_bank.py` with a **unique** `id`.
2. Prefer `exact_answer` / regex / structural checks over soft keywords alone.
3. Keep prompts short; QA is for engine health, not overnight leaderboards.
4. Run `python3 scripts/test_qa_sampling.py`.
