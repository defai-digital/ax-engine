"""QA prompts API: dataclass, legacy PROMPTS alias, stratified sampling from the bank.

Design goals
------------
* **Large bank, small runs**: standard public-eval *dimensions* (math, code,
  instruction following, reasoning, knowledge, …) live in `question_bank.py`.
  Each run samples a subset so the suite does not collapse to a fixed dozen
  prompts that a model or pipeline can overfit.
* **Reproducible**: sampling always records a seed; pass `--seed` to replay.
* **Coverage**: default sampling is stratified across categories when possible.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, replace
from typing import Optional, Sequence


@dataclass
class QaPrompt:
    id: str
    category: str
    system: Optional[str]
    user: str
    keywords: list[str] = field(default_factory=list)
    regex_patterns: list[str] = field(default_factory=list)
    min_length: int = 20
    max_repetition_ratio: float = 0.3
    description: str = ""
    min_test_count: int = 0
    json_expected_total: Optional[float] = None
    # Case-insensitive substring match against the model output (after strip).
    exact_answer: Optional[str] = None
    # Alternative accepted answers (also case-insensitive substring).
    exact_answer_aliases: list[str] = field(default_factory=list)


# Import bank after QaPrompt exists (question_bank imports QaPrompt from here).
from question_bank import QUESTION_BANK, all_bank_categories, bank_size  # noqa: E402

# Backward-compatible name: full bank (not the sampled run set).
PROMPTS: list[QaPrompt] = QUESTION_BANK

# Default per-run sample size — large enough for multi-category coverage,
# small enough for interactive QA against a loaded model.
DEFAULT_SAMPLE_SIZE = 12


def get_prompt_by_id(prompt_id: str) -> Optional[QaPrompt]:
    for p in QUESTION_BANK:
        if p.id == prompt_id:
            return p
    return None


def get_prompts_by_category(category: str) -> list[QaPrompt]:
    return [p for p in QUESTION_BANK if p.category == category]


def all_categories() -> list[str]:
    return all_bank_categories()


def _normalize_seed(seed: Optional[int]) -> int:
    if seed is None:
        # 31-bit positive seed for portable logging / CLI flags.
        return random.SystemRandom().randint(0, 2**31 - 1)
    return int(seed) & 0x7FFFFFFF


def sample_prompts(
    n: int = DEFAULT_SAMPLE_SIZE,
    seed: Optional[int] = None,
    bank: Optional[Sequence[QaPrompt]] = None,
    categories: Optional[Sequence[str]] = None,
    prompt_ids: Optional[Sequence[str]] = None,
    stratified: bool = True,
) -> tuple[list[QaPrompt], int]:
    """Sample prompts from the bank.

    Returns
    -------
    (prompts, seed_used)
        `prompts` is a new list (never mutates the bank). `seed_used` is the
        RNG seed actually applied (generated when `seed` is None).

    Selection rules
    ---------------
    1. If `prompt_ids` is set, return those prompts in given order (seed unused
       for selection but still returned for logging).
    2. Otherwise filter by `categories` if provided.
    3. If `stratified` and n is large enough, take one random item per category
       first, then fill remaining slots uniformly without replacement.
    4. If n >= len(pool), return a shuffled copy of the whole pool.
    """
    seed_used = _normalize_seed(seed)
    rng = random.Random(seed_used)

    if prompt_ids:
        selected: list[QaPrompt] = []
        for pid in prompt_ids:
            p = get_prompt_by_id(pid)
            if p is None:
                raise ValueError(f"unknown prompt id: {pid}")
            selected.append(p)
        return selected, seed_used

    pool = list(bank) if bank is not None else list(QUESTION_BANK)
    if categories:
        wanted = {c.lower() for c in categories}
        pool = [p for p in pool if p.category.lower() in wanted]
    if not pool:
        raise ValueError("empty prompt pool after filters")

    n = max(1, int(n))
    if n >= len(pool):
        rng.shuffle(pool)
        return pool, seed_used

    if not stratified:
        return rng.sample(pool, n), seed_used

    # Stratified: one from as many categories as possible, then fill.
    by_cat: dict[str, list[QaPrompt]] = {}
    for p in pool:
        by_cat.setdefault(p.category, []).append(p)

    cat_order = list(by_cat.keys())
    rng.shuffle(cat_order)

    selected = []
    selected_ids: set[str] = set()
    for cat in cat_order:
        if len(selected) >= n:
            break
        choice = rng.choice(by_cat[cat])
        selected.append(choice)
        selected_ids.add(choice.id)

    remaining = [p for p in pool if p.id not in selected_ids]
    rng.shuffle(remaining)
    while len(selected) < n and remaining:
        selected.append(remaining.pop())

    rng.shuffle(selected)
    return selected, seed_used


def clone_prompt(prompt: QaPrompt) -> QaPrompt:
    """Defensive copy if a caller needs to mutate a prompt instance."""
    return replace(
        prompt,
        keywords=list(prompt.keywords),
        regex_patterns=list(prompt.regex_patterns),
        exact_answer_aliases=list(prompt.exact_answer_aliases),
    )


def describe_bank() -> str:
    cats = all_categories()
    lines = [f"question bank: {bank_size()} prompts, {len(cats)} categories"]
    for c in cats:
        lines.append(f"  {c:16s} {len(get_prompts_by_category(c)):3d}")
    return "\n".join(lines)
