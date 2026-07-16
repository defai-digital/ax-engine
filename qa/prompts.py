"""QA prompts API: dataclass re-export, bank access, stratified sampling.

Design goals
------------
* **Large bank, small runs**: skill dimensions live in ``question_bank.py``.
  Each run samples a subset so the suite does not collapse to a fixed dozen
  prompts that a model or pipeline can overfit.
* **Reproducible**: sampling always records a seed; pass ``--seed`` to replay.
* **Coverage**: default sampling is stratified across categories when possible.
"""

from __future__ import annotations

import random
from dataclasses import replace
from typing import Optional, Sequence

from prompt_def import QaPrompt
from question_bank import QUESTION_BANK, all_bank_categories, bank_size

# Backward-compatible re-exports.
__all__ = [
    "QaPrompt",
    "PROMPTS",
    "DEFAULT_SAMPLE_SIZE",
    "QUESTION_BANK",
    "bank_size",
    "all_categories",
    "get_prompt_by_id",
    "get_prompts_by_category",
    "sample_prompts",
    "clone_prompt",
    "describe_bank",
]

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
        ``prompts`` is a new list (never mutates the bank). ``seed_used`` is the
        RNG seed actually applied (generated when ``seed`` is None).

    Selection rules
    ---------------
    1. If ``prompt_ids`` is set, return those prompts in given order (seed unused
       for selection but still returned for logging).
    2. Otherwise filter by ``categories`` if provided.
    3. If ``stratified`` and n is large enough, take one random item per category
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

    selected: list[QaPrompt] = []
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


def validate_bank(bank: Optional[Sequence[QaPrompt]] = None) -> list[str]:
    """Return validation error strings (empty if bank is well-formed)."""
    items = list(bank) if bank is not None else list(QUESTION_BANK)
    errors: list[str] = []
    seen: set[str] = set()
    valid_match = {"auto", "substring", "token", "last_line", "full", "case"}
    for p in items:
        if not p.id:
            errors.append("empty prompt id")
            continue
        if p.id in seen:
            errors.append(f"duplicate prompt id: {p.id}")
        seen.add(p.id)
        if not p.category:
            errors.append(f"{p.id}: empty category")
        if not p.user:
            errors.append(f"{p.id}: empty user prompt")
        mode = getattr(p, "exact_match", "auto") or "auto"
        if mode not in valid_match:
            errors.append(f"{p.id}: invalid exact_match={mode!r}")
        if mode != "auto" and not p.exact_answer:
            errors.append(f"{p.id}: exact_match set without exact_answer")
        needle = (p.exact_answer or "").strip()
        if needle and mode == "auto" and len(needle) <= 2 and needle.isalpha():
            # Single/double-letter answers need token mode to avoid false passes.
            errors.append(
                f"{p.id}: short alphabetic exact_answer {needle!r} should use "
                "exact_match='token' (or full/last_line)"
            )
    return errors
