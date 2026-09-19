"""Shared QA prompt definition (no bank import — breaks circular deps)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


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
    # Closed-ended expected answer (see checkers.check_exact_answer).
    exact_answer: Optional[str] = None
    # Alternative accepted answers.
    exact_answer_aliases: list[str] = field(default_factory=list)
    # Matching mode for exact_answer:
    #   auto      — short tokens use word-boundary; else last-line then substring
    #   substring — case-insensitive substring anywhere
    #   token     — word-boundary match (good for yes/no/a/b)
    #   last_line — case-insensitive match on the last non-empty line
    #   full      — entire stripped output equals candidate
    #   case      — like last_line but case-sensitive (instruction constraints)
    exact_match: str = "auto"
