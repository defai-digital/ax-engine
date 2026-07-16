"""AX Engine inference quality harness.

Import path preference (package-style when ``qa`` is on ``sys.path`` parent):

    from qa.prompt_def import QaPrompt
    from qa.prompts import sample_prompts
    from qa.checkers import run_all_checks

Scripts under ``qa/`` also support flat imports via ``sys.path`` insert of
this directory (legacy). Prefer ``python3 -m`` style when possible.
"""

from __future__ import annotations

__all__ = [
    "QaPrompt",
    "sample_prompts",
    "run_all_checks",
    "bank_size",
]

# Lazy-friendly re-exports for ``from qa import …`` when parent is on path.
try:
    from qa.prompt_def import QaPrompt
    from qa.prompts import bank_size, sample_prompts
    from qa.checkers import run_all_checks
except ImportError:  # flat import when ``qa/`` itself is on sys.path
    from prompt_def import QaPrompt  # type: ignore
    from prompts import bank_size, sample_prompts  # type: ignore
    from checkers import run_all_checks  # type: ignore
