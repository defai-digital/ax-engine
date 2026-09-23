#!/usr/bin/env python3
"""Assert the Flash Next multimodel review report is complete.

Checks that the consolidated report under the multimodel review directory:
  * exists and is non-trivial;
  * carries a progress, a best-practice and a plan section;
  * cites every one of the six reviewer receipts (glm, qwen, kimi, muse,
    grok, claude);
  * keeps the product-level gates explicitly open (MTP-S/P/D tokens,
    not_assessed, fail-closed default admission, no release).

Exits nonzero with a reason list when any of that is missing.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REVIEW_DIR = REPO_ROOT / ".internal/reports/flash-next-multimodel-review-20260923"
REPORT_CANDIDATES = ("REPORT.md", "report.md", "README.md")

REVIEWERS = ("glm", "qwen", "kimi", "muse", "grok", "claude")

# Each entry is (label, tuple of case-insensitive tokens that must all appear).
REQUIRED_SECTIONS = (
    ("progress", ("## progress", "progress")),
    ("best practices", ("best practice",)),
    ("plan", ("## plan", "plan")),
)

OPEN_GATE_TOKENS = (
    "MTP-S",
    "MTP-P",
    "MTP-D",
    "not_assessed",
    "fail-closed",
)


def find_report() -> Path | None:
    if not REVIEW_DIR.is_dir():
        return None
    for name in REPORT_CANDIDATES:
        candidate = REVIEW_DIR / name
        if candidate.is_file():
            return candidate
    return None


def main() -> int:
    problems: list[str] = []

    report = find_report()
    if report is None:
        print(f"no report found under {REVIEW_DIR.relative_to(REPO_ROOT)}")
        return 1

    text = report.read_text(encoding="utf-8")
    lowered = text.lower()

    if len(text) < 500:
        problems.append(f"{report.name} is too short ({len(text)} bytes)")

    for label, tokens in REQUIRED_SECTIONS:
        if not any(token in lowered for token in tokens):
            problems.append(f"missing {label} section")

    for reviewer in REVIEWERS:
        receipt = REVIEW_DIR / f"{reviewer}.receipt"
        if not receipt.is_file():
            problems.append(f"missing {reviewer} receipt")
            continue
        if reviewer not in lowered:
            problems.append(f"report does not cite the {reviewer} receipt")

    for token in OPEN_GATE_TOKENS:
        if token.lower() not in lowered:
            problems.append(f"missing open-gate token: {token}")

    # The gates must be stated as open, not closed.
    for closed in ("release_ready=true", "release-ready: yes", "gate closed"):
        if closed in lowered:
            problems.append(f"report appears to close a gate: {closed}")

    if problems:
        for problem in problems:
            print(f"FAIL: {problem}")
        return 1

    print(
        f"OK: {report.relative_to(REPO_ROOT)} cites all six receipts, "
        "has progress/best-practice/plan sections, and keeps the gates open"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
