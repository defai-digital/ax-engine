"""Quality checkers for AX Engine QA.

Checks are either **hard** (fail the item / suite) or **soft** (reported only).
Hard failures drive ``auto_pass`` and process exit codes. Soft checks surface
keyword coverage without blocking engine-health gates.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional

from prompt_def import QaPrompt


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""
    score: float = 0.0
    # Soft checks never fail auto_pass; they are informational.
    hard: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "detail": self.detail,
            "score": self.score,
            "hard": self.hard,
        }


@dataclass
class QualityReport:
    prompt_id: str
    checks: list[CheckResult] = field(default_factory=list)
    auto_pass: bool = False
    manual_review: bool = False
    output_preview: str = ""

    @property
    def passed_count(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    @property
    def total_count(self) -> int:
        return len(self.checks)

    @property
    def hard_checks(self) -> list[CheckResult]:
        return [c for c in self.checks if c.hard]

    @property
    def soft_checks(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.hard]

    @property
    def hard_pass(self) -> bool:
        hard = self.hard_checks
        return all(c.passed for c in hard) if hard else True

    @property
    def summary(self) -> str:
        if not self.checks:
            return "no checks"
        hard = self.hard_checks
        soft = self.soft_checks
        hard_ok = sum(1 for c in hard if c.passed)
        soft_ok = sum(1 for c in soft if c.passed)
        parts = [f"hard {hard_ok}/{len(hard)}"]
        if soft:
            parts.append(f"soft {soft_ok}/{len(soft)}")
        return ", ".join(parts)

    def as_dict(self) -> dict[str, Any]:
        return {
            "prompt_id": self.prompt_id,
            "auto_pass": self.auto_pass,
            "hard_pass": self.hard_pass,
            "manual_review": self.manual_review,
            "summary": self.summary,
            "output_preview": self.output_preview,
            "checks": [c.as_dict() for c in self.checks],
        }


def check_length(text: str, prompt: QaPrompt) -> CheckResult:
    text = "" if text is None else str(text)
    length = len(text.strip())
    passed = length >= prompt.min_length
    return CheckResult(
        "length",
        passed,
        f"{length} chars (min {prompt.min_length})",
        min(1.0, length / max(1, prompt.min_length)),
        hard=True,
    )


def check_repetition(text: str, prompt: QaPrompt) -> CheckResult:
    if not text.strip():
        return CheckResult("repetition", False, "empty output", 0.0, hard=True)
    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
    if not lines:
        return CheckResult("repetition", False, "no non-empty lines", 0.0, hard=True)
    if len(lines) < 5:
        return CheckResult(
            "repetition", True, f"{len(lines)} non-empty lines", 1.0, hard=True
        )
    counter = Counter(lines)
    most_common_count = counter.most_common(1)[0][1]
    ratio = most_common_count / len(lines)
    passed = ratio <= prompt.max_repetition_ratio
    return CheckResult(
        "repetition",
        passed,
        f"max line repeat ratio {ratio:.2f} (threshold {prompt.max_repetition_ratio})",
        1.0 - ratio,
        hard=True,
    )


def check_keywords(text: str, prompt: QaPrompt) -> CheckResult:
    """Soft coverage signal — does not fail the suite on its own."""
    if not prompt.keywords:
        return CheckResult("keywords", True, "no keywords required", 1.0, hard=False)
    text_lower = text.lower()
    aliases = {
        "music": ["music", "melody", "song", "rhythm", "rhythmic", "harmonic"],
        "robot": ["robot", "unit", "android", "machine"],
        "sound": ["sound", "sonic", "audio", "vibration"],
    }
    found = [
        kw
        for kw in prompt.keywords
        if any(alias in text_lower for alias in aliases.get(kw.lower(), [kw.lower()]))
    ]
    missing = [kw for kw in prompt.keywords if kw not in found]
    ratio = len(found) / len(prompt.keywords)
    # Soft: pass when majority present; still never hard-fails.
    passed = ratio >= 0.5
    detail = f"found {len(found)}/{len(prompt.keywords)}"
    if missing:
        detail += f" (missing: {', '.join(missing[:3])})"
    return CheckResult("keywords", passed, detail, ratio, hard=False)


def check_regex(text: str, prompt: QaPrompt) -> CheckResult:
    if not prompt.regex_patterns:
        return CheckResult("regex", True, "no patterns required", 1.0, hard=True)
    matched = sum(
        1 for pattern in prompt.regex_patterns if re.search(pattern, text, re.IGNORECASE)
    )
    total = len(prompt.regex_patterns)
    ratio = matched / total
    # Hard structural checks require all patterns (was 50% — too loose for gates).
    passed = matched == total
    return CheckResult(
        "regex",
        passed,
        f"matched {matched}/{total} patterns",
        ratio,
        hard=True,
    )


def check_coherence(text: str, prompt: QaPrompt) -> CheckResult:
    if not text.strip():
        return CheckResult("coherence", False, "empty output", 0.0, hard=True)
    # Closed-ended numeric / token answers ("2", "8", "H2O", JSON arrays) are
    # intentionally low-alpha; do not fail them on the prose alpha ratio.
    if getattr(prompt, "exact_answer", None) or prompt.category in {
        "math",
        "json",
        "format",
        "knowledge",
    }:
        stripped = text.strip()
        if len(stripped) <= 64 and not re.search(r"[\u4e00-\u9fff\u0600-\u06ff]", stripped):
            return CheckResult(
                "coherence",
                True,
                f"short closed-ended answer ({len(stripped)} chars)",
                1.0,
                hard=True,
            )
    alpha_chars = sum(1 for c in text if c.isalpha())
    total_chars = len(text)
    if total_chars == 0:
        return CheckResult("coherence", False, "zero length", 0.0, hard=True)
    alpha_ratio = alpha_chars / total_chars
    unique_ratio = len(set(text)) / min(total_chars, 100)
    min_alpha_ratio = 0.2 if prompt.category == "code" else 0.3
    passed = alpha_ratio > min_alpha_ratio and unique_ratio > 0.05
    return CheckResult(
        "coherence",
        passed,
        f"alpha={alpha_ratio:.2f}, unique={unique_ratio:.2f}",
        (alpha_ratio + unique_ratio) / 2,
        hard=True,
    )


def check_unicode_replacement(text: str, prompt: Optional[QaPrompt] = None) -> CheckResult:
    del prompt  # unused; signature stable for call sites
    replacement_count = text.count("\uFFFD")
    if replacement_count == 0:
        return CheckResult("unicode_replacement", True, "none", 1.0, hard=True)
    return CheckResult(
        "unicode_replacement",
        False,
        f"{replacement_count} replacement chars (possible UTF-8 stream corruption)",
        0.0,
        hard=True,
    )


def check_garbage(text: str) -> CheckResult:
    if not text.strip():
        return CheckResult("garbage", True, "empty (no garbage)", 1.0, hard=True)
    scripts: set[str] = set()
    for ch in text[:500]:
        cp = ord(ch)
        if 0x4E00 <= cp <= 0x9FFF:
            scripts.add("CJK")
        elif 0x3040 <= cp <= 0x30FF:
            scripts.add("Japanese")
        elif 0xAC00 <= cp <= 0xD7AF:
            scripts.add("Korean")
        elif 0x0600 <= cp <= 0x06FF:
            scripts.add("Arabic")
        elif 0x0900 <= cp <= 0x097F:
            scripts.add("Devanagari")
        elif 0x0E00 <= cp <= 0x0E7F:
            scripts.add("Thai")
        elif 0x0400 <= cp <= 0x04FF:
            scripts.add("Cyrillic")
        elif 0x0041 <= cp <= 0x024F:
            scripts.add("Latin")
    if len(scripts) >= 4 and "Latin" in scripts:
        return CheckResult(
            "garbage",
            False,
            f"mixed scripts detected: {', '.join(sorted(scripts))}",
            0.2,
            hard=True,
        )
    return CheckResult(
        "garbage",
        True,
        f"scripts: {', '.join(sorted(scripts)) or 'none'}",
        1.0,
        hard=True,
    )


def check_pytest_test_count(text: str, prompt: QaPrompt) -> CheckResult:
    if not prompt.min_test_count:
        return CheckResult("pytest_test_count", True, "not required", 1.0, hard=True)
    found = len(re.findall(r"def test_\w+\s*\(", text))
    passed = found >= prompt.min_test_count
    detail = f"found {found} test function(s) (min {prompt.min_test_count})"
    score = min(1.0, found / max(1, prompt.min_test_count))
    return CheckResult("pytest_test_count", passed, detail, score, hard=True)


def _last_nonempty_line(text: str) -> str:
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    return lines[-1] if lines else ""


def _normalize_spaces(text: str) -> str:
    """Collapse Unicode/NBSP whitespace so exact answers match model variants.

    Models sometimes emit U+00A0 (NBSP) or other separators instead of ASCII
    space (observed: gpt-oss-20b ``red green\\xa0blue``).
    """
    return re.sub(r"\s+", " ", text.replace("\u00a0", " ").replace("\u2007", " ").replace("\u202f", " ")).strip()


def _match_candidate(hay: str, needle: str, mode: str) -> bool:
    """Return True if needle matches hay under the given mode."""
    if not needle:
        return False
    hay_n = _normalize_spaces(hay)
    needle_n = _normalize_spaces(needle)
    if mode == "full":
        return hay_n == needle_n
    if mode == "case":
        # Case-sensitive after space normalize; preserve letter case.
        last = _normalize_spaces(_last_nonempty_line(hay))
        return last == needle_n or hay_n == needle_n
    if mode == "last_line":
        last = _normalize_spaces(_last_nonempty_line(hay))
        return needle_n.lower() in last.lower()
    if mode == "token":
        # Word-boundary match; allow surrounding punctuation on short answers.
        pattern = r"(?<!\w)" + re.escape(needle_n) + r"(?!\w)"
        return re.search(pattern, hay_n, flags=re.IGNORECASE) is not None
    if mode == "substring":
        return needle_n.lower() in hay_n.lower()
    # auto
    stripped_needle = needle_n
    if len(stripped_needle) <= 3 and re.fullmatch(
        r"[A-Za-z0-9/+.-]+", stripped_needle
    ):
        return _match_candidate(hay, stripped_needle, "token")
    last = _normalize_spaces(_last_nonempty_line(hay))
    if stripped_needle.lower() in last.lower():
        return True
    return stripped_needle.lower() in hay_n.lower()


def check_exact_answer(text: str, prompt: QaPrompt) -> CheckResult:
    """Match closed-ended items with safer-than-raw-substring modes."""
    expected = getattr(prompt, "exact_answer", None)
    if not expected:
        return CheckResult("exact_answer", True, "not required", 1.0, hard=True)
    aliases = list(getattr(prompt, "exact_answer_aliases", None) or [])
    candidates = [expected, *aliases]
    mode = getattr(prompt, "exact_match", None) or "auto"
    for cand in candidates:
        needle = str(cand).strip()
        if not needle:
            continue
        if _match_candidate(text, needle, mode):
            return CheckResult(
                "exact_answer",
                True,
                f"found {cand!r} (mode={mode})",
                1.0,
                hard=True,
            )
    return CheckResult(
        "exact_answer",
        False,
        f"missing any of {candidates!r} (mode={mode})",
        0.0,
        hard=True,
    )


def check_invoice_total(text: str, prompt: QaPrompt) -> CheckResult:
    if prompt.json_expected_total is None:
        return CheckResult("invoice_total", True, "not required", 1.0, hard=True)
    expected = prompt.json_expected_total

    # Prefer full JSON objects (code fences, then balanced braces), then a
    # bare `"total": N` match when models wrap JSON in prose or truncate.
    json_candidates = re.findall(r"```(?:json)?\s*([\s\S]*?)```", text)
    json_candidates.extend(re.findall(r"\{[\s\S]*\}", text))
    json_candidates.append(text)

    seen_totals: list[float] = []
    for candidate in json_candidates:
        candidate = candidate.strip()
        if not candidate:
            continue
        try:
            data = json.loads(candidate)
        except (json.JSONDecodeError, ValueError, TypeError):
            continue
        if not isinstance(data, dict) or "total" not in data:
            continue
        actual = float(data["total"])
        seen_totals.append(actual)
        if abs(actual - expected) < 0.001:
            return CheckResult(
                "invoice_total",
                True,
                f"total={actual} matches expected {expected}",
                1.0,
                hard=True,
            )

    for match in re.finditer(
        r'["\']?total["\']?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)', text, flags=re.IGNORECASE
    ):
        actual = float(match.group(1))
        seen_totals.append(actual)
        if abs(actual - expected) < 0.001:
            return CheckResult(
                "invoice_total",
                True,
                f"total={actual} matches expected {expected} (loose field match)",
                1.0,
                hard=True,
            )

    if seen_totals:
        return CheckResult(
            "invoice_total",
            False,
            f"total={seen_totals[-1]} != expected {expected} (wrong arithmetic)",
            0.0,
            hard=True,
        )

    return CheckResult(
        "invoice_total",
        False,
        f"no JSON with 'total' field found (expected {expected})",
        0.0,
        hard=True,
    )


def run_all_checks(text: str, prompt: QaPrompt) -> QualityReport:
    text = "" if text is None else str(text)
    checks = [
        check_length(text, prompt),
        check_repetition(text, prompt),
        check_keywords(text, prompt),
        check_regex(text, prompt),
        check_coherence(text, prompt),
        check_unicode_replacement(text, prompt),
        check_garbage(text),
        check_pytest_test_count(text, prompt),
        check_invoice_total(text, prompt),
        check_exact_answer(text, prompt),
    ]
    hard_ok = all(c.passed for c in checks if c.hard)
    report = QualityReport(
        prompt_id=prompt.id,
        checks=checks,
        auto_pass=hard_ok,
        manual_review=not hard_ok,
        output_preview=text[:500],
    )
    return report
