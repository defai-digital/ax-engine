"""Quality checkers for AX Engine QA."""
import re
from collections import Counter
from dataclasses import dataclass, field

from prompts import QaPrompt


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""
    score: float = 0.0


@dataclass
class QualityReport:
    prompt_id: str
    checks: list = field(default_factory=list)
    auto_pass: bool = False
    manual_review: bool = False
    output_preview: str = ""

    @property
    def passed_count(self):
        return sum(1 for c in self.checks if c.passed)

    @property
    def total_count(self):
        return len(self.checks)

    @property
    def summary(self):
        if not self.checks:
            return "no checks"
        return f"{self.passed_count}/{self.total_count} passed"


def check_length(text, prompt):
    length = len(text.strip())
    passed = length >= prompt.min_length
    return CheckResult("length", passed, f"{length} chars (min {prompt.min_length})",
                       min(1.0, length / max(1, prompt.min_length)))


def check_repetition(text, prompt):
    if not text.strip():
        return CheckResult("repetition", False, "empty output", 0.0)
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    if not lines:
        return CheckResult("repetition", False, "no non-empty lines", 0.0)
    if len(lines) < 5:
        return CheckResult("repetition", True, f"{len(lines)} non-empty lines", 1.0)
    counter = Counter(lines)
    most_common_count = counter.most_common(1)[0][1]
    ratio = most_common_count / len(lines)
    passed = ratio <= prompt.max_repetition_ratio
    return CheckResult("repetition", passed,
                       f"max line repeat ratio {ratio:.2f} (threshold {prompt.max_repetition_ratio})",
                       1.0 - ratio)


def check_keywords(text, prompt):
    if not prompt.keywords:
        return CheckResult("keywords", True, "no keywords required", 1.0)
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
    passed = ratio >= 0.5
    detail = f"found {len(found)}/{len(prompt.keywords)}"
    if missing:
        detail += f" (missing: {', '.join(missing[:3])})"
    return CheckResult("keywords", passed, detail, ratio)


def check_regex(text, prompt):
    if not prompt.regex_patterns:
        return CheckResult("regex", True, "no patterns required", 1.0)
    matched = sum(1 for p in prompt.regex_patterns if re.search(p, text, re.IGNORECASE))
    ratio = matched / len(prompt.regex_patterns)
    return CheckResult("regex", ratio >= 0.5,
                       f"matched {matched}/{len(prompt.regex_patterns)} patterns", ratio)


def check_coherence(text, prompt):
    if not text.strip():
        return CheckResult("coherence", False, "empty output", 0.0)
    alpha_chars = sum(1 for c in text if c.isalpha())
    total_chars = len(text)
    if total_chars == 0:
        return CheckResult("coherence", False, "zero length", 0.0)
    alpha_ratio = alpha_chars / total_chars
    unique_ratio = len(set(text)) / min(total_chars, 100)
    min_alpha_ratio = 0.2 if prompt.category == "code" else 0.3
    passed = alpha_ratio > min_alpha_ratio and unique_ratio > 0.05
    return CheckResult("coherence", passed,
                       f"alpha={alpha_ratio:.2f}, unique={unique_ratio:.2f}",
                       (alpha_ratio + unique_ratio) / 2)


def check_garbage(text):
    if not text.strip():
        return CheckResult("garbage", True, "empty (no garbage)", 1.0)
    scripts = set()
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
        return CheckResult("garbage", False,
                           f"mixed scripts detected: {', '.join(sorted(scripts))}", 0.2)
    return CheckResult("garbage", True, f"scripts: {', '.join(sorted(scripts)) or 'none'}", 1.0)


def run_all_checks(text, prompt):
    checks = [
        check_length(text, prompt),
        check_repetition(text, prompt),
        check_keywords(text, prompt),
        check_regex(text, prompt),
        check_coherence(text, prompt),
        check_garbage(text),
    ]
    report = QualityReport(
        prompt_id=prompt.id,
        checks=checks,
        auto_pass=all(c.passed for c in checks),
        manual_review=not all(c.passed for c in checks),
        output_preview=text[:500],
    )
    return report
