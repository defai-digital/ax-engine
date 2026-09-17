"""Read explicitly supplied ds4 question data without executing reference code.

Question data stays external to this repository. This parser accepts only the
known, data-only designated-initializer shape and fails on unknown fields.
"""

from __future__ import annotations

import ast
import hashlib
import re
from pathlib import Path

STRING = r'"(?:\\.|[^"\\])*"'
FIELD = re.compile(r"\.(\w+)(?:\[(\d+)\])?\s*=\s*((?:" + STRING + r"\s*)+|[^,]+),")
KINDS = {"CHOICE", "INTEGER", "RATIONAL", "EXACT_TEXT", "ORDERED_SEQUENCE", "LINE_SET"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_table(text: str, name: str, suite: str) -> list[dict]:
    match = re.search(r"\b" + re.escape(name) + r"\[\]\s*=\s*\{(.*?)\n\};", text, re.S)
    if not match:
        raise ValueError(f"Missing data table: {name}")
    record_pattern = re.compile(r"^    \{\n(.*?)^    \},", re.M | re.S)
    records = record_pattern.findall(match[1])
    if record_pattern.sub("", match[1]).strip():
        raise ValueError("Unparsed table data")
    cases = []
    for record in records:
        data, choices, aliases = {}, {}, {}
        for field in FIELD.finditer(record):
            key, index, value = field.groups()
            if key not in {
                "source",
                "id",
                "domain",
                "title",
                "question",
                "answer",
                "choice",
                "alias",
                "answer_kind",
                "suites",
                "max_tokens",
            }:
                raise ValueError(f"Unsupported data field: {key}")
            if value.startswith('"'):
                # C octal escapes in these fixtures encode UTF-8 bytes.
                value = "".join(ast.literal_eval(t) for t in re.findall(STRING, value))
                value = value.encode("latin1").decode("utf-8")
            else:
                value = value.strip()
            target = choices if key == "choice" else aliases if key == "alias" else data
            slot = int(index) if index is not None else key
            if slot in target:
                raise ValueError(f"Duplicate field: {key}")
            target[slot] = value
        if FIELD.sub("", record).strip():
            raise ValueError("Unparsed initializer data")
        for required in ("source", "id", "domain", "question", "answer"):
            if not data.get(required):
                raise ValueError(f"Missing {required}")
        if sorted(choices) != list(range(len(choices))):
            raise ValueError("Non-contiguous choices")
        inferred = "CHOICE" if choices else "LINE_SET" if data["source"] == "COMPSEC" else "INTEGER"
        kind = data.get("answer_kind", "EVAL_ANSWER_" + inferred).removeprefix("EVAL_ANSWER_")
        if kind not in KINDS:
            raise ValueError(f"Unsupported answer kind: {kind}")
        cases.append(
            {
                **data,
                "key": data["source"] + ":" + data["id"],
                "kind": kind,
                "choices": [choices[i] for i in sorted(choices)],
                "aliases": [aliases[i] for i in sorted(aliases)],
                "suite": suite,
                "hard_smoke": "EVAL_SUITE_HARD_SMOKE" in data.get("suites", ""),
                "source_max_tokens": int(data.get("max_tokens", 16000)),
            }
        )
    if not cases:
        raise ValueError("Empty question table")
    return cases


def load_questions(root: Path) -> dict:
    files = {
        "ds4_eval.c": ("eval_core_cases", "core", 92),
        "ds4_eval_cases.c": ("eval_hard_cases", "hard", 50),
    }
    cases = []
    hashes = {}
    for filename, (table, suite, count) in files.items():
        path = root / filename
        rows = parse_table(path.read_text(), table, suite)
        if len(rows) != count:
            raise ValueError(f"{filename}: expected {count}, got {len(rows)}")
        cases.extend(rows)
        hashes[filename] = sha256(path)
    if len({row["key"] for row in cases}) != len(cases):
        raise ValueError("Duplicate question keys")
    if sum(row["hard_smoke"] for row in cases) != 12:
        raise ValueError("Expected exactly 12 hard-smoke questions")
    return {"schema": "ax.external_questions.v1", "source_hashes": hashes, "cases": cases}
