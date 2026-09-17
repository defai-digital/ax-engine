#!/usr/bin/env python3
"""Recompute saved native QA retest statistics without weights or a server.

Input: pack directories with numbered rows, questions.json and manifest.json,
plus <pack>_provenance.json. Output hashes identify the exact evidence read;
they do not authenticate a binary or recover missing source provenance.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "qa"))
from run_ds4_qa import grade  # noqa: E402


def audit(root: Path, packs: list[str]) -> dict:
    hashes = {}

    def read(path: Path):
        data = path.read_bytes()
        hashes[path.relative_to(root).as_posix()] = hashlib.sha256(data).hexdigest()
        return json.loads(data)

    result = {}
    paired_keys = None
    for pack in packs:
        folder = root / pack
        manifest = read(folder / "manifest.json")
        questions = read(folder / "questions.json")["cases"]
        cases = {case["key"]: case for case in questions}
        if len(cases) != len(questions):
            raise ValueError(f"{pack}: duplicate question keys")
        rows = [read(p) for p in sorted(folder.glob("[0-9][0-9][0-9].json"))]
        keys = [row["key"] for row in rows]
        if not rows or len(set(keys)) != len(keys) or keys != manifest["question_order"]:
            raise ValueError(f"{pack}: missing, duplicate or misordered rows")
        if set(keys) != set(cases):
            raise ValueError(f"{pack}: row/question identity mismatch")
        if paired_keys is not None and keys != paired_keys:
            raise ValueError("Packs do not contain the same ordered cases")
        paired_keys = keys
        primary, final, recovery = Counter(), Counter(), Counter()
        line_grades = []
        line_replies = []
        for row in rows:
            case = cases[row["key"]]
            if (row["kind"], row["answer"]) != (case["kind"], case["answer"]):
                raise ValueError(f"{pack}: row answer key differs from question bank")
            # Recovery replaces decoded_response in the campaign runner. The
            # original primary grade survives, but its decoded text may not.
            first = row.get("primary_grade", row["grade"])
            last = grade(case, row["decoded_response"])
            if "recovery_decoded_response" in row:
                recovered = grade(case, row["recovery_decoded_response"])
                recovery[recovered["status"]] += 1
            if any(last[k] != row["grade"][k] for k in ("status", "answer")):
                raise ValueError(f"{pack}: saved grade differs from decoded reply")
            primary[first["status"]] += 1
            final[last["status"]] += 1
            if case["kind"] == "LINE_SET":
                line_grades.append(last)
                line_replies.append(row["decoded_response"]["choices"][0]["message"]["content"])
        recalls = [g["span_recall"] for g in line_grades if g.get("span_recall") is not None]
        spans = [g["span_expected"] for g in line_grades if g.get("span_expected") is not None]
        provenance = read(root / f"{pack}_provenance.json")
        result[pack] = {
            "rows": len(rows),
            "primary": dict(primary),
            "final": dict(final),
            "recovery": dict(recovery),
            "finish_reason": dict(Counter(r["response"]["finish_reason"] for r in rows)),
            "output_cap_hits": sum(
                len(r["response"]["output_tokens"]) == manifest["max_tokens"] for r in rows
            ),
            "loop_suspect": sum(bool(r.get("loop_suspect")) for r in rows),
            "repetition_max_run": {
                "median": statistics.median(r.get("repetition_max_run", 0) for r in rows),
                "max": max(r.get("repetition_max_run", 0) for r in rows),
            },
            "line_set": {
                "rows": len(line_grades),
                "graded": sum(g["status"] in ("correct", "wrong") for g in line_grades),
                "detected": sum(g.get("detection", False) for g in line_grades),
                "single_line": sum(g.get("span_reported") == 1 for g in line_grades),
                "full_recall": sum(value == 1 for value in recalls),
                "answers_with_comma": sum("," in (g.get("answer") or "") for g in line_grades),
                "replies_without_exactly_one_answer_line": sum(
                    sum(line.strip().startswith("Answer:") for line in reply.splitlines()) != 1
                    for reply in line_replies
                ),
                "gold_span_min": min(spans) if spans else None,
                "gold_span_max": max(spans) if spans else None,
                "gold_span_median": statistics.median(spans) if spans else None,
                "recall_denominator": len(recalls),
                "recall_mean": statistics.mean(recalls) if recalls else None,
                "recall_median": statistics.median(recalls) if recalls else None,
            },
            "provenance": {
                key: provenance.get(key)
                for key in (
                    "source_commit",
                    "binary_sha256",
                    "model_repo",
                    "model_revision",
                    "host_chip",
                    "host_memory_bytes",
                )
            },
            "source_binding": "unverified" if provenance.get("source_commit") else "unknown",
        }
    return {
        "schema": 1,
        "packs": result,
        "input_sha256": hashes,
        "limitations": [
            "Selected retest subset; not overall model accuracy.",
            "Final decoded responses regraded; raw tokens not independently decoded.",
            "Primary grades retained from records; recovery can replace decoded text.",
            "Recorded provenance is not proof of the binary's source tree.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--packs", nargs="+", default=["axq6", "mxfp4"])
    args = parser.parse_args()
    print(json.dumps(audit(args.root, args.packs), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
