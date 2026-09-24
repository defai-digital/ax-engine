#!/usr/bin/env python3
"""Consistency check for a frozen Flash Next full-QA adjudication record.

A frozen full-QA validation stays ``qualification: false`` while any hard
check fails; that verdict is pinned evidence and is never rewritten (the
frozen replayer in benchmarks/results/qualification/2026-09-19-flash-next-
mxfp4-installed-qa rejects verdict regrading). This script checks a separate
record without closing a qualification gate. Every retained failure must be
covered by a sha-bound entry documenting its disposition and the absence of
an established runtime cause. Identity and accounting invariants must hold.

Pass conditions (all must hold):

  * ``completed`` and both identity flags (``text_identity``,
    ``checker_identity``) are true, and ``normal_stops`` equals
    ``2 * cases_per_mode`` (both modes ran every case to a normal stop);
  * exactly the disabled and required modes exist; each has unique failed
    IDs and ``hard_pass + len(failed_ids) == cases_per_mode``;
  * the union of ``failed_ids`` across modes is exactly the set of
    adjudicated case ids (no uncovered failure, no adjudicated pass);
  * ``adjudication.source_qa.sha256 == validation.raw_sha256`` and
    ``adjudication.source_items.sha256`` equals the expected frozen-items
    hash; both bindings require valid SHA-256 values;
  * every adjudicated entry has a non-empty disposition,
    ``runtime_cause_established`` false (a runtime defect may never be
    retained as a model failure), and the adjudication itself keeps
    ``threshold_changed``, ``qualification`` and ``release_ready`` false.

Exit 0 iff record consistency is ``passed`` or ``passed_with_retained``.
This does not replay raw results, establish a model-only cause, change frozen
quality grades, or qualify a target. Those require independent evidence.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_VALIDATION = (
    "benchmarks/results/flash-next-installed-full-qa-4bit-c101-m2-20260917.json"
)
DEFAULT_ADJUDICATION = (
    "benchmarks/results/"
    "flash-next-installed-full-qa-4bit-c101-m2-20260917.adjudication.json"
)
DEFAULT_FROZEN_ITEMS = (
    "benchmarks/results/qualification/"
    "2026-09-19-flash-next-mxfp4-installed-qa/full-qa-items.json"
)


def adjudicated_verdict(
    validation: dict[str, Any],
    adjudication: dict[str, Any],
    *,
    frozen_items_sha256: str | None = None,
) -> dict[str, Any]:
    """Compute the adjudicated verdict for one frozen full-QA validation."""
    problems: list[str] = []
    if not isinstance(validation, dict) or not isinstance(adjudication, dict):
        return {"verdict": "failed", "problems": ["records must be JSON objects"],
                "cases_per_mode": 0, "retained_failures": [],
                "qualification": False, "release_ready": False}

    if validation.get("schema") != "ax-engine.flash-next.installed-full-qa.v1":
        problems.append("unknown validation schema")
    if adjudication.get("schema") != "ax-engine.flash-next.full-qa-adjudication.v1":
        problems.append("unknown adjudication schema")
    for key in ("completed", "text_identity", "checker_identity"):
        if validation.get(key) is not True:
            problems.append(f"validation {key} must be true")
    for label, record, keys in (
        ("validation", validation, ("qualification", "release_ready")),
        ("adjudication", adjudication, ("threshold_changed", "qualification", "release_ready")),
    ):
        for key in keys:
            if record.get(key) is not False:
                problems.append(f"{label} {key} must be explicitly false")

    cases_per_mode = validation.get("cases_per_mode")
    if type(cases_per_mode) is not int or cases_per_mode <= 0:
        problems.append("cases_per_mode missing or not positive")
        cases_per_mode = 0
    stops = validation.get("normal_stops")
    if type(stops) is not int or stops != 2 * cases_per_mode:
        problems.append("normal_stops must equal 2 * cases_per_mode")

    modes = validation.get("modes")
    if not isinstance(modes, dict) or set(modes) != {"disabled", "required"}:
        problems.append("validation must contain exactly disabled and required modes")
        modes = modes if isinstance(modes, dict) else {}
    failed_union: set[str] = set()
    failed_by_mode: dict[str, set[str]] = {}
    for mode, stats in modes.items():
        if not isinstance(stats, dict):
            problems.append(f"mode {mode} is not an object")
            continue
        hard_pass, failed_ids = stats.get("hard_pass"), stats.get("failed_ids")
        if (type(hard_pass) is not int or hard_pass < 0
                or not isinstance(failed_ids, list)
                or any(not isinstance(fid, str) or not fid.strip() for fid in failed_ids)):
            problems.append(f"mode {mode} has malformed hard_pass/failed_ids")
            continue
        if len(failed_ids) != len(set(failed_ids)):
            problems.append(f"mode {mode} has duplicate failed ids")
        if hard_pass + len(failed_ids) != cases_per_mode:
            problems.append(f"mode {mode} accounting does not close")
        failed_union.update(failed_ids)
        failed_by_mode[mode] = set(failed_ids)
    if (set(failed_by_mode) == {"disabled", "required"}
            and failed_by_mode["disabled"] != failed_by_mode["required"]):
        problems.append("mode failures contradict checker_identity")

    def valid_sha(value: Any) -> bool:
        return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None

    raw_sha = validation.get("raw_sha256")
    source_qa = adjudication.get("source_qa")
    if (not valid_sha(raw_sha) or not isinstance(source_qa, dict)
            or source_qa.get("sha256") != raw_sha):
        problems.append("source_qa must bind a valid validation raw_sha256")
    source_items = adjudication.get("source_items")
    if (not valid_sha(frozen_items_sha256) or not isinstance(source_items, dict)
            or source_items.get("sha256") != frozen_items_sha256):
        problems.append("source_items must bind a valid expected frozen cohort sha256")

    cases = adjudication.get("cases")
    if not isinstance(cases, list):
        problems.append("adjudication cases is not a list")
        cases = []
    by_id: dict[str, dict[str, Any]] = {}
    for case in cases:
        if (not isinstance(case, dict) or not isinstance(case.get("id"), str)
                or not case["id"].strip()):
            problems.append("adjudication has a case without a string id")
            continue
        if case["id"] in by_id:
            problems.append(f"duplicate adjudication id {case['id']}")
        by_id[case["id"]] = case

    for failed_id in sorted(failed_union):
        entry = by_id.get(failed_id)
        if entry is None:
            problems.append(f"failed id {failed_id} has no adjudication entry")
            continue
        if entry.get("runtime_cause_established") is not False:
            problems.append(f"{failed_id}: runtime cause must be explicitly false")
        disposition = entry.get("disposition")
        if not isinstance(disposition, str) or not disposition.strip():
            problems.append(f"{failed_id}: adjudication entry lacks a disposition")
    extra = sorted(set(by_id) - failed_union)
    if extra:
        problems.append(f"adjudication covers ids that did not fail: {', '.join(extra)}")

    if problems:
        verdict = "failed"
    elif failed_union:
        verdict = "passed_with_retained"
    else:
        verdict = "passed"
    return {
        "verdict": verdict,
        "problems": problems,
        "cases_per_mode": cases_per_mode,
        "retained_failures": sorted(failed_union),
        "qualification": False,
        "release_ready": False,
    }


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validation",
        default=str(REPO_ROOT / DEFAULT_VALIDATION),
        help="frozen full-QA validation JSON",
    )
    parser.add_argument(
        "--adjudication",
        default=str(REPO_ROOT / DEFAULT_ADJUDICATION),
        help="adjudication record JSON",
    )
    parser.add_argument(
        "--frozen-items-sha",
        default=None,
        help=(
            "expected sha256 of the frozen QA items cohort; defaults to hashing "
            "the tracked frozen cohort next to the validation record"
        ),
    )
    parser.add_argument("--json", action="store_true", help="print JSON verdict")
    args = parser.parse_args()

    validation_path = Path(args.validation)
    adjudication_path = Path(args.adjudication)
    if not validation_path.is_file():
        print(f"validation not found: {validation_path}", file=sys.stderr)
        return 2
    if not adjudication_path.is_file():
        print(f"adjudication not found: {adjudication_path}", file=sys.stderr)
        return 2
    try:
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        adjudication = json.loads(adjudication_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        print(f"cannot read adjudication inputs: {error}", file=sys.stderr)
        return 2

    frozen_sha = args.frozen_items_sha
    if frozen_sha is None:
        default_items = REPO_ROOT / DEFAULT_FROZEN_ITEMS
        if default_items.is_file():
            frozen_sha = _sha256(default_items)

    result = adjudicated_verdict(
        validation,
        adjudication,
        frozen_items_sha256=frozen_sha,
    )
    if args.json:
        json.dump(result, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        print(f"verdict: {result['verdict']}")
        for problem in result["problems"]:
            print(f"FAIL: {problem}")
        if result["retained_failures"]:
            print(f"retained failures: {', '.join(result['retained_failures'])}")
    return 0 if result["verdict"] in ("passed", "passed_with_retained") else 1


if __name__ == "__main__":
    sys.exit(main())
