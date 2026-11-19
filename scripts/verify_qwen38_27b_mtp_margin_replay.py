#!/usr/bin/env python3
"""Verify paired Qwen 3.8 27B MXFP4 logit margins against measured token pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
from collections import Counter
from pathlib import Path
from typing import Any

MANIFEST_SHA256 = "524163237ca2b879bff41f782a5eee6a1266b4f80f5d2c99e07117b7b899127a"
MTP_SOURCES = {"target", "verifier", "rollback_replay"}
RUST_PASS = re.compile(r"^test result: ok\. [1-9]\d* passed; 0 failed;", re.M)


def require(condition: bool, detail: str) -> None:
    if not condition:
        raise ValueError(detail)


def token_ids(value: Any) -> list[int]:
    require(
        isinstance(value, list) and all(type(token) is int and 0 <= token < 2**32 for token in value),
        "invalid token IDs",
    )
    return value


def verify_scores(branch: dict[str, Any], sources: set[str]) -> tuple[float, str]:
    require(isinstance(branch, dict) and set(branch) == {"Ok"}, "missing logit row")
    row = branch["Ok"]
    source = row["source"]
    margin = row["paired_gap"]
    runner_up_margin = row["runner_up_gap"]
    emitted = row["emitted_logit"]
    rival = row["rival_logit"]
    require(source in sources, f"invalid logit row source {source!r}")
    require(
        all(type(value) in (int, float) and math.isfinite(value)
            for value in (margin, runner_up_margin, emitted, rival)),
        "nonfinite or nonnumeric logit or margin",
    )
    # Rust serializes f32 scores and subtracts their exact f64 conversions.
    emitted_f32, rival_f32 = struct.unpack("!ff", struct.pack("!ff", emitted, rival))
    require(
        0 <= runner_up_margin <= margin and margin == emitted_f32 - rival_f32,
        "invalid paired logit margin",
    )
    return margin, source


def verify_case(case: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    prompt = token_ids(case["prompt_tokens"])
    direct = token_ids(case["direct_tokens"])
    mtp = token_ids(case["mtp_tokens"])
    budget = case["max_output_tokens"]
    ignore_eos = case["ignore_eos"]
    require(
        bool(prompt) and type(budget) is int and 0 < budget < 2**32
        and type(ignore_eos) is bool and 0 < len(direct) <= budget and 0 < len(mtp) <= budget,
        "invalid prompt or generation budget",
    )
    if ignore_eos:
        require(len(direct) == len(mtp) == budget, "fixed-budget output was truncated")
    require(
        report["case_id"] == case["case_id"]
        and token_ids(report["prompt_tokens"]) == prompt
        and type(report["max_output_tokens"]) is int and report["max_output_tokens"] == budget
        and report["ignore_eos"] is ignore_eos
        and report["manifest_sha256"] == MANIFEST_SHA256
        and token_ids(report["direct"]["tokens"]) == direct
        and token_ids(report["mtp"]["tokens"]) == mtp
        and report["bound_to_recorded_outputs"] is True
        and report["complete_aligned_margins"] is True,
        "replay does not match the measured request, output, or model",
    )
    first_unpaired = min(len(direct), len(mtp)) if len(direct) != len(mtp) else None
    require(report["first_unpaired_index"] == first_unpaired, "incorrect length divergence")
    indices = [index for index, (left, right) in enumerate(zip(direct, mtp)) if left != right]
    differences = report["differences"]
    require(
        [item["index"] for item in differences] == indices,
        "incorrect or incomplete aligned divergence list",
    )
    sources: Counter[str] = Counter()
    margins = []
    for occurrence, (index, item) in enumerate(zip(indices, differences, strict=True)):
        require(
            item["direct_token"] == direct[index] and item["mtp_token"] == mtp[index]
            and item["same_prior_tokens"] is (occurrence == 0),
            f"incorrect token attribution at output index {index}",
        )
        left, left_source = verify_scores(item["direct_row"], {"target"})
        right, right_source = verify_scores(item["mtp_row"], MTP_SOURCES)
        paired = max(left, right)
        require(
            item["max_paired_logit_gap"] == paired,
            f"incorrect maximum margin at output index {index}",
        )
        margins.append(paired)
        sources.update((f"direct:{left_source}", f"mtp:{right_source}"))
    return {
        "case_id": case["case_id"],
        "paired_exact": direct == mtp,
        "first_differing_index": indices[0] if indices else first_unpaired,
        "first_difference_same_prefix_gap": margins[0] if margins else None,
        "largest_aligned_gap": max(margins) if margins else None,
        "aligned_differences": len(indices),
        "first_unpaired_index": first_unpaired,
        "row_sources": dict(sources),
    }


def reports_from_log(log: str) -> list[dict[str, Any]]:
    require(RUST_PASS.search(log) is not None, "missing successful Rust test result")
    reports = []
    for line in log.splitlines():
        if line.startswith("test ") and " ... {" in line:
            line = line.split(" ... ", 1)[1]
        if line.startswith("{"):
            event = json.loads(line)
            if "case_id" in event:
                reports.append(event)
    return reports


def verify(
    cases_bytes: bytes, report_bytes: bytes, *, rust_log: bool
) -> dict[str, Any]:
    cases = json.loads(cases_bytes)
    case_ids = [case["case_id"] for case in cases]
    require(
        bool(case_ids) and all(isinstance(case_id, str) and case_id for case_id in case_ids)
        and len(set(case_ids)) == len(case_ids),
        "empty or duplicate replay case IDs",
    )
    if rust_log:
        reports = reports_from_log(report_bytes.decode())
    else:
        parsed = json.loads(report_bytes)
        reports = [parsed] if isinstance(parsed, dict) else parsed
    require(len(cases) == len(reports), "wrong number of completed replays")
    verified = []
    for case, report in zip(cases, reports, strict=True):
        try:
            verified.append(verify_case(case, report))
        except (KeyError, OverflowError, TypeError, ValueError) as error:
            raise ValueError(f"{case['case_id']}: {error}") from error
    artifact_digest = "raw_replay_log_sha256" if rust_log else "replay_report_sha256"
    return {
        "schema": 1,
        "replay_input_sha256": hashlib.sha256(cases_bytes).hexdigest(),
        artifact_digest: hashlib.sha256(report_bytes).hexdigest(),
        "cases_verified": len(verified),
        "paired_exact_count": sum(row["paired_exact"] for row in verified),
        "aligned_differences_scored": sum(row["aligned_differences"] for row in verified),
        "cases": verified,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path, help="JSON input cases with both measured outputs")
    report = parser.add_mutually_exclusive_group(required=True)
    report.add_argument("--rust-log", type=Path, help="Successful native replay test output")
    report.add_argument("--report-json", type=Path, help="Curated replay report or array of reports")
    args = parser.parse_args()
    path = args.rust_log or args.report_json
    try:
        result = verify(args.cases.read_bytes(), path.read_bytes(), rust_log=args.rust_log is not None)
    except (KeyError, OSError, OverflowError, TypeError, ValueError) as error:
        parser.exit(1, f"Invalid replay evidence: {error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
