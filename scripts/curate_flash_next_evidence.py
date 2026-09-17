#!/usr/bin/env python3
"""Preserve Flash Next development evidence without promoting partial runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sanitize(value):
    if isinstance(value, dict):
        return {sanitize(k): sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize(v) for v in value]
    if isinstance(value, str) and re.search(
        r"/Users/|/Volumes/|192\.168\.|(?:df|um|tn)-mac", value
    ):
        return "[private location " + hashlib.sha256(value.encode()).hexdigest()[:16] + "]"
    return value


def summarize(kind: str, raw: dict) -> dict:
    require(raw.get("qualification") is False, "expected development evidence")
    require(not raw.get("fabricated"), "fabricated evidence is not publishable")
    require(bool(re.fullmatch(r"[0-9a-f]{64}", raw.get("binary_sha256", ""))),
            "missing binary identity")
    if kind == "qa":
        sections = {**raw["modes"], "reference": raw["reference"]}
        require(set(sections) == {"disabled", "required", "reference"}, "missing QA route")
        indexed = {}
        for route, section in sections.items():
            cases = section["cases"]
            require(section.get("completed") is True, "incomplete QA route")
            indexed[route] = {case["id"]: case for case in cases}
            require(len(cases) == len(indexed[route]) > 0, "duplicate or empty QA cases")
        ids = set(indexed["disabled"])
        require(all(set(cases) == ids for cases in indexed.values()), "QA coverage differs")
        mismatches = sorted(i for i in ids if indexed["disabled"][i]["text"]
                            != indexed["required"][i]["text"])
        ppl = raw["perplexity"]
        require(ppl["scored_tokens"] == ppl["reference_scored_tokens"] > 0,
                "perplexity coverage differs")
        a, b = ppl["ax_direct_mean_nll"], ppl["reference_mean_nll"]
        require(all(math.isfinite(v) and v > 0 for v in (a, b)), "invalid NLL")
        return {
            "completed": raw.get("completed") is True,
            "items_per_route": len(ids),
            "hard_pass": {r: sum(c.get("hard_pass") is True for c in cs.values())
                          for r, cs in indexed.items()},
            "direct_mtp_mismatch_ids": mismatches,
            "direct_mtp_text_identity": not mismatches,
            "nll_within_two_percent": a <= b * 1.02,
            "recorded_acceptance": raw["acceptance"],
            "passed": raw.get("completed") is True and not mismatches and a <= b * 1.02,
        }
    if kind == "head":
        for name, permuted in (("real_head", False), ("permuted_head", True)):
            side = raw[name]
            require(side["head_permuted"] is permuted, "incorrect oracle control")
            require(0 <= side["accepted"] <= side["proposed"] and side["proposed"] > 0,
                    "invalid oracle counters")
            require(side["requests"] and side["greedy_identity"] is True,
                    "missing oracle requests or greedy identity")
            require(sum(r["proposed"] for r in side["requests"] if not r.get("too_short")) == side["proposed"],
                    "oracle proposal totals differ")
            require(sum(r["accepted"] for r in side["requests"] if not r.get("too_short")) == side["accepted"],
                    "oracle acceptance totals differ")
        real = raw["real_head"]["accepted"] / raw["real_head"]["proposed"]
        permuted = raw["permuted_head"]["accepted"] / raw["permuted_head"]["proposed"]
        threshold = raw["thresholds"]
        passed = real >= threshold["real_min"] and permuted < threshold["permuted_max"]
        require(raw["falsification_ok"] is passed, "oracle verdict contradicts counters")
        return {"completed": raw.get("completed") is True,
                "real_acceptance": real, "permuted_acceptance": permuted,
                "recorded_thresholds": threshold,
                "scored_requests": {name: sum(not r.get("too_short") for r in raw[name]["requests"])
                                    for name in ("real_head", "permuted_head")},
                "passed": passed and raw.get("completed") is True}
    if kind == "throughput":
        cfg = raw["config"]
        expected = {f"{p}/{n}/{r}" for p in cfg["packs"]
                    for n in cfg["prompt_tokens"] for r in cfg["routes"]}
        cells = {c["cell_id"]: c for c in raw["cells"]}
        require(len(cells) == len(raw["cells"]), "duplicate throughput cell")
        require(set(cells) <= expected, "unexpected throughput cell")
        matrix = []
        for name in sorted(expected):
            cell = cells.get(name)
            samples = [] if cell is None else [s for s in cell.get("samples", [])
                                              if s.get("warmup") is False]
            warmups = [] if cell is None else [s for s in cell.get("samples", [])
                                              if s.get("warmup") is True]
            complete = bool(cell and not cell.get("failed") and not cell.get("skipped")) and (
                len(warmups) == cfg.get("warmup_repetitions", 0)
                and len(samples) == cfg["measurement_repetitions"]
            ) and all(
                (s.get("done") is True or (cell["route"] == "reference" and "done" not in s))
                and s.get("generated_tokens") == cfg["generation_tokens"]
                for s in samples
            )
            matrix.append({"cell_id": name, "status": "complete" if complete else
                           "missing" if cell is None else "incomplete_output_or_samples",
                           "warmup_samples": len(warmups),
                           "measured_samples": len(samples),
                           "generated_tokens": [s.get("generated_tokens") for s in samples]})
        complete = all(c["status"] == "complete" for c in matrix)
        return {"completed": complete, "recorded_completed": raw.get("completed"),
                "coverage_complete": raw.get("completed") is True and set(cells) == expected,
                "attempted_cells": len(cells), "expected_cells": len(expected),
                "complete_cells": sum(c["status"] == "complete" for c in matrix),
                "matrix": matrix, "passed": complete and raw.get("completed") is True,
                "note": "Early EOS and skipped cells are not fixed-decode throughput samples.",
                "memory_boundaries": (
                    "AX peak_rss_bytes is a post-request RSS snapshot, not a sampled peak. "
                    "AX MLX peak is server-lifetime high water; the reference resets its "
                    "MLX peak per request. These are not equivalent peak measurements."
                )}
    if kind == "mtp":
        rows = []
        for pack in ("2bit", "4bit", "6bit"):
            entry = raw.get("packs", {}).get(pack, {})
            for prompt in ("primary", "tie"):
                records = entry if prompt == "primary" else entry.get("tie_prompt") or {}
                for route in ("state", "runner"):
                    record = records.get(route)
                    passed = bool(record and not record.get("failed")
                                  and record.get("identity_until_first_tie") is True
                                  and record.get("within_tolerance") is True)
                    rows.append({"pack": pack, "prompt": prompt, "route": route,
                                 "status": "pass" if passed else "failed" if record else "missing",
                                 "failure": (record or {}).get("failure_log_tail") or
                                            (record or {}).get("validation_error")})
        complete = raw.get("completed") is True and all(r["status"] != "missing" for r in rows)
        return {"completed": complete, "matrix": rows,
                "passed": complete and all(r["status"] == "pass" for r in rows)}
    if kind == "http":
        expected = {(p, m) for p in ("2bit", "4bit", "6bit")
                    for m in ("disabled", "required")}
        cells = {(c["pack"], c["mode"]): c for c in raw["cells"]}
        require(len(cells) == len(raw["cells"]), "duplicate HTTP cell")
        require(set(cells) <= expected, "unexpected HTTP cell")
        rows = []
        for pack, mode in sorted(expected):
            cell = cells.get((pack, mode))
            checks = {} if cell is None else {
                "recorded_pass": cell.get("passed") is True,
                "clean_shutdown": cell.get("exit_code") == 0,
                "metadata_unchanged": cell.get("pack_metadata_unchanged") is True,
                "both_requests": len(cell.get("requests", [])) == 2,
            }
            for key in ("repeat_identity", "direct_identity"):
                if cell is not None and key in cell:
                    checks[key] = cell[key] is True
            passed = bool(cell and all(checks.values()))
            rows.append({"pack": pack, "mode": mode,
                         "status": "pass" if passed else "failed" if cell else "missing",
                         "error": (cell or {}).get("error"),
                         "failed_checks": [name for name, ok in checks.items() if not ok]})
        return {"completed": raw.get("completed") is True and set(cells) == expected,
                "matrix": rows, "passed": raw.get("completed") is True
                and all(row["status"] == "pass" for row in rows)}
    raise ValueError(f"unknown evidence kind: {kind}")


def curate(kind: str, path: Path, harness: Path | None = None) -> dict:
    raw = json.loads(path.read_text())
    result = summarize(kind, raw)
    recorded = raw.get("harness_sha256")
    observed = digest(harness) if harness else None
    return {
        "schema": "ax-engine.flash-next.evidence-review.v1",
        "kind": kind,
        "qualification": False,
        "release_ready": False,
        "performance_claim": False,
        "hardware_scope": "Apple M2 Ultra 192 GB development evidence; not M5 Ultra qualification",
        "raw_sha256": digest(path),
        "binary_sha256": raw["binary_sha256"],
        "binary_identity_scope": (
            "recorded aggregate identity; resumed cells are not independently bound to this binary"
            if raw.get("resumed") or raw.get("resumed_cell_ids")
            else "recorded by the original harness"
        ),
        "harness_provenance": {"recorded_sha256": recorded, "observed_sha256": observed,
                               "verified": bool(recorded and recorded == observed)},
        "summary": sanitize(result),
        "evidence": sanitize(raw),
        "limitations": ["Historical results do not validate a newer binary or merged commit.",
                        "Recorded thresholds are retained, not a new certification decision.",
                        "An unverified harness hash is an open reproducibility gate."],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("qa", "head", "throughput", "mtp", "http"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--harness", type=Path)
    args = parser.parse_args()
    result = curate(args.kind, args.input, args.harness)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
