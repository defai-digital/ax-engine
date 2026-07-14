#!/usr/bin/env python3
"""Validate decode hot-path kernel admission manifests.

This checker implements the admission contract from the internal decode
hot-path kernel strategy. It does not run benchmarks; it fails closed on the
structured evidence that must exist before a kernel or graph-fusion route can
be promoted into production routing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATES_ROOT = REPO_ROOT / ".internal/analysis/decode-hot-path-kernels"
SCHEMA_VERSION = "ax.decode_hot_path_kernel_candidate.v1"

ALLOWED_CLASSES = {
    "graph_compile",
    "phase1_metal",
    "mlx_sidecar",
    "sampling",
}
ALLOWED_STATUSES = {
    "prototype",
    "not_promoted",
    "promotion_candidate",
    "promoted",
    "no_go",
}
COMPLETE_STATUSES = {"promotion_candidate", "promoted"}
PROMOTION_DECISIONS = {
    "prototype",
    "not_promoted",
    "promote",
    "no_go",
}
REQUIRED_SECTIONS = (
    "profile",
    "mechanism",
    "correctness",
    "microbench",
    "real_graph_ab",
    "rollback",
    "promotion",
)
REQUIRED_COUNTER_HINTS = ("attempt", "hit", "fallback")


class DecodeHotPathAdmissionError(RuntimeError):
    pass


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise DecodeHotPathAdmissionError(f"failed to read {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise DecodeHotPathAdmissionError(f"failed to parse {path}: {error}") from error
    if not isinstance(payload, dict):
        raise DecodeHotPathAdmissionError(f"{path} must contain a JSON object")
    return payload


def _require_string(
    doc: dict[str, Any],
    key: str,
    errors: list[str],
    *,
    allow_empty: bool = False,
) -> str | None:
    value = doc.get(key)
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        errors.append(f"{key} must be a non-empty string")
        return None
    return value


def _require_bool(doc: dict[str, Any], key: str, errors: list[str]) -> bool | None:
    value = doc.get(key)
    if not isinstance(value, bool):
        errors.append(f"{key} must be a boolean")
        return None
    return value


def _require_number(
    doc: dict[str, Any],
    key: str,
    errors: list[str],
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float | None:
    value = doc.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        errors.append(f"{key} must be numeric")
        return None
    result = float(value)
    if minimum is not None and result < minimum:
        errors.append(f"{key} must be >= {minimum:g}")
    if maximum is not None and result > maximum:
        errors.append(f"{key} must be <= {maximum:g}")
    return result


def _require_int(
    doc: dict[str, Any],
    key: str,
    errors: list[str],
    *,
    minimum: int | None = None,
) -> int | None:
    value = doc.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        errors.append(f"{key} must be an integer")
        return None
    if minimum is not None and value < minimum:
        errors.append(f"{key} must be >= {minimum}")
    return value


def _section(doc: dict[str, Any], name: str, errors: list[str]) -> dict[str, Any] | None:
    value = doc.get(name)
    if not isinstance(value, dict):
        errors.append(f"{name} must be an object")
        return None
    return value


def _validate_relative_path(
    manifest_path: Path,
    section_name: str,
    section: dict[str, Any],
    errors: list[str],
    *,
    required: bool,
) -> None:
    value = section.get("path")
    if value is None and not required:
        return
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{section_name}.path must be a non-empty relative path")
        return
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        errors.append(f"{section_name}.path must stay inside the candidate directory")
        return
    if not (manifest_path.parent / path).is_file():
        errors.append(f"{section_name}.path does not exist: {value}")


def _string_list(value: Any, key: str, errors: list[str]) -> list[str]:
    if not isinstance(value, list) or not value:
        errors.append(f"{key} must be a non-empty array")
        return []
    items: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            errors.append(f"{key}[{index}] must be a non-empty string")
            continue
        items.append(item)
    return items


def _validate_profile(
    manifest_path: Path,
    profile: dict[str, Any],
    errors: list[str],
    *,
    complete: bool,
) -> None:
    _validate_relative_path(manifest_path, "profile", profile, errors, required=complete)
    source = _require_string(profile, "source", errors)
    if source and not (
        source.startswith("AX_MLX_") or source.startswith("model_family_")
    ):
        errors.append("profile.source must name an AX_MLX_* or model_family_* telemetry source")
    if not complete:
        return
    _require_string(profile, "model", errors)
    _require_int(profile, "prompt_tokens", errors, minimum=1)
    _require_int(profile, "generation_tokens", errors, minimum=1)
    _require_string(profile, "build_commit", errors)
    _require_string(profile, "host", errors)
    _require_string(profile, "dominant_stage", errors)
    _require_number(profile, "stage_wall_share_pct", errors, minimum=0.0, maximum=100.0)


def _validate_mechanism(
    manifest_path: Path,
    mechanism: dict[str, Any],
    errors: list[str],
    *,
    complete: bool,
) -> None:
    _validate_relative_path(manifest_path, "mechanism", mechanism, errors, required=complete)
    costs = _string_list(mechanism.get("removed_costs"), "mechanism.removed_costs", errors)
    allowed_cost_words = ("byte", "material", "dispatch", "readback", "barrier", "compute")
    if costs and not any(
        any(word in cost.lower() for word in allowed_cost_words) for cost in costs
    ):
        errors.append(
            "mechanism.removed_costs must name bytes, materialization, dispatch, "
            "readback, eval barrier, or duplicated compute"
        )
    if complete:
        _require_string(mechanism, "why_mlx_cannot_remove", errors)


def _validate_correctness(
    manifest_path: Path,
    correctness: dict[str, Any],
    errors: list[str],
    *,
    complete: bool,
) -> None:
    _validate_relative_path(
        manifest_path,
        "correctness",
        correctness,
        errors,
        required=complete,
    )
    oracle = _require_string(correctness, "oracle", errors)
    if oracle and oracle not in {"current_mlx", "cpu_reference", "scalar_reference"}:
        errors.append("correctness.oracle must be current_mlx, cpu_reference, or scalar_reference")
    greedy_parity = _require_string(correctness, "greedy_parity", errors)
    if greedy_parity and greedy_parity not in {"required", "passed", "not_applicable"}:
        errors.append("correctness.greedy_parity must be required, passed, or not_applicable")
    if complete:
        _require_number(correctness, "numeric_tolerance", errors, minimum=0.0)


def _validate_microbench(
    manifest_path: Path,
    microbench: dict[str, Any],
    errors: list[str],
    *,
    complete: bool,
) -> None:
    _validate_relative_path(manifest_path, "microbench", microbench, errors, required=complete)
    if complete:
        _require_int(microbench, "warmup_runs", errors, minimum=2)
        _require_int(microbench, "measure_runs", errors, minimum=5)
        _require_string(microbench, "host", errors)
        _require_number(microbench, "median_speedup_pct", errors)


def _validate_real_graph_ab(
    manifest_path: Path,
    real_graph_ab: dict[str, Any],
    errors: list[str],
    *,
    complete: bool,
) -> float | None:
    _validate_relative_path(
        manifest_path,
        "real_graph_ab",
        real_graph_ab,
        errors,
        required=complete,
    )
    if not complete:
        return None
    _require_string(real_graph_ab, "baseline_row", errors)
    _require_string(real_graph_ab, "candidate_row", errors)
    _require_number(real_graph_ab, "prefill_tok_s", errors, minimum=0.0)
    _require_number(real_graph_ab, "ttft_ms", errors, minimum=0.0)
    _require_bool(real_graph_ab, "greedy_parity_passed", errors)
    return _require_number(real_graph_ab, "decode_speedup_pct", errors)


def _validate_rollback(
    manifest_path: Path,
    rollback: dict[str, Any],
    errors: list[str],
    *,
    complete: bool,
    production_default: bool,
) -> None:
    _validate_relative_path(manifest_path, "rollback", rollback, errors, required=False)
    default_off = _require_bool(rollback, "default_off", errors)
    if default_off is False and not production_default:
        errors.append("rollback.default_off must be true unless production_default is true")
    _require_string(rollback, "fallback", errors)
    kill_switch = _require_string(rollback, "kill_switch", errors)
    if kill_switch and not kill_switch.startswith("AX_"):
        errors.append("rollback.kill_switch must be an AX_* environment flag")
    counters = _string_list(rollback.get("telemetry_counters"), "rollback.telemetry_counters", errors)
    if complete:
        lowered = " ".join(counters).lower()
        for hint in REQUIRED_COUNTER_HINTS:
            if hint not in lowered:
                errors.append(f"rollback.telemetry_counters must include a {hint} counter")


def _validate_promotion(
    manifest_path: Path,
    promotion: dict[str, Any],
    errors: list[str],
    *,
    status: str,
    complete: bool,
    decode_speedup_pct: float | None,
    real_graph_ab: dict[str, Any] | None,
) -> None:
    _validate_relative_path(manifest_path, "promotion", promotion, errors, required=complete)
    decision = _require_string(promotion, "decision", errors)
    _require_string(promotion, "reason", errors)
    if decision and decision not in PROMOTION_DECISIONS:
        errors.append(f"promotion.decision must be one of {sorted(PROMOTION_DECISIONS)}")
    if status == "promoted" and decision != "promote":
        errors.append("status=promoted requires promotion.decision=promote")
    if status == "promotion_candidate" and decision not in {"promote", "not_promoted"}:
        errors.append("status=promotion_candidate requires promote or not_promoted decision")
    if status == "no_go" and decision not in {"no_go", "not_promoted"}:
        errors.append("status=no_go requires no_go or not_promoted decision")
    if decision != "promote":
        return
    if decode_speedup_pct is None:
        errors.append("promotion.decision=promote requires real_graph_ab.decode_speedup_pct")
        return
    if decode_speedup_pct >= 5.0:
        return
    lower_threshold_met = decode_speedup_pct >= 3.0 and bool(
        real_graph_ab and real_graph_ab.get("improved_ttft_or_variance") is True
    )
    if not lower_threshold_met:
        errors.append(
            "promotion.decision=promote requires >=5% decode speedup, or >=3% "
            "with real_graph_ab.improved_ttft_or_variance=true"
        )


def validate_candidate_manifest(
    manifest_path: Path,
    *,
    require_complete: bool = False,
) -> dict[str, Any]:
    doc = _load_json(manifest_path)
    errors: list[str] = []

    schema_version = _require_string(doc, "schema_version", errors)
    if schema_version and schema_version != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    candidate_id = _require_string(doc, "candidate_id", errors)
    if candidate_id and candidate_id != manifest_path.parent.name:
        errors.append("candidate_id must match the candidate directory name")
    _require_string(doc, "title", errors)
    candidate_class = _require_string(doc, "class", errors)
    if candidate_class and candidate_class not in ALLOWED_CLASSES:
        errors.append(f"class must be one of {sorted(ALLOWED_CLASSES)}")
    status = _require_string(doc, "status", errors)
    if status and status not in ALLOWED_STATUSES:
        errors.append(f"status must be one of {sorted(ALLOWED_STATUSES)}")
    production_default = _require_bool(doc, "production_default", errors)
    feature_flag = doc.get("feature_flag")
    if feature_flag is not None:
        if not isinstance(feature_flag, str) or not feature_flag.startswith("AX_"):
            errors.append("feature_flag must be an AX_* environment flag when present")

    complete = require_complete or status in COMPLETE_STATUSES or status == "promoted"
    if production_default is True:
        complete = True
    if production_default is True and status != "promoted":
        errors.append("production_default=true requires status=promoted")
    if status != "promoted" and production_default is True:
        errors.append("non-promoted candidates must not be production_default")
    if status in {"prototype", "promotion_candidate", "promoted"} and feature_flag is None:
        errors.append(f"status={status} requires feature_flag")

    missing_sections = [name for name in REQUIRED_SECTIONS if name not in doc]
    if complete and missing_sections:
        errors.append("complete candidate is missing sections: " + ", ".join(missing_sections))

    sections: dict[str, dict[str, Any]] = {}
    for name in REQUIRED_SECTIONS:
        if name in doc:
            value = _section(doc, name, errors)
            if value is not None:
                sections[name] = value

    if "profile" in sections:
        _validate_profile(manifest_path, sections["profile"], errors, complete=complete)
    if "mechanism" in sections:
        _validate_mechanism(manifest_path, sections["mechanism"], errors, complete=complete)
    if "correctness" in sections:
        _validate_correctness(manifest_path, sections["correctness"], errors, complete=complete)
    if "microbench" in sections:
        _validate_microbench(manifest_path, sections["microbench"], errors, complete=complete)

    real_graph = sections.get("real_graph_ab")
    decode_speedup_pct = None
    if real_graph is not None:
        decode_speedup_pct = _validate_real_graph_ab(
            manifest_path,
            real_graph,
            errors,
            complete=complete,
        )
    if "rollback" in sections:
        _validate_rollback(
            manifest_path,
            sections["rollback"],
            errors,
            complete=complete,
            production_default=production_default is True,
        )
    if "promotion" in sections and status:
        _validate_promotion(
            manifest_path,
            sections["promotion"],
            errors,
            status=status,
            complete=complete,
            decode_speedup_pct=decode_speedup_pct,
            real_graph_ab=real_graph,
        )

    if complete and sections.get("correctness", {}).get("greedy_parity") != "passed":
        errors.append("complete candidate requires correctness.greedy_parity=passed")
    if complete and real_graph and real_graph.get("greedy_parity_passed") is not True:
        errors.append("complete candidate requires real_graph_ab.greedy_parity_passed=true")

    if errors:
        rendered = "\n".join(f"- {error}" for error in errors)
        raise DecodeHotPathAdmissionError(f"{manifest_path} failed admission:\n{rendered}")

    return {
        "path": str(manifest_path),
        "candidate_id": candidate_id,
        "status": status,
        "complete": complete,
    }


def discover_candidate_manifests(candidates_root: Path) -> list[Path]:
    if not candidates_root.exists():
        return []
    return sorted(
        path
        for path in candidates_root.glob("*/candidate.json")
        if path.is_file()
    )


def check_candidates(
    candidates_root: Path,
    *,
    require_complete: bool = False,
    allow_empty: bool = True,
) -> list[dict[str, Any]]:
    manifests = discover_candidate_manifests(candidates_root)
    if not manifests and not allow_empty:
        raise DecodeHotPathAdmissionError(
            f"no candidate manifests found under {candidates_root}"
        )
    return [
        validate_candidate_manifest(path, require_complete=require_complete)
        for path in manifests
    ]


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates-root",
        type=Path,
        default=DEFAULT_CANDIDATES_ROOT,
        help="decode hot-path candidate root to scan",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="require every discovered candidate to satisfy the full checklist",
    )
    parser.add_argument(
        "--no-allow-empty",
        action="store_true",
        help="fail when no candidate manifests are found",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        help="optional path for a JSON validation report",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        checked = check_candidates(
            args.candidates_root,
            require_complete=args.require_complete,
            allow_empty=not args.no_allow_empty,
        )
    except DecodeHotPathAdmissionError as error:
        print(f"Decode hot-path kernel admission check failed: {error}", file=sys.stderr)
        return 1

    report = {
        "schema_version": "ax.decode_hot_path_kernel_admission_report.v1",
        "candidates_root": str(args.candidates_root),
        "candidate_count": len(checked),
        "candidates": checked,
    }
    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if checked:
        statuses = ", ".join(f"{item['candidate_id']}={item['status']}" for item in checked)
        print(f"ok: validated {len(checked)} decode hot-path candidate(s): {statuses}")
    else:
        print(f"ok: no decode hot-path candidate manifests found under {args.candidates_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
