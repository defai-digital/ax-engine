#!/usr/bin/env python3
"""Compare two multi-model serving artifacts for the Qwen3+Gemma4 flip gates.

Candidate is typically AX single-process multi-model; baseline is typically a
peer (e.g. mlxcel multi-process) run on the same scenario/sampling. Thresholds
are provisional until Week-1 calibration on the primary host.

Exit codes:
  0 — all required gates passed (or --report-only)
  1 — validation / gate failure
  2 — usage error
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import check_ax_multimodel_serving_artifact as multimodel

SCHEMA_VERSION = "ax.qwen_gemma_flip_compare.v1"


class FlipCompareError(RuntimeError):
    """Raised when flip comparison cannot be completed or gates fail."""


def load_validated(path: Path) -> dict[str, Any]:
    return multimodel.validate_multimodel_serving_artifact(
        path,
        min_requests=1,
        require_zero_errors=False,
        require_focus_families=[],
        max_interactive_stream_gap_p95_ms=None,
        max_request_http_503=None,
    )


def metric_from_summary(summary: dict[str, Any], key: str) -> float | None:
    if key in {"output_token_throughput_tok_s", "request_throughput_rps"}:
        value = summary.get(key)
        if value is None:
            return None
        return float(value)
    if key in {"ttft_p95", "stream_gap_p95", "e2e_p95", "tpot_p95"}:
        mapping = {
            "ttft_p95": "ttft_ms",
            "stream_gap_p95": "stream_step_interval_ms",
            "e2e_p95": "e2e_latency_ms",
            "tpot_p95": "client_tpot_ms",
        }
        dist = summary.get(mapping[key])
        if not isinstance(dist, dict) or dist.get("p95") is None:
            return None
        return float(dist["p95"])
    raise FlipCompareError(f"unknown metric key {key!r}")


def interactive_gap(artifact: dict[str, Any]) -> float | None:
    try:
        return multimodel.interactive_stream_gap_p95(
            artifact,
            list(artifact.get("observations") or []),
            dict(artifact.get("summary") or {}),
        )
    except multimodel.ArtifactCheckError:
        return None


def ratio(candidate: float | None, baseline: float | None) -> float | None:
    if candidate is None or baseline is None:
        return None
    if baseline == 0.0:
        return None
    return candidate / baseline


def request_error_rate(summary: dict[str, Any]) -> float:
    requests = int(summary.get("requests") or 0)
    if requests <= 0:
        return 0.0
    return float(summary.get("error_requests") or 0) / float(requests)


def http_503_count(artifact: dict[str, Any]) -> int:
    availability = artifact.get("availability")
    if isinstance(availability, dict) and availability.get("request_http_503") is not None:
        return int(availability["request_http_503"])
    count = 0
    for obs in artifact.get("observations") or []:
        if obs.get("kind") == "request" and obs.get("status") == 503:
            count += 1
    return count


def focus_families_present(artifact: dict[str, Any]) -> list[str]:
    models = artifact.get("target", {}).get("models") or []
    found: set[str] = set()
    for model_id in models:
        family = multimodel.classify_focus_model(str(model_id))
        if family:
            found.add(family)
    return sorted(found)


def _package_signature(identity: Any) -> dict[str, Any] | None:
    if not isinstance(identity, dict):
        return None
    keys = (
        "path",
        "identity_file_sha256",
        "safetensors_files",
        "safetensors_bytes",
        "safetensors_layout_sha256",
    )
    return {key: identity.get(key) for key in keys}


def evaluate_comparison_contract(
    candidate: dict[str, Any], baseline: dict[str, Any]
) -> dict[str, Any]:
    mismatches: list[str] = []

    cand_scenario = candidate.get("scenario")
    base_scenario = baseline.get("scenario")
    if not isinstance(cand_scenario, dict) or not isinstance(base_scenario, dict):
        mismatches.append("scenario metadata is missing")
    else:
        for key in ("sha256", "events"):
            if cand_scenario.get(key) != base_scenario.get(key):
                mismatches.append(
                    f"scenario.{key}: candidate={cand_scenario.get(key)!r} "
                    f"baseline={base_scenario.get(key)!r}"
                )

    cand_sampling = candidate.get("sampling")
    base_sampling = baseline.get("sampling")
    if not isinstance(cand_sampling, dict) or not isinstance(base_sampling, dict):
        mismatches.append("sampling metadata is missing")
    else:
        for key in ("temperature", "top_p", "top_k", "seed", "input_kind"):
            if cand_sampling.get(key) != base_sampling.get(key):
                mismatches.append(
                    f"sampling.{key}: candidate={cand_sampling.get(key)!r} "
                    f"baseline={base_sampling.get(key)!r}"
                )

    cand_method = candidate.get("methodology")
    base_method = baseline.get("methodology")
    if not isinstance(cand_method, dict) or not isinstance(base_method, dict):
        mismatches.append("methodology metadata is missing")
    else:
        for key in ("request_endpoint", "protocol"):
            if cand_method.get(key) != base_method.get(key):
                mismatches.append(
                    f"methodology.{key}: candidate={cand_method.get(key)!r} "
                    f"baseline={base_method.get(key)!r}"
                )

    cand_target = candidate.get("target")
    base_target = baseline.get("target")
    if not isinstance(cand_target, dict) or not isinstance(base_target, dict):
        mismatches.append("target metadata is missing")
        cand_target = {}
        base_target = {}

    cand_contract = cand_target.get("comparison_contract")
    base_contract = base_target.get("comparison_contract")
    if not isinstance(cand_contract, dict) or not cand_contract:
        mismatches.append("candidate target comparison_contract is missing")
    if not isinstance(base_contract, dict) or not base_contract:
        mismatches.append("baseline target comparison_contract is missing")
    if (
        isinstance(cand_contract, dict)
        and isinstance(base_contract, dict)
        and cand_contract != base_contract
    ):
        mismatches.append("target comparison_contract objects differ")

    cand_host = cand_target.get("host")
    base_host = base_target.get("host")
    if not isinstance(cand_host, dict) or not isinstance(base_host, dict):
        mismatches.append("target host identity is missing")
    else:
        for key in ("node", "machine", "platform"):
            if cand_host.get(key) != base_host.get(key):
                mismatches.append(
                    f"target.host.{key}: candidate={cand_host.get(key)!r} "
                    f"baseline={base_host.get(key)!r}"
                )

    cand_packages = cand_target.get("model_packages")
    base_packages = base_target.get("model_packages")
    if not isinstance(cand_packages, dict) or not isinstance(base_packages, dict):
        mismatches.append("target model_packages identity is missing")
    else:
        model_ids = sorted(set(cand_packages) | set(base_packages))
        for model_id in model_ids:
            cand_signature = _package_signature(cand_packages.get(model_id))
            base_signature = _package_signature(base_packages.get(model_id))
            if cand_signature is None or base_signature is None:
                mismatches.append(f"model package identity missing for {model_id}")
            elif cand_signature != base_signature:
                mismatches.append(f"model package identity differs for {model_id}")

    cand_input_tokens = {
        str(item.get("event_id")): item.get("input_tokens")
        for item in candidate.get("observations") or []
        if item.get("kind") == "request"
    }
    base_input_tokens = {
        str(item.get("event_id")): item.get("input_tokens")
        for item in baseline.get("observations") or []
        if item.get("kind") == "request"
    }
    if not cand_input_tokens or any(
        not isinstance(value, int) for value in cand_input_tokens.values()
    ):
        mismatches.append("candidate authoritative per-event input token counts are missing")
    if not base_input_tokens or any(
        not isinstance(value, int) for value in base_input_tokens.values()
    ):
        mismatches.append("baseline authoritative per-event input token counts are missing")
    if cand_input_tokens != base_input_tokens:
        mismatches.append(
            f"per-event input token counts differ: candidate={cand_input_tokens!r} "
            f"baseline={base_input_tokens!r}"
        )

    return {
        "passed": not mismatches,
        "mismatches": mismatches,
        "candidate_runtime": cand_target.get("runtime"),
        "candidate_process_count": cand_target.get("process_count"),
        "baseline_runtime": base_target.get("runtime"),
        "baseline_process_count": base_target.get("process_count"),
    }


def evaluate_gates(
    *,
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    min_throughput_ratio: float,
    max_ttft_p95_ratio: float,
    max_stream_gap_p95_ratio: float,
    max_candidate_error_rate: float,
    max_candidate_http_503: int | None,
    require_focus_families: list[str],
    require_contract_match: bool,
) -> dict[str, Any]:
    cand_summary = candidate["summary"]
    base_summary = baseline["summary"]

    cand_tok = metric_from_summary(cand_summary, "output_token_throughput_tok_s")
    base_tok = metric_from_summary(base_summary, "output_token_throughput_tok_s")
    cand_ttft = metric_from_summary(cand_summary, "ttft_p95")
    base_ttft = metric_from_summary(base_summary, "ttft_p95")
    cand_gap = interactive_gap(candidate)
    base_gap = interactive_gap(baseline)

    throughput_ratio = ratio(cand_tok, base_tok)
    ttft_ratio = ratio(cand_ttft, base_ttft)
    gap_ratio = ratio(cand_gap, base_gap)
    cand_error_rate = request_error_rate(cand_summary)
    cand_503 = http_503_count(candidate)
    families = focus_families_present(candidate)
    contract = evaluate_comparison_contract(candidate, baseline)

    gates: list[dict[str, Any]] = []

    def add_gate(
        name: str,
        passed: bool | None,
        *,
        detail: str,
        required: bool = True,
    ) -> None:
        gates.append(
            {
                "name": name,
                "passed": passed,
                "required": required,
                "detail": detail,
            }
        )

    add_gate(
        "comparison_contract",
        contract["passed"],
        detail=("matched" if contract["passed"] else "; ".join(contract["mismatches"])),
        required=require_contract_match,
    )

    for family in require_focus_families:
        add_gate(
            f"focus_family_{family}",
            family in families,
            detail=f"present_families={families}",
        )

    if throughput_ratio is None:
        add_gate("throughput_ratio", None, detail="missing throughput on candidate or baseline")
    else:
        add_gate(
            "throughput_ratio",
            throughput_ratio >= min_throughput_ratio,
            detail=(
                f"candidate/baseline={throughput_ratio:.4f} "
                f"(need >= {min_throughput_ratio:.4f}); "
                f"cand={cand_tok:.4f} base={base_tok:.4f}"
            ),
        )

    if ttft_ratio is None:
        add_gate("ttft_p95_ratio", None, detail="missing ttft p95 on candidate or baseline")
    else:
        add_gate(
            "ttft_p95_ratio",
            ttft_ratio <= max_ttft_p95_ratio,
            detail=(
                f"candidate/baseline={ttft_ratio:.4f} "
                f"(need <= {max_ttft_p95_ratio:.4f}); "
                f"cand_p95={cand_ttft:.3f}ms base_p95={base_ttft:.3f}ms"
            ),
        )

    if gap_ratio is None:
        add_gate(
            "interactive_stream_gap_p95_ratio",
            None,
            detail="missing interactive stream-gap p95 on candidate or baseline",
            required=False,
        )
    else:
        add_gate(
            "interactive_stream_gap_p95_ratio",
            gap_ratio <= max_stream_gap_p95_ratio,
            detail=(
                f"candidate/baseline={gap_ratio:.4f} "
                f"(need <= {max_stream_gap_p95_ratio:.4f}); "
                f"cand_p95={cand_gap:.3f}ms base_p95={base_gap:.3f}ms"
            ),
        )

    add_gate(
        "candidate_error_rate",
        cand_error_rate <= max_candidate_error_rate,
        detail=f"error_rate={cand_error_rate:.4f} (need <= {max_candidate_error_rate:.4f})",
    )

    if max_candidate_http_503 is not None:
        add_gate(
            "candidate_http_503",
            cand_503 <= max_candidate_http_503,
            detail=f"http_503={cand_503} (need <= {max_candidate_http_503})",
        )

    required_failed = [gate for gate in gates if gate["required"] and gate["passed"] is not True]
    return {
        "schema_version": SCHEMA_VERSION,
        "comparison_contract": contract,
        "focus_families_present": families,
        "metrics": {
            "candidate_output_token_throughput_tok_s": cand_tok,
            "baseline_output_token_throughput_tok_s": base_tok,
            "throughput_ratio": throughput_ratio,
            "candidate_ttft_p95_ms": cand_ttft,
            "baseline_ttft_p95_ms": base_ttft,
            "ttft_p95_ratio": ttft_ratio,
            "candidate_interactive_stream_gap_p95_ms": cand_gap,
            "baseline_interactive_stream_gap_p95_ms": base_gap,
            "interactive_stream_gap_p95_ratio": gap_ratio,
            "candidate_error_rate": cand_error_rate,
            "candidate_http_503": cand_503,
        },
        "gates": gates,
        "passed": not required_failed,
        "failed_required_gates": [gate["name"] for gate in required_failed],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        type=Path,
        required=True,
        help="AX (or improved) multi-model artifact",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        required=True,
        help="Peer baseline multi-model artifact (e.g. mlxcel multi-process)",
    )
    parser.add_argument(
        "--min-throughput-ratio",
        type=float,
        default=1.15,
        help="Provisional: candidate/baseline output tok/s minimum (default 1.15)",
    )
    parser.add_argument(
        "--max-ttft-p95-ratio",
        type=float,
        default=0.90,
        help="Provisional: candidate/baseline TTFT p95 maximum (default 0.90)",
    )
    parser.add_argument(
        "--max-stream-gap-p95-ratio",
        type=float,
        default=0.90,
        help="Provisional: candidate/baseline interactive stream-gap p95 max (default 0.90)",
    )
    parser.add_argument(
        "--max-candidate-error-rate",
        type=float,
        default=0.0,
        help="Maximum allowed candidate request error rate (default 0)",
    )
    parser.add_argument(
        "--max-candidate-http-503",
        type=int,
        default=None,
        help="Optional absolute HTTP 503 budget on candidate request observations",
    )
    parser.add_argument(
        "--require-focus-family",
        action="append",
        default=None,
        choices=sorted(multimodel.FOCUS_FAMILY_PATTERNS),
        help=(
            "Focus families that must appear on the candidate. Repeatable. "
            "Default when omitted: qwen3 and gemma4."
        ),
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Always exit 0 after writing the comparison (calibration mode).",
    )
    parser.add_argument(
        "--allow-contract-mismatch",
        action="store_true",
        help=(
            "Do not make scenario/sampling/host/package/memory contract mismatches "
            "a required failure. Intended only for exploratory legacy artifacts."
        ),
    )
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    for name, value in (
        ("min_throughput_ratio", args.min_throughput_ratio),
        ("max_ttft_p95_ratio", args.max_ttft_p95_ratio),
        ("max_stream_gap_p95_ratio", args.max_stream_gap_p95_ratio),
        ("max_candidate_error_rate", args.max_candidate_error_rate),
    ):
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
            print(f"invalid {name}: {value!r}", file=sys.stderr)
            return 2

    raw_families = (
        args.require_focus_family
        if args.require_focus_family is not None
        else [
            "qwen3",
            "gemma4",
        ]
    )
    require_families: list[str] = []
    for family in raw_families:
        if family not in require_families:
            require_families.append(family)

    try:
        candidate = load_validated(args.candidate)
        baseline = load_validated(args.baseline)
        report = evaluate_gates(
            candidate=candidate,
            baseline=baseline,
            min_throughput_ratio=float(args.min_throughput_ratio),
            max_ttft_p95_ratio=float(args.max_ttft_p95_ratio),
            max_stream_gap_p95_ratio=float(args.max_stream_gap_p95_ratio),
            max_candidate_error_rate=float(args.max_candidate_error_rate),
            max_candidate_http_503=args.max_candidate_http_503,
            require_focus_families=require_families,
            require_contract_match=not args.allow_contract_mismatch,
        )
    except (FlipCompareError, multimodel.ArtifactCheckError) as error:
        print(f"Qwen/Gemma flip compare failed: {error}", file=sys.stderr)
        return 1

    report["candidate_path"] = str(args.candidate)
    report["baseline_path"] = str(args.baseline)
    report["candidate_scenario"] = candidate.get("scenario")
    report["baseline_scenario"] = baseline.get("scenario")
    report["thresholds"] = {
        "min_throughput_ratio": args.min_throughput_ratio,
        "max_ttft_p95_ratio": args.max_ttft_p95_ratio,
        "max_stream_gap_p95_ratio": args.max_stream_gap_p95_ratio,
        "max_candidate_error_rate": args.max_candidate_error_rate,
        "max_candidate_http_503": args.max_candidate_http_503,
        "require_focus_families": require_families,
        "require_contract_match": not args.allow_contract_mismatch,
    }

    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        sys.stdout.write(text)

    if args.report_only or report["passed"]:
        status = "PASS" if report["passed"] else "REPORT_ONLY_FAIL"
        print(
            f"Qwen/Gemma flip compare {status}: failed_required={report['failed_required_gates']}",
            file=sys.stderr,
        )
        return 0

    print(
        f"Qwen/Gemma flip compare FAIL: failed_required={report['failed_required_gates']}",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
