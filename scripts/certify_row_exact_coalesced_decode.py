#!/usr/bin/env python3
"""Certify and profile the row-exact Qwen/Gemma decode-coalescing fallback."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROBE = REPO_ROOT / "target/release/batched_decode_e2e_probe"
DECODE_RE = re.compile(
    r"decode: (?P<seconds>[0-9.]+)s for (?P<tokens>[0-9]+) tokens = "
    r"(?P<tokens_per_second>[0-9.]+) agg tok/s"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probe", type=Path, default=DEFAULT_PROBE)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--with-baseline",
        action="store_true",
        help="Also run the exact same matrix with all decode batching disabled.",
    )
    return parser.parse_args(argv)


def certification_context(probe: Path, model_dir: Path, timeout: float) -> dict[str, Any]:
    completed = subprocess.run(
        [str(probe), str(model_dir), "--certification-context-json"],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "failed to read certification context: "
            + (completed.stderr.strip() or completed.stdout.strip())
        )
    value = json.loads(completed.stdout)
    required = value.get("required_scenarios")
    if not isinstance(required, list) or not required:
        raise RuntimeError("certification context has no required scenarios")
    return value


def scenario_environment(
    base: dict[str, str], scenario: dict[str, Any], *, coalesced: bool
) -> dict[str, str]:
    environment = dict(base)
    environment.update(
        {
            "AX_MLX_BATCHED_DECODE": "1" if coalesced else "0",
            "AX_BATCH": str(scenario["batch"]),
            "AX_PROMPT_LEN": str(scenario["prompt_len"]),
            "AX_GEN": str(scenario["gen_len"]),
            "AX_PROMPT_SEED": str(scenario["prompt_seed"]),
            "AX_SAMPLING": str(scenario["sampling"]),
        }
    )
    # The diagnostic tensor-batch override would defeat the purpose of this
    # certificate. Delete it even if the operator's shell exported it.
    environment.pop("AX_MLX_BATCHED_DECODE_ALLOW_UNCERTIFIED", None)
    if scenario["ragged"]:
        environment["AX_RAGGED"] = "1"
    else:
        environment.pop("AX_RAGGED", None)
    return environment


def parse_decode_metrics(stdout: str) -> dict[str, float | int]:
    match = DECODE_RE.search(stdout)
    if match is None:
        raise RuntimeError("probe output has no decode timing")
    return {
        "seconds": float(match.group("seconds")),
        "tokens": int(match.group("tokens")),
        "tokens_per_second": float(match.group("tokens_per_second")),
    }


def oracle_passed(returncode: int, stdout: str) -> bool:
    return (
        returncode == 0
        and "HARNESS-FIDELITY: PASS" in stdout
        and "BATCHED==SEQUENTIAL: PASS" in stdout
    )


def coalesced_passed(returncode: int, stdout: str) -> bool:
    return (
        oracle_passed(returncode, stdout)
        and "ROW-EXACT-COALESCED-PATH: PASS" in stdout
        and "BATCHED-PATH: PASS" not in stdout
    )


def failure_summary(stdout: str, stderr: str) -> str | None:
    lines = []
    for line in (stdout + "\n" + stderr).splitlines():
        if "FAIL" in line or "MISMATCH" in line or "error:" in line.lower():
            lines.append(line.strip())
        if len(lines) == 4:
            break
    return " | ".join(lines) if lines else None


def run_probe(
    probe: Path,
    model_dir: Path,
    scenario: dict[str, Any],
    timeout: float,
    *,
    coalesced: bool,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(probe), str(model_dir)],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=scenario_environment(os.environ, scenario, coalesced=coalesced),
    )


def run_scenario(
    probe: Path,
    model_dir: Path,
    scenario: dict[str, Any],
    timeout: float,
    *,
    with_baseline: bool,
) -> dict[str, Any]:
    completed = run_probe(probe, model_dir, scenario, timeout, coalesced=True)
    passed = coalesced_passed(completed.returncode, completed.stdout)
    result = dict(scenario)
    result["passed"] = passed
    result["returncode"] = completed.returncode
    with contextlib.suppress(RuntimeError):
        result["coalesced"] = parse_decode_metrics(completed.stdout)
    summary = failure_summary(completed.stdout, completed.stderr)
    if not passed and summary:
        result["failure_summary"] = summary

    if with_baseline:
        baseline = run_probe(probe, model_dir, scenario, timeout, coalesced=False)
        baseline_passed = oracle_passed(baseline.returncode, baseline.stdout)
        result["baseline_passed"] = baseline_passed
        with contextlib.suppress(RuntimeError):
            result["baseline"] = parse_decode_metrics(baseline.stdout)
        coalesced_metrics = result.get("coalesced")
        baseline_metrics = result.get("baseline")
        if isinstance(coalesced_metrics, dict) and isinstance(baseline_metrics, dict):
            baseline_rate = float(baseline_metrics["tokens_per_second"])
            if baseline_rate > 0:
                result["speedup"] = (
                    float(coalesced_metrics["tokens_per_second"]) / baseline_rate
                )
        result["passed"] = bool(result["passed"] and baseline_passed)
        baseline_summary = failure_summary(baseline.stdout, baseline.stderr)
        if not baseline_passed and baseline_summary:
            result["baseline_failure_summary"] = baseline_summary
    return result


def build_evidence(
    context: dict[str, Any], scenarios: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "schema_version": "ax.mlx.row_exact_coalesced_decode_certification.v1",
        "verdict": "pass" if scenarios and all(row["passed"] for row in scenarios) else "fail",
        "model_family": context["model_family"],
        "artifact_fingerprint_sha256": context["artifact_fingerprint_sha256"],
        "engine_version": context["engine_version"],
        "mlx_version": context["mlx_version"],
        "device_architecture": context["device_architecture"],
        "runtime_contract": "ax.mlx.row_exact_coalesced_decode.runtime.v1",
        "numerics_env_sha256": context["numerics_env_sha256"],
        "scenarios": scenarios,
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.probe.is_file():
        print(f"probe not found: {args.probe}", file=sys.stderr)
        return 2
    if not args.model_dir.is_dir():
        print(f"model directory not found: {args.model_dir}", file=sys.stderr)
        return 2
    try:
        context = certification_context(args.probe, args.model_dir, args.timeout)
        scenarios = [
            run_scenario(
                args.probe,
                args.model_dir,
                scenario,
                args.timeout,
                with_baseline=args.with_baseline,
            )
            for scenario in context["required_scenarios"]
        ]
    except (OSError, RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError) as error:
        print(f"certification failed: {error}", file=sys.stderr)
        return 2

    evidence = build_evidence(context, scenarios)
    write_json(args.output, evidence)
    print(f"wrote {evidence['verdict']} evidence to {args.output}")
    return 0 if evidence["verdict"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
