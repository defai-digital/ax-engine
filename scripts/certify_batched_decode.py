#!/usr/bin/env python3
"""Generate fail-closed batched-decode certification evidence."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROBE = REPO_ROOT / "target/release/batched_decode_e2e_probe"
CERTIFICATION_FILE = "batched-decode-certification.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probe", type=Path, default=DEFAULT_PROBE)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--install",
        action="store_true",
        help="Install passing evidence into the model directory.",
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
    base: dict[str, str], scenario: dict[str, Any]
) -> dict[str, str]:
    environment = dict(base)
    environment.update(
        {
            "AX_MLX_BATCHED_DECODE": "1",
            "AX_MLX_BATCHED_DECODE_ALLOW_UNCERTIFIED": "1",
            "AX_BATCH": str(scenario["batch"]),
            "AX_PROMPT_LEN": str(scenario["prompt_len"]),
            "AX_GEN": str(scenario["gen_len"]),
            "AX_PROMPT_SEED": str(scenario["prompt_seed"]),
            "AX_SAMPLING": str(scenario["sampling"]),
        }
    )
    if scenario["ragged"]:
        environment["AX_RAGGED"] = "1"
    else:
        environment.pop("AX_RAGGED", None)
    return environment


def scenario_passed(returncode: int, stdout: str) -> bool:
    return (
        returncode == 0
        and "BATCHED-PATH: PASS" in stdout
        and "HARNESS-FIDELITY: PASS" in stdout
        and "BATCHED==SEQUENTIAL: PASS" in stdout
    )


def failure_summary(stdout: str, stderr: str) -> str | None:
    lines = []
    for line in (stdout + "\n" + stderr).splitlines():
        if "FAIL" in line or "MISMATCH" in line or "error:" in line.lower():
            lines.append(line.strip())
        if len(lines) == 4:
            break
    return " | ".join(lines) if lines else None


def run_scenario(
    probe: Path,
    model_dir: Path,
    scenario: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    completed = subprocess.run(
        [str(probe), str(model_dir)],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=scenario_environment(os.environ, scenario),
    )
    passed = scenario_passed(completed.returncode, completed.stdout)
    result = dict(scenario)
    result["passed"] = passed
    result["returncode"] = completed.returncode
    summary = failure_summary(completed.stdout, completed.stderr)
    if not passed and summary:
        result["failure_summary"] = summary
    return result


def build_evidence(
    context: dict[str, Any], scenarios: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "schema_version": context["schema_version"],
        "verdict": "pass" if scenarios and all(row["passed"] for row in scenarios) else "fail",
        "model_family": context["model_family"],
        "artifact_fingerprint_sha256": context["artifact_fingerprint_sha256"],
        "engine_version": context["engine_version"],
        "mlx_version": context["mlx_version"],
        "device_architecture": context["device_architecture"],
        "runtime_contract": context["runtime_contract"],
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
            run_scenario(args.probe, args.model_dir, scenario, args.timeout)
            for scenario in context["required_scenarios"]
        ]
    except (OSError, RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError) as error:
        print(f"certification failed: {error}", file=sys.stderr)
        return 2

    evidence = build_evidence(context, scenarios)
    write_json(args.output, evidence)
    print(f"wrote {evidence['verdict']} evidence to {args.output}")

    if args.install:
        if evidence["verdict"] != "pass":
            print("refusing to install failed certification evidence", file=sys.stderr)
            return 1
        install_path = args.model_dir / CERTIFICATION_FILE
        write_json(install_path, evidence)
        print(f"installed certification at {install_path}")
    return 0 if evidence["verdict"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
