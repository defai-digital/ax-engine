#!/usr/bin/env python3
"""Run repeated, cache-isolated Qwen/Gemma S0-S3 trials across targets."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import bench_qwen_gemma_flip_target as target_runner
import check_ax_multimodel_serving_artifact as checker
import compare_qwen_gemma_flip as comparator

SCHEMA_VERSION = "ax.qwen_gemma_flip_campaign.v1"
REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class ScenarioSpec:
    scenario_id: str
    path: Path
    required_focus_families: tuple[str, ...]


DEFAULT_SCENARIOS = (
    ScenarioSpec(
        "s0",
        REPO_ROOT / "benchmarks/manifests/replay/multimodel_decode_baseline.jsonl",
        ("qwen3",),
    ),
    ScenarioSpec(
        "s1",
        REPO_ROOT / "benchmarks/manifests/replay/multimodel_prefill_isolation.jsonl",
        ("qwen3", "gemma4"),
    ),
    ScenarioSpec(
        "s2",
        REPO_ROOT / "benchmarks/manifests/replay/multimodel_lifecycle_isolation.jsonl",
        ("qwen3",),
    ),
    ScenarioSpec(
        "s3",
        REPO_ROOT / "benchmarks/manifests/replay/qwen_gemma_agent_coexist.jsonl",
        ("qwen3", "gemma4"),
    ),
)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def parse_scenario(value: str) -> ScenarioSpec:
    scenario_id, separator, path_text = value.partition("=")
    if not separator or not scenario_id or not path_text:
        raise argparse.ArgumentTypeError("scenario must be ID=PATH")
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", scenario_id):
        raise argparse.ArgumentTypeError("scenario ID must use lowercase letters, digits, _ or -")
    path = Path(path_text).resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"scenario does not exist: {path}")
    events = target_runner.multimodel.load_scenario(path)
    families = {
        target_runner.multimodel.classify_focus_family(event.model_id)
        for event in events
        if event.kind == "request"
    }
    return ScenarioSpec(
        scenario_id,
        path,
        tuple(family for family in ("qwen3", "gemma4") if family in families),
    )


def slug(value: str) -> str:
    rendered = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return rendered or "target"


def build_run_command(
    *,
    target_path: Path,
    scenario: ScenarioSpec,
    artifact_path: Path,
    log_dir: Path,
    workers: int,
    timeout: float,
) -> list[str]:
    return [
        sys.executable,
        str(REPO_ROOT / "scripts/bench_qwen_gemma_flip_target.py"),
        "--target",
        str(target_path),
        "--scenario",
        str(scenario.path),
        "--workers",
        str(workers),
        "--timeout",
        str(timeout),
        "--log-dir",
        str(log_dir),
        "--output",
        str(artifact_path),
        "--report-only",
    ]


def validate_trial(path: Path, scenario: ScenarioSpec) -> dict[str, Any]:
    artifact = checker.validate_multimodel_serving_artifact(
        path,
        min_requests=1,
        require_zero_errors=True,
        require_focus_families=list(scenario.required_focus_families),
        max_interactive_stream_gap_p95_ms=None,
        max_request_http_503=0,
    )
    lifecycle = artifact.get("lifecycle")
    if not isinstance(lifecycle, dict) or int(lifecycle.get("error_events") or 0) != 0:
        raise checker.ArtifactCheckError(f"{path}: lifecycle control event failed")
    return artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, action="append", required=True)
    parser.add_argument(
        "--scenario",
        type=parse_scenario,
        action="append",
        help="Override the default S0-S3 suite with ID=PATH (repeatable).",
    )
    parser.add_argument("--repetitions", type=positive_int, default=3)
    parser.add_argument("--cooldown", type=non_negative_float, default=15.0)
    parser.add_argument("--workers", type=positive_int, default=16)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--allow-fewer-than-three",
        action="store_true",
        help="Allow fewer than three repetitions for smoke testing only.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue after a failed trial so the campaign ledger records every attempted row.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.repetitions < 3 and not args.allow_fewer_than_three:
        raise SystemExit("--repetitions must be at least 3 unless --allow-fewer-than-three is set")
    if args.timeout <= 0:
        raise SystemExit("--timeout must be positive")

    target_paths = [path.resolve() for path in args.target]
    targets = [target_runner.load_target(path) for path in target_paths]
    contracts = [target.comparison_contract for target in targets]
    if any(contract != contracts[0] for contract in contracts[1:]):
        raise SystemExit("all campaign targets must use the same comparison_contract")
    target_labels: list[str] = []
    for target in targets:
        label = slug(target.runtime)
        if label in target_labels:
            label = slug(target.name)
        if label in target_labels:
            raise SystemExit(f"target label collision for {target.name!r}")
        target_labels.append(label)

    scenarios = list(args.scenario or DEFAULT_SCENARIOS)
    if len({scenario.scenario_id for scenario in scenarios}) != len(scenarios):
        raise SystemExit("scenario IDs must be unique")
    output_dir = args.output_dir.resolve()
    artifact_dir = output_dir / "artifacts"
    log_root = output_dir / "logs"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    started_at = target_runner.serving.utc_now()
    records: list[dict[str, Any]] = []
    artifacts: dict[tuple[int, str, str], dict[str, Any]] = {}
    stop_requested = False
    total_trials = args.repetitions * len(scenarios) * len(targets)
    trial_index = 0

    for repetition in range(1, args.repetitions + 1):
        indexed_targets = list(zip(target_labels, targets, target_paths, strict=True))
        if repetition % 2 == 0:
            indexed_targets.reverse()
        for scenario in scenarios:
            for label, target, target_path in indexed_targets:
                trial_index += 1
                artifact_path = artifact_dir / f"{label}-{scenario.scenario_id}-r{repetition}.json"
                log_dir = log_root / f"{label}-{scenario.scenario_id}-r{repetition}"
                command = build_run_command(
                    target_path=target_path,
                    scenario=scenario,
                    artifact_path=artifact_path,
                    log_dir=log_dir,
                    workers=args.workers,
                    timeout=args.timeout,
                )
                print(
                    f"[{trial_index}/{total_trials}] {label} {scenario.scenario_id} "
                    f"repetition {repetition}",
                    flush=True,
                )
                process = subprocess.run(command, check=False)
                error: str | None = None
                artifact: dict[str, Any] | None = None
                if process.returncode != 0:
                    error = f"runner exited {process.returncode}"
                elif not artifact_path.is_file():
                    error = "runner did not write an artifact"
                else:
                    try:
                        artifact = validate_trial(artifact_path, scenario)
                    except (checker.ArtifactCheckError, OSError, ValueError) as caught:
                        error = str(caught)
                record = {
                    "target": label,
                    "runtime": target.runtime,
                    "runtime_revision": target.runtime_revision,
                    "scenario": scenario.scenario_id,
                    "repetition": repetition,
                    "artifact": str(artifact_path),
                    "artifact_sha256": (
                        hashlib.sha256(artifact_path.read_bytes()).hexdigest()
                        if artifact_path.is_file()
                        else None
                    ),
                    "command": command,
                    "runner_exit_code": process.returncode,
                    "passed": error is None,
                    "error": error,
                }
                records.append(record)
                if artifact is not None:
                    artifacts[(repetition, scenario.scenario_id, label)] = artifact
                if error is not None and not args.keep_going:
                    stop_requested = True
                    break
                if args.cooldown > 0 and trial_index < total_trials:
                    time.sleep(args.cooldown)
            if stop_requested:
                break
        if stop_requested:
            break

    pair_contracts: list[dict[str, Any]] = []
    if len(targets) == 2:
        for repetition in range(1, args.repetitions + 1):
            for scenario in scenarios:
                left = artifacts.get((repetition, scenario.scenario_id, target_labels[0]))
                right = artifacts.get((repetition, scenario.scenario_id, target_labels[1]))
                if left is None or right is None:
                    continue
                contract = comparator.evaluate_comparison_contract(left, right)
                pair_contracts.append(
                    {
                        "scenario": scenario.scenario_id,
                        "repetition": repetition,
                        **contract,
                    }
                )

    passed = (
        len(records) == total_trials
        and all(record["passed"] for record in records)
        and all(contract["passed"] for contract in pair_contracts)
        and (len(targets) != 2 or len(pair_contracts) == args.repetitions * len(scenarios))
    )
    campaign = {
        "schema_version": SCHEMA_VERSION,
        "created_at": started_at,
        "passed": passed,
        "comparison_contract": contracts[0],
        "methodology": {
            "repetitions": args.repetitions,
            "cooldown_s": args.cooldown,
            "fresh_process_per_trial": True,
            "target_order": "forward on odd repetitions, reverse on even repetitions",
            "reported_stat": "raw trials; aggregate median is computed by the flip report",
        },
        "targets": [
            {
                "label": label,
                "name": target.name,
                "runtime": target.runtime,
                "runtime_revision": target.runtime_revision,
                "path": str(path),
            }
            for label, target, path in zip(target_labels, targets, target_paths, strict=True)
        ],
        "scenarios": [
            {
                "id": scenario.scenario_id,
                "path": str(scenario.path),
                "sha256": hashlib.sha256(scenario.path.read_bytes()).hexdigest(),
            }
            for scenario in scenarios
        ],
        "runs": records,
        "pair_contracts": pair_contracts,
    }
    campaign_path = output_dir / "campaign.json"
    campaign_path.write_text(json.dumps(campaign, indent=2, sort_keys=True) + "\n")
    print(f"campaign {'PASS' if passed else 'FAIL'}: {campaign_path}", flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
