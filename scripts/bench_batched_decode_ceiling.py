#!/usr/bin/env python3
"""Benchmark Shared versus RowExact batched decode with publication evidence.

The Rust probe reports internally timed aggregate decode throughput for batches
1, 2, 4, and 8. This wrapper runs both projection policies in alternating
order, preserves raw logs, records host/build conditions, verifies full-cohort
greedy-token hashes, and emits a fail-closed JSON artifact plus Markdown
summary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from scripts import bench_mlx_inference_stack as bench_support
    from scripts import bench_mtp_6bit_ax_refresh as condition_gate
except ModuleNotFoundError:
    import bench_mlx_inference_stack as bench_support
    import bench_mtp_6bit_ax_refresh as condition_gate

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROBE = REPO_ROOT / "target" / "release" / "batched-decode-ceiling-probe"
SCHEMA_VERSION = "ax.batched_decode_ceiling.v1"
INCOMPLETE_SCHEMA_VERSION = "ax.batched_decode_ceiling.incomplete.v1"
POLICIES = ("shared", "row_exact")
EXPECTED_BATCHES = (1, 2, 4, 8)
PROBE_WARMUP_STEPS_PER_BATCH = 8
PROBE_MEASURED_STEPS_PER_BATCH = 64
MIN_REPETITIONS = 5
MIN_COOLDOWN_SECONDS = 15.0
DEFAULT_MAX_LOAD_AVERAGE = 2.0
DEFAULT_MAX_TOP_PROCESS_CPU_PERCENT = 50.0
PROBE_ROW_RE = re.compile(
    r"^\s*(?P<batch>\d+)\s+"
    r"(?P<agg>\d+(?:\.\d+)?)\s+"
    r"(?P<per_req>\d+(?:\.\d+)?)\s+"
    r"(?P<step_us>\d+(?:\.\d+)?)\s+"
    r"(?P<scaling>\d+(?:\.\d+)?)x\s+"
    r"cohort_fnv=(?P<cohort>[0-9a-f]{16})\s*$",
    re.MULTILINE,
)


class BatchedDecodeBenchmarkError(RuntimeError):
    pass


def repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_probe_output(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    observed_batches: set[int] = set()
    for match in PROBE_ROW_RE.finditer(text):
        batch = int(match.group("batch"))
        if batch in observed_batches:
            raise BatchedDecodeBenchmarkError(
                f"probe output contains duplicate batch={batch}"
            )
        observed_batches.add(batch)
        agg = float(match.group("agg"))
        per_req = float(match.group("per_req"))
        step_us = float(match.group("step_us"))
        scaling = float(match.group("scaling"))
        if not all(
            math.isfinite(value) and value > 0.0
            for value in (agg, per_req, step_us, scaling)
        ):
            raise BatchedDecodeBenchmarkError(
                f"probe output has a non-positive metric for batch={batch}"
            )
        expected_agg = batch * 1_000_000.0 / step_us
        if not math.isclose(agg, expected_agg, rel_tol=0.02):
            raise BatchedDecodeBenchmarkError(
                f"probe output has inconsistent aggregate rate for batch={batch}"
            )
        if not math.isclose(per_req, agg / batch, rel_tol=0.02):
            raise BatchedDecodeBenchmarkError(
                f"probe output has inconsistent per-request rate for batch={batch}"
            )
        rows.append(
            {
                "batch": batch,
                "aggregate_tok_s": agg,
                "per_request_tok_s": per_req,
                "step_us": step_us,
                "scaling_vs_batch1": scaling,
                "cohort_fnv": match.group("cohort"),
            }
        )
    if observed_batches != set(EXPECTED_BATCHES):
        raise BatchedDecodeBenchmarkError(
            "probe output batch matrix mismatch: "
            f"expected={list(EXPECTED_BATCHES)} observed={sorted(observed_batches)}"
        )
    rows.sort(key=lambda row: int(row["batch"]))
    batch1 = float(rows[0]["aggregate_tok_s"])
    for row in rows:
        expected_scaling = float(row["aggregate_tok_s"]) / batch1
        if not math.isclose(
            float(row["scaling_vs_batch1"]),
            expected_scaling,
            rel_tol=0.03,
        ):
            raise BatchedDecodeBenchmarkError(
                "probe output has inconsistent scaling for "
                f"batch={row['batch']}"
            )
    return rows


def median(values: list[float]) -> float:
    if not values:
        raise BatchedDecodeBenchmarkError("cannot summarize an empty metric set")
    return float(statistics.median(values))


def summarize_trials(trials: list[dict[str, Any]]) -> dict[str, Any]:
    policy_rows: dict[str, dict[int, list[dict[str, Any]]]] = {
        policy: {batch: [] for batch in EXPECTED_BATCHES}
        for policy in POLICIES
    }
    by_rep: dict[int, dict[str, dict[int, dict[str, Any]]]] = {}
    for trial in trials:
        policy = str(trial["policy"])
        repetition = int(trial["repetition"])
        by_rep.setdefault(repetition, {}).setdefault(policy, {})
        for row in trial["rows"]:
            batch = int(row["batch"])
            policy_rows[policy][batch].append(row)
            by_rep[repetition][policy][batch] = row

    policies: dict[str, Any] = {}
    for policy in POLICIES:
        batches: dict[str, Any] = {}
        for batch in EXPECTED_BATCHES:
            rows = policy_rows[policy][batch]
            aggregate = [float(row["aggregate_tok_s"]) for row in rows]
            step_us = [float(row["step_us"]) for row in rows]
            batches[str(batch)] = {
                "median_aggregate_tok_s": median(aggregate),
                "min_aggregate_tok_s": min(aggregate),
                "max_aggregate_tok_s": max(aggregate),
                "median_step_us": median(step_us),
                "cohort_fnv": sorted(
                    {str(row["cohort_fnv"]) for row in rows}
                ),
            }
        b1 = float(batches["1"]["median_aggregate_tok_s"])
        for batch in EXPECTED_BATCHES:
            batches[str(batch)]["median_scaling_vs_batch1"] = (
                float(batches[str(batch)]["median_aggregate_tok_s"]) / b1
            )
        policies[policy] = {"batches": batches}

    paired_batch8_ratios = []
    for repetition in sorted(by_rep):
        rep = by_rep[repetition]
        if set(rep) != set(POLICIES):
            continue
        shared = float(rep["shared"][8]["aggregate_tok_s"])
        row_exact = float(rep["row_exact"][8]["aggregate_tok_s"])
        paired_batch8_ratios.append(shared / row_exact)
    shared_b8 = float(
        policies["shared"]["batches"]["8"]["median_aggregate_tok_s"]
    )
    row_exact_b8 = float(
        policies["row_exact"]["batches"]["8"]["median_aggregate_tok_s"]
    )
    ties = sum(math.isclose(ratio, 1.0) for ratio in paired_batch8_ratios)
    return {
        "policies": policies,
        "paired_batch8_shared_over_row_exact": {
            "ratios": paired_batch8_ratios,
            "median_ratio": median(paired_batch8_ratios),
            "wins": sum(
                ratio > 1.0 and not math.isclose(ratio, 1.0)
                for ratio in paired_batch8_ratios
            ),
            "ties": ties,
            "losses": sum(
                ratio < 1.0 and not math.isclose(ratio, 1.0)
                for ratio in paired_batch8_ratios
            ),
        },
        "ratio_of_batch8_medians": shared_b8 / row_exact_b8,
    }


def publication_reasons(artifact: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if artifact.get("schema_version") != SCHEMA_VERSION:
        reasons.append("unexpected_schema")
    if artifact.get("status") != "complete":
        reasons.append("benchmark_incomplete")
    if artifact.get("prefill_len") != 32:
        reasons.append("publication_requires_prefill_len_32")
    if artifact.get("probe_contract") != {
        "batches": list(EXPECTED_BATCHES),
        "warmup_steps_per_batch": PROBE_WARMUP_STEPS_PER_BATCH,
        "measured_steps_per_batch": PROBE_MEASURED_STEPS_PER_BATCH,
        "timing_scope": "internal_batched_decode_step_wall",
    }:
        reasons.append("probe_contract_mismatch")
    repetitions = artifact.get("repetitions")
    if (
        not isinstance(repetitions, int)
        or isinstance(repetitions, bool)
        or repetitions < MIN_REPETITIONS
    ):
        reasons.append("publication_requires_five_repetitions")
        repetitions = 0
    cooldown = artifact.get("cooldown_seconds")
    if (
        not isinstance(cooldown, (int, float))
        or isinstance(cooldown, bool)
        or not math.isfinite(float(cooldown))
        or float(cooldown) < MIN_COOLDOWN_SECONDS
    ):
        reasons.append("publication_requires_15s_cooldown")

    build = artifact.get("build")
    commit = build.get("commit") if isinstance(build, dict) else None
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        reasons.append("missing_full_build_commit")
    if not isinstance(build, dict) or build.get("build_profile") != "release":
        reasons.append("non_release_build")
    if not isinstance(build, dict) or build.get("git_tracked_dirty") is not False:
        reasons.append("dirty_build")
    binary_sha = (
        build.get("benchmark_binary_sha256")
        if isinstance(build, dict)
        else None
    )
    binary_path = (
        build.get("benchmark_binary")
        if isinstance(build, dict)
        else None
    )
    if not isinstance(binary_sha, str) or re.fullmatch(
        r"[0-9a-f]{64}", binary_sha
    ) is None:
        reasons.append("missing_benchmark_binary_hash")
    if not isinstance(binary_path, str) or not binary_path:
        reasons.append("missing_benchmark_binary_path")

    host = artifact.get("host")
    if not isinstance(host, dict) or "Apple" not in str(host.get("chip", "")):
        reasons.append("requires_apple_silicon_host")
    model = artifact.get("model")
    model_path = model.get("path") if isinstance(model, dict) else None
    if (
        not isinstance(model, dict)
        or not isinstance(model.get("manifest_sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", model["manifest_sha256"]) is None
    ):
        reasons.append("missing_model_manifest_hash")

    trials = artifact.get("trials")
    if not isinstance(trials, list):
        return sorted({*reasons, "missing_trials"})
    expected_trial_count = repetitions * len(POLICIES)
    if len(trials) != expected_trial_count:
        reasons.append("trial_count_mismatch")
    keys: set[tuple[int, str]] = set()
    hashes_by_batch: dict[int, set[str]] = {
        batch: set() for batch in EXPECTED_BATCHES
    }
    for trial in trials:
        if not isinstance(trial, dict):
            reasons.append("invalid_trial")
            continue
        repetition = trial.get("repetition")
        policy = trial.get("policy")
        key = (repetition, policy)
        if (
            not isinstance(repetition, int)
            or isinstance(repetition, bool)
            or not 1 <= repetition <= repetitions
            or policy not in POLICIES
            or key in keys
        ):
            reasons.append("invalid_or_duplicate_trial_identity")
            continue
        keys.add(key)
        expected_policy_value = "1" if policy == "shared" else "0"
        if trial.get("environment") != {
            "AX_MLX_BATCHED_SHARED_PROJ": expected_policy_value,
            "AX_MLX_BATCHED_PROFILE": "0",
        }:
            reasons.append(f"rep{repetition}_{policy}_environment_mismatch")
        if trial.get("command") != [
            binary_path,
            model_path,
            str(artifact.get("prefill_len")),
        ]:
            reasons.append(f"rep{repetition}_{policy}_command_mismatch")
        fake_artifact = {
            "benchmark_window": {
                "performance_conditions_start": trial.get(
                    "performance_conditions_start"
                ),
                "performance_conditions_end": trial.get(
                    "performance_conditions_end"
                ),
            }
        }
        reasons.extend(
            condition_gate.publication_condition_reasons(
                f"rep{repetition}_{policy}",
                fake_artifact,
            )
        )
        rows = trial.get("rows")
        if not isinstance(rows, list):
            reasons.append(f"rep{repetition}_{policy}_missing_rows")
            continue
        try:
            parsed_batches = {int(row["batch"]) for row in rows}
        except (KeyError, TypeError, ValueError):
            reasons.append(f"rep{repetition}_{policy}_invalid_rows")
            continue
        if parsed_batches != set(EXPECTED_BATCHES):
            reasons.append(f"rep{repetition}_{policy}_batch_matrix_mismatch")
            continue
        for row in rows:
            batch = int(row["batch"])
            cohort = row.get("cohort_fnv")
            metrics = (
                row.get("aggregate_tok_s"),
                row.get("per_request_tok_s"),
                row.get("step_us"),
                row.get("scaling_vs_batch1"),
            )
            if (
                not isinstance(cohort, str)
                or re.fullmatch(r"[0-9a-f]{16}", cohort) is None
                or any(
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not math.isfinite(float(value))
                    or float(value) <= 0.0
                    for value in metrics
                )
            ):
                reasons.append(f"rep{repetition}_{policy}_invalid_batch{batch}")
                continue
            hashes_by_batch[batch].add(cohort)
    expected_keys = {
        (repetition, policy)
        for repetition in range(1, repetitions + 1)
        for policy in POLICIES
    }
    if keys != expected_keys:
        reasons.append("repetition_policy_matrix_mismatch")
    for batch, hashes in hashes_by_batch.items():
        if len(hashes) != 1:
            reasons.append(f"batch{batch}_cohort_hash_divergence")
    return sorted(set(reasons))


def render_summary(artifact: dict[str, Any]) -> str:
    summary = artifact["summary"]
    lines = [
        "# Batched decode ceiling: Shared vs RowExact",
        "",
        f"- Host: {artifact['host'].get('chip', 'unknown')}",
        f"- Engine: {artifact['build'].get('engine_version', 'unknown')}",
        f"- Commit: `{artifact['build'].get('commit', 'unknown')}`",
        f"- Repetitions: {artifact['repetitions']} per policy",
        (
            "- Publication candidate: "
            f"{str(artifact['publication_candidate']).lower()}"
        ),
        "",
        "| Policy | Batch | Median agg tok/s | Median per-policy scaling | "
        "Median step µs |",
        "|---|---:|---:|---:|---:|",
    ]
    for policy in POLICIES:
        for batch in EXPECTED_BATCHES:
            row = summary["policies"][policy]["batches"][str(batch)]
            lines.append(
                f"| {policy} | {batch} | "
                f"{row['median_aggregate_tok_s']:.1f} | "
                f"{row['median_scaling_vs_batch1']:.2f}× | "
                f"{row['median_step_us']:.0f} |"
            )
    paired = summary["paired_batch8_shared_over_row_exact"]
    lines.extend(
        [
            "",
            (
                "At batch 8, Shared / RowExact has a paired median ratio of "
                f"**{paired['median_ratio']:.2f}×** "
                f"({paired['wins']} wins, {paired['ties']} ties, "
                f"{paired['losses']} losses)."
            ),
        ]
    )
    if artifact["publication_reasons"]:
        lines.extend(
            [
                "",
                "Publication blockers: "
                + ", ".join(artifact["publication_reasons"]),
            ]
        )
    return "\n".join(lines) + "\n"


def write_artifact(path: Path, artifact: dict[str, Any]) -> None:
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")


def policy_environment(policy: str) -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("AX_MLX_")
    }
    env["AX_MLX_BATCHED_SHARED_PROJ"] = "1" if policy == "shared" else "0"
    env["AX_MLX_BATCHED_PROFILE"] = "0"
    return env


def run_probe(
    *,
    probe: Path,
    model_dir: Path,
    prefill_len: int,
    policy: str,
    repetition: int,
    log_path: Path,
    max_load_average: float,
    max_top_process_cpu_percent: float,
    load_wait_timeout: float,
    load_poll_interval: float,
) -> dict[str, Any]:
    context = f"repetition {repetition} {policy}"
    conditions_start = bench_support.wait_for_performance_load(
        max_one_minute=max_load_average,
        max_top_process_cpu_percent_value=max_top_process_cpu_percent,
        timeout_seconds=load_wait_timeout,
        poll_interval_seconds=load_poll_interval,
        context=context,
    )
    command = [str(probe), str(model_dir), str(prefill_len)]
    env = policy_environment(policy)
    started = time.perf_counter()
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    elapsed = time.perf_counter() - started
    conditions_end = bench_support.collect_performance_condition_metadata()
    log_path.write_text(
        f"AX_MLX_BATCHED_SHARED_PROJ={env['AX_MLX_BATCHED_SHARED_PROJ']}\n"
        f"AX_MLX_BATCHED_PROFILE={env['AX_MLX_BATCHED_PROFILE']}\n"
        f"$ {shlex.join(command)}\n\n"
        f"{result.stdout}\n"
        f"[exit {result.returncode} after {elapsed:.3f}s]\n"
    )
    if result.returncode != 0:
        tail = "\n".join(result.stdout.splitlines()[-60:])
        raise BatchedDecodeBenchmarkError(
            f"probe failed for {context}; see {log_path}\n{tail}"
        )
    rows = parse_probe_output(result.stdout)
    return {
        "repetition": repetition,
        "policy": policy,
        "command": command,
        "environment": {
            "AX_MLX_BATCHED_SHARED_PROJ": env[
                "AX_MLX_BATCHED_SHARED_PROJ"
            ],
            "AX_MLX_BATCHED_PROFILE": env["AX_MLX_BATCHED_PROFILE"],
        },
        "elapsed_seconds": elapsed,
        "raw_log": repo_relative(log_path),
        "performance_conditions_start": conditions_start,
        "performance_conditions_end": conditions_end,
        "rows": rows,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--probe", type=Path, default=DEFAULT_PROBE)
    parser.add_argument("--prefill-len", type=int, default=32)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--cooldown", type=float, default=15.0)
    parser.add_argument(
        "--max-load-average",
        type=float,
        default=DEFAULT_MAX_LOAD_AVERAGE,
    )
    parser.add_argument(
        "--max-top-process-cpu-percent",
        type=float,
        default=DEFAULT_MAX_TOP_PROCESS_CPU_PERCENT,
    )
    parser.add_argument("--load-wait-timeout", type=float, default=900.0)
    parser.add_argument("--load-poll-interval", type=float, default=5.0)
    parser.add_argument("--no-build", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    for name in (
        "cooldown",
        "max_load_average",
        "max_top_process_cpu_percent",
        "load_wait_timeout",
        "load_poll_interval",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0.0:
            parser.error(f"--{name.replace('_', '-')} must be finite and non-negative")
    if args.load_poll_interval <= 0.0:
        parser.error("--load-poll-interval must be greater than zero")
    if args.prefill_len <= 0:
        parser.error("--prefill-len must be positive")
    if args.repetitions <= 0:
        parser.error("--repetitions must be positive")

    model_dir = args.model_dir.resolve()
    manifest_path = model_dir / "model-manifest.json"
    if not manifest_path.is_file():
        parser.error(f"model-manifest.json not found: {manifest_path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = args.output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = args.output_dir / "batched_decode_ceiling.json"
    summary_path = args.output_dir / "summary.md"
    artifact: dict[str, Any] = {
        "schema_version": INCOMPLETE_SCHEMA_VERSION,
        "status": "running",
        "started_at": datetime.now().astimezone().isoformat(),
        "model": {
            "path": str(model_dir),
            "manifest_sha256": sha256_file(manifest_path),
        },
        "prefill_len": args.prefill_len,
        "probe_contract": {
            "batches": list(EXPECTED_BATCHES),
            "warmup_steps_per_batch": PROBE_WARMUP_STEPS_PER_BATCH,
            "measured_steps_per_batch": PROBE_MEASURED_STEPS_PER_BATCH,
            "timing_scope": "internal_batched_decode_step_wall",
        },
        "repetitions": args.repetitions,
        "cooldown_seconds": args.cooldown,
        "max_load_average": args.max_load_average,
        "max_top_process_cpu_percent": args.max_top_process_cpu_percent,
        "trials": [],
    }
    write_artifact(artifact_path, artifact)

    try:
        if not args.no_build:
            subprocess.run(
                [
                    "cargo",
                    "build",
                    "--release",
                    "-p",
                    "ax-engine-microbench",
                    "--bin",
                    "batched-decode-ceiling-probe",
                ],
                cwd=REPO_ROOT,
                check=True,
            )
        probe = args.probe.resolve()
        if not probe.is_file():
            raise BatchedDecodeBenchmarkError(
                f"benchmark probe binary not found: {probe}"
            )
        build = bench_support.collect_build_metadata()
        build["benchmark_binary"] = str(probe)
        build["benchmark_binary_sha256"] = sha256_file(probe)
        artifact["build"] = build
        artifact["host"] = bench_support.collect_host_metadata()

        for repetition in range(1, args.repetitions + 1):
            order = POLICIES if repetition % 2 == 1 else tuple(reversed(POLICIES))
            for policy in order:
                if args.cooldown > 0.0:
                    time.sleep(args.cooldown)
                print(
                    f"[run] repetition={repetition} policy={policy}",
                    flush=True,
                )
                trial = run_probe(
                    probe=probe,
                    model_dir=model_dir,
                    prefill_len=args.prefill_len,
                    policy=policy,
                    repetition=repetition,
                    log_path=logs_dir / f"rep-{repetition:02d}-{policy}.log",
                    max_load_average=args.max_load_average,
                    max_top_process_cpu_percent=args.max_top_process_cpu_percent,
                    load_wait_timeout=args.load_wait_timeout,
                    load_poll_interval=args.load_poll_interval,
                )
                artifact["trials"].append(trial)
                write_artifact(artifact_path, artifact)

        artifact["schema_version"] = SCHEMA_VERSION
        artifact["status"] = "complete"
        artifact["completed_at"] = datetime.now().astimezone().isoformat()
        artifact["summary"] = summarize_trials(artifact["trials"])
        artifact["publication_reasons"] = publication_reasons(artifact)
        artifact["publication_candidate"] = not artifact["publication_reasons"]
        write_artifact(artifact_path, artifact)
        summary_path.write_text(render_summary(artifact))
    except Exception as error:
        artifact["status"] = "failed"
        artifact["failure"] = f"{type(error).__name__}: {error}"
        write_artifact(artifact_path, artifact)
        raise

    print(f"Wrote {artifact_path}")
    print(f"Wrote {summary_path}")
    if not artifact["publication_candidate"]:
        for reason in artifact["publication_reasons"]:
            print(f"publication blocker: {reason}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
