#!/usr/bin/env python3
"""Run the disk-prefix-cache online serving soak bundle.

This script assumes an AX server is already running with the desired runtime
environment, for example `AX_MLX_PREFIX_CACHE_DIR=/tmp/ax-prefix-cache-soak`.
It creates a shared-prefix corpus, runs the online serving benchmark, validates
the route-decision gate, and renders a Markdown report in one run directory.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


DEFAULT_ROUTE_DECISION_KEY = "ax_mlx_prefix_cache_disk_hits"
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class SoakPaths:
    run_dir: Path
    corpus: Path
    artifact: Path
    report: Path
    command_log: Path


@dataclass(frozen=True)
class SoakPlan:
    paths: SoakPaths
    commands: list[list[str]]


def repo_script(name: str) -> str:
    return str(Path(__file__).with_name(name))


def utc_run_id() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d-disk-prefix-serving-soak-%H%M%S")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def run_id_arg(value: str) -> str:
    if not RUN_ID_PATTERN.fullmatch(value):
        raise argparse.ArgumentTypeError(
            "run id must be a single path component using letters, numbers, '.', '_', or '-'"
        )
    if value in {".", ".."}:
        raise argparse.ArgumentTypeError("run id must not be '.' or '..'")
    return value


def route_decision_key_arg(value: str) -> str:
    if not value or "=" in value or "/" in value:
        raise argparse.ArgumentTypeError("route decision key must be non-empty and omit '=' or '/'")
    return value


def non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def ratio_arg(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0 or parsed > 1.0:
        raise argparse.ArgumentTypeError("value must be between 0 and 1")
    return parsed


def optional_positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_plan(args: argparse.Namespace) -> SoakPlan:
    run_dir = Path(args.output_root) / args.run_id
    paths = SoakPaths(
        run_dir=run_dir,
        corpus=run_dir / "corpus.jsonl",
        artifact=run_dir / "artifact.json",
        report=run_dir / "report.md",
        command_log=run_dir / "commands.json",
    )

    warmup_requests = args.warmup_requests if args.warmup_requests is not None else args.prompts
    min_input_tokens_p95 = (
        args.min_input_tokens_p95 if args.min_input_tokens_p95 is not None else args.prefix_tokens
    )
    route_gate = f"{args.route_decision_key}={args.route_decision_min:g}"

    bench_command = [
        sys.executable,
        repo_script("bench_ax_serving.py"),
        "--base-url",
        args.base_url,
        "--model-id",
        args.model_id,
        "--corpus",
        str(paths.corpus),
        "--input-kind",
        "tokens",
        "--warmup-requests",
        str(warmup_requests),
        "--requests",
        str(args.requests),
        "--concurrency",
        str(args.concurrency),
        "--timeout",
        str(args.timeout),
        "--slo-ttft-ms",
        str(args.slo_ttft_ms),
        "--slo-tpot-ms",
        str(args.slo_tpot_ms),
        "--slo-e2e-ms",
        str(args.slo_e2e_ms),
        "--output",
        str(paths.artifact),
    ]
    if args.request_rate_rps is not None:
        bench_command.extend(["--request-rate-rps", str(args.request_rate_rps)])

    check_command = [
        sys.executable,
        repo_script("check_ax_serving_benchmark_artifact.py"),
        str(paths.artifact),
        "--min-requests",
        str(args.requests),
        "--min-concurrency",
        str(args.concurrency),
        "--require-slo",
        "--min-input-tokens-p95",
        str(min_input_tokens_p95),
        "--require-route-decision-min",
        route_gate,
    ]
    render_command = [
        sys.executable,
        repo_script("render_ax_serving_benchmark_report.py"),
        str(paths.artifact),
        "--output",
        str(paths.report),
        "--min-requests",
        str(args.requests),
        "--min-concurrency",
        str(args.concurrency),
        "--require-slo",
        "--min-input-tokens-p95",
        str(min_input_tokens_p95),
        "--require-route-decision-min",
        route_gate,
    ]
    if args.min_goodput_ratio is not None:
        check_command.extend(["--min-goodput-ratio", str(args.min_goodput_ratio)])
        render_command.extend(["--min-goodput-ratio", str(args.min_goodput_ratio)])

    commands = [
        [
            sys.executable,
            repo_script("build_serving_shared_prefix_corpus.py"),
            "--output",
            str(paths.corpus),
            "--prompts",
            str(args.prompts),
            "--prefix-tokens",
            str(args.prefix_tokens),
            "--suffix-tokens",
            str(args.suffix_tokens),
            "--max-output-tokens",
            str(args.max_output_tokens),
        ],
        bench_command,
        check_command,
        render_command,
    ]
    return SoakPlan(paths=paths, commands=commands)


def shell_join(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def ensure_run_dir_is_new(run_dir: Path) -> None:
    if run_dir.exists() and not run_dir.is_dir():
        raise RuntimeError(f"run output path exists and is not a directory: {run_dir}")
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(
            f"run directory already contains files: {run_dir}. Use a new --run-id for auditable evidence."
        )


def write_command_log(plan: SoakPlan, *, dry_run: bool) -> None:
    plan.paths.run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "dry_run": dry_run,
        "run_dir": str(plan.paths.run_dir),
        "corpus": str(plan.paths.corpus),
        "artifact": str(plan.paths.artifact),
        "report": str(plan.paths.report),
        "commands": [{"argv": command, "shell": shell_join(command)} for command in plan.commands],
    }
    plan.paths.command_log.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def run_plan(plan: SoakPlan, *, dry_run: bool) -> int:
    ensure_run_dir_is_new(plan.paths.run_dir)
    write_command_log(plan, dry_run=dry_run)
    for command in plan.commands:
        print(shell_join(command))
        if not dry_run:
            subprocess.run(command, check=True)
    if dry_run:
        print(f"AX disk-prefix serving soak dry run written: {plan.paths.command_log}")
    else:
        print(f"AX disk-prefix serving soak report written: {plan.paths.report}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("benchmarks/results/serving"))
    parser.add_argument("--run-id", type=run_id_arg, default=utc_run_id())
    parser.add_argument("--prompts", type=positive_int, default=8)
    parser.add_argument("--prefix-tokens", type=positive_int, default=8192)
    parser.add_argument("--suffix-tokens", type=positive_int, default=64)
    parser.add_argument("--max-output-tokens", type=positive_int, default=64)
    parser.add_argument("--warmup-requests", type=positive_int)
    parser.add_argument("--requests", type=positive_int, default=24)
    parser.add_argument("--concurrency", type=positive_int, default=2)
    parser.add_argument("--request-rate-rps", type=optional_positive_float)
    parser.add_argument("--timeout", type=optional_positive_float, default=600.0)
    parser.add_argument("--slo-ttft-ms", type=optional_positive_float, default=10000.0)
    parser.add_argument("--slo-tpot-ms", type=optional_positive_float, default=250.0)
    parser.add_argument("--slo-e2e-ms", type=optional_positive_float, default=60000.0)
    parser.add_argument("--min-goodput-ratio", type=ratio_arg)
    parser.add_argument("--min-input-tokens-p95", type=positive_int)
    parser.add_argument(
        "--route-decision-key",
        type=route_decision_key_arg,
        default=DEFAULT_ROUTE_DECISION_KEY,
    )
    parser.add_argument("--route-decision-min", type=non_negative_float, default=1.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    plan = build_plan(args)
    try:
        return run_plan(plan, dry_run=args.dry_run)
    except RuntimeError as error:
        print(f"AX disk-prefix serving soak failed: {error}", file=sys.stderr)
        return 1


def main_with_args_for_test(argv: list[str]) -> int:
    return main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
