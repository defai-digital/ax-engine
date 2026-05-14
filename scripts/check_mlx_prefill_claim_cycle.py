#!/usr/bin/env python3
"""Run the checked-in W0-W4 MLX prefill claim-cycle gates."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


DEFAULT_P1_PREFILL_SCALING = Path(
    "benchmarks/results/mlx-inference/2026-05-07-real-p1/"
    "qwen3-4b-4bit-prefill-scaling/prefill-scaling.json"
)
DEFAULT_P2_CONCURRENT_PREFILL = Path(
    "benchmarks/results/mlx-inference/2026-05-07-real-p2/"
    "qwen3-4b-4bit-p2-latency/concurrent-prefill.json"
)
DEFAULT_W4_FORWARD_PROFILE = Path(
    "benchmarks/results/mlx-inference/2026-05-13-ttft-breakdown/"
    "qwen3_6-35b-a3b-8bit-linear-profile-prefill.json"
)


@dataclass(frozen=True)
class ClaimCheck:
    name: str
    command: list[str]


def default_checks(repo_root: Path) -> list[ClaimCheck]:
    scripts = repo_root / "scripts"
    return [
        ClaimCheck(
            name="W0-W3 README table and narrative claim gate",
            command=[
                sys.executable,
                str(scripts / "check_readme_performance_artifacts.py"),
            ],
        ),
        ClaimCheck(
            name="W1 long-context prefill boundary",
            command=[
                sys.executable,
                str(scripts / "check_mlx_prefill_scaling_artifact.py"),
                str(repo_root / DEFAULT_P1_PREFILL_SCALING),
            ],
        ),
        ClaimCheck(
            name="W3 concurrent-prefill boundary",
            command=[
                sys.executable,
                str(scripts / "check_mlx_concurrent_prefill_artifact.py"),
                str(repo_root / DEFAULT_P2_CONCURRENT_PREFILL),
                "--min-concurrency-levels",
                "3",
                "--min-max-concurrent-requests",
                "4",
                "--allow-missing-scheduler-evidence",
            ],
        ),
        ClaimCheck(
            name="W4 forward-profile diagnostic boundary",
            command=[
                sys.executable,
                str(scripts / "check_mlx_forward_profile_artifact.py"),
                str(repo_root / DEFAULT_W4_FORWARD_PROFILE),
            ],
        ),
    ]


def run_check(check: ClaimCheck, *, cwd: Path) -> tuple[bool, str]:
    completed = subprocess.run(
        check.command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    output = "\n".join(
        part.strip()
        for part in (completed.stdout, completed.stderr)
        if part.strip()
    )
    return completed.returncode == 0, output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    failures = 0
    for check in default_checks(repo_root):
        ok, output = run_check(check, cwd=repo_root)
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {check.name}")
        if output:
            print(output)
        if not ok:
            failures += 1
    if failures:
        print(f"MLX prefill claim-cycle gate failed: {failures} check(s) failed")
        return 1
    print("MLX prefill claim-cycle gate passed: 4 checks")
    return 0


def main_with_args_for_test(argv: list[str]) -> int:
    return main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
