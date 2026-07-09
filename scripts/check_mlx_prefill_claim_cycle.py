#!/usr/bin/env python3
"""Run the current W0-W4 MLX prefill claim-cycle gates."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ClaimCheck:
    name: str
    command: list[str] | None
    skip_reason: str | None = None


def _optional_artifact_check(
    *,
    scripts: Path,
    name: str,
    artifact: Path | None,
    checker: str,
    extra_args: list[str] | None = None,
) -> ClaimCheck:
    if artifact is None:
        return ClaimCheck(
            name=name,
            command=None,
            skip_reason="no current artifact provided",
        )
    return ClaimCheck(
        name=name,
        command=[
            sys.executable,
            str(scripts / checker),
            str(artifact),
            *(extra_args or []),
        ],
    )


def default_checks(
    repo_root: Path,
    *,
    prefill_scaling_artifact: Path | None = None,
    concurrent_prefill_artifact: Path | None = None,
    forward_profile_artifact: Path | None = None,
) -> list[ClaimCheck]:
    scripts = repo_root / "scripts"
    return [
        ClaimCheck(
            name="W0-W3 README table and narrative claim gate",
            command=[
                sys.executable,
                str(scripts / "check_readme_performance_artifacts.py"),
            ],
        ),
        _optional_artifact_check(
            scripts=scripts,
            name="W1 long-context prefill boundary",
            artifact=prefill_scaling_artifact,
            checker="check_mlx_prefill_scaling_artifact.py",
        ),
        _optional_artifact_check(
            scripts=scripts,
            name="W3 concurrent-prefill boundary",
            artifact=concurrent_prefill_artifact,
            checker="check_mlx_concurrent_prefill_artifact.py",
            extra_args=[
                "--min-concurrency-levels",
                "3",
                "--min-max-concurrent-requests",
                "4",
                "--allow-missing-scheduler-evidence",
            ],
        ),
        _optional_artifact_check(
            scripts=scripts,
            name="W4 forward-profile diagnostic boundary",
            artifact=forward_profile_artifact,
            checker="check_mlx_forward_profile_artifact.py",
        ),
    ]


def run_check(check: ClaimCheck, *, cwd: Path) -> tuple[bool, str]:
    if check.command is None:
        return True, check.skip_reason or "skipped"
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
    parser.add_argument(
        "--prefill-scaling-artifact",
        type=Path,
        help="Current W1 prefill-scaling artifact to validate.",
    )
    parser.add_argument(
        "--concurrent-prefill-artifact",
        type=Path,
        help="Current W3 concurrent-prefill artifact to validate.",
    )
    parser.add_argument(
        "--forward-profile-artifact",
        type=Path,
        help="Current W4 forward-profile artifact to validate.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    checks = default_checks(
        repo_root,
        prefill_scaling_artifact=args.prefill_scaling_artifact,
        concurrent_prefill_artifact=args.concurrent_prefill_artifact,
        forward_profile_artifact=args.forward_profile_artifact,
    )
    failures = 0
    skipped = 0
    for check in checks:
        if check.command is None:
            skipped += 1
            print(f"[SKIP] {check.name}")
            if check.skip_reason:
                print(check.skip_reason)
            continue
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
    passed = len(checks) - skipped
    print(
        f"MLX prefill claim-cycle gate passed: {passed} checks"
        f" ({skipped} optional boundaries skipped)"
    )
    return 0


def main_with_args_for_test(argv: list[str]) -> int:
    return main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
