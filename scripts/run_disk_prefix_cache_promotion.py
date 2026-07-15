#!/usr/bin/env python3
"""Orchestrator scaffold for durable tiered prefix-cache promotion evidence.

Produces a schema-valid ``ax.disk_prefix_cache_promotion.v2`` artifact directory.
Full model matrix execution requires Apple Silicon + configured models; by default
this tool emits an *incomplete* scaffold that the fail-closed checker accepts only
with ``--allow-incomplete``.

When ``--execute`` is set, the script shells out to the existing cross-restart
correctness harness per model (if artifacts dirs exist) and records commands.
It does **not** invent performance numbers.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


SCHEMA = "ax.disk_prefix_cache_promotion.v2"
ROOT = Path(__file__).resolve().parents[1]


def git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return out.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return bool(out.strip())
    except (OSError, subprocess.CalledProcessError):
        return False


def empty_mode(name: str, *, fs_state: str | None = None) -> dict:
    row: dict = {
        "name": name,
        "status": "not_run",
        "p50_ttft_ms": None,
        "p95_ttft_ms": None,
        "p99_ttft_ms": None,
        "p95_tpot_ms": None,
        "notes": "scaffold; run with --execute on supported hardware",
    }
    if fs_state is not None:
        row["filesystem_cache_state"] = fs_state
        row["filesystem_cache_method"] = None
    return row


def build_scaffold(
    *,
    run_id: str,
    models: list[str],
    execute_notes: list[str],
) -> dict:
    return {
        "schema": SCHEMA,
        "run_id": run_id,
        "status": "incomplete",
        "ax_commit": git_commit(),
        "ax_dirty": git_dirty(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
        },
        "models": models,
        "prefix_buckets": ["256", "2k", "8k", "32k", "gte_32k"],
        "modes": {
            "cold_prefill": empty_mode("cold_prefill"),
            "l1_hit": empty_mode("l1_hit"),
            "l2_hit_warm_fs": empty_mode("l2_hit_warm_fs", fs_state="warm_fs"),
            "l2_hit_cold_fs": empty_mode("l2_hit_cold_fs", fs_state="unknown"),
            "producer_l2_enabled": empty_mode("producer_l2_enabled"),
        },
        "correctness": {
            "deterministic_match": False,
            "wrong_prefix_hits": 0,
            "corrupt_restores": 0,
            "notes": "not yet evaluated in this scaffold run",
        },
        "performance_gates": {
            "decision": "not_promoted",
            "reason": "incomplete scaffold; no measured TTFT curves",
            "admitted_bucket_improvements": [],
        },
        "execute_notes": execute_notes,
        "commands": [],
    }


def try_execute_correctness(models: list[str], out_dir: Path) -> list[str]:
    """Best-effort: invoke cross-restart harness when model dirs exist."""
    notes: list[str] = []
    harness = ROOT / "scripts" / "verify_disk_prefix_cache_cross_restart.py"
    if not harness.is_file():
        notes.append("verify_disk_prefix_cache_cross_restart.py missing")
        return notes
    for model_id in models:
        # Conventional local layout; skip quietly when absent.
        candidates = [
            ROOT / ".internal" / "models" / model_id,
            ROOT / "models" / model_id,
        ]
        model_dir = next((p for p in candidates if p.is_dir()), None)
        if model_dir is None:
            notes.append(f"skip {model_id}: model dir not found")
            continue
        artifact = out_dir / f"cross-restart-{model_id}.json"
        cmd = [
            sys.executable,
            str(harness),
            "--model-id",
            model_id,
            "--mlx-artifacts-dir",
            str(model_dir),
            "--output",
            str(artifact),
        ]
        notes.append("run: " + " ".join(cmd))
        try:
            proc = subprocess.run(
                cmd,
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=3600,
                check=False,
            )
            notes.append(f"{model_id}: exit={proc.returncode}")
            if proc.returncode != 0:
                notes.append((proc.stderr or proc.stdout or "")[:500])
        except (OSError, subprocess.TimeoutExpired) as exc:
            notes.append(f"{model_id}: failed: {exc}")
    return notes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for run artifacts (created)",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=["gemma-4-e2b-it-4bit", "qwen3.5-9b", "glm-4.7-flash"],
        help="Model ids for the promotion matrix",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Attempt correctness harness when local model dirs exist",
    )
    args = parser.parse_args(argv)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.output_dir or (
        ROOT
        / "benchmarks"
        / "results"
        / "profiling"
        / "disk-prefix-cache-promotion"
        / run_id
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    notes: list[str] = [
        "Scaffold only: full cold/warm FS + TTFT matrix is not automated here yet.",
        "Use check_disk_prefix_cache_promotion.py --allow-incomplete on the decision JSON.",
    ]
    if args.execute:
        notes.extend(try_execute_correctness(list(args.models), out_dir))

    artifact = build_scaffold(run_id=run_id, models=list(args.models), execute_notes=notes)
    decision_path = out_dir / "decision.json"
    decision_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = out_dir / "SUMMARY.md"
    summary.write_text(
        "\n".join(
            [
                f"# Disk prefix-cache promotion run `{run_id}`",
                "",
                f"- status: **{artifact['status']}**",
                f"- commit: `{artifact['ax_commit']}` dirty={artifact['ax_dirty']}",
                f"- models: {', '.join(args.models)}",
                "",
                "## Notes",
                "",
                *[f"- {n}" for n in notes],
                "",
                "This run does **not** authorize a public performance claim.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(out_dir), "decision": str(decision_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
