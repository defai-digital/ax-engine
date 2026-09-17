#!/usr/bin/env python3
"""Qwen 3.8 27B primary qualification contract.

`--dry-run` prints the pinned pack and required commands (CI-safe, no weights).
A live `--model-dir` run belongs on Mac mini M4 Pro 64 GB with a clean checkout.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

PRIMARY_ALIAS = "qwen3.8-27b:axq"
PRIMARY_REPO = "AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-6bit-MTP"
PRIMARY_REVISION = "3e290738e96972307c6aeb9934ab170ca0eae1c1"
HOST_CLASS = "Mac mini M4 Pro, 64 GB"
STATUS_SENTENCE = (
    "Primary optimization target. Checkpoint Tier 1. MTP Tier 2 pending. "
    "AX certification record: Candidate (gates open)."
)


def contract() -> dict[str, Any]:
    return {
        "alias": PRIMARY_ALIAS,
        "repo_id": PRIMARY_REPO,
        "revision": PRIMARY_REVISION,
        "model_manifest_sha256": "621470389598a8042634f1b65c71f0dd5f02ae47b44ed9e8080083e28725cc26",
        "host_class": HOST_CLASS,
        "status": STATUS_SENTENCE,
        "ci": "dry-run only; do not mount 27B weights on CI",
        "release_blocking": [
            "ax-engine doctor ready on the pinned snapshot",
            "QA surface: direct + MTP for qwen3.8-27b",
            "no silent MTP direct-fallback",
            "clean worktree + matching engine commit",
        ],
        "campaign_only": [
            "MTP Tier 2 promotion",
            "8h/72h endurance",
            "long-context decode-at-depth",
            "peer ranking",
            "multi-model add-mode",
            "P0 multimodal quality",
        ],
        "commands": {
            "doctor": "ax-engine doctor --mlx-model-artifacts-dir $MODEL_DIR --json",
            "serve": f"ax-engine serve {PRIMARY_ALIAS}",
            "qa_inventory": (
                f"OK|direct|{PRIMARY_ALIAS}|$MODEL_DIR\n"
                f"OK|mtp|{PRIMARY_ALIAS}|$MODEL_DIR"
            ),
            "stack": (
                "python3 scripts/bench_mlx_inference_stack.py "
                f"--model-repo-id {PRIMARY_REPO} --model-dir $MODEL_DIR"
            ),
        },
    }


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the qualification contract and exit (CI default)",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        help="pinned snapshot directory for a live operator run",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit the contract as JSON",
    )
    parser.add_argument("--run", action="store_true", help="execute release gates; requires build provenance")
    for name in ("output", "build-manifest", "server-bin", "bench-bin", "wheel", "cli"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--port", type=int, default=31494)
    args = parser.parse_args(argv)
    if args.run:
        if args.dry_run or any(getattr(args, name) is None for name in
                              ("model_dir", "output", "build_manifest", "server_bin", "bench_bin", "wheel", "cli")):
            parser.error("--run requires model-dir, output, build-manifest, server-bin, bench-bin, wheel and cli; no dry-run")
    return args


def _print_contract(as_json: bool) -> None:
    payload = contract()
    if as_json:
        json.dump(payload, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return
    print(payload["status"])
    print(f"alias: {payload['alias']}")
    print(f"repo:  {payload['repo_id']}")
    print(f"rev:   {payload['revision']}")
    print(f"host:  {payload['host_class']}")
    print(f"ci:    {payload['ci']}")
    print("release-blocking:")
    for item in payload["release_blocking"]:
        print(f"  - {item}")
    print("campaign-only:")
    for item in payload["campaign_only"]:
        print(f"  - {item}")
    print("commands:")
    for name, command in payload["commands"].items():
        print(f"  {name}: {command}")


def _live_preflight(model_dir: Path) -> None:
    if not model_dir.is_dir():
        raise SystemExit(f"model dir is not a directory: {model_dir}")
    snapshot_name = model_dir.name
    if snapshot_name != PRIMARY_REVISION and PRIMARY_REVISION not in str(model_dir):
        raise SystemExit(
            f"model dir does not look like pinned revision {PRIMARY_REVISION}: "
            f"{model_dir}"
        )
    config = model_dir / "config.json"
    if not config.is_file():
        raise SystemExit(f"missing config.json under {model_dir}")
    print(f"live preflight ok: {model_dir}")
    print("next: ax-engine doctor, then QA surface direct+mtp on this snapshot")
    print("preflight only: NOT qualified; use --run with build provenance to execute gates")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.run:
        from qwen38_live_gate import run_live
        return run_live(args, contract(), Path(__file__).resolve().parents[1])
    if args.model_dir is not None and not args.dry_run:
        _live_preflight(args.model_dir)
        if args.json:
            _print_contract(True)
        return 0
    _print_contract(args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
