#!/usr/bin/env python3
"""Qwen 3.8 Flash Next incubating contract.

`--dry-run` prints the SKU and fail-closed rule (CI-safe, no weights).
A live `--model-dir` run is expected to fail until a repo-owned graph exists.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

FAMILY = "qwen4_exp"
HOST_CLASS = "Mac Studio M5 Ultra, 256 GB"
STATUS = (
    "Incubating. Convert maps qwen4_exp metadata; load/serve stay fail-closed "
    "(qwen4_exp_native_trunk_not_implemented). Not Qwen 3.8 27B and not Super-class 2.4T."
)


def contract() -> dict[str, Any]:
    return {
        "family": FAMILY,
        "aliases": [
            "qwen4_exp",
            "qwen3.8-flash-next",
            "qwen38_flash_next",
        ],
        "host_class": HOST_CLASS,
        "status": STATUS,
        "ci": "dry-run only; do not mount Flash Next weights on CI",
        "fail_closed": True,
        "convert": "metadata mapping; runtime_status.ready=false",
        "load_blocker": "qwen4_exp_native_trunk_not_implemented",
        "not": [
            "qwen3.8-27b:axq",
            "qwen3_5",
            "Super-class Qwen 3.8 2.4T",
        ],
        "later_kernels": [
            "GDN decode",
            "sparse attention",
            "hyper-connection",
            "n-gram mmap/gather",
        ],
    }


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def _print_contract(as_json: bool) -> None:
    payload = contract()
    if as_json:
        json.dump(payload, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return
    print(payload["status"])
    print(f"family: {payload['family']}")
    print(f"host:   {payload['host_class']}")
    print("fail-closed: convert must not remap onto qwen3_5")


def _live_preflight(model_dir: Path) -> None:
    if not model_dir.is_dir():
        raise SystemExit(f"model dir is not a directory: {model_dir}")
    config = model_dir / "config.json"
    if not config.is_file():
        raise SystemExit(f"missing config.json under {model_dir}")
    raise SystemExit(
        "Qwen 3.8 Flash Next is incubating: AX has no repo-owned graph yet. "
        f"Best-experience SKU is {HOST_CLASS}."
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.model_dir is not None and not args.dry_run:
        _live_preflight(args.model_dir)
    _print_contract(args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
