#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    default_repo_dir = Path(os.getenv("REPO_DIR", Path(__file__).resolve().parent.parent))
    parser = argparse.ArgumentParser(
        description="List apple-to-apple benchmark results from benchmarks/results."
    )
    parser.add_argument(
        "--repo-dir",
        default=str(default_repo_dir),
        help="repository directory (defaults to script location or $REPO_DIR)",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="results directory to scan (default: benchmarks/results)",
    )
    parser.add_argument("--json", action="store_true", help="print JSON instead of a table")
    parser.add_argument(
        "--label-contains",
        help="only show runs whose label contains this substring",
    )
    parser.add_argument(
        "--model-contains",
        help="only show runs whose model path contains this substring",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="limit the number of rows shown after sorting by folder name descending",
    )
    args = parser.parse_args()
    if args.results_dir is None:
        args.results_dir = str(Path(args.repo_dir) / "benchmarks" / "results")
    return args


def load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def build_entry(run_dir: Path) -> dict | None:
    manifest_path = run_dir / "manifest.json"
    comparison_path = run_dir / "comparison.json"
    ax_path = run_dir / "ax.json"
    if not ax_path.exists():
        ax_path = None
    manifest = load_json(manifest_path)
    comparison = load_json(comparison_path)
    ax_only = load_json(ax_path) if ax_path is not None else None
    if manifest is None and comparison is None and ax_only is None:
        return None

    label = None
    model = None
    ax_engine = None
    llama_engine = None
    ax_prefill = None
    llama_prefill = None
    ax_decode = None
    llama_decode = None
    prefill_ratio = None
    decode_ratio = None

    if manifest is not None:
        label = manifest.get("label")
        model = manifest.get("model")
        ax_engine = ((manifest.get("engines") or {}).get("ax") or {}).get("name")
        llama_engine = ((manifest.get("engines") or {}).get("llama") or {}).get("name")

    if comparison is not None:
        label = comparison.get("label", label)
        model = comparison.get("model", model)
        ax_engine = (comparison.get("ax") or {}).get("engine", ax_engine)
        llama_engine = (comparison.get("llama") or {}).get("engine", llama_engine)
        ax_prefill = (comparison.get("ax") or {}).get("prefill_median_tok_per_s")
        llama_prefill = (comparison.get("llama") or {}).get("prefill_median_tok_per_s")
        ax_decode = (comparison.get("ax") or {}).get("decode_median_tok_per_s")
        llama_decode = (comparison.get("llama") or {}).get("decode_median_tok_per_s")
        prefill_ratio = (comparison.get("ratios") or {}).get("prefill_percent")
        decode_ratio = (comparison.get("ratios") or {}).get("decode_percent")

    if comparison is None and ax_only is not None:
        model = ax_only.get("model", model)
        ax_engine = "ax-engine"
        ax_prefill = ax_only.get("prefill_tok_per_sec_median")
        ax_decode = ax_only.get("decode_tok_per_sec_median")

    return {
        "folder": run_dir.name,
        "run_dir": str(run_dir),
        "label": label,
        "model": model,
        "ax_engine": ax_engine,
        "llama_engine": llama_engine,
        "ax_prefill_tok_per_s": ax_prefill,
        "llama_prefill_tok_per_s": llama_prefill,
        "prefill_percent": prefill_ratio,
        "ax_decode_tok_per_s": ax_decode,
        "llama_decode_tok_per_s": llama_decode,
        "decode_percent": decode_ratio,
        "manifest_json": str(manifest_path) if manifest_path.exists() else None,
        "comparison_json": str(comparison_path) if comparison_path.exists() else None,
    }


def format_num(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, (int, float)):
        return f"{value:.1f}"
    return str(value)


def print_table(entries: list[dict]) -> None:
    headers = [
        "folder",
        "label",
        "model",
        "ax",
        "llama",
        "ax_prefill",
        "llama_prefill",
        "prefill_%",
        "ax_decode",
        "llama_decode",
        "decode_%",
    ]
    rows = []
    for entry in entries:
        rows.append(
            [
                entry["folder"],
                entry.get("label") or "-",
                entry.get("model") or "-",
                entry.get("ax_engine") or "-",
                entry.get("llama_engine") or "-",
                format_num(entry.get("ax_prefill_tok_per_s")),
                format_num(entry.get("llama_prefill_tok_per_s")),
                format_num(entry.get("prefill_percent")),
                format_num(entry.get("ax_decode_tok_per_s")),
                format_num(entry.get("llama_decode_tok_per_s")),
                format_num(entry.get("decode_percent")),
            ]
        )

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def emit(cells: list[str]) -> str:
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells))

    print(emit(headers))
    print(emit(["-" * w for w in widths]))
    for row in rows:
        print(emit(row))


def main() -> int:
    args = parse_args()
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"error: results directory not found: {results_dir}", file=sys.stderr)
        return 1

    entries = []
    for child in sorted(results_dir.iterdir(), key=lambda p: p.name, reverse=True):
        if not child.is_dir():
            continue
        entry = build_entry(child)
        if entry is None:
            continue
        if args.label_contains and args.label_contains not in (entry.get("label") or ""):
            continue
        if args.model_contains and args.model_contains not in (entry.get("model") or ""):
            continue
        entries.append(entry)

    if args.limit is not None:
        entries = entries[: args.limit]

    if args.json:
        print(json.dumps(entries, indent=2))
    else:
        print_table(entries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
