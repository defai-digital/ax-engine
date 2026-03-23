#!/usr/bin/env python3
from __future__ import annotations

import csv
import pathlib
import sys
from collections import defaultdict


def parse_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: scripts/analyze_prefill_sweep.py <tsv> [<tsv> ...]", file=sys.stderr)
        return 2

    rows: list[dict[str, str]] = []
    for path_str in argv[1:]:
        path = pathlib.Path(path_str)
        with path.open(newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            rows.extend(reader)

    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["model_name"], row["prompt_tokens"])].append(row)

    print("| Model | P | Best Prefill Config | Prefill tok/s | Decode tok/s | Margin vs P1 |")
    print("|---|---:|---|---:|---:|---:|")

    for (model_name, prompt_tokens), group in sorted(grouped.items()):
        valid = [r for r in group if r.get("exit_code") == "0"]
        if not valid:
            continue
        for row in valid:
            row["_prefill"] = parse_float(row.get("prefill_toks", ""))
            row["_decode"] = parse_float(row.get("decode_toks", ""))
        valid = [r for r in valid if r["_prefill"] is not None]
        if not valid:
            continue

        baseline = next((r for r in valid if r["config"] == "P1_defaults"), None)
        best = max(valid, key=lambda r: r["_prefill"])
        margin = None
        if baseline and baseline["_prefill"] is not None:
            margin = best["_prefill"] - baseline["_prefill"]

        margin_text = f"{margin:+.1f}" if margin is not None else "n/a"
        decode_text = f"{best['_decode']:.1f}" if best["_decode"] is not None else "n/a"
        print(
            f"| {model_name} | {prompt_tokens} | {best['config']} | "
            f"{best['_prefill']:.1f} | {decode_text} | {margin_text} |"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
