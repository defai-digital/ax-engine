#!/usr/bin/env python3
from __future__ import annotations

import csv
import pathlib
import statistics
import sys
from collections import defaultdict


def median_text(values: list[float]) -> str:
    return f"{statistics.median(values):.1f}" if values else "n/a"


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(
            "usage: scripts/summarize_phase3_samples.py <tsv> [<tsv> ...]",
            file=sys.stderr,
        )
        return 2

    grouped: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for path_str in argv[1:]:
        path = pathlib.Path(path_str)
        with path.open(newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                key = (row["model_name"], row["prompt_tokens"])
                grouped[key]["ax_prefill"].append(float(row["ax_prefill"]))
                grouped[key]["ax_decode"].append(float(row["ax_decode"]))
                grouped[key]["llama_prefill"].append(float(row["llama_prefill"]))
                grouped[key]["llama_decode"].append(float(row["llama_decode"]))

    print("| Model | P | AX prefill | llama.cpp prefill | AX/llama | AX decode | llama.cpp decode | AX/llama | Samples |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")

    for (model_name, prompt_tokens), data in sorted(grouped.items()):
        ax_pp = statistics.median(data["ax_prefill"])
        ll_pp = statistics.median(data["llama_prefill"])
        ax_tg = statistics.median(data["ax_decode"])
        ll_tg = statistics.median(data["llama_decode"])
        samples = len(data["ax_decode"])
        print(
            f"| {model_name} | {prompt_tokens} | "
            f"{ax_pp:.1f} | {ll_pp:.1f} | {ax_pp / ll_pp * 100:.1f}% | "
            f"{ax_tg:.1f} | {ll_tg:.1f} | {ax_tg / ll_tg * 100:.1f}% | "
            f"{samples} |"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
