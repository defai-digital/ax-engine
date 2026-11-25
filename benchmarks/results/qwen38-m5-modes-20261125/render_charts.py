#!/usr/bin/env python3
"""Render the public M5 greedy mode tables and charts from the serving receipt.

Needs matplotlib. `python render_charts.py --check` fails when the README
tables or SVG charts differ from this receipt.
"""

import argparse
import json
import sys
from pathlib import Path
from statistics import median

CAMPAIGN = Path(__file__).resolve().parent
sys.path.insert(0, str(CAMPAIGN))
from chart_style import bar_chart  # noqa: E402

ROOT = CAMPAIGN.parents[2]
BEGIN = "<!-- qwen38-m5-modes:begin -->"
END = "<!-- qwen38-m5-modes:end -->"
MODEL_ORDER = ("flash", "dense")
MODEL_NAMES = {"flash": "Qwen 3.8 Flash Next", "dense": "Qwen 3.8 27B"}
MODES = ("direct", "dflash2")
SERIES = [("Direct greedy", "#176b63"), ("DFlash2", "#514388")]
ENGINE_NAMES = {
    "ax-engine": "AX Engine",
    "splash": "Splash",
    "omlx": "oMLX",
    "mlx-lm": "mlx-lm",
    "mtplx": "MTPLX",
    "tensorfold": "TensorFold",
    "ds4": "ds4",
}
MODE_NAMES = {"direct": "Direct greedy", "dflash2": "DFlash2"}
RERUN_MARK = " †"
RERUN_NOTE = (
    "† Median of three runs interleaved with the neighbouring † rows in one later session "
    "on the same day; the other rows are single runs from the original session. See the "
    "[campaign notes](benchmarks/results/qwen38-m5-modes-20261125/README.md#paired-rerun)."
)


def label(row, versions):
    key = row["engine"]
    mark = RERUN_MARK if row.get("session") == "paired-rerun" else ""
    return f"{ENGINE_NAMES[key]} {versions[key]} ({MODE_NAMES[row['mode']]}){mark}"


def figure_name(model):
    return f"perf-qwen38-{'27b' if model == 'dense' else 'flash-next'}-m5-modes.svg"


def rate(value):
    return f"{value:.2f}" if value else "0"


def first_text(row, size):
    """Median first-text latency of the short or long prefill probe, or a dash."""
    probes = [pair[size]["first_text_seconds"] for pair in row.get("prefill") or []]
    return f"{median(probes):.2f}" if probes else "—"


def render(model, rows, path):
    by_engine = {}
    for row in rows:
        by_engine.setdefault(row["engine"], {})[row["mode"]] = row
    engines = list(by_engine)
    categories = [ENGINE_NAMES[engine] for engine in engines]

    def panel(metric, title):
        values, suffix = [], []
        for mode in MODES:
            mode_values, mode_suffix = [], []
            for engine in engines:
                row = by_engine[engine].get(mode)
                if row is None:
                    mode_values.append(None)
                    mode_suffix.append(None)
                else:
                    mode_values.append(row[metric])
                    mode_suffix.append("†" if row.get("session") == "paired-rerun" else None)
            values.append(mode_values)
            suffix.append(mode_suffix)
        return {"title": title, "categories": categories, "values": values, "suffix": suffix,
                "missing": "not run"}

    name = MODEL_NAMES[model]
    bar_chart(
        path,
        title=f"{name} AXQ MXFP4",
        subtitle="MacBook Pro M5 Max 128 GB · one active streaming request · greedy",
        series=SERIES,
        panels=[panel("decode_tok_s", "Decode"), panel("prefill_tok_s", "Effective prefill")],
        xlabel="Tokens per second · each panel has its own scale",
        note="† Median of three interleaved runs. Zero means that configuration did not finish.",
        share_scale=False,
        fmt=lambda value: "0" if not value else f"{value:,.2f}",
        legend_columns=2,
        description=f"{name} decode and effective prefill by engine and greedy mode on M5 Max. "
                    "Versions are in the README table.",
    )
    return path.read_text()


def tables(versions, grouped):
    parts = [BEGIN, ""]
    for model, rows in grouped.items():
        name = MODEL_NAMES[model]
        chart = figure_name(model)
        parts += [
            f"#### {name} MXFP4",   # under the README's per-machine heading
            "",
            f"![{name} decode and effective prefill by engine version and mode on M5 Max.](docs/assets/{chart})",
            "",
            "| Engine and mode | Decode (output tok/s) | Effective prefill (input tok/s) "
            "| First text, 127–128 input tokens (s) | First text, 1,492 input tokens (s) |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        parts += [
            f"| {label(row, versions)} | {rate(row['decode_tok_s'])} | {rate(row['prefill_tok_s'])} "
            f"| {first_text(row, 'short')} | {first_text(row, 'long')} |"
            for row in rows
        ]
        parts.append("")
        if any(row.get("session") == "paired-rerun" for row in rows):
            parts += [RERUN_NOTE, ""]
    return "\n".join(parts + [END])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check the figures and README table")
    args = parser.parse_args()
    campaign = json.loads((CAMPAIGN / "provenance.json").read_text())
    versions = campaign["engine_versions"]
    records = [json.loads(line) for line in (CAMPAIGN / "results.jsonl").read_text().splitlines()]
    grouped = {
        model: [row for row in records if row["model"] == model and row["engine"] in ENGINE_NAMES]
        for model in MODEL_ORDER
    }
    for model, rows in grouped.items():
        path = ROOT / "docs/assets" / figure_name(model)
        if args.check:
            probe = path.with_suffix(".check.svg")
            chart = render(model, rows, probe)
            probe.unlink()
            if path.read_text() != chart:
                parser.error(f"{path.name} differs from the recorded metrics")
        else:
            render(model, rows, path)
    readme = ROOT / "README.md"
    original = readme.read_text()
    start, end = original.index(BEGIN), original.index(END) + len(END)
    generated = tables(versions, grouped)
    if args.check:
        if original[start:end] != generated:
            parser.error("README benchmark tables differ from the recorded metrics")
    else:
        readme.write_text(original[:start] + generated + original[end:])


if __name__ == "__main__":
    main()
