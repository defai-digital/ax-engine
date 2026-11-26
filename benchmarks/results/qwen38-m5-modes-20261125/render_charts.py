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
ROOT = CAMPAIGN.parents[2]
sys.path.insert(0, str(CAMPAIGN))
sys.path.insert(0, str(ROOT / "scripts"))
from chart_palette import M5_ENGINE_COLORS  # noqa: E402
from chart_style import bar_chart  # noqa: E402
BEGIN = "<!-- qwen38-m5-modes:begin -->"
END = "<!-- qwen38-m5-modes:end -->"
MODEL_ORDER = ("flash", "dense")
MODEL_NAMES = {"flash": "Qwen 3.8 Flash Next", "dense": "Qwen 3.8 27B"}
MODES = ("direct", "dflash2")
MODE_CATEGORIES = ["Direct greedy", "DFlash2"]
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


LENGTH_FIGURE = "perf-qwen38-flash-next-m5-first-text-by-length.svg"
LENGTH_NOTE = "docs/performance/flash-next-qsa-masked-dense-2026-11-25.md"
# Past this many input tokens Flash Next attends over selected key blocks.
LENGTH_SPLIT = 2_500


def label(row, versions):
    key = row["engine"]
    mark = RERUN_MARK if row.get("session") == "paired-rerun" else ""
    version = row.get("engine_version", versions[key])
    return f"{ENGINE_NAMES[key]} {version} ({MODE_NAMES[row['mode']]}){mark}"


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
    series = [(ENGINE_NAMES[engine], M5_ENGINE_COLORS[engine]) for engine in engines]

    def panel(metric, title):
        values, suffix = [], []
        for engine in engines:
            mode_values, mode_suffix = [], []
            for mode in MODES:
                row = by_engine[engine].get(mode)
                if row is None:
                    mode_values.append(None)
                    mode_suffix.append(None)
                else:
                    mode_values.append(row[metric])
                    mode_suffix.append("†" if row.get("session") == "paired-rerun" else None)
            values.append(mode_values)
            suffix.append(mode_suffix)
        return {"title": title, "categories": MODE_CATEGORIES, "values": values, "suffix": suffix,
                "missing": "not run"}

    name = MODEL_NAMES[model]
    bar_chart(
        path,
        title=f"{name} AXQ MXFP4",
        subtitle="MacBook Pro M5 Max 128 GB · one active streaming request · greedy",
        series=series,
        panels=[panel("decode_tok_s", "Decode"), panel("prefill_tok_s", "Effective prefill")],
        xlabel="Tokens per second · each panel has its own scale",
        note="† Median of three interleaved runs. Zero means that configuration did not finish.",
        share_scale=False,
        fmt=lambda value: "0" if not value else f"{value:,.2f}",
        legend_columns=4,
        description=f"{name} decode and effective prefill by engine and greedy mode on M5 Max. "
                    "Versions are in the README table.",
    )
    return path.read_text()


def by_length(records):
    """Per input length and engine: the mean over that engine's runs of the run median."""
    runs = {}
    for row in records:
        per_engine = runs.setdefault(row["input_tokens"], {})
        per_engine.setdefault(row["engine"], []).append(median(row["first_text_seconds"]))
    return {
        tokens: {engine: sum(values) / len(values) for engine, values in per_engine.items()}
        for tokens, per_engine in sorted(runs.items())
    }


def length_engines(records):
    """Engines of the length scan in first-run order, with the version each row recorded."""
    return list({row["engine"]: row["engine_version"] for row in records}.items())


def render_length(records, path):
    lengths = by_length(records)
    engines = [engine for engine, _ in length_engines(records)]

    def panel(title, keep):
        tokens = [count for count in lengths if keep(count)]
        return {
            "title": title,
            "categories": [f"{count:,}" for count in tokens],
            "values": [[lengths[count].get(engine) for count in tokens] for engine in engines],
            "missing": "not run",
        }

    bar_chart(
        path,
        title="Qwen 3.8 Flash Next AXQ MXFP4 · first text by input length",
        subtitle="MacBook Pro M5 Max 128 GB · one cold direct greedy request · rows are input tokens",
        series=[(ENGINE_NAMES[engine], M5_ENGINE_COLORS[engine]) for engine in engines],
        panels=[
            panel(f"Up to {LENGTH_SPLIT:,} input tokens", lambda count: count <= LENGTH_SPLIT),
            panel(f"More than {LENGTH_SPLIT:,} input tokens", lambda count: count > LENGTH_SPLIT),
        ],
        xlabel="Seconds to first text · lower is better · each panel has its own scale",
        note="Servers ran one after another with 150 s of rest before each. "
             "An engine measured twice shows the mean of its runs.",
        share_scale=False,
        fmt=lambda value: f"{value:.2f}",
        legend_columns=4,
        description="Qwen 3.8 Flash Next cold first-text latency by input length for AX Engine "
                    "and oMLX on M5 Max. Versions are in the README table.",
    )
    return path.read_text()


def length_table(records):
    lengths = by_length(records)
    engines = length_engines(records)
    first, *others = [engine for engine, _ in engines]
    head = [f"{ENGINE_NAMES[engine]} {version} (s)" for engine, version in engines]
    head += [f"{ENGINE_NAMES[first]} against {ENGINE_NAMES[other]}" for other in others]
    parts = [
        "**First text by input length**, one cold direct greedy request. The servers ran one "
        "after another in the order AX Engine, oMLX, AX Engine, with 150 seconds of rest before "
        "each; the AX Engine column is the mean of its two runs. Lower is better. "
        f"[Measurements and method]({LENGTH_NOTE}).",
        "",
        f"![Qwen 3.8 Flash Next cold first text by input length on M5 Max.](docs/assets/{LENGTH_FIGURE})",
        "",
        "| Input tokens | " + " | ".join(head) + " |",
        "| ---: |" + " ---: |" * len(head),
    ]
    for tokens, seconds in lengths.items():
        cells = [f"{seconds[engine]:.3f}" for engine, _ in engines]
        for other in others:
            change = 100 * (seconds[first] / seconds[other] - 1)
            cells.append(f"{change:+.0f}%".replace("-", "−"))
        parts.append(f"| {tokens:,} | " + " | ".join(cells) + " |")
    return parts + [""]


def tables(versions, grouped, lengths):
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
        if model == "flash":
            parts += length_table(lengths)
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
    lengths = [
        json.loads(line)
        for line in (CAMPAIGN / "first-text-by-length.jsonl").read_text().splitlines()
    ]
    charts = [
        (figure_name(model), lambda path, model=model, rows=rows: render(model, rows, path))
        for model, rows in grouped.items()
    ]
    charts.append((LENGTH_FIGURE, lambda path: render_length(lengths, path)))
    for name, draw in charts:
        path = ROOT / "docs/assets" / name
        if args.check:
            probe = path.with_suffix(".check.svg")
            chart = draw(probe)
            probe.unlink()
            if path.read_text() != chart:
                parser.error(f"{path.name} differs from the recorded metrics")
        else:
            draw(path)
    readme = ROOT / "README.md"
    original = readme.read_text()
    start, end = original.index(BEGIN), original.index(END) + len(END)
    generated = tables(versions, grouped, lengths)
    if args.check:
        if original[start:end] != generated:
            parser.error("README benchmark tables differ from the recorded metrics")
    else:
        readme.write_text(original[:start] + generated + original[end:])


if __name__ == "__main__":
    main()
