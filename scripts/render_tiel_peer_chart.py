#!/usr/bin/env python3
"""Render the four-machine Tiel / Cyber-Tiel AX-vs-MTPLX completion chart.

The chart is derived from the published median tables in
``docs/performance/tiel-vs-mtplx-2026-09-20.md`` so the figure, the public
report, and the README table can never disagree. It emits a deterministic
SVG (no plotting library) and supports ``--check`` to fail if the committed
SVG drifts from the source tables.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import NamedTuple

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DOC = REPO_ROOT / "docs" / "performance" / "tiel-vs-mtplx-2026-09-20.md"
OUTPUT_SVG = REPO_ROOT / "docs" / "assets" / "perf-tiel-vs-mtplx-2026-09-20.svg"

AX_COLOR = "#2eaf5f"
MTPLX_COLOR = "#f2b705"

# Machine short label used in the source table -> friendly label for the chart.
MACHINE_LABELS = {
    "M5 Max / 128 GiB": "M5 Max 128 GiB",
    "M4 Pro / 64 GiB": "M4 Pro 64 GiB",
    "M2 Ultra / 192 GiB": "M2 Ultra 192 GiB",
    "M3 Ultra / 512 GiB": "M3 Ultra 512 GiB",
}


class Cell(NamedTuple):
    machine: str
    model: str
    workload: str
    ax_tok_s: float
    mtplx_tok_s: float
    delta_pct: float


def _parse_delta(cell: str) -> float:
    return float(cell.replace("%", "").replace("+", "").strip())


def parse_completion_table(markdown_text: str) -> list[Cell]:
    """Read the '## Completion throughput' markdown table from the report."""
    lines = markdown_text.splitlines()
    try:
        start = next(
            index for index, line in enumerate(lines) if line.strip() == "## Completion throughput"
        )
    except StopIteration as error:  # pragma: no cover - guard against doc drift
        raise SystemExit("completion table heading missing from source doc") from error

    cells: list[Cell] = []
    in_table = False
    for line in lines[start:]:
        stripped = line.strip()
        if stripped.startswith("|") and "Machine" in stripped:
            in_table = True
            continue
        if not in_table:
            continue
        if not stripped.startswith("|"):
            if cells:
                break
            continue
        row = [part.strip() for part in stripped.strip("|").split("|")]
        if len(row) != 7 or set(row[0]) <= set("-: "):
            continue
        machine, model, workload = row[0], row[1], row[2]
        try:
            cell = Cell(
                machine=machine,
                model=model,
                workload=workload,
                ax_tok_s=float(row[3]),
                mtplx_tok_s=float(row[4]),
                delta_pct=_parse_delta(row[6]),
            )
        except ValueError:
            continue
        cells.append(cell)
    if len(cells) < 8:
        raise SystemExit(f"expected the full four-machine completion table, got {len(cells)} rows")
    return cells


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )


def render_svg(cells: list[Cell]) -> str:
    label_width = 250
    plot_left = label_width + 8
    plot_right = 812
    row_height = 46
    bar_height = 15
    top = 96
    width = 840
    height = top + row_height * len(cells) + 58

    max_value = max(max(cell.ax_tok_s, cell.mtplx_tok_s) for cell in cells)
    scale_top = ((int(max_value) // 50) + 1) * 50  # round up to the next 50
    plot_width = plot_right - plot_left

    def x_for(value: float) -> float:
        return plot_left + plot_width * (value / scale_top)

    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="Helvetica, Arial, sans-serif">'
    )
    parts.append(f'<rect width="{width}" height="{height}" fill="#ffffff"/>')
    parts.append(
        f'<text x="24" y="34" font-size="18" font-weight="700" fill="#12211a">'
        f"Tiel / Cyber-Tiel completion throughput — AX Engine vs MTPLX 2.11.3"
        f"</text>"
    )
    parts.append(
        f'<text x="24" y="56" font-size="12.5" fill="#4a5a52">'
        f"Median tokens/s including TTFT (higher is better) · four Apple Silicon Macs · "
        f"identical packs, cold KV, MLX 0.32.2 · campaign 2026-09-20"
        f"</text>"
    )

    # Gridlines + axis labels.
    step = 50
    for value in range(0, scale_top + 1, step):
        gx = x_for(value)
        parts.append(
            f'<line x1="{gx:.1f}" y1="{top - 12}" x2="{gx:.1f}" '
            f'y2="{top + row_height * len(cells) - 6}" stroke="#e6ece9" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{gx:.1f}" y="{top + row_height * len(cells) + 12}" '
            f'font-size="10.5" fill="#7a8a82" text-anchor="middle">{value}</text>'
        )

    # Legend.
    legend_y = 78
    parts.append(
        f'<rect x="{plot_left}" y="{legend_y - 9}" width="12" height="12" fill="{AX_COLOR}"/>'
    )
    parts.append(
        f'<text x="{plot_left + 16}" y="{legend_y + 1}" font-size="12" fill="#12211a">'
        f"AX Engine</text>"
    )
    parts.append(
        f'<rect x="{plot_left + 96}" y="{legend_y - 9}" width="12" height="12" '
        f'fill="{MTPLX_COLOR}"/>'
    )
    parts.append(
        f'<text x="{plot_left + 112}" y="{legend_y + 1}" font-size="12" fill="#12211a">'
        f"MTPLX sustained</text>"
    )

    for index, cell in enumerate(cells):
        row_top = top + index * row_height
        ax_y = row_top + 4
        mt_y = row_top + 4 + bar_height + 3
        machine = MACHINE_LABELS.get(cell.machine, cell.machine)
        label = f"{machine} · {cell.model} · {cell.workload}"
        parts.append(
            f'<text x="24" y="{row_top + bar_height + 6}" font-size="11.5" '
            f'fill="#22332c">{_escape(label)}</text>'
        )
        sign = "+" if cell.delta_pct >= 0 else ""
        parts.append(
            f'<text x="{plot_right}" y="{row_top + bar_height + 6}" font-size="11" '
            f'fill="#5a6a62" text-anchor="end">{sign}{cell.delta_pct:.1f}%</text>'
        )
        ax_w = x_for(cell.ax_tok_s) - plot_left
        mt_w = x_for(cell.mtplx_tok_s) - plot_left
        parts.append(
            f'<rect x="{plot_left}" y="{ax_y}" width="{ax_w:.1f}" height="{bar_height}" '
            f'fill="{AX_COLOR}"/>'
        )
        parts.append(
            f'<text x="{ax_w + plot_left + 5:.1f}" y="{ax_y + bar_height - 3}" '
            f'font-size="10.5" fill="#176c37">{cell.ax_tok_s:.2f}</text>'
        )
        parts.append(
            f'<rect x="{plot_left}" y="{mt_y}" width="{mt_w:.1f}" height="{bar_height}" '
            f'fill="{MTPLX_COLOR}"/>'
        )
        parts.append(
            f'<text x="{mt_w + plot_left + 5:.1f}" y="{mt_y + bar_height - 3}" '
            f'font-size="10.5" fill="#9a6a00">{cell.mtplx_tok_s:.2f}</text>'
        )

    parts.append(
        f'<text x="24" y="{height - 16}" font-size="10.5" fill="#7a8a82">'
        f"Derived from docs/performance/tiel-vs-mtplx-2026-09-20.md; per-cell ranges, "
        f"not pooled. Generated by scripts/render_tiel_peer_chart.py."
        f"</text>"
    )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if the committed SVG does not match the source tables",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_SVG)
    args = parser.parse_args(argv)

    cells = parse_completion_table(SOURCE_DOC.read_text(encoding="utf-8"))
    svg = render_svg(cells)

    if args.check:
        existing = args.output.read_text(encoding="utf-8") if args.output.is_file() else ""
        if existing != svg:
            print(
                f"ERROR: {args.output.relative_to(REPO_ROOT)} is stale; "
                f"run scripts/render_tiel_peer_chart.py",
                file=sys.stderr,
            )
            return 1
        print("Tiel peer chart is up to date")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg, encoding="utf-8")
    print(f"wrote {args.output.relative_to(REPO_ROOT)} ({len(cells)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
