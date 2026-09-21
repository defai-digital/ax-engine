#!/usr/bin/env python3
"""Render the current three-way MTP decode chart on the qualification SKU.

The chart compares the three runtimes that load the shared
``qwen3.8-27b:axq`` AXQ 6-bit pack and run MTP — AX Engine, MTPLX, and OMLX —
against the direct-AR ``mlx-lm`` baseline, using the published medians in
``benchmarks/results/mtp-axq-peer/2026-09-17-mac-mini-m4-pro-64gb/summary.json``.
It emits a deterministic SVG (no plotting library) and supports ``--check`` so
the committed figure cannot drift from the artifact.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import NamedTuple

REPO_ROOT = Path(__file__).resolve().parents[1]
SUMMARY = (
    REPO_ROOT
    / "benchmarks"
    / "results"
    / "mtp-axq-peer"
    / "2026-09-17-mac-mini-m4-pro-64gb"
    / "summary.json"
)
OUTPUT_SVG = REPO_ROOT / "docs" / "assets" / "perf-mtp-peer-decode-m4-pro-2026-09-17.svg"

# engine key -> (friendly name, config line, bar color, text color)
PEERS = {
    "ax_engine": ("AX Engine", "MTP depth 3", "#2eaf5f", "#176c37"),
    "mtplx": ("MTPLX", "MTP depth 3 · sustained", "#f2b705", "#9a6a00"),
    "omlx": ("OMLX", "Lightning MTP depth 1 · imported sidecar", "#3a7bd5", "#235a99"),
}
REFERENCE_COLOR = "#9aa6a0"
REFERENCE_TEXT = "#5a6a62"


class Peer(NamedTuple):
    name: str
    config: str
    color: str
    text_color: str
    decode_tok_s: float
    version: str


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )


def load_peers(summary: dict) -> list[Peer]:
    """Return the three MTP-capable runtimes, in AX/MTPLX/OMLX order."""
    measured = summary.get("measured")
    if not isinstance(measured, dict):
        raise SystemExit("summary.json has no 'measured' object")
    peers: list[Peer] = []
    for key, (name, config, color, text_color) in PEERS.items():
        row = measured.get(key)
        if not isinstance(row, dict):
            raise SystemExit(f"summary.json measured is missing engine {key!r}")
        value = row.get("decode_tok_s_median_20")
        if not _positive_finite(value):
            raise SystemExit(f"summary.json {key} lacks a positive decode median")
        peers.append(
            Peer(
                name=name,
                config=config,
                color=color,
                text_color=text_color,
                decode_tok_s=float(value),
                version=str(row.get("version", "")),
            )
        )
    return peers


def _positive_finite(value: object) -> bool:
    """A real, finite, positive number. `bool` is an int subclass and JSON
    `NaN` / `Infinity` parse as floats, so both need explicit rejection."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def _display_path(path: Path) -> str:
    """Repo-relative when inside the repo; otherwise the path as given."""
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def load_reference(summary: dict) -> Peer | None:
    """Return the direct-AR mlx-lm baseline row, or None if absent."""
    row = (summary.get("measured") or {}).get("mlx_lm")
    if not isinstance(row, dict):
        return None
    value = row.get("decode_tok_s_median_20")
    if not _positive_finite(value):
        return None
    return Peer(
        name="mlx-lm",
        config="direct AR · no MTP (baseline)",
        color=REFERENCE_COLOR,
        text_color=REFERENCE_TEXT,
        decode_tok_s=float(value),
        version=str(row.get("version", "")),
    )


def render_svg(peers: list[Peer], reference: Peer | None) -> str:
    rows = list(peers) + ([reference] if reference else [])
    label_width = 250
    plot_left = label_width + 8
    plot_right = 748
    row_height = 58
    bar_height = 20
    top = 104
    width = 780
    height = top + row_height * len(rows) + 74

    max_value = max(row.decode_tok_s for row in rows)
    scale_top = ((int(max_value) // 5) + 1) * 5  # round up to the next 5
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
        f"MTP decode — AX Engine vs MTPLX vs OMLX</text>"
    )
    parts.append(
        f'<text x="24" y="56" font-size="12.5" fill="#4a5a52">'
        f"qwen3.8-27b:axq (AXQ 6-bit MTP) · Mac mini M4 Pro 64 GB · "
        f"median decode tok/s over 20 runs · 2026-09-17</text>"
    )
    parts.append(
        f'<text x="24" y="74" font-size="12.5" fill="#4a5a52">'
        f"All three load the same pack and run MTP; OMLX uses Lightning draft "
        f"depth 1, and the grey bar is the direct-AR baseline (no MTP)</text>"
    )

    axis_bottom = top + row_height * len(rows) - 10
    for value in range(0, scale_top + 1, 5):
        gx = x_for(value)
        parts.append(
            f'<line x1="{gx:.1f}" y1="{top - 12}" x2="{gx:.1f}" '
            f'y2="{axis_bottom}" stroke="#e6ece9" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{gx:.1f}" y="{axis_bottom + 16}" '
            f'font-size="10.5" fill="#7a8a82" text-anchor="middle">{value}</text>'
        )
    parts.append(
        f'<text x="{(plot_left + plot_right) / 2:.0f}" y="{axis_bottom + 34}" '
        f'font-size="11" fill="#4a5a52" text-anchor="middle">decode tokens/s</text>'
    )

    for index, row in enumerate(rows):
        row_top = top + index * row_height
        bar_y = row_top + 4
        label = f"{row.name} {row.version}".strip()
        parts.append(
            f'<text x="24" y="{row_top + 16}" font-size="13" font-weight="600" '
            f'fill="#22332c">{_escape(label)}</text>'
        )
        parts.append(
            f'<text x="24" y="{row_top + 33}" font-size="11" '
            f'fill="#5a6a62">{_escape(row.config)}</text>'
        )
        bar_w = x_for(row.decode_tok_s) - plot_left
        parts.append(
            f'<rect x="{plot_left}" y="{bar_y}" width="{bar_w:.1f}" '
            f'height="{bar_height}" fill="{row.color}"/>'
        )
        parts.append(
            f'<text x="{bar_w + plot_left + 6:.1f}" y="{bar_y + bar_height - 5}" '
            f'font-size="12.5" font-weight="700" fill="{row.text_color}">'
            f"{row.decode_tok_s:.2f}</text>"
        )

    parts.append(
        f'<text x="24" y="{height - 16}" font-size="10.5" fill="#7a8a82">'
        f"From benchmarks/results/mtp-axq-peer/2026-09-17-mac-mini-m4-pro-64gb/"
        f"summary.json · generated by scripts/render_mtp_peer_decode_chart.py"
        f"</text>"
    )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if the committed SVG does not match the artifact",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_SVG)
    args = parser.parse_args(argv)

    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    peers = load_peers(summary)
    reference = load_reference(summary)
    svg = render_svg(peers, reference)

    if args.check:
        existing = args.output.read_text(encoding="utf-8") if args.output.is_file() else ""
        if existing != svg:
            print(
                f"ERROR: {_display_path(args.output)} is stale; "
                f"run scripts/render_mtp_peer_decode_chart.py",
                file=sys.stderr,
            )
            return 1
        print("MTP peer decode chart is up to date")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg, encoding="utf-8")
    suffix = " + reference" if reference is not None else ""
    print(f"wrote {_display_path(args.output)} ({len(peers)} MTP peers{suffix})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
