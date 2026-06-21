#!/usr/bin/env python3
"""Render the Qwen3-Coder-Next MoE-decode efficiency map (scatter).

The decode story is one identity:

    decode tok/s  =  effective bandwidth (GB/s)  /  bytes read per token

so the clearest presentation is a scatter of speed (y) vs weight-bytes read per
token (x), with the 577 GB/s M5 Max bandwidth ceiling drawn as the curve
tok/s = 577 / bytes. Reading it:
  - upper-left is best (few bytes, high speed);
  - AX and mlx-lm sit at the SAME x (identical MLX 4-bit weights), so the
    vertical gap between them is pure kernel efficiency (AX's MoE gather-GEMV);
  - llama.cpp is shifted right because Q4_K_M reads 1.44x the bytes/token, which
    is why it decodes slowest even though it sustains more raw bandwidth;
  - every point sits far below the ceiling curve -> decode is gather/dispatch
    bound, NOT bandwidth bound; the vertical room to the curve is headroom.

Data provenance (M5 Max, decode generation=128, prompt=128 representative;
decode tok/s is essentially depth-independent for this model):
  - peak 577 GB/s: MLX reduction probe (same figure as the Gemma 4 12B chart).
  - MLX 1.9648 GB/token (AX, mlx-lm): harness `bandwidth_accounting`
    (other_bytes + routed_bytes * 10/512) from the MLX model-manifest.
  - llama.cpp 2.8275 GB/token: same formula over the GGUF tensor table
    (`llama-gguf <model> r`).
  - decode tok/s: AX 105.7 / mlx-lm 99.2 / llama.cpp 85.5 (prompt=128 rows).

Usage:
  python3 scripts/render_qwen_coder_next_bandwidth_chart.py [--assets-dir docs/assets]
"""

from __future__ import annotations

import argparse
import html
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FONT = "Inter,Segoe UI,Arial,sans-serif"

PEAK_GBS = 577.0

# (key, label, quant, bytes_per_token_gb, decode_tok_s, fill, stroke)
POINTS = [
    ("ax", "AX Engine 6.5.2", "MLX 4-bit + fused expert block", 1.9648, 105.7, "#2eaf5f", "#176c37"),
    ("mlx", "mlx-lm 0.31.3", "MLX 4-bit", 1.9648, 99.2, "#f2b705", "#9a6a00"),
    ("llama", "llama.cpp b9700", "Q4_K_M", 2.8275, 85.5, "#f97316", "#c2410c"),
]

# Axis domains
X_MIN, X_MAX = 1.5, 3.1   # bytes/token (GB)
Y_MIN, Y_MAX = 0.0, 320.0  # decode tok/s

WIDTH = 840
HEIGHT = 470

PLOT_LEFT = 74
PLOT_RIGHT = 600
PLOT_TOP = 104
PLOT_BOTTOM = 392

RED = "#dc2626"
TITLE = "Qwen3-Coder-Next 4-bit — MoE Decode Efficiency: speed = bandwidth ÷ bytes/token"
SUBTITLE = (
    "decode throughput vs weight bytes read per token · the curve is the 577 GB/s "
    "M5 Max bandwidth ceiling"
)
NOTE = (
    "every engine sits far below the ceiling — MoE decode is gather/dispatch-bound, "
    "not bandwidth-bound; the room above each point is headroom"
)
FOOTNOTE = (
    "AX 6.5.2 · mlx-lm 0.31.3 · llama.cpp b9700 · M5 Max · peak via MLX reduction "
    "probe · llama bytes/token from GGUF tensor table"
)


def fx(bytes_gb: float) -> float:
    return PLOT_LEFT + (bytes_gb - X_MIN) / (X_MAX - X_MIN) * (PLOT_RIGHT - PLOT_LEFT)


def fy(tok_s: float) -> float:
    return PLOT_BOTTOM - (tok_s - Y_MIN) / (Y_MAX - Y_MIN) * (PLOT_BOTTOM - PLOT_TOP)


def render() -> str:
    parts: list[str] = []

    def e(s: object) -> str:
        return html.escape(str(s))

    parts += [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}"'
        f' viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="title desc">',
        f"<title>{e(TITLE)}</title>",
        f"<desc>Scatter plot of Qwen3-Coder-Next 4-bit MoE decode. X axis is weight"
        f" bytes read per token, Y axis is decode tok/s; the curve is the 577 GB/s"
        f" bandwidth ceiling (tok/s = 577 / bytes). AX and mlx-lm share x = 1.96"
        f" GB/token (identical MLX weights) so the vertical gap between them is kernel"
        f" efficiency: AX 105.7 tok/s vs mlx-lm 99.2 tok/s. llama.cpp reads 2.83"
        f" GB/token (1.44x more, Q4_K_M) and decodes slowest at 85.5 tok/s. All three"
        f" sit far below the ceiling, so decode is gather-bound, not bandwidth-bound.</desc>",
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="#f8fafc"/>',
        f'<text x="20" y="28" font-family="{FONT}" font-size="15" font-weight="700"'
        f' fill="#111827">{e(TITLE)}</text>',
        f'<text x="20" y="48" font-family="{FONT}" font-size="11" fill="#4b5563">'
        f"{e(SUBTITLE)}</text>",
        f'<text x="20" y="64" font-family="{FONT}" font-size="10" fill="#6b7280">'
        f"{e(NOTE)}</text>",
        f'<rect x="{PLOT_LEFT}" y="{PLOT_TOP}" width="{PLOT_RIGHT - PLOT_LEFT}"'
        f' height="{PLOT_BOTTOM - PLOT_TOP}" rx="4" fill="#ffffff" stroke="#dbe3ef"/>',
    ]

    # Y gridlines + labels (tok/s)
    for tv in range(0, 321, 40):
        gy = fy(tv)
        parts.append(
            f'<line x1="{PLOT_LEFT}" y1="{gy:.1f}" x2="{PLOT_RIGHT}" y2="{gy:.1f}"'
            f' stroke="#eef2f7" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{PLOT_LEFT - 8}" y="{gy + 3:.1f}" text-anchor="end"'
            f' font-family="{FONT}" font-size="10" fill="#6b7280">{tv}</text>'
        )
    parts.append(
        f'<text x="20" y="{(PLOT_TOP + PLOT_BOTTOM) / 2:.1f}" text-anchor="middle"'
        f' font-family="{FONT}" font-size="11" font-weight="700" fill="#374151"'
        f' transform="rotate(-90 20 {(PLOT_TOP + PLOT_BOTTOM) / 2:.1f})">decode tok/s'
        f" (higher is better)</text>"
    )

    # X gridlines + labels (bytes/token)
    xticks = [1.5, 1.8, 2.1, 2.4, 2.7, 3.0]
    for xv in xticks:
        gx = fx(xv)
        parts.append(
            f'<line x1="{gx:.1f}" y1="{PLOT_TOP}" x2="{gx:.1f}" y2="{PLOT_BOTTOM}"'
            f' stroke="#eef2f7" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{gx:.1f}" y="{PLOT_BOTTOM + 16:.1f}" text-anchor="middle"'
            f' font-family="{FONT}" font-size="10" fill="#6b7280">{xv:.1f}</text>'
        )
    parts.append(
        f'<text x="{(PLOT_LEFT + PLOT_RIGHT) / 2:.1f}" y="{PLOT_BOTTOM + 34:.1f}"'
        f' text-anchor="middle" font-family="{FONT}" font-size="11" font-weight="700"'
        f' fill="#374151">weight bytes read per token (GB) — fewer is better ←</text>'
    )

    # Bandwidth ceiling curve: tok/s = 577 / bytes (clipped to the plot box)
    pts: list[str] = []
    steps = 80
    for i in range(steps + 1):
        bx = X_MIN + (X_MAX - X_MIN) * i / steps
        ty = PEAK_GBS / bx
        if ty > Y_MAX:
            continue
        pts.append(f"{fx(bx):.1f},{fy(ty):.1f}")
    parts.append(
        f'<polyline points="{" ".join(pts)}" fill="none" stroke="{RED}"'
        f' stroke-width="1.8" stroke-dasharray="6 4"/>'
    )
    # Ceiling label near the right end of the curve
    lbx = 2.92
    lby = PEAK_GBS / lbx
    parts.append(
        f'<text x="{fx(lbx) + 6:.1f}" y="{fy(lby) - 6:.1f}" text-anchor="start"'
        f' font-family="{FONT}" font-size="10" font-weight="700" fill="{RED}">'
        f"577 GB/s ceiling</text>"
    )
    parts.append(
        f'<text x="{fx(lbx) + 6:.1f}" y="{fy(lby) + 6:.1f}" text-anchor="start"'
        f' font-family="{FONT}" font-size="9" fill="{RED}" fill-opacity="0.8">'
        f"max speed if saturated</text>"
    )

    parts.append(
        '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7"'
        ' markerHeight="7" orient="auto-start-reverse">'
        '<path d="M 0 0 L 10 5 L 0 10 z" fill="#c2410c"/></marker></defs>'
    )

    ax_pt = POINTS[0]
    mlx_pt = POINTS[1]
    llama_pt = POINTS[2]
    mlx_x = fx(ax_pt[3])
    llama_x = fx(llama_pt[3])

    # Headroom: dashed vertical from the AX point up to the ceiling at x = 1.96
    ceil_at_mlx = PEAK_GBS / ax_pt[3]
    parts.append(
        f'<line x1="{mlx_x:.1f}" y1="{fy(ax_pt[4]):.1f}"'
        f' x2="{mlx_x:.1f}" y2="{fy(ceil_at_mlx) + 4:.1f}"'
        f' stroke="#94a3b8" stroke-width="1.2" stroke-dasharray="2 3"/>'
    )
    parts.append(
        f'<text x="{mlx_x + 8:.1f}" y="{fy(ceil_at_mlx) + 22:.1f}"'
        f' text-anchor="start" font-family="{FONT}" font-size="10" font-weight="700"'
        f' fill="#64748b">64% idle</text>'
    )
    parts.append(
        f'<text x="{mlx_x + 8:.1f}" y="{fy(ceil_at_mlx) + 35:.1f}"'
        f' text-anchor="start" font-family="{FONT}" font-size="10" fill="#64748b">'
        f"headroom</text>"
    )

    # Bytes-penalty arrow in the empty headroom band: 1.96 -> 2.83 GB/token
    ay = fy(150)
    parts.append(
        f'<line x1="{mlx_x + 4:.1f}" y1="{ay:.1f}" x2="{llama_x - 9:.1f}"'
        f' y2="{ay:.1f}" stroke="#c2410c" stroke-width="1.4" marker-end="url(#arrow)"/>'
    )
    parts.append(
        f'<text x="{(mlx_x + llama_x) / 2:.1f}" y="{ay - 6:.1f}" text-anchor="middle"'
        f' font-family="{FONT}" font-size="10" font-weight="700" fill="#c2410c">'
        f"Q4_K_M reads 1.44× the bytes/token</text>"
    )

    # Kernel-gap bracket: AX vs mlx-lm at the same x (identical weights)
    bx = mlx_x - 14
    midy = (fy(ax_pt[4]) + fy(mlx_pt[4])) / 2
    parts.append(
        f'<path d="M {bx + 6:.1f} {fy(ax_pt[4]):.1f} H {bx:.1f} V {fy(mlx_pt[4]):.1f}'
        f' H {bx + 6:.1f}" fill="none" stroke="#176c37" stroke-width="1.4"/>'
    )
    parts.append(
        f'<text x="{bx - 4:.1f}" y="{midy - 3:.1f}" text-anchor="end"'
        f' font-family="{FONT}" font-size="10" font-weight="700" fill="#176c37">'
        f"+6.6% decode</text>"
    )
    parts.append(
        f'<text x="{bx - 4:.1f}" y="{midy + 9:.1f}" text-anchor="end"'
        f' font-family="{FONT}" font-size="9" fill="#176c37">same weights, AX kernels</text>'
    )

    # Point label offsets: AX above the pair, mlx-lm below it, llama to the right.
    offsets = {
        "ax": (13, -11, 1),
        "mlx": (13, 15, 27),
        "llama": (14, -4, 9),
    }
    for key, label, quant, bpt, tok_s, fill, stroke in POINTS:
        px, py = fx(bpt), fy(tok_s)
        pct_peak = bpt * tok_s / PEAK_GBS * 100.0
        dx, name_dy, met_dy = offsets[key]
        parts.append(
            f'<circle cx="{px:.1f}" cy="{py:.1f}" r="7.5" fill="{fill}"'
            f' fill-opacity="0.92" stroke="{stroke}" stroke-width="2"/>'
        )
        parts.append(
            f'<text x="{px + dx:.1f}" y="{py + name_dy:.1f}" text-anchor="start"'
            f' font-family="{FONT}" font-size="11" font-weight="700" fill="#111827">'
            f"{e(label)}</text>"
        )
        parts.append(
            f'<text x="{px + dx:.1f}" y="{py + met_dy:.1f}" text-anchor="start"'
            f' font-family="{FONT}" font-size="10" fill="#4b5563">'
            f"{tok_s:.1f} tok/s · {bpt:.2f} GB/tok · {pct_peak:.0f}% of peak</text>"
        )

    parts.append(
        f'<text x="20" y="{HEIGHT - 14}" font-family="{FONT}" font-size="9"'
        f' fill="#9ca3af">{e(FOOTNOTE)}</text>'
    )

    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-dir", type=Path, default=REPO_ROOT / "docs" / "assets")
    args = parser.parse_args()
    args.assets_dir.mkdir(parents=True, exist_ok=True)
    out = args.assets_dir / "perf-qwen-coder-next-bandwidth.svg"
    out.write_text(render())
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
