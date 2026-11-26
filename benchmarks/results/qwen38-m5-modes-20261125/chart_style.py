"""Shared style for the README benchmark charts: horizontal grouped bars, one panel per setting.

Every chart has a title, a subtitle, a legend row, one or more panels (for example one per
concurrency) with the same categories, values printed at the bar ends, and an optional footnote.
Text stays text in the SVG (DejaVu Sans), and the output is deterministic.
"""
import math
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BACKGROUND = "#f5f8fc"
INK = "#14243b"
MUTED = "#485971"
GRID = "#d6dfec"
AXIS = "#8191a3"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "svg.fonttype": "none",
    "svg.hashsalt": "ax-engine-charts",
    "axes.edgecolor": AXIS,
    "axes.labelcolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
})

WIDTH = 10.5        # inches
BAR = 0.19          # bar thickness, inches
GAP = 0.20          # space between categories, inches


def number(value, digits=1):
    return f"{value:,.{digits}f}"


def percent(value, digits=2):
    return f"{value:+.{digits}f}%".replace("-", "−")


def bar_chart(path, *, title, subtitle, series, panels, xlabel, note=None, description=None,
              fmt=number, share_scale=True, xmax=None, legend_columns=4, tick_suffix=""):
    """series: [(label, color)]; panels: [{"title", "categories", "values": [[per series]], "suffix"}].

    values[s][c] is the value of series s in category c, or None when the route did not run;
    "missing" gives the text shown for a None bar, "suffix" an optional text appended to each label
    (same shape as values).
    """
    n_series = len(series)
    n_cat = max(len(p["categories"]) for p in panels)
    group = n_series * BAR + GAP
    plot_h = n_cat * group
    legend_rows = math.ceil(n_series / legend_columns)
    top = 0.95 + 0.26 * legend_rows + 0.35          # title, subtitle, legend, panel titles
    bottom = 0.72 + (0.28 if note else 0)
    height = top + plot_h + bottom
    fig = plt.figure(figsize=(WIDTH, height), facecolor=BACKGROUND)

    left_label = 1.25 if max(len(c) for p in panels for c in p["categories"]) > 6 else 0.75
    panel_gap = 0.35
    usable = WIDTH - 0.35 - len(panels) * left_label - (len(panels) - 1) * panel_gap - 0.25
    panel_w = usable / len(panels)

    values = [v for p in panels for row in p["values"] for v in row if v is not None]
    lo = min(0.0, min(values))
    hi = xmax if xmax is not None else max(values)
    span = (hi - lo) or 1.0

    fig.text(0.35 / WIDTH, 1 - 0.38 / height, title, fontsize=15, fontweight="bold", color=INK)
    fig.text(0.35 / WIDTH, 1 - 0.65 / height, subtitle, fontsize=9.5, color=MUTED)
    for i, (label, color) in enumerate(series):
        row, col = divmod(i, legend_columns)
        x = 0.35 + col * (WIDTH - 0.7) / legend_columns
        y = 0.98 + 0.26 * row
        fig.patches.append(plt.Rectangle((x / WIDTH, 1 - (y + 0.06) / height), 0.22 / WIDTH, 0.12 / height,
                                         transform=fig.transFigure, color=color))
        fig.text((x + 0.3) / WIDTH, 1 - (y + 0.01) / height, label, fontsize=9, color=INK, va="center")

    x0 = 0.35
    for p in panels:
        x0 += left_label
        ax = fig.add_axes([x0 / WIDTH, bottom / height, panel_w / WIDTH, plot_h / height])
        x0 += panel_w + panel_gap
        ax.set_facecolor(BACKGROUND)
        pv = [v for row in p["values"] for v in row if v is not None]
        p_lo = lo if share_scale else min(0.0, min(pv))
        p_hi = hi if share_scale else max(pv)
        p_span = (p_hi - p_lo) or 1.0
        ax.set_xlim(p_lo - (0.12 * p_span if p_lo < 0 else 0), p_hi + 0.16 * p_span)
        ax.set_ylim(plot_h, 0)
        cats = p["categories"]
        for c, cat in enumerate(cats):
            for s, (label, color) in enumerate(series):
                v = p["values"][s][c]
                y = c * group + GAP / 2 + s * BAR + BAR / 2
                if v is None:
                    text = p.get("missing")
                    if text:
                        ax.text(0, y, " " + text, fontsize=7.7, color=MUTED, va="center", style="italic")
                    continue
                ax.barh(y, v, height=BAR * 0.86, color=color, zorder=2)
                label_text = fmt(v)
                suffix = p.get("suffix")
                if suffix and suffix[s][c]:
                    label_text += f" · {suffix[s][c]}"
                pad = 0.008 * (ax.get_xlim()[1] - ax.get_xlim()[0])
                ax.text(v + (pad if v >= 0 else -pad), y, label_text, fontsize=7.7, color=INK, va="center",
                        ha="left" if v >= 0 else "right", zorder=3)
        ax.set_yticks([c * group + GAP / 2 + n_series * BAR / 2 for c in range(len(cats))])
        ax.set_yticklabels(cats, fontsize=9)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", labelsize=8.5, length=0)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
            lambda v, _: f"{v:,.12g}{tick_suffix}".replace("-", "−")))
        ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
        for side in ("top", "right", "bottom"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color(AXIS)
        if p_lo < 0:
            ax.axvline(0, color=AXIS, linewidth=0.8, zorder=1)
        if p.get("title"):
            ax.set_title(p["title"], loc="left", fontsize=10.5, fontweight="bold", color=INK, pad=8)
    fig.text(0.5, (bottom - 0.5) / height, xlabel, fontsize=9, color=MUTED, ha="center")
    if note:
        fig.text(0.35 / WIDTH, 0.12 / height, note, fontsize=8.5, color=MUTED)
    fig.savefig(path, facecolor=BACKGROUND, metadata={"Title": title, "Description": description or subtitle,
                                                       "Date": None, "Creator": None})
    preview = os.environ.get("CHART_PREVIEW")
    if preview:
        fig.savefig(os.path.join(preview, os.path.basename(str(path)).replace(".svg", ".png")), dpi=90,
                    facecolor=BACKGROUND)
    plt.close(fig)
    svg = open(path).read()   # browsers without DejaVu Sans fall back to a sans-serif face, not a serif one
    open(path, "w").write(re.sub(r"font-family: 'DejaVu Sans'", "font-family: 'DejaVu Sans', 'Helvetica Neue', "
                                 "Helvetica, Arial, sans-serif", svg))
