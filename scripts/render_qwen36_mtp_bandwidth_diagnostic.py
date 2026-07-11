#!/usr/bin/env python3
"""Render the Qwen3.6 27B MTP same-sidecar effective output-work diagnostic.

The chart compares AX Engine, MTPLX, and lightning-mlx using a derived metric:

    effective output bandwidth = decode tok/s * active target-weight bytes

This chart is limited to the 27B dense same-sidecar rows, where active bytes
match across engines. The derived value is not a GPU counter; MTP can emit
multiple committed tokens per target verifier cycle.
"""

from __future__ import annotations

import argparse
import html
import json
import statistics
import struct
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = (
    REPO_ROOT
    / "benchmarks/results/mtp-qwen36-matrix/"
    / "2026-07-09-peer-comparison-apples-to-apples-refresh/summary.json"
)
DEFAULT_JSON = DEFAULT_SUMMARY.parent / "bandwidth_diagnostic.json"
DEFAULT_SVG = REPO_ROOT / "docs/assets/perf-qwen36-mtp-bandwidth-diagnostic.svg"

FONT = "Inter,Segoe UI,Arial,sans-serif"
PEAK_GBS = 577.0
PEAK_SOURCE = "M5 Max MLX reduction probe"

# Fallbacks keep this renderer reproducible when the local HF cache is absent.
# The 35B peer value was computed from the 2026-07-09 Youssofal optimized
# package safetensors headers: non-routed bytes + switch_mlp routed bytes * 8/256.
PEER_PACKAGE_ACTIVE_BYTE_FALLBACKS = {
    "Qwen3.6 35B-A3B 4-bit": {
        "bytes": 2_943_165_152,
        "source": "safetensors_header_fallback_peer_youssofal_optimized",
    },
}

ENGINE_LABELS = {
    "ax_engine": "AX Engine v6.8.2 (2026-07-11)",
    "mtplx": "MTPLX",
    "lightning_mlx": "lightning-mlx",
}
ENGINE_ORDER = ("ax_engine", "mtplx", "lightning_mlx")
TARGET_ORDER = ("Qwen3.6 27B 4-bit", "Qwen3.6 35B-A3B 4-bit")
ENGINE_COLORS = {
    "ax_engine": ("#2eaf5f", "#176c37"),
    "mtplx": ("#f2b705", "#9a6a00"),
    "lightning_mlx": ("#2563eb", "#1d4ed8"),
}

WIDTH = 940
LEFT = 36
PLOT_LEFT = 210
PLOT_RIGHT = 680


def metric_median(cell: dict[str, Any], key: str) -> float:
    value = cell.get(key)
    if isinstance(value, dict):
        value = value.get("median")
    if not isinstance(value, int | float):
        raise ValueError(f"missing numeric metric: {key}")
    return float(value)


def safetensor_header_byte_estimate(model_dir: Path, *, moe_active: bool) -> tuple[int, str]:
    if not model_dir.is_dir():
        raise FileNotFoundError(model_dir)
    total = 0
    other = 0
    routed = 0
    for path in sorted(model_dir.glob("*.safetensors")):
        with path.open("rb") as handle:
            header_len = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(header_len))
        for key, metadata in header.items():
            if key == "__metadata__":
                continue
            start, end = metadata["data_offsets"]
            size = int(end) - int(start)
            total += size
            is_routed = (
                ("switch_mlp" in key or ".mlp.experts." in key)
                and "shared_expert" not in key
            )
            if is_routed:
                routed += size
            else:
                other += size
    if total <= 0:
        raise ValueError(f"no safetensor bytes found in {model_dir}")
    if not moe_active:
        return total, "safetensors_header_dense_total"
    return other + int(routed * 8 / 256), "safetensors_header_moe_active_8_of_256"


def peer_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in summary.get("rows", []):
        if (
            isinstance(row, dict)
            and row.get("model_label") in TARGET_ORDER
            and row.get("engine") in ENGINE_ORDER
        ):
            rows.append(row)
    rows.sort(
        key=lambda row: (
            TARGET_ORDER.index(str(row["model_label"])),
            ENGINE_ORDER.index(str(row["engine"])),
        )
    )
    return rows


def ax_artifact_estimate(peer_row: dict[str, Any]) -> tuple[int, str, dict[str, Any] | None]:
    artifact = REPO_ROOT / str(peer_row["artifact"])
    raw = json.loads(artifact.read_text())
    accounting = raw["bandwidth_accounting"]
    bytes_used = int(accounting["bytes_used_for_estimate"])
    verifier = build_ax_verifier_summary(raw, bytes_used)
    return bytes_used, str(accounting["estimate_kind"]), verifier


def build_ax_verifier_summary(raw: dict[str, Any], bytes_per_cycle: int) -> dict[str, Any]:
    cycle_rates = []
    emitted_per_cycle = []
    for result in raw["results"]:
        telemetry = result["ngram_acceleration_telemetry"]
        mlx_telemetry = result["ax_mlx_telemetry"]
        cycles = int(telemetry["ax_mtp_decode_steps"])
        emitted = int(telemetry["ax_mtp_emitted_tokens"])
        wall_s = float(mlx_telemetry["ax_mlx_decode_wall_us"]) / 1_000_000.0
        if cycles > 0 and wall_s > 0:
            cycle_rates.append(cycles / wall_s)
            emitted_per_cycle.append(emitted / cycles)
    if not cycle_rates:
        return {}
    cycles_per_s = statistics.median(cycle_rates)
    return {
        "scope": "ax_verifier_cycle",
        "bytes_per_verifier_cycle": bytes_per_cycle,
        "gb_per_verifier_cycle": round(bytes_per_cycle / 1e9, 3),
        "verifier_cycles_per_s": round(cycles_per_s, 3),
        "output_tokens_per_verifier_cycle": round(statistics.median(emitted_per_cycle), 3),
        "effective_verifier_bandwidth_gb_s": round(bytes_per_cycle * cycles_per_s / 1e9, 3),
        "percent_of_peak": round(bytes_per_cycle * cycles_per_s / 1e9 / PEAK_GBS * 100.0, 2),
    }


def mtplx_artifact_estimate(peer_row: dict[str, Any]) -> tuple[int, str, dict[str, Any] | None]:
    artifact = REPO_ROOT / str(peer_row["artifact"])
    raw = json.loads(artifact.read_text())
    model_dir = Path(raw["model_inspection"]["model_dir"])
    is_moe = "35B-A3B" in str(peer_row["model_label"])
    try:
        bytes_used, source = safetensor_header_byte_estimate(model_dir, moe_active=is_moe)
    except (FileNotFoundError, ValueError):
        fallback = PEER_PACKAGE_ACTIVE_BYTE_FALLBACKS.get(str(peer_row["model_label"]))
        if fallback is None:
            raise
        bytes_used = int(fallback["bytes"])
        source = str(fallback["source"])
    return bytes_used, source, build_mtplx_cycle_summary(raw, bytes_used)


def build_mtplx_cycle_summary(raw: dict[str, Any], bytes_per_cycle: int) -> dict[str, Any]:
    cycle_rates = []
    emitted_per_cycle = []
    for case in raw["results"]:
        for run in case["runs"]:
            if not run.get("measured"):
                continue
            generated = int(run["generated_tokens"])
            target_cycles = generated - int(run["accepted_drafts"])
            decode_elapsed_s = float(run["decode_elapsed_s"])
            if target_cycles > 0 and decode_elapsed_s > 0:
                cycle_rates.append(target_cycles / decode_elapsed_s)
                emitted_per_cycle.append(generated / target_cycles)
    if not cycle_rates:
        return {}
    cycles_per_s = statistics.median(cycle_rates)
    return {
        "scope": "mtplx_target_cycle_from_accepted_drafts",
        "bytes_per_target_cycle": bytes_per_cycle,
        "gb_per_target_cycle": round(bytes_per_cycle / 1e9, 3),
        "target_cycles_per_s": round(cycles_per_s, 3),
        "output_tokens_per_target_cycle": round(statistics.median(emitted_per_cycle), 3),
        "effective_target_cycle_bandwidth_gb_s": round(bytes_per_cycle * cycles_per_s / 1e9, 3),
        "percent_of_peak": round(bytes_per_cycle * cycles_per_s / 1e9 / PEAK_GBS * 100.0, 2),
    }


def build_diagnostic(summary_path: Path) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text())
    rows = peer_rows(summary)

    ax_bytes_by_target: dict[str, int] = {}
    peer_bytes_by_target: dict[str, int] = {}
    built: list[dict[str, Any]] = []

    for row in rows:
        target = str(row["model_label"])
        engine = str(row["engine"])
        if engine == "ax_engine":
            bytes_used, source, cycle_summary = ax_artifact_estimate(row)
            ax_bytes_by_target[target] = bytes_used
        elif engine == "mtplx":
            bytes_used, source, cycle_summary = mtplx_artifact_estimate(row)
            peer_bytes_by_target[target] = bytes_used
        else:
            continue
        built.append(build_output_row(row, bytes_used, source, cycle_summary))

    for row in rows:
        target = str(row["model_label"])
        engine = str(row["engine"])
        if engine != "lightning_mlx":
            continue
        if target == "Qwen3.6 27B 4-bit":
            bytes_used = ax_bytes_by_target[target]
            source = "same_ax_sidecar_proxy"
        else:
            bytes_used = peer_bytes_by_target.get(target)
            if bytes_used is None:
                fallback = PEER_PACKAGE_ACTIVE_BYTE_FALLBACKS[target]
                bytes_used = int(fallback["bytes"])
            source = "retained_lightning_row_peer_package_proxy"
        built.append(build_output_row(row, bytes_used, source, None))

    built.sort(
        key=lambda row: (
            TARGET_ORDER.index(str(row["target"])),
            ENGINE_ORDER.index(str(row["engine"])),
        )
    )
    return {
        "schema": "ax.qwen36_mtp_output_bandwidth_diagnostic.v1",
        "source_summary": str(summary_path.relative_to(REPO_ROOT)),
        "run_date": summary_path.parent.name[:10],
        "peak_bandwidth_gb_s": PEAK_GBS,
        "peak_bandwidth_source": PEAK_SOURCE,
        "bandwidth_scope": "effective_output_not_gpu_counter",
        "caveat": (
            "Effective output bandwidth multiplies committed-token decode throughput "
            "by active target-weight bytes. It is useful for peer readout but can "
            "exceed physical memory bandwidth when MTP accepts multiple tokens per "
            "target verifier cycle."
        ),
        "rows": built,
    }


def build_output_row(
    peer_row: dict[str, Any],
    bytes_used: int,
    estimate_source: str,
    cycle_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    decode = float(peer_row["metrics"]["decode_tok_s"])
    output_bandwidth = bytes_used * decode / 1e9
    return {
        "target": peer_row["model_label"],
        "engine": peer_row["engine"],
        "engine_label": ENGINE_LABELS[str(peer_row["engine"])],
        "decode_tok_s": round(decode, 3),
        "active_target_bytes_per_output_token": bytes_used,
        "active_target_gb_per_output_token": round(bytes_used / 1e9, 3),
        "effective_output_bandwidth_gb_s": round(output_bandwidth, 3),
        "percent_of_peak_reference": round(output_bandwidth / PEAK_GBS * 100.0, 2),
        "byte_estimate_source": estimate_source,
        "artifact": peer_row.get("artifact"),
        "cycle_summary": cycle_summary,
    }


def render_27b_output_svg(diagnostic: dict[str, Any]) -> str:
    target = "Qwen3.6 27B 4-bit"
    rows = [row for row in diagnostic["rows"] if row["target"] == target]
    rows.sort(key=lambda row: ENGINE_ORDER.index(str(row["engine"])))
    axis_max = max(
        200.0,
        ((int(max(row["percent_of_peak_reference"] for row in rows)) + 24) // 25)
        * 25.0,
    )
    def e(value: object) -> str:
        return html.escape(str(value))

    def fx(percent: float) -> float:
        return PLOT_LEFT + max(0.0, percent) / axis_max * (PLOT_RIGHT - PLOT_LEFT)

    best_output = max(float(row["effective_output_bandwidth_gb_s"]) for row in rows)
    height = 345
    top = 124
    bar_h = 22
    row_gap = 12
    footer_y = 308
    meta_output_x = 720
    meta_decode_x = 840

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" viewBox="0 0 {WIDTH} {height}" role="img" aria-labelledby="title desc">',
        "<title>Qwen3.6 27B MTP effective output work</title>",
        (
            "<desc>Same-sidecar effective output-work diagnostic comparing AX Engine, MTPLX, "
            "and lightning-mlx on Qwen3.6 27B 4-bit MTP.</desc>"
        ),
        f'<rect width="{WIDTH}" height="{height}" fill="#f8fafc"/>',
        f'<text id="title" x="{LEFT}" y="28" font-family="{FONT}" font-size="17" font-weight="700" fill="#111827">Qwen3.6 27B MTP - Effective output work (same sidecar, {e(diagnostic["run_date"])})</text>',
        f'<text id="desc" x="{LEFT}" y="50" font-family="{FONT}" font-size="11" fill="#4b5563">All engines use the same dense 27B sidecar, so active bytes match and output work tracks decode speed.</text>',
        f'<text x="{LEFT}" y="68" font-family="{FONT}" font-size="10" fill="#6b7280">Output work = decode tok/s × 16.90 GB/token; percentages are output-scaled against the {PEAK_GBS:.0f} GB/s reference.</text>',
        f'<rect x="{PLOT_LEFT}" y="{top - 28}" width="{PLOT_RIGHT - PLOT_LEFT}" height="142" rx="5" fill="#ffffff" stroke="#dbe3ef"/>',
        f'<text x="{PLOT_RIGHT}" y="{top - 45}" text-anchor="end" font-family="{FONT}" font-size="11" font-weight="700" fill="#dc2626">Higher output work is better here</text>',
        f'<text x="{meta_output_x}" y="{top - 11}" font-family="{FONT}" font-size="10" font-weight="700" fill="#374151">Output work</text>',
        f'<text x="{meta_decode_x}" y="{top - 11}" font-family="{FONT}" font-size="10" font-weight="700" fill="#374151">Decode</text>',
    ]

    for percent in (0.0, 50.0, 100.0, 150.0, 200.0):
        x = fx(percent)
        parts.append(
            f'<line x1="{x:.1f}" y1="{top - 28}" x2="{x:.1f}" y2="{top + 114}" stroke="#e5e7eb" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{top + 134}" text-anchor="middle" font-family="{FONT}" font-size="10" fill="#6b7280">{percent:.0f}%</text>'
        )
    peak_x = fx(100.0)
    parts.append(
        f'<line x1="{peak_x:.1f}" y1="{top - 28}" x2="{peak_x:.1f}" y2="{top + 114}" stroke="#dc2626" stroke-width="1.2" stroke-dasharray="5 4"/>'
    )
    parts.append(
        f'<text x="{peak_x + 5:.1f}" y="{top - 8}" font-family="{FONT}" font-size="9" font-weight="700" fill="#dc2626">100% physical reference</text>'
    )

    y = top
    for row in rows:
        fill, stroke = ENGINE_COLORS[str(row["engine"])]
        value = float(row["effective_output_bandwidth_gb_s"])
        percent = float(row["percent_of_peak_reference"])
        width = fx(percent) - PLOT_LEFT
        label_y = y + bar_h - 6
        percent_label_x = max(PLOT_LEFT + 42.0, PLOT_LEFT + width - 8.0)
        is_best = value == best_output
        output_fill = "#dc2626" if is_best else "#374151"
        output_weight = "700" if is_best else "400"
        parts.extend(
            [
                f'<text x="{PLOT_LEFT - 12}" y="{label_y}" text-anchor="end" font-family="{FONT}" font-size="10" fill="#374151">{e(row["engine_label"])}</text>',
                f'<rect x="{PLOT_LEFT}" y="{y}" width="{width:.1f}" height="{bar_h}" rx="4" fill="{fill}" fill-opacity="0.82"/>',
                f'<line x1="{PLOT_LEFT + width:.1f}" y1="{y}" x2="{PLOT_LEFT + width:.1f}" y2="{y + bar_h}" stroke="{stroke}" stroke-width="1.4"/>',
                f'<text x="{percent_label_x:.1f}" y="{label_y}" text-anchor="end" font-family="{FONT}" font-size="10" font-weight="700" fill="#ffffff">{percent:.0f}%</text>',
                f'<text x="{meta_output_x}" y="{label_y}" font-family="{FONT}" font-size="10" font-weight="{output_weight}" fill="{output_fill}">{value:.0f} GB/s</text>',
                f'<text x="{meta_decode_x}" y="{label_y}" font-family="{FONT}" font-size="10" fill="#6b7280">{row["decode_tok_s"]:.1f} tok/s</text>',
            ]
        )
        y += bar_h + row_gap

    parts.extend(render_legend(footer_y - 26))
    parts.append(
        f'<text x="{LEFT}" y="{footer_y}" font-family="{FONT}" font-size="9" fill="#9ca3af">Percentages can exceed 100% because MTP can commit multiple output tokens per target verifier pass.</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def render_legend(legend_y: int) -> list[str]:
    parts: list[str] = []
    legend_x = LEFT
    for engine in ENGINE_ORDER:
        fill, stroke = ENGINE_COLORS[engine]
        parts.extend(
            [
                f'<rect x="{legend_x}" y="{legend_y - 9}" width="10" height="10" rx="2" fill="{fill}" fill-opacity="0.82" stroke="{stroke}"/>',
                f'<text x="{legend_x + 14}" y="{legend_y}" font-family="{FONT}" font-size="10" fill="#374151">{html.escape(ENGINE_LABELS[engine])}</text>',
            ]
        )
        legend_x += 130
    return parts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--svg-output", type=Path, default=DEFAULT_SVG)
    args = parser.parse_args()

    diagnostic = build_diagnostic(args.summary)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.svg_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(diagnostic, indent=2, sort_keys=True) + "\n")
    args.svg_output.write_text(render_27b_output_svg(diagnostic))
    print(f"Wrote {args.json_output}")
    print(f"Wrote {args.svg_output}")


if __name__ == "__main__":
    main()
