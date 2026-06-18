#!/usr/bin/env python3
"""Benchmark AX direct DiffusionGemma first-block telemetry and render README charts.

DiffusionGemma is block-diffusion, not token-autoregressive. This script reports:

* prefill_tok_s from AX runner prefill wall time
* block_decode_tok_s from one committed 256-token diffusion block
* time_to_first_block_ms from prefill wall time plus the first block wall time

It intentionally does not use fixed-token ignore-EOS mode. That path is a
separate autoregressive benchmark contract and is not currently the published
DiffusionGemma direct-mode metric.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--mlx-community--diffusiongemma-26B-A4B-it-4bit"
    / "snapshots"
    / "252183330817f96e9cba0b20cc400b2947a575cf"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "benchmarks"
    / "results"
    / "diffusion-gemma-direct"
    / "2026-06-18-direct-first-block"
)
DEFAULT_ASSETS_DIR = REPO_ROOT / "docs" / "assets"
BENCH_SCHEMA = "ax.diffusion_gemma_direct_first_block.v1"
M5_MAX_THEORETICAL_MEMORY_BANDWIDTH_GB_S = 614.4
FONT = "Inter,Segoe UI,Arial,sans-serif"
WIDTH = 800
HEIGHT = 430
LEFT = 64
RIGHT = 640
TOP = 86
BOTTOM = 316
AX_COLOR = "#2eaf5f"
AX_DARK = "#176c37"
RED = "#dc2626"


@dataclass(frozen=True)
class TrialMetrics:
    prompt_tokens: int
    output_tokens: int
    prefill_wall_us: int
    prefill_forward_wall_us: int
    decode_wall_us: int
    block_wall_us: int
    denoise_wall_us: int
    commit_wall_us: int
    denoise_steps: int
    converged_blocks: int
    diffusion_blocks: int
    step_count: int
    ttft_step: int | None
    finish_reason: str
    prefill_tok_s: float
    block_decode_tok_s: float
    time_to_first_block_ms: float


def log(message: str) -> None:
    print(f"[diffusion-gemma-direct] {message}", flush=True)


def parse_prompt_tokens(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("--prompt-tokens must contain at least one integer")
    if any(value <= 0 for value in values):
        raise ValueError("--prompt-tokens values must be positive")
    return values


def synthetic_tokens(prompt_tokens: int) -> list[int]:
    # Stay away from common low special-token IDs while keeping the prompt fully
    # deterministic and tokenizer-free.
    return [1000 + ((prompt_tokens * 9973 + i * 37) % 120000) for i in range(prompt_tokens)]


def median(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot calculate median of an empty list")
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    variance = sum((value - avg) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def short_number(value: float) -> str:
    if value >= 1000:
        compact = value / 1000
        return f"{compact:.0f}k" if compact.is_integer() else f"{compact:.1f}k"
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def nice_axis_ceiling(value: float) -> float:
    if value <= 0:
        return 1.0
    magnitude = 10 ** math.floor(math.log10(value))
    normalized = value / magnitude
    for candidate in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0):
        if normalized <= candidate:
            return candidate * magnitude
    return 10.0 * magnitude


def token_hash(tokens: list[int]) -> str:
    payload = ",".join(str(token) for token in tokens).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def safetensors_weight_bytes(model_dir: Path) -> int:
    files = sorted(model_dir.glob("*.safetensors"))
    if not files:
        return 0
    return sum(path.stat().st_size for path in files)


def add_bandwidth_estimates(rows: list[dict[str, Any]], *, model_weight_bytes: int) -> None:
    if model_weight_bytes <= 0:
        return
    model_weight_gb = model_weight_bytes / 1_000_000_000.0
    for row in rows:
        denoise_steps = float(row["diffusion_denoise_steps"]["median"])
        block_wall_s = 256.0 / float(row["block_decode_tok_s"]["median"])
        estimated_bytes_per_block = model_weight_bytes * (denoise_steps + 1.0)
        estimated_gb_per_block = estimated_bytes_per_block / 1_000_000_000.0
        effective_gb_s = estimated_gb_per_block / block_wall_s
        row["estimated_weight_bytes"] = {
            "model_weight_bytes": model_weight_bytes,
            "model_weight_gb": model_weight_gb,
            "passes_per_block": denoise_steps + 1.0,
            "estimated_gb_per_block": estimated_gb_per_block,
        }
        row["effective_bandwidth_gb_s"] = {
            "median": effective_gb_s,
            "theoretical_peak_gb_s": M5_MAX_THEORETICAL_MEMORY_BANDWIDTH_GB_S,
        }
        row["memory_bandwidth_utilization_pct"] = {
            "median": effective_gb_s / M5_MAX_THEORETICAL_MEMORY_BANDWIDTH_GB_S * 100.0,
            "theoretical_peak_gb_s": M5_MAX_THEORETICAL_MEMORY_BANDWIDTH_GB_S,
        }


def extract_counter(response: dict[str, Any], key: str) -> int:
    counters = response["route"]["crossover_decisions"]
    value = counters.get(key)
    if not isinstance(value, int):
        raise ValueError(f"missing integer route counter {key}")
    return value


def parse_trial_response(response: dict[str, Any], prompt_tokens: int, canvas_size: int) -> TrialMetrics:
    output_tokens = response.get("output_tokens") or []
    if not isinstance(output_tokens, list):
        raise ValueError("response output_tokens must be a list")
    finish_reason = str(response.get("finish_reason", "unknown"))
    step_count = int(response.get("step_count") or 0)
    ttft_step_raw = response.get("ttft_step")
    ttft_step = int(ttft_step_raw) if isinstance(ttft_step_raw, int) else None
    prefill_wall_us = extract_counter(response, "ax_mlx_prefill_wall_us")
    prefill_forward_wall_us = extract_counter(response, "ax_mlx_prefill_forward_wall_us")
    decode_wall_us = extract_counter(response, "ax_mlx_decode_wall_us")
    block_wall_us = extract_counter(response, "ax_mlx_diffusion_block_wall_us")
    denoise_wall_us = extract_counter(response, "ax_mlx_diffusion_denoise_wall_us")
    commit_wall_us = extract_counter(response, "ax_mlx_diffusion_commit_wall_us")
    denoise_steps = extract_counter(response, "ax_mlx_diffusion_denoise_steps")
    converged_blocks = extract_counter(response, "ax_mlx_diffusion_converged_blocks")
    diffusion_blocks = extract_counter(response, "ax_mlx_diffusion_blocks")
    if prefill_wall_us <= 0 or block_wall_us <= 0:
        raise ValueError("prefill and diffusion block wall counters must be positive")
    return TrialMetrics(
        prompt_tokens=prompt_tokens,
        output_tokens=len(output_tokens),
        prefill_wall_us=prefill_wall_us,
        prefill_forward_wall_us=prefill_forward_wall_us,
        decode_wall_us=decode_wall_us,
        block_wall_us=block_wall_us,
        denoise_wall_us=denoise_wall_us,
        commit_wall_us=commit_wall_us,
        denoise_steps=denoise_steps,
        converged_blocks=converged_blocks,
        diffusion_blocks=diffusion_blocks,
        step_count=step_count,
        ttft_step=ttft_step,
        finish_reason=finish_reason,
        prefill_tok_s=prompt_tokens * 1_000_000.0 / prefill_wall_us,
        block_decode_tok_s=canvas_size * 1_000_000.0 / block_wall_us,
        time_to_first_block_ms=(prefill_wall_us + block_wall_us) / 1000.0,
    )


def trial_to_json(trial: TrialMetrics, *, phase: str, index: int) -> dict[str, Any]:
    return {
        "phase": phase,
        "index": index,
        "prompt_tokens": trial.prompt_tokens,
        "output_tokens": trial.output_tokens,
        "prefill_wall_us": trial.prefill_wall_us,
        "prefill_forward_wall_us": trial.prefill_forward_wall_us,
        "decode_wall_us": trial.decode_wall_us,
        "diffusion_block_wall_us": trial.block_wall_us,
        "diffusion_denoise_wall_us": trial.denoise_wall_us,
        "diffusion_commit_wall_us": trial.commit_wall_us,
        "diffusion_denoise_steps": trial.denoise_steps,
        "diffusion_converged_blocks": trial.converged_blocks,
        "diffusion_blocks": trial.diffusion_blocks,
        "step_count": trial.step_count,
        "ttft_step": trial.ttft_step,
        "finish_reason": trial.finish_reason,
        "prefill_tok_s": trial.prefill_tok_s,
        "block_decode_tok_s": trial.block_decode_tok_s,
        "time_to_first_block_ms": trial.time_to_first_block_ms,
    }


def summarize_trials(prompt_tokens: int, trials: list[TrialMetrics]) -> dict[str, Any]:
    return {
        "prompt_tokens": prompt_tokens,
        "measure_runs": len(trials),
        "prefill_tok_s": {
            "median": median([trial.prefill_tok_s for trial in trials]),
            "mean": mean([trial.prefill_tok_s for trial in trials]),
            "stdev": stdev([trial.prefill_tok_s for trial in trials]),
        },
        "block_decode_tok_s": {
            "median": median([trial.block_decode_tok_s for trial in trials]),
            "mean": mean([trial.block_decode_tok_s for trial in trials]),
            "stdev": stdev([trial.block_decode_tok_s for trial in trials]),
        },
        "time_to_first_block_ms": {
            "median": median([trial.time_to_first_block_ms for trial in trials]),
            "mean": mean([trial.time_to_first_block_ms for trial in trials]),
            "stdev": stdev([trial.time_to_first_block_ms for trial in trials]),
        },
        "diffusion_denoise_steps": {
            "median": median([float(trial.denoise_steps) for trial in trials]),
            "min": min(trial.denoise_steps for trial in trials),
            "max": max(trial.denoise_steps for trial in trials),
        },
        "diffusion_commit_ms": {
            "median": median([trial.commit_wall_us / 1000.0 for trial in trials]),
            "mean": mean([trial.commit_wall_us / 1000.0 for trial in trials]),
        },
        "output_tokens": {
            "median": median([float(trial.output_tokens) for trial in trials]),
            "min": min(trial.output_tokens for trial in trials),
            "max": max(trial.output_tokens for trial in trials),
        },
    }


def run_one(
    *,
    bench_bin: Path,
    model_dir: Path,
    prompt_tokens: int,
    max_output_tokens: int,
    canvas_size: int,
    phase: str,
    index: int,
    log_dir: Path,
) -> TrialMetrics:
    tokens = synthetic_tokens(prompt_tokens)
    command = [
        str(bench_bin),
        "generate",
        "--mlx",
        "--model-id",
        "diffusiongemma",
        "--tokens",
        ",".join(str(token) for token in tokens),
        "--max-output-tokens",
        str(max_output_tokens),
        "--mlx-model-artifacts-dir",
        str(model_dir),
        "--json",
    ]
    log_path = log_dir / f"p{prompt_tokens}-{phase}-{index}.log"
    started = time.time()
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    elapsed_s = time.time() - started
    log_path.write_text(result.stdout, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(
            f"benchmark command failed for prompt={prompt_tokens} {phase}#{index}; "
            f"exit={result.returncode}; log={log_path}"
        )
    try:
        response = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"benchmark output is not JSON for {log_path}: {error}") from error
    trial = parse_trial_response(response, prompt_tokens, canvas_size)
    log(
        f"p{prompt_tokens} {phase}#{index}: prefill={trial.prefill_tok_s:.1f} tok/s "
        f"block={trial.block_decode_tok_s:.1f} tok/s first_block={trial.time_to_first_block_ms:.1f} ms "
        f"steps={trial.denoise_steps} wall={elapsed_s:.1f}s"
    )
    return trial


def render_chart(
    *,
    title: str,
    metric_key: str,
    unit: str,
    rows: list[dict[str, Any]],
    lower_is_better: bool,
    note: str,
    direction_label: str | None = None,
) -> str:
    values = [float(row[metric_key]["median"]) for row in rows]
    if not values:
        raise ValueError("no data to chart")
    axis_max = nice_axis_ceiling(max(values) * 1.05)
    plot_width = RIGHT - LEFT
    plot_height = BOTTOM - TOP
    group_step = plot_width / len(rows)
    bar_w = 34.0
    header_right = WIDTH - 34
    unit_w = max(48, len(unit) * 7 + 24)
    best = min(values) if lower_is_better else max(values)
    direction = direction_label or ("Lower is better" if lower_is_better else "Higher is better")
    direction_fill = RED if lower_is_better else "#374151"

    def fy(value: float) -> float:
        return BOTTOM - (max(0.0, min(value, axis_max)) / axis_max) * plot_height

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="title desc">',
        f"<title>{escape(title)}</title>",
        f"<desc>{escape(title)}; AX Engine direct first-block measurements at 128/512/2048 prompt tokens.</desc>",
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="#f8fafc"/>',
        f'<text id="title" x="{LEFT}" y="24" font-family="{FONT}" font-size="16" '
        f'font-weight="700" fill="#111827">{escape(title)}</text>',
        f'<text x="{LEFT}" y="46" font-family="{FONT}" font-size="11" fill="#4b5563">'
        f"median over reps | grouped by prompt tokens | first committed block</text>",
        f'<text id="desc" x="{LEFT}" y="62" font-family="{FONT}" font-size="10" fill="#6b7280">'
        f"AX-only DiffusionGemma direct telemetry; peer runtime blockers are documented in the README table</text>",
        f'<text x="{LEFT}" y="76" font-family="{FONT}" font-size="9" fill="#9ca3af">'
        f"{escape(note)}</text>",
        f'<rect x="{header_right - unit_w}" y="13" width="{unit_w}" height="22" '
        f'rx="11" fill="#eef2ff" stroke="#c7d2fe"/>',
        f'<text x="{header_right - unit_w / 2:.1f}" y="28" text-anchor="middle" '
        f'font-family="{FONT}" font-size="10" font-weight="700" fill="#3730a3">{escape(unit)}</text>',
        f'<text x="{header_right}" y="52" text-anchor="end" font-family="{FONT}" '
        f'font-size="10" font-weight="700" fill="{direction_fill}">{direction}</text>',
        f'<rect x="{LEFT}" y="{TOP}" width="{plot_width}" height="{plot_height}" '
        f'rx="6" fill="#ffffff" stroke="#dbe3ef"/>',
    ]

    for index in range(5):
        value = axis_max * index / 4
        gy = fy(value)
        parts.append(
            f'<line x1="{LEFT}" y1="{gy:.1f}" x2="{RIGHT}" y2="{gy:.1f}" '
            f'stroke="#e5e7eb" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{LEFT - 8}" y="{gy + 3:.1f}" text-anchor="end" '
            f'font-family="{FONT}" font-size="11" fill="#6b7280">{short_number(value)}</text>'
        )

    best_y = fy(best)
    parts.append(
        f'<line x1="{LEFT}" y1="{best_y:.1f}" x2="{RIGHT}" y2="{best_y:.1f}" '
        f'stroke="{RED}" stroke-width="1.2" stroke-dasharray="1 4" stroke-linecap="round"/>'
    )
    parts.append(
        f'<text x="{RIGHT + 8}" y="{max(TOP + 11, best_y - 5):.1f}" '
        f'text-anchor="start" font-family="{FONT}" font-size="11" font-weight="700" '
        f'fill="{RED}">{"lowest" if lower_is_better else "highest"}: {short_number(best)}</text>'
    )

    for group_index, row in enumerate(rows):
        prompt = int(row["prompt_tokens"])
        value = float(row[metric_key]["median"])
        center = LEFT + (group_index + 0.5) * group_step
        bar_x = center - bar_w / 2
        y_med = fy(value)
        bar_h = max(1.0, BOTTOM - y_med)
        parts.extend(
            [
                f'<rect x="{bar_x:.1f}" y="{y_med:.1f}" width="{bar_w:.1f}" '
                f'height="{bar_h:.1f}" rx="3" fill="{AX_COLOR}" fill-opacity="0.40" '
                f'stroke="{AX_DARK}" stroke-width="1.8"/>',
                f'<line x1="{bar_x:.1f}" y1="{y_med:.1f}" x2="{bar_x + bar_w:.1f}" '
                f'y2="{y_med:.1f}" stroke="{AX_DARK}" stroke-width="2.6"/>',
                f'<text x="{center:.1f}" y="{max(TOP + 11, y_med - 6):.1f}" '
                f'text-anchor="middle" font-family="{FONT}" font-size="10" '
                f'font-weight="700" fill="#111827">{short_number(value)}</text>',
                f'<text x="{center:.1f}" y="{BOTTOM + 30}" text-anchor="middle" '
                f'font-family="{FONT}" font-size="11" font-weight="700" fill="#111827">{prompt} tok</text>',
            ]
        )

    legend_y = HEIGHT - 22
    legend_x = LEFT
    label = "AX Engine direct"
    parts.append(
        f'<rect x="{legend_x:.1f}" y="{legend_y - 9}" width="12" height="10" rx="2" '
        f'fill="{AX_COLOR}" fill-opacity="0.40" stroke="{AX_DARK}" stroke-width="1.4"/>'
    )
    parts.append(
        f'<text x="{legend_x + 16:.1f}" y="{legend_y}" font-family="{FONT}" '
        f'font-size="10" fill="#374151">{label}</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts)


def render_bandwidth_share_chart(rows: list[dict[str, Any]]) -> str:
    plot_left = LEFT + 78
    plot_right = RIGHT
    plot_width = plot_right - plot_left
    panel_y = TOP
    panel_h = BOTTOM - TOP
    bar_h = 24.0
    row_gap = 58.0
    first_y = TOP + 32.0
    header_right = WIDTH - 34
    headroom_color = "#e5e7eb"
    headroom_stroke = "#cbd5e1"

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="title desc">',
        "<title>DiffusionGemma 4-bit - Memory bandwidth share</title>",
        "<desc>DiffusionGemma 4-bit - Memory bandwidth share; 100% stacked bars show "
        "effective bandwidth used versus theoretical headroom at 128/512/2048 prompt tokens.</desc>",
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="#f8fafc"/>',
        f'<text id="title" x="{LEFT}" y="24" font-family="{FONT}" font-size="16" '
        f'font-weight="700" fill="#111827">DiffusionGemma 4-bit - Memory bandwidth share</text>',
        f'<text x="{LEFT}" y="46" font-family="{FONT}" font-size="11" fill="#4b5563">'
        f"median over reps | grouped by prompt tokens | first committed block</text>",
        f'<text id="desc" x="{LEFT}" y="62" font-family="{FONT}" font-size="10" fill="#6b7280">'
        f"AX-only DiffusionGemma direct telemetry; peer runtime blockers are documented in the README table</text>",
        f'<text x="{LEFT}" y="76" font-family="{FONT}" font-size="9" fill="#9ca3af">'
        f"100% = M5 Max theoretical unified-memory bandwidth</text>",
        f'<rect x="{header_right - 48}" y="13" width="48" height="22" rx="11" '
        f'fill="#eef2ff" stroke="#c7d2fe"/>',
        f'<text x="{header_right - 24}" y="28" text-anchor="middle" font-family="{FONT}" '
        f'font-size="10" font-weight="700" fill="#3730a3">%</text>',
        f'<text x="{header_right}" y="52" text-anchor="end" font-family="{FONT}" '
        f'font-size="10" font-weight="700" fill="#374151">Used vs headroom</text>',
        f'<rect x="{LEFT}" y="{panel_y}" width="{RIGHT - LEFT}" height="{panel_h}" '
        f'rx="6" fill="#ffffff" stroke="#dbe3ef"/>',
    ]

    for percent in (0, 25, 50, 75, 100):
        x = plot_left + plot_width * percent / 100.0
        parts.append(
            f'<line x1="{x:.1f}" y1="{panel_y}" x2="{x:.1f}" y2="{BOTTOM}" '
            f'stroke="#e5e7eb" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{BOTTOM + 18}" text-anchor="middle" '
            f'font-family="{FONT}" font-size="10" fill="#6b7280">{percent}%</text>'
        )

    for index, row in enumerate(rows):
        prompt = int(row["prompt_tokens"])
        used_pct = float(row["memory_bandwidth_utilization_pct"]["median"])
        used_w = plot_width * used_pct / 100.0
        y = first_y + index * row_gap
        label_x = max(plot_left + 33.0, plot_left + used_w - 8.0)
        parts.extend(
            [
                f'<text x="{LEFT + 8}" y="{y + bar_h / 2 + 4:.1f}" '
                f'font-family="{FONT}" font-size="11" font-weight="700" '
                f'fill="#111827">{prompt} tok</text>',
                f'<rect x="{plot_left:.1f}" y="{y:.1f}" width="{plot_width:.1f}" '
                f'height="{bar_h:.1f}" rx="5" fill="{headroom_color}" '
                f'stroke="{headroom_stroke}" stroke-width="1.2"/>',
                f'<rect x="{plot_left:.1f}" y="{y:.1f}" width="{used_w:.1f}" '
                f'height="{bar_h:.1f}" rx="5" fill="{AX_COLOR}" fill-opacity="0.76"/>',
                f'<line x1="{plot_left + used_w:.1f}" y1="{y:.1f}" '
                f'x2="{plot_left + used_w:.1f}" y2="{y + bar_h:.1f}" '
                f'stroke="{AX_DARK}" stroke-width="1.4"/>',
                f'<text x="{label_x:.1f}" y="{y + bar_h / 2 + 4:.1f}" '
                f'text-anchor="end" font-family="{FONT}" font-size="10" '
                f'font-weight="700" fill="#ffffff">{used_pct:.1f}%</text>',
            ]
        )

    legend_y = HEIGHT - 22
    legend_x = LEFT
    parts.extend(
        [
            f'<rect x="{legend_x:.1f}" y="{legend_y - 9}" width="12" height="10" rx="2" '
            f'fill="{AX_COLOR}" fill-opacity="0.76"/>',
            f'<text x="{legend_x + 16:.1f}" y="{legend_y}" font-family="{FONT}" '
            f'font-size="10" fill="#374151">Used bandwidth</text>',
            f'<rect x="{legend_x + 128:.1f}" y="{legend_y - 9}" width="12" height="10" rx="2" '
            f'fill="{headroom_color}" stroke="{headroom_stroke}" stroke-width="1"/>',
            f'<text x="{legend_x + 144:.1f}" y="{legend_y}" font-family="{FONT}" '
            f'font-size="10" fill="#374151">Theoretical headroom</text>',
            "</svg>",
        ]
    )
    return "\n".join(parts)


def write_summary(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# DiffusionGemma Direct First-Block Benchmark",
        "",
        f"- schema: `{BENCH_SCHEMA}`",
        "- runtime: AX Engine native MLX direct",
        "- model: `mlx-community/diffusiongemma-26B-A4B-it-4bit`",
        "- method: 2 warmup + 5 measure runs, median reported",
        "- metric boundary: first committed diffusion block, not fixed-token autoregressive TTFT",
        "",
        "| Prompt tokens | Block decode | Prefill | Time to first block | Denoise steps |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {prompt_tokens} | {decode:.1f} tok/s | {prefill:,.1f} tok/s | "
            "{ttfb:,.0f} ms | {steps:.0f} |".format(
                prompt_tokens=row["prompt_tokens"],
                decode=row["block_decode_tok_s"]["median"],
                prefill=row["prefill_tok_s"]["median"],
                ttfb=row["time_to_first_block_ms"]["median"],
                steps=row["diffusion_denoise_steps"]["median"],
            )
        )
    lines.extend(
        [
            "",
            "| Prompt tokens | Estimated effective bandwidth | % of 614.4 GB/s M5 Max theoretical bandwidth |",
            "|---:|---:|---:|",
        ]
    )
    for row in rows:
        bandwidth = row.get("effective_bandwidth_gb_s", {}).get("median")
        utilization = row.get("memory_bandwidth_utilization_pct", {}).get("median")
        if isinstance(bandwidth, (int, float)) and isinstance(utilization, (int, float)):
            lines.append(
                f"| {row['prompt_tokens']} | {bandwidth:.1f} GB/s | {utilization:.1f}% |"
            )
    lines.extend(
        [
            "",
            "Effective bandwidth estimates use local safetensors bytes times "
            "`denoise_steps + 1 commit` per block, divided by measured block wall time, "
            "against the 614.4 GB/s M5 Max theoretical unified-memory bandwidth.",
            "",
            "Peer runtimes are intentionally N/A: current llama.cpp and mlx-lm releases "
            "cannot load DiffusionGemma model artifacts.",
            "",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench-bin", type=Path, default=REPO_ROOT / "target" / "debug" / "ax-engine-bench")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--assets-dir", type=Path, default=DEFAULT_ASSETS_DIR)
    parser.add_argument("--prompt-tokens", default="128,512,2048")
    parser.add_argument("--warmup-iters", type=int, default=2)
    parser.add_argument("--measure-iters", type=int, default=5)
    parser.add_argument("--max-output-tokens", type=int, default=1)
    parser.add_argument("--canvas-size", type=int, default=256)
    parser.add_argument("--skip-benchmark", action="store_true")
    args = parser.parse_args()

    prompt_sizes = parse_prompt_tokens(args.prompt_tokens)
    output_dir = args.output_root
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    args.assets_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_benchmark:
        if not args.bench_bin.is_file():
            raise SystemExit(f"bench binary not found: {args.bench_bin}")
        if not (args.model_dir / "model-manifest.json").is_file():
            raise SystemExit(f"DiffusionGemma model manifest not found: {args.model_dir}")
        model_weight_bytes = safetensors_weight_bytes(args.model_dir)

        rows: list[dict[str, Any]] = []
        all_trials: list[dict[str, Any]] = []
        for prompt_tokens in prompt_sizes:
            tokens = synthetic_tokens(prompt_tokens)
            warmups: list[TrialMetrics] = []
            measures: list[TrialMetrics] = []
            log(f"prompt={prompt_tokens}: warmup={args.warmup_iters} measure={args.measure_iters}")
            for index in range(1, args.warmup_iters + 1):
                trial = run_one(
                    bench_bin=args.bench_bin,
                    model_dir=args.model_dir,
                    prompt_tokens=prompt_tokens,
                    max_output_tokens=args.max_output_tokens,
                    canvas_size=args.canvas_size,
                    phase="warmup",
                    index=index,
                    log_dir=log_dir,
                )
                warmups.append(trial)
                all_trials.append(trial_to_json(trial, phase="warmup", index=index))
            for index in range(1, args.measure_iters + 1):
                trial = run_one(
                    bench_bin=args.bench_bin,
                    model_dir=args.model_dir,
                    prompt_tokens=prompt_tokens,
                    max_output_tokens=args.max_output_tokens,
                    canvas_size=args.canvas_size,
                    phase="measure",
                    index=index,
                    log_dir=log_dir,
                )
                measures.append(trial)
                all_trials.append(trial_to_json(trial, phase="measure", index=index))
            row = summarize_trials(prompt_tokens, measures)
            row["prompt_sha256"] = token_hash(tokens)
            row["warmup_runs"] = len(warmups)
            rows.append(row)
        add_bandwidth_estimates(rows, model_weight_bytes=model_weight_bytes)
        artifact = {
            "schema_version": BENCH_SCHEMA,
            "model": {
                "id": "mlx-community/diffusiongemma-26B-A4B-it-4bit",
                "family": "diffusion_gemma",
                "quant": "4bit",
                "canvas_size": args.canvas_size,
                "model_dir": str(args.model_dir),
                "safetensors_weight_bytes": model_weight_bytes,
            },
            "runtime": {
                "engine": "ax_engine_native_mlx",
                "bench_bin": str(args.bench_bin),
                "benchmark_contract": "first_committed_diffusion_block",
                "fixed_token_ignore_eos": "not_used",
            },
            "methodology": {
                "warmup_iters": args.warmup_iters,
                "measure_iters": args.measure_iters,
                "reported_stat": "median",
                "max_output_tokens": args.max_output_tokens,
                "prompt_token_generator": "1000 + ((prompt_tokens * 9973 + i * 37) % 120000)",
            },
            "bandwidth_model": {
                "schema": "ax.estimated_weight_bandwidth.v1",
                "model_weight_bytes": model_weight_bytes,
                "theoretical_peak_gb_s": M5_MAX_THEORETICAL_MEMORY_BANDWIDTH_GB_S,
                "estimated_bytes_per_block": "model_weight_bytes * (diffusion_denoise_steps + 1 commit)",
                "effective_bandwidth_gb_s": "estimated_bytes_per_block / measured_diffusion_block_wall_s",
            },
            "rows": rows,
            "trials": all_trials,
            "peer_runtime_status": {
                "llama_cpp_metal": "N/A: GGUF load fails with unknown model architecture diffusion-gemma",
                "mlx_lm": "N/A: mlx-lm 0.31.3 does not support model type diffusion_gemma",
            },
        }
        (output_dir / "summary.json").write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
        write_summary(output_dir, rows)
    else:
        artifact = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
        model_dir = Path(artifact["model"].get("model_dir", str(args.model_dir)))
        model_weight_bytes = int(
            artifact["model"].get("safetensors_weight_bytes") or safetensors_weight_bytes(model_dir)
        )
        rows = artifact["rows"]
        add_bandwidth_estimates(rows, model_weight_bytes=model_weight_bytes)
        artifact["model"]["safetensors_weight_bytes"] = model_weight_bytes
        artifact["bandwidth_model"] = {
            "schema": "ax.estimated_weight_bandwidth.v1",
            "model_weight_bytes": model_weight_bytes,
            "theoretical_peak_gb_s": M5_MAX_THEORETICAL_MEMORY_BANDWIDTH_GB_S,
            "estimated_bytes_per_block": "model_weight_bytes * (diffusion_denoise_steps + 1 commit)",
            "effective_bandwidth_gb_s": "estimated_bytes_per_block / measured_diffusion_block_wall_s",
        }
        (output_dir / "summary.json").write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
        write_summary(output_dir, rows)

    chart_specs = [
        (
            "DiffusionGemma 4-bit - First block decode",
            "block_decode_tok_s",
            "tok/s",
            False,
            "Throughput is one 256-token diffusion block divided by block wall time.",
            "perf-diffusiongemma-direct-decode-tok-s.svg",
            None,
        ),
        (
            "DiffusionGemma 4-bit - Prefill",
            "prefill_tok_s",
            "tok/s",
            False,
            "Prefill uses AX runner wall time from the same first-block runs.",
            "perf-diffusiongemma-direct-prefill-tok-s.svg",
            None,
        ),
        (
            "DiffusionGemma 4-bit - Time to first block",
            "time_to_first_block_ms",
            "ms",
            True,
            "Runner time to first committed block: prefill wall plus first diffusion block wall.",
            "perf-diffusiongemma-direct-ttft-ms.svg",
            None,
        ),
    ]
    for (
        title,
        metric_key,
        unit,
        lower_is_better,
        note,
        filename,
        direction_label,
    ) in chart_specs:
        svg = render_chart(
            title=title,
            metric_key=metric_key,
            unit=unit,
            rows=rows,
            lower_is_better=lower_is_better,
            note=note,
            direction_label=direction_label,
        )
        (args.assets_dir / filename).write_text(svg + "\n", encoding="utf-8")
        log(f"wrote {args.assets_dir / filename}")

    bandwidth_svg = render_bandwidth_share_chart(rows)
    bandwidth_filename = args.assets_dir / "perf-diffusiongemma-direct-memory-bandwidth-share.svg"
    bandwidth_filename.write_text(bandwidth_svg + "\n", encoding="utf-8")
    log(f"wrote {bandwidth_filename}")


if __name__ == "__main__":
    main()
