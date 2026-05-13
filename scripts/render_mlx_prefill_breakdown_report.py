#!/usr/bin/env python3
"""Render AX MLX prefill breakdown rows from inference-stack artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class PrefillBreakdownReportError(RuntimeError):
    pass


@dataclass(frozen=True)
class PrefillBreakdownRow:
    model: str
    artifact: Path
    prompt_tokens: int
    ax_prefill_tok_s: float
    mlx_lm_prefill_tok_s: float | None
    mlx_swift_lm_prefill_tok_s: float | None
    llama_cpp_prefill_tok_s: float | None
    prefill_wall_ms: float
    forward_ms: float
    prefix_cache_ms: float
    generation_state_ms: float
    other_ms: float
    forward_share: float
    eval_barriers: int
    drain_async_evals: int

    @property
    def ax_to_mlx_lm(self) -> float | None:
        return ratio(self.ax_prefill_tok_s, self.mlx_lm_prefill_tok_s)

    @property
    def ax_to_llama_cpp(self) -> float | None:
        return ratio(self.ax_prefill_tok_s, self.llama_cpp_prefill_tok_s)


def ratio(numerator: float, denominator: float | None) -> float | None:
    if denominator is None or denominator <= 0:
        return None
    return numerator / denominator


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except OSError as error:
        raise PrefillBreakdownReportError(f"failed to read {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise PrefillBreakdownReportError(f"failed to parse {path}: {error}") from error
    if not isinstance(payload, dict):
        raise PrefillBreakdownReportError(f"{path} is not a JSON object")
    return payload


def metric_median(row: dict[str, Any], key: str) -> float | None:
    metric = row.get(key)
    if isinstance(metric, dict) and isinstance(metric.get("median"), (int, float)):
        return float(metric["median"])
    if isinstance(metric, (int, float)):
        return float(metric)
    return None


def telemetry_int(row: dict[str, Any], key: str) -> int:
    telemetry = row.get("ax_mlx_telemetry")
    if not isinstance(telemetry, dict):
        raise PrefillBreakdownReportError(
            f"{row.get('engine')} prompt={row.get('prompt_tokens')} lacks ax_mlx_telemetry"
        )
    value = telemetry.get(key, 0)
    if not isinstance(value, int):
        raise PrefillBreakdownReportError(
            f"{row.get('engine')} prompt={row.get('prompt_tokens')} has non-integer {key}"
        )
    return value


def row_key(row: dict[str, Any]) -> tuple[int, int]:
    prompt_tokens = row.get("prompt_tokens")
    generation_tokens = row.get("generation_tokens")
    if not isinstance(prompt_tokens, int) or not isinstance(generation_tokens, int):
        raise PrefillBreakdownReportError(f"row lacks integer prompt/generation tokens: {row}")
    return (prompt_tokens, generation_tokens)


def rows_by_shape(artifact: dict[str, Any]) -> dict[tuple[int, int], dict[str, dict[str, Any]]]:
    grouped: dict[tuple[int, int], dict[str, dict[str, Any]]] = {}
    for row in artifact.get("results", []):
        if not isinstance(row, dict):
            continue
        grouped.setdefault(row_key(row), {})[str(row.get("engine"))] = row
    return grouped


def llama_cpp_prefill_by_prompt(path: Path | None) -> dict[int, float]:
    if path is None or not path.exists():
        return {}
    artifact = load_json(path)
    values: dict[int, float] = {}
    for row in artifact.get("results", []):
        if not isinstance(row, dict) or row.get("engine") != "llama_cpp_metal":
            continue
        prompt_tokens = row.get("prompt_tokens")
        prefill = metric_median(row, "prefill_tok_s")
        if isinstance(prompt_tokens, int) and prefill is not None:
            values[prompt_tokens] = prefill
    return values


def build_rows(artifact_path: Path, *, llama_dir: Path | None = None) -> list[PrefillBreakdownRow]:
    artifact = load_json(artifact_path)
    llama_values = llama_cpp_prefill_by_prompt(
        llama_dir / artifact_path.name if llama_dir is not None else None
    )
    model = str(artifact.get("model", artifact_path.stem))
    rows: list[PrefillBreakdownRow] = []
    for (prompt_tokens, _generation_tokens), engines in sorted(rows_by_shape(artifact).items()):
        ax_row = engines.get("ax_engine_mlx")
        if ax_row is None:
            continue
        ax_prefill = metric_median(ax_row, "prefill_tok_s")
        if ax_prefill is None:
            raise PrefillBreakdownReportError(f"{artifact_path} prompt={prompt_tokens} lacks AX prefill")

        prefill_wall_us = telemetry_int(ax_row, "ax_mlx_prefill_wall_us")
        forward_us = telemetry_int(ax_row, "ax_mlx_prefill_forward_wall_us")
        prefix_cache_us = telemetry_int(ax_row, "ax_mlx_prefill_prefix_cache_wall_us")
        generation_state_us = telemetry_int(ax_row, "ax_mlx_prefill_generation_state_wall_us")
        measured_us = forward_us + prefix_cache_us + generation_state_us
        other_us = max(0, prefill_wall_us - measured_us)
        forward_share = ratio(float(forward_us), float(prefill_wall_us)) or 0.0

        rows.append(
            PrefillBreakdownRow(
                model=model,
                artifact=artifact_path,
                prompt_tokens=prompt_tokens,
                ax_prefill_tok_s=ax_prefill,
                mlx_lm_prefill_tok_s=metric_median(engines.get("mlx_lm", {}), "prefill_tok_s"),
                mlx_swift_lm_prefill_tok_s=metric_median(
                    engines.get("mlx_swift_lm", {}),
                    "prefill_tok_s",
                ),
                llama_cpp_prefill_tok_s=llama_values.get(prompt_tokens),
                prefill_wall_ms=prefill_wall_us / 1000.0,
                forward_ms=forward_us / 1000.0,
                prefix_cache_ms=prefix_cache_us / 1000.0,
                generation_state_ms=generation_state_us / 1000.0,
                other_ms=other_us / 1000.0,
                forward_share=forward_share,
                eval_barriers=telemetry_int(ax_row, "ax_mlx_prefill_eval_barriers"),
                drain_async_evals=telemetry_int(ax_row, "ax_mlx_prefill_drain_async_evals"),
            )
        )
    return rows


DIAGNOSTIC_ARTIFACT_SUFFIXES = (
    "-decode-profile",
    "-gemma4-moe-profile",
    "-linear-profile",
    "-prefill-profile",
)


def is_diagnostic_artifact(path: Path) -> bool:
    return any(suffix in path.stem for suffix in DIAGNOSTIC_ARTIFACT_SUFFIXES)


def artifact_paths(inputs: list[Path], *, include_diagnostics: bool = False) -> list[Path]:
    paths: list[Path] = []
    for input_path in inputs:
        if input_path.is_dir():
            paths.extend(
                path
                for path in sorted(input_path.glob("*.json"))
                if not path.name.startswith("sweep_")
                and (include_diagnostics or not is_diagnostic_artifact(path))
            )
        else:
            if not include_diagnostics and is_diagnostic_artifact(input_path):
                continue
            paths.append(input_path)
    return paths


def fmt_number(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    return f"{value:,.{digits}f}"


def fmt_ratio(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}x"


def sort_rows(rows: list[PrefillBreakdownRow]) -> list[PrefillBreakdownRow]:
    return sorted(
        rows,
        key=lambda row: (
            row.ax_to_llama_cpp is None,
            row.ax_to_llama_cpp if row.ax_to_llama_cpp is not None else row.ax_to_mlx_lm or 999.0,
            row.model,
            row.prompt_tokens,
        ),
    )


def render_report(rows: list[PrefillBreakdownRow], *, title: str) -> str:
    if not rows:
        raise PrefillBreakdownReportError("no AX MLX prefill rows found")
    sorted_rows = sort_rows(rows)
    worst_llama = next((row for row in sorted_rows if row.ax_to_llama_cpp is not None), None)
    lines = [
        f"# {title}",
        "",
        (
            "This report decomposes AX MLX prefill timing from inference-stack artifacts. "
            "`llama.cpp Metal` values are shape-compatible external GGUF references when supplied; "
            "they are not prompt-hash parity evidence."
        ),
        "",
        "| Model | Prompt tok | AX prefill tok/s | AX/MLX | AX/llama.cpp | Prefill ms | Forward ms | Prefix cache ms | Generation state ms | Other ms | Forward % | Eval barriers | Async drains |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted_rows:
        lines.append(
            "| "
            f"{row.model} | "
            f"{row.prompt_tokens:,} | "
            f"{fmt_number(row.ax_prefill_tok_s)} | "
            f"{fmt_ratio(row.ax_to_mlx_lm)} | "
            f"{fmt_ratio(row.ax_to_llama_cpp)} | "
            f"{fmt_number(row.prefill_wall_ms)} | "
            f"{fmt_number(row.forward_ms)} | "
            f"{fmt_number(row.prefix_cache_ms)} | "
            f"{fmt_number(row.generation_state_ms)} | "
            f"{fmt_number(row.other_ms)} | "
            f"{row.forward_share * 100:.1f}% | "
            f"{row.eval_barriers} | "
            f"{row.drain_async_evals} |"
        )

    lines.extend(["", "## Reading Notes", ""])
    if worst_llama is not None:
        lines.append(
            "- Worst AX/llama.cpp row: "
            f"`{worst_llama.model}` prompt={worst_llama.prompt_tokens:,}, "
            f"{fmt_ratio(worst_llama.ax_to_llama_cpp)}."
        )
    lines.extend(
        [
            "- `Forward ms` is the model forward plus final prefill token materialization path.",
            "- `Prefix cache ms` covers prompt-prefix snapshot storage after forward.",
            "- `Generation state ms` covers decode-state initialization after a completing prefill.",
            "- Non-forward overhead that is high at 128 tokens is serving-path overhead, not a tensor-kernel bottleneck by itself.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--llama-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--title", default="AX MLX Prefill Breakdown Report")
    parser.add_argument(
        "--include-diagnostics",
        action="store_true",
        help="Include diagnostic/profile artifacts such as *-linear-profile.json.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        rows: list[PrefillBreakdownRow] = []
        for path in artifact_paths(
            args.artifacts,
            include_diagnostics=args.include_diagnostics,
        ):
            rows.extend(build_rows(path, llama_dir=args.llama_dir))
        report = render_report(rows, title=args.title)
    except PrefillBreakdownReportError as error:
        print(f"Prefill breakdown report failed: {error}", file=sys.stderr)
        return 1

    if args.output:
        args.output.write_text(report)
        print(f"Prefill breakdown report written: {args.output}")
    else:
        print(report, end="")
    return 0


def main_with_args_for_test(argv: list[str]) -> int:
    return main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
