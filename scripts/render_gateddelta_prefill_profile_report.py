#!/usr/bin/env python3
"""Render validated GatedDelta prefill profile artifacts as Markdown reports."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from check_gateddelta_prefill_profile_artifact import (
    GatedDeltaPrefillProfileArtifactError,
    load_json,
    validate_gateddelta_prefill_profile_artifact,
)


class GatedDeltaPrefillProfileReportError(RuntimeError):
    pass


STAGE_KEYS = [
    ("projection", "ax_mlx_linear_attention_profile_projection_wall_us"),
    ("conv", "ax_mlx_linear_attention_profile_conv_wall_us"),
    ("qk_norm", "ax_mlx_linear_attention_profile_qk_norm_wall_us"),
    ("recurrent", "ax_mlx_linear_attention_profile_recurrent_wall_us"),
    ("output", "ax_mlx_linear_attention_profile_output_wall_us"),
]


@dataclass(frozen=True)
class ReportRow:
    prompt_tokens: int
    generation_tokens: int
    ax_prefill_tok_s: float
    prefill_ratio_to_mlx_lm: float
    prefill_wall_ms: float
    profile_layers: int
    profile_tokens: int
    stage_ms: dict[str, float]
    recurrent_share: float
    dominant_stage: str
    decision_hint: str


def metric_value(row: dict[str, Any], metric: str, stat: str) -> float:
    payload = row.get(metric)
    if not isinstance(payload, dict) or not isinstance(payload.get(stat), (int, float)):
        raise GatedDeltaPrefillProfileReportError(
            f"{row.get('engine')} prompt={row.get('prompt_tokens')} lacks {metric}.{stat}"
        )
    return float(payload[stat])


def int_value(payload: dict[str, Any], key: str, *, owner: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int):
        raise GatedDeltaPrefillProfileReportError(f"{owner} lacks integer field {key}")
    return value


def row_by_shape(artifact: dict[str, Any]) -> dict[tuple[int, int], dict[str, dict[str, Any]]]:
    groups: dict[tuple[int, int], dict[str, dict[str, Any]]] = {}
    for row in artifact.get("results", []):
        if not isinstance(row, dict):
            continue
        shape = (int(row["prompt_tokens"]), int(row["generation_tokens"]))
        groups.setdefault(shape, {})[str(row["engine"])] = row
    return groups


def decision_hint(recurrent_share: float, dominant_stage: str) -> str:
    if recurrent_share >= 0.5:
        return "prioritize recurrent scan"
    if dominant_stage in {"projection", "output"}:
        return "inspect projection/fusion"
    if dominant_stage == "conv":
        return "inspect conv/state update"
    return "compare mixed-stage overhead"


def build_report_rows(artifact: dict[str, Any]) -> list[ReportRow]:
    rows: list[ReportRow] = []
    for (prompt_tokens, generation_tokens), engines in sorted(row_by_shape(artifact).items()):
        ax_row = engines.get("ax_engine_mlx")
        if ax_row is None:
            raise GatedDeltaPrefillProfileReportError(
                f"prompt={prompt_tokens} generation={generation_tokens} lacks AX row"
            )
        profile = ax_row.get("ax_mlx_linear_attention_profile")
        if not isinstance(profile, dict):
            raise GatedDeltaPrefillProfileReportError(
                f"prompt={prompt_tokens} generation={generation_tokens} lacks profile"
            )
        telemetry = ax_row.get("ax_mlx_telemetry")
        if not isinstance(telemetry, dict):
            raise GatedDeltaPrefillProfileReportError(
                f"prompt={prompt_tokens} generation={generation_tokens} lacks AX telemetry"
            )
        baseline = ax_row.get("baseline")
        if not isinstance(baseline, dict) or not isinstance(
            baseline.get("prefill_ratio_to_mlx_lm"), (int, float)
        ):
            raise GatedDeltaPrefillProfileReportError(
                f"prompt={prompt_tokens} generation={generation_tokens} lacks prefill ratio"
            )

        stage_us = {
            label: int_value(profile, key, owner=f"prompt={prompt_tokens}.profile")
            for label, key in STAGE_KEYS
        }
        total_stage_us = sum(stage_us.values())
        if total_stage_us <= 0:
            raise GatedDeltaPrefillProfileReportError(
                f"prompt={prompt_tokens} generation={generation_tokens} has zero profile time"
            )
        dominant = max(stage_us, key=lambda key: stage_us[key])
        recurrent_share = stage_us["recurrent"] / total_stage_us
        rows.append(
            ReportRow(
                prompt_tokens=prompt_tokens,
                generation_tokens=generation_tokens,
                ax_prefill_tok_s=metric_value(ax_row, "prefill_tok_s", "median"),
                prefill_ratio_to_mlx_lm=float(baseline["prefill_ratio_to_mlx_lm"]),
                prefill_wall_ms=int_value(
                    telemetry,
                    "ax_mlx_prefill_wall_us",
                    owner=f"prompt={prompt_tokens}.ax_mlx_telemetry",
                )
                / 1000.0,
                profile_layers=int_value(
                    profile,
                    "ax_mlx_linear_attention_profile_layers",
                    owner=f"prompt={prompt_tokens}.profile",
                ),
                profile_tokens=int_value(
                    profile,
                    "ax_mlx_linear_attention_profile_tokens",
                    owner=f"prompt={prompt_tokens}.profile",
                ),
                stage_ms={label: value / 1000.0 for label, value in stage_us.items()},
                recurrent_share=recurrent_share,
                dominant_stage=dominant,
                decision_hint=decision_hint(recurrent_share, dominant),
            )
        )
    return rows


def fmt_number(value: float, digits: int = 1) -> str:
    return f"{value:,.{digits}f}"


def render_report(artifact_path: Path) -> str:
    validate_gateddelta_prefill_profile_artifact(artifact_path)
    artifact = load_json(artifact_path)
    model_config = artifact.get("model_config", {})
    profile = artifact.get("gateddelta_prefill_profile", {})
    model_preflight = profile.get("model_preflight", {})
    linear_attention = (
        model_preflight.get("linear_attention", {})
        if isinstance(model_preflight, dict)
        else {}
    )
    host = artifact.get("host", {})
    rows = build_report_rows(artifact)

    lines = [
        "# GatedDelta Prefill Profile Report",
        "",
        f"- Artifact: `{artifact_path}`",
        f"- Model: `{artifact.get('model', 'unknown')}`",
        f"- Family: `{model_config.get('model_family', model_config.get('model_type', 'unknown'))}`",
        (
            "- Model preflight: "
            f"`{model_preflight.get('status', 'unknown')}` via "
            f"`{model_preflight.get('checker', 'unknown')}`, "
            f"schema={model_preflight.get('schema_version', 'unknown')}, "
            f"key_head_dim={linear_attention.get('key_head_dim', 'unknown')}"
        ),
        f"- Host: {host.get('chip', 'unknown')} / {host.get('memory_gb', 'unknown')} GB",
        (
            f"- Benchmark: generation={artifact.get('generation_tokens', 'unknown')}, "
            f"repetitions={artifact.get('repetitions', 'unknown')}, "
            f"prefill_step_size={artifact.get('prefill_step_size', 'unknown')}"
        ),
        "",
        (
            "Timing scope: stage counters are opt-in diagnostic timing-barrier "
            "measurements from `AX_MLX_LINEAR_ATTENTION_PROFILE=1`; use them to "
            "choose the next kernel experiment, not as headline throughput."
        ),
        "",
        "| Prompt tok | AX prefill tok/s | AX/MLX prefill | Prefill wall ms | LA layers | LA tokens | Projection ms | Conv ms | QK norm ms | Recurrent ms | Output ms | Recurrent % | Dominant | Next hint |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row.prompt_tokens:,} | "
            f"{fmt_number(row.ax_prefill_tok_s)} | "
            f"{row.prefill_ratio_to_mlx_lm:.3f}x | "
            f"{fmt_number(row.prefill_wall_ms)} | "
            f"{row.profile_layers:,} | "
            f"{row.profile_tokens:,} | "
            f"{fmt_number(row.stage_ms['projection'])} | "
            f"{fmt_number(row.stage_ms['conv'])} | "
            f"{fmt_number(row.stage_ms['qk_norm'])} | "
            f"{fmt_number(row.stage_ms['recurrent'])} | "
            f"{fmt_number(row.stage_ms['output'])} | "
            f"{row.recurrent_share * 100:.1f}% | "
            f"{row.dominant_stage} | "
            f"{row.decision_hint} |"
        )

    max_recurrent = max(rows, key=lambda row: row.recurrent_share) if rows else None
    lines.extend(["", "## Interpretation Guardrails", ""])
    if max_recurrent:
        lines.append(
            f"- Highest recurrent share: {max_recurrent.recurrent_share * 100:.1f}% "
            f"at {max_recurrent.prompt_tokens:,} prompt tokens."
        )
    lines.extend(
        [
            "- If recurrent share dominates at long prompts, prototype chunked/parallel scan first.",
            "- If projection or output dominates, inspect fusion around the recurrent update before scan work.",
            "- Do not compare this profile directly with non-profile throughput rows; timing barriers perturb latency.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a validated GatedDelta prefill profile artifact as Markdown."
    )
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = render_report(args.artifact)
    except (
        GatedDeltaPrefillProfileArtifactError,
        GatedDeltaPrefillProfileReportError,
    ) as error:
        print(f"GatedDelta prefill profile report failed: {error}", file=sys.stderr)
        return 1
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n")
        print(f"GatedDelta prefill profile report written: {args.output}")
    else:
        print(report)
    return 0


def main_with_args_for_test(argv: list[str]) -> int:
    return main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
