#!/usr/bin/env python3
"""Render diagnostic AX MLX forward-profile artifacts as Markdown reports."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class MlxForwardProfileReportError(RuntimeError):
    pass


STAGE_KEYS = [
    ("projection", "ax_mlx_linear_attention_profile_projection_wall_us"),
    ("conv", "ax_mlx_linear_attention_profile_conv_wall_us"),
    ("qk_norm", "ax_mlx_linear_attention_profile_qk_norm_wall_us"),
    ("recurrent", "ax_mlx_linear_attention_profile_recurrent_wall_us"),
    ("output", "ax_mlx_linear_attention_profile_output_wall_us"),
]

PROJECTION_STAGE_KEYS = [
    ("qkvz", "ax_mlx_linear_attention_profile_projection_qkvz_wall_us"),
    ("ba", "ax_mlx_linear_attention_profile_projection_ba_wall_us"),
    ("qkv", "ax_mlx_linear_attention_profile_projection_qkv_wall_us"),
    ("z", "ax_mlx_linear_attention_profile_projection_z_wall_us"),
    ("a", "ax_mlx_linear_attention_profile_projection_a_wall_us"),
    ("b", "ax_mlx_linear_attention_profile_projection_b_wall_us"),
]

INVALID_PROFILE_TOKEN_SENTINELS = {4_294_967_295}


@dataclass(frozen=True)
class ForwardProfileRow:
    model: str
    artifact: Path
    prompt_tokens: int
    generation_tokens: int
    ax_prefill_tok_s: float
    mlx_lm_prefill_tok_s: float | None
    mlx_swift_lm_prefill_tok_s: float | None
    prefill_wall_ms: float | None
    forward_ms: float | None
    profile_layers: int
    profile_tokens: int
    stage_ms: dict[str, float]
    projection_ms: dict[str, float | None]
    projection_layout: str | None
    offline_pack_candidate: bool | None
    stage_total_ms: float
    dominant_stage: str
    dominant_share: float
    decision_hint: str

    @property
    def ax_to_mlx_lm(self) -> float | None:
        return ratio(self.ax_prefill_tok_s, self.mlx_lm_prefill_tok_s)

    @property
    def ax_to_mlx_swift_lm(self) -> float | None:
        return ratio(self.ax_prefill_tok_s, self.mlx_swift_lm_prefill_tok_s)


def ratio(numerator: float, denominator: float | None) -> float | None:
    if denominator is None or denominator <= 0:
        return None
    return numerator / denominator


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except OSError as error:
        raise MlxForwardProfileReportError(f"failed to read {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise MlxForwardProfileReportError(f"failed to parse {path}: {error}") from error
    if not isinstance(payload, dict):
        raise MlxForwardProfileReportError(f"{path} is not a JSON object")
    return payload


def metric_median(row: dict[str, Any], key: str) -> float | None:
    metric = row.get(key)
    if isinstance(metric, dict) and isinstance(metric.get("median"), (int, float)):
        return float(metric["median"])
    if isinstance(metric, (int, float)):
        return float(metric)
    return None


def int_value(payload: dict[str, Any], key: str, *, owner: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int):
        raise MlxForwardProfileReportError(f"{owner} lacks integer field {key}")
    return value


def optional_int_value(payload: dict[str, Any], key: str) -> int | None:
    value = payload.get(key)
    return value if isinstance(value, int) else None


def optional_telemetry_ms(row: dict[str, Any], key: str) -> float | None:
    telemetry = row.get("ax_mlx_telemetry")
    if not isinstance(telemetry, dict):
        return None
    value = telemetry.get(key)
    if not isinstance(value, int):
        return None
    return value / 1000.0


def row_shape(row: dict[str, Any]) -> tuple[int, int]:
    prompt_tokens = row.get("prompt_tokens")
    generation_tokens = row.get("generation_tokens")
    if not isinstance(prompt_tokens, int) or not isinstance(generation_tokens, int):
        raise MlxForwardProfileReportError(f"row lacks integer prompt/generation tokens: {row}")
    return (prompt_tokens, generation_tokens)


def rows_by_shape(artifact: dict[str, Any]) -> dict[tuple[int, int], dict[str, dict[str, Any]]]:
    grouped: dict[tuple[int, int], dict[str, dict[str, Any]]] = {}
    for row in artifact.get("results", []):
        if not isinstance(row, dict):
            continue
        grouped.setdefault(row_shape(row), {})[str(row.get("engine"))] = row
    return grouped


def validate_profile(
    profile: dict[str, Any],
    *,
    artifact_path: Path,
    prompt_tokens: int,
) -> None:
    if int_value(
        profile,
        "ax_mlx_linear_attention_profile_enabled",
        owner=f"{artifact_path} prompt={prompt_tokens}.profile",
    ) != 1:
        raise MlxForwardProfileReportError(
            f"{artifact_path} prompt={prompt_tokens} has disabled linear-attention profile"
        )
    layers = int_value(
        profile,
        "ax_mlx_linear_attention_profile_layers",
        owner=f"{artifact_path} prompt={prompt_tokens}.profile",
    )
    tokens = int_value(
        profile,
        "ax_mlx_linear_attention_profile_tokens",
        owner=f"{artifact_path} prompt={prompt_tokens}.profile",
    )
    if layers <= 0:
        raise MlxForwardProfileReportError(
            f"{artifact_path} prompt={prompt_tokens} has non-positive profile layer count"
        )
    if tokens in INVALID_PROFILE_TOKEN_SENTINELS:
        raise MlxForwardProfileReportError(
            f"{artifact_path} prompt={prompt_tokens} has stale profile token sentinel {tokens}; "
            "rerun the diagnostic artifact with the fixed token counter"
        )
    if tokens < prompt_tokens:
        raise MlxForwardProfileReportError(
            f"{artifact_path} prompt={prompt_tokens} has profile tokens below prompt tokens"
        )


def projection_decision_hint(
    projection_ms: dict[str, float | None],
    *,
    projection_layout: str | None,
    offline_pack_candidate: bool | None,
) -> str:
    qkv = projection_ms.get("qkv") or 0.0
    qkvz = projection_ms.get("qkvz") or 0.0
    ba = projection_ms.get("ba") or 0.0
    split_total = sum((projection_ms.get(key) or 0.0) for key in ["qkv", "z", "a", "b"])
    packed_total = qkvz + ba
    if offline_pack_candidate and projection_layout == "split_qkv_z_a_b" and split_total > 0.0:
        return "evaluate offline packed qkvz/ba projection"
    if packed_total > 0.0:
        return "inspect packed projection fusion"
    if qkv > 0.0:
        return "inspect qkv projection layout"
    return "inspect projection/layout fusion"


def decision_hint(
    dominant_stage: str,
    dominant_share: float,
    *,
    projection_ms: dict[str, float | None],
    projection_layout: str | None,
    offline_pack_candidate: bool | None,
) -> str:
    if dominant_stage == "recurrent" and dominant_share >= 0.5:
        return "prioritize recurrent scan experiment"
    if dominant_stage == "projection":
        return projection_decision_hint(
            projection_ms,
            projection_layout=projection_layout,
            offline_pack_candidate=offline_pack_candidate,
        )
    if dominant_stage == "output":
        return "inspect output projection fusion"
    if dominant_stage == "conv":
        return "inspect conv/state update"
    return "compare mixed-stage overhead"


def build_rows(artifact_path: Path) -> list[ForwardProfileRow]:
    artifact = load_json(artifact_path)
    model = str(artifact.get("model", artifact_path.stem))
    model_config = artifact.get("model_config")
    projection_layout_payload = (
        model_config.get("linear_attention_projection_layout")
        if isinstance(model_config, dict)
        else None
    )
    projection_layout = (
        str(projection_layout_payload.get("layout"))
        if isinstance(projection_layout_payload, dict)
        and isinstance(projection_layout_payload.get("layout"), str)
        else None
    )
    offline_pack_candidate = (
        bool(projection_layout_payload.get("offline_pack_candidate"))
        if isinstance(projection_layout_payload, dict)
        and isinstance(projection_layout_payload.get("offline_pack_candidate"), bool)
        else None
    )
    rows: list[ForwardProfileRow] = []
    for (prompt_tokens, generation_tokens), engines in sorted(rows_by_shape(artifact).items()):
        ax_row = engines.get("ax_engine_mlx")
        if ax_row is None:
            continue
        profile = ax_row.get("ax_mlx_linear_attention_profile")
        if not isinstance(profile, dict):
            continue
        validate_profile(profile, artifact_path=artifact_path, prompt_tokens=prompt_tokens)
        ax_prefill = metric_median(ax_row, "prefill_tok_s")
        if ax_prefill is None:
            raise MlxForwardProfileReportError(
                f"{artifact_path} prompt={prompt_tokens} lacks AX prefill metric"
            )
        stage_us = {
            label: int_value(
                profile,
                key,
                owner=f"{artifact_path} prompt={prompt_tokens}.profile",
            )
            for label, key in STAGE_KEYS
        }
        stage_total_us = sum(stage_us.values())
        if stage_total_us <= 0:
            raise MlxForwardProfileReportError(
                f"{artifact_path} prompt={prompt_tokens} has zero profile stage time"
            )
        projection_us = {
            label: optional_int_value(profile, key) for label, key in PROJECTION_STAGE_KEYS
        }
        projection_ms = {
            label: (value / 1000.0 if value is not None else None)
            for label, value in projection_us.items()
        }
        dominant_stage = max(stage_us, key=lambda key: stage_us[key])
        dominant_share = stage_us[dominant_stage] / stage_total_us
        rows.append(
            ForwardProfileRow(
                model=model,
                artifact=artifact_path,
                prompt_tokens=prompt_tokens,
                generation_tokens=generation_tokens,
                ax_prefill_tok_s=ax_prefill,
                mlx_lm_prefill_tok_s=metric_median(
                    engines.get("mlx_lm", {}),
                    "prefill_tok_s",
                ),
                mlx_swift_lm_prefill_tok_s=metric_median(
                    engines.get("mlx_swift_lm", {}),
                    "prefill_tok_s",
                ),
                prefill_wall_ms=optional_telemetry_ms(ax_row, "ax_mlx_prefill_wall_us"),
                forward_ms=optional_telemetry_ms(ax_row, "ax_mlx_prefill_forward_wall_us"),
                profile_layers=int_value(
                    profile,
                    "ax_mlx_linear_attention_profile_layers",
                    owner=f"{artifact_path} prompt={prompt_tokens}.profile",
                ),
                profile_tokens=int_value(
                    profile,
                    "ax_mlx_linear_attention_profile_tokens",
                    owner=f"{artifact_path} prompt={prompt_tokens}.profile",
                ),
                stage_ms={label: value / 1000.0 for label, value in stage_us.items()},
                projection_ms=projection_ms,
                projection_layout=projection_layout,
                offline_pack_candidate=offline_pack_candidate,
                stage_total_ms=stage_total_us / 1000.0,
                dominant_stage=dominant_stage,
                dominant_share=dominant_share,
                decision_hint=decision_hint(
                    dominant_stage,
                    dominant_share,
                    projection_ms=projection_ms,
                    projection_layout=projection_layout,
                    offline_pack_candidate=offline_pack_candidate,
                ),
            )
        )
    return rows


def is_forward_profile_artifact(path: Path) -> bool:
    if path.name.startswith("sweep_") or path.suffix != ".json":
        return False
    if "linear-profile" in path.stem or "prefill-profile" in path.stem:
        return True
    try:
        artifact = load_json(path)
    except MlxForwardProfileReportError:
        return False
    return artifact.get("ax_linear_attention_profile") is True


def artifact_paths(inputs: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for input_path in inputs:
        if input_path.is_dir():
            paths.extend(
                path for path in sorted(input_path.glob("*.json")) if is_forward_profile_artifact(path)
            )
        else:
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


def fmt_percent_ratio(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def fmt_bool(value: bool | None) -> str:
    if value is None:
        return "n/a"
    return "yes" if value else "no"


def sort_rows(rows: list[ForwardProfileRow]) -> list[ForwardProfileRow]:
    return sorted(
        rows,
        key=lambda row: (
            row.ax_to_mlx_lm is None,
            row.ax_to_mlx_lm if row.ax_to_mlx_lm is not None else 999.0,
            -row.dominant_share,
            row.model,
            row.prompt_tokens,
        ),
    )


def render_report(rows: list[ForwardProfileRow], *, title: str) -> str:
    if not rows:
        raise MlxForwardProfileReportError("no AX MLX forward profile rows found")
    sorted_rows = sort_rows(rows)
    lines = [
        f"# {title}",
        "",
        (
            "This report renders diagnostic `AX_MLX_LINEAR_ATTENTION_PROFILE=1` "
            "stage counters from inference-stack artifacts. Timing barriers perturb "
            "latency, so use this to choose the next kernel experiment, not as a "
            "headline throughput claim."
        ),
        "",
        "| Model | Prompt tok | AX prefill tok/s | AX/MLX | AX/SwiftLM | Prefill ms | Forward ms | LA layers | LA tokens | Projection ms | Conv ms | QK norm ms | Recurrent ms | Output ms | Dominant | Dominant % | Next hint | Artifact |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|---|",
    ]
    for row in sorted_rows:
        lines.append(
            "| "
            f"{row.model} | "
            f"{row.prompt_tokens:,} | "
            f"{fmt_number(row.ax_prefill_tok_s)} | "
            f"{fmt_ratio(row.ax_to_mlx_lm)} | "
            f"{fmt_ratio(row.ax_to_mlx_swift_lm)} | "
            f"{fmt_number(row.prefill_wall_ms)} | "
            f"{fmt_number(row.forward_ms)} | "
            f"{row.profile_layers:,} | "
            f"{row.profile_tokens:,} | "
            f"{fmt_number(row.stage_ms['projection'])} | "
            f"{fmt_number(row.stage_ms['conv'])} | "
            f"{fmt_number(row.stage_ms['qk_norm'])} | "
            f"{fmt_number(row.stage_ms['recurrent'])} | "
            f"{fmt_number(row.stage_ms['output'])} | "
            f"{row.dominant_stage} | "
            f"{row.dominant_share * 100:.1f}% | "
            f"{row.decision_hint} | "
            f"`{row.artifact}` |"
        )

    lines.extend(
        [
            "",
            "## Projection Breakdown",
            "",
            "| Model | Prompt tok | Layout | Offline pack candidate | Projection ms | QKVZ ms | BA ms | QKV ms | Z ms | A ms | B ms | QKV share | Split tail share |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted_rows:
        projection_ms = row.stage_ms["projection"]
        qkv_ms = row.projection_ms.get("qkv")
        split_tail_ms = sum((row.projection_ms.get(key) or 0.0) for key in ["z", "a", "b"])
        lines.append(
            "| "
            f"{row.model} | "
            f"{row.prompt_tokens:,} | "
            f"{row.projection_layout or 'n/a'} | "
            f"{fmt_bool(row.offline_pack_candidate)} | "
            f"{fmt_number(projection_ms)} | "
            f"{fmt_number(row.projection_ms.get('qkvz'))} | "
            f"{fmt_number(row.projection_ms.get('ba'))} | "
            f"{fmt_number(qkv_ms)} | "
            f"{fmt_number(row.projection_ms.get('z'))} | "
            f"{fmt_number(row.projection_ms.get('a'))} | "
            f"{fmt_number(row.projection_ms.get('b'))} | "
            f"{fmt_percent_ratio(ratio(qkv_ms or 0.0, projection_ms))} | "
            f"{fmt_percent_ratio(ratio(split_tail_ms, projection_ms))} |"
        )

    worst_mlx = next((row for row in sorted_rows if row.ax_to_mlx_lm is not None), None)
    strongest_stage = max(sorted_rows, key=lambda row: row.dominant_share)
    lines.extend(["", "## Reading Notes", ""])
    if worst_mlx is not None:
        lines.append(
            "- Lowest AX/MLX row in this diagnostic set: "
            f"`{worst_mlx.model}` prompt={worst_mlx.prompt_tokens:,}, "
            f"{fmt_ratio(worst_mlx.ax_to_mlx_lm)}."
        )
    lines.append(
        "- Strongest single-stage concentration: "
        f"`{strongest_stage.model}` prompt={strongest_stage.prompt_tokens:,}, "
        f"{strongest_stage.dominant_stage}={strongest_stage.dominant_share * 100:.1f}%."
    )
    lines.extend(
        [
            "- Compare this with the prefill breakdown report first: if forward is not dominant, do not use this report to justify kernel work.",
            "- Projection substage cells are `n/a` for artifacts captured before the projection split counters existed.",
            "- Reject stale artifacts with `ax_mlx_linear_attention_profile_tokens=4294967295`; that value came from an old signed/unsigned clamp bug.",
            "- Keep barrier-profile artifacts out of README headline tables.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--title", default="AX MLX Forward Profile Report")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        rows: list[ForwardProfileRow] = []
        for path in artifact_paths(args.artifacts):
            rows.extend(build_rows(path))
        report = render_report(rows, title=args.title)
    except MlxForwardProfileReportError as error:
        print(f"MLX forward profile report failed: {error}", file=sys.stderr)
        return 1
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n")
        print(f"MLX forward profile report written: {args.output}")
    else:
        print(report)
    return 0


def main_with_args_for_test(argv: list[str]) -> int:
    return main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
