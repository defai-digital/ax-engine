#!/usr/bin/env python3
"""Render diagnostic AX MLX decode-profile artifacts as Markdown reports."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class MlxDecodeProfileReportError(RuntimeError):
    pass


MAIN_STAGES = [
    ("per-layer input", "ax_mlx_decode_profile_per_layer_input_wall_us"),
    ("pre-SDPA", "ax_mlx_decode_profile_pre_sdpa_wall_us"),
    ("SDPA", "ax_mlx_decode_profile_sdpa_wall_us"),
    ("post-attention", "ax_mlx_decode_profile_post_attn_wall_us"),
    ("LM head", "ax_mlx_decode_profile_lm_head_wall_us"),
]

SUBSTAGES = [
    (
        "QKV projection",
        "ax_mlx_decode_profile_pre_sdpa_qkv_proj_wall_us",
        "ax_mlx_decode_profile_pre_sdpa_wall_us",
    ),
    (
        "QK norm",
        "ax_mlx_decode_profile_pre_sdpa_qk_norm_wall_us",
        "ax_mlx_decode_profile_pre_sdpa_wall_us",
    ),
    (
        "RoPE + KV append",
        "ax_mlx_decode_profile_pre_sdpa_rope_kv_wall_us",
        "ax_mlx_decode_profile_pre_sdpa_wall_us",
    ),
    (
        "Attention output projection",
        "ax_mlx_decode_profile_post_attn_output_proj_wall_us",
        "ax_mlx_decode_profile_post_attn_wall_us",
    ),
    (
        "Attention residual + FFN norm",
        "ax_mlx_decode_profile_post_attn_residual_norm_wall_us",
        "ax_mlx_decode_profile_post_attn_wall_us",
    ),
    (
        "FFN",
        "ax_mlx_decode_profile_post_attn_ffn_wall_us",
        "ax_mlx_decode_profile_post_attn_wall_us",
    ),
    (
        "FFN residual + layer gate",
        "ax_mlx_decode_profile_post_attn_residual_gate_wall_us",
        "ax_mlx_decode_profile_post_attn_wall_us",
    ),
]


@dataclass(frozen=True)
class DecodeProfileRow:
    model: str
    artifact: Path
    prompt_tokens: int
    generation_tokens: int
    decode_steps: int
    layers: int
    profile: dict[str, int]


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except OSError as error:
        raise MlxDecodeProfileReportError(f"failed to read {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise MlxDecodeProfileReportError(f"failed to parse {path}: {error}") from error
    if not isinstance(payload, dict):
        raise MlxDecodeProfileReportError(f"{path} is not a JSON object")
    return payload


def profile_int(profile: dict[str, Any], key: str, *, default: int | None = None) -> int:
    value = profile.get(key, default)
    if isinstance(value, int):
        return value
    raise MlxDecodeProfileReportError(f"decode profile lacks integer field {key}")


def row_shape(row: dict[str, Any]) -> tuple[int, int]:
    prompt_tokens = row.get("prompt_tokens")
    generation_tokens = row.get("generation_tokens")
    if not isinstance(prompt_tokens, int) or not isinstance(generation_tokens, int):
        raise MlxDecodeProfileReportError(f"row lacks integer prompt/generation tokens: {row}")
    return (prompt_tokens, generation_tokens)


def build_rows(artifact_path: Path) -> list[DecodeProfileRow]:
    artifact = load_json(artifact_path)
    model = str(artifact.get("model", artifact_path.stem))
    rows: list[DecodeProfileRow] = []
    for row in artifact.get("results", []):
        if not isinstance(row, dict) or row.get("engine") != "ax_engine_mlx":
            continue
        profile = row.get("ax_mlx_decode_profile")
        if not isinstance(profile, dict):
            continue
        if profile_int(profile, "ax_mlx_decode_profile_enabled") != 1:
            raise MlxDecodeProfileReportError(f"{artifact_path} has disabled decode profile")
        prompt_tokens, generation_tokens = row_shape(row)
        rows.append(
            DecodeProfileRow(
                model=model,
                artifact=artifact_path,
                prompt_tokens=prompt_tokens,
                generation_tokens=generation_tokens,
                decode_steps=profile_int(profile, "ax_mlx_decode_profile_decode_steps"),
                layers=profile_int(profile, "ax_mlx_decode_profile_layers"),
                profile={
                    key: value
                    for key, value in profile.items()
                    if key.startswith("ax_mlx_decode_profile_") and isinstance(value, int)
                },
            )
        )
    return rows


def is_decode_profile_artifact(path: Path) -> bool:
    if path.name.startswith("sweep_") or path.suffix != ".json":
        return False
    if "decode-profile" in path.stem:
        return True
    try:
        artifact = load_json(path)
    except MlxDecodeProfileReportError:
        return False
    return artifact.get("ax_decode_profile") is True


def artifact_paths(inputs: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for input_path in inputs:
        if input_path.is_dir():
            paths.extend(
                path for path in sorted(input_path.glob("*.json")) if is_decode_profile_artifact(path)
            )
        else:
            paths.append(input_path)
    return paths


def fmt_us(value: int | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:,}"


def fmt_share(value: int | None, denominator: int) -> str:
    if value is None or denominator <= 0:
        return "n/a"
    return f"{value / denominator * 100:.1f}%"


def value_or_none(profile: dict[str, int], key: str) -> int | None:
    value = profile.get(key)
    return value if isinstance(value, int) else None


def parent_remainder(profile: dict[str, int], parent_key: str, child_keys: list[str]) -> int | None:
    parent = value_or_none(profile, parent_key)
    if parent is None:
        return None
    child_total = sum(value_or_none(profile, key) or 0 for key in child_keys)
    return max(0, parent - child_total)


def render_row(row: DecodeProfileRow) -> list[str]:
    profile = row.profile
    main_total = sum(value_or_none(profile, key) or 0 for _, key in MAIN_STAGES)
    if main_total <= 0:
        raise MlxDecodeProfileReportError(f"{row.artifact} has zero decode profile stage time")

    lines = [
        f"## {row.model} p{row.prompt_tokens} g{row.generation_tokens}",
        "",
        f"- Artifact: `{row.artifact}`",
        f"- Decode steps: {row.decode_steps:,}",
        f"- Profiled layers: {row.layers:,}",
        "",
        "| Stage | Wall us | Share of profiled stage time |",
        "|---|---:|---:|",
    ]
    for label, key in MAIN_STAGES:
        value = value_or_none(profile, key)
        lines.append(f"| {label} | {fmt_us(value)} | {fmt_share(value, main_total)} |")

    lines.extend(
        [
            "",
            "| Substage | Wall us | Parent share | Total profile share |",
            "|---|---:|---:|---:|",
        ]
    )
    for label, key, parent_key in SUBSTAGES:
        value = value_or_none(profile, key)
        parent = value_or_none(profile, parent_key) or 0
        lines.append(
            f"| {label} | {fmt_us(value)} | {fmt_share(value, parent)} | {fmt_share(value, main_total)} |"
        )

    pre_tail = parent_remainder(
        profile,
        "ax_mlx_decode_profile_pre_sdpa_wall_us",
        [
            "ax_mlx_decode_profile_pre_sdpa_qkv_proj_wall_us",
            "ax_mlx_decode_profile_pre_sdpa_qk_norm_wall_us",
            "ax_mlx_decode_profile_pre_sdpa_rope_kv_wall_us",
        ],
    )
    post_tail = parent_remainder(
        profile,
        "ax_mlx_decode_profile_post_attn_wall_us",
        [
            "ax_mlx_decode_profile_post_attn_output_proj_wall_us",
            "ax_mlx_decode_profile_post_attn_residual_norm_wall_us",
            "ax_mlx_decode_profile_post_attn_ffn_wall_us",
            "ax_mlx_decode_profile_post_attn_residual_gate_wall_us",
        ],
    )
    lines.append(f"| Unsplit pre-SDPA tail | {fmt_us(pre_tail)} | n/a | {fmt_share(pre_tail, main_total)} |")
    lines.append(
        f"| Unsplit post-attention tail | {fmt_us(post_tail)} | n/a | {fmt_share(post_tail, main_total)} |"
    )

    dominant_label, dominant_key = max(
        MAIN_STAGES,
        key=lambda item: value_or_none(profile, item[1]) or 0,
    )
    dominant_value = value_or_none(profile, dominant_key) or 0
    lines.extend(
        [
            "",
            "### Reading Notes",
            "",
            (
                f"- Dominant parent stage: {dominant_label}, "
                f"{fmt_share(dominant_value, main_total)} of profiled stage time."
            ),
            "- `n/a` means the artifact predates that finer-grained counter.",
            "- This profile uses timing barriers and disables production decode pipelining; do not use it as headline throughput evidence.",
            "",
        ]
    )
    return lines


def render_report(rows: list[DecodeProfileRow], *, title: str) -> str:
    if not rows:
        raise MlxDecodeProfileReportError("no AX MLX decode profile rows found")
    lines = [
        f"# {title}",
        "",
        (
            "This report renders diagnostic `AX_MLX_DECODE_PROFILE=1` counters "
            "from inference-stack artifacts. Use it to choose direct-decode "
            "cache/fusion experiments, not as a production throughput claim."
        ),
        "",
    ]
    for row in sorted(rows, key=lambda item: (item.model, item.prompt_tokens, item.generation_tokens)):
        lines.extend(render_row(row))
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--title", default="AX MLX Decode Profile Report")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        rows: list[DecodeProfileRow] = []
        for path in artifact_paths(args.artifacts):
            rows.extend(build_rows(path))
        report = render_report(rows, title=args.title)
    except MlxDecodeProfileReportError as error:
        print(f"MLX decode profile report failed: {error}", file=sys.stderr)
        return 1
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n")
        print(f"MLX decode profile report written: {args.output}")
    else:
        print(report)
    return 0


def main_with_args_for_test(argv: list[str]) -> int:
    return main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
