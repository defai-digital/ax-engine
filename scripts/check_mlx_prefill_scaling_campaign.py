#!/usr/bin/env python3
"""Validate and summarize a multi-model MLX prefill scaling campaign."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from check_mlx_prefill_scaling_artifact import (
    PrefillScalingArtifactError,
    load_json,
    validate_prefill_scaling_artifact,
)
from render_mlx_prefill_scaling_report import build_report_rows


DEFAULT_REQUIRED_FAMILIES = ["gemma", "qwen", "glm"]


class PrefillScalingCampaignError(RuntimeError):
    pass


@dataclass(frozen=True)
class CampaignSummary:
    path: Path
    model_id: str
    family: str
    host_key: str
    contexts: tuple[int, ...]
    max_context_tokens: int
    first_bend_context_tokens: int | None
    best_prefill_ratio: float
    worst_ttft_ratio: float


def model_text(artifact: dict[str, Any]) -> str:
    model = artifact.get("model", {})
    values: list[str] = []
    if isinstance(model, dict):
        for key in ("id", "dir", "model_type", "model_family", "quantization"):
            value = model.get(key)
            if value is not None:
                values.append(str(value))
    values.append(str(artifact.get("source_artifact", "")))
    return " ".join(values).lower()


def detect_family(artifact: dict[str, Any]) -> str:
    text = model_text(artifact)
    for family in ("gemma", "qwen", "glm"):
        if family in text:
            return family
    return "unknown"


def host_key(artifact: dict[str, Any]) -> str:
    host = artifact.get("host", {})
    if not isinstance(host, dict):
        return "unknown"
    return "|".join(
        str(host.get(key, "unknown"))
        for key in ("chip", "memory_gb", "os_version")
    )


def summarize_artifact(
    path: Path,
    *,
    bend_drop_ratio: float,
    min_context_count: int,
    min_largest_context_tokens: int,
) -> CampaignSummary:
    validate_prefill_scaling_artifact(
        path,
        min_context_count=min_context_count,
        min_largest_context_tokens=min_largest_context_tokens,
    )
    artifact = load_json(path)
    rows = build_report_rows(artifact, bend_drop_ratio=bend_drop_ratio)
    model = artifact.get("model", {})
    model_id = str(model.get("id", "unknown")) if isinstance(model, dict) else "unknown"
    contexts = tuple(row.context_tokens for row in rows)
    bend_rows = [row for row in rows if row.bend_note]
    return CampaignSummary(
        path=path,
        model_id=model_id,
        family=detect_family(artifact),
        host_key=host_key(artifact),
        contexts=contexts,
        max_context_tokens=max(contexts),
        first_bend_context_tokens=bend_rows[0].context_tokens if bend_rows else None,
        best_prefill_ratio=max(row.prefill_ratio for row in rows),
        worst_ttft_ratio=max(row.ttft_ratio for row in rows),
    )


def validate_campaign(
    artifact_paths: list[Path],
    *,
    required_families: list[str],
    allow_mixed_host: bool,
    bend_drop_ratio: float,
    min_context_count: int,
    min_largest_context_tokens: int,
) -> list[CampaignSummary]:
    if not artifact_paths:
        raise PrefillScalingCampaignError("at least one artifact is required")
    summaries = [
        summarize_artifact(
            path,
            bend_drop_ratio=bend_drop_ratio,
            min_context_count=min_context_count,
            min_largest_context_tokens=min_largest_context_tokens,
        )
        for path in artifact_paths
    ]

    seen_families = {summary.family for summary in summaries}
    missing = sorted(set(required_families) - seen_families)
    if missing:
        raise PrefillScalingCampaignError(
            f"campaign lacks required model families: {missing}; seen={sorted(seen_families)}"
        )

    host_keys = {summary.host_key for summary in summaries}
    if not allow_mixed_host and len(host_keys) > 1:
        raise PrefillScalingCampaignError(
            "campaign mixes host identities; pass --allow-mixed-host to render "
            f"cross-host evidence explicitly: {sorted(host_keys)}"
        )
    return summaries


def render_campaign_report(summaries: list[CampaignSummary]) -> str:
    lines = [
        "# MLX Prefill Scaling Campaign",
        "",
        "| Family | Model | Max context | Contexts | First bend | Best AX/MLX prefill | Worst AX/MLX TTFT | Artifact |",
        "|---|---|---:|---|---:|---:|---:|---|",
    ]
    for summary in summaries:
        first_bend = (
            f"{summary.first_bend_context_tokens:,}"
            if summary.first_bend_context_tokens is not None
            else ""
        )
        contexts = ", ".join(f"{context:,}" for context in summary.contexts)
        lines.append(
            "| "
            f"{summary.family} | "
            f"`{summary.model_id}` | "
            f"{summary.max_context_tokens:,} | "
            f"{contexts} | "
            f"{first_bend} | "
            f"{summary.best_prefill_ratio:.3f}x | "
            f"{summary.worst_ttft_ratio:.3f}x | "
            f"`{summary.path}` |"
        )
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- This campaign covers direct AX prefill/TTFT scaling only.",
            "- It does not validate n-gram decode acceleration, cold start, concurrency, or KV eviction.",
            "- Required model-family coverage should be chosen before running the campaign.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a multi-model MLX prefill/TTFT scaling campaign."
    )
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument(
        "--required-family",
        action="append",
        default=[],
        help="Required model family label. Repeat for gemma/qwen/glm campaign coverage.",
    )
    parser.add_argument("--default-required-families", action="store_true")
    parser.add_argument("--allow-mixed-host", action="store_true")
    parser.add_argument("--min-context-count", type=int, default=2)
    parser.add_argument("--min-largest-context-tokens", type=int, default=8192)
    parser.add_argument("--bend-drop-ratio", type=float, default=0.75)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    required_families = list(args.required_family)
    if args.default_required_families:
        required_families.extend(DEFAULT_REQUIRED_FAMILIES)
    required_families = sorted({family.lower() for family in required_families})

    try:
        summaries = validate_campaign(
            args.artifacts,
            required_families=required_families,
            allow_mixed_host=args.allow_mixed_host,
            bend_drop_ratio=args.bend_drop_ratio,
            min_context_count=args.min_context_count,
            min_largest_context_tokens=args.min_largest_context_tokens,
        )
        report = render_campaign_report(summaries)
    except (PrefillScalingArtifactError, PrefillScalingCampaignError) as error:
        print(f"MLX prefill scaling campaign check failed: {error}", file=sys.stderr)
        return 1

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n")
        print(f"MLX prefill scaling campaign report written: {args.output}")
    else:
        print(report)
    return 0


def main_with_args_for_test(argv: list[str]) -> int:
    return main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
