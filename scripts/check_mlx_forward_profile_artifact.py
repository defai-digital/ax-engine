#!/usr/bin/env python3
"""Validate diagnostic AX MLX forward-profile and projection-pack artifacts."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).parent
RENDERER_PATH = SCRIPT_DIR / "render_mlx_forward_profile_report.py"
MODULE_SPEC = importlib.util.spec_from_file_location(
    "render_mlx_forward_profile_report", RENDERER_PATH
)
assert MODULE_SPEC and MODULE_SPEC.loader
renderer = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = renderer
MODULE_SPEC.loader.exec_module(renderer)

SCHEMA_VERSION = "ax.mlx_inference_stack.v2"
PACKED_PUBLIC_CLAIM_NAMES = {
    "linear_attention_projection_pack",
    "packed_projection_performance",
    "packed_projection_prefill_win",
    "packed_linear_attention_projection_win",
}


class MlxForwardProfileArtifactError(RuntimeError):
    pass


@dataclass(frozen=True)
class CheckedPackComparison:
    model: str
    prompt_tokens: int
    verdict: str


@dataclass(frozen=True)
class ForwardProfileCheckResult:
    artifact_count: int
    diagnostic_count: int
    pack_comparisons: list[CheckedPackComparison]


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise MlxForwardProfileArtifactError(f"{path} is not valid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise MlxForwardProfileArtifactError(f"{path} must contain a JSON object")
    return payload


def public_claim_names(artifact: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for key in ("public_claims", "public_performance_claims"):
        claims = artifact.get(key) or []
        if not isinstance(claims, list):
            raise MlxForwardProfileArtifactError(f"{key} must be a list when present")
        for claim in claims:
            if isinstance(claim, str):
                names.add(claim)
            elif isinstance(claim, dict) and isinstance(claim.get("name"), str):
                names.add(str(claim["name"]))
            else:
                raise MlxForwardProfileArtifactError(f"invalid {key} entry: {claim!r}")
    return names


def reject_public_packed_claims(path: Path, artifact: dict[str, Any]) -> None:
    forbidden = sorted(PACKED_PUBLIC_CLAIM_NAMES & public_claim_names(artifact))
    if forbidden:
        raise MlxForwardProfileArtifactError(
            f"{path} makes packed projection public claims from diagnostic evidence: "
            + ", ".join(forbidden)
        )


def row_shape(row: dict[str, Any]) -> tuple[int, int] | None:
    prompt_tokens = row.get("prompt_tokens")
    generation_tokens = row.get("generation_tokens")
    if isinstance(prompt_tokens, int) and isinstance(generation_tokens, int):
        return (prompt_tokens, generation_tokens)
    return None


def validate_direct_ax_row(path: Path, row: dict[str, Any], *, label: str) -> None:
    if row.get("ax_decode_policy") != "direct_no_ngram_acceleration":
        raise MlxForwardProfileArtifactError(
            f"{path} {label} must use direct_no_ngram_acceleration"
        )
    if not isinstance(row.get("ax_mlx_linear_attention_profile"), dict):
        raise MlxForwardProfileArtifactError(f"{path} {label} lacks linear profile")


def validate_raw_pack_pair_contract(path: Path, artifact: dict[str, Any]) -> None:
    if artifact.get("ax_linear_attention_projection_pack_compare") is not True:
        raise MlxForwardProfileArtifactError(
            f"{path} pack comparison artifacts must set ax_linear_attention_projection_pack_compare=true"
        )
    rows = [row for row in artifact.get("results", []) if isinstance(row, dict)]
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in rows:
        shape = row_shape(row)
        if shape is not None:
            grouped.setdefault(shape, []).append(row)

    found_pair = False
    for shape, group in grouped.items():
        split = next(
            (
                row
                for row in group
                if row.get("engine") == "ax_engine_mlx"
                and row.get("ax_linear_attention_projection_pack") is not True
            ),
            None,
        )
        packed = next(
            (
                row
                for row in group
                if row.get("engine") == "ax_engine_mlx_linear_pack"
                or row.get("ax_linear_attention_projection_pack") is True
            ),
            None,
        )
        if split is None and packed is None:
            continue
        if split is None or packed is None:
            raise MlxForwardProfileArtifactError(
                f"{path} shape={shape} must include matched split and packed AX rows"
            )
        validate_direct_ax_row(path, split, label=f"shape={shape} split row")
        validate_direct_ax_row(path, packed, label=f"shape={shape} packed row")
        if packed.get("ax_linear_attention_projection_pack") is not True:
            raise MlxForwardProfileArtifactError(
                f"{path} shape={shape} packed row lacks projection pack marker"
            )
        found_pair = True

    if not found_pair:
        raise MlxForwardProfileArtifactError(f"{path} has no matched pack comparison rows")


def validate_mlx_forward_profile_artifact(
    path: Path,
    *,
    require_pack_comparison: bool = False,
) -> list[CheckedPackComparison]:
    artifact = load_json(path)
    if artifact.get("schema_version") != SCHEMA_VERSION:
        raise MlxForwardProfileArtifactError(
            f"{path} has schema_version={artifact.get('schema_version')!r}, expected {SCHEMA_VERSION}"
        )
    reject_public_packed_claims(path, artifact)

    try:
        rows = renderer.build_rows(path)
    except renderer.MlxForwardProfileReportError as error:
        raise MlxForwardProfileArtifactError(str(error)) from error

    comparisons = renderer.build_pack_comparisons(rows)
    if require_pack_comparison:
        validate_raw_pack_pair_contract(path, artifact)
        if not comparisons:
            raise MlxForwardProfileArtifactError(f"{path} has no pack comparison verdicts")

    return [
        CheckedPackComparison(
            model=comparison.model,
            prompt_tokens=comparison.prompt_tokens,
            verdict=comparison.verdict,
        )
        for comparison in comparisons
    ]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument(
        "--require-pack-comparison",
        action="store_true",
        help="Require matched split and packed direct AX rows.",
    )
    return parser.parse_args(argv)


def summarize_pack_comparisons(comparisons: list[CheckedPackComparison]) -> str:
    if not comparisons:
        return "0 pack comparisons"
    return "; ".join(
        f"{comparison.model} prompt={comparison.prompt_tokens}: {comparison.verdict}"
        for comparison in comparisons
    )


def check_mlx_forward_profile_artifacts(
    artifacts: list[Path],
    *,
    require_pack_comparison: bool = False,
) -> ForwardProfileCheckResult:
    diagnostic_count = 0
    pack_comparisons: list[CheckedPackComparison] = []
    for artifact in artifacts:
        comparisons = validate_mlx_forward_profile_artifact(
            artifact,
            require_pack_comparison=require_pack_comparison,
        )
        diagnostic_count += max(1, len(comparisons))
        pack_comparisons.extend(comparisons)
    return ForwardProfileCheckResult(
        artifact_count=len(artifacts),
        diagnostic_count=diagnostic_count,
        pack_comparisons=pack_comparisons,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        checked = check_mlx_forward_profile_artifacts(
            args.artifacts,
            require_pack_comparison=args.require_pack_comparison,
        )
    except MlxForwardProfileArtifactError as error:
        print(f"MLX forward profile artifact check failed: {error}", file=sys.stderr)
        return 1
    print(
        "MLX forward profile artifact check passed: "
        f"{checked.diagnostic_count} diagnostics validated across "
        f"{checked.artifact_count} artifact(s); "
        f"{summarize_pack_comparisons(checked.pack_comparisons)}"
    )
    return 0


def main_with_args_for_test(argv: list[str]) -> int:
    return main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
