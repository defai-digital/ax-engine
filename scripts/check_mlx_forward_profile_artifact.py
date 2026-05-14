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
    generation_tokens: int
    verdict: str


@dataclass(frozen=True)
class ForwardProfileCheckResult:
    artifact_count: int
    diagnostic_count: int
    pack_candidate_win_count: int
    pack_candidate_win_prompt_count: int
    pack_candidate_win_shape_count: int
    pack_comparisons: list[CheckedPackComparison]


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except OSError as error:
        raise MlxForwardProfileArtifactError(f"failed to read {path}: {error}") from error
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
    require_pack_candidate_win: bool = False,
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
    if require_pack_comparison or require_pack_candidate_win:
        validate_raw_pack_pair_contract(path, artifact)
        if not comparisons:
            raise MlxForwardProfileArtifactError(f"{path} has no pack comparison verdicts")
    if require_pack_candidate_win:
        non_wins = [
            comparison
            for comparison in comparisons
            if comparison.verdict != "candidate win"
        ]
        if non_wins:
            summary = "; ".join(
                (
                    f"{comparison.model} prompt={comparison.prompt_tokens} "
                    f"gen={comparison.generation_tokens}: {comparison.verdict}"
                )
                for comparison in non_wins
            )
            raise MlxForwardProfileArtifactError(
                f"{path} pack comparison is not a candidate win: {summary}"
            )

    return [
        CheckedPackComparison(
            model=comparison.model,
            prompt_tokens=comparison.prompt_tokens,
            generation_tokens=comparison.generation_tokens,
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
    parser.add_argument(
        "--require-pack-candidate-win",
        action="store_true",
        help=(
            "Require every matched split/packed comparison to have a candidate win "
            "verdict. Implies --require-pack-comparison."
        ),
    )
    parser.add_argument(
        "--min-pack-candidate-wins",
        type=int,
        default=None,
        help=(
            "Require at least this many candidate-win split/packed comparisons "
            "across the provided artifacts. Implies --require-pack-comparison."
        ),
    )
    parser.add_argument(
        "--min-pack-candidate-win-prompts",
        type=int,
        default=None,
        help=(
            "Require candidate wins across at least this many distinct prompt-token "
            "lengths. Implies --require-pack-comparison."
        ),
    )
    parser.add_argument(
        "--min-pack-candidate-win-shapes",
        type=int,
        default=None,
        help=(
            "Require candidate wins across at least this many distinct "
            "(prompt_tokens, generation_tokens) shapes. Implies "
            "--require-pack-comparison."
        ),
    )
    return parser.parse_args(argv)


def summarize_pack_comparisons(comparisons: list[CheckedPackComparison]) -> str:
    if not comparisons:
        return "0 pack comparisons"
    return "; ".join(
        (
            f"{comparison.model} prompt={comparison.prompt_tokens} "
            f"gen={comparison.generation_tokens}: {comparison.verdict}"
        )
        for comparison in comparisons
    )


def check_mlx_forward_profile_artifacts(
    artifacts: list[Path],
    *,
    require_pack_comparison: bool = False,
    require_pack_candidate_win: bool = False,
    min_pack_candidate_wins: int | None = None,
    min_pack_candidate_win_prompts: int | None = None,
    min_pack_candidate_win_shapes: int | None = None,
) -> ForwardProfileCheckResult:
    if min_pack_candidate_wins is not None and min_pack_candidate_wins < 0:
        raise MlxForwardProfileArtifactError(
            "--min-pack-candidate-wins must be non-negative"
        )
    if (
        min_pack_candidate_win_prompts is not None
        and min_pack_candidate_win_prompts < 0
    ):
        raise MlxForwardProfileArtifactError(
            "--min-pack-candidate-win-prompts must be non-negative"
        )
    if (
        min_pack_candidate_win_shapes is not None
        and min_pack_candidate_win_shapes < 0
    ):
        raise MlxForwardProfileArtifactError(
            "--min-pack-candidate-win-shapes must be non-negative"
        )
    require_comparison = (
        require_pack_comparison
        or require_pack_candidate_win
        or (min_pack_candidate_wins is not None and min_pack_candidate_wins > 0)
        or (
            min_pack_candidate_win_prompts is not None
            and min_pack_candidate_win_prompts > 0
        )
        or (
            min_pack_candidate_win_shapes is not None
            and min_pack_candidate_win_shapes > 0
        )
    )
    diagnostic_count = 0
    pack_comparisons: list[CheckedPackComparison] = []
    for artifact in artifacts:
        comparisons = validate_mlx_forward_profile_artifact(
            artifact,
            require_pack_comparison=require_comparison,
            require_pack_candidate_win=require_pack_candidate_win,
        )
        diagnostic_count += max(1, len(comparisons))
        pack_comparisons.extend(comparisons)
    pack_candidate_win_count = sum(
        1 for comparison in pack_comparisons if comparison.verdict == "candidate win"
    )
    pack_candidate_win_prompt_count = len(
        {
            comparison.prompt_tokens
            for comparison in pack_comparisons
            if comparison.verdict == "candidate win"
        }
    )
    pack_candidate_win_shape_count = len(
        {
            (comparison.prompt_tokens, comparison.generation_tokens)
            for comparison in pack_comparisons
            if comparison.verdict == "candidate win"
        }
    )
    if (
        min_pack_candidate_wins is not None
        and pack_candidate_win_count < min_pack_candidate_wins
    ):
        raise MlxForwardProfileArtifactError(
            "pack comparison has "
            f"{pack_candidate_win_count} candidate win(s), expected at least "
            f"{min_pack_candidate_wins}"
        )
    if (
        min_pack_candidate_win_prompts is not None
        and pack_candidate_win_prompt_count < min_pack_candidate_win_prompts
    ):
        raise MlxForwardProfileArtifactError(
            "pack comparison has candidate wins across "
            f"{pack_candidate_win_prompt_count} prompt length(s), expected at least "
            f"{min_pack_candidate_win_prompts}"
        )
    if (
        min_pack_candidate_win_shapes is not None
        and pack_candidate_win_shape_count < min_pack_candidate_win_shapes
    ):
        raise MlxForwardProfileArtifactError(
            "pack comparison has candidate wins across "
            f"{pack_candidate_win_shape_count} shape(s), expected at least "
            f"{min_pack_candidate_win_shapes}"
        )
    return ForwardProfileCheckResult(
        artifact_count=len(artifacts),
        diagnostic_count=diagnostic_count,
        pack_candidate_win_count=pack_candidate_win_count,
        pack_candidate_win_prompt_count=pack_candidate_win_prompt_count,
        pack_candidate_win_shape_count=pack_candidate_win_shape_count,
        pack_comparisons=pack_comparisons,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        checked = check_mlx_forward_profile_artifacts(
            args.artifacts,
            require_pack_comparison=args.require_pack_comparison,
            require_pack_candidate_win=args.require_pack_candidate_win,
            min_pack_candidate_wins=args.min_pack_candidate_wins,
            min_pack_candidate_win_prompts=args.min_pack_candidate_win_prompts,
            min_pack_candidate_win_shapes=args.min_pack_candidate_win_shapes,
        )
    except MlxForwardProfileArtifactError as error:
        print(f"MLX forward profile artifact check failed: {error}", file=sys.stderr)
        return 1
    print(
        "MLX forward profile artifact check passed: "
        f"{checked.diagnostic_count} diagnostics validated across "
        f"{checked.artifact_count} artifact(s); "
        f"{checked.pack_candidate_win_count} candidate win(s); "
        f"{checked.pack_candidate_win_prompt_count} candidate-win prompt length(s); "
        f"{checked.pack_candidate_win_shape_count} candidate-win shape(s); "
        f"{summarize_pack_comparisons(checked.pack_comparisons)}"
    )
    return 0


def main_with_args_for_test(argv: list[str]) -> int:
    return main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
