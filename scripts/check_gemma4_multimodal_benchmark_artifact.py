#!/usr/bin/env python3
"""Validate Gemma 4 12B multimodal benchmark artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCHEMA = "ax.gemma4_multimodal_benchmark.v1"
ALLOWED_STATUS = {"measured", "skipped"}
ALLOWED_SKIP_REASONS = {
    "llama_cpp_video_not_supported",
    "missing_llama_cpp_mmproj",
    "no_llama_cpp_server_url",
    "server_unavailable",
    "unsupported_modality",
}
POSITIVE_METRICS = {
    "runner_prefill_ttft_ms",
    "client_wall_ttft_ms",
    "client_wall_total_ms",
    "client_wall_ms",
    "prefill_tok_s",
}


def metric_stats(row: dict[str, Any], key: str) -> dict[str, Any] | None:
    raw = row.get("summary", {}).get(key)
    return raw if isinstance(raw, dict) else None


def validate_positive_metric(
    errors: list[str],
    *,
    row_index: int,
    row: dict[str, Any],
    key: str,
) -> None:
    stats = metric_stats(row, key)
    if stats is None:
        return
    for stat_name in ("mean", "median", "min", "max"):
        value = stats.get(stat_name)
        if value is None:
            continue
        if not isinstance(value, (int, float)) or value <= 0:
            errors.append(f"rows[{row_index}].summary.{key}.{stat_name} must be positive")


def validate_prompt(errors: list[str], *, row_index: int, prompt: Any) -> None:
    if not isinstance(prompt, dict):
        errors.append(f"rows[{row_index}].prompt must be an object")
        return
    for key in ("original_tokens", "expanded_tokens"):
        value = prompt.get(key)
        if not isinstance(value, int) or value <= 0:
            errors.append(f"rows[{row_index}].prompt.{key} must be a positive integer")
    if (
        isinstance(prompt.get("original_tokens"), int)
        and isinstance(prompt.get("expanded_tokens"), int)
        and prompt["expanded_tokens"] < prompt["original_tokens"]
    ):
        errors.append(f"rows[{row_index}].prompt.expanded_tokens must be >= original_tokens")
    for key in ("image_soft_tokens", "audio_soft_tokens", "video_soft_tokens", "video_frame_counts"):
        value = prompt.get(key)
        if not isinstance(value, list):
            errors.append(f"rows[{row_index}].prompt.{key} must be a list")
            continue
        if any(not isinstance(item, int) or item < 0 for item in value):
            errors.append(f"rows[{row_index}].prompt.{key} entries must be non-negative integers")


def validate_row(errors: list[str], *, row_index: int, row: Any, min_repetitions: int) -> None:
    if not isinstance(row, dict):
        errors.append(f"rows[{row_index}] must be an object")
        return

    for key in ("engine", "backend", "layer", "case_id"):
        if not isinstance(row.get(key), str) or not row[key]:
            errors.append(f"rows[{row_index}].{key} must be a non-empty string")
    modalities = row.get("modalities")
    if not isinstance(modalities, list) or not modalities:
        errors.append(f"rows[{row_index}].modalities must be a non-empty list")
    elif any(modality not in {"image", "audio", "video"} for modality in modalities):
        errors.append(f"rows[{row_index}].modalities contains an unknown modality")

    status = row.get("status")
    if status not in ALLOWED_STATUS:
        errors.append(f"rows[{row_index}].status must be one of {sorted(ALLOWED_STATUS)}")
        return

    validate_prompt(errors, row_index=row_index, prompt=row.get("prompt"))
    runs = row.get("runs")
    if status == "measured":
        if not isinstance(runs, list) or len(runs) < min_repetitions:
            errors.append(
                f"rows[{row_index}].runs must contain at least {min_repetitions} measured run(s)"
            )
        if not isinstance(row.get("summary"), dict) or not row["summary"]:
            errors.append(f"rows[{row_index}].summary must be a non-empty object")
        for key in POSITIVE_METRICS:
            validate_positive_metric(errors, row_index=row_index, row=row, key=key)
        if row.get("layer") == "native_runtime_prefill" and metric_stats(
            row, "runner_prefill_ttft_ms"
        ) is None:
            errors.append(
                f"rows[{row_index}] native_runtime_prefill requires runner_prefill_ttft_ms"
            )
        if row.get("layer") == "openai_chat_e2e" and metric_stats(row, "client_wall_ms") is None:
            errors.append(f"rows[{row_index}] openai_chat_e2e requires client_wall_ms")
    else:
        reason = row.get("skip_reason")
        if reason not in ALLOWED_SKIP_REASONS:
            errors.append(
                f"rows[{row_index}].skip_reason must be one of {sorted(ALLOWED_SKIP_REASONS)}"
            )
        if not isinstance(row.get("skip_detail"), str) or not row["skip_detail"]:
            errors.append(f"rows[{row_index}].skip_detail must be a non-empty string")
        if runs not in ([], None):
            errors.append(f"rows[{row_index}].runs must be empty for skipped rows")


def validate_provenance(
    errors: list[str],
    artifact: dict[str, Any],
    *,
    require_build_provenance: bool,
    readme_ready: bool,
) -> None:
    provenance = artifact.get("provenance")
    if not isinstance(provenance, dict):
        errors.append("provenance must be an object")
        return
    git = provenance.get("git")
    if not isinstance(git, dict):
        errors.append("provenance.git must be an object")
        return
    if require_build_provenance and not git.get("commit"):
        errors.append("provenance.git.commit is required")
    if readme_ready and git.get("tracked_dirty"):
        errors.append("readme-ready artifacts must not have provenance.git.tracked_dirty=true")
    fingerprints = provenance.get("model_fingerprints")
    if require_build_provenance and not isinstance(fingerprints, dict):
        errors.append("provenance.model_fingerprints is required")


def validate_artifact(
    artifact: dict[str, Any],
    *,
    min_repetitions: int = 1,
    require_modalities: set[str] | None = None,
    require_build_provenance: bool = False,
    readme_ready: bool = False,
) -> list[str]:
    errors: list[str] = []
    if artifact.get("schema") != SCHEMA:
        errors.append(f"schema must be {SCHEMA}")
    benchmark = artifact.get("benchmark")
    if not isinstance(benchmark, dict):
        errors.append("benchmark must be an object")
    else:
        for key in ("name", "model"):
            if not isinstance(benchmark.get(key), str) or not benchmark[key]:
                errors.append(f"benchmark.{key} must be a non-empty string")
        for key in ("warmup", "repetitions", "max_output_tokens"):
            value = benchmark.get(key)
            if not isinstance(value, int) or value < 0:
                errors.append(f"benchmark.{key} must be a non-negative integer")
        if isinstance(benchmark.get("repetitions"), int) and benchmark["repetitions"] < min_repetitions:
            errors.append(f"benchmark.repetitions must be >= {min_repetitions}")

    rows = artifact.get("rows")
    if not isinstance(rows, list) or not rows:
        errors.append("rows must be a non-empty list")
        rows = []
    for row_index, row in enumerate(rows):
        validate_row(errors, row_index=row_index, row=row, min_repetitions=min_repetitions)

    validate_provenance(
        errors,
        artifact,
        require_build_provenance=require_build_provenance,
        readme_ready=readme_ready,
    )

    if require_modalities:
        measured = {
            modality
            for row in rows
            if isinstance(row, dict) and row.get("status") == "measured"
            for modality in row.get("modalities", [])
        }
        missing = sorted(require_modalities - measured)
        if missing:
            errors.append(f"missing measured modality coverage: {', '.join(missing)}")

    return errors


def load_artifact(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError("artifact root must be an object")
    return raw


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--min-repetitions", type=int, default=1)
    parser.add_argument(
        "--require-modalities",
        help="Comma-separated measured modality coverage to require, e.g. image,audio,video",
    )
    parser.add_argument("--require-build-provenance", action="store_true")
    parser.add_argument("--readme-ready", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    require_modalities = None
    if args.require_modalities:
        require_modalities = {
            item.strip() for item in args.require_modalities.split(",") if item.strip()
        }
    errors = validate_artifact(
        load_artifact(args.artifact),
        min_repetitions=args.min_repetitions,
        require_modalities=require_modalities,
        require_build_provenance=args.require_build_provenance,
        readme_ready=args.readme_ready,
    )
    if errors:
        for error in errors:
            print(f"error: {error}")
        return 1
    print(f"ok: {args.artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
