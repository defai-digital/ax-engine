#!/usr/bin/env python3
"""Validate Gemma 4 12B multimodal benchmark artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCHEMA = "ax.gemma4_multimodal_benchmark.v1"
ALLOWED_STATUS = {"measured", "skipped"}
ALLOWED_LAYERS = {"native_runtime_prefill", "openai_chat_e2e", "peer_comparison"}
ALLOWED_MODALITIES = {"image", "audio", "video"}
MODALITY_ORDER = ("image", "audio", "video")
ALLOWED_SKIP_REASONS = {
    "llama_cpp_video_not_supported",
    "missing_llama_cpp_mmproj_for_gemma4_12b",
    "missing_llama_cpp_gguf_for_gemma4_12b",
    "no_llama_cpp_server_url",
    "peer_binary_missing",
    "peer_prompt_contract_unmatched",
    "server_unavailable",
    "unsupported_modality",
}
POSITIVE_METRICS = {
    "runner_prefill_ttft_ms",
    "client_wall_ttft_ms",
    "client_wall_total_ms",
    "client_wall_ms",
    "non_streaming_total_ms",
    "prefill_tok_s",
    "payload_bytes",
}


def metric_stats(row: dict[str, Any], key: str) -> dict[str, Any] | None:
    raw = row.get("summary", {}).get(key)
    return raw if isinstance(raw, dict) else None


def require_object(errors: list[str], value: Any, path: str) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        errors.append(f"{path} must be an object")
        return None
    return value


def validate_top_level(
    errors: list[str],
    artifact: dict[str, Any],
    *,
    require_build_provenance: bool,
    readme_ready: bool,
) -> None:
    if artifact.get("schema") != SCHEMA:
        errors.append(f"schema must be {SCHEMA}")
    if not isinstance(artifact.get("created_at"), str) or not artifact["created_at"]:
        errors.append("created_at must be a non-empty string")

    host = require_object(errors, artifact.get("host"), "host")
    if host is not None:
        for key in ("platform", "machine", "python", "os_version"):
            if not isinstance(host.get(key), str) or not host[key]:
                errors.append(f"host.{key} must be a non-empty string")

    build = require_object(errors, artifact.get("build"), "build")
    if build is not None:
        if require_build_provenance and not isinstance(build.get("commit"), str):
            errors.append("build.commit is required")
        if not isinstance(build.get("git_tracked_dirty"), bool):
            errors.append("build.git_tracked_dirty must be a boolean")
        if not isinstance(build.get("git_tracked_status"), list):
            errors.append("build.git_tracked_status must be a list")
        if readme_ready and build.get("git_tracked_dirty"):
            errors.append("readme-ready artifacts must have build.git_tracked_dirty=false")

    server = require_object(errors, artifact.get("server"), "server")
    if server is not None:
        if not isinstance(server.get("url"), str) or not server["url"]:
            errors.append("server.url must be a non-empty string")
        if not isinstance(server.get("endpoint_layers"), list) or not server["endpoint_layers"]:
            errors.append("server.endpoint_layers must be a non-empty list")
        if not isinstance(server.get("request_timeout_s"), int) or server["request_timeout_s"] <= 0:
            errors.append("server.request_timeout_s must be a positive integer")
        if readme_ready and server.get("command") is None:
            errors.append("readme-ready artifacts must record server.command")

    model = require_object(errors, artifact.get("model"), "model")
    if model is not None:
        for key in ("id", "model_dir", "model_type"):
            if not isinstance(model.get(key), str) or not model[key]:
                errors.append(f"model.{key} must be a non-empty string")
        if require_build_provenance:
            for key in (
                "model_manifest_sha256",
                "config_sha256",
                "processor_config_sha256",
                "tokenizer_sha256",
            ):
                if not isinstance(model.get(key), str) or not model[key]:
                    errors.append(f"model.{key} is required")


def validate_fixtures(errors: list[str], fixtures: Any) -> dict[str, str]:
    if not isinstance(fixtures, list) or not fixtures:
        errors.append("fixtures must be a non-empty list")
        return {}
    ids: dict[str, str] = {}
    for index, fixture in enumerate(fixtures):
        if not isinstance(fixture, dict):
            errors.append(f"fixtures[{index}] must be an object")
            continue
        fixture_id = fixture.get("id")
        if not isinstance(fixture_id, str) or not fixture_id:
            errors.append(f"fixtures[{index}].id must be a non-empty string")
        elif fixture_id in ids:
            errors.append(f"fixtures[{index}].id duplicates {fixture_id}")
        else:
            ids[fixture_id] = str(fixture.get("modality") or "")
        if fixture.get("modality") not in ALLOWED_MODALITIES:
            errors.append(f"fixtures[{index}].modality must be one of {sorted(ALLOWED_MODALITIES)}")
        for key in ("source", "sha256", "mime"):
            if not isinstance(fixture.get(key), str) or not fixture[key]:
                errors.append(f"fixtures[{index}].{key} must be a non-empty string")
        if not isinstance(fixture.get("raw"), dict) or not fixture["raw"]:
            errors.append(f"fixtures[{index}].raw must be a non-empty object")
    return ids


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


def expected_modality_set(modalities: list[Any]) -> list[str]:
    present = {modality for modality in modalities if modality in ALLOWED_MODALITIES}
    return [modality for modality in MODALITY_ORDER if modality in present]


def validate_prompt(
    errors: list[str],
    *,
    row_index: int,
    prompt: Any,
    fixture_modalities: dict[str, str],
    expected_row_fixture_ids: list[Any] | None,
    row_modalities: list[Any] | None,
) -> None:
    if not isinstance(prompt, dict):
        errors.append(f"rows[{row_index}].prompt must be an object")
        return
    for key in ("original_tokens", "expanded_tokens"):
        value = prompt.get(key)
        if not isinstance(value, int) or value <= 0:
            errors.append(f"rows[{row_index}].prompt.{key} must be a positive integer")
    soft_tokens = prompt.get("soft_tokens")
    if not isinstance(soft_tokens, dict):
        errors.append(f"rows[{row_index}].prompt.soft_tokens must be an object")
        soft_total = 0
    else:
        soft_total = 0
        for modality in ALLOWED_MODALITIES:
            value = soft_tokens.get(modality)
            if not isinstance(value, int) or value < 0:
                errors.append(
                    f"rows[{row_index}].prompt.soft_tokens.{modality} must be a non-negative integer"
                )
            else:
                soft_total += value
    if (
        isinstance(prompt.get("original_tokens"), int)
        and isinstance(prompt.get("expanded_tokens"), int)
        and prompt["expanded_tokens"] < prompt["original_tokens"] + soft_total
    ):
        errors.append(
            f"rows[{row_index}].prompt.expanded_tokens must be >= original_tokens + soft tokens"
        )
    span_order = prompt.get("span_order")
    if not isinstance(span_order, list):
        errors.append(f"rows[{row_index}].prompt.span_order must be a list")
    elif any(item not in ALLOWED_MODALITIES for item in span_order):
        errors.append(f"rows[{row_index}].prompt.span_order contains an unknown modality")
    elif isinstance(row_modalities, list) and set(span_order) != set(row_modalities):
        errors.append(f"rows[{row_index}].prompt.span_order modalities must match row modalities")
    prompt_fixture_ids = prompt.get("fixture_ids")
    if not isinstance(prompt_fixture_ids, list) or not prompt_fixture_ids:
        errors.append(f"rows[{row_index}].prompt.fixture_ids must be a non-empty list")
    else:
        missing = [fixture_id for fixture_id in prompt_fixture_ids if fixture_id not in fixture_modalities]
        if missing:
            errors.append(f"rows[{row_index}].prompt.fixture_ids missing fixtures: {missing}")
    if isinstance(prompt_fixture_ids, list) and prompt_fixture_ids != expected_row_fixture_ids:
        errors.append(f"rows[{row_index}].prompt.fixture_ids must match row fixture_ids")
    if isinstance(prompt_fixture_ids, list) and isinstance(span_order, list):
        expected_span_order = [
            fixture_modalities[fixture_id]
            for fixture_id in prompt_fixture_ids
            if fixture_id in fixture_modalities
        ]
        if len(expected_span_order) == len(prompt_fixture_ids) and span_order != expected_span_order:
            errors.append(f"rows[{row_index}].prompt.span_order must match fixture order")
    for key in ("image_soft_tokens", "audio_soft_tokens", "video_soft_tokens", "video_frame_counts"):
        value = prompt.get(key)
        if not isinstance(value, list):
            errors.append(f"rows[{row_index}].prompt.{key} must be a list")
            continue
        if any(not isinstance(item, int) or item < 0 for item in value):
            errors.append(f"rows[{row_index}].prompt.{key} entries must be non-negative integers")


def validate_peer_capability(errors: list[str], *, row_index: int, row: dict[str, Any]) -> None:
    capability = row.get("capability")
    if not isinstance(capability, dict):
        errors.append(f"rows[{row_index}].capability is required for peer_comparison")
        return
    if row.get("status") == "measured" and capability.get("proof") is not True:
        errors.append(f"rows[{row_index}] measured peer row requires capability.proof=true")
    if row.get("status") == "measured":
        for key in ("text_gguf_sha256", "mmproj_sha256", "prompt_contract"):
            if not isinstance(capability.get(key), str) or not capability[key]:
                errors.append(f"rows[{row_index}].capability.{key} is required")
    if "video" in row.get("modalities", []) and row.get("status") == "measured":
        if capability.get("supports_video") is not True:
            errors.append(f"rows[{row_index}] cannot measure video peer row without supports_video")


def validate_row(
    errors: list[str],
    *,
    row_index: int,
    row: Any,
    min_repetitions: int,
    fixture_modalities: dict[str, str],
) -> None:
    if not isinstance(row, dict):
        errors.append(f"rows[{row_index}] must be an object")
        return

    for key in ("row_id", "engine", "backend", "layer", "case_id"):
        if not isinstance(row.get(key), str) or not row[key]:
            errors.append(f"rows[{row_index}].{key} must be a non-empty string")
    if row.get("layer") not in ALLOWED_LAYERS:
        errors.append(f"rows[{row_index}].layer must be one of {sorted(ALLOWED_LAYERS)}")
    modalities = row.get("modalities")
    if not isinstance(modalities, list) or not modalities:
        errors.append(f"rows[{row_index}].modalities must be a non-empty list")
    elif any(modality not in ALLOWED_MODALITIES for modality in modalities):
        errors.append(f"rows[{row_index}].modalities contains an unknown modality")
    modality_set = row.get("modality_set")
    if not isinstance(modality_set, list) or not modality_set:
        errors.append(f"rows[{row_index}].modality_set must be a non-empty list")
    elif any(modality not in ALLOWED_MODALITIES for modality in modality_set):
        errors.append(f"rows[{row_index}].modality_set contains an unknown modality")
    elif isinstance(modalities, list) and modality_set != expected_modality_set(modalities):
        errors.append(f"rows[{row_index}].modality_set must match row modalities")
    row_fixture_ids = row.get("fixture_ids")
    if not isinstance(row_fixture_ids, list) or not row_fixture_ids:
        errors.append(f"rows[{row_index}].fixture_ids must be a non-empty list")
    else:
        missing = [fixture_id for fixture_id in row_fixture_ids if fixture_id not in fixture_modalities]
        if missing:
            errors.append(f"rows[{row_index}].fixture_ids missing fixtures: {missing}")
        fixture_modality_set = {
            fixture_modalities[fixture_id]
            for fixture_id in row_fixture_ids
            if fixture_id in fixture_modalities
        }
        if isinstance(modalities, list):
            row_modality_set = set(modalities)
            missing_modalities = sorted(row_modality_set - fixture_modality_set)
            extra_modalities = sorted(fixture_modality_set - row_modality_set)
            if missing_modalities:
                errors.append(
                    f"rows[{row_index}].fixture_ids missing modality fixtures: {missing_modalities}"
                )
            if extra_modalities:
                errors.append(
                    f"rows[{row_index}].fixture_ids include unrelated modalities: {extra_modalities}"
                )

    status = row.get("status")
    if status not in ALLOWED_STATUS:
        errors.append(f"rows[{row_index}].status must be one of {sorted(ALLOWED_STATUS)}")
        return

    validate_prompt(
        errors,
        row_index=row_index,
        prompt=row.get("prompt"),
        fixture_modalities=fixture_modalities,
        expected_row_fixture_ids=row_fixture_ids if isinstance(row_fixture_ids, list) else None,
        row_modalities=modalities if isinstance(modalities, list) else None,
    )
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
        if row.get("layer") in {"openai_chat_e2e", "peer_comparison"} and metric_stats(
            row, "client_wall_ms"
        ) is None:
            errors.append(f"rows[{row_index}] {row.get('layer')} requires client_wall_ms")
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
    if row.get("layer") == "peer_comparison":
        validate_peer_capability(errors, row_index=row_index, row=row)


def validate_artifact(
    artifact: dict[str, Any],
    *,
    min_repetitions: int = 1,
    require_modalities: set[str] | None = None,
    require_build_provenance: bool = False,
    readme_ready: bool = False,
) -> list[str]:
    errors: list[str] = []
    validate_top_level(
        errors,
        artifact,
        require_build_provenance=require_build_provenance,
        readme_ready=readme_ready,
    )
    fixture_modalities = validate_fixtures(errors, artifact.get("fixtures"))

    benchmark = require_object(errors, artifact.get("benchmark"), "benchmark")
    if benchmark is not None:
        for key in ("name", "model", "model_dir"):
            if not isinstance(benchmark.get(key), str) or not benchmark[key]:
                errors.append(f"benchmark.{key} must be a non-empty string")
        for key in ("warmup", "repetitions", "max_output_tokens", "timeout_s"):
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
        validate_row(
            errors,
            row_index=row_index,
            row=row,
            min_repetitions=min_repetitions,
            fixture_modalities=fixture_modalities,
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
