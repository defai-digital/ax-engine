#!/usr/bin/env python3
"""Fail-closed preflight for Qwen/GatedDelta prefill profile runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PREFLIGHT_SCHEMA_VERSION = "ax.gateddelta_prefill_model_preflight.v1"
SUPPORTED_MODEL_FAMILIES = {"qwen3_5", "qwen3_next"}
REQUIRED_LINEAR_ATTENTION_FIELDS = (
    "num_value_heads",
    "num_key_heads",
    "key_head_dim",
    "value_head_dim",
    "conv_kernel_dim",
)


class GatedDeltaPrefillModelError(RuntimeError):
    """Raised when a model directory cannot produce GatedDelta profile evidence."""


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.exists():
        raise GatedDeltaPrefillModelError(f"{label} is missing: {path}")
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise GatedDeltaPrefillModelError(
            f"{label} is not valid JSON: {path}: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise GatedDeltaPrefillModelError(f"{label} must be a JSON object: {path}")
    return data


def _require_positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise GatedDeltaPrefillModelError(
            f"linear_attention.{field} must be a positive integer"
        )
    return value


def validate_gateddelta_prefill_model(model_dir: Path) -> dict[str, Any]:
    """Return normalized metadata when model_dir is usable for profile runs."""
    if not model_dir.is_dir():
        raise GatedDeltaPrefillModelError(f"model directory does not exist: {model_dir}")

    config = _read_json_object(model_dir / "config.json", "config.json")
    manifest = _read_json_object(
        model_dir / "model-manifest.json", "model-manifest.json"
    )

    model_family = manifest.get("model_family")
    if model_family not in SUPPORTED_MODEL_FAMILIES:
        supported = "/".join(sorted(SUPPORTED_MODEL_FAMILIES))
        raise GatedDeltaPrefillModelError(
            "model-manifest.json model_family must be "
            f"{supported} for GatedDelta prefill profiling; got {model_family!r}"
        )

    linear_attention = manifest.get("linear_attention")
    if not isinstance(linear_attention, dict) or not any(
        value is not None for value in linear_attention.values()
    ):
        raise GatedDeltaPrefillModelError(
            "model-manifest.json must include enabled linear_attention metadata"
        )

    missing_fields = [
        field
        for field in REQUIRED_LINEAR_ATTENTION_FIELDS
        if linear_attention.get(field) is None
    ]
    if missing_fields:
        raise GatedDeltaPrefillModelError(
            "linear_attention is missing required fields: "
            + ", ".join(missing_fields)
        )

    normalized_linear_attention = {
        field: _require_positive_int(linear_attention[field], field)
        for field in REQUIRED_LINEAR_ATTENTION_FIELDS
    }
    full_attention_interval = linear_attention.get("full_attention_interval")
    if full_attention_interval is not None:
        normalized_linear_attention["full_attention_interval"] = _require_positive_int(
            full_attention_interval, "full_attention_interval"
        )

    key_head_dim = normalized_linear_attention["key_head_dim"]
    if key_head_dim % 32 != 0:
        raise GatedDeltaPrefillModelError(
            "linear_attention.key_head_dim must be divisible by 32 for the "
            f"MLX gated-delta kernel; got {key_head_dim}"
        )

    return {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "model_dir": str(model_dir),
        "model_type": config.get("model_type"),
        "model_family": model_family,
        "linear_attention": normalized_linear_attention,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate that an MLX model directory is suitable for GatedDelta prefill profiling."
    )
    parser.add_argument("model_dir", type=Path, help="Local MLX model artifact directory")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print normalized metadata as JSON instead of a short ok line.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        metadata = validate_gateddelta_prefill_model(args.model_dir)
    except GatedDeltaPrefillModelError as exc:
        print(f"ERROR: {exc}")
        return 1

    if args.json:
        print(json.dumps(metadata, indent=2, sort_keys=True))
    else:
        print(
            "ok: GatedDelta prefill profile model "
            f"family={metadata['model_family']} model_type={metadata.get('model_type')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
