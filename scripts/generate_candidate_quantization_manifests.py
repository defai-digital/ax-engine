#!/usr/bin/env python3
"""Generate candidate quantization manifests for lower-bit recipes.

Takes a base model directory and produces candidate quantization manifests
for recipes that could plausibly improve bytes/token. Each candidate manifest
describes the target quantization recipe without performing actual weight
quantization.

Output schema: ax.candidate_quantization_manifests.v1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "ax.candidate_quantization_manifests.v1"

CANDIDATE_RECIPES = [
    {
        "recipe_id": "uniform_3bit_g64",
        "description": "Uniform 3-bit quantization with group size 64",
        "bits": 3,
        "group_size": 64,
        "layout": "uniform",
        "per_layer_overrides": {},
    },
    {
        "recipe_id": "uniform_3bit_g32",
        "description": "Uniform 3-bit quantization with group size 32",
        "bits": 3,
        "group_size": 32,
        "layout": "uniform",
        "per_layer_overrides": {},
    },
    {
        "recipe_id": "mixed_4bit_attn_3bit_ffn_g64",
        "description": "4-bit attention, 3-bit FFN with group size 64",
        "bits": 4,
        "group_size": 64,
        "layout": "mixed",
        "per_layer_overrides": {},
        "ffn_bits": 3,
    },
    {
        "recipe_id": "mixed_4bit_attn_3bit_ffn_g32",
        "description": "4-bit attention, 3-bit FFN with group size 32",
        "bits": 4,
        "group_size": 32,
        "layout": "mixed",
        "per_layer_overrides": {},
        "ffn_bits": 3,
    },
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "model_dir",
        type=Path,
        help="Base model artifact directory.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write JSON artifact to this path instead of stdout.",
    )
    p.add_argument(
        "--recipes",
        nargs="*",
        default=None,
        help="Recipe IDs to generate (default: all).",
    )
    return p.parse_args()


def read_config(model_dir: Path) -> dict[str, Any]:
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        return {}
    try:
        return json.loads(config_path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def read_manifest(model_dir: Path) -> dict[str, Any]:
    manifest_path = model_dir / "model-manifest.json"
    if not manifest_path.is_file():
        return {}
    try:
        return json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def estimate_candidate_bytes(
    manifest: dict[str, Any],
    recipe: dict[str, Any],
) -> dict[str, Any]:
    tensors = manifest.get("tensors", [])
    if not tensors:
        return {"estimated_bytes": None, "reason": "no_tensors_in_manifest"}

    target_bits = recipe["bits"]
    group_size = recipe["group_size"]
    ffn_bits = recipe.get("ffn_bits")

    total_bits = 0
    for t in tensors:
        role = t.get("role", "")
        length_bytes = t.get("length_bytes", 0)
        if length_bytes == 0:
            continue

        bits = target_bits
        if ffn_bits is not None and any(
            tag in role for tag in ("ffn_gate", "ffn_up", "ffn_down", ".mlp.")
        ):
            bits = ffn_bits

        original_bits = length_bytes * 8
        quantized_elements = length_bytes * 8 // 16
        quantized_bytes = (quantized_elements * bits + 7) // 8
        scales_bytes = (quantized_elements // group_size) * 2
        total_bits += (quantized_bytes + scales_bytes) * 8

    estimated_bytes = total_bits // 8
    return {
        "estimated_bytes": estimated_bytes,
        "target_bits": target_bits,
        "group_size": group_size,
        "ffn_bits": ffn_bits,
    }


def build_candidate_manifest(
    model_dir: Path,
    recipe: dict[str, Any],
    config: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    model_type = config.get("model_type", "unknown")
    model_family = manifest.get("model_family", model_type)

    quantization = {
        "bits": recipe["bits"],
        "group_size": recipe["group_size"],
        "layout": recipe["layout"],
    }
    if recipe.get("per_layer_overrides"):
        quantization.update(recipe["per_layer_overrides"])
    if recipe.get("ffn_bits"):
        quantization["ffn_bits"] = recipe["ffn_bits"]

    byte_estimate = estimate_candidate_bytes(manifest, recipe)

    return {
        "recipe_id": recipe["recipe_id"],
        "description": recipe["description"],
        "base_model_dir": str(model_dir),
        "model_type": model_type,
        "model_family": model_family,
        "quantization": quantization,
        "byte_estimate": byte_estimate,
    }


def build_candidates(
    model_dir: Path,
    recipe_ids: list[str] | None = None,
) -> dict[str, Any]:
    if not model_dir.is_dir():
        return {
            "schema_version": SCHEMA_VERSION,
            "error": "directory_not_found",
            "model_dir": str(model_dir),
        }

    config = read_config(model_dir)
    manifest = read_manifest(model_dir)

    recipes = CANDIDATE_RECIPES
    if recipe_ids:
        recipes = [r for r in recipes if r["recipe_id"] in recipe_ids]

    candidates = [
        build_candidate_manifest(model_dir, recipe, config, manifest)
        for recipe in recipes
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "model_dir": str(model_dir),
        "candidate_count": len(candidates),
        "candidates": candidates,
    }


def main() -> int:
    args = parse_args()
    result = build_candidates(args.model_dir, args.recipes)

    output_text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output_text)
        print(f"wrote {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(output_text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
