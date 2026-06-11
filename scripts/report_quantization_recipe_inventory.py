#!/usr/bin/env python3
"""Report quantization recipe inventory for supported direct-mode models.

Scans model artifact directories and produces a quantization recipe inventory
including bits-per-weight, layout, group sizes, and active-expert accounting.

Output schema: ax.quantization_recipe_inventory.v1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "ax.quantization_recipe_inventory.v1"
WEIGHT_FILE_SUFFIXES = (".safetensors", ".gguf", ".npz", ".bin")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "model_dirs",
        nargs="+",
        type=Path,
        help="Model artifact directories to inventory.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write JSON artifact to this path instead of stdout.",
    )
    return p.parse_args()


def sum_weight_bytes(model_dir: Path) -> int:
    total = 0
    for path in model_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in WEIGHT_FILE_SUFFIXES:
            total += path.stat().st_size
    return total


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


def extract_quantization_recipe(config: dict[str, Any]) -> dict[str, Any]:
    quant = config.get("quantization", {})
    if not isinstance(quant, dict):
        return {"bits": None, "group_size": None, "layout": "unknown"}

    bits = quant.get("bits")
    group_size = quant.get("group_size")
    layout = quant.get("layout", "uniform")

    per_layer_overrides: dict[str, Any] = {}
    for key, value in quant.items():
        if isinstance(value, dict) and ("bits" in value or "group_size" in value):
            per_layer_overrides[key] = {
                "bits": value.get("bits"),
                "group_size": value.get("group_size"),
            }

    return {
        "bits": bits,
        "group_size": group_size,
        "layout": layout,
        "per_layer_overrides": per_layer_overrides,
    }


def compute_active_expert_accounting(manifest: dict[str, Any]) -> dict[str, Any] | None:
    moe_block = manifest.get("moe")
    if not moe_block:
        return None

    expert_count = int(moe_block.get("expert_count", 0))
    experts_per_token = int(moe_block.get("experts_per_token", 0))
    if expert_count <= 0 or experts_per_token <= 0:
        return {"is_moe": True, "valid": False, "reason": "invalid_moe_block"}

    tensors = manifest.get("tensors", [])
    routed_bytes = 0
    other_bytes = 0
    for t in tensors:
        role = t.get("role", "")
        nbytes = t.get("length_bytes", 0)
        if "shared_expert" in role:
            other_bytes += nbytes
        elif any(tag in role for tag in ("_exps", "routed_expert")):
            routed_bytes += nbytes
        else:
            other_bytes += nbytes

    if routed_bytes == 0:
        return {"is_moe": True, "valid": False, "reason": "no_routed_expert_tensors"}

    active_ratio = experts_per_token / expert_count
    active_routed = int(routed_bytes * active_ratio)
    active_bytes = other_bytes + active_routed

    return {
        "is_moe": True,
        "valid": True,
        "expert_count": expert_count,
        "experts_per_token": experts_per_token,
        "active_ratio": round(active_ratio, 6),
        "routed_expert_bytes": routed_bytes,
        "other_bytes": other_bytes,
        "active_bytes_per_forward_pass": active_bytes,
    }


def build_recipe_entry(model_dir: Path) -> dict[str, Any]:
    if not model_dir.is_dir():
        return {
            "model_dir": str(model_dir),
            "error": "directory_not_found",
        }

    config = read_config(model_dir)
    manifest = read_manifest(model_dir)
    weight_bytes = sum_weight_bytes(model_dir)

    recipe = extract_quantization_recipe(config)
    active_accounting = compute_active_expert_accounting(manifest)

    model_type = config.get("model_type", "unknown")
    model_family = manifest.get("model_family", model_type)

    entry: dict[str, Any] = {
        "model_dir": str(model_dir),
        "model_type": model_type,
        "model_family": model_family,
        "weight_bytes": weight_bytes,
        "quantization_recipe": recipe,
        "active_expert_accounting": active_accounting,
        "has_manifest": bool(manifest),
        "has_config": bool(config),
    }

    if active_accounting and active_accounting.get("valid"):
        entry["bytes_used_for_bandwidth_estimate"] = active_accounting[
            "active_bytes_per_forward_pass"
        ]
        entry["estimate_kind"] = "moe_active_estimate"
    elif weight_bytes > 0:
        entry["bytes_used_for_bandwidth_estimate"] = weight_bytes
        entry["estimate_kind"] = "dense_weight_total"
    else:
        entry["bytes_used_for_bandwidth_estimate"] = 0
        entry["estimate_kind"] = "no_weights"

    return entry


def build_inventory(model_dirs: list[Path]) -> dict[str, Any]:
    entries = [build_recipe_entry(d) for d in model_dirs]
    return {
        "schema_version": SCHEMA_VERSION,
        "model_count": len(entries),
        "entries": entries,
    }


def main() -> int:
    args = parse_args()
    inventory = build_inventory(args.model_dirs)

    output_text = json.dumps(inventory, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output_text)
        print(f"wrote {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(output_text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
