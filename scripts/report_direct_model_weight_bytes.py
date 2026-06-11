#!/usr/bin/env python3
"""Report resolved safetensor byte totals for a direct-mode model directory.

Resolves Hugging Face cache symlinks and computes safetensor byte totals for
dense and MoE models. For dense models, safetensor_bytes is a first-order
bytes/token estimate. For MoE models, the helper marks total bytes as
non-comparable unless active-expert accounting exists.

Output schema: ax.direct_weight_bytes.v1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCHEMA_VERSION = "ax.direct_weight_bytes.v1"
SAFETENSOR_SUFFIX = ".safetensors"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "model_dir",
        type=Path,
        help="Model directory (may be an HF cache snapshot with symlinks).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write JSON artifact to this path instead of stdout.",
    )
    return p.parse_args()


def resolve_symlinks(model_dir: Path) -> tuple[Path, bool]:
    resolved = model_dir.resolve()
    return resolved, resolved != model_dir


def sum_safetensor_bytes(model_dir: Path) -> tuple[int, int, list[str], bool]:
    total = 0
    count = 0
    files: list[str] = []
    file_symlinks_followed = False
    for path in sorted(model_dir.rglob(f"*{SAFETENSOR_SUFFIX}")):
        if path.is_file():
            size = path.stat().st_size
            total += size
            count += 1
            files.append(str(path.relative_to(model_dir)))
            file_symlinks_followed = file_symlinks_followed or path.is_symlink()
    return total, count, files, file_symlinks_followed


def detect_moe_from_manifest(model_dir: Path) -> dict | None:
    manifest_path = model_dir / "model-manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    return manifest.get("moe")


def compute_active_expert_bytes(
    model_dir: Path, total_bytes: int
) -> tuple[int | None, str]:
    moe_block = detect_moe_from_manifest(model_dir)
    if moe_block is None:
        return None, "no_manifest_or_no_moe_block"

    expert_count = int(moe_block.get("expert_count", 0))
    experts_per_token = int(moe_block.get("experts_per_token", 0))
    if expert_count <= 0 or experts_per_token <= 0:
        return None, "moe_block_invalid"

    manifest_path = model_dir / "model-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    tensors = manifest.get("tensors", [])
    if not tensors:
        return None, "manifest_no_tensors"

    routed_expert_bytes = 0
    other_bytes = 0
    for t in tensors:
        role = t.get("role", "")
        nbytes = t.get("length_bytes", 0)
        if "shared_expert" in role:
            other_bytes += nbytes
        elif any(tag in role for tag in ("_exps", "routed_expert")):
            routed_expert_bytes += nbytes
        else:
            other_bytes += nbytes

    if routed_expert_bytes == 0:
        return None, "no_routed_expert_tensors"

    active_ratio = experts_per_token / expert_count
    active_routed = int(routed_expert_bytes * active_ratio)
    return other_bytes + active_routed, "manifest_active_expert"


def build_report(model_dir: Path) -> dict:
    if not model_dir.is_dir():
        raise SystemExit(f"model_dir does not exist or is not a directory: {model_dir}")

    resolved_dir, model_dir_symlink = resolve_symlinks(model_dir)
    (
        safetensor_bytes,
        safetensor_count,
        safetensor_files,
        safetensor_symlink,
    ) = sum_safetensor_bytes(resolved_dir)
    symlinks_followed = model_dir_symlink or safetensor_symlink

    if safetensor_bytes == 0:
        raise SystemExit(
            f"No safetensor files found under {resolved_dir}. "
            "Ensure the model directory contains *.safetensors files."
        )

    active_bytes, active_source = compute_active_expert_bytes(
        resolved_dir, safetensor_bytes
    )
    moe_block = detect_moe_from_manifest(resolved_dir)

    dense_estimate_supported = moe_block is None
    moe_active_bytes_supported = active_bytes is not None

    report = {
        "schema_version": SCHEMA_VERSION,
        "model_dir": str(model_dir),
        "resolved_model_dir": str(resolved_dir),
        "symlinks_followed": symlinks_followed,
        "safetensor_bytes": safetensor_bytes,
        "safetensor_files": safetensor_count,
        "safetensor_file_list": safetensor_files,
        "dense_estimate_supported": dense_estimate_supported,
        "moe_active_bytes_supported": moe_active_bytes_supported,
        "moe_active_bytes": active_bytes,
        "moe_active_bytes_source": active_source,
        "moe_block": moe_block,
    }
    return report


def main() -> int:
    args = parse_args()
    report = build_report(args.model_dir)

    output_text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output_text)
        print(f"wrote {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(output_text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
