#!/usr/bin/env python3
"""Validate an AX MTP sidecar provenance manifest.

The fair MTP benchmark track depends on generated sidecars being traceable to a
standard base model plus standard HF MTP source shards. This checker validates
the manifest written by scripts/prepare_qwen36_mtp_sidecar.py before those
sidecars are used for publication-grade benchmarks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

MANIFEST_FILE = "ax_mtp_sidecar_manifest.json"
SCHEMA_VERSION = "ax.mtp_sidecar_provenance.v1"


class ProvenanceError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(path_or_dir: Path) -> tuple[Path, dict[str, Any]]:
    path = path_or_dir / MANIFEST_FILE if path_or_dir.is_dir() else path_or_dir
    if not path.is_file():
        raise ProvenanceError(f"manifest not found: {path}")
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ProvenanceError(f"manifest is not valid JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ProvenanceError("manifest root must be an object")
    return path, payload


def require_object(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ProvenanceError(f"manifest.{key} must be an object")
    return value


def require_string(parent: dict[str, Any], dotted: str) -> str:
    current: Any = parent
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            raise ProvenanceError(f"manifest.{dotted} is required")
        current = current[part]
    if not isinstance(current, str) or not current:
        raise ProvenanceError(f"manifest.{dotted} must be a non-empty string")
    return current


def require_int(parent: dict[str, Any], dotted: str) -> int:
    current: Any = parent
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            raise ProvenanceError(f"manifest.{dotted} is required")
        current = current[part]
    if not isinstance(current, int):
        raise ProvenanceError(f"manifest.{dotted} must be an integer")
    return current


def validate_file_record(record: dict[str, Any], label: str, *, strict_local: bool) -> None:
    path_raw = record.get("path")
    if not isinstance(path_raw, str) or not path_raw:
        raise ProvenanceError(f"{label}.path must be a non-empty string")
    path = Path(path_raw).expanduser()
    exists = record.get("exists")
    if exists is not True:
        raise ProvenanceError(f"{label}.exists must be true")
    expected_size = record.get("size_bytes")
    expected_sha = record.get("sha256")
    if not isinstance(expected_size, int) or expected_size <= 0:
        raise ProvenanceError(f"{label}.size_bytes must be a positive integer")
    if not isinstance(expected_sha, str) or len(expected_sha) != 64:
        raise ProvenanceError(f"{label}.sha256 must be a SHA-256 hex string")
    if strict_local:
        if not path.is_file():
            raise ProvenanceError(f"{label}.path does not exist locally: {path}")
        actual_size = path.stat().st_size
        if actual_size != expected_size:
            raise ProvenanceError(
                f"{label}.size_bytes mismatch: manifest={expected_size} actual={actual_size}"
            )
        actual_sha = sha256_file(path)
        if actual_sha != expected_sha:
            raise ProvenanceError(
                f"{label}.sha256 mismatch: manifest={expected_sha} actual={actual_sha}"
            )


def validate_manifest(
    manifest: dict[str, Any],
    *,
    strict_local: bool = True,
    fair_base_only: bool = False,
) -> dict[str, Any]:
    schema = require_string(manifest, "schema_version")
    if schema != SCHEMA_VERSION:
        raise ProvenanceError(f"unsupported schema_version: {schema}")

    require_string(manifest, "generated_by")
    require_string(manifest, "model_key")
    base = require_object(manifest, "base")
    source = require_object(manifest, "source")
    output = require_object(manifest, "output")
    transform = require_object(manifest, "transform")
    runtime = require_object(manifest, "runtime")

    base_model_id = require_string(manifest, "base.model_id")
    source_model_id = require_string(manifest, "source.model_id")
    transform_policy = require_string(manifest, "transform.norm_policy")
    require_int(manifest, "runtime.mtp_depth_max")
    require_int(manifest, "runtime.mtp_tensor_count")

    source_shards = source.get("mtp_shards")
    if not isinstance(source_shards, list) or not source_shards:
        raise ProvenanceError("manifest.source.mtp_shards must be a non-empty list")
    for index, shard in enumerate(source_shards):
        if not isinstance(shard, dict):
            raise ProvenanceError(f"manifest.source.mtp_shards[{index}] must be an object")
        require_string(shard, "name")
        validate_file_record(shard, f"source.mtp_shards[{index}]", strict_local=strict_local)

    for label in ("base.config", "output.mtp", "output.runtime", "output.config"):
        parent_key, child_key = label.split(".", 1)
        record_parent = base if parent_key == "base" else output
        record = record_parent.get(child_key)
        if not isinstance(record, dict):
            raise ProvenanceError(f"{label} must be an object")
        validate_file_record(record, label, strict_local=strict_local)

    if fair_base_only:
        if not base_model_id.startswith("mlx-community/"):
            raise ProvenanceError(
                f"fair base track requires mlx-community base model, got {base_model_id}"
            )
        if not source_model_id.startswith("Qwen/"):
            raise ProvenanceError(
                f"fair base track requires standard Qwen source model, got {source_model_id}"
            )
        if base_model_id.startswith("Youssofal/") or source_model_id.startswith(
            "Youssofal/"
        ):
            raise ProvenanceError("fair base track must not use Youssofal optimized bundles")

    return {
        "schema_version": schema,
        "model_key": manifest["model_key"],
        "base_model_id": base_model_id,
        "source_model_id": source_model_id,
        "source_shard_count": len(source_shards),
        "mtp_tensor_count": runtime["mtp_tensor_count"],
        "mtp_depth_max": runtime["mtp_depth_max"],
        "norm_policy": transform_policy,
        "fair_base_only": fair_base_only,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest_or_dir", type=Path)
    parser.add_argument(
        "--no-strict-local",
        action="store_true",
        help="Do not re-hash referenced local files.",
    )
    parser.add_argument(
        "--fair-base-only",
        action="store_true",
        help="Require mlx-community base and standard Qwen source model IDs.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON summary.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest_path, manifest = load_manifest(args.manifest_or_dir)
        summary = validate_manifest(
            manifest,
            strict_local=not args.no_strict_local,
            fair_base_only=args.fair_base_only,
        )
    except ProvenanceError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    summary = {"manifest": str(manifest_path), **summary}
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(
            "MTP sidecar provenance OK: "
            f"{summary['base_model_id']} + {summary['source_model_id']} "
            f"({summary['source_shard_count']} shards, {summary['mtp_tensor_count']} tensors)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
