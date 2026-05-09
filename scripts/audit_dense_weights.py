#!/usr/bin/env python3
"""Audit dense BF16/F16 tensors in an AX Engine model manifest.

Reads model-manifest.json and classifies every projection-like tensor by its
source dtype, quantization state, and runtime dtype.  Produces the artifact
format required by MLX-RUNTIME-PERFORMANCE-PRD.md §W3.

Usage:
  python3 scripts/audit_dense_weights.py \
    --model-dir .internal/models/gemma-4-e2b-it-4bit \
    --output-root benchmarks/results/dense-weight-audit

  # Override model-id label (defaults to model-dir basename):
  python3 scripts/audit_dense_weights.py \
    --model-dir .internal/models/gemma-4-e2b-it-4bit \
    --model-id gemma-4-e2b-it-4bit
"""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

# Roles that are matrix multiplications in GEMV-heavy decode paths.
PROJECTION_ROLES: frozenset[str] = frozenset({
    "attention_q",
    "attention_k",
    "attention_v",
    "attention_o",
    "ffn_gate",
    "ffn_up",
    "ffn_down",
    "per_layer_input_gate",
    "per_layer_input_projection",
    "per_layer_model_projection",
    "token_embedding",
    "per_layer_embedding",
})

# Roles that are normalisation weights or biases — always dense by design.
NORM_ROLES: frozenset[str] = frozenset({
    "attention_norm",
    "ffn_norm",
    "attention_k_norm",
    "attention_q_norm",
    "attention_post_norm",
    "ffn_post_norm",
    "final_norm",
    "per_layer_projection_norm",
    "per_layer_input_post_norm",
})

# Roles that are scalars with no GEMV value.
SCALAR_ROLES: frozenset[str] = frozenset({
    "layer_scalar",
})


def _git_commit(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _classify_tensor(t: dict[str, Any]) -> tuple[str, str]:
    """Return (classification, action) for a single manifest tensor entry."""
    role: str = t.get("role", "")
    dtype: str = t.get("dtype", "")
    source_quantized: bool | None = t.get("source_quantized")
    quantization: dict[str, Any] | None = t.get("quantization")

    # u32 is AX Engine's packed 4-bit format.
    if dtype == "u32":
        if source_quantized is True and isinstance(quantization, dict):
            return "intentional_quantized", "none"
        # u32 without source_quantized metadata is a converter gap.
        return "converter_metadata_gap", "fix_converter"

    # Dense (bf16 / f16) path.
    if role in SCALAR_ROLES:
        return "non_gemv", "none"

    if role in NORM_ROLES:
        return "intentional_dense", "none"

    if role in PROJECTION_ROLES:
        shape: list[int] = t.get("shape", [])
        # 1-D tensors (bias vectors, scale vectors) are never GEMV targets.
        if len(shape) <= 1:
            return "intentional_dense", "none"
        # source_quantized=False means the source checkpoint explicitly stored
        # this projection as dense (intentional by the converter or model author).
        if source_quantized is False:
            return "intentional_dense", "document"
        # No source_quantized field: manifest predates the field or the source
        # checkpoint was never quantized for this role.  Flag for documentation.
        if source_quantized is None:
            return "intentional_dense", "document"

    # Fallback for unknown or unclassified roles.
    return "intentional_dense", "document"


def _source_dtype(t: dict[str, Any]) -> str:
    dtype = t.get("dtype", "")
    if dtype == "u32":
        return "BF16"
    return dtype.upper().replace("BFLOAT16", "BF16")


def _runtime_dtype(t: dict[str, Any]) -> str:
    dtype = t.get("dtype", "")
    if dtype == "u32":
        return "BF16"
    return dtype.upper().replace("BFLOAT16", "BF16")


def _quantization_bits(t: dict[str, Any]) -> int:
    q = t.get("quantization")
    if isinstance(q, dict):
        return int(q.get("bits", 0))
    return 0


def _sidecar_present(t: dict[str, Any]) -> bool:
    # AX Engine's affine quantization stores scales/biases packed inside the
    # u32 weight tensor (no separate sidecar file).  A sidecar is considered
    # present when source_quantized=true and a quantization block exists.
    return bool(t.get("source_quantized") and t.get("quantization"))


def build_audit(manifest_path: Path, model_id: str, commit: str) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    tensors_raw: list[dict[str, Any]] = manifest.get("tensors", [])

    tensor_entries: list[dict[str, Any]] = []
    for t in tensors_raw:
        role = t.get("role", "")
        classification, action = _classify_tensor(t)
        entry: dict[str, Any] = {
            "role": role,
            "layer": t.get("layer_index"),
            "source_dtype": _source_dtype(t),
            "quantization_bits": _quantization_bits(t),
            "sidecar_present": _sidecar_present(t),
            "runtime_dtype": _runtime_dtype(t),
            "classification": classification,
            "action": action,
        }
        tensor_entries.append(entry)

    return {
        "model": model_id,
        "commit": commit,
        "manifest": manifest_path.name,
        "tensors": tensor_entries,
    }


def _summary(audit: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for t in audit["tensors"]:
        key = t["classification"]
        counts[key] = counts.get(key, 0) + 1
    return counts


def _print_findings(audit: dict[str, Any]) -> None:
    findings = [
        t for t in audit["tensors"]
        if t["action"] not in ("none",) or t["classification"] == "converter_metadata_gap"
    ]
    if not findings:
        print("  No actionable findings.")
        return
    for f in findings:
        layer_str = f"layer={f['layer']} " if f["layer"] is not None else ""
        print(
            f"  {f['role']} {layer_str}"
            f"dtype={f['source_dtype']} bits={f['quantization_bits']} "
            f"→ {f['classification']} action={f['action']}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        required=True,
        type=Path,
        help="Path to model directory containing model-manifest.json",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Model identifier for the artifact (defaults to model-dir basename)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "benchmarks" / "results" / "dense-weight-audit",
        help="Root directory for audit output artifacts",
    )
    args = parser.parse_args(argv)

    model_dir = args.model_dir
    if not model_dir.is_absolute():
        model_dir = REPO_ROOT / model_dir

    manifest_path = model_dir / "model-manifest.json"
    if not manifest_path.exists():
        print(f"error: manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    model_id = args.model_id or model_dir.name
    commit = _git_commit(REPO_ROOT)
    date_str = datetime.date.today().isoformat()

    print(f"Auditing {model_id} (commit {commit}) ...")
    audit = build_audit(manifest_path, model_id, commit)

    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    artifact_path = output_root / f"{model_id}-{date_str}.json"
    artifact_path.write_text(json.dumps(audit, indent=2))
    try:
        label = artifact_path.relative_to(REPO_ROOT)
    except ValueError:
        label = artifact_path
    print(f"Artifact written: {label}")

    summary = _summary(audit)
    print("\nClassification summary:")
    for cls, count in sorted(summary.items()):
        print(f"  {cls}: {count}")

    print("\nActionable findings:")
    _print_findings(audit)

    has_gap = any(
        t["classification"] == "converter_metadata_gap"
        for t in audit["tensors"]
    )
    if has_gap:
        print(
            "\nWARNING: converter_metadata_gap tensors found — "
            "review fix_converter actions before merging.",
            file=sys.stderr,
        )
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
