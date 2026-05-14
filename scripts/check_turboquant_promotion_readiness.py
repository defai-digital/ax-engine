#!/usr/bin/env python3
"""Report whether TurboQuant has model-level promotion evidence.

This is a readiness probe, not a benchmark runner. It keeps the production
support decision fail-closed by checking two separate surfaces:

- local model manifests, to see whether any model can exercise the current
  fused K8/V4 decode implementation; and
- existing quality-gate artifacts, to see whether any real model artifact
  already satisfies the promotion contract.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "ax.turboquant_promotion_readiness.v1"
CHECKER_PATH = Path(__file__).with_name("check_turboquant_quality_artifact.py")
ATTENTION_TENSOR_ROLES = {
    "attention_o",
    "attention_q",
    "attention_k",
    "attention_v",
    "attention_qkv_packed",
}
CHECKER_SPEC = importlib.util.spec_from_file_location(
    "check_turboquant_quality_artifact",
    CHECKER_PATH,
)
assert CHECKER_SPEC and CHECKER_SPEC.loader
checker = importlib.util.module_from_spec(CHECKER_SPEC)
CHECKER_SPEC.loader.exec_module(checker)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _full_attention_layers(manifest: dict[str, Any]) -> int | None:
    layer_types = manifest.get("layer_types")
    if isinstance(layer_types, list):
        return sum(1 for item in layer_types if item == "full_attention")
    layer_count = manifest.get("layer_count")
    tensors = manifest.get("tensors")
    has_attention_tensors = isinstance(tensors, list) and any(
        isinstance(tensor, dict) and tensor.get("role") in ATTENTION_TENSOR_ROLES
        for tensor in tensors
    )
    if isinstance(layer_count, int) and layer_count > 0 and has_attention_tensors:
        return layer_count
    return None


def inspect_manifest(path: Path) -> dict[str, Any]:
    manifest = _load_json(path)
    model_dir = path.parent
    attention_head_dim = manifest.get("attention_head_dim")
    fused_head_dim = manifest.get("global_head_dim") or attention_head_dim
    n_heads = manifest.get("attention_head_count")
    n_kv_heads = manifest.get("kv_head_count")
    family = manifest.get("model_family")
    full_layers = _full_attention_layers(manifest)
    has_linear_attention = bool(manifest.get("linear_attention"))
    has_mla_attention = bool(manifest.get("mla_attention"))

    blockers: list[str] = []
    if fused_head_dim not in checker.SUPPORTED_HEAD_DIMS:
        blockers.append(
            f"fused attention head_dim {fused_head_dim} is not supported by the current fused gate"
        )
    if not isinstance(n_heads, int) or not isinstance(n_kv_heads, int) or n_kv_heads <= 0:
        blockers.append("attention_head_count and kv_head_count must be valid integers")
    elif n_heads % n_kv_heads != 0:
        blockers.append(
            f"grouped-query mapping is invalid: attention_head_count={n_heads}, kv_head_count={n_kv_heads}"
        )
    if full_layers is None:
        blockers.append("full_attention layer coverage could not be determined")
    elif full_layers == 0:
        blockers.append("no full_attention layers are available for first production preset")
    if has_linear_attention:
        blockers.append("linear-attention state remains outside TurboQuant promotion scope")
    if has_mla_attention:
        blockers.append("GLM MLA attention remains outside TurboQuant promotion scope")

    return {
        "model_dir": _relative(model_dir),
        "model_family": family,
        "attention_head_dim": attention_head_dim,
        "fused_attention_head_dim": fused_head_dim,
        "attention_head_count": n_heads,
        "kv_head_count": n_kv_heads,
        "full_attention_layers": full_layers,
        "eligible_for_current_fused_gate": not blockers,
        "blockers": blockers,
    }


def discover_manifests(models_root: Path) -> list[Path]:
    if models_root.is_file():
        return [models_root]
    if not models_root.exists():
        return []
    return sorted(models_root.glob("*/model-manifest.json"))


def discover_quality_artifacts(results_root: Path) -> list[Path]:
    if not results_root.exists():
        return []
    return sorted(results_root.rglob("*quality-gate*.json"))


def inspect_quality_artifact(
    path: Path,
    *,
    require_files: bool,
    root: Path = REPO_ROOT,
) -> dict[str, Any]:
    try:
        doc = _load_json(path)
        checker.validate_artifact(doc, root=root, require_files=require_files)
    except Exception as exc:  # noqa: BLE001 - report fail-closed reason verbatim.
        return {
            "path": _relative(path),
            "passes_quality_gate": False,
            "passes_performance_gate": False,
            "quality_blocker": str(exc),
            "performance_blockers": [],
        }
    performance_blockers = checker.performance_gate_blockers(doc.get("metrics", {}))
    metrics = doc.get("metrics", {})
    observed_decode_ratio = (
        metrics.get("decode_tok_s_ratio_to_baseline")
        if isinstance(metrics, dict)
        else None
    )
    return {
        "path": _relative(path),
        "passes_quality_gate": True,
        "passes_performance_gate": not performance_blockers,
        "quality_blocker": None,
        "performance_blockers": performance_blockers,
        "promotion_gap": {
            "observed_decode_tok_s_ratio_to_baseline": observed_decode_ratio,
            "required_min_decode_tok_s_ratio_to_baseline": checker.MIN_DECODE_RATIO_TO_BASELINE,
            "performance_promotion_ready": not performance_blockers,
            "next_action": (
                "ready_for_companion_prd_review"
                if not performance_blockers
                else "rerun or improve fused compressed decode until performance blockers clear"
            ),
        },
    }


def build_report(
    *,
    models_root: Path,
    results_root: Path,
    artifacts: list[Path],
    require_artifact_files: bool,
    root: Path = REPO_ROOT,
) -> dict[str, Any]:
    manifests = [inspect_manifest(path) for path in discover_manifests(models_root)]
    artifact_paths = artifacts or discover_quality_artifacts(results_root)
    quality_artifacts = [
        inspect_quality_artifact(
            path,
            require_files=require_artifact_files,
            root=root,
        )
        for path in artifact_paths
    ]
    eligible_models = [item for item in manifests if item["eligible_for_current_fused_gate"]]
    passing_quality_artifacts = [
        item for item in quality_artifacts if item["passes_quality_gate"]
    ]
    passing_performance_artifacts = [
        item
        for item in passing_quality_artifacts
        if item["passes_performance_gate"]
    ]

    blockers: list[str] = []
    if not eligible_models:
        blockers.append(
            "no local model manifest can exercise the current fused K8/V4 promotion gate"
        )
    if not passing_quality_artifacts:
        blockers.append("no passing long-context fused-path quality artifact was found")
    elif not passing_performance_artifacts:
        blockers.append(
            "no passing long-context fused-path performance promotion artifact was found"
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "decision": {
            "can_make_public_support_claim": not blockers,
            "public_docs_should_remain_experimental": bool(blockers),
            "blockers": blockers,
        },
        "required_current_gate": {
            "candidate_mode": checker.REQUIRED_CANDIDATE_COMPRESSION_MODE,
            "preset": "k8v4",
            "head_dim": sorted(checker.SUPPORTED_HEAD_DIMS),
            "decode_path": "fused_compressed_decode",
            "fused_decode_successes": "> 0",
            "fused_decode_fallbacks": 0,
            "minimum_context_tokens": checker.MIN_CONTEXT_TOKENS,
            "minimum_generation_tokens": checker.MIN_GENERATION_TOKENS,
        },
        "models": manifests,
        "quality_artifacts": quality_artifacts,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-root", type=Path, default=REPO_ROOT / ".internal/models")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=REPO_ROOT / "benchmarks/results/turboquant",
    )
    parser.add_argument("--artifact", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fail-on-blockers", action="store_true")
    parser.add_argument(
        "--no-require-artifact-files",
        action="store_true",
        help="validate artifact shape without checking referenced paths exist",
    )
    args = parser.parse_args(argv)

    try:
        report = build_report(
            models_root=args.models_root,
            results_root=args.results_root,
            artifacts=args.artifact,
            require_artifact_files=not args.no_require_artifact_files,
            root=REPO_ROOT,
        )
        text = json.dumps(report, indent=2) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text)
        else:
            sys.stdout.write(text)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    has_blockers = bool(report["decision"]["blockers"])
    if args.fail_on_blockers and has_blockers:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
