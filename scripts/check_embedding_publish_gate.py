#!/usr/bin/env python3
"""Publication gate for embedding fair and ingest-scale benchmark artifacts.

Use this before wiring an artifact into README tables or performance charts.

Claim modes
-----------
- ``paired_delta``: same-session AX vs reference (mlx-lm / mlx-embeddings).
  Requires ``ax_only=false``, both engines on every row, host + runtime
  identity (including libmlx linkage fingerprints), and build commit.
- ``ax_absolute_trend``: AX-only absolute throughput/latency trend.
  Requires ``ax_only=true`` (or only AX results), host + runtime identity,
  and build commit. Must **not** be used to invent a reference delta.

Legacy v1 artifacts without runtime_identity are accepted only with
``--allow-legacy`` (historical retained rows). New harnesses emit v2.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

FAIR_SCHEMAS = {"ax.embedding_fair.v1", "ax.embedding_fair.v2"}
SCALE_SCHEMAS = {
    "ax.embedding_ingest_scale.v1",
    "ax.embedding_ingest_scale.v2",
}
SUPPORTED_SCHEMAS = FAIR_SCHEMAS | SCALE_SCHEMAS
V2_SCHEMAS = {"ax.embedding_fair.v2", "ax.embedding_ingest_scale.v2"}
CLAIM_PAIRED = "paired_delta"
CLAIM_AX_ONLY = "ax_absolute_trend"
VALID_CLAIMS = {CLAIM_PAIRED, CLAIM_AX_ONLY}


class PublishGateError(ValueError):
    pass


def load_artifact(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise PublishGateError(f"{path}: failed to load JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise PublishGateError(f"{path}: root must be an object")
    return payload


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PublishGateError(message)


def reference_key(artifact: dict[str, Any]) -> str:
    reference = artifact.get("reference", "mlx_lm")
    if reference == "mlx_embeddings":
        return "mlx_embeddings"
    return "mlx_lm"


def validate_host(artifact: dict[str, Any], *, path: Path) -> list[str]:
    warnings: list[str] = []
    host = artifact.get("host")
    if not isinstance(host, dict) or not host:
        raise PublishGateError(f"{path}: missing host metadata")
    if not host.get("chip") or host.get("chip") == "unknown":
        warnings.append(f"{path}: host.chip is unknown")
    return warnings


def validate_build(artifact: dict[str, Any], *, path: Path) -> list[str]:
    warnings: list[str] = []
    build = artifact.get("build")
    commit = None
    if isinstance(build, dict):
        commit = build.get("commit")
        if build.get("git_tracked_dirty"):
            warnings.append(
                f"{path}: build.git_tracked_dirty=true "
                "(publication prefers a clean tree)"
            )
    if not commit or commit == "unknown":
        commit = artifact.get("git_commit")
    require(
        isinstance(commit, str) and bool(commit) and commit != "unknown",
        f"{path}: missing build commit / git_commit",
    )
    return warnings


def _linked_sources(identity: dict[str, Any], side: str) -> list[str]:
    block = identity.get(side)
    if not isinstance(block, dict):
        return []
    linked = block.get("linked_mlx") or []
    sources = []
    for entry in linked:
        if isinstance(entry, dict) and entry.get("source_class"):
            sources.append(str(entry["source_class"]))
    return sources


def validate_runtime_identity(
    artifact: dict[str, Any],
    *,
    path: Path,
    claim: str,
) -> list[str]:
    warnings: list[str] = []
    identity = artifact.get("runtime_identity")
    require(
        isinstance(identity, dict) and bool(identity),
        f"{path}: missing runtime_identity "
        "(libmlx path/sha required for publication claims)",
    )
    ax_native = identity.get("ax_engine_native")
    require(
        isinstance(ax_native, dict),
        f"{path}: runtime_identity.ax_engine_native is required",
    )
    linked = ax_native.get("linked_mlx") or []
    if not linked:
        warnings.append(
            f"{path}: runtime_identity.ax_engine_native.linked_mlx is empty "
            "(could not fingerprint libmlx via otool)"
        )
    ax_sources = set(_linked_sources(identity, "ax_engine_native"))
    if "homebrew" in ax_sources:
        warnings.append(
            f"{path}: AX native extension appears linked to Homebrew libmlx; "
            "paired deltas vs pip mlx-lm historically showed ~3× false gaps. "
            "Prefer the venv/pip wheel (see mlx-sys / ax-engine-py rpath)."
        )
    if claim == CLAIM_PAIRED:
        ref_rt = identity.get("reference_runtime")
        require(
            isinstance(ref_rt, dict),
            f"{path}: paired_delta requires runtime_identity.reference_runtime",
        )
        ref_sources = set(_linked_sources(identity, "reference_runtime"))
        if ax_sources and ref_sources and ax_sources.isdisjoint(ref_sources):
            # Not always fatal (source_class can differ in labeling), but flag.
            if "homebrew" in ax_sources and "pip_or_venv" in ref_sources:
                raise PublishGateError(
                    f"{path}: AX uses Homebrew libmlx while reference uses "
                    "pip/venv MLX — reject paired_delta publication"
                )
    return warnings


def validate_rows(
    artifact: dict[str, Any],
    *,
    path: Path,
    claim: str,
) -> list[str]:
    warnings: list[str] = []
    models = artifact.get("models")
    require(isinstance(models, list) and bool(models), f"{path}: models must be non-empty")
    ref_key = reference_key(artifact)
    schema = str(artifact.get("schema_version", ""))
    is_fair = schema in FAIR_SCHEMAS

    for model in models:
        require(isinstance(model, dict), f"{path}: model entries must be objects")
        label = model.get("model_label", "<unknown>")
        rows = model.get("rows")
        require(
            isinstance(rows, list) and bool(rows),
            f"{path}: model {label} has no rows",
        )
        for row in rows:
            require(isinstance(row, dict), f"{path}: row must be an object")
            workload = row.get("workload", "<unknown>")
            results = row.get("results")
            require(
                isinstance(results, dict),
                f"{path}: {label}/{workload} missing results",
            )
            ax = results.get("ax_engine_py")
            require(
                isinstance(ax, dict),
                f"{path}: {label}/{workload} missing ax_engine_py results",
            )
            if claim == CLAIM_PAIRED:
                ref = results.get(ref_key)
                require(
                    isinstance(ref, dict),
                    f"{path}: {label}/{workload} missing {ref_key} results "
                    f"for paired_delta",
                )
                comparison = row.get("comparison")
                require(
                    isinstance(comparison, dict) and bool(comparison),
                    f"{path}: {label}/{workload} missing comparison for paired_delta",
                )
            if is_fair:
                # Short-query rows must carry latency metrics as primary.
                if str(workload).startswith("short_query"):
                    if "median_ms_per_item" not in ax:
                        warnings.append(
                            f"{path}: {label}/{workload} lacks median_ms_per_item "
                            "(short-query primary metric)"
                        )
                    primary = row.get("primary_metric")
                    if primary and primary != "median_ms_per_item":
                        warnings.append(
                            f"{path}: {label}/{workload} primary_metric="
                            f"{primary!r} expected median_ms_per_item"
                        )
                require(
                    "median_tokens_per_sec" in ax,
                    f"{path}: {label}/{workload} missing median_tokens_per_sec",
                )
            else:
                require(
                    "median_tokens_per_sec" in ax,
                    f"{path}: {label}/{workload} missing median_tokens_per_sec",
                )
                require(
                    "median_batch_p95_ms" in ax,
                    f"{path}: {label}/{workload} missing median_batch_p95_ms",
                )
    return warnings


def validate_claim_shape(
    artifact: dict[str, Any],
    *,
    path: Path,
    claim: str,
) -> None:
    ax_only = bool(artifact.get("ax_only"))
    declared = artifact.get("publication_claim")
    if isinstance(declared, str) and declared in VALID_CLAIMS and declared != claim:
        raise PublishGateError(
            f"{path}: artifact publication_claim={declared!r} does not match "
            f"requested claim={claim!r}"
        )
    if claim == CLAIM_PAIRED:
        require(
            not ax_only,
            f"{path}: paired_delta claim requires ax_only=false "
            "(use same-session paired run, not AX-only overlay)",
        )
    elif claim == CLAIM_AX_ONLY:
        # Prefer explicit ax_only, but also accept artifacts that only have AX
        # results if ax_only was omitted (legacy).
        if not ax_only:
            # Soft: still allow if no reference results exist at all.
            ref_key = reference_key(artifact)
            for model in artifact.get("models") or []:
                for row in model.get("rows") or []:
                    results = row.get("results") or {}
                    if ref_key in results:
                        raise PublishGateError(
                            f"{path}: ax_absolute_trend claim but artifact "
                            f"contains {ref_key} results with ax_only=false; "
                            "use paired_delta or re-run with --ax-only"
                        )


def validate_artifact(
    path: Path,
    *,
    claim: str,
    allow_legacy: bool = False,
    require_clean_tree: bool = False,
) -> dict[str, Any]:
    if claim not in VALID_CLAIMS:
        raise PublishGateError(f"unknown claim mode: {claim}")
    artifact = load_artifact(path)
    schema = artifact.get("schema_version")
    require(
        schema in SUPPORTED_SCHEMAS,
        f"{path}: unsupported schema_version {schema!r}",
    )
    require(
        artifact.get("output_contract") == "contiguous_cpu_f32_batch_hidden",
        f"{path}: output_contract must be contiguous_cpu_f32_batch_hidden",
    )
    validate_claim_shape(artifact, path=path, claim=claim)

    warnings: list[str] = []
    is_v2 = schema in V2_SCHEMAS
    has_identity = isinstance(artifact.get("runtime_identity"), dict)

    if is_v2 or has_identity:
        warnings.extend(validate_host(artifact, path=path))
        warnings.extend(validate_build(artifact, path=path))
        warnings.extend(validate_runtime_identity(artifact, path=path, claim=claim))
    else:
        if not allow_legacy:
            raise PublishGateError(
                f"{path}: legacy artifact without runtime_identity; "
                "re-run the v2 harness or pass --allow-legacy for retained "
                "historical rows"
            )
        warnings.append(
            f"{path}: legacy schema {schema} accepted via --allow-legacy "
            "(no runtime_identity)"
        )
        # Still require git_commit when present for basic provenance.
        if not artifact.get("git_commit"):
            warnings.append(f"{path}: legacy artifact missing git_commit")

    if require_clean_tree:
        build = artifact.get("build") if isinstance(artifact.get("build"), dict) else {}
        if build.get("git_tracked_dirty"):
            raise PublishGateError(
                f"{path}: --require-clean-tree but build.git_tracked_dirty=true"
            )

    warnings.extend(validate_rows(artifact, path=path, claim=claim))

    # Methodology floor for publication.
    warmup = artifact.get("warmup")
    trials = artifact.get("trials")
    if isinstance(warmup, int) and warmup < 2:
        warnings.append(f"{path}: warmup={warmup} < 2 (publication convention is 2)")
    if isinstance(trials, int) and trials < 5:
        warnings.append(f"{path}: trials={trials} < 5 (publication convention is 5)")

    return {
        "path": str(path),
        "schema_version": schema,
        "claim": claim,
        "ax_only": bool(artifact.get("ax_only")),
        "ok": True,
        "warnings": warnings,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "artifacts",
        nargs="+",
        type=Path,
        help="Paths to embedding_fair.json or embedding_ingest_scale.json",
    )
    parser.add_argument(
        "--claim",
        choices=sorted(VALID_CLAIMS),
        default=CLAIM_PAIRED,
        help="Publication claim mode to validate (default: paired_delta).",
    )
    parser.add_argument(
        "--allow-legacy",
        action="store_true",
        help="Accept v1 artifacts without runtime_identity (historical only).",
    )
    parser.add_argument(
        "--require-clean-tree",
        action="store_true",
        help="Fail when build.git_tracked_dirty is true.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON summary on stdout.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    reports = []
    failed = False
    for artifact_path in args.artifacts:
        try:
            report = validate_artifact(
                artifact_path,
                claim=args.claim,
                allow_legacy=args.allow_legacy,
                require_clean_tree=args.require_clean_tree,
            )
            reports.append(report)
            for warning in report["warnings"]:
                print(f"warning: {warning}", file=sys.stderr)
            print(f"ok: {artifact_path} claim={args.claim}", file=sys.stderr)
        except PublishGateError as exc:
            failed = True
            print(f"error: {exc}", file=sys.stderr)
            reports.append(
                {
                    "path": str(artifact_path),
                    "claim": args.claim,
                    "ok": False,
                    "error": str(exc),
                }
            )
    if args.json:
        print(json.dumps({"reports": reports}, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
