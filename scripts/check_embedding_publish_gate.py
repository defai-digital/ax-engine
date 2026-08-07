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
import math
import re
import statistics
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
MIN_WARMUPS = 2
MIN_TRIALS = 5
MIN_SCALE_COOLDOWN_SECONDS = 15.0
MAX_PUBLICATION_LOAD_AVERAGE = 2.0
MAX_PUBLICATION_TOP_PROCESS_CPU_PERCENT = 50.0


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


def validate_build(
    artifact: dict[str, Any],
    *,
    path: Path,
    strict: bool,
) -> list[str]:
    warnings: list[str] = []
    raw_build = artifact.get("build")
    build = raw_build if isinstance(raw_build, dict) else {}
    commit = None
    if build:
        commit = build.get("commit")
        if build.get("git_tracked_dirty"):
            warnings.append(
                f"{path}: build.git_tracked_dirty=true (publication prefers a clean tree)"
            )
    if not commit or commit == "unknown":
        commit = artifact.get("git_commit")
    require(
        isinstance(commit, str) and bool(commit) and commit != "unknown",
        f"{path}: missing build commit / git_commit",
    )
    if strict:
        require(
            isinstance(raw_build, dict),
            f"{path}: v2 publication requires build metadata",
        )
        require(
            re.fullmatch(r"[0-9a-f]{40}", commit) is not None,
            f"{path}: v2 publication requires a full measured build commit",
        )
        engine_version = build.get("engine_version")
        require(
            isinstance(engine_version, str)
            and re.fullmatch(r"\d+\.\d+\.\d+", engine_version) is not None,
            f"{path}: v2 publication requires a semantic engine version",
        )
        require(
            isinstance(build.get("git_tracked_dirty"), bool),
            f"{path}: v2 publication requires build.git_tracked_dirty",
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


def _linked_hashes(identity: dict[str, Any], side: str) -> set[str]:
    block = identity.get(side)
    if not isinstance(block, dict):
        return set()
    linked = block.get("linked_mlx") or []
    return {
        str(entry["sha256"])
        for entry in linked
        if isinstance(entry, dict)
        and isinstance(entry.get("sha256"), str)
        and bool(entry["sha256"])
    }


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
        f"{path}: missing runtime_identity (libmlx path/sha required for publication claims)",
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
        ref_linked = ref_rt.get("linked_mlx") or []
        require(
            bool(linked) and bool(ref_linked),
            f"{path}: paired_delta requires linked MLX fingerprints for both AX and the reference",
        )
        ref_sources = set(_linked_sources(identity, "reference_runtime"))
        if ax_sources != ref_sources and ("homebrew" in ax_sources or "homebrew" in ref_sources):
            raise PublishGateError(
                f"{path}: AX and reference use different Homebrew / pip "
                "libmlx sources — reject paired_delta publication"
            )
        ax_hashes = _linked_hashes(identity, "ax_engine_native")
        ref_hashes = _linked_hashes(identity, "reference_runtime")
        require(
            bool(ax_hashes) and bool(ref_hashes),
            f"{path}: paired_delta requires sha256 fingerprints for linked MLX",
        )
        require(
            ax_hashes == ref_hashes,
            f"{path}: AX and reference resolve different linked MLX binaries",
        )
    return warnings


def validate_benchmark_conditions(
    artifact: dict[str, Any],
    *,
    path: Path,
) -> None:
    max_load = artifact.get("max_load_average")
    require(
        isinstance(max_load, (int, float))
        and not isinstance(max_load, bool)
        and math.isfinite(float(max_load))
        and float(max_load) <= MAX_PUBLICATION_LOAD_AVERAGE,
        f"{path}: missing or relaxed max_load_average publication gate",
    )
    max_top_cpu = artifact.get("max_top_process_cpu_percent")
    require(
        isinstance(max_top_cpu, (int, float))
        and not isinstance(max_top_cpu, bool)
        and math.isfinite(float(max_top_cpu))
        and float(max_top_cpu) <= MAX_PUBLICATION_TOP_PROCESS_CPU_PERCENT,
        f"{path}: missing or relaxed max_top_process_cpu_percent publication gate",
    )
    window = artifact.get("benchmark_window")
    require(
        isinstance(window, dict),
        f"{path}: missing benchmark_window",
    )
    for boundary in (
        "performance_conditions_start",
        "performance_conditions_end",
    ):
        conditions = window.get(boundary)
        require(
            isinstance(conditions, dict),
            f"{path}: benchmark_window.{boundary} must be an object",
        )
        load_average = conditions.get("load_average")
        one_minute = load_average.get("one_minute") if isinstance(load_average, dict) else None
        require(
            isinstance(one_minute, (int, float))
            and not isinstance(one_minute, bool)
            and math.isfinite(float(one_minute))
            and float(one_minute) <= MAX_PUBLICATION_LOAD_AVERAGE,
            f"{path}: benchmark_window.{boundary} load exceeds publication limit",
        )
        require(
            conditions.get("power_source") == "AC Power",
            f"{path}: benchmark_window.{boundary} requires AC Power",
        )
        for warning_key in (
            "thermal_warning_recorded",
            "performance_warning_recorded",
            "cpu_power_status_recorded",
        ):
            require(
                conditions.get(warning_key) is False,
                f"{path}: benchmark_window.{boundary}.{warning_key} must be false",
            )
        top_processes = conditions.get("top_processes_cpu")
        cpu_values = (
            [
                float(process["cpu_percent"])
                for process in top_processes
                if isinstance(process, dict)
                and isinstance(process.get("cpu_percent"), (int, float))
                and not isinstance(process.get("cpu_percent"), bool)
                and math.isfinite(float(process["cpu_percent"]))
            ]
            if isinstance(top_processes, list)
            else []
        )
        require(
            bool(cpu_values) and max(cpu_values) <= MAX_PUBLICATION_TOP_PROCESS_CPU_PERCENT,
            f"{path}: benchmark_window.{boundary} top-process CPU exceeds publication limit",
        )


def validate_trial_summary(
    result: dict[str, Any],
    *,
    path: Path,
    context: str,
    declared_trials: int,
    summary_key: str,
    trial_key: str,
    allow_zero: bool = False,
) -> None:
    trial_rows = result.get("trials")
    require(
        isinstance(trial_rows, list) and len(trial_rows) == declared_trials,
        f"{path}: {context} must contain exactly {declared_trials} trial rows",
    )
    values = [row.get(trial_key) if isinstance(row, dict) else None for row in trial_rows]
    require(
        all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and (float(value) >= 0.0 if allow_zero else float(value) > 0.0)
            for value in values
        ),
        f"{path}: {context} trial {trial_key} values must be finite and "
        f"{'non-negative' if allow_zero else 'positive'}",
    )
    recorded = result.get(summary_key)
    require(
        isinstance(recorded, (int, float))
        and not isinstance(recorded, bool)
        and math.isfinite(float(recorded)),
        f"{path}: {context} {summary_key} must be finite",
    )
    expected = statistics.median(float(value) for value in values)
    require(
        math.isclose(float(recorded), expected, rel_tol=1e-9, abs_tol=1e-9),
        f"{path}: {context} {summary_key} is inconsistent with trial rows",
    )


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
    is_v2 = schema in V2_SCHEMAS
    declared_trials = artifact.get("trials")
    if is_v2:
        require(
            isinstance(declared_trials, int) and not isinstance(declared_trials, bool),
            f"{path}: v2 publication requires an integer trials count",
        )

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
            ax_tokens_per_sec = ax.get("median_tokens_per_sec")
            require(
                isinstance(ax_tokens_per_sec, (int, float))
                and not isinstance(ax_tokens_per_sec, bool)
                and math.isfinite(float(ax_tokens_per_sec))
                and float(ax_tokens_per_sec) > 0.0,
                f"{path}: {label}/{workload} ax_engine_py "
                "median_tokens_per_sec must be finite and positive",
            )
            if is_v2:
                validate_trial_summary(
                    ax,
                    path=path,
                    context=f"{label}/{workload} ax_engine_py",
                    declared_trials=declared_trials,
                    summary_key="median_tokens_per_sec",
                    trial_key="tokens_per_sec",
                )
            if claim == CLAIM_PAIRED:
                ref = results.get(ref_key)
                require(
                    isinstance(ref, dict),
                    f"{path}: {label}/{workload} missing {ref_key} results for paired_delta",
                )
                reference_tokens_per_sec = ref.get("median_tokens_per_sec")
                require(
                    isinstance(reference_tokens_per_sec, (int, float))
                    and not isinstance(reference_tokens_per_sec, bool)
                    and math.isfinite(float(reference_tokens_per_sec))
                    and float(reference_tokens_per_sec) > 0.0,
                    f"{path}: {label}/{workload} {ref_key} "
                    "median_tokens_per_sec must be finite and positive",
                )
                if is_v2:
                    validate_trial_summary(
                        ref,
                        path=path,
                        context=f"{label}/{workload} {ref_key}",
                        declared_trials=declared_trials,
                        summary_key="median_tokens_per_sec",
                        trial_key="tokens_per_sec",
                    )
                comparison = row.get("comparison")
                require(
                    isinstance(comparison, dict) and bool(comparison),
                    f"{path}: {label}/{workload} missing comparison for paired_delta",
                )
                recorded_delta = comparison.get("ax_vs_reference_tokens_pct")
                require(
                    isinstance(recorded_delta, (int, float))
                    and not isinstance(recorded_delta, bool)
                    and math.isfinite(float(recorded_delta)),
                    f"{path}: {label}/{workload} comparison "
                    "ax_vs_reference_tokens_pct must be finite",
                )
                expected_delta = (
                    (float(ax_tokens_per_sec) - float(reference_tokens_per_sec))
                    / float(reference_tokens_per_sec)
                    * 100.0
                )
                require(
                    math.isclose(
                        float(recorded_delta),
                        expected_delta,
                        rel_tol=1e-9,
                        abs_tol=1e-6,
                    ),
                    f"{path}: {label}/{workload} comparison "
                    "ax_vs_reference_tokens_pct is inconsistent with "
                    "the recorded medians",
                )
            if is_fair:
                # Short-query rows must carry latency metrics as primary.
                if str(workload).startswith("short_query"):
                    if "median_ms_per_item" not in ax:
                        warnings.append(
                            f"{path}: {label}/{workload} lacks median_ms_per_item "
                            "(short-query primary metric)"
                        )
                    elif is_v2:
                        validate_trial_summary(
                            ax,
                            path=path,
                            context=f"{label}/{workload} ax_engine_py",
                            declared_trials=declared_trials,
                            summary_key="median_ms_per_item",
                            trial_key="ms_per_item",
                        )
                        if claim == CLAIM_PAIRED:
                            validate_trial_summary(
                                ref,
                                path=path,
                                context=f"{label}/{workload} {ref_key}",
                                declared_trials=declared_trials,
                                summary_key="median_ms_per_item",
                                trial_key="ms_per_item",
                            )
                    primary = row.get("primary_metric")
                    if primary and primary != "median_ms_per_item":
                        warnings.append(
                            f"{path}: {label}/{workload} primary_metric="
                            f"{primary!r} expected median_ms_per_item"
                        )
            else:
                batch_p95_ms = ax.get("median_batch_p95_ms")
                require(
                    isinstance(batch_p95_ms, (int, float))
                    and not isinstance(batch_p95_ms, bool)
                    and math.isfinite(float(batch_p95_ms))
                    and float(batch_p95_ms) >= 0.0,
                    f"{path}: {label}/{workload} ax_engine_py "
                    "median_batch_p95_ms must be finite and non-negative",
                )
                if is_v2:
                    validate_trial_summary(
                        ax,
                        path=path,
                        context=f"{label}/{workload} ax_engine_py",
                        declared_trials=declared_trials,
                        summary_key="median_batch_p95_ms",
                        trial_key="batch_p95_ms",
                        allow_zero=True,
                    )
                    if claim == CLAIM_PAIRED:
                        reference_batch_p95_ms = ref.get("median_batch_p95_ms")
                        require(
                            isinstance(reference_batch_p95_ms, (int, float))
                            and not isinstance(reference_batch_p95_ms, bool)
                            and math.isfinite(float(reference_batch_p95_ms))
                            and float(reference_batch_p95_ms) >= 0.0,
                            f"{path}: {label}/{workload} {ref_key} "
                            "median_batch_p95_ms must be finite and non-negative",
                        )
                        validate_trial_summary(
                            ref,
                            path=path,
                            context=f"{label}/{workload} {ref_key}",
                            declared_trials=declared_trials,
                            summary_key="median_batch_p95_ms",
                            trial_key="batch_p95_ms",
                            allow_zero=True,
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
    elif claim == CLAIM_AX_ONLY and not ax_only:
        # Prefer explicit ax_only, but also accept artifacts that only have AX
        # results if ax_only was omitted (legacy).
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
        warnings.extend(validate_build(artifact, path=path, strict=is_v2))
        warnings.extend(validate_runtime_identity(artifact, path=path, claim=claim))
    else:
        if not allow_legacy:
            raise PublishGateError(
                f"{path}: legacy artifact without runtime_identity; "
                "re-run the v2 harness or pass --allow-legacy for retained "
                "historical rows"
            )
        warnings.append(
            f"{path}: legacy schema {schema} accepted via --allow-legacy (no runtime_identity)"
        )
        # Still require git_commit when present for basic provenance.
        if not artifact.get("git_commit"):
            warnings.append(f"{path}: legacy artifact missing git_commit")

    if require_clean_tree:
        build = artifact.get("build") if isinstance(artifact.get("build"), dict) else {}
        if build.get("git_tracked_dirty"):
            raise PublishGateError(f"{path}: --require-clean-tree but build.git_tracked_dirty=true")

    warnings.extend(validate_rows(artifact, path=path, claim=claim))

    # V2 schemas are current publication evidence, so provenance and
    # methodology floors fail closed. Legacy artifacts remain explicitly
    # opt-in historical context and keep warning-only behavior.
    warmup = artifact.get("warmup")
    trials = artifact.get("trials")
    if is_v2:
        if claim == CLAIM_PAIRED:
            require(
                artifact.get("trial_order") == "interleaved_alternating",
                f"{path}: paired v2 publication requires trial_order=interleaved_alternating",
            )
        require(
            isinstance(warmup, int) and warmup >= MIN_WARMUPS,
            f"{path}: v2 publication requires warmup >= {MIN_WARMUPS}",
        )
        require(
            isinstance(trials, int) and trials >= MIN_TRIALS,
            f"{path}: v2 publication requires trials >= {MIN_TRIALS}",
        )
        if schema in SCALE_SCHEMAS:
            require(
                artifact.get("status") == "complete",
                f"{path}: v2 ingest-scale artifact is not complete",
            )
            cooldown = artifact.get("cooldown_s")
            require(
                isinstance(cooldown, (int, float))
                and not isinstance(cooldown, bool)
                and float(cooldown) >= MIN_SCALE_COOLDOWN_SECONDS,
                f"{path}: v2 ingest-scale publication requires cooldown_s "
                f">= {MIN_SCALE_COOLDOWN_SECONDS:.0f}",
            )
            validate_benchmark_conditions(artifact, path=path)
    else:
        if isinstance(warmup, int) and warmup < MIN_WARMUPS:
            warnings.append(f"{path}: warmup={warmup} < {MIN_WARMUPS} (publication convention)")
        if isinstance(trials, int) and trials < MIN_TRIALS:
            warnings.append(f"{path}: trials={trials} < {MIN_TRIALS} (publication convention)")

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
