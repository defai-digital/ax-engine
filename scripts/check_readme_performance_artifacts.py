#!/usr/bin/env python3
"""Validate README performance tables against benchmark artifact provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ARTIFACT_LABELS = {
    "gemma-4-e2b-it-4bit": ("Gemma 4 E2B", "4-bit"),
    "gemma-4-e2b-it-6bit": ("Gemma 4 E2B", "6-bit"),
    "gemma-4-e4b-it-4bit": ("Gemma 4 E4B", "4-bit"),
    "gemma-4-e4b-it-6bit": ("Gemma 4 E4B", "6-bit"),
    "gemma-4-26b-a4b-it-4bit": ("Gemma 4 26B A4B", "4-bit"),
    "gemma-4-26b-a4b-it-6bit": ("Gemma 4 26B A4B", "6-bit"),
    "gemma-4-31b-it-4bit": ("Gemma 4 31B", "4-bit"),
    "gemma-4-31b-it-6bit": ("Gemma 4 31B", "6-bit"),
    "qwen3_5-9b-mlx-4bit": ("Qwen 3.5 9B", "4-bit"),
    "qwen3_6-27b-4bit": ("Qwen 3.6 27B", "4-bit"),
    "qwen3_6-27b-6bit": (
        "Qwen 3.6 27B",
        "6-bit",
    ),
    "qwen3_6-35b-a3b-4bit": ("Qwen 3.6 35B A3B", "4-bit"),
    "qwen3_6-35b-a3b-6bit": ("Qwen 3.6 35B A3B", "6-bit"),
    "qwen3-coder-next-4bit": ("Qwen Coder Next", "4-bit"),
    "glm-4.7-flash-4bit": ("GLM 4.7 Flash", "4-bit"),
}

DECODE_TABLE_COLUMNS = {
    "mlx_lm": "mlx_lm",
    "ax direct baseline": "ax_engine_mlx",
}

PREFILL_TABLE_COLUMNS = {
    "mlx_lm": "mlx_lm",
    "ax engine": "ax_engine_mlx",
}

TTFT_TABLE_COLUMNS = PREFILL_TABLE_COLUMNS

PHASE0_CLAIM_GATE_SCHEMA_VERSION = "ax.phase0_claim_gate.v1"
EVIDENCE_SET_SCHEMA_VERSION = "ax.benchmark_evidence_set.v1"
RUN_STABILITY_SCHEMA_VERSION = "ax.benchmark_run_stability.v1"
RUN_STABILITY_SUMMARY_SCHEMA_VERSION = "ax.benchmark_run_stability_summary.v1"
MLX_INFERENCE_STACK_SCHEMA_VERSION = "ax.mlx_inference_stack.v2"
AX_ONLY_REFRESH_SCHEMA_VERSION = "ax.ax_only_refresh.v1"
AX_ONLY_REFRESH_REGRESSION_SCHEMA_VERSION = "ax.ax_only_refresh_regression.v1"
AX_ONLY_REFRESH_DECODE_MIN_RATIO_TO_REFERENCE = 0.97
MTP_6BIT_APPROXIMATE_SCHEMA_VERSION = (
    "ax.mtp_6bit_approximate_diagnostic_summary.v2"
)
MTP_6BIT_EXACT_SCHEMA_VERSION = "ax.mtp_6bit_ax_acceleration_summary.v3"
MTP_6BIT_EXACT_TARGET_IDS = (
    "qwen3.6-27b-6bit",
    "qwen3.6-35b-a3b",
    "gemma-4-12b",
    "gemma-4-26b",
    "gemma-4-31b",
)
MTP_6BIT_EXACT_SUITES = ("flappy", "long_code", "python_modules_long")
MTP_6BIT_NGRAM_ZERO_KEYS = (
    "ax_ngram_accepted_tokens",
    "ax_ngram_draft_tokens",
    "ax_ngram_rejected_tokens",
    "ax_mtp_ngram_accepted_tokens",
    "ax_mtp_ngram_proposed_tokens",
    "ax_mtp_ngram_submitted_tokens",
    "ax_mtp_ngram_submitted_accepted_tokens",
    "ax_mtp_ngram_hit_steps",
    "ax_mtp_ngram_attempt_steps",
)
README_MAX_PUBLICATION_LOAD_AVERAGE = 2.0
README_MAX_PUBLICATION_TOP_PROCESS_CPU_PERCENT = 50.0
PREFIX_REUSE_EQUIVALENCE_SCHEMA_VERSION = "ax.prefix_reuse_equivalence.v1"
PREFILL_SCALING_SCHEMA_VERSION = "ax.mlx_prefill_scaling.v1"
CONCURRENT_PREFILL_SCHEMA_VERSION = "ax.mlx_concurrent_prefill.v1"
REUSED_REFERENCE_MIN_REPETITIONS = 3
MLX_LM_STYLE_PREFILL_WORK_CONTRACT = (
    "mlx_lm_style_cache_only_prefix_plus_final_prompt_token"
)
HISTORICAL_PREFILL_WORK_CONTRACT = "historical_full_logits_prefill_or_sampler_required"
LONG_CONTEXT_CAMPAIGN_BOUNDARY_SNIPPET = (
    "single-model long-context boundary, not a Gemma/Qwen/GLM-wide campaign"
)
LEGACY_MLX_INFERENCE_PATH_PREFIX = ("benchmarks", "results", "mlx-inference")
CATEGORIZED_MLX_INFERENCE_PATH_PREFIX = (
    "benchmarks",
    "results",
    "inference",
    "mlx-inference",
)

AX_NGRAM_TELEMETRY_COUNTERS = {
    "ax_ngram_draft_attempts",
    "ax_ngram_draft_tokens",
    "ax_ngram_accepted_tokens",
    "ax_ngram_rejected_tokens",
    "ax_ngram_full_accepts",
    "ax_ngram_partial_rejects",
    "ax_ngram_complete_misses",
    "ax_ngram_no_draft_steps",
    "ax_ngram_cooldown_steps",
    "ax_ngram_cooldown_events",
    "ax_ngram_cooldown_steps_scheduled",
    "ax_ngram_request_disable_events",
    "ax_ngram_request_disabled_steps",
    "ax_ngram_fallback_no_candidate_steps",
    "ax_ngram_fallback_confidence_filtered_steps",
    "ax_ngram_fallback_short_output_steps",
    "ax_ngram_fallback_linear_no_draft_steps",
    "ax_ngram_policy_variant",
    "ax_ngram_adaptive_draft_len_steps",
    "ax_ngram_adaptive_draft_len_total",
}

AX_DIRECT_CLAIM_STATUSES = {
    "direct_same_policy_baseline",
}

AX_DIRECT_ENGINE_KEYS = {
    "ax_engine_mlx",
    "ax_engine_mlx_linear_pack",
    "ax_engine_mlx_dense_ffn_pack",
    "ax_engine_mlx_direct_gemma4_ffn",
    "ax_engine_mlx_direct_linear_attention_inputs",
    "ax_engine_mlx_direct_linear_attention_post_input",
}

AX_MTP_ENGINE_KEYS = {
    "ax_engine_mlx_pure_mtp",
    "ax_engine_gemma4_assistant_mtp",
}

AX_MTP_CLAIM_STATUSES = {
    "mtp_head_only_effective",
}

AX_MTP_EFFECTIVE_ROUTES = {
    "mtp_head_only_verify_loop",
}

AX_OWNED_MLX_ROW_ENGINE_KEYS = AX_DIRECT_ENGINE_KEYS | {
    "mlx_lm",
    "ax_engine_mlx_ngram_accel",
} | AX_MTP_ENGINE_KEYS

AX_NGRAM_CLAIM_STATUSES = {
    "ngram_acceleration_effective_throughput",
    "ngram_no_draft_direct_fallback",
    "ngram_no_accept_fallback",
    "ngram_no_observed_draft_path",
}

AX_DECODE_EFFECTIVE_ROUTES = {
    "direct_pipeline_baseline",
    "direct_single_decode_baseline",
    "linear_no_draft_direct_pipeline_fallback",
    "linear_no_draft_mixed_fallback",
    "linear_no_draft_single_decode_fallback",
    "no_draft_fallback",
    "ngram_route_not_observed",
    "ngram_attempted_no_accept_fallback",
    "ngram_verified_bonus_tokens",
    "ngram_accepted_without_decode_route",
}

AX_NGRAM_NO_DRAFT_EFFECTIVE_ROUTES = {
    "linear_no_draft_direct_pipeline_fallback",
    "linear_no_draft_mixed_fallback",
    "linear_no_draft_single_decode_fallback",
    "no_draft_fallback",
}

AX_DIRECT_HOTPATH_FALLBACK_COUNTERS = {
    "ax_mlx_single_decode_steps": "single-decode fallback steps",
    "ax_mlx_ngram_decode_steps": "n-gram decode steps",
    "ax_mlx_dense_ffn_split_gate_up_layers": "dense FFN split gate/up fallback layers",
    "ax_mlx_direct_cpp_linear_attention_inputs_fallbacks": (
        "direct C++ linear-attention input fallback calls"
    ),
    "ax_mlx_direct_cpp_linear_attention_inputs_profile_blocked": (
        "direct C++ linear-attention input profile-blocked calls"
    ),
    "ax_mlx_direct_cpp_linear_attention_post_input_fallbacks": (
        "direct C++ linear-attention post-input fallback calls"
    ),
    "ax_mlx_direct_cpp_linear_attention_post_input_profile_blocked": (
        "direct C++ linear-attention post-input profile-blocked calls"
    ),
}

AX_DIRECT_CPP_LINEAR_ATTENTION_INPUT_COUNTERS = {
    "attempts": "ax_mlx_direct_cpp_linear_attention_inputs_attempts",
    "hits": "ax_mlx_direct_cpp_linear_attention_inputs_hits",
    "fallbacks": "ax_mlx_direct_cpp_linear_attention_inputs_fallbacks",
    "profile_blocked": "ax_mlx_direct_cpp_linear_attention_inputs_profile_blocked",
}

AX_DIRECT_CPP_LINEAR_ATTENTION_POST_INPUT_COUNTERS = {
    "attempts": "ax_mlx_direct_cpp_linear_attention_post_input_attempts",
    "hits": "ax_mlx_direct_cpp_linear_attention_post_input_hits",
    "fallbacks": "ax_mlx_direct_cpp_linear_attention_post_input_fallbacks",
    "profile_blocked": (
        "ax_mlx_direct_cpp_linear_attention_post_input_profile_blocked"
    ),
}

PUBLIC_CLAIM_EVIDENCE = {
    "continuous_batching": "concurrent_prefill_overlap_classification",
    "prefix_reuse": "prefix_reuse_evidence",
    "long_context_prefill_improvement": "long_context_prefill_evidence",
}
KNOWN_PUBLIC_CLAIMS = frozenset(PUBLIC_CLAIM_EVIDENCE)

PREFIX_REUSE_EVIDENCE_COUNTERS = {
    "hit_count",
    "miss_count",
    "blocked_count",
    "stored_prefix_count",
    "eviction_count",
    "reused_token_count",
    "warmup_token_count",
    "cache_entry_count",
    "cache_bytes_kib",
    "blocked_policy_disabled_count",
    "blocked_unsupported_layout_count",
    "blocked_trim_failure_count",
    "blocked_reason_count",
    "blocked_reason_accounting_gap_count",
}

PREFIX_REUSE_COVERAGE_VALUES = {
    "none_observed",
    "hit_only",
    "miss_warmup_only",
    "blocked_only",
    "hit_and_miss_warmup",
}
OVERLAP_CLASSIFICATION_VALUES = {
    "single_request_no_overlap",
    "serialized",
    "partial_overlap",
    "overlapped",
}
POSITIVE_CONTINUOUS_BATCHING_CLASSIFICATIONS = {
    "partial_overlap",
    "overlapped",
}


class ArtifactCheckError(RuntimeError):
    pass


@dataclass(frozen=True)
class ReadmeMetric:
    table: str
    model: str
    quantization: str
    prompt_tokens: int
    generation_tokens: int | None
    column: str
    engine: str
    displayed_value: float
    displayed_delta_percent: float | None


@dataclass(frozen=True)
class ArtifactRow:
    artifact_path: Path
    model: str
    quantization: str
    prompt_tokens: int
    generation_tokens: int
    engine: str
    row: dict[str, Any]


@dataclass(frozen=True)
class ArtifactSource:
    artifact_dir: Path
    include_engines: frozenset[str] | None = None
    include_tables: frozenset[str] | None = None
    include_prompt_tokens: frozenset[int] | None = None


@dataclass(frozen=True)
class ArtifactMarkerEntry:
    kind: str
    artifact_dir: Path
    include_prompt_tokens: frozenset[int] | None = None


@dataclass(frozen=True)
class ReadmeCheckResult:
    metric_checks: list[str]
    narrative_claim_checks: list[str]
    condition_metadata_checks: list[str]


@dataclass(frozen=True)
class ConditionMetadataGap:
    artifact_path: Path
    reason: str


README_METRIC_TABLES = frozenset({"decode", "prefill", "ttft"})
AX_ENGINE_ROWS = frozenset({"ax_engine_mlx", "ax_engine_mlx_ngram_accel"})
AX_DIRECT_ROW = frozenset({"ax_engine_mlx"})


def token_sha256(tokens: list[int]) -> str:
    payload = json.dumps(tokens, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def parse_numeric_cell(cell: str) -> float:
    cleaned = cell.replace("**", "").replace(",", "")
    match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
    if not match:
        raise ArtifactCheckError(f"metric cell has no numeric value: {cell!r}")
    return float(match.group(0))


def parse_percent_cell(cell: str) -> float | None:
    cleaned = cell.replace("**", "")
    match = re.search(r"\(([+-]?\d+(?:\.\d+)?)%\)", cleaned)
    if not match:
        return None
    return float(match.group(1))


def is_unavailable_cell(cell: str) -> bool:
    normalized = cell.strip().lower()
    return normalized in {"n/a", "na", "-", "—"} or normalized.startswith("— ")


def split_markdown_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def extract_table_lines(readme_text: str, heading_prefix: str) -> list[str]:
    lines = readme_text.splitlines()
    start = None
    for index, line in enumerate(lines):
        if line.startswith(heading_prefix):
            start = index + 1
            break
    if start is None:
        raise ArtifactCheckError(f"missing README section: {heading_prefix}")

    table_lines: list[str] = []
    in_table = False
    for line in lines[start:]:
        if line.startswith("|"):
            in_table = True
            table_lines.append(line)
            continue
        if in_table:
            break
    if len(table_lines) < 3:
        raise ArtifactCheckError(f"missing markdown table under: {heading_prefix}")
    return table_lines


def parse_heading_generation_tokens(heading: str) -> int | None:
    match = re.search(r"generation\s*=\s*(\d+)\s*tokens", heading)
    if not match:
        return None
    return int(match.group(1))


def find_heading_line(readme_text: str, heading_prefix: str) -> str:
    for line in readme_text.splitlines():
        if line.startswith(heading_prefix):
            return line
    raise ArtifactCheckError(f"missing README section: {heading_prefix}")


def parse_readme_table(
    readme_text: str,
    *,
    heading_prefix: str,
    table_name: str,
    column_map: dict[str, str],
) -> list[ReadmeMetric]:
    heading = find_heading_line(readme_text, heading_prefix)
    generation_tokens = parse_heading_generation_tokens(heading)
    table_lines = extract_table_lines(readme_text, heading_prefix)
    headers = split_markdown_row(table_lines[0])
    normalized_headers = [header.lower() for header in headers]
    rows: list[ReadmeMetric] = []
    current_model = ""
    current_quantization = ""

    for line in table_lines[2:]:
        cells = split_markdown_row(line)
        if len(cells) != len(headers):
            raise ArtifactCheckError(f"unexpected README table row shape: {line}")
        if cells[0]:
            current_model = cells[0]
        if cells[1]:
            current_quantization = cells[1]
        if not current_model or not current_quantization:
            raise ArtifactCheckError(f"README row is missing model context: {line}")
        prompt_tokens = int(parse_numeric_cell(cells[2]))
        for header, cell in zip(normalized_headers[3:], cells[3:]):
            engine = column_map.get(header)
            if engine is None:
                continue
            if is_unavailable_cell(cell):
                continue
            rows.append(
                ReadmeMetric(
                    table=table_name,
                    model=current_model,
                    quantization=current_quantization,
                    prompt_tokens=prompt_tokens,
                    generation_tokens=generation_tokens,
                    column=headers[normalized_headers.index(header)],
                    engine=engine,
                    displayed_value=parse_numeric_cell(cell),
                    displayed_delta_percent=parse_percent_cell(cell),
                )
            )
    return rows


def parse_readme_metrics(readme_path: Path) -> list[ReadmeMetric]:
    text = readme_path.read_text()
    return [
        *parse_readme_table(
            text,
            heading_prefix="#### Decode throughput",
            table_name="decode",
            column_map=DECODE_TABLE_COLUMNS,
        ),
        *parse_readme_table(
            text,
            heading_prefix="#### Prefill throughput",
            table_name="prefill",
            column_map=PREFILL_TABLE_COLUMNS,
        ),
        *parse_readme_table(
            text,
            heading_prefix="#### Time to first token",
            table_name="ttft",
            column_map=TTFT_TABLE_COLUMNS,
        ),
    ]


def repo_root_for(path: Path) -> Path:
    """Locate the workspace root for a results doc under the repo or docs/."""
    for parent in (path if path.is_dir() else path.parent, *path.parents):
        if (parent / "Cargo.toml").is_file() and (parent / "crates").is_dir():
            return parent
    return path.parent if path.is_file() else path


def resolve_results_doc_path(results_doc: Path, path_value: str) -> Path:
    """Resolve a marker or link path from a results doc.

    Paths may be relative to the doc file (legacy root README) or to the
    workspace root (docs/PERFORMANCE-RESULTS.md with benchmarks/... links).
    """
    raw = Path(path_value)
    if raw.is_absolute():
        return raw.resolve()
    relative_to_doc = (results_doc.parent / path_value).resolve()
    if relative_to_doc.exists():
        return relative_to_doc
    if path_value.startswith(("benchmarks/", "docs/", "assets/")):
        return (repo_root_for(results_doc) / path_value).resolve()
    return relative_to_doc


def readme_performance_artifact_marker_entries(
    readme_path: Path,
) -> list[ArtifactMarkerEntry]:
    text = readme_path.read_text()
    sources_match = re.search(
        r"<!--\s*readme-performance-artifacts:\s*(?P<sources>.*?)\s*-->",
        text,
        flags=re.DOTALL,
    )
    if sources_match is None:
        return []
    entries: list[ArtifactMarkerEntry] = []
    source_parts = [
        part.strip() for part in sources_match.group("sources").split(";") if part.strip()
    ]
    for part in source_parts:
        if "=" not in part:
            raise ArtifactCheckError(
                f"invalid readme-performance-artifacts entry: {part.strip()!r}"
            )
        raw_kind, path_value = [value.strip() for value in part.split("=", 1)]
        kind_match = re.fullmatch(
            r"(?P<kind>[a-z0-9-]+)(?:@p(?P<prompt>\d+))?",
            raw_kind,
        )
        if not kind_match:
            raise ArtifactCheckError(
                f"invalid readme-performance-artifacts source kind: {raw_kind!r}"
            )
        include_prompt_tokens = None
        if kind_match.group("prompt") is not None:
            include_prompt_tokens = frozenset({int(kind_match.group("prompt"))})
        entries.append(
            ArtifactMarkerEntry(
                kind=kind_match.group("kind"),
                artifact_dir=resolve_results_doc_path(readme_path, path_value),
                include_prompt_tokens=include_prompt_tokens,
            )
        )
    return entries


def default_artifact_sources(readme_path: Path) -> list[ArtifactSource]:
    entries = readme_performance_artifact_marker_entries(readme_path)
    if entries:
        sources: list[ArtifactSource] = []
        source_kinds = {
            "base": (None, None),
            "reference": (frozenset({"mlx_lm"}), None),
            "ax": (AX_ENGINE_ROWS, None),
            "ax-base": (AX_ENGINE_ROWS, None),
            "ax-overlay": (AX_ENGINE_ROWS, None),
            "ax-decode-overlay": (AX_ENGINE_ROWS, frozenset({"decode"})),
        }
        for entry in entries:
            kind = entry.kind
            if kind not in source_kinds:
                raise ArtifactCheckError(
                    f"unknown readme-performance-artifacts source kind: {kind!r}"
                )
            if kind == "base" and len(entries) > 1:
                raise ArtifactCheckError(
                    "readme-performance-artifacts base= is only allowed as a "
                    "single legacy source; composite markers must use scoped "
                    "reference=, ax-base=, ax-overlay=, or ax-decode-overlay= sources"
                )
            include_engines, include_tables = source_kinds[kind]
            sources.append(
                ArtifactSource(
                    artifact_dir=entry.artifact_dir,
                    include_engines=include_engines,
                    include_tables=include_tables,
                    include_prompt_tokens=entry.include_prompt_tokens,
                )
            )
        if not sources:
            raise ArtifactCheckError(
                "readme-performance-artifacts comment has no sources"
        )
        return sources

    text = readme_path.read_text()
    match = re.search(
        r"`(benchmarks/results/(?:inference/)?mlx-inference/[^`]+/)`",
        text,
    )
    if not match:
        raise ArtifactCheckError(
            "README does not name a benchmarks/results/inference/mlx-inference artifact directory"
        )
    return [ArtifactSource(resolve_results_doc_path(readme_path, match.group(1)))]


def default_hot_prefix_artifact_paths(readme_path: Path) -> list[Path]:
    return default_marker_artifact_paths(
        readme_path,
        marker_name="readme-hot-prefix-artifact",
    )


def default_long_context_boundary_artifact_paths(readme_path: Path) -> list[Path]:
    return default_marker_artifact_paths(
        readme_path,
        marker_name="readme-long-context-boundary-artifact",
    )


def default_concurrent_prefill_boundary_artifact_paths(readme_path: Path) -> list[Path]:
    return default_marker_artifact_paths(
        readme_path,
        marker_name="readme-concurrent-prefill-boundary-artifact",
    )


def default_marker_artifact_paths(readme_path: Path, *, marker_name: str) -> list[Path]:
    text = readme_path.read_text()
    artifacts_match = re.search(
        rf"<!--\s*{re.escape(marker_name)}:\s*(?P<paths>.*?)\s*-->",
        text,
        flags=re.DOTALL,
    )
    if artifacts_match is None:
        return []
    paths: list[Path] = []
    for path_value in artifacts_match.group("paths").split(";"):
        path_value = path_value.strip()
        if not path_value:
            continue
        paths.append(resolve_results_doc_path(readme_path, path_value))
    if not paths:
        raise ArtifactCheckError(f"{marker_name} comment has no paths")
    return paths


def metric_median(row: dict[str, Any], table: str) -> float:
    if table == "ttft":
        metric_key = "ttft_ms"
    else:
        metric_key = "decode_tok_s" if table == "decode" else "prefill_tok_s"
    metric = row.get(metric_key)
    if not isinstance(metric, dict) or metric.get("median") is None:
        raise ArtifactCheckError(f"artifact row lacks {metric_key}.median")
    return float(metric["median"])


def has_metric_summary(row: dict[str, Any], key: str) -> bool:
    metric = row.get(key)
    return isinstance(metric, dict) and "median" in metric


def phase0_claim_gate_enabled(artifact: dict[str, Any]) -> bool:
    gate = artifact.get("claim_gate")
    return (
        isinstance(gate, dict)
        and gate.get("schema_version") == PHASE0_CLAIM_GATE_SCHEMA_VERSION
    )


def validate_claim_gate_methodology(
    *, artifact_path: Path, artifact: dict[str, Any]
) -> None:
    gate = artifact.get("claim_gate")
    if not isinstance(gate, dict):
        return
    minimum_warmups = gate.get("minimum_warmup_repetitions")
    minimum_measurements = gate.get("minimum_measurement_repetitions")
    requirements = (
        ("warmup_repetitions", "minimum_warmup_repetitions", minimum_warmups, 2),
        (
            "repetitions",
            "minimum_measurement_repetitions",
            minimum_measurements,
            5,
        ),
    )
    for artifact_key, gate_key, minimum, floor in requirements:
        if minimum is None:
            continue
        if not isinstance(minimum, int) or isinstance(minimum, bool) or minimum < floor:
            raise ArtifactCheckError(
                f"{artifact_path} claim_gate.{gate_key} must be at least {floor}"
            )
        observed = artifact.get(artifact_key)
        if (
            not isinstance(observed, int)
            or isinstance(observed, bool)
            or observed < minimum
        ):
            raise ArtifactCheckError(
                f"{artifact_path} {artifact_key} must be at least {minimum} for publication"
            )
    if gate.get("requires_clean_build_commit") is True:
        build = artifact.get("build")
        if not isinstance(build, dict):
            raise ArtifactCheckError(
                f"{artifact_path} requires build provenance for publication"
            )
        commit = build.get("commit")
        if not isinstance(commit, str) or not commit.strip():
            raise ArtifactCheckError(
                f"{artifact_path} requires a non-empty build.commit for publication"
            )
        if build.get("git_tracked_dirty") is not False:
            raise ArtifactCheckError(
                f"{artifact_path} requires build.git_tracked_dirty=false for publication"
            )


def validate_evidence_set_manifest(manifest_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != EVIDENCE_SET_SCHEMA_VERSION:
        raise ArtifactCheckError(f"{manifest_path} has unexpected schema_version")
    if manifest.get("status") != "superseded":
        raise ArtifactCheckError(f"{manifest_path} must remain marked superseded")
    if manifest.get("publication_candidate") is not False:
        raise ArtifactCheckError(
            f"{manifest_path} superseded evidence must set publication_candidate=false"
        )
    superseded_by = manifest.get("superseded_by_commits")
    if not isinstance(superseded_by, list) or not all(
        isinstance(commit, str) and commit for commit in superseded_by
    ):
        raise ArtifactCheckError(
            f"{manifest_path} must identify the commits that superseded the evidence"
        )

    aggregate_paths = sorted(
        path
        for path in manifest_path.parent.rglob("ax_*.json")
        if not any(part.endswith("-prompts") for part in path.parts)
    )
    completed: list[dict[str, Any]] = []
    incomplete: list[dict[str, Any]] = []
    for aggregate_path in aggregate_paths:
        aggregate = json.loads(aggregate_path.read_text())
        if aggregate.get("schema_version") == MLX_INFERENCE_STACK_SCHEMA_VERSION:
            completed.append(aggregate)
        elif aggregate.get("schema_version") == "ax.mlx_inference_stack.incomplete.v1":
            if aggregate.get("publication_candidate") is not False:
                raise ArtifactCheckError(
                    f"{aggregate_path} incomplete evidence must not be publishable"
                )
            incomplete.append(aggregate)
        else:
            raise ArtifactCheckError(f"{aggregate_path} has unexpected schema_version")

    expected_completed = manifest.get("expected_completed_aggregates")
    expected_incomplete = manifest.get("expected_incomplete_aggregates")
    if len(completed) != expected_completed or len(incomplete) != expected_incomplete:
        raise ArtifactCheckError(
            f"{manifest_path} aggregate counts do not match the evidence set"
        )
    source_commits = sorted(
        {
            str(aggregate.get("build", {}).get("commit", ""))
            for aggregate in completed
        }
    )
    if source_commits != sorted(manifest.get("source_build_commits") or []):
        raise ArtifactCheckError(
            f"{manifest_path} source_build_commits do not match completed aggregates"
        )
    methodology = manifest.get("methodology")
    if not isinstance(methodology, dict):
        raise ArtifactCheckError(f"{manifest_path} lacks methodology")
    warmups = {aggregate.get("warmup_repetitions") for aggregate in completed}
    repetitions = {aggregate.get("repetitions") for aggregate in completed}
    if warmups != {methodology.get("warmup_repetitions")}:
        raise ArtifactCheckError(
            f"{manifest_path} warmup methodology does not match completed aggregates"
        )
    if repetitions != {methodology.get("measurement_repetitions")}:
        raise ArtifactCheckError(
            f"{manifest_path} measurement methodology does not match completed aggregates"
        )


def public_claim_names(artifact: dict[str, Any]) -> set[str]:
    claims = artifact.get("public_claims") or []
    names: set[str] = set()
    if not isinstance(claims, list):
        raise ArtifactCheckError("public_claims must be a list when present")
    for claim in claims:
        if isinstance(claim, str):
            name = claim
        elif isinstance(claim, dict) and isinstance(claim.get("name"), str):
            name = str(claim["name"])
        else:
            raise ArtifactCheckError(f"invalid public claim entry: {claim!r}")
        if name not in KNOWN_PUBLIC_CLAIMS:
            known = ", ".join(sorted(KNOWN_PUBLIC_CLAIMS))
            raise ArtifactCheckError(
                f"unknown public claim {name!r}; known claims: {known}"
            )
        names.add(name)
    return names


def validate_public_claim_evidence(
    *, artifact_path: Path, artifact: dict[str, Any]
) -> None:
    for claim, evidence_key in PUBLIC_CLAIM_EVIDENCE.items():
        if claim not in public_claim_names(artifact):
            continue
        evidence = artifact.get(evidence_key)
        if not isinstance(evidence, dict):
            raise ArtifactCheckError(
                f"{artifact_path} claims {claim} without {evidence_key}"
            )
        if claim == "continuous_batching":
            validate_concurrent_prefill_overlap_classification(
                artifact_path=artifact_path,
                evidence=evidence,
            )
            if not evidence.get("continuous_batching_claim"):
                raise ArtifactCheckError(
                    f"{artifact_path} claims continuous_batching without positive overlap evidence"
                )
        elif claim == "prefix_reuse":
            validate_prefix_reuse_evidence_shape(
                artifact_path=artifact_path,
                evidence=evidence,
            )
            if not evidence.get("physical_snapshot_hit_observed"):
                raise ArtifactCheckError(
                    f"{artifact_path} claims prefix_reuse without physical snapshot hit evidence"
                )


def validate_build_provenance(*, artifact_path: Path, artifact: dict[str, Any]) -> None:
    build = artifact.get("build")
    if not isinstance(build, dict):
        return
    if build.get("git_tracked_dirty") is True:
        if build.get("git_tracked_dirty_accepted") is True:
            return
        status = build.get("git_tracked_status")
        suffix = ""
        if isinstance(status, list) and status:
            suffix = f"; tracked changes include: {', '.join(map(str, status[:5]))}"
        raise ArtifactCheckError(
            f"{artifact_path} was produced from a tracked-dirty source tree{suffix}"
        )


def validate_host_performance_conditions(
    *, artifact_path: Path, artifact: dict[str, Any]
) -> None:
    host = artifact.get("host")
    if not isinstance(host, dict):
        return
    conditions = host.get("performance_conditions")
    if conditions is None:
        return
    if not isinstance(conditions, dict):
        raise ArtifactCheckError(
            f"{artifact_path} host.performance_conditions must be an object"
        )
    for key in (
        "thermal_warning_recorded",
        "performance_warning_recorded",
        "cpu_power_status_recorded",
    ):
        if key in conditions and not isinstance(conditions[key], bool):
            raise ArtifactCheckError(
                f"{artifact_path} host.performance_conditions.{key} must be boolean"
            )
    if "thermal_status_lines" in conditions:
        lines = conditions["thermal_status_lines"]
        if not isinstance(lines, list) or not all(
            isinstance(line, str) for line in lines
        ):
            raise ArtifactCheckError(
                f"{artifact_path} host.performance_conditions.thermal_status_lines "
                "must be a string list"
            )
    if "load_average" in conditions:
        load_average = conditions["load_average"]
        if not isinstance(load_average, dict):
            raise ArtifactCheckError(
                f"{artifact_path} host.performance_conditions.load_average "
                "must be an object"
            )
        for key in ("one_minute", "five_minutes", "fifteen_minutes"):
            value = load_average.get(key)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ArtifactCheckError(
                    f"{artifact_path} host.performance_conditions.load_average."
                    f"{key} must be numeric"
                )
        one_minute = float(load_average["one_minute"])
        if one_minute > README_MAX_PUBLICATION_LOAD_AVERAGE:
            raise ArtifactCheckError(
                f"{artifact_path} host.performance_conditions.load_average."
                f"one_minute {one_minute:.3f} exceeds publication limit "
                f"{README_MAX_PUBLICATION_LOAD_AVERAGE:.3f}"
            )
    if "top_processes_cpu" in conditions:
        processes = conditions["top_processes_cpu"]
        if not isinstance(processes, list):
            raise ArtifactCheckError(
                f"{artifact_path} host.performance_conditions.top_processes_cpu "
                "must be a list"
            )
        for process in processes:
            if not isinstance(process, dict):
                raise ArtifactCheckError(
                    f"{artifact_path} host.performance_conditions."
                    "top_processes_cpu entries must be objects"
                )
            cpu_percent = process.get("cpu_percent")
            if not isinstance(cpu_percent, (int, float)) or isinstance(
                cpu_percent, bool
            ):
                raise ArtifactCheckError(
                    f"{artifact_path} host.performance_conditions."
                    "top_processes_cpu.cpu_percent must be numeric"
                )
            if cpu_percent > README_MAX_PUBLICATION_TOP_PROCESS_CPU_PERCENT:
                command = process.get("command")
                if not isinstance(command, str) or not command:
                    command = "unknown process"
                raise ArtifactCheckError(
                    f"{artifact_path} host.performance_conditions."
                    f"top_processes_cpu {command} at {cpu_percent:.1f}% exceeds "
                    "publication limit "
                    f"{README_MAX_PUBLICATION_TOP_PROCESS_CPU_PERCENT:.1f}%"
                )


def validate_benchmark_window(*, artifact_path: Path, artifact: dict[str, Any]) -> None:
    window = artifact.get("benchmark_window")
    if window is None:
        return
    if not isinstance(window, dict):
        raise ArtifactCheckError(f"{artifact_path} benchmark_window must be an object")
    for key in ("started_at", "finished_at"):
        value = window.get(key)
        if not isinstance(value, str) or not value:
            raise ArtifactCheckError(
                f"{artifact_path} benchmark_window.{key} must be a non-empty string"
            )
    elapsed = window.get("elapsed_seconds")
    if not isinstance(elapsed, (int, float)) or isinstance(elapsed, bool):
        raise ArtifactCheckError(
            f"{artifact_path} benchmark_window.elapsed_seconds must be numeric"
        )
    if elapsed < 0:
        raise ArtifactCheckError(
            f"{artifact_path} benchmark_window.elapsed_seconds must be non-negative"
        )
    for key in ("performance_conditions_start", "performance_conditions_end"):
        conditions = window.get(key)
        if conditions is None:
            continue
        validate_host_performance_conditions(
            artifact_path=artifact_path,
            artifact={"host": {"performance_conditions": conditions}},
        )


def artifact_condition_metadata_gaps(
    *, artifact_path: Path, artifact: dict[str, Any]
) -> list[ConditionMetadataGap]:
    gaps: list[ConditionMetadataGap] = []
    host = artifact.get("host")
    if not isinstance(host, dict) or not isinstance(
        host.get("performance_conditions"),
        dict,
    ):
        gaps.append(
            ConditionMetadataGap(
                artifact_path=artifact_path,
                reason="missing host.performance_conditions",
            )
        )
    window = artifact.get("benchmark_window")
    if not isinstance(window, dict):
        gaps.append(
            ConditionMetadataGap(
                artifact_path=artifact_path,
                reason="missing benchmark_window",
            )
        )
        return gaps
    for key in ("performance_conditions_start", "performance_conditions_end"):
        if not isinstance(window.get(key), dict):
            gaps.append(
                ConditionMetadataGap(
                    artifact_path=artifact_path,
                    reason=f"missing benchmark_window.{key}",
                )
            )
    return gaps


def collect_condition_metadata_gaps(
    *,
    artifact_sources: list[ArtifactSource],
    needed_labels: frozenset[tuple[str, str]] | None = None,
) -> list[ConditionMetadataGap]:
    _, gaps = collect_condition_metadata_checks(
        artifact_sources=artifact_sources,
        needed_labels=needed_labels,
    )
    return gaps


def collect_condition_metadata_checks(
    *,
    artifact_sources: list[ArtifactSource],
    needed_labels: frozenset[tuple[str, str]] | None = None,
) -> tuple[list[str], list[ConditionMetadataGap]]:
    checks: list[str] = []
    gaps: list[ConditionMetadataGap] = []
    for source in artifact_sources:
        if not source.artifact_dir.exists():
            continue
        for path in sorted(source.artifact_dir.glob("*.json")):
            label = ARTIFACT_LABELS.get(path.stem)
            if label is None:
                continue
            if needed_labels is not None and label not in needed_labels:
                continue
            artifact = json.loads(path.read_text())
            if artifact.get("schema_version") != MLX_INFERENCE_STACK_SCHEMA_VERSION:
                continue
            artifact_gaps = artifact_condition_metadata_gaps(
                artifact_path=path,
                artifact=artifact,
            )
            gaps.extend(artifact_gaps)
            if not artifact_gaps:
                checks.append(f"{source.artifact_dir}:{path.name}")
    return checks, gaps


def assert_condition_metadata_complete(gaps: list[ConditionMetadataGap]) -> None:
    if not gaps:
        return
    reasons_by_path: dict[Path, list[str]] = {}
    for gap in gaps:
        reasons_by_path.setdefault(gap.artifact_path, []).append(gap.reason)
    sample = "; ".join(
        f"{path}: {', '.join(reasons)}"
        for path, reasons in list(reasons_by_path.items())[:5]
    )
    extra = (
        ""
        if len(reasons_by_path) <= 5
        else f"; and {len(reasons_by_path) - 5} more artifact(s)"
    )
    raise ArtifactCheckError(
        f"{len(reasons_by_path)} README performance artifact(s) lack "
        "condition metadata: "
        f"{sample}{extra}"
    )


def tracked_dirty_is_benchmark_doc_only(status: Any) -> bool:
    if not isinstance(status, list) or not status:
        return False
    return all(is_benchmark_doc_only_status_line(line) for line in status)


def is_benchmark_doc_only_status_line(line: Any) -> bool:
    if not isinstance(line, str) or len(line) < 4:
        return False
    status_code = line[:2]
    # Allow M (modify) and D (delete); reject A (add), R (rename), C (copy),
    # U (unmerged), ? (untracked), and others that indicate structural changes.
    if any(code not in {" ", "M", "D"} for code in status_code):
        return False
    if not any(code in {"M", "D"} for code in status_code):
        return False
    path = line[3:]
    if " -> " in path:
        return False
    return is_benchmark_doc_only_path(path)


# Paths that cannot affect bench-output JSON for the artifact classes this
# checker validates. README + box-whisker SVGs are pure docs. The
# `update_readme_*.py` family only consumes bench JSON to rewrite README cells;
# none of them produce or orchestrate a bench run. `test_*.py` files exercise
# other scripts and never run a bench themselves. `bench_llama_cpp_metal_sweep.py`
# only orchestrates llama.cpp Metal full-stack runs and is not invoked by the
# AX-only / mlx_lm-only paths whose artifacts feed README rows directly.
# `benchmarks/results/` JSON files are bench output artifacts; modifying or
# deleting a sibling result cannot affect a separate AX/mlx_lm bench run.
# `pyproject.toml` records the Python package version in artifact metadata
# but does not influence bench execution (the package is already installed).
BENCHMARK_DOC_ONLY_SCRIPT_PREFIXES = (
    "scripts/update_readme_",
    "scripts/test_",
)
BENCHMARK_DOC_ONLY_SCRIPT_PATHS = frozenset({"scripts/bench_llama_cpp_metal_sweep.py"})


def is_benchmark_doc_only_path(path: str) -> bool:
    if path in {
        "README.md",
        "docs/PERFORMANCE-RESULTS.md",
        "docs/PERFORMANCE.md",
        "pyproject.toml",
    }:
        return True
    if path.startswith("docs/assets/perf-") and path.endswith(".svg"):
        return True
    if path.startswith("benchmarks/results/") and path.endswith(".json"):
        return True
    if path in BENCHMARK_DOC_ONLY_SCRIPT_PATHS:
        return True
    return path.endswith(".py") and any(
        path.startswith(prefix) for prefix in BENCHMARK_DOC_ONLY_SCRIPT_PREFIXES
    )


@dataclass(frozen=True)
class HotPrefixClaimSummary:
    artifact_path: Path
    prompts_matching: int
    prompts_total: int
    hit_count: int
    reused_token_count: int
    warmup_token_count: int
    miss_count: int
    blocked_count: int


def validate_hot_prefix_equivalence_artifact(
    *, artifact_path: Path
) -> HotPrefixClaimSummary:
    if not artifact_path.exists():
        raise ArtifactCheckError(f"hot-prefix artifact does not exist: {artifact_path}")
    artifact = json.loads(artifact_path.read_text())
    if artifact.get("schema_version") != PREFIX_REUSE_EQUIVALENCE_SCHEMA_VERSION:
        raise ArtifactCheckError(f"{artifact_path} has unexpected schema_version")
    config = artifact.get("config")
    if not isinstance(config, dict) or config.get("mode") != "warm_repeat":
        raise ArtifactCheckError(f"{artifact_path} is not a warm_repeat artifact")
    aggregate = artifact.get("aggregate")
    if not isinstance(aggregate, dict) or aggregate.get("verdict") != "PASS":
        raise ArtifactCheckError(f"{artifact_path} hot-prefix verdict is not PASS")
    prompts_matching = int(aggregate.get("prompts_matching_exactly", -1))
    prompts_total = int(aggregate.get("prompts_total", -1))
    if prompts_total <= 0 or prompts_matching != prompts_total:
        raise ArtifactCheckError(f"{artifact_path} does not have token-exact parity")
    per_prompt = artifact.get("per_prompt")
    if not isinstance(per_prompt, list) or len(per_prompt) != prompts_total:
        raise ArtifactCheckError(
            f"{artifact_path} per_prompt count does not match aggregate"
        )

    hit_count = 0
    reused_token_count = 0
    warmup_token_count = 0
    miss_count = 0
    blocked_count = 0
    for item in per_prompt:
        if not isinstance(item, dict) or item.get("tokens_match") is not True:
            raise ArtifactCheckError(f"{artifact_path} has a non-matching prompt")
        telemetry = item.get("warm_telemetry")
        if not isinstance(telemetry, dict):
            raise ArtifactCheckError(f"{artifact_path} prompt lacks warm telemetry")
        hit_count += int(telemetry.get("ax_mlx_prefix_cache_hits", 0))
        reused_token_count += int(telemetry.get("ax_mlx_prefix_cache_reused_tokens", 0))
        warmup_token_count += int(telemetry.get("ax_mlx_prefix_cache_warmup_tokens", 0))
        miss_count += int(telemetry.get("ax_mlx_prefix_cache_misses", 0))
        blocked_count += int(telemetry.get("ax_mlx_prefix_cache_blocked", 0))

    if hit_count <= 0 or reused_token_count <= 0:
        raise ArtifactCheckError(
            f"{artifact_path} lacks physical hot-prefix hit evidence"
        )
    if warmup_token_count != 0 or miss_count != 0 or blocked_count != 0:
        raise ArtifactCheckError(
            f"{artifact_path} hot-prefix claim is not hit-only physical reuse"
        )
    return HotPrefixClaimSummary(
        artifact_path=artifact_path,
        prompts_matching=prompts_matching,
        prompts_total=prompts_total,
        hit_count=hit_count,
        reused_token_count=reused_token_count,
        warmup_token_count=warmup_token_count,
        miss_count=miss_count,
        blocked_count=blocked_count,
    )


def validate_readme_hot_prefix_claim_text(
    *, readme_text: str, summary: HotPrefixClaimSummary
) -> None:
    expected_snippets = {
        f"{summary.prompts_matching}/{summary.prompts_total} prompts": "prompt match count",
        f"reused {summary.reused_token_count} tokens": "reused token count",
        f"{summary.warmup_token_count} warmup": "warmup token count",
    }
    for snippet, label in expected_snippets.items():
        if snippet not in readme_text:
            raise ArtifactCheckError(
                f"README hot-prefix claim has stale {label}; expected {snippet!r}"
            )


def validate_readme_hot_prefix_claims(
    *, readme_path: Path, artifact_paths: list[Path]
) -> list[str]:
    if not artifact_paths:
        return []
    readme_text = readme_path.read_text()
    checked: list[str] = []
    for artifact_path in artifact_paths:
        summary = validate_hot_prefix_equivalence_artifact(artifact_path=artifact_path)
        validate_readme_hot_prefix_claim_text(
            readme_text=readme_text,
            summary=summary,
        )
        checked.append(
            "hot-prefix:"
            f"{artifact_path.name}:"
            f"{summary.prompts_matching}/{summary.prompts_total}:"
            f"{summary.reused_token_count}"
        )
    return checked


@dataclass(frozen=True)
class LongContextBoundarySummary:
    artifact_path: Path
    context_tokens: int
    prefill_ratio: float


def validate_long_context_boundary_artifact(
    *, artifact_path: Path
) -> LongContextBoundarySummary:
    if not artifact_path.exists():
        raise ArtifactCheckError(
            f"long-context artifact does not exist: {artifact_path}"
        )
    artifact = json.loads(artifact_path.read_text())
    if artifact.get("schema_version") != PREFILL_SCALING_SCHEMA_VERSION:
        raise ArtifactCheckError(f"{artifact_path} has unexpected schema_version")
    rows = artifact.get("rows")
    if not isinstance(rows, list):
        raise ArtifactCheckError(f"{artifact_path} lacks rows")
    candidates = [
        row
        for row in rows
        if isinstance(row, dict)
        and row.get("engine") == "ax_engine_mlx"
        and row.get("context_tokens") == 8192
    ]
    if len(candidates) != 1:
        raise ArtifactCheckError(f"{artifact_path} must have one AX 8k context row")
    ratios = candidates[0].get("ratios_to_mlx_lm")
    if not isinstance(ratios, dict) or not isinstance(
        ratios.get("prefill_tok_s"), (int, float)
    ):
        raise ArtifactCheckError(f"{artifact_path} AX 8k row lacks prefill ratio")
    return LongContextBoundarySummary(
        artifact_path=artifact_path,
        context_tokens=8192,
        prefill_ratio=float(ratios["prefill_tok_s"]),
    )


@dataclass(frozen=True)
class ConcurrentPrefillBoundarySummary:
    artifact_path: Path
    concurrent_requests: int
    classification: str


def validate_concurrent_prefill_boundary_artifact(
    *, artifact_path: Path
) -> ConcurrentPrefillBoundarySummary:
    if not artifact_path.exists():
        raise ArtifactCheckError(
            f"concurrent-prefill artifact does not exist: {artifact_path}"
        )
    artifact = json.loads(artifact_path.read_text())
    if artifact.get("schema_version") != CONCURRENT_PREFILL_SCHEMA_VERSION:
        raise ArtifactCheckError(f"{artifact_path} has unexpected schema_version")
    rows = artifact.get("rows")
    if not isinstance(rows, list):
        raise ArtifactCheckError(f"{artifact_path} lacks rows")
    candidates = [
        row
        for row in rows
        if isinstance(row, dict)
        and row.get("engine") == "ax_engine_mlx"
        and row.get("concurrent_requests") == 4
    ]
    if len(candidates) != 1:
        raise ArtifactCheckError(f"{artifact_path} must have one AX 4-request row")
    overlap = candidates[0].get("prefill_overlap")
    if not isinstance(overlap, dict) or not isinstance(
        overlap.get("classification"), str
    ):
        raise ArtifactCheckError(
            f"{artifact_path} AX 4-request row lacks classification"
        )
    return ConcurrentPrefillBoundarySummary(
        artifact_path=artifact_path,
        concurrent_requests=4,
        classification=overlap["classification"],
    )


def normalized_markdown_text(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def validate_readme_boundary_claims(
    *,
    readme_path: Path,
    long_context_artifact_paths: list[Path],
    concurrent_prefill_artifact_paths: list[Path],
) -> list[str]:
    readme_text = readme_path.read_text()
    normalized = normalized_markdown_text(readme_text)
    checked: list[str] = []
    for artifact_path in long_context_artifact_paths:
        summary = validate_long_context_boundary_artifact(artifact_path=artifact_path)
        snippet = f"The 8k P1 AX/MLX prefill ratio was {summary.prefill_ratio:.3f}x"
        if snippet not in normalized:
            raise ArtifactCheckError(
                f"README long-context boundary claim is stale; expected {snippet!r}"
            )
        if LONG_CONTEXT_CAMPAIGN_BOUNDARY_SNIPPET not in normalized:
            raise ArtifactCheckError(
                "README long-context boundary must not imply a family-wide campaign; "
                f"expected {LONG_CONTEXT_CAMPAIGN_BOUNDARY_SNIPPET!r}"
            )
        checked.append(
            "long-context-boundary:"
            f"{artifact_path.name}:"
            f"{summary.context_tokens}:"
            f"{summary.prefill_ratio:.3f}"
        )
    for artifact_path in concurrent_prefill_artifact_paths:
        summary = validate_concurrent_prefill_boundary_artifact(
            artifact_path=artifact_path,
        )
        snippet = (
            f"the {summary.concurrent_requests}-request P2 concurrent prefill row "
            f"was classified as {summary.classification}"
        )
        if snippet not in normalized:
            raise ArtifactCheckError(
                f"README concurrent-prefill boundary claim is stale; expected {snippet!r}"
            )
        checked.append(
            "concurrent-prefill-boundary:"
            f"{artifact_path.name}:"
            f"{summary.concurrent_requests}:"
            f"{summary.classification}"
        )
    return checked


def extract_readme_section(
    readme_text: str, *, heading_prefix: str
) -> tuple[str, str] | None:
    lines = readme_text.splitlines()
    start = next(
        (index for index, line in enumerate(lines) if line.startswith(heading_prefix)),
        None,
    )
    if start is None:
        return None
    end = next(
        (
            index
            for index, line in enumerate(lines[start + 1 :], start + 1)
            if line.startswith("#### ")
        ),
        len(lines),
    )
    return lines[start], "\n".join(lines[start:end])


def validate_mtp_6bit_summary_contract(
    *, summary_path: Path, expected_run_dir: str
) -> tuple[dict[str, Any], bool]:
    if not summary_path.exists():
        raise ArtifactCheckError(
            f"README MTP 6-bit summary does not exist: {summary_path}"
        )
    try:
        summary = json.loads(summary_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactCheckError(
            f"README MTP 6-bit summary is unreadable: {summary_path}: {error}"
        ) from error
    schema = summary.get("schema")
    if schema not in {
        MTP_6BIT_APPROXIMATE_SCHEMA_VERSION,
        MTP_6BIT_EXACT_SCHEMA_VERSION,
    }:
        raise ArtifactCheckError(
            f"README MTP 6-bit summary has unsupported schema {schema!r}: "
            f"{summary_path}"
        )
    approximate = schema == MTP_6BIT_APPROXIMATE_SCHEMA_VERSION
    publication_candidate = not approximate
    if summary.get("publication_candidate") is not publication_candidate:
        raise ArtifactCheckError(
            "README MTP 6-bit summary publication_candidate does not match "
            f"schema {schema}: {summary_path}"
        )
    claim_type = (
        "approximate_optimistic_diagnostic"
        if approximate
        else "exact_mtp_acceleration"
    )
    if summary.get("claim_type") != claim_type:
        raise ArtifactCheckError(
            f"README MTP 6-bit summary claim_type must be {claim_type!r}: "
            f"{summary_path}"
        )
    if summary.get("run_dir") != expected_run_dir:
        raise ArtifactCheckError(
            f"README MTP 6-bit summary run_dir does not match its link: {summary_path}"
        )
    rows = summary.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ArtifactCheckError(f"README MTP 6-bit summary has no rows: {summary_path}")
    for row in rows:
        if not isinstance(row, dict):
            raise ArtifactCheckError(
                f"README MTP 6-bit summary row is not an object: {summary_path}"
            )
        if row.get("publication_candidate") is not publication_candidate:
            raise ArtifactCheckError(
                "README MTP 6-bit summary row publication_candidate does not "
                f"match schema {schema}: {summary_path}"
            )
    if not approximate:
        methodology = summary.get("methodology")
        if not isinstance(methodology, dict):
            raise ArtifactCheckError(
                f"README exact MTP summary lacks methodology: {summary_path}"
            )
        expected_methodology = {
            "targets": list(MTP_6BIT_EXACT_TARGET_IDS),
            "suites": list(MTP_6BIT_EXACT_SUITES),
            "generated_tokens": 1000,
            "repetitions": 5,
            "warmup_repetitions": 2,
            "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20},
        }
        for key, expected in expected_methodology.items():
            if methodology.get(key) != expected:
                raise ArtifactCheckError(
                    f"README exact MTP methodology {key} must be {expected!r}: "
                    f"{summary_path}"
                )

        expected_rows = {
            (model_id, suite_id)
            for model_id in MTP_6BIT_EXACT_TARGET_IDS
            for suite_id in MTP_6BIT_EXACT_SUITES
        }
        actual_rows: set[tuple[str, str]] = set()
        for row in rows:
            row_key = (str(row.get("model_id")), str(row.get("suite_id")))
            if row_key in actual_rows:
                raise ArtifactCheckError(
                    f"README exact MTP summary has duplicate row {row_key!r}"
                )
            actual_rows.add(row_key)
            if row.get("publication_reasons") != []:
                raise ArtifactCheckError(
                    f"README exact MTP row has publication reasons: {row_key!r}"
                )
            try:
                direct = float(row["ax_direct_decode_tok_s"])
                mtp = float(row["ax_mtp_decode_tok_s"])
                speedup = float(row["ax_mtp_speedup_x"])
                coverage = float(row["ax_mtp_step_coverage_pct"])
            except (KeyError, TypeError, ValueError) as error:
                raise ArtifactCheckError(
                    f"README exact MTP row has invalid metrics: {row_key!r}"
                ) from error
            if direct <= 0.0 or mtp <= 0.0 or speedup <= 1.0:
                raise ArtifactCheckError(
                    f"README exact MTP row does not accelerate decode: {row_key!r}"
                )
            if abs(speedup - mtp / direct) > 0.001:
                raise ArtifactCheckError(
                    f"README exact MTP row speedup is inconsistent: {row_key!r}"
                )
            if coverage != 100.0:
                raise ArtifactCheckError(
                    f"README exact MTP row lacks 100% step coverage: {row_key!r}"
                )
            if row.get("ax_mtp_fallback_prompt_count") != 0:
                raise ArtifactCheckError(
                    f"README exact MTP row has fallback prompts: {row_key!r}"
                )
            if row.get("ax_mtp_direct_fallback_steps") != 0:
                raise ArtifactCheckError(
                    f"README exact MTP row has direct fallback steps: {row_key!r}"
                )
            ngram = row.get("ax_mtp_ngram_telemetry")
            if not isinstance(ngram, dict) or any(
                ngram.get(key) != 0 for key in MTP_6BIT_NGRAM_ZERO_KEYS
            ):
                raise ArtifactCheckError(
                    f"README exact MTP row has nonzero n-gram telemetry: {row_key!r}"
                )
        if actual_rows != expected_rows:
            raise ArtifactCheckError(
                "README exact MTP summary does not contain the complete supported "
                f"matrix: {summary_path}"
            )
    return summary, approximate


def mtp_6bit_diagnostic_table_cells(row: dict[str, Any]) -> list[str]:
    required = {
        "model_id",
        "suite_id",
        "ax_direct_decode_tok_s",
        "ax_mtp_decode_tok_s",
        "ax_mtp_speedup_x",
        "ax_mtp_draft_quality_pct",
        "ax_mtp_draft_quality_kind",
        "ax_mtp_step_coverage_pct",
        "ax_mtp_fallback_prompt_count",
        "prompt_count",
    }
    missing = sorted(required - row.keys())
    if missing:
        raise ArtifactCheckError(
            "README MTP approximate diagnostic row is missing fields: "
            + ", ".join(missing)
        )
    quality_kind = row["ax_mtp_draft_quality_kind"]
    if quality_kind == "target_argmax_match_ewma":
        quality_suffix = "match"
    elif quality_kind == "verified_accept_rate":
        quality_suffix = "verified"
    else:
        raise ArtifactCheckError(
            "README MTP approximate diagnostic row has unsupported draft quality "
            f"kind {quality_kind!r}"
        )
    try:
        return [
            f"`{row['model_id']}`",
            f"`{row['suite_id']}`",
            f"{float(row['ax_direct_decode_tok_s']):.1f} tok/s",
            f"{float(row['ax_mtp_decode_tok_s']):.1f} tok/s",
            f"{float(row['ax_mtp_speedup_x']):.2f}x",
            f"{float(row['ax_mtp_draft_quality_pct']):.1f}% {quality_suffix}",
            f"{float(row['ax_mtp_step_coverage_pct']):.1f}%",
            (
                f"{int(row['ax_mtp_fallback_prompt_count'])}/"
                f"{int(row['prompt_count'])}"
            ),
        ]
    except (TypeError, ValueError) as error:
        raise ArtifactCheckError(
            "README MTP approximate diagnostic row has invalid numeric fields"
        ) from error


def mtp_6bit_exact_table_cells(row: dict[str, Any]) -> list[str]:
    required = {
        "model_id",
        "suite_id",
        "ax_direct_decode_tok_s",
        "ax_mtp_decode_tok_s",
        "ax_mtp_speedup_x",
        "ax_mtp_prefill_tok_s",
        "ax_mtp_ttft_ms",
        "ax_mtp_accept_rate_pct",
    }
    missing = sorted(required - row.keys())
    if missing:
        raise ArtifactCheckError(
            "README exact MTP row is missing fields: " + ", ".join(missing)
        )
    try:
        return [
            f"`{row['model_id']}`",
            f"`{row['suite_id']}`",
            f"{float(row['ax_direct_decode_tok_s']):.1f} tok/s",
            f"{float(row['ax_mtp_decode_tok_s']):.1f} tok/s",
            f"{float(row['ax_mtp_speedup_x']):.2f}x",
            f"{float(row['ax_mtp_prefill_tok_s']):.1f} tok/s",
            f"{float(row['ax_mtp_ttft_ms']):.0f} ms",
            f"{float(row['ax_mtp_accept_rate_pct']):.1f}%",
        ]
    except (TypeError, ValueError) as error:
        raise ArtifactCheckError(
            "README exact MTP row has invalid numeric fields"
        ) from error


def validate_readme_mtp_6bit_claims(*, readme_path: Path) -> list[str]:
    readme_text = readme_path.read_text()
    section = extract_readme_section(
        readme_text,
        heading_prefix="#### AX Engine 6-bit",
    )
    if section is None:
        return []
    heading, section_text = section
    summary_match = re.search(
        r"\]\((benchmarks/results/(?:speculative/)?mtp-6bit/[^)]+/summary\.json)\)",
        section_text,
    )
    if summary_match is None:
        raise ArtifactCheckError(
            "README MTP 6-bit section does not link a summary.json artifact"
        )
    summary_link = Path(summary_match.group(1).removeprefix("../"))
    summary_path = resolve_results_doc_path(readme_path, summary_match.group(1))
    summary, approximate = validate_mtp_6bit_summary_contract(
        summary_path=summary_path,
        expected_run_dir=summary_link.parent.as_posix(),
    )
    artifact_date_match = re.match(r"\d{4}-\d{2}-\d{2}", summary_path.parent.name)
    heading_date_match = re.search(r"\((\d{4}-\d{2}-\d{2})\)$", heading)
    if artifact_date_match is None or heading_date_match is None:
        raise ArtifactCheckError(
            "README MTP 6-bit heading and artifact directory must carry a date"
        )
    if artifact_date_match.group(0) != heading_date_match.group(1):
        raise ArtifactCheckError(
            "README MTP 6-bit heading date does not match its summary artifact"
        )
    if not approximate:
        normalized = normalized_markdown_text(section_text).lower()
        if "exact sampled-mtp acceleration" not in heading.lower():
            raise ArtifactCheckError(
                "README publishable MTP section heading must say "
                "'exact sampled-MTP acceleration'"
            )
        for snippet in (
            "distribution-exact sampled mtp",
            f"all {len(summary['rows'])} target/suite rows accelerate decode",
            "100% mtp step coverage",
            "zero direct-fallback prompts or steps",
            # Accept either root-README or docs/PERFORMANCE-RESULTS asset paths.
            "perf-mtp-6bit-ax-acceleration.svg",
            "`temperature=0.6`",
            "`top_p=0.95`",
            "`top_k=20`",
            "2 warmups",
            "5 measured repetitions",
            "prefill and ttft are reported as context, not mtp acceleration claims",
        ):
            if snippet not in normalized:
                raise ArtifactCheckError(
                    f"README exact MTP acceleration section is missing {snippet!r}"
                )

        table_lines = extract_table_lines(section_text, "#### AX Engine 6-bit")
        expected_headers = [
            "Target",
            "Suite",
            "AX direct decode",
            "AX MTP decode",
            "AX speedup",
            "AX MTP prefill",
            "AX MTP TTFT",
            "AX accept",
        ]
        headers = split_markdown_row(table_lines[0])
        if headers != expected_headers:
            raise ArtifactCheckError(
                "README exact MTP table headers are stale; expected "
                + " | ".join(expected_headers)
            )
        displayed_rows = table_lines[2:]
        artifact_rows = summary["rows"]
        if len(displayed_rows) != len(artifact_rows):
            raise ArtifactCheckError(
                "README exact MTP table row count does not match summary"
            )
        for line, row in zip(displayed_rows, artifact_rows):
            cells = split_markdown_row(line)
            expected_cells = mtp_6bit_exact_table_cells(row)
            if cells != expected_cells:
                raise ArtifactCheckError(
                    "README exact MTP table row is stale; expected "
                    + " | ".join(expected_cells)
                )
        return [
            f"mtp-6bit:{summary['schema']}:{len(artifact_rows)}:publishable"
        ]

    normalized = normalized_markdown_text(section_text).lower()
    if "approximate mtp diagnostic" not in heading.lower():
        raise ArtifactCheckError(
            "README nonpublishable MTP section heading must say "
            "'approximate MTP diagnostic'"
        )
    if "acceleration" in heading.lower():
        raise ArtifactCheckError(
            "README nonpublishable MTP section must not be titled as acceleration"
        )
    for snippet in (
        "approximate optimistic",
        "not publication eligible",
        "`temperature=0.0`",
        "`top_p=1.0`",
        "`top_k=0`",
        "perf-mtp-6bit-ax-approximate-diagnostic.svg",
    ):
        if snippet not in normalized:
            raise ArtifactCheckError(
                f"README MTP approximate diagnostic is missing {snippet!r}"
            )
    if "sampled decode" in normalized:
        raise ArtifactCheckError(
            "README MTP approximate diagnostic still claims sampled decode"
        )
    methodology = summary.get("methodology")
    if not isinstance(methodology, dict):
        raise ArtifactCheckError(
            "README MTP approximate diagnostic summary lacks methodology"
        )
    sampling = methodology.get("sampling")
    if sampling != {"temperature": 0.0, "top_p": 1.0, "top_k": 0}:
        raise ArtifactCheckError(
            "README MTP approximate diagnostic summary must use the recorded "
            "greedy sampler"
        )

    table_lines = extract_table_lines(
        section_text,
        "#### AX Engine 6-bit",
    )
    expected_headers = [
        "Target",
        "Suite",
        "AX direct decode",
        "Approx. MTP decode",
        "Diagnostic ratio",
        "Draft quality",
        "MTP step coverage",
        "Fallback prompts",
    ]
    headers = split_markdown_row(table_lines[0])
    if headers != expected_headers:
        raise ArtifactCheckError(
            "README MTP approximate diagnostic table headers are stale; expected "
            + " | ".join(expected_headers)
        )
    displayed_rows = table_lines[2:]
    artifact_rows = summary["rows"]
    if len(displayed_rows) != len(artifact_rows):
        raise ArtifactCheckError(
            "README MTP approximate diagnostic row count does not match summary"
        )
    for line, row in zip(displayed_rows, artifact_rows):
        cells = split_markdown_row(line)
        expected_cells = mtp_6bit_diagnostic_table_cells(row)
        if cells != expected_cells:
            raise ArtifactCheckError(
                "README MTP approximate diagnostic table row is stale; expected "
                + " | ".join(expected_cells)
            )
    return [
        f"mtp-6bit:{summary['schema']}:{len(artifact_rows)}:nonpublishable"
    ]


def validate_readme_direct_generation_claims(*, readme_path: Path) -> list[str]:
    text = readme_path.read_text()
    if "### Session Mode: Direct Generation" not in text:
        return []

    retired_charts = ("perf-direct-validation-2026-07-12.svg",)
    restored_box_charts = (
        "perf-gemma4-decode-box-whisker.svg",
        "perf-gemma4-prefill-box-whisker.svg",
        "perf-gemma4-ttft-box-whisker.svg",
        "perf-qwen-decode-box-whisker.svg",
        "perf-qwen-prefill-box-whisker.svg",
        "perf-qwen-ttft-box-whisker.svg",
    )
    referenced_retired_charts = [chart for chart in retired_charts if chart in text]
    if referenced_retired_charts:
        raise ArtifactCheckError(
            "README references retired direct-mode aggregate charts: "
            + ", ".join(referenced_retired_charts)
        )
    referenced_box_charts = [chart for chart in restored_box_charts if chart in text]
    if referenced_box_charts:
        required_boxplot_boundary = (
            "cross-run distribution diagnostics, not exact per-model deltas or a",
            "same-session peer benchmark",
            "the exact AX values are in the table below",
        )
        missing_boundary = [
            phrase for phrase in required_boxplot_boundary if phrase not in text
        ]
        if missing_boundary:
            raise ArtifactCheckError(
                "README direct peer boxplots require an explicit cross-run boundary: "
                + ", ".join(missing_boundary)
            )

    unsupported_claims = (
        "AX Engine leads `mlx_lm` on prefill, runner-time TTFT, and direct decode",
        "AX leads `mlx_lm` on prefill, runner-time TTFT, and decode",
    )
    for claim in unsupported_claims:
        if claim in text:
            raise ArtifactCheckError(
                f"README contains unsupported matrix-wide direct claim: {claim}"
            )

    required_boundary = (
        "No current-head, matrix-wide direct peer comparison is published."
    )
    if required_boundary not in text:
        raise ArtifactCheckError(
            "README direct-mode section must state the current-head publication boundary"
        )
    return ["direct-generation:current-head-claim-boundary"]


def validate_concurrent_prefill_overlap_classification(
    *, artifact_path: Path, evidence: dict[str, Any]
) -> None:
    classification = evidence.get("classification")
    if classification not in OVERLAP_CLASSIFICATION_VALUES:
        raise ArtifactCheckError(
            f"{artifact_path} concurrent prefill overlap classification is invalid"
        )
    claim = evidence.get("continuous_batching_claim")
    if not isinstance(claim, bool):
        raise ArtifactCheckError(
            f"{artifact_path} concurrent prefill overlap continuous_batching_claim must be a boolean"
        )
    concurrency = evidence.get("concurrency")
    if (
        not isinstance(concurrency, int)
        or isinstance(concurrency, bool)
        or concurrency <= 0
    ):
        raise ArtifactCheckError(
            f"{artifact_path} concurrent prefill overlap concurrency must be a positive integer"
        )
    expected_claim = classification in POSITIVE_CONTINUOUS_BATCHING_CLASSIFICATIONS
    if claim != expected_claim:
        raise ArtifactCheckError(
            f"{artifact_path} concurrent prefill overlap claim is inconsistent with classification"
        )
    if classification == "single_request_no_overlap" and concurrency != 1:
        raise ArtifactCheckError(
            f"{artifact_path} single_request_no_overlap must use concurrency=1"
        )


def validate_prefix_reuse_evidence_shape(
    *, artifact_path: Path, evidence: dict[str, Any]
) -> None:
    for counter in sorted(PREFIX_REUSE_EVIDENCE_COUNTERS):
        if counter not in evidence:
            raise ArtifactCheckError(
                f"{artifact_path} prefix reuse evidence lacks {counter}"
            )
        if not isinstance(evidence[counter], int) or isinstance(
            evidence[counter], bool
        ):
            raise ArtifactCheckError(
                f"{artifact_path} prefix reuse evidence {counter} must be an integer"
            )
        if evidence[counter] < 0:
            raise ArtifactCheckError(
                f"{artifact_path} prefix reuse evidence {counter} must be non-negative"
            )
    for flag in (
        "physical_snapshot_hit_observed",
        "physical_snapshot_miss_warmup_observed",
        "physical_snapshot_blocked_observed",
    ):
        if not isinstance(evidence.get(flag), bool):
            raise ArtifactCheckError(
                f"{artifact_path} prefix reuse evidence {flag} must be a boolean"
            )
    coverage = evidence.get("physical_snapshot_coverage")
    if coverage not in PREFIX_REUSE_COVERAGE_VALUES:
        raise ArtifactCheckError(
            f"{artifact_path} prefix reuse evidence has invalid physical_snapshot_coverage"
        )
    expected = expected_prefix_reuse_classification(evidence)
    for key, expected_value in expected.items():
        if evidence.get(key) != expected_value:
            raise ArtifactCheckError(
                f"{artifact_path} prefix reuse evidence {key} is inconsistent with counters"
            )


def expected_prefix_reuse_classification(evidence: dict[str, Any]) -> dict[str, Any]:
    hit_observed = evidence["hit_count"] > 0
    miss_warmup_observed = (
        evidence["miss_count"] > 0 and evidence["warmup_token_count"] > 0
    )
    blocked_reason_count = (
        evidence["blocked_policy_disabled_count"]
        + evidence["blocked_unsupported_layout_count"]
        + evidence["blocked_trim_failure_count"]
    )
    blocked_count = evidence["blocked_count"]
    blocked_observed = blocked_count > 0

    if hit_observed and miss_warmup_observed:
        coverage = "hit_and_miss_warmup"
    elif miss_warmup_observed:
        coverage = "miss_warmup_only"
    elif hit_observed:
        coverage = "hit_only"
    elif blocked_observed:
        coverage = "blocked_only"
    else:
        coverage = "none_observed"

    return {
        "physical_snapshot_hit_observed": hit_observed,
        "physical_snapshot_miss_warmup_observed": miss_warmup_observed,
        "physical_snapshot_blocked_observed": blocked_observed,
        "physical_snapshot_coverage": coverage,
        "blocked_reason_count": blocked_reason_count,
        "blocked_reason_accounting_gap_count": max(
            0, blocked_count - blocked_reason_count
        ),
    }


def expected_run_stability_summary(results: list[Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "schema_version": RUN_STABILITY_SUMMARY_SCHEMA_VERSION,
        "scope": "ax_engine_rows",
        "row_count": 0,
        "stable_enough_count": 0,
        "unstable_count": 0,
        "missing_count": 0,
        "classification_counts": {},
        "unstable_rows": [],
        "publication_candidate": True,
    }
    for row in results:
        if not isinstance(row, dict):
            continue
        engine = str(row.get("engine", ""))
        if not engine.startswith("ax_engine"):
            continue
        summary["row_count"] += 1
        stability = row.get("run_stability")
        if not isinstance(stability, dict):
            summary["missing_count"] += 1
            summary["publication_candidate"] = False
            continue
        shape_problem = None
        if stability.get("schema_version") != RUN_STABILITY_SCHEMA_VERSION:
            shape_problem = "invalid_run_stability_schema"
        elif stability.get("metric") != "decode_tok_s":
            shape_problem = "invalid_run_stability_metric"
        classification = shape_problem or str(
            stability.get("classification", "unknown")
        )
        counts = summary["classification_counts"]
        counts[classification] = int(counts.get(classification, 0)) + 1
        if shape_problem is None and classification == "stable_enough":
            summary["stable_enough_count"] += 1
            continue
        summary["unstable_count"] += 1
        summary["publication_candidate"] = False
        unstable_row = {
            "engine": engine,
            "prompt_tokens": row.get("prompt_tokens"),
            "generation_tokens": row.get("generation_tokens"),
            "classification": classification,
        }
        drift = stability.get("last_vs_first_pct")
        if isinstance(drift, (int, float)):
            unstable_row["last_vs_first_pct"] = float(drift)
        summary["unstable_rows"].append(unstable_row)
    if summary["row_count"] == 0:
        summary["publication_candidate"] = False
    return summary


def validate_run_stability_summary_if_present(
    *, artifact_path: Path, artifact: dict[str, Any]
) -> None:
    summary = artifact.get("run_stability_summary")
    if summary is None:
        return
    if not isinstance(summary, dict):
        raise ArtifactCheckError(
            f"{artifact_path} run_stability_summary must be an object"
        )
    results = artifact.get("results")
    if not isinstance(results, list):
        raise ArtifactCheckError(
            f"{artifact_path} run_stability_summary requires results list"
        )
    for row in results:
        if not isinstance(row, dict):
            continue
        engine = str(row.get("engine", ""))
        if not engine.startswith("ax_engine"):
            continue
        if isinstance(row.get("run_stability"), dict):
            validate_run_stability_shape(artifact_path=artifact_path, row=row)
    expected = expected_run_stability_summary(results)
    for key, expected_value in expected.items():
        if summary.get(key) != expected_value:
            raise ArtifactCheckError(
                f"{artifact_path} run_stability_summary {key} is inconsistent with result rows"
            )
    if summary.get("publication_candidate") is not True:
        raise ArtifactCheckError(
            f"{artifact_path} run_stability_summary is not a publication candidate"
        )


def validate_phase0_artifact_gate(
    *, artifact_path: Path, artifact: dict[str, Any]
) -> None:
    if not phase0_claim_gate_enabled(artifact):
        return
    validate_claim_gate_methodology(artifact_path=artifact_path, artifact=artifact)
    overlap = artifact.get("concurrent_prefill_overlap_classification")
    if not isinstance(overlap, dict):
        raise ArtifactCheckError(
            f"{artifact_path} lacks concurrent prefill overlap classification"
        )
    validate_concurrent_prefill_overlap_classification(
        artifact_path=artifact_path,
        evidence=overlap,
    )
    prefix_reuse = artifact.get("prefix_reuse_evidence")
    if not isinstance(prefix_reuse, dict):
        raise ArtifactCheckError(f"{artifact_path} lacks prefix reuse evidence")
    validate_prefix_reuse_evidence_shape(
        artifact_path=artifact_path,
        evidence=prefix_reuse,
    )
    validate_public_claim_evidence(artifact_path=artifact_path, artifact=artifact)


def validate_metric_summary(
    *, artifact_path: Path, row: dict[str, Any], key: str
) -> None:
    if not has_metric_summary(row, key):
        raise ArtifactCheckError(
            f"{artifact_path} {row.get('engine')} lacks {key}.median"
        )


def metric_summary_median(row: dict[str, Any], key: str) -> float:
    metric = row.get(key)
    if not isinstance(metric, dict) or metric.get("median") is None:
        raise ArtifactCheckError(f"{row.get('engine')} lacks {key}.median")
    return float(metric["median"])


def metric_summary_median_or_zero(row: dict[str, Any], key: str) -> float:
    metric = row.get(key)
    if not isinstance(metric, dict):
        return 0.0
    if metric.get("median") is not None:
        return float(metric["median"])
    if metric.get("mean") is not None:
        return float(metric["mean"])
    return 0.0


def ax_refresh_row_key(row: dict[str, Any]) -> tuple[str, int, int]:
    return (
        str(row.get("engine", "")),
        int(row.get("prompt_tokens", -1)),
        int(row.get("generation_tokens", -1)),
    )


def expected_ax_only_refresh_regression_summary(
    *, artifact: dict[str, Any], reference_artifact: dict[str, Any]
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "schema_version": AX_ONLY_REFRESH_REGRESSION_SCHEMA_VERSION,
        "scope": "matching_ax_rows_from_reused_reference_artifact",
        "metric": "decode_tok_s",
        "decode_min_ratio_to_reference": AX_ONLY_REFRESH_DECODE_MIN_RATIO_TO_REFERENCE,
        "row_count": 0,
        "matched_count": 0,
        "missing_reference_count": 0,
        "duplicate_current_count": 0,
        "duplicate_reference_count": 0,
        "decode_regression_count": 0,
        "classification_counts": {},
        "missing_reference_rows": [],
        "duplicate_current_rows": [],
        "duplicate_reference_rows": [],
        "rows": [],
        "publication_candidate": True,
    }
    reference_rows = {}
    for row in reference_artifact.get("results", []):
        if not isinstance(row, dict):
            continue
        if str(row.get("engine", "")).startswith("ax_engine"):
            key = ax_refresh_row_key(row)
            if key in reference_rows:
                summary["duplicate_reference_count"] += 1
                counts = summary["classification_counts"]
                counts["duplicate_reference"] = (
                    int(counts.get("duplicate_reference", 0)) + 1
                )
                summary["duplicate_reference_rows"].append(
                    {
                        "engine": key[0],
                        "prompt_tokens": key[1],
                        "generation_tokens": key[2],
                        "classification": "duplicate_reference",
                    }
                )
                summary["publication_candidate"] = False
                continue
            reference_rows[key] = row

    current_rows: set[tuple[str, int, int]] = set()
    for row in artifact.get("results", []):
        if not isinstance(row, dict):
            continue
        if not str(row.get("engine", "")).startswith("ax_engine"):
            continue
        summary["row_count"] += 1
        key = ax_refresh_row_key(row)
        if key in current_rows:
            summary["duplicate_current_count"] += 1
            counts = summary["classification_counts"]
            counts["duplicate_current"] = int(counts.get("duplicate_current", 0)) + 1
            summary["duplicate_current_rows"].append(
                {
                    "engine": key[0],
                    "prompt_tokens": key[1],
                    "generation_tokens": key[2],
                    "classification": "duplicate_current",
                }
            )
            summary["publication_candidate"] = False
            continue
        current_rows.add(key)
        reference_row = reference_rows.get(key)
        if reference_row is None:
            summary["missing_reference_count"] += 1
            counts = summary["classification_counts"]
            counts["missing_reference"] = int(counts.get("missing_reference", 0)) + 1
            summary["missing_reference_rows"].append(
                {
                    "engine": key[0],
                    "prompt_tokens": key[1],
                    "generation_tokens": key[2],
                    "classification": "missing_reference",
                }
            )
            summary["publication_candidate"] = False
            continue
        current_decode = metric_summary_median_or_zero(row, "decode_tok_s")
        reference_decode = metric_summary_median_or_zero(reference_row, "decode_tok_s")
        if current_decode <= 0.0 or reference_decode <= 0.0:
            classification = "missing_decode_metric"
            decode_ratio = None
        else:
            decode_ratio = current_decode / reference_decode
            classification = (
                "decode_regression"
                if decode_ratio < AX_ONLY_REFRESH_DECODE_MIN_RATIO_TO_REFERENCE
                else "within_tolerance"
            )
        counts = summary["classification_counts"]
        counts[classification] = int(counts.get(classification, 0)) + 1
        if classification == "decode_regression":
            summary["decode_regression_count"] += 1
            summary["publication_candidate"] = False
        elif classification == "missing_decode_metric":
            summary["publication_candidate"] = False
        summary["matched_count"] += 1
        current_prefill = metric_summary_median_or_zero(row, "prefill_tok_s")
        reference_prefill = metric_summary_median_or_zero(reference_row, "prefill_tok_s")
        current_ttft = metric_summary_median_or_zero(row, "ttft_ms")
        reference_ttft = metric_summary_median_or_zero(reference_row, "ttft_ms")
        summary["rows"].append(
            {
                "engine": key[0],
                "prompt_tokens": key[1],
                "generation_tokens": key[2],
                "classification": classification,
                "current_decode_tok_s": current_decode,
                "reference_decode_tok_s": reference_decode,
                "decode_ratio_to_reference": decode_ratio,
                "current_prefill_tok_s": current_prefill,
                "reference_prefill_tok_s": reference_prefill,
                "prefill_ratio_to_reference": (
                    current_prefill / reference_prefill
                    if current_prefill > 0.0 and reference_prefill > 0.0
                    else None
                ),
                "current_ttft_ms": current_ttft,
                "reference_ttft_ms": reference_ttft,
                "ttft_ratio_to_reference": (
                    current_ttft / reference_ttft
                    if current_ttft > 0.0 and reference_ttft > 0.0
                    else None
                ),
            }
        )
    if summary["row_count"] == 0:
        summary["publication_candidate"] = False
    return summary


def validate_ax_only_refresh_regression_summary_if_present(
    *, repo_root: Path, artifact_path: Path, artifact: dict[str, Any]
) -> None:
    ax_only_refresh = artifact.get("ax_only_refresh")
    if not isinstance(ax_only_refresh, dict):
        return
    summary = ax_only_refresh.get("ax_reference_regression_summary")
    if summary is None:
        if ax_only_refresh.get("schema_version") == AX_ONLY_REFRESH_SCHEMA_VERSION:
            raise ArtifactCheckError(
                f"{artifact_path} ax_only_refresh lacks ax_reference_regression_summary"
            )
        return
    if not isinstance(summary, dict):
        raise ArtifactCheckError(
            f"{artifact_path} ax_reference_regression_summary must be an object"
        )
    reference_source = ax_only_refresh.get("reference_results_source")
    if not isinstance(reference_source, str):
        raise ArtifactCheckError(
            f"{artifact_path} ax_only_refresh lacks reference_results_source"
        )
    reference_path = resolve_repo_path(repo_root, reference_source)
    if not reference_path.exists():
        raise ArtifactCheckError(
            f"{artifact_path} ax_only_refresh reference does not exist: {reference_path}"
        )
    reference_artifact = json.loads(reference_path.read_text())
    if not isinstance(reference_artifact, dict):
        raise ArtifactCheckError(
            f"{artifact_path} ax_only_refresh reference must be an object"
        )
    if reference_artifact.get("schema_version") != MLX_INFERENCE_STACK_SCHEMA_VERSION:
        raise ArtifactCheckError(
            f"{artifact_path} ax_only_refresh reference has unexpected schema_version"
        )
    if not isinstance(reference_artifact.get("results"), list):
        raise ArtifactCheckError(
            f"{artifact_path} ax_only_refresh reference lacks results list"
        )
    expected = expected_ax_only_refresh_regression_summary(
        artifact=artifact,
        reference_artifact=reference_artifact,
    )
    if summary != expected:
        raise ArtifactCheckError(
            f"{artifact_path} ax_reference_regression_summary is inconsistent"
        )
    if summary.get("publication_candidate") is not True:
        raise ArtifactCheckError(
            f"{artifact_path} ax_reference_regression_summary is not a publication candidate"
        )


def validate_positive_metric_summary(
    *, artifact_path: Path, row: dict[str, Any], key: str
) -> None:
    validate_metric_summary(artifact_path=artifact_path, row=row, key=key)
    if metric_summary_median(row, key) <= 0.0:
        raise ArtifactCheckError(
            f"{artifact_path} {row.get('engine')} {key}.median must be positive"
        )


def validate_run_stability_shape(
    *, artifact_path: Path, row: dict[str, Any]
) -> dict[str, Any]:
    stability = row.get("run_stability")
    if not isinstance(stability, dict):
        raise ArtifactCheckError(
            f"{artifact_path} {row.get('engine')} run_stability must be an object"
        )
    if stability.get("schema_version") != RUN_STABILITY_SCHEMA_VERSION:
        raise ArtifactCheckError(
            f"{artifact_path} {row.get('engine')} has stale run_stability schema"
        )
    if stability.get("metric") != "decode_tok_s":
        raise ArtifactCheckError(
            f"{artifact_path} {row.get('engine')} run_stability must track decode_tok_s"
        )
    return stability


def validate_run_stability_if_present(
    *, artifact_path: Path, row: dict[str, Any], require_phase0: bool
) -> None:
    if not require_phase0:
        return
    stability = row.get("run_stability")
    if stability is None:
        return
    stability = validate_run_stability_shape(artifact_path=artifact_path, row=row)
    classification = stability.get("classification")
    if classification != "stable_enough":
        raise ArtifactCheckError(
            f"{artifact_path} {row.get('engine')} unstable benchmark row: "
            f"{classification}"
        )


def validate_mtp_row_contract(
    *, artifact_path: Path, row: dict[str, Any], require_phase0: bool
) -> None:
    engine = row.get("engine")
    expected_policy = (
        "gemma4_assistant_mtp_no_ngram_stacking"
        if engine == "ax_engine_gemma4_assistant_mtp"
        else "mtp_head_only_no_ngram_stacking"
    )
    if row.get("ax_decode_policy") != expected_policy:
        raise ArtifactCheckError(f"{artifact_path} MTP AX row lacks MTP-only policy")
    if not require_phase0:
        return
    if row.get("ax_decode_claim_status") not in AX_MTP_CLAIM_STATUSES:
        raise ArtifactCheckError(f"{artifact_path} MTP AX row lacks effective MTP claim")
    if row.get("ax_decode_effective_route") not in AX_MTP_EFFECTIVE_ROUTES:
        raise ArtifactCheckError(
            f"{artifact_path} MTP AX row lacks MTP verify-loop route"
        )
    telemetry = row.get("ngram_acceleration_telemetry")
    if not isinstance(telemetry, dict):
        raise ArtifactCheckError(f"{artifact_path} MTP AX row lacks MTP telemetry")
    if int(telemetry.get("ax_mtp_ngram_hit_steps", 0)) != 0:
        raise ArtifactCheckError(
            f"{artifact_path} MTP AX row has n-gram stacking hits"
        )
    if int(telemetry.get("ax_mtp_draft_tokens", 0)) <= 0:
        raise ArtifactCheckError(f"{artifact_path} MTP AX row has no draft tokens")


def validate_phase0_runtime_identity(
    *, artifact_path: Path, row: dict[str, Any]
) -> None:
    runtime = row.get("runtime_identity")
    if not isinstance(runtime, dict):
        raise ArtifactCheckError(
            f"{artifact_path} {row.get('engine')} lacks runtime_identity"
        )
    if runtime.get("selected_backend") != "mlx":
        raise ArtifactCheckError(
            f"{artifact_path} {row.get('engine')} runtime identity is not MLX"
        )
    if runtime.get("route_identity") != "repo_owned_mlx":
        raise ArtifactCheckError(
            f"{artifact_path} {row.get('engine')} lacks repo-owned MLX route identity"
        )


def validate_ax_prefill_decode_split(
    *, artifact_path: Path, row: dict[str, Any], require_phase0: bool
) -> None:
    validate_positive_metric_summary(
        artifact_path=artifact_path,
        row=row,
        key="prefill_s",
    )
    generation_tokens = int(row.get("generation_tokens", 0))
    telemetry = row.get("ax_mlx_telemetry")
    if not isinstance(telemetry, dict):
        raise ArtifactCheckError(
            f"{artifact_path} {row.get('engine')} lacks AX MLX telemetry"
        )
    for key in ("ax_mlx_prefill_steps", "ax_mlx_decode_steps"):
        if key not in telemetry:
            raise ArtifactCheckError(f"{artifact_path} {row.get('engine')} lacks {key}")
    decode_route = row.get("ax_mlx_decode_route")
    no_decode_steps = (
        generation_tokens > 1
        and int(telemetry.get("ax_mlx_decode_steps", 0)) == 0
        and isinstance(decode_route, dict)
        and decode_route.get("classification") == "no_decode_steps"
    )
    if generation_tokens > 1 and not no_decode_steps:
        validate_positive_metric_summary(
            artifact_path=artifact_path,
            row=row,
            key="decode_s",
        )
    else:
        validate_metric_summary(artifact_path=artifact_path, row=row, key="decode_s")
    if int(telemetry.get("ax_mlx_prefill_steps", 0)) <= 0:
        raise ArtifactCheckError(
            f"{artifact_path} {row.get('engine')} lacks positive ax_mlx_prefill_steps"
        )
    if (
        generation_tokens > 1
        and not no_decode_steps
        and int(telemetry.get("ax_mlx_decode_steps", 0)) <= 0
    ):
        raise ArtifactCheckError(
            f"{artifact_path} {row.get('engine')} lacks positive ax_mlx_decode_steps"
        )
    if require_phase0:
        if row.get("timing_scope") != "ax_engine_runner_time_us":
            raise ArtifactCheckError(
                f"{artifact_path} {row.get('engine')} lacks AX runner timing scope"
            )
        if row.get("ttft_source") != "ax_engine_runner_prefill_time":
            raise ArtifactCheckError(
                f"{artifact_path} {row.get('engine')} lacks AX TTFT source"
            )
        validate_positive_metric_summary(
            artifact_path=artifact_path,
            row=row,
            key="ttft_ms",
        )
        validate_phase0_runtime_identity(artifact_path=artifact_path, row=row)


def expected_ax_prefill_work_contract(row: dict[str, Any]) -> str:
    """Mirror ``generate.rs::mlx_lm_style_cache_only_prefix_len`` for claim gates.

    Greedy (or missing sampler) with ``prompt_tokens > 1`` uses the
    mlx-lm-style cache-only prefix. Sampling without repetition penalty uses
    cache-only only when ``prompt_tokens >= 512``.
    """
    sampler = row.get("sampler_settings")
    prompt_tokens = int(row.get("prompt_tokens", -1))
    if prompt_tokens <= 1:
        return HISTORICAL_PREFILL_WORK_CONTRACT
    if sampler in (None, "greedy"):
        return MLX_LM_STYLE_PREFILL_WORK_CONTRACT
    if isinstance(sampler, dict):
        temperature = float(sampler.get("temperature") or 0.0)
        repetition_penalty = float(sampler.get("repetition_penalty") or 1.0)
        if abs(repetition_penalty - 1.0) > 1e-6:
            return HISTORICAL_PREFILL_WORK_CONTRACT
        if temperature <= 0.0 or prompt_tokens >= 512:
            return MLX_LM_STYLE_PREFILL_WORK_CONTRACT
        return HISTORICAL_PREFILL_WORK_CONTRACT
    # String sampler labels other than greedy (e.g. "sampled") keep historical
    # unless the prompt is long enough that generate.rs would cache-only.
    if prompt_tokens >= 512:
        return MLX_LM_STYLE_PREFILL_WORK_CONTRACT
    return HISTORICAL_PREFILL_WORK_CONTRACT


def validate_ax_prefill_work_contract(
    *, artifact_path: Path, row: dict[str, Any]
) -> None:
    expected = expected_ax_prefill_work_contract(row)
    actual = row.get("prefill_work_contract")
    if actual == expected:
        return
    # Legacy short-prompt greedy rows were mislabeled as historical even though
    # the runtime already used cache-only. Accept the old label so published
    # artifacts remain valid while new runs emit the correct contract.
    sampler = row.get("sampler_settings")
    prompt_tokens = int(row.get("prompt_tokens", -1))
    legacy_short_greedy_mislabeled = (
        expected == MLX_LM_STYLE_PREFILL_WORK_CONTRACT
        and actual == HISTORICAL_PREFILL_WORK_CONTRACT
        and sampler in (None, "greedy")
        and 1 < prompt_tokens <= 512
    )
    if legacy_short_greedy_mislabeled:
        return
    raise ArtifactCheckError(
        f"{artifact_path} {row.get('engine')} prompt={row.get('prompt_tokens')} "
        f"has prefill_work_contract={actual!r}; expected {expected!r}"
    )


def validate_direct_hotpath_no_hidden_fallbacks(
    *,
    artifact_path: Path,
    row: dict[str, Any],
    require_phase0: bool,
    model_repo_id: str = "",
) -> None:
    if not require_phase0:
        return
    telemetry = row.get("ax_mlx_telemetry")
    if not isinstance(telemetry, dict):
        raise ArtifactCheckError(
            f"{artifact_path} direct AX row lacks AX MLX telemetry"
        )

    # Some dense FFN gate/up encodings intentionally stay split: 5-bit packing
    # is not validated against mlx_lm, and Qwen3 dense FFNs are guarded in
    # weights.rs until their packed path is token-exact. A non-zero split
    # counter is expected for those models, not a hidden fallback.
    normalized_model_repo_id = model_repo_id.lower()
    ffn_pack_intentionally_disabled = (
        "5bit" in normalized_model_repo_id
        or "5-bit" in normalized_model_repo_id
        or "qwen3.5" in normalized_model_repo_id
        or "qwen3.6-27b" in normalized_model_repo_id
    )

    fallback_counts = []
    for key, label in AX_DIRECT_HOTPATH_FALLBACK_COUNTERS.items():
        value = int(telemetry.get(key, 0))
        if value > 0:
            if (
                key == "ax_mlx_dense_ffn_split_gate_up_layers"
                and ffn_pack_intentionally_disabled
            ):
                continue
            fallback_counts.append(f"{key}={value} ({label})")
    if fallback_counts:
        raise ArtifactCheckError(
            f"{artifact_path} direct AX row has hidden hotpath fallback counters: "
            + ", ".join(fallback_counts)
        )

    kv_compression = row.get("kv_compression_telemetry")
    if (
        row.get("kv_compression_claim_status") == "integrated_fused_compressed_decode"
        and isinstance(kv_compression, dict)
        and int(kv_compression.get("ax_mlx_kv_compression_fused_decode_fallbacks", 0))
        > 0
    ):
        raise ArtifactCheckError(
            f"{artifact_path} direct AX row claims fused KV decode with fallback telemetry"
        )


def validate_direct_cpp_linear_attention_input_summary(
    *, artifact_path: Path, row: dict[str, Any], require_phase0: bool
) -> None:
    if not require_phase0:
        return
    telemetry = row.get("ax_mlx_telemetry")
    if not isinstance(telemetry, dict):
        raise ArtifactCheckError(
            f"{artifact_path} direct AX row lacks AX MLX telemetry"
        )

    counters = {
        field: int(telemetry.get(counter_key, 0))
        for field, counter_key in AX_DIRECT_CPP_LINEAR_ATTENTION_INPUT_COUNTERS.items()
    }
    if counters["attempts"] <= 0:
        if "ax_mlx_direct_cpp_linear_attention_inputs" in row:
            raise ArtifactCheckError(
                f"{artifact_path} direct AX row has direct C++ linear-attention "
                "summary without route attempts"
            )
        return

    summary = row.get("ax_mlx_direct_cpp_linear_attention_inputs")
    if not isinstance(summary, dict):
        raise ArtifactCheckError(
            f"{artifact_path} direct AX row lacks direct C++ linear-attention summary"
        )
    if summary.get("schema_version") != "ax.mlx_direct_cpp_linear_attention_inputs.v1":
        raise ArtifactCheckError(
            f"{artifact_path} direct AX row has invalid direct C++ linear-attention summary schema"
        )
    for field, expected in counters.items():
        if int(summary.get(field, -1)) != expected:
            raise ArtifactCheckError(
                f"{artifact_path} direct AX row direct C++ linear-attention "
                f"summary has {field}={summary.get(field)!r}; expected {expected}"
            )
    if summary.get("classification") != "all_hits":
        raise ArtifactCheckError(
            f"{artifact_path} direct AX row direct C++ linear-attention "
            f"summary is not all_hits: {summary.get('classification')!r}"
        )
    if counters["hits"] <= 0:
        raise ArtifactCheckError(
            f"{artifact_path} direct AX row attempted direct C++ linear-attention "
            "inputs but recorded no hits"
        )


def validate_direct_cpp_linear_attention_post_input_summary(
    *, artifact_path: Path, row: dict[str, Any], require_phase0: bool
) -> None:
    if not require_phase0:
        return
    telemetry = row.get("ax_mlx_telemetry")
    if not isinstance(telemetry, dict):
        raise ArtifactCheckError(
            f"{artifact_path} direct AX row lacks AX MLX telemetry"
        )

    counters = {
        field: int(telemetry.get(counter_key, 0))
        for field, counter_key in AX_DIRECT_CPP_LINEAR_ATTENTION_POST_INPUT_COUNTERS.items()
    }
    if counters["attempts"] <= 0:
        if "ax_mlx_direct_cpp_linear_attention_post_input" in row:
            raise ArtifactCheckError(
                f"{artifact_path} direct AX row has direct C++ linear-attention "
                "post-input summary without route attempts"
            )
        return

    summary = row.get("ax_mlx_direct_cpp_linear_attention_post_input")
    if not isinstance(summary, dict):
        raise ArtifactCheckError(
            f"{artifact_path} direct AX row lacks direct C++ linear-attention post-input summary"
        )
    if (
        summary.get("schema_version")
        != "ax.mlx_direct_cpp_linear_attention_post_input.v1"
    ):
        raise ArtifactCheckError(
            f"{artifact_path} direct AX row has invalid direct C++ "
            "linear-attention post-input summary schema"
        )
    for field, expected in counters.items():
        if int(summary.get(field, -1)) != expected:
            raise ArtifactCheckError(
                f"{artifact_path} direct AX row direct C++ linear-attention "
                f"post-input summary has {field}={summary.get(field)!r}; expected {expected}"
            )
    if summary.get("classification") != "all_hits":
        raise ArtifactCheckError(
            f"{artifact_path} direct AX row direct C++ linear-attention "
            "post-input summary is not all_hits: "
            f"{summary.get('classification')!r}"
        )
    if counters["hits"] <= 0:
        raise ArtifactCheckError(
            f"{artifact_path} direct AX row attempted direct C++ linear-attention "
            "post-input but recorded no hits"
        )


def validate_ngram_claim_telemetry(
    *, artifact_path: Path, row: dict[str, Any], require_phase0: bool
) -> None:
    telemetry = row.get("ngram_acceleration_telemetry")
    if not isinstance(telemetry, dict):
        raise ArtifactCheckError(f"{artifact_path} n-gram row lacks telemetry")
    if not require_phase0:
        return
    missing = sorted(AX_NGRAM_TELEMETRY_COUNTERS - set(telemetry))
    if missing:
        raise ArtifactCheckError(
            f"{artifact_path} n-gram row lacks telemetry counters: {', '.join(missing)}"
        )
    status = row.get("ax_decode_claim_status")
    effective_route = row.get("ax_decode_effective_route")
    if (
        effective_route is not None
        and effective_route not in AX_DECODE_EFFECTIVE_ROUTES
    ):
        raise ArtifactCheckError(
            f"{artifact_path} has unknown AX effective decode route: {effective_route}"
        )
    decode_route = row.get("ax_mlx_decode_route")
    no_decode_steps = (
        isinstance(decode_route, dict)
        and decode_route.get("classification") == "no_decode_steps"
    )
    attempts = int(telemetry.get("ax_ngram_draft_attempts", 0))
    accepted = int(telemetry.get("ax_ngram_accepted_tokens", 0))
    fallback_steps = int(telemetry.get("ax_ngram_no_draft_steps", 0)) + int(
        telemetry.get("ax_ngram_request_disabled_steps", 0)
    )
    if status == "ngram_acceleration_effective_throughput" and (
        attempts <= 0 or accepted <= 0
    ):
        raise ArtifactCheckError(
            f"{artifact_path} claims n-gram throughput without draft acceptance"
        )
    if (
        status == "ngram_acceleration_effective_throughput"
        and effective_route is not None
        and effective_route != "ngram_verified_bonus_tokens"
    ):
        raise ArtifactCheckError(
            f"{artifact_path} claims n-gram throughput with inconsistent effective route"
        )
    if status == "ngram_no_draft_direct_fallback" and (
        attempts != 0 or fallback_steps <= 0
    ):
        raise ArtifactCheckError(
            f"{artifact_path} claims n-gram fallback without fallback telemetry"
        )
    if (
        status == "ngram_no_draft_direct_fallback"
        and effective_route is not None
        and effective_route not in AX_NGRAM_NO_DRAFT_EFFECTIVE_ROUTES
    ):
        raise ArtifactCheckError(
            f"{artifact_path} claims n-gram fallback with inconsistent effective route"
        )
    if status == "ngram_no_accept_fallback" and (attempts <= 0 or accepted != 0):
        raise ArtifactCheckError(
            f"{artifact_path} claims n-gram no-accept fallback with inconsistent telemetry"
        )
    if (
        status == "ngram_no_accept_fallback"
        and effective_route is not None
        and effective_route != "ngram_attempted_no_accept_fallback"
    ):
        raise ArtifactCheckError(
            f"{artifact_path} claims n-gram no-accept fallback with inconsistent effective route"
        )
    if status == "ngram_no_observed_draft_path" and (
        not no_decode_steps or attempts != 0 or accepted != 0
    ):
        raise ArtifactCheckError(
            f"{artifact_path} claims no observed n-gram draft path with inconsistent telemetry"
        )


def validate_delegated_metrics_if_present(
    *, artifact_path: Path, row: dict[str, Any], require_phase0: bool
) -> None:
    if not require_phase0:
        return
    engine = str(row.get("engine", ""))
    delegated = row.get("delegated_backend") == "llama.cpp" or engine.startswith(
        "llama_cpp"
    )
    has_legacy_metrics = isinstance(row.get("llama_cpp_delegated_metrics"), dict)
    has_current_metadata = isinstance(row.get("llama_cpp"), dict)
    if delegated and not (has_legacy_metrics or has_current_metadata):
        raise ArtifactCheckError(
            f"{artifact_path} llama.cpp delegated row lacks metrics"
        )


def assert_display_matches(metric: ReadmeMetric, artifact_row: ArtifactRow) -> None:
    actual = metric_median(artifact_row.row, metric.table)
    if abs(actual - metric.displayed_value) > 0.051:
        raise ArtifactCheckError(
            f"{metric.table} README value mismatch for {metric.model} "
            f"{metric.quantization} prompt={metric.prompt_tokens} "
            f"{metric.column}: README={metric.displayed_value:.1f} "
            f"artifact median={actual:.3f} ({artifact_row.artifact_path})"
        )


def assert_display_delta_matches(
    metric: ReadmeMetric,
    artifact_row: ArtifactRow,
    artifact_rows: dict[tuple[str, str, int, int, str, str], ArtifactRow],
) -> None:
    if metric.displayed_delta_percent is None:
        return
    reference_row = artifact_rows.get(
        (
            metric.model,
            metric.quantization,
            metric.prompt_tokens,
            artifact_row.generation_tokens,
            "mlx_lm",
            metric.table,
        )
    )
    if reference_row is None:
        raise ArtifactCheckError(
            f"README {metric.table} delta has no mlx_lm reference: "
            f"{metric.model} {metric.quantization} prompt={metric.prompt_tokens} "
            f"generation_tokens={artifact_row.generation_tokens} engine={metric.engine}"
        )
    reference = metric_median(reference_row.row, metric.table)
    if reference == 0:
        raise ArtifactCheckError(
            f"README {metric.table} delta has zero mlx_lm reference: "
            f"{metric.model} {metric.quantization} prompt={metric.prompt_tokens}"
        )
    actual = metric_median(artifact_row.row, metric.table)
    expected = (actual - reference) / reference * 100.0
    if abs(expected - metric.displayed_delta_percent) > 0.06:
        raise ArtifactCheckError(
            f"{metric.table} README percentage mismatch for {metric.model} "
            f"{metric.quantization} prompt={metric.prompt_tokens} "
            f"{metric.column}: README={metric.displayed_delta_percent:.1f}% "
            f"artifact={expected:.3f}% ({artifact_row.artifact_path}; "
            f"reference={reference_row.artifact_path})"
        )


def remap_legacy_mlx_inference_path(repo_root: Path, path: Path) -> Path | None:
    parts = path.parts
    prefix_len = len(LEGACY_MLX_INFERENCE_PATH_PREFIX)
    for start in range(0, len(parts) - prefix_len + 1):
        if parts[start : start + prefix_len] != LEGACY_MLX_INFERENCE_PATH_PREFIX:
            continue
        candidate = repo_root.joinpath(
            *CATEGORIZED_MLX_INFERENCE_PATH_PREFIX,
            *parts[start + prefix_len :],
        )
        if candidate.exists():
            return candidate
    return None


def resolve_repo_path(repo_root: Path, path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        candidate = repo_root / path
        if candidate.exists():
            return candidate
        return remap_legacy_mlx_inference_path(repo_root, path) or candidate
    if path.exists():
        return path
    for marker in ("benchmarks", "docs", "scripts"):
        if marker in path.parts:
            marker_index = path.parts.index(marker)
            candidate = repo_root.joinpath(*path.parts[marker_index:])
            if candidate.exists():
                return candidate
            remapped = remap_legacy_mlx_inference_path(repo_root, candidate)
            if remapped is not None:
                return remapped
    return remap_legacy_mlx_inference_path(repo_root, path) or path


def validate_prompt_artifact(
    *,
    repo_root: Path,
    artifact_path: Path,
    prompt_doc: dict[str, Any],
    prompt_tokens: int,
    generation_tokens: int,
) -> str:
    token_path = resolve_repo_path(repo_root, str(prompt_doc.get("token_ids_path", "")))
    if not token_path.exists():
        raise ArtifactCheckError(f"missing prompt-token artifact: {token_path}")
    payload = json.loads(token_path.read_text())
    if payload.get("schema_version") != "ax.mlx_reference_prompt.v1":
        raise ArtifactCheckError(f"invalid prompt artifact schema: {token_path}")
    if int(payload.get("prompt_tokens", -1)) != prompt_tokens:
        raise ArtifactCheckError(f"prompt token count mismatch in {token_path}")
    if int(payload.get("generation_tokens", -1)) != generation_tokens:
        raise ArtifactCheckError(f"generation token count mismatch in {token_path}")
    tokens = payload.get("token_ids")
    if not isinstance(tokens, list):
        raise ArtifactCheckError(f"prompt artifact has no token_ids list: {token_path}")
    prompt_hash = str(prompt_doc.get("token_ids_sha256") or prompt_doc.get("sha256"))
    if payload.get("sha256") != prompt_hash:
        raise ArtifactCheckError(f"prompt hash mismatch in {token_path}")
    if token_sha256([int(token) for token in tokens]) != prompt_hash:
        raise ArtifactCheckError(
            f"prompt token hash does not match payload: {token_path}"
        )
    return prompt_hash


def prompt_contract_by_shape(
    *,
    repo_root: Path,
    artifact_path: Path,
    artifact: dict[str, Any],
) -> dict[tuple[int, int], str]:
    contract = artifact.get("reference_contract", {}).get("prompt_contract", {})
    prompt_docs = contract.get("artifacts")
    if not isinstance(prompt_docs, list):
        raise ArtifactCheckError(f"{artifact_path} lacks prompt contract artifacts")
    hashes: dict[tuple[int, int], str] = {}
    for prompt_doc in prompt_docs:
        prompt_tokens = int(prompt_doc.get("prompt_tokens", -1))
        generation_tokens = int(prompt_doc.get("generation_tokens", -1))
        prompt_hash = validate_prompt_artifact(
            repo_root=repo_root,
            artifact_path=artifact_path,
            prompt_doc=prompt_doc,
            prompt_tokens=prompt_tokens,
            generation_tokens=generation_tokens,
        )
        hashes[(prompt_tokens, generation_tokens)] = prompt_hash
    return hashes


def validate_artifact_row(
    *,
    artifact_path: Path,
    artifact: dict[str, Any],
    row: dict[str, Any],
    prompt_hashes: dict[tuple[int, int], str],
) -> None:
    engine = row.get("engine")
    require_phase0 = phase0_claim_gate_enabled(artifact)
    prompt_tokens = int(row.get("prompt_tokens", -1))
    generation_tokens = int(row.get("generation_tokens", -1))
    shape = (prompt_tokens, generation_tokens)
    expected_hash = prompt_hashes.get(shape)
    if expected_hash is None:
        raise ArtifactCheckError(f"{artifact_path} row has no prompt contract: {shape}")
    if row.get("prompt_token_ids_sha256") != expected_hash:
        raise ArtifactCheckError(
            f"{artifact_path} {engine} prompt={prompt_tokens} has stale prompt hash"
        )
    if int(row.get("batch_size", -1)) != 1:
        raise ArtifactCheckError(
            f"{artifact_path} {engine} prompt={prompt_tokens} is not batch=1"
        )
    if int(row.get("prefill_step_size", -1)) != int(
        artifact.get("prefill_step_size", -2)
    ):
        raise ArtifactCheckError(
            f"{artifact_path} {engine} prompt={prompt_tokens} has mismatched prefill_step_size"
        )
    trials = row.get("trials")
    required_trials = required_trial_count_for_row(artifact=artifact, row=row)
    if not isinstance(trials, list) or len(trials) < required_trials:
        raise ArtifactCheckError(
            f"{artifact_path} {engine} prompt={prompt_tokens} lacks repetition trials"
        )

    if engine == "mlx_lm":
        baseline = row.get("baseline", {})
        if (
            row.get("method") != "mlx_lm.benchmark"
            or baseline.get("role") != "primary_reference"
        ):
            raise ArtifactCheckError(
                f"{artifact_path} mlx_lm row lacks primary reference identity"
            )
    elif engine in AX_DIRECT_ENGINE_KEYS:
        if row.get("ax_decode_policy") != "direct_no_ngram_acceleration":
            raise ArtifactCheckError(
                f"{artifact_path} direct AX row lacks direct policy"
            )
        if (
            require_phase0
            and row.get("ax_decode_claim_status") not in AX_DIRECT_CLAIM_STATUSES
        ):
            raise ArtifactCheckError(
                f"{artifact_path} direct AX row lacks claim status"
            )
        validate_ax_prefill_decode_split(
            artifact_path=artifact_path,
            row=row,
            require_phase0=require_phase0,
        )
        validate_run_stability_if_present(
            artifact_path=artifact_path,
            row=row,
            require_phase0=require_phase0,
        )
        validate_direct_hotpath_no_hidden_fallbacks(
            artifact_path=artifact_path,
            row=row,
            require_phase0=require_phase0,
            model_repo_id=artifact.get("model_repo_id", ""),
        )
        validate_direct_cpp_linear_attention_input_summary(
            artifact_path=artifact_path,
            row=row,
            require_phase0=require_phase0,
        )
        validate_direct_cpp_linear_attention_post_input_summary(
            artifact_path=artifact_path,
            row=row,
            require_phase0=require_phase0,
        )
        if require_phase0:
            telemetry = row.get("ngram_acceleration_telemetry")
            if not isinstance(telemetry, dict):
                raise ArtifactCheckError(
                    f"{artifact_path} direct AX row lacks n-gram telemetry"
                )
            if int(telemetry.get("ax_ngram_draft_attempts", 0)) != 0:
                raise ArtifactCheckError(
                    f"{artifact_path} direct AX row has draft attempts"
                )
    elif engine == "ax_engine_mlx_ngram_accel":
        if not str(row.get("ax_decode_policy", "")).startswith("ngram_acceleration"):
            raise ArtifactCheckError(f"{artifact_path} n-gram row lacks n-gram policy")
        if (
            require_phase0
            and row.get("ax_decode_claim_status") not in AX_NGRAM_CLAIM_STATUSES
        ):
            raise ArtifactCheckError(f"{artifact_path} n-gram row lacks claim status")
        validate_ax_prefill_decode_split(
            artifact_path=artifact_path,
            row=row,
            require_phase0=require_phase0,
        )
        validate_run_stability_if_present(
            artifact_path=artifact_path,
            row=row,
            require_phase0=require_phase0,
        )
        validate_ngram_claim_telemetry(
            artifact_path=artifact_path,
            row=row,
            require_phase0=require_phase0,
        )
    elif engine in AX_MTP_ENGINE_KEYS:
        validate_phase0_runtime_identity(artifact_path=artifact_path, row=row)
        validate_ax_prefill_decode_split(
            artifact_path=artifact_path,
            row=row,
            require_phase0=require_phase0,
        )
        validate_run_stability_if_present(
            artifact_path=artifact_path,
            row=row,
            require_phase0=require_phase0,
        )
        validate_mtp_row_contract(
            artifact_path=artifact_path,
            row=row,
            require_phase0=require_phase0,
        )


def required_trial_count_for_row(
    *, artifact: dict[str, Any], row: dict[str, Any]
) -> int:
    artifact_repetitions = max(1, int(artifact.get("repetitions", 0)))
    engine = row.get("engine")
    if row_uses_reused_reference_source(artifact=artifact, row=row):
        return min(artifact_repetitions, REUSED_REFERENCE_MIN_REPETITIONS)
    return artifact_repetitions


def row_uses_reused_reference_source(
    *, artifact: dict[str, Any], row: dict[str, Any]
) -> bool:
    if row.get("engine") != "mlx_lm":
        return False
    ax_only_refresh = artifact.get("ax_only_refresh")
    return bool(
        artifact.get("reference_results_source")
        or (
            isinstance(ax_only_refresh, dict)
            and ax_only_refresh.get("reference_results_source")
        )
    )


def collect_artifact_rows(
    repo_root: Path,
    artifact_dir: Path,
    include_engines: frozenset[str] | None = None,
    include_tables: frozenset[str] | None = None,
    include_prompt_tokens: frozenset[int] | None = None,
    needed_labels: frozenset[tuple[str, str]] | None = None,
) -> dict[tuple[str, str, int, int, str, str], ArtifactRow]:
    if not artifact_dir.exists():
        raise ArtifactCheckError(f"artifact directory does not exist: {artifact_dir}")
    rows: dict[tuple[str, str, int, int, str, str], ArtifactRow] = {}
    table_names = include_tables or README_METRIC_TABLES
    json_paths = sorted(artifact_dir.glob("*.json"))
    if not json_paths:
        raise ArtifactCheckError(
            f"artifact directory has no JSON artifacts: {artifact_dir}"
        )

    for path in json_paths:
        label = ARTIFACT_LABELS.get(path.stem)
        if label is None:
            continue
        if needed_labels is not None and label not in needed_labels:
            continue
        artifact = json.loads(path.read_text())
        if artifact.get("schema_version") != MLX_INFERENCE_STACK_SCHEMA_VERSION:
            raise ArtifactCheckError(f"{path} has unexpected schema_version")
        validate_build_provenance(artifact_path=path, artifact=artifact)
        validate_host_performance_conditions(artifact_path=path, artifact=artifact)
        validate_benchmark_window(artifact_path=path, artifact=artifact)
        validate_public_claim_evidence(artifact_path=path, artifact=artifact)
        validate_run_stability_summary_if_present(
            artifact_path=path,
            artifact=artifact,
        )
        validate_ax_only_refresh_regression_summary_if_present(
            repo_root=repo_root,
            artifact_path=path,
            artifact=artifact,
        )
        validate_phase0_artifact_gate(artifact_path=path, artifact=artifact)
        model, quantization = label
        prompt_hashes = prompt_contract_by_shape(
            repo_root=repo_root,
            artifact_path=path,
            artifact=artifact,
        )
        seen_reference_shapes: set[tuple[int, int]] = set()
        for row in artifact.get("results", []):
            if not isinstance(row, dict):
                if (
                    phase0_claim_gate_enabled(artifact)
                    or "run_stability_summary" in artifact
                ):
                    raise ArtifactCheckError(f"{path} has non-object result row")
                continue
            validate_delegated_metrics_if_present(
                artifact_path=path,
                row=row,
                require_phase0=phase0_claim_gate_enabled(artifact),
            )
            engine = row.get("engine")
            if engine not in AX_OWNED_MLX_ROW_ENGINE_KEYS:
                if (
                    isinstance(engine, str)
                    and engine.startswith("ax_engine")
                    and (
                        phase0_claim_gate_enabled(artifact)
                        or "run_stability_summary" in artifact
                    )
                ):
                    raise ArtifactCheckError(
                        f"{path} has unvalidated AX row engine: {engine}"
                    )
                continue
            if engine == "mlx_lm":
                seen_reference_shapes.add(
                    (int(row["prompt_tokens"]), int(row["generation_tokens"]))
                )
            prompt_tokens = int(row["prompt_tokens"])
            validate_artifact_row(
                artifact_path=path,
                artifact=artifact,
                row=row,
                prompt_hashes=prompt_hashes,
            )
            if include_engines is not None and engine not in include_engines:
                continue
            if (
                include_prompt_tokens is not None
                and prompt_tokens not in include_prompt_tokens
            ):
                continue
            generation_tokens = int(row["generation_tokens"])
            for table_name in table_names:
                key = (
                    model,
                    quantization,
                    prompt_tokens,
                    generation_tokens,
                    str(engine),
                    table_name,
                )
                if key in rows:
                    raise ArtifactCheckError(
                        f"{path} has duplicate artifact row for {key}"
                    )
                rows[key] = ArtifactRow(
                    artifact_path=path,
                    model=model,
                    quantization=quantization,
                    prompt_tokens=prompt_tokens,
                    generation_tokens=generation_tokens,
                    engine=str(engine),
                    row=row,
                )
        missing_references = set(prompt_hashes) - seen_reference_shapes
        requires_reference_rows = include_engines is None or "mlx_lm" in include_engines
        if requires_reference_rows and missing_references:
            raise ArtifactCheckError(
                f"{path} lacks mlx_lm rows for {sorted(missing_references)}"
            )
    return rows


def collect_artifact_rows_from_sources(
    repo_root: Path,
    sources: list[ArtifactSource],
    needed_labels: frozenset[tuple[str, str]] | None = None,
) -> dict[tuple[str, str, int, int, str, str], ArtifactRow]:
    rows: dict[tuple[str, str, int, int, str, str], ArtifactRow] = {}
    for source in sources:
        source_rows = collect_artifact_rows(
            repo_root,
            source.artifact_dir,
            include_engines=source.include_engines,
            include_tables=source.include_tables,
            include_prompt_tokens=source.include_prompt_tokens,
            needed_labels=needed_labels,
        )
        for key, row in source_rows.items():
            merge_artifact_row(rows, key, row)
    return rows


def merge_artifact_row(
    rows: dict[tuple[str, str, int, int, str, str], ArtifactRow],
    key: tuple[str, str, int, int, str, str],
    row: ArtifactRow,
) -> None:
    existing = rows.get(key)
    if existing is None:
        rows[key] = row
        return

    _model, _quantization, _prompt_tokens, _generation_tokens, engine, table = key
    if engine not in AX_ENGINE_ROWS:
        rows[key] = row
        return

    candidate = metric_median(row.row, table)
    previous = metric_median(existing.row, table)
    if not metric_is_regressed(table, candidate, previous):
        rows[key] = row


def find_artifact_row_for_metric(
    artifact_rows: dict[tuple[str, str, int, int, str, str], ArtifactRow],
    metric: ReadmeMetric,
) -> ArtifactRow | None:
    if metric.generation_tokens is not None:
        return artifact_rows.get(
            (
                metric.model,
                metric.quantization,
                metric.prompt_tokens,
                metric.generation_tokens,
                metric.engine,
                metric.table,
            )
        )
    candidates = [
        row
        for (
            model,
            quantization,
            prompt_tokens,
            _generation_tokens,
            engine,
            table_name,
        ), row in artifact_rows.items()
        if (
            model == metric.model
            and quantization == metric.quantization
            and prompt_tokens == metric.prompt_tokens
            and engine == metric.engine
            and table_name == metric.table
        )
    ]
    if len(candidates) <= 1:
        return candidates[0] if candidates else None
    generation_values = sorted({row.generation_tokens for row in candidates})
    raise ArtifactCheckError(
        f"README {metric.table} row is ambiguous for {metric.model} "
        f"{metric.quantization} prompt={metric.prompt_tokens} engine={metric.engine}; "
        f"artifact has generation tokens {generation_values}"
    )


def artifact_source_for_marker_entry(entry: ArtifactMarkerEntry) -> ArtifactSource:
    include_tables = frozenset({"decode"}) if entry.kind == "ax-decode-overlay" else None
    return ArtifactSource(
        artifact_dir=entry.artifact_dir,
        include_engines=AX_DIRECT_ROW,
        include_tables=include_tables,
        include_prompt_tokens=entry.include_prompt_tokens,
    )


def metric_is_regressed(table: str, current: float, previous: float) -> bool:
    if table == "ttft":
        return current > previous + 1.0e-9
    return current + 1.0e-9 < previous


def metric_improvement_ratio(table: str, current: float, previous: float) -> float:
    if previous == 0:
        return 0.0
    if table == "ttft":
        return (previous - current) / previous
    return (current - previous) / previous


def assert_published_ax_rows_not_below_prior_record(
    *,
    repo_root: Path,
    readme_path: Path,
    needed_labels: frozenset[tuple[str, str]],
) -> None:
    entries = readme_performance_artifact_marker_entries(readme_path)
    ax_entries = [
        entry
        for entry in entries
        if entry.kind in {"ax", "ax-base", "ax-overlay", "ax-decode-overlay"}
    ]
    if len(ax_entries) < 2:
        return
    prior_entries = ax_entries[:-1]
    prior_rows: dict[tuple[str, str, int, int, str, str], ArtifactRow] = {}
    for entry in prior_entries:
        rows = collect_artifact_rows_from_sources(
            repo_root,
            [artifact_source_for_marker_entry(entry)],
            needed_labels=needed_labels,
        )
        for key, row in rows.items():
            existing = prior_rows.get(key)
            if existing is None:
                prior_rows[key] = row
                continue
            candidate = metric_median(row.row, key[-1])
            previous = metric_median(existing.row, key[-1])
            if metric_is_regressed(key[-1], previous, candidate):
                prior_rows[key] = row

    published_rows = collect_artifact_rows_from_sources(
        repo_root,
        [artifact_source_for_marker_entry(entry) for entry in ax_entries],
        needed_labels=needed_labels,
    )
    regressions: list[str] = []
    for key, current_row in sorted(published_rows.items()):
        previous_row = prior_rows.get(key)
        if previous_row is None:
            continue
        table = key[-1]
        current = metric_median(current_row.row, table)
        previous = metric_median(previous_row.row, table)
        if not metric_is_regressed(table, current, previous):
            continue
        model, quantization, prompt_tokens, generation_tokens, _engine, _table = key
        delta = metric_improvement_ratio(table, current, previous) * 100.0
        unit = "ms" if table == "ttft" else "tok/s"
        regressions.append(
            f"{table} {model} {quantization} prompt={prompt_tokens} "
            f"gen={generation_tokens}: current={current:.3f} {unit} "
            f"previous={previous:.3f} {unit} ({delta:.2f}%) "
            f"current={current_row.artifact_path} previous={previous_row.artifact_path}"
        )
    if regressions:
        sample = "; ".join(regressions[:5])
        extra = "" if len(regressions) <= 5 else f"; and {len(regressions) - 5} more"
        raise ArtifactCheckError(
            "published AX rows regress prior README AX high-water records: "
            f"{sample}{extra}"
        )


def check_readme_performance(
    *,
    repo_root: Path,
    readme_path: Path,
    artifact_dir: Path | None = None,
    expected_metric_count: int | None = 207,
    require_condition_metadata: bool = False,
) -> list[str]:
    return check_readme_performance_summary(
        repo_root=repo_root,
        readme_path=readme_path,
        artifact_dir=artifact_dir,
        expected_metric_count=expected_metric_count,
        require_condition_metadata=require_condition_metadata,
    ).metric_checks


def check_readme_performance_summary(
    *,
    repo_root: Path,
    readme_path: Path,
    artifact_dir: Path | None = None,
    expected_metric_count: int | None = 207,
    require_condition_metadata: bool = False,
) -> ReadmeCheckResult:
    resolved_readme = readme_path.resolve()
    artifact_sources = (
        [ArtifactSource(artifact_dir.resolve())]
        if artifact_dir is not None
        else default_artifact_sources(resolved_readme)
    )
    hot_prefix_artifact_paths = default_hot_prefix_artifact_paths(resolved_readme)
    long_context_artifact_paths = default_long_context_boundary_artifact_paths(
        resolved_readme
    )
    concurrent_prefill_artifact_paths = (
        default_concurrent_prefill_boundary_artifact_paths(resolved_readme)
    )
    metrics = parse_readme_metrics(resolved_readme)
    needed_labels = frozenset((metric.model, metric.quantization) for metric in metrics)
    condition_metadata_checks, condition_metadata_gaps = (
        collect_condition_metadata_checks(
            artifact_sources=artifact_sources,
            needed_labels=needed_labels,
        )
    )
    if require_condition_metadata:
        assert_condition_metadata_complete(condition_metadata_gaps)
    artifact_rows = collect_artifact_rows_from_sources(
        repo_root.resolve(),
        artifact_sources,
        needed_labels=needed_labels,
    )
    if artifact_dir is None:
        assert_published_ax_rows_not_below_prior_record(
            repo_root=repo_root.resolve(),
            readme_path=resolved_readme,
            needed_labels=needed_labels,
        )
    narrative_claim_checks = [
        *validate_readme_hot_prefix_claims(
            readme_path=resolved_readme,
            artifact_paths=hot_prefix_artifact_paths,
        ),
        *validate_readme_boundary_claims(
            readme_path=resolved_readme,
            long_context_artifact_paths=long_context_artifact_paths,
            concurrent_prefill_artifact_paths=concurrent_prefill_artifact_paths,
        ),
        *validate_readme_mtp_6bit_claims(readme_path=resolved_readme),
        *validate_readme_direct_generation_claims(readme_path=resolved_readme),
    ]
    checked: list[str] = []

    for metric in metrics:
        artifact_row = find_artifact_row_for_metric(artifact_rows, metric)
        if artifact_row is None:
            raise ArtifactCheckError(
                f"README {metric.table} row has no artifact: "
                f"{metric.model} {metric.quantization} prompt={metric.prompt_tokens} "
                f"generation_tokens={metric.generation_tokens} engine={metric.engine}"
            )
        if metric.engine in AX_ENGINE_ROWS:
            validate_ax_prefill_work_contract(
                artifact_path=artifact_row.artifact_path,
                row=artifact_row.row,
            )
        assert_display_matches(metric, artifact_row)
        assert_display_delta_matches(metric, artifact_row, artifact_rows)
        checked.append(
            f"{metric.table}:{metric.model}:{metric.quantization}:"
            f"{metric.prompt_tokens}:{metric.engine}"
        )

    if expected_metric_count is not None and len(checked) != expected_metric_count:
        raise ArtifactCheckError(
            f"checked {len(checked)} README metrics, expected {expected_metric_count}"
        )
    return ReadmeCheckResult(
        metric_checks=checked,
        narrative_claim_checks=narrative_claim_checks,
        condition_metadata_checks=condition_metadata_checks,
    )


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Validate README performance table values and provenance artifacts."
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument(
        "--readme",
        type=Path,
        default=repo_root / "docs" / "PERFORMANCE-RESULTS.md",
        help=(
            "Public results document with performance tables/charts "
            "(default: docs/PERFORMANCE-RESULTS.md)."
        ),
    )
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument(
        "--require-condition-metadata",
        action="store_true",
        help=(
            "Fail when README MLX inference artifacts lack host performance "
            "conditions or benchmark condition windows."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        checked = check_readme_performance_summary(
            repo_root=args.repo_root,
            readme_path=args.readme,
            artifact_dir=args.artifact_dir,
            require_condition_metadata=args.require_condition_metadata,
        )
    except ArtifactCheckError as error:
        print(f"README performance artifact check failed: {error}", file=sys.stderr)
        return 1
    print(
        "README performance artifact check passed: "
        f"{len(checked.metric_checks)} metrics validated; "
        f"{len(checked.narrative_claim_checks)} narrative claims validated; "
        f"{len(checked.condition_metadata_checks)} condition metadata artifacts "
        "validated"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
