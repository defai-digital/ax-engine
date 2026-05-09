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
    "gemma-4-e2b-it-4bit": ("Gemma 4 E2B", "4-bit · group=64 · affine"),
    "gemma-4-e2b-it-5bit": ("Gemma 4 E2B", "5-bit · group=64 · affine"),
    "gemma-4-e2b-it-6bit": ("Gemma 4 E2B", "6-bit · group=64 · affine"),
    "gemma-4-e2b-it-8bit": ("Gemma 4 E2B", "8-bit · group=64 · affine"),
    "gemma-4-e4b-it-4bit": ("Gemma 4 E4B", "4-bit · group=64 · affine"),
    "gemma-4-26b-a4b-it-4bit": ("Gemma 4 26B A4B", "4-bit · group=64 · affine"),
    "gemma-4-31b-it-4bit": ("Gemma 4 31B", "4-bit · group=64 · affine"),
    "qwen3_5-9b-mlx-4bit": ("Qwen 3.5 9B", "4-bit · group=64 · affine"),
    "qwen3_6-35b-a3b-ud-mlx-4bit": (
        "Qwen 3.6 35B A3B",
        "UD-MLX 4-bit · group=64 · affine",
    ),
    "qwen3_6-35b-a3b-5bit": (
        "Qwen 3.6 35B A3B",
        "MLX 5-bit · group=64 · affine",
    ),
    "qwen3_6-35b-a3b-6bit": (
        "Qwen 3.6 35B A3B",
        "MLX 6-bit · group=64 · affine",
    ),
    "qwen3_6-35b-a3b-8bit": (
        "Qwen 3.6 35B A3B",
        "MLX 8-bit · group=64 · affine",
    ),
    "qwen3-coder-next-4bit": ("Qwen Coder Next", "4-bit · group=64 · affine"),
    "glm-4.7-flash-4bit": ("GLM 4.7 Flash", "4-bit · group=64 · affine"),
}

DECODE_TABLE_COLUMNS = {
    "mlx_lm": "mlx_lm",
    "mlx_swift_lm": "mlx_swift_lm",
    "ax direct baseline": "ax_engine_mlx",
    "ax default n-gram": "ax_engine_mlx_ngram_accel",
}

PREFILL_TABLE_COLUMNS = {
    "mlx_lm": "mlx_lm",
    "mlx_swift_lm": "mlx_swift_lm",
    "ax engine": "ax_engine_mlx",
}

PHASE0_CLAIM_GATE_SCHEMA_VERSION = "ax.phase0_claim_gate.v1"
REUSED_REFERENCE_MIN_REPETITIONS = 3

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
    column: str
    engine: str
    displayed_value: float


@dataclass(frozen=True)
class ArtifactRow:
    artifact_path: Path
    model: str
    quantization: str
    prompt_tokens: int
    generation_tokens: int
    engine: str
    row: dict[str, Any]


def token_sha256(tokens: list[int]) -> str:
    payload = json.dumps(tokens, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def parse_numeric_cell(cell: str) -> float:
    cleaned = cell.replace("**", "").replace(",", "")
    match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
    if not match:
        raise ArtifactCheckError(f"metric cell has no numeric value: {cell!r}")
    return float(match.group(0))


def is_unavailable_cell(cell: str) -> bool:
    normalized = cell.strip().lower()
    return normalized in {"n/a", "na", "-"}


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


def parse_readme_table(
    readme_text: str,
    *,
    heading_prefix: str,
    table_name: str,
    column_map: dict[str, str],
) -> list[ReadmeMetric]:
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
                    column=headers[normalized_headers.index(header)],
                    engine=engine,
                    displayed_value=parse_numeric_cell(cell),
                )
            )
    return rows


def parse_readme_metrics(readme_path: Path) -> list[ReadmeMetric]:
    text = readme_path.read_text()
    return [
        *parse_readme_table(
            text,
            heading_prefix="### Decode throughput",
            table_name="decode",
            column_map=DECODE_TABLE_COLUMNS,
        ),
        *parse_readme_table(
            text,
            heading_prefix="### Prefill throughput",
            table_name="prefill",
            column_map=PREFILL_TABLE_COLUMNS,
        ),
    ]


def default_artifact_dir(readme_path: Path) -> Path:
    text = readme_path.read_text()
    match = re.search(
        r"`(benchmarks/results/mlx-inference/[^`]+/)`",
        text,
    )
    if not match:
        raise ArtifactCheckError(
            "README does not name a benchmarks/results/mlx-inference artifact directory"
        )
    return (readme_path.parent / match.group(1)).resolve()


def metric_median(row: dict[str, Any], table: str) -> float:
    metric_key = "decode_tok_s" if table == "decode" else "prefill_tok_s"
    metric = row.get(metric_key)
    if not isinstance(metric, dict) or "median" not in metric:
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
    if not isinstance(concurrency, int) or isinstance(concurrency, bool) or concurrency <= 0:
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
        if not isinstance(evidence[counter], int) or isinstance(evidence[counter], bool):
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


def validate_phase0_artifact_gate(
    *, artifact_path: Path, artifact: dict[str, Any]
) -> None:
    if not phase0_claim_gate_enabled(artifact):
        return
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
        raise ArtifactCheckError(f"{artifact_path} {row.get('engine')} lacks {key}.median")


def validate_phase0_runtime_identity(
    *, artifact_path: Path, row: dict[str, Any]
) -> None:
    runtime = row.get("runtime_identity")
    if not isinstance(runtime, dict):
        raise ArtifactCheckError(f"{artifact_path} {row.get('engine')} lacks runtime_identity")
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
    validate_metric_summary(artifact_path=artifact_path, row=row, key="prefill_s")
    validate_metric_summary(artifact_path=artifact_path, row=row, key="decode_s")
    telemetry = row.get("ax_mlx_telemetry")
    if not isinstance(telemetry, dict):
        raise ArtifactCheckError(f"{artifact_path} {row.get('engine')} lacks AX MLX telemetry")
    for key in ("ax_mlx_prefill_steps", "ax_mlx_decode_steps"):
        if key not in telemetry:
            raise ArtifactCheckError(f"{artifact_path} {row.get('engine')} lacks {key}")
    if require_phase0:
        if row.get("timing_scope") != "ax_engine_runner_time_us":
            raise ArtifactCheckError(
                f"{artifact_path} {row.get('engine')} lacks AX runner timing scope"
            )
        if row.get("ttft_source") != "ax_engine_runner_prefill_time":
            raise ArtifactCheckError(
                f"{artifact_path} {row.get('engine')} lacks AX TTFT source"
            )
        validate_phase0_runtime_identity(artifact_path=artifact_path, row=row)


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
    if status == "ngram_no_draft_direct_fallback" and (
        attempts != 0 or fallback_steps <= 0
    ):
        raise ArtifactCheckError(
            f"{artifact_path} claims n-gram fallback without fallback telemetry"
        )


def validate_delegated_metrics_if_present(
    *, artifact_path: Path, row: dict[str, Any], require_phase0: bool
) -> None:
    if not require_phase0:
        return
    engine = str(row.get("engine", ""))
    delegated = row.get("delegated_backend") == "llama.cpp" or engine.startswith("llama_cpp")
    if delegated and not isinstance(row.get("llama_cpp_delegated_metrics"), dict):
        raise ArtifactCheckError(f"{artifact_path} llama.cpp delegated row lacks metrics")


def assert_display_matches(metric: ReadmeMetric, artifact_row: ArtifactRow) -> None:
    actual = metric_median(artifact_row.row, metric.table)
    if abs(actual - metric.displayed_value) > 0.051:
        raise ArtifactCheckError(
            f"{metric.table} README value mismatch for {metric.model} "
            f"{metric.quantization} prompt={metric.prompt_tokens} "
            f"{metric.column}: README={metric.displayed_value:.1f} "
            f"artifact median={actual:.3f} ({artifact_row.artifact_path})"
        )


def resolve_repo_path(repo_root: Path, path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return repo_root / path


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
        raise ArtifactCheckError(f"prompt token hash does not match payload: {token_path}")
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
        raise ArtifactCheckError(f"{artifact_path} {engine} prompt={prompt_tokens} is not batch=1")
    if int(row.get("prefill_step_size", -1)) != int(artifact.get("prefill_step_size", -2)):
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
        if row.get("method") != "mlx_lm.benchmark" or baseline.get("role") != "primary_reference":
            raise ArtifactCheckError(f"{artifact_path} mlx_lm row lacks primary reference identity")
    elif engine == "mlx_swift_lm":
        if row.get("method") != "mlx_swift_lm_benchmark_adapter":
            raise ArtifactCheckError(f"{artifact_path} mlx_swift_lm row lacks adapter method")
        if "BenchmarkHelpers/MLXLMCommon" not in str(row.get("secondary_reference_role", "")):
            raise ArtifactCheckError(f"{artifact_path} mlx_swift_lm row lacks adapter role")
    elif engine == "ax_engine_mlx":
        if row.get("ax_decode_policy") != "direct_no_ngram_acceleration":
            raise ArtifactCheckError(f"{artifact_path} direct AX row lacks direct policy")
        if row.get("ax_decode_claim_status") != "direct_same_policy_baseline":
            raise ArtifactCheckError(f"{artifact_path} direct AX row lacks claim status")
        validate_ax_prefill_decode_split(
            artifact_path=artifact_path,
            row=row,
            require_phase0=require_phase0,
        )
        if require_phase0:
            telemetry = row.get("ngram_acceleration_telemetry")
            if not isinstance(telemetry, dict):
                raise ArtifactCheckError(f"{artifact_path} direct AX row lacks n-gram telemetry")
            if int(telemetry.get("ax_ngram_draft_attempts", 0)) != 0:
                raise ArtifactCheckError(f"{artifact_path} direct AX row has draft attempts")
    elif engine == "ax_engine_mlx_ngram_accel":
        if not str(row.get("ax_decode_policy", "")).startswith("ngram_acceleration"):
            raise ArtifactCheckError(f"{artifact_path} n-gram row lacks n-gram policy")
        if row.get("ax_decode_claim_status") not in {
            "ngram_acceleration_effective_throughput",
            "ngram_no_draft_direct_fallback",
        }:
            raise ArtifactCheckError(f"{artifact_path} n-gram row lacks claim status")
        validate_ax_prefill_decode_split(
            artifact_path=artifact_path,
            row=row,
            require_phase0=require_phase0,
        )
        validate_ngram_claim_telemetry(
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
    if row.get("engine") not in {"mlx_lm", "mlx_swift_lm"}:
        return False
    ax_only_refresh = artifact.get("ax_only_refresh")
    return bool(
        artifact.get("reference_results_source")
        or (
            isinstance(ax_only_refresh, dict)
            and ax_only_refresh.get("reference_results_source")
        )
    )


def collect_artifact_rows(repo_root: Path, artifact_dir: Path) -> dict[tuple[str, str, int, str], ArtifactRow]:
    if not artifact_dir.exists():
        raise ArtifactCheckError(f"artifact directory does not exist: {artifact_dir}")
    rows: dict[tuple[str, str, int, str], ArtifactRow] = {}
    json_paths = sorted(artifact_dir.glob("*.json"))
    if not json_paths:
        raise ArtifactCheckError(f"artifact directory has no JSON artifacts: {artifact_dir}")

    for path in json_paths:
        label = ARTIFACT_LABELS.get(path.stem)
        if label is None:
            continue
        artifact = json.loads(path.read_text())
        if artifact.get("schema_version") != "ax.mlx_inference_stack.v2":
            raise ArtifactCheckError(f"{path} has unexpected schema_version")
        validate_public_claim_evidence(artifact_path=path, artifact=artifact)
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
                continue
            validate_delegated_metrics_if_present(
                artifact_path=path,
                row=row,
                require_phase0=phase0_claim_gate_enabled(artifact),
            )
            engine = row.get("engine")
            if engine not in {
                "mlx_lm",
                "mlx_swift_lm",
                "ax_engine_mlx",
                "ax_engine_mlx_ngram_accel",
            }:
                continue
            validate_artifact_row(
                artifact_path=path,
                artifact=artifact,
                row=row,
                prompt_hashes=prompt_hashes,
            )
            prompt_tokens = int(row["prompt_tokens"])
            generation_tokens = int(row["generation_tokens"])
            if engine == "mlx_lm":
                seen_reference_shapes.add((prompt_tokens, generation_tokens))
            rows[(model, quantization, prompt_tokens, str(engine))] = ArtifactRow(
                artifact_path=path,
                model=model,
                quantization=quantization,
                prompt_tokens=prompt_tokens,
                generation_tokens=generation_tokens,
                engine=str(engine),
                row=row,
            )
        missing_references = set(prompt_hashes) - seen_reference_shapes
        if missing_references:
            raise ArtifactCheckError(f"{path} lacks mlx_lm rows for {sorted(missing_references)}")
    return rows


def check_readme_performance(
    *,
    repo_root: Path,
    readme_path: Path,
    artifact_dir: Path | None = None,
    expected_metric_count: int | None = 196,
) -> list[str]:
    resolved_readme = readme_path.resolve()
    resolved_artifact_dir = (artifact_dir or default_artifact_dir(resolved_readme)).resolve()
    metrics = parse_readme_metrics(resolved_readme)
    artifact_rows = collect_artifact_rows(repo_root.resolve(), resolved_artifact_dir)
    checked: list[str] = []

    for metric in metrics:
        key = (metric.model, metric.quantization, metric.prompt_tokens, metric.engine)
        artifact_row = artifact_rows.get(key)
        if artifact_row is None:
            raise ArtifactCheckError(
                f"README {metric.table} row has no artifact: "
                f"{metric.model} {metric.quantization} prompt={metric.prompt_tokens} "
                f"engine={metric.engine}"
            )
        assert_display_matches(metric, artifact_row)
        checked.append(
            f"{metric.table}:{metric.model}:{metric.quantization}:"
            f"{metric.prompt_tokens}:{metric.engine}"
        )

    if expected_metric_count is not None and len(checked) != expected_metric_count:
        raise ArtifactCheckError(
            f"checked {len(checked)} README metrics, expected {expected_metric_count}"
        )
    return checked


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Validate README performance table values and provenance artifacts."
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--readme", type=Path, default=repo_root / "README.md")
    parser.add_argument("--artifact-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        checked = check_readme_performance(
            repo_root=args.repo_root,
            readme_path=args.readme,
            artifact_dir=args.artifact_dir,
        )
    except ArtifactCheckError as error:
        print(f"README performance artifact check failed: {error}", file=sys.stderr)
        return 1
    print(f"README performance artifact check passed: {len(checked)} metrics validated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
