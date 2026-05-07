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
    if not isinstance(trials, list) or len(trials) < int(artifact.get("repetitions", 0)):
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
    elif engine == "ax_engine_mlx_ngram_accel":
        if not str(row.get("ax_decode_policy", "")).startswith("ngram_acceleration"):
            raise ArtifactCheckError(f"{artifact_path} n-gram row lacks n-gram policy")
        if row.get("ax_decode_claim_status") not in {
            "ngram_acceleration_effective_throughput",
            "ngram_no_draft_direct_fallback",
        }:
            raise ArtifactCheckError(f"{artifact_path} n-gram row lacks claim status")
        if not isinstance(row.get("ngram_acceleration_telemetry"), dict):
            raise ArtifactCheckError(f"{artifact_path} n-gram row lacks telemetry")


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
    expected_metric_count: int | None = 13 * 2 * (4 + 3),
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
