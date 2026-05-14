#!/usr/bin/env python3
"""Build long-context comparison artifacts from MLX inference-stack runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from check_long_context_comparison_artifact import (
    SCHEMA_VERSION,
    LongContextComparisonArtifactError,
    validate_long_context_comparison_artifact,
)


SOURCE_SCHEMA_VERSION = "ax.mlx_inference_stack.v2"
COMPARISON_ENGINES = {"mlx_lm", "ax_engine_mlx", "llama_cpp_metal"}


class LongContextComparisonBuildError(RuntimeError):
    pass


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise LongContextComparisonBuildError(f"{path} is not valid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise LongContextComparisonBuildError(f"{path} must contain a JSON object")
    return payload


def metric_median(row: dict[str, Any], key: str, *, owner: str) -> float:
    metric = row.get(key)
    if not isinstance(metric, dict):
        raise LongContextComparisonBuildError(f"{owner} lacks metric object {key!r}")
    value = metric.get("median")
    if not isinstance(value, (int, float)) or float(value) <= 0.0:
        raise LongContextComparisonBuildError(f"{owner}.{key} lacks positive median")
    return float(value)


def copy_metric(row: dict[str, Any], key: str, *, owner: str) -> dict[str, float]:
    metric = row.get(key)
    if not isinstance(metric, dict):
        raise LongContextComparisonBuildError(f"{owner} lacks metric object {key!r}")
    copied: dict[str, float] = {}
    for stat in ("mean", "median", "p75", "min", "max"):
        value = metric.get(stat)
        if isinstance(value, (int, float)):
            copied[stat] = float(value)
    if "median" not in copied:
        raise LongContextComparisonBuildError(f"{owner}.{key} lacks median")
    return copied


def optional_metric(row: dict[str, Any], key: str, *, owner: str) -> dict[str, float] | None:
    if not isinstance(row.get(key), dict):
        return None
    return copy_metric(row, key, owner=owner)


def derive_ttft_metric(row: dict[str, Any], *, owner: str) -> tuple[dict[str, float], str]:
    ttft = row.get("ttft_ms")
    if isinstance(ttft, dict) and isinstance(ttft.get("median"), (int, float)):
        return copy_metric(row, "ttft_ms", owner=owner), str(
            row.get("ttft_source", "reported_ttft_ms")
        )

    prefill_s = row.get("prefill_s")
    if isinstance(prefill_s, dict) and isinstance(prefill_s.get("median"), (int, float)):
        derived = {
            stat: float(value) * 1000.0
            for stat, value in prefill_s.items()
            if stat in {"mean", "median", "p75", "min", "max"}
            and isinstance(value, (int, float))
        }
        if "median" in derived:
            return derived, "derived_from_prefill_s"

    prompt_tokens = int(row.get("prompt_tokens", 0))
    prefill_tok_s = metric_median(row, "prefill_tok_s", owner=owner)
    return (
        {"median": prompt_tokens / prefill_tok_s * 1000.0},
        "derived_from_prefill_tok_s",
    )


def source_model(source: dict[str, Any]) -> dict[str, Any]:
    model: dict[str, Any] = {"id": str(source.get("model", ""))}
    if source.get("model_dir"):
        model["dir"] = str(source["model_dir"])
    config = source.get("model_config")
    if isinstance(config, dict):
        for key in ("model_type", "model_family", "quantization"):
            if key in config:
                model[key] = config[key]
    return model


def build_rows(
    source_path: Path,
    source: dict[str, Any],
    *,
    min_context_tokens: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(source.get("results", [])):
        if not isinstance(row, dict):
            continue
        engine = str(row.get("engine", ""))
        if engine not in COMPARISON_ENGINES:
            continue
        context_tokens = int(row.get("prompt_tokens", 0))
        if context_tokens < min_context_tokens:
            continue

        owner = f"{source_path}.results[{index}]"
        ttft_ms, ttft_source = derive_ttft_metric(row, owner=owner)
        built: dict[str, Any] = {
            "engine": engine,
            "context_tokens": context_tokens,
            "generation_tokens": int(row.get("generation_tokens", 0)),
            "repetitions": int(row.get("repetitions") or source.get("repetitions") or 0),
            "prefill_tok_s": copy_metric(row, "prefill_tok_s", owner=owner),
            "ttft_ms": ttft_ms,
            "ttft_source": ttft_source,
            "source_engine": engine,
            "timing_scope": str(row.get("timing_scope", "")),
        }
        prompt_hash = row.get("prompt_token_ids_sha256")
        if isinstance(prompt_hash, str):
            built["prompt_token_ids_sha256"] = prompt_hash
        if row.get("prompt_token_ids_path"):
            built["prompt_token_ids_path"] = row["prompt_token_ids_path"]
        decode_metric = optional_metric(row, "decode_tok_s", owner=owner)
        if decode_metric is not None:
            built["decode_tok_s"] = decode_metric
        memory_metric = optional_metric(row, "peak_memory_gb", owner=owner)
        if memory_metric is not None:
            built["peak_memory_gb"] = memory_metric

        if engine == "mlx_lm":
            built["comparison_role"] = "primary_mlx_reference"
            built["baseline"] = {"role": "primary_reference", "method": "mlx_lm.benchmark"}
            built["prompt_contract"] = row.get("prompt_contract", "mlx_lm_random_tokens_seed_0")
        elif engine == "ax_engine_mlx":
            built["comparison_role"] = "repo_owned_ax_direct"
            built["ax_decode_policy"] = row.get("ax_decode_policy", "direct_no_ngram_acceleration")
            built["route"] = row.get("route") if isinstance(row.get("route"), dict) else {
                "selected_backend": "mlx",
                "source": "inferred_from_ax_engine_mlx_row",
            }
            built["prompt_contract"] = row.get("prompt_contract", "mlx_lm_random_tokens_seed_0")
        elif engine == "llama_cpp_metal":
            built["comparison_role"] = "shape_compatible_external_gguf"
            built["prompt_contract"] = row.get(
                "prompt_contract", "shape_compatible_llama_bench_internal_tokens"
            )
            built["external_baseline_role"] = row.get(
                "external_baseline_role", "gguf_non_mlx_metal_reference"
            )
            built["claim_boundary"] = row.get(
                "claim_boundary",
                "Shape-compatible external GGUF baseline, not prompt-hash parity evidence.",
            )
            if isinstance(row.get("llama_cpp"), dict):
                built["llama_cpp"] = row["llama_cpp"]
            if row.get("gguf_model"):
                built["gguf_model"] = row["gguf_model"]
        rows.append(built)
    return rows


def attach_ratios(rows: list[dict[str, Any]]) -> None:
    baselines = {
        (int(row["context_tokens"]), int(row["generation_tokens"])): row
        for row in rows
        if row.get("engine") == "mlx_lm"
    }
    for row in rows:
        if row.get("engine") == "mlx_lm":
            continue
        key = (int(row["context_tokens"]), int(row["generation_tokens"]))
        baseline = baselines.get(key)
        if baseline is None:
            raise LongContextComparisonBuildError(
                "missing mlx_lm baseline for "
                f"context_tokens={key[0]} generation_tokens={key[1]}"
            )
        row["ratios_to_mlx_lm"] = {
            "prefill_tok_s": metric_median(row, "prefill_tok_s", owner="row")
            / metric_median(baseline, "prefill_tok_s", owner="baseline"),
            "ttft_ms": metric_median(row, "ttft_ms", owner="row")
            / metric_median(baseline, "ttft_ms", owner="baseline"),
        }


def build_long_context_comparison_artifact(
    source_path: Path,
    *,
    min_context_tokens: int = 1024,
) -> dict[str, Any]:
    source = load_json(source_path)
    if source.get("schema_version") != SOURCE_SCHEMA_VERSION:
        raise LongContextComparisonBuildError(
            f"{source_path} has schema_version={source.get('schema_version')!r}, "
            f"expected {SOURCE_SCHEMA_VERSION}"
        )
    rows = build_rows(source_path, source, min_context_tokens=min_context_tokens)
    if not rows:
        raise LongContextComparisonBuildError(
            f"{source_path} has no comparison rows at or above {min_context_tokens} tokens"
        )
    attach_ratios(rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "claim_scope": "long_context_cold_prefill_comparison",
        "source_artifact": str(source_path),
        "source_schema_version": SOURCE_SCHEMA_VERSION,
        "host": source.get("host", {}),
        "model": source_model(source),
        "benchmark": {
            "batch_size": 1,
            "temperature": 0.0,
            "prefill_step_size": int(source.get("prefill_step_size", 0)),
            "repetitions": int(source.get("repetitions", 0)),
        },
        "comparison_contract": {
            "parity_engines": ["mlx_lm", "ax_engine_mlx"],
            "shape_compatible_external_engines": ["llama_cpp_metal"],
            "llama_cpp_boundary": (
                "llama.cpp Metal rows are external GGUF llama-bench rows. "
                "They are shape-compatible but not prompt-hash parity evidence."
            ),
        },
        "rows": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build ax.long_context_comparison.v1 from ax.mlx_inference_stack.v2."
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-context-tokens", type=int, default=1024)
    parser.add_argument("--require-llama-cpp", action="store_true")
    parser.add_argument("--skip-validate", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        artifact = build_long_context_comparison_artifact(
            args.source,
            min_context_tokens=args.min_context_tokens,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
        if not args.skip_validate:
            validate_long_context_comparison_artifact(
                args.output,
                require_llama_cpp=args.require_llama_cpp,
            )
    except (LongContextComparisonBuildError, LongContextComparisonArtifactError) as error:
        print(f"Long-context comparison artifact build failed: {error}", file=sys.stderr)
        return 1
    print(f"Long-context comparison artifact written: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
