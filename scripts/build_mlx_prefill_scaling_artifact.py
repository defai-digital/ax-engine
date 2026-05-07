#!/usr/bin/env python3
"""Build long-context prefill/TTFT scaling artifacts from MLX inference runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from check_mlx_prefill_scaling_artifact import (
    SCHEMA_VERSION,
    PrefillScalingArtifactError,
    validate_prefill_scaling_artifact,
)


SOURCE_SCHEMA_VERSION = "ax.mlx_inference_stack.v2"
SCALING_ENGINES = {"mlx_lm", "ax_engine_mlx"}


class PrefillScalingBuildError(RuntimeError):
    pass


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise PrefillScalingBuildError(f"{path} is not valid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise PrefillScalingBuildError(f"{path} must contain a JSON object")
    return payload


def metric_median(row: dict[str, Any], key: str, *, owner: str) -> float:
    metric = row.get(key)
    if not isinstance(metric, dict):
        raise PrefillScalingBuildError(f"{owner} lacks metric object {key!r}")
    value = metric.get("median")
    if not isinstance(value, (int, float)) or float(value) <= 0.0:
        raise PrefillScalingBuildError(f"{owner}.{key} lacks positive median")
    return float(value)


def copy_metric(row: dict[str, Any], key: str, *, owner: str) -> dict[str, float]:
    metric = row.get(key)
    if not isinstance(metric, dict):
        raise PrefillScalingBuildError(f"{owner} lacks metric object {key!r}")
    copied: dict[str, float] = {}
    for stat in ("mean", "median", "p75", "min", "max"):
        value = metric.get(stat)
        if isinstance(value, (int, float)):
            copied[stat] = float(value)
    if "median" not in copied:
        raise PrefillScalingBuildError(f"{owner}.{key} lacks median")
    if key == "peak_memory_gb" and "max" not in copied:
        copied["max"] = copied["median"]
    return copied


def derive_ttft_metric(row: dict[str, Any], *, owner: str) -> tuple[dict[str, float], str]:
    ttft = row.get("ttft_ms")
    if isinstance(ttft, dict) and isinstance(ttft.get("median"), (int, float)):
        return copy_metric(row, "ttft_ms", owner=owner), str(
            row.get("ttft_source", "reported_ttft_ms")
        )

    prefill_s = row.get("prefill_s")
    if isinstance(prefill_s, dict) and isinstance(prefill_s.get("median"), (int, float)):
        metric = {
            stat: float(value) * 1000.0
            for stat, value in prefill_s.items()
            if stat in {"mean", "median", "p75", "min", "max"}
            and isinstance(value, (int, float))
        }
        if "median" in metric:
            return metric, "derived_from_prefill_s"

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
    model_config = source.get("model_config")
    if isinstance(model_config, dict):
        for key in ("model_type", "model_family", "quantization"):
            if key in model_config:
                model[key] = model_config[key]
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
        if engine not in SCALING_ENGINES:
            continue
        context_tokens = int(row.get("prompt_tokens", 0))
        if context_tokens < min_context_tokens:
            continue

        owner = f"{source_path}.results[{index}]"
        prompt_hash = row.get("prompt_token_ids_sha256")
        if not isinstance(prompt_hash, str) or not prompt_hash:
            raise PrefillScalingBuildError(f"{owner} lacks prompt_token_ids_sha256")
        ttft_ms, ttft_source = derive_ttft_metric(row, owner=owner)
        built: dict[str, Any] = {
            "engine": engine,
            "context_tokens": context_tokens,
            "generation_tokens": int(row.get("generation_tokens", 0)),
            "prompt_token_ids_sha256": prompt_hash,
            "repetitions": int(row.get("repetitions") or source.get("repetitions") or 0),
            "prefill_tok_s": copy_metric(row, "prefill_tok_s", owner=owner),
            "ttft_ms": ttft_ms,
            "ttft_source": ttft_source,
            "peak_memory_gb": copy_metric(row, "peak_memory_gb", owner=owner),
            "memory_source": str(row.get("memory_source", "reported_peak_memory_gb")),
            "source_engine": engine,
        }
        if row.get("prompt_token_ids_path"):
            built["prompt_token_ids_path"] = row["prompt_token_ids_path"]
        if engine == "mlx_lm":
            built["baseline"] = {
                "role": "primary_reference",
                "method": "mlx_lm.benchmark",
            }
        else:
            built["ax_decode_policy"] = row.get(
                "ax_decode_policy", "direct_no_ngram_acceleration"
            )
            built["ax_decode_claim_status"] = row.get("ax_decode_claim_status")
            route = row.get("route")
            built["route"] = (
                route
                if isinstance(route, dict)
                else {"selected_backend": "mlx", "source": "inferred_from_ax_engine_mlx_row"}
            )
        rows.append(built)
    return rows


def attach_ratios(rows: list[dict[str, Any]]) -> None:
    baselines = {
        (int(row["context_tokens"]), int(row["generation_tokens"])): row
        for row in rows
        if row.get("engine") == "mlx_lm"
    }
    for row in rows:
        if row.get("engine") != "ax_engine_mlx":
            continue
        key = (int(row["context_tokens"]), int(row["generation_tokens"]))
        baseline = baselines.get(key)
        if baseline is None:
            raise PrefillScalingBuildError(
                "missing mlx_lm baseline for "
                f"context_tokens={key[0]} generation_tokens={key[1]}"
            )
        row["ratios_to_mlx_lm"] = {
            "prefill_tok_s": metric_median(row, "prefill_tok_s", owner="ax_row")
            / metric_median(baseline, "prefill_tok_s", owner="baseline_row"),
            "ttft_ms": metric_median(row, "ttft_ms", owner="ax_row")
            / metric_median(baseline, "ttft_ms", owner="baseline_row"),
        }


def build_prefill_scaling_artifact(
    source_path: Path,
    *,
    min_context_tokens: int = 1024,
) -> dict[str, Any]:
    source = load_json(source_path)
    if source.get("schema_version") != SOURCE_SCHEMA_VERSION:
        raise PrefillScalingBuildError(
            f"{source_path} has schema_version={source.get('schema_version')!r}, "
            f"expected {SOURCE_SCHEMA_VERSION}"
        )
    rows = build_rows(source_path, source, min_context_tokens=min_context_tokens)
    attach_ratios(rows)
    if not rows:
        raise PrefillScalingBuildError(
            f"{source_path} has no scaling rows at or above {min_context_tokens} tokens"
        )

    artifact = {
        "schema_version": SCHEMA_VERSION,
        "claim_scope": "long_context_prefill_ttft_scaling",
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
        "rows": rows,
    }
    return artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build ax.mlx_prefill_scaling.v1 from ax.mlx_inference_stack.v2."
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-context-tokens", type=int, default=1024)
    parser.add_argument("--skip-validate", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        artifact = build_prefill_scaling_artifact(
            args.source,
            min_context_tokens=args.min_context_tokens,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
        if not args.skip_validate:
            validate_prefill_scaling_artifact(args.output)
    except (PrefillScalingBuildError, PrefillScalingArtifactError) as error:
        print(f"MLX prefill scaling artifact build failed: {error}", file=sys.stderr)
        return 1
    print(f"MLX prefill scaling artifact written: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
