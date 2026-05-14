#!/usr/bin/env python3
"""Build decode-at-depth artifacts from long-context benchmark rows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from check_long_context_decode_at_depth_artifact import (
    LLAMA_CPP_DEPTH_CONTRACT,
    SCHEMA_VERSION,
    LongContextDecodeAtDepthArtifactError,
    validate_long_context_decode_at_depth_artifact,
)


SOURCE_SCHEMAS = {"ax.mlx_inference_stack.v2", "ax.long_context_comparison.v1"}
COMPARISON_ENGINES = {"mlx_lm", "ax_engine_mlx", "llama_cpp_metal"}


class LongContextDecodeAtDepthBuildError(RuntimeError):
    pass


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise LongContextDecodeAtDepthBuildError(f"{path} is not valid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise LongContextDecodeAtDepthBuildError(f"{path} must contain a JSON object")
    return payload


def metric_median(row: dict[str, Any], key: str, *, owner: str) -> float:
    metric = row.get(key)
    if not isinstance(metric, dict):
        raise LongContextDecodeAtDepthBuildError(f"{owner} lacks metric object {key!r}")
    value = metric.get("median")
    if not isinstance(value, (int, float)) or float(value) <= 0.0:
        raise LongContextDecodeAtDepthBuildError(f"{owner}.{key} lacks positive median")
    return float(value)


def copy_metric(row: dict[str, Any], key: str, *, owner: str) -> dict[str, float]:
    metric = row.get(key)
    if not isinstance(metric, dict):
        raise LongContextDecodeAtDepthBuildError(f"{owner} lacks metric object {key!r}")
    copied: dict[str, float] = {}
    for stat in ("mean", "median", "p75", "min", "max", "stddev", "reported_mean"):
        value = metric.get(stat)
        if isinstance(value, (int, float)):
            copied[stat] = float(value)
    if "median" not in copied:
        raise LongContextDecodeAtDepthBuildError(f"{owner}.{key} lacks median")
    return copied


def source_rows(source: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(source.get("results"), list):
        return [row for row in source["results"] if isinstance(row, dict)]
    if isinstance(source.get("rows"), list):
        return [row for row in source["rows"] if isinstance(row, dict)]
    return []


def source_model(source: dict[str, Any]) -> dict[str, Any]:
    raw_model = source.get("model", {})
    if isinstance(raw_model, dict):
        model = dict(raw_model)
        if not isinstance(model.get("id"), str) or not model["id"].strip():
            model["id"] = "unknown"
    else:
        model = {"id": str(raw_model)}
    if source.get("model_dir") and "dir" not in model:
        model["dir"] = str(source["model_dir"])
    config = source.get("model_config")
    if isinstance(config, dict):
        for key in ("model_type", "model_family", "quantization"):
            if key in config:
                model[key] = config[key]
    return model


def row_context_depth(row: dict[str, Any]) -> int:
    if isinstance(row.get("context_depth_tokens"), int):
        return int(row["context_depth_tokens"])
    if isinstance(row.get("context_tokens"), int):
        return int(row["context_tokens"])
    return int(row.get("prompt_tokens", 0))


def row_has_llama_depth_contract(row: dict[str, Any], context_depth_tokens: int) -> bool:
    if row.get("depth_contract") == LLAMA_CPP_DEPTH_CONTRACT:
        return True
    if row.get("decode_depth_contract") == LLAMA_CPP_DEPTH_CONTRACT:
        return True
    if row.get("decode_at_depth_contract") == LLAMA_CPP_DEPTH_CONTRACT:
        return True
    llama_cpp = row.get("llama_cpp")
    if isinstance(llama_cpp, dict):
        n_depth = llama_cpp.get("n_depth")
        if isinstance(n_depth, int) and n_depth == context_depth_tokens:
            return True
    llama_cpp_depth = row.get("llama_cpp_depth")
    if isinstance(llama_cpp_depth, dict):
        n_depth = llama_cpp_depth.get("n_depth")
        if isinstance(n_depth, int) and n_depth == context_depth_tokens:
            return True
    return False


def decode_metric_key(row: dict[str, Any], engine: str) -> str | None:
    if engine == "llama_cpp_metal" and isinstance(row.get("decode_at_depth_tok_s"), dict):
        return "decode_at_depth_tok_s"
    if isinstance(row.get("decode_tok_s"), dict):
        return "decode_tok_s"
    return None


def build_rows(
    source_path: Path,
    source: dict[str, Any],
    *,
    min_context_tokens: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(source_rows(source)):
        engine = str(row.get("engine", ""))
        if engine not in COMPARISON_ENGINES:
            continue
        metric_key = decode_metric_key(row, engine)
        if metric_key is None:
            continue
        context_depth_tokens = row_context_depth(row)
        if context_depth_tokens < min_context_tokens:
            continue
        if engine == "llama_cpp_metal" and not row_has_llama_depth_contract(
            row, context_depth_tokens
        ):
            continue

        owner = f"{source_path}.rows_or_results[{index}]"
        built: dict[str, Any] = {
            "engine": engine,
            "context_depth_tokens": context_depth_tokens,
            "generation_tokens": int(row.get("generation_tokens", 0)),
            "repetitions": int(row.get("repetitions") or source.get("repetitions") or 0),
            "decode_tok_s": copy_metric(row, metric_key, owner=owner),
            "source_engine": engine,
            "timing_scope": str(row.get("timing_scope", "")),
        }
        prompt_hash = row.get("prompt_token_ids_sha256")
        if isinstance(prompt_hash, str):
            built["prompt_token_ids_sha256"] = prompt_hash
        if row.get("prompt_token_ids_path"):
            built["prompt_token_ids_path"] = row["prompt_token_ids_path"]

        if engine == "mlx_lm":
            built["comparison_role"] = "primary_mlx_reference"
            built["baseline"] = {"role": "primary_reference", "method": "mlx_lm.benchmark"}
            built["depth_contract"] = row.get(
                "depth_contract", "generation_after_prompt_hash_parity_prefill"
            )
        elif engine == "ax_engine_mlx":
            built["comparison_role"] = "repo_owned_ax_direct"
            built["ax_decode_policy"] = row.get("ax_decode_policy", "direct_no_ngram_acceleration")
            built["depth_contract"] = row.get(
                "depth_contract", "generation_after_prompt_hash_parity_prefill"
            )
            built["route"] = row.get("route") if isinstance(row.get("route"), dict) else {
                "selected_backend": "mlx",
                "source": "inferred_from_ax_engine_mlx_row",
            }
        elif engine == "llama_cpp_metal":
            built["comparison_role"] = "shape_compatible_external_gguf_depth"
            built["depth_contract"] = LLAMA_CPP_DEPTH_CONTRACT
            built["external_baseline_role"] = row.get(
                "external_baseline_role", "gguf_non_mlx_metal_depth_reference"
            )
            built["claim_boundary"] = row.get(
                "claim_boundary",
                "Shape-compatible external GGUF decode-depth baseline, not prompt-hash parity evidence.",
            )
            if isinstance(row.get("llama_cpp"), dict):
                built["llama_cpp"] = row["llama_cpp"]
            if isinstance(row.get("llama_cpp_depth"), dict):
                built["llama_cpp_depth"] = row["llama_cpp_depth"]
            if row.get("gguf_model"):
                built["gguf_model"] = row["gguf_model"]
            if isinstance(row.get("decode_at_depth_trials"), list):
                built["decode_trials"] = row["decode_at_depth_trials"]
        rows.append(built)
    return rows


def attach_ratios(rows: list[dict[str, Any]]) -> None:
    baselines = {
        (int(row["context_depth_tokens"]), int(row["generation_tokens"])): row
        for row in rows
        if row.get("engine") == "mlx_lm"
    }
    for row in rows:
        if row.get("engine") == "mlx_lm":
            continue
        key = (int(row["context_depth_tokens"]), int(row["generation_tokens"]))
        baseline = baselines.get(key)
        if baseline is None:
            raise LongContextDecodeAtDepthBuildError(
                "missing mlx_lm baseline for "
                f"context_depth_tokens={key[0]} generation_tokens={key[1]}"
            )
        row["ratios_to_mlx_lm"] = {
            "decode_tok_s": metric_median(row, "decode_tok_s", owner="row")
            / metric_median(baseline, "decode_tok_s", owner="baseline"),
        }


def build_long_context_decode_at_depth_artifact(
    source_path: Path,
    *,
    min_context_tokens: int = 1024,
) -> dict[str, Any]:
    source = load_json(source_path)
    source_schema = source.get("schema_version")
    if source_schema not in SOURCE_SCHEMAS:
        raise LongContextDecodeAtDepthBuildError(
            f"{source_path} has schema_version={source_schema!r}, "
            f"expected one of {sorted(SOURCE_SCHEMAS)}"
        )
    rows = build_rows(source_path, source, min_context_tokens=min_context_tokens)
    if not rows:
        raise LongContextDecodeAtDepthBuildError(
            f"{source_path} has no decode-at-depth rows at or above {min_context_tokens} tokens"
        )
    attach_ratios(rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "claim_scope": "long_context_decode_at_existing_depth",
        "source_artifact": str(source_path),
        "source_schema_version": source_schema,
        "host": source.get("host", {}),
        "model": source_model(source),
        "benchmark": {
            "batch_size": 1,
            "temperature": 0.0,
            "repetitions": int(
                source.get("repetitions")
                or source.get("benchmark", {}).get("repetitions", 0)
            ),
        },
        "comparison_contract": {
            "parity_engines": ["mlx_lm", "ax_engine_mlx"],
            "shape_compatible_external_engines": ["llama_cpp_metal"],
            "llama_cpp_boundary": (
                "llama.cpp Metal rows must come from llama-bench n_depth evidence. "
                "They are shape-compatible but not prompt-hash parity evidence."
            ),
        },
        "rows": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build ax.long_context_decode_at_depth.v1 from long-context "
            "comparison or MLX inference-stack artifacts."
        )
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
        artifact = build_long_context_decode_at_depth_artifact(
            args.source,
            min_context_tokens=args.min_context_tokens,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
        if not args.skip_validate:
            validate_long_context_decode_at_depth_artifact(
                args.output,
                require_llama_cpp=args.require_llama_cpp,
            )
    except (LongContextDecodeAtDepthBuildError, LongContextDecodeAtDepthArtifactError) as error:
        print(f"Long-context decode-at-depth artifact build failed: {error}", file=sys.stderr)
        return 1
    print(f"Long-context decode-at-depth artifact written: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
