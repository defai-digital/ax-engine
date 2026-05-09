#!/usr/bin/env python3
"""Build an MLX prefix warmup artifact from an ax-engine-bench result directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import check_mlx_prefix_warmup_artifact as checker


SOURCE_SCHEMAS = {
    "manifest": "ax.engine_bench.manifest.v1",
    "metrics": "ax.engine_bench.metrics.v1",
    "routes": "ax.engine_bench.routes.v1",
    "trace": "ax.engine_bench.trace.v1",
}


class PrefixWarmupBuildError(RuntimeError):
    pass


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise PrefixWarmupBuildError(f"{path} is not valid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise PrefixWarmupBuildError(f"{path} must contain a JSON object")
    return payload


def require_schema(path: Path, payload: dict[str, Any], expected: str) -> None:
    if payload.get("schema_version") != expected:
        raise PrefixWarmupBuildError(
            f"{path} has schema_version={payload.get('schema_version')!r}, expected {expected}"
        )


def require_mapping(payload: dict[str, Any], key: str, *, owner: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise PrefixWarmupBuildError(f"{owner} lacks object field {key!r}")
    return value


def require_list(payload: dict[str, Any], key: str, *, owner: str) -> list[Any]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise PrefixWarmupBuildError(f"{owner} lacks list field {key!r}")
    return value


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(json.dumps(value, separators=(",", ":")).encode())


def route_counter(route: dict[str, Any], key: str) -> int:
    value = route.get(key, 0)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise PrefixWarmupBuildError(f"routes.route.{key} must be a non-negative integer")
    return value


def find_logical_prefix_observation(trace: dict[str, Any]) -> dict[str, int]:
    best: dict[str, int] | None = None
    for step in require_list(trace, "steps", owner="trace"):
        if not isinstance(step, dict):
            continue
        for item in step.get("items", []):
            if not isinstance(item, dict):
                continue
            reused_tokens = item.get("prefix_tokens_reused", 0)
            reused_blocks = item.get("prefix_blocks_reused", 0)
            request_id = item.get("request_id")
            if (
                isinstance(request_id, int)
                and not isinstance(request_id, bool)
                and isinstance(reused_tokens, int)
                and isinstance(reused_blocks, int)
                and reused_tokens > 0
                and reused_blocks > 0
            ):
                candidate = {
                    "request_id": request_id,
                    "matched_token_count": reused_tokens,
                    "reused_block_count": reused_blocks,
                }
                if best is None or reused_tokens > best["matched_token_count"]:
                    best = candidate
    if best is None:
        raise PrefixWarmupBuildError("trace lacks a logical prefix reuse item")
    return best


def final_requests_by_id(trace: dict[str, Any]) -> dict[int, dict[str, Any]]:
    observation = require_mapping(trace, "observation", owner="trace")
    requests = require_list(observation, "requests", owner="trace.observation")
    by_id: dict[int, dict[str, Any]] = {}
    for request in requests:
        if not isinstance(request, dict):
            continue
        request_id = request.get("request_id")
        if isinstance(request_id, int) and not isinstance(request_id, bool):
            by_id[request_id] = request
    return by_id


def events_by_external_id(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    events = require_list(manifest, "events", owner="manifest")
    by_external_id: dict[str, dict[str, Any]] = {}
    for event in events:
        if not isinstance(event, dict) or event.get("type") != "submit":
            continue
        request_id = event.get("request_id")
        if isinstance(request_id, str) and request_id:
            by_external_id[request_id] = event
    return by_external_id


def prompt_digest_for_event(
    event: dict[str, Any],
    *,
    manifest_root: Path,
) -> dict[str, str]:
    token_hash = event.get("prompt_token_ids_sha256")
    if isinstance(token_hash, str):
        return {
            "prompt_token_ids_sha256": token_hash,
            "prompt_digest_kind": "token_ids",
        }
    prompt_ref = event.get("prompt_ref")
    if not isinstance(prompt_ref, str) or not prompt_ref:
        raise PrefixWarmupBuildError("submit event lacks prompt_ref or prompt_token_ids_sha256")
    prompt_path = (manifest_root / prompt_ref).resolve()
    if not prompt_path.is_file():
        raise PrefixWarmupBuildError(f"prompt_ref does not exist: {prompt_path}")
    return {
        "prompt_ref_sha256": sha256_bytes(prompt_path.read_bytes()),
        "prompt_digest_kind": "prompt_ref_bytes",
    }


def validate_source_gates(metrics: dict[str, Any]) -> None:
    correctness = require_mapping(metrics, "correctness", owner="metrics")
    determinism = require_mapping(metrics, "determinism", owner="metrics")
    if correctness.get("passed") is not True:
        raise PrefixWarmupBuildError("metrics.correctness.passed must be true")
    if determinism.get("passed") is not True:
        raise PrefixWarmupBuildError("metrics.determinism.passed must be true")


def normalize_model(manifest: dict[str, Any]) -> dict[str, Any]:
    model = require_mapping(manifest, "model", owner="manifest")
    normalized = dict(model)
    if not isinstance(normalized.get("id"), str):
        family = normalized.get("family")
        if not isinstance(family, str) or not family:
            raise PrefixWarmupBuildError("manifest.model must include id or family")
        normalized["id"] = family
    return normalized


def build_prefix_warmup_artifact(
    *,
    result_dir: Path,
    manifest_root: Path,
) -> dict[str, Any]:
    manifest_path = result_dir / "manifest.json"
    metrics_path = result_dir / "metrics.json"
    routes_path = result_dir / "routes.json"
    trace_path = result_dir / "trace.json"
    manifest = load_json(manifest_path)
    metrics = load_json(metrics_path)
    routes = load_json(routes_path)
    trace = load_json(trace_path)
    require_schema(manifest_path, manifest, SOURCE_SCHEMAS["manifest"])
    require_schema(metrics_path, metrics, SOURCE_SCHEMAS["metrics"])
    require_schema(routes_path, routes, SOURCE_SCHEMAS["routes"])
    require_schema(trace_path, trace, SOURCE_SCHEMAS["trace"])
    validate_source_gates(metrics)

    logical = find_logical_prefix_observation(trace)
    final_request = final_requests_by_id(trace).get(logical["request_id"])
    if final_request is None:
        raise PrefixWarmupBuildError(
            f"trace lacks final request state for request_id={logical['request_id']}"
        )
    external_id = final_request.get("external_id")
    if not isinstance(external_id, str) or not external_id:
        raise PrefixWarmupBuildError("final request lacks external_id")
    event = events_by_external_id(manifest).get(external_id)
    if event is None:
        raise PrefixWarmupBuildError(f"manifest lacks submit event for {external_id!r}")

    route = require_mapping(routes, "route", owner="routes")
    runtime = require_mapping(routes, "runtime", owner="routes")
    selected_backend = runtime.get("selected_backend")
    if selected_backend != "mlx":
        raise PrefixWarmupBuildError("routes.runtime.selected_backend must be mlx")

    generated_tokens = final_request.get("generated_tokens")
    if not isinstance(generated_tokens, list):
        raise PrefixWarmupBuildError("final request lacks generated_tokens list")
    benchmark_generation_tokens = event.get("output_tokens_target")
    if not isinstance(benchmark_generation_tokens, int) or benchmark_generation_tokens <= 0:
        benchmark_generation_tokens = max(len(generated_tokens), 1)

    observation = {
        "request_id": external_id,
        **prompt_digest_for_event(event, manifest_root=manifest_root),
        "route": {
            "selected_backend": "mlx",
            "route_identity": "repo_owned_mlx",
        },
        "logical_prefix_reuse": {
            "matched_token_count": logical["matched_token_count"],
            "reused_block_count": logical["reused_block_count"],
        },
        "physical_prefix_snapshot": {
            "hit_count": route_counter(route, "ax_mlx_prefix_cache_hits"),
            "miss_count": route_counter(route, "ax_mlx_prefix_cache_misses"),
            "warmup_token_count": route_counter(route, "ax_mlx_prefix_cache_warmup_tokens"),
            "reused_token_count": route_counter(route, "ax_mlx_prefix_cache_reused_tokens"),
            "blocked_count": route_counter(route, "ax_mlx_prefix_cache_blocked"),
            "physical_snapshot_coverage": "miss_warmup_only",
        },
        "correctness": {
            "status": "passed",
            "deterministic_replay": True,
            "output_token_ids_sha256": sha256_json(generated_tokens),
        },
    }

    return {
        "schema_version": checker.SCHEMA_VERSION,
        "claim_scope": "physical_prefix_miss_warmup_correctness",
        "source": {
            "schema_version": "ax.mlx_prefix_warmup_builder.v1",
            "result_dir": str(result_dir),
            "manifest_root": str(manifest_root),
            "source_schemas": SOURCE_SCHEMAS,
        },
        "model": normalize_model(manifest),
        "host": metrics.get("host", routes.get("host", {"chip": "unknown"})),
        "benchmark": {
            "shared_prefix_tokens": logical["matched_token_count"],
            "generation_tokens": benchmark_generation_tokens,
            "repetitions": 1,
        },
        "observations": [observation],
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument("--manifest-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        artifact = build_prefix_warmup_artifact(
            result_dir=args.result_dir,
            manifest_root=args.manifest_root,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(artifact, indent=2) + "\n")
        checker.validate_prefix_warmup_artifact(args.output)
    except (PrefixWarmupBuildError, checker.PrefixWarmupArtifactError) as error:
        print(f"MLX prefix warmup artifact build failed: {error}", file=sys.stderr)
        return 1
    print(f"MLX prefix warmup artifact written: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
