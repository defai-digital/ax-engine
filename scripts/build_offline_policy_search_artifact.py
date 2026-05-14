#!/usr/bin/env python3
"""Build and validate a minimal offline policy-search artifact.

This builder is intentionally small: it packages already-collected baseline and
candidate rows into the `ax.offline_policy_search.v1` contract. It does not run
models, benchmarks, or optimizers.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

CHECKER_PATH = Path(__file__).with_name("check_offline_policy_search_artifact.py")
CHECKER_SPEC = importlib.util.spec_from_file_location(
    "check_offline_policy_search_artifact",
    CHECKER_PATH,
)
assert CHECKER_SPEC and CHECKER_SPEC.loader
checker = importlib.util.module_from_spec(CHECKER_SPEC)
sys.modules[CHECKER_SPEC.name] = checker
CHECKER_SPEC.loader.exec_module(checker)

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_OBJECTIVE = {
    "maximize": ["decode_tok_s", "kv_saved_bytes"],
    "minimize": ["ttft_ms", "fallback_tokens"],
    "hard_constraints": [
        "quality_gate_pass",
        "deterministic_replay_pass",
        "selected_backend_mlx",
    ],
}


class OfflinePolicySearchBuildError(RuntimeError):
    """Raised when input rows cannot form a valid search artifact."""


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except OSError as error:
        raise OfflinePolicySearchBuildError(f"failed to read {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise OfflinePolicySearchBuildError(f"{path} is not valid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise OfflinePolicySearchBuildError(f"{path} must contain a JSON object")
    return payload


def git_commit(root: Path = REPO_ROOT) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            check=True,
            text=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise OfflinePolicySearchBuildError(f"failed to resolve git commit: {error}") from error
    return result.stdout.strip()


def git_changed_files(root: Path = REPO_ROOT) -> list[str]:
    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            cwd=root,
            check=True,
            text=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise OfflinePolicySearchBuildError(f"failed to inspect git status: {error}") from error
    files: list[str] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        files.append(line[3:].strip())
    return files


def default_repo_metadata(*, allow_dirty: bool, root: Path = REPO_ROOT) -> dict[str, Any]:
    changed_files = git_changed_files(root)
    if changed_files and not allow_dirty:
        raise OfflinePolicySearchBuildError(
            "git worktree is dirty; pass --allow-dirty to record changed files"
        )
    metadata: dict[str, Any] = {
        "commit": git_commit(root),
        "dirty": bool(changed_files),
    }
    if changed_files:
        metadata["changed_files"] = changed_files
    return metadata


def require_mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise OfflinePolicySearchBuildError(f"{field} must be an object")
    return value


def require_list(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise OfflinePolicySearchBuildError(f"{field} must be a list")
    return value


def search_metadata(
    *,
    algorithm: str,
    seed: int,
    max_candidates: int,
    max_wall_time_seconds: int,
    space: dict[str, Any],
) -> dict[str, Any]:
    return {
        "algorithm": algorithm,
        "seed": seed,
        "budget": {
            "max_candidates": max_candidates,
            "max_wall_time_seconds": max_wall_time_seconds,
        },
        "space": space,
    }


def choose_best_candidate(candidates: list[dict[str, Any]], requested: str | None) -> dict[str, str]:
    if requested is not None:
        for candidate in candidates:
            if candidate.get("policy_id") == requested:
                return {"policy_id": requested}
        raise OfflinePolicySearchBuildError(
            f"--best-policy-id {requested!r} does not match any candidate policy_id"
        )
    if len(candidates) != 1:
        raise OfflinePolicySearchBuildError(
            "--best-policy-id is required when more than one candidate row is provided"
        )
    policy_id = candidates[0].get("policy_id")
    if not isinstance(policy_id, str) or not policy_id:
        raise OfflinePolicySearchBuildError("candidate policy_id must be a non-empty string")
    return {"policy_id": policy_id}


def build_offline_policy_search_artifact(
    *,
    metadata: dict[str, Any],
    baseline: dict[str, Any],
    candidates: list[dict[str, Any]],
    decision_classification: str,
    decision_reason: str,
    best_policy_id: str | None = None,
    repo: dict[str, Any] | None = None,
    algorithm: str = "grid",
    seed: int = 42,
    max_wall_time_seconds: int = 3600,
) -> dict[str, Any]:
    target = str(metadata.get("target", "turboquant_kv_policy"))
    model = require_mapping(metadata.get("model"), "metadata.model")
    route = require_mapping(
        metadata.get(
            "route",
            {
                "selected_backend": "mlx",
                "support_tier": "repo_owned_runtime",
            },
        ),
        "metadata.route",
    )
    space = require_mapping(metadata.get("space", {}), "metadata.space")
    objective = require_mapping(metadata.get("objective", DEFAULT_OBJECTIVE), "metadata.objective")

    artifact: dict[str, Any] = {
        "schema": checker.SCHEMA_VERSION,
        "target": target,
        "status": str(metadata.get("status", "diagnostic_only")),
        "created_at": str(metadata.get("created_at", "1970-01-01T00:00:00Z")),
        "repo": repo if repo is not None else require_mapping(metadata.get("repo"), "metadata.repo"),
        "model": model,
        "route": route,
        "search": search_metadata(
            algorithm=str(metadata.get("algorithm", algorithm)),
            seed=int(metadata.get("seed", seed)),
            max_candidates=int(metadata.get("max_candidates", len(candidates))),
            max_wall_time_seconds=int(metadata.get("max_wall_time_seconds", max_wall_time_seconds)),
            space=space,
        ),
        "objective": objective,
        "baseline": baseline,
        "candidates": candidates,
        "best_candidate": choose_best_candidate(candidates, best_policy_id),
        "decision": {
            "classification": decision_classification,
            "reason": decision_reason,
        },
    }
    if "promotion_evidence" in metadata:
        artifact["promotion_evidence"] = metadata["promotion_evidence"]
    return artifact


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--best-policy-id")
    parser.add_argument("--decision-classification", default="diagnostic_only")
    parser.add_argument("--decision-reason", default="diagnostic evidence only")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--skip-git-repo-metadata", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        metadata = load_json(args.metadata)
        baseline = load_json(args.baseline)
        candidates = [load_json(path) for path in args.candidate]
        repo = None
        if not args.skip_git_repo_metadata:
            repo = default_repo_metadata(allow_dirty=args.allow_dirty)
        artifact = build_offline_policy_search_artifact(
            metadata=metadata,
            baseline=baseline,
            candidates=candidates,
            decision_classification=args.decision_classification,
            decision_reason=args.decision_reason,
            best_policy_id=args.best_policy_id,
            repo=repo,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
        checker.validate_offline_policy_search_artifact(args.output)
    except (OfflinePolicySearchBuildError, checker.OfflinePolicySearchArtifactError) as error:
        print(f"offline policy search artifact build failed: {error}", file=sys.stderr)
        return 1
    print(f"wrote offline policy search artifact: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
