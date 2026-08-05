#!/usr/bin/env python3
"""Smoke-test matrix for AX Engine's three-tier model support coverage.

Every listed model carries an explicit support tier (certified / compatible /
experimental) from `crates/ax-engine-core/src/support_tier.rs`. This script
validates that curated mlx-community checkpoints actually load and generate
through the normal runtime path:

  1. Resolve a local snapshot (or download one with `--download`).
  2. Run `ax-engine-bench generate-manifest` (idempotent).
  3. Assert the manifest `model_family` and registry tier match the matrix.
  4. Start `ax-engine-server`, run a short greedy chat generation, and assert
     non-empty coherent output (structural assertions; no goldens yet).
  5. Print a summary table including the tier; fail loudly per model.

Modes:
  --list     print the curated matrix and exit (no weights needed)
  --dry-run  validate the matrix and cross-check tier assignments against the
             Rust architecture registry source (no weights, no cargo; this is
             what CI runs on every push)
  (default)  full real-weight run; artifact-gated like the other model smoke
             checks. Models without a local snapshot are reported as skipped;
             when every selected model is skipped the script prints a JSON
             skip result and exits zero. `--required` turns missing artifacts
             into a hard failure for release gates.

Exit codes: 0 = pass/skip, 1 = one or more models failed the matrix.

Examples:
    python3 scripts/smoke_compatible_models.py --list
    python3 scripts/smoke_compatible_models.py --dry-run
    python3 scripts/smoke_compatible_models.py --models qwen3-0.6b,llama3.2-1b
    python3 scripts/smoke_compatible_models.py --download --required
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import re
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_RS = REPO_ROOT / "crates" / "ax-engine-core" / "src" / "architecture_registry.rs"
DOWNLOAD_MODEL_PY = REPO_ROOT / "scripts" / "download_model.py"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import download_model  # noqa: E402  (repo-local shared downloader helpers)

TIER_CERTIFIED = "certified"
TIER_COMPATIBLE = "compatible"
TIER_EXPERIMENTAL = "experimental"
VALID_TIERS = (TIER_CERTIFIED, TIER_COMPATIBLE, TIER_EXPERIMENTAL)

DEFAULT_MAX_COMPLETION_TOKENS = 64
MIN_CONTENT_CHARS = 8

GREEDY_PROMPT = "Name three primary colors. Answer in one short sentence."


@dataclass(frozen=True)
class SmokeModel:
    """One curated mlx-community checkpoint in the smoke matrix."""

    slug: str
    repo_id: str
    family: str
    tier: str
    prompt: str = GREEDY_PROMPT
    min_chars: int = MIN_CONTENT_CHARS


# Curated matrix: small certified checkpoints first, then compatible ones
# that load through the generic `standard` family path.
SMOKE_MATRIX: tuple[SmokeModel, ...] = (
    SmokeModel(
        slug="qwen3-0.6b",
        repo_id="mlx-community/Qwen3-0.6B-4bit",
        family="qwen3",
        tier=TIER_CERTIFIED,
    ),
    SmokeModel(
        slug="qwen3-4b",
        repo_id="mlx-community/Qwen3-4B-4bit",
        family="qwen3",
        tier=TIER_CERTIFIED,
    ),
    SmokeModel(
        slug="gemma4-e2b",
        repo_id="mlx-community/gemma-4-e2b-it-4bit",
        family="gemma4",
        tier=TIER_CERTIFIED,
    ),
    SmokeModel(
        slug="llama3.2-1b",
        repo_id="mlx-community/Llama-3.2-1B-Instruct-4bit",
        family="llama3",
        tier=TIER_COMPATIBLE,
    ),
    SmokeModel(
        slug="ministral-8b",
        repo_id="mlx-community/Ministral-8B-Instruct-2410-4bit",
        family="mistral3",
        tier=TIER_COMPATIBLE,
    ),
)


class SmokeFailure(Exception):
    """Contract or correctness failure for one matrix row."""


def parse_registry_tiers(registry_path: Path = REGISTRY_RS) -> dict[str, str]:
    """Extract (family_label -> support tier label) from the Rust registry.

    This is a deliberately simple source-level cross-check so `--dry-run` can
    validate tier assignment on CI runners without cargo or model weights.
    """
    text = registry_path.read_text(encoding="utf-8")
    tiers: dict[str, str] = {}
    for block in text.split("ArchitectureRegistration {")[1:]:
        family = re.search(r'family_label:\s*"([^"]+)"', block)
        tier = re.search(r"support_tier:\s*ModelSupportTier::(\w+)", block)
        if family is None or tier is None:
            continue
        tiers[family.group(1)] = tier.group(1).lower()
    if not tiers:
        raise SmokeFailure(
            f"could not parse any registry entries from {registry_path}; "
            "the ArchitectureRegistration layout may have changed"
        )
    return tiers


def validate_matrix(
    matrix: Sequence[SmokeModel] = SMOKE_MATRIX,
    registry_tiers: Mapping[str, str] | None = None,
) -> list[str]:
    """Structural + tier cross-checks. Returns a list of problems (empty = ok)."""
    problems: list[str] = []
    slugs = [model.slug for model in matrix]
    if len(slugs) != len(set(slugs)):
        problems.append("duplicate slugs in the smoke matrix")
    if registry_tiers is None:
        registry_tiers = parse_registry_tiers()
    for model in matrix:
        if model.tier not in VALID_TIERS:
            problems.append(f"{model.slug}: invalid tier {model.tier!r}")
        if not re.fullmatch(r"[A-Za-z0-9._-]+/[A-Za-z0-9._-]+", model.repo_id):
            problems.append(f"{model.slug}: malformed repo id {model.repo_id!r}")
        registry_tier = registry_tiers.get(model.family)
        if registry_tier is None:
            problems.append(
                f"{model.slug}: family {model.family!r} is not in ARCHITECTURE_REGISTRY"
            )
        elif registry_tier != model.tier:
            problems.append(
                f"{model.slug}: matrix tier {model.tier!r} != registry tier "
                f"{registry_tier!r} for family {model.family!r}"
            )
        if model.min_chars < 1:
            problems.append(f"{model.slug}: min_chars must be positive")
        if not model.prompt.strip():
            problems.append(f"{model.slug}: prompt must be non-empty")
    return problems


def render_table(rows: Sequence[Mapping[str, str]]) -> str:
    columns = ("model", "family", "tier", "status", "detail")
    widths = {
        column: max(len(column), *(len(str(row.get(column, ""))) for row in rows))
        for column in columns
    }
    header = "  ".join(column.upper().ljust(widths[column]) for column in columns)
    lines = [header, "  ".join("-" * widths[column] for column in columns)]
    for row in rows:
        lines.append(
            "  ".join(str(row.get(column, "")).ljust(widths[column]) for column in columns)
        )
    return "\n".join(lines)


def list_matrix(matrix: Sequence[SmokeModel]) -> None:
    rows = [
        {
            "model": model.slug,
            "family": model.family,
            "tier": model.tier,
            "status": model.repo_id,
            "detail": "",
        }
        for model in matrix
    ]
    print(render_table(rows))


def hf_repo_cache_dir(repo_id: str) -> Path:
    return download_model.default_mlx_lm_repo_cache_dir(repo_id)


def latest_usable_snapshot(repo_cache: Path) -> Path | None:
    snapshots = repo_cache / "snapshots"
    if not snapshots.is_dir():
        return None
    candidates = []
    for child in snapshots.iterdir():
        if child.is_dir() and (child / "config.json").is_file():
            try:
                candidates.append((child.stat().st_mtime, child))
            except OSError:
                continue
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def resolve_snapshot(model: SmokeModel, models_dir: Path | None) -> Path | None:
    """Find a local snapshot for a matrix model without downloading."""
    repo_name = model.repo_id.split("/", 1)[1].lower()
    if models_dir is not None and models_dir.is_dir():
        for child in sorted(models_dir.iterdir()):
            if not child.is_dir() or repo_name not in child.name.lower():
                continue
            if (child / "config.json").is_file():
                return child
            snapshot = latest_usable_snapshot(child)
            if snapshot is not None:
                return snapshot
    return latest_usable_snapshot(hf_repo_cache_dir(model.repo_id))


def download_snapshot(model: SmokeModel) -> Path:
    command = [
        sys.executable,
        str(DOWNLOAD_MODEL_PY),
        model.repo_id,
        "--json",
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise SmokeFailure(
            f"download failed for {model.repo_id}: {completed.stderr.strip() or completed.stdout.strip()}"
        )
    snapshot = resolve_snapshot(model, None)
    if snapshot is None:
        raise SmokeFailure(f"download of {model.repo_id} did not produce a usable snapshot")
    return snapshot


def ensure_binary(name: str, package: str, no_build: bool, release: bool) -> Path:
    profile = "release" if release else "debug"
    binary = REPO_ROOT / "target" / profile / name
    if binary.is_file():
        return binary
    if no_build:
        raise SmokeFailure(f"binary does not exist and --no-build was given: {binary}")
    command = ["cargo", "build", "-p", package]
    if release:
        command.append("--release")
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    if not binary.is_file():
        raise SmokeFailure(f"cargo build did not produce {binary}")
    return binary


def generate_manifest(bench_bin: Path, model_dir: Path) -> dict[str, Any]:
    command = [
        str(bench_bin),
        "generate-manifest",
        str(model_dir),
        "--json",
        "--validate",
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise SmokeFailure(
            f"generate-manifest failed for {model_dir}: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    manifest_path = model_dir / "model-manifest.json"
    if not manifest_path.is_file():
        raise SmokeFailure(f"generate-manifest did not write {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SmokeFailure(f"could not read {manifest_path}: {error}") from error
    return manifest


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def http_json(
    method: str, url: str, payload: Mapping[str, Any] | None, timeout_sec: float
) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, method=method)
    if body is not None:
        request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise SmokeFailure(f"{method} {url} returned {error.code}: {detail}") from error
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
        raise SmokeFailure(f"{method} {url} failed: {error}") from error


def wait_for_health(
    process: subprocess.Popen[str], url: str, timeout_sec: float, log_path: Path
) -> None:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise SmokeFailure(
                f"server exited before readiness with status {process.returncode}; "
                f"see {log_path}"
            )
        try:
            health = http_json("GET", url, None, timeout_sec=5.0)
            if health.get("status") == "ok":
                return
        except SmokeFailure:
            pass
        time.sleep(0.5)
    raise SmokeFailure(f"server did not become ready within {timeout_sec}s; see {log_path}")


def assert_coherent_content(response: Mapping[str, Any], model: SmokeModel) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise SmokeFailure(f"chat response has no choices: {response}")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise SmokeFailure(f"chat response content is empty: {response}")
    text = content.strip()
    if len(text) < model.min_chars:
        raise SmokeFailure(
            f"chat response shorter than {model.min_chars} chars: {text!r}"
        )
    if not any(character.isalnum() for character in text):
        raise SmokeFailure(f"chat response has no alphanumeric content: {text!r}")
    return text


def run_model(
    args: argparse.Namespace,
    model: SmokeModel,
    server_bin: Path,
    bench_bin: Path,
    registry_tiers: Mapping[str, str],
) -> dict[str, str]:
    snapshot = resolve_snapshot(model, args.models_dir)
    if snapshot is None and args.download:
        snapshot = download_snapshot(model)
    if snapshot is None:
        if args.required:
            raise SmokeFailure(
                f"no local snapshot for {model.repo_id}; pass --download or --models-dir"
            )
        return {
            "model": model.slug,
            "family": model.family,
            "tier": model.tier,
            "status": "skipped",
            "detail": "no local snapshot",
        }

    manifest = generate_manifest(bench_bin, snapshot)
    family = str(manifest.get("model_family", ""))
    if family != model.family:
        raise SmokeFailure(
            f"manifest family {family!r} != expected {model.family!r} for {model.slug}"
        )
    registry_tier = registry_tiers.get(family)
    if registry_tier is None:
        raise SmokeFailure(f"family {family!r} missing from ARCHITECTURE_REGISTRY")
    if registry_tier != model.tier:
        raise SmokeFailure(
            f"registry tier {registry_tier!r} != matrix tier {model.tier!r} for {model.slug}"
        )

    port = free_port()
    base_url = f"http://127.0.0.1:{port}"
    log_path = Path(tempfile.gettempdir()) / f"ax-smoke-compatible-{model.slug}-{port}.log"
    command = [
        str(server_bin),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--model-id",
        model.slug,
        "--mlx",
        "--mlx-model-artifacts-dir",
        str(snapshot),
    ]
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            wait_for_health(process, f"{base_url}/health", args.ready_timeout_sec, log_path)
            response = http_json(
                "POST",
                f"{base_url}/v1/chat/completions",
                {
                    "model": model.slug,
                    "messages": [{"role": "user", "content": model.prompt}],
                    "temperature": 0,
                    "max_completion_tokens": DEFAULT_MAX_COMPLETION_TOKENS,
                    "stream": False,
                },
                args.request_timeout_sec,
            )
            text = assert_coherent_content(response, model)
            return {
                "model": model.slug,
                "family": family,
                "tier": registry_tier,
                "status": "passed",
                "detail": f"{len(text)} chars: {text[:60]!r}",
            }
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)


def parser() -> argparse.ArgumentParser:
    parsed = argparse.ArgumentParser(
        description=(
            "Smoke-test the curated compatible/certified model matrix through "
            "the normal AX Engine runtime path."
        )
    )
    parsed.add_argument(
        "--models",
        help="comma-separated matrix slugs to run (default: all); see --list",
    )
    parsed.add_argument(
        "--list",
        action="store_true",
        help="print the curated matrix and exit (no weights needed)",
    )
    parsed.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "validate the matrix and cross-check tier assignments against the "
            "Rust registry source; no weights or cargo needed"
        ),
    )
    parsed.add_argument(
        "--models-dir",
        type=Path,
        help="root directory containing pre-downloaded model snapshots",
    )
    parsed.add_argument(
        "--download",
        action="store_true",
        help="download missing snapshots via scripts/download_model.py",
    )
    parsed.add_argument(
        "--required",
        action="store_true",
        help="fail when a selected model has no local snapshot",
    )
    parsed.add_argument(
        "--release",
        action="store_true",
        help="use/build target/release binaries instead of target/debug",
    )
    parsed.add_argument(
        "--no-build",
        action="store_true",
        help="fail if the required binaries do not already exist",
    )
    parsed.add_argument("--ready-timeout-sec", type=float, default=180.0)
    parsed.add_argument("--request-timeout-sec", type=float, default=180.0)
    parsed.add_argument(
        "--json",
        action="store_true",
        help="emit the run summary as JSON after the table",
    )
    return parsed


def select_models(args: argparse.Namespace) -> list[SmokeModel]:
    if not args.models:
        return list(SMOKE_MATRIX)
    requested = {slug.strip() for slug in args.models.split(",") if slug.strip()}
    known = {model.slug for model in SMOKE_MATRIX}
    unknown = requested - known
    if unknown:
        raise SmokeFailure(
            f"unknown --models slugs: {sorted(unknown)}; known: {sorted(known)}"
        )
    return [model for model in SMOKE_MATRIX if model.slug in requested]


def emit_summary(status: str, results: Sequence[Mapping[str, str]], as_json: bool) -> None:
    print()
    print(render_table(results))
    if as_json:
        print(
            json.dumps(
                {
                    "schema": "ax.compatible_model_smoke.v1",
                    "status": status,
                    "results": list(results),
                },
                indent=2,
                sort_keys=True,
            )
        )


def main() -> int:
    args = parser().parse_args()
    try:
        matrix = select_models(args)
    except SmokeFailure as error:
        print(str(error), file=sys.stderr)
        return 1

    if args.list:
        list_matrix(matrix)
        return 0

    try:
        registry_tiers = parse_registry_tiers()
        problems = validate_matrix(matrix, registry_tiers)
    except SmokeFailure as error:
        print(str(error), file=sys.stderr)
        return 1
    if problems:
        for problem in problems:
            print(f"matrix validation: {problem}", file=sys.stderr)
        return 1

    if args.dry_run:
        print("matrix validation: ok")
        print()
        list_matrix(matrix)
        print()
        print(
            f"dry-run: {len(matrix)} model(s), tiers cross-checked against "
            f"{REGISTRY_RS.relative_to(REPO_ROOT)}"
        )
        return 0

    try:
        bench_bin = ensure_binary(
            "ax-engine-bench", "ax-engine-bench", args.no_build, args.release
        )
        server_bin = ensure_binary(
            "ax-engine-server", "ax-engine-server", args.no_build, args.release
        )
    except (OSError, subprocess.CalledProcessError, SmokeFailure) as error:
        print(f"build failure: {error}", file=sys.stderr)
        return 1

    results: list[dict[str, str]] = []
    for model in matrix:
        print(f"[{model.slug}] tier={model.tier} repo={model.repo_id}")
        try:
            row = run_model(args, model, server_bin, bench_bin, registry_tiers)
        except (OSError, SmokeFailure) as error:
            row = {
                "model": model.slug,
                "family": model.family,
                "tier": model.tier,
                "status": "failed",
                "detail": str(error),
            }
            print(f"[{model.slug}] FAILED: {error}", file=sys.stderr)
        results.append(row)

    failed = [row for row in results if row["status"] == "failed"]
    skipped = [row for row in results if row["status"] == "skipped"]
    if failed:
        emit_summary("failed", results, args.json)
        return 1
    status = "skipped" if len(skipped) == len(results) else "passed"
    emit_summary(status, results, args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
