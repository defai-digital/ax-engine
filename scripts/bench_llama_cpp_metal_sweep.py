#!/usr/bin/env python3
"""Sweep llama.cpp Metal benchmarks across README MLX-inference rows.

Reads benchmarks/manifests/llama_cpp_metal/inventory.json and, for each row:
  1) Resolves the first GGUF candidate from the local Hugging Face cache, or
     from Hugging Face metadata when not running --cache-only.
  2) Reuses the cached GGUF, or downloads it to --cache-dir when not running
     --cache-only.
  3) Invokes scripts/bench_mlx_inference_stack.py with --llama-cpp-bench /
     --llama-cpp-gguf, --skip-mlx-lm, --skip-ax-engine to produce ONLY the
     external GGUF baseline row.
  4) Optionally deletes the GGUF after the row finishes (--no-keep-gguf) to
     keep peak disk low.

Pass --full-stack to benchmark the same GGUF-resolved model set with
llama.cpp Metal, mlx_lm.benchmark, AX direct mode, and AX default n-gram mode
in one artifact per row. Pass --update-readme with --full-stack to update the
README performance tables from those artifacts.

Writes one result JSON per row plus a combined sweep_results.json and a
sweep_summary.md. Unresolved rows are recorded as explicit n/a entries.

This script modifies README.md only when --update-readme is provided.

Claim boundary: rows produced here are shape-compatible external GGUF
baselines only. See inventory.json for the full disclaimer.
"""
from __future__ import annotations

import argparse
import contextlib
import fnmatch
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

try:
    from scripts import bench_mtp_6bit_ax_refresh as publication_gate
except ModuleNotFoundError:
    import bench_mtp_6bit_ax_refresh as publication_gate

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "benchmarks" / "manifests" / "llama_cpp_metal" / "inventory.json"
DEFAULT_BENCH_SCRIPT = REPO_ROOT / "scripts" / "bench_mlx_inference_stack.py"
DEFAULT_LLAMA_BENCH = Path("/opt/homebrew/bin/llama-bench")
DEFAULT_REQUIRED_GGUF_PUBLISHER = "unsloth"
DEFAULT_REQUIRED_MLX_PUBLISHER = "mlx-community"
DEFAULT_MAX_LOAD_AVERAGE = 2.0
DEFAULT_MAX_TOP_PROCESS_CPU_PERCENT = 50.0
DEFAULT_LOAD_WAIT_TIMEOUT_SECONDS = 900.0
README_PUBLICATION_MODEL_COUNT = 12
LLAMA_CPP_PUBLICATION_MATRIX_SCHEMA = "ax.llama_cpp_metal_publication_matrix.v1"


class LlamaCppMetalSweepError(RuntimeError):
    pass


def log(msg: str) -> None:
    print(f"[sweep] {msg}", flush=True)


def _row_slug(row: dict[str, Any]) -> str:
    if not isinstance(row, dict):
        raise LlamaCppMetalSweepError("manifest row must be an object")
    slug = row.get("slug")
    if not isinstance(slug, str) or not slug:
        raise LlamaCppMetalSweepError("manifest row lacks non-empty slug")
    return slug


def filter_manifest_rows(
    rows: list[dict[str, Any]],
    rows_filter: list[str] | None,
) -> list[dict[str, Any]]:
    if not rows:
        raise LlamaCppMetalSweepError("manifest contains no rows")

    seen: set[str] = set()
    duplicate_slugs: set[str] = set()
    for row in rows:
        slug = _row_slug(row)
        if slug in seen:
            duplicate_slugs.add(slug)
        seen.add(slug)
    if duplicate_slugs:
        raise LlamaCppMetalSweepError(
            "manifest contains duplicate slug(s): "
            + ", ".join(sorted(duplicate_slugs))
        )

    if rows_filter is None:
        return rows
    if not rows_filter:
        raise LlamaCppMetalSweepError("--rows-filter requires at least one slug")

    requested: set[str] = set()
    duplicate_filters: set[str] = set()
    for slug in rows_filter:
        if slug in requested:
            duplicate_filters.add(slug)
        requested.add(slug)
    if duplicate_filters:
        raise LlamaCppMetalSweepError(
            "--rows-filter contains duplicate slug(s): "
            + ", ".join(sorted(duplicate_filters))
        )

    missing = sorted(requested - seen)
    if missing:
        raise LlamaCppMetalSweepError(
            "--rows-filter references unknown slug(s): " + ", ".join(missing)
        )

    selected = [row for row in rows if _row_slug(row) in requested]
    if not selected:
        raise LlamaCppMetalSweepError("--rows-filter selected no rows")
    return selected


def validate_gguf_publisher_inventory(manifest: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    publisher = manifest.get("required_gguf_publisher", DEFAULT_REQUIRED_GGUF_PUBLISHER)
    if not isinstance(publisher, str) or not publisher:
        raise RuntimeError(
            f"required_gguf_publisher must be a non-empty string, got {publisher!r}"
        )

    bad: list[str] = []
    prefix = f"{publisher}/"
    for row in rows:
        for candidate in row.get("gguf_candidates", []):
            repo = candidate.get("repo", "")
            if not repo.startswith(prefix):
                bad.append(f"{row.get('slug', '<unknown>')} -> {repo}")

    if bad:
        details = "; ".join(bad)
        raise RuntimeError(
            f"llama.cpp Metal sweep inventory must use {publisher} GGUF repos only: "
            f"{details}"
        )


def validate_mlx_publisher_inventory(manifest: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    publisher = manifest.get("required_mlx_publisher", DEFAULT_REQUIRED_MLX_PUBLISHER)
    if not isinstance(publisher, str) or not publisher:
        raise RuntimeError(
            f"required_mlx_publisher must be a non-empty string, got {publisher!r}"
        )

    bad: list[str] = []
    prefix = f"{publisher}/"
    for row in rows:
        repo = row.get("mlx_repo_id", "")
        if not isinstance(repo, str) or not repo.startswith(prefix):
            bad.append(f"{row.get('slug', '<unknown>')} -> {repo}")

    if bad:
        details = "; ".join(bad)
        raise RuntimeError(
            f"llama.cpp Metal sweep inventory must use {publisher} MLX repos only: "
            f"{details}"
        )


def resolve_gguf_candidate(
    candidates: list[dict[str, Any]],
    *,
    cache_dir: Path,
    hf_token: str | None,
    cache_only: bool,
) -> tuple[str, str, list[dict[str, Any]]] | None:
    """Walk candidates in priority order. Return (repo, filename, probe_log)
    for the first candidate that resolves; None if all fail."""
    probe_log: list[dict[str, Any]] = []
    for candidate in candidates:
        repo = candidate["repo"]
        pattern = candidate["filename_pattern"]
        allow_dynamic = candidate_allows_dynamic_quant(candidate)
        entry: dict[str, Any] = {"repo": repo, "filename_pattern": pattern}

        cached_match = resolve_cached_hf_file(repo, pattern, cache_dir)
        if cached_match is not None:
            entry["result"] = "resolved_from_cache"
            entry["filename"] = cached_match.name
            probe_log.append(entry)
            return repo, cached_match.name, probe_log

        if cache_only:
            entry["result"] = "cache_miss"
            probe_log.append(entry)
            continue

        from huggingface_hub import HfApi
        from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

        api = HfApi(token=hf_token)
        try:
            files = api.list_repo_files(repo)
        except RepositoryNotFoundError:
            entry["result"] = "repo_not_found"
            probe_log.append(entry)
            continue
        except GatedRepoError:
            entry["result"] = "gated_repo"
            probe_log.append(entry)
            continue
        except Exception as exc:
            entry["result"] = f"error: {type(exc).__name__}: {exc}"
            probe_log.append(entry)
            continue
        matches = [f for f in files if fnmatch.fnmatch(f, pattern)]
        if not matches:
            entry["result"] = "no_match"
            entry["sample_files"] = [f for f in files if f.endswith(".gguf")][:5]
            probe_log.append(entry)
            continue
        # Prefer root-level standard K-quants over Unsloth Dynamic (UD-*) and
        # auxiliary MTP/projector files that can share the same quant marker.
        matches.sort(key=gguf_candidate_sort_key)
        if not is_allowed_root_gguf(matches[0], allow_dynamic=allow_dynamic):
            entry["result"] = "no_standard_root_match"
            entry["sample_files"] = matches[:5]
            probe_log.append(entry)
            continue
        entry["result"] = "resolved"
        entry["filename"] = matches[0]
        probe_log.append(entry)
        return repo, matches[0], probe_log
    return None


def gguf_candidate_sort_key(filename: str) -> tuple[bool, bool, bool, str]:
    basename = Path(filename).name
    return (
        "/" in filename,
        "UD-" in basename,
        "MTP" in filename or "mtp" in filename,
        filename,
    )


def is_standard_root_gguf(filename: str) -> bool:
    return is_allowed_root_gguf(filename, allow_dynamic=False)


def is_allowed_root_gguf(filename: str, *, allow_dynamic: bool) -> bool:
    basename = Path(filename).name
    return (
        "/" not in filename
        and "MTP" not in filename
        and "mtp" not in filename
        and (allow_dynamic or "UD-" not in basename)
    )


def candidate_allows_dynamic_quant(candidate: dict[str, Any]) -> bool:
    pattern = str(candidate.get("filename_pattern", ""))
    return bool(candidate.get("allow_dynamic_quant")) or "UD-" in pattern


_SHARD_RE = __import__("re").compile(r"-(\d{5})-of-(\d{5})\.gguf$")


def _shard_siblings(filename: str) -> list[str]:
    """If filename is the first shard of an N-shard GGUF set, return all N
    shard filenames. Otherwise return [filename] unchanged."""
    match = _SHARD_RE.search(filename)
    if not match:
        return [filename]
    total = int(match.group(2))
    prefix = filename[: match.start(1)]
    suffix = ".gguf"
    return [f"{prefix}{i:05d}-of-{total:05d}{suffix}" for i in range(1, total + 1)]


def resolve_cached_hf_file(repo: str, filename_pattern: str, cache_dir: Path) -> Path | None:
    snapshot = latest_hf_cache_snapshot(repo, cache_dir)
    if snapshot is None:
        return None
    allow_dynamic = "UD-" in filename_pattern
    matches = sorted(
        (
            path
            for path in snapshot.rglob("*.gguf")
            if fnmatch.fnmatch(path.name, filename_pattern)
            or fnmatch.fnmatch(
                str(path.relative_to(snapshot)),
                filename_pattern,
            )
        ),
        key=lambda path: gguf_candidate_sort_key(str(path.relative_to(snapshot))),
    )
    if matches and not is_allowed_root_gguf(
        str(matches[0].relative_to(snapshot)),
        allow_dynamic=allow_dynamic,
    ):
        return None
    return matches[0] if matches else None


def cached_hf_file(repo: str, filename: str, cache_dir: Path) -> Path | None:
    snapshot = latest_hf_cache_snapshot(repo, cache_dir)
    if snapshot is None:
        return None
    candidate = snapshot / filename
    if candidate.is_file():
        return candidate
    matches = sorted(path for path in snapshot.rglob(Path(filename).name) if path.is_file())
    return matches[0] if matches else None


def download_gguf(
    repo: str,
    filename: str,
    *,
    cache_dir: Path,
    hf_token: str | None,
    cache_only: bool,
) -> Path:
    shards = _shard_siblings(filename)
    first_path: Path | None = None
    for shard in shards:
        cached = cached_hf_file(repo, shard, cache_dir)
        if cached is not None:
            log(f"  reuse cached {repo} :: {shard}")
            if first_path is None:
                first_path = cached
            continue
        if cache_only:
            raise FileNotFoundError(f"cached GGUF shard not found for {repo}: {shard}")

        from huggingface_hub import hf_hub_download

        log(f"  download {repo} :: {shard}")
        local = hf_hub_download(
            repo_id=repo,
            filename=shard,
            cache_dir=str(cache_dir),
            token=hf_token,
        )
        if first_path is None:
            first_path = Path(local)
    assert first_path is not None
    return first_path


def gguf_disk_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return 0


def _slug_repo_id(repo_id: str) -> str:
    return repo_id.replace("/", "--")


def latest_hf_cache_snapshot(repo_id: str, cache_dir: Path) -> Path | None:
    repo_cache = cache_dir / f"models--{_slug_repo_id(repo_id)}"
    refs_main = repo_cache / "refs" / "main"
    if refs_main.is_file():
        revision = refs_main.read_text().strip()
        snapshot = repo_cache / "snapshots" / revision
        if snapshot.is_dir():
            return snapshot

    snapshots = repo_cache / "snapshots"
    if not snapshots.is_dir():
        return None
    candidates = [path for path in snapshots.iterdir() if path.is_dir()]
    return max(candidates, key=lambda path: path.stat().st_mtime, default=None)


def missing_ax_model_artifacts(model_dir: Path) -> list[str]:
    missing: list[str] = []
    if not (model_dir / "config.json").is_file():
        missing.append("config.json")
    if not (model_dir / "model-manifest.json").is_file():
        missing.append("model-manifest.json")
    if not any(model_dir.glob("*.safetensors")):
        missing.append("*.safetensors")
    return missing


def latest_ax_ready_hf_cache_snapshot(repo_id: str, cache_dir: Path) -> Path | None:
    repo_cache = cache_dir / f"models--{_slug_repo_id(repo_id)}"
    snapshots = repo_cache / "snapshots"
    if not snapshots.is_dir():
        return None
    candidates = [
        path
        for path in snapshots.iterdir()
        if path.is_dir() and not missing_ax_model_artifacts(path)
    ]
    return max(candidates, key=lambda path: path.stat().st_mtime, default=None)


def resolve_mlx_model_args(
    row: dict[str, Any],
    *,
    cache_dir: Path,
) -> tuple[list[str] | None, str | None]:
    local_dir_value = row.get("mlx_local_dir")
    if local_dir_value:
        model_dir = REPO_ROOT / local_dir_value
        if model_dir.exists():
            return ["--model-dir", str(model_dir)], None

    repo_id = row.get("mlx_repo_id")
    if not repo_id:
        local_desc = str(REPO_ROOT / local_dir_value) if local_dir_value else "<unset>"
        return None, f"Local MLX dir {local_desc} not found and no mlx_repo_id is configured."

    snapshot = latest_hf_cache_snapshot(repo_id, cache_dir)
    if snapshot is None:
        return None, f"No Hugging Face cache snapshot found for MLX repo {repo_id}."
    missing = missing_ax_model_artifacts(snapshot)
    if missing:
        ax_ready_snapshot = latest_ax_ready_hf_cache_snapshot(repo_id, cache_dir)
        if ax_ready_snapshot is None:
            return None, (
                f"MLX cache snapshot for {repo_id} is not AX-ready: {snapshot}; "
                f"missing {', '.join(missing)}."
            )
        return ["--model-dir", str(ax_ready_snapshot), "--model-repo-id", repo_id], None
    return ["--model-repo-id", repo_id, "--hf-cache-root", str(cache_dir)], None


def _delete_cached_repo(repo: str, cache_dir: Path) -> int:
    """Remove the entire HF cache subtree for one repo, return bytes freed.

    HF cache layout: <cache_dir>/models--<org>--<name>/{blobs,snapshots,refs}.
    Deleting the whole repo dir reclaims real disk because blobs are the
    backing files (snapshots are symlinks)."""
    repo_dir = cache_dir / ("models--" + repo.replace("/", "--"))
    if not repo_dir.exists():
        return 0
    freed = 0
    for root, _dirs, files in os.walk(repo_dir, followlinks=False):
        for name in files:
            with contextlib.suppress(OSError):
                freed += os.lstat(os.path.join(root, name)).st_size
    shutil.rmtree(repo_dir, ignore_errors=True)
    return freed


def run_bench_for_row(
    row: dict[str, Any],
    gguf_path: Path,
    *,
    output_dir: Path,
    bench_script: Path,
    llama_bench: Path,
    prompt_tokens: str,
    generation_tokens: int,
    repetitions: int,
    cooldown: float,
    n_gpu_layers: int,
    extra_args: str | None,
    flash_attn: bool,
    decode_at_depth: bool,
    model_args: list[str],
    full_stack: bool,
    build_ax_engine: bool,
    skip_mlx_lm: bool = False,
    include_mlx_lm: bool = False,
    max_load_average: float | None = DEFAULT_MAX_LOAD_AVERAGE,
    max_top_process_cpu_percent: float | None = DEFAULT_MAX_TOP_PROCESS_CPU_PERCENT,
    load_average_wait_timeout: float | None = DEFAULT_LOAD_WAIT_TIMEOUT_SECONDS,
    load_average_poll_interval: float | None = None,
) -> dict[str, Any]:
    """Invoke bench_mlx_inference_stack.py for one GGUF-mapped README row.

    We pass either --model-dir or --model-repo-id/--hf-cache-root so the
    harness can generate the shape-matching prompt artifact (random tokens at
    the right vocab size). By default the llama.cpp row is the only entry in results[].
    With --full-stack, the same invocation also runs mlx_lm plus AX direct and
    AX n-gram rows unless --skip-mlx-lm is explicitly passed for unsupported
    reference-baseline collection.
    """
    slug = row["slug"]
    out_json = output_dir / f"{slug}.json"
    log_path = output_dir / "logs" / f"{slug}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(bench_script),
        *model_args,
        "--prompt-tokens",
        prompt_tokens,
        "--generation-tokens",
        str(generation_tokens),
        "--repetitions",
        str(repetitions),
        "--cooldown",
        str(cooldown),
        "--llama-cpp-bench",
        str(llama_bench),
        "--llama-cpp-gguf",
        str(gguf_path),
        "--llama-cpp-n-gpu-layers",
        str(n_gpu_layers),
        "--output",
        str(out_json),
    ]
    llama_cpp_extra_args = extra_args
    if flash_attn:
        llama_cpp_extra_args = (
            "-fa 1"
            if not llama_cpp_extra_args
            else f"-fa 1 {llama_cpp_extra_args}"
        )
    if decode_at_depth:
        cmd.append("--llama-cpp-decode-at-depth")
    if full_stack:
        cmd.append("--ax-compare-policies")
        if skip_mlx_lm:
            cmd.append("--skip-mlx-lm")
        if not build_ax_engine:
            cmd.append("--no-build-ax-engine")
    else:
        cmd.extend(["--skip-ax-engine", "--no-build-ax-engine"])
        if not include_mlx_lm:
            cmd.append("--skip-mlx-lm")
    if max_load_average is not None:
        cmd.extend(["--max-load-average", str(max_load_average)])
    if max_top_process_cpu_percent is not None:
        cmd.extend(
            [
                "--max-top-process-cpu-percent",
                str(max_top_process_cpu_percent),
            ]
        )
    if load_average_wait_timeout is not None:
        cmd.extend(
            [
                "--load-average-wait-timeout",
                str(load_average_wait_timeout),
            ]
        )
    if load_average_poll_interval is not None:
        cmd.extend(
            [
                "--load-average-poll-interval",
                str(load_average_poll_interval),
            ]
        )
    if llama_cpp_extra_args:
        cmd.extend(["--llama-cpp-extra-args", llama_cpp_extra_args])

    log(f"  invoke: {' '.join(cmd)}")
    with log_path.open("w") as fh:
        result = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        return {
            "status": "bench_failed",
            "exit_code": result.returncode,
            "log_path": str(log_path),
            "output_path": str(out_json) if out_json.exists() else None,
        }
    if not out_json.exists():
        return {
            "status": "bench_failed_no_output",
            "log_path": str(log_path),
        }
    with out_json.open() as fh:
        doc = json.load(fh)
    return {
        "status": "ok",
        "output_path": str(out_json),
        "log_path": str(log_path),
        "result_doc": doc,
    }


def _repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def update_readme_source_marker(readme: Path, output_root: Path) -> None:
    import re

    rel = _repo_relative(output_root)
    text = readme.read_text()
    marker_re = re.compile(
        r"<!--\s*readme-performance-artifacts:\s*(?P<body>.*?)\s*-->",
        re.DOTALL,
    )
    marker = (
        "<!-- readme-performance-artifacts: "
        f"reference={rel}/; ax-base={rel}/ -->"
    )
    text, count = marker_re.subn(marker, text, count=1)
    if count == 0:
        raise RuntimeError("README does not contain readme-performance-artifacts marker")

    text = re.sub(
        r"These rows are a provenance-tracked (?:composite|result set) from\n`[^`]+`\.",
        f"These rows are a provenance-tracked result set from\n`{rel}/`.",
        text,
        count=1,
    )
    readme.write_text(text)


def check_readme_performance_tables(readme: Path) -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "check_readme_performance_artifacts.py"),
        "--readme",
        str(readme),
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


PERFORMANCE_TABLE_PREFIXES = (
    "### Prefill throughput",
    "### Decode throughput",
    "### Time to first token",
)


def _split_table_cells(line: str) -> list[str]:
    parts = line.split("|")
    if len(parts) < 3:
        return []
    return [part.strip() for part in parts[1:-1]]


def prune_readme_performance_rows(
    readme: Path,
    *,
    allowed_rows: set[tuple[str, str]],
) -> None:
    """Remove README performance table rows outside the selected sweep set.

    Full-stack README refreshes should describe exactly the models that were
    freshly measured. This prevents old rows, such as Qwen 3.6 5/6/8-bit, from
    surviving under a new artifact marker when the sweep intentionally covers
    only Qwen 3.6 4-bit.
    """
    lines = readme.read_text().splitlines()
    out: list[str] = []
    in_perf_table = False
    current_pair: tuple[str, str] | None = None

    for line in lines:
        if line.startswith("### ") or line.startswith("## "):
            in_perf_table = any(line.startswith(prefix) for prefix in PERFORMANCE_TABLE_PREFIXES)
            current_pair = None
            out.append(line)
            continue

        if not in_perf_table or not line.startswith("|"):
            out.append(line)
            continue

        cells = _split_table_cells(line)
        if len(cells) < 3:
            out.append(line)
            continue

        if cells[0] == "Model" or set(cells[0]) <= {"-"}:
            out.append(line)
            continue

        if cells[0]:
            current_pair = (cells[0], cells[1])

        if current_pair is not None and current_pair not in allowed_rows:
            continue
        out.append(line)

    readme.write_text("\n".join(out) + "\n")


def update_readme_from_sweep(
    *,
    readme: Path,
    sweep_path: Path,
    sweep_doc: dict[str, Any],
    full_stack: bool,
    output_root: Path,
    allow_partial: bool,
) -> None:
    if full_stack and not allow_partial:
        incomplete = [
            f"{row.get('slug')}={row.get('status')}"
            for row in sweep_doc["rows"]
            if row.get("status") != "ok"
        ]
        if incomplete:
            details = ", ".join(incomplete)
            raise RuntimeError(
                "Refusing to update README from an incomplete full-stack sweep. "
                f"Pass --allow-partial-readme-update to override. Incomplete rows: {details}"
            )

    if full_stack:
        allowed_rows = {
            (row["readme_model"], row["readme_quant"])
            for row in sweep_doc["rows"]
            if row.get("status") == "ok"
        }
        prune_readme_performance_rows(readme, allowed_rows=allowed_rows)
        for row in sweep_doc["rows"]:
            if row.get("status") != "ok" or not row.get("output_path"):
                continue
            cmd = [
                sys.executable,
                str(REPO_ROOT / "scripts" / "update_readme_from_bench.py"),
                "--slug",
                row["slug"],
                "--json",
                row["output_path"],
                "--readme",
                str(readme),
            ]
            subprocess.run(cmd, cwd=REPO_ROOT, check=True)

    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "update_readme_inject_llama_cpp.py"),
            "--sweep",
            str(sweep_path),
            "--readme",
            str(readme),
        ],
        cwd=REPO_ROOT,
        check=True,
    )

    if full_stack:
        update_readme_source_marker(readme, output_root)
        check_readme_performance_tables(readme)


def positive_metric_median(row: dict[str, Any], key: str) -> float | None:
    metric = row.get(key)
    value = metric.get("median") if isinstance(metric, dict) else None
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        return None
    return float(value)


def valid_positive_trials(
    row: dict[str, Any],
    *,
    trials_key: str,
    metric_key: str,
    repetitions: int,
) -> bool:
    trials = row.get(trials_key)
    if not isinstance(trials, list) or len(trials) != repetitions:
        return False
    observed_trials: set[int] = set()
    for trial in trials:
        if not isinstance(trial, dict):
            return False
        trial_number = trial.get("trial")
        metric = trial.get(metric_key)
        sample_ns = trial.get("sample_ns")
        if (
            not isinstance(trial_number, int)
            or isinstance(trial_number, bool)
            or trial_number in observed_trials
            or not isinstance(metric, (int, float))
            or isinstance(metric, bool)
            or not math.isfinite(float(metric))
            or float(metric) <= 0.0
            or not isinstance(sample_ns, (int, float))
            or isinstance(sample_ns, bool)
            or not math.isfinite(float(sample_ns))
            or float(sample_ns) <= 0.0
        ):
            return False
        observed_trials.add(trial_number)
    return observed_trials == set(range(1, repetitions + 1))


def llama_result_doc_publication_reasons(
    doc: dict[str, Any],
    *,
    expected_prompt_tokens: set[int],
    generation_tokens: int,
    repetitions: int,
    cooldown: float,
    require_flash_attn: bool,
    require_decode_at_depth: bool,
) -> tuple[list[str], set[tuple[int, str, str]]]:
    reasons: list[str] = []
    identities: set[tuple[int, str, str]] = set()
    if doc.get("schema_version") != "ax.mlx_inference_stack.v2":
        reasons.append("unexpected_artifact_schema")
    if doc.get("generation_tokens") != generation_tokens:
        reasons.append("generation_tokens_mismatch")
    if doc.get("repetitions") != repetitions:
        reasons.append("repetitions_mismatch")
    warmup_repetitions = doc.get("warmup_repetitions")
    if (
        not isinstance(warmup_repetitions, int)
        or isinstance(warmup_repetitions, bool)
        or warmup_repetitions < 2
    ):
        reasons.append("requires_two_warmups")
    if doc.get("cooldown") != cooldown:
        reasons.append("cooldown_mismatch")

    build = doc.get("build")
    commit = build.get("commit") if isinstance(build, dict) else None
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        reasons.append("missing_harness_build_commit")
    if not isinstance(build, dict) or build.get("build_profile") != "release":
        reasons.append("non_release_harness_build")
    if not isinstance(build, dict) or build.get("git_tracked_dirty") is not False:
        reasons.append("dirty_harness_build")
    reasons.extend(publication_gate.publication_condition_reasons("llama", doc))

    results = doc.get("results")
    if not isinstance(results, list):
        return [*reasons, "missing_results"], identities
    observed_prompt_tokens: set[int] = set()
    for row in results:
        if not isinstance(row, dict) or row.get("engine") != "llama_cpp_metal":
            reasons.append("unexpected_non_llama_result")
            continue
        prompt_tokens = row.get("prompt_tokens")
        if (
            not isinstance(prompt_tokens, int)
            or isinstance(prompt_tokens, bool)
            or prompt_tokens in observed_prompt_tokens
        ):
            reasons.append("invalid_or_duplicate_prompt_tokens")
            continue
        observed_prompt_tokens.add(prompt_tokens)
        if row.get("generation_tokens") != generation_tokens:
            reasons.append(f"p{prompt_tokens}_generation_tokens_mismatch")
        prompt_hash = row.get("prompt_token_ids_sha256")
        if not isinstance(prompt_hash, str) or re.fullmatch(
            r"[0-9a-f]{64}", prompt_hash
        ) is None:
            reasons.append(f"p{prompt_tokens}_missing_prompt_hash")
        for metric in ("prefill_tok_s", "decode_tok_s", "ttft_ms"):
            if positive_metric_median(row, metric) is None:
                reasons.append(f"p{prompt_tokens}_invalid_{metric}")
        if not valid_positive_trials(
            row,
            trials_key="prefill_trials",
            metric_key="prefill_tok_s",
            repetitions=repetitions,
        ):
            reasons.append(f"p{prompt_tokens}_invalid_prefill_trials")
        if not valid_positive_trials(
            row,
            trials_key="decode_trials",
            metric_key="decode_tok_s",
            repetitions=repetitions,
        ):
            reasons.append(f"p{prompt_tokens}_invalid_decode_trials")
        if require_decode_at_depth:
            if positive_metric_median(row, "decode_at_depth_tok_s") is None:
                reasons.append(f"p{prompt_tokens}_missing_depth_decode")
            if not valid_positive_trials(
                row,
                trials_key="decode_at_depth_trials",
                metric_key="decode_at_depth_tok_s",
                repetitions=repetitions,
            ):
                reasons.append(f"p{prompt_tokens}_invalid_depth_trials")

        llama = row.get("llama_cpp")
        if not isinstance(llama, dict):
            reasons.append(f"p{prompt_tokens}_missing_llama_identity")
            continue
        build_number = llama.get("build_number")
        build_commit = llama.get("build_commit")
        backends = llama.get("backends")
        gpu_info = llama.get("gpu_info")
        backend_names = (
            {name.strip() for name in backends.split(",")}
            if isinstance(backends, str)
            else set()
        )
        if (
            not isinstance(build_number, int)
            or isinstance(build_number, bool)
            or build_number <= 0
            or not isinstance(build_commit, str)
            or re.fullmatch(r"[0-9a-f]{7,40}", build_commit) is None
            or "MTL" not in backend_names
            or not isinstance(gpu_info, str)
            or "Apple" not in gpu_info
        ):
            reasons.append(f"p{prompt_tokens}_invalid_llama_identity")
            continue
        if require_flash_attn and llama.get("flash_attn") != 1:
            reasons.append(f"p{prompt_tokens}_flash_attention_disabled")
        if require_decode_at_depth:
            depth = row.get("llama_cpp_depth")
            if (
                not isinstance(depth, dict)
                or depth.get("build_number") != build_number
                or depth.get("build_commit") != build_commit
                or depth.get("gpu_info") != gpu_info
                or depth.get("n_depth") != prompt_tokens
                or (
                    require_flash_attn
                    and depth.get("flash_attn") != 1
                )
            ):
                reasons.append(f"p{prompt_tokens}_invalid_depth_identity")
        identities.add((build_number, build_commit, gpu_info))

    if observed_prompt_tokens != expected_prompt_tokens:
        reasons.append("prompt_matrix_mismatch")
    if len(identities) != 1:
        reasons.append("mixed_or_missing_llama_identity")
    return sorted(set(reasons)), identities


def build_llama_publication_matrix(
    *,
    manifest_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    prompt_tokens: str,
    generation_tokens: int,
    repetitions: int,
    cooldown: float,
    require_flash_attn: bool,
    require_decode_at_depth: bool,
    max_load_average: float | None,
    max_top_process_cpu_percent: float | None,
    full_scope: bool,
) -> dict[str, Any]:
    expected_slugs = [
        _row_slug(row)
        for row in manifest_rows
        if row.get("readme_direct_table") is not False
    ]
    try:
        expected_prompt_tokens = {
            int(value.strip()) for value in prompt_tokens.split(",") if value.strip()
        }
    except ValueError:
        expected_prompt_tokens = set()
    rows_by_slug: dict[str, dict[str, Any]] = {}
    row_counts: dict[str, int] = {}
    for row in summary_rows:
        if not isinstance(row, dict) or not isinstance(row.get("slug"), str):
            continue
        slug = str(row["slug"])
        row_counts[slug] = row_counts.get(slug, 0) + 1
        rows_by_slug[slug] = row
    models: list[dict[str, Any]] = []
    failure_reasons: list[str] = []
    identities: set[tuple[int, str, str]] = set()
    for slug in expected_slugs:
        record = rows_by_slug.get(slug)
        reasons: list[str] = []
        if row_counts.get(slug, 0) != 1:
            reasons.append("missing_or_duplicate_sweep_row")
        elif not isinstance(record, dict):
            reasons.append("missing_sweep_row")
        elif record.get("status") != "ok":
            reasons.append(f"sweep_status_{record.get('status', 'missing')}")
        else:
            result_doc = record.get("result_doc")
            if not isinstance(result_doc, dict):
                reasons.append("missing_result_doc")
            else:
                row_reasons, row_identities = llama_result_doc_publication_reasons(
                    result_doc,
                    expected_prompt_tokens=expected_prompt_tokens,
                    generation_tokens=generation_tokens,
                    repetitions=repetitions,
                    cooldown=cooldown,
                    require_flash_attn=require_flash_attn,
                    require_decode_at_depth=require_decode_at_depth,
                )
                reasons.extend(row_reasons)
                identities.update(row_identities)
        failure_reasons.extend(f"{slug}:{reason}" for reason in reasons)
        models.append(
            {
                "slug": slug,
                "publication_candidate": not reasons,
                "publication_reasons": reasons,
            }
        )

    global_reasons: list[str] = []
    if not full_scope:
        global_reasons.append("filtered_scope_is_not_full_readme_matrix")
    if len(expected_slugs) != README_PUBLICATION_MODEL_COUNT:
        global_reasons.append(
            f"publication_requires_{README_PUBLICATION_MODEL_COUNT}_readme_models"
        )
    if expected_prompt_tokens != {128, 512, 2048}:
        global_reasons.append("publication_requires_prompt_tokens_128_512_2048")
    if generation_tokens != 128:
        global_reasons.append("publication_requires_generation_tokens_128")
    if repetitions < 5:
        global_reasons.append("publication_requires_five_repetitions")
    if not math.isfinite(cooldown) or cooldown < 15.0:
        global_reasons.append("publication_requires_15s_cooldown")
    if not require_flash_attn:
        global_reasons.append("publication_requires_flash_attention")
    if not require_decode_at_depth:
        global_reasons.append("publication_requires_depth_matched_decode")
    if (
        max_load_average is None
        or not math.isfinite(max_load_average)
        or max_load_average > DEFAULT_MAX_LOAD_AVERAGE
    ):
        global_reasons.append("publication_requires_default_load_gate")
    if (
        max_top_process_cpu_percent is None
        or not math.isfinite(max_top_process_cpu_percent)
        or max_top_process_cpu_percent > DEFAULT_MAX_TOP_PROCESS_CPU_PERCENT
    ):
        global_reasons.append("publication_requires_default_process_cpu_gate")
    if len(identities) != 1:
        global_reasons.append("matrix_has_mixed_or_missing_llama_identity")
    all_reasons = [*failure_reasons, *global_reasons]
    identity = next(iter(identities)) if len(identities) == 1 else None
    return {
        "schema_version": LLAMA_CPP_PUBLICATION_MATRIX_SCHEMA,
        "scope": "readme_llama_cpp_metal_snapshot",
        "expected_slugs": expected_slugs,
        "expected_model_count": len(expected_slugs),
        "publication_model_count": sum(
            model["publication_candidate"] for model in models
        ),
        "publication_candidate": bool(expected_slugs) and not all_reasons,
        "publication_reasons": all_reasons,
        "llama_cpp_identity": (
            {
                "build_number": identity[0],
                "build_commit": identity[1],
                "gpu_info": identity[2],
            }
            if identity is not None
            else None
        ),
        "models": models,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--bench-script", type=Path, default=DEFAULT_BENCH_SCRIPT)
    parser.add_argument("--llama-bench", type=Path, default=DEFAULT_LLAMA_BENCH)
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Directory to write per-row JSON, sweep_results.json, and sweep_summary.md.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path.home() / ".cache" / "huggingface" / "hub",
        help="HF download cache root. Existing files are reused.",
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help=(
            "Resolve GGUF and MLX artifacts only from --cache-dir. Do not call "
            "Hugging Face metadata APIs and do not download missing files."
        ),
    )
    parser.add_argument(
        "--rows-filter",
        nargs="*",
        help="If set, only process rows whose slug is in this list.",
    )
    parser.add_argument("--prompt-tokens", default="128,512,2048")
    parser.add_argument("--generation-tokens", type=int, default=128)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--cooldown", type=float, default=15.0)
    parser.add_argument(
        "--max-load-average",
        type=float,
        default=DEFAULT_MAX_LOAD_AVERAGE,
        help=(
            "Wait for the one-minute load average to be at or below this value "
            "before benchmark phases. Use a negative value only to trigger an "
            "argument error."
        ),
    )
    parser.add_argument(
        "--max-top-process-cpu-percent",
        type=float,
        default=DEFAULT_MAX_TOP_PROCESS_CPU_PERCENT,
        help=(
            "Wait until every sampled top process is at or below this CPU "
            "percentage before benchmark phases."
        ),
    )
    parser.add_argument(
        "--load-average-wait-timeout",
        type=float,
        default=DEFAULT_LOAD_WAIT_TIMEOUT_SECONDS,
        help=(
            "Maximum seconds to wait for the publication performance gates "
            "before failing a row. "
            f"Default: {DEFAULT_LOAD_WAIT_TIMEOUT_SECONDS:.0f}."
        ),
    )
    parser.add_argument(
        "--load-average-poll-interval",
        type=float,
        default=None,
        help="Forwarded to the benchmark harness when a load gate is enabled.",
    )
    parser.add_argument("--n-gpu-layers", type=int, default=99)
    parser.add_argument(
        "--extra-args",
        help="Forwarded to bench_mlx_inference_stack.py --llama-cpp-extra-args.",
    )
    parser.add_argument(
        "--llama-cpp-flash-attn",
        action="store_true",
        help="Forward '-fa 1' to llama-bench for llama.cpp Metal rows.",
    )
    parser.add_argument(
        "--llama-cpp-decode-at-depth",
        action="store_true",
        help=(
            "Forward --llama-cpp-decode-at-depth so each llama.cpp row also "
            "records `llama-bench -p 0 -n <generation> -d <prompt>` decode "
            "evidence."
        ),
    )
    parser.add_argument(
        "--keep-gguf",
        action="store_true",
        help="Keep downloaded GGUFs after each row (default: delete to save disk).",
    )
    parser.add_argument(
        "--full-stack",
        action="store_true",
        help=(
            "For each GGUF-resolved row, run llama.cpp Metal, mlx_lm.benchmark, "
            "AX direct, and AX default n-gram rows in one artifact."
        ),
    )
    parser.add_argument(
        "--include-mlx-lm",
        action="store_true",
        help=(
            "In non-full-stack mode, also run mlx_lm.benchmark alongside "
            "llama.cpp so the resulting JSON has both engines (AX still skipped)."
        ),
    )
    parser.add_argument(
        "--skip-mlx-lm",
        action="store_true",
        help=(
            "With --full-stack, skip the mlx_lm.benchmark baseline and still "
            "collect AX plus llama.cpp rows. Default full-stack behavior remains "
            "fail-closed when mlx_lm.benchmark fails."
        ),
    )
    parser.add_argument(
        "--no-build-ax-engine",
        action="store_true",
        help=(
            "With --full-stack, skip the release server build and use the "
            "existing target/release/ax-engine-server binary."
        ),
    )
    parser.add_argument(
        "--update-readme",
        action="store_true",
        help=(
            "Update README.md from the sweep. With --full-stack this updates "
            "mlx_lm/AX rows plus llama.cpp columns; otherwise only the "
            "llama.cpp columns are refreshed."
        ),
    )
    parser.add_argument(
        "--allow-partial-readme-update",
        action="store_true",
        help=(
            "Allow --update-readme even when some full-stack rows failed or "
            "were skipped. By default, full-stack README updates fail closed."
        ),
    )
    parser.add_argument(
        "--readme",
        type=Path,
        default=REPO_ROOT / "docs" / "PERFORMANCE-RESULTS.md",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve candidates and print plan; do not download or benchmark.",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help=(
            "Resolve and download GGUF candidates into the Hugging Face cache, "
            "then write sweep_results.json without running llama-bench."
        ),
    )
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
        help="HF token for gated repos. Defaults to $HF_TOKEN or $HUGGING_FACE_HUB_TOKEN.",
    )
    args = parser.parse_args()
    if not math.isfinite(args.max_load_average) or args.max_load_average < 0.0:
        parser.error("--max-load-average must be finite and non-negative")
    if (
        not math.isfinite(args.max_top_process_cpu_percent)
        or args.max_top_process_cpu_percent < 0.0
    ):
        parser.error(
            "--max-top-process-cpu-percent must be finite and non-negative"
        )
    if (
        not math.isfinite(args.load_average_wait_timeout)
        or args.load_average_wait_timeout < 0.0
    ):
        parser.error(
            "--load-average-wait-timeout must be finite and non-negative"
        )
    if (
        args.load_average_poll_interval is not None
        and (
            not math.isfinite(args.load_average_poll_interval)
            or args.load_average_poll_interval <= 0.0
        )
    ):
        parser.error("--load-average-poll-interval must be finite and positive")

    try:
        with args.manifest.open() as fh:
            manifest = json.load(fh)
        if not isinstance(manifest, dict):
            raise LlamaCppMetalSweepError("manifest root must be an object")
        raw_rows = manifest.get("rows")
        if not isinstance(raw_rows, list):
            raise LlamaCppMetalSweepError("manifest.rows must be an array")
        rows = filter_manifest_rows(raw_rows, args.rows_filter)
        validate_gguf_publisher_inventory(manifest, rows)
        validate_mlx_publisher_inventory(manifest, rows)
    except (json.JSONDecodeError, LlamaCppMetalSweepError, RuntimeError) as exc:
        log(f"ERROR: {exc}")
        sys.exit(2)

    if not args.dry_run and not args.download_only and not args.llama_bench.exists():
        log(f"ERROR: llama-bench binary not found: {args.llama_bench}")
        sys.exit(2)

    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "logs").mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    total_bytes_downloaded = 0
    total_bytes_freed = 0
    started = time.time()

    for index, row in enumerate(rows, start=1):
        slug = row["slug"]
        log(f"({index}/{len(rows)}) {slug}")

        record: dict[str, Any] = {
            "slug": slug,
            "readme_model": row["readme_model"],
            "readme_quant": row["readme_quant"],
            "llama_cpp_arch": row.get("llama_cpp_arch"),
            "gguf_quant_target": row.get("gguf_quant_target"),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }

        try:
            resolved = resolve_gguf_candidate(
                row["gguf_candidates"],
                cache_dir=args.cache_dir,
                hf_token=args.hf_token,
                cache_only=args.cache_only,
            )
        except Exception as exc:
            record["status"] = "resolution_error"
            record["error"] = f"{type(exc).__name__}: {exc}"
            summary_rows.append(record)
            continue

        if resolved is None:
            record["status"] = "unresolved"
            record["note"] = (
                "No GGUF candidate matched on HF. Architecture may not yet "
                "have a public GGUF conversion."
            )
            summary_rows.append(record)
            log("  -> unresolved")
            continue

        repo, filename, probe_log = resolved
        record["resolved_repo"] = repo
        record["resolved_filename"] = filename
        record["probe_log"] = probe_log

        if args.dry_run:
            record["status"] = "dry_run_resolved"
            summary_rows.append(record)
            log(f"  -> dry-run resolved: {repo} :: {filename}")
            continue

        model_args: list[str] | None = None
        if not args.download_only:
            model_args, missing_model_note = resolve_mlx_model_args(row, cache_dir=args.cache_dir)
            if model_args is None:
                record["status"] = "mlx_model_dir_missing"
                record["note"] = f"{missing_model_note} Cannot generate prompt artifact."
                summary_rows.append(record)
                log(f"  -> skipped: {record['note']}")
                continue

        try:
            gguf_path = download_gguf(
                repo,
                filename,
                cache_dir=args.cache_dir,
                hf_token=args.hf_token,
                cache_only=args.cache_only,
            )
        except Exception as exc:
            record["status"] = "download_failed"
            record["error"] = f"{type(exc).__name__}: {exc}"
            summary_rows.append(record)
            log(f"  -> download failed: {exc}")
            continue

        record["gguf_path"] = str(gguf_path)
        size_bytes = gguf_disk_bytes(gguf_path)
        record["gguf_size_bytes"] = size_bytes
        total_bytes_downloaded += size_bytes
        log(f"  -> GGUF ready ({size_bytes / 1e9:.2f} GB)")

        if args.download_only:
            record["status"] = "downloaded"
            summary_rows.append(record)
            continue

        bench_result = run_bench_for_row(
            row,
            gguf_path,
            output_dir=args.output_root,
            bench_script=args.bench_script,
            llama_bench=args.llama_bench,
            prompt_tokens=args.prompt_tokens,
            generation_tokens=args.generation_tokens,
            repetitions=args.repetitions,
            cooldown=args.cooldown,
            n_gpu_layers=args.n_gpu_layers,
            extra_args=args.extra_args,
            flash_attn=args.llama_cpp_flash_attn,
            decode_at_depth=args.llama_cpp_decode_at_depth,
            model_args=model_args or [],
            full_stack=args.full_stack,
            build_ax_engine=not args.no_build_ax_engine,
            skip_mlx_lm=args.skip_mlx_lm,
            include_mlx_lm=args.include_mlx_lm,
            max_load_average=args.max_load_average,
            max_top_process_cpu_percent=args.max_top_process_cpu_percent,
            load_average_wait_timeout=args.load_average_wait_timeout,
            load_average_poll_interval=args.load_average_poll_interval,
        )
        record.update(bench_result)

        if not args.keep_gguf:
            freed = _delete_cached_repo(repo, args.cache_dir)
            record["gguf_bytes_freed"] = freed
            total_bytes_freed += freed
            log(f"  -> deleted cached repo {repo} (freed {freed / 1e9:.2f} GB)")

        summary_rows.append(record)

    elapsed = time.time() - started
    publication_matrix = build_llama_publication_matrix(
        manifest_rows=raw_rows,
        summary_rows=summary_rows,
        prompt_tokens=args.prompt_tokens,
        generation_tokens=args.generation_tokens,
        repetitions=args.repetitions,
        cooldown=args.cooldown,
        require_flash_attn=args.llama_cpp_flash_attn,
        require_decode_at_depth=args.llama_cpp_decode_at_depth,
        max_load_average=args.max_load_average,
        max_top_process_cpu_percent=args.max_top_process_cpu_percent,
        full_scope=args.rows_filter is None,
    )
    sweep_doc = {
        "schema_version": "ax.llama_cpp_metal_sweep.v1",
        "claim_boundary": manifest.get("claim_boundary"),
        "quant_mapping_policy": manifest.get("quant_mapping_policy"),
        "manifest_path": str(args.manifest),
        "llama_bench": str(args.llama_bench),
        "prompt_tokens": args.prompt_tokens,
        "generation_tokens": args.generation_tokens,
        "repetitions": args.repetitions,
        "cooldown": args.cooldown,
        "max_load_average": args.max_load_average,
        "max_top_process_cpu_percent": args.max_top_process_cpu_percent,
        "load_average_wait_timeout": args.load_average_wait_timeout,
        "load_average_poll_interval": args.load_average_poll_interval,
        "n_gpu_layers": args.n_gpu_layers,
        "extra_args": args.extra_args,
        "llama_cpp_flash_attn": args.llama_cpp_flash_attn,
        "llama_cpp_decode_at_depth": args.llama_cpp_decode_at_depth,
        "full_stack": args.full_stack,
        "skip_mlx_lm": args.skip_mlx_lm,
        "download_only": args.download_only,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(started)),
        "elapsed_seconds": round(elapsed, 1),
        "total_bytes_downloaded": total_bytes_downloaded,
        "total_bytes_freed": total_bytes_freed,
        "keep_gguf": args.keep_gguf,
        "cache_only": args.cache_only,
        "publication_candidate": publication_matrix["publication_candidate"],
        "readme_llama_cpp_publication_candidate": publication_matrix[
            "publication_candidate"
        ],
        "llama_cpp_publication_matrix": publication_matrix,
        "rows": summary_rows,
    }
    sweep_path = args.output_root / "sweep_results.json"
    sweep_path.write_text(json.dumps(sweep_doc, indent=2))
    log(f"wrote {sweep_path}")

    summary_md = args.output_root / "sweep_summary.md"
    summary_md.write_text(_render_summary_md(sweep_doc))
    log(f"wrote {summary_md}")

    if args.update_readme and not publication_matrix["publication_candidate"]:
        log("ERROR: refusing to update README from a non-publication sweep")
        for reason in publication_matrix["publication_reasons"]:
            log(f"  - {reason}")
        sys.exit(2)

    if args.update_readme:
        update_readme_from_sweep(
            readme=args.readme,
            sweep_path=sweep_path,
            sweep_doc=sweep_doc,
            full_stack=args.full_stack,
            output_root=args.output_root,
            allow_partial=args.allow_partial_readme_update,
        )
        log(f"updated {args.readme}")

    if (
        not args.dry_run
        and not args.download_only
        and args.rows_filter is None
        and not publication_matrix["publication_candidate"]
    ):
        log("ERROR: llama.cpp README publication matrix is incomplete or ineligible")
        for reason in publication_matrix["publication_reasons"]:
            log(f"  - {reason}")
        sys.exit(2)


def _render_summary_md(doc: dict[str, Any]) -> str:
    lines = ["# llama.cpp Metal sweep summary", ""]
    lines.append(f"- elapsed: {doc['elapsed_seconds']:.0f}s")
    lines.append(f"- downloaded: {doc['total_bytes_downloaded'] / 1e9:.1f} GB")
    lines.append(f"- freed: {doc['total_bytes_freed'] / 1e9:.1f} GB")
    matrix = doc["llama_cpp_publication_matrix"]
    lines.append(
        "- README publication candidate: "
        f"{str(matrix['publication_candidate']).lower()} "
        f"({matrix['publication_model_count']}/{matrix['expected_model_count']} models)"
    )
    identity = matrix.get("llama_cpp_identity")
    if isinstance(identity, dict):
        lines.append(
            "- llama.cpp identity: "
            f"build {identity['build_number']} ({identity['build_commit']}), "
            f"{identity['gpu_info']}"
        )
    if matrix["publication_reasons"]:
        lines.append(
            "- publication blockers: "
            + ", ".join(str(reason) for reason in matrix["publication_reasons"])
        )
    lines.append("")
    lines.append("| slug | status | repo | quant | notes |")
    lines.append("|---|---|---|---|---|")
    for r in doc["rows"]:
        lines.append(
            "| {slug} | {status} | {repo} | {quant} | {notes} |".format(
                slug=r["slug"],
                status=r.get("status", "?"),
                repo=r.get("resolved_repo", "-"),
                quant=r.get("gguf_quant_target", "-"),
                notes=r.get("note") or r.get("error") or "",
            )
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
