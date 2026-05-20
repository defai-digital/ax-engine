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
import fnmatch
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "benchmarks" / "manifests" / "llama_cpp_metal" / "inventory.json"
DEFAULT_BENCH_SCRIPT = REPO_ROOT / "scripts" / "bench_mlx_inference_stack.py"
DEFAULT_LLAMA_BENCH = Path("/opt/homebrew/bin/llama-bench")
REQUIRED_GGUF_PUBLISHER = "bartowski"


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


def validate_bartowski_inventory(manifest: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    publisher = manifest.get("required_gguf_publisher", REQUIRED_GGUF_PUBLISHER)
    if publisher != REQUIRED_GGUF_PUBLISHER:
        raise RuntimeError(
            f"unsupported required_gguf_publisher={publisher!r}; expected {REQUIRED_GGUF_PUBLISHER!r}"
        )

    bad: list[str] = []
    prefix = f"{REQUIRED_GGUF_PUBLISHER}/"
    for row in rows:
        for candidate in row.get("gguf_candidates", []):
            repo = candidate.get("repo", "")
            if not repo.startswith(prefix):
                bad.append(f"{row.get('slug', '<unknown>')} -> {repo}")

    if bad:
        details = "; ".join(bad)
        raise RuntimeError(
            "llama.cpp Metal sweep inventory must use bartowski GGUF repos only: "
            f"{details}"
        )


def resolve_gguf_candidate(
    candidates: list[dict[str, str]],
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
        # If the GGUF is split into shards (e.g. *-00001-of-00002.gguf), prefer
        # the first shard; hf_hub_download will fetch the matching shard and
        # llama-bench can read split files when pointed at the first one.
        matches.sort()
        entry["result"] = "resolved"
        entry["filename"] = matches[0]
        probe_log.append(entry)
        return repo, matches[0], probe_log
    return None


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
    matches = sorted(
        path
        for path in snapshot.rglob("*.gguf")
        if fnmatch.fnmatch(path.name, filename_pattern) or fnmatch.fnmatch(str(path.relative_to(snapshot)), filename_pattern)
    )
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
        return None, (
            f"MLX cache snapshot for {repo_id} is not AX-ready: {snapshot}; "
            f"missing {', '.join(missing)}."
        )
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
            try:
                freed += os.lstat(os.path.join(root, name)).st_size
            except OSError:
                pass
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
    include_mlx_lm: bool = False,
) -> dict[str, Any]:
    """Invoke bench_mlx_inference_stack.py for one GGUF-mapped README row.

    We pass either --model-dir or --model-repo-id/--hf-cache-root so the
    harness can generate the shape-matching prompt artifact (random tokens at
    the right vocab size). By default the llama.cpp row is the only entry in results[].
    With --full-stack, the same invocation also runs mlx_lm plus AX direct and
    AX n-gram rows.
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
        llama_cpp_extra_args = "-fa 1" if not llama_cpp_extra_args else f"-fa 1 {llama_cpp_extra_args}"
    if decode_at_depth:
        cmd.append("--llama-cpp-decode-at-depth")
    if full_stack:
        cmd.append("--ax-compare-policies")
        if not build_ax_engine:
            cmd.append("--no-build-ax-engine")
    else:
        cmd.extend(["--skip-ax-engine", "--no-build-ax-engine"])
        if not include_mlx_lm:
            cmd.append("--skip-mlx-lm")
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
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--cooldown", type=float, default=15.0)
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
    parser.add_argument("--readme", type=Path, default=REPO_ROOT / "README.md")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve candidates and print plan; do not download or benchmark.",
    )
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
        help="HF token for gated repos. Defaults to $HF_TOKEN or $HUGGING_FACE_HUB_TOKEN.",
    )
    args = parser.parse_args()

    try:
        with args.manifest.open() as fh:
            manifest = json.load(fh)
        if not isinstance(manifest, dict):
            raise LlamaCppMetalSweepError("manifest root must be an object")
        raw_rows = manifest.get("rows")
        if not isinstance(raw_rows, list):
            raise LlamaCppMetalSweepError("manifest.rows must be an array")
        rows = filter_manifest_rows(raw_rows, args.rows_filter)
        validate_bartowski_inventory(manifest, rows)
    except (json.JSONDecodeError, LlamaCppMetalSweepError, RuntimeError) as exc:
        log(f"ERROR: {exc}")
        sys.exit(2)

    if not args.dry_run and not args.llama_bench.exists():
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
            record["note"] = "No GGUF candidate matched on HF. Architecture may not yet have a public GGUF conversion."
            summary_rows.append(record)
            log(f"  -> unresolved")
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
            model_args=model_args,
            full_stack=args.full_stack,
            build_ax_engine=not args.no_build_ax_engine,
            include_mlx_lm=args.include_mlx_lm,
        )
        record.update(bench_result)

        if not args.keep_gguf:
            freed = _delete_cached_repo(repo, args.cache_dir)
            record["gguf_bytes_freed"] = freed
            total_bytes_freed += freed
            log(f"  -> deleted cached repo {repo} (freed {freed / 1e9:.2f} GB)")

        summary_rows.append(record)

    elapsed = time.time() - started
    sweep_doc = {
        "schema_version": "ax.llama_cpp_metal_sweep.v1",
        "claim_boundary": manifest.get("claim_boundary"),
        "quant_mapping_policy": manifest.get("quant_mapping_policy"),
        "manifest_path": str(args.manifest),
        "llama_bench": str(args.llama_bench),
        "prompt_tokens": args.prompt_tokens,
        "generation_tokens": args.generation_tokens,
        "repetitions": args.repetitions,
        "n_gpu_layers": args.n_gpu_layers,
        "extra_args": args.extra_args,
        "llama_cpp_flash_attn": args.llama_cpp_flash_attn,
        "llama_cpp_decode_at_depth": args.llama_cpp_decode_at_depth,
        "full_stack": args.full_stack,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(started)),
        "elapsed_seconds": round(elapsed, 1),
        "total_bytes_downloaded": total_bytes_downloaded,
        "total_bytes_freed": total_bytes_freed,
        "keep_gguf": args.keep_gguf,
        "cache_only": args.cache_only,
        "rows": summary_rows,
    }
    sweep_path = args.output_root / "sweep_results.json"
    sweep_path.write_text(json.dumps(sweep_doc, indent=2))
    log(f"wrote {sweep_path}")

    summary_md = args.output_root / "sweep_summary.md"
    summary_md.write_text(_render_summary_md(sweep_doc))
    log(f"wrote {summary_md}")

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


def _render_summary_md(doc: dict[str, Any]) -> str:
    lines = ["# llama.cpp Metal sweep summary", ""]
    lines.append(f"- elapsed: {doc['elapsed_seconds']:.0f}s")
    lines.append(f"- downloaded: {doc['total_bytes_downloaded'] / 1e9:.1f} GB")
    lines.append(f"- freed: {doc['total_bytes_freed'] / 1e9:.1f} GB")
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
