#!/usr/bin/env python3
"""Sweep llama.cpp Metal benchmarks across the 14 README MLX-inference rows.

Reads benchmarks/manifests/llama_cpp_metal/inventory.json and, for each row:
  1) Resolves the first GGUF candidate that exists on Hugging Face.
  2) Downloads the GGUF to --cache-dir (default ~/.cache/huggingface/hub).
  3) Invokes scripts/bench_mlx_inference_stack.py with --llama-cpp-bench /
     --llama-cpp-gguf, --skip-mlx-lm, --skip-ax-engine to produce ONLY the
     external GGUF baseline row.
  4) Optionally deletes the GGUF after the row finishes (--no-keep-gguf) to
     keep peak disk low.

Writes one result JSON per row plus a combined sweep_results.json and a
sweep_summary.md. Unresolved rows are recorded as explicit n/a entries.

This script does NOT modify the README. README integration is the job of a
separate updater that consumes sweep_results.json.

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


def log(msg: str) -> None:
    print(f"[sweep] {msg}", flush=True)


def resolve_gguf_candidate(
    candidates: list[dict[str, str]],
    *,
    hf_token: str | None,
) -> tuple[str, str, list[dict[str, Any]]] | None:
    """Walk candidates in priority order. Return (repo, filename, probe_log)
    for the first candidate that resolves; None if all fail."""
    from huggingface_hub import HfApi
    from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

    api = HfApi(token=hf_token)
    probe_log: list[dict[str, Any]] = []
    for candidate in candidates:
        repo = candidate["repo"]
        pattern = candidate["filename_pattern"]
        entry: dict[str, Any] = {"repo": repo, "filename_pattern": pattern}
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


def download_gguf(
    repo: str,
    filename: str,
    *,
    cache_dir: Path,
    hf_token: str | None,
) -> Path:
    from huggingface_hub import hf_hub_download

    shards = _shard_siblings(filename)
    first_path: Path | None = None
    for shard in shards:
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
    model_dir_for_prompts: Path,
) -> dict[str, Any]:
    """Invoke bench_mlx_inference_stack.py with only the llama.cpp row enabled.

    We point --model-dir at the local MLX model so the harness can still
    generate the shape-matching prompt artifact (random tokens at the right
    vocab size), but pass --skip-mlx-lm and --skip-ax-engine so no MLX run
    happens. The llama.cpp row is the only entry in results[].
    """
    slug = row["slug"]
    out_json = output_dir / f"{slug}.json"
    log_path = output_dir / "logs" / f"{slug}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(bench_script),
        "--model-dir",
        str(model_dir_for_prompts),
        "--prompt-tokens",
        prompt_tokens,
        "--generation-tokens",
        str(generation_tokens),
        "--repetitions",
        str(repetitions),
        "--cooldown",
        str(cooldown),
        "--skip-mlx-lm",
        "--skip-ax-engine",
        "--no-build-ax-engine",
        "--llama-cpp-bench",
        str(llama_bench),
        "--llama-cpp-gguf",
        str(gguf_path),
        "--llama-cpp-n-gpu-layers",
        str(n_gpu_layers),
        "--output",
        str(out_json),
    ]
    if extra_args:
        cmd.extend(["--llama-cpp-extra-args", extra_args])

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
        "--rows-filter",
        nargs="*",
        help="If set, only process rows whose slug is in this list.",
    )
    parser.add_argument("--prompt-tokens", default="128,512")
    parser.add_argument("--generation-tokens", type=int, default=128)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--cooldown", type=float, default=15.0)
    parser.add_argument("--n-gpu-layers", type=int, default=99)
    parser.add_argument(
        "--extra-args",
        help="Forwarded to bench_mlx_inference_stack.py --llama-cpp-extra-args.",
    )
    parser.add_argument(
        "--keep-gguf",
        action="store_true",
        help="Keep downloaded GGUFs after each row (default: delete to save disk).",
    )
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

    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "logs").mkdir(parents=True, exist_ok=True)

    with args.manifest.open() as fh:
        manifest = json.load(fh)

    rows = manifest["rows"]
    if args.rows_filter:
        rows = [r for r in rows if r["slug"] in set(args.rows_filter)]
        if not rows:
            log("ERROR: --rows-filter matched zero rows")
            sys.exit(2)

    if not args.dry_run and not args.llama_bench.exists():
        log(f"ERROR: llama-bench binary not found: {args.llama_bench}")
        sys.exit(2)

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
                hf_token=args.hf_token,
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

        try:
            gguf_path = download_gguf(
                repo,
                filename,
                cache_dir=args.cache_dir,
                hf_token=args.hf_token,
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
        log(f"  -> downloaded ({size_bytes / 1e9:.2f} GB)")

        model_dir = REPO_ROOT / row["mlx_local_dir"]
        if not model_dir.exists():
            record["status"] = "mlx_model_dir_missing"
            record["note"] = f"Local MLX dir {model_dir} not found; cannot generate prompt artifact."
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
            model_dir_for_prompts=model_dir,
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
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(started)),
        "elapsed_seconds": round(elapsed, 1),
        "total_bytes_downloaded": total_bytes_downloaded,
        "total_bytes_freed": total_bytes_freed,
        "keep_gguf": args.keep_gguf,
        "rows": summary_rows,
    }
    sweep_path = args.output_root / "sweep_results.json"
    sweep_path.write_text(json.dumps(sweep_doc, indent=2))
    log(f"wrote {sweep_path}")

    summary_md = args.output_root / "sweep_summary.md"
    summary_md.write_text(_render_summary_md(sweep_doc))
    log(f"wrote {summary_md}")


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
