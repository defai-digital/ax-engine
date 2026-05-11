#!/usr/bin/env python3
"""Download an MLX model through mlx-lm for use with ax-engine.

Downloads model weights and automatically generates the ax-engine manifest
(model-manifest.json). Tries ax-engine-bench (Homebrew install) then cargo (dev).

Usage:
  python scripts/download_model.py mlx-community/Qwen3-4B-4bit
  python scripts/download_model.py mlx-community/Qwen3-4B-4bit --dest /path/to/dest
  python scripts/download_model.py mlx-community/Qwen3-4B-4bit --force

For raw HuggingFace checkpoints (not from mlx-community), convert first:
  pip install mlx-lm
  mlx_lm.convert --hf-path <org/model> --mlx-path <dest> -q --q-bits 4
  ax-engine-bench generate-manifest <dest>
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

MODEL_MANIFEST_FILE = "model-manifest.json"
READY_STATUS = "ready"
MANIFEST_MISSING_STATUS = "manifest_missing"
INVALID_STATUS = "invalid"
DOWNLOAD_FAILED_STATUS = "download_failed"


def _slug(repo_id: str) -> str:
    return repo_id.replace("/", "--")


def _reject_non_llm_repo(repo_id: str) -> None:
    if "embedding" in repo_id.lower() or "embed" in repo_id.lower():
        raise RuntimeError(
            "embedding model downloads are not managed by ax-engine. "
            "Download embedding model artifacts manually and pass the local model directory."
        )


def default_mlx_lm_cache_root() -> Path:
    """Return the Hub cache root where mlx-lm stores downloaded model snapshots."""
    if hf_hub_cache := os.environ.get("HF_HUB_CACHE"):
        return Path(hf_hub_cache).expanduser()
    if hf_home := os.environ.get("HF_HOME"):
        return Path(hf_home).expanduser() / "hub"
    cache_home = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")).expanduser()
    return cache_home / "huggingface" / "hub"


def default_mlx_lm_repo_cache_dir(repo_id: str) -> Path:
    """Return the repository cache directory that contains snapshot revisions."""
    return default_mlx_lm_cache_root() / f"models--{_slug(repo_id)}"


def _latest_mlx_lm_snapshot(repo_id: str) -> Path | None:
    repo_cache = default_mlx_lm_repo_cache_dir(repo_id)
    refs_main = repo_cache / "refs" / "main"
    if refs_main.is_file():
        revision = refs_main.read_text().strip()
        if revision:
            snapshot = repo_cache / "snapshots" / revision
            if snapshot.is_dir():
                return snapshot
    snapshots = repo_cache / "snapshots"
    if not snapshots.is_dir():
        return None
    candidates = [path for path in snapshots.iterdir() if path.is_dir()]
    return max(candidates, key=lambda path: path.stat().st_mtime, default=None)


def _format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "estimating"
    seconds = int(seconds)
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def _snapshot_weight_progress(snapshot: Path) -> tuple[int, int] | None:
    index_path = snapshot / "model.safetensors.index.json"
    total = 0
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text())
            total = int(index.get("metadata", {}).get("total_size") or 0)
        except (OSError, ValueError, TypeError):
            total = 0
    downloaded = 0
    for path in snapshot.glob("*.safetensors"):
        try:
            downloaded += path.stat().st_size
        except OSError:
            pass
    if total <= 0 and downloaded > 0:
        total = downloaded
    if total <= 0:
        return None
    return min(downloaded, total), total


def _download_progress_message(repo_id: str, started_at: float) -> tuple[int, str]:
    elapsed = time.monotonic() - started_at
    snapshot = _latest_mlx_lm_snapshot(repo_id)
    if snapshot is not None and (progress := _snapshot_weight_progress(snapshot)) is not None:
        downloaded, total = progress
        ratio = 0.0 if total == 0 else downloaded / total
        eta = elapsed * (1.0 - ratio) / ratio if ratio > 0 else None
        gib = 1024 ** 3
        return (
            5 + int(min(ratio, 1.0) * 80),
            "Downloading weights "
            f"({downloaded / gib:.1f}/{total / gib:.1f} GiB, "
            f"elapsed {_format_duration(elapsed)}, ETA {_format_duration(eta)})",
        )
    synthetic = min(25, 5 + int(elapsed // 20))
    return (
        synthetic,
        f"Downloading with mlx-lm (elapsed {_format_duration(elapsed)}, ETA estimating)",
    )


def _emit_progress(done: int, total: int, file: str) -> None:
    print(json.dumps({"event": "progress", "done": done, "total": total, "file": file}), flush=True)


def _run_mlx_lm_download(
    repo_id: str,
    *,
    quiet: bool = False,
    progress_json: bool = False,
) -> None:
    env = os.environ.copy()
    if quiet:
        env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    command = [
        sys.executable,
        "-m",
        "mlx_lm",
        "generate",
        "--model",
        repo_id,
        "--prompt",
        "x",
        "--max-tokens",
        "1",
    ]
    if not progress_json:
        result = subprocess.run(command, capture_output=True, text=True, env=env)
    else:
        started_at = time.monotonic()
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        output_parts: list[str] = []

        def drain(stream: object) -> None:
            if stream is None:
                return
            for line in stream:
                output_parts.append(line)

        stdout_thread = threading.Thread(target=drain, args=(proc.stdout,), daemon=True)
        stderr_thread = threading.Thread(target=drain, args=(proc.stderr,), daemon=True)
        stdout_thread.start()
        stderr_thread.start()
        while proc.poll() is None:
            done, message = _download_progress_message(repo_id, started_at)
            _emit_progress(done, 100, message)
            time.sleep(2)
        stdout_thread.join(timeout=1)
        stderr_thread.join(timeout=1)
        result = subprocess.CompletedProcess(command, proc.returncode, "".join(output_parts), "")
    if result.returncode != 0:
        output = "\n".join(part for part in [result.stderr.strip(), result.stdout.strip()] if part)
        raise RuntimeError(
            "mlx-lm download failed. Install mlx-lm with:\n"
            "  pip install mlx-lm\n"
            f"Command: {' '.join(command)}\n"
            f"{output}".rstrip()
        )


def _copy_snapshot_to_dest(snapshot: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for path in snapshot.iterdir():
        target = dest / path.name
        if path.is_dir():
            shutil.copytree(path, target, dirs_exist_ok=True)
        elif path.is_symlink():
            shutil.copy2(path.resolve(), target)
        else:
            shutil.copy2(path, target)


def download(
    repo_id: str,
    dest: Path | None,
    force: bool = False,
    *,
    quiet: bool = False,
    progress_json: bool = False,
) -> Path:
    _reject_non_llm_repo(repo_id)

    if dest is not None:
        dest = Path(dest)

    if dest is not None and dest.exists() and not force:
        safetensors = list(dest.glob("*.safetensors"))
        if safetensors and (dest / MODEL_MANIFEST_FILE).exists():
            if not quiet:
                print(f"  already present with manifest: {dest}")
            if progress_json:
                _emit_progress(100, 100, "Ready")
            return dest
        if safetensors:
            if not quiet:
                print(f"  weights present but manifest missing: {dest}")
            if progress_json:
                _emit_progress(85, 100, "Weights already present")
            return dest

    if force:
        shutil.rmtree(default_mlx_lm_repo_cache_dir(repo_id), ignore_errors=True)

    snapshot = None if force else _latest_mlx_lm_snapshot(repo_id)
    if snapshot is not None and not _validation_errors(snapshot):
        if progress_json:
            _emit_progress(85, 100, "Using existing mlx-lm cache snapshot")
        if dest is None:
            if not quiet:
                print(f"  already present in mlx-lm cache: {snapshot}")
            return snapshot
        _copy_snapshot_to_dest(snapshot, dest)
        return dest

    if not quiet:
        destination = "mlx-lm cache" if dest is None else f"{dest} via mlx-lm cache"
        print(f"  downloading {repo_id} -> {destination}")
    _run_mlx_lm_download(repo_id, quiet=quiet, progress_json=progress_json)
    snapshot = _latest_mlx_lm_snapshot(repo_id)
    if snapshot is None:
        raise RuntimeError(f"mlx-lm completed but no cache snapshot was found for {repo_id}")

    if dest is None:
        return snapshot

    _copy_snapshot_to_dest(snapshot, dest)
    return dest


def _try_generate_manifest(dest: Path, *, quiet: bool = False) -> bool:
    """Try ax-engine-bench (installed) then cargo (dev). Returns True on success."""
    if shutil.which("ax-engine-bench"):
        command = ["ax-engine-bench", "generate-manifest", str(dest)]
        if quiet:
            command.append("--json")
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            out = result.stdout.strip()
            if out and not quiet:
                print(f"  {out}")
            return True
        if not quiet:
            print(
                f"  ax-engine-bench generate-manifest failed: {result.stderr.strip()}",
                file=sys.stderr,
            )
        return False

    if shutil.which("cargo"):
        result = subprocess.run(
            [
                "cargo", "run", "-q",
                "-p", "ax-engine-core",
                "--bin", "generate-manifest",
                "--", str(dest),
            ],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            out = result.stdout.strip()
            if out and not quiet:
                print(f"  {out}")
            return True

    return False


def _print_manifest_hint(dest: Path) -> None:
    print(
        "\nManifest generation not available automatically. Run manually:\n"
        f"  ax-engine-bench generate-manifest {dest}\n"
        "or (from source):\n"
        f"  cargo run -p ax-engine-core --bin generate-manifest -- {dest}\n"
        "\nThen start the server:\n"
        f"  ax-engine-server --mlx --mlx-model-artifacts-dir {dest} --port 8080"
    )


def _validate(dest: Path) -> bool:
    return not _validation_errors(dest)


def _validation_errors(dest: Path) -> list[str]:
    errors = []
    safetensors = list(dest.glob("*.safetensors"))
    if not safetensors:
        errors.append(f"no .safetensors files found in {dest}")
    if not (dest / "config.json").exists():
        errors.append(f"config.json missing in {dest}")
    return errors


def _server_command(dest: Path) -> list[str]:
    return [
        "ax-engine-server",
        "--mlx",
        "--mlx-model-artifacts-dir",
        str(dest),
        "--port",
        "8080",
    ]


def _summary(repo_id: str, dest: Path, *, status: str, errors: list[str] | None = None) -> dict:
    manifest_path = dest / MODEL_MANIFEST_FILE
    return {
        "schema_version": "ax.download_model.v1",
        "repo_id": repo_id,
        "dest": str(dest),
        "manifest_path": str(manifest_path),
        "manifest_present": manifest_path.exists(),
        "safetensors_count": len(list(dest.glob("*.safetensors"))) if dest.exists() else 0,
        "config_present": (dest / "config.json").exists(),
        "status": status,
        "errors": errors or [],
        "server_command": _server_command(dest),
    }


def _print_json(summary: dict) -> None:
    print(json.dumps(summary, indent=2, sort_keys=True))


def _print_json_line(summary: dict) -> None:
    print(json.dumps(summary, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download an MLX model through mlx-lm for ax-engine"
    )
    parser.add_argument("repo_id", help="MLX LLM repo id, e.g. mlx-community/Qwen3-4B-4bit")
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="Destination directory (default: mlx-lm cache snapshot)",
    )
    parser.add_argument("--force", action="store_true", help="Re-download even if present")
    parser.add_argument("--json", action="store_true", help="Emit a machine-readable run summary")
    parser.add_argument(
        "--progress-json",
        action="store_true",
        help="Emit newline-delimited progress JSON before the final summary",
    )
    args = parser.parse_args()

    dest = args.dest
    summary_dest = dest or default_mlx_lm_repo_cache_dir(args.repo_id)
    if not args.json:
        print(f"\n[{args.repo_id}]")
    try:
        if args.progress_json:
            _emit_progress(0, 100, "Starting mlx-lm download")
        dest = download(
            args.repo_id,
            dest,
            force=args.force,
            quiet=args.json,
            progress_json=args.progress_json,
        )
    except RuntimeError as error:
        if args.json:
            summary = _summary(
                args.repo_id,
                summary_dest,
                status=DOWNLOAD_FAILED_STATUS,
                errors=[str(error)],
            )
            if args.progress_json:
                _print_json_line(summary)
            else:
                _print_json(summary)
        else:
            print(f"error: {error}", file=sys.stderr)
        return 1

    errors = _validation_errors(dest)
    if errors:
        if args.json:
            summary = _summary(args.repo_id, dest, status=INVALID_STATUS, errors=errors)
            if args.progress_json:
                _print_json_line(summary)
            else:
                _print_json(summary)
        else:
            for error in errors:
                print(f"warning: {error}", file=sys.stderr)
        return 1
    if not args.json:
        print(f"  safetensors shards: {len(list(dest.glob('*.safetensors')))}")

    if not (dest / MODEL_MANIFEST_FILE).exists():
        if not args.json:
            print("  generating manifest...")
        if args.progress_json:
            _emit_progress(90, 100, "Generating manifest")
        if not _try_generate_manifest(dest, quiet=args.json):
            if args.json:
                summary = _summary(args.repo_id, dest, status=MANIFEST_MISSING_STATUS)
                if args.progress_json:
                    _print_json_line(summary)
                else:
                    _print_json(summary)
            else:
                _print_manifest_hint(dest)
            return 0  # not fatal; user has instructions

    if args.json:
        summary = _summary(args.repo_id, dest, status=READY_STATUS)
        if args.progress_json:
            _emit_progress(100, 100, "Ready")
            _print_json_line(summary)
        else:
            _print_json(summary)
    else:
        print(f"\nReady - model artifacts at: {dest}")
        print(f"  ax-engine-server --mlx --mlx-model-artifacts-dir {dest} --port 8080")
    return 0


if __name__ == "__main__":
    sys.exit(main())
