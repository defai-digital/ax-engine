#!/usr/bin/env python3
"""Download an MLX model from Hugging Face Hub for use with ax-engine.

Downloads model weights and automatically generates the ax-engine manifest
(model-manifest.json). Tries ax-engine-bench (Homebrew install) then cargo (dev).

Usage:
  python scripts/download_model.py mlx-community/Qwen3-4B-4bit
  python scripts/download_model.py mlx-community/Qwen3-4B-4bit --dest /path/to/dest
  python scripts/download_model.py mlx-community/Qwen3-4B-4bit --force

For raw HuggingFace checkpoints (not from mlx-community), convert first:
  pip install mlx-lm
  mlx_lm.convert --hf-path <org/model> --mlx-path <dest> -q --q-bits 4
  python scripts/download_model.py <dest>  # (or point ax-engine-bench directly)

If you already have mlx_lm and want to use its download:
  python -m mlx_lm.generate --model mlx-community/Qwen3-4B-4bit --prompt "x" --max-tokens 1
  # model lands in ~/.cache/huggingface/hub/ — use --resolve-model-artifacts hf-cache
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

IGNORE_PATTERNS = ["*.bin", "*.pt", "*.gguf", "*.msgpack", "flax_model*"]
MODEL_MANIFEST_FILE = "model-manifest.json"
READY_STATUS = "ready"
MANIFEST_MISSING_STATUS = "manifest_missing"
INVALID_STATUS = "invalid"
DOWNLOAD_FAILED_STATUS = "download_failed"


def _slug(repo_id: str) -> str:
    return repo_id.replace("/", "--")


def default_hf_cache_root() -> Path:
    """Return the Hugging Face Hub cache root used by snapshot_download."""
    if hf_hub_cache := os.environ.get("HF_HUB_CACHE"):
        return Path(hf_hub_cache).expanduser()
    if hf_home := os.environ.get("HF_HOME"):
        return Path(hf_home).expanduser() / "hub"
    cache_home = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")).expanduser()
    return cache_home / "huggingface" / "hub"


def default_hf_repo_cache_dir(repo_id: str) -> Path:
    """Return the repository cache directory that contains snapshot revisions."""
    return default_hf_cache_root() / f"models--{_slug(repo_id)}"


def _disable_hf_progress_bars() -> None:
    try:
        from huggingface_hub.utils import disable_progress_bars

        disable_progress_bars()
    except Exception:
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"


def download(repo_id: str, dest: Path | None, force: bool = False, *, quiet: bool = False) -> Path:
    if dest is not None:
        dest = Path(dest)

    if dest is not None and dest.exists() and not force:
        safetensors = list(dest.glob("*.safetensors"))
        if safetensors and (dest / MODEL_MANIFEST_FILE).exists():
            if not quiet:
                print(f"  already present with manifest: {dest}")
            return dest
        if safetensors:
            if not quiet:
                print(f"  weights present but manifest missing: {dest}")
            return dest

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise RuntimeError(
            "huggingface_hub is not installed. Run:\n"
            "  pip install huggingface_hub"
        )

    if quiet:
        _disable_hf_progress_bars()

    if dest is None:
        if not quiet:
            print(f"  downloading {repo_id} -> Hugging Face Hub cache")
        return Path(
            snapshot_download(
                repo_id=repo_id,
                ignore_patterns=IGNORE_PATTERNS,
                force_download=force,
            )
        )

    dest.mkdir(parents=True, exist_ok=True)
    if not quiet:
        print(f"  downloading {repo_id} -> {dest}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(dest),
        ignore_patterns=IGNORE_PATTERNS,
        force_download=force,
    )
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download an MLX model from Hugging Face Hub for ax-engine"
    )
    parser.add_argument("repo_id", help="HuggingFace repo id, e.g. mlx-community/Qwen3-4B-4bit")
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="Destination directory (default: Hugging Face Hub cache)",
    )
    parser.add_argument("--force", action="store_true", help="Re-download even if present")
    parser.add_argument("--json", action="store_true", help="Emit a machine-readable run summary")
    args = parser.parse_args()

    dest = args.dest
    summary_dest = dest or default_hf_repo_cache_dir(args.repo_id)
    if not args.json:
        print(f"\n[{args.repo_id}]")
    try:
        dest = download(args.repo_id, dest, force=args.force, quiet=args.json)
    except RuntimeError as error:
        if args.json:
            _print_json(_summary(args.repo_id, summary_dest, status=DOWNLOAD_FAILED_STATUS, errors=[str(error)]))
        else:
            print(f"error: {error}", file=sys.stderr)
        return 1

    errors = _validation_errors(dest)
    if errors:
        if args.json:
            _print_json(_summary(args.repo_id, dest, status=INVALID_STATUS, errors=errors))
        else:
            for error in errors:
                print(f"warning: {error}", file=sys.stderr)
        return 1
    if not args.json:
        print(f"  safetensors shards: {len(list(dest.glob('*.safetensors')))}")

    if not (dest / MODEL_MANIFEST_FILE).exists():
        if not args.json:
            print("  generating manifest...")
        if not _try_generate_manifest(dest, quiet=args.json):
            if args.json:
                _print_json(_summary(args.repo_id, dest, status=MANIFEST_MISSING_STATUS))
            else:
                _print_manifest_hint(dest)
            return 0  # not fatal; user has instructions

    if args.json:
        _print_json(_summary(args.repo_id, dest, status=READY_STATUS))
    else:
        print(f"\nReady - model artifacts at: {dest}")
        print(f"  ax-engine-server --mlx --mlx-model-artifacts-dir {dest} --port 8080")
    return 0


if __name__ == "__main__":
    sys.exit(main())
