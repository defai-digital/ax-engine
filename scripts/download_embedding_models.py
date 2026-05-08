#!/usr/bin/env python3
"""Download embedding models from mlx-community to .internal/models/."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / ".internal" / "models"

MODELS = {
    "qwen3-embedding-0.6b-8bit": "mlx-community/Qwen3-Embedding-0.6B-8bit",
    "qwen3-embedding-4b-4bit": "mlx-community/Qwen3-Embedding-4B-4bit-DWQ",
}


def download(local_name: str, hf_repo: str, force: bool = False) -> Path:
    from huggingface_hub import snapshot_download

    dest = MODELS_DIR / local_name
    if dest.exists() and not force:
        safetensors = list(dest.glob("*.safetensors"))
        if safetensors:
            print(f"  already present: {dest} ({len(safetensors)} shard(s))")
            return dest

    dest.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {hf_repo} → {dest}")
    snapshot_download(
        repo_id=hf_repo,
        local_dir=str(dest),
        ignore_patterns=["*.bin", "*.pt", "*.gguf", "*.msgpack", "flax_model*"],
    )
    return dest


def probe(model_dir: Path) -> None:
    idx_file = model_dir / "model.safetensors.index.json"
    if idx_file.exists():
        idx = json.loads(idx_file.read_text())
        keys = list(idx["weight_map"].keys())
        files = set(idx["weight_map"].values())
        print(f"    tensors: {len(keys)}, shards: {len(files)}")
        # Show a few sample keys
        sample = [k for k in keys if "layers.0" in k][:6]
        for k in sample:
            print(f"      {k}")
    else:
        safetensors = list(model_dir.glob("*.safetensors"))
        if safetensors:
            print(f"    single-file safetensors: {[f.name for f in safetensors]}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download embedding models")
    parser.add_argument("--force", action="store_true", help="Re-download even if present")
    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()) + ["all"],
        default="all",
        help="Which model(s) to download (default: all)",
    )
    args = parser.parse_args()

    targets = MODELS if args.model == "all" else {args.model: MODELS[args.model]}

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for local_name, hf_repo in targets.items():
        print(f"\n[{local_name}]")
        dest = download(local_name, hf_repo, force=args.force)
        probe(dest)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
