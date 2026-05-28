#!/usr/bin/env python3
"""Prepare an ax-engine-compatible MTP sidecar from standard Qwen3.6 HF models.

Downloads only the 2 safetensors shards that contain mtp.* tensors from the
original BF16 HuggingFace repo, extracts those tensors, and writes:
  - mtp.safetensors      (MTP head weights)
  - mtplx_runtime.json   (depth/sampler config ax-engine reads)
  - config.json          (updated from the base 4-bit model)

The output is placed alongside an existing mlx-community quantized model,
making it immediately usable by ax-engine without re-downloading the full
54/70 GB BF16 checkpoint.

Usage:
  # Qwen3.6-27B (dense, depth-3)
  python3 scripts/prepare_qwen36_mtp_sidecar.py \\
      --model 27b \\
      --base-dir ~/.cache/huggingface/hub/models--mlx-community--Qwen3.6-27B-4bit/snapshots/<hash>

  # Qwen3.6-35B-A3B (MoE, depth-1 default)
  python3 scripts/prepare_qwen36_mtp_sidecar.py \\
      --model 35b \\
      --base-dir ~/.cache/huggingface/hub/models--mlx-community--Qwen3.6-35B-A3B-4bit/snapshots/<hash>

  # Auto-detect base dir from HF cache
  python3 scripts/prepare_qwen36_mtp_sidecar.py --model 27b

Output is written to <base-dir>/../../../models--ax-local--Qwen3.6-{27B,35B}-MTP/snapshots/v1/
(a synthetic HF cache layout so ax-engine-bench --model-dir works without change).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

MODEL_CONFIGS: dict[str, dict] = {
    "27b": {
        "hf_repo": "Qwen/Qwen3.6-27B",
        "mtp_shards": [
            "model-00013-of-00015.safetensors",
            "model-00015-of-00015.safetensors",
        ],
        "mlx_community_repo": "mlx-community/Qwen3.6-27B-4bit",
        "mlx_community_slug": "models--mlx-community--Qwen3.6-27B-4bit",
        "arch_id": "qwen3-next-mtp",
        "mtp_depth_max": 3,
        "is_moe": False,
        "output_slug": "models--ax-local--Qwen3.6-27B-MTP",
    },
    "35b": {
        "hf_repo": "Qwen/Qwen3.6-35B-A3B",
        "mtp_shards": [
            "model-00025-of-00026.safetensors",
            "model-00026-of-00026.safetensors",
        ],
        "mlx_community_repo": "mlx-community/Qwen3.6-35B-A3B-4bit",
        "mlx_community_slug": "models--mlx-community--Qwen3.6-35B-A3B-4bit",
        "arch_id": "qwen3-next-mtp",
        "mtp_depth_max": 1,  # MoE MTP: start conservative at depth-1
        "is_moe": True,
        "output_slug": "models--ax-local--Qwen3.6-35B-MTP",
    },
}

HF_CACHE = Path(
    os.environ.get("HF_HUB_CACHE")
    or (Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub")
)

DRAFT_SAMPLER = {"temperature": 0.7, "top_k": 20, "top_p": 0.95}
TARGET_SAMPLER = {"temperature": 0.6, "top_k": 20, "top_p": 0.95}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _file_record(path: Path) -> dict:
    if not path.exists():
        return {"path": str(path), "exists": False}
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _find_snapshot(slug: str) -> Path | None:
    model_dir = HF_CACHE / slug
    if not model_dir.exists():
        return None
    snapshots = sorted((model_dir / "snapshots").iterdir()) if (model_dir / "snapshots").exists() else []
    return snapshots[-1] if snapshots else None


def _download_shard(hf_repo: str, filename: str) -> Path:
    from huggingface_hub import hf_hub_download
    print(f"  Downloading {hf_repo}/{filename} ...", flush=True)
    path = hf_hub_download(hf_repo, filename)
    size_gb = Path(path).stat().st_size / 1024**3
    print(f"  -> {path} ({size_gb:.2f} GB)", flush=True)
    return Path(path)


def _load_mtp_tensors(shard_paths: list[Path]) -> dict:
    import mlx.core as mx

    tensors = {}
    for shard in shard_paths:
        loaded = mx.load(str(shard))
        for k, v in loaded.items():
            if k.startswith("mtp."):
                tensors[k] = v
    return tensors


def _scale_norm_weights(tensors: dict) -> dict:
    """Scale RMSNorm weights by 2 — matches lightning-mlx convert-mtplx convention."""
    norm_suffixes = (
        "mtp.norm.weight",
        "mtp.pre_fc_norm_embedding.weight",
        "mtp.pre_fc_norm_hidden.weight",
    )
    layer_norm_patterns = ("input_layernorm.weight", "post_attention_layernorm.weight")
    import mlx.core as mx

    out = {}
    for k, v in tensors.items():
        if k in norm_suffixes or any(p in k for p in layer_norm_patterns):
            out[k] = (v * 2).astype(v.dtype)
        else:
            out[k] = v
    return out


def _unpack_moe_experts(tensors: dict) -> dict:
    """Unpack packed gate_up_proj [E, 2*D, in] -> separate gate_proj and up_proj [E, D, in]."""
    import mlx.core as mx

    out = {}
    for k, v in tensors.items():
        if k.endswith("mlp.experts.gate_up_proj"):
            # shape [E, 2*D, in] -> gate [E, D, in], up [E, D, in]
            prefix = k[: -len("mlp.experts.gate_up_proj")]
            half = v.shape[1] // 2
            out[prefix + "mlp.gate_proj.weight"] = v[:, :half, :]
            out[prefix + "mlp.up_proj.weight"] = v[:, half:, :]
        elif k.endswith("mlp.experts.down_proj"):
            # rename to mlp.down_proj.weight (keep as-is, reshape if needed)
            prefix = k[: -len("mlp.experts.down_proj")]
            out[prefix + "mlp.down_proj.weight"] = v
        else:
            out[k] = v
    return out


def prepare_sidecar(model_key: str, base_dir: Path | None) -> Path:
    cfg = MODEL_CONFIGS[model_key]

    # Resolve base dir (the 4-bit model snapshot)
    if base_dir is None:
        snap = _find_snapshot(cfg["mlx_community_slug"])
        if snap is None:
            sys.exit(
                f"ERROR: {cfg['mlx_community_slug']} not found in HF cache at {HF_CACHE}.\n"
                f"Download it first: python3 scripts/download_model.py {cfg['mlx_community_repo']}"
            )
        base_dir = snap
    base_dir = Path(base_dir).expanduser().resolve()
    print(f"Base 4-bit model: {base_dir}", flush=True)

    # Output dir: a synthetic HF cache entry so ax-engine-bench can use --model-dir directly.
    out_dir = HF_CACHE / cfg["output_slug"] / "snapshots" / "v1"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir:       {out_dir}", flush=True)

    # Copy base model files (weights + configs) if not already there.
    print("\nCopying base model files...", flush=True)
    for item in base_dir.iterdir():
        dest = out_dir / item.name
        if dest.exists():
            continue
        if item.is_dir():
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)
        print(f"  copied {item.name}", flush=True)

    # Download MTP shards from HF if not cached.
    print("\nLocating MTP shards...", flush=True)
    shard_paths: list[Path] = []
    for shard_name in cfg["mtp_shards"]:
        # hf_hub_download will return cached path if already downloaded.
        path = _download_shard(cfg["hf_repo"], shard_name)
        shard_paths.append(path)

    # Extract MTP tensors.
    print("\nExtracting mtp.* tensors...", flush=True)
    tensors = _load_mtp_tensors(shard_paths)
    print(f"  Found {len(tensors)} MTP tensors: {sorted(tensors)[:6]}...", flush=True)

    # Apply transformations.
    tensors = _scale_norm_weights(tensors)
    if cfg["is_moe"]:
        tensors = _unpack_moe_experts(tensors)
        print(f"  After MoE unpack: {len(tensors)} tensors", flush=True)

    # Save mtp.safetensors.
    import mlx.core as mx

    mtp_path = out_dir / "mtp.safetensors"
    mx.save_safetensors(str(mtp_path), {k: mx.array(v) for k, v in tensors.items()})
    size_mb = mtp_path.stat().st_size / 1024**2
    print(f"\nSaved mtp.safetensors ({size_mb:.1f} MB) -> {mtp_path}", flush=True)

    # Write mtplx_runtime.json.
    runtime = {
        "arch_id": cfg["arch_id"],
        "mtp_depth_max": cfg["mtp_depth_max"],
        "recommended_profile": "sustained",
        "recommended_draft_sampler": DRAFT_SAMPLER,
        "sampler": TARGET_SAMPLER,
        "base_trunk": cfg["mlx_community_repo"],
        "mtp_tensor_count": len(tensors),
    }
    runtime_path = out_dir / "mtplx_runtime.json"
    runtime_path.write_text(json.dumps(runtime, indent=2))
    print(f"Saved mtplx_runtime.json -> {runtime_path}", flush=True)

    # Patch config.json with MTP fields so mlx-lm (if used) recognises it.
    config_path = out_dir / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())
        config["mlx_lm_extra_tensors"] = {"mtp_file": "mtp.safetensors", "mtp_tensor_count": len(tensors)}
        config["num_nextn_predict_layers"] = 1
        config_path.write_text(json.dumps(config, indent=2))
        print(f"Updated config.json with MTP fields.", flush=True)

    # Write provenance manifest. This is intentionally separate from
    # mtplx_runtime.json: runtime config controls loading, provenance supports
    # fair benchmark review.
    manifest = {
        "schema_version": "ax.mtp_sidecar_provenance.v1",
        "generated_by": "scripts/prepare_qwen36_mtp_sidecar.py",
        "model_key": model_key,
        "base": {
            "model_id": cfg["mlx_community_repo"],
            "snapshot_dir": str(base_dir),
            "snapshot": base_dir.name,
            "config": _file_record(base_dir / "config.json"),
            "index": _file_record(base_dir / "model.safetensors.index.json"),
        },
        "source": {
            "model_id": cfg["hf_repo"],
            "mtp_shards": [
                {
                    "name": shard.name,
                    **_file_record(shard),
                }
                for shard in shard_paths
            ],
        },
        "output": {
            "dir": str(out_dir),
            "mtp": _file_record(mtp_path),
            "runtime": _file_record(runtime_path),
            "config": _file_record(config_path),
        },
        "transform": {
            "norm_policy": "scale_selected_mtp_norm_weights_by_2",
            "moe_expert_unpack": cfg["is_moe"],
            "notes": (
                "Validate this policy against AX loader expectations and the "
                "reference MLX conversion path before publication."
            ),
        },
        "runtime": runtime,
    }
    manifest_path = out_dir / "ax_mtp_sidecar_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Saved provenance manifest -> {manifest_path}", flush=True)

    print(f"\nSidecar ready at:\n  {out_dir}", flush=True)
    return out_dir


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=["27b", "35b"], required=True,
                        help="Which Qwen3.6 model to prepare (27b or 35b).")
    parser.add_argument("--base-dir", type=Path, default=None,
                        help="Path to the local mlx-community 4-bit model snapshot. "
                             "Auto-detected from HF cache if omitted.")
    args = parser.parse_args()
    prepare_sidecar(args.model, args.base_dir)


if __name__ == "__main__":
    main()
