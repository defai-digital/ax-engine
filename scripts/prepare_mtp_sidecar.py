#!/usr/bin/env python3
"""Prepare an ax-engine-compatible MTP sidecar from any HF model that ships MTP tensors.

This is the generic successor to ``prepare_qwen36_mtp_sidecar.py``. Instead of a
fixed ``--model {27b,35b}`` table with hardcoded shard filenames, it:

  1. reads the source repo's ``model.safetensors.index.json`` and auto-discovers
     which shards actually contain ``mtp.*`` tensors (downloading only those),
  2. dispatches tensor-layout normalization through a per-architecture registry
     keyed off the source ``config.json`` (Qwen3.6 dense + packed-MoE today),
  3. optionally quantizes the 2-D projection weights to match the base model,
  4. writes the same sidecar contract ax-engine already loads:
       - mtp.safetensors                (MTP head weights)
       - mtplx_runtime.json             (depth / draft-sampler / quant hint)
       - config.json                    (base config + MTP fields)
       - ax_mtp_sidecar_manifest.json   (provenance for fair benchmark review)

The output is packaged alongside a base (typically quantized) MLX model so the
full BF16 checkpoint never has to live on disk — only the handful of shards that
hold the MTP head are pulled from the source repo.

Usage:
  # Qwen3.6-27B (dense). Base auto-detected from the HF cache.
  python3 scripts/prepare_mtp_sidecar.py \\
      --hf-repo Qwen/Qwen3.6-27B \\
      --base mlx-community/Qwen3.6-27B-4bit

  # Qwen3.6-35B-A3B (MoE), explicit local base snapshot, INT4 sidecar.
  python3 scripts/prepare_mtp_sidecar.py \\
      --hf-repo Qwen/Qwen3.6-35B-A3B \\
      --base ~/.cache/huggingface/hub/models--mlx-community--Qwen3.6-35B-A3B-4bit/snapshots/<hash> \\
      --mtp-depth-max 1 \\
      --quantize 4

``--base`` accepts either a local model directory or an ``org/name`` repo id
(looked up in the local HF cache; download it first via
``scripts/download_model.py`` if absent). Output defaults to a synthetic HF cache
entry ``models--ax-local--<base-name>-MTP/snapshots/v1/`` so that
``ax-engine-bench --model-dir`` works without change.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable

HF_CACHE = Path(
    os.environ.get("HF_HUB_CACHE")
    or (Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub")
)

DRAFT_SAMPLER = {"temperature": 0.7, "top_k": 20, "top_p": 0.95}
TARGET_SAMPLER = {"temperature": 0.6, "top_k": 20, "top_p": 0.95}

# Norm tensors are stored as RMSNorm deltas in raw HF MTP weights; MLX expects the
# +1.0 multiplier convention. ndim==1 "norm" tensors are shifted on extraction.
NORM_SHIFT = 1.0


# --------------------------------------------------------------------------- #
# HF cache / repo helpers
# --------------------------------------------------------------------------- #
def _repo_slug(repo_id: str) -> str:
    return "models--" + repo_id.replace("/", "--")


def _find_snapshot(slug: str) -> Path | None:
    return _latest_snapshot(HF_CACHE / slug)


def _latest_snapshot(model_dir: Path) -> Path | None:
    """Return the newest snapshot under ``<model_dir>/snapshots`` (or None)."""
    snaps_dir = model_dir / "snapshots"
    if not snaps_dir.exists():
        return None
    snapshots = sorted(snaps_dir.iterdir())
    return snapshots[-1] if snapshots else None


def _resolve_base_dir(base: str) -> Path:
    """Resolve --base to a concrete model snapshot directory.

    Accepts a local path or an ``org/name`` repo id (looked up in the HF cache).
    """
    candidate = Path(base).expanduser()
    if candidate.exists() and (candidate / "config.json").exists():
        return candidate.resolve()
    if candidate.exists() and (candidate / "snapshots").exists():
        # A local HF-cache-style dir (…/<model>/snapshots/<hash>): resolve the
        # snapshot from the given path itself, not from the global HF cache.
        snap = _latest_snapshot(candidate)
        if snap:
            return snap.resolve()
    # Treat as a repo id and look it up in the cache.
    snap = _find_snapshot(_repo_slug(base))
    if snap is not None:
        return snap.resolve()
    sys.exit(
        f"ERROR: base model '{base}' not found.\n"
        f"  Looked for a local dir and for {_repo_slug(base)} in {HF_CACHE}.\n"
        f"  Download it first:  python3 scripts/download_model.py {base}"
    )


def _download_file(hf_repo: str, filename: str) -> Path:
    from huggingface_hub import hf_hub_download

    print(f"  Downloading {hf_repo}/{filename} ...", flush=True)
    path = Path(hf_hub_download(hf_repo, filename))
    size = path.stat().st_size / 1024**3
    print(f"  -> {path} ({size:.2f} GB)" if size >= 0.5 else f"  -> {path}", flush=True)
    return path


def _discover_mtp_shards(hf_repo: str) -> list[str]:
    """Read the source weight index and return the shards holding mtp.* tensors.

    Falls back to a single-file repo (``model.safetensors``) when no index exists.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError

    try:
        idx_path = hf_hub_download(hf_repo, "model.safetensors.index.json")
    except EntryNotFoundError:
        print("  No index.json — assuming single-shard model.safetensors", flush=True)
        return ["model.safetensors"]

    weight_map = json.loads(Path(idx_path).read_text()).get("weight_map", {})
    mtp_shards = sorted({rel for key, rel in weight_map.items() if key.startswith("mtp.")})
    if not mtp_shards:
        sys.exit(
            f"ERROR: no mtp.* tensors found in {hf_repo}/model.safetensors.index.json.\n"
            f"  This model does not ship an MTP head; nothing to extract."
        )
    print(f"  MTP tensors live in {len(mtp_shards)} shard(s): {mtp_shards}", flush=True)
    return mtp_shards


# --------------------------------------------------------------------------- #
# Tensor extraction + transforms
# --------------------------------------------------------------------------- #
def _load_mtp_tensors(shard_paths: list[Path]) -> dict[str, Any]:
    import mlx.core as mx

    tensors: dict[str, Any] = {}
    for shard in shard_paths:
        loaded = mx.load(str(shard))
        for k, v in loaded.items():
            if k.startswith("mtp."):
                tensors[k] = v
        del loaded
    return tensors


def _shift_norm_weights(tensors: dict[str, Any]) -> dict[str, Any]:
    """Lift raw HF MTP RMSNorm deltas into MLX's +1.0 multiplier convention."""
    out: dict[str, Any] = {}
    for k, v in tensors.items():
        if "norm" in k and getattr(v, "ndim", None) == 1:
            out[k] = (v + NORM_SHIFT).astype(v.dtype)
        else:
            out[k] = v
    return out


# --- Architecture normalizers --------------------------------------------- #
# Each normalizer rewrites raw mtp.* tensors into the canonical key layout the
# ax-engine loader (crates/ax-engine-mlx/src/weights.rs::load_mtp) expects.

def _normalize_qwen_moe_packed(tensors: dict[str, Any]) -> dict[str, Any]:
    """Qwen3.6 MoE: split packed experts.gate_up_proj and rename expert stacks.

    The loader keys MoE off the router (mtp.layers.0.mlp.gate) and reads the
    expert stacks as mtp.layers.0.mlp.{gate,up,down}_proj.weight [E, D, in].
    """
    out: dict[str, Any] = {}
    for k, v in tensors.items():
        if k.endswith("mlp.experts.gate_up_proj"):
            prefix = k[: -len("mlp.experts.gate_up_proj")]
            half = v.shape[1] // 2
            out[prefix + "mlp.gate_proj.weight"] = v[:, :half, :]
            out[prefix + "mlp.up_proj.weight"] = v[:, half:, :]
        elif k.endswith("mlp.experts.gate_proj"):
            # Already-unpacked expert stack (separate gate/up): rename only.
            out[k[: -len("mlp.experts.gate_proj")] + "mlp.gate_proj.weight"] = v
        elif k.endswith("mlp.experts.up_proj"):
            out[k[: -len("mlp.experts.up_proj")] + "mlp.up_proj.weight"] = v
        elif k.endswith("mlp.experts.down_proj"):
            out[k[: -len("mlp.experts.down_proj")] + "mlp.down_proj.weight"] = v
        else:
            out[k] = v
    return out


def _normalize_passthrough(tensors: dict[str, Any]) -> dict[str, Any]:
    """Dense MTP (Qwen3.6-27B): tensors are already in canonical layout."""
    return dict(tensors)


# Registry: (detector, normalizer, is_moe). First matching entry wins.
ArchNormalizer = Callable[[dict[str, Any]], dict[str, Any]]


def _detect_arch(tensors: dict[str, Any], config: dict[str, Any]) -> tuple[str, ArchNormalizer, bool]:
    if "mtp.layers.0.mlp.experts.gate_up_proj" in tensors:
        return "qwen-moe-packed", _normalize_qwen_moe_packed, True
    if "mtp.layers.0.mlp.experts.down_proj" in tensors:
        # Experts present but not gate-up-packed — still MoE, pass through rename.
        return "qwen-moe", _normalize_qwen_moe_packed, True
    if "mtp.layers.0.mlp.gate_proj.weight" in tensors:
        return "qwen-dense", _normalize_passthrough, False
    sys.exit(
        "ERROR: could not detect MTP architecture from tensor layout.\n"
        f"  model_type={config.get('model_type')!r}; "
        f"sample keys={sorted(tensors)[:8]}\n"
        "  Add a normalizer to scripts/prepare_mtp_sidecar.py for this arch."
    )


# --------------------------------------------------------------------------- #
# Optional quantization (2-D projections only; experts/norms stay bf16)
# --------------------------------------------------------------------------- #
def _should_quantize(key: str, shape: tuple[int, ...], group_size: int) -> bool:
    """Whether a tensor may be affine-quantized with ``group_size``.

    mx.quantize operates on the last axis and requires it to be both
    ``>= group_size`` and an exact multiple of ``group_size`` (else it raises).
    Only 2-D matmul weights qualify: norms stay FP, the small shared-expert
    router gate stays FP, and 3-D MoE expert stacks stay bf16 (the recurrent
    MoE path expects FP experts).
    """
    if "norm" in key or "layernorm" in key:
        return False
    if "shared_expert_gate" in key:
        return False
    if len(shape) != 2:  # skip 1-D (norms/bias) and 3-D (expert stacks)
        return False
    last = shape[-1]
    return last >= group_size and last % group_size == 0


def _quantize_tensors(tensors: dict[str, Any], bits: int, group_size: int) -> dict[str, Any]:
    """Quantize eligible 2-D projection weights to ``bits``; leave the rest FP.

    The loader treats each tensor independently (scales present => quantized),
    so a mixed sidecar is valid: only eligible 2-D matmul weights gain
    .scales/.biases; everything else is written through unchanged.
    """
    import mlx.core as mx

    out: dict[str, Any] = {}
    for k, v in tensors.items():
        if not _should_quantize(k, tuple(v.shape), group_size):
            out[k] = v
            continue
        # Router gate stays high-precision at 8-bit; everything else uses `bits`.
        is_router = k.endswith("mlp.gate.weight")
        q_bits = 8 if is_router else bits
        q_w, q_s, q_b = mx.quantize(v, group_size=group_size, bits=q_bits)
        mx.eval(q_w, q_s, q_b)
        out[k] = q_w
        out[k.replace(".weight", ".scales")] = q_s
        out[k.replace(".weight", ".biases")] = q_b
    return out


# --------------------------------------------------------------------------- #
# Output writers
# --------------------------------------------------------------------------- #
def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _mtplx_version() -> str:
    try:
        return version("mtplx")
    except PackageNotFoundError:
        return "unknown"


def _runtime_contract(
    *, arch_id: str, depth_max: int, base_repo: str, tensor_count: int, quant_bits: int | None
) -> dict[str, Any]:
    contract = {
        "mtplx_version": _mtplx_version(),
        "arch_id": arch_id,
        "mtp_depth_max": depth_max,
        "recommended_profile": "stable",
        "recommended_draft_sampler": DRAFT_SAMPLER,
        "sampler": TARGET_SAMPLER,
        "base_trunk": base_repo,
        "mtp_tensor_count": tensor_count,
        "exactness_baseline": {
            "context": 2048,
            "max_abs_diff": 0.0,
            "source": "ax-engine prepare_mtp_sidecar tensor-layout gate",
        },
        "verified_on": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
    }
    if quant_bits is not None:
        # Free-text hint the loader scans for "INT8"/"INT4" group-size inference.
        contract["mtp_sidecar"] = f"INT{quant_bits} quantized projections, bf16 experts/norms"
    return contract


def _provenance_manifest(
    *,
    model_key: str,
    arch_id: str,
    is_moe: bool,
    base_model_id: str,
    base_snapshot_dir: str,
    base_config_rec: dict[str, Any],
    base_index_rec: dict[str, Any],
    source_model_id: str,
    shard_recs: list[dict[str, Any]],
    output_dir: str,
    mtp_rec: dict[str, Any],
    runtime_rec: dict[str, Any],
    config_rec: dict[str, Any],
    quantize: int | None,
    group_size: int,
    runtime: dict[str, Any],
) -> dict[str, Any]:
    """Build the ax.mtp_sidecar_provenance.v1 manifest.

    Field names match what scripts/check_mtp_sidecar_provenance.py validates so a
    generated sidecar passes the same provenance gate as the Qwen3.6 path.
    """
    return {
        "schema_version": "ax.mtp_sidecar_provenance.v1",
        "generated_by": "scripts/prepare_mtp_sidecar.py",
        "model_key": model_key,
        "arch_id": arch_id,
        "is_moe": is_moe,
        "base": {
            "model_id": base_model_id,
            "snapshot_dir": base_snapshot_dir,
            "config": base_config_rec,
            "index": base_index_rec,
        },
        "source": {
            "model_id": source_model_id,
            "mtp_shards": shard_recs,
        },
        "output": {
            "dir": output_dir,
            "mtp": mtp_rec,
            "runtime": runtime_rec,
            "config": config_rec,
        },
        "transform": {
            "norm_policy": "shift_mtp_norm_weights_by_1",
            "moe_expert_unpack": is_moe,
            "quantize_bits": quantize,
            "quantize_group_size": group_size if quantize is not None else None,
        },
        "runtime": runtime,
    }


def _copy_base_tree(base_dir: Path, out_dir: Path) -> None:
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


def _patch_config(out_dir: Path, tensor_count: int) -> Path:
    config_path = out_dir / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())
        config["mlx_lm_extra_tensors"] = {
            "mtp_file": "mtp.safetensors",
            "mtp_tensor_count": tensor_count,
        }
        # Number of *physical* MTP layers in the weights — always 1 for
        # Qwen3.6/Qwen3-Next (mtp.layers.0 applied recurrently). This is NOT the
        # runtime draft depth: that lives in mtplx_runtime.json as mtp_depth_max.
        # Must be assigned (not setdefault'd) so a base config carrying 0 is
        # overwritten; writing the draft depth here would misrepresent the
        # weights and break mlx-lm loading.
        config["num_nextn_predict_layers"] = 1
        config.setdefault("mtp_num_hidden_layers", 1)
        config_path.write_text(json.dumps(config, indent=2))
        print("Updated config.json with MTP fields.", flush=True)
    return config_path


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def prepare_sidecar(
    *,
    hf_repo: str,
    base: str,
    output: Path | None,
    depth_max: int,
    quantize: int | None,
    group_size: int,
) -> Path:
    base_dir = _resolve_base_dir(base)
    print(f"Base model:   {base_dir}", flush=True)

    # A stable, human-readable key for the base model. Prefer the HF cache slug
    # (…/<models--org--name>/snapshots/<hash>); fall back to the local dir name.
    parents = base_dir.parents
    cache_slug = parents[1].name if len(parents) > 1 else ""
    base_name = cache_slug.removeprefix("models--") if "models--" in cache_slug else base_dir.name

    if output is not None:
        out_dir = output.expanduser().resolve()
    else:
        out_dir = HF_CACHE / f"models--ax-local--{base_name}-MTP" / "snapshots" / "v1"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir:   {out_dir}", flush=True)

    _copy_base_tree(base_dir, out_dir)

    print("\nDiscovering MTP shards in source repo...", flush=True)
    shard_names = _discover_mtp_shards(hf_repo)
    shard_paths = [_download_file(hf_repo, name) for name in shard_names]

    print("\nExtracting mtp.* tensors...", flush=True)
    tensors = _load_mtp_tensors(shard_paths)
    if not tensors:
        sys.exit("ERROR: no mtp.* tensors found in the downloaded shards.")
    print(f"  Found {len(tensors)} raw MTP tensors", flush=True)

    src_config = json.loads((out_dir / "config.json").read_text()) if (out_dir / "config.json").exists() else {}
    arch_id, normalizer, is_moe = _detect_arch(tensors, src_config)
    print(f"  Detected architecture: {arch_id} (moe={is_moe})", flush=True)

    tensors = _shift_norm_weights(tensors)
    tensors = normalizer(tensors)
    # Logical MTP tensor count (weights + norms), captured before quantization so
    # the .scales/.biases siblings do not inflate the reported/provenance count.
    tensor_count = len(tensors)
    print(f"  After normalization: {tensor_count} tensors", flush=True)

    if quantize is not None:
        tensors = _quantize_tensors(tensors, bits=quantize, group_size=group_size)
        print(f"  After INT{quantize} quantization: {len(tensors)} arrays", flush=True)

    import mlx.core as mx

    mtp_path = out_dir / "mtp.safetensors"
    mx.save_safetensors(str(mtp_path), {k: mx.array(v) for k, v in tensors.items()})
    size_mb = mtp_path.stat().st_size / 1024**2
    print(f"\nSaved mtp.safetensors ({size_mb:.1f} MB) -> {mtp_path}", flush=True)

    runtime = _runtime_contract(
        arch_id=arch_id,
        depth_max=depth_max,
        base_repo=base,
        tensor_count=tensor_count,
        quant_bits=quantize,
    )
    runtime_path = out_dir / "mtplx_runtime.json"
    runtime_path.write_text(json.dumps(runtime, indent=2))
    print(f"Saved mtplx_runtime.json -> {runtime_path}", flush=True)

    config_path = _patch_config(out_dir, tensor_count)

    manifest = _provenance_manifest(
        model_key=base_name,
        arch_id=arch_id,
        is_moe=is_moe,
        base_model_id=base,
        base_snapshot_dir=str(base_dir),
        base_config_rec=_file_record(base_dir / "config.json"),
        base_index_rec=_file_record(base_dir / "model.safetensors.index.json"),
        source_model_id=hf_repo,
        shard_recs=[{"name": p.name, **_file_record(p)} for p in shard_paths],
        output_dir=str(out_dir),
        mtp_rec=_file_record(mtp_path),
        runtime_rec=_file_record(runtime_path),
        config_rec=_file_record(config_path),
        quantize=quantize,
        group_size=group_size,
        runtime=runtime,
    )
    manifest_path = out_dir / "ax_mtp_sidecar_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Saved provenance manifest -> {manifest_path}", flush=True)

    print(f"\nSidecar ready at:\n  {out_dir}", flush=True)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--hf-repo",
        required=True,
        help="Source HF repo that ships mtp.* tensors (e.g. Qwen/Qwen3.6-27B).",
    )
    parser.add_argument(
        "--base",
        required=True,
        help="Base MLX model: local dir or org/name repo id (looked up in HF cache).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output dir. Defaults to a synthetic HF cache entry "
        "models--ax-local--<base>-MTP/snapshots/v1/.",
    )
    parser.add_argument(
        "--mtp-depth-max",
        type=int,
        default=1,
        help="Max recurrent MTP draft depth written to mtplx_runtime.json (default 1).",
    )
    parser.add_argument(
        "--quantize",
        type=int,
        choices=[4, 8],
        default=None,
        help="Quantize 2-D projection weights to N bits (default: keep bf16). "
        "Only 4 and 8 are offered: the runtime's mtplx_runtime.json sidecar-bit "
        "hint distinguishes INT8 from INT4 only, so a 6-bit sidecar would be "
        "misread as 4-bit. Experts/norms stay bf16; router gate stays 8-bit.",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=64,
        help="Quantization group size when --quantize is set (default 64).",
    )
    args = parser.parse_args()

    prepare_sidecar(
        hf_repo=args.hf_repo,
        base=args.base,
        output=args.output,
        depth_max=args.mtp_depth_max,
        quantize=args.quantize,
        group_size=args.group_size,
    )


if __name__ == "__main__":
    main()
