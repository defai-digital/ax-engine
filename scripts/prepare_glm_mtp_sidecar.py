#!/usr/bin/env python3
"""Prepare an ax-engine-compatible GLM MTP sidecar from GLM-4.7-Flash.

GLM-4.7-Flash embeds its single MTP head inside the base checkpoint as
``model.layers.47.*`` (the last layer of a 48-layer model).  Unlike the Qwen3
family there is no separate ``mtp.*`` namespace, so this script:

  1. Downloads ``zai-org/GLM-4.7-Flash`` (or reads a local snapshot) and
     identifies which shard(s) contain ``model.layers.47.*`` tensors.
  2. Extracts only that layer, renames every tensor to the ``glm_mtp.*``
     canonical key layout expected by the ax-engine loader:
       - model.layers.47.eh_proj.weight          → glm_mtp.eh_proj.weight
       - model.layers.47.enorm.weight             → glm_mtp.enorm.weight
       - model.layers.47.hnorm.weight             → glm_mtp.hnorm.weight
       - model.layers.47.shared_head.head.weight  → glm_mtp.shared_head.head.weight
       - model.layers.47.shared_head.norm.weight  → glm_mtp.shared_head.norm.weight
       - model.layers.47.input_layernorm.weight   → glm_mtp.layer.input_layernorm.weight
       - model.layers.47.post_attention_layernorm.weight
                                                  → glm_mtp.layer.post_attention_layernorm.weight
       - model.layers.47.self_attn.*              → glm_mtp.layer.self_attn.*
       - model.layers.47.mlp.*                    → glm_mtp.layer.mlp.*
         (per-expert tensors are stacked into [E, D, in] arrays)
     Note: model.layers.47.embed_tokens.weight is skipped (shared; already in
     the main model).
  3. Optionally quantizes 2-D projection weights to match the base model bits.
  4. Writes the GLM MTP sidecar contract alongside the quantized MLX model:
       - glm_mtp.safetensors        (MTP head weights)
       - glm_mtp_runtime.json       (depth / draft-sampler / quant hint)
       - ax_glm_mtp_manifest.json   (provenance for fair benchmark review)

Usage:
  # Minimal — auto-detects base from HF cache:
  python3 scripts/prepare_glm_mtp_sidecar.py \\
      --base mlx-community/GLM-4.7-Flash-4bit

  # Explicit HF source repo, local base dir, INT4 quantization:
  python3 scripts/prepare_glm_mtp_sidecar.py \\
      --hf-repo zai-org/GLM-4.7-Flash \\
      --base ~/.cache/huggingface/hub/models--mlx-community--GLM-4.7-Flash-4bit/snapshots/<hash> \\
      --quantize 4

``--base`` accepts either a local model directory or an ``org/name`` repo id
(looked up in the local HF cache; download it first via
``scripts/download_model.py`` if absent).  Output defaults to a synthetic HF
cache entry ``models--ax-local--<base-name>-GlmMTP/snapshots/v1/``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

# ── constants ───────────────────────────────────────────────────────────────
DEFAULT_HF_REPO = "zai-org/GLM-4.7-Flash"
MTP_LAYER_IDX = 47          # model.layers.47.* holds the single GLM MTP head
SKIP_KEY_SUFFIX = "embed_tokens.weight"   # shared with main model; skip it

DRAFT_SAMPLER = {"temperature": 0.7, "top_k": 20, "top_p": 0.95}
TARGET_SAMPLER = {"temperature": 0.6, "top_k": 20, "top_p": 0.95}

HF_CACHE = Path(
    os.environ.get("HF_HUB_CACHE")
    or (Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub")
)


# ── HF cache helpers ─────────────────────────────────────────────────────────
def _repo_slug(repo_id: str) -> str:
    return "models--" + repo_id.replace("/", "--")


def _latest_snapshot(model_dir: Path) -> Path | None:
    snaps_dir = model_dir / "snapshots"
    if not snaps_dir.exists():
        return None
    snapshots = sorted(snaps_dir.iterdir())
    return snapshots[-1] if snapshots else None


def _resolve_base_dir(base: str) -> Path:
    candidate = Path(base).expanduser()
    if candidate.exists() and (candidate / "config.json").exists():
        return candidate.resolve()
    if candidate.exists() and (candidate / "snapshots").exists():
        snap = _latest_snapshot(candidate)
        if snap:
            return snap.resolve()
    snap = _latest_snapshot(HF_CACHE / _repo_slug(base))
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


def _discover_layer47_shards(hf_repo: str) -> list[str]:
    """Return the shard filename(s) that hold model.layers.47.* tensors."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError

    prefix = f"model.layers.{MTP_LAYER_IDX}."
    try:
        idx_path = hf_hub_download(hf_repo, "model.safetensors.index.json")
    except EntryNotFoundError:
        print("  No index.json — assuming single-shard model.safetensors", flush=True)
        return ["model.safetensors"]

    weight_map = json.loads(Path(idx_path).read_text()).get("weight_map", {})
    shards = sorted({rel for key, rel in weight_map.items() if key.startswith(prefix)})
    if not shards:
        sys.exit(
            f"ERROR: no {prefix}* tensors found in {hf_repo}/model.safetensors.index.json.\n"
            f"  This checkpoint does not appear to be a GLM-4.7-Flash model."
        )
    print(f"  Layer-{MTP_LAYER_IDX} tensors live in {len(shards)} shard(s): {shards}", flush=True)
    return shards


# ── tensor extraction + renaming ─────────────────────────────────────────────
_LAYER_PREFIX = f"model.layers.{MTP_LAYER_IDX}."
_MTP_PREFIX = "glm_mtp."

# Direct tensor renames (special-cased tensors outside the sub-module tree).
_DIRECT_RENAMES: dict[str, str] = {
    "eh_proj.weight":                "eh_proj.weight",
    "enorm.weight":                  "enorm.weight",
    "hnorm.weight":                  "hnorm.weight",
    "shared_head.head.weight":       "shared_head.head.weight",
    "shared_head.norm.weight":       "shared_head.norm.weight",
    "input_layernorm.weight":        "layer.input_layernorm.weight",
    "post_attention_layernorm.weight": "layer.post_attention_layernorm.weight",
}


def _rename_key(raw_key: str) -> str | None:
    """Map a raw HF key under model.layers.47.* to glm_mtp.*.

    Returns None if the key should be skipped.
    """
    if not raw_key.startswith(_LAYER_PREFIX):
        return None  # not our layer
    suffix = raw_key[len(_LAYER_PREFIX):]

    # Skip shared embed_tokens
    if suffix == SKIP_KEY_SUFFIX or suffix.endswith("." + SKIP_KEY_SUFFIX):
        return None

    # Direct renames first
    if suffix in _DIRECT_RENAMES:
        return _MTP_PREFIX + _DIRECT_RENAMES[suffix]

    # self_attn.* and mlp.* sub-modules → glm_mtp.layer.*
    if suffix.startswith("self_attn.") or suffix.startswith("mlp."):
        return _MTP_PREFIX + "layer." + suffix

    # Anything else (future-proofing): keep under glm_mtp.layer.*
    return _MTP_PREFIX + "layer." + suffix


def _load_layer47_tensors(shard_paths: list[Path]) -> dict[str, Any]:
    """Load and rename all model.layers.47.* tensors; skip embed_tokens."""
    import mlx.core as mx

    tensors: dict[str, Any] = {}
    for shard in shard_paths:
        loaded = mx.load(str(shard))
        for k, v in loaded.items():
            new_key = _rename_key(k)
            if new_key is None:
                continue
            tensors[new_key] = v
        del loaded
    return tensors


# ── per-expert stacking (GLM MoE FFN) ────────────────────────────────────────
def _stack_per_expert_tensors(tensors: dict[str, Any]) -> dict[str, Any]:
    """Stack per-expert gate/up/down tensors into [E, D, in] arrays.

    GLM-4.7-Flash ships MoE experts as separate keys:
      glm_mtp.layer.mlp.experts.<idx>.gate_proj.weight
      glm_mtp.layer.mlp.experts.<idx>.up_proj.weight
      glm_mtp.layer.mlp.experts.<idx>.down_proj.weight

    The ax-engine loader (LayerWeights::gate_exps / up_exps / down_exps) expects
    stacked arrays:
      glm_mtp.layer.mlp.gate_proj.weight  [E, D, in]
      glm_mtp.layer.mlp.up_proj.weight    [E, D, in]
      glm_mtp.layer.mlp.down_proj.weight  [E, D, in]
    """
    import mlx.core as mx

    expert_pat = re.compile(
        r"^(glm_mtp\.layer\.mlp)\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$"
    )
    per_expert: dict[tuple[str, str], dict[int, Any]] = {}
    passthrough: dict[str, Any] = {}

    for k, v in tensors.items():
        m = expert_pat.match(k)
        if m:
            prefix, idx_str, role = m.group(1), m.group(2), m.group(3)
            key = (prefix, role)
            per_expert.setdefault(key, {})[int(idx_str)] = v
        else:
            passthrough[k] = v

    out: dict[str, Any] = dict(passthrough)
    for (prefix, role), expert_map in per_expert.items():
        n_experts = max(expert_map) + 1
        stacked = mx.stack([expert_map[i] for i in range(n_experts)], 0)
        mx.eval(stacked)
        out[f"{prefix}.{role}.weight"] = stacked

    return out


# ── optional quantization ────────────────────────────────────────────────────
def _should_quantize(key: str, shape: tuple[int, ...], group_size: int) -> bool:
    """True when a tensor is a 2-D projection weight eligible for quantization."""
    if "norm" in key or "layernorm" in key:
        return False
    if "router" in key or "gate.weight" in key:
        return False
    if len(shape) != 2:
        return False
    last = shape[-1]
    return last >= group_size and last % group_size == 0


def _quantize_tensors(tensors: dict[str, Any], bits: int, group_size: int) -> dict[str, Any]:
    import mlx.core as mx

    out: dict[str, Any] = {}
    for k, v in tensors.items():
        if not _should_quantize(k, tuple(v.shape), group_size):
            out[k] = v
            continue
        q_w, q_s, q_b = mx.quantize(v, group_size=group_size, bits=bits)
        mx.eval(q_w, q_s, q_b)
        out[k] = q_w
        out[k.replace(".weight", ".scales")] = q_s
        out[k.replace(".weight", ".biases")] = q_b
    return out


# ── output helpers ───────────────────────────────────────────────────────────
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


def _ax_version() -> str:
    try:
        return version("ax-engine")
    except PackageNotFoundError:
        try:
            return version("mtplx")
        except PackageNotFoundError:
            return "unknown"


def _runtime_contract(
    *,
    depth_max: int,
    base_repo: str,
    tensor_count: int,
    quant_bits: int | None,
) -> dict[str, Any]:
    contract: dict[str, Any] = {
        "ax_version": _ax_version(),
        "arch_id": "glm-4.7-flash-mtp",
        "mtp_depth_max": depth_max,
        "recommended_profile": "stable",
        "recommended_draft_sampler": DRAFT_SAMPLER,
        "sampler": TARGET_SAMPLER,
        "base_trunk": base_repo,
        "mtp_tensor_count": tensor_count,
        "verified_on": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
    }
    if quant_bits is not None:
        contract["mtp_sidecar"] = f"INT{quant_bits} quantized projections, bf16 norms/router"
    return contract


def _provenance_manifest(
    *,
    base_model_id: str,
    base_snapshot_dir: str,
    source_model_id: str,
    shard_recs: list[dict[str, Any]],
    output_dir: str,
    mtp_rec: dict[str, Any],
    runtime_rec: dict[str, Any],
    quantize: int | None,
    group_size: int,
    runtime: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "ax.glm_mtp_sidecar_provenance.v1",
        "generated_by": "scripts/prepare_glm_mtp_sidecar.py",
        "arch_id": "glm-4.7-flash-mtp",
        "base": {
            "model_id": base_model_id,
            "snapshot_dir": base_snapshot_dir,
            "config": _file_record(Path(base_snapshot_dir) / "config.json"),
        },
        "source": {
            "model_id": source_model_id,
            "mtp_layer_index": MTP_LAYER_IDX,
            "shards": shard_recs,
        },
        "output": {
            "dir": output_dir,
            "mtp": mtp_rec,
            "runtime": runtime_rec,
        },
        "transform": {
            "moe_expert_stack": True,
            "embed_tokens_skipped": True,
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


# ── main driver ───────────────────────────────────────────────────────────────
def prepare_glm_mtp_sidecar(
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

    parents = base_dir.parents
    cache_slug = parents[1].name if len(parents) > 1 else ""
    base_name = cache_slug.removeprefix("models--") if "models--" in cache_slug else base_dir.name

    if output is not None:
        out_dir = output.expanduser().resolve()
    else:
        out_dir = HF_CACHE / f"models--ax-local--{base_name}-GlmMTP" / "snapshots" / "v1"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir:   {out_dir}", flush=True)

    _copy_base_tree(base_dir, out_dir)

    print(f"\nDiscovering layer-{MTP_LAYER_IDX} shards in {hf_repo}...", flush=True)
    shard_names = _discover_layer47_shards(hf_repo)
    shard_paths = [_download_file(hf_repo, name) for name in shard_names]

    print(f"\nExtracting model.layers.{MTP_LAYER_IDX}.* tensors...", flush=True)
    tensors = _load_layer47_tensors(shard_paths)
    if not tensors:
        sys.exit(
            f"ERROR: no model.layers.{MTP_LAYER_IDX}.* tensors found in the downloaded shards.\n"
            "  Verify the HF repo contains GLM-4.7-Flash weights."
        )
    print(f"  Loaded {len(tensors)} tensors after renaming", flush=True)

    tensors = _stack_per_expert_tensors(tensors)
    tensor_count = len(tensors)
    print(f"  After expert stacking: {tensor_count} tensors", flush=True)

    if quantize is not None:
        tensors = _quantize_tensors(tensors, bits=quantize, group_size=group_size)
        print(f"  After INT{quantize} quantization: {len(tensors)} arrays", flush=True)

    import mlx.core as mx

    mtp_path = out_dir / "glm_mtp.safetensors"
    mx.save_safetensors(str(mtp_path), {k: mx.array(v) for k, v in tensors.items()})
    size_mb = mtp_path.stat().st_size / 1024**2
    print(f"\nSaved glm_mtp.safetensors ({size_mb:.1f} MB) -> {mtp_path}", flush=True)

    runtime = _runtime_contract(
        depth_max=depth_max,
        base_repo=base,
        tensor_count=tensor_count,
        quant_bits=quantize,
    )
    runtime_path = out_dir / "glm_mtp_runtime.json"
    runtime_path.write_text(json.dumps(runtime, indent=2))
    print(f"Saved glm_mtp_runtime.json -> {runtime_path}", flush=True)

    manifest = _provenance_manifest(
        base_model_id=base,
        base_snapshot_dir=str(base_dir),
        source_model_id=hf_repo,
        shard_recs=[{"name": p.name, **_file_record(p)} for p in shard_paths],
        output_dir=str(out_dir),
        mtp_rec=_file_record(mtp_path),
        runtime_rec=_file_record(runtime_path),
        quantize=quantize,
        group_size=group_size,
        runtime=runtime,
    )
    manifest_path = out_dir / "ax_glm_mtp_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Saved provenance manifest -> {manifest_path}", flush=True)

    print(f"\nGLM MTP sidecar ready at:\n  {out_dir}", flush=True)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--hf-repo",
        default=DEFAULT_HF_REPO,
        help=f"Source HF repo containing GLM-4.7-Flash weights (default: {DEFAULT_HF_REPO}).",
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
        help="Output dir.  Defaults to a synthetic HF cache entry "
        "models--ax-local--<base>-GlmMTP/snapshots/v1/.",
    )
    parser.add_argument(
        "--mtp-depth-max",
        type=int,
        default=1,
        help="Max speculative depth written to glm_mtp_runtime.json (default 1).",
    )
    parser.add_argument(
        "--quantize",
        type=int,
        choices=[4, 8],
        default=None,
        help="Quantize 2-D projection weights to N bits (default: keep bf16).  "
        "Experts/norms stay bf16; router gate is excluded.",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=64,
        help="Quantization group size when --quantize is set (default 64).",
    )
    args = parser.parse_args()

    prepare_glm_mtp_sidecar(
        hf_repo=args.hf_repo,
        base=args.base,
        output=args.output,
        depth_max=args.mtp_depth_max,
        quantize=args.quantize,
        group_size=args.group_size,
    )


if __name__ == "__main__":
    main()
