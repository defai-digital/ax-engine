#!/usr/bin/env python3
"""Generate a model-manifest.json for an mlx-community embedding model.

Usage:
    python scripts/generate_embedding_manifest.py .internal/models/qwen3-embedding-0.6b-8bit
    python scripts/generate_embedding_manifest.py .internal/models/qwen3-embedding-4b-4bit
"""
from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Safetensors header reader
# ---------------------------------------------------------------------------

def read_safetensors_header(path: Path) -> dict[str, Any]:
    """Return the JSON header dict from a safetensors file."""
    with open(path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        raw = f.read(header_size)
    header = json.loads(raw)
    header.pop("__metadata__", None)
    return header


def dtype_map(st_dtype: str) -> str:
    """Map safetensors dtype string to ax-engine manifest dtype string."""
    return {
        "BF16": "bf16",
        "F16": "f16",
        "F32": "f32",
        "I32": "i32",
        "U32": "u32",
        "U8": "u8",
        "I8": "i8",
    }.get(st_dtype.upper(), st_dtype.lower())


# ---------------------------------------------------------------------------
# Tensor role mapping for Qwen3 dense  (model.* prefix, no linear attn/MoE)
# ---------------------------------------------------------------------------

QWEN3_DENSE_ROLES: list[tuple[re.Pattern[str], str, bool]] = [
    # (pattern, role, is_per_layer)
    (re.compile(r"^model\.embed_tokens\.weight$"),         "token_embedding",  False),
    (re.compile(r"^model\.norm\.weight$"),                 "final_norm",       False),
    (re.compile(r"^lm_head\.weight$"),                     "lm_head",          False),
    (re.compile(r"^model\.layers\.(\d+)\.input_layernorm\.weight$"),          "attention_norm",   True),
    (re.compile(r"^model\.layers\.(\d+)\.post_attention_layernorm\.weight$"), "ffn_norm",         True),
    (re.compile(r"^model\.layers\.(\d+)\.self_attn\.q_norm\.weight$"),        "attention_q_norm", True),
    (re.compile(r"^model\.layers\.(\d+)\.self_attn\.k_norm\.weight$"),        "attention_k_norm", True),
    (re.compile(r"^model\.layers\.(\d+)\.self_attn\.q_proj\.weight$"),        "attention_q",      True),
    (re.compile(r"^model\.layers\.(\d+)\.self_attn\.k_proj\.weight$"),        "attention_k",      True),
    (re.compile(r"^model\.layers\.(\d+)\.self_attn\.v_proj\.weight$"),        "attention_v",      True),
    (re.compile(r"^model\.layers\.(\d+)\.self_attn\.o_proj\.weight$"),        "attention_o",      True),
    (re.compile(r"^model\.layers\.(\d+)\.mlp\.gate_proj\.weight$"),           "ffn_gate",         True),
    (re.compile(r"^model\.layers\.(\d+)\.mlp\.up_proj\.weight$"),             "ffn_up",           True),
    (re.compile(r"^model\.layers\.(\d+)\.mlp\.down_proj\.weight$"),           "ffn_down",         True),
]

UNQUANTIZED_ROLES = {
    "final_norm", "attention_norm", "ffn_norm",
    "attention_q_norm", "attention_k_norm",
}


def match_qwen3_dense(name: str) -> tuple[str, int | None] | None:
    for pattern, role, is_per_layer in QWEN3_DENSE_ROLES:
        m = pattern.match(name)
        if m:
            layer_idx = int(m.group(1)) if is_per_layer else None
            return role, layer_idx
    return None


# ---------------------------------------------------------------------------
# Build tensor specs from safetensors files
# ---------------------------------------------------------------------------

def build_tensor_specs(
    model_dir: Path,
    quant_bits: int,
    quant_group_size: int,
) -> list[dict[str, Any]]:
    """Return list of NativeTensorSpec dicts for all weight tensors."""
    # Collect all safetensors shards and their file paths
    idx_file = model_dir / "model.safetensors.index.json"
    if idx_file.exists():
        idx = json.loads(idx_file.read_text())
        weight_map: dict[str, str] = idx["weight_map"]
        shard_files = sorted(set(weight_map.values()))
    else:
        single = list(model_dir.glob("model.safetensors"))
        if not single:
            single = list(model_dir.glob("*.safetensors"))
        if not single:
            raise FileNotFoundError(f"No safetensors files in {model_dir}")
        weight_map = {}  # will be filled from header
        shard_files = [f.name for f in sorted(single)]

    # Read headers from each shard
    # Maps: tensor_name → (file_name, dtype_str, shape, data_offsets_in_data_section)
    tensor_info: dict[str, tuple[str, str, list[int], list[int]]] = {}
    for shard_name in shard_files:
        shard_path = model_dir / shard_name
        header = read_safetensors_header(shard_path)
        # Header size = the 8-byte prefix value
        with open(shard_path, "rb") as f:
            header_size = struct.unpack("<Q", f.read(8))[0]
        data_section_start = 8 + header_size  # byte offset where data begins

        for tname, tinfo in header.items():
            dtype_str = tinfo["dtype"]
            shape = tinfo["shape"]
            data_offsets = tinfo["data_offsets"]  # [begin, end] relative to data section
            offset_bytes = data_section_start + data_offsets[0]
            length_bytes = data_offsets[1] - data_offsets[0]
            tensor_info[tname] = (shard_name, dtype_str, shape, [offset_bytes, length_bytes])

    specs: list[dict[str, Any]] = []

    # Skip sidecar tensors (.scales, .biases) — loader finds them automatically
    skip_suffixes = (".scales", ".biases")
    weight_names = [n for n in sorted(tensor_info.keys()) if not any(n.endswith(s) for s in skip_suffixes)]

    for name in weight_names:
        matched = match_qwen3_dense(name)
        if matched is None:
            continue  # skip unknown tensors (e.g., rotary_emb, attention_mask, etc.)

        role, layer_idx = matched
        file_name, dtype_str, shape, (offset_bytes, length_bytes) = tensor_info[name]
        ax_dtype = dtype_map(dtype_str)

        # Determine if this tensor has quantization sidecars
        base = name[: -len(".weight")] if name.endswith(".weight") else name
        has_scales = f"{base}.scales" in tensor_info
        is_quantized = has_scales and (role not in UNQUANTIZED_ROLES)

        spec: dict[str, Any] = {
            "name": name,
            "role": role,
        }
        if layer_idx is not None:
            spec["layer_index"] = layer_idx

        if is_quantized:
            spec["dtype"] = "u32"
            spec["source_quantized"] = True
            spec["quantization"] = {
                "mode": "affine",
                "group_size": quant_group_size,
                "bits": quant_bits,
            }
        else:
            spec["dtype"] = ax_dtype

        spec["shape"] = shape
        spec["file"] = file_name
        spec["offset_bytes"] = offset_bytes
        spec["length_bytes"] = length_bytes

        specs.append(spec)

    # Sort: global tensors first, then by layer_index, then by role
    def sort_key(s: dict[str, Any]) -> tuple[int, int, str]:
        li = s.get("layer_index", -1)
        return (0 if li == -1 else 1, li if isinstance(li, int) else -1, s["role"])

    specs.sort(key=sort_key)
    return specs


# ---------------------------------------------------------------------------
# Config.json → manifest fields
# ---------------------------------------------------------------------------

def parse_config(model_dir: Path) -> dict[str, Any]:
    cfg_path = model_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.json not found in {model_dir}")
    cfg = json.loads(cfg_path.read_text())

    # Qwen3 models sometimes wrap params under text_config
    text_cfg = cfg.get("text_config", cfg)

    model_type = text_cfg.get("model_type", cfg.get("model_type", ""))
    if not model_type:
        raise ValueError("model_type not found in config.json")

    if model_type not in ("qwen3", "qwen3_moe", "qwen3_5", "qwen3_5_moe"):
        raise ValueError(
            f"Unsupported model_type '{model_type}'. "
            "This script currently supports qwen3 dense models only."
        )

    hidden_size = int(text_cfg.get("hidden_size", cfg.get("hidden_size")))
    num_layers = int(text_cfg.get("num_hidden_layers", cfg.get("num_hidden_layers")))
    num_heads = int(text_cfg.get("num_attention_heads", cfg.get("num_attention_heads")))
    num_kv_heads = int(text_cfg.get("num_key_value_heads", cfg.get("num_key_value_heads", num_heads)))
    head_dim = int(text_cfg.get("head_dim", cfg.get("head_dim", hidden_size // num_heads)))
    intermediate_size = int(text_cfg.get("intermediate_size", cfg.get("intermediate_size", 0)))
    vocab_size = int(text_cfg.get("vocab_size", cfg.get("vocab_size")))
    rope_theta = int(text_cfg.get("rope_theta", cfg.get("rope_theta", 10000)))
    tie_word_embeddings = bool(text_cfg.get("tie_word_embeddings", cfg.get("tie_word_embeddings", False)))

    quant_cfg = cfg.get("quantization_config") or {}
    quant_bits = int(quant_cfg.get("bits", 8))
    quant_group_size = int(quant_cfg.get("group_size", 64))

    return {
        "model_type": model_type,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "intermediate_size": intermediate_size,
        "vocab_size": vocab_size,
        "rope_theta": rope_theta,
        "tie_word_embeddings": tie_word_embeddings,
        "quant_bits": quant_bits,
        "quant_group_size": quant_group_size,
    }


# ---------------------------------------------------------------------------
# Assemble manifest
# ---------------------------------------------------------------------------

def build_manifest(model_dir: Path, cfg: dict[str, Any], specs: list[dict[str, Any]]) -> dict[str, Any]:
    has_lm_head = any(s["role"] == "lm_head" for s in specs)
    tie = cfg["tie_word_embeddings"] or not has_lm_head

    manifest: dict[str, Any] = {
        "schema_version": "ax.native_model.v1",
        "model_family": "qwen3_dense",
        "tensor_format": "safetensors",
        "layer_count": cfg["num_layers"],
        "hidden_size": cfg["hidden_size"],
        "attention_head_count": cfg["num_heads"],
        "attention_head_dim": cfg["head_dim"],
        "kv_head_count": cfg["num_kv_heads"],
        "vocab_size": cfg["vocab_size"],
        "tie_word_embeddings": tie,
        "rope_theta": cfg["rope_theta"],
        "tensors": specs,
    }
    if cfg["intermediate_size"] > 0:
        manifest["intermediate_size"] = cfg["intermediate_size"]

    # Remove lm_head spec if we're tying (it won't exist in the file anyway)
    if tie:
        manifest["tensors"] = [s for s in specs if s["role"] != "lm_head"]

    return manifest


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Generate model-manifest.json for embedding models")
    parser.add_argument("model_dir", type=Path, help="Path to downloaded model directory")
    parser.add_argument("--dry-run", action="store_true", help="Print manifest without writing")
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    if not model_dir.exists():
        print(f"ERROR: model_dir not found: {model_dir}", file=sys.stderr)
        return 1

    print(f"Parsing config: {model_dir / 'config.json'}")
    cfg = parse_config(model_dir)
    print(
        f"  model_type={cfg['model_type']}, layers={cfg['num_layers']}, "
        f"hidden={cfg['hidden_size']}, heads={cfg['num_heads']}/{cfg['num_kv_heads']}, "
        f"head_dim={cfg['head_dim']}, vocab={cfg['vocab_size']}, "
        f"quant={cfg['quant_bits']}bit/g{cfg['quant_group_size']}"
    )

    print("Reading safetensors headers…")
    specs = build_tensor_specs(model_dir, cfg["quant_bits"], cfg["quant_group_size"])
    print(f"  {len(specs)} tensor specs")

    roles = {s["role"] for s in specs}
    print(f"  roles: {sorted(roles)}")

    manifest = build_manifest(model_dir, cfg, specs)

    out_path = model_dir / "model-manifest.json"
    if args.dry_run:
        print(json.dumps(manifest, indent=2)[:2000], "…")
    else:
        out_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Wrote {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
