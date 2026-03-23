#!/usr/bin/env python3
"""
Generate kernel profile JSON from llama.cpp parameter rules.

This script reads GGUF model headers and generates optimized kernel dispatch
profiles based on llama.cpp's proven parameter choices.
"""

import json
import struct
import sys
from pathlib import Path


GGUF_MAGIC = 0x46554747  # "GGUF" in little-endian

GGML_TYPE_Q4_K = 12
GGML_TYPE_Q6_K = 14
GGML_TYPE_Q8_0 = 8


def read_gguf_header(path: str) -> dict:
    """Read basic model config from GGUF header."""
    with open(path, "rb") as f:
        magic = struct.unpack("<I", f.read(4))[0]
        if magic != GGUF_MAGIC:
            raise ValueError(f"Invalid GGUF magic: 0x{magic:08x}")
        
        version = struct.unpack("<I", f.read(4))[0]
        tensor_count = struct.unpack("<Q", f.read(8))[0]
        metadata_kv_count = struct.unpack("<Q", f.read(8))[0]
        
        return {
            "version": version,
            "tensor_count": tensor_count,
            "metadata_kv_count": metadata_kv_count,
        }


def llama_cpp_matvec_params(quant_type: str) -> dict:
    """llama.cpp mul_mv kernel parameters for decode (N=1)."""
    return {
        "threadgroup_size": 256,
        "rows_per_tg": 2,
        "blocks_per_thread": 4,
        "x_in_threadgroup_mem": True,
    }


def llama_cpp_batch_params(quant_type: str) -> dict:
    """llama.cpp mul_mm kernel parameters for prefill (N>1)."""
    return {
        "tile_m": 32,
        "tile_n": 32,
        "tile_k": 32,
        "threadgroup_size": 256,
        "use_f16_io": True,
        "use_pair_kernel": True,
    }


def llama_cpp_attention_decode_params() -> dict:
    """llama.cpp flash_attn_ext parameters for decode."""
    return {
        "kernel": "splitk",
        "threadgroup_size": 256,
        "splitk_chunk_size": 256,
        "splitk_threshold": 256,
        "fallback_kernel": "single_tg",
    }


def llama_cpp_attention_prefill_params() -> dict:
    """llama.cpp flash_attn parameters for prefill."""
    return {
        "kernel": "fa2_hd128",
        "threadgroup_size": 256,
        "queries_per_tg": 8,
        "fa2_auto_threshold": 512,
    }


def generate_profile(model_name: str, quant: str) -> dict:
    """Generate a kernel profile for the given model."""
    profile = {
        "model": model_name,
        "source": "llama.cpp-inspired",
        "generated": "2026-03-22",
        "decode_matvec": {},
        "batch_matmul": {},
        "attention_decode": llama_cpp_attention_decode_params(),
        "attention_prefill": llama_cpp_attention_prefill_params(),
        "elementwise": {
            "threadgroup_size": 256,
            "use_fused_residual_norm": False,
        },
    }
    
    for quant_type in ["q4_k", "q6_k", "q8_0"]:
        profile["decode_matvec"][quant_type] = llama_cpp_matvec_params(quant_type)
        profile["batch_matmul"][quant_type] = llama_cpp_batch_params(quant_type)
    
    return profile


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate kernel profile JSON from llama.cpp rules"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name (e.g., Qwen3-8B)",
    )
    parser.add_argument(
        "--quant",
        required=True,
        help="Quantization type (e.g., q4_k_m)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON file path",
    )
    parser.add_argument(
        "--gguf",
        help="Optional GGUF file to read dimensions from",
    )
    
    args = parser.parse_args()
    
    profile = generate_profile(args.model, args.quant)
    
    if args.gguf:
        try:
            header = read_gguf_header(args.gguf)
            profile["gguf_version"] = header["version"]
            profile["tensor_count"] = header["tensor_count"]
        except Exception as e:
            print(f"Warning: Could not read GGUF header: {e}", file=sys.stderr)
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(profile, f, indent=2)
    
    print(f"Generated profile: {output_path}")


if __name__ == "__main__":
    main()
