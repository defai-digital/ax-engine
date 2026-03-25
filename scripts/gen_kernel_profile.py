#!/usr/bin/env python3
"""
Generate AX-compatible kernel profile JSON using llama.cpp-informed defaults.

This script emits only schema-backed fields that the AX runtime can consume.
The output is intentionally conservative: benchmark-gated defaults stay aligned
with current AX routing behavior, while still preserving model-specific profile
metadata for future tuning.
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
    """AX-compatible decode matvec parameters informed by llama.cpp NR2 routing."""
    if quant_type == "q8_0":
        return {
            "threadgroup_size": 128,
            "rows_per_simdgroup": 2,
        }
    return {
        "threadgroup_size": 64,
        "rows_per_simdgroup": 2,
    }


def batch_prefill_params() -> dict:
    """AX-compatible prefill batch-matmul routing knobs."""
    return {
        "prefer_f16_io": False,
        "prefer_pair_kernel": False,
        "small_n_threshold": 1,
        "small_m_max": 0,
        "use_bn32": True,
        "use_bk32": True,
        "q8_f16in_full_min_n": 64,
    }


def llama_cpp_attention_decode_params() -> dict:
    """AX-compatible decode attention parameters."""
    return {
        "splitk_chunk_size": 256,
        "splitk_threshold": 512,
    }


def llama_cpp_attention_prefill_params() -> dict:
    """AX-compatible prefill attention routing thresholds."""
    return {
        "fa2_mode": "off",
        "fa2_hd128_mode": "off",
        "fa2_auto_min_tokens": 512,
        "fa2_auto_min_base_seq": 256,
        "fa2_hd128_auto_min_tokens": 512,
    }


def generate_profile(model_name: str, quant: str) -> dict:
    """Generate a kernel profile for the given model."""
    profile = {
        "model": model_name,
        "source": "llama.cpp-inspired-benchmark-gated",
        "generated": "2026-03-23",
        "decode_matvec": {},
        "batch_prefill": batch_prefill_params(),
        "attention_decode": llama_cpp_attention_decode_params(),
        "attention_prefill": llama_cpp_attention_prefill_params(),
    }
    
    for quant_type in ["q4_k", "q6_k", "q8_0"]:
        profile["decode_matvec"][quant_type] = llama_cpp_matvec_params(quant_type)
    
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
            read_gguf_header(args.gguf)
        except Exception as e:
            print(f"Warning: Could not read GGUF header: {e}", file=sys.stderr)
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(profile, f, indent=2)
    
    print(f"Generated profile: {output_path}")


if __name__ == "__main__":
    main()
