#!/usr/bin/env python3
"""Re-quantize the Gemma 4 12B FFN weights from 8-bit to 4-bit (group 64).

The mlx-community/gemma-4-12B-it-4bit artifact keeps attention at 4-bit but
overrides every layer's mlp.{gate,up,down}_proj to 8-bit. FFN is ~75% of the
weights and decode is bandwidth-bound, so the 8-bit FFN is the dominant cost of
the AX-vs-llama.cpp direct-decode gap. This produces a uniformly-4-bit artifact:
dequantize each 8-bit FFN tensor and re-quantize to 4-bit, copy everything else
unchanged, and strip the per-layer 8-bit overrides from config.json.

Usage: python3 scripts/requantize_gemma4_12b_ffn_4bit.py <BASE_DIR> <OUT_DIR>
"""
import glob
import json
import os
import shutil
import struct
import sys

import mlx.core as mx

GROUP = 64


def safetensors_metadata(path: str) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return header.get("__metadata__", {}) or {}


def is_ffn_weight(name: str) -> bool:
    return ".mlp." in name and name.endswith(".weight")


def main() -> None:
    base, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)

    shards = sorted(glob.glob(os.path.join(base, "*.safetensors")))
    if not shards:
        sys.exit(f"no safetensors in {base}")

    total_requant = 0
    for shard in shards:
        name = os.path.basename(shard)
        tensors = dict(mx.load(shard).items())
        meta = safetensors_metadata(shard)

        ffn_bases = {k[: -len(".weight")] for k in tensors if is_ffn_weight(k)}
        out_tensors = {}
        for b in ffn_bases:
            w = tensors[b + ".weight"]
            s = tensors[b + ".scales"]
            bi = tensors[b + ".biases"]
            full = mx.dequantize(w, scales=s, biases=bi, group_size=GROUP, bits=8)
            wq, sq, bq = mx.quantize(full, group_size=GROUP, bits=4)
            out_tensors[b + ".weight"] = wq
            out_tensors[b + ".scales"] = sq
            out_tensors[b + ".biases"] = bq
        for k, v in tensors.items():
            out_tensors.setdefault(k, v)

        mx.eval(list(out_tensors.values()))
        mx.save_safetensors(os.path.join(out, name), out_tensors, metadata=meta)
        total_requant += len(ffn_bases)
        print(f"  {name}: {len(out_tensors)} tensors, {len(ffn_bases)} FFN re-quantized 8->4")

    # Copy aux files (tokenizer, index, etc.); skip config + any stale manifest.
    for f in glob.glob(os.path.join(base, "*")):
        bn = os.path.basename(f)
        if bn.endswith(".safetensors") or bn in ("config.json", "model-manifest.json"):
            continue
        if os.path.isfile(f):
            shutil.copy2(f, os.path.join(out, bn))

    # Strip the per-layer 8-bit FFN overrides so the artifact is uniformly 4-bit.
    cfg = json.load(open(os.path.join(base, "config.json")))
    q = cfg.get("quantization", {})
    removed = [k for k in list(q) if isinstance(q[k], dict) and ".mlp." in k]
    for k in removed:
        del q[k]
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=2)
    print(f"config: removed {len(removed)} per-layer 8-bit FFN overrides -> uniform 4-bit")
    print(f"done: {total_requant} FFN tensors re-quantized; artifact at {out}")


if __name__ == "__main__":
    main()
