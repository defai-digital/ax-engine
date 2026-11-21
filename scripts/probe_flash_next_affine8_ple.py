#!/usr/bin/env python3
"""Compare bounded affine8 PLE row reads with MLX on the pinned Flash Next pack."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import struct
from typing import BinaryIO

import mlx.core as mx
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INVENTORY = (ROOT / "benchmarks/results/qualification"
             / "2026-10-04-flash-next-mxfp4-native-support/pack-inventory.json")
MARKER = ".ple.ple_embedding.ngram_embedding.shards."
AFFINE8 = {"bits": 8, "group_size": 32, "mode": "affine"}
LAYOUT = {"weight": ("U32", 40, 160), "scales": ("BF16", 5, 10), "biases": ("BF16", 5, 10)}


@dataclass
class ShardFile:
    handle: BinaryIO
    header: dict
    payload_start: int
    loaded: dict


def check_member(root: Path, name: str, members: dict) -> str:
    expected = members[name]
    with (root / name).open("rb") as handle:
        if os.fstat(handle.fileno()).st_size != expected["size"]:
            raise ValueError(f"publisher size differs for {name}")
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if digest != expected["sha256"]:
        raise ValueError(f"publisher SHA-256 differs for {name}")
    return digest


def read_rows(handle: BinaryIO, start: int, row_count: int, row_bytes: int,
              indices: list[int]) -> bytes:
    rows = []
    for index in indices:
        if not 0 <= index < row_count:
            raise IndexError(f"row {index} outside table shard of {row_count} rows")
        row = os.pread(handle.fileno(), row_bytes, start + index * row_bytes)
        if len(row) != row_bytes:
            raise ValueError(f"truncated packed row {index}")
        rows.append(row)
    return b"".join(rows)


def bfloat16_values(payload: bytes) -> np.ndarray:
    return (np.frombuffer(payload, dtype="<u2").astype("<u4") << 16).view("<f4")


def compare_shard(shard: int, prefix: str, files: dict[str, ShardFile]) -> dict:
    row_count = files["weight"].header[f"{prefix}.weight"]["shape"][0]
    interior = max(1, row_count * (shard + 1) // 129)
    requested = [row_count - 1, interior, 0, interior, 1]
    unique = sorted(set(requested))
    payload = {}
    for plane, (dtype, columns, row_bytes) in LAYOUT.items():
        file = files[plane]
        spec = file.header[f"{prefix}.{plane}"]
        start, end = spec["data_offsets"]
        if (spec["dtype"] != dtype or spec["shape"] != [row_count, columns]
                or start < 0 or end - start != row_count * row_bytes
                or file.payload_start + end > os.fstat(file.handle.fileno()).st_size):
            raise ValueError(f"invalid {plane} layout for PLE shard {shard}")
        payload[plane] = read_rows(file.handle, file.payload_start + start, row_count, row_bytes, unique)

    take = mx.array(requested, dtype=mx.int32)
    resident = mx.dequantize(*(files[key].loaded[f"{prefix}.{key}"][take] for key in LAYOUT),
                            group_size=32, bits=8)
    weights = np.frombuffer(payload["weight"], dtype="<u4").reshape(-1, 40)
    scales = bfloat16_values(payload["scales"]).reshape(-1, 5)
    biases = bfloat16_values(payload["biases"]).reshape(-1, 5)
    bounded = mx.dequantize(mx.array(weights), mx.array(scales, dtype=mx.bfloat16),
                           mx.array(biases, dtype=mx.bfloat16), group_size=32, bits=8)
    restore = [unique.index(index) for index in requested]
    bounded = bounded[mx.array(restore, dtype=mx.int32)]

    # A mathematically equivalent FP32 computation is a numerical negative control.
    u8 = weights.view(np.uint8).reshape(-1, 5, 32).astype(np.float32)
    analytic = (u8 * scales[..., None] + biases[..., None]).reshape(-1, 160)[restore]
    analytic = mx.array(analytic, dtype=mx.bfloat16)
    if resident.dtype != mx.bfloat16 or bounded.dtype != mx.bfloat16:
        raise ValueError(f"unexpected dequantized dtype for PLE shard {shard}")
    ref_bits, row_bits, analytic_bits = (
        np.array(array.view(mx.uint16), dtype="<u2") for array in (resident, bounded, analytic)
    )
    if not np.array_equal(ref_bits, row_bits):
        raise ValueError(f"bounded dequantization differs for PLE shard {shard}")
    return {
        "shard": shard, "row_count": row_count, "rows": requested,
        "bounded_payload_bytes": sum(map(len, payload.values())),
        "resident_sha256": hashlib.sha256(ref_bits.tobytes()).hexdigest(),
        "bounded_sha256": hashlib.sha256(row_bits.tobytes()).hexdigest(),
        "analytic_bit_differences": int(np.count_nonzero(ref_bits != analytic_bits)),
        "analytic_max_absolute_difference": float(np.max(np.abs(
            np.array(resident.astype(mx.float32)) - np.array(analytic.astype(mx.float32))
        ))),
    }


def probe(root: Path, selected: set[int]) -> dict:
    raw_inventory = INVENTORY.read_bytes()
    inventory = json.loads(raw_inventory)
    members = {entry["name"]: entry for entry in inventory["members"]}
    checked = {name: check_member(root, name, members)
               for name in ("config.json", "model.safetensors.index.json")}
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    config = json.loads((root / "config.json").read_text())
    quantization = config.get("quantization_config") or config.get("quantization", {})
    shards = {}
    for key in index:
        if MARKER in key and key.endswith(".weight"):
            shard = int(key.partition(MARKER)[2].split(".", 1)[0])
            if shard in shards:
                raise ValueError(f"duplicate PLE shard {shard}")
            shards[shard] = key.rsplit(".", 1)[0]
    if set(shards) != set(range(128)) or not selected <= shards.keys():
        raise ValueError("expected the pinned pack's 128 affine8 PLE shards")

    locations = {}
    last_use = {}
    for shard in sorted(selected):
        prefix = shards[shard]
        settings = quantization.get(prefix, quantization)
        if any(settings.get(key) != value for key, value in AFFINE8.items()):
            raise ValueError(f"unsupported PLE quantization for shard {shard}")
        names = {plane: index[f"{prefix}.{plane}"] for plane in LAYOUT}
        locations[shard] = names
        for name in names.values():
            last_use[name] = shard

    mx.set_default_device(mx.cpu)
    results = []
    opened = {}
    with ExitStack() as stack:
        for shard, names in locations.items():
            for name in sorted(set(names.values())):
                if name in opened:
                    continue
                checked[name] = check_member(root, name, members)
                path = root / name
                handle = stack.enter_context(path.open("rb"))
                size, = struct.unpack("<Q", handle.read(8))
                header = json.loads(handle.read(size))
                opened[name] = ShardFile(handle, header, 8 + size, mx.load(str(path)))
            results.append(compare_shard(shard, shards[shard], {plane: opened[name] for plane, name in names.items()}))
            for name in set(names.values()):
                if last_use[name] == shard:
                    opened.pop(name).handle.close()
                    mx.clear_cache()

    return {
        "pack_revision": inventory["revision"],
        "inventory_sha256": hashlib.sha256(raw_inventory).hexdigest(),
        "mlx": importlib.metadata.version("mlx"), "backend": "cpu",
        "members_sha256": checked, "table_shards": len(shards),
        "shards_with_separate_plane_files": sum(len(set(names.values())) > 1 for names in locations.values()),
        "decoded_values": sum(len(row["rows"]) for row in results) * 160,
        "bounded_payload_bytes": sum(row["bounded_payload_bytes"] for row in results),
        "shards": sorted(results, key=lambda row: row["shard"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--shard", type=int, action="append", help="limit the row probe to selected shards")
    args = parser.parse_args()
    report = probe(args.model_dir, set(args.shard or range(128)))
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
