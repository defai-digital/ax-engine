#!/usr/bin/env python3
"""Check Flash Next MXFP4 projection with the six retained layer-zero inputs."""

from argparse import ArgumentParser
from hashlib import file_digest, sha256
from importlib.metadata import version
import json
from pathlib import Path
import struct

import mlx.core as mx
import numpy as np

REVISION = "ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3"
NAME = "language_model.model.layers.0.linear_attn.in_proj_a"
POSITIONS = (493, 494, 495, 1039, 1040, 1041)
PARTITION = (512, 512, 512, 511)


def digest(path):
    with path.open("rb") as source:
        return file_digest(source, "sha256").hexdigest()


def projection_weights(snapshot):
    tables = [json.loads(p.read_text()).get("weight_map", {}) for p in snapshot.glob("*.index.json")]
    file = snapshot / next(table[f"{NAME}.weight"] for table in tables if f"{NAME}.weight" in table)
    tensors, descriptors = {}, {}
    with file.open("rb") as source:
        header_size = struct.unpack("<Q", source.read(8))[0]
        header = json.loads(source.read(header_size))
        for name, (dtype, shape, np_type) in {
            "weight": ("U32", [48, 320], "<u4"),
            "scales": ("U8", [48, 80], "u1"),
        }.items():
            entry = header[f"{NAME}.{name}"]
            assert entry["shape"] == shape and entry["dtype"] == dtype
            start, end = entry["data_offsets"]
            source.seek(8 + header_size + start)
            body = source.read(end - start)
            tensors[name] = mx.array(np.frombuffer(body, dtype=np_type).reshape(shape))
            descriptors[name] = {"dtype": dtype, "shape": shape, "sha256": sha256(body).hexdigest()}
    return file, tensors, descriptors


def read_inputs(observed):
    records, expectations = {}, {}
    for width in ("512", "2048"):
        approved = json.loads((observed / width / "receipt.json").read_text())
        expectations[width] = {
            item["position"]: item["bits"] for item in json.loads(
                (observed / width / "result/recurrence.json").read_text()
            )["decay_stages"]["projection"]["witnesses"]
        }
        records[width] = {}
        for file, expected in approved["projection_input_probes"].items():
            data = (observed / width / "operator" / file).read_bytes()
            assert sha256(data).hexdigest() == expected
            if file.endswith(".f32le"):
                position = int(file.rsplit("-pos-", 1)[1].split(".", 1)[0])
                records[width][position] = data
    assert records["512"] == records["2048"]
    assert sorted(records["512"]) == list(POSITIONS)
    return records["512"], expectations


def positions(output):
    assert output.dtype == mx.bfloat16
    result = mx.take(output, mx.array(POSITIONS), axis=1).astype(mx.float32)
    mx.eval(result)
    return {position: row.tolist() for position, row in zip(
        POSITIONS, np.asarray(result[0]).view(np.uint32), strict=True
    )}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("observed", type=Path)
    parser.add_argument("inventory", type=Path)
    args = parser.parse_args()
    assert version("mlx") == version("mlx-metal") == "0.32.4"
    assert args.snapshot.name == REVISION
    path, tensors, descriptors = projection_weights(args.snapshot)
    file_sha256 = digest(path)
    inventory = json.loads(args.inventory.read_text())
    assert inventory["revision"] == REVISION
    member = next(entry for entry in inventory["members"] if entry["name"] == path.name)
    assert path.stat().st_size == member["size"]
    assert file_sha256 == member["publisher_sha256"] == member["sha256"]
    records, expected = read_inputs(args.observed)
    inputs = np.zeros((1, 2048, 2560), dtype=np.float32)
    for index, data in records.items():
        inputs[0, index] = np.frombuffer(data, dtype="<f4")
    inputs = mx.array(inputs).astype(mx.bfloat16)
    mx.eval(inputs, *tensors.values())

    def project(x):
        return mx.quantized_matmul(
            x, tensors["weight"], tensors["scales"],
            transpose=True, group_size=32, bits=4, mode="mxfp4",
        )

    def tiled():
        parts, start = [], 0
        for count in PARTITION:
            parts.append(project(mx.contiguous(inputs[:, start:start + count, :])))
            start += count
        assert start == 2047
        return mx.concatenate(parts, axis=1)

    variants = {
        "whole_2047": positions(project(mx.contiguous(inputs[:, :2047, :]))),
        "padded_2048": positions(project(inputs)),
        "canonical_tiles": positions(tiled()),
    }
    print(json.dumps({
        "mlx": version("mlx"),
        "pack_revision": REVISION,
        "shard_sha256": file_sha256,
        "tensors": descriptors,
        "partition": PARTITION,
        "verified_positions": POSITIONS,
        "input_row_sha256": {i: sha256(r).hexdigest() for i, r in records.items()},
        "output_dtype": "BF16",
        "layout_comparisons": {
            name: {
                "positions_different_from_512": [
                    i for i in POSITIONS if rows[i] != expected["512"][i]
                ],
                "positions_different_from_2048": [
                    i for i in POSITIONS if rows[i] != expected["2048"][i]
                ],
                "output_row_sha256": {
                    i: sha256(struct.pack(f"<{len(row)}I", *row)).hexdigest()
                    for i, row in rows.items()
                },
            } for name, rows in variants.items()
        },
    }, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
