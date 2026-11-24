#!/usr/bin/env python3
"""Replay the exact Flash Next layer-zero affine8 router on retained input rows."""

from argparse import ArgumentParser
from hashlib import file_digest, sha256
from importlib.metadata import version
import json
from pathlib import Path
import struct

import mlx.core as mx
import numpy as np

PREFIX = "language_model.model.layers.0.mlp.gate."
POSITIONS = (0, 457, 458, 459, 1333, 1334, 1335, 1900, 1901, 1902, 2046)
ARMS = ("narrow-incoming-bridge", "wide-tiling-incoming-bridge")
PARTITION = (512, 512, 512, 511)
PIN = "ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3"


def weights(snapshot, inventory):
    tables = [json.loads(p.read_bytes())["weight_map"] for p in snapshot.glob("*.index.json")]
    expected = {entry["name"]: entry for entry in inventory["members"]}
    members, descriptors, tensors = {}, {}, {}
    for variable, dtype, shape in (
        ("weight", "U32", [512, 640]), ("scales", "BF16", [512, 80]), ("biases", "BF16", [512, 80]),
    ):
        name = PREFIX + variable
        file = next(table[name] for table in tables if name in table)
        path = snapshot / file
        if file not in members:
            with path.open("rb") as fp:
                digest = file_digest(fp, "sha256").hexdigest()
            item = expected[file]
            assert digest == item["publisher_sha256"] == item["sha256"]
            assert path.stat().st_size == item["size"]
            members[file] = digest
        with path.open("rb") as fp:
            text_size = struct.unpack("<Q", fp.read(8))[0]
            entry = json.loads(fp.read(text_size))[name]
            assert entry["dtype"] == dtype and entry["shape"] == shape
            start, end = entry["data_offsets"]
            fp.seek(8 + text_size + start)
            data = fp.read(end - start)
        bits = mx.array(np.frombuffer(data, dtype="<u4" if dtype == "U32" else "<u2").reshape(shape))
        tensors[variable] = bits if dtype == "U32" else mx.view(bits, mx.bfloat16)
        descriptors[variable] = {"dtype": dtype, "shape": shape, "sha256": sha256(data).hexdigest()}
    return tensors, descriptors, members


def probes(observed, name):
    base = observed / name
    receipt = json.loads((base/"receipt.json").read_bytes())
    sealed = receipt["operator_sha256"]
    by_stage = {"mlp_hc_read": {}, "moe_router_logits": {}}
    for item, digest in sealed.items():
        if not item.endswith(".json"):
            continue
        entry = (base/"operator"/item).read_bytes()
        assert sha256(entry).hexdigest() == digest
        meta = json.loads(entry)
        stage, pos = meta["stage"], meta["position"]
        if stage not in by_stage:
            continue
        raw = item.removesuffix(".json")+".f32le"
        body = (base/"operator"/raw).read_bytes()
        assert sha256(body).hexdigest() == sealed[raw]
        by_stage[stage][pos] = body
    assert all(tuple(sorted(rows)) == POSITIONS for rows in by_stage.values())
    return by_stage["mlp_hc_read"], by_stage["moe_router_logits"]


def outputs(x):
    assert x.dtype == mx.bfloat16
    retained = mx.take(x, mx.array(POSITIONS), axis=1).astype(mx.float32)
    mx.eval(retained)
    return {pos: np.asarray(retained[0, index]).astype("<f4", copy=False).tobytes()
            for index, pos in enumerate(POSITIONS)}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("observed", type=Path)
    parser.add_argument("inventory", type=Path)
    args = parser.parse_args()
    assert version("mlx") == version("mlx-metal") == "0.32.4"
    inventory = json.loads(args.inventory.read_bytes())
    assert args.snapshot.name == inventory["revision"] == PIN
    tensors, descriptors, files = weights(args.snapshot, inventory)
    (narrow, expected_n), (wide, expected_w) = [probes(args.observed, name) for name in ARMS]
    assert narrow == wide
    raw = np.zeros((1, 2048, 2560), dtype=np.float32)
    for pos,row in narrow.items():
        raw[0,pos] = np.frombuffer(row, dtype="<f4")
    x = mx.array(raw).astype(mx.bfloat16)
    mx.eval(x,*tensors.values())

    def projection(data):
        return mx.quantized_matmul(data, tensors["weight"], tensors["scales"], tensors["biases"],
                                  transpose=True, group_size=32, bits=8, mode="affine")

    start, pieces = 0, []
    for length in PARTITION:
        pieces.append(projection(mx.contiguous(x[:,start:start+length,:])))
        start += length
    assert start == 2047
    variants = {
        "whole_2047": outputs(projection(mx.contiguous(x[:,:2047,:]))),
        "padded_2048": outputs(projection(x)),
        "canonical_tiles": outputs(mx.concatenate(pieces,axis=1)),
    }
    digest_rows=lambda rows:{pos:sha256(value).hexdigest() for pos,value in rows.items()}
    print(json.dumps({"pack_revision":PIN,"mlx":version("mlx"),"mode":"affine8/group32",
        "verified_positions":POSITIONS,"partition":PARTITION,
        "shards":files,"tensors":descriptors,"input_rows":digest_rows(narrow),
        "archived_narrow_rows":digest_rows(expected_n),"archived_wide_rows":digest_rows(expected_w),
        "variants":{name:{
            "output_rows":digest_rows(result),
            "different_from_narrow":[p for p in POSITIONS if result[p]!=expected_n[p]],
            "different_from_wide":[p for p in POSITIONS if result[p]!=expected_w[p]],
        } for name,result in variants.items()},
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
