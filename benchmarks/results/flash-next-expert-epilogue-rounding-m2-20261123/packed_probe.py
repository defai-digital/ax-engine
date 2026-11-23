"""Measure bounded real MXFP4 routed gate/up packing at Flash Next shapes."""
from argparse import ArgumentParser
from collections import defaultdict
from hashlib import file_digest, sha256
from importlib.metadata import version
import json
import math
from pathlib import Path
from statistics import median
import struct
from time import perf_counter

import mlx.core as mx
import numpy as np

REVISION = "ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3"
LAYERS = (0, 32)
EXPERTS, HIDDEN, INNER, TOP_K = 512, 2560, 640, 10
PAIR_ORDER = ("split", "packed", "scratch", "scratch", "packed", "split") * 3
FORMATS = {"U32": np.dtype("<u4"), "U8": np.dtype("u1")}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def tensor_names(layer):
    stem = f"language_model.model.layers.{layer}.mlp.switch_mlp"
    return [f"{stem}.{part}_proj.{role}" for part in ("gate", "up", "down")
            for role in ("weight", "scales")]


def read_tensor(pack, index, name, headers):
    shard = index[name]
    if shard not in headers:
        with (pack / shard).open("rb") as stream:
            size, = struct.unpack("<Q", stream.read(8))
            require(0 < size < 16 * 1024 * 1024, "Unexpected safetensors header length")
            headers[shard] = json.loads(stream.read(size)), 8 + size
    fields, payload = headers[shard]
    row = fields[name]
    shape, dtype = tuple(row["shape"]), FORMATS[row["dtype"]]
    start, end = row["data_offsets"]
    inbound, outbound = (INNER, HIDDEN) if ".down_proj." in name else (HIDDEN, INNER)
    is_payload = name.endswith(".weight")
    expected_shape = (EXPERTS, outbound, inbound // (8 if is_payload else 32))
    require(shape == expected_shape and dtype == FORMATS["U32" if is_payload else "U8"]
            and end - start == math.prod(shape) * dtype.itemsize
            and 0 <= start < end <= (pack / shard).stat().st_size - payload,
            f"Unexpected Flash Next MXFP4 geometry for {name}")
    array = np.memmap(pack / shard, offset=payload + start, shape=shape, dtype=dtype, mode="r")
    owned = mx.array(array)
    mx.eval(owned)
    return owned


def packed_pair(gate, up):
    return tuple(mx.concatenate((a, b), axis=1) for a, b in zip(gate, up, strict=True))


def inputs(rows, pool):
    values = np.random.default_rng(rows + pool).standard_normal((rows, HIDDEN), dtype=np.float32)
    x = mx.array(values).astype(mx.bfloat16).reshape(rows, 1, HIDDEN)
    row = np.arange(rows, dtype=np.uint32)[:, None]
    route = np.arange(TOP_K, dtype=np.uint32)[None, :]
    choices = ((17 * row + 13 * route + 11) % pool).astype(np.uint32)
    if rows == 1:
        x = x.reshape(1, 1, 1, 1, HIDDEN)
        indices = mx.array(choices.reshape(1, 1, TOP_K))
    else:
        order = np.argsort(choices.reshape(-1), kind="stable")
        indices = mx.array(choices.reshape(-1)[order])
        x = x[mx.array(order // TOP_K)]
    mx.eval(x, indices)
    return x, indices, len(np.unique(choices))


def gather(x, pair, indices, sorted_indices):
    return mx.gather_qmm(x, *pair, None, rhs_indices=indices, transpose=True,
                        group_size=32, bits=4, mode="mxfp4", sorted_indices=sorted_indices)


def same_bits(a, b):
    require(a.shape == b.shape and a.dtype == b.dtype == mx.bfloat16, "Activation geometry changed")
    return bool(mx.all(a.view(mx.uint16) == b.view(mx.uint16)).item())


def compare_case(gate, up, down, packed, rows, pool):
    x, indices, selected = inputs(rows, pool)
    sorted_indices = rows != 1

    def project(mode):
        if mode == "split":
            return gather(x, gate, indices, sorted_indices), gather(x, up, indices, sorted_indices)
        weights = packed if mode == "packed" else packed_pair(gate, up)
        both = gather(x, weights, indices, sorted_indices)
        return both[..., :INNER], both[..., INNER:]

    @mx.compile
    def activation(g, u):
        g32 = g.astype(mx.float32)
        return (g32 * mx.sigmoid(g32)).astype(g.dtype) * u

    def execute(mode):
        gate_rows, up_rows = project(mode)
        activated = activation(gate_rows, up_rows)
        down_rows = gather(activated, down, indices, sorted_indices)
        return gate_rows, up_rows, activated, down_rows

    control = execute("split")
    checks = {}
    for mode in ("packed", "scratch"):
        candidate = execute(mode)
        labels = ("gate", "up", "activation", "down")
        checks[mode] = dict(zip(labels, (same_bits(a, b) for a, b in
                                        zip(control, candidate, strict=True)), strict=True))
    del control, candidate

    timings = defaultdict(list)
    peak = defaultdict(list)
    for mode in PAIR_ORDER:
        mx.synchronize()
        mx.reset_peak_memory()
        baseline = mx.get_active_memory()
        start = perf_counter()
        gate_rows, up_rows = project(mode)
        y = gather(activation(gate_rows, up_rows), down, indices, sorted_indices)
        mx.eval(y)
        timings[mode].append(perf_counter() - start)
        peak[mode].append(max(0, mx.get_peak_memory() - baseline))
        del y, gate_rows, up_rows
    return {"rows": rows, "route_pool": pool, "distinct_experts": selected,
            "sorted_indices": sorted_indices, "bit_identity": checks,
            "elapsed_seconds": timings, "median_seconds": {mode: median(v) for mode, v in timings.items()},
            "peak_additional_bytes": {mode: max(v) for mode, v in peak.items()}}


def main():
    parser = ArgumentParser(description=__doc__)
    for name in ("pack", "inventory", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    versions = {package: version(package) for package in ("mlx", "mlx-metal", "numpy")}
    require(versions["mlx"] == versions["mlx-metal"] == "0.32.4", "MLX wheel differs from the pin")
    mx.set_default_device(mx.gpu)
    inventory = json.loads(args.inventory.read_bytes())
    require(inventory["revision"] == REVISION and len(inventory["members"]) == 49,
            "Publisher inventory differs from the pin")
    members = {row["name"]: row for row in inventory["members"]}
    index_path = args.pack / "model.safetensors.index.json"
    require(sha(index_path) == members[index_path.name]["sha256"], "Publisher index digest changed")
    index = json.loads(index_path.read_bytes())["weight_map"]
    shard_ids = sorted({index[name] for layer in LAYERS for name in tensor_names(layer)})
    for shard in shard_ids:
        path = args.pack / shard
        member = members[shard]
        require(path.stat().st_size == member["size"] and sha(path) == member["sha256"]
                and member["publisher_sha256"] == member["sha256"], f"Publisher shard differs: {shard}")
    report = {"scope": "One-layer resident MXFP4 routed expert probe using synthetic activations",
              "qualification": False, "versions": versions, "pack_revision": REVISION,
              "inventory_sha256": sha(args.inventory), "probe_sha256": sha(Path(__file__)),
              "publisher_shards_verified": shard_ids, "layer_results": []}
    for layer in LAYERS:
        headers = {}
        gate_w, gate_s, up_w, up_s, down_w, down_s = (
            read_tensor(args.pack, index, name, headers) for name in tensor_names(layer))
        gate, up, down = (gate_w, gate_s), (up_w, up_s), (down_w, down_s)
        mx.synchronize()
        started = perf_counter()
        packed = packed_pair(gate, up)
        mx.eval(*packed)
        pack_seconds = perf_counter() - started
        matrix = [compare_case(gate, up, down, packed, rows, pool)
                  for rows, pool in ((1, EXPERTS), (511, EXPERTS), (512, EXPERTS), (512, 64))]
        report["layer_results"].append({"layer": layer, "pack_seconds": pack_seconds,
                                        "packed_bytes": sum(t.nbytes for t in packed), "cases": matrix})
        del gate, up, down, packed, gate_w, gate_s, up_w, up_s, down_w, down_s
        mx.clear_cache()
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
