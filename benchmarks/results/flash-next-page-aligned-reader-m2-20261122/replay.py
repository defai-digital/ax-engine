"""Replay native page-aligned resident weight I/O on Darwin."""
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from statistics import median
import subprocess
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent
PRIOR = ROOT.parent / "flash-next-paired-resident-reads-m2-20261122"
FROZEN_SELECTION = ROOT.parent / "flash-next-parallel-mmap-copy-m2-20261122/selection.json"
PUBLISHER = ROOT.parent / "qualification/2026-10-04-flash-next-mxfp4-native-support/pack-inventory.json"
BASE = "mmap"
BUFFERED = "aligned-buffered"
NOCACHE = "aligned-nocache"
MODES = (BASE, BUFFERED, NOCACHE)
SELECTION = [("model-00017-of-00025.safetensors", 1677721600),
             ("model-00018-of-00025.safetensors", 1677721600)]
PIN = "ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3"
INVENTORY = "97b890234af731ac6e07d2ed7dbe32b4b0fcb1f0aff01addaaa340a838b28cc6"


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def read(name):
    return json.loads((ROOT / name).read_bytes())


def digest(name):
    return sha256((ROOT / name).read_bytes()).hexdigest()


def percent(before, after):
    return round((after / before - 1) * 100, 3)


def checked_inputs(manifest, build):
    need(manifest.pop("direct_weights") == build["binary_sha256"], "Probe binary checksum changed")
    inventory = PUBLISHER.read_bytes()
    selection = FROZEN_SELECTION.read_bytes()
    need(sha256(selection).hexdigest() == manifest.pop("selection.json"), "Pinned startup-sensitive selection changed")
    need(sha256(inventory).hexdigest() == manifest.pop("inventory.json") == INVENTORY, "Pinned inventory changed")
    with TemporaryDirectory() as directory:
        for name in ("direct_weights.cpp", "run_probe.py"):
            with (ROOT / f"{name}.patch").open("rb") as patch:
                subprocess.run(["patch", "-s", "-o", name, str(PRIOR / name)], cwd=directory, stdin=patch, check=True)
            data = (Path(directory) / name).read_bytes()
            need(sha256(data).hexdigest() == manifest.pop(name), f"Reconstructed input changed: {name}")
    need(all(digest(name) == checksum for name, checksum in manifest.items()), "Staged input changed")
    return json.loads(inventory), json.loads(selection)


def replay():
    manifest = {name: checksum for checksum, name in (line.split() for line in
                (ROOT / "inputs.sha256").read_text().splitlines())}
    build = read("build.json")
    inventory, selection = checked_inputs(dict(manifest), build)
    need(build["reader_source_sha256"] == manifest["direct_weights.cpp"]
         and build["driver_sha256"] == manifest["run_probe.py"]
         and build["mlx"] == build["mlx_metal"] == "0.32.4", "Wrong executable or native MLX wheel")
    need(inventory["revision"] == PIN, "Pinned inventory revision changed")
    publishers = {row["name"]: row["publisher_sha256"] for row in inventory["members"]}
    files = selection["files"]
    need([(entry["name"], entry["selected_bytes"]) for entry in files] == SELECTION
         and all(len(entry["names"]) == len(set(entry["names"])) == 4 for entry in files),
         "Unexpected startup-sensitive shard controls")
    result = read("result.json")
    need((ROOT / "status.txt").read_text() == "0\n" and result["completed"] is True
         and result["short_read_rejected"] is True and result["qualification"] is False
         and result["revision"] == PIN and result["inventory_sha256"] == INVENTORY
         and result["selection_sha256"] == manifest["selection.json"]
         and result["binary_sha256"] == build["binary_sha256"]
         and result["scope"] == "functional" and result["page_size"] == 16384
         and result["hardware"]["soc"] == "Apple M2 Ultra" and result["hardware"]["memory_bytes"] == 192 * 1024**3,
         "Incomplete or mismatched M2 diagnostic")
    arms = result["arms"]
    need(len(arms) == 24, "Missing repeated page-aligned read runs")
    analysis = {}
    for index, (name, num_bytes) in enumerate(SELECTION):
        rows = arms[index * 12:index * 12 + 12]
        order = MODES if index == 0 else MODES[::-1]
        need([row["mode"] for row in rows] == list((*order, *order[::-1])) * 2
             and [row["cache"] for row in rows] == ["clone"] * 6 + ["reused"] * 6,
             "Reader mode or cache order changed")
        first = rows[0]
        same = ("selected_bytes", "allocated_bytes", "tensors", "weights_sha256", "metal_sha256", "unaligned_source_tensors")
        need(all(row["shard"] == name and row["publisher_sha256"] == publishers[name]
                 and row["selected_bytes"] == num_bytes and row["tensors"] == 4
                 and row["pressure_before"] == row["pressure_after"] == 1
                 and row["nocache"] is (row["mode"] == NOCACHE)
                 and row["staging_bytes"] == (0 if row["mode"] == BASE else 4 * 1024**2)
                 and isfinite(row["elapsed_seconds"]) and row["elapsed_seconds"] > 0
                 and all(row[key] == first[key] for key in same) for row in rows),
                 "Reader output, allocation, selected bytes, staging, or host pressure changed")
        count = first["file_pages_before"]["selected_pages"]
        need(type(count) is int and count > 0, "Missing sampled selected-page geometry")
        cache_results = {}
        for cache, series in (("clone", rows[:6]), ("reused", rows[6:])):
            before = [row["file_pages_before"] for row in series]
            after = [row["file_pages_after"] for row in series]
            need(all(page["selected_pages"] == count for page in before + after)
                 and all(page["resident_pages"] == (0 if cache == "clone" else count) for page in before)
                 and all(page["resident_pages"] == (0 if cache == "clone" and row["mode"] == NOCACHE else count)
                         for row, page in zip(series, after, strict=True)),
                 "Initial residency or the observed F_NOCACHE page effect changed")
            medians = {mode: median(row["elapsed_seconds"] for row in series if row["mode"] == mode) for mode in MODES}
            pairs = [{row["mode"]: row["elapsed_seconds"] for row in series[i:i + 3]} for i in (0, 3)]
            cache_results[cache] = {"selected_pages": count,
                                    "initial_resident_pages": [row["resident_pages"] for row in before],
                                    "final_resident_pages": {mode: [page["resident_pages"] for row, page in
                                        zip(series, after, strict=True) if row["mode"] == mode] for mode in MODES},
                                    "median_seconds": medians,
                                    "nocache_difference_percent": percent(medians[BASE], medians[NOCACHE]),
                                    "nocache_per_order_difference_percent": [percent(pair[BASE], pair[NOCACHE]) for pair in pairs]}
        analysis[name] = {"weights_sha256": first["weights_sha256"], "metal_sha256": first["metal_sha256"],
                          "selected_bytes": num_bytes, "cache_conditions": cache_results}
    tail = read("unaligned-tail.json")
    need(tail["binary_sha256"] == build["binary_sha256"] and tail["out_of_range_rejected"] is True
         and tail["source_offset"] == 39 and set(tail["cases"]) == set(MODES), "Unbound unaligned and EOF control")
    base = tail["cases"][BASE]
    need(all(row["tensors"] == 1 and row["selected_bytes"] == 4 * 1024**2 + 257
             and row["mode"] == mode and row["nocache"] is (mode == NOCACHE)
             and all(base[key] == row[key] for key in ("allocated_bytes", "weights_sha256", "metal_sha256"))
             for mode, row in tail["cases"].items()), "Unaligned tail or post-read Metal check changed")
    return {"scope": "M2 Ultra Darwin reader component only", "qualification": False,
            "native_control_runs": len(arms), "matching_payload_observations": len(arms) * 4,
            "unaligned_tail_bytes": base["selected_bytes"], "shards": analysis}


if __name__ == "__main__":
    print(json.dumps(replay(), indent=2, allow_nan=False))
