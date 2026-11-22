"""Replay eager and bulk wired residency for the same mapped weight tensors."""
from gzip import decompress
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from statistics import median
import subprocess
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent
PRIOR = ROOT.parent / "flash-next-paired-resident-reads-m2-20261122"
FROZEN_SELECTION = ROOT.parent / "flash-next-resident-startup-attribution-m5-20261122/holdout-selection.json.gz"
PUBLISHER = ROOT.parent / "qualification/2026-10-04-flash-next-mxfp4-native-support/pack-inventory.json"
EARLY = "early-wired"
BULK = "bulk-wired"
MODES = (EARLY, BULK)
WIRED_BYTES = 8 * 1024**3
TENSORS = 158
SELECTION = [("model-00017-of-00025.safetensors", 5144009280),
             ("model-00018-of-00025.safetensors", 5144009280)]
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
    selection = decompress(FROZEN_SELECTION.read_bytes())
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
         and all(len(entry["names"]) == len(set(entry["names"])) == TENSORS for entry in files),
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
    need(len(arms) == 16, "Missing repeated wired-residency runs")
    analysis = {}
    for index, (name, num_bytes) in enumerate(SELECTION):
        rows = arms[index * 8:index * 8 + 8]
        expected_order = (EARLY, BULK, BULK, EARLY) if index == 0 else (BULK, EARLY, EARLY, BULK)
        need([row["mode"] for row in rows] == list(expected_order) * 2
             and [row["cache"] for row in rows] == ["reused"] * 4 + ["clone"] * 4,
             "Wiring or cache order changed")
        base = rows[0]
        identity = ("selected_bytes", "allocated_bytes", "tensors", "weights_sha256", "metal_sha256", "unaligned_source_tensors")
        need(all(row["shard"] == name and row["publisher_sha256"] == publishers[name]
                 and row["selected_bytes"] == num_bytes and row["tensors"] == TENSORS
                 and row["pressure_before"] == row["pressure_after"] == 1
                 and row["wired_budget_bytes"] == WIRED_BYTES
                 and all(isfinite(row[key]) and row[key] > 0 for key in
                         ("elapsed_seconds", "copy_seconds", "set_wired_limit_seconds"))
                 and abs(row["elapsed_seconds"] - row["copy_seconds"] - row["set_wired_limit_seconds"]) < 0.00001
                 and all(row[key] == base[key] for key in identity) for row in rows),
                 "Reader bytes, Metal output, allocation, pressure, or combined wiring time changed")
        cache_results = {}
        for cache, series in (("reused", rows[:4]), ("clone", rows[4:])):
            before = [row["file_pages_before"] for row in series]
            eligible = all(0 == row["resident_pages"] if cache == "clone"
                           else row["resident_pages"] == row["selected_pages"] for row in before)
            medians = {mode: median(row["elapsed_seconds"] for row in series if row["mode"] == mode) for mode in MODES}
            pairs = [{row["mode"]: row["elapsed_seconds"] for row in series[i:i + 2]} for i in (0, 2)]
            cache_results[cache] = {"attested_initial_page_state": eligible,
                                    "initial_resident_pages": [row["resident_pages"] for row in before],
                                    "median_seconds": medians,
                                    "median_copy_seconds": {mode: median(row["copy_seconds"]
                                        for row in series if row["mode"] == mode) for mode in MODES},
                                    "median_setting_seconds": {mode: median(row["set_wired_limit_seconds"]
                                        for row in series if row["mode"] == mode) for mode in MODES},
                                    "bulk_median_difference_percent": percent(medians[EARLY], medians[BULK]),
                                    "per_order_difference_percent": [percent(pair[EARLY], pair[BULK]) for pair in pairs]}
        analysis[name] = {"weights_sha256": base["weights_sha256"], "metal_sha256": base["metal_sha256"],
                          "selected_bytes": num_bytes, "cache_conditions": cache_results}
    return {"scope": "M2 Ultra mapped load plus Metal wiring, component-only", "qualification": False,
            "native_control_runs": len(arms), "matching_payload_observations": len(arms) * TENSORS,
            "wired_budget_bytes": WIRED_BYTES, "shards": analysis}


if __name__ == "__main__":
    print(json.dumps(replay(), indent=2, allow_nan=False))
