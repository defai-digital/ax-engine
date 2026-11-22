"""Replay a guarded Flash Next sidecar mapping advice diagnostic."""

from gzip import decompress
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PRIOR = ROOT.parent / "flash-next-owned-positional-sidecar-m2-20261122/result.json"
NORMAL, SEQUENTIAL = "mmap", "mmap-sequential"
ORDER = [
    ("clone", NORMAL), ("clone", SEQUENTIAL), ("clone", SEQUENTIAL), ("clone", NORMAL),
    ("reused", SEQUENTIAL), ("reused", NORMAL), ("reused", NORMAL), ("reused", SEQUENTIAL),
]


def read(path):
    return json.loads(path.read_bytes())


def need(valid, message):
    if not valid:
        raise ValueError(message)


def digest(data):
    return sha256(data).hexdigest()


def replay():
    previous, result = read(PRIOR), read(ROOT / "result.json")
    build_path = ROOT / "build.json"
    build = read(build_path)
    sums = dict(line.split()[::-1] for line in (ROOT / "inputs.sha256").read_text().splitlines())
    need(result["scope"] == "M2 pinned MTP sidecar guarded mapping advice diagnostic"
         and (ROOT / "status.txt").read_text() == "0\n" and result["completed"] is True
         and result["binary_sha256"] == build["binary_sha256"] == sums["direct_weights"]
         and sums["build.json"] == digest(build_path.read_bytes())
         and build["mlx"] == build["mlx_metal"] == "0.32.4"
         and build["libmlx_sha256"] == "d8c9393d584e781393c3effccfccd8ba85ad0c70264edaf9f897dca9875375c6",
         "The staged diagnostic executable or result changed")
    for name, label in (("run_probe.py", "geometry_helper_sha256"),
                        ("compare_sidecar.py", "driver_sha256"), ("direct_weights.cpp", "reader_source_sha256")):
        data = decompress((ROOT / (name + ".gz")).read_bytes())
        need(digest(data) == build[label] == sums[name], f"Staged source changed: {name}")
    need(all(result[key] == previous[key] for key in
             ("revision", "inventory_sha256", "file_bytes", "publisher_sha256",
              "payload_sha256", "page_size", "hardware"))
         and result["inventory_sha256"] == sums["inventory.json"], "Publisher identity or test host changed")
    arms = result["arms"]
    need([(row["cache"], row["mode"]) for row in arms] == ORDER, "The mapping advice order changed")
    signatures = previous["arms"][0]
    for arm in arms:
        need(arm["tensors"] == 31 and arm["pressure_before"] == arm["pressure_after"] == 1
             and arm["paired_tensors"] == 0 and isfinite(arm["elapsed_seconds"]) and arm["elapsed_seconds"] > 0
             and all(arm[key] == signatures[key] for key in
                     ("selected_bytes", "allocated_bytes", "weights_sha256", "metal_sha256")),
             "Mapping advice changed payloads, Metal output or observed memory pressure")
        pages = arm["file_pages_before"]
        need(pages["selected_pages"] == 318256
             and 0 <= pages["resident_pages"] <= pages["selected_pages"], "Invalid page-residency sample")

    pairs = []
    for control_index, candidate_index in ((0, 1), (3, 2), (5, 4), (6, 7)):
        control, candidate = arms[control_index], arms[candidate_index]
        pages = [row["file_pages_before"]["resident_pages"] for row in (control, candidate)]
        comparable = max(pages) <= 1 if control["cache"] == "clone" else pages == [318256, 318256]
        pairs.append({
            "cache": control["cache"],
            "observed_page_residency_comparable": comparable,
            "resident_pages_before": pages,
            "ordinary_s": control["elapsed_seconds"],
            "sequential_s": candidate["elapsed_seconds"],
            "sequential_over_ordinary": candidate["elapsed_seconds"] / control["elapsed_seconds"] if comparable else None,
        })
    return {"scope": result["scope"], "qualification": False,
            "matching_tensors_per_arm": 31, "matching_selected_bytes_per_arm": signatures["selected_bytes"],
            "normal_pressure_boundaries": 2 * len(arms), "pairs": pairs}


if __name__ == "__main__":
    print(json.dumps(replay(), indent=2, allow_nan=False))
