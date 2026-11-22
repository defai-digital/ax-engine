"""Replay targeted mapped-cache release with exact MLX and restart controls."""

from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from statistics import median
import sys

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / "flash-next-page-aligned-reader-m2-20261122"
INVENTORY_SHA256 = "97b890234af731ac6e07d2ed7dbe32b4b0fcb1f0aff01addaaa340a838b28cc6"
LIBMLX_SHA256 = "d8c9393d584e781393c3effccfccd8ba85ad0c70264edaf9f897dca9875375c6"
MODES = ("mmap", "mmap-reclaim-shared")
IDENTITY = ("weights_sha256", "metal_sha256", "allocated_bytes", "selected_bytes", "tensors")


def require(condition, why):
    if not condition:
        raise ValueError(why)


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def checked_synthetic(synthetic):
    expected = {
        "mmap": [1, 4],
        "mmap-reclaim-private": [1, 4],
        "mmap-reclaim-shared": [0, 4],
    }
    reference = synthetic["arms"]["mmap"]["arrays"]
    require(synthetic["status"] == "complete"
            and synthetic["targeted_release_observed"] == {name: name == MODES[1] for name in expected}
            and set(synthetic["arms"]) == set(expected), "Synthetic page-release modes changed")
    for name, arm in synthetic["arms"].items():
        arrays = arm["arrays"]
        require(arm["pages_before"] == [1, 4] and arm["pages_after"] == expected[name]
                and arrays["reclaim_errno"] == 0 and arrays["requested_invalidate_pages"] == int(name != "mmap")
                and all(arrays[key] == reference[key] for key in IDENTITY),
                "The synthetic copied bytes, target page or adjacent pages changed")
    return expected


def timed(observation):
    total, reclaim = (observation[k] for k in ("elapsed_seconds", "reclaim_seconds"))
    require(isfinite(total) and isfinite(reclaim) and total > 0 and 0 <= reclaim <= total,
            "Invalid initial read, cache-release or restart time")
    return total


def validate_arm(record, expected, shard, condition, trial, mode):
    initial, restart = record["initial_read"], record["mapped_restart"]
    for readback in (initial, restart):
        require(all(readback[key] == expected[key] for key in IDENTITY)
                and readback["reclaim_errno"] == 0 and readback["staging_bytes"] == 0
                and readback["nocache"] is False, "Native payload, Metal output or allocations changed")
        timed(readback)
    require(record["shard"] == shard and record["condition"] == condition and record["trial"] == trial
            and record["pressure"] == 1 and initial["mode"] == mode and restart["mode"] == MODES[0],
            "Counterbalanced order, selected shard or normal pressure changed")
    before, after, post_restart = (record[key] for key in ("before", "after", "restart_after"))
    selected, full = before["selected_pages"], record["eligible_pages"]
    reclaimed = (mode == MODES[1])
    require(full > 0 and selected >= full
            and before["resident_pages"] == (0 if condition == "nonresident" else selected)
            and all(row["selected_pages"] == selected for row in (after, post_restart))
            and initial["requested_invalidate_pages"] == (full if reclaimed else 0)
            and restart["requested_invalidate_pages"] == restart["reclaim_seconds"] == 0
            and post_restart["resident_pages"] == selected
            and after["resident_pages"] == (selected - full if reclaimed else selected),
            "Observed page accounting did not match the selected full-page hint")


def summarize(reports):
    return {
        "median_initial_read_seconds": {mode: median(row["initial_read"]["elapsed_seconds"] for row in rows)
                                        for mode, rows in reports.items()},
        "median_mapped_restart_seconds": {mode: median(row["mapped_restart"]["elapsed_seconds"] for row in rows)
                                           for mode, rows in reports.items()},
        "median_release_seconds": median(row["initial_read"]["reclaim_seconds"] for row in reports[MODES[1]]),
        "resident_pages_after": {mode: [row["after"]["resident_pages"] for row in rows]
                                 for mode, rows in reports.items()},
    }


def replay(root=ROOT, previous=PREVIOUS):
    record, build, synthetic = (read(root / name) for name in ("result.json", "build.json", "synthetic.json"))
    measured_inputs = {
        "direct_weights.cpp": "reader_source_sha256", "base_probe.py": "base_probe_sha256",
        "run_reclaim.py": "runner_sha256",
    }
    for name, field in measured_inputs.items():
        require(build[field] == digest(root / name), f"Measured input changed: {name}")
    require(build["mlx"] == build["mlx_metal"] == "0.32.4"
            and build["libmlx_sha256"] == LIBMLX_SHA256 and record["build"] == build,
            "The measured MLX runtime or binary identity changed")
    synthetic_after = checked_synthetic(synthetic)
    before = read(previous / "result.json")
    require((root / "status.txt").read_text() == "0\n" and record["status"] == "complete"
            and record["scope"] == "M2 Ultra selected mapped cache release" and record["qualification"] is False
            and record["hardware"]["memory_bytes"] == 192 * 1024**3
            and record["hardware"]["soc"] == before["hardware"]["soc"] == "Apple M2 Ultra"
            and record["hardware"]["storage"] == "external SSD",
            "The component run did not complete in the intended scope")
    require(record["page_bytes"] == before["page_size"] == 16384
            and record["inventory_sha256"] == before["inventory_sha256"] == INVENTORY_SHA256
            and record["selection_sha256"] == before["selection_sha256"] == digest(root / "selection.json")
            and record["synthetic_sha256"] == digest(root / "synthetic.json"),
            "The selection, publisher binding or synthetic input changed")
    shards = sorted({row["shard"] for row in before["arms"]})
    require(len(shards) == 2 and len(record["arms"]) == 16, "The two-shard comparison is incomplete")

    summaries = {}
    current = iter(record["arms"])
    for shard_index, shard in enumerate(shards):
        oracle = next(row for row in before["arms"] if row["shard"] == shard and row["mode"] == "mmap")
        order = MODES if shard_index == 0 else MODES[::-1]
        summaries[shard] = {}
        for condition in ("nonresident", "primed"):
            measured = {mode: [] for mode in MODES}
            for trial, mode in enumerate((*order, *order[::-1]), 1):
                arm = next(current)
                validate_arm(arm, oracle, shard, condition, trial, mode)
                measured[mode].append(arm)
            summaries[shard][condition] = summarize(measured)
    return {
        "scope": record["scope"], "qualification": False, "hardware": record["hardware"],
        "measured_runs": 16, "matched_mapped_restart_runs": 16,
        "cpu_and_metal_matches": sum(row["initial_read"]["tensors"] + row["mapped_restart"]["tensors"]
                                     for row in record["arms"]),
        "synthetic_resident_pages_after": synthetic_after,
        "shards": summaries,
    }


if __name__ == "__main__":
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT
    print(json.dumps(replay(root), indent=2, sort_keys=True, allow_nan=False))
