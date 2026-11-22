"""Recompute the two Flash Next resident-shard read orders on M5 Max."""

from hashlib import file_digest
import json
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent
PRIOR = ROOT.parent / "flash-next-paired-resident-reads-m2-20261122"
MATCH = ("publisher_sha256", "tensors", "selected_bytes", "allocated_bytes",
         "weights_sha256", "metal_sha256", "unaligned_source_tensors", "cache", "mode", "paired_tensors")


def load(path):
    return json.loads(path.read_bytes())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with path.open("rb") as f:
        return file_digest(f, "sha256").hexdigest()


def replay(root, prior):
    result = load(root / "result.json")
    baseline = load(prior / "result-m2.json")
    build = load(prior / "build.json")
    require((root / "status.txt").read_text() == "0\n" and result["scope"] == "m5-diagnostic"
            and result["completed"] and result["short_read_rejected"] and not result["qualification"]
            and result["hardware"]["soc"] == "Apple M5 Max"
            and result["hardware"]["memory_bytes"] == 128 * 2**30
            and result["hardware"]["storage"] == "internal SSD",
            "The M5 reader comparison did not complete its declared scope")
    require(result["binary_sha256"] == baseline["binary_sha256"] == build["binary_sha256"]
            and digest(prior / "direct_weights.cpp") == build["reader_source_sha256"]
            and digest(prior / "run_probe.py") == build["driver_sha256"]
            and result["selection_sha256"] == baseline["selection_sha256"] == digest(prior / "selection.json")
            and result["inventory_sha256"] == baseline["inventory_sha256"]
            and result["revision"] == baseline["revision"],
            "The native reader or pinned publisher identity changed")
    before, after = load(root / "cpu-before.json"), load(root / "cpu-after.json")
    require(before["min_idle_percent"] == 90 and 90 <= before["cpu_idle_percent"] <= 100
            and 90 <= after <= 100, "The host missed the timing floor")
    arms, prior_arms = result["arms"], baseline["arms"]
    require(len(arms) == len(prior_arms) == 18, "The counterbalanced comparison is incomplete")
    groups = {}
    for observed, expected in zip(arms, prior_arms, strict=True):
        name = observed["shard"]
        require(name == expected["shard"] and all(observed[x] == expected[x] for x in MATCH),
                f"Shard content, allocation, reader, or order changed for {name}")
        pages = observed["file_pages_before"]
        expected_resident = 0 if observed["cache"] == "clone" else pages["selected_pages"]
        require(observed["pressure_before"] == observed["pressure_after"] == 1
                and pages["selected_pages"] > 0 and pages["resident_pages"] == expected_resident
                and observed["elapsed_seconds"] > 0, f"Cache state, pressure, or timing changed for {name}")
        groups.setdefault(name, []).append(observed)

    shards = {}
    for name, rows in groups.items():
        result_by_cache = {}
        for cache in ("clone", "reused"):
            timings = {
                mode: [r["elapsed_seconds"] for r in rows if (r["cache"], r["mode"]) == (cache, mode)]
                for mode in ("mmap", "pread", "pread-pair")
            }
            require(len(timings["pread"]) == len(timings["pread-pair"]) == 2
                    and len(timings["mmap"]) == int(cache == "reused"), "The treatment repeat changed")
            result_by_cache[cache] = {mode: median(values) for mode, values in timings.items() if values}
        shards[name] = {"matched_tensors_per_arm": rows[0]["tensors"], "median_seconds": result_by_cache}
    return {"qualification": False, "same_binary_as_m2": True,
            "cpu_idle_percent": [before["cpu_idle_percent"], after], "total_arms": len(arms), "shards": shards}


if __name__ == "__main__":
    print(json.dumps(replay(ROOT, PRIOR), indent=2, allow_nan=False))
