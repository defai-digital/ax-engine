"""Replay the identical scoped reader on the primary M5 Max hardware."""
from gzip import decompress
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent
M2 = ROOT.parent / "flash-next-page-aligned-reader-m2-20261122"
MODES = ("mmap", "aligned-buffered", "aligned-nocache")


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def read(path):
    return json.loads(path.read_bytes())


def replay(root=ROOT):
    raw = decompress((root / "result.json.gz").read_bytes())
    result = json.loads(raw)
    baseline = read(M2 / "result.json")
    build = read(M2 / "build.json")
    need((root / "status.txt").read_text() == "0\n" and result["completed"] is True
         and result["qualification"] is False and result["scope"] == "m5-diagnostic"
         and result["hardware"]["soc"] == "Apple M5 Max" and result["hardware"]["memory_bytes"] == 128 * 1024**3
         and result["hardware"]["storage"] == "internal SSD",
         "The primary-tier reader run did not complete on its standard host")
    need(result["binary_sha256"] == baseline["binary_sha256"] == build["binary_sha256"]
         and all(result[key] == baseline[key] for key in ("inventory_sha256", "selection_sha256", "revision", "page_size")),
         "The M5 run used different binary, publisher data or geometry")
    idle = [read(root / name)["cpu_idle_percent"] for name in ("cpu-before.json", "cpu-after.json")]
    need(all(90 <= value <= 100 for value in idle), "Reader timing host fell below the 90 percent idle floor")
    need(read(root / "m5-tail.json") == read(M2 / "unaligned-tail.json"), "Unaligned tail or EOF control changed")
    actual = result["arms"]
    need(len(actual) == len(baseline["arms"]) == 24, "Missing primary-tier counterbalanced arms")
    for row, oracle in zip(actual, baseline["arms"], strict=True):
        need(row.keys() == oracle.keys() and all(row[key] == oracle[key] for key in oracle if key != "elapsed_seconds")
             and isfinite(row["elapsed_seconds"]) and row["elapsed_seconds"] > 0,
             "M5 byte, Metal, allocation, pressure or cache-state evidence changed")
    shards = {}
    for name in sorted({row["shard"] for row in actual}):
        by_cache = {}
        for cache in ("clone", "reused"):
            series = [row for row in actual if row["shard"] == name and row["cache"] == cache]
            medians = {mode: median(row["elapsed_seconds"] for row in series if row["mode"] == mode) for mode in MODES}
            pairs = [{row["mode"]: row["elapsed_seconds"] for row in series[i:i + 3]} for i in (0, 3)]
            by_cache[cache] = {"median_seconds": medians,
                               "nocache_difference_percent": round(100 * (medians["aligned-nocache"] / medians["mmap"] - 1), 3),
                               "nocache_per_order_difference_percent": [round(100 * (pair["aligned-nocache"] / pair["mmap"] - 1), 3)
                                                                         for pair in pairs],
                               "nocache_after_resident_pages": [row["file_pages_after"]["resident_pages"] for row in series
                                                                 if row["mode"] == "aligned-nocache"]}
        shards[name] = by_cache
    return {"scope": "M5 Max page-aligned resident weight reader component", "qualification": False,
            "hardware": result["hardware"], "cpu_idle_percent": idle, "build": build,
            "result_sha256": sha256(raw).hexdigest(), "native_control_runs": len(actual),
            "matching_m2_payload_observations": sum(row["tensors"] for row in actual), "shards": shards}


if __name__ == "__main__":
    print(json.dumps(replay(), indent=2, sort_keys=True, allow_nan=False))
