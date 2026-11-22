"""Attribute Flash Next startup timer differences to their recorded shards."""

from hashlib import sha256
from gzip import decompress
import json
from math import isfinite
from pathlib import Path
import re
from statistics import median
from tarfile import open as open_tar

ROOT = Path(__file__).resolve().parent
HISTORY = ROOT.parent / "flash-next-mxfp4-startup-phases-m5-20261121"
DELIVERED = ROOT.parent / "flash-next-delivery-bound-startup-m5-20261122"
READER = ROOT.parent / "flash-next-paired-resident-reads-m2-20261122"
HISTORY_SHA = "1872c5a219698e84c63a6122d36cd896759b0a981a995ebb25a64f7f99a3ded3"
DELIVERED_SHA = "f2ea687d64414e9ce4b91785e230d9eae571d44a09e43b665a48f2cee72cc1de"
ORDERS = {"disabled-first": ("disabled", "default"), "default-first": ("default", "disabled")}
DELIVERY_MODES = ("disabled", "required", "default")
GROUP = range(13, 19)
PAIRED_PROBE = (21, 24)
HOLDOUT = (17, 18)
SHARDS = (1, *range(12, 26))
HOLDOUT_PUBLISHER_SHA = (
    "febb6963db738629ac60b87e2991ad6bbff124e545f601c2fa33d90978604d4f",
    "448f8f0b37f598d7142b675f659b89c22a5538272cd29f0f374fa9a72553f7fd",
)
SHARD_LOG = re.compile(
    r'shard=Some\("(model-\d{5}-of-00025\.safetensors)"\) '
    r'tensor_count=(\d+) read_us=(\d+) eval_us=(\d+)'
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def bound_archive(folder, expected_digest):
    path = folder / "evidence.tar.xz"
    require(sha256(path.read_bytes()).hexdigest() == expected_digest, f"Changed archive {folder.name}")
    return open_tar(path)


def member(archive, name):
    handle = archive.extractfile(name)
    if handle is None:
        raise ValueError(f"Missing archived log or receipt: {name}")
    with handle:
        return handle.read()


def indexed(rows):
    parsed = {}
    for row in rows:
        number = int(row["name"].split("-")[1])
        require(number not in parsed and row["read_us"] > 0 and row["eval_us"] >= 0,
                f"Duplicate shard or invalid timing: {row['name']}")
        parsed[number] = row
    require(tuple(parsed) == SHARDS and sum(row["tensors"] for row in parsed.values()) == 1942,
            "Unexpected resident shard selection")
    return parsed


def replay_holdout():
    result = json.loads((ROOT / "holdout-m2.json").read_bytes())
    selection_raw = decompress((ROOT / "holdout-selection.json.gz").read_bytes())
    selection = json.loads(selection_raw)["files"]
    build = json.loads((READER / "build.json").read_bytes())
    prior = json.loads((READER / "result-m2.json").read_bytes())
    require(result["scope"] == "functional" and result["qualification"] is False
            and result["completed"] is True and result["short_read_rejected"] is True,
            "Incomplete M2 functional holdout")
    require(result["selection_sha256"] == sha256(selection_raw).hexdigest()
            and result["binary_sha256"] == build["binary_sha256"] == prior["binary_sha256"]
            and all(result[field] == prior[field]
                    for field in ("revision", "inventory_sha256", "hardware", "page_size")),
            "M2 holdout build, pack or host identity changed")
    names = [f"model-{number:05d}-of-00025.safetensors" for number in HOLDOUT]
    require([item["name"] for item in selection] == names
            and [row["shard"] for row in result["arms"]] == [name for name in names for _ in range(9)],
            "M2 holdout shard roster changed")
    timings = {}
    for index, shard in enumerate(selection):
        name = shard["name"]
        rows = [row for row in result["arms"] if row["shard"] == name]
        modes = ("pread", "pread-pair", "pread-pair", "pread")
        if index == 1:
            modes = ("pread-pair", "pread", "pread", "pread-pair")
        sequence = [("clone", mode) for mode in modes] + [("reused", mode) for mode in ("mmap", *modes)]
        require([(row["cache"], row["mode"]) for row in rows] == sequence,
                f"Unexpected holdout arm order for {name}")
        require(len(shard["names"]) == len(set(shard["names"])) == 158
                and all(row["tensors"] == 158 and row["selected_bytes"] == shard["selected_bytes"]
                        and row["publisher_sha256"] == HOLDOUT_PUBLISHER_SHA[index]
                        and row["pressure_before"] == row["pressure_after"] == 1
                        and row["paired_tensors"] == (11 if row["mode"] == "pread-pair" else 0)
                        and isfinite(row["elapsed_seconds"]) and row["elapsed_seconds"] > 0 for row in rows),
                f"Unexpected tensor, reader or pressure measurements for {name}")
        for key in ("weights_sha256", "metal_sha256", "allocated_bytes", "unaligned_source_tensors"):
            require(len({row[key] for row in rows}) == 1, f"Holdout output differs for {name}: {key}")
        require(all(row["file_pages_before"]["resident_pages"] ==
                    (0 if row["cache"] == "clone" else row["file_pages_before"]["selected_pages"])
                    for row in rows), f"Observed cache state differs for {name}")
        timings[name] = {
            cache: {
                mode: round(median(row["elapsed_seconds"] for row in rows
                                   if row["cache"] == cache and row["mode"] == mode), 6)
                for mode in mode_set
            }
            for cache, mode_set in (("clone", ("pread", "pread-pair")),
                                    ("reused", ("mmap", "pread", "pread-pair")))
        }
    return {"scope": "M2 functional", "arms": len(result["arms"]), "unique_tensors": 316,
            "unique_published_bytes": sum(shard["selected_bytes"] for shard in selection),
            "wall_seconds_medians": timings}


def replay():
    with bound_archive(HISTORY, HISTORY_SHA) as archive:
        history = json.loads(member(archive, "raw/startup-phase.json"))
    require(not history["qualification"] and len(history["starts"]) == 4,
            "Expected the four counterbalanced diagnostic starts")
    historical = {name: indexed(row["shards"]) for name, row in history["starts"].items()}

    with bound_archive(DELIVERED, DELIVERED_SHA) as archive:
        require(member(archive, "phase-status.txt") == b"delivery-trace 0\n",
                "Expected the matching-manifest diagnostic")
        delivered = {}
        for mode in DELIVERY_MODES:
            log = member(archive, f"native-support/server-{mode}.log").decode()
            delivered[mode] = indexed(
                {"name": name, "tensors": int(count), "read_us": int(read), "eval_us": int(evaluate)}
                for name, count, read, evaluate in SHARD_LOG.findall(log)
            )
    counts = {number: row["tensors"] for number, row in delivered["default"].items()}
    require(all({number: row["tensors"] for number, row in start.items()} == counts
                for start in (*historical.values(), *delivered.values())),
            "The recorded tensor roster differs across starts")

    pairs = {}
    for order, (first, second) in ORDERS.items():
        baseline = historical[f"{order}/{first}"]
        later = historical[f"{order}/{second}"]
        diff = {number: later[number]["read_us"] - baseline[number]["read_us"] for number in SHARDS}
        pairs[order] = {
            "first_mode": first,
            "second_mode": second,
            "total_second_minus_first_us": sum(diff.values()),
            "shards_13_through_18_second_minus_first_us": sum(diff[number] for number in GROUP),
            "shards_21_and_24_second_minus_first_us": sum(diff[number] for number in PAIRED_PROBE),
            "shards_17_and_18_second_minus_first_us": sum(diff[number] for number in HOLDOUT),
            "per_shard_second_minus_first_us": diff,
        }

    by_mode = {}
    for mode, rows in delivered.items():
        total = sum(row["read_us"] for row in rows.values())
        by_mode[mode] = {
            "total_copy_and_allocation_us": total,
            "shards_21_and_24_share_percent": round(
                100 * sum(rows[number]["read_us"] for number in PAIRED_PROBE) / total, 3
            ),
            "shards_17_and_18_share_percent": round(
                100 * sum(rows[number]["read_us"] for number in HOLDOUT) / total, 3
            ),
            "per_shard_read_us": {number: row["read_us"] for number, row in rows.items()},
        }
    return {
        "scope": "M5 guarded mmap copy-and-allocation timer attribution",
        "qualification": False,
        "evidence_sha256": {"historical": HISTORY_SHA, "matching_manifest": DELIVERED_SHA},
        "counterbalanced_direct_starts": pairs,
        "matching_manifest_starts": by_mode,
        "selected_resident_tensors": sum(counts.values()),
        "paired_reader_original_shards": PAIRED_PROBE,
        "paired_reader_holdout_shards": HOLDOUT,
        "holdout": replay_holdout(),
    }


if __name__ == "__main__":
    print(json.dumps(replay(), indent=2, allow_nan=False))
