"""Validate real-shard positional-read and mapped-copy parity."""
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INVENTORY = ROOT.parent / "qualification/2026-10-04-flash-next-mxfp4-native-support/pack-inventory.json"
INVENTORY_SHA = "97b890234af731ac6e07d2ed7dbe32b4b0fcb1f0aff01addaaa340a838b28cc6"
ORDERS = (
    (("clone", "mmap"), ("clone", "pread"), ("reused", "pread"), ("reused", "mmap")),
    (("clone", "pread"), ("clone", "mmap"), ("reused", "mmap"), ("reused", "pread")),
)
COMPARABLE_FIELDS = ("weights_sha256", "metal_sha256", "allocated_bytes",
                     "selected_bytes", "unaligned_source_tensors", "tensors")


def digest(payload):
    return sha256(payload).hexdigest()


def replay(root=ROOT):
    assert (root / "phase-status.txt").read_bytes() == b"direct-weights 0\n"
    original = INVENTORY.read_bytes()
    assert digest(original) == INVENTORY_SHA
    inventory = json.loads(original)
    publisher = {entry["name"]: entry["publisher_sha256"] for entry in inventory["members"]
                 if entry["name"].endswith(".safetensors")}
    bindings = {name: value for value, name in
                (line.split() for line in (root / "inputs.sha256").read_text().splitlines())}
    for name in ("direct_weights.cpp", "selection.json", "run_probe.py"):
        assert digest((root / name).read_bytes()) == bindings[name]
    assert bindings["inventory.json"] == INVENTORY_SHA
    result = json.loads((root / "result.json").read_bytes())
    assert result["completed"] is True and result["short_read_rejected"] is True
    assert result["binary_sha256"] == bindings["direct_weights"]
    assert result["inventory_sha256"] == INVENTORY_SHA and result["revision"] == inventory["revision"]
    assert result["hardware"] == {"soc": "Apple M2 Ultra", "memory_bytes": 192 * 2**30,
                                  "os": "27.2.1", "storage": "external SSD"}
    assert result["page_size"] == 16384
    assert result["selection_sha256"] == bindings["selection.json"]
    spec = json.loads((root / "selection.json").read_bytes())["files"]
    assert len(spec) == 2 and len(result["arms"]) == 8
    comparisons = {}
    for index, (file, expected_order) in enumerate(zip(spec, ORDERS, strict=True)):
        name = file["name"]
        rows = result["arms"][4 * index:4 * (index + 1)]
        assert tuple((row["cache"], row["mode"]) for row in rows) == expected_order
        values = {field: rows[0][field] for field in COMPARABLE_FIELDS}
        assert values["tensors"] == len(file["names"]) == len(set(file["names"]))
        assert values["selected_bytes"] == file["selected_bytes"] <= values["allocated_bytes"]
        assert 0 <= values["unaligned_source_tensors"] <= values["tensors"]
        for field in ("weights_sha256", "metal_sha256"):
            assert len(bytes.fromhex(values[field])) == 32
        seconds = {}
        for row in rows:
            assert row["shard"] == name and row["publisher_sha256"] == publisher[name]
            assert all(row[field] == value for field, value in values.items())
            assert row["pressure_before"] == row["pressure_after"] == 1
            pages = row["file_pages_before"]
            assert pages["selected_pages"] > 0
            assert pages["resident_pages"] == (0 if row["cache"] == "clone" else pages["selected_pages"])
            elapsed = row["elapsed_seconds"]
            assert elapsed > 0 and isfinite(elapsed)
            seconds[f'{row["cache"]}_{row["mode"]}'] = elapsed
        comparisons[name] = {**values, **seconds}
    return {"scope": result["scope"], "verified_short_read_rejection": True,
            "shards": comparisons}


if __name__ == "__main__":
    print(json.dumps(replay(), indent=2, sort_keys=True))
