"""Replay the Flash Next serial-versus-paired resident operand control."""
from hashlib import sha256
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PRIOR = ROOT.parent / "flash-next-owned-positional-shards-m2-20261122" / "result.json"


def load(path):
    return json.loads(path.read_bytes())


def require(valid, message):
    if not valid:
        raise ValueError(message)


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def main():
    baseline = load(PRIOR)
    result = load(ROOT / "result-m2.json")
    build = load(ROOT / "build.json")
    selection = load(ROOT / "selection.json")
    require(result["scope"] == "functional" and not result["qualification"]
            and result["completed"] and result["short_read_rejected"],
            "Functional control did not finish")
    require(result["binary_sha256"] == build["binary_sha256"]
            and result["selection_sha256"] == baseline["selection_sha256"] == digest(ROOT / "selection.json")
            and result["inventory_sha256"] == baseline["inventory_sha256"]
            and result["revision"] == baseline["revision"]
            and digest(ROOT / "direct_weights.cpp") == build["reader_source_sha256"]
            and digest(ROOT / "run_probe.py") == build["driver_sha256"]
            and build["mlx"] == build["mlx_metal"] == "0.32.4",
            "Build, input or publisher identity changed")
    require(len(result["arms"]) == 18, "Expected all measurement arms")

    for index, source in enumerate(selection["files"][:2]):
        name = source["name"]
        rows = [row for row in result["arms"] if row["shard"] == name]
        original = [row for row in baseline["arms"] if row["shard"] == name]
        calls = ["pread", "pread-pair", "pread-pair", "pread"]
        if index == 1:
            calls = ["pread-pair", "pread", "pread", "pread-pair"]
        sequence = [("clone", mode) for mode in calls] + [
            ("reused", mode) for mode in ("mmap", *calls)]
        require([(row["cache"], row["mode"]) for row in rows] == sequence,
                f"Counterbalanced request order changed for {name}")
        for row in rows:
            require(row["paired_tensors"] == (3 if row["mode"] == "pread-pair" else 0)
                    and row["pressure_before"] == row["pressure_after"] == 1
                    and row["elapsed_seconds"] > 0,
                    f"Result does not exercise the expected reader for {name}")
            for key in ("allocated_bytes", "selected_bytes", "weights_sha256", "metal_sha256",
                        "tensors", "publisher_sha256", "unaligned_source_tensors"):
                require(all(other[key] == row[key] for other in original),
                        f"Array identity field changed for {name}: {key}")
        print(f"{name}: {len(rows)} matching controls, {rows[0]['tensors']} tensors, "
              f"{rows[0]['selected_bytes']} exact bytes")


if __name__ == "__main__":
    main()
