"""Replay the completed M5 default-mode warmup phase."""

import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent
M2 = ROOT.parent / "flash-next-worker-warmup-once-m2-20261122"
THRESHOLDS = ROOT.parents[2] / "scripts/flash_next_release_thresholds.json"
THRESHOLDS_SHA = "17a39e0229cf734dca04ffa5abb4718a6c7b1c9c773891385a37396a1625345b"


def replay():
    spec = importlib.util.spec_from_file_location("previous_warmup", M2 / "replay.py")
    if spec is None or spec.loader is None:
        raise ValueError("Missing exact harness verifier")
    previous = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(previous)
    need, read, digest = previous.need, previous.read, previous.digest
    hashes = dict(line.split()[::-1] for line in (ROOT / "inputs.sha256").read_text().splitlines())
    build = read(ROOT / "build.json")
    need(read(M2 / "build.json") == build and build["server_sha256"] == hashes["ax-engine-server-startup"]
         and build["driver_sha256"] == hashes["run_warmup.py"] == digest(M2 / "run_warmup.py")
         and digest(ROOT / "build.json") == hashes["build.json"]
         and digest(THRESHOLDS) == THRESHOLDS_SHA,
         "The candidate build or readiness thresholds changed")
    need(all(digest(ROOT / name) == hashes[name]
             for name in ("run_m5.sh", "run_m5.py", "run_load_order.py")), "The M5 stage source changed")

    with TemporaryDirectory() as directory:
        source = Path(directory)
        previous.unpack(previous.DEPENDENCIES, source, previous.HARNESS)
        need(all(digest(source / name) == hashes[name] for name in previous.HARNESS), "A staged helper changed")
        names = {"progress.json", *(f"default-{i}.json" for i in range(1, 5)),
                 *(f"server-default-{i}.log" for i in range(1, 5))}
        previous.unpack(ROOT / "evidence.tar.xz", source, names)
        sys.path[:0] = [str(source), str(M2)]
        from compare_m5 import compare
        report = compare(source, build, read(THRESHOLDS), modes=("default",), receipt="progress.json")
        progress = read(source / "progress.json")
        need(progress["inventory_sha256"] == hashes["inventory.json"], "Inventory digest changed")
        return report | {
            "completed_phase": "default",
            "ready_s_in_process_order": [row["ready_seconds"] for row in progress["arms"]],
            "worker_s_in_process_order": [row["worker_elapsed_us"] / 1e6 for row in progress["arms"]],
            "normal_pressure_samples": sum(row["normal_pressure_samples"] for row in progress["arms"]),
        }


if __name__ == "__main__":
    print(json.dumps(replay(), indent=2, allow_nan=False))
