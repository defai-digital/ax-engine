"""Replay the exact M5 arrivals, route telemetry and explicit held cancellation."""
from hashlib import sha256
import importlib.util
import json
from math import isfinite
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parent
EARLY = ROOT.parent / "flash-next-server-prefill-early-m2-20261122"
CANCEL = ROOT.parent / "flash-next-stepwise-held-cancel-m2-20261122"
LATE = ROOT.parent / "flash-next-server-prefill-tail-fixed-m5-20261122"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def open_evidence(path):
    data = {}
    with tarfile.open(path) as archive:
        for item in archive:
            name = Path(item.name)
            require(item.isfile() and len(name.parts) == 1 and name.name == item.name
                    and item.name not in data, "Unexpected evidence member")
            with archive.extractfile(item) as stream:
                data[item.name] = stream.read()
    return data


def hashes(data):
    rows = [row.split() for row in data.decode().splitlines()]
    result = {name: checksum for checksum, name in rows}
    require(len(rows) == len(result), "Duplicate measured input")
    return result


def load_early():
    spec = importlib.util.spec_from_file_location("tail_holdout_inputs", EARLY / "replay.py")
    require(spec and spec.loader, "Missing sealed workload verifier")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(value):
    return sha256(value).hexdigest()


def replay():
    data = open_evidence(ROOT / "evidence.tar.xz")
    older = open_evidence(LATE / "evidence.tar.xz")
    earlier_inputs = hashes(older["inputs.sha256"])
    newer_inputs = hashes(data["primary-inputs.sha256"])
    new_files = {"run_early.py", "run_held_cancel.py", "run_primary_holdout.py",
                 "await_tail.sh", "run_tail_serialized.sh"}
    require(newer_inputs.keys() - earlier_inputs.keys() == new_files
            and earlier_inputs.keys() - newer_inputs.keys() == set()
            and all(earlier_inputs[name] == newer_inputs[name] for name in earlier_inputs if name != "run_m5.sh")
            and all(newer_inputs[name] == digest(data[name])
                    for name in new_files | {"run_m5.sh", "build.json"})
            and data["build.json"] == older["build.json"]
            and data["await_tail.sh"] == older["await_tail.sh"]
            and data["run_tail_serialized.sh"] == older["run_tail_serialized.sh"]
            and data["run_early.py"] == (EARLY / "run_early.py").read_bytes()
            and data["run_held_cancel.py"] == (CANCEL / "run_held_cancel.py").read_bytes(),
            "Measured binary, workload, verifier or dispatcher changed")
    receipt = json.loads(data["receipt.json"])
    prior = json.loads(older["receipt.json"])
    require(data["status.txt"] == b"0\n" and receipt["status"] == "complete"
            and receipt["scope"] == "M5 early-arrival and cancellation holdouts"
            and receipt["qualification"] is False
            and all(receipt[k] == prior[k] for k in ("build", "pack_revision", "manifest", "inventory_sha256"))
            and receipt["build"] == json.loads(data["build.json"])
            and receipt["build"]["server_sha256"] == newer_inputs["ax-engine-server-tail"]
            and receipt["build"]["runtime_commit"] == "14235e84b01456cfcb40ba6fcb717ae4798fa8f7",
            "The M5 runtime or pack differs from the four-arm experiment")
    capacity = json.loads(data["dispatch-capacity.json"])["capacity"]
    require(capacity["pressure"] == 1 and capacity["eligible_for_attempt"] is False
            and capacity["reason"] == "background-rss-exceeds-signed-margin",
            "Measured dispatch eligibility changed")
    samples = data["primary-tail-holdout-samples.jsonl"]
    rows = [json.loads(row) for row in samples.splitlines()]
    require(rows and len(rows) == receipt["pressure_samples"]
            and digest(samples) == receipt["pressure_sha256"] and {row["pressure"] for row in rows} == {1},
            "A native sample or pressure receipt changed")
    idle = [receipt[f"cpu_idle_{position}_percent"] for position in ("before", "after")]
    require(all(isfinite(percent) and 0 <= percent <= 100 for percent in idle),
            "Missing primary-tier CPU observations")
    early = load_early()
    with early.inputs(EARLY) as (early_driver, files):
        for field in ("run_held_cancel.py", "run_primary_holdout.py"):
            (files / field).write_bytes(data[field])
        import run_primary_holdout as driver
        (files / "build.json").write_bytes(data["build.json"])
        cases, _, oracle, _, _ = driver.read_inputs()
        labels = ("trigger-48", "trigger-65", "held-cancel")
        require(tuple(receipt["runs"]) == labels, "The native workload order changed")
        results = {}
        for label in labels:
            payload = data[f"{label}.json"]
            run = receipt["runs"][label]
            require(digest(payload) == run["sha256"], "Measured case receipt changed")
            raw = json.loads(payload)
            if label == "held-cancel":
                result = driver.cancel.verify(raw, cases, oracle)
            else:
                result = early_driver.verify(raw, int(label.split("-")[1]), cases, oracle)
                require(all(raw[k] == v for k, v in result.items())
                        and raw["drained"] == dict.fromkeys(driver.mixed.COUNTERS, 0),
                        "The native arrival did not preserve its request geometry or drain")
            require(result == run["checks"], "Replay and live route assertions disagree")
            results[label] = result
    return {"scope": receipt["scope"], "qualification": False,
            "runtime_commit": receipt["build"]["runtime_commit"], "matching_generated_ids": 576,
            "cpu_idle_percent": idle, "normal_pressure_samples": len(rows),
            "peak_process_footprint_bytes": max(row["footprint_bytes"] or 0 for row in rows),
            "runs": results}


if __name__ == "__main__":
    print(json.dumps(replay(), indent=2, sort_keys=True, allow_nan=False))
